from fastapi import FastAPI, Request, Response
from typing import Dict
import uvicorn

import json
import soundfile as sf

from whisperx_align import align

from fastapi.middleware.cors import CORSMiddleware
import uuid
from pydantic import BaseModel, Field
import re, io
from utils import *
import phonemizer
import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

# from phonemizer.backend.espeak.wrapper import EspeakWrapper
# EspeakWrapper.set_library("/opt/homebrew/bin/espeak-ng")
# EspeakWrapper.set_library("/usr/bin/espeak-ng")


from text_utils import TextCleaner
textcleaner = TextCleaner()
import torch 

# from firebase_module.firebase_api import authenticate_api_key, update_api_key_info
# from firebase_module.firebase_api_params import UpdateAPIKeyInfoParams

from models import build_model
from kokorooo import generatekoko
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = build_model('kokoro-v0_19.pth', device)
VOICE_NAME = "bf_isabella"
VOICEPACK = torch.load(f'voices/{VOICE_NAME}.pt', weights_only=True).to(device)
print(f'Loaded voice: {VOICE_NAME}')


import torchaudio
model_name = "WAV2VEC2_ASR_BASE_960H"
pipeline_type = "torchaudio"
bundle = torchaudio.pipelines.__dict__[model_name]
align_model = bundle.get_model(dl_kwargs={"model_dir": None}).to(device)
labels = bundle.get_labels()
align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
align_metadata = {"language": "en", "dictionary": align_dictionary, "type": "torchaudio"}

global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')


app = FastAPI()
# app.middleware("http")(authenticate_api_key)

allow_all = ['*']
app.add_middleware(
   CORSMiddleware,
   allow_origins=allow_all,
   allow_credentials=True,
   allow_methods=allow_all,
   allow_headers=allow_all,
   expose_headers=["X-Word-Timestamps"]
)

def merge_eps(timestamps):
     result = []
     skip_first_eps = True

     for i in range(len(timestamps)):
          start_time, end_time, word = timestamps[i]

          if word == "<eps>":
               if skip_first_eps:
                    # Skip merging the very first `<eps>` and just set the flag to False
                    skip_first_eps = False
                    continue
               else:
                    # Merge this `<eps>` with the last word in the result
                    if result:
                         last_start_time, _, last_word = result[-1]
                         result[-1] = [last_start_time, end_time, last_word]
          else:
          # Normal word, add to result list
               result.append([start_time, end_time, word])
     return result

class T2SRequestParams(BaseModel):
    text: str = Field(...)

def num_tokens(text: str) -> int:
    try:
        text = text.strip()
        text = text.replace('"', '')
        ps = global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = textcleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
        return len(tokens[0])
    except Exception as e:
        raise ValueError(text, e)

def remove_emojis(text):
    # This regex pattern captures a wide range of emoji patterns.
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

@app.get("/")
async def hello():
    return "Hello"

@app.post(f"/t2s/api/v1/generate", response_model=Dict)
async def generate(request: Request, params: T2SRequestParams):
    input_characters = len(params.text)
    # params.text = remove_emojis(params.text)

    noise = torch.randn(1,1,256).to('cuda' if torch.cuda.is_available() else 'cpu')
    audios = []
    sentences = split_text_into_chunks(params.text, num_tokens)

    for sentence_chunk in sentences:
        # audios.append(ljspeechimportable.inference(remove_emojis(sentence_chunk), 
        #                                     noise, word_tokenize,
        #                                     global_phonemizer,
        #                                     diffusion_steps=5, 
        #                                     embedding_scale=1.3, speed=1.3))
        audio, out_ps = generatekoko(MODEL, remove_emojis(sentence_chunk), VOICEPACK, lang=VOICE_NAME[0])
        audios.append(audio)

    # print(f"###TTS: {time.time()-elapse_tmp}")
    final_audios_array = np.concatenate(audios)
    byte_io = io.BytesIO()
    sf.write(byte_io, final_audios_array, 24000, format='WAV')
    byte_io.seek(0)

    # del audios
    # torch.cuda.empty_cache()
    
    output_duration = len(final_audios_array) / 24000

    #update the usage info in firestore
    # _ = await update_api_key_info(UpdateAPIKeyInfoParams(api_key=request.headers.get('API-KEY'),
    #                                                      input_characters=input_characters,
    #                                                      output_duration=output_duration))

    # # Generate unique IDs for temporary storage
    # unique_id = str(uuid.uuid4())
    # tmp_dir = "/tmp_holder"
    # os.makedirs(tmp_dir, exist_ok=True)
    
    # # Paths for temporary files
    # audio_path = os.path.join(tmp_dir, f"{unique_id}_audio.wav")
    # text_path = os.path.join(tmp_dir, f"{unique_id}_text.txt")
    # output_json_path = os.path.join(tmp_dir, f"{unique_id}_output.json")
    # # zip_file_path = os.path.join(tmp_dir, f"{unique_id}_files.zip")

    # # Save audio file temporarily
    # with open(audio_path, "wb") as audio_file:
    #     audio_file.write(byte_io.read())
    
    # # Save text file temporarily
    # with open(text_path, "w") as text_file:
    #     text_file.write(params.text)
    
    # # Run MFA align_one command
    # mfa_command = [
    #     "mfa", "align_one",
    #     audio_path,
    #     text_path,
    #     "english_us_arpa",
    #     "english_us_arpa",
    #     output_json_path,
    #     "--output_format", "json"
    # ]
    # subprocess.run(mfa_command, check=True)
    
    # with open(output_json_path, 'r') as output_json_file:
    #     output_json_data = json.load(output_json_file)
    
    # merged_eps_phons = merge_eps(output_json_data['tiers']['words']['entries'])
    # txt_lst = re.split(r'[ \-]', params.text)
    # txt_time_stamps = [[merged_eps_phons[i][0], merged_eps_phons[i][1], txt_lst[i]] for i in range(len(merged_eps_phons))]

    # # Create a ZIP archive containing the audio and JSON files
    # # with zipfile.ZipFile(zip_file_path, 'w') as zipf:
    # #     zipf.write(audio_path, arcname=f"{unique_id}_audio.wav")
    # #     zipf.write(output_json_path, arcname=f"{unique_id}_alignment.json")
    
    # # Clean up temporary files
    # os.remove(audio_path)
    # os.remove(text_path)
    # os.remove(output_json_path)
    
    # # return Response(content=open(zip_file_path, "rb").read(), media_type="application/zip", headers={"Content-Disposition": f"attachment; filename={unique_id}_files.zip"})

    # word_timestamps_json = json.dumps(txt_time_stamps)

    # elapse_tmp = time.time()
    alignments = []
    for idx, chunk in enumerate(sentences):
        transcript = {}
        transcript['segments'] = [{'text': chunk, "start": 0.0, "end": len(audios[idx])/24000}]
        alignment_info = align(transcript['segments'], align_model, align_metadata, audios[idx], device, return_char_alignments=False)
        alignments.append(alignment_info['word_segments'])

    # print(f"Alignment: {time.time() - elapse_tmp}")

    # for sen in sentences:
    #     print("\n\n")
    #     print(sen)
    #     print("\n\n")
    # elapse_tmp = time.time()
    # for i in range(1, len(alignments)):
    #     segment = alignments[i]
    #     prev_segment = alignments[i-1]
        
    #     if 'start' not in segment or 'end' not in segment:
    #         segment['start'] = prev_segment['end'] + 0.01
    #         segment['end'] = prev_segment['end'] + 0.1
        
    #     if i == len(alignments)-1:
    #         segment['end'] += 0.3

    # print(f"###Alignment Post Processing: {time.time()-elapse_tmp}")

    word_alignment_segments = []
    current_time = 0.0
    for i in range(len(alignments)): #len=1
        for word_seg_idx, word_seg in enumerate(alignments[i]):

            if word_seg_idx-1 < 0:
                if i-1<0:
                    prev_word_segment = {"start": 0.0, "end": 0.0}
                else:
                    prev_word_segment = alignments[i-1][-1]
            else:
                prev_word_segment = alignments[i][word_seg_idx-1]
            
            if 'start' not in word_seg or 'end' not in word_seg:
                word_seg['start'] = round(prev_word_segment['end'] + 0.01, 3)

                if (word_seg_idx+1) < len(alignments[i]):
                    if alignments[i][word_seg_idx+1].get('start'):
                        word_seg['end'] = alignments[i][word_seg_idx+1]['start'] - 0.01
                elif (i+1) < len(alignments):
                    if alignments[i+1][0].get('start'):
                        word_seg['end'] = alignments[i+1][0]['start'] - 0.01
                
                else:
                    word_seg['end'] = round(prev_word_segment['end'] + 0.03, 3)

            
            seg_info = {"word": word_seg['word'], 'start': current_time+word_seg['start'], 'end': current_time+word_seg['end']}
                
            word_alignment_segments.append(seg_info)
        
        # audios_duration = sum([len(aud)/24000 for aud in audios[:i+1]])
        # current_time += word_seg['end'] + ((audios_duration - (current_time+word_seg['end']))/2)
        current_time += (len(audios[i])/24000) - 0.001 #0.163 #Delta error corrections
    
    #Free up GPU memory
    del noise, audios
    torch.cuda.empty_cache()

    headers = {
            "X-Word-Timestamps": json.dumps(word_alignment_segments) #word_timestamps_json
        }

    return Response(
        content=byte_io.getvalue(),
        media_type="audio/wav",
        headers=headers
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1000)
    