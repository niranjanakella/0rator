#!/usr/bin/env python3
"""
Main entry point for 0rator macOS application.
This is a simplified launcher to avoid py2app recursion issues.
"""

import sys
import os

# Debug information for troubleshooting
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {__file__}")

# Add the orator_app directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(script_dir, 'orator_app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

print(f"Added to path: {app_dir}")
print(f"App dir exists: {os.path.exists(app_dir)}")

def main():
    """Main application entry point."""
    try:
        print("Starting 0rator application...")
        
        # First, try to run the test to verify bundle
        try:
            import test_bundle
            test_bundle.test_bundle_structure()
        except Exception as e:
            print(f"Bundle test failed: {e}")
        
        # Try to import and run the main GUI application
        print("Attempting to import OratorApp...")
        from core.orator_app import OratorApp
        
        print("Creating OratorApp instance...")
        app = OratorApp()
        
        print("Running application...")
        app.run()
        
    except ImportError as e:
        print(f"Error importing OratorApp: {e}")
        print("Attempting fallback to example...")
        
        # Fallback to example if GUI fails
        try:
            from example import main as example_main
            print("Running example application...")
            example_main()
        except ImportError as e2:
            print(f"Example import failed: {e2}")
            print("Creating minimal test application...")
            
            # Last resort - create a minimal test app
            try:
                import tkinter as tk
                root = tk.Tk()
                root.title("0rator - Test Mode")
                label = tk.Label(root, text="0rator is running in test mode.\nMain application failed to load.")
                label.pack(padx=20, pady=20)
                root.mainloop()
            except:
                print("All application modes failed")
                sys.exit(1)
                
    except Exception as e:
        print(f"Error running application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()