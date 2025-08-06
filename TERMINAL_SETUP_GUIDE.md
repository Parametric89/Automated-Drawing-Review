# Terminal Setup Guide for Cursor

This guide will help you set up your terminal in Cursor to automatically use your `rhino_mcp` conda environment.

## Current Status
✅ Conda is installed at: `C:\Users\ersk\AppData\Local\anaconda3\Scripts\conda.exe`  
✅ Environment `rhino_mcp` exists and works  
✅ Python 3.10.18 with OpenCV, NumPy, and PyTorch is available  

## Solutions

### Option 1: Use the Full Conda Path (Recommended for AI Assistant)
When I need to run Python scripts, I'll use this command format:
```cmd
C:\Users\ersk\AppData\Local\anaconda3\Scripts\conda.exe run -n rhino_mcp python <script_name>
```

**Example:**
```cmd
C:\Users\ersk\AppData\Local\anaconda3\Scripts\conda.exe run -n rhino_mcp python test_env.py
```

### Option 2: Set Up PowerShell Profile (For Manual Use)
1. Open PowerShell as Administrator
2. Run: `Set-ExecutionPolicy RemoteSigned`
3. Run the setup script: `powershell -ExecutionPolicy Bypass -File setup_powershell_profile.ps1`
4. Restart your terminal

### Option 3: Use the Batch File
Run: `start_with_env.bat` to start a command prompt with the environment activated.

### Option 4: Configure Cursor Settings
1. Open Cursor Settings (Ctrl+,)
2. Search for "terminal"
3. Set the default shell to: `C:\Users\ersk\OneDrive - Kvadrat\Desktop\ML review\start_with_env.bat`

## Testing
Run this command to test if everything works:
```cmd
C:\Users\ersk\AppData\Local\anaconda3\Scripts\conda.exe run -n rhino_mcp python test_env.py
```

Expected output:
```
Python version: 3.10.18 | packaged by conda-forge | (main, Jun  4 2025, 14:42:04) [MSC v.1943 64 bit (AMD64)]
Python executable: C:\Users\ersk\AppData\Local\anaconda3\envs\rhino_mcp\python.exe
Current working directory: C:\Users\ersk\OneDrive - Kvadrat\Desktop\ML review
✓ OpenCV imported successfully
✓ NumPy imported successfully
✓ PyTorch imported successfully
  PyTorch version: 2.5.1+cu121
Environment test completed!
```

## Quick Reference Commands

### For AI Assistant (me):
- Run Python script: `C:\Users\ersk\AppData\Local\anaconda3\Scripts\conda.exe run -n rhino_mcp python <script>`
- Install package: `C:\Users\ersk\AppData\Local\anaconda3\Scripts\conda.exe run -n rhino_mcp pip install <package>`
- Check environment: `C:\Users\ersk\AppData\Local\anaconda3\Scripts\conda.exe run -n rhino_mcp python test_env.py`

### For Manual Use:
- Start environment: `start_with_env.bat`
- Or use PowerShell with profile setup

## Troubleshooting
If you get "conda not found" errors:
1. Make sure the path is correct: `C:\Users\ersk\AppData\Local\anaconda3\Scripts\conda.exe`
2. Try running: `C:\Users\ersk\AppData\Local\anaconda3\Scripts\conda.exe --version`
3. If that fails, reinstall Anaconda or check the installation path 