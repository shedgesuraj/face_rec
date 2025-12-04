#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper script - automatically activates venv and runs face recognition
"""

import sys
import os
import subprocess

# Get project directory
project_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(project_dir, '.venv', 'Scripts', 'python.exe')

# Check if venv Python exists
if not os.path.exists(venv_python):
    print("[ERROR] Virtual environment not found at:", venv_python)
    sys.exit(1)

# Run run_recognition.py with venv Python
script_path = os.path.join(project_dir, 'run_recognition.py')

print("[INFO] Using virtual environment Python...")
print("[INFO] Running face recognition system...\n")

# Execute with venv Python
result = subprocess.run([venv_python, script_path], cwd=project_dir)
sys.exit(result.returncode)
