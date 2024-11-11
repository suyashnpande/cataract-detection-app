@echo off

REM Define paths
set REQUIREMENTS_DIR=..
set TRAINING_SCRIPT_DIR=model_training
set TRAINING_SCRIPT=train_model.py
set VENV_DIR=venv

REM Navigate to the directory with requirements.txt and create a virtual environment
cd /d "%REQUIREMENTS_DIR%" || (echo Directory not found: %REQUIREMENTS_DIR% && exit /b 1)

REM Create a virtual environment if it doesn't already exist
if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

REM Activate the virtual environment (Windows uses Scripts\activate)
call "%VENV_DIR%\Scripts\activate.bat"

REM Upgrade pip and install dependencies from requirements.txt
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM Navigate to the model training folder
cd /d "%TRAINING_SCRIPT_DIR%" || (echo Directory not found: %TRAINING_SCRIPT_DIR% && exit /b 1)

REM Run the Python training script
python "%TRAINING_SCRIPT%"

REM End message
echo Training script executed successfully.
pause
