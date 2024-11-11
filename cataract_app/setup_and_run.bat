@echo off

REM Check for Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not added to PATH. Exiting...
    exit /b 1
)

REM Define variables
set VENV_DIR=venv
set MODEL_FILE=final_model.keras
set REQUIREMENTS_FILE=requirements.txt
set STREAMLIT_APP=app.py

REM Check if virtual environment directory exists
if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
call %VENV_DIR%\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install required packages
if exist "%REQUIREMENTS_FILE%" (
    echo Installing packages from %REQUIREMENTS_FILE%...
    python -m pip install -r %REQUIREMENTS_FILE%
) else (
    echo Requirements file %REQUIREMENTS_FILE% not found! Ensure it's in the same directory.
    exit /b 1
)

REM Check if model file exists
if not exist "%MODEL_FILE%" (
    echo Model file %MODEL_FILE% not found! Place it in the same directory as this script.
    exit /b 1
)

REM Run the Streamlit app
echo Starting Streamlit app...
streamlit run %STREAMLIT_APP%

pause
