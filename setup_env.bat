@echo off
REM Surgery Duration Prediction Environment Setup Script for Windows
REM This script sets up the environment for reproducible results

echo Setting up Surgery Duration Prediction environment...

REM Set Python hash seed for reproducibility
set PYTHONHASHSEED=0
echo Set PYTHONHASHSEED=0

REM Set other environment variables
set PYTHONPATH=%PYTHONPATH%;%CD%\src

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

echo Environment setup complete!
echo To activate the environment in the future, run: venv\Scripts\activate.bat
echo To deactivate, run: deactivate
pause
