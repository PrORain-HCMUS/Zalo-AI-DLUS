@echo off
REM Setup script for AeroEyes project (Windows)

echo ===================================
echo AeroEyes Setup Script
echo ===================================

REM Check Python version
echo Checking Python version...
python --version

REM Create virtual environment
echo Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo Creating directories...
if not exist "data" mkdir data
if not exist "checkpoints" mkdir checkpoints
if not exist "results" mkdir results
if not exist "logs" mkdir logs

echo ===================================
echo Setup complete!
echo ===================================
echo.
echo Next steps:
echo 1. Activate environment: .venv\Scripts\activate
echo 2. Download dataset to data\
echo 3. Download weights to checkpoints\best.pt
echo 4. Run inference: python src\predict.py --help

pause
