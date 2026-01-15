@echo off
REM Quick setup script for Data Quality Research (Windows)

echo ================================
echo Data Quality Research - Setup
echo ================================

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

REM Run verification
echo Running verification tests...
python test_setup.py

echo.
echo Setup complete! Next steps:
echo   1. Activate environment: venv\Scripts\activate
echo   2. Run experiments: python run_experiments.py --sentence-only
pause
