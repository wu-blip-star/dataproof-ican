@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python --version >nul 2>&1
    if errorlevel 1 goto no_python
    python -m venv .venv
    if errorlevel 1 goto error
)
".venv\Scripts\python.exe" -c "import streamlit, scipy, pandas, numpy, statsmodels, plotly, sklearn, openpyxl, xlrd" >nul 2>&1
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto error
)
echo Open http://127.0.0.1:8504 in your browser after the server starts.
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8504
if errorlevel 1 goto error
exit /b 0
:no_python
echo Please install Python 3.13 and select Add Python to PATH, then run start.bat again.
pause
exit /b 1
:error
echo DataProof could not start. Please keep this window open and share the error above.
pause
exit /b 1
