@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && set PY=py -3
%PY% -c "import numpy, scipy, matplotlib, soundfile, sounddevice" 2>nul
if errorlevel 1 (
    echo A instalar as bibliotecas necessarias - so acontece da primeira vez...
    %PY% -m pip install --user numpy scipy matplotlib soundfile sounddevice
)
%PY% app.py
if errorlevel 1 pause
