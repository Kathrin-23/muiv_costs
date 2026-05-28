@echo off
chcp 65001 > nul
python -m venv venv
call venv\Scriptsctivate
pip install -r requirements.txt
python main.py
pause
