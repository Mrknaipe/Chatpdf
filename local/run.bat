

@echo off
cd /d %~dp0
call .venv\Scripts\activate
pip install -r requirement.txt
python -m streamlit run app.py 2>nul
pause