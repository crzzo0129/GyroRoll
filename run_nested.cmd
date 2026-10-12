@echo off
"%~dp0.venv\Scripts\python.exe" "%~dp0sim.py" wheelbot --controller nested --speed 1 --duration 60 --camera-distance 1.1
if errorlevel 1 pause
