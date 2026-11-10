@echo off
chcp 65001 >nul
title Detection de faux scans CT - application
cd /d "%~dp0"
set OUVRIR=1
set PYTHONUTF8=1
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
echo Chargement des modeles (environ 10 s), puis ouverture de http://127.0.0.1:7860 dans le navigateur...
echo Pour arreter l'application : fermez cette fenetre.
"%PY%" -W ignore app.py
pause
