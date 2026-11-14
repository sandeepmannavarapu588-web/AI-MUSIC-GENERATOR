@echo off
title Push to GitHub
cd /d "%~dp0"
echo =====================================================================
echo   Pushing AI Music Generator to GitHub...
echo   Repository: https://github.com/sandeepmannavarapu588-web/AI-MUSIC-GENERATOR
echo =====================================================================
echo.
git push -u origin main
echo.
echo =====================================================================
echo   Done!
echo =====================================================================
pause
