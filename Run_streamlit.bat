@echo off
REM Récupère le chemin du script
set "SCRIPT_DIR=%~dp0"

REM Recherche Conda
for /f "delims=" %%i in ('where conda 2^>nul') do set "CONDA_PATH=%%i"

REM Fallback si non trouvé
if "%CONDA_PATH%"=="" (
    if exist "C:\Users\%USERNAME%\Anaconda3\condabin\conda.bat" (
        set "CONDA_PATH=C:\Users\%USERNAME%\Anaconda3\condabin\conda.bat"
    ) else if exist "C:\Users\%USERNAME%\Miniconda3\condabin\conda.bat" (
        set "CONDA_PATH=C:\Users\%USERNAME%\Miniconda3\condabin\conda.bat"
    ) else (
        echo Conda introuvable. Veuillez l'installer.
        pause
        exit /b
    )
)

REM Active la base pour éviter des erreurs conda init
call "%CONDA_PATH%" activate base

REM Vérifie si l'environnement existe
conda env list | findstr /C:"python_stru_env" >nul
if errorlevel 1 (
    echo Création de l'environnement Conda 'python_stru_env'...
    conda create -y -n python_stru_env python=3.10.16
)

REM Active l'environnement
echo Activation de l'environnement...
call conda activate python_stru_env

REM Installation propre des dépendances
echo Installation des dépendances...
if exist "%SCRIPT_DIR%requirements.txt" (
    "%CONDA_PREFIX%\python.exe" -m pip install --upgrade pip
    "%CONDA_PREFIX%\python.exe" -m pip install -r "%SCRIPT_DIR%requirements.txt"
) else (
    echo [AVERTISSEMENT] requirements.txt introuvable dans %SCRIPT_DIR%
)

REM Va dans le dossier du script
cd /d "%SCRIPT_DIR%"

REM Lance Streamlit
echo Lancement de Streamlit...
call streamlit run main.py

pause
