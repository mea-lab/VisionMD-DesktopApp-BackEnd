@echo off
setlocal enabledelayedexpansion

set MODEL_URL=https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_eff2s_y4.zip
set MODEL_DIR=.\app\analysis\models
set MODEL_NAME=metrabs_eff2s_y4
set ZIP_FILE=%MODEL_NAME%.zip

if exist "%MODEL_DIR%\%MODEL_NAME%" (
    echo %MODEL_DIR%\%MODEL_NAME% already exists, skipping download.
    exit /b 0
)

if not exist "%MODEL_DIR%" (
    echo Model directory %MODEL_DIR% does not exist.
    exit /b 1
)

echo Downloading %MODEL_NAME% from %MODEL_URL% ...

curl -L "%MODEL_URL%" -o "%MODEL_DIR%\%ZIP_FILE%"
if %errorlevel% neq 0 (
    echo Download failed.
    exit /b 1
)

echo Unzipping...
powershell -Command "Expand-Archive -Path '%MODEL_DIR%\%ZIP_FILE%' -DestinationPath '%MODEL_DIR%' -Force"
if %errorlevel% neq 0 (
    echo Unzip failed.
    exit /b 1
)

echo Cleaning up...
del "%MODEL_DIR%\%ZIP_FILE%"

echo %MODEL_NAME% downloaded and extracted to %MODEL_DIR%