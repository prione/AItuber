@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

if exist "%~dp0Temp" (
    rd /s /q "%~dp0Temp"
)
md "%~dp0Temp"

call conda activate vtuber
cd %~dp0

set /p choice="Enter 1 (Debug), 2 (OBS), 3 (Only Language) or 4 (Only Avator):"

if "%choice%"=="1" (
    REM EasyVtuberをデバッグモードで実行
    start cmd /k "cd ./Language && python main.py --debug --voice"
    start cmd /k "cd ./Avator && python main.py --debug --character 1 --output_size 512x512"
    cmd /k "python ./VOICEVOX/run.py --voicevox_dir="./VOICEVOX/core" --use_gpu"

) else if "%choice%"=="2" (
    REM EasyVtuberをOBSモードで実行
        start cmd /k "cd ./Avator && python main.py --output_webcam obs --character 1 --output_size 512x512"
        set /p video_id="video_id:"
    start cmd /k "cd ./Language && python main.py --voice --load_memory --video_id !video_id!"
    REM start cmd /k "cd ./Language && python comment.py --video_id !video_id!"
    cmd /k "python ./VOICEVOX/run.py --voicevox_dir="./VOICEVOX/core" --use_gpu"
) else if "%choice%"=="3" (
    REM Languageモジュールだけ実行
    cmd /k "cd ./Language && python main.py"

) else if "%choice%"=="4" (
    REM Avatorモジュールだけ実行
    cmd /k "cd ./Avator && python main.py --output_webcam obs --character 1"

) else (
    echo Invalid choice
)