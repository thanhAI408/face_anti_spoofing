@echo off
title He Thong Cham Cong FaceID - Liveness Cam4K
cd /d "%~dp0ChamCong_FaceID"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Khong tim thay moi truong ao tai: %~dp0ChamCong_FaceID\.venv
    echo Vui long kiem tra lai du an.
    pause
    exit /b 1
)

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

cls
echo ==============================================================================
echo       HE THONG CHAM CONG FACEID VA CHONG GIA MAO (LIVENESS CAM4K)
echo ==============================================================================
echo.
echo [INFO] Dang mo trinh duyet toi http://localhost:8000 ...
start "" "http://localhost:8000"

echo [INFO] Dang khoi dong Web Server tren port 8000...
echo Nhan Ctrl + C de dung server.
echo.
.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
pause
