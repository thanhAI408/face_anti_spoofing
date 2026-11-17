@echo off
title He Thong Cham Cong FaceID - Liveness Cam4K
cd /d "%~dp0ChamCong_FaceID"

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Chua co moi truong ao .venv trong ChamCong_FaceID. Dang tu dong khoi tao...
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
    if not exist "face_prototypes.npz" (
        .venv\Scripts\python.exe init_demo_data.py
    )
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
