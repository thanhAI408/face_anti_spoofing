@echo off
chcp 65001 >nul
title He Thong Cham Cong FaceID - Liveness Cam4K
echo ==============================================================================
echo       KHOI DONG HE THONG CHAM CONG FACEID CO CHONG GIA MAO (LIVENESS)
echo ==============================================================================
echo.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Chua co moi truong ao .venv. Dang tu dong khoi tao va cai thu vien...
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
    if not exist "face_prototypes.npz" (
        .venv\Scripts\python.exe init_demo_data.py
    )
)

:: Giai phong port 8000 neu co tien trinh cu dang chay ngam
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

echo [INFO] Dang mo trinh duyet toi http://localhost:8000 ...
start "" "http://localhost:8000"

echo [INFO] Dang khoi dong Web Server uvicorn tren port 8000...
echo Nhan Ctrl + C de dung server.
echo.
.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
pause
