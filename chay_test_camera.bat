@echo off
chcp 65001 >nul
cd /d "%~dp0"
cls
echo ==========================================================
echo      CHUONG TRINH TEST CAMERA FACE ANTI-SPOOFING
echo      Model: liveness_cam4k.h5
echo ==========================================================
echo.
echo   [0] Chay ngay Camera USB SC-M200 (Mac dinh - Nhan Enter)
echo   [1] Chay Camera lien may (Laptop Built-in Webcam)
echo   [2] Chay giao dien Web tren trinh duyet (http://127.0.0.1:8080)
echo.
echo   * Meo: Khi cua so Camera dang mo, ban co the nhan phim [0], [1]
echo          hoac [C] de chuyen doi qua lai giua 2 camera bat cu luc nao!
echo ==========================================================
set /p choice="Nhap lua chon cua ban (Enter = 0): "

if "%choice%"=="1" (
    .\venv\Scripts\python.exe run.py --camera 1
) else if "%choice%"=="2" (
    start http://127.0.0.1:8080
    .\venv\Scripts\python.exe web_app.py
) else (
    .\venv\Scripts\python.exe run.py --camera 0
)
pause
