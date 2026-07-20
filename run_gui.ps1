# run_gui.ps1 - Launcher for RAEZ Hand Bone Tracker (FastAPI + Vite)
# Can be run from any directory

$ProjectRoot = "G:\RAEZ-Next\hand_bone_tracker"
$Python = "$ProjectRoot\.venv\Scripts\python.exe"
$Dashboard = "$ProjectRoot\gui\dashboard"

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "   RAEZ Hand Bone Tracker - GUI Launcher      " -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# Verification
if (-not (Test-Path $Python)) {
    Write-Host "[-] Python .venv not found at: $Python" -ForegroundColor Red
    Write-Host "    Please check your virtual environment path." -ForegroundColor Yellow
    pause
    exit 1
}

if (-not (Test-Path "$Dashboard\node_modules")) {
    Write-Host "[!] node_modules missing. Installing npm dependencies..." -ForegroundColor Yellow
    Set-Location $Dashboard
    npm install
}

Write-Host "[+] Starting FastAPI Backend (separate window)..." -ForegroundColor Green
Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", "Set-Location '$ProjectRoot'; Write-Host '🚀 Backend FastAPI' -ForegroundColor Cyan; & '$Python' gui\server.py"

Write-Host "[*] Waiting 2 seconds for backend to start..." -ForegroundColor Gray
Start-Sleep -Seconds 2

Write-Host "[+] Starting Vite Frontend..." -ForegroundColor Yellow
Set-Location $Dashboard
npm run dev
