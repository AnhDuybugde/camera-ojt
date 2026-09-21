[CmdletBinding()]
param(
    [string]$StreamHost = "127.0.0.1",
    [int]$StreamPort = 8765,
    [int]$UiPort = 8501
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $workspace "backend"
$frontend = Join-Path $workspace "frontend"
$backendPython = Join-Path $backend ".venv\Scripts\python.exe"
$frontendPython = Join-Path $frontend ".venv\Scripts\python.exe"
$runtimeDir = Join-Path $workspace ".runtime"
$logDir = Join-Path $workspace "logs"

foreach ($required in @($backendPython, $frontendPython)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing environment: $required. Run .\setup.ps1 first."
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $backend ".env"))) {
    throw "Missing backend\.env. Run .\setup.ps1 and fill camera/audio credentials."
}

New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$backendProcess = Start-Process `
    -FilePath $backendPython `
    -ArgumentList @(
        "-X", "utf8",
        "-u",
        "scripts/run_workstate.py",
        "--stream-host", $StreamHost,
        "--stream-port", "$StreamPort",
        "--no-greet",
        "--no-halinh",
        "--no-supervisor"
    ) `
    -WorkingDirectory $backend `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logDir "backend.out.log") `
    -RedirectStandardError (Join-Path $logDir "backend.err.log") `
    -PassThru

Start-Sleep -Seconds 2

$audioProcess = Start-Process `
    -FilePath $backendPython `
    -ArgumentList @(
        "-X", "utf8",
        "-u",
        "scripts/run_be_xinh_bridge.py",
        "--status-url", "http://127.0.0.1:$StreamPort/status.json"
    ) `
    -WorkingDirectory $backend `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logDir "be-xinh.out.log") `
    -RedirectStandardError (Join-Path $logDir "be-xinh.err.log") `
    -PassThru

$uiProcess = Start-Process `
    -FilePath $frontendPython `
    -ArgumentList @(
        "-X", "utf8",
        "-u",
        "-m", "streamlit", "run", "app.py",
        "--server.port", "$UiPort",
        "--server.headless", "true"
    ) `
    -WorkingDirectory $frontend `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logDir "frontend.out.log") `
    -RedirectStandardError (Join-Path $logDir "frontend.err.log") `
    -PassThru

@{
    started_at = (Get-Date).ToString("o")
    backend = $backendProcess.Id
    audio = $audioProcess.Id
    frontend = $uiProcess.Id
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimeDir "processes.json") -Encoding UTF8

Write-Host "Integrated system started."
Write-Host "UI:      http://127.0.0.1:$UiPort"
Write-Host "Model:   http://127.0.0.1:$StreamPort/status.json"
Write-Host "Logs:    $logDir"
Write-Host "Stop:    .\stop.ps1"
