[CmdletBinding()]
param(
    [int]$UiPort = 8501,
    [int]$BackendPort = 8765
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontend = Join-Path $workspace "frontend"
$python = Join-Path $frontend ".venv\Scripts\python.exe"
$runtimeDir = Join-Path $workspace ".runtime"
$statePath = Join-Path $runtimeDir "processes.json"
$logDir = Join-Path $workspace "logs"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing frontend environment: $python"
}
New-Item -ItemType Directory -Force -Path $runtimeDir, $logDir | Out-Null

$state = if (Test-Path -LiteralPath $statePath) {
    Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
} else {
    [pscustomobject]@{ backend = 0; audio = 0; chat = 0; frontend = 0 }
}
$frontendPid = if ($null -ne $state.PSObject.Properties['frontend']) { [int]$state.frontend } else { 0 }
$existing = if ($frontendPid -gt 0) {
    Get-CimInstance Win32_Process -Filter "ProcessId=$frontendPid" -ErrorAction SilentlyContinue
} else { $null }

$lanIp = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } |
    Sort-Object InterfaceMetric |
    Select-Object -First 1 -ExpandProperty IPAddress
if (-not $lanIp) { $lanIp = "127.0.0.1" }

if ($null -eq $existing -or $existing.CommandLine -notlike '*streamlit*app.py*') {
    $env:PUBLIC_TRACKING_BACKEND_URL = "http://${lanIp}:$BackendPort"
    $process = Start-Process `
        -FilePath $python `
        -ArgumentList @(
            "-X", "utf8", "-u", "-m", "streamlit", "run", "app.py",
            "--server.port", "$UiPort",
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ) `
        -WorkingDirectory $frontend `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDir "frontend.out.log") `
        -RedirectStandardError (Join-Path $logDir "frontend.err.log") `
        -PassThru
    if ($null -eq $state.PSObject.Properties['frontend']) {
        $state | Add-Member -NotePropertyName frontend -NotePropertyValue $process.Id
    } else { $state.frontend = $process.Id }
    $state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
    Write-Host "Dashboard started (PID $($process.Id))."
} else {
    Write-Host "Dashboard is already running (PID $frontendPid)."
}

Write-Host "This PC: http://127.0.0.1:$UiPort"
Write-Host "Office LAN: http://${lanIp}:$UiPort"
Write-Host "Camera AI and Be Xinh can now be controlled from the Live Attendance page."
