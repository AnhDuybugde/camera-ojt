[CmdletBinding()]
param(
    [string]$StreamHost = "127.0.0.1",
    [int]$StreamPort = 8765,
    [int]$UiPort = 8501,
    [string]$UiHost = "127.0.0.1"
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

function Wait-ServiceEndpoint {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($Process.HasExited) {
            return $false
        }
        try {
            $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
                return $true
            }
        }
        catch {
            # Model loading and camera connection are allowed to take time.
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Stop-StartedProcess {
    param([System.Diagnostics.Process]$Process)
    if ($null -ne $Process -and -not $Process.HasExited) {
        Stop-Process -Id $Process.Id -ErrorAction SilentlyContinue
    }
}

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

if (-not (Wait-ServiceEndpoint `
    -Uri "http://127.0.0.1:$StreamPort/healthz" `
    -Process $backendProcess `
    -TimeoutSeconds 180)) {
    Stop-StartedProcess -Process $backendProcess
    throw "Backend failed to become healthy. Check $logDir\backend.err.log and backend.out.log."
}

$audioEnabled = $true
$audioSuppressedByChat = $false
$audioControlPath = Join-Path $runtimeDir "be-xinh.json"
if (Test-Path -LiteralPath $audioControlPath) {
    try {
        $audioControl = Get-Content -LiteralPath $audioControlPath -Raw | ConvertFrom-Json
        $audioEnabled = [bool]$audioControl.enabled
    }
    catch {
        Write-Warning "Cannot read $audioControlPath; Be Xinh will use the default enabled state."
    }
}

$geminiKeyConfigured = $false
$geminiLine = Get-Content -LiteralPath (Join-Path $backend ".env") |
    Where-Object { $_ -match '^\s*GEMINI_API_KEY\s*=' } |
    Select-Object -Last 1
if ($null -ne $geminiLine) {
    $geminiValue = (($geminiLine -split '=', 2)[1]).Trim().Trim('"').Trim("'")
    $geminiKeyConfigured = $geminiValue.Length -gt 10 -and $geminiValue -notmatch '^your_'
}
$chatEnabled = $false
$chatPreferencePath = Join-Path $runtimeDir "be-xinh-chat.json"
if (Test-Path -LiteralPath $chatPreferencePath) {
    try {
        $chatPreference = Get-Content -LiteralPath $chatPreferencePath -Raw | ConvertFrom-Json
        $chatEnabled = [bool]$chatPreference.enabled
    }
    catch {
        Write-Warning "Cannot read $chatPreferencePath; voice chat stays disabled."
    }
}
if ($geminiKeyConfigured -and $chatEnabled -and $audioEnabled) {
    $audioEnabled = $false
    $audioSuppressedByChat = $true
}

$audioProcess = $null
if ($audioEnabled) {
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
}

$chatProcess = $null
if ($geminiKeyConfigured -and $chatEnabled) {
    $chatProcess = Start-Process `
        -FilePath $backendPython `
        -ArgumentList @("-X", "utf8", "-u", "scripts/be_xinh_assistant.py", "--model", "gemini-3.5-flash-lite") `
        -WorkingDirectory $backend `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDir "be-xinh-chat.out.log") `
        -RedirectStandardError (Join-Path $logDir "be-xinh-chat.err.log") `
        -PassThru
}

$uiProcess = Start-Process `
    -FilePath $frontendPython `
    -ArgumentList @(
        "-X", "utf8",
        "-u",
        "-m", "streamlit", "run", "app.py",
        "--server.port", "$UiPort",
        "--server.address", $UiHost,
        "--server.headless", "true"
    ) `
    -WorkingDirectory $frontend `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logDir "frontend.out.log") `
    -RedirectStandardError (Join-Path $logDir "frontend.err.log") `
    -PassThru

if (-not (Wait-ServiceEndpoint `
    -Uri "http://127.0.0.1:$UiPort/_stcore/health" `
    -Process $uiProcess `
    -TimeoutSeconds 60)) {
    Stop-StartedProcess -Process $uiProcess
    Stop-StartedProcess -Process $chatProcess
    Stop-StartedProcess -Process $audioProcess
    Stop-StartedProcess -Process $backendProcess
    throw "Frontend failed to become healthy. Check $logDir\frontend.err.log."
}

@{
    started_at = (Get-Date).ToString("o")
    backend = $backendProcess.Id
    audio = if ($null -ne $audioProcess) { $audioProcess.Id } else { 0 }
    chat = if ($null -ne $chatProcess) { $chatProcess.Id } else { 0 }
    frontend = $uiProcess.Id
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimeDir "processes.json") -Encoding UTF8

Write-Host "Integrated system started."
Write-Host "UI:      http://$UiHost`:$UiPort"
Write-Host "Model:   http://127.0.0.1:$StreamPort/status.json"
Write-Host "Health:  http://127.0.0.1:$StreamPort/readyz"
Write-Host "Be Xinh greeting: $(if ($audioEnabled) { 'enabled' } elseif ($audioSuppressedByChat) { 'disabled while voice chat is active' } else { 'disabled by dashboard' })"
Write-Host "Voice chat: $(if ($null -ne $chatProcess) { 'enabled (Gemini 3.5 Flash)' } elseif (-not $geminiKeyConfigured) { 'disabled - add GEMINI_API_KEY' } else { 'disabled by dashboard' })"
Write-Host "Logs:    $logDir"
Write-Host "Stop:    .\stop.ps1"
