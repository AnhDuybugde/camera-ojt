[CmdletBinding()]
param(
    [ValidateSet("start", "stop", "status", "foreground")]
    [string]$Action = "start"
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $workspace "backend"
$python = Join-Path $backend ".venv\Scripts\python.exe"
$runtimeDir = Join-Path $workspace ".runtime"
$statePath = Join-Path $runtimeDir "processes.json"
$preferencePath = Join-Path $runtimeDir "be-xinh-chat.json"
$logDir = Join-Path $workspace "logs"

function Get-DotEnvValue {
    param([string]$Name, [string]$Default = "")
    $inherited = [Environment]::GetEnvironmentVariable($Name)
    if (-not [string]::IsNullOrWhiteSpace($inherited)) { return $inherited.Trim() }
    $line = Get-Content -LiteralPath (Join-Path $backend ".env") |
        Where-Object { $_ -match "^\s*$([regex]::Escape($Name))\s*=" } |
        Select-Object -Last 1
    if ($null -eq $line) { return $Default }
    return (($line -split '=', 2)[1]).Trim().Trim('"').Trim("'")
}

$realtimeMode = (Get-DotEnvValue "BE_XINH_REALTIME_MODE" "classic").ToLowerInvariant()
if ($realtimeMode -eq "live") {
    $script = Join-Path $backend "scripts\be_xinh_live_assistant.py"
    $chatModel = Get-DotEnvValue "GEMINI_LIVE_MODEL" "gemini-3.8-live"
}
elseif ($realtimeMode -eq "classic") {
    $script = Join-Path $backend "scripts\be_xinh_assistant.py"
    $chatModel = Get-DotEnvValue "GEMINI_MODEL" "gemini-3.5-flash-lite"
}
else {
    throw "Unsupported BE_XINH_REALTIME_MODE=$realtimeMode"
}

function Read-State {
    if (Test-Path -LiteralPath $statePath) {
        return Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    }
    return [pscustomobject]@{ backend = 0; audio = 0; chat = 0; frontend = 0 }
}

function Test-ChatProcess([int]$ProcessId) {
    if ($ProcessId -le 0) { return $false }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction SilentlyContinue
    return $null -ne $process -and (
        $process.CommandLine -like "*be_xinh_assistant.py*" -or
        $process.CommandLine -like "*be_xinh_live_assistant.py*"
    )
}

function Write-ChatPid($state, [int]$ProcessId) {
    if ($null -eq $state.PSObject.Properties["chat"]) {
        $state | Add-Member -NotePropertyName chat -NotePropertyValue $ProcessId
    }
    else {
        $state.chat = $ProcessId
    }
    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
    $state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
}

function Write-ChatPreference([bool]$Enabled) {
    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
    @{
        enabled = $Enabled
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $preferencePath -Encoding utf8
}

$state = Read-State
$chatPid = if ($null -ne $state.PSObject.Properties["chat"]) { [int]$state.chat } else { 0 }
$running = Test-ChatProcess $chatPid

if ($Action -eq "status") {
    Write-Host "Bé Xinh voice chat: $(if ($running) { "running (PID $chatPid)" } else { 'stopped' })"
    exit 0
}

if ($Action -eq "stop") {
    if ($running) {
        taskkill.exe /PID $chatPid /T /F | Out-Null
    }
    Write-ChatPid $state 0
    Write-ChatPreference $false
    Write-Host "Bé Xinh voice chat stopped."
    exit 0
}

if ($running) {
    Write-ChatPreference $true
    Write-Host "Bé Xinh voice chat is already running (PID $chatPid)."
    exit 0
}
foreach ($required in @($python, $script, (Join-Path $backend ".env"))) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing required file: $required"
    }
}
$geminiLine = Get-Content -LiteralPath (Join-Path $backend ".env") |
    Where-Object { $_ -match '^\s*GEMINI_API_KEY\s*=' } |
    Select-Object -Last 1
$geminiValue = if ($null -ne $geminiLine) {
    (($geminiLine -split '=', 2)[1]).Trim().Trim('"').Trim("'")
} else { "" }
if ($geminiValue.Length -le 10 -or $geminiValue -match '^your_') {
    throw "Missing GEMINI_API_KEY in backend\.env. Add it locally, then run .\chat.ps1 start."
}

# Voice chat owns the camera speaker. Stop the automatic greeting bridge first
# so it cannot interrupt a question or make the microphone hear repeated hellos.
$audioPid = if ($null -ne $state.PSObject.Properties["audio"]) { [int]$state.audio } else { 0 }
if ($audioPid -gt 0) {
    $audioProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$audioPid" -ErrorAction SilentlyContinue
    if ($null -ne $audioProcess -and $audioProcess.CommandLine -like "*run_be_xinh_bridge.py*") {
        taskkill.exe /PID $audioPid /T /F | Out-Null
    }
}
if ($null -eq $state.PSObject.Properties["audio"]) {
    $state | Add-Member -NotePropertyName audio -NotePropertyValue 0
}
else {
    $state.audio = 0
}
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$preferencePath = Join-Path $runtimeDir "be-xinh.json"
@{
    enabled = $false
    updated_at = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -LiteralPath $preferencePath -Encoding utf8

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
if ($Action -eq "foreground") {
    Write-Host "Bé Xinh is running in the foreground. Press Ctrl+C to stop immediately."
    $process = Start-Process `
        -FilePath $python `
        -ArgumentList @("-X", "utf8", "-u", $script, "--model", $chatModel) `
        -WorkingDirectory $backend `
        -NoNewWindow `
        -PassThru
    Write-ChatPid $state $process.Id
    Write-ChatPreference $true
    try {
        Wait-Process -Id $process.Id
    }
    finally {
        $current = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
        if ($null -ne $current) {
            taskkill.exe /PID $process.Id /T /F | Out-Null
        }
        $latestState = Read-State
        Write-ChatPid $latestState 0
        Write-ChatPreference $false
        Write-Host "Bé Xinh voice chat stopped."
    }
    exit 0
}

$process = Start-Process `
    -FilePath $python `
    -ArgumentList @("-X", "utf8", "-u", $script, "--model", $chatModel) `
    -WorkingDirectory $backend `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logDir "be-xinh-chat.out.log") `
    -RedirectStandardError (Join-Path $logDir "be-xinh-chat.err.log") `
    -PassThru
Write-ChatPid $state $process.Id
Write-ChatPreference $true
Write-Host "Stop background mode: .\chat.ps1 stop"
Write-Host "For Ctrl+C mode: .\chat.ps1 foreground"
Write-Host "Bé Xinh voice chat started (PID $($process.Id))."
Write-Host "Say: Bé Xinh ơi"
