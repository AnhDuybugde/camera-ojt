[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$statePath = Join-Path $workspace ".runtime\processes.json"
if (-not (Test-Path -LiteralPath $statePath)) {
    Write-Host "No launcher state found."
    exit 0
}

$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
foreach ($name in @("frontend", "chat", "audio", "backend")) {
    if ($null -eq $state.PSObject.Properties[$name]) {
        continue
    }
    $processId = [int]$state.$name
    if ($processId -le 0) {
        continue
    }
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Stop-Process -Id $processId
        Write-Host "Stopped $name (PID $processId)."
    }
}
