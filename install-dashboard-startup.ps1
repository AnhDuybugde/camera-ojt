[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$launcher = Join-Path $workspace "TeamIntegrationDashboard.cmd"
$startup = [Environment]::GetFolderPath("Startup")
if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Missing startup launcher: $launcher"
}
if (-not $startup) {
    throw "Windows Startup folder is unavailable."
}
Copy-Item -LiteralPath $launcher -Destination (Join-Path $startup "TeamIntegrationDashboard.cmd") -Force
Write-Host "Installed dashboard startup launcher for the current Windows user."
