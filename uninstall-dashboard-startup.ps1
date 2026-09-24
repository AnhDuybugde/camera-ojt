[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$target = Join-Path ([Environment]::GetFolderPath("Startup")) "TeamIntegrationDashboard.cmd"
if (Test-Path -LiteralPath $target) {
    Remove-Item -LiteralPath $target -Force
    Write-Host "Removed dashboard startup launcher."
} else {
    Write-Host "Dashboard startup launcher is not installed."
}
