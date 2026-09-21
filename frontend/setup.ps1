$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Set-Location -LiteralPath $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    $version = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($version -notin @("3.10", "3.11", "3.12")) {
        throw "Python $version is not supported by the face-recognition stack. Install Python 3.10-3.12 first."
    }
    python -m venv .venv
}

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r requirements.txt

if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
}

Write-Host "Setup complete. Start with: .\run.ps1" -ForegroundColor Green

