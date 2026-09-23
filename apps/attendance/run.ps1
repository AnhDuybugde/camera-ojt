$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing .venv. Create a Python 3.10-3.12 virtual environment and install requirements first."
}

Set-Location -LiteralPath $PSScriptRoot
& $python -m streamlit run app.py
