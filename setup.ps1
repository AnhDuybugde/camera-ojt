[CmdletBinding()]
param(
    [string]$PythonVersion = "auto",
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $workspace "backend"
$frontend = Join-Path $workspace "frontend"
$backendPython = Join-Path $backend ".venv\Scripts\python.exe"
$frontendPython = Join-Path $frontend ".venv\Scripts\python.exe"

function Resolve-CompatiblePython {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    $requestedVersions = if ($PythonVersion -and $PythonVersion -ne "auto") {
        @($PythonVersion)
    }
    else {
        @("3.12", "3.11", "3.10")
    }

    if ($null -ne $launcher) {
        foreach ($version in $requestedVersions) {
            & $launcher.Source "-$version" -c "import sys; print(sys.executable)" *> $null
            if ($LASTEXITCODE -eq 0) {
                return @{
                    File = $launcher.Source
                    PrefixArgs = @("-$version")
                    Label = "Python $version via py launcher"
                }
            }
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        $versionText = & $python.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($LASTEXITCODE -eq 0 -and $versionText -in @("3.10", "3.11", "3.12")) {
            return @{
                File = $python.Source
                PrefixArgs = @()
                Label = "Python $versionText at $($python.Source)"
            }
        }
    }

    $detected = if ($null -ne $launcher) {
        (& $launcher.Source -0p 2>&1 | Out-String).Trim()
    }
    else {
        "Python launcher 'py' is not installed."
    }
    throw @"
No compatible Python runtime was found. This project requires Python 3.10-3.12.
Detected runtimes:
$detected

Install Python 3.12, or run with an installed supported version:
  .\setup.ps1 -PythonVersion 3.12
"@
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$Description
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Test-PythonCommand {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$Code
    )
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonPath -c $Code *> $null
    $commandExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    return ($commandExitCode -eq 0)
}

function Test-PythonScript {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$ScriptPath
    )
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonPath $ScriptPath *> $null
    $commandExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    return ($commandExitCode -eq 0)
}

$runtime = Resolve-CompatiblePython
Write-Host "Using $($runtime.Label)"
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$useNvidia = (-not $CpuOnly) -and ($null -ne $nvidiaSmi)
if ($useNvidia) {
    Write-Host "NVIDIA GPU detected: backend will use CUDA 12.8 packages."
}
else {
    Write-Host "CUDA setup disabled: backend will use CPU packages."
}

Write-Host "[1/4] Creating isolated backend environment..."
if (-not (Test-Path -LiteralPath $backendPython)) {
    $venvArgs = @($runtime.PrefixArgs) + @("-m", "venv", (Join-Path $backend ".venv"))
    Invoke-Checked -FilePath $runtime.File -ArgumentList $venvArgs -Description "Backend venv creation"
}
if (-not (Test-Path -LiteralPath $backendPython)) {
    throw "Backend venv was not created at $backendPython"
}
$packagingArgs = @("-m", "pip", "install", "--upgrade", "pip", "wheel")
$packagingArgs += if ($useNvidia) { "setuptools<82" } else { "setuptools" }
Invoke-Checked -FilePath $backendPython `
    -ArgumentList $packagingArgs `
    -Description "Backend packaging tools installation"

if ($useNvidia) {
    $cudaTorchReady = Test-PythonCommand -PythonPath $backendPython `
        -Code "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    if (-not $cudaTorchReady) {
        Write-Host "Installing CUDA-enabled PyTorch for the RTX GPU..."
        Invoke-Checked -FilePath $backendPython `
            -ArgumentList @(
                "-m", "pip", "install", "--upgrade", "--force-reinstall",
                "torch==2.11.0", "torchvision==0.26.0",
                "--index-url", "https://download.pytorch.org/whl/cu128"
            ) `
            -Description "CUDA PyTorch installation"
    }

    $cudaOnnxReady = Test-PythonScript -PythonPath $backendPython `
        -ScriptPath (Join-Path $backend "scripts\check_gpu_runtime.py")
    if (-not $cudaOnnxReady) {
        # The CPU and GPU distributions expose the same Python package. Remove
        # both before installing the selected extra to avoid mixed DLLs.
        # PowerShell 5 surfaces harmless pip warnings on stderr as terminating
        # NativeCommandError records when ErrorActionPreference is Stop.
        $previousPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $backendPython -m pip uninstall -y onnxruntime onnxruntime-gpu *> $null
        $uninstallExitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousPreference
        if ($uninstallExitCode -ne 0) {
            throw "ONNX Runtime cleanup failed with exit code $uninstallExitCode."
        }
    }
}
Push-Location $backend
try {
    $backendExtra = if ($useNvidia) {
        ".[face-gpu,store,reid,voice,dev]"
    }
    else {
        ".[face,store,reid,voice,dev]"
    }
    Invoke-Checked -FilePath $backendPython `
        -ArgumentList @("-m", "pip", "install", "--upgrade", "-e", $backendExtra) `
        -Description "Backend dependencies installation"
}
finally {
    Pop-Location
}

Write-Host "Installing Vietnamese Zipformer INT8 speech model..."
Invoke-Checked -FilePath $backendPython `
    -ArgumentList @((Join-Path $backend "scripts\install_zipformer_model.py")) `
    -Description "Zipformer Vietnamese model installation"

if ($useNvidia) {
    # faster-whisper declares the CPU distribution name ``onnxruntime`` even
    # though the GPU wheel exposes the same module. Pip may therefore install
    # both and let the CPU files win. Reinstall the GPU wheel last, without
    # touching dependencies, then verify the provider contract explicitly.
    $cudaOnnxReady = Test-PythonScript -PythonPath $backendPython `
        -ScriptPath (Join-Path $backend "scripts\check_gpu_runtime.py")
    if (-not $cudaOnnxReady) {
        Write-Host "Making ONNX Runtime GPU the active provider..."
        Invoke-Checked -FilePath $backendPython `
            -ArgumentList @(
                "-m", "pip", "install", "--force-reinstall", "--no-deps",
                "onnxruntime-gpu==1.26.0"
            ) `
            -Description "ONNX Runtime GPU activation"
    }
    $cudaOnnxReady = Test-PythonScript -PythonPath $backendPython `
        -ScriptPath (Join-Path $backend "scripts\check_gpu_runtime.py")
    if (-not $cudaOnnxReady) {
        throw "CUDAExecutionProvider could not execute an inference after ONNX Runtime GPU installation."
    }
    Invoke-Checked -FilePath $backendPython `
        -ArgumentList @((Join-Path $backend "scripts\check_gpu_runtime.py")) `
        -Description "CUDA runtime verification"
}

Write-Host "[2/4] Creating isolated Streamlit environment..."
if (-not (Test-Path -LiteralPath $frontendPython)) {
    $venvArgs = @($runtime.PrefixArgs) + @("-m", "venv", (Join-Path $frontend ".venv"))
    Invoke-Checked -FilePath $runtime.File -ArgumentList $venvArgs -Description "Frontend venv creation"
}
if (-not (Test-Path -LiteralPath $frontendPython)) {
    throw "Frontend venv was not created at $frontendPython"
}
Invoke-Checked -FilePath $frontendPython `
    -ArgumentList @("-m", "pip", "install", "--upgrade", "pip") `
    -Description "Frontend pip installation"
Push-Location $frontend
try {
    Invoke-Checked -FilePath $frontendPython `
        -ArgumentList @("-m", "pip", "install", "-r", "requirements.txt") `
        -Description "Frontend dependencies installation"
}
finally {
    Pop-Location
}

Write-Host "[3/4] Creating local configuration files..."
$backendEnv = Join-Path $backend ".env"
if (-not (Test-Path -LiteralPath $backendEnv)) {
    Copy-Item -LiteralPath (Join-Path $backend ".env.example") -Destination $backendEnv
}
$frontendEnv = Join-Path $frontend ".env"
if (-not (Test-Path -LiteralPath $frontendEnv)) {
    Copy-Item -LiteralPath (Join-Path $frontend ".env.example") -Destination $frontendEnv
}

Write-Host "[4/4] Done. Fill backend\.env with camera/audio credentials, then run .\run.ps1"
