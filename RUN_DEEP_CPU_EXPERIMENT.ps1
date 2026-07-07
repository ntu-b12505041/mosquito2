param(
    [Parameter(Mandatory = $true)]
    [string]$RunName,

    [Parameter(Mandatory = $true)]
    [ValidateSet("all12", "wearable3_vdiff")]
    [string]$LeadMode,

    [Parameter(Mandatory = $true)]
    [ValidateSet("raw_zscore", "bandpass_0.5_40_zscore")]
    [string]$Preprocess,

    [Parameter(Mandatory = $true)]
    [ValidateSet("resnet1d", "inception1d", "spectrogram2d")]
    [string]$Model
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

. "$ScriptDir\LOCAL_CPU_SETTINGS.ps1"

if ($PTBXL_DATA_DIR -like "*<<<*") {
    throw "Please edit LOCAL_CPU_SETTINGS.ps1 first and replace <<<PTBXL_DATA_DIR>>> with your local PTB-XL path."
}

$ResolvedDataDir = (Resolve-Path -LiteralPath $PTBXL_DATA_DIR).Path
if ((Split-Path -Leaf $ResolvedDataDir).ToLower() -eq "records100") {
    $ResolvedDataDir = Split-Path -Parent $ResolvedDataDir
}

Write-Host "Run: $RunName"
Write-Host "Project dir: $ScriptDir"
Write-Host "PTB-XL data dir: $ResolvedDataDir"
Write-Host "Lead mode: $LeadMode"
Write-Host "Preprocess: $Preprocess"
Write-Host "Model: $Model"
Write-Host "Max records: $DEEP_MAX_RECORDS"
Write-Host "Epochs: $DEEP_EPOCHS"
Write-Host "Batch size: $DEEP_BATCH_SIZE"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating local virtual environment..."
    python -m venv .venv
}

$Python = Join-Path $ScriptDir ".venv\Scripts\python.exe"

Write-Host "Installing CPU requirements..."
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements_cpu.txt

Write-Host "Installing CPU PyTorch if needed..."
& $Python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

Write-Host "Checking local PTB-XL records100..."
& $Python scripts\check_local_ptbxl_data.py --data-dir "$ResolvedDataDir"

Write-Host "Running deep CPU experiment..."
& $Python scripts\run_deep_experiments.py `
    --data-dir "$ResolvedDataDir" `
    --no-download `
    --device cpu `
    --max-records $DEEP_MAX_RECORDS `
    --lead-mode $LeadMode `
    --epochs $DEEP_EPOCHS `
    --batch-size $DEEP_BATCH_SIZE `
    --preprocess $Preprocess `
    --models $Model

Write-Host "Generating report.md..."
& $Python scripts\make_report.py

Write-Host ""
Write-Host "Done: $RunName"
Write-Host "Metrics:"
Write-Host "  reports\metrics_deep.csv"
Write-Host "  reports\metrics_deep.json"
Write-Host "Report:"
Write-Host "  report.md"
