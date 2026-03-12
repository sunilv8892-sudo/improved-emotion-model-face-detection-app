$ErrorActionPreference = 'Stop'

param(
    [Parameter(Mandatory = $true)]
    [string]$CsvPath,

    [string]$LabelColumn = 'emotion_label',
    [string]$ImageColumn = 'image_name',
    [switch]$ExportTflite
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvDir = Join-Path $repoRoot '.venv-emotion'
$requirements = Join-Path $repoRoot 'training\requirements.txt'
$trainer = Join-Path $repoRoot 'training\train_emotion_model.py'
$modelsDir = Join-Path $repoRoot 'models'

if (-not (Test-Path $venvDir)) {
    python -m venv $venvDir
}

$pythonExe = Join-Path $venvDir 'Scripts\python.exe'

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r $requirements

$args = @(
    $trainer,
    $CsvPath,
    '--output-dir', $modelsDir,
    '--label-column', $LabelColumn,
    '--image-column', $ImageColumn
)

if ($ExportTflite) {
    $args += '--export-tflite'
}

& $pythonExe @args

Write-Host 'Training/export completed. Run scripts/copy_emotion_assets.ps1 next.'