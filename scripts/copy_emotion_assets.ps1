$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$modelsDir = Join-Path $repoRoot 'models'
$assetsDir = Join-Path $repoRoot 'assets\models'

$requiredFiles = @(
    'emotion_runtime_params.json',
    'efficientnet_feature_extractor.tflite'
)

foreach ($file in $requiredFiles) {
    $source = Join-Path $modelsDir $file
    if (-not (Test-Path $source)) {
        throw "Missing generated file: $source"
    }
}

foreach ($file in $requiredFiles) {
    Copy-Item (Join-Path $modelsDir $file) (Join-Path $assetsDir $file) -Force
}

Write-Host 'Emotion assets copied to assets/models successfully.'