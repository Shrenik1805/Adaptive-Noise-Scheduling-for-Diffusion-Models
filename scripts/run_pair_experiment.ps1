Param(
    [string]$PythonExe = "python",
    [string]$Device = "auto",
    [int]$Epochs = 100,
    [int]$BatchSize = 128,
    [double]$LearningRate = 2e-4,
    [string]$WandbMode = "online",
    [int]$NumFidSamples = 10000,
    [int]$SamplesPerClass = 1000,
    [int]$Repeats = 3,
    [string]$RunTag = "default",
    [int]$Seed = 42,
    [int]$NumWorkers = 2,
    [int]$ValidateEvery = 5,
    [int]$ValidateBatches = 50
)

$ErrorActionPreference = "Stop"

$adaptiveCkptDir = "./checkpoints_adaptive_$RunTag"
$fixedCkptDir = "./checkpoints_fixed_$RunTag"
$adaptiveSampleDir = "./samples_adaptive_$RunTag"
$fixedSampleDir = "./samples_fixed_$RunTag"
$analysisDir = "./adaptive_diffusion/analysis_$RunTag"

$env:WANDB_MODE = $WandbMode

Write-Host "Training adaptive model..."
& $PythonExe -m adaptive_diffusion.train --schedule-mode adaptive --device $Device --epochs $Epochs --batch-size $BatchSize --lr $LearningRate --seed $Seed --num-workers $NumWorkers --validate-every $ValidateEvery --validate-batches $ValidateBatches --checkpoint-dir $adaptiveCkptDir --sample-dir $adaptiveSampleDir

Write-Host "Training fixed-cosine baseline..."
& $PythonExe -m adaptive_diffusion.train --schedule-mode fixed_cosine --device $Device --epochs $Epochs --batch-size $BatchSize --lr $LearningRate --seed $Seed --num-workers $NumWorkers --validate-every $ValidateEvery --validate-batches $ValidateBatches --checkpoint-dir $fixedCkptDir --sample-dir $fixedSampleDir

function Find-BestCheckpoint([string]$CheckpointDir) {
    $pattern = 'epoch_\d+_fid_([0-9]+(?:\.[0-9]+)?)\.pt$'
    $bestPath = $null
    $bestFid = [double]::PositiveInfinity
    Get-ChildItem -Path $CheckpointDir -Filter "epoch_*_fid_*.pt" | ForEach-Object {
        if ($_.Name -match $pattern) {
            $fid = [double]$Matches[1]
            if ($fid -lt $bestFid) {
                $bestFid = $fid
                $bestPath = $_.FullName
            }
        }
    }
    if (-not $bestPath) {
        throw "No checkpoint found in $CheckpointDir"
    }
    return $bestPath
}

$adaptiveBest = Find-BestCheckpoint -CheckpointDir $adaptiveCkptDir
$fixedBest = Find-BestCheckpoint -CheckpointDir $fixedCkptDir

Write-Host "Best adaptive checkpoint: $adaptiveBest"
Write-Host "Best fixed checkpoint: $fixedBest"

Write-Host "Running paired evaluation..."
& $PythonExe -m adaptive_diffusion.evaluate --adaptive-checkpoint $adaptiveBest --fixed-checkpoint $fixedBest --device $Device --output-dir $analysisDir --num-fid-samples $NumFidSamples --samples-per-class $SamplesPerClass --repeats $Repeats

Write-Host "Generating summary..."
& $PythonExe scripts/summarize_results.py --analysis-dir $analysisDir --adaptive-checkpoint $adaptiveBest --fixed-checkpoint $fixedBest

Write-Host "Done. Artifacts in $analysisDir"
