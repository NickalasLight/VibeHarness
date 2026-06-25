# Run a single karriere.at job application and update multi_job_state.json
# Usage: powershell -File run_karriere_job.ps1 -JobIndex <1-9>

param(
    [Parameter(Mandatory=$true)]
    [int]$JobIndex
)

$ErrorActionPreference = "Stop"
$Root = "C:\git\vibethinkharnessProto1"
$LogFile = "$Root\vibe_karriere_job${JobIndex}.log"
$ErrFile = "$Root\vibe_karriere_job${JobIndex}_err.log"
$TaskFile = "$Root\task_karriere_job${JobIndex}.txt"

if (-not (Test-Path $TaskFile)) {
    Write-Error "Task file not found: $TaskFile"
    exit 1
}

# Update state: mark as running
$state = Get-Content "$Root\multi_job_state.json" | ConvertFrom-Json
$job = $state.jobs[$JobIndex - 1]
$job.status = "running"
$job.current_log = $LogFile
$state | ConvertTo-Json -Depth 10 | Set-Content "$Root\multi_job_state.json"

Write-Host "=== Starting job $JobIndex: $($job.url) ===" -ForegroundColor Cyan
Write-Host "Log: $LogFile"

# Evict models from VRAM before starting
try {
    Invoke-RestMethod -Uri "http://localhost:11434/api/generate" `
        -Method POST -ContentType "application/json" `
        -Body '{"model":"vibethinker:latest","keep_alive":0,"prompt":""}' `
        -TimeoutSec 15 | Out-Null
} catch {}
try {
    Invoke-RestMethod -Uri "http://localhost:11434/api/generate" `
        -Method POST -ContentType "application/json" `
        -Body '{"model":"qwen3:4b","keep_alive":0,"prompt":""}' `
        -TimeoutSec 15 | Out-Null
} catch {}

Start-Sleep -Seconds 3

# Run vibe agent
Write-Host "Running vibe agent..."
$proc = Start-Process -FilePath "powershell" `
    -ArgumentList "-Command", "cd '$Root'; python -m vibeharness --agent web --task-file '$TaskFile' --max-steps 60 2>&1 | Tee-Object -FilePath '$LogFile'" `
    -PassThru -NoNewWindow -Wait

$exitCode = $proc.ExitCode
Write-Host "Vibe process exited with code: $exitCode"

# Analyze log
$analysis = python "$Root\analyze_job_log.py" $LogFile | ConvertFrom-Json

Write-Host "Analysis: escalations=$($analysis.escalations) validator_saves=$($analysis.validator_saves) cost=$($analysis.estimated_cost_usd) completed=$($analysis.completed)"

# Update state with results
$state = Get-Content "$Root\multi_job_state.json" | ConvertFrom-Json
$job = $state.jobs[$JobIndex - 1]
$job.iterations = $job.iterations + 1
$job.escalations = $analysis.escalations
$job.validator_saves = $analysis.validator_saves
$job.estimated_cost_usd = $analysis.estimated_cost_usd

if ($analysis.completed) {
    $job.status = "completed"
    $job.completed_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    $job.notes = "Application submitted successfully"
} elseif ($analysis.blocked) {
    $job.status = "blocked"
    $job.notes = "Blocked: phone verification or login required"
} else {
    $job.status = "failed"
    $job.notes = "Did not reach confirmation page"
}

# Update global metrics
$state.metrics_summary.total_escalations = ($state.jobs | Measure-Object -Property escalations -Sum).Sum
$state.metrics_summary.total_validator_saves = ($state.jobs | Measure-Object -Property validator_saves -Sum).Sum
$state.metrics_summary.total_estimated_cost_usd = ($state.jobs | Measure-Object -Property estimated_cost_usd -Sum).Sum
$state.metrics_summary.jobs_completed = ($state.jobs | Where-Object { $_.status -eq "completed" }).Count
$state.current_job_index = $JobIndex

$state | ConvertTo-Json -Depth 10 | Set-Content "$Root\multi_job_state.json"
Write-Host "State updated. Job $JobIndex status: $($job.status)"
