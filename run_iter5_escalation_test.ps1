$env:ZHIPUAI_API_KEY = "5d79364a7c864f429c1711a4e660a6a1.fb75jZvQT9m3WYzI"
Set-Location C:\git\vibethinkharnessProto1
Write-Host "=== vibe iter5 escalation test ===" -ForegroundColor Cyan
Write-Host "API key loaded" -ForegroundColor Green
python -m vibeharness --agent web --task-file task_karriere_job2_iter5.txt --max-steps 60 2>&1 | Tee-Object -FilePath vibe_karriere_job2_iter5.log
Write-Host "=== Done ===" -ForegroundColor Cyan
