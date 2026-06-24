# scripts/leak_check.ps1 — lightweight host-health / leak check for heavy Claude Code sessions.
#
# WHY: long, high-concurrency sessions (many background subagents + browser trees + a
# leaky standalone llama-server, see issue #32) drove host RAM exhaustion that OOM-killed
# the Claude Code process — which takes all in-flight subagents with it. This check warns
# BEFORE that happens. See scripts/OPS_HEALTH_CHECK.md for the full mitigations list.
#
# OUTPUT: prints an "ALERT ..." line ONLY when something is wrong; silent when healthy,
# so it's safe to drive from a background Monitor (each printed line becomes a notification).
#
# USAGE:
#   One-shot:        powershell -NoProfile -ExecutionPolicy Bypass -File scripts/leak_check.ps1
#   Recurring (5m):  while ($true) { powershell -NoProfile -ExecutionPolicy Bypass -File scripts/leak_check.ps1; Start-Sleep 300 }
#   As a Claude Code Monitor (bash): while true; do powershell -NoProfile -ExecutionPolicy Bypass -File scripts/leak_check.ps1; sleep 300; done
#
# IMPORTANT: this only WARNS. Do NOT auto-kill Ollama or its bundled llama.cpp server
# (VibeThinker + output constraints depend on it). The llama-server leak is tracked as a
# known bug (#32) for a later sprint.

$os    = Get-CimInstance Win32_OperatingSystem
$free  = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
$pct   = [math]::Round((($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / $os.TotalVisibleMemorySize) * 100, 0)
$llama = [math]::Round((((Get-Process llama-server -ErrorAction SilentlyContinue) | Measure-Object WorkingSet64 -Sum).Sum) / 1MB, 0)
$cl    = @(Get-Process claude -ErrorAction SilentlyContinue).Count
$ch    = @(Get-Process chrome -ErrorAction SilentlyContinue).Count
$nd    = @(Get-Process node   -ErrorAction SilentlyContinue).Count

# Alert ONLY when free RAM is CRITICALLY low (< 4 GB) — the near-OOM danger zone —
# with the process breakdown inline (so the llama-server/chrome culprit is visible).
# Higher headroom is still recommended (OPS_HEALTH_CHECK.md), but we only INTERRUPT on a
# genuinely critical level to avoid noise. A high claude/chrome count alone is NOT alerted.
if ($free -lt 4) {
    Write-Output ("ALERT CRITICAL-RAM: free={0} GB ({1}% used) | llama-server={2}MB chrome={3} node={4} claude={5}" -f $free,$pct,$llama,$ch,$nd,$cl)
}
# healthy / non-critical -> no output
