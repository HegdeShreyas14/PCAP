param(
    [int]$WaitForPid = 0
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

Set-Location -LiteralPath $projectRoot

function Invoke-Checked {
    param(
        [string]$Label,
        [string[]]$Arguments
    )
    Write-Output "[$(Get-Date -Format s)] START $Label"
    # Python's unittest runner writes normal progress to stderr. Merge the
    # streams so the runner's dedicated error log indicates an actual runner
    # failure rather than a successful test report.
    & $pythonExe @Arguments 2>&1 | ForEach-Object { Write-Output $_ }
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
    Write-Output "[$(Get-Date -Format s)] DONE  $Label"
}

if ($WaitForPid -gt 0) {
    $existing = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Output "[$(Get-Date -Format s)] Waiting for RQ1 process $WaitForPid"
        Wait-Process -Id $WaitForPid
    }
}

Invoke-Checked "RQ1 paired analysis" @(
    "src\analyze.py", "--compare",
    "results\rq1_primary_1t.csv", "results\rq1_primary_8t.csv"
)
Invoke-Checked "Percolation analysis" @(
    "src\percolation.py", "--seeds", "0", "1", "2",
    "--out", "results\percolation.csv", "--blas-threads", "8", "--resume"
)
Invoke-Checked "Phase-3 benchmark" @(
    "src\benchmark_phase3.py", "--out", "results\phase3_results.csv",
    "--machine", "primary", "--seed", "0", "--trials", "9",
    "--blas-threads", "8", "--resume"
)
Invoke-Checked "Phase-3 artifact generation" @(
    "src\analyze_phase3.py", "results\phase3_results.csv",
    "--out-dir", "results"
)
Invoke-Checked "Correctness tests" @(
    "-m", "unittest", "discover", "-s", "tests", "-v"
)

Write-Output "[$(Get-Date -Format s)] ALL FINAL EXPERIMENTS COMPLETE"
