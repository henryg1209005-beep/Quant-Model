$repoRoot = "C:\Users\yxngh\Documents\ai_trading_engine"
$portLine = netstat -ano | Select-String -Pattern "127\.0\.0\.1:8000\s+.*LISTENING" -SimpleMatch:$false
if ($portLine) {
    exit 0
}

Set-Location $repoRoot
if (!(Test-Path "data")) {
    New-Item -ItemType Directory -Path "data" | Out-Null
}

$pythonExe = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"
if (!(Test-Path $pythonExe)) {
    $pythonExe = "python"
}

Start-Process -WindowStyle Hidden -FilePath $pythonExe `
    -ArgumentList "-m","uvicorn","ai_trading_engine.api:app","--app-dir","src","--host","127.0.0.1","--port","8000" `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput (Join-Path $repoRoot "data\server.out.log") `
    -RedirectStandardError (Join-Path $repoRoot "data\server.err.log")
