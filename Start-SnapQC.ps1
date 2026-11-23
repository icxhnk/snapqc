param([int]$Port = 57840)
$ErrorActionPreference = 'Stop'
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw 'Lingkungan Python .venv belum tersedia.' }
$env:PORT = "$Port"
$env:HOST = '127.0.0.1'
& $py -u (Join-Path $PSScriptRoot 'server.py')
