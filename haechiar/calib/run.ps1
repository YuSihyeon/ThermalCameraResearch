# Bootstrap a repository-local environment, then forward CLI arguments unchanged.
$ErrorActionPreference = 'Stop'
$calibArgs = @($args)
if ($calibArgs.Count -eq 0) { $calibArgs = @('wizard', '--source', 'usb') }
$calibRoot = $PSScriptRoot
$calibPython = Join-Path $calibRoot '.venv\Scripts\python.exe'
$calibMarker = Join-Path $calibRoot '.venv\firesight-bootstrap.txt'
$calibHash = (Get-FileHash -LiteralPath (Join-Path $calibRoot 'pyproject.toml') -Algorithm SHA256).Hash
if (-not (Test-Path -LiteralPath $calibPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv (Join-Path $calibRoot '.venv')
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv (Join-Path $calibRoot '.venv')
    } else {
        throw 'Install Python 3.10+ (64-bit) from python.org, then rerun this script.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment. Install Python 3.10+ (64-bit).' }
}
& $calibPython -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'
if ($LASTEXITCODE -ne 0) { throw 'Python 3.10+ is required. Recreate .venv using a supported Python.' }
if (-not (Test-Path -LiteralPath $calibMarker) -or (Get-Content -LiteralPath $calibMarker -Raw).Trim() -ne $calibHash) {
    & $calibPython -m pip install -e $calibRoot
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check network access and pip output above.' }
    Set-Content -LiteralPath $calibMarker -Value $calibHash
}
& $calibPython -m firesight_calib.cli @calibArgs
exit $LASTEXITCODE
