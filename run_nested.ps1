param(
    [switch]$Headless,
    [double]$Speed = 1,
    [double]$Turn = 0,
    [double]$Duration = 60,
    [string]$Video = ''
)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create .venv and install requirements-sim.txt first; see README.md.'
}
$simPath = Join-Path $PSScriptRoot 'sim.py'
$arguments = @($simPath, 'wheelbot', '--controller', 'nested',
    '--speed', "$Speed", '--turn', "$Turn", '--duration', "$Duration", '--camera-distance', '1.1')
if ($Headless) { $arguments += '--headless' }
if ($Video) { $arguments += @('--video', $Video) }
& $pythonPath @arguments
exit $LASTEXITCODE
