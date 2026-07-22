param(
    [string]$InstallRoot = "C:\AMDDesignTools\2025.2\Vitis",
    [string]$Part = "xczu7ev-ffvc1156-2-e",
    [string]$ClockNs = "5.0",
    [ValidateSet("csim", "csynth", "full")]
    [string]$Stage = "full"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$tclPath = Join-Path $scriptDir "run_hls.tcl"
$settingsBat = Join-Path $InstallRoot "settings64.bat"

if (-not (Test-Path $settingsBat)) {
    throw "Cannot find settings64.bat at $settingsBat"
}

$env:HLS_PART = $Part
$env:HLS_CLOCK_NS = $ClockNs
$env:HLS_STAGE = $Stage

$cmd = "`"$settingsBat`" && vitis-run --mode hls --tcl `"$tclPath`""
Push-Location $repoRoot
try {
    cmd /c $cmd
} finally {
    Pop-Location
}
