param(
    [ValidateSet("positive", "negative")]
    [string]$Mode = "positive",
    [string]$ProjectRoot = "C:\Users\holbo\00VS_projjects\00_HFT_project",
    [string]$VivadoBin = "C:\AMDDesignTools\2025.2\Vivado\bin"
)

$ErrorActionPreference = "Stop"

$phaseRoot = Join-Path $ProjectRoot "n2ew_era\06s2_pynq_z2_wrapper_bringup"
$coreRtlDir = Join-Path $ProjectRoot "n2ew_era\06u_phc_latency_breakthrough\variants\v3_diagonal16_unroll8_localpart_100mhz\rtl_export"
$workRoot = Join-Path $phaseRoot "xsim_direct_work\v3diag_full_$Mode"
$logRoot = Join-Path $phaseRoot "logs"
$resultRoot = Join-Path $phaseRoot "results_v3_diag"

$xvlog = Join-Path $VivadoBin "xvlog.bat"
$xelab = Join-Path $VivadoBin "xelab.bat"
$xsim = Join-Path $VivadoBin "xsim.bat"
foreach ($tool in @($xvlog, $xelab, $xsim)) {
    if (-not (Test-Path $tool)) { throw "Missing Vivado simulation tool: $tool" }
}

$rtlFiles = @(Get-ChildItem -Path $coreRtlDir -Filter *.v -File | Sort-Object Name)
$datFiles = @(Get-ChildItem -Path $coreRtlDir -Filter *.dat -File | Sort-Object Name)
if ($rtlFiles.Count -ne 44) { throw "Expected 44 V3 Verilog files, found $($rtlFiles.Count)" }
if ($datFiles.Count -ne 21) { throw "Expected 21 V3 ROM data files, found $($datFiles.Count)" }

New-Item -ItemType Directory -Force $workRoot | Out-Null
New-Item -ItemType Directory -Force $logRoot | Out-Null
New-Item -ItemType Directory -Force $resultRoot | Out-Null
$datFiles | Copy-Item -Destination $workRoot -Force

$fileList = Join-Path $workRoot "xsim_files.f"
$sources = @($rtlFiles.FullName)
$sources += Join-Path $phaseRoot "rtl\pynq_z2_phc_demo_wrapper_v3diag.sv"
$sources += Join-Path $phaseRoot "sim\tb_pynq_z2_phc_demo_wrapper_v3diag.sv"
$sources | ForEach-Object { '"' + $_.Replace('\', '/') + '"' } | Set-Content -Path $fileList -Encoding ascii

$xvlogLog = Join-Path $logRoot "xsim_v3diag_full_${Mode}_xvlog.log"
$xelabLog = Join-Path $logRoot "xsim_v3diag_full_${Mode}_xelab.log"
$xsimLog = Join-Path $logRoot "xsim_v3diag_full_${Mode}_xsim.log"

Push-Location $workRoot
try {
    $defines = @("-d", "PHC_USE_HLS_MEM_DLY")
    if ($Mode -eq "positive") {
        $defines += @("-d", "PHC_WRAPPER_DUMP_ALL_READOUT")
    } else {
        $defines += @("-d", "NEGATIVE_TEST")
    }

    & $xvlog -sv -work xil_defaultlib @defines -f $fileList 2>&1 | Tee-Object -FilePath $xvlogLog
    if ($LASTEXITCODE -ne 0) { throw "xvlog failed with exit code $LASTEXITCODE" }

    & $xelab -debug typical xil_defaultlib.tb_pynq_z2_phc_demo_wrapper_v3diag -s phc_demo_v3diag_full_snapshot 2>&1 | Tee-Object -FilePath $xelabLog
    if ($LASTEXITCODE -ne 0) { throw "xelab failed with exit code $LASTEXITCODE" }

    & $xsim phc_demo_v3diag_full_snapshot -runall 2>&1 | Tee-Object -FilePath $xsimLog
    if ($LASTEXITCODE -ne 0) { throw "xsim failed with exit code $LASTEXITCODE" }

    $marker = if ($Mode -eq "positive") { "V3DIAG_POSITIVE_TEST_PASS" } else { "V3DIAG_NEGATIVE_TEST_PASS" }
    if (-not (Select-String -Path $xsimLog -SimpleMatch $marker -Quiet)) {
        throw "Simulation completed without expected marker: $marker"
    }

    if ($Mode -eq "positive") {
        $dump = Join-Path $workRoot "phc_wrapper_readout_dump.csv"
        if (-not (Test-Path $dump)) { throw "Positive run did not produce readout dump: $dump" }
        Copy-Item $dump (Join-Path $resultRoot "v3diag_wrapper_readout_dump_xsim.csv") -Force
    }
    Write-Output "XSIM_V3DIAG_FULL_PASS mode=$Mode log=$xsimLog"
} finally {
    Pop-Location
}
