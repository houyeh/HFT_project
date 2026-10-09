param(
    [ValidateSet("positive", "negative")]
    [string]$Mode = "positive",
    [string]$ProjectRoot = "C:\Users\holbo\00VS_projjects\00_HFT_project",
    [string]$VivadoBin = "C:\AMDDesignTools\2025.2\Vivado\bin",
    [string]$CoreRtlDir = "",
    [switch]$CoreHasStatePrevPort1,
    [switch]$TraceWrapper,
    [int]$TraceIndex = 0,
    [int]$ReadoutToleranceLsb = 1,
    [int]$VectorCount = 4096,
    [switch]$DumpAllReadout,
    [switch]$DrainExtraCycle,
    [string]$WorkSuffix = "",
    [string]$DumpCopyPath = ""
)

$ErrorActionPreference = "Stop"

$phaseRoot = Join-Path $ProjectRoot "n2ew_era\06s2_pynq_z2_wrapper_bringup"
if (-not $CoreRtlDir) {
    $CoreRtlDir = Join-Path $ProjectRoot "n2ew_era\06u_phc_latency_breakthrough\variants\v1_unroll16_dataflow_90mhz\hls_component\hls_component\hls\sim\verilog"
}

$xvlog = Join-Path $VivadoBin "xvlog.bat"
$xelab = Join-Path $VivadoBin "xelab.bat"
$xsim = Join-Path $VivadoBin "xsim.bat"
foreach ($tool in @($xvlog, $xelab, $xsim)) {
    if (-not (Test-Path $tool)) {
        throw "Missing Vivado simulation tool: $tool"
    }
}

$workName = $Mode
if ($WorkSuffix) {
    $safeSuffix = $WorkSuffix -replace '[^A-Za-z0-9_.-]', '_'
    $workName = "${Mode}_${safeSuffix}"
}

$workRoot = Join-Path $phaseRoot "xsim_direct_work\$workName"
$logRoot = Join-Path $phaseRoot "logs"
New-Item -ItemType Directory -Force $workRoot | Out-Null
New-Item -ItemType Directory -Force $logRoot | Out-Null

Get-ChildItem -Path $CoreRtlDir -Filter *.dat -File | Copy-Item -Destination $workRoot -Force

$fileList = Join-Path $workRoot "xsim_files.f"
$rtlFiles = @()
$rtlFiles += Get-ChildItem -Path $CoreRtlDir -Filter *.v -File | Sort-Object Name | ForEach-Object { $_.FullName }
$rtlFiles += Join-Path $phaseRoot "rtl\pynq_z2_phc_demo_wrapper.sv"
$rtlFiles += Join-Path $phaseRoot "sim\tb_pynq_z2_phc_demo_wrapper.sv"
$rtlFiles | ForEach-Object { '"' + $_.Replace('\', '/') + '"' } | Set-Content -Path $fileList -Encoding ascii

Push-Location $workRoot
try {
    $defineArgs = @()
    if ($Mode -eq "negative") {
        $defineArgs = @("-d", "NEGATIVE_TEST")
    }
    if ($CoreHasStatePrevPort1) {
        $defineArgs += @("-d", "PHC_HAS_STATE_PREV_PORT1")
    }
    if ($ReadoutToleranceLsb -eq 1) {
        $defineArgs += @("-d", "READOUT_TOL_LSB_1")
    } elseif ($ReadoutToleranceLsb -ne 0) {
        throw "Unsupported ReadoutToleranceLsb=$ReadoutToleranceLsb. Supported values: 0 or 1."
    }
    $defineArgs += @("-d", "PHC_USE_HLS_MEM_DLY")
    if ($TraceWrapper) {
        $defineArgs += @("-d", "PHC_WRAPPER_TRACE")
        if ($TraceIndex -eq 3223) {
            $defineArgs += @("-d", "PHC_TRACE_INDEX_3223")
        } elseif ($TraceIndex -ne 0) {
            throw "TraceIndex=$TraceIndex is not wired into compile-time macros yet."
        }
    }
    if ($DumpAllReadout) {
        $defineArgs += @("-d", "PHC_WRAPPER_DUMP_ALL_READOUT")
    }
    if ($DrainExtraCycle) {
        $defineArgs += @("-d", "PHC_DRAIN_EXTRA_CYCLE")
    }
    if ($VectorCount -in @(32, 256, 1024)) {
        $defineArgs += @("-d", "VECTOR_COUNT_$VectorCount")
    } elseif ($VectorCount -ne 4096) {
        throw "Unsupported VectorCount=$VectorCount. Supported values: 32, 256, 1024, 4096."
    }

    $xvlogLog = Join-Path $logRoot "xsim_direct_${workName}_xvlog.log"
    $xelabLog = Join-Path $logRoot "xsim_direct_${workName}_xelab.log"
    $xsimLog = Join-Path $logRoot "xsim_direct_${workName}_xsim.log"

    & $xvlog -sv -work xil_defaultlib @defineArgs -f $fileList 2>&1 | Tee-Object -FilePath $xvlogLog
    if ($LASTEXITCODE -ne 0) { throw "xvlog failed with exit code $LASTEXITCODE" }

    & $xelab -debug typical xil_defaultlib.tb_pynq_z2_phc_demo_wrapper -s phc_demo_wrapper_snapshot 2>&1 | Tee-Object -FilePath $xelabLog
    if ($LASTEXITCODE -ne 0) { throw "xelab failed with exit code $LASTEXITCODE" }

    & $xsim phc_demo_wrapper_snapshot -runall 2>&1 | Tee-Object -FilePath $xsimLog
    if ($LASTEXITCODE -ne 0) { throw "xsim failed with exit code $LASTEXITCODE" }

    if ($Mode -eq "positive") {
        $passPattern = "POSITIVE_TEST_PASS"
    } else {
        $passPattern = "NEGATIVE_TEST_PASS"
    }
    if (-not (Select-String -Path $xsimLog -Pattern $passPattern -Quiet)) {
        throw "Simulation completed without $passPattern in $xsimLog"
    }

    if ($DumpAllReadout) {
        $dumpInWork = Join-Path $workRoot "phc_wrapper_readout_dump.csv"
        if (-not (Test-Path $dumpInWork)) {
            throw "DumpAllReadout was requested but no dump file was written: $dumpInWork"
        }
        if ($DumpCopyPath) {
            $dumpParent = Split-Path -Parent $DumpCopyPath
            if ($dumpParent) {
                New-Item -ItemType Directory -Force $dumpParent | Out-Null
            }
            Copy-Item -Path $dumpInWork -Destination $DumpCopyPath -Force
            Write-Output "READOUT_DUMP path=$DumpCopyPath"
        } else {
            Write-Output "READOUT_DUMP path=$dumpInWork"
        }
    }

    Write-Output "XSIM_DIRECT_PASS mode=$Mode log=$xsimLog"
} finally {
    Pop-Location
}
