# Board-level P&R for the V3 diagonal replay demo with final-stat VIO readback.

proc slash_path {p} {
    return [string map {\\ /} $p]
}

proc phc_add_tclstore_pkg_dirs {dir} {
    if {![file isdirectory $dir]} {
        return
    }
    if {[file exists [file join $dir pkgIndex.tcl]] && [lsearch -exact $::auto_path $dir] == -1} {
        lappend ::auto_path $dir
    }
    foreach child [glob -nocomplain -directory $dir *] {
        if {[file isdirectory $child]} {
            phc_add_tclstore_pkg_dirs $child
        }
    }
}

if {[info exists ::env(XILINX_VIVADO)] && ![string equal $::env(XILINX_VIVADO) ""]} {
    set vivado_root [slash_path $::env(XILINX_VIVADO)]
} else {
    set vivado_root "C:/AMDDesignTools/2025.2/Vivado"
}
set tclstore_root [file join $vivado_root data XilinxTclStore]
phc_add_tclstore_pkg_dirs $tclstore_root
set appinit_dir [file join $tclstore_root support appinit]
if {[file exists [file join $appinit_dir pkgIndex.tcl]] && [lsearch -exact $::auto_path $appinit_dir] == -1} {
    lappend ::auto_path $appinit_dir
}
catch {package require ::tclapp::support::appinit 1.2} appinit_msg
puts "TCLSTORE_APPINIT_CHECK dir=$appinit_dir msg=$appinit_msg"

if {[info exists ::env(PHC_06S2_PHASE_ROOT)] && ![string equal $::env(PHC_06S2_PHASE_ROOT) ""]} {
    set phase_root [slash_path $::env(PHC_06S2_PHASE_ROOT)]
} else {
    set script_path [slash_path [info script]]
    set phase_root [file join [file dirname $script_path] ..]
}
set repo_root [file dirname [file dirname $phase_root]]
set core_rtl_dir [file join $repo_root n2ew_era 06u_phc_latency_breakthrough variants v3_diagonal16_unroll8_localpart_100mhz rtl_export]
set wrapper_rtl [file join $phase_root rtl pynq_z2_phc_demo_wrapper_v3diag_readback.sv]
set clock_rtl [file join $phase_root rtl pynq_z2_clock_100mhz.sv]
set board_top_rtl [file join $phase_root rtl pynq_z2_phc_demo_board_top_v3diag_readback.sv]
set xdc_file [file join $phase_root xdc pynq_z2_demo_wrapper_board.xdc]
set bram_mem [file join $phase_root bram_v3_diag demo_replay_x_readout_q24_12.mem]
set negative_test 0
if {[info exists ::env(PHC_NEGATIVE_TEST)] && $::env(PHC_NEGATIVE_TEST) eq "1"} {
    set negative_test 1
}
set build_suffix [expr {$negative_test ? "negative" : "positive"}]
# Keep the generated project short enough for Windows' 260-byte path limit.
set work_root [file join $repo_root "_vivado_v3diag_vio_$build_suffix"]
set report_root [file join $phase_root results_v3_diag "pnr_100mhz_readback_vio_$build_suffix"]

foreach required [list $wrapper_rtl $clock_rtl $board_top_rtl $xdc_file $bram_mem] {
    if {![file exists $required]} {
        error "Missing required input: $required"
    }
}
set core_v [lsort [glob -nocomplain [file join $core_rtl_dir *.v]]]
set core_dat [lsort [glob -nocomplain [file join $core_rtl_dir *.dat]]]
if {[llength $core_v] != 44 || [llength $core_dat] != 21} {
    error "Expected 44 .v and 21 .dat; found [llength $core_v] and [llength $core_dat]"
}

file mkdir $work_root
file mkdir $report_root
create_project -force phc_v3diag_readback_vio $work_root -part xc7z020clg400-1
add_files -fileset sources_1 $core_v
add_files -fileset sources_1 $wrapper_rtl
add_files -fileset sources_1 $clock_rtl
add_files -fileset sources_1 $board_top_rtl
add_files -fileset sources_1 $bram_mem
foreach dat $core_dat {
    add_files -fileset sources_1 $dat
}
add_files -fileset constrs_1 $xdc_file

# create_project does not populate runs in this batch-only installation.
if {[llength [get_runs -quiet synth_1]] == 0} {
    create_run synth_1 -flow {Vivado Synthesis 2025} -part xc7z020clg400-1 -constrset constrs_1
}
if {[llength [get_runs -quiet impl_1]] == 0} {
    create_run impl_1 -flow {Vivado Implementation 2025} -part xc7z020clg400-1 \
        -parent_run synth_1 -constrset constrs_1
}

create_ip -name vio -vendor xilinx.com -library ip -module_name phc_demo_vio
set_property -dict [list \
    CONFIG.C_NUM_PROBE_IN {12} \
    CONFIG.C_NUM_PROBE_OUT {0} \
    CONFIG.C_PROBE_IN0_WIDTH {32} \
    CONFIG.C_PROBE_IN1_WIDTH {1} \
    CONFIG.C_PROBE_IN2_WIDTH {1} \
    CONFIG.C_PROBE_IN3_WIDTH {1} \
    CONFIG.C_PROBE_IN4_WIDTH {32} \
    CONFIG.C_PROBE_IN5_WIDTH {32} \
    CONFIG.C_PROBE_IN6_WIDTH {32} \
    CONFIG.C_PROBE_IN7_WIDTH {32} \
    CONFIG.C_PROBE_IN8_WIDTH {32} \
    CONFIG.C_PROBE_IN9_WIDTH {32} \
    CONFIG.C_PROBE_IN10_WIDTH {32} \
    CONFIG.C_PROBE_IN11_WIDTH {32} \
] [get_ips phc_demo_vio]
generate_target all [get_ips phc_demo_vio]
create_ip_run [get_ips phc_demo_vio]
set vio_run [get_runs -quiet phc_demo_vio_synth_1]
if {[llength $vio_run] == 0} {
    error "Vivado did not create the VIO out-of-context synthesis run"
}
puts "STAGE_E_VIO_OOC_RUN name=$vio_run"
launch_runs $vio_run -jobs 2
wait_on_run $vio_run
set vio_status [get_property STATUS $vio_run]
puts "STAGE_E_VIO_OOC_STATUS status=$vio_status"
if {![string match "*Complete*" $vio_status]} {
    error "VIO out-of-context synthesis did not complete: $vio_status"
}

set_property top pynq_z2_phc_demo_board_top_v3diag_readback [current_fileset]
set_property generic "MEM_FILE=$bram_mem INJECT_NEGATIVE_TEST=$negative_test" [get_filesets sources_1]
update_compile_order -fileset sources_1

puts "STAGE_E_READBACK_BUILD core_v=[llength $core_v] core_dat=[llength $core_dat] vio=1"
puts "STAGE_E_READBACK_MODE negative_test=$negative_test"
puts "STAGE_E_READBACK_CLOCK DIVCLK=5 MULT=40 CLKOUT_DIVIDE=10 output_mhz=100"

launch_runs synth_1 -jobs 2
wait_on_run synth_1
set synth_status [get_property STATUS [get_runs synth_1]]
puts "STAGE_E_TOP_SYNTH_STATUS status=$synth_status"
if {![string match "*Complete*" $synth_status]} {
    error "Top-level synthesis did not complete: $synth_status"
}

launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1
set impl_status [get_property STATUS [get_runs impl_1]]
puts "STAGE_E_TOP_IMPL_STATUS status=$impl_status"
if {![string match "*Complete*" $impl_status]} {
    error "Top-level implementation did not complete: $impl_status"
}
open_run impl_1

write_checkpoint -force [file join $report_root v3diag_readback_routed.dcp]
report_route_status -file [file join $report_root v3diag_readback_route_status.rpt]
report_drc -file [file join $report_root v3diag_readback_drc_routed.rpt]
report_utilization -file [file join $report_root v3diag_readback_utilization_routed.rpt]
report_timing_summary -max_paths 20 -report_unconstrained -warn_on_violation -file [file join $report_root v3diag_readback_timing_summary_routed.rpt]

set worst_paths [get_timing_paths -delay_type max -max_paths 1]
if {[llength $worst_paths] == 0} {
    error "No setup timing path found after route"
}
set wns [get_property SLACK [lindex $worst_paths 0]]
set failing_paths [get_timing_paths -delay_type max -slack_lesser_than 0 -max_paths 10000]
set tns 0.0
foreach path $failing_paths {
    set tns [expr {$tns + double([get_property SLACK $path])}]
}
puts "STAGE_E_READBACK_TIMING WNS=$wns TNS=$tns failing_endpoints_capped=[llength $failing_paths]"
if {$wns < 0.0 || $tns < -0.0005} {
    error "V3 readback VIO build failed 100 MHz timing"
}

set generated_bit [file join $work_root phc_v3diag_readback_vio.runs impl_1 pynq_z2_phc_demo_board_top_v3diag_readback.bit]
if {![file exists $generated_bit]} {
    error "Implementation completed without the expected bitstream: $generated_bit"
}
set bit_path [file join $report_root "phc_v3diag_demo_100mhz_readback_vio_$build_suffix.bit"]
set ltx_path [file join $report_root "phc_v3diag_demo_100mhz_readback_vio_$build_suffix.ltx"]
file copy -force $generated_bit $bit_path
write_debug_probes -force $ltx_path
puts "STAGE_E_READBACK_BUILD_PASS bitstream=$bit_path probes=$ltx_path"
close_project
