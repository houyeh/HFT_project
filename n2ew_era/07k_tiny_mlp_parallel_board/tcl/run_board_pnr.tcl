proc slash_path {p} { return [string map {\\ /} $p] }

proc add_tclstore_pkg_dirs {dir} {
    if {![file isdirectory $dir]} { return }
    if {[file exists [file join $dir pkgIndex.tcl]] && [lsearch -exact $::auto_path $dir] == -1} {
        lappend ::auto_path $dir
    }
    foreach child [glob -nocomplain -directory $dir *] {
        if {[file isdirectory $child]} { add_tclstore_pkg_dirs $child }
    }
}
set vivado_root [expr {[info exists ::env(XILINX_VIVADO)] ? [slash_path $::env(XILINX_VIVADO)] : "C:/AMDDesignTools/2025.2/Vivado"}]
set tclstore_root [file join $vivado_root data XilinxTclStore]
add_tclstore_pkg_dirs $tclstore_root
set appinit_dir [file join $tclstore_root support appinit]
if {[file exists [file join $appinit_dir pkgIndex.tcl]] && [lsearch -exact $::auto_path $appinit_dir] == -1} {
    lappend ::auto_path $appinit_dir
}
catch {package require ::tclapp::support::appinit 1.2} appinit_msg
puts "TCLSTORE_APPINIT_CHECK dir=$appinit_dir msg=$appinit_msg"

if {[info exists ::env(PROJECT_ROOT)] && $::env(PROJECT_ROOT) ne ""} {
    set repo_root [slash_path $::env(PROJECT_ROOT)]
} else {
    set repo_root [slash_path [pwd]]
}
set phase_root [file join $repo_root n2ew_era 07k_tiny_mlp_parallel_board]
set core_dir [file join $repo_root n2ew_era 07j_tiny_mlp_parallel_hls hls_component tiny_mlp_parallel_csynth_work hls impl verilog]
set clock_rtl [file join $repo_root n2ew_era 06s2_pynq_z2_wrapper_bringup rtl pynq_z2_clock_100mhz.sv]
set xdc_file [file join $repo_root n2ew_era 06s2_pynq_z2_wrapper_bringup xdc pynq_z2_demo_wrapper_board.xdc]
set work_root [file join $repo_root _vivado_tinymlp_board]
set result_root [file join $phase_root results board_100mhz]
set mem_file [file join $phase_root bram tiny_mlp_board_vectors.mem]
file mkdir $result_root

set core_files [lsort [glob -nocomplain [file join $core_dir *.v]]]
if {[llength $core_files] != 20} { error "Expected 20 HLS RTL files, found [llength $core_files]" }
set wrapper_rtl [file join $phase_root rtl tiny_mlp_board_replay_wrapper.sv]
set board_top [file join $phase_root rtl pynq_z2_tiny_mlp_board_top.sv]
foreach required [concat $core_files [list $wrapper_rtl $board_top $clock_rtl $xdc_file $mem_file]] {
    if {![file exists $required]} { error "Missing required input: $required" }
}

create_project -force tiny_mlp_board $work_root -part xc7z020clg400-1
add_files -fileset sources_1 $core_files
add_files -fileset sources_1 [list $wrapper_rtl $board_top $clock_rtl $mem_file]
add_files -fileset constrs_1 $xdc_file

if {[llength [get_runs -quiet synth_1]] == 0} {
    create_run synth_1 -flow {Vivado Synthesis 2025} -part xc7z020clg400-1 -constrset constrs_1
}
if {[llength [get_runs -quiet impl_1]] == 0} {
    create_run impl_1 -flow {Vivado Implementation 2025} -part xc7z020clg400-1 -parent_run synth_1 -constrset constrs_1
}

create_ip -name vio -vendor xilinx.com -library ip -module_name tiny_mlp_demo_vio
set_property -dict [list \
    CONFIG.C_NUM_PROBE_IN {12} CONFIG.C_NUM_PROBE_OUT {0} \
    CONFIG.C_PROBE_IN0_WIDTH {32} CONFIG.C_PROBE_IN1_WIDTH {1} \
    CONFIG.C_PROBE_IN2_WIDTH {1} CONFIG.C_PROBE_IN3_WIDTH {1} \
    CONFIG.C_PROBE_IN4_WIDTH {32} CONFIG.C_PROBE_IN5_WIDTH {32} \
    CONFIG.C_PROBE_IN6_WIDTH {40} CONFIG.C_PROBE_IN7_WIDTH {32} \
    CONFIG.C_PROBE_IN8_WIDTH {40} CONFIG.C_PROBE_IN9_WIDTH {40} \
    CONFIG.C_PROBE_IN10_WIDTH {40} CONFIG.C_PROBE_IN11_WIDTH {2}] [get_ips tiny_mlp_demo_vio]
generate_target all [get_ips tiny_mlp_demo_vio]
create_ip_run [get_ips tiny_mlp_demo_vio]
set vio_run [get_runs -quiet tiny_mlp_demo_vio_synth_1]
launch_runs $vio_run -jobs 2
wait_on_run $vio_run
if {![string match "*Complete*" [get_property STATUS $vio_run]]} {
    error "VIO synthesis failed: [get_property STATUS $vio_run]"
}

set_property top pynq_z2_tiny_mlp_board_top [current_fileset]
set_property generic "MEM_FILE=$mem_file INJECT_NEGATIVE_TEST=0" [get_filesets sources_1]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 2
wait_on_run synth_1
if {![string match "*Complete*" [get_property STATUS [get_runs synth_1]]]} {
    error "Top synthesis failed: [get_property STATUS [get_runs synth_1]]"
}
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1
if {![string match "*Complete*" [get_property STATUS [get_runs impl_1]]]} {
    error "Implementation failed: [get_property STATUS [get_runs impl_1]]"
}
open_run impl_1
report_route_status -file [file join $result_root route_status.rpt]
report_drc -file [file join $result_root drc_routed.rpt]
report_utilization -file [file join $result_root utilization_routed.rpt]
report_timing_summary -max_paths 20 -report_unconstrained -warn_on_violation -file [file join $result_root timing_summary_routed.rpt]
write_checkpoint -force [file join $result_root tiny_mlp_board_routed.dcp]

set worst [get_timing_paths -delay_type max -max_paths 1]
set wns [get_property SLACK [lindex $worst 0]]
set failing [get_timing_paths -delay_type max -slack_lesser_than 0 -max_paths 10000]
set tns 0.0
foreach path $failing { set tns [expr {$tns + double([get_property SLACK $path])}] }
puts "TINY_MLP_BOARD_TIMING WNS=$wns TNS=$tns failing_endpoints_capped=[llength $failing]"
if {$wns < 0.0 || $tns < -0.0005} { error "100 MHz board timing failed" }

set generated_bit [file join $work_root tiny_mlp_board.runs impl_1 pynq_z2_tiny_mlp_board_top.bit]
set bit_file [file join $result_root tiny_mlp_parallel_100mhz.bit]
set ltx_file [file join $result_root tiny_mlp_parallel_100mhz.ltx]
file copy -force $generated_bit $bit_file
write_debug_probes -force $ltx_file
puts "TINY_MLP_BOARD_BUILD_PASS bitstream=$bit_file probes=$ltx_file"
close_project
