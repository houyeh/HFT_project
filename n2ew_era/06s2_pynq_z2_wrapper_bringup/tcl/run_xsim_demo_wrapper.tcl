# Vivado XSim wrapper integration simulation.
#
# Usage from repo root:
#   vivado -mode batch -source n2ew_era/06s2_pynq_z2_wrapper_bringup/tcl/run_xsim_demo_wrapper.tcl
#
# Negative comparator test:
#   vivado -mode batch -source n2ew_era/06s2_pynq_z2_wrapper_bringup/tcl/run_xsim_demo_wrapper.tcl -tclargs negative
#
# Important: before the positive test can be bit-exact, regenerate
# bram/demo_replay_x_readout_q24_12.mem from HLS CSim DUT output by running
# 02_update_bram_expected_from_hls_csim.py. The initial file is based on
# PyTorch float reference and is not guaranteed bit-exact to fixed-point RTL.

proc slash_path {p} {
    return [string map {\\ /} $p]
}

if {[info exists ::env(PROJECT_ROOT)] && $::env(PROJECT_ROOT) ne ""} {
    set repo_root [slash_path $::env(PROJECT_ROOT)]
} else {
    set pwd_root [slash_path [pwd]]
    if {[file exists [file join $pwd_root n2ew_era 06s2_pynq_z2_wrapper_bringup]]} {
        set repo_root $pwd_root
    } else {
        set repo_root [slash_path [file join [file dirname [info script]] .. .. ..]]
    }
}
set phase_root [slash_path [file join $repo_root n2ew_era 06s2_pynq_z2_wrapper_bringup]]
set work_root [slash_path [file join $phase_root xsim_work]]
file mkdir $work_root

set negative 0
if {$argc > 0 && [string equal [lindex $argv 0] "negative"]} {
    set negative 1
}

if {[info exists ::env(PHC_CORE_RTL_DIR)] && $::env(PHC_CORE_RTL_DIR) ne ""} {
    set phc_verilog_dir [slash_path $::env(PHC_CORE_RTL_DIR)]
} else {
    # Use HLS simulation RTL for wrapper-level XSim. The 06U impl/verilog
    # top-level directory does not contain every generated child module unless
    # the nested IP-shared folders are added separately.
    set phc_verilog_dir [slash_path [file join $repo_root n2ew_era 06u_phc_latency_breakthrough variants v1_unroll16_dataflow_90mhz hls_component hls_component hls sim verilog]]
}
set phc_rtl_files [glob -nocomplain [file join $phc_verilog_dir *.v]]
set phc_dat_files [glob -nocomplain [file join $phc_verilog_dir *.dat]]
if {[llength $phc_rtl_files] == 0} {
    error "No HLS Verilog files found in $phc_verilog_dir. Set PHC_CORE_RTL_DIR to a valid generated RTL directory."
}
set wrapper_rtl [slash_path [file join $phase_root rtl pynq_z2_phc_demo_wrapper.sv]]
set tb_rtl [slash_path [file join $phase_root sim tb_pynq_z2_phc_demo_wrapper.sv]]

create_project -force phc_demo_xsim $work_root -part xc7z020clg400-1
add_files -fileset sources_1 $phc_rtl_files
add_files -fileset sources_1 $wrapper_rtl
if {[llength $phc_dat_files] > 0} {
    add_files -fileset sim_1 $phc_dat_files
}
add_files -fileset sim_1 $tb_rtl
set_property top tb_pynq_z2_phc_demo_wrapper [get_filesets sim_1]
if {$negative} {
    set_property verilog_define {NEGATIVE_TEST} [get_filesets sim_1]
}
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
launch_simulation -simset sim_1 -mode behavioral
run all
close_sim
puts "XSim wrapper simulation completed. negative=$negative"
