# V3 diagonal Stage B interface/handshake simulation.
# Numerical equivalence is intentionally deferred until Stage C produces
# model-matched replay expectations.

proc slash_path {p} {
    return [string map {\\ /} $p]
}

if {[info exists ::env(PROJECT_ROOT)] && $::env(PROJECT_ROOT) ne ""} {
    set repo_root [slash_path $::env(PROJECT_ROOT)]
} else {
    set repo_root [slash_path [pwd]]
}

set phase_root [slash_path [file join $repo_root n2ew_era 06s2_pynq_z2_wrapper_bringup]]
set core_rtl_dir [slash_path [file join $repo_root n2ew_era 06u_phc_latency_breakthrough variants v3_diagonal16_unroll8_localpart_100mhz rtl_export]]
set work_root [slash_path [file join $phase_root xsim_work_v3diag_stage_b]]
file mkdir $work_root

set core_rtl_files [glob -nocomplain [file join $core_rtl_dir *.v]]
set core_dat_files [glob -nocomplain [file join $core_rtl_dir *.dat]]
if {[llength $core_rtl_files] != 44} {
    error "Expected 44 V3 Verilog files, found [llength $core_rtl_files] in $core_rtl_dir"
}
if {[llength $core_dat_files] != 21} {
    error "Expected 21 V3 ROM data files, found [llength $core_dat_files] in $core_rtl_dir"
}

set wrapper_rtl [slash_path [file join $phase_root rtl pynq_z2_phc_demo_wrapper_v3diag.sv]]
set tb_rtl [slash_path [file join $phase_root sim tb_pynq_z2_phc_demo_wrapper_v3diag_stage_b.sv]]

create_project -force phc_demo_v3diag_stage_b $work_root -part xc7z020clg400-1
add_files -fileset sources_1 $core_rtl_files
add_files -fileset sources_1 $wrapper_rtl
add_files -fileset sim_1 $core_dat_files
add_files -fileset sim_1 $tb_rtl
set_property top tb_pynq_z2_phc_demo_wrapper_v3diag_stage_b [get_filesets sim_1]
set_property verilog_define {PHC_USE_HLS_MEM_DLY} [get_filesets sim_1]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
launch_simulation -simset sim_1 -mode behavioral
run all
close_sim
puts "V3DIAG_STAGE_B_XSIM_COMPLETED"
