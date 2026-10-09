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
set work_root [file join $repo_root _vivado_tinymlp_xsim]
set mem_file [file join $phase_root bram tiny_mlp_board_vectors.mem]

set core_files [lsort [glob -nocomplain [file join $core_dir *.v]]]
if {[llength $core_files] != 20} { error "Expected 20 HLS RTL files, found [llength $core_files]" }
foreach required [list \
    [file join $phase_root rtl tiny_mlp_board_replay_wrapper.sv] \
    [file join $phase_root tb tb_tiny_mlp_board_replay.sv] \
    $mem_file] {
    if {![file exists $required]} { error "Missing required input: $required" }
}

create_project -force tiny_mlp_board_xsim $work_root -part xc7z020clg400-1
add_files -fileset sources_1 $core_files
add_files -fileset sources_1 [file join $phase_root rtl tiny_mlp_board_replay_wrapper.sv]
add_files -fileset sim_1 [file join $phase_root tb tb_tiny_mlp_board_replay.sv]
add_files -fileset sim_1 $mem_file
set_property top tb_tiny_mlp_board_replay [get_filesets sim_1]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
launch_simulation -simset sim_1 -mode behavioral
run all
close_sim
puts "TINY_MLP_BOARD_XSIM_COMPLETED"
close_project
