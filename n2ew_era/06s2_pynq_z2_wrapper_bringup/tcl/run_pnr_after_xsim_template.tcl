# Guarded P&R/bitstream script for the demo wrapper.
#
# This script intentionally refuses to run until wrapper XSim is fixed and a
# marker file is created. It is a preparation artifact, not a green-light to
# bypass simulation.
#
# Required marker after positive and negative XSim both pass:
#   n2ew_era/06s2_pynq_z2_wrapper_bringup/results/XSIM_POSITIVE_AND_NEGATIVE_PASS.marker
#
# Required environment variable:
#   PHC_CORE_RTL_DIR=<complete RTL directory for the exact final PHC core>
#
# Selected demo RTL after 06U V1 source-set audit:
#   06S seed2 / 06L unroll16 90 MHz complete RTL
#   headline: 25.733 us @ 90 MHz, routed timing met in 06L evidence.

# Vivado 2025.2 on this workstation reports a corrupted user TclStore and then
# sometimes fails to locate ::tclapp::support::appinit from the installation
# TclStore. Add the install appinit directory explicitly as a local workaround;
# this does not modify the Vivado installation.
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
    set vivado_root_raw $::env(XILINX_VIVADO)
} else {
    set vivado_root_raw "C:/AMDDesignTools/2025.2/Vivado"
}
regsub -all {\\} $vivado_root_raw {/} vivado_root_raw
set tclstore_root [file join $vivado_root_raw data XilinxTclStore]
phc_add_tclstore_pkg_dirs $tclstore_root
set appinit_dir [file join $vivado_root_raw data XilinxTclStore support appinit]
if {[file exists [file join $appinit_dir pkgIndex.tcl]] && [lsearch -exact $::auto_path $appinit_dir] == -1} {
    lappend ::auto_path $appinit_dir
}
catch {package require ::tclapp::support::appinit 1.2} appinit_msg
puts "TCLSTORE_APPINIT_CHECK dir=$appinit_dir msg=$appinit_msg"
puts "TCLSTORE_AUTOPATH_PATCHED root=$tclstore_root"

if {[info exists ::env(PHC_06S2_PHASE_ROOT)] && ![string equal $::env(PHC_06S2_PHASE_ROOT) ""]} {
    set phase_root_raw $::env(PHC_06S2_PHASE_ROOT)
    regsub -all {\\} $phase_root_raw {/} phase_root_raw
    set phase_root $phase_root_raw
} else {
    set script_path_raw [info script]
    regsub -all {\\} $script_path_raw {/} script_path_raw
    set script_path $script_path_raw
    set phase_root [file join [file dirname $script_path] ..]
}
set marker [file join $phase_root results XSIM_POSITIVE_AND_NEGATIVE_PASS.marker]

if {![file exists $marker]} {
    error "XSim gate has not passed. Run positive and negative XSim first, then create $marker deliberately."
}

if {![info exists ::env(PHC_CORE_RTL_DIR)] || [string equal $::env(PHC_CORE_RTL_DIR) ""]} {
    error "PHC_CORE_RTL_DIR is not set. Point it at the complete RTL directory for the exact final PHC core before P&R."
}

set work_root [file join $phase_root vivado_pnr_work]
file mkdir $work_root

set phc_verilog_dir_raw $::env(PHC_CORE_RTL_DIR)
regsub -all {\\} $phc_verilog_dir_raw {/} phc_verilog_dir_raw
set phc_verilog_dir $phc_verilog_dir_raw
set phc_rtl_files [glob -nocomplain [file join $phc_verilog_dir *.v]]
set phc_dat_files [glob -nocomplain [file join $phc_verilog_dir *.dat]]
set wrapper_rtl [file join $phase_root rtl pynq_z2_phc_demo_wrapper.sv]
set clock_rtl [file join $phase_root rtl pynq_z2_clock_90mhz.sv]
set board_top_rtl [file join $phase_root rtl pynq_z2_phc_demo_board_top.sv]
set xdc_file [file join $phase_root xdc pynq_z2_demo_wrapper_board.xdc]
set bram_mem [file join $phase_root bram demo_replay_x_readout_q24_12.mem]
set enable_ila 0
if {[info exists ::env(PHC_ENABLE_ILA)] && [string equal $::env(PHC_ENABLE_ILA) "1"]} {
    set enable_ila 1
}

create_project -force phc_demo_wrapper_pnr $work_root -part xc7z020clg400-1
add_files -fileset sources_1 $phc_rtl_files
add_files -fileset sources_1 $wrapper_rtl
add_files -fileset sources_1 $clock_rtl
add_files -fileset sources_1 $board_top_rtl
add_files -fileset sources_1 $bram_mem
add_files -fileset constrs_1 $xdc_file
foreach dat $phc_dat_files {
    add_files -fileset sources_1 $dat
}

set verilog_defines {PHC_HAS_STATE_PREV_PORT1}
if {$enable_ila} {
    create_ip -name ila -vendor xilinx.com -library ip -module_name phc_demo_ila
    set_property -dict [list \
        CONFIG.C_NUM_OF_PROBES {12} \
        CONFIG.C_DATA_DEPTH {1024} \
        CONFIG.C_PROBE0_WIDTH {32} \
        CONFIG.C_PROBE1_WIDTH {1} \
        CONFIG.C_PROBE2_WIDTH {1} \
        CONFIG.C_PROBE3_WIDTH {1} \
        CONFIG.C_PROBE4_WIDTH {32} \
        CONFIG.C_PROBE5_WIDTH {32} \
        CONFIG.C_PROBE6_WIDTH {24} \
        CONFIG.C_PROBE7_WIDTH {24} \
        CONFIG.C_PROBE8_WIDTH {1} \
        CONFIG.C_PROBE9_WIDTH {1} \
        CONFIG.C_PROBE10_WIDTH {1} \
        CONFIG.C_PROBE11_WIDTH {1} \
    ] [get_ips phc_demo_ila]
    generate_target all [get_ips phc_demo_ila]
    lappend verilog_defines PHC_ENABLE_ILA
}

set_property top pynq_z2_phc_demo_board_top [current_fileset]
set_property verilog_define $verilog_defines [get_filesets sources_1]
set_property generic "MEM_FILE=$bram_mem" [get_filesets sources_1]
update_compile_order -fileset sources_1

puts "WRAPPER_PNR_FLOW direct_synth_design_no_launch_runs"
synth_design -top pynq_z2_phc_demo_board_top -part xc7z020clg400-1 -verilog_define $verilog_defines -generic "MEM_FILE=$bram_mem"
write_checkpoint -force [file join $phase_root results wrapper_synth.dcp]
report_utilization -file [file join $phase_root results wrapper_synth_utilization.rpt]

opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force [file join $phase_root results wrapper_routed.dcp]
report_timing_summary -file [file join $phase_root results wrapper_timing_summary_routed.rpt]
report_utilization -file [file join $phase_root results wrapper_utilization_routed.rpt]
write_bitstream -force [file join $phase_root results phc_demo_wrapper.bit]
if {$enable_ila} {
    write_debug_probes -force [file join $phase_root results phc_demo_wrapper.ltx]
}

puts "WRAPPER_PNR_DONE bitstream=[file join $phase_root results phc_demo_wrapper.bit]"
