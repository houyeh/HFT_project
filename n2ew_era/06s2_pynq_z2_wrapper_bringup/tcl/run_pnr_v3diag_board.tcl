# Board-level P&R for the V3 diagonal replay demo.
#
# Usage:
#   vivado -mode batch -source run_pnr_v3diag_board.tcl -tclargs baseline
# Strategies: baseline, aggressive, fanout

if {[llength $argv] < 1} {
    error "Usage: run_pnr_v3diag_board.tcl <baseline|aggressive|fanout>"
}
set strategy [lindex $argv 0]
if {$strategy ni {baseline aggressive fanout}} {
    error "Unknown strategy: $strategy"
}

proc slash_path {p} {
    return [string map {\\ /} $p]
}

set script_path [slash_path [info script]]
set phase_root [file normalize [file join [file dirname $script_path] ..]]
set repo_root [file normalize [file join $phase_root .. ..]]
set core_rtl_dir [file join $repo_root n2ew_era 06u_phc_latency_breakthrough variants v3_diagonal16_unroll8_localpart_100mhz rtl_export]
set wrapper_rtl [file join $phase_root rtl pynq_z2_phc_demo_wrapper_v3diag.sv]
set clock_rtl [file join $phase_root rtl pynq_z2_clock_100mhz.sv]
set board_top_rtl [file join $phase_root rtl pynq_z2_phc_demo_board_top_v3diag.sv]
set xdc_file [file join $phase_root xdc pynq_z2_demo_wrapper_board.xdc]
set bram_mem [file join $phase_root bram_v3_diag demo_replay_x_readout_q24_12.mem]
set marker [file join $phase_root results_v3_diag XSIM_POSITIVE_AND_NEGATIVE_PASS.marker]
set work_root [file join $phase_root vivado_pnr_work_v3_diag_100mhz $strategy]
set report_root [file join $phase_root results_v3_diag pnr_100mhz_$strategy]

if {![file exists $marker]} {
    error "Missing XSim pass marker: $marker"
}
foreach required [list $wrapper_rtl $clock_rtl $board_top_rtl $xdc_file $bram_mem] {
    if {![file exists $required]} {
        error "Missing required input: $required"
    }
}
set core_v [lsort [glob -nocomplain [file join $core_rtl_dir *.v]]]
set core_dat [lsort [glob -nocomplain [file join $core_rtl_dir *.dat]]]
if {[llength $core_v] != 44 || [llength $core_dat] != 21} {
    error "Expected 44 .v and 21 .dat in $core_rtl_dir; found [llength $core_v] and [llength $core_dat]"
}

file mkdir $work_root
file mkdir $report_root
create_project -force phc_v3diag_board_$strategy $work_root -part xc7z020clg400-1
add_files -fileset sources_1 $core_v
add_files -fileset sources_1 $wrapper_rtl
add_files -fileset sources_1 $clock_rtl
add_files -fileset sources_1 $board_top_rtl
add_files -fileset sources_1 $bram_mem
foreach dat $core_dat {
    add_files -fileset sources_1 $dat
}
add_files -fileset constrs_1 $xdc_file
set_property top pynq_z2_phc_demo_board_top_v3diag [current_fileset]
set_property generic "MEM_FILE=$bram_mem" [get_filesets sources_1]
update_compile_order -fileset sources_1

puts "STAGE_D_INPUT strategy=$strategy core_v=[llength $core_v] core_dat=[llength $core_dat]"
puts "STAGE_D_BRAM replay_readout=1 expected_state=0 mem=$bram_mem"
puts "STAGE_D_CLOCK DIVCLK=5 MULT=40 CLKOUT_DIVIDE=10 output_mhz=100"

synth_design -top pynq_z2_phc_demo_board_top_v3diag -part xc7z020clg400-1 -generic "MEM_FILE=$bram_mem"
write_checkpoint -force [file join $report_root v3diag_board_synth.dcp]
report_utilization -file [file join $report_root v3diag_board_utilization_synth.rpt]

if {$strategy eq "baseline"} {
    opt_design
    place_design
    phys_opt_design
    route_design
} elseif {$strategy eq "aggressive"} {
    opt_design -directive Explore
    place_design -directive ExtraNetDelay_high
    phys_opt_design -directive AggressiveExplore
    route_design -directive AggressiveExplore
} else {
    opt_design -directive Explore
    set high_fanout_nets [get_nets -hier -quiet -filter {FLAT_PIN_COUNT >= 100}]
    if {[llength $high_fanout_nets] > 0} {
        set_property MAX_FANOUT 16 $high_fanout_nets
    }
    puts "STAGE_D_FANOUT nets_with_flat_pin_count_ge_100=[llength $high_fanout_nets]"
    place_design -directive ExtraNetDelay_high
    if {[llength $high_fanout_nets] > 0} {
        catch {phys_opt_design -force_replication_on_nets $high_fanout_nets} replication_msg
        puts "STAGE_D_FANOUT_REPLICATION result=$replication_msg"
    }
    phys_opt_design -directive AggressiveFanoutOpt
    route_design -directive AggressiveExplore
}

write_checkpoint -force [file join $report_root v3diag_board_routed.dcp]
report_route_status -file [file join $report_root v3diag_board_route_status.rpt]
report_drc -file [file join $report_root v3diag_board_drc_routed.rpt]
report_utilization -file [file join $report_root v3diag_board_utilization_routed.rpt]
report_timing_summary -max_paths 20 -report_unconstrained -warn_on_violation -file [file join $report_root v3diag_board_timing_summary_routed.rpt]
report_timing -max_paths 20 -sort_by group -file [file join $report_root v3diag_board_timing_paths_routed.rpt]
report_power -file [file join $report_root v3diag_board_power_routed.rpt]

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
puts "STAGE_D_RESULT strategy=$strategy WNS=$wns TNS=$tns failing_endpoints_capped=[llength $failing_paths]"

if {$wns >= 0.0 && $tns >= -0.0005} {
    set bit_path [file join $report_root phc_v3diag_demo_100mhz_$strategy.bit]
    write_bitstream -force $bit_path
    puts "STAGE_D_TIMING_MET strategy=$strategy bitstream=$bit_path"
} else {
    puts "STAGE_D_TIMING_FAILED strategy=$strategy"
}

close_project
