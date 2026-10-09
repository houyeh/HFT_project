# Direct Vivado P&R for 07C2D TCN streaming iface-slim RTL.
# Usage:
# vivado -mode batch -source run_direct_tcn_streaming_pnr_custom_clock.tcl \
#        -tclargs <run_name> <clock_period_ns> <rtl_dir>

if {$argc < 3} {
    puts "Usage: run_direct_tcn_streaming_pnr_custom_clock.tcl <run_name> <clock_period_ns> <rtl_dir>"
    exit 1
}

set run_name [lindex $argv 0]
set clock_period_ns [lindex $argv 1]
set rtl_dir [file normalize [lindex $argv 2]]
set top_module tcn_streaming_step
set target_device xc7z020clg400-1

set report_dir [file join $rtl_dir direct_tcn_streaming_reports_${run_name}]
set checkpoint_dir [file join $rtl_dir direct_tcn_streaming_checkpoints_${run_name}]
file mkdir $report_dir
file mkdir $checkpoint_dir

puts "Run name: $run_name"
puts "Clock period ns: $clock_period_ns"
puts "RTL dir: $rtl_dir"
puts "Report dir: $report_dir"

set verilog_files [glob -nocomplain [file join $rtl_dir *.v]]
if {[llength $verilog_files] == 0} {
    puts "ERROR: no Verilog files found in $rtl_dir"
    exit 1
}

read_verilog $verilog_files

set half_period [expr {$clock_period_ns / 2.0}]
set custom_clock_xdc [file join $checkpoint_dir tcn_streaming_custom_clock.xdc]
set fh [open $custom_clock_xdc w]
puts $fh "create_clock -name ap_clk -period $clock_period_ns -waveform \{0.000 $half_period\} \[get_ports ap_clk\]"
close $fh
read_xdc $custom_clock_xdc

synth_design \
    -top $top_module \
    -part $target_device \
    -mode out_of_context \
    -directive sdx_optimization_effort_high

report_utilization -file [file join $report_dir tcn_streaming_step_utilization_synth.rpt]
report_timing_summary -max_paths 10 -report_unconstrained -file [file join $report_dir tcn_streaming_step_timing_summary_synth.rpt]
write_checkpoint -force [file join $checkpoint_dir tcn_streaming_step_synth.dcp]

opt_design
place_design

report_utilization -file [file join $report_dir tcn_streaming_step_utilization_placed.rpt]
report_timing_summary -max_paths 10 -report_unconstrained -file [file join $report_dir tcn_streaming_step_timing_summary_placed.rpt]
write_checkpoint -force [file join $checkpoint_dir tcn_streaming_step_placed.dcp]

phys_opt_design
route_design

report_route_status -file [file join $report_dir tcn_streaming_step_route_status.rpt]
report_drc -file [file join $report_dir tcn_streaming_step_drc_routed.rpt]
report_utilization -file [file join $report_dir tcn_streaming_step_utilization_routed.rpt]
report_timing_summary \
    -max_paths 10 \
    -routable_nets \
    -report_unconstrained \
    -warn_on_violation \
    -file [file join $report_dir tcn_streaming_step_timing_summary_routed.rpt]
report_timing \
    -max_paths 10 \
    -sort_by group \
    -file [file join $report_dir tcn_streaming_step_timing_paths_routed.rpt]
report_power -file [file join $report_dir tcn_streaming_step_power_routed.rpt]
write_checkpoint -force [file join $checkpoint_dir tcn_streaming_step_routed.dcp]

puts "Direct Vivado P&R complete for $run_name"
