# Direct non-project Vivado P&R for 06L unroll-16 PHC-lite with a custom clock.
#
# This is used for fallback timing closure experiments such as 95 MHz. It reads
# the frozen HLS-generated Verilog RTL and creates the requested clock in Tcl
# instead of reading phc_step.xdc, whose period is fixed at 10ns.
#
# Usage:
#   vivado -mode batch -source run_direct_phc_step_pnr_custom_clock.tcl \
#          -tclargs <run_name> <clock_period_ns>

if { [llength $argv] < 2 } {
    error "Usage: run_direct_phc_step_pnr_custom_clock.tcl <run_name> <clock_period_ns>"
}

set run_name [lindex $argv 0]
set clock_period_ns [lindex $argv 1]
set top_module phc_step
set target_device xc7z020clg400-1
set report_dir ./direct_phc_step_reports_${run_name}
set checkpoint_dir ./direct_phc_step_checkpoints_${run_name}

file mkdir $report_dir
file mkdir $checkpoint_dir

puts "DIRECT_CUSTOM_CLOCK_PNR: run=$run_name top=$top_module part=$target_device clock_period_ns=$clock_period_ns"
puts "DIRECT_CUSTOM_CLOCK_PNR: reading Verilog RTL from [pwd]"

set rtl_files [glob -nocomplain ./*.v]
if { [llength $rtl_files] == 0 } {
    error "No Verilog RTL files found in [pwd]"
}
read_verilog $rtl_files

set half_period [expr {$clock_period_ns / 2.0}]
set custom_clock_xdc [file join $checkpoint_dir phc_step_custom_clock.xdc]
set fh [open $custom_clock_xdc w]
puts $fh "create_clock -name ap_clk -period $clock_period_ns -waveform \{0.000 $half_period\} \[get_ports ap_clk\]"
close $fh
read_xdc $custom_clock_xdc
puts "DIRECT_CUSTOM_CLOCK_PNR: loaded custom clock xdc=$custom_clock_xdc"

puts "DIRECT_CUSTOM_CLOCK_PNR: synth_design"
synth_design -top $top_module -part $target_device -mode out_of_context -directive sdx_optimization_effort_high
write_checkpoint -force $checkpoint_dir/phc_step_synth.dcp
report_utilization -file $report_dir/phc_step_utilization_synth.rpt
report_timing_summary -max_paths 10 -report_unconstrained -file $report_dir/phc_step_timing_summary_synth.rpt

puts "DIRECT_CUSTOM_CLOCK_PNR: opt_design"
opt_design
write_checkpoint -force $checkpoint_dir/phc_step_opt.dcp

puts "DIRECT_CUSTOM_CLOCK_PNR: place_design"
place_design
write_checkpoint -force $checkpoint_dir/phc_step_placed.dcp
report_utilization -file $report_dir/phc_step_utilization_placed.rpt
report_timing_summary -max_paths 10 -report_unconstrained -file $report_dir/phc_step_timing_summary_placed.rpt

puts "DIRECT_CUSTOM_CLOCK_PNR: phys_opt_design"
phys_opt_design
write_checkpoint -force $checkpoint_dir/phc_step_physopt.dcp

puts "DIRECT_CUSTOM_CLOCK_PNR: route_design"
route_design
write_checkpoint -force $checkpoint_dir/phc_step_routed.dcp

puts "DIRECT_CUSTOM_CLOCK_PNR: post-route reports"
report_route_status -file $report_dir/phc_step_route_status.rpt
report_drc -file $report_dir/phc_step_drc_routed.rpt
report_utilization -file $report_dir/phc_step_utilization_routed.rpt
report_timing_summary -max_paths 10 -routable_nets -report_unconstrained -warn_on_violation -file $report_dir/phc_step_timing_summary_routed.rpt
report_timing -max_paths 10 -sort_by group -file $report_dir/phc_step_timing_paths_routed.rpt
report_power -file $report_dir/phc_step_power_routed.rpt

puts "DIRECT_CUSTOM_CLOCK_PNR: complete run=$run_name report_dir=$report_dir"
