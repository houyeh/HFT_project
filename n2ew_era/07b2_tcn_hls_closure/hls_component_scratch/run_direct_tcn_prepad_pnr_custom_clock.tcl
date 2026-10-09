# Direct non-project Vivado P&R for 07B2 TCN PrePad scratch.
#
# Usage from this component directory after Vitis HLS synthesis:
#   vivado -mode batch -source run_direct_tcn_prepad_pnr_custom_clock.tcl \
#          -tclargs <run_name> <clock_period_ns> [rtl_dir]

if { [llength $argv] < 2 } {
    error "Usage: run_direct_tcn_prepad_pnr_custom_clock.tcl <run_name> <clock_period_ns> [rtl_dir]"
}

set run_name [lindex $argv 0]
set clock_period_ns [lindex $argv 1]
if { [llength $argv] >= 3 } {
    set rtl_dir [lindex $argv 2]
} else {
    set rtl_dir "./hls_component/hls/syn/verilog"
}

set top_module tcn_prepad_infer
set target_device xc7z020clg400-1
set report_dir [file join $rtl_dir direct_tcn_prepad_reports_${run_name}]
set checkpoint_dir [file join $rtl_dir direct_tcn_prepad_checkpoints_${run_name}]

file mkdir $report_dir
file mkdir $checkpoint_dir

puts "DIRECT_TCN_PREPAD_PNR: run=$run_name top=$top_module part=$target_device clock_period_ns=$clock_period_ns"
puts "DIRECT_TCN_PREPAD_PNR: reading Verilog RTL from $rtl_dir"

set rtl_files [glob -nocomplain [file join $rtl_dir *.v]]
if { [llength $rtl_files] == 0 } {
    error "No Verilog RTL files found in $rtl_dir"
}
read_verilog $rtl_files

set half_period [expr {$clock_period_ns / 2.0}]
set custom_clock_xdc [file join $checkpoint_dir tcn_prepad_custom_clock.xdc]
set fh [open $custom_clock_xdc w]
puts $fh "create_clock -name ap_clk -period $clock_period_ns -waveform \{0.000 $half_period\} \[get_ports ap_clk\]"
close $fh
read_xdc $custom_clock_xdc
puts "DIRECT_TCN_PREPAD_PNR: loaded custom clock xdc=$custom_clock_xdc"

puts "DIRECT_TCN_PREPAD_PNR: synth_design"
synth_design -top $top_module -part $target_device -mode out_of_context -directive sdx_optimization_effort_high
write_checkpoint -force [file join $checkpoint_dir tcn_prepad_infer_synth.dcp]
report_utilization -file [file join $report_dir tcn_prepad_infer_utilization_synth.rpt]
report_timing_summary -max_paths 10 -report_unconstrained -file [file join $report_dir tcn_prepad_infer_timing_summary_synth.rpt]

puts "DIRECT_TCN_PREPAD_PNR: opt_design"
opt_design
write_checkpoint -force [file join $checkpoint_dir tcn_prepad_infer_opt.dcp]

puts "DIRECT_TCN_PREPAD_PNR: place_design"
place_design
write_checkpoint -force [file join $checkpoint_dir tcn_prepad_infer_placed.dcp]
report_utilization -file [file join $report_dir tcn_prepad_infer_utilization_placed.rpt]
report_timing_summary -max_paths 10 -report_unconstrained -file [file join $report_dir tcn_prepad_infer_timing_summary_placed.rpt]

puts "DIRECT_TCN_PREPAD_PNR: phys_opt_design"
phys_opt_design
write_checkpoint -force [file join $checkpoint_dir tcn_prepad_infer_physopt.dcp]

puts "DIRECT_TCN_PREPAD_PNR: route_design"
route_design
write_checkpoint -force [file join $checkpoint_dir tcn_prepad_infer_routed.dcp]

puts "DIRECT_TCN_PREPAD_PNR: post-route reports"
report_route_status -file [file join $report_dir tcn_prepad_infer_route_status.rpt]
report_drc -file [file join $report_dir tcn_prepad_infer_drc_routed.rpt]
report_utilization -file [file join $report_dir tcn_prepad_infer_utilization_routed.rpt]
report_timing_summary -max_paths 10 -routable_nets -report_unconstrained -warn_on_violation -file [file join $report_dir tcn_prepad_infer_timing_summary_routed.rpt]
report_timing -max_paths 10 -sort_by group -file [file join $report_dir tcn_prepad_infer_timing_paths_routed.rpt]
report_power -file [file join $report_dir tcn_prepad_infer_power_routed.rpt]

puts "DIRECT_TCN_PREPAD_PNR: complete run=$run_name report_dir=$report_dir"
