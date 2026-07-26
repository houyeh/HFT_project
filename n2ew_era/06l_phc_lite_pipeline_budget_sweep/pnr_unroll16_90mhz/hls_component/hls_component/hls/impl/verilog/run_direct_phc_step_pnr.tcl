# Direct non-project Vivado P&R for the 06L unroll-16 PHC-lite RTL.
#
# This script intentionally avoids Vivado managed runs because the generated
# run launcher (rundef.js) is blocked by Windows policy in this environment.
# It does not modify HLS source, pragmas, interfaces, or clock constraints.

set top_module phc_step
set target_device xc7z020clg400-1
set report_dir ./direct_phc_step_reports
set checkpoint_dir ./direct_phc_step_checkpoints

file mkdir $report_dir
file mkdir $checkpoint_dir

puts "DIRECT_PNR: top=$top_module part=$target_device"
puts "DIRECT_PNR: reading Verilog RTL from [pwd]"

set rtl_files [glob -nocomplain ./*.v]
if { [llength $rtl_files] == 0 } {
    error "No Verilog RTL files found in [pwd]"
}
read_verilog $rtl_files

if { ![file exists ./phc_step.xdc] } {
    error "Missing phc_step.xdc"
}
read_xdc ./phc_step.xdc

puts "DIRECT_PNR: synth_design"
synth_design -top $top_module -part $target_device -mode out_of_context -directive sdx_optimization_effort_high
write_checkpoint -force $checkpoint_dir/phc_step_synth.dcp
report_utilization -file $report_dir/phc_step_utilization_synth.rpt
report_timing_summary -max_paths 10 -report_unconstrained -file $report_dir/phc_step_timing_summary_synth.rpt

puts "DIRECT_PNR: opt_design"
opt_design
write_checkpoint -force $checkpoint_dir/phc_step_opt.dcp

puts "DIRECT_PNR: place_design"
place_design
write_checkpoint -force $checkpoint_dir/phc_step_placed.dcp
report_utilization -file $report_dir/phc_step_utilization_placed.rpt
report_timing_summary -max_paths 10 -report_unconstrained -file $report_dir/phc_step_timing_summary_placed.rpt

puts "DIRECT_PNR: phys_opt_design"
phys_opt_design
write_checkpoint -force $checkpoint_dir/phc_step_physopt.dcp

puts "DIRECT_PNR: route_design"
route_design
write_checkpoint -force $checkpoint_dir/phc_step_routed.dcp

puts "DIRECT_PNR: post-route reports"
report_route_status -file $report_dir/phc_step_route_status.rpt
report_drc -file $report_dir/phc_step_drc_routed.rpt
report_utilization -file $report_dir/phc_step_utilization_routed.rpt
report_timing_summary -max_paths 10 -routable_nets -report_unconstrained -warn_on_violation -file $report_dir/phc_step_timing_summary_routed.rpt
report_timing -max_paths 10 -sort_by group -file $report_dir/phc_step_timing_paths_routed.rpt
report_power -file $report_dir/phc_step_power_routed.rpt

puts "DIRECT_PNR: complete"
