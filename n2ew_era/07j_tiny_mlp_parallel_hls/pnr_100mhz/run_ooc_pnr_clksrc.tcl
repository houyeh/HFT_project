set script_dir [file dirname [file normalize [info script]]]
set phase_dir [file dirname $script_dir]
set rtl_dir [file join $phase_dir hls_component tiny_mlp_parallel_csynth_work hls syn verilog]
set out_dir [file join $script_dir output_clksrc]
file mkdir $out_dir

create_project -in_memory -part xc7z020clg400-1
set rtl_files [glob -nocomplain [file join $rtl_dir *.v]]
if {[llength $rtl_files] == 0} {
    error "No generated Verilog files found in $rtl_dir"
}
read_verilog $rtl_files

synth_design -mode out_of_context -top tiny_mlp_parallel -part xc7z020clg400-1
create_clock -name ap_clk -period 10.000 [get_ports ap_clk]
set_property HD.CLK_SRC BUFGCTRL_X0Y0 [get_ports ap_clk]

opt_design
place_design
phys_opt_design
route_design

report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
    -max_paths 20 -input_pins -file [file join $out_dir timing_summary_routed.rpt]
report_timing -delay_type max -max_paths 20 -sort_by group \
    -path_type full_clock_expanded -input_pins \
    -file [file join $out_dir critical_paths_routed.rpt]
report_utilization -hierarchical -file [file join $out_dir utilization_routed.rpt]
write_checkpoint -force [file join $out_dir tiny_mlp_parallel_routed_clksrc.dcp]

set worst_path [get_timing_paths -delay_type max -max_paths 1]
set wns [get_property SLACK $worst_path]
set achieved_period [expr {10.000 - $wns}]
set fh [open [file join $out_dir routed_summary.txt] w]
puts $fh "target_part=xc7z020clg400-1"
puts $fh "target_clock_period_ns=10.000"
puts $fh "hd_clk_src=BUFGCTRL_X0Y0"
puts $fh "wns_ns=$wns"
puts $fh "achieved_period_ns=$achieved_period"
puts $fh "derived_latency_cycles=18"
puts $fh "derived_latency_us=0.180"
puts $fh "verification_tier=core_level_routed"
close $fh

puts "OOC_PNR_CLKSRC_COMPLETE WNS_NS=$wns ACHIEVED_PERIOD_NS=$achieved_period"
exit
