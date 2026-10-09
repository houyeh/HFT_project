if {[llength $argv] != 3} {
    error "Usage: run_ooc_route.tcl <mhz> <period_ns> <latency_cycles>"
}
set mhz [lindex $argv 0]
set period [lindex $argv 1]
set latency [lindex $argv 2]
set script_dir [file dirname [file normalize [info script]]]
set rtl_dir [file join $script_dir "hls_hs_${mhz}mhz" hls syn verilog]
set out_dir [file join $script_dir route "${mhz}mhz"]
file mkdir $out_dir

create_project -in_memory -part xc7z020clg400-1
set rtl_files [glob -nocomplain [file join $rtl_dir *.v]]
if {[llength $rtl_files] == 0} { error "No generated Verilog in $rtl_dir" }
read_verilog $rtl_files
synth_design -mode out_of_context -top qat12_scalar_hs_pipeline -part xc7z020clg400-1
create_clock -name ap_clk -period $period [get_ports ap_clk]
set_property HD.CLK_SRC BUFGCTRL_X0Y0 [get_ports ap_clk]
opt_design
place_design
phys_opt_design
route_design
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
    -max_paths 20 -input_pins -file [file join $out_dir timing_summary_routed.rpt]
report_utilization -hierarchical -file [file join $out_dir utilization_routed.rpt]
set worst_path [get_timing_paths -delay_type max -max_paths 1]
set wns [get_property SLACK $worst_path]
set achieved_period [expr {$period - $wns}]
set derived_us [expr {$latency * $period / 1000.0}]
set fh [open [file join $out_dir routed_summary.txt] w]
puts $fh "target_frequency_mhz=$mhz"
puts $fh "target_clock_period_ns=$period"
puts $fh "hd_clk_src=BUFGCTRL_X0Y0"
puts $fh "wns_ns=$wns"
puts $fh "achieved_period_ns=$achieved_period"
puts $fh "derived_latency_cycles=$latency"
puts $fh "derived_latency_us=$derived_us"
close $fh
puts "ROUTE_COMPLETE FREQ_MHZ=$mhz WNS_NS=$wns"
exit
