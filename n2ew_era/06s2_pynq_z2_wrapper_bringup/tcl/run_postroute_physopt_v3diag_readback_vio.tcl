# One bounded post-route physical-optimization attempt for the V3 readback build.

proc slash_path {p} {
    return [string map {\\ /} $p]
}

if {[info exists ::env(PHC_06S2_PHASE_ROOT)] && ![string equal $::env(PHC_06S2_PHASE_ROOT) ""]} {
    set phase_root [slash_path $::env(PHC_06S2_PHASE_ROOT)]
} else {
    set script_path [slash_path [info script]]
    set phase_root [file join [file dirname $script_path] ..]
}

set negative_test 0
if {[info exists ::env(PHC_NEGATIVE_TEST)] && [string equal $::env(PHC_NEGATIVE_TEST) "1"]} {
    set negative_test 1
}

if {$negative_test} {
    set input_root [file join $phase_root results_v3_diag pnr_100mhz_readback_vio_negative]
    set output_root [file join $phase_root results_v3_diag pnr_100mhz_readback_vio_negative_postroute_physopt]
    set artifact_stem phc_v3diag_demo_100mhz_readback_vio_negative_postroute_physopt
} else {
    set input_root [file join $phase_root results_v3_diag pnr_100mhz_readback_vio]
    set output_root [file join $phase_root results_v3_diag pnr_100mhz_readback_vio_postroute_physopt]
    set artifact_stem phc_v3diag_demo_100mhz_readback_vio_postroute_physopt
}
set input_dcp [file join $input_root v3diag_readback_routed.dcp]
if {![file exists $input_dcp]} {
    error "Missing routed checkpoint: $input_dcp"
}
file mkdir $output_root

open_checkpoint $input_dcp
set before_path [lindex [get_timing_paths -delay_type max -max_paths 1] 0]
puts "STAGE_E_POSTROUTE_BEFORE WNS=[get_property SLACK $before_path]"

phys_opt_design -directive AggressiveExplore
route_design

set worst_path [lindex [get_timing_paths -delay_type max -max_paths 1] 0]
set wns [get_property SLACK $worst_path]
set failing_paths [get_timing_paths -delay_type max -slack_lesser_than 0 -max_paths 10000]
set tns 0.0
foreach path $failing_paths {
    set tns [expr {$tns + double([get_property SLACK $path])}]
}
puts "STAGE_E_POSTROUTE_AFTER WNS=$wns TNS=$tns failing_endpoints=[llength $failing_paths]"

write_checkpoint -force [file join $output_root v3diag_readback_postroute_physopt_routed.dcp]
report_route_status -file [file join $output_root v3diag_readback_postroute_physopt_route_status.rpt]
report_drc -file [file join $output_root v3diag_readback_postroute_physopt_drc.rpt]
report_utilization -file [file join $output_root v3diag_readback_postroute_physopt_utilization.rpt]
report_timing_summary -max_paths 20 -report_unconstrained -warn_on_violation \
    -file [file join $output_root v3diag_readback_postroute_physopt_timing_summary.rpt]

if {$wns < 0.0 || $tns < -0.0005} {
    error "Post-route physical optimization did not close 100 MHz timing"
}

set bit_path [file join $output_root ${artifact_stem}.bit]
set ltx_path [file join $output_root ${artifact_stem}.ltx]
write_bitstream -force $bit_path
write_debug_probes -force $ltx_path
puts "STAGE_E_POSTROUTE_PASS bitstream=$bit_path probes=$ltx_path"
close_design
