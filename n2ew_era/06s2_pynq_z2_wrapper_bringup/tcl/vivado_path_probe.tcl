puts "pwd=[pwd]"
if {[info exists ::env(PHC_06S2_PHASE_ROOT)]} {
    puts "env_phase=$::env(PHC_06S2_PHASE_ROOT)"
    puts "norm_env=[file normalize $::env(PHC_06S2_PHASE_ROOT)]"
} else {
    puts "env_phase=<missing>"
}
puts "literal=[file normalize {C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06s2_pynq_z2_wrapper_bringup}]"
puts "exists_marker=[file exists {C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06s2_pynq_z2_wrapper_bringup/results/XSIM_POSITIVE_AND_NEGATIVE_PASS.marker}]"
