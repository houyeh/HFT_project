# Program the V3 diagonal VIO bitstream and read final replay statistics.

set script_path [string map {\\ /} [info script]]
set phase_root [file normalize [file join [file dirname $script_path] ..]]
set negative_test 0
if {[info exists ::env(PHC_NEGATIVE_TEST)] && [string equal $::env(PHC_NEGATIVE_TEST) "1"]} {
    set negative_test 1
}
if {$negative_test} {
    set result_root [file join $phase_root results_v3_diag pnr_100mhz_readback_vio_negative_postroute_physopt]
    set artifact_stem phc_v3diag_demo_100mhz_readback_vio_negative_postroute_physopt
} else {
    set result_root [file join $phase_root results_v3_diag pnr_100mhz_readback_vio_postroute_physopt]
    set artifact_stem phc_v3diag_demo_100mhz_readback_vio_postroute_physopt
}
set bit_file [file join $result_root ${artifact_stem}.bit]
set ltx_file [file join $result_root ${artifact_stem}.ltx]

foreach required [list $bit_file $ltx_file] {
    if {![file exists $required]} {
        error "Missing required programming artifact: $required"
    }
}

open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target
set devices [get_hw_devices *xc7z020*]
if {[llength $devices] == 0} {
    puts "V3DIAG_PROGRAM_STATUS no_xc7z020_device_detected"
    close_hw_manager
    error "No xc7z020 device detected"
}

set dev [lindex $devices 0]
current_hw_device $dev
refresh_hw_device -update_hw_probes false $dev
set_property PROGRAM.FILE $bit_file $dev
set_property PROBES.FILE $ltx_file $dev
puts "V3DIAG_PROGRAM_DEVICE $dev"
puts "V3DIAG_PROGRAM_BITSTREAM $bit_file"
puts "V3DIAG_PROGRAM_PROBES $ltx_file"
program_hw_devices $dev
refresh_hw_device -update_hw_probes true $dev
after 2000

set vios [get_hw_vios]
puts "V3DIAG_HW_VIO_COUNT [llength $vios]"
if {[llength $vios] == 0} {
    close_hw_manager
    error "No VIO core discovered after programming"
}

foreach vio $vios {
    refresh_hw_vio $vio
    puts "V3DIAG_HW_VIO $vio"
    foreach probe [lsort [get_hw_probes -of_objects $vio]] {
        set value "NA"
        catch {set value [get_property INPUT_VALUE $probe]}
        puts "V3DIAG_HW_PROBE name=$probe value=$value"
    }
}
puts "V3DIAG_PROGRAM_STATUS programmed_and_read"
close_hw_manager
