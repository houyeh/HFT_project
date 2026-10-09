if {[info exists ::env(PROJECT_ROOT)] && $::env(PROJECT_ROOT) ne ""} {
    set repo_root [string map {\\ /} $::env(PROJECT_ROOT)]
} else {
    set repo_root [string map {\\ /} [pwd]]
}
set phase_root [file join $repo_root n2ew_era 07k_tiny_mlp_parallel_board]
set result_root [file join $phase_root results board_100mhz]
set bit_file [file join $result_root tiny_mlp_parallel_100mhz.bit]
set ltx_file [file join $result_root tiny_mlp_parallel_100mhz.ltx]
foreach required [list $bit_file $ltx_file] {
    if {![file exists $required]} { error "Missing programming artifact: $required" }
}

open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target
set devices [get_hw_devices *xc7z020*]
if {[llength $devices] == 0} {
    puts "TINY_MLP_BOARD_STATUS no_xc7z020_device_detected"
    close_hw_manager
    error "No xc7z020 device detected"
}
set dev [lindex $devices 0]
current_hw_device $dev
refresh_hw_device -update_hw_probes false $dev
set_property PROGRAM.FILE $bit_file $dev
set_property PROBES.FILE $ltx_file $dev
program_hw_devices $dev
refresh_hw_device -update_hw_probes true $dev
after 2000

set vios [get_hw_vios]
puts "TINY_MLP_BOARD_DEVICE $dev"
puts "TINY_MLP_BOARD_VIO_COUNT [llength $vios]"
if {[llength $vios] == 0} {
    close_hw_manager
    error "No VIO core found"
}
foreach vio $vios {
    refresh_hw_vio $vio
    foreach probe [lsort [get_hw_probes -of_objects $vio]] {
        set value "NA"
        catch {set value [get_property INPUT_VALUE $probe]}
        puts "TINY_MLP_BOARD_PROBE name=$probe value=$value"
    }
}
puts "TINY_MLP_BOARD_STATUS programmed_and_read"
close_hw_manager
