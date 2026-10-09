# PYNQ-Z2 hardware presence check only. This does not program the board.
#
# Usage:
#   vivado -mode batch -source n2ew_era/06s2_pynq_z2_wrapper_bringup/tcl/check_pynq_z2_hw.tcl

proc phc_add_tclstore_pkg_dirs {dir} {
    if {![file isdirectory $dir]} {
        return
    }
    if {[file exists [file join $dir pkgIndex.tcl]] && [lsearch -exact $::auto_path $dir] == -1} {
        lappend ::auto_path $dir
    }
    foreach child [glob -nocomplain -directory $dir *] {
        if {[file isdirectory $child]} {
            phc_add_tclstore_pkg_dirs $child
        }
    }
}

if {[info exists ::env(XILINX_VIVADO)] && ![string equal $::env(XILINX_VIVADO) ""]} {
    set vivado_root_raw $::env(XILINX_VIVADO)
} else {
    set vivado_root_raw "C:/AMDDesignTools/2025.2/Vivado"
}
regsub -all {\\} $vivado_root_raw {/} vivado_root_raw
set tclstore_root [file join $vivado_root_raw data XilinxTclStore]
phc_add_tclstore_pkg_dirs $tclstore_root
set appinit_dir [file join $vivado_root_raw data XilinxTclStore support appinit]
if {[file exists [file join $appinit_dir pkgIndex.tcl]] && [lsearch -exact $::auto_path $appinit_dir] == -1} {
    lappend ::auto_path $appinit_dir
}
catch {package require ::tclapp::support::appinit 1.2} appinit_msg
puts "TCLSTORE_APPINIT_CHECK dir=$appinit_dir msg=$appinit_msg"

open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target

set devices [get_hw_devices]
puts "HW_DEVICES_BEGIN"
foreach dev $devices {
    puts $dev
}
puts "HW_DEVICES_END"

if {[llength $devices] == 0} {
    puts "PYNQ_Z2_DETECTION_STATUS: no_hw_device_detected"
} else {
    puts "PYNQ_Z2_DETECTION_STATUS: hw_device_present"
}

close_hw_manager
