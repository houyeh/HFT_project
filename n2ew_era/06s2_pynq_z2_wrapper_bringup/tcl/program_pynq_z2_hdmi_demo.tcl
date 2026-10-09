# Program the generated HDMI-status PHC demo bitstream onto a connected PYNQ-Z2.
#
# Usage from the repository root:
#   vivado -mode batch -source n2ew_era/06s2_pynq_z2_wrapper_bringup/tcl/program_pynq_z2_hdmi_demo.tcl

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

if {[info exists ::env(PHC_06S2_PHASE_ROOT)] && ![string equal $::env(PHC_06S2_PHASE_ROOT) ""]} {
    set phase_root_raw $::env(PHC_06S2_PHASE_ROOT)
    regsub -all {\\} $phase_root_raw {/} phase_root_raw
    set phase_root $phase_root_raw
} else {
    set script_path_raw [info script]
    regsub -all {\\} $script_path_raw {/} script_path_raw
    set phase_root [file join [file dirname $script_path_raw] ..]
}

if {[info exists ::env(PHC_06S2_PROGRAM_BIT_FILE)] && ![string equal $::env(PHC_06S2_PROGRAM_BIT_FILE) ""]} {
    set bit_file_raw $::env(PHC_06S2_PROGRAM_BIT_FILE)
    regsub -all {\\} $bit_file_raw {/} bit_file_raw
    set bit_file $bit_file_raw
} else {
    set bit_file [file join $phase_root results phc_demo_wrapper_hdmi.bit]
}
if {![file exists $bit_file]} {
    error "Bitstream not found: $bit_file"
}

open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target

set devices [get_hw_devices *xc7z020*]
if {[llength $devices] == 0} {
    puts "PYNQ_Z2_HDMI_PROGRAM_STATUS: no_xc7z020_device_detected"
    close_hw_manager
    error "No xc7z020 hardware device detected. Check PYNQ-Z2 power, USB/JTAG cable, and drivers."
}

set dev [lindex $devices 0]
current_hw_device $dev
refresh_hw_device -update_hw_probes false $dev
set_property PROGRAM.FILE $bit_file $dev

puts "PYNQ_Z2_HDMI_PROGRAM_DEVICE: $dev"
puts "PYNQ_Z2_HDMI_PROGRAM_BITSTREAM: $bit_file"
program_hw_devices $dev
refresh_hw_device $dev
puts "PYNQ_Z2_HDMI_PROGRAM_STATUS: programmed"

close_hw_manager
