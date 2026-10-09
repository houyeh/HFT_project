# Probe Vivado Hardware Manager objects relevant to live replay upload/readback.
#
# This script is read-only. It does not program the FPGA, does not write memory,
# and does not modify any hardware state.

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
puts "HW_DEVICE_COUNT [llength $devices]"
foreach dev $devices {
    puts "HW_DEVICE $dev"
}

set zdevs [get_hw_devices *xc7z020*]
if {[llength $zdevs] == 0} {
    puts "PYNQ_Z2_DETECTION_STATUS no_xc7z020_device_detected"
    close_hw_manager
    exit 2
}

set dev [lindex $zdevs 0]
current_hw_device $dev
refresh_hw_device -update_hw_probes true $dev
puts "PYNQ_Z2_DETECTION_STATUS hw_device_present"
puts "CURRENT_HW_DEVICE $dev"
catch {puts "PROGRAM_FILE [get_property PROGRAM.FILE $dev]"} program_file_msg

set object_kinds {
    hw_ilas
    hw_vios
    hw_axis
    hw_mems
    hw_probes
    hw_sio_links
    hw_sio_gts
    hw_sio_plls
}

foreach kind $object_kinds {
    set objs {}
    if {[catch {set objs [eval get_$kind]} err]} {
        puts "OBJECT_KIND $kind QUERY_ERROR $err"
    } else {
        puts "OBJECT_KIND $kind COUNT [llength $objs]"
        foreach obj $objs {
            puts "OBJECT $kind $obj"
        }
    }
}

if {![catch {set probes [get_hw_probes *]}]} {
    foreach probe $probes {
        set name $probe
        set width "NA"
        catch {set width [get_property WIDTH $probe]}
        puts "HW_PROBE_DETAIL name=$name width=$width"
    }
}

close_hw_manager
