# P&R/bitstream flow for the 06S2 HDMI numeric display variants.
#
# Usage:
#   vivado -mode batch -source tcl/run_pnr_hdmi_numeric_variant.tcl -tclargs numeric_test
#   vivado -mode batch -source tcl/run_pnr_hdmi_numeric_variant.tcl -tclargs numeric_test_regpipe
#   vivado -mode batch -source tcl/run_pnr_hdmi_numeric_variant.tcl -tclargs numeric_test_regpipe_vga
#   vivado -mode batch -source tcl/run_pnr_hdmi_numeric_variant.tcl -tclargs numeric_test_regpipe_bufio
#   vivado -mode batch -source tcl/run_pnr_hdmi_numeric_variant.tcl -tclargs phc_numeric
#
# This script is intentionally separate from run_pnr_hdmi_variant.tcl so the
# verified solid-color HDMI flow and bitstreams remain untouched.

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

if {[llength $argv] < 1} {
    error "Expected variant argument: numeric_test or phc_numeric"
}
set variant [lindex $argv 0]
if {![string equal $variant "numeric_test"] && ![string equal $variant "numeric_test_regpipe"] && ![string equal $variant "numeric_test_regpipe_vga"] && ![string equal $variant "numeric_test_regpipe_bufio"] && ![string equal $variant "phc_numeric"]} {
    error "Unsupported HDMI numeric variant '$variant'. Expected numeric_test, numeric_test_regpipe, numeric_test_regpipe_vga, numeric_test_regpipe_bufio, or phc_numeric."
}

if {[info exists ::env(PHC_06S2_PHASE_ROOT)] && ![string equal $::env(PHC_06S2_PHASE_ROOT) ""]} {
    set phase_root_raw $::env(PHC_06S2_PHASE_ROOT)
    regsub -all {\\} $phase_root_raw {/} phase_root_raw
    set phase_root $phase_root_raw
} else {
    set script_path_raw [info script]
    regsub -all {\\} $script_path_raw {/} script_path_raw
    set phase_root [file join [file dirname $script_path_raw] ..]
}

set work_root [file join $phase_root vivado_pnr_work_hdmi_$variant]
file mkdir $work_root
file mkdir [file join $phase_root results]

set rgb2dvi_src [file join $phase_root third_party digilent_rgb2dvi src]
set vhdl_files [list \
    [file join $rgb2dvi_src DVI_Constants.vhd] \
    [file join $rgb2dvi_src SyncAsync.vhd] \
    [file join $rgb2dvi_src SyncAsyncReset.vhd] \
    [file join $rgb2dvi_src OutputSERDES.vhd] \
    [file join $rgb2dvi_src TMDS_Encoder.vhd] \
    [file join $rgb2dvi_src rgb2dvi.vhd] \
    [file join $phase_root rtl rgb2dvi_fixed.vhd] \
]

set common_sv_files [list \
    [file join $phase_root rtl pynq_z2_clock_hdmi_640x480.sv] \
    [file join $phase_root rtl pynq_z2_hdmi_timing_640x480.sv] \
    [file join $phase_root rtl pynq_z2_hdmi_numeric_overlay.sv] \
]

set xdc_file [file join $phase_root xdc pynq_z2_hdmi_demo_board.xdc]
set verilog_defines {}
set synth_generics {}

if {[string equal $variant "numeric_test"]} {
    set top pynq_z2_hdmi_numeric_test_top
    set project_name phc_demo_hdmi_numeric_test_pnr
    set result_prefix hdmi_numeric_test
    set sv_files [concat $common_sv_files [list [file join $phase_root rtl pynq_z2_hdmi_numeric_test_top.sv]]]
} elseif {[string equal $variant "numeric_test_regpipe"]} {
    set top pynq_z2_hdmi_numeric_test_top_regpipe
    set project_name phc_demo_hdmi_numeric_test_regpipe_pnr
    set result_prefix hdmi_numeric_test_regpipe
    set sv_files [concat $common_sv_files [list [file join $phase_root rtl pynq_z2_hdmi_numeric_test_top_regpipe.sv]]]
} elseif {[string equal $variant "numeric_test_regpipe_vga"]} {
    set top pynq_z2_hdmi_numeric_test_top_regpipe_vga
    set project_name phc_demo_hdmi_numeric_test_regpipe_vga_pnr
    set result_prefix hdmi_numeric_test_regpipe_vga
    set sv_files [concat \
        $common_sv_files \
        [list \
            [file join $phase_root rtl pynq_z2_clock_hdmi_640x480_vga.sv] \
            [file join $phase_root rtl pynq_z2_hdmi_numeric_test_top_regpipe_vga.sv] \
        ] \
    ]
} elseif {[string equal $variant "numeric_test_regpipe_bufio"]} {
    set top pynq_z2_hdmi_numeric_test_top_regpipe_bufio
    set project_name phc_demo_hdmi_numeric_test_regpipe_bufio_pnr
    set result_prefix hdmi_numeric_test_regpipe_bufio
    set sv_files [concat \
        $common_sv_files \
        [list \
            [file join $phase_root rtl pynq_z2_clock_hdmi_640x480_bufio.sv] \
            [file join $phase_root rtl pynq_z2_hdmi_numeric_test_top_regpipe_bufio.sv] \
        ] \
    ]
} else {
    set marker [file join $phase_root results XSIM_POSITIVE_AND_NEGATIVE_PASS.marker]
    if {![file exists $marker]} {
        error "XSim gate has not passed. Missing $marker"
    }
    if {![info exists ::env(PHC_CORE_RTL_DIR)] || [string equal $::env(PHC_CORE_RTL_DIR) ""]} {
        error "PHC_CORE_RTL_DIR is not set for HDMI PHC numeric variant."
    }
    set phc_verilog_dir_raw $::env(PHC_CORE_RTL_DIR)
    regsub -all {\\} $phc_verilog_dir_raw {/} phc_verilog_dir_raw
    set phc_verilog_dir $phc_verilog_dir_raw
    set phc_rtl_files [glob -nocomplain [file join $phc_verilog_dir *.v]]
    set phc_dat_files [glob -nocomplain [file join $phc_verilog_dir *.dat]]
    set bram_mem [file join $phase_root bram demo_replay_x_readout_q24_12.mem]
    set top pynq_z2_phc_demo_board_top_hdmi_numeric
    set project_name phc_demo_hdmi_numeric_pnr
    set result_prefix phc_demo_wrapper_hdmi_numeric
    set sv_files [concat \
        $phc_rtl_files \
        $common_sv_files \
        [list \
            [file join $phase_root rtl pynq_z2_clock_90mhz.sv] \
            [file join $phase_root rtl pynq_z2_phc_demo_wrapper.sv] \
            [file join $phase_root rtl pynq_z2_phc_demo_board_top_hdmi_numeric.sv] \
            $bram_mem \
        ] \
        $phc_dat_files \
    ]
    set verilog_defines {PHC_HAS_STATE_PREV_PORT1}
    set synth_generics [list MEM_FILE=$bram_mem]
}

create_project -force $project_name $work_root -part xc7z020clg400-1
add_files -fileset sources_1 $vhdl_files
add_files -fileset sources_1 $sv_files
add_files -fileset constrs_1 $xdc_file
set_property top $top [current_fileset]
if {[llength $verilog_defines] > 0} {
    set_property verilog_define $verilog_defines [get_filesets sources_1]
}
if {[llength $synth_generics] > 0} {
    set_property generic $synth_generics [get_filesets sources_1]
}
update_compile_order -fileset sources_1

puts "HDMI_NUMERIC_PNR_FLOW variant=$variant top=$top direct_synth_design_no_launch_runs"
if {[llength $verilog_defines] > 0 && [llength $synth_generics] > 0} {
    synth_design -top $top -part xc7z020clg400-1 -verilog_define $verilog_defines -generic $synth_generics
} elseif {[llength $verilog_defines] > 0} {
    synth_design -top $top -part xc7z020clg400-1 -verilog_define $verilog_defines
} elseif {[llength $synth_generics] > 0} {
    synth_design -top $top -part xc7z020clg400-1 -generic $synth_generics
} else {
    synth_design -top $top -part xc7z020clg400-1
}
write_checkpoint -force [file join $phase_root results ${result_prefix}_synth.dcp]
report_utilization -file [file join $phase_root results ${result_prefix}_synth_utilization.rpt]

if {[string equal $variant "phc_numeric"]} {
    # The numeric overlay consumes display-only status/readout/index snapshots
    # synchronized from the 90 MHz replay clock into the 25 MHz pixel clock.
    # Treat those crossings as asynchronous, as in the verified solid-color HDMI
    # status flow.
    set_clock_groups -asynchronous \
        -group [get_clocks clk90_raw] \
        -group [get_clocks clk_pixel_raw]
}

opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force [file join $phase_root results ${result_prefix}_routed.dcp]
report_timing_summary -file [file join $phase_root results ${result_prefix}_timing_summary_routed.rpt]
report_utilization -file [file join $phase_root results ${result_prefix}_utilization_routed.rpt]
write_bitstream -force [file join $phase_root results ${result_prefix}.bit]

puts "HDMI_NUMERIC_PNR_DONE variant=$variant bitstream=[file join $phase_root results ${result_prefix}.bit]"
