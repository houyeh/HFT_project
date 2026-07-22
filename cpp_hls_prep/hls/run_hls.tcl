set script_dir [file normalize [file dirname [info script]]]
set repo_root [file normalize [file join $script_dir .. ..]]
set cpp_root [file normalize [file join $repo_root cpp_hls_prep]]
set include_dir [file normalize [file join $cpp_root include]]
set project_dir [file normalize [file join $script_dir project]]

set top_name "tcn_fixed_top"
set solution_name "solution1"
set part_name "xczu7ev-ffvc1156-2-e"
if {[info exists ::env(HLS_PART)] && $::env(HLS_PART) ne ""} {
    set part_name $::env(HLS_PART)
}

set clock_period "5.0"
if {[info exists ::env(HLS_CLOCK_NS)] && $::env(HLS_CLOCK_NS) ne ""} {
    set clock_period $::env(HLS_CLOCK_NS)
}

set run_stage "full"
if {[info exists ::env(HLS_STAGE)] && $::env(HLS_STAGE) ne ""} {
    set run_stage $::env(HLS_STAGE)
}

open_project -reset [file join $project_dir fixed_tcn_hls]
set_top $top_name

set cflags "-I$include_dir"
add_files [file join $cpp_root src fixed_tcn_modules.cpp] -cflags $cflags
add_files [file join $cpp_root src fixed_tcn.cpp] -cflags $cflags
add_files [file join $cpp_root src fixed_tcn_top.cpp] -cflags $cflags
add_files -tb [file join $cpp_root src hls_testbench.cpp] -cflags $cflags

open_solution -reset $solution_name -flow_target vivado
set_part $part_name
create_clock -period $clock_period -name default

if {$run_stage eq "csim"} {
    csim_design
} elseif {$run_stage eq "csynth"} {
    csynth_design
} elseif {$run_stage eq "full"} {
    csim_design
    csynth_design
    export_design -format ip_catalog
} else {
    puts "Unsupported HLS_STAGE='$run_stage'. Use csim, csynth, or full."
    exit 1
}

exit
