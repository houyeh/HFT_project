set script_dir [file dirname [file normalize [info script]]]
set src_dir [file join $script_dir hls_src]

foreach item {{110 9.091} {120 8.333} {125 8.000} {130 7.692} {140 7.143}} {
    set mhz [lindex $item 0]
    set period [lindex $item 1]
    set project [file join $script_dir "hls_hs_${mhz}mhz"]
    open_project -reset $project
    set_top qat12_scalar_hs_pipeline
    add_files [file join $src_dir qat12_scalar_hs_pipeline.cpp]
    add_files [file join $src_dir qat12_stream.hpp]
    add_files [file join $src_dir qat12_stream_impl.hpp]
    add_files [file join $src_dir qat12_weights.hpp]
    open_solution -flow_target vivado hls
    set_part xc7z020-clg400-1
    create_clock -period $period -name default
    config_compile -pipeline_loops 0
    csynth_design
    close_project
}
exit
