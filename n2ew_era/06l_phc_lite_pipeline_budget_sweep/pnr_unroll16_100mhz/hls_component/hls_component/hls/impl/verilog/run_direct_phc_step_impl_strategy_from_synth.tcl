# Direct Vivado implementation strategy runner for 06L unroll-16 PHC-lite.
#
# This starts from the already synthesized phc_step checkpoint produced from
# the frozen HLS RTL, then reruns implementation only. It avoids Vivado managed
# run launchers (rundef.js) and keeps all changes at the implementation /
# constraint level: no HLS C source, pragma, or RTL edits.
#
# Usage:
#   vivado -mode batch -source run_direct_phc_step_impl_strategy_from_synth.tcl \
#          -tclargs <strategy_name> <clock_period_ns>
#
# Strategies:
#   fanout_dup_100mhz
#   aggressive_explore_100mhz
#   baseline_95mhz

if { [llength $argv] < 2 } {
    error "Usage: run_direct_phc_step_impl_strategy_from_synth.tcl <strategy_name> <clock_period_ns>"
}

set strategy_name [lindex $argv 0]
set clock_period_ns [lindex $argv 1]
set source_checkpoint ./direct_phc_step_checkpoints/phc_step_synth.dcp
set report_dir ./direct_phc_step_reports_${strategy_name}
set checkpoint_dir ./direct_phc_step_checkpoints_${strategy_name}

file mkdir $report_dir
file mkdir $checkpoint_dir

puts "DIRECT_IMPL_STRATEGY: strategy=$strategy_name clock_period_ns=$clock_period_ns"
puts "DIRECT_IMPL_STRATEGY: source_checkpoint=$source_checkpoint"

if { ![file exists $source_checkpoint] } {
    error "Missing synthesized checkpoint: $source_checkpoint"
}

open_checkpoint $source_checkpoint

# Override only the implementation timing constraint when a non-10ns fallback
# clock is requested. The original RTL and HLS-generated xdc are not edited.
set existing_clocks [get_clocks -quiet ap_clk]
if { [llength $existing_clocks] } {
    set current_period [get_property PERIOD [lindex $existing_clocks 0]]
    set period_delta [expr {abs(double($current_period) - double($clock_period_ns))}]
    if { $period_delta > 0.001 } {
        error "Cannot change clock period from synthesized checkpoint: current=$current_period requested=$clock_period_ns. Use the RTL-based full P&R script for a different clock."
    }
    puts "DIRECT_IMPL_STRATEGY: keeping existing ap_clk period=$current_period from synthesized checkpoint"
} else {
    set half_period [expr {$clock_period_ns / 2.0}]
    create_clock -name ap_clk -period $clock_period_ns -waveform [list 0.000 $half_period] [get_ports ap_clk]
    puts "DIRECT_IMPL_STRATEGY: created ap_clk period=$clock_period_ns waveform=0.000/$half_period"
}

proc maybe_set_property {property value objects label} {
    if { [llength $objects] == 0 } {
        puts "DIRECT_IMPL_STRATEGY: $label not found for property $property"
        return 0
    }
    set rc [catch {set_property $property $value $objects} msg]
    if { $rc } {
        puts "DIRECT_IMPL_STRATEGY: could not set $property on $label: $msg"
        return 0
    }
    puts "DIRECT_IMPL_STRATEGY: set $property=$value on $label ([llength $objects] object(s))"
    return 1
}

proc run_or_error {description command} {
    puts "DIRECT_IMPL_STRATEGY: $description"
    set rc [catch {uplevel 1 $command} msg]
    if { $rc } {
        puts "DIRECT_IMPL_STRATEGY_ERROR: $description failed: $msg"
        return -code error $msg
    }
}

set fsm_cells [get_cells -hier -quiet -regexp {.*grp_phc_step_Outline_VITIS_LOOP_127_11_fu_2060/ap_CS_fsm_reg\[9\]$}]
set fsm_pins {}
set fsm_nets {}
if { [llength $fsm_cells] } {
    set fsm_pins [get_pins -quiet -of_objects $fsm_cells -filter {REF_PIN_NAME == Q}]
    if { [llength $fsm_pins] } {
        set fsm_nets [get_nets -quiet -segments -of_objects $fsm_pins]
    }
}
puts "DIRECT_IMPL_STRATEGY: fsm_cells=[llength $fsm_cells] fsm_q_pins=[llength $fsm_pins] fsm_nets=[llength $fsm_nets]"

if { $strategy_name eq "fanout_dup_100mhz" } {
    maybe_set_property REGISTER_DUPLICATION true $fsm_cells "FSM state register ap_CS_fsm_reg[9]"
    maybe_set_property MAX_FANOUT 16 $fsm_nets "FSM state register ap_CS_fsm_reg[9] Q net"

    run_or_error "opt_design -directive Explore" {
        opt_design -directive Explore
    }
    write_checkpoint -force $checkpoint_dir/phc_step_opt.dcp

    run_or_error "place_design -directive ExtraNetDelay_high" {
        place_design -directive ExtraNetDelay_high
    }
    write_checkpoint -force $checkpoint_dir/phc_step_placed.dcp
    report_utilization -file $report_dir/phc_step_utilization_placed.rpt
    report_timing_summary -max_paths 10 -report_unconstrained -file $report_dir/phc_step_timing_summary_placed.rpt

    if { [llength $fsm_nets] } {
        set rc [catch {phys_opt_design -force_replication_on_nets $fsm_nets} msg]
        if { $rc } {
            puts "DIRECT_IMPL_STRATEGY: phys_opt_design -force_replication_on_nets failed or unsupported: $msg"
        } else {
            puts "DIRECT_IMPL_STRATEGY: forced replication on FSM state Q net"
        }
    }
    run_or_error "phys_opt_design -directive AggressiveFanoutOpt" {
        phys_opt_design -directive AggressiveFanoutOpt
    }
    write_checkpoint -force $checkpoint_dir/phc_step_physopt.dcp

    run_or_error "route_design -directive AggressiveExplore" {
        route_design -directive AggressiveExplore
    }
} elseif { $strategy_name eq "aggressive_explore_100mhz" } {
    run_or_error "opt_design -directive Explore" {
        opt_design -directive Explore
    }
    write_checkpoint -force $checkpoint_dir/phc_step_opt.dcp

    run_or_error "place_design -directive ExtraNetDelay_high" {
        place_design -directive ExtraNetDelay_high
    }
    write_checkpoint -force $checkpoint_dir/phc_step_placed.dcp
    report_utilization -file $report_dir/phc_step_utilization_placed.rpt
    report_timing_summary -max_paths 10 -report_unconstrained -file $report_dir/phc_step_timing_summary_placed.rpt

    run_or_error "phys_opt_design -directive AggressiveExplore" {
        phys_opt_design -directive AggressiveExplore
    }
    write_checkpoint -force $checkpoint_dir/phc_step_physopt.dcp

    run_or_error "route_design -directive AggressiveExplore" {
        route_design -directive AggressiveExplore
    }
} elseif { $strategy_name eq "baseline_95mhz" } {
    run_or_error "opt_design" {
        opt_design
    }
    write_checkpoint -force $checkpoint_dir/phc_step_opt.dcp

    run_or_error "place_design" {
        place_design
    }
    write_checkpoint -force $checkpoint_dir/phc_step_placed.dcp
    report_utilization -file $report_dir/phc_step_utilization_placed.rpt
    report_timing_summary -max_paths 10 -report_unconstrained -file $report_dir/phc_step_timing_summary_placed.rpt

    run_or_error "phys_opt_design" {
        phys_opt_design
    }
    write_checkpoint -force $checkpoint_dir/phc_step_physopt.dcp

    run_or_error "route_design" {
        route_design
    }
} else {
    error "Unknown strategy: $strategy_name"
}

write_checkpoint -force $checkpoint_dir/phc_step_routed.dcp

report_route_status -file $report_dir/phc_step_route_status.rpt
report_drc -file $report_dir/phc_step_drc_routed.rpt
report_utilization -file $report_dir/phc_step_utilization_routed.rpt
report_timing_summary -max_paths 10 -routable_nets -report_unconstrained -warn_on_violation -file $report_dir/phc_step_timing_summary_routed.rpt
report_timing -max_paths 10 -sort_by group -file $report_dir/phc_step_timing_paths_routed.rpt
report_power -file $report_dir/phc_step_power_routed.rpt

puts "DIRECT_IMPL_STRATEGY: complete strategy=$strategy_name report_dir=$report_dir"
