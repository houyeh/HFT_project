puts "Resetting Vivado TclStore for this user profile..."
set rc [catch {tclapp::reset_tclstore} msg opts]
puts "reset_tclstore_rc=$rc"
puts "reset_tclstore_msg=$msg"
if {$rc != 0} {
    puts "reset_tclstore_opts=$opts"
    exit 1
}
puts "Vivado TclStore reset completed."
exit 0
