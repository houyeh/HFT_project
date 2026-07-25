puts "AUTO_PATH_BEGIN"
foreach p $::auto_path { puts $p }
puts "AUTO_PATH_END"
foreach pkg {::tclapp::support::appinit ::tclapp::xilinx::xsim} {
  if {[catch {package require $pkg} res]} {
    puts "PACKAGE_REQUIRE_FAILED $pkg: $res"
    puts $::errorInfo
  } else {
    puts "PACKAGE_REQUIRE_OK $pkg: $res"
  }
}
exit
