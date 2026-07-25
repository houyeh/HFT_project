puts "AUTO_PATH_BEGIN"
foreach p $::auto_path { puts $p }
puts "AUTO_PATH_END"
if {[catch {package require ::tclapp::support::appinit 1.2} res]} {
  puts "PACKAGE_REQUIRE_FAILED: $res"
  puts $::errorInfo
} else {
  puts "PACKAGE_REQUIRE_OK: $res"
}
if {[lsearch -exact $::auto_path "C:/AMDDesignTools/2025.2/Vivado/data/XilinxTclStore/support/appinit"] == -1} {
  lappend ::auto_path "C:/AMDDesignTools/2025.2/Vivado/data/XilinxTclStore/support/appinit"
}
if {[catch {package require ::tclapp::support::appinit 1.2} res2]} {
  puts "PACKAGE_REQUIRE_AFTER_APPEND_FAILED: $res2"
  puts $::errorInfo
} else {
  puts "PACKAGE_REQUIRE_AFTER_APPEND_OK: $res2"
}
exit
