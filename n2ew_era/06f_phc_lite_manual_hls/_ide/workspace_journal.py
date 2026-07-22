# 2026-06-21T02:01:50.624097800
import vitis

client = vitis.create_client()
client.set_workspace(path="06f_phc_lite_manual_hls")

vitis.dispose()

