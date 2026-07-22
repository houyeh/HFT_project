# 2026-06-09T22:46:42.768640300
import vitis

client = vitis.create_client()
client.set_workspace(path="06f_phc_lite_manual_hls")

comp = client.get_component(name="hlsComponent")
comp.run(operation="CO_SIMULATION")

vitis.dispose()

