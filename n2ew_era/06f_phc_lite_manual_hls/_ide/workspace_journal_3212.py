# 2026-06-11T15:57:52.922969800
import vitis

client = vitis.create_client()
client.set_workspace(path="06f_phc_lite_manual_hls")

comp = client.get_component(name="hlsComponent")
comp.run(operation="C_SIMULATION")

comp.run(operation="C_SIMULATION")

vitis.dispose()

