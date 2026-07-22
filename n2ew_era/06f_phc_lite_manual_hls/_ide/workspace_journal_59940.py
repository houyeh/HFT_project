# 2026-06-08T20:17:36.816890
import vitis

client = vitis.create_client()
client.set_workspace(path="06f_phc_lite_manual_hls")

comp = client.create_hls_component(name = "hlsComponent",cfg_file = ["hls_config.cfg"],template = "empty_hls_component")

comp = client.get_component(name="hlsComponent")
comp.run(operation="C_SIMULATION")

comp.run(operation="C_SIMULATION")

comp.run(operation="C_SIMULATION")

comp.run(operation="SYNTHESIS")

comp.run(operation="SYNTHESIS")

comp.run(operation="SYNTHESIS")

comp.run(operation="SYNTHESIS")

comp.run(operation="CO_SIMULATION")

vitis.dispose()

