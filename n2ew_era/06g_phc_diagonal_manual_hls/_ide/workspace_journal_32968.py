# 2026-06-21T01:55:47.203970700
import vitis

client = vitis.create_client()
client.set_workspace(path="06g_phc_diagonal_manual_hls")

comp = client.create_hls_component(name = "hlsComponent",cfg_file = ["hls_config.cfg"],template = "empty_hls_component")

comp = client.get_component(name="hlsComponent")
comp.run(operation="C_SIMULATION")

comp.run(operation="C_SIMULATION")

comp.run(operation="C_SIMULATION")

vitis.dispose()

