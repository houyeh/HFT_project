# 2026-06-24T00:15:37.397252600
import vitis

client = vitis.create_client()
client.set_workspace(path="06n2_phc_lite_gain_dataflow")

comp = client.get_component(name="hls_component")
comp.run(operation="C_SIMULATION")

comp.run(operation="C_SIMULATION")

comp.run(operation="SYNTHESIS")

vitis.dispose()

