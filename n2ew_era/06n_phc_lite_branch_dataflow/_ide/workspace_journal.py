# 2026-06-23T17:16:37.179415100
import vitis

client = vitis.create_client()
client.set_workspace(path="06n_phc_lite_branch_dataflow")

comp = client.get_component(name="hls_component")
comp.run(operation="C_SIMULATION")

comp.run(operation="SYNTHESIS")

comp.run(operation="SYNTHESIS")

comp.run(operation="CO_SIMULATION")

vitis.dispose()

