# 2026-06-22T12:51:54.739746200
import vitis

client = vitis.create_client()
client.set_workspace(path="06l_phc_lite_pipeline_budget_sweep")

comp = client.get_component(name="hls_component")
comp.run(operation="C_SIMULATION")

comp.run(operation="SYNTHESIS")

vitis.dispose()

