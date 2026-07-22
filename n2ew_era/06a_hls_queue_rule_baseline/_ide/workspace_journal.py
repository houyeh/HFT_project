# 2026-06-08T16:32:41.827033400
import vitis

client = vitis.create_client()
client.set_workspace(path="06a_hls_queue_rule_baseline")

comp = client.get_component(name="hls_component")
comp.run(operation="C_SIMULATION")

comp.run(operation="C_SIMULATION")

comp.run(operation="CO_SIMULATION")

comp.run(operation="IMPLEMENTATION")

comp.run(operation="IMPLEMENTATION")

vitis.dispose()

vitis.dispose()

