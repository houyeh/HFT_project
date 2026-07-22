# 2026-06-22T17:11:41.600088300
import vitis

client = vitis.create_client()
client.set_workspace(path="factor_2")

comp = client.get_component(name="hls_component")
comp.run(operation="CO_SIMULATION")

vitis.dispose()

