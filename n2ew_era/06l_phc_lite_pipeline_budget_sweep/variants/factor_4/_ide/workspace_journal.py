# 2026-06-22T21:53:45.642801
import vitis

client = vitis.create_client()
client.set_workspace(path="factor_4")

comp = client.get_component(name="hls_component")
comp.run(operation="C_SIMULATION")

comp.run(operation="C_SIMULATION")

comp.run(operation="SYNTHESIS")

comp.run(operation="CO_SIMULATION")

vitis.dispose()

