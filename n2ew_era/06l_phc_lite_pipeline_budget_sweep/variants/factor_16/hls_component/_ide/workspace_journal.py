# 2026-06-23T01:00:08.713075700
import vitis

client = vitis.create_client()
client.set_workspace(path="hls_component")

comp = client.get_component(name="hls_component")
comp.run(operation="C_SIMULATION")

comp.run(operation="SYNTHESIS")

comp.run(operation="CO_SIMULATION")

vitis.dispose()

