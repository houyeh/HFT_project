# 2026-06-23T12:14:22.580933500
import vitis

client = vitis.create_client()
client.set_workspace(path="hls_component")

comp = client.get_component(name="hls_component")
comp.run(operation="CO_SIMULATION")

vitis.dispose()

