# 2026-06-24T12:42:24.833936400
import vitis

client = vitis.create_client()
client.set_workspace(path="hls_component_scratch")

comp = client.get_component(name="hls_component")
comp.run(operation="C_SIMULATION")

comp.run(operation="SYNTHESIS")

vitis.dispose()

