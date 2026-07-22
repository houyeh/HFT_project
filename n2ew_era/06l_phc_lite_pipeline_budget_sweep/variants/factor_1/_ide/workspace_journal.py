# 2026-06-22T21:30:39.127772600
import vitis

client = vitis.create_client()
client.set_workspace(path="factor_1")

comp = client.get_component(name="hls_component")
comp.run(operation="C_SIMULATION")

comp.run(operation="SYNTHESIS")

vitis.dispose()

