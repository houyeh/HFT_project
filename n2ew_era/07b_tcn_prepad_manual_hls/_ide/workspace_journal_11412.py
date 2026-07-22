# 2026-06-24T10:02:47.576765300
import vitis

client = vitis.create_client()
client.set_workspace(path="07b_tcn_prepad_manual_hls")

comp = client.get_component(name="hls_component")
comp.run(operation="C_SIMULATION")

comp.run(operation="C_SIMULATION")

vitis.dispose()

