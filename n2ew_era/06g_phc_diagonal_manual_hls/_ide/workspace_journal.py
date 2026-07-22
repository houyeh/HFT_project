# 2026-06-22T03:21:32.353597400
import vitis

client = vitis.create_client()
client.set_workspace(path="06g_phc_diagonal_manual_hls")

comp = client.get_component(name="hls_component")
comp.run(operation="SYNTHESIS")

comp.run(operation="CO_SIMULATION")

vitis.dispose()

