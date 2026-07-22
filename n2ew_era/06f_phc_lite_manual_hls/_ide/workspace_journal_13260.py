# 2026-06-15T13:44:54.113757800
import vitis

client = vitis.create_client()
client.set_workspace(path="06f_phc_lite_manual_hls")

comp = client.get_component(name="hls_component")
comp.run(operation="IMPLEMENTATION")

comp.run(operation="IMPLEMENTATION")

comp.run(operation="IMPLEMENTATION")

comp.run(operation="IMPLEMENTATION")

vitis.dispose()

