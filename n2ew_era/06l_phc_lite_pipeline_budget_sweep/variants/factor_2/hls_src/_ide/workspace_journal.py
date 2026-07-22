# 2026-06-22T17:10:34.457063300
import vitis

client = vitis.create_client()
client.set_workspace(path="hls_src")

vitis.dispose()

