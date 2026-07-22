# 2026-06-22T03:19:02.440704400
import vitis

client = vitis.create_client()
client.set_workspace(path="06g_phc_diagonal_manual_hls")

client.delete_component(name="hlsComponent")

cfg = client.get_config_file(path="/c:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06g_phc_diagonal_manual_hls/hls_component/hls_config.cfg")

cfg = client.get_config_file(path="/c:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06g_phc_diagonal_manual_hls/hls_component/hls_config.cfg")

cfg = client.get_config_file(path="/c:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06g_phc_diagonal_manual_hls/hls_component/hls_config.cfg")

cfg = client.get_config_file(path="/c:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06g_phc_diagonal_manual_hls/hls_component/hls_config.cfg")

cfg.set_values(key="tb.file", values=[])

cfg.set_values(key="tb.file_cflags", values=[])

cfg.set_value(section="hls", key="tb.cflags", value="")

cfg.set_value(section="hls", key="syn.csimflags", value="")

vitis.dispose()

