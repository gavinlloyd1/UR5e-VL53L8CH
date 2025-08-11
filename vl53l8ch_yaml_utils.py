import yaml
import os

# modify this path according to the user
EVK_CONFIG_PATH = "C:/Users/lloy7803/OneDrive - University of St. Thomas/2025_Summer/GUIs/MZAI_EVK_v1.0.1/evk_config"

YAML_PATH = os.path.join(EVK_CONFIG_PATH, "custom_log_settings.yml")
PRESET_PATH = os.path.join(EVK_CONFIG_PATH, "custom_presets.yml")

def update_log_settings(num_frames, log_type = "csv"):
    with open(YAML_PATH, "r") as f:
        full_config = yaml.safe_load(f) or {}

    if "CustomLogSettings" not in full_config:
        full_config["CustomLogSettings"] = {}

    settings = full_config["CustomLogSettings"]
    settings["NumberOfFramesToLog"] = num_frames
    settings["LogType"] = log_type

    with open(YAML_PATH, "w") as f:
        yaml.dump(full_config, f, default_flow_style = False)

def update_cnh_bin_settings(preset_name, start_bin, sub_sample, num_bins):
    with open(PRESET_PATH, "r") as f:
        config = yaml.safe_load(f) or {}

    if "VL53L8CH_AIKit" not in config:
        config["VL53L8CH_AIKit"] = {}

    presets = config["VL53L8CH_AIKit"]

    if preset_name not in presets:
        print(f"Preset '{preset_name}' not found.")
        return

    preset = presets[preset_name]

    # Update or add the fields explicitly
    preset["cnhStartBin"] = int(start_bin)
    preset["cnhSubSample"] = int(sub_sample)
    preset["cnhNumBins"] = int(num_bins)

    print(f"Updated preset '{preset_name}' values:")
    print(f"  cnhStartBin: {preset.get('cnhStartBin')}")
    print(f"  cnhSubSample: {preset.get('cnhSubSample')}")
    print(f"  cnhNumBins: {preset.get('cnhNumBins')}")

    with open(PRESET_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
