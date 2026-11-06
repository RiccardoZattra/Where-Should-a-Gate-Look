import itertools
import os

from omegaconf import OmegaConf

# 1. Define the parameter grids
VAR_1 = [4, 8]  # learning_rate
VAR_2 = [64, 128]  # gate_bias

output_dir = "configs/experiment/bulk/"
os.makedirs(output_dir, exist_ok=True)

combinations = list(itertools.product(VAR_1, VAR_2))
print(f"Generating {len(combinations)} configuration files...")
input("Press Enter to continue or Ctrl+C to abort...")

# Iterate over each combination
for v1, v2 in combinations:
    # 2. Load the base config fresh for each iteration
    # Ensure you have a 'base_config.yaml' file in your directory
    config = OmegaConf.load(
        "configs/experiment/dynamic-impulse-response-prediction.yaml"
    )

    # 3. Override the specific variables
    config.transformer_seed = 1
    config.data_seed = 1

    config.lr = v1

    config.model.gate_bias = v2

    # 4. Format the filename
    filename = f"{config.dataset.name}-{v1}-{v2}.yaml"
    filepath = os.path.join(output_dir, filename)

    # 5. Save the new configuration
    with open(filepath, "w") as f:
        # OmegaConf strips comments when loading/saving.
        # We manually re-inject the Hydra global directive at the top.
        f.write("# @package _global_\n\n")
        OmegaConf.save(config, f)

print(f"Done! All files have been saved to the '{output_dir}' directory.")
