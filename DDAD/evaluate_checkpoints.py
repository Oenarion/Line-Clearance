import os
import subprocess
from ruamel.yaml import YAML

# Path to your config file
config_path = "DDAD_main\\config.yaml"
data_dir = ["dataset\\AV_dataset_mvtec_like", "dataset\\AV_dataset_mvtec_like_aligned"]
# List of categories
categories = ["C"]
# categories = ["B2"]
load_chps = [2000,2250,2500,2750,3000, 3250, 3500, 3750, 4000]
masked = [True, False]
threshold = [0, 1, 2]
img_reg = [True, False]
pre_masking = [True, False]

yaml = YAML()
yaml.preserve_quotes = True  # Preserve quotes if used in the file

def update_config_thresh(category, chp, is_masked, reg, data_directory, mask_type):
    with open(config_path, "r") as f:
        config = yaml.load(f)
    
    # Update the category field
    config["data"]["pre_masking"] = mask_type
    config["data"]["data_dir"] = data_directory
    config["data"]["category"] = category
    config["model"]["load_chp"] = chp
    config["metrics"]["masking"] = is_masked
    config["data"]["image_registration"] = reg
    # Save the updated config back to the file
    with open(config_path, "w") as f:
        yaml.dump(config, f)

def update_config_is_masked(is_masked, thresh):
    """Update the category in the config file while preserving comments."""
    with open(config_path, "r") as f:
        config = yaml.load(f)
    
    # Update the category field

    config["metrics"]["masking"] = is_masked
    config["metrics"]["threshold_type"] = thresh
    # Save the updated config back to the file
    with open(config_path, "w") as f:
        yaml.dump(config, f)

def run_ddad_command(name='detection'):
    """
    Run the appropriate DDAD command.
    
    name is either 'detection' or 'threshold'
    """
    command = ["python", "DDAD_main\\main.py"]
    command.append(f"--{name}")
    command.append("True")
    
    # Run the command
    subprocess.run(command)

def main():
    """Main script to automate evaluation for all categories."""
    for mask_type in pre_masking:
        for reg in img_reg:
            dataset_dir = data_dir[1] if reg else data_dir[0]
            for category in categories:
                for chp in load_chps:
                    for is_masked in masked:
                        update_config_thresh(category, chp, is_masked, reg, dataset_dir, mask_type)
                        print(f"Threshold selection, Category: {category}, checkpoint: {chp}, masked: {is_masked}, image registration: {reg}")
                        run_ddad_command('threshold')
                        for thresh in threshold:
                            if thresh == 0:
                                thresh_name = 'max_threshold'
                            elif thresh == 1:
                                thresh_name = 'p-quantile_threshold'
                            else:
                                thresh_name = 'k-sigma_threshold'
                            print(f"Processing category: {category}, checkpoint: {chp}, masked: {is_masked}, threshold: {thresh_name}, image registration: {reg}")

                            update_config_is_masked(is_masked, thresh)
                            
                            run_ddad_command()
                        
                print(f"Finished processing category: {category}")

if __name__ == "__main__":
    main()
