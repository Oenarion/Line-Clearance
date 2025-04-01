import os
import subprocess
from ruamel.yaml import YAML


networks = ["DDAD_main", "DRAEM"]
categories = ["A", "A2", "B", "B2", "C"]
data_dir = ["dataset\\AV_dataset_mvtec_like", "dataset\\AV_dataset_mvtec_like_aligned"]
load_chps = [2000,2250,2500,2750,3000]
masked = [True, False]
threshold = [0, 1, 2]
img_reg = [True, False]
pre_masking = [True, False]

yaml = YAML()
yaml.preserve_quotes = True  # Preserve quotes if used in the file

def update_config_thresh(net, category, chp, is_masked, reg, data_directory, mask_type):
    if net == 'DRAEM':
        config_path = "DRAEM\\config.yaml"
    else:
        config_path = "DDAD_main\\config.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.load(f)
    
    if net == 'DRAEM':
        config["data"]["pre_masking"] = mask_type
        config["data"]["data_path"] = data_directory
        config["data"]["category"] = category
        config["data"]["load_epoch"] = chp
        config["metrics"]["masking"] = is_masked
        config["data"]["image_registration"] = reg
    else:
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

def update_config_is_masked(net, thresh):
    """Update the category in the config file while preserving comments."""
    if net == 'DRAEM':
        config_path = "DRAEM\\config.yaml"
    else:
        config_path = "DDAD_main\\config.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.load(f)
    
    # Update the category field
    
    config["metrics"]["threshold_type"] = thresh
    # Save the updated config back to the file
    with open(config_path, "w") as f:
        yaml.dump(config, f)

def run_ddad_command(net, threshold=False, detection=False):

    """Run the appropriate DDAD/DRAEM command."""
    command = ["python", f"{net}\\main.py"]
    
    if threshold:
        command.append("--threshold")
    
    if detection:
        command.append("--detection")
    command.append("True")

    # Run the command
    subprocess.run(command)

def main():
    """Main script to automate training for all categories."""
    for net in networks:
        print(f"Training Network: {net}")
        for mask_type in pre_masking:
            for reg in img_reg:
                if reg:
                    data_directory = data_dir[1]
                else:
                    data_directory = data_dir[0]
                for category in categories:
                    for chp in load_chps:
                        for is_masked in masked:
                            update_config_thresh(net, category, chp, is_masked, reg, data_directory, mask_type)
                            print(f"Threshold selection, Category: {category}, checkpoint: {chp}, masked: {is_masked}, image registration: {reg}")
                            run_ddad_command(net, threshold=True)
                            for thresh in threshold:
                                if thresh == 0:
                                    thresh_name = 'max_threshold'
                                elif thresh == 1:
                                    thresh_name = 'p-quantile_threshold'
                                else:
                                    thresh_name = 'k-sigma_threshold'
                                

                                # Step 1: Update the category in the config file
                                update_config_is_masked(net, thresh)
                                print(f"Processing category: {category}, checkpoint: {chp}, masked: {is_masked}, threshold: {thresh_name}, image registration: {reg}, pre_masking: {mask_type}")
                                # Step 2: Run the commands in sequence
                                run_ddad_command(net, detection=True)
                                
                    print(f"Finished processing category: {category}")
                print(f"Finished image registration: {reg}")
            print(f"Finished pre-masking: {mask_type}")
            print("="*100)

        print(f"End of evaluation of: {net}")

if __name__ == "__main__":
    main()
