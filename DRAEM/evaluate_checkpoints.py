import os
import subprocess
from ruamel.yaml import YAML
import time

# Path to your config file
config_path = "DRAEM\\config.yaml"

# List of categories
categories = ["A", "A2", "B", "B2", "C"]
# categories = ["B2"]
load_chps = [2000,2250,2500,2750,3000]
masked = [False, True]
threshold = [0, 1, 2]

yaml = YAML()
yaml.preserve_quotes = True  # Preserve quotes if used in the file

def update_config_thresh(category, chp, is_masked):
    with open(config_path, "r") as f:
        config = yaml.load(f)
    
    # Update the category field
    config["data"]["category"] = category
    config["data"]["load_epoch"] = chp
    config["metrics"]["masking"] = is_masked
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

def run_draem_command(name='detection'):
    """
    Run the appropriate DDAD command.
    
    name is either 'detection' or 'threshold'
    """
    command = ["python", "DRAEM\\main.py"]
    command.append(f"--{name}")
    command.append("True")
    
    # Run the command
    subprocess.run(command)

def main():
    """Main script to automate evaluation for all categories."""
    for category in categories:
        for chp in load_chps:
            for is_masked in masked:
                update_config_thresh(category, chp, is_masked)
                print(f"Threshold selection, Category: {category}, checkpoint: {chp}, masked: {is_masked}")
                run_draem_command('threshold')
                for thresh in threshold:
                    if thresh == 0:
                        thresh_name = 'max_threshold'
                    elif thresh == 1:
                        thresh_name = 'p-quantile_threshold'
                    else:
                        thresh_name = 'k-sigma_threshold'
                    print(f"Processing category: {category}, checkpoint: {chp}, masked: {is_masked}, threshold: {thresh_name}")

                    
                    update_config_is_masked(is_masked, thresh)
                    inference_start = time.time()
                    run_draem_command()
                    inference_time = time.time() - inference_start

                    with open('DRAEM\\inference_times.txt', 'a') as f:
                        f.write(f"INFERENCE TIME FOR CATEGORY {category}, MASKING {is_masked}, THRESHOLD {thresh_name}: {inference_time}s \n")
                
        print(f"Finished processing category: {category}")

if __name__ == "__main__":
    main()
