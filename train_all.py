import os
import subprocess
from ruamel.yaml import YAML


networks = ["DRAEM","DDAD-main"]
# List of categories
categories = ["A", "A2", "B", "B2", "C"]


yaml = YAML()
yaml.preserve_quotes = True  # Preserve quotes if used in the file

def update_config_category(net, category):
    """Update the category in the config file while preserving comments."""
    with open(f'{net}\\config.yaml', "r") as f:
        config = yaml.load(f)
    
    # Update the category field
    config["data"]["category"] = category
    
    # Save the updated config back to the file
    with open(f'{net}\\config.yaml', "w") as f:
        yaml.dump(config, f)

def run_ddad_command(net):
    """Run the appropriate DDAD/DRAEM command."""
    command = ["python", f"{net}\\main.py"]

    command.append("--train")
    command.append("True")

    # Run the command
    subprocess.run(command)

def main():
    """Main script to automate training for all categories."""
    for net in networks:
        print(f"Training Network: {net}")
        for category in categories:
            print(f"Processing category: {category}")
            
            # Step 1: Update the category in the config file
            update_config_category(net, category)
            
            # Step 2: Run the commands in sequence
            print(f"Training for category: {category}")
            run_ddad_command(net)
            
            print(f"Finished processing category: {category}")
        print(f"End of training of: {net}")

if __name__ == "__main__":
    main()
