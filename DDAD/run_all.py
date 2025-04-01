import os
import subprocess
from ruamel.yaml import YAML
import time

# Path to your config file
config_path = "DDAD_main\\config.yaml"

# List of categories
categories = ["A", "A2", "B", "B2", "C"]

yaml = YAML()
yaml.preserve_quotes = True  # Preserve quotes if used in the file

def update_config_category(category):
    """Update the category in the config file while preserving comments."""
    with open(config_path, "r") as f:
        config = yaml.load(f)
    
    # Update the category field
    config["data"]["category"] = category
    
    # Save the updated config back to the file
    with open(config_path, "w") as f:
        yaml.dump(config, f)

def run_ddad_command(train=False, domain_adaptation=False):
    """Run the appropriate DDAD command."""
    command = ["python", "DDAD_main\\main.py"]
    
    if train:
        command.append("--train")
        command.append("True")
    elif domain_adaptation:
        command.append("--domain_adaptation")
        command.append("True")

    # Run the command
    subprocess.run(command)

def main():
    """Main script to automate training for all categories."""
    train_times = []
    da_times = []

    for category in categories:
        print(f"Processing category: {category}")
        
        # Step 1: Update the category in the config file
        update_config_category(category)
        
        start_train_time = time.time()
        # Step 2: Run the commands in sequence
        print(f"Training for category: {category}")
        run_ddad_command(train=True)
        end_train_time = time.time()

        m_t, s_t = divmod(end_train_time - start_train_time, 60)
        h_t, m_t = divmod(m_t, 60)
        print(f"Total training time for category {category} is: {h_t}h{m_t}m{s_t}s")
        
        start_da_time = time.time()
        print(f"Domain Adaptation for category: {category}")
        run_ddad_command(domain_adaptation=True)
        end_da_time = time.time()

        m_d, s_d = divmod(end_da_time - start_da_time, 60)
        h_d, m_d = divmod(m_d, 60)
        print(f"Total domain adaptation time for category {category} is: {h_d}h{m_d}m{s_d}s")
        
        print(f"Finished processing category: {category}")

        train_times.append([end_train_time - start_train_time, f"{h_t}:{m_t}:{s_t}"])
        da_times.append([end_da_time - start_da_time, f"{h_d}:{m_d}:{s_d}"])

    with open('training_times.txt', 'w') as f:
        f.write(f"TRAIN TIMES: {train_times} \n")
        f.write(f"DOMAIN ADAPTATION TIMES: {da_times}")
    

if __name__ == "__main__":
    main()
