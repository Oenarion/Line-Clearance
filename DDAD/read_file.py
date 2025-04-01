import os
import numpy as np
import re

# Path to your config file
results_path = "DDAD_main\\results"

# List of categories
categories = ["A", "A2", "B", "B2", "C"]
load_chps = [2000,2250,2500,2750,3000]
masked = ['no_masking', 'masking']

def parse_array(lines, label):
    """Helper function to parse arrays between 'label' and the next label."""
    array_data = []
    array_started = False
    
    #print(lines[:3])
    for line in lines:
        #print(line)
        if line.startswith(label):
            #print("START!")
            array_started = True
            if ']' in line:
                #print(line.split('[')[1].split(']')[0])
                cleaned_line = line.split('[')[1].split(']')[0].strip().replace('  ', '')
                array_data.extend(map(float, cleaned_line.split()))
                break
        elif array_started:
            # Check if the array ends
            if ']' in line:
                cleaned_line = line.split(']')[0].strip().replace('  ', '')
                array_data.extend(map(float, cleaned_line.split()))
                break
            else:
                # Clean and split the line, then extend the array list
                cleaned_line = line.strip().replace('  ', '')
                array_data.extend(map(float, cleaned_line.split()))
    
    return np.array(array_data)


def main():
    """Main script to automatically get the best scores out of all the trained networks."""
    print("-"*50)
    for is_masked in masked:
        for chp in load_chps:
            for category in categories:

                with open(f'DDAD_main/graphs_comparison/dump/checkpoint_{chp}/{is_masked}/metrics_dump.txt', 'r') as f:
                    lines = f.readlines()

                img_auroc_fpr = parse_array(lines, 'fpr: [')
                print(img_auroc_fpr)
                img_auroc_tpr = parse_array(lines, 'tpr: [')
                print(img_auroc_tpr)
                img_auroc_thresholds = parse_array(lines, 'thresholds: [')
                print(img_auroc_fpr)

                pixel_auroc_fpr = parse_array(lines, 'fpr: [')
                pixel_auroc_tpr = parse_array(lines, 'tpr: [')
                pixel_auroc_thresholds = parse_array(lines, 'thresholds: [')

                pixel_pro_fpr = parse_array(lines, 'fpr: [')
                pixel_pro_pro = parse_array(lines, 'pro: [')
                pixel_pro_thresholds = parse_array(lines, 'thresholds: [')

                precision_recall_precision = parse_array(lines, 'precision: [')
                precision_recall_recall = parse_array(lines, 'recall: [')
                precision_recall_thresholds = parse_array(lines, 'thresholds: [')

                # Combine parsed arrays into a dictionary for easy access
                metrics = {
                    'IMG_AUROC': {'fpr': img_auroc_fpr, 'tpr': img_auroc_tpr, 'thresholds': img_auroc_thresholds},
                    'PIXEL_AUROC': {'fpr': pixel_auroc_fpr, 'tpr': pixel_auroc_tpr, 'thresholds': pixel_auroc_thresholds},
                    'PIXEL_PRO': {'fpr': pixel_pro_fpr, 'pro': pixel_pro_pro, 'thresholds': pixel_pro_thresholds},
                    'PRECISION_RECALL': {'precision': precision_recall_precision, 'recall': precision_recall_recall, 'thresholds': precision_recall_thresholds}
                }

                print(metrics)


if __name__ == "__main__":
    main()
