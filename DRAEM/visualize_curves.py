import os
import subprocess
from ruamel.yaml import YAML
import torch
import numpy as np
import argparse
import gc

from main import *

# Path to your config file
config_path = "DRAEM\\config.yaml"

# List of categories
data_dir = ["dataset\\AV_dataset_mvtec_like", "dataset\\AV_dataset_mvtec_like_aligned"]
categories = ["A", "A2", "B", "B2", "C"]
load_chps = [2000,2250,2500,2750,3000]
masked = [True, False]
img_reg = [True, False]
pre_masking = [False, True]

yaml = YAML()
yaml.preserve_quotes = True  # Preserve quotes if used in the file

def update_config_category(category, chp, is_masked, reg, mask_type, data_directory):
    """Update the category in the config file while preserving comments."""
    with open(config_path, "r") as f:
        config = yaml.load(f)
    
    # Update the category field
    config["data"]["pre_masking"] = mask_type
    config["data"]["image_registration"] = reg
    config["data"]["data_path"] = data_directory
    config["data"]["category"] = category
    config["data"]["load_epoch"] = chp
    config["metrics"]["masking"] = is_masked
    config["metrics"]["auroc"] = True
    config["metrics"]["pro"] = True
    config["metrics"]["pr"] = True
    config["metrics"]["iou"] = True
    config["metrics"]["comparison"] = True
    # Save the updated config back to the file
    with open(config_path, "w") as f:
        yaml.dump(config, f)

def run_ddad_command():
    """Run the appropriate DDAD command."""
    command = ["python", "DDAD_main\\main.py"]
    command.append("--detection")
    command.append("True")
    
    # Run the command
    img_auroc_curve, pixel_auroc_curve, df_pro, pr_curve = subprocess.run(command)
    return img_auroc_curve, pixel_auroc_curve, df_pro, pr_curve 

def create_view_mask(mask_chp, out_mask, mask_type):
    mask_chp = mask_chp[mask_type]
    chps = list(mask_chp.keys())
    #print(chps)
    views = mask_chp[chps[0]].keys()
    inverted_map = {}

    for view in views:
        inverted_map[view] = {}
        for chp in chps:
            metrics = mask_chp[chp][view]        
            inverted_map[view][chp] = metrics
    
    out_mask[mask_type] = inverted_map
    return out_mask

def main():
    """Main script to automate comparison of networks."""
    for is_pre_masked in pre_masking:
        after_masking = "pre_masking" if is_pre_masked else "post_masking" 
        for reg in img_reg:
            registration = "image_reg" if reg else "no_image_reg"
            dataset_dir = data_dir[1] if reg else data_dir[0]
            mask_map_chp = {}
            mask_map_view = {}
            for is_masked in masked:
                mask_type = 'masking' if is_masked else 'no_masking'
                mask_map_chp[mask_type] = {}
                mask_map_view[mask_type] = {}
                for chp in load_chps:
                    chp_map = {}
                    for category in categories:
                        category_map = {}
                        print(f"Processing category: {category}, checkpoint: {chp}, masked: {is_masked}, img_reg: {reg}, pre_masking: {is_pre_masked}")

                        update_config_category(category, chp, is_masked, reg, is_pre_masked, dataset_dir)
                        
                        torch.cuda.empty_cache()
                        args = parse_args()
                        config = OmegaConf.load(args.config)
                        img_auroc_curve, pixel_auroc_curve, df_pro, pr_curve, iou_curve = test(config)
                        df_pro = [df_pro['fpr'].to_numpy(), df_pro['pro'].to_numpy(), df_pro['threshold'].to_numpy()]
                        category_map = {
                            'img_auroc': img_auroc_curve,
                            'pixel_auroc': pixel_auroc_curve,
                            'pixel_pro': df_pro,
                            'precision_recall':pr_curve,
                            'iou_curve': iou_curve
                        }
                        chp_map[category] = category_map

                        # Release unused variables
                        del img_auroc_curve, pixel_auroc_curve, df_pro, pr_curve, iou_curve, category_map
                        torch.cuda.empty_cache()  # Clear CUDA memory
                        gc.collect()  # Trigger garbage collection

                    mask_map_chp[mask_type][chp] = chp_map 
                    # print(mask_map_chp[mask_type])
                    visualize_multiple_curves(mask_map_chp, 'chp_comparison', mask_type, registration, after_masking)
                    del chp_map  # Clear chp_map after use

                mask_map_view = create_view_mask(mask_map_chp, mask_map_view, mask_type)
                visualize_multiple_curves(mask_map_view, 'view_comparison', mask_type, registration, after_masking)

                del mask_map_view, mask_map_chp  # Clear view and chp map after use
                print(f"Finished [{mask_type}]")
                torch.cuda.empty_cache()
                gc.collect()


if __name__ == "__main__":
    main()
