import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import numpy as np
import torch
import os
from dataset import *



def save_plot_chp(curve_map, dir_path, comparison_type, metric_type):
    plt.figure(figsize=(12, 12))
    plt.title(f'{metric_type}_{comparison_type}', fontsize=30)
    for label in curve_map.keys():
        # print(label)
        # print(curve_map[label])
        plt.plot(curve_map[label][metric_type][0], curve_map[label][metric_type][1], label=label)
    plt.legend()
    plt.savefig(f'{dir_path}/{metric_type}.png')
    plt.close()
    
def visualize_multiple_curves(mask_map, label, mask_type, reg, is_pre_masked):
    if label == 'chp_comparison':
        if not os.path.exists(f'DDAD_main/visualize_curves/{is_pre_masked}/{reg}/{mask_type}/chp_comparison'):
                os.makedirs(f'DDAD_main/visualize_curves/{is_pre_masked}/{reg}/{mask_type}/chp_comparison')
        for chp in mask_map[mask_type].keys():
            dir_path = f'DDAD_main/visualize_curves/{is_pre_masked}/{reg}/{mask_type}/chp_comparison/{chp}'
            os.makedirs(dir_path, exist_ok=True)
            save_plot_chp(mask_map[mask_type][chp], dir_path, chp, 'img_auroc')
            save_plot_chp(mask_map[mask_type][chp], dir_path, chp, 'pixel_auroc')
            save_plot_chp(mask_map[mask_type][chp], dir_path, chp, 'pixel_pro')
            save_plot_chp(mask_map[mask_type][chp], dir_path, chp, 'precision_recall')
            save_plot_chp(mask_map[mask_type][chp], dir_path, chp, 'iou_curve')
    elif label == 'view_comparison':
        if not os.path.exists(f'DDAD_main/visualize_curves/{is_pre_masked}/{reg}/{mask_type}/view_comparison'):
                os.makedirs(f'DDAD_main/visualize_curves/{is_pre_masked}/{reg}/{mask_type}/view_comparison')
        for view in mask_map[mask_type].keys():
            dir_path = f'DDAD_main/visualize_curves/{is_pre_masked}/{reg}/{mask_type}/view_comparison/{view}'
            os.makedirs(dir_path, exist_ok=True)
            save_plot_chp(mask_map[mask_type][view], dir_path, view, 'img_auroc')
            save_plot_chp(mask_map[mask_type][view], dir_path, view, 'pixel_auroc')
            save_plot_chp(mask_map[mask_type][view], dir_path, view, 'pixel_pro')
            save_plot_chp(mask_map[mask_type][view], dir_path, view, 'precision_recall')
            save_plot_chp(mask_map[mask_type][view], dir_path, view, 'iou_curve')
            

def visualize_curves(img_auroc_curve, pixel_auroc_curve, pr_curve, df_pro, img_auc, pixel_auc, pr_auc, pro_auc, ious, iou_fprs, iou_auc, category, masked, load_chp, img_reg, mask_type):
    """
    Visualize evaluation curves with AUC values and diagonal line for ROC curves. 
    2x2 grid layout with each plot containing its respective AUC value.

    Args:
    - img_auroc_curve: Tuple with FPR and TPR for image-level AUROC curve.
    - pixel_auroc_curve: Tuple with FPR and TPR for pixel-level AUROC curve.
    - pr_curve: Tuple with precision and recall for PR curve.
    - df_pro: DataFrame with 'pro' and 'fpr' for PRO curve.
    - img_auc: AUC value for image-level AUROC.
    - pixel_auc: AUC value for pixel-level AUROC.
    - pr_auc: AUC value for Precision-Recall.
    - pro_auc: AUC value for PRO curve.
    """
    plt.figure(figsize=(12, 12))

    # IMAGE LEVEL AUROC
    plt.subplot(1, 2, 1)
    plt.plot(img_auroc_curve[0], img_auroc_curve[1], label=f'AUC = {img_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')  # Diagonal line
    plt.title('IMAGE LEVEL AUROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    # PIXEL LEVEL AUROC
    plt.subplot(1, 2, 2)
    plt.plot(pixel_auroc_curve[0], pixel_auroc_curve[1], label=f'AUC = {pixel_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')  # Diagonal line
    plt.title('PIXEL LEVEL AUROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(f'DDAD_main/results/{mask_type}/{img_reg}/{category}/checkpoint_{load_chp}/{masked}/auroc_curves.png')
    plt.close()

    plt.figure(figsize=(20, 12))
    # PRECISION-RECALL CURVE
    plt.subplot(1, 3, 1)
    plt.plot(pr_curve[0], pr_curve[1], label=f'AUC = {pr_auc:.3f}')
    plt.title('PRECISION RECALL CURVE')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')

    # PER REGION OVERLAP (PRO) CURVE
    plt.subplot(1, 3, 2)
    plt.plot(df_pro["fpr"], df_pro["pro"], label=f'AUC = {pro_auc:.3f}')
    plt.title('PER REGION OVERLAP (PRO) CURVE')
    plt.xlabel('False Positive Rate')
    plt.ylabel('PRO')
    plt.legend(loc='lower left')

    plt.subplot(1, 3, 3)
    plt.plot(iou_fprs, ious, label=f'AUC = {iou_auc:.3f}')
    plt.title('IoU Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('IoU')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(f'DDAD_main/results/{mask_type}/{img_reg}/{category}/checkpoint_{load_chp}/{masked}/pro_pr_iou_curves.png')
    plt.close()



def visualize(image, noisy_image, GT, pred_mask, anomaly_map, category, masked, load_chp, threshold_name, img_reg, mask_type) :
    for idx, img in enumerate(image):
        plt.figure(figsize=(11,11))
        plt.subplot(1, 3, 1).axis('off')
        plt.subplot(1, 3, 2).axis('off')
        plt.subplot(1, 3, 3).axis('off')
        plt.subplot(1, 3, 1)
        plt.imshow(show_tensor_image(image[idx]))
        plt.title('clear image')

        plt.subplot(1, 3, 2)

        plt.imshow(show_tensor_image(noisy_image[idx]))
        plt.title('reconstructed image')
        
        plt.subplot(1, 3, 3)

        anomaly_overlay = np.zeros_like(show_tensor_image(image[idx]))
        # print(np.unique(show_tensor_mask(pred_mask[idx])))
        anomaly_overlay[:,:,0] = np.where(show_tensor_mask(pred_mask[idx])[:,:,0] == 0, anomaly_overlay[:,:,0], 255)  # Red color for anomalies
        anomaly_overlay[:,:,1] = np.where(show_tensor_mask(pred_mask[idx])[:,:,0] == 0, anomaly_overlay[:,:,1], 0)  
        anomaly_overlay[:,:,2] = np.where(show_tensor_mask(pred_mask[idx])[:,:,0] == 0, anomaly_overlay[:,:,2], 0)  

        # Blend the original image and the anomaly overlay
        blended_img = 0.7 * show_tensor_image(image[idx]) + 0.3 * anomaly_overlay
        plt.imshow(blended_img.astype(np.uint8))
        plt.title('Original with anomaly overlay')

        plt.savefig(f'DDAD_main/results/{mask_type}/{img_reg}/{category}/checkpoint_{load_chp}/{threshold_name}/{masked}/sample{idx}.png')
        plt.close()

        plt.figure(figsize=(11,11))
        plt.subplot(1, 3, 1).axis('off')
        plt.subplot(1, 3, 2).axis('off')
        plt.subplot(1, 3, 3).axis('off')

        plt.subplot(1, 3, 1)
        plt.imshow(show_tensor_mask(GT[idx]))
        plt.title('ground truth')

        plt.subplot(1, 3, 2)
        plt.imshow(show_tensor_mask(pred_mask[idx]))
        plt.title('normal' if torch.max(pred_mask[idx]) == 0 else 'abnormal', color="g" if torch.max(pred_mask[idx]) == 0 else "r")

        plt.subplot(1, 3, 3)
        plt.imshow(show_tensor_image(anomaly_map[idx]))
        plt.title('heat map')
        plt.savefig(f'DDAD_main/results/{mask_type}/{img_reg}/{category}/checkpoint_{load_chp}/{threshold_name}/{masked}/sample{idx}heatmap.png')
        plt.close()



def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
    ])

    # Takes the first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    return reverse_transforms(image)

def show_tensor_mask(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.int8)),
    ])

    # Takes the first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    return reverse_transforms(image)
        

