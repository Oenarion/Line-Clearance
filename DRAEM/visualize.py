import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import numpy as np
import torch
import os
import cv2




def extract_mask(out_mask, threshold, size, guassian_blur = False, morph = False):
    """
    Takes as input the structural similarity matrix computed with ssim(full=True) and the l1_matrix, creates a blank matrix and sets it's value to 1 if ssim matrix or l1_matrix are higher than the threshold
    Returns the mask matrix
    """
    # print(out_mask.shape)
    if guassian_blur:
        out_mask = cv2.GaussianBlur(out_mask, (5, 5), 0)

    mask = np.zeros((size,size))
    # print(mask.shape)
    # Assign the weighted values to the mask based on the threshold
    mask[out_mask > threshold] = 1

    
    if morph:
        binary_mask = (mask > 0).astype(np.uint8)  # Set an appropriate threshold to convert mask to binary
        # Apply morphological operations on the binary mask
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        # Multiply the result with your original mask to restore the weighted values
        mask = mask * binary_mask

    mask = torch.from_numpy(mask)

    return mask

def create_masked_img(image, pred_mask):
    """
    Utility function to draw the original image (colored) with the mask and rectangles computed before
    Returns the masked image
    """
    anomaly_overlay = np.zeros_like(image)
    anomaly_overlay[:,:,0] = np.where(pred_mask == 0, anomaly_overlay[:,:,0], 255)  # Red color for anomalies
    anomaly_overlay[:,:,1] = np.where(pred_mask == 0, anomaly_overlay[:,:,1], 0)  
    anomaly_overlay[:,:,2] = np.where(pred_mask == 0, anomaly_overlay[:,:,2], 0)  

    # Blend the original image and the anomaly overlay
    blended_img = 0.7 * image + 0.3 * anomaly_overlay
    return blended_img

def show_results(orig_img, recon_img, out_mask, gt_mask, recon_mask, prediction, run_name, load_chp, masked, idx, threshold_name, img_reg, mask_type):
    # print(orig_img.shape, recon_img.shape, out_mask.shape, gt_mask.shape, recon_mask.shape)
    if not os.path.exists(f'DRAEM/results/{mask_type}/{img_reg}/{run_name}/checkpoint_{load_chp}/{threshold_name}/{masked}'):
        os.makedirs(f'DRAEM/results/{mask_type}/{img_reg}/{run_name}/checkpoint_{load_chp}/{threshold_name}/{masked}')

    orig_img = np.transpose(np.array(orig_img.detach().cpu()), (1,2,0))
    recon_img = np.transpose(np.array(recon_img.detach().cpu()), (1,2,0))
    out_mask = np.array(out_mask[0].detach().cpu())
    mask_img = create_masked_img(orig_img, recon_mask)

    recon_img = np.clip(recon_img, 0, 1)
    out_mask = np.clip(out_mask, 0, 1)
    mask_img = np.clip(mask_img, 0, 1)
    # print("orig img")
    # print(np.max(orig_img), np.min(orig_img), orig_img.dtype)
    # print("recon img")
    # print(np.max(recon_img), np.min(recon_img), recon_img.dtype)
    # print("out mask")
    # print(np.max(out_mask), np.min(out_mask), out_mask.dtype)
    # print("gt_mask")
    # print(np.max(gt_mask), np.min(gt_mask), gt_mask.dtype)
    # print("recon mask")
    # print(np.max(recon_mask), np.min(recon_mask), recon_mask.dtype)
    
    _, ax = plt.subplots(2, 3,figsize = [30,30])

    ax[0][0].imshow(orig_img)
    ax[0][0].set_title("Original image", fontsize=30)
    ax[0][1].imshow(recon_img)
    ax[0][1].set_title(f"Reconstructed image", fontsize=30)
    ax[0][2].imshow(out_mask, cmap='jet')
    ax[0][2].set_title(f"Anomaly map", fontsize=30)
    ax[1][0].imshow(gt_mask[0])
    ax[1][0].set_title("GT mask", fontsize=30)
    ax[1][1].imshow(recon_mask)
    ax[1][1].set_title("Thresholded binary mask", fontsize=30)
    ax[1][2].imshow(mask_img)
    ax[1][2].set_title('Final result = normal' if prediction[idx] == 0 else 'Final result = abnormal', color="g" if prediction[idx] == 0 else "r", fontsize=30)
    plt.tight_layout()
    plt.savefig(f'DRAEM/results/{mask_type}/{img_reg}/{run_name}/checkpoint_{load_chp}/{threshold_name}/{masked}/sample{idx}heatmap.png')
    plt.close()


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
        if not os.path.exists(f'DRAEM/visualize_curves/{is_pre_masked}/{reg}/{mask_type}/chp_comparison'):
                os.makedirs(f'DRAEM/visualize_curves/{is_pre_masked}/{reg}/{mask_type}/chp_comparison')
        for chp in mask_map[mask_type].keys():
            dir_path = f'DRAEM/visualize_curves/{is_pre_masked}/{reg}/{mask_type}/chp_comparison/{chp}'
            os.makedirs(dir_path, exist_ok=True)
            save_plot_chp(mask_map[mask_type][chp], dir_path, chp, 'img_auroc')
            save_plot_chp(mask_map[mask_type][chp], dir_path, chp, 'pixel_auroc')
            save_plot_chp(mask_map[mask_type][chp], dir_path, chp, 'pixel_pro')
            save_plot_chp(mask_map[mask_type][chp], dir_path, chp, 'precision_recall')
            save_plot_chp(mask_map[mask_type][chp], dir_path, chp, 'iou_curve')
    elif label == 'view_comparison':
        if not os.path.exists(f'DRAEM/visualize_curves/{is_pre_masked}/{reg}/{mask_type}/view_comparison'):
                os.makedirs(f'DRAEM/visualize_curves/{is_pre_masked}/{reg}/{mask_type}/view_comparison')
        for view in mask_map[mask_type].keys():
            dir_path = f'DRAEM/visualize_curves/{is_pre_masked}/{reg}/{mask_type}/view_comparison/{view}'
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
    plt.savefig(f'DRAEM/results/{mask_type}/{img_reg}/{category}/checkpoint_{load_chp}/{masked}/auroc_curves.png')
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
    plt.savefig(f'DRAEM/results/{mask_type}/{img_reg}/{category}/checkpoint_{load_chp}/{masked}/pro_pr_iou_curves.png')
    plt.close()
