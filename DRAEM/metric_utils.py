import torch
from torchmetrics import ROC, AUROC, F1Score
import os
from torchvision.transforms import transforms
from skimage import measure
import pandas as pd
from statistics import mean
import numpy as np
from sklearn.metrics import auc
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

def optimal_threshold(labels_list, predictions):
    fpr, tpr, thresholds = roc_curve(labels_list, predictions)

    # Calculate Youden's J statistic for each threshold
    youden_j = tpr - fpr
    # Find the optimal threshold that maximizes Youden's J statistic
    optimal_threshold_index = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_threshold_index]
    print(f"OPTIMAL THRESHOLD: {optimal_threshold}")
    print(f"OPTIMAL THRESHOLD WITH SMALL VARIATIONS: {optimal_threshold - 0.1*optimal_threshold}")
    # threshold = optimal_threshold - 0.1*optimal_threshold
    # return threshold
    return optimal_threshold

def IoU_curve(thresholds, fprs, results_embeddings, gt_embeddings):
    ious = []
    new_fprs = []
    new_threshs = []
    #passing through each threshold takes too much time so we just take a threshold_idx every 400 idxs 
    i = 1
    while fprs[i] < 0.3:

        thresh_pred = results_embeddings.copy()
        thresh_pred[thresh_pred < thresholds[i]] = 0.0
        thresh_pred[thresh_pred >= thresholds[i]] = 1.0
        area_intersection = px_intersection(gt_embeddings, thresh_pred)
        area_union = px_union(gt_embeddings, thresh_pred)
        ious.append(area_intersection / area_union if area_union != 0 else 0)
        new_fprs.append(fprs[i])
        new_threshs.append(thresholds[i])
        i += 1000
    
    iou_auc = auc(new_threshs, ious)
    return ious, new_fprs, iou_auc

def px_intersection(gt_mask, pred_mask):
    """
    Computes pixel-wise intersection between two masks

    Args:
        - gt_mask (torch.Tensor): ground truth mask of the anomaly
        - pred_mask (torch.Tensor): predicted anomaly mask

    Returns the area of intersection
    """
    return np.sum((gt_mask == 1) & (pred_mask == 1))

def px_union(gt_mask, pred_mask):
    """
    Computes area of union between two masks

    Args:
        - gt_mask (torch.Tensor): ground truth mask of the anomaly
        - pred_mask (torch.Tensor): predicted anomaly mask

    Returns the area of union
    """
    area_img1 = np.sum(gt_mask)
    area_img2 = np.sum(pred_mask)
    area_intersection = px_intersection(gt_mask, pred_mask)

    return area_img1 + area_img2 - area_intersection


def pixel_IoU(results_embeddings, gt_embeddings, threshold):
    """
    Computes IoU for different FPR values up to a maximum FPR of 0.3.
    """
    # Convert the numpy arrays into binary masks based on the threshold
    pred_mask = (results_embeddings > threshold).astype(float)
    gt_mask = gt_embeddings.astype(float)

    # Compute intersection and union areas
    area_intersection = np.sum((gt_mask == 1) & (pred_mask == 1))  # True positives
    area_union = np.sum((gt_mask == 1) | (pred_mask == 1))         # True positives + False positives + False negatives

    # Compute IoU
    iou = area_intersection / area_union if area_union != 0 else 0

    return iou


def pixel_pro(total_gt_pixel_scores, total_pixel_scores, img_size, num_th=200):
    def _compute_pro(masks, amaps, num_th):
        # Convert to appropriate tensor formats and reshape
        amaps = torch.tensor(amaps).view(-1, 1, img_size, img_size)  # Reshape as (B, 1, H, W)
        masks = torch.tensor(masks).view(-1, 1, img_size, img_size)

        # Normalize anomaly maps
        amaps = (amaps - amaps.min()) / (amaps.max() - amaps.min())
        amaps = amaps.squeeze(1).cpu().detach().numpy()
        masks = masks.squeeze(1).cpu().detach().numpy()

        # Thresholding step
        min_th = amaps.min()
        max_th = amaps.max()
        delta = (max_th - min_th) / num_th
        binary_amaps = np.zeros_like(amaps)
        df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])

        for th in np.arange(min_th, max_th, delta):
            binary_amaps[amaps <= th] = 0
            binary_amaps[amaps > th] = 1

            pros = []
            for binary_amap, mask in zip(binary_amaps, masks):
                for region in measure.regionprops(measure.label(mask)):
                    axes0_ids = region.coords[:, 0]
                    axes1_ids = region.coords[:, 1]
                    tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                    pros.append(tp_pixels / region.area)

            inverse_masks = 1 - masks
            fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
            fpr = fp_pixels / inverse_masks.sum()

            df = pd.concat([df, pd.DataFrame({"pro": np.mean(pros), "fpr": fpr, "threshold": th}, index=[0])], ignore_index=True)

        # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
        df = df[df["fpr"] < 0.3]
        df["fpr"] = df["fpr"] / df["fpr"].max()

        pro_auc = auc(df["fpr"], df["pro"])

        return pro_auc, df

    # Call the internal function with the scores
    pro_auc, df = _compute_pro(total_gt_pixel_scores, total_pixel_scores, num_th)
    return pro_auc, df

def misclassified(predictions, labels_list, threshold):
        predictions = torch.tensor(predictions)
        labels_list = torch.tensor(labels_list)
        predictions0_1 = (predictions > threshold).int()
        errors = []
        for i,(l,p) in enumerate(zip(labels_list, predictions0_1)):
            if l != p:
                phrase = f'Sample : {i} predicted as: {p.item()} label is: {l.item()} anomaly_score is: {predictions[i]}\n' 
                errors.append(phrase)
                print(phrase)
        
        return errors