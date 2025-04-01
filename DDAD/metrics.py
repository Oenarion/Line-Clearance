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

class Metric:
    def __init__(self,labels_list, predictions, anomaly_map_list, gt_list, config) -> None:
        self.labels_list = labels_list
        self.predictions = predictions
        self.anomaly_map_list = anomaly_map_list
        self.gt_list = gt_list
        self.config = config
        self.threshold = 0.5
    
    def px_intersection(self, gt_mask, pred_mask):
        """
        Computes pixel-wise intersection between two masks

        Args:
            - gt_mask (torch.Tensor): ground truth mask of the anomaly
            - pred_mask (torch.Tensor): predicted anomaly mask

        Returns the area of intersection
        """
        return torch.sum((gt_mask == 1) & (pred_mask == 1))

    def px_union(self, gt_mask, pred_mask):
        """
        Computes area of union between two masks

        Args:
            - gt_mask (torch.Tensor): ground truth mask of the anomaly
            - pred_mask (torch.Tensor): predicted anomaly mask

        Returns the area of union
        """
        area_img1 = torch.sum(gt_mask)
        area_img2 = torch.sum(pred_mask)
        area_intersection = self.px_intersection(gt_mask, pred_mask)

        return area_img1 + area_img2 - area_intersection

        
    def compute_fpr(self, gt_mask, pred_mask):
        """
        Computes False Positive Rate (FPR)

        Args:
            - gt_mask (torch.Tensor): ground truth mask of the anomaly
            - pred_mask (torch.Tensor): predicted anomaly mask

        Returns FPR
        """
        false_positives = torch.sum((gt_mask == 0) & (pred_mask == 1))
        true_negatives = torch.sum((gt_mask == 0) & (pred_mask == 0))
        
        if (false_positives + true_negatives) == 0:
            return 0
        fpr = (false_positives / (false_positives + true_negatives))
        
        return fpr.item()

    def pixel_IoU(self):
        """
        Computes IoU for different FPR values up to a maximum FPR of 0.3.

        """
        results_embeddings, gt_embeddings = self.concat_embeddings()

        pred_mask = (results_embeddings > self.threshold).float()
        area_intersection = self.px_intersection(gt_embeddings, pred_mask)
        area_union = self.px_union(gt_embeddings, pred_mask)
        iou = area_intersection / area_union if area_union != 0 else 0

        return iou
    
    def IoU_curve(self, thresholds, fprs):
        ious = []
        new_fprs = []
        new_threshs = []
        results_embeddings, gt_embeddings = self.concat_embeddings()
        #passing through each threshold takes too much time so we just take a threshold_idx every 700 idxs 
        i = 1
        while fprs[i] < 0.3:
    
            thresh_pred = results_embeddings.detach().clone()
            thresh_pred[thresh_pred < thresholds[i]] = 0.0
            thresh_pred[thresh_pred >= thresholds[i]] = 1.0
            area_intersection = self.px_intersection(gt_embeddings, thresh_pred)
            area_union = self.px_union(gt_embeddings, thresh_pred)
            ious.append(area_intersection / area_union if area_union != 0 else 0)
            new_fprs.append(fprs[i])
            new_threshs.append(thresholds[i])
            i += 1000
        
        iou_auc = auc(new_threshs, ious)
        return ious, new_fprs, iou_auc


    def image_auroc(self):
        auroc_image = roc_auc_score(self.labels_list, self.predictions)
        auroc_image_curve = roc_curve(self.labels_list, self.predictions)
        return auroc_image, auroc_image_curve
    
    def concat_embeddings(self):
        results_embeddings = self.anomaly_map_list[0]
        for feature in self.anomaly_map_list[1:]:
            results_embeddings = torch.cat((results_embeddings, feature), 0)
        results_embeddings =  ((results_embeddings - results_embeddings.min())/ (results_embeddings.max() - results_embeddings.min())) 

        gt_embeddings = self.gt_list[0]
        for feature in self.gt_list[1:]:
            
            # print(f"gt_embeddings shape, {gt_embeddings.shape}, feature shape: {feature.shape}")
            gt_embeddings = torch.cat((gt_embeddings, feature), 0)

        results_embeddings = results_embeddings.clone().detach().requires_grad_(False)
        gt_embeddings = gt_embeddings.clone().detach().requires_grad_(False)

        gt_embeddings = torch.flatten(gt_embeddings).cpu().type(torch.bool).detach()
        results_embeddings = torch.flatten(results_embeddings).cpu().detach()

        return results_embeddings, gt_embeddings

    def pixel_auroc(self):
        results_embeddings, gt_embeddings = self.concat_embeddings()
        auroc_p = AUROC(task="binary")

        auroc_pixel = auroc_p(results_embeddings, gt_embeddings)
        auroc_pixel_curve = roc_curve(gt_embeddings, results_embeddings)
        return auroc_pixel, auroc_pixel_curve
    
    def update_threshold(self, thresh):
        self.threshold = thresh

    def optimal_threshold(self):
        fpr, tpr, thresholds = roc_curve(self.labels_list, self.predictions)

        # Calculate Youden's J statistic for each threshold
        youden_j = tpr - fpr
        # Find the optimal threshold that maximizes Youden's J statistic
        optimal_threshold_index = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_threshold_index]
        print(f"OPTIMAL THRESHOLD: {optimal_threshold}")
        print(f"OPTIMAL THRESHOLD WITH SMALL VARIATIONS: {optimal_threshold - 0.1*optimal_threshold}")
        self.threshold = optimal_threshold - 0.1*optimal_threshold
        return optimal_threshold
    

    def pixel_pro(self):
        #https://github.com/hq-deng/RD4AD/blob/main/test.py#L337
        def _compute_pro(masks, amaps, num_th = 200):
            results_embeddings = amaps[0]
            for feature in amaps[1:]:
                results_embeddings = torch.cat((results_embeddings, feature), 0)
            amaps =  ((results_embeddings - results_embeddings.min())/ (results_embeddings.max() - results_embeddings.min())) 
            amaps = amaps.squeeze(1)
            amaps = amaps.cpu().detach().numpy()
            gt_embeddings = masks[0]
            for feature in masks[1:]:
                gt_embeddings = torch.cat((gt_embeddings, feature), 0)
            masks = gt_embeddings.squeeze(1).cpu().detach().numpy()
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
                fp_pixels = np.logical_and(inverse_masks , binary_amaps).sum()
                fpr = fp_pixels / inverse_masks.sum()
                # print(f"Threshold: {th}, FPR: {fpr}, PRO: {mean(pros)}")

                # print(pros,fpr,th)
                df = pd.concat([df, pd.DataFrame({"pro": mean(pros), "fpr": fpr, "threshold": th}, index=[0])], ignore_index=True)
                # df = df.concat({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

            # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
            df = df[df["fpr"] < 0.3]
            new_df = df.copy(deep=True)
            new_df["fpr"] = new_df["fpr"] / new_df["fpr"].max()

            pro_auc = auc(new_df["fpr"], new_df["pro"])

            return pro_auc, df
        
        pro = _compute_pro(self.gt_list, self.anomaly_map_list, num_th = 200)
        return pro
    
    def precision_recall(self):
        results_embeddings, gt_embeddings = self.concat_embeddings()
        pr_curve = precision_recall_curve(gt_embeddings, results_embeddings)
        average_precision = average_precision_score(gt_embeddings, results_embeddings)

        return average_precision, pr_curve

    def misclassified(self):
        predictions = torch.tensor(self.predictions)
        labels_list = torch.tensor(self.labels_list)
        predictions0_1 = (predictions > self.threshold).int()
        errors = []
        for i,(l,p) in enumerate(zip(labels_list, predictions0_1)):
            if l != p:
                phrase = f'Sample : {i} predicted as: {p.item()} label is: {l.item()} \n' 
                errors.append(phrase)
                print(phrase)
        
        return errors

