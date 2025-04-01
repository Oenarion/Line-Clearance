import torch
import torch.nn.functional as F
from data_loader import DRAEMTrainDataset
from torch.utils.data import DataLoader
import numpy as np
from model import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
from metric_utils import *
from visualize import *
from masking import *


def max_threshold(anomaly_scores):
    """
    Compute the maximum threshold based on the maximum anomaly score.
    
    Args:
        - anomaly_scores (np.array): Array of anomaly scores from normal training images.
        
    Returns:
        - threshold (float): Maximum threshold.
    """

    return np.max(anomaly_scores)

def p_quantile_threshold(anomaly_scores, p=99):
    """
    Compute the p-quantile threshold.
    
    Args:
        - anomaly_scores (np.array): Array of anomaly scores from normal training images.
        - p (float): The quantile percentage (between 0 and 100).
        
    Returns:
        - threshold (float): p-quantile threshold.
    """
    
    return np.percentile(anomaly_scores, p)

def k_sigma_threshold(anomaly_scores, k=3):
    """
    Compute the k-sigma threshold.
    
    Args:
        - anomaly_scores (np.array): Array of anomaly scores from normal training images.
        - k (float): The k value for the threshold (number of standard deviations from the mean).
        
    Returns:
        - threshold (float): k-sigma threshold.
    """
    
    mean_score = np.mean(anomaly_scores)
    std_dev = np.std(anomaly_scores)
    return mean_score + k * std_dev

def threshold_computation(anomaly_scores, p=99, k=3):
    """
    Writes a report for all the threshold selection methods

    Args:
        - anomaly_scores (np.array): the anomaly scores computed beforehand

    Returns:
        - max_thresh (float)
        - p_quant_thresh (float)
        - k_sigma_thresh (float)
    """

    print("--- APPLYING MAX THRESHOLD SELECTION ---")
    max_thresh = max_threshold(anomaly_scores)
    print(f"MAX THRESHOLD IS {max_thresh}")

    print("--- APPLYING P-QUANTILE THRESHOLD SELECTION ---")
    p_quant_thresh = p_quantile_threshold(anomaly_scores, p)
    print(f"P-QUANTILE THRESHOLD IS {p_quant_thresh}")

    print("--- APPLYING K-SIGMA THRESHOLD SELECTION ---")
    k_sigma_thresh = k_sigma_threshold(anomaly_scores, k)
    print(f"K-SIGMA THRESHOLD IS {k_sigma_thresh}")

    return max_thresh, p_quant_thresh, k_sigma_thresh



def threshold(config):

    run_name_rec = config.data.category + "_" + config.data.run_name +"_REC_MODEL_"+str(config.data.load_epoch)+".pth"
    run_name_seg = config.data.category + "_" + config.data.run_name +"_SEG_MODEL_"+str(config.data.load_epoch)+".pth"

    # run_name_rec = config.data.run_name +"_REC_MODEL_"+str(config.data.load_epoch)+".pth"
    # run_name_seg = config.data.run_name +"_SEG_MODEL_"+str(config.data.load_epoch)+".pth"

    masked = 'masking' if config.metrics.masking else 'no_masking'

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(os.path.join(config.data.checkpoint_path, config.data.category, run_name_rec), map_location='cuda:0'))
    model.cuda()
    model.eval()

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.load_state_dict(torch.load(os.path.join(config.data.checkpoint_path, config.data.category, run_name_seg), map_location='cuda:0'))
    model_seg.cuda()
    model_seg.eval()

    dataset = DRAEMTrainDataset(config.data.data_path + '\\' + config.data.category + '\\train\\good', config.data.anomaly_source_path, 1, resize_shape=[config.data.img_size, config.data.img_size])
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    total_pixel_scores = np.zeros((config.data.img_size * config.data.img_size * len(dataset)))
    mask_cnt = 0

    anomaly_score_prediction = []

    if config.data.pre_masking == True:
        mask_type = "pre_masking"
    else:
        mask_type = "post_masking"

    if config.data.image_registration == True:
        img_reg = "image_reg" 
    else:
        img_reg = "no_image_reg"

    for _, sample_batched in enumerate(dataloader):

        gray_batch = sample_batched["image"].cuda()
        gray_rec = model(gray_batch)
        joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

        out_mask = model_seg(joined_in)

        ## Check for NaN before masking
        if torch.isnan(out_mask).any():
            print("Found NaN in out_mask before masking")

        ## masking needs to be applied here before we have the softmax computation
        if config.metrics.masking and config.data.pre_masking:
            mask = Mask(config.data.img_size, 'cuda')
            pred_mask, _ = mask.masking(gray_rec, out_mask, config.data.category)
            pred_mask_clamped = torch.clamp(pred_mask, min=-1e4, max=1e4)
            out_mask_sm = torch.softmax(pred_mask_clamped, dim=1)
        else:
            out_mask_sm = torch.softmax(out_mask, dim=1)

        ## Check for NaN after softmax
        if torch.isnan(out_mask_sm).any():
            print("Found NaN in out_mask_sm after softmax")

        out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()

        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                            padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)

        anomaly_score_prediction.append(image_score)

        flat_out_mask = out_mask_cv.flatten()
        total_pixel_scores[mask_cnt * config.data.img_size * config.data.img_size:(mask_cnt + 1) * config.data.img_size * config.data.img_size] = flat_out_mask
        mask_cnt += 1

    max_thresh, p_quant_thresh, k_sigma_thresh = threshold_computation(total_pixel_scores)


    if not os.path.exists(f'DRAEM/results/{mask_type}/{img_reg}/{config.data.category}/checkpoint_{config.data.load_epoch}/{masked}'):
        os.makedirs(f'DRAEM/results/{mask_type}/{img_reg}/{config.data.category}/checkpoint_{config.data.load_epoch}/{masked}')

    with open(f'DRAEM/results/{mask_type}/{img_reg}/{config.data.category}/checkpoint_{config.data.load_epoch}/{masked}/threshold_selection.txt', 'w') as f:
        f.write("THRESHOLDS COMPUTED ON NOMINAL VALIDATION DATA \n")
        f.write(f"MAX THRESHOLD: {max_thresh} \n")
        f.write(f"P-QUANT THRESHOLD: {p_quant_thresh} \n")
        f.write(f"K-SIGMA THRESHOLD: {k_sigma_thresh}")