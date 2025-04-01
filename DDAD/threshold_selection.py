from asyncio import constants
from typing import Any
import torch
from unet import *
from dataset import *
from visualize import *
from anomaly_map import *
from metrics import *
from feature_extractor import *
from reconstruction import *
from masking import *
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"


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

class Threshold:
    def __init__(self, unet, config) -> None:
        self.val_dataset = Dataset_maker(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=1,
        )
        self.testloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size= config.data.test_batch_size,
            shuffle=False,
        )

        self.unet = unet
        self.config = config
        self.reconstruction = Reconstruction(self.unet, self.config)


    def __call__(self) -> Any:
        feature_extractor = domain_adaptation(self.unet, self.config, fine_tune=False, is_train = 1)
        feature_extractor.eval()
        
        mask_cnt = 0
        pixel_anomaly_scores = np.zeros((self.config.data.image_size * self.config.data.image_size * len(self.val_dataset)))
        masked = 'masking' if self.config.metrics.masking else 'no_masking'
        with torch.no_grad():
            for input_img, _, _ in self.testloader:
                input_img = input_img.to(self.config.model.device)
                x0 = self.reconstruction(input_img, input_img, self.config.model.w)[-1]
                if self.config.metrics.masking  and self.config.data.pre_masking:
                    mask = Mask(self.config.data.image_size, self.config.model.device)
                    input_masked, x0 = mask.masking(input_img, x0, self.config.data.category)
                else:
                    input_masked = input_img
                anomaly_map = heat_map(x0, input_masked, feature_extractor, self.config)
                flatten_map = np.array(anomaly_map.detach().cpu()).flatten()

                pixel_anomaly_scores[mask_cnt * self.config.data.image_size * self.config.data.image_size:(mask_cnt + 1) * self.config.data.image_size * self.config.data.image_size] = flatten_map
       
                mask_cnt += 1

        max_thresh, p_quant_thresh, k_sigma_thresh = threshold_computation(pixel_anomaly_scores)

        if self.config.data.pre_masking == True:
            mask_type = "pre_masking"
        else:
            mask_type = "post_masking"

        if self.config.data.image_registration == True:
            img_reg = "image_reg" 
        else:
            img_reg = "no_image_reg"

        if not os.path.exists(f'DDAD_main/results/{mask_type}/{img_reg}/{self.config.data.category}/checkpoint_{self.config.model.load_chp}/{masked}'):
            os.makedirs(f'DDAD_main/results/{mask_type}/{img_reg}/{self.config.data.category}/checkpoint_{self.config.model.load_chp}/{masked}')

        with open(f'DDAD_main/results/{mask_type}/{img_reg}/{self.config.data.category}/checkpoint_{self.config.model.load_chp}/{masked}/threshold_selection.txt', 'w') as f:
            f.write("THRESHOLDS COMPUTED ON NOMINAL VALIDATION DATA \n")
            f.write(f"MAX THRESHOLD: {max_thresh} \n")
            f.write(f"P-QUANT THRESHOLD: {p_quant_thresh} \n")
            f.write(f"K-SIGMA THRESHOLD: {k_sigma_thresh}")