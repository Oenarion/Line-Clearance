import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import os
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from omegaconf import OmegaConf
import glob
from PIL import Image

# Add the 'DRAEM' directory to the system path
draem_path = os.path.join(os.getcwd(), 'DRAEM')
sys.path.append(draem_path)
ddad_path = os.path.join(os.getcwd(), 'DDAD_main')
sys.path.append(ddad_path)

# Now you can import the module
from DRAEM import model, data_loader, visualize, masking
from DDAD_main import unet, feature_extractor, reconstruction, dataset, anomaly_map
from DDAD_main import masking as mask_DDAD

categories = ['A','A2','B','B2','C']
masked = ['masking', 'no_masking']
img_reg = [False, True]
pre_masking = [False, True]
IMG_SIZE = 512

DRAEM_results_path = "DRAEM\\results"
DDAD_results_path = "DDAD_main\\results"
DRAEM_models_path = "DRAEM\\DRAEM_models"
DDAD_models_path = "DDAD_main\\checkpoints"
dataset_path = ["dataset\\AV_dataset_mvtec_like", "dataset\\AV_dataset_mvtec_like_aligned"]

save_folder = "networks_comparison"

thresh_map = {
    'max_threshold': 0,
    'p-quantile_threshold': 1,
    'k-sigma_threshold': 2
}

def threshold_decision(save_path, mask):
    
    with open(save_path, 'r') as f:
        lines = f.readlines()
        if mask == 'masking':
            acc = lines[6]
        else:
            acc = lines[-1]
        # Extract the portion containing the checkpoints
        chps_str = acc.split(':')[2].split(', BEST')[0].strip()
        threshs_str = acc.split(':')[3].strip()
        # Convert the string representation of the list into an actual list of integers
        chps = [int(x) for x in chps_str.strip('[]').split(',')]
        threshs = [x.strip(" []'") for x in threshs_str.split(',')]
        
        chp_iou = lines[4].split(':')[2].split(', BEST')[0].strip()
        chp_iou = [int(x) for x in chp_iou.strip('[]').split(',')]
        thresh_iou = lines[4].split(':')[3].strip()
        thresh_iou = [x.strip(" []'") for x in thresh_iou.split(',')]

        #invert them to use the last epochs first
        threshs = threshs[::-1]
        chps = chps[::-1]

        #if same results, use the best iou as threshold
        if chp_iou in chps and thresh_iou in threshs:
            print("USING THRESH WITH BEST IOU")
            chosen_chp = chp_iou[0]
            chosen_thresh = thresh_iou[0]
        else:
            print("BEST IOU DIDN'T GIVE BEST ACC")
            if 'max_threshold' in threshs:
                idx = threshs.index('max_threshold')
            elif 'k-sigma_threshold' in threshs:
                idx = threshs.index('k-sigma_threshold')
            else:
                idx = threshs.index('p-quantile_threshold')
            
            chosen_chp = chps[idx]
            chosen_thresh = threshs[idx]

        print(f"CHOSEN CHECKPOINT: {chosen_chp}")
        print(f"CHOSEN THRESHOLD TYPE: {chosen_thresh}")

    return chosen_chp, chosen_thresh

def show_tensor_mask(image):
    reverse_transforms = transforms.Compose([
        # transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.int8)),
    ])

    # Takes the first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    return reverse_transforms(image)

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

def create_masked_img(image, pred_mask):
    """
    Utility function to draw the original image (colored) with the mask and rectangles computed before
    Returns the masked image
    """

    anomaly_overlay = np.zeros_like(image)
    # print(np.unique(show_tensor_mask(pred_mask[idx])))
    anomaly_overlay[:,:,0] = np.where(pred_mask == 0, anomaly_overlay[:,:,0], 255)  # Red color for anomalies
    anomaly_overlay[:,:,1] = np.where(pred_mask == 0, anomaly_overlay[:,:,1], 0)  
    anomaly_overlay[:,:,2] = np.where(pred_mask == 0, anomaly_overlay[:,:,2], 0)  

    # Blend the original image and the anomaly overlay
    blended_img = 0.9 * image + 0.1 * anomaly_overlay
    return blended_img

def prepare_DRAEM(chp, category, dataset_path):
    model_rec = model.ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model_rec.load_state_dict(torch.load(os.path.join(DRAEM_models_path, category, f"{category}_TEST_REC_MODEL_{chp}.pth"), map_location='cuda:0'))
    model_rec.cuda()
    model_rec.eval()

    model_seg = model.DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.load_state_dict(torch.load(os.path.join(DRAEM_models_path, category, f"{category}_TEST_SEG_MODEL_{chp}.pth"), map_location='cuda:0'))
    model_seg.cuda()
    model_seg.eval()

    dataset = data_loader.DRAEMTestDataset(dataset_path + '\\' + category, resize_shape=[512, 512])
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    return model_rec, model_seg, dataloader

def prepare_DDAD(chp, category, config, dataset_path):
    unet_model = unet.UNetModel(config.data.image_size, 64, dropout=0.0, n_heads=4 ,in_channels=config.data.input_channel)
    chp_path = os.path.join(os.getcwd(), DDAD_models_path, category)
    checkpoint = torch.load(os.path.join(chp_path, str(chp)))
    unet_model = torch.nn.DataParallel(unet_model)
    unet_model.load_state_dict(checkpoint)    
    unet_model.to('cuda')
    checkpoint = torch.load(os.path.join(chp_path, str(chp)))
    unet_model.eval()

    fe = feature_extractor.domain_adaptation(unet_model, config, fine_tune=False, is_train=2)
    fe.eval()

    rec = reconstruction.Reconstruction(unet_model, config)

    data = dataset.Dataset_maker(dataset_path, category, config, 2)
    dataloader = torch.utils.data.DataLoader(
            data,
            batch_size= 1,
            shuffle=False,
        )
    
    return fe, rec, dataloader

def DRAEM_output(model_rec, model_seg, batch, category, mask, thresh, chp, black_masks, mask_label, registration, idx):
    gray_batch = batch["image"].cuda()

    gray_rec = model_rec(gray_batch)
    joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

    out_mask = model_seg(joined_in)

    if mask == 'masking':
        mask_DRAEM = masking.Mask(IMG_SIZE, 'cuda')
        pred_mask, gray_rec = mask_DRAEM.masking(gray_rec, out_mask, category)
        if mask_label:
            # Clamping values to avoid numerical issues
            #pred_mask_clamped = torch.clamp(pred_mask, min=-1e4, max=1e4)
            if registration:
                black_mask = torch.tensor(black_masks[idx]).unsqueeze(0).unsqueeze(0)
                black_mask = black_mask.to(pred_mask.device)
                pred_mask[:, 1, :, :] = torch.where(black_mask == 0, float('-inf'), pred_mask[:, 1, :, :])
            out_mask_sm = torch.softmax(pred_mask, dim=1)
        else:
            bit_mask = mask.create_mask(mask.image_type_masks[category])
            out_mask_sm = torch.softmax(out_mask, dim=1)
            if registration:
                black_mask = torch.tensor(black_masks[idx]).unsqueeze(0).unsqueeze(0)
                black_mask = black_mask.to(pred_mask.device)
                out_mask_sm[:, 1, :, :] = torch.where(black_mask == 0, 0, out_mask_sm[:, 1, :, :])

            out_mask_sm[:, 1, :, :] = torch.where(bit_mask == 0, 0, out_mask_sm[:, 1, :, :])
    else:
        out_mask_sm = torch.softmax(out_mask, dim=1)

    t_mask = out_mask_sm[:, 1:, :, :]
    rec_image = gray_rec[0]
    orig_image = gray_batch[0]
    out_mask = t_mask[0]


    with open(f'DRAEM/results/{mask_label}/{registration}/{category}/checkpoint_{chp}/{mask}/threshold_selection.txt', 'r') as f:
        lines = f.readlines()
        threshold = float(lines[thresh_map[thresh] + 1].split(':')[1])
        # try to have a better evaluation
        threshold = threshold - 0.1*threshold

    thresh_mask = visualize.extract_mask(np.array(out_mask[0].detach().cpu()), threshold, 512)
    return orig_image, rec_image, out_mask, thresh_mask

def DDAD_output(input_img, rec, category, thresh, chp, fe, mask, config, black_masks, mask_label, registration, idx):
    input_img = input_img.to(config.model.device)
    x0 = rec(input_img, input_img, config.model.w)[-1]
    reg_mask = None

    if mask == 'masking' and mask_label == 'pre_masking':
        mask_net = mask_DDAD.Mask(config.data.image_size, config.model.device)
        input_masked, x0 = mask_net.masking(input_img, x0, category)
    else:
        input_masked = input_img

    if registration and mask == 'masking':
        mask_tensor = torch.tensor(black_masks[idx], dtype=torch.float32).to(config.model.device)
        mask_tensor = mask_tensor.unsqueeze(0) # channel dimensions
        if mask_label == 'pre_masking':
            reg_mask = mask_tensor
    
    anom_map = anomaly_map.heat_map(x0, input_masked, fe, config, reg_mask)

    if mask == 'masking' and mask_label == 'post_masking':
        anom_map = anom_map.squeeze(0)
        mask_post = mask_DDAD.Mask(config.data.image_size, config.model.device)
        applied_mask = mask_post.create_mask(mask_post.image_type_masks[category]).unsqueeze(0)
        anom_map = anom_map * applied_mask
        if registration == 'image_registration':
            anom_map = anom_map * mask_tensor
        anom_map = anom_map.unsqueeze(0)

    with open(f'DDAD_main/results/{mask_label}/{registration}/{category}/checkpoint_{chp}/{mask}/threshold_selection.txt', 'r') as f:
        lines = f.readlines()
        threshold = float(lines[thresh_map[thresh] + 1].split(':')[1])

    pred_mask = (anom_map > threshold).float()

    return x0, pred_mask, anom_map

def prepare_DRAEM_images(orig_img, rec_img, out_mask, thresh_mask):
    orig_img = np.transpose(np.array(orig_img.detach().cpu()), (1,2,0))
    rec_img = np.transpose(np.array(rec_img.detach().cpu()), (1,2,0))
    out_mask = np.array(out_mask[0].detach().cpu())
    mask_img = create_masked_img(orig_img, thresh_mask)

    return orig_img, rec_img, out_mask, mask_img

def prepare_DDAD_images(input_img, x0, anomaly_map, pred_mask):
    
    anomaly_overlay = np.zeros_like(show_tensor_image(input_img))
    anomaly_overlay[:,:,0] = np.where(show_tensor_mask(pred_mask)[:,:,0] == 0, anomaly_overlay[:,:,0], 255)  # Red color for anomalies
    anomaly_overlay[:,:,1] = np.where(show_tensor_mask(pred_mask)[:,:,0] == 0, anomaly_overlay[:,:,1], 0)  
    anomaly_overlay[:,:,2] = np.where(show_tensor_mask(pred_mask)[:,:,0] == 0, anomaly_overlay[:,:,2], 0)  
    blended_img = 0.7 * show_tensor_image(input_img) + 0.3 * anomaly_overlay
    blended_img = blended_img.astype(np.uint8)
    input_img = show_tensor_image(input_img)
    x0 = show_tensor_image(x0)
    anomaly_map = show_tensor_image(anomaly_map.detach())
    pred_mask = show_tensor_mask(pred_mask)

    return input_img, x0, anomaly_map, pred_mask, blended_img

def save_thresh_used(thresh_DRAEM, chp_DRAEM, thresh_DDAD, chp_DDAD, save_path):

    with open(f'{save_path}\\thresh_chp_info.txt', 'w') as f:
        f.write("="*50)
        f.write("\n")
        f.write(f"THRESHOLD TYPE USED FOR DRAEM: {thresh_DRAEM} \n")
        f.write(f"CHECKPOINT USED FOR DRAEM: {chp_DRAEM} \n")
        f.write("="*50)
        f.write("\n")
        f.write(f"THRESHOLD TYPE USED FOR DDAD: {thresh_DDAD} \n")
        f.write(f"CHECKPOINT USED FOR DDAD: {chp_DDAD} \n")
        f.write("="*50)

def save_reconstructions(orig_img, rec_DRAEM, rec_DDAD, save_path, idx):
    
    _, ax = plt.subplots(1, 3,figsize = [30,30])
    
    ax[0].imshow(orig_img)
    ax[0].set_title("Original image", fontsize=30)
    ax[1].imshow(rec_DRAEM)
    ax[1].set_title(f"DRAEM reconstruction", fontsize=30)
    ax[2].imshow(rec_DDAD)
    ax[2].set_title(f"DDAD reconstruction", fontsize=30)

    plt.tight_layout()
    plt.savefig(f'{save_path}\\reconstruction_{idx}', bbox_inches='tight')
    plt.close()

def save_anomaly_maps(anomaly_map_DRAEM, thresh_map_DRAEM, anomaly_map_DDAD, thresh_map_DDAD, save_path, idx):
    
    _, ax = plt.subplots(2, 2,figsize = [30,30])
    
    ax[0][0].imshow(anomaly_map_DRAEM)
    ax[0][0].set_title("Anomaly map DRAEM", fontsize=30)
    ax[0][1].imshow(thresh_map_DRAEM)
    ax[0][1].set_title(f"Thresholded map DRAEM", fontsize=30)
    ax[1][0].imshow(anomaly_map_DDAD)
    ax[1][0].set_title(f"Anomaly map DDAD", fontsize=30)
    ax[1][1].imshow(thresh_map_DDAD)
    ax[1][1].set_title(f"Thresholded map DDAD", fontsize=30)

    plt.tight_layout()
    plt.savefig(f'{save_path}\\anomaly_maps_{idx}', bbox_inches='tight')
    plt.close()

def save_final_result(final_img_DRAEM, final_img_DDAD, save_path, idx):

    _, ax = plt.subplots(1, 2, figsize = [30, 30])

    ax[0].imshow(final_img_DRAEM)
    ax[0].set_title("final reconstruction DRAEM", fontsize=30)
    ax[1].imshow(final_img_DDAD)
    ax[1].set_title(f"final reconstruction DDAD", fontsize=30)

    plt.tight_layout()
    plt.savefig(f'{save_path}\\anomaly_results_{idx}', bbox_inches='tight')
    plt.close()


def main():
    if not os.path.exists(f'{save_folder}'):
        os.makedirs(f'{save_folder}')

    config = OmegaConf.load(os.path.join(os.getcwd(),'DDAD_main\\config.yaml'))
    
    print("STARTING COMPARISON...")
    for mask_type in pre_masking:
        if mask_type:
            mask_label = "pre_masking"
        else:
            mask_label = "post_masking"
        print(f"PRE MASKING? {mask_type}")
        for reg in img_reg:
            if reg:
                data_dir = dataset_path[1]
                registration = "image_reg"
            else:
                data_dir = dataset_path[0]
                registration = "no_image_reg"

            print(f"IMAGE REGISTRATION? {reg}")
            for category in categories:
                # if not os.path.exists(f'{save_folder}\\{mask_label}\\{registration}\\{category}'):
                #     os.makedirs(f'{save_folder}\\{mask_label}\\{registration}\\{category}')
                black_masks = []
                image_files = glob.glob(os.path.join(data_dir, category, "test", "*", "*.bmp"))
                for im in image_files:
                    curr_im = np.array(Image.open(im).resize((IMG_SIZE, IMG_SIZE)))
                    black_mask = np.all(curr_im == [0, 0, 0], axis=-1).astype(np.uint8)
                    anomalous_mask = 1 - black_mask
                    black_masks.append(anomalous_mask)
                for mask in masked:
                    if not os.path.exists(f'{save_folder}\\{mask_label}\\{registration}\\{category}\\{mask}'):
                        os.makedirs(f'{save_folder}\\{mask_label}\\{registration}\\{category}\\{mask}')

                    save_path_DRAEM = os.path.join(os.getcwd(),DRAEM_results_path,mask_label,registration,category,'metrics_report.txt')
                    print("CHOSING DRAEM CHP AND THRESHOLD")
                    chosen_chp_DRAEM, chosen_thresh_DRAEM = threshold_decision(save_path_DRAEM, mask)

                    save_path_DDAD = os.path.join(os.getcwd(),DDAD_results_path,mask_label,registration,category,'metrics_report.txt')
                    print("CHOSING DDAD CHP AND THRESHOLD")
                    chosen_chp_DDAD, chosen_thresh_DDAD = threshold_decision(save_path_DDAD, mask)
                    
                    model_rec, model_seg, dataloader_DRAEM = prepare_DRAEM(chosen_chp_DRAEM, category, data_dir)
                    fe, rec, dataloader_DDAD = prepare_DDAD(chosen_chp_DDAD, category, config, data_dir)
                    print(f"PREPARED DRAEM AND DDAD NETWORKS AND DATALOADERS")

                    ddad_iter = iter(dataloader_DDAD)
                    idx = 0
                    save_path = f'{save_folder}\\{mask_label}\\{registration}\\{category}\\{mask}'
                    save_thresh_used(chosen_thresh_DRAEM, chosen_chp_DRAEM, chosen_thresh_DDAD, chosen_chp_DDAD, save_path)
                    #iterate for both draem and ddad
                    print("STARTING EVALUATION...")
                    for  _, sample_batched in enumerate(dataloader_DRAEM):
                        image_path = f'{save_path}\\image_{idx}'
                        if not os.path.exists(f'{image_path}'):
                            os.makedirs(f'{image_path}')

                        orig_img, rec_img, out_mask, thresh_mask = DRAEM_output(model_rec, model_seg, sample_batched, category, mask, chosen_thresh_DRAEM, chosen_chp_DRAEM, black_masks, mask_label, registration, idx)
                        input_img, _, _ = next(ddad_iter)
                        x0, pred_mask, anomaly_map = DDAD_output(input_img, rec, category, chosen_thresh_DDAD, chosen_chp_DDAD, fe, mask, config, black_masks, mask_label, registration, idx)

                        #prepare data in the right format
                        orig_img, rec_img, out_mask, mask_img = prepare_DRAEM_images(orig_img, rec_img, out_mask, thresh_mask)
                        input_img, x0, anomaly_map, pred_mask, blended_img = prepare_DDAD_images(input_img, x0, anomaly_map, pred_mask)
                        
                        print(f"SAVING IMG_{idx}...")
                        #save the files
                        save_reconstructions(orig_img, rec_img, x0, image_path, idx)
                        save_anomaly_maps(out_mask, thresh_mask, anomaly_map, pred_mask, image_path, idx)
                        save_final_result(mask_img, blended_img, image_path, idx)
                        idx += 1


if __name__ == '__main__':
    main()