import torch
import torch.nn.functional as F
from data_loader import DRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
from metric_utils import *
from visualize import *
from masking import * 
import glob
from PIL import Image


def test(config):
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []

        # run_name_rec = category + "_" + config.data.run_name +"_REC_MODEL_"+str(config.test.load_epoch)+".pth"
        # run_name_seg = category + "_" + config.data.run_name +"_SEG_MODEL_"+str(config.test.load_epoch)+".pth"

    run_name_rec = config.data.category + "_" + config.data.run_name +"_REC_MODEL_"+str(config.data.load_epoch)+".pth"
    run_name_seg = config.data.category + "_" + config.data.run_name +"_SEG_MODEL_"+str(config.data.load_epoch)+".pth"

    masked = 'masking' if config.metrics.masking else 'no_masking'

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(os.path.join(config.data.checkpoint_path, config.data.category, run_name_rec), map_location='cuda:0'))
    model.cuda()
    model.eval()

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.load_state_dict(torch.load(os.path.join(config.data.checkpoint_path, config.data.category, run_name_seg), map_location='cuda:0'))
    model_seg.cuda()
    model_seg.eval()

    dataset = DRAEMTestDataset(config.data.data_path + '\\' + config.data.category, resize_shape=[config.data.img_size, config.data.img_size])
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    total_pixel_scores = np.zeros((config.data.img_size * config.data.img_size * len(dataset)))
    total_gt_pixel_scores = np.zeros((config.data.img_size * config.data.img_size * len(dataset)))
    mask_cnt = 0

    anomaly_score_gt = []
    anomaly_score_prediction = []

    black_masks = []
    image_files = glob.glob(os.path.join(config.data.data_path, config.data.category, "test", "*", "*.bmp"))
    for im in image_files:
        curr_im = np.array(Image.open(im).resize((config.data.img_size,config.data.img_size)))
        black_mask = np.all(curr_im == [0, 0, 0], axis=-1).astype(np.uint8)
        anomalous_mask = 1 - black_mask
        black_masks.append(anomalous_mask)

    for idx, sample_batched in enumerate(dataloader):

        gray_batch = sample_batched["image"].cuda()

        is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
        anomaly_score_gt.append(is_normal)
        true_mask = sample_batched["mask"]
        true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

        gray_rec = model(gray_batch)
        joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

        out_mask = model_seg(joined_in)

        # plt.imshow(np.transpose(gray_batch.detach().cpu().numpy()[0], (1,2,0)))
        # plt.show()
        # plt.imshow(np.transpose(gray_rec.detach().cpu().numpy()[0], (1,2,0)))
        # plt.show()

        if config.metrics.masking:
            mask = Mask(config.data.img_size, 'cuda')
            pred_mask, _ = mask.masking(gray_rec, out_mask, config.data.category)
            if config.data.pre_masking:
                # Clamping values to avoid numerical issues
                #pred_mask_clamped = torch.clamp(pred_mask, min=-1e4, max=1e4)
                if config.data.image_registration:
                    black_mask = torch.tensor(black_masks[idx]).unsqueeze(0).unsqueeze(0)
                    black_mask = black_mask.to(pred_mask.device)
                    pred_mask[:, 1, :, :] = torch.where(black_mask == 0, float('-inf'), pred_mask[:, 1, :, :])
                out_mask_sm = torch.softmax(pred_mask, dim=1)
            else:
                bit_mask = mask.create_mask(mask.image_type_masks[config.data.category])
                out_mask_sm = torch.softmax(out_mask, dim=1)
                if config.data.image_registration:
                    black_mask = torch.tensor(black_masks[idx]).unsqueeze(0).unsqueeze(0)
                    black_mask = black_mask.to(pred_mask.device)
                    out_mask_sm[:, 1, :, :] = torch.where(black_mask == 0, 0, out_mask_sm[:, 1, :, :])

                out_mask_sm[:, 1, :, :] = torch.where(bit_mask == 0, 0, out_mask_sm[:, 1, :, :])
        else:
            out_mask_sm = torch.softmax(out_mask, dim=1)


        out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()
        # plt.imshow(out_mask_cv)
        # plt.show()

        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                            padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)

        anomaly_score_prediction.append(image_score)

        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        total_pixel_scores[mask_cnt * config.data.img_size * config.data.img_size:(mask_cnt + 1) * config.data.img_size * config.data.img_size] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * config.data.img_size * config.data.img_size:(mask_cnt + 1) * config.data.img_size * config.data.img_size] = flat_true_mask
        mask_cnt += 1


    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)
    auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
    img_auroc_curve = roc_curve(anomaly_score_gt, anomaly_score_prediction)
    #ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

    if config.data.pre_masking == True:
        mask_type = "pre_masking"
    else:
        mask_type = "post_masking"

    if config.data.image_registration == True:
        img_reg = "image_reg" 
    else:
        img_reg = "no_image_reg"

    if config.metrics.threshold:
        if config.metrics.threshold_type == 0:
            threshold_name = "max_threshold"
        elif config.metrics.threshold_type == 1:
            threshold_name = 'p-quantile_threshold'
        else:
            threshold_name = 'k-sigma_threshold'
        with open(f'DRAEM/results/{mask_type}/{img_reg}/{config.data.category}/checkpoint_{config.data.load_epoch}/{masked}/threshold_selection.txt', 'r') as f:
            lines = f.readlines()
            threshold = float(lines[config.metrics.threshold_type + 1].split(':')[1])
            print(f"THRESHOLD SELECTED: {threshold}")
            # try to have a better evaluation
            threshold = threshold - 0.1*threshold
            print(f"THRESHOLD USED: {threshold}")
    else:
        threshold = optimal_threshold()
    print(f"THRESHOLD: {threshold}")
    errors = misclassified(anomaly_score_prediction, anomaly_score_gt, threshold)
    predictions_int = (torch.tensor(anomaly_score_prediction) > torch.tensor(threshold)).int()
    if not os.path.exists(f'DRAEM/results/{mask_type}/{img_reg}/{config.data.category}/checkpoint_{config.data.load_epoch}/{threshold_name}/{masked}'):
        os.makedirs(f'DRAEM/results/{mask_type}/{img_reg}/{config.data.category}/checkpoint_{config.data.load_epoch}/{threshold_name}/{masked}')
    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:config.data.img_size * config.data.img_size * mask_cnt]
    total_pixel_scores = total_pixel_scores[:config.data.img_size * config.data.img_size * mask_cnt]
    
    

    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    pixel_auroc_curve = roc_curve(total_gt_pixel_scores, total_pixel_scores)
    ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
    pr_curve = precision_recall_curve(total_gt_pixel_scores, total_pixel_scores)
    # tensor_gt_pixel_scores = torch.from_numpy(total_gt_pixel_scores)
    # tensor_pixel_scores = torch.from_numpy(total_pixel_scores)
    # print(tensor_gt_pixel_scores)
    pro_auc, df_pro = pixel_pro(total_gt_pixel_scores, total_pixel_scores, config.data.img_size)
    iou = pixel_IoU(total_pixel_scores, total_gt_pixel_scores, threshold)
    ious, iou_fprs, iou_auc = IoU_curve(pixel_auroc_curve[-1], pixel_auroc_curve[0], total_pixel_scores, total_gt_pixel_scores)
    

    #obj_ap_pixel_list.append(ap_pixel)
    #obj_auroc_pixel_list.append(auroc_pixel)
    #obj_auroc_image_list.append(auroc)
    #obj_ap_image_list.append(ap)
    print('AUROC: ({:.1f},{:.1f}) \n'.format(auroc * 100, auroc_pixel * 100))
    print('PRO: {:.1f} \n'.format(pro_auc * 100))
    print('IOU: {:.1f} \n'.format(iou * 100))
    print('AP: {:.1f} \n'.format(ap_pixel * 100))
    print('ACCURACY: {:.1f}% \n'.format(((len(anomaly_score_gt)-len(errors))/len(anomaly_score_gt))*100))

    with open(f'DRAEM/results/{mask_type}/{img_reg}/{config.data.category}/checkpoint_{config.data.load_epoch}/{threshold_name}/{masked}/metrics.txt', 'w') as f:
        f.write(f'Total images: {len(anomaly_score_gt)}, Good: {(anomaly_score_gt == 0).sum()}, Anomalies: {len(anomaly_score_gt) - (anomaly_score_gt == 0).sum()} \n')
        f.write('AUROC: ({:.1f},{:.1f}) \n'.format(auroc * 100, auroc_pixel * 100))
        f.write('PRO: {:.1f} \n'.format(pro_auc * 100))
        f.write('IOU: {:.1f} \n'.format(iou * 100))
        f.write('AP: {:.1f} \n'.format(ap_pixel * 100))
        f.write('ACCURACY: {:.1f}% \n'.format(((len(anomaly_score_gt)-len(errors))/len(anomaly_score_gt))*100))
        f.write('ERRORS: \n')
        f.writelines(errors)

    # print("PIXEL PRO:  " +str(pro))
    # print("IOU:  " +str(iou))
    print("==============================")
    if config.metrics.comparison:
        return img_auroc_curve, pixel_auroc_curve, df_pro, pr_curve, [iou_fprs, ious]
    
    if config.metrics.visualisation:
        print("Now visualizing results...")
        ## HAVE TO DO IT LIKE THIS BECAUSE IT SATURATES THE MEMORY OTHERWISE
        for idx, sample_batched in enumerate(dataloader):

            gray_batch = sample_batched["image"].cuda()

            is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]

            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            gray_rec = model(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

            out_mask = model_seg(joined_in)

            if config.metrics.masking:
                mask = Mask(config.data.img_size, 'cuda')
                pred_mask, gray_rec = mask.masking(gray_rec, out_mask, config.data.category)
                # Clamping values to avoid numerical issues
                pred_mask_clamped = torch.clamp(pred_mask, min=-1e4, max=1e4)

                out_mask_sm = torch.softmax(pred_mask_clamped, dim=1)
            else:
                out_mask_sm = torch.softmax(out_mask, dim=1)

            out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()

            t_mask = out_mask_sm[:, 1:, :, :]
            rec_image = gray_rec[0]
            orig_image = gray_batch[0]
            out_mask = t_mask[0]
            gt_mask = true_mask[0]
            thresh_mask = extract_mask(np.array(out_mask[0].detach().cpu()), threshold, config.data.img_size)
            show_results(orig_image, rec_image, out_mask, gt_mask, thresh_mask, predictions_int, config.data.category, str(config.data.load_epoch), masked, idx, threshold_name, img_reg, mask_type)
            visualize_curves(img_auroc_curve, pixel_auroc_curve, pr_curve, df_pro, auroc, auroc_pixel, ap_pixel, pro_auc, ious, iou_fprs, iou_auc, config.data.category, masked, config.data.load_epoch, img_reg, mask_type)
# write_results_to_file(config.data.run_name+"_"+str(config.test.load_epoch), obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)