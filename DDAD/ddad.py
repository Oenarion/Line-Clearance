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
import glob 
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

class DDAD:
    def __init__(self, unet, config) -> None:
        
        

        self.test_dataset = Dataset_maker(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=2,
        )
        self.testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size= config.data.test_batch_size,
            shuffle=False,
            # num_workers= config.model.num_workers,
            # drop_last=False,
        )

        self.unet = unet
        self.config = config
        self.reconstruction = Reconstruction(self.unet, self.config)
        # self.transform = transforms.Compose([
        #                     transforms.CenterCrop((224)), 
        #                 ])
        self.black_masks = []
        image_files = glob.glob(os.path.join(config.data.data_dir, config.data.category, "test", "*", "*.bmp"))
        for im in image_files:
            curr_im = np.array(Image.open(im).resize((self.config.data.image_size,self.config.data.image_size)))
            black_mask = np.all(curr_im == [0, 0, 0], axis=-1).astype(np.uint8)
            anomalous_mask = 1 - black_mask
            self.black_masks.append(anomalous_mask)

    def __call__(self) -> Any:
        feature_extractor = domain_adaptation(self.unet, self.config, fine_tune=False)
        feature_extractor.eval()
        
        labels_list = []
        predictions= []
        anomaly_map_list = []
        gt_list = []
        reconstructed_list = []
        forward_list = []

        if self.config.metrics.masking == True:
            masked = 'masking'
        else:
            masked = 'no_masking'
        
        threshold_name = 'test_threshold'
        idx = 0
        with torch.no_grad():
            for input_img, gt, labels in self.testloader:
                reg_mask = None
                # plt.imshow(np.transpose(np.array(gt[0]),(1,2,0)))
                # plt.show()
                input_img = input_img.to(self.config.model.device)
                x0 = self.reconstruction(input_img, input_img, self.config.model.w)[-1]
                if self.config.metrics.masking and self.config.data.pre_masking:
                    mask = Mask(self.config.data.image_size, self.config.model.device)
                    input_masked, x0 = mask.masking(input_img, x0, self.config.data.category)
                    masked = 'masking'
                else:
                    input_masked = input_img

                if self.config.data.image_registration and self.config.metrics.masking:
                    mask_tensor = torch.tensor(self.black_masks[idx], dtype=torch.float32).to(self.config.model.device)
                    mask_tensor = mask_tensor.unsqueeze(0) # channel dimensions
                    if self.config.data.pre_masking:
                        reg_mask = mask_tensor


                anomaly_map = heat_map(x0, input_masked, feature_extractor, self.config, reg_mask)

                if self.config.metrics.masking and not self.config.data.pre_masking:
                    anomaly_map = anomaly_map.squeeze(0)
                    mask = Mask(self.config.data.image_size, self.config.model.device)
                    applied_mask = mask.create_mask(mask.image_type_masks[self.config.data.category]).unsqueeze(0)
                    anomaly_map = anomaly_map * applied_mask
                    if self.config.data.image_registration == True:
                        anomaly_map = anomaly_map * mask_tensor
                    anomaly_map = anomaly_map.unsqueeze(0)

                forward_list.append(input_img)
                anomaly_map_list.append(anomaly_map)


                gt_list.append(gt)
                reconstructed_list.append(x0)
                for pred, label in zip(anomaly_map, labels):
                    labels_list.append(0 if label == 'good' else 1)
                    predictions.append(torch.max(pred).item())

                idx += 1

        if self.config.data.pre_masking == True:
            mask_type = "pre_masking"
        else:
            mask_type = "post_masking"

        if self.config.data.image_registration == True:
            img_reg = "image_reg" 
        else:
            img_reg = "no_image_reg"

        metric = Metric(labels_list, predictions, anomaly_map_list, gt_list, self.config)

        if self.config.metrics.threshold:
            if self.config.metrics.threshold_type == 0:
                threshold_name = "max_threshold"
            elif self.config.metrics.threshold_type == 1:
                threshold_name = 'p-quantile_threshold'
            else:
                threshold_name = 'k-sigma_threshold'
            with open(f'DDAD_main/results/{mask_type}/{img_reg}/{self.config.data.category}/checkpoint_{self.config.model.load_chp}/{masked}/threshold_selection.txt', 'r') as f:
                lines = f.readlines()
                threshold = float(lines[self.config.metrics.threshold_type + 1].split(':')[1])
                print(f"THRESHOLD USED: {threshold}")
                metric.update_threshold(threshold)
        else:
            metric.optimal_threshold()
        # print(f"OPTIMAL THRESHOLD: {thresh}")
        if not os.path.exists(f'DDAD_main/results/{mask_type}/{img_reg}/{self.config.data.category}/checkpoint_{self.config.model.load_chp}/{threshold_name}/{masked}'):
                os.makedirs(f'DDAD_main/results/{mask_type}/{img_reg}/{self.config.data.category}/checkpoint_{self.config.model.load_chp}/{threshold_name}/{masked}')
        # if not os.path.exists(f'DDAD_main/results/{self.config.data.category}/checkpoint_{self.config.model.load_chp}/{masked}'):
        #         os.mkdir(f'DDAD_main/results/{self.config.data.category}/checkpoint_{self.config.model.load_chp}/{masked}')     
        if self.config.metrics.auroc:
            img_auroc, img_auroc_curve = metric.image_auroc()
            pixel_auroc, pixel_auroc_curve = metric.pixel_auroc()
            print('AUROC: ({:.1f},{:.1f})'.format(img_auroc * 100, pixel_auroc * 100))
        if self.config.metrics.pro:
            pixel_pro, df_pro = metric.pixel_pro()
            print('PRO: {:.1f}'.format(pixel_pro * 100))
        if self.config.metrics.iou:
            pixel_iou = metric.pixel_IoU()
            print('IoU: {:.1f}'.format(pixel_iou * 100))
            ious, iou_fprs, iou_auc = metric.IoU_curve(pixel_auroc_curve[-1], pixel_auroc_curve[0])
        if self.config.metrics.pr:
            pr, pr_curve = metric.precision_recall()
            print('Precision Recall: {:.1f}'.format(pr * 100))
        if self.config.metrics.misclassifications:
            errors = metric.misclassified()
            print('ACCURACY: {:.1f}% \n'.format(((len(gt_list)-len(errors))/len(gt_list))*100))
        print("==============================")
        #write metrics results in a .txt
        if self.config.metrics.save_txt:
            with open(f'DDAD_main/results/{mask_type}/{img_reg}/{self.config.data.category}/checkpoint_{self.config.model.load_chp}/{threshold_name}/{masked}/metrics.txt', 'w') as f:
                f.write(f'Total images: {len(gt_list)}, Good: {labels_list.count(0)}, Anomalies: {len(gt_list) - labels_list.count(0)} \n')
                f.write('AUROC: ({:.1f},{:.1f}) \n'.format(img_auroc * 100, pixel_auroc * 100))
                f.write('PRO: {:.1f} \n'.format(pixel_pro * 100))
                f.write('IOU: {:.1f} \n'.format(pixel_iou * 100))
                f.write('AP: {:.1f} \n'.format(pr * 100))
                f.write('ACCURACY: {:.1f}% \n'.format(((len(gt_list)-len(errors))/len(gt_list))*100))
                f.write('ERRORS: \n')
                f.writelines(errors)
            f.close()

        reconstructed_list = torch.cat(reconstructed_list, dim=0)
        forward_list = torch.cat(forward_list, dim=0)
        anomaly_map_list = torch.cat(anomaly_map_list, dim=0)
        pred_mask = (anomaly_map_list > metric.threshold).float()
        gt_list = torch.cat(gt_list, dim=0)
        if self.config.metrics.visualisation:
            #image, noisy_image, GT, pred_mask, anomaly_map
            visualize(forward_list, reconstructed_list, gt_list, pred_mask, anomaly_map_list, self.config.data.category, masked, self.config.model.load_chp, threshold_name, img_reg, mask_type)
            visualize_curves(img_auroc_curve, pixel_auroc_curve, pr_curve, df_pro, img_auroc, pixel_auroc, pr, pixel_pro, ious, iou_fprs, iou_auc, self.config.data.category, masked, self.config.model.load_chp, img_reg, mask_type)

        if self.config.metrics.comparison:
            return img_auroc_curve, pixel_auroc_curve, df_pro, pr_curve, [iou_fprs, ious]