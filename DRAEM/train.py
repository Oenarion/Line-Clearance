import torch
from data_loader import DRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from model import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
import os
import time
import numpy as np
import matplotlib.pyplot as plt

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def plot_images(orig_img, aug_orig_img, rec_img, gt_mask, mask_out, epoch, dir):
    
    orig_img = np.transpose(np.array(orig_img.detach().cpu()), (1,2,0))
    aug_orig_img = np.transpose(np.array(aug_orig_img.detach().cpu()), (1,2,0))
    rec_img = np.clip(np.transpose(np.array(rec_img.detach().cpu()), (1,2,0)), 0, 1)
    gt_mask = np.array(gt_mask[0].detach().cpu())
    mask_out = np.array(mask_out[0].detach().cpu())


    _, ax = plt.subplots(1,5 ,figsize = [20,20])

    ax[0].imshow(orig_img)
    ax[0].set_title("Original image")
    ax[1].imshow(aug_orig_img)
    ax[1].set_title(f"Augmented image")
    ax[2].imshow(rec_img)
    ax[2].set_title("Reconstructed image")
    ax[3].imshow(gt_mask, cmap='gray')
    ax[3].set_title("GT anomaly mask")
    ax[4].imshow(mask_out, cmap='gray')
    ax[4].set_title(f"Predicted anomaly mask")

    os.makedirs(f"DRAEM\\DRAEM_images\\{dir}", exist_ok=True)
    plt.savefig(f'DRAEM\\DRAEM_images\\{dir}\\EPOCH_{epoch+1}.png')
    plt.close()


def train_on_device(config):

    print("Building network...")
    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.cuda()
    model.apply(weights_init)

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.cuda()
    model_seg.apply(weights_init)
    print("Network built!")

    optimizer = torch.optim.Adam([
                                    {"params": model.parameters(), "lr": 0.0001},
                                    {"params": model_seg.parameters(), "lr": 0.0001}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[config.train.epochs*0.4,config.train.epochs*0.7],gamma=0.2, last_epoch=-1)

    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    loss_focal = FocalLoss()

    print("Preparing dataset")
    dataset = DRAEMTrainDataset(config.data.data_path + '\\' + config.data.category + '\\train\\good', config.data.anomaly_source_path, 0, resize_shape=[config.data.img_size, config.data.img_size], cut=config.train.cut)

    dataloader = DataLoader(dataset, batch_size=config.train.batch_size,
                            shuffle=True)
    print("Dataset prepared!")
    best_epoch = 0
    n_iter = 0
    # best_loss = 9999

    train_start = time.time()

    run_name = config.data.category + "_" + config.data.run_name
    print("Starting training...")
    for epoch in range(config.train.epochs):
        start = time.time()
        counter = 0
        running_loss = 0
        print("Epoch: "+str(epoch))
        for _, sample_batched in enumerate(dataloader):
            counter += 1
            gray_batch = sample_batched["image"].cuda()
            aug_gray_batch = sample_batched["augmented_image"].cuda()
            anomaly_mask = sample_batched["anomaly_mask"].cuda()

            gray_rec = model(aug_gray_batch)
            joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            l2_loss = loss_l2(gray_rec,gray_batch)
            ssim_loss = loss_ssim(gray_rec, gray_batch)

            segment_loss = loss_focal(out_mask_sm, anomaly_mask)
            loss = l2_loss + ssim_loss + segment_loss

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            t_mask = out_mask_sm[:, 1:, :, :]
            running_loss += loss
            n_iter +=1
        if ((epoch+1) % 500) == 0:
            plot_images(gray_batch[0], aug_gray_batch[0], gray_rec[0], anomaly_mask[0], t_mask[0], epoch, config.data.run_name)
        
        avg_train_loss = running_loss / counter
        
        end = time.time()
        print(f"[ End of epoch {epoch} ], [ elapsed time: {end - start} ], [LOSS train: {avg_train_loss} ]")
        scheduler.step()

        #save best epoch
        # if avg_train_loss < best_loss:
        #     best_loss = avg_train_loss
        #     best_epoch = epoch
        #     print(f"new best epoch {epoch}")
        #     torch.save(model.state_dict(), f"DRAEM\\DRAEM_models\\{run_name}_REC_MODEL_{config.train.epochs}_best_epoch.pth")
        #     torch.save(model_seg.state_dict(), f"DRAEM\\DRAEM_models\\{run_name}_SEG_MODEL_{config.train.epochs}_best_epoch.pth")    
        #save after x epochs
        if (epoch+1) >= config.train.save_from:
            #save every 250 epochs
            if ((epoch+1) % 250) == 0 and (epoch+1) >= 2000:
                torch.save(model.state_dict(), f"DRAEM\\DRAEM_models\\{config.data.category}\\{run_name}_REC_MODEL_{(epoch+1)}.pth")
                torch.save(model_seg.state_dict(), f"DRAEM\\DRAEM_models\\{config.data.category}\\{run_name}_SEG_MODEL_{(epoch+1)}.pth")
    
    train_end = time.time()
    print(f"TOTAL ELAPSED TIME OF TRAINING: {train_end - train_start}")
    #print(f"BEST EPOCH: {best_epoch}")

