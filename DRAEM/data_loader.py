import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np
import random

class DRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        dir_str = os.getcwd() + "\\"+root_dir+"\\test"
        bad_images = sorted(glob.glob(os.getcwd() + "\\"+root_dir+"\\test\\reject\\*.bmp"))
        good_images = sorted(glob.glob(os.getcwd() + "\\"+root_dir+"\\test\\good\\*.bmp"))
        self.images = good_images + bad_images
        gts = sorted(glob.glob(os.getcwd() + "\\"+root_dir+"\\ground_truth\\*\\*.bmp"))
        good_gts = [0] * len(good_images)
        self.gts = good_gts + gts
        self.resize_shape=resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        mask_path = self.gts[idx]
        if mask_path == 0:
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample



class DRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, is_train, resize_shape=None, cut=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.is_train = is_train

        image_files = sorted(glob.glob(os.getcwd() + "\\"+root_dir+"\\*.bmp"))
        random.seed(42)
        train_data_len = int(len(image_files) * 0.8)
        
        if self.is_train == 0:
            self.image_paths = random.sample(image_files, train_data_len)
        else:
            self.image_paths = list(set(image_files) - set(random.sample(image_files, train_data_len)))

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"\\*\\*.jpg"))
        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        
        self.cut = cut

    def cutout(self, image):
        """
        Puts a white or grey rectangle patch in an anomaly free image.
        The reason is that they should represent the anomalies which can appear in the image (e.g. blisters of product)

        Returns:
            Augmented image
        """
        
        cutout_image = np.copy(image)
        img_height, img_width = image.shape[:2]

        # Ensure that the cutout size doesn't exceed the image size
        rand_width = random.randint(10, img_width//2)
        rand_height = random.randint(10, img_height//2)

        rand_color = random.choice([0.5, 1])
        rand_x = random.randint(0, img_width - rand_width)
        rand_y = random.randint(0, img_height - rand_height)
        final_x = rand_x + rand_width
        final_y = rand_y + rand_height

        # Apply the cutout
        cutout_image[rand_y:final_y, rand_x:final_x] = rand_color

        # Create a mask of the same size as the image
        mask = np.zeros((img_height, img_width), dtype=np.float32)
        mask[rand_y:final_y, rand_x:final_x] = 1.0

        # Make sure both the image and mask have the same shape and are transposed correctly
        cutout_image = np.transpose(cutout_image, (2, 0, 1)).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension for mask (1, H, W)

        return cutout_image, mask

    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        do_perl = torch.rand(1).numpy()[0] > 0.2
        if do_perl:
            image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
            augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
            augmented_image = np.transpose(augmented_image, (2, 0, 1))
            image = np.transpose(image, (2, 0, 1))
            anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
            return image, augmented_image, anomaly_mask, has_anomaly
        else:
            image =  np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
            if self.cut:      
                augmented_image, mask = self.cutout(image)
            image = np.transpose(image, (2, 0, 1))
            return image, augmented_image, mask, np.array(1, dtype=np.float32)

    def __getitem__(self, idx):
        if self.is_train == 0:
            idx = torch.randint(0, len(self.image_paths), (1,)).item()
            anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
            image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                            self.anomaly_source_paths[anomaly_source_idx])
            sample = {'image': image, "anomaly_mask": anomaly_mask,
                    'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}
        else:
            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_path = self.image_paths[idx]
            image = cv2.imread(img_path)
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            image =  np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))

            sample = {'image': image, 'idx': idx}

        return sample