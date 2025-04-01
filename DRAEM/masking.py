import numpy as np
import torch
import matplotlib.pyplot as plt

"""
THIS ARE ALL EMPIRICAL VALUES BASED ON THE CURRENT DATASET, THEY NEED TO BE CHANGED ACCORDINGLY WITH FUTURES CHANGES!!!
"""
ORIG_SHAPE = (2048, 3072)
A_MASKS = np.array([[[0,1550],[0,760]],[[1400, 1600], [0, 850]],[[0,110], [2100,2500]], [[0,160],[1550,3072]]]) 
A2_MASKS = np.array([[[0,1400],[0,1200]],[[0,1300], [1200,2200]], [[0,750],[1700,3072]], [[0,1500],[2900,3072]], [[0,1550], [0,400]]]) 
B_MASKS = np.array([[[0,2048],[1780,3072]]]) 
B2_MASKS = np.array([[[0,2048],[0,250]], [[1480,2048],[0,1400]], [[1900,2048],[1400,1800]], [[1950,2048],[1800,3072]], [[650,2048],[2500,3072]]]) 
C_MASKS = np.array([[[0,2048],[0,1000]], [[0,2048],[2100,3072]], [[0,250],[1000,2100]], [[2000, 2048], [0,3072]]])
MASK_ARR = [A_MASKS, A2_MASKS, B_MASKS, B2_MASKS, C_MASKS]


class Mask():
    def __init__(self, size, device):
        self.size = size
        self.orig_shape = ORIG_SHAPE
        self.device = device
        masks = []
        for mask in MASK_ARR:
            masks.append(self.resize_masks(mask, self.size, self.orig_shape))

        self.image_type_masks = {
            'A':masks[0],
            'A2':masks[1],
            'B':masks[2],
            'B2':masks[3],
            'C':masks[4],
        }

    def create_mask(self, mask_type):
        """
        Creates the mask array which will be later used for masking
        """

        new_mask = torch.ones((self.size,self.size))
        new_mask = new_mask.to(self.device)
        for mask in mask_type:
            x = mask[0]
            y = mask[1]
            new_mask[int(x[0]):int(x[1]),int(y[0]):int(y[1])] = 0
        
        new_mask = new_mask.unsqueeze(0).unsqueeze(0)

        return new_mask
    
    def masking(self, input_img, pred_mask, category):
        """
        Applies masking on both the input image and the reconstructed image.

        Args:
            - pred_mask (tensor): predicted mask
            - category (str): the category of the image  

        Returns the masked images
        """
        mask_type = self.image_type_masks[category]
        mask = self.create_mask(mask_type)
        input_img = input_img * mask
        pred_mask[:, 1, :, :] *= mask.squeeze(0)  # Remove batch dim for the mask when applying

        # Set the masked-out regions to -inf in the anomaly channel, so softmax outputs 0 probability
        pred_mask[:, 1, :, :][pred_mask[:, 1, :, :] == 0] = float('-inf')

        return pred_mask, input_img

    def resize_masks(self, masks, size, original_shape):
        """
        Since masks are computed on full size images, if the image gets resized during evaluation the mask should as well.

        Args:
            - masks (np.array): The masks which we have to resize
            - size (tuple of int): The new width and height
            - original_shape (tuple of int): starting width and height 
        
        Returns:
            - the resized masks (np.array)
        """

        resized_masks = []
        resize_height, resize_width = size, size
        height, width = original_shape
        for curr_mask in masks:
            x = curr_mask[0]
            y = curr_mask[1]
            resize_x = [(resize_height/height)* x[0], (resize_height/height)* x[1]]
            resize_y = [(resize_width/width)* y[0], (resize_width/width)* y[1]]
            resized_masks.append([resize_x,resize_y])
        return np.array(resized_masks)