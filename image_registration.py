import glob
import os
import cv2
import numpy as np
import sys

def transform_img(img1_color, img2_color, gt=None):

    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with 
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    # Match the two sets of descriptors.
    matches = list(matcher.match(d1, d2))

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key = lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*0.9)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color,
                        homography, (width, height))
    if gt is not None:
        transformed_gt = cv2.warpPerspective(gt,
                        homography, (width, height))
        return transformed_img, transformed_gt

    return transformed_img

def main():

    dataset_path = sys.argv[1]
    save_path = sys.argv[2]

    print(dataset_path)
    reference_img = glob.glob(os.path.join(dataset_path,"train","good","*.bmp"))[0]

    img2_color = cv2.imread(reference_img)  # Reference image

    good_files = glob.glob(os.path.join(dataset_path,"test", "good","*.bmp"))
    anomaly_files = glob.glob(os.path.join(dataset_path,"test", "reject","*.bmp"))
    gts_path = glob.glob(os.path.join(dataset_path,"ground_truth", "reject","*.bmp"))

    for img in good_files:
        img1_color = cv2.imread(img)
        transformed_img = transform_img(img1_color, img2_color)
        # Save the output.
        img_name = img.split("\\")[-1]
        cv2.imwrite(f'{save_path}\\test\\good\\{img_name}', transformed_img)

    for i in range(len(anomaly_files)):
        img1_color = cv2.imread(anomaly_files[i])
        gt = cv2.imread(gts_path[i])
        transformed_img, transformed_gt = transform_img(img1_color, img2_color, gt)
        img_name = anomaly_files[i].split("\\")[-1]
        # Save the output.
        cv2.imwrite(f'{save_path}\\test\\reject\\{img_name}', transformed_img)
        cv2.imwrite(f'{save_path}\\ground_truth\\reject\\{img_name}', transformed_gt)

if __name__ == "__main__":
    main()