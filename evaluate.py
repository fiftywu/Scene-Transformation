import cv2
import os
import numpy as np
import skimage


# prefix = 'results/EmptyCities/images_RGB_MASK_TARGET/images'
# prefix = 'results/EmptyCities/images_RGB_TARGET/images'

prefix = '/mnt/mnt2/fiftywu/Files/EmptyCities_SLAM/results/ORB_512x512/latest_net_G_test/images'
names = os.listdir(os.path.join(prefix, 'input'))  # name the same as ouput and target

L1loss_save = []
PSNR_save = []
SSIM_save = []
for num, name in enumerate(names):
    # real_A = cv2.imread(os.path.join(prefix,'input',name), cv2.IMREAD_GRAYSCALE)
    real_B = cv2.imread(os.path.join(prefix,'target',name), cv2.IMREAD_GRAYSCALE)
    fake_B = cv2.imread(os.path.join(prefix,'output',name), cv2.IMREAD_GRAYSCALE)
    L1loss_save.append( np.sum(np.abs(real_B/255.-fake_B/255.))/real_B.size )

    if num % 500 == 0:
        print(num)

print('L1loss:', np.mean(L1loss_save), np.median(L1loss_save), len(names))


