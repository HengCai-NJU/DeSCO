import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
from skimage import morphology, exposure
import scipy
from scipy import ndimage
import h5py

v = "v3"

class CFG:
    do_windowing = True
    window_width = 300  # -100
    window_center = 50  # 200

    do_background_cropping = True
    cropping_width = 0.4
    cropping_center = 0.5

    do_cropping = True
    do_mask_cropping = True

    do_spacing = False
    target_spacing = [2, 2, 2]

    do_reshape = True
    new_size = [192, 192, 64]       # v3


# windowing
def transform_ctdata(image, windowWidth, windowCenter, normal=False):
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (image - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg


# background removing
def image_background_segmentation(img, WW=40, WL=80, display=False):
    # Calculate the outside values by hand (again)
    lB = WW - WL
    uB = WW + WL

    # Keep only values inside of the window
    background_seperation = np.logical_and(img > lB, img < uB)
    background_seperation = morphology.dilation(background_seperation, np.ones((5, 5)))
    labels, label_nb = scipy.ndimage.label(background_seperation)
    label_count = np.bincount(labels.ravel().astype(np.int))

    # discard the 0 label
    label_count[0] = 0
    mask = labels == label_count.argmax()  # find the most frequency number mask

    mask = morphology.dilation(mask, np.ones((4, 4)))  # dilate the mask for less fuzzy edges
    mask = scipy.ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))  # dilate the mask again

    if display:
        plt.figure(figsize=(15, 2.5))
        plt.subplot(141)
        plt.imshow(img, cmap='bone')
        plt.title('Original Images')
        plt.axis('off')

        plt.subplot(142)
        plt.imshow(background_seperation)
        plt.title('Segmentation')
        plt.axis('off')

        plt.subplot(143)
        plt.imshow(mask)
        plt.title('Mask')
        plt.axis('off')

        plt.subplot(144)
        plt.imshow(mask * img, cmap='bone')
        plt.title('Image * Mask')
        plt.axis('off')

    return mask, mask * img


# cropping
def crop(mask, vol):
    for i in range(mask.shape[0]):      # 125*512*512
        coords = np.array(np.nonzero(mask[i]))
        if i == 0:
            top_left = np.min(coords, axis = 1)
            bottom_right = np.max(coords, axis = 1)
        else:
            top_left = np.vstack((top_left, np.min(coords, axis = 1)))
            bottom_right = np.vstack((bottom_right, np.max(coords, axis = 1)))
    top = max(0, min(top_left[:, 0]) - 20)
    left = max(0, min(top_left[:, 1]) - 20)
    bottom = min(mask.shape[1], max(bottom_right[:, 0]) + 20)
    right = min(mask.shape[2], max(bottom_right[:, 1]) + 20)
    croped_vol = vol[:, top : bottom, left : right]
    return croped_vol

def getRangImageDepth(image):
    """
    :param image:
    :return:rangofimage depth
    """
    fistflag = True
    startposition = 0
    endposition = 0
    for z in range(image.shape[0]):
        notzeroflag = np.max(image[z])
        if notzeroflag and fistflag:
            startposition = z
            fistflag = False
        if notzeroflag:
            endposition = z
    return startposition, endposition

def make_patch(image, mask, startpostion, endpostion):
    """
    make number patch
    :param image:[depth,512,512]
    :param blockz: 64
    :return:[n,512,512]
    """
    imagezsrc = np.shape(image)[0]
    subimage_startpostion = startpostion - 10
    subimage_endpostion = endpostion + 10
    if subimage_startpostion < 0:
        subimage_startpostion = 0
    if subimage_endpostion > imagezsrc:
        subimage_endpostion = imagezsrc
    # if (subimage_endpostion - subimage_startpostion) < blockz:
    #     subimage_startpostion = 0
    #     subimage_endpostion = imagezsrc
    imageroi = image[subimage_startpostion:subimage_endpostion, :, :]
    maskroi = mask[subimage_startpostion:subimage_endpostion, :, :]
    return imageroi, maskroi

# reshape
def resampling(roiImg, new_size, lbl=False):
    new_spacing = [old_sz * old_spc / new_sz for old_sz, old_spc, new_sz in
                   zip(roiImg.GetSize(), roiImg.GetSpacing(), new_size)]
    if lbl:
        resampled_sitk = sitk.Resample(roiImg, new_size, sitk.Transform(), sitk.sitkNearestNeighbor, roiImg.GetOrigin(),
                                       new_spacing, roiImg.GetDirection(), 0.0, roiImg.GetPixelIDValue())
    else:
        resampled_sitk = sitk.Resample(roiImg, new_size, sitk.Transform(), sitk.sitkLinear, roiImg.GetOrigin(),
                                       new_spacing, roiImg.GetDirection(), 0.0, roiImg.GetPixelIDValue())

    return resampled_sitk


# space resampling
def space_resampling(roiImg, new_spacing, lbl=False):
    new_size = [int(old_sz * old_spc / new_spc) for old_sz, old_spc, new_spc in
                zip(roiImg.GetSize(), roiImg.GetSpacing(), new_spacing)]

    if lbl:
        resampled_sitk = sitk.Resample(roiImg, new_size, sitk.Transform(), sitk.sitkNearestNeighbor, roiImg.GetOrigin(),
                                       new_spacing, roiImg.GetDirection(), 0.0, roiImg.GetPixelIDValue())
    else:
        resampled_sitk = sitk.Resample(roiImg, new_size, sitk.Transform(), sitk.sitkLinear, roiImg.GetOrigin(),
                                       new_spacing, roiImg.GetDirection(), 0.0, roiImg.GetPixelIDValue())

    return resampled_sitk



loadPath = "../data/LiTS/raw/"
savePath = "../data/LiTS/processed_" + v + "_h5/"
if not os.path.exists(savePath):
    os.makedirs(savePath)

if __name__ == "__main__":
    subpath = loadPath + "ct/"
    subpath_lbl = loadPath + "seg/"
    fileList = os.listdir(subpath)
    for f in fileList:
        ct = sitk.ReadImage(subpath + f)
        lbl = sitk.ReadImage(subpath_lbl + "segmentation-" + f[7:])
        ct_array = sitk.GetArrayFromImage(ct)
        lbl_array = sitk.GetArrayFromImage(lbl)
        new_ct_array = ct_array.copy()
        new_lbl_array = lbl_array.copy()

        if CFG.do_windowing:    # ct
            ct_array_w = transform_ctdata(new_ct_array, CFG.window_width, CFG.window_center, True)
            new_ct_array = ct_array_w.copy()

        if CFG.do_background_cropping:      # ct
            ct_array_bgcrop = new_ct_array.copy()
            mask = new_ct_array.copy()
            for i in range(ct_array_bgcrop.shape[0]):
                mask[i], ct_array_bgcrop[i] = image_background_segmentation(ct_array_bgcrop[i], WW=CFG.cropping_center, WL=CFG.cropping_width, display=False)
            new_ct_array = ct_array_bgcrop.copy()

        if CFG.do_cropping:         # ct, lbl
            lbl_mask = mask.copy()
            ct_array_crop = new_ct_array.copy()
            lbl_array_crop = new_lbl_array.copy()
            ct_array_crop = crop(lbl_mask, ct_array_crop)
            lbl_array_crop = crop(lbl_mask, lbl_array_crop)
            new_ct_array = ct_array_crop.copy()
            new_lbl_array = lbl_array_crop.copy()

        if CFG.do_mask_cropping:        # ct, lbl
            ct_array_maskcrop = new_ct_array.copy()
            lbl_array_maskcrop = new_lbl_array.copy()
            startpostion, endpostion = getRangImageDepth(lbl_array_maskcrop)  # (75, 512, 512), 45, 73
            new_ct_array, new_lbl_array = make_patch(ct_array_maskcrop, lbl_array_maskcrop, startpostion=startpostion, endpostion=endpostion)

        new_ct = sitk.GetImageFromArray(new_ct_array)
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing(ct.GetSpacing())
        new_ct.SetDirection(ct.GetDirection())
        new_lbl = sitk.GetImageFromArray(new_lbl_array)
        new_lbl.SetOrigin(lbl.GetOrigin())
        new_lbl.SetSpacing(lbl.GetSpacing())
        new_lbl.SetDirection(lbl.GetDirection())

        if CFG.do_spacing:
            new_ct = space_resampling(new_ct, CFG.target_spacing, lbl=False)
            new_lbl = space_resampling(new_lbl, CFG.target_spacing, lbl=True)
        elif CFG.do_reshape:
            new_ct = resampling(new_ct, CFG.new_size, lbl=False)
            new_lbl = resampling(new_lbl, CFG.new_size, lbl=True)
        save_ct_array = sitk.GetArrayFromImage(new_ct)
        save_lbl_array = sitk.GetArrayFromImage(new_lbl)

        # output shape
        img_shape = save_ct_array.shape
        lbl_shape = save_lbl_array.shape

        # save
        save_file = h5py.File(savePath + f[:-4] + ".h5", 'w')
        save_file.create_dataset('image', data=save_ct_array)
        save_file.create_dataset('label', data=save_lbl_array)
        save_file.close()

