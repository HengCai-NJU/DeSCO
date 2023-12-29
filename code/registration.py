import h5py
import numpy as np
import cv2
import ants
import SimpleITK as sitk
import os
import shutil
import time
from glob import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--gtnum', type=int, default=32)
parser.add_argument('--idx', type=str, default='??')
parser.add_argument('--open1', type=int, default=1)
parser.add_argument('--close', type=int, default=1)
parser.add_argument('--threshold', type=float, default=1)
args = parser.parse_args()

def h52dir(h5file,pseudolabeldir,gtnum,opentime,closetime,typeoftransform):
    image_path = h5file
    h5f = h5py.File(image_path, 'r')
    if not os.path.exists(pseudolabeldir):
        os.mkdir(pseudolabeldir)
    image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)
    h5f.close()
    image = np.where(image < 255, (image - np.min(image)) * 255 / (np.max(image) - np.min(image)), 0)
    label = np.where(label < 1, 0, 255)
    paths=os.listdir('h5img')
    for path in paths:
        filepath=os.path.join('h5img',path)
        if os.path.isfile(filepath):
            os.remove(filepath)
    paths = os.listdir('h5label')
    for path in paths:
        filepath = os.path.join('h5label', path)
        if os.path.isfile(filepath):
            os.remove(filepath)
    paths = os.listdir('pseudolabel')
    for path in paths:
        filepath = os.path.join('pseudolabel', path)
        if os.path.isfile(filepath):
            os.remove(filepath)
    for i in range(0, image.shape[2]):
        cv2.imwrite('h5img/' + str(i) + '.png', image[:, :, i])
        cv2.imwrite('h5label/' + str(i) + '.png', label[:, :, i])
    shutil.copyfile('h5label/'+str(gtnum)+'.png', pseudolabeldir+str(gtnum)+'.png')

    for i in range(gtnum+1, 88):
        move_path = 'h5img/' + str(i - 1) + '.png'
        fix_path = 'h5img/' + str(i) + '.png'
        move_label_path = pseudolabeldir + str(i - 1) + '.png'
        save_label_path = pseudolabeldir + str(i) + '.nii.gz'
        fix_img = ants.image_read(fix_path)

        move_img = ants.image_read(move_path)
        move_label_img = ants.image_read(move_label_path)

        outs = ants.registration(fix_img, move_img, type_of_transforme=typeoftransform)
        reg_img = outs['warpedmovout']
        reg_label_img = ants.apply_transforms(fix_img, move_label_img, transformlist=outs['fwdtransforms'],
                                              interpolator='nearestNeighbor')
        ants.image_write(reg_label_img, save_label_path)
        # nii2png
        filedir = save_label_path
        outdir = pseudolabeldir
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        img = sitk.ReadImage(filedir)
        img = sitk.GetArrayFromImage(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=opentime)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=closetime)
        cv2.imwrite(outdir + str(i) + '.png', img)
    for i in range(gtnum, 0, -1):
        #print(i)
        move_path = 'h5img/' + str(i) + '.png'
        fix_path = 'h5img/' + str(i - 1) + '.png'
        move_label_path = pseudolabeldir + str(i) + '.png'
        save_label_path = pseudolabeldir + str(i - 1) + '.nii.gz'
        fix_img = ants.image_read(fix_path)

        move_img = ants.image_read(move_path)
        move_label_img = ants.image_read(move_label_path)

        outs = ants.registration(fix_img, move_img, type_of_transforme=typeoftransform)
        reg_img = outs['warpedmovout']
        reg_label_img = ants.apply_transforms(fix_img, move_label_img, transformlist=outs['fwdtransforms'],
                                              interpolator='nearestNeighbor')
        ants.image_write(reg_label_img, save_label_path)
        filedir = save_label_path
        outdir = pseudolabeldir
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        img = sitk.ReadImage(filedir)
        img = sitk.GetArrayFromImage(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=opentime2)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=closetime)
        cv2.imwrite(outdir + str(i - 1) + '.png', img)

def dir2h5(h5file,pseudolabeldir,outh5file):
    image_path = h5file
    h5f = h5py.File(image_path, 'r')
    image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)
    hf = h5py.File(outh5file, 'w')
    hf.create_dataset('image', data=image)
    h5f.close()
    mask = np.zeros_like(image)
    for i in range(0, 88):
        outputpath = pseudolabeldir + str(i) + '.png'
        msk = cv2.imread(outputpath, 0)
        msk = np.where(msk > 100, 1, 0)
        mask[:, :, i] = msk
    dice = 0
    for i in range(0, 88):
        smooth = 1e-5
        gtpath = 'h5label/' + str(i) + '.png'
        outputpath = 'pseudolabel/' + str(i) + '.png'
        gt = cv2.imread(gtpath, 0)
        output = cv2.imread(outputpath, 0)
        intersection = np.count_nonzero(gt * output)
        dicecoef = (2 * intersection + smooth) / (np.count_nonzero(gt) + np.count_nonzero(output) + smooth)
        dice += dicecoef
    print('dice:{}'.format(dice /88))
    hf.create_dataset('label', data=mask)
    hf.close()
    return dice/88

logpath='reglog.txt'
tic=time.time()
list=glob('../data/LA/'+args.idx+'.h5')
dice_list=[]
for item in list:
    dice = 0
    while (1):
        h52dir(item, 'pseudolabel/', args.gtnum, args.open1, args.close, 'SyNRA')
        dice = dir2h5(item, 'pseudolabel/', 'registration_data/'+args.idx+'px.h5')
        if dice > args.threshold:
            break
    dice_list.append(dice)
toc=time.time()
print('time:',toc-tic)
with open(logpath, "a") as f:
    f.writelines('la_x'+str(args.idx)+str(args.gtnum)+str(args.open1)+str(args.close)+str(dice_list)+'\n')
print(dice_list)

