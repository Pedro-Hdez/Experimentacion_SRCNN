# -*- coding: utf-8 -*-
import os
import cv2
import h5py
import numpy



Random_Crop = 30
Patch_size = 32
label_size = 20
conv_side = 6

from PIL import Image

def to_bmp(path):
    from progress.bar import Bar
    with Bar ("Converting images", max = len(os.listdir(path)), ) as bar:
        for file in os.listdir(path):
            Image.open(path + "/" + file).save(path + "/" + file.split(".")[0] + ".bmp")
            bar.next()

    filtered_files = [file for file in os.listdir(path) if not file.endswith(".bmp")]
    
    with Bar ("Deliting not bmp images", max = len(filtered_files), ) as bar:
        for file in filtered_files:
            os.remove(os.path.join(path, file))
            bar.next()

def prepare_data(_path, scale):
    names = os.listdir(_path)
    names = sorted(names)
    nums = names.__len__()

    data = numpy.zeros((nums * Random_Crop, 1, Patch_size, Patch_size), dtype=numpy.double)
    label = numpy.zeros((nums * Random_Crop, 1, label_size, label_size), dtype=numpy.double)

    from progress.bar import Bar
    with Bar ("Preparing data", max = nums, suffix='%(percent)d%%') as bar:
        for i in range(nums):
            name = _path + "/" + names[i]
            hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
            shape = hr_img.shape

            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
            hr_img = hr_img[:, :, 0]

            # two resize operation to produce training data and labels
            lr_img = cv2.resize(hr_img, (int(shape[1] / scale), int(shape[0] / scale)))
            lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

            # produce Random_Crop random coordinate to crop training img
            Points_x = numpy.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_Crop)
            Points_y = numpy.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_Crop)

            for j in range(Random_Crop):
                lr_patch = lr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]
                hr_patch = hr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]

                lr_patch = lr_patch.astype(float) / 255.
                hr_patch = hr_patch.astype(float) / 255.

                data[i * Random_Crop + j, 0, :, :] = lr_patch
                label[i * Random_Crop + j, 0, :, :] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]
                # cv2.imshow("lr", lr_img)
                # cv2.imshow("hr", hr_patch)
                # cv2.waitKey(0)
            bar.next()
    return data, label

# BORDER_CUT = 8
BLOCK_STEP = 16
BLOCK_SIZE = 32


def prepare_crop_data(_path, scale):
    names = os.listdir(_path)
    names = sorted(names)
    nums = names.__len__()

    data = []
    label = []

    from progress.bar import Bar
    with Bar ("Preparing crop data", max = nums, suffix='%(percent)d%%') as bar:

        for i in range(nums):
            name = _path + "/" + names[i]
            hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
            hr_img = hr_img[:, :, 0]
            shape = hr_img.shape

            # two resize operation to produce training data and labels
            lr_img = cv2.resize(hr_img, (int(shape[1] / scale), int(shape[0] / scale)))
            lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

            width_num = (shape[0] - (BLOCK_SIZE - BLOCK_STEP) * 2) / BLOCK_STEP
            height_num = (shape[1] - (BLOCK_SIZE - BLOCK_STEP) * 2) / BLOCK_STEP
            for k in range(int(width_num)):
                for j in range(int(height_num)):
                    x = k * BLOCK_STEP
                    y = j * BLOCK_STEP
                    hr_patch = hr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]
                    lr_patch = lr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]

                    lr_patch = lr_patch.astype(float) / 255.
                    hr_patch = hr_patch.astype(float) / 255.

                    lr = numpy.zeros((1, Patch_size, Patch_size), dtype=numpy.double)
                    hr = numpy.zeros((1, label_size, label_size), dtype=numpy.double)

                    lr[0, :, :] = lr_patch
                    hr[0, :, :] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]

                    data.append(lr)
                    label.append(hr)
            bar.next()

    data = numpy.array(data, dtype=float)
    label = numpy.array(label, dtype=float)
    return data, label


def write_hdf5(data, labels, output_filename):
    """
    This function is used to save image data and its label(s) to hdf5 file.
    output_file.h5,contain data and label
    """

    x = data.astype(numpy.float32)
    y = labels.astype(numpy.float32)

    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        h.create_dataset('label', data=y, shape=y.shape)
        # h.create_dataset()


def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data = numpy.array(hf.get('data'))
        label = numpy.array(hf.get('label'))
        train_data = numpy.transpose(data, (0, 2, 3, 1))
        train_label = numpy.transpose(label, (0, 2, 3, 1))
        return train_data, train_label


if __name__ == "__main__":
    import getopt, sys

    # Remove 1st argument from the 
    # list of command line arguments 
    argumentList = sys.argv[1:] 
    
    # Options.
    #If option x require an argument, then should be written as x: in the next string
    options = "i:f:c:h"
    
    # Long options 
    long_options = ["input=", "factor=","convert=", "help"]

    training_data_path = ""
    factor = None
    convert = None

    try: 
        # Parsing argument 
        arguments, values = getopt.getopt(argumentList, options, long_options) 
        
        # checking each argument 
        for currentArgument, currentValue in arguments: 
    
            if currentArgument in ("-h", "--help"):
                print("This script prepare training data and saves it in .h5 files\n")
                print("OPTIONS")
                print("--input / -i: Folder which contains traning images.\n")
                print("--factor / -f: Degrading factor")
                print("--convert / -c: 1 to convert input images to bmp. If not given, images will not be converted")
                print("USAGE")
                print("python prepare_data.py -i <training_images_path> -f <factor> -c 1(OPTIONAL)")
                exit() 
            elif currentArgument in ("-i", "--input"): 
                training_data_path = currentValue 
            elif currentArgument in ("-f", "--factor"):
                factor = int(currentValue)
            elif currentArgument in ("-c", "--convert"):
                convert = int(currentValue)

    except getopt.error as err: 
        # output error, and return with an error code 
        print (str(err)) 
        print("\nType prepare_data.py -h for usage indications")

    if convert == 1:
        print("Converting all images in ", training_data_path, " to .bmp images")
        to_bmp(training_data_path)

    data, label = prepare_crop_data(training_data_path, factor)
    write_hdf5(data, label, "crop_train.h5")
    data, label = prepare_data(training_data_path, factor)
    write_hdf5(data, label, "test.h5")
