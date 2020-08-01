import os
import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
import pandas as pd

#define a function for peak signal-to-noise ratio (PSNR)
def psnr(target, ref):
         
    # assume RGB image
    target_data = target.astype(float)
    ref_data = ref.astype(float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)

# define function for mean squared error (MSE)
def mse(target, ref):
    # the MSE between the two images is the sum of the squared difference between the two images
    err = np.sum((target.astype('float') - ref.astype('float')) ** 2)
    err /= float(target.shape[0] * target.shape[1])
    
    return err

# define function that combines all three image quality metrics
def compare_images(target, ref):
    scores = []
    scores.append(psnr(target, ref))
    scores.append(mse(target, ref))
    scores.append(ssim(target, ref, multichannel =True))
    
    return scores

#----------------------------------------------------------------------

# define necessary image processing functions
def modcrop(img, scale):
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 1:sz[1]]
    return img


def shave(image, border):
    img = image[border: -border, border: -border]
    return img

#--------------------------------------------------------------------

if __name__ == "__main__":
    import getopt, sys
    # Remove 1st argument from the 
    # list of command line arguments 
    argumentList = sys.argv[1:] 
    
    # Options.
    #If option x require an argument, then should be written as x: in the next string
    options = "m:w:r:t:o:h"
    
    # Long options 
    long_options = ["model=", "weights=", "reference=", "test=","output=","help"] 
    
    model = 0
    pretrained_weights = ""
    reference_dirName = ""
    test_dirName = ""
    output_dirName = ""

    try: 
        # Parsing argument 
        arguments, values = getopt.getopt(argumentList, options, long_options) 
        
        # checking each argument 
        for currentArgument, currentValue in arguments: 
    
            if currentArgument in ("-h", "--help"):
                print("This script use a trained neural network to improve resolution of images in a given test folder," + 
                      "also, calculates MSE, PSNR and SSIM indicators and saves the results in an output given folder.\n")
                print("OPTIONS")
                print("--model / -m: Which model-like will be used for prediction (1 for original model / 2 for my model / 3 for my model 2)")
                print("--weights / -w: .h5 file containing trained neural net model")
                print("--reference / -r: Folder path which contains original and not degraded images")
                print("--test / -t: Folder path which contains degraded images")
                print("--output / -o: Folder path where results will be saved\n")
                print("USAGE")
                print("python predict.py -w <trained_model.h5> -r <not_degraded_images_folder> -t <degraded_images_folder> -o <results_folder>")
                exit()
            elif currentArgument in ("-m", "--model"):
                model = int(currentValue)
            elif currentArgument in ("-w", "--weights"): 
                pretrained_weights = currentValue 
            elif currentArgument in ("-r", "--reference"): 
                reference_dirName = currentValue
            elif currentArgument in ("-t", "--test"):
                test_dirName = currentValue
            elif currentArgument in ("-o", "--output"):
                output_dirName = currentValue
    except getopt.error as err: 
        # output error, and return with an error code 
        print (str(err)) 
        print("\nType predict.py -h for usage indications")
    
    try:
        os.mkdir(output_dirName)
        print("Directory '" , output_dirName ,  "' Created ") 
    except FileExistsError:
        print("Directory '" , output_dirName ,  "' already exists, it will be overwritten")
    try:
        os.mkdir(output_dirName + "/analysis")
        print("DIrectory '", output_dirName, "/analysis' created")
    except FileExistsError:
        print("Directory '", output_dirName, "/analysis' already exists, it will be overwritten")
    try:
        os.mkdir(output_dirName + "/individual_images")
        print("Directory '", output_dirName, "/individual_images' created")
    except FileExistsError:
        print("Directory '", output_dirName, "/individual_images' already exists, it will be overwritten")
    

    # load the srcnn model with weights
    if model == 1:
        from train import original_model as model
    elif model == 2:
        from train import my_model as model
    elif model == 3:
        from train import my_model_2 as model
    srcnn = model()
    srcnn.load_weights(pretrained_weights)
    
    #Pandas DataFrame is created to ease the score saving. This will be used to compare 
    #performance of two different models
    cols = ["imgName", "psnr", "mse", "ssim"]

    scores_df = pd.DataFrame(columns=cols)
    degraded_scores_df = pd.DataFrame(columns=cols)
    
    #Prediction loop
    from progress.bar import Bar
    with Bar ("Performing resolution augmentation", max = len(os.listdir(test_dirName)), ) as bar:
        for file in os.listdir(test_dirName):
            
            # load the degraded image
            degraded = cv2.imread(test_dirName + '/{}'.format(file))
            #Reference images
            ref = cv2.imread(reference_dirName + '/{}'.format(file))
    
            # preprocess the image with modcrop
            ref = modcrop(ref, 3)
            degraded = modcrop(degraded, 3)
            
            # convert the image to YCrCb - (srcnn trained on Y channel)
            temp = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)
            
            # create image slice and normalize  
            Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
            Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255
            
            # perform super-resolution with srcnn
            pre = srcnn.predict(Y, batch_size=1)
            
            # post-process output
            pre *= 255
            pre[pre[:] > 255] = 255
            pre[pre[:] < 0] = 0
            pre = pre.astype(np.uint8)
            
            # copy Y channel back to image and convert to BGR
            temp = shave(temp, 6)
            temp[:, :, 0] = pre[0, :, :, 0]
            output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)
            
            # remove border from reference and degraged image
            ref = shave(ref.astype(np.uint8), 6)
            degraded = shave(degraded.astype(np.uint8), 6)
            
            # image quality calculations
            scores = []
            scores.append(compare_images(degraded, ref))
            scores.append(compare_images(output, ref))

            #degraded image scores are stored
            degraded_scores = pd.Series([file.split('.')[0], scores[0][0], scores[0][1], scores[0][2]], index=cols)
            degraded_scores_df = degraded_scores_df.append(degraded_scores, ignore_index=True)

            #Scores are stored in scroes dataframe
            img_scores = pd.Series([file.split(".")[0], scores[1][0], scores[1][1], scores[1][2]], index=cols)
            scores_df = scores_df.append(img_scores, ignore_index=True)
                    
            # display images as subplots
            fig, axs = plt.subplots(1, 3, figsize=(20, 8))
            axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
            axs[0].set_title('Original')
            axs[1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
            axs[1].set_title('Degraded')
            axs[1].set(xlabel = 'PSNR: {}\nMSE: {} \nSSIM: {}'.format(scores[0][0], scores[0][1], scores[0][2]))
            axs[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            axs[2].set_title('Predicted')
            axs[2].set(xlabel = 'PSNR: {} \nMSE: {} \nSSIM: {}'.format(scores[1][0], scores[1][1], scores[1][2]))


            cv2.imwrite(output_dirName + "/individual_images/" + file, output)
            cv2.waitKey(0)
            # remove the x and y ticks
            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])
            
            fig.savefig(output_dirName + '/analysis/{}.png'.format(os.path.splitext(file)[0])) 
            plt.close()
            bar.next()
    
    scores_df.to_csv(output_dirName + "/{}".format("scores.csv"), index = False)
    degraded_scores_df.to_csv(output_dirName + "/{}".format("degraded_img_scores.csv"), index = False)
    print("\nPrediction completed")
    

    

