from predict import modcrop, compare_images, mse, psnr
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt




if __name__ == "__main__":
    import getopt, sys, os, cv2, fnmatch
    import numpy as np
    import pandas as pd

    def find_image(pattern, path):
        for root, _, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    return os.path.join(root, name)

    # Remove 1st argument from the 
    # list of command line arguments 
    argumentList = sys.argv[1:] 
    
    # Options.
    #If option x require an argument, then should be written as x: in the next string
    options = "r:d:a:b:o:h"
    
    # Long options 
    long_options = ["reference=", "degraded=", "originalmodel=", "mymodel=", "output=", "help"]

    original_images_path = ""
    degraded_images_path = ""
    original_model_predicted_path = ""
    my_model_predicted_path = ""
    output_dirName = ""


    try: 
        # Parsing argument 
        arguments, values = getopt.getopt(argumentList, options, long_options) 
        
        # checking each argument 
        for currentArgument, currentValue in arguments: 
    
            if currentArgument in ("-h", "--help"):
                print("This script compare the outputs of two different trained models\n")
                print("OPTIONS")
                print("--reference / -r: Folder path which contains original and not degraded images.\n")
                print("--degraded / -d: Folder path which contains degraded images.\n")
                print("--originalmodel / -a: Folder path which contains original model's output images.\n")
                print("--mymodel / -b: Folder path which contains my model's output images.\n")
                print("--output / -o: Folder path where comparisons will be saved.\n")
                print("USAGE")
                print("python comparator.py -r <original_images_folder_path> -d <degraded_images_folder_path> " + 
                      "-a <original_model_path> -b <my_model_path> -o <ouput_folder_path>")
                
                exit()
                
            elif currentArgument in ("-r", "--reference"):
                original_images_path = currentValue
            elif currentArgument in ("-d", "--degraded"):
                degraded_images_path = currentValue
            elif currentArgument in ("-a", "--originalmodel"):
                original_model_predicted_path = currentValue
            elif currentArgument in ("-b", "--mymodel"):
                my_model_predicted_path = currentValue
            elif currentArgument in ("-o", "--output"):
                output_dirName = currentValue

    except getopt.error as err: 
        # output error, and return with an error code 
        print (str(err)) 
        print("\nType comparator.py -h for usage indications")
    
    try:
        os.mkdir(output_dirName)
        print("Directory '" , output_dirName ,  "' Created ") 
    except FileExistsError:
        print("Directory '" , output_dirName ,  "' already exists, it will be overwritten")
    
    from progress.bar import Bar
    print(original_images_path)
    with Bar ("Comparing", max = len(os.listdir(original_images_path))) as bar:
        for file in os.listdir(original_images_path):
            
            file = file.split(".")[0]
            match_pattern = file + "*"

            # Find degraded image
            degraded = find_image( match_pattern, degraded_images_path)
            #Load degraded image
            degraded = cv2.imread(degraded)

            #Find original image
            original = find_image(match_pattern, original_images_path)
            #Load original image
            original = cv2.imread(original)

            #Find original model predicted image
            original_model_prediction = find_image(match_pattern, original_model_predicted_path + "/individual_images")
            #load the original model prediction
            original_model_prediction = cv2.imread(original_model_prediction)

            #Find my model predicted image
            my_model_prediction = find_image(match_pattern, my_model_predicted_path + "/individual_images")
            #Load my model prediction
            my_model_prediction = cv2.imread(my_model_prediction)

            # preprocess the image with modcrop
            degraded = modcrop(degraded, 3)
            original = modcrop(original, 3)
            original_model_prediction = modcrop(original_model_prediction, 3)
            my_model_prediction = modcrop(my_model_prediction, 3)

            #Get scores for original and degraded image
            degraded_image_scores = compare_images(degraded, original)


            #Load output scores for both models
            original_model_scores = pd.read_csv(original_model_predicted_path + "/scores.csv")
            my_model_scores = pd.read_csv(my_model_predicted_path + "/scores.csv")
            
            #filters for search scores
            original_model_filter = original_model_scores["imgName"] == file
            my_model_filter = my_model_scores["imgName"] == file


            # display images as subplots
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))

            axs[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            axs[0, 0].set_title('Original')

            axs[0, 1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
            axs[0, 1].set_title('Degraded')
            axs[0, 1].set(xlabel = 'PSNR: {}\nMSE: {} \nSSIM: {}'.format(degraded_image_scores[0], degraded_image_scores[1], 
                                                                         degraded_image_scores[2]))
            
            axs[1, 0].imshow(cv2.cvtColor(original_model_prediction, cv2.COLOR_BGR2RGB))
            axs[1, 0].set_title('Original model (200ep + 400ep HomeObjects06 )')
            axs[1, 0].set(xlabel = 'PSNR: {} \nMSE: {} \nSSIM: {}'.format(original_model_scores.loc[original_model_scores["imgName"] == file, "psnr"].iloc[0], 
                                                                          original_model_scores.loc[original_model_scores["imgName"] == file, "mse"].iloc[0],
                                                                          original_model_scores.loc[original_model_scores["imgName"] == file, "ssim"].iloc[0]))

            axs[1, 1].imshow(cv2.cvtColor(my_model_prediction, cv2.COLOR_BGR2RGB))
            axs[1, 1].set_title('my model (6lyrs 800ep)')
            axs[1, 1].set(xlabel = 'PSNR: {} \nMSE: {} \nSSIM: {}'.format(my_model_scores.loc[my_model_scores["imgName"] == file, "psnr"].iloc[0],
                                                                          my_model_scores.loc[my_model_scores["imgName"] == file, "mse"].iloc[0],
                                                                          my_model_scores.loc[my_model_scores["imgName"] == file, "ssim"].iloc[0]))



            # remove the x and y ticks
            axs = axs.flatten()
            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])

            #Saving the resuld
            fig.savefig(output_dirName + '/{}.png'.format(file))
            plt.close()
            bar.next()
