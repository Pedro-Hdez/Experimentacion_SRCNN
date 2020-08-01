
# Python program to demonstrate 
# command line arguments 
  
  
import getopt, sys
import os
import cv2
  
  
# Remove 1st argument from the 
# list of command line arguments 
argumentList = sys.argv[1:] 
  
# Options.
#If option x require an argument, then should be written as x: in the next string
options = "i:o:f:h"
  
# Long options 
long_options = ["input=", "output=", "factor=", "help"] 

input_folder = ""
output_folder = ""
factor = 1

try: 
    # Parsing argument 
    arguments, values = getopt.getopt(argumentList, options, long_options) 
      
    # checking each argument 
    for currentArgument, currentValue in arguments: 
  
        if currentArgument in ("-h", "--help"):
            print("This script degrade all images in an input folder given " +  
                  "a factor using bilinear interpolation method; then, " + 
                  "degraded images are saved in an output folder.\n")
            print("OPTIONS")
            print("--input / -i: Input folder path\n--output / -o: Output folder path\n"+
                  "--factor / -f: Degrade factor\n")
            print("USAGE")
            print("python degrade_images.py -i <input_folder_path> -o <output_folder_path> -f <factor>")  
            exit()
        elif currentArgument in ("-i", "--input"): 
            input_folder = currentValue 
            print(currentValue)
        elif currentArgument in ("-o", "--output"): 
            output_folder = currentValue
        elif currentArgument in ("-f", "--factor"):
            factor = int(currentValue)
except getopt.error as err: 
    # output error, and return with an error code 
    print (str(err)) 
    print("\nType degrade_images.py -h for usage indications")


# prepare degraded images by introducing quality distortions via resizing

# Create new directory to save degraded images
dirName = output_folder
try:
    os.mkdir(dirName)
    print("Directory '" , dirName ,  "' Created ") 
except FileExistsError:
    print("Directory '" , dirName ,  "' already exists, images will be overwritten")

# loop through the files in the directory
from progress.bar import Bar
with Bar ("Processing images", max=len(os.listdir(input_folder))) as bar:
    for file in os.listdir(input_folder):
        
        # open the file
        img = cv2.imread(input_folder + '/' + file)
        
        # find old and new image dimensions
        h, w, _ = img.shape
        new_height = int(h / factor)
        new_width = int(w / factor)
        
        # resize the image - down
        img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR)
        
        # resize the image - up
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_LINEAR)
        
        # save the image
        cv2.imwrite(output_folder + '/' + file, img)
        
        bar.next()
        
