import getopt, sys
import pandas as pd
import numpy as np
from operator import itemgetter

# Remove 1st argument from the 
# list of command line arguments 
argumentList = sys.argv[1:] 

# Options.
#If option x require an argument, then should be written as x: in the next string
options = "i:m:h"

# Long options 
long_options = ["input=", "metric=", "help"]

scores_path = ""
metric = ""
try: 
    # Parsing argument 
    arguments, values = getopt.getopt(argumentList, options, long_options) 
    
    # checking each argument 
    for currentArgument, currentValue in arguments: 

        if currentArgument in ("-h", "--help"):
            print("This script find best's and worst's predictions")
            print("OPTIONS")
            print("-i / --input: Folder which contains model's 'scores.csv' file .\n")
            print("-m / --metrics: Metric to be evaluated (psnr, mse, ssim)")
            print("USAGE")
            print("python find_best.py -i <model's_predicted_images_path> -m <metric>")
            exit()

        elif currentArgument in ("-i", "--input"): 
            scores_path = currentValue 
        elif currentArgument in ("-m", "--metric"):
            metric = currentValue
except getopt.error as err: 
    # output error, and return with an error code 
    print (str(err)) 
    print("\nType find_best.py -h for usage indications")

scores = pd.read_csv(scores_path + "/{}".format("scores.csv"))
degraded_scores = pd.read_csv(scores_path + "/{}".format("degraded_img_scores.csv"))

scores.sort_values(by=["imgName"])
degraded_scores.sort_values(by=["imgName"])

images = []
for i in range (len(scores.index)):
    images.append((scores.iloc[i]["imgName"], degraded_scores.iloc[i][metric] - scores.iloc[i][metric]))

images = sorted(images,key=itemgetter(1), reverse= False if metric == "ssim" else True )

print("Images sorted from best to worst improvement according  to", metric, "\n")
c = 1
for img in images:
    print("{}.- ".format(c) + img[0])
    c+= 1
