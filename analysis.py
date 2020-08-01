import getopt, sys
import pandas as pd
import numpy as np

# Remove 1st argument from the 
# list of command line arguments 
argumentList = sys.argv[1:] 

# Options.
#If option x require an argument, then should be written as x: in the next string
options = "a:b:o:h"

# Long options 
long_options = ["model1=", "model2=","output=", "help"]

model2_path = ""

try: 
    # Parsing argument 
    arguments, values = getopt.getopt(argumentList, options, long_options) 
    
    # checking each argument 
    for currentArgument, currentValue in arguments: 

        if currentArgument in ("-h", "--help"):
            print("This script compare, numerically, performances of one or two model predictions\n")
            print("OPTIONS")
            print("-a / --model1: Folder which contains model's 1 'scores.csv' file .\n")
            print("-b / --model2: (OPTIONAL) Folder which contains model's 2 'scores.csv' file.\n")
            print("USAGE")
            print("python analysis.py -a <model1_path> -b <(OPTIONAL) model2_path>")
            exit()

        elif currentArgument in ("-a", "--model1"): 
            model1_path = currentValue 
        elif currentArgument in ("-b", "--model2"):
            model2_path = currentValue
        elif currentArgument in ("-o", "--output"):
            output_path = currentValue

except getopt.error as err: 
    # output error, and return with an error code 
    print (str(err)) 
    print("\nType analysis.py -h for usage indications")

m1 = pd.read_csv(model1_path + "/scores.csv")
degraded_scores = pd.read_csv(model1_path + "/degraded_img_scores.csv")

m1.sort_values(by=["imgName"])
degraded_scores.sort_values(by=["imgName"])

improved_psnr_m1 = 0
improved_mse_m1 = 0
improved_ssim_m1 = 0


if model2_path != "":
    m2 = pd.read_csv(model2_path + "/scores.csv")
    m2.sort_values(by=["imgName"])

    improved_psnr_m2 = 0
    improved_mse_m2 = 0
    improved_ssim_m2 = 0

    m1_psnr_win = 0
    m1_psnr_loose = 0
    m1_mse_win = 0
    m1_mse_loose = 0
    m1_ssim_win = 0
    m1_ssim_loose = 0

    m2_psnr_win = 0
    m2_psnr_loose = 0
    m2_mse_win = 0
    m2_mse_loose = 0
    m2_ssim_win = 0
    m2_ssim_loose = 0

examples = len(m1.index)


for i in range (examples):
    degraded_psnr = degraded_scores

    if m1.iloc[i]["psnr"] < degraded_scores.iloc[i]["psnr"]:
        improved_psnr_m1 += 1
    if m1.iloc[i]["mse"] < degraded_scores.iloc[i]["mse"]:
        improved_mse_m1 += 1
    if m1.iloc[i]["ssim"] > degraded_scores.iloc[i]["ssim"]:
        improved_ssim_m1 += 1
    
    if model2_path != "":
        if m2.iloc[i]["psnr"] < degraded_scores.iloc[i]["psnr"]:
            improved_psnr_m2 += 1
        if m2.iloc[i]["mse"] < degraded_scores.iloc[i]["mse"]:
            improved_mse_m2 += 1
        if m2.iloc[i]["ssim"] > degraded_scores.iloc[i]["ssim"]:
            improved_ssim_m2 += 1

        if m1.iloc[i]["psnr"] < m2.iloc[i]["psnr"]:
            m1_psnr_win += 1
            m2_psnr_loose += 1
        elif m1.iloc[i]["psnr"] > m2.iloc[i]["psnr"]:
            m2_psnr_win += 1
            m1_psnr_loose += 1
        
        if m1.iloc[i]["mse"] < m2.iloc[i]["mse"]:
            m1_mse_win += 1
            m2_mse_loose += 1
        elif m1.iloc[i]["mse"] > m2.iloc[i]["mse"]:
            m2_mse_win += 1
            m1_mse_loose += 1
        
        if m1.iloc[i]["ssim"] < m2.iloc[i]["ssim"]:
            m1_ssim_loose += 1
            m2_ssim_win += 1
        elif m1.iloc[i]["ssim"] > m2.iloc[i]["ssim"]:
            m2_ssim_loose += 1
            m1_ssim_win += 1
    


degraded_psnr = np.array(degraded_scores["psnr"].tolist())
degraded_mse = np.array(degraded_scores["mse"].tolist())
degraded_ssim = np.array(degraded_scores["ssim"].tolist())

m1_psnr = np.array(m1["psnr"].tolist())
m1_mse = np.array(m1["mse"].tolist())
m1_ssim = np.array(m1["ssim"].tolist())

if model2_path != "":
    m2_psnr = np.array(m2["psnr"].tolist())
    m2_mse = np.array(m2["mse"].tolist())
    m2_ssim = np.array(m2["ssim"].tolist())

print("\nDEGRADED IMAGES SCORES")
print("Total images:", len(degraded_psnr))
print("Average psnr:", degraded_psnr.mean())
print("Average mse:", degraded_mse.mean())
print("Average ssim:", degraded_ssim.mean())
print("\n----------------------------------------------\n")

print("Model 1")
print("Images with psnr improvement: ", improved_psnr_m1, "(", (improved_psnr_m1 * 100 / examples), " %)")
print("Images with mse improvement: ", improved_mse_m1, "(", (improved_mse_m1 * 100 / examples), " %)")
print("Images with ssim improvement: ", improved_ssim_m1, "(", (improved_ssim_m1 * 100 / examples), " %)\n")
print("Average psnr: ", np.mean(m1_psnr))
print("Average mse: ", np.mean(m1_mse))
print("Average ssim: ", np.mean(m1_ssim), "\n")
print("average psnr improvement: ", np.mean(degraded_psnr - m1_psnr))
print("aveage mse improvement: ",np.mean(degraded_mse - m1_mse))
print("average ssim improvement: ", np.mean(m1_ssim - degraded_ssim) * 100, "%")
print("\n----------------------------------------------\n")

if model2_path != "":
    print("Model 2")
    print("Images with psnr improvement: ", improved_psnr_m2, "(", (improved_psnr_m2 * 100 / examples), " %)")
    print("Images with mse improvement: ", improved_mse_m2, "(", (improved_mse_m2 * 100 / examples), " %)")
    print("Images with ssim improvement: ", improved_ssim_m2, "(", (improved_ssim_m2 * 100 / examples), " %)\n")
    print("Average psnr: ", np.mean(m2_psnr))
    print("Average mse: ", np.mean(m2_mse))
    print("Average ssim: ", np.mean(m2_ssim), "\n")
    print("average psnr improvement: ", np.mean(degraded_psnr - m2_psnr))
    print("aveage mse improvement: ",np.mean(degraded_mse - m2_mse))
    print("average ssim improvement: ", np.mean(m2_ssim - degraded_ssim) * 100, "%")

    print("\n-----------------------------------------------\n")

    print("PSNR")
    print("M1 Wins:", m1_psnr_win)
    print("M1 Loose:", m1_psnr_loose)
    print("M1 Wins:", m2_psnr_win)
    print("M2 Loose", m2_psnr_loose)
    if m1_psnr_win > m2_psnr_win:
        print("****M1 WINS****")
    elif m1_psnr_win < m2_psnr_win:
        print("****M2 WINS****")
    else:
        print("****DRAW****")
    print("\n")

    print("MSE")
    print("M1 Wins:", m1_mse_win)
    print("M1 Loose:", m1_mse_loose)
    print("M1 Wins:", m2_mse_win)
    print("M2 Loose", m2_mse_loose)
    if m1_mse_win > m2_mse_win:
        print("****M1 WINS****")
    elif m1_mse_win < m2_mse_win:
        print("****M2 WINS****")
    else:
        print("****DRAW****")
    print("\n")

    print("SSIM")
    print("M1 Wins:", m1_ssim_win)
    print("M1 Loose:", m1_ssim_loose)
    print("M1 Wins:", m2_ssim_win)
    print("M2 Loose", m2_ssim_loose)
    if m1_ssim_win > m2_ssim_win:
        print("****M1 WINS****")
    elif m1_ssim_win < m2_ssim_win:
        print("****M2 WINS****")
    else:
        print("****DRAW****")
    print("\n")









