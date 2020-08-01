from keras.models import Sequential
from keras.layers import Conv2D, Input, BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import prepare_data as pd
import numpy
import math

def original_model():
    SRCNN = Sequential()

    SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size = (3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN
    

def my_model():
    SRCNN = Sequential()

    # MY MODEL
    # add model layers
    SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size = (7, 7), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=32, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=16, kernel_size = (3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    #Esta es la que se quita
    SRCNN.add(Conv2D(filters=8, kernel_size = (3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN

def my_model_2():
    # lrelu = LeakyReLU(alpha=0.1)
    SRCNN = Sequential()

    # MY MODEL
    # add model layers
    SRCNN.add(Conv2D(filters=128, kernel_size = (7,7), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size = (7, 7), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=64, kernel_size = (7, 7), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=64, kernel_size = (7, 7), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=64, kernel_size = (7, 7), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=64, kernel_size = (7, 7), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    
    adam = Adam(lr=0.0003)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN



def train(model, epochs, checkpointFile, preWeights):
    srcnn_model = None
    if model == 1:
        srcnn_model = original_model()
    elif model == 2:
        srcnn_model = my_model()
    elif model == 3:
        srcnn_model = my_model_2()

    if len(preWeights) != 0:
    	srcnn_model.load_weights(preWeights)
    	print("Loaded model " + preWeights + "to continue training by {} epochs".format(epochs))
    else: 
    	print("New model created")
    	
    print(srcnn_model.summary())
    data, label = pd.read_training_data("./crop_train.h5")
    val_data, val_label = pd.read_training_data("./test.h5")

    checkpoint = ModelCheckpoint(checkpointFile, monitor='val_loss', verbose=1, save_best_only=False,
                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    srcnn_model.fit(data, label, batch_size=128, validation_data=(val_data, val_label),
                    callbacks=callbacks_list, shuffle=True, epochs=epochs, verbose=0)
    


if __name__ == "__main__":
    import getopt, sys, time
    # Remove 1st argument from the 
    # list of command line arguments 
    argumentList = sys.argv[1:] 
    
    # Options.
    #If option x require an argument, then should be written as x: in the next string
    options = "n:m:e:l:h"
    
    # Long options 
    long_options = ["name=", "model=", "epochs=", "load=", "help"] 

    checkpoint_file = ""
    model = 0
    epochs = 0
    pretrained_model = ""

    try: 
        # Parsing argument 
        arguments, values = getopt.getopt(argumentList, options, long_options) 
        
        # checking each argument 
        for currentArgument, currentValue in arguments: 
    
            if currentArgument in ("-h", "--help"):
                print("This script trains a neural network model.\n")
                print("OPTIONS")
                print("--name / -n: File name where trained model will be saved.")
                print("--model / -m: Wich \"model-like\" will be trained (1 for Original Model / 2 for my model / 3 for my model 2)")
                print("--epochs / -e: Number of training epochs")
                print("--load / -l (Optional): Pass it if you want to load a pretrained model and continue training it\n")
                print("USAGE")
                print("python train.py -n <output_file_name.h5> -e <# of epochs> -l <*OPTIONAL* pretrained_model.h5 >")
                exit()
            elif currentArgument in ("-n", "--name"): 
                checkpoint_file = currentValue 
            elif currentArgument in ("-m", "--model"):
                model = int(currentValue)
            elif currentArgument in ("-e", "--epochs"): 
                epochs = int(currentValue)
            elif currentArgument in ("-l", "--load"):
                pretrained_model = currentValue
    except getopt.error as err: 
        # output error, and return with an error code 
        print (str(err)) 
        print("\nType train.py -h for usage indications")
    
    start_time = time.time()
    train(model, epochs, checkpoint_file, pretrained_model)
    print('Training completed')
    print("time elapsed: {} hours".format( ((time.time() - start_time) / 60 ) / 60))
    #predict()