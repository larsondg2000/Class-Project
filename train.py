"""
Trains a model using VGG19 or AlexNet the following configurable settings:
    -model
    -epochs
    -learning rate
    -data nd save directories
    -hidden layers
    -gpu or cpu

Train Main Module
    1) Get arg parse values
    2) Call train function
"""

# imports
from train_functions import *


# Main Function, Call train function in train_functions
def main():
    args = args_parse()
    train(args.model, args.epochs, args.gpu, args.learn_rate, args.hidden_layers, args.save_dir, args.data_dir)
    print("Model Successfully Trained and Saved!!!!!")


# use the get_input_args function to run the program
if __name__ == "__main__":
    main()