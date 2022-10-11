import pickle
import argparse
from neural_model import NeuralModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural net inference arguments.')

    parser.add_argument('-m', type=str, help='trained model file')
    parser.add_argument('-i', type=str, help='test file to be read')
    parser.add_argument('-o', type=str, help='output file')

    args = parser.parse_args()
    
    model = NeuralModel.load_model(args.m)

    preds = model.classify(args.i)
    
    ## Save the predictions: one label prediction per line
    with open(args.o, "w") as file:
        for pred in preds:
            file.write(pred+"\n")
