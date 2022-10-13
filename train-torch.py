import pickle
import argparse
from pytorch_model import NeuralModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural net training arguments.')

    parser.add_argument('-u', type=str, help='number of hidden units')
    parser.add_argument('-l', type=float, help='learning rate')
    parser.add_argument('-f', type=int, help='max sequence length')
    parser.add_argument('-b', type=int, help='mini-batch size')
    parser.add_argument('-e', type=int, help='number of epochs to train for')
    parser.add_argument('-E', type=str, help='word embedding file')
    parser.add_argument('-i', type=str, help='training file')
    parser.add_argument('-o', type=str, help='model file to be written')
    parser.add_argument('-d', default=None, type=str, help='debug file path')

    args = parser.parse_args()

    model = NeuralModel(n_hidden_neurons=args.u,
                        max_sequence=args.f,
                        embedding_files=args.E,
                        debug_file=args.d)
    model.train(args.i, 
                lr=args.l,
                batch_size=args.b,
                epochs=args.e)
    
    model.save_model(args.o)
