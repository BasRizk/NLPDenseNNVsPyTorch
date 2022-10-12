# from tqdm import tqdm
# import numpy as np
import torch
from torch import nn, softmax
from torch.utils.data import DataLoader

from model import Model
from features import Features, tokenize
from utils import split_data
from tqdm import tqdm

# np.random.seed(10)
    
class NeuralModel(Model):
    def __init__(self, n_hidden_neurons, max_sequence, embedding_files):
        self.n_hidden_neurons = n_hidden_neurons.split(',')
        self.max_sequence = max_sequence       
        self.embedding_files = embedding_files.split(',')
        # self.network = FeedForwardNetwork()
        self.is_network_init = False  
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")
    
    
    def build_network(self, num_of_classes):
        self.network = NeuralNetwork(
                self.embedding_size, self.max_sequence, 
                num_of_classes, self.n_hidden_neurons
            ).to(self.device)
        print()
        self.is_network_init = True
    
    def _get_features(self, input_file, labeled=True):
        print('Loading Embeddings:', end=" ", flush=True)     
        word_to_vec, self.embedding_size =\
            Features.load_embeddings(self.embedding_files)
        print('Successful')
        
        print('Applying Embedding:', end=" ", flush=True)
        features =\
            Features.get_features(input_file,
                                  word_to_vec,
                                  self.embedding_size,
                                  self.max_sequence,
                                  labeled=labeled)
        print('Successful')  
        if labeled:
            self.label_to_index = features[-1]
            return features[:-1]
        return features
    
    def train(self, input_file, lr=0.0001, batch_size=10, epochs=1):   
        features_list, labels = self._get_features(input_file)
        if not self.is_network_init:
            num_of_classes = labels.shape[-1]
            self.build_network(num_of_classes)  
        
        print(self.network, flush=True)

        X_train, Y_train, X_valid, Y_valid =\
            split_data(features_list, labels)
            
        self._train(X_train, Y_train, X_valid, Y_valid,
                    lr_decay_rate=1,
                    lr=lr, batch_size=batch_size, epochs=epochs)
        
    
    def _train(self, X_train, Y_train, X_valid, Y_valid,
                lr=0.01, lr_decay_rate=0.9, momentum=1,
                batch_size=10, epochs=1, 
                early_stopping=False, stopping_tolerance=0.5, 
                verbose=1):
        
        train_dataloader =\
            DataLoader(
                SimpleDataset(X_train, Y_train),
                batch_size=batch_size
            )
        # valid_dataloader =\
        #     DataLoader(
        #         SimpleDataset(X_valid, Y_valid),
        #         batch_size=batch_size
        #     )
        X_valid = torch.tensor(X_valid)
        Y_valid = torch.tensor(Y_valid)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=lr)

        n_samples = X_train.shape[0]
        num_of_batches = len(train_dataloader)
        self.network.train()
        
        epochs_trange = tqdm(range(epochs), desc='Epochs')
        for epoch in epochs_trange:
            accumulated_training_corrects = 0
            accumulated_training_loss = 0
            for batch_i, (X, y) in enumerate(train_dataloader):
                X, y = X.to(self.device), y.to(self.device)

                # Compute prediction error
                pred = self.network(X)
                
                # compute loss function (cross-entropy), (logistic loss)
                loss = self.loss_fn(pred, y)
                accumulated_training_loss += loss
                accumulated_training_corrects +=\
                    (pred.argmax(1) == torch.argmax(y, axis=1)).type(torch.float).sum().item()
                
                if verbose > 0:
                    epochs_trange.set_postfix({
                        'batch' : f'{batch_i+1}/{num_of_batches}',
                        'loss': f'{accumulated_training_loss/(batch_i+1): < .2f}'
                    })
                      
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            train_acc = accumulated_training_corrects*100/n_samples
            val_pred = self._classify(X_valid)            
            val_loss = self.loss_fn(val_pred, Y_valid).item()
            val_acc = (val_pred.argmax(1) == torch.argmax(Y_valid, axis=1)).type(torch.float).sum().item()*100/len(val_pred)
            
            epochs_trange.set_postfix({
                'batch' : f'{batch_i+1}/{num_of_batches}',
                'loss': f'{accumulated_training_loss/(batch_i+1): < .2f}',
                'acc:': f'{train_acc: < .2f}%',
                'val_loss': f'{val_loss: < .2f}',
                'val_acc:': f'{val_acc: < .2f}%',
            })
            print()

    def classify(self, input_file):
        X_test = self._get_features(input_file, labeled=False)        
        index_to_label = {index: label for label, index in self.label_to_index.items()}
        predictions = map(lambda x: index_to_label[x.item()], self._classify(X_test).argmax(1))
        print('Finished Prediction')
        return predictions
    
    def _classify(self, X):
        self.network.eval()
        with torch.no_grad():
            pred = self.network(torch.tensor(X).to(self.device))
        return pred
            
class NeuralNetwork(nn.Module):
    def __init__(self, embedding_size, max_sequence, 
                 n_classes, n_hidden_neurons):
        super().__init__()
        
        layers = [
            nn.Linear(embedding_size*max_sequence, int(n_hidden_neurons[0])),
            nn.ReLU(),
        ]
        for i, layer_units_num in enumerate(n_hidden_neurons[1:]):
            layers += [
                nn.Linear(n_hidden_neurons[i-1], layer_units_num),
                nn.ReLU()
            ]
        layers += [
            nn.Linear(int(n_hidden_neurons[-1]), n_classes),
            nn.Softmax(dim=1)
        ]
            
        self.network_stack = nn.Sequential(*layers)
        self.network_stack.double()
        

    def forward(self, x):
        logits = self.network_stack(x)
        return logits
    
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]