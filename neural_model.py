from tqdm import tqdm
import numpy as np

from model import Model
from nn_layers import FeedForwardNetwork, Dense
from features import Features, tokenize
from utils import split_data
# np.random.seed(10)
    
class NeuralModel(Model):
    def __init__(self, n_hidden_neurons, max_sequence, embedding_files):
        self.n_hidden_neurons = n_hidden_neurons.split(',')
        self.max_sequence = max_sequence       
        self.embedding_files = embedding_files.split(',')
        self.network = FeedForwardNetwork()
        self.is_network_init = False  

    def build_network(self, num_of_classes):
        self.network.add(
            Dense(
                int(self.n_hidden_neurons[0]),
                input_size=self.embedding_size*self.max_sequence,
                activation='relu'
            )
        )
        
        for layer_units_num in self.n_hidden_neurons[1:]:
            self.network.add(
                Dense(
                    int(layer_units_num),
                    activation='relu'
                )
            )
        
        self.network.add(
            Dense(num_of_classes, activation='softmax')
        )
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

    def calc_cross_entropy_loss(self, Y_label, Y_predict):
        return -np.sum(Y_label*np.log(Y_predict))/len(Y_label)
    
    def train(self, input_file, lr=0.0001, batch_size=10, epochs=1):    
        features_list, labels = self._get_features(input_file)
        if not self.is_network_init:
            num_of_classes = labels.shape[-1]
            self.build_network(num_of_classes)
        
        print(self.network.summary(), flush=True)
        
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
        
        def split_into_batches(x, y, num_of_batches, shuffle=True):
            if shuffle:
                shuffle_indices = np.random.choice(x.shape[0], x.shape[0], replace=False)
                x, y = (x[shuffle_indices], y[shuffle_indices])
            x_batches = np.array_split(x, indices_or_sections=num_of_batches)
            y_batches = np.array_split(y, indices_or_sections=num_of_batches)
            return x_batches, y_batches
    
        n_samples = X_train.shape[0]
        print('Training on', n_samples, 'samples')
        if n_samples == 0:
            epochs = 0
        self.epochs=epochs
            
        if verbose > 0:
            print('lr:', lr, '-', 'lr_decay_rate:', lr_decay_rate,
                  '-', 'batch_size:', batch_size, '-', 'epochs:', epochs,
                  '-', 'early_stopping:', early_stopping,
                  '-', 'stopping_tolerance:', stopping_tolerance)
            
        batch_size = batch_size if batch_size < n_samples else max(n_samples, 1)
        # 1 SPLIT DATASET INTO BATCHES
        num_of_batches = max(np.ceil(n_samples/batch_size).astype(int), 1)
        old_val_acc = 0
        
        epochs_trange = tqdm(range(epochs), desc='Epochs')
        for epoch in epochs_trange:
            
            accumulated_training_corrects = 0
            accumulated_training_loss = 0
                 
            # 3 select batch of data and calc forward pass 
            x_train_batches, y_train_batches =\
                split_into_batches(X_train, Y_train, num_of_batches,
                                        shuffle=True)
            for batch_i in range(len(x_train_batches)):
                X = x_train_batches[batch_i]
                Y = y_train_batches[batch_i]

                outputs_cache = self.network.forward(X)

                # 4 compute loss function (cross-entropy), (logistic loss)
                accumulated_training_loss +=\
                    self.calc_cross_entropy_loss(Y, outputs_cache[-1])
                accumulated_training_corrects +=\
                    np.sum(np.argmax(outputs_cache[-1], axis=1) == np.argmax(Y, axis=1))
                
                if verbose > 0:
                    epochs_trange.set_postfix({
                        'batch' : f'{batch_i+1}/{num_of_batches}',
                        'loss': f'{accumulated_training_loss/(batch_i+1): < .2f}'
                    })
                    # print(f'{batch_i+1}/{num_of_batches}' +\
                    #       f'loss: {accumulated_training_loss/(batch_i+1): < .2f}',
                    #       end='\r')
                
                # 5 Backward-Propagation
                # learning-rate decay
                updated_lr = lr/(1 + lr_decay_rate*epoch)
                
                incoming_grad = Y
                self.network.backward(
                    outputs_cache, X, incoming_grad,
                    updated_lr, momentum
                )

            train_acc = accumulated_training_corrects*100/n_samples
            
            val_pred = self._classify(X_valid)
            val_loss = self.calc_cross_entropy_loss(Y_valid, val_pred)

            val_acc = np.sum(np.argmax(val_pred, axis=1) == np.argmax(Y_valid, axis=1))*100/len(val_pred)
            # print(str(batch_i+1) + '/' + str(num_of_batches), '-',
            #       'loss:', '%.2f' % (accumulated_training_loss/(batch_i+1)), '-',
            #       'acc:' f'{train_acc: < .2f}%', '-',
            #       'val_loss:', '%.2f' % val_loss, '-', 'val_acc:', '%.2f' % val_acc + '%')    
            epochs_trange.set_postfix({
                'batch' : f'{batch_i+1}/{num_of_batches}',
                'loss': f'{accumulated_training_loss/(batch_i+1): < .2f}',
                'acc:': f'{train_acc: < .2f}%',
                'val_loss': f'{val_loss: < .2f}',
                'val_acc:': f'{val_acc: < .2f}%',
            })
            print()
            
            
            if early_stopping:
                if abs(old_val_acc - val_acc) < stopping_tolerance:
                    print('Early stopping')
                    break   
                old_val_acc = val_acc
        # breakpoint()


    def classify(self, input_file):
        X_test = self._get_features(input_file, labeled=False)        
        index_to_label = {index: label for label, index in self.label_to_index.items()}
        predictions = map(lambda x: index_to_label[x], np.argmax(self._classify(X_test), axis=1))
        print('Finished Prediction')
        return predictions
    
    def _classify(self, X):
        return self.network.classify(X)