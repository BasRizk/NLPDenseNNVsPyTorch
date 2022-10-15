import numpy as np

from abc import ABC, abstractmethod


# NEGLIGIBLE_VALUE = 1E-03

class NNComp(ABC):
    """
    This is one design option you might consider. Though it's
    solely a suggestion, and you are by no means required to stick
    to it. You can implement your NN modules as concrete
    implementations of this class, and fill forward and backward
    methods for each module accordingly.
    """

    @abstractmethod
    def forward(self, x):
        raise NotImplemented

    @abstractmethod
    def backward(self, incoming_grad):
        raise NotImplemented

        
class FeedForwardNetwork(NNComp):
    """
    This is one design option you might consider. Though it's
    solely a suggestion, and you are by no means required to stick
    to it. You can implement your FeedForwardNetwork as concrete
    implementations of the NNComp class, and fill forward and backward
    methods for each module accordingly. It will likely be composed of
    other NNComp objects.
    """
    def __init__(self):
        self._layers = []
        self.train_accuracies = []
        self.valid_accuracies = []
        self.past_epoch = -1
    
    def summary(self):
        statement = '=============== Model Summary ===============\n'
        statement += '# of layers ' + str(len(self._layers)) + '\n'
        num_of_params = 0
        for i, layer in enumerate(self._layers):
            statement += str(i+1) + " " + str(layer) + '\n'
            num_of_params += layer.num_of_params
        statement += 'Total # of params = ' + str(num_of_params)
        return statement
    
    def classify(self, X):
        for layer in self._layers:
            X = layer.predict(X) 
        return X
    
    def init_outputs_cache_space(self, batch_size):
        outputs = []
        for layer in self._layers:
            outputs.append(np.zeros((batch_size, layer.n_neurons)))
        return outputs
    
    def add(self, layer):
        if len(self._layers) > 0:
            # breakpoint()
            layer._config_params(input_size=self._layers[-1].n_neurons)
        self._layers.append(layer)
        
    def forward(self, x):
        outputs_cache = self.init_outputs_cache_space(x.shape[0])                        
        _input = x
        for layer_i, layer in enumerate(self._layers):
            _input = layer.forward(_input)    

            # Cache output for backward propagation
            outputs_cache[layer_i] = _input
            
        return outputs_cache

    def backward(self, outputs_cache, X, incoming_grad, lr, momentum):
        for i in reversed(range(len(self._layers))):
            input_values = outputs_cache[i-1] if i > 0 else X
            predicted_output = outputs_cache[i]
            incoming_grad = self._layers[i].backpropagate(
                input_values, predicted_output, incoming_grad,
                lr, momentum
            )
    
class Dense:
    def __init__(self, n_neurons, input_size=None, activation='linear',
                 weights_init='xavier'
                # weights_init='random'
                 ): 
        self.type = 'Dense'
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.weights_init = weights_init
        if self.input_size is not None:
            self._config_params()
        self.activation_type = activation
        self.set_activation_func(activation)
        self.debug_i = 0


    def _config_params(self, input_size=None):
        # 2 INIT WEIGHTS AND BIAS
        self.input_size = input_size if input_size is not None else self.input_size
        # Xavier Initialization
        self.weights = np.random.randn(self.input_size, self.n_neurons)
        self.bias = np.random.randn(1, self.n_neurons)
        if self.weights_init == 'xavier':
            bound = np.sqrt(6)/np.sqrt(self.input_size + self.n_neurons)
            self.weights = np.random.uniform(-1, 1, size=(self.input_size, self.n_neurons))
            self.weights = self.weights*(2*bound) - bound
            self.bias = np.random.uniform(-1, 1, size=(1, self.n_neurons))
            self.bias = self.bias*(2*bound) - bound
        elif self.weights_init == 'scaling':
            self.weights = self.weights*(1/self.input_size)
            self.bias = self.bias*0.01 # just a small value but not too small
        elif self.weights_init == 'zeros':
            self.weights = np.zeros((self.input_size, self.n_neurons))
            self.bias = np.zeros((1, self.n_neurons))
        elif self.weights_init == 'normal':
            self.weights = np.random.normal(scale=1/self.input_size, size=(self.input_size, self.n_neurons))
            self.bias = np.random.normal(scale=1/self.input_size, size=(1, self.n_neurons))
            
        self.num_of_params = self.weights.size + self.bias.size

    def set_activation_func(self, activation_type):
        self.activation_type = activation_type
        if activation_type == 'sigmoid':
            self.activation = self.sigmoid
        elif activation_type == 'softmax':
            self.activation = self.softmax
        elif activation_type == 'relu':
            self.activation = self.relu
      
    def __str__(self):
        return self.type + ' layer - neurons: ' + str(self.n_neurons) +\
            ' - activation type: ' + self.activation_type +\
            ' - inputs: ' + str(self.input_size) +\
            ' - weights_init_method: ' + self.weights_init +\
            ' - ' + str(self.num_of_params) + ' params'

    def predict(self, input_values):
        return self.forward(input_values)
    
    def forward(self, input_values):
        output = input_values.dot(self.weights) + self.bias
        output = self.activation(output)
        return output
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def softmax(self, x):
        exp_input = np.exp(x)
        sum_of_exp = np.sum(exp_input, axis=1)
        return np.divide(exp_input.T, sum_of_exp).T

    def relu(self, x):
        def relu1d(x1d):
            return np.where(x1d > 0, x1d, 0) 
        return np.apply_along_axis(relu1d, 0, x)
        
    def d_E_d_Ij(self, predicted_output, incoming_gradient):
        # d_E_d_Ij = (d_E_d_oj) (d_oj_d_Ij)

        if self.activation_type == 'sigmoid':
            sig_tmp = self.sigmoid(predicted_output)
            return ((sig_tmp * (1- sig_tmp)))*incoming_gradient
        
        if self.activation_type == 'softmax':
            return (predicted_output - incoming_gradient)
        
        if self.activation_type == 'relu':
            return np.where(predicted_output > 0, 1, 0)*incoming_gradient
        
    def backpropagate(self, input_values, predicted_output,
                      incoming_gradient, lr, momentum):
        # d_E_d_Ij = (d_E_d_oj) (d_oj_d_Ij)
        d_E_d_Ij = self.d_E_d_Ij(predicted_output, incoming_gradient)
        # o_i * [d_E_d_Ij]
        # breakpoint()
        d_w = input_values.T.dot(d_E_d_Ij)/input_values.shape[0]
        d_b = np.sum(d_E_d_Ij, axis=0)/input_values.shape[0]
        
        # Incoming gradient for previous layer
        d_E_d_Ii = d_E_d_Ij.dot(self.weights.T)
        
        # ADAM OPTIMIZER
        # m = beta1*m + (1-beta1)*d_w
        # v = beta2*v + (1-beta2)*(d_w**2)
        # mhat = m/(1-beta1**epoch)
        # vhat = v/(1-beta2**epoch)
        # d_w = mhat/(np.sqrt(vhat) + 1E-8)
        
        # 6 update weights
        self.weights -= lr*d_w
        self.bias -= lr*d_b

        # self.debug_i += 1
        # if self.debug_i > 10000:
        #     breakpoint()
        return d_E_d_Ii
    
class Dropout:
    def __init__(self, probability, input_size=None):
        self.type = 'Dropout'
        self.probability=probability
        self.mask = None
        self.input_size = input_size
        if self.input_size is not None:
            self._config_params()
    
    def _config_params(self, input_size=None):
        self.input_size = input_size if input_size is not None else self.input_size
        self.n_neurons = self.input_size
        self.num_of_params = 0
               
    def __str__(self):
        return self.type + ' layer - probability: ' + str(self.probability) +\
            ' - input size: ' + str(self.input_size)
        
    def predict(self, input_values):
        return input_values#*(1-self.probability)
    
    def forward(self, input_values):
        if self.mask is None:
            self.mask = np.random.binomial(1, 1-self.probability,input_values.shape)
            self.mask = self.mask/(1-self.probability) # Inverse Dropout
        output = input_values*self.mask
        return output
    
    def backpropagate(self, input_values, predicted_output,
                      prev_gradient, lr, momentum): 
        old_mask = self.mask
        self.mask = None
        # breakpoint()
        d_E_d_Ii = prev_gradient*old_mask
        return d_E_d_Ii
    