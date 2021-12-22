import numpy as np
import random
import mnist_loader
import math

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network:
    def __init__(self, sizes, trained_parameters=None):
        if trained_parameters:
            pass
        else:
            self.num_layers = len(sizes)
            self.sizes = sizes
            self.biases = [np.random.randn(y,1) for y in sizes[1:]]
            self.weights = [np.random.randn(y,x) for y,x in zip(sizes[:-1], sizes[1:])]
            self.usums = [np.zeros((y,1)) for y in sizes[1:]]
            self.activations = [np.zeros((y,1)) for y in sizes[1:]]

    # feedforward from input to output
    def feedforward(self, input):
        
        current_layer_activation = input
        index = 0
        for lb, lw in zip(self.biases, self.weights):
            # print(lw.shape)
            # print(current_layer_activation.shape)
            # print(lb.shape)
            layer_size = lw.shape[1] # or lb.shape[0]
            output_weighted_sum = np.zeros((layer_size,1)) 
            # neuron weight, activation
            for nw, a in zip(lw, current_layer_activation):
                addend = nw * a      
                output_weighted_sum += np.ndarray(shape=(layer_size,1), buffer=addend,dtype=float)
            current_layer_activation = sigmoid(output_weighted_sum + lb)

            self.usums[index] = output_weighted_sum + lb
            self.activations[index] = current_layer_activation

            index += 1

        return current_layer_activation

    # trains and test the network in batches 
    def StochasticGradientDescent(self, training_data, epochs, batch_size, learning_rate, test_data=None):
        num_test = len(test_data)
        num_train = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k: k+batch_size] for k in range(0, num_train, batch_size)]
            for batch in batches:
                self.update_network(batch, learning_rate)
            if test_data != None:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {num_test}")
            print(f"Epoch {j}")


    def update_network(self, batch, learning_rate):
        # this 2 value below is the final changes applied to the network's biases and weights
        total_changes_b = [np.zeros(b.shape) for b in self.biases]
        total_changes_w = [np.zeros(w.shape) for w in self.weights]
        for training_input, result in batch:
            # this 2 value is how 1 training data want to modify the weight and biases
            single_change_b, single_change_w = self.backpropagate(training_input, result)
            total_changes_b = [tcb+scb for tcb, scb in zip(total_changes_b, single_change_b)]
            total_changes_w = [tcw+scw for tcw, scw in zip(total_changes_w, single_change_w)]
        self.weights = [w - (learning_rate/len(batch))*changes for w, changes in zip(self.weights, total_changes_w)]
        self.biases = [b - (learning_rate/len(batch))*changes for b, changes in zip(self.biases, total_changes_b)]


    # my own implementation
    def backpropagate(self,training_input, result):
        # feedforward
        output = self.feedforward(training_input)
    
        # backpropagate
        # calculate errors (bias error, same thing) in output layer
        # same shape as bías array
        bias_errors = [np.zeros((y,1)) for y in self.sizes[1:]]
        L_error = np.multiply(self.activations[-1] - result, sigmoid_prime(self.usums[-1]))
        bias_errors[-1] = L_error
        # calcualte error of backward layer until finish
        # excluding output layer
        # using len(sizes) as endpoint here because it's easier to understand how we are accessing the weight, activation and bias arrays
        for l in reversed(range(0,len(bias_errors)-1)):
            # we = np.multiply(self.weights[l], errors[l]) * sigmoid_prime(self.usums[l])
            # errors[l-1] = [we*sp for we,sp in zip(np.multiply(self.weights[l], errors[l]) * sigmoid_prime(self.usums[l]))]
            for e in range(0, len(bias_errors[l])):
                transposed_weight = np.ndarray((self.weights[l+1].shape[1],1), buffer=self.weights[l+1][e], dtype=float)
                bias_errors[l][e] = np.sum(np.multiply(transposed_weight, bias_errors[l+1])* sigmoid_prime(self.usums[l][e]))
                    
        
        # calculate weight error
        weight_error = [np.zeros(w.shape) for w in self.weights]
        for l in range(0, len(self.weights)):
            for k in range(0, len(self.weights[l])):
                # print(f"{l} {k}")
                if l == 0:
                    weight_error[l][k] = np.multiply(bias_errors[l], training_input[k]).flatten()
                else:      
                    weight_error[l][k] = np.multiply(bias_errors[l], self.activations[l-1][k]).flatten()
        # return changes
        return bias_errors, weight_error

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in test_results)
    
    def export(self, filename):
        np.savez(filename, sizes=np.array(self.sizes, dtype=object), weights=np.array(self.weights, dtype=object), biases=np.array(self.biases,dtype=object))

network = Network([784, 20, 10])
tr_zip, val_zip, test_zip = mnist_loader.load_data_wrapper()
training_data = [(x,y) for x,y in tr_zip]
test_data = [(x,y) for x,y in test_zip]
network.StochasticGradientDescent(training_data, 1,10,3.0, test_data=test_data)
network.export("parameters")

