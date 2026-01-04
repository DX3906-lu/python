import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))  # 防止指数溢出

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.square(x)


class ImprovedAutoEncoder:
    def __init__(self, network_structure, activation=tanh, activation_deriv=tanh_derivative):
        self.structure = network_structure 
        self.activation = activation      
        self.activation_deriv = activation_deriv  

        self.weights = []
        for i in range(1, len(network_structure)):
            prev_units = network_structure[i-1] + 1
            curr_units = network_structure[i]
            weight = 2 * np.random.rand(prev_units, curr_units) - 1
            self.weights.append(weight)

    def forward_propagation(self, x):
        activations = [x] 
        current = np.hstack([1, x])
        
        for i, weight in enumerate(self.weights):
            z = np.dot(current, weight)
            a = self.activation(z)
            activations.append(a)
            if i != len(self.weights) - 1:
                current = np.hstack([1, a])
            else:
                current = a
        
        return activations

    def train_batch(self, X, lr=0.01, epochs=10000, batch_size=32, print_interval=1000):
        n_samples = X.shape[0]
        X_with_bias = np.hstack([np.ones((n_samples, 1)), X])
        
        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            X_shuffled = X_with_bias[idx]
            
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch = X_shuffled[batch_start:batch_end]
                
                weight_updates = [np.zeros_like(w) for w in self.weights]
                
                for x in batch:
                    activations = self.forward_propagation(x[1:])  
                    output = activations[-1]
                    input_x = x[1:]  
                    
                    delta = (output - input_x) * self.activation_deriv(output)
                    deltas = [delta] 
                    
                    for i in range(len(self.weights)-1, 0, -1):
                        delta = np.dot(deltas[-1], self.weights[i][1:].T) * self.activation_deriv(activations[i])
                        deltas.append(delta)
                    
                    deltas.reverse()
                    
                    for i in range(len(self.weights)):
                        prev_activation = np.hstack([1, activations[i]]) if i == 0 else np.hstack([1, activations[i]])
                        weight_updates[i] += np.outer(prev_activation, deltas[i])
                
                for i in range(len(self.weights)):
                    self.weights[i] -= lr * (weight_updates[i] / batch_size)
            
            if (epoch + 1) % print_interval == 0:
                total_mse = 0
                for x in X:
                    output = self.predict(x)
                    total_mse += mean_squared_error(x, output)
                avg_mse = total_mse / n_samples
                print(f"Epoch {epoch+1:5d} | Average Reconstruction MSE: {avg_mse:.6f}")

    def predict(self, x):
        activations = self.forward_propagation(x)
        return activations[-1]


iris = datasets.load_iris()
X = iris['data'] 


scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)


if __name__ == "__main__":
    network_structure = [4, 2, 4]
    
    autoencoder = ImprovedAutoEncoder(
        network_structure=network_structure,
        activation=tanh,
        activation_deriv=tanh_derivative
    )
    
    print("Start Training AutoEncoder...")
    autoencoder.train_batch(
        X=X_scaled,
        lr=0.008,
        epochs=8000,
        batch_size=16,
        print_interval=1000
    )
    
    print("\n" + "-"*50)
    print("Test Reconstruction Result (First 10 Samples):")
    print("-"*50)
    for i in range(10):
        input_sample = X_scaled[i]
        reconstructed = autoencoder.predict(input_sample)
        print(f"Sample {i+1:2d} | Input: {input_sample.round(4)} | Reconstructed: {reconstructed.round(4)}")
    
    total_mse = mean_squared_error(X_scaled, [autoencoder.predict(x) for x in X_scaled])
    print("\n" + "-"*50)
    print(f"Final Overall Reconstruction MSE: {total_mse:.6f}")
    print("-"*50)