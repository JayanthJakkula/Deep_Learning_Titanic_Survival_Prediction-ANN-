import pandas as pd
import numpy as np
import pickle

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("Titanic-Dataset.csv")

# ==========================================
# SELECT FEATURES
# ==========================================

X = df[['Pclass', 'Age', 'Fare']].values
y = df['Survived'].values.reshape(-1, 1)

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# ==========================================
# MIN MAX NORMALIZATION
# ==========================================

X_min = X.min(axis=0)
X_max = X.max(axis=0)

X = (X - X_min) / (X_max - X_min)

# Save scaler values
scaler_data = {
    "min": X_min,
    "max": X_max
}

pickle.dump(scaler_data, open("scaler.pkl", "wb"))

# ==========================================
# ANN USING NUMPY
# ==========================================

np.random.seed(42)

input_neurons = 3
hidden_neurons = 4
output_neurons = 1

# Weights
W1 = np.random.randn(input_neurons, hidden_neurons)
b1 = np.zeros((1, hidden_neurons))

W2 = np.random.randn(hidden_neurons, output_neurons)
b2 = np.zeros((1, output_neurons))

# ==========================================
# ACTIVATION FUNCTIONS
# ==========================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ==========================================
# TRAINING
# ==========================================

epochs = 5000
lr = 0.1

for epoch in range(epochs):

    # FORWARD PROPAGATION

    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    predicted_output = sigmoid(final_input)

    # ERROR

    error = y - predicted_output

    # BACKPROPAGATION

    d_output = error * sigmoid_derivative(predicted_output)

    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # UPDATE WEIGHTS

    W2 += hidden_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr

    W1 += X.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

    if epoch % 500 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch} Loss: {loss}")

# ==========================================
# SAVE MODEL
# ==========================================

model = {
    "W1": W1,
    "b1": b1,
    "W2": W2,
    "b2": b2
}

pickle.dump(model, open("titanic_model.pkl", "wb"))

print("\nModel Saved Successfully")