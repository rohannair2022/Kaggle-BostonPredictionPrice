import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pre_proccess_script import data_preprocess

# Preprocess the data
dataFrame = data_preprocess(pd.read_csv('Dataset/train.csv'))

# Extracting the columns to be used for training
all_columns = dataFrame.columns
columns_to_include = [col for col in all_columns if col != 'SalePrice' and col != 'Id']

dependant = dataFrame[columns_to_include].values
target = dataFrame['SalePrice'].values

# Initialize weight vector and other parameters
weight_vector = np.zeros(dependant.shape[1])
intercept = 0
alpha = 0.001


# Define loss function
def loss(y, y_predicted):
    return np.mean((y - y_predicted) ** 2) / 2


# Define function to predict y
def predicted_y(weight, x, intercept):
    return np.dot(x, weight) + intercept


# Define gradients
def dldw(x, y, y_predicted):
    return (1 / len(y)) * np.dot(x.T, (y_predicted - y))


def dldb(y, y_predicted):
    return (1 / len(y)) * np.sum(y_predicted - y)


# Training loop
losses = []

for i in range(10000):
    y_predicted = predicted_y(weight_vector, dependant, intercept)
    weight_vector -= alpha * dldw(dependant, target, y_predicted)
    intercept -= alpha * dldb(target, y_predicted)

    current_loss = loss(target, y_predicted)
    losses.append(current_loss)

    # Optionally, print the loss every few iterations
    if i % 10 == 0:
        print(f"Iteration {i}: Loss = {current_loss}")

# Plotting the loss over iterations
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Loss over iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Gradient Descent Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

# Load and preprocess the test data
test_Data = pd.read_csv('Dataset/test.csv')

test_data_initial = data_preprocess(pd.read_csv('Dataset/test.csv'))

# Ensure the test data has the same columns as the training data
missing_columns = set(columns_to_include) - set(test_data_initial.columns)
for col in missing_columns:
    test_data_initial[col] = 0

# Reorder the test data columns to match the training data
test_data = test_data_initial[columns_to_include]

# Convert test data to numpy array
test_dependant = test_data.values

# Predict the values for the test set
test_predictions = predicted_y(weight_vector, test_dependant, intercept)

# Ensure the predictions are in the correct format
predictions_df = pd.DataFrame({
    'Id': test_data_initial['Id'],
    'SalePrice': test_predictions
})

# Save the predictions to a CSV file
predictions_df.to_csv('predictions.csv', index=False)
