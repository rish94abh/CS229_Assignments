import numpy as np
import util
import sys
from random import random
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../logreg_stability')

### NOTE : You need to complete logreg implementation first! If so, make sure to set the regularization weight to 0.
from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    #x_train, y_train = util.load_csv(train_path, add_intercept=True)
    #x_val, y_val = util.load_csv(validation_path, add_intercept=True)
    
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(validation_path, add_intercept=True)
    
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    model_naive = LogisticRegression(learning_rate=1e-2, max_iter=100000, eps=1e-5, verbose=False)
    model_naive.fit(x_train, y_train)
    y_val_pred_naive = model_naive.predict(x_val)
    np.savetxt(output_path_naive, y_val_pred_naive, delimiter=',')
    
    # Part (d): Upsampling minority class
    minority_class = 1
    minority_indices = np.where(y_train == minority_class)[0]
    majority_indices = np.where(y_train != minority_class)[0]
    
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    upsample_factor = int(1 / kappa)
    upsampled_minority_indices = np.tile(minority_indices, upsample_factor)
    new_indices = np.concatenate([majority_indices, upsampled_minority_indices])
    x_upsampled = x_train[new_indices]
    y_upsampled = y_train[new_indices]
    # Repeat minority examples 1 / kappa times
    model_upsampled = LogisticRegression(learning_rate=1e-2, max_iter=100000, eps=1e-5, verbose=False)
    model_upsampled.fit(x_upsampled, y_upsampled)
    y_val_pred_upsampled = model_upsampled.predict(x_val)
    np.savetxt(output_path_upsampling, y_val_pred_upsampled, delimiter=',')
    
    
    #Extra
    x_min, x_max = x_train[:, 1].min() , x_train[:, 1].max()
    #y_min, y_max = x_train[:, 2].min() - 1, x_train[:, 2].max() + 1
    y_min, y_max = x_train[:, 2].min() , x_train[:, 2].max() 
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict the probabilities for each point in the mesh grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid = np.hstack([np.ones((grid.shape[0], 1)), grid])
    #probs = clf.predict_prob(grid).reshape(xx.shape)
    predictions = model_upsampled.predict(grid).reshape(xx.shape)
    
    #predictions = model.predict(x_train)
    data = pd.read_csv(train_path)

    # Extract the columns for x1, x2, and y
    x1 = data['x_1']
    x2 = data['x_2']
    y = data['y']
    #p = data['p(y|x)']

    # Prepare the data for logistic regression
    X = data[['x_1', 'x_2']].values
    y = data['y'].values
    #p = data['p(y|x)'].values
    
    # Create a scatter plot with different symbols for the two classes
    plt.figure(figsize=(10, 6))

    # Plot class 0
    plt.scatter(x1[y == 0], x2[y == 0], label='Class 0', marker='o', color='blue')

    # Plot class 1
    plt.scatter(x1[y == 1], x2[y == 1], label='Class 1', marker='x', color='green')
    plt.contour(xx, yy, predictions, levels=[0.5], linestyles=['dashed'], colors='red')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Scatter Plot with Decision Boundary')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()
    
    
    # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')
