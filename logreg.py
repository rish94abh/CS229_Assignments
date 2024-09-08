import numpy as np
import util
import matplotlib.pyplot as plt
import pandas as pd

def main(train_path, save_path):
    """Problem: Logistic regression with gradient descent.

    Args:
        train_path: Path to CSV file containing dataset for training.
        save_path: Path to save outputs; visualizations, predictions, etc.
    """
    x_train, y_train = util.load_csv(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    # Plot decision boundary on top of validation set.
        # Create a mesh grid
    #x_min, x_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
    x_min, x_max = x_train[:, 1].min() , x_train[:, 1].max()
    #y_min, y_max = x_train[:, 2].min() - 1, x_train[:, 2].max() + 1
    y_min, y_max = x_train[:, 2].min() , x_train[:, 2].max() 
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict the probabilities for each point in the mesh grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid = np.hstack([np.ones((grid.shape[0], 1)), grid])
    #probs = clf.predict_prob(grid).reshape(xx.shape)
    predictions = model.predict(grid).reshape(xx.shape)
    
    #predictions = model.predict(x_train)
    data = pd.read_csv(train_path)

    # Extract the columns for x1, x2, and y
    x1 = data['x0']
    x2 = data['x1']
    y = data['y']
    #p = data['p(y|x)']

    # Prepare the data for logistic regression
    X = data[['x0', 'x1']].values
    y = data['y'].values
    #p = data['p(y|x)'].values
    
    # Create a scatter plot with different symbols for the two classes
    plt.figure(figsize=(10, 6))

    # Plot class 0
    plt.scatter(x1[y == 0], x2[y == 0], label='Class 0', marker='o', color='blue')

    # Plot class 1
    plt.scatter(x1[y == 1], x2[y == 1], label='Class 1', marker='x', color='red')
    plt.contour(xx, yy, predictions, levels=[0.5], linestyles=['dashed'], colors='black')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Scatter Plot with Decision Boundary')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()
    
    #Added later
    #accuracy = np.mean(predictions == y_train)
    #print("Accuracy on the training set:", accuracy)
    #plot_decision_boundary(x_train, y_train, model)
    #
    plot_loss(model.losses, title="Dataset A Loss")
    # Use save_path argument to save various visualizations for your own reference.
    np.savetxt(save_path, predictions, delimiter=',')
    
    # Plot decision boundary
    # plot_decision_boundary(model, x_train, y_train, save_path.replace('.txt', '.png'))
    # *** END CODE HERE ***





class LogisticRegression:
    """Logistic regression using gradient descent.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, learning_rate=1, max_iter=100000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            learning_rate: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.losses=[]

        # *** START CODE HERE ***
    def compute_loss(self, X, y, theta):
        m = X.shape[0]
        h = self.sigmoid(X @ theta)
        loss = -1/m * (y @ np.log(h) + (1 - y) @ np.log(1 - h))
        return loss
        
        # *** END CODE HERE ***

    def sigmoid(self, z):
        """Compute the sigmoid of z.

        Args:
        z: A scalar or numpy array of any size.

        Returns:
            Sigmoid of z.
            """
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y):
        """Run gradient descent to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n_examples, dim = x.shape
        if self.theta is None:
           self.theta = np.zeros(dim)

        for i in range(self.max_iter):
           z = np.dot(x, self.theta)
           h = self.sigmoid(z)
           #gradient = np.dot(x.T, (h - y)) / n_examples + 0.01*self.theta
           gradient = np.dot(x.T, (h - y)) / n_examples
           #self.theta -= self.learning_rate * gradient 
           new_theta = self.theta - self.learning_rate * gradient
           loss = self.compute_loss(x, y, new_theta)
           self.losses.append(loss)
           
           # Check for convergence
           if np.linalg.norm(self.learning_rate * gradient) < self.eps:
               print(self.theta)
               break
           
           self.theta = new_theta
           if self.verbose and i % 1000 == 0:
                loss = self.compute_loss(x, y, self.theta)
                print(f"Iteration {i}, Loss: {loss}")
                print(self.theta)
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return self.sigmoid(np.dot(x, self.theta))
        # *** END CODE HERE ***


    def plot_decision_boundary(X, y, model):
        plt.figure(figsize=(10, 6))
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0', marker='o', color='blue')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', marker='x', color='red')
        x_values = [np.min(X[:, 0] - 1), np.max(X[:, 1] + 1)]
        y_values = -(model.theta[0] * x_values + model.theta[2]) / model.theta[1]
        plt.plot(x_values, y_values, label='Decision Boundary', color='black')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Scatter Plot with Decision Boundary')
        plt.legend()
        plt.show()

def plot_loss(losses, title=""):
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    #plt.title(title)
    plt.show()

if __name__ == '__main__':
    print('==== Training model on data set A ====')
    main(train_path='ds1_a.csv',
         save_path='logreg_pred_a.txt')
    

    print('\n==== Training model on data set B ====')
    main(train_path='ds1_b.csv',
         save_path='logreg_pred_b.txt')
