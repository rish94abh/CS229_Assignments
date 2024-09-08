import matplotlib.pyplot as plt
import numpy as np
import util

from lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem: Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    best_tau = None
    lowest_mse = float('inf')

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        y_valid_pred = model.predict(x_valid)
        mse = np.mean((y_valid_pred - y_valid) ** 2)
        print(f'Tau: {tau}, Validation MSE: {mse}')

        if mse < lowest_mse:
            lowest_mse = mse
            best_tau = tau

    print(f'Best tau: {best_tau}, with MSE: {lowest_mse}')
    
    # Fit a LWR model with the best tau value
    best_model = LocallyWeightedLinearRegression(best_tau)
    best_model.fit(x_train, y_train)
    
    # Run on the test set to get the MSE value
    y_test_pred = best_model.predict(x_test)
    test_mse = np.mean((y_test_pred - y_test) ** 2)
    print(f'Test MSE: {test_mse}')
    # Save predictions to pred_path
    np.savetxt(pred_path, y_test_pred)
    # Plot data
    plt.figure()
    plt.plot(x_train[:, 1], y_train, 'bx', label='Training data')
    plt.plot(x_valid[:, 1], y_valid, 'go', label='Validation data')
    plt.plot(x_test[:, 1], y_test_pred, 'ro', label='Test predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Locally Weighted Linear Regression (best tau={best_tau})')
    plt.show()
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='./train.csv',
         valid_path='./valid.csv',
         test_path='./test.csv',
         pred_path='./pred.txt')
