# 3rd party imports
import torch as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
import data
def test_model(model, settings):
    # Load the test dataset (should return a dictionary with 'X' and 'U')
    dataset = data.get_test_dataset(settings)
    
    # Extract X and U from the dataset
    X = dataset['X']  # State variables (Tr and Tj)
    U = dataset['U']  # Control input (Fc)
    
    # Move X and U to the correct device
    device = settings['device']
    X = X.to(device)
    U = U.to(device)
    
    # Define the loss function
    loss_function = settings['loss_function']()

    # Initialize MSE dictionary for state variables and total MSE
    MSEs = {k: [] for k in settings[settings['process']]['state_names']}
    MSEs['total'] = []

    # Make predictions using the model
    X_pred = model.multi_step_prediction(X[0, :], U[:-1, :])

    # Ensure X_pred is on the same device as X
    X_pred = X_pred.to(device)

    print("Model predictions (X_pred):", X_pred)
    print("Target values (X):", X)


    # Calculate MSE for each state variable and total MSE
    for i, name in enumerate(settings[settings['process']]['state_names']):
        MSEs[name].append(loss_function(X_pred[:, i], X[:, i]).item())
    MSEs['total'].append(loss_function(X_pred, X).item())

    # Plot if plotting is enabled
    if settings['plot_test']:
        plot_test_trajectory(X, U, X_pred,
                             {k: v[-1] for k, v in MSEs.items()},
                             'test', settings)

    # Print the test results
    print(f"\nTest results for {settings['model_name']}:")
    for name in settings[settings['process']]['state_names']:
        print(f"{name} MSE: {np.mean(MSEs[name])}")
    print(f"Total MSE: {np.mean(MSEs['total'])}")

    return None

import torch
from torch.nn import MSELoss

def test_model_single(model, settings):
    dataset = data.get_test_dataloader1(settings)
    criterion = MSELoss()
    model.eval()

    total_loss = 0.0
    num_batches = 0

    for row in dataset:
        X0, U0, X1 = row  
        X0 = X0.to(settings['device'])
        U0 = U0.to(settings['device'])
        X1 = X1.to(settings['device'])

        x1_pred = model(X0, U0)

        val_loss = criterion(x1_pred, X1)

        total_loss += val_loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    print(f'Validation Loss: {avg_loss:.4f}')

    return avg_loss



def predict_timeseries_closed_loop(model, X, U):
    x = X[0, :]
    X_pred = T.zeros_like(X)
    X_pred[0, :] = x
    for t in range(X.shape[0]-1):
        x = model(x, U[t, :])
        X_pred[t+1, :] = x
    return X_pred

def plot_test_trajectory(X, U, X_pred, MSEs, i_tds, settings):
    # convert to pandas dataframes
    X = pd.DataFrame(X.detach().to(T.device('cpu')).numpy(), columns=settings[settings['process']]['state_names'])
    U = pd.DataFrame(U.detach().to(T.device('cpu')).numpy(), columns=settings[settings['process']]['control_names'])
    X_pred = pd.DataFrame(X_pred.detach().to(T.device('cpu')).numpy(), columns=settings[settings['process']]['state_names'])

    # set up figure
    fig, ax = plt.subplots(X.shape[1] + U.shape[1], 1,
                           sharex=True, sharey=False,
                           figsize=(10,10))
    linewidth = 0.8
    fig.suptitle(f"Overall MSE: {MSEs['total']:.7f}")

    # plot states
    for i, name in enumerate(settings[settings['process']]['state_names']):
        ax[i].plot(X[name], label='true', linewidth=linewidth)
        ax[i].plot(X_pred[name], label='pred', linewidth=linewidth)
        ax[i].set_ylabel(name)
        ax[i].legend()
        ax[i].set_title(f"State {name}, MSE: {MSEs[name]:.7f}")

    # plot controls
    for i, name in enumerate(settings[settings['process']]['control_names']):
        ax[i+X.shape[1]].plot(U[name], label='true', linewidth=linewidth)
        ax[i+X.shape[1]].set_ylabel(name)
        ax[i+X.shape[1]].legend()
        ax[i+X.shape[1]].set_title(f"Control {name}")

    plt.xlabel('Time steps')
    plt.tight_layout()
    plt.savefig(f"{settings['plots_dir']}test_{i_tds}.pdf")
    plt.close(fig)

    return None
