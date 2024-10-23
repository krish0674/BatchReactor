# 3rd party imports
import torch as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
import data
def test_model(model, settings):
    # Load the test dataset
    dataset = data.get_test_dataset(settings)
    print(dataset)

    # Move the model to the appropriate device (GPU or CPU)
    device = settings['device']  # Assuming settings['device'] is set correctly ('cuda' or 'cpu')
    model.to(device)  # Move the model to the same device

    loss_function = settings['loss_function']()

    MSEs = {k: [] for k in settings[settings['process']]['state_names']}
    MSEs['total'] = []

    # Access X and U directly from the dataset (no need for iteration)
    X = dataset['X'].to(device)  # Move the input data to the same device as the model
    U = dataset['U'].to(device)  # Move control inputs to the same device

    # Make predictions using the model
    X_pred = model.multi_step_prediction(X[0, :], U[:-1, :])

    # Calculate MSE for each state variable and the total MSE
    for i, name in enumerate(settings[settings['process']]['state_names']):
        MSEs[name].append(loss_function(X_pred[:, i], X[:, i]).item())
    MSEs['total'].append(loss_function(X_pred, X).item())

    # If plotting is enabled, visualize the results
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