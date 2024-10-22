# 3rd party imports
import os
import torch as T
import numpy as np

# Local imports
import networks
import test_model
import data
import training

def get_settings():
    # Settings
    settings = {
        # General
        'train_new_model':              True,
        'process':                      'CSTR1',     # 'CSTR1', 'ASU'
        'device':                       T.device('cuda' if T.cuda.is_available() else 'cpu'),
        'model_type':                   'Koopman',    # 'MLP', 'Koopman', 'Linear'
        'accuracy':                     T.float32,  # T.float32, T.float64

        # Training
        'learning_rate':                0.001,
        'max_epochs':                   5_000,
        'batch_size':                   64,
        'validate_every_n_epochs':      1,
        'print_every_n_epochs':         10,
        'early_stopping_patience':      200,
        'train_val_ratio':              0.8,
        'optimizer':                    T.optim.Adam,
        'loss_function':                T.nn.MSELoss,

        # Testing
        'plot_test':                    True,

        # MLP model architecture
        'MLP': {
            'hidden_layer_sizes':       [10, 10],
            'activation':               T.nn.Tanh,
            'output_activation':        T.nn.Identity,
        },

        # Koopman model architecture
        'Koopman': {
            'latent_dim':               8,
            'encoder_n_hidden_layers':  2,
            'encoder_activation':       T.nn.Tanh,
        },

        # Koopman training parameters
        'Koopman_loss_weighting': {
            'comb_loss':                0.34,
            'ae_loss':                  0.33,
            'pred_loss':                0.33,
        },

        # Process
        'CSTR1': {
            'n_states':      2,
            'n_controls':    2,
            'state_names':   ['c', 'T'],
            'control_names': ['roh', 'Fc'],
            'state_scaling': {
                'min_unscaled': np.array([0.9*0.1367, 0.8*0.7293]),     # c_lower, T_lower
                'max_unscaled': np.array([1.1*0.1367, 1.2*0.7293]),     # c_upper, T_upper
                'min_scaled':   np.array([-1.0, -1.0]),
                'max_scaled':   np.array([1.0, 1.0]),
            },
            'action_scaling': {
                'min_unscaled': np.array([0.8/(60*60), 0.0/(60*60)]),   # rho_lower, Fc_lower
                'max_unscaled': np.array([1.2/(60*60), 700.0/(60*60)]), # rho_upper, Fc_upper
                'min_scaled':   np.array([-1.0, -1.0]),
                'max_scaled':   np.array([1.0, 1.0]),
            },
            'dataset_name': '5d_29July_train',
            'testset_name': '5d_29July_test',
        }
    }

    # Directories
    settings['models_dir'] = f"./models/{settings['process']}/"
    settings['model_name'] = f"{settings['model_type']}"
    settings['logdir'] = f"./logs/{settings['process']}/{settings['model_name']}/"
    settings['data_dir'] = f"./data/{settings['process']}/"
    settings['plots_dir'] = f"./plots/{settings['process']}/{settings['model_name']}/"
    settings['train_data_path'] = f"{settings['data_dir']}{settings[settings['process']]['dataset_name']}.pickle"
    settings['test_data_path'] = f"{settings['data_dir']}{settings[settings['process']]['testset_name']}.pickle"

    os.makedirs(settings['models_dir'], exist_ok=True)
    os.makedirs(settings['logdir'], exist_ok=True)
    os.makedirs(settings['plots_dir'], exist_ok=True)
    return settings

if __name__ == "__main__":
    settings = get_settings()

    train_dataloader, val_dataloader = data.get_train_val_dataloaders(settings)

    model = getattr(networks, settings['model_type'])(settings).to(settings['accuracy'])

    if settings['train_new_model']:
        trainer = getattr(training, f"{settings['model_type']}Trainer")(
            model, train_dataloader, val_dataloader, settings)
        trainer.train()

    model.load_self(name_suffix='best_val')

    test_model.test_model(model, settings)

    print('Done!')