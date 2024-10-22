import os
import torch as T
import numpy as np
import argparse

# Local imports
import networks
import test_model
import data
import training

def get_settings(args):
    # Settings with command-line overrides
    settings = {
        # General
        'train_new_model':              args.train_new_model,
        'process':                      args.process,
        'device':                       T.device('cuda' if T.cuda.is_available() else 'cpu'),
        'model_type':                   args.model_type,
        'accuracy':                     T.float32 if args.accuracy == 'float32' else T.float64,

        # Training
        'learning_rate':                args.learning_rate,
        'max_epochs':                   args.max_epochs,
        'batch_size':                   args.batch_size,
        'validate_every_n_epochs':      args.validate_every_n_epochs,
        'print_every_n_epochs':         args.print_every_n_epochs,
        'early_stopping_patience':      args.early_stopping_patience,
        'train_val_ratio':              args.train_val_ratio,
        'optimizer':                    T.optim.Adam,
        'loss_function':                T.nn.MSELoss,

        # Testing
        'plot_test':                    args.plot_test,

        # MLP model architecture
        'MLP': {
            'hidden_layer_sizes':       args.MLP_hidden_layer_sizes,
            'activation':               T.nn.Tanh,
            'output_activation':        T.nn.Identity,
        },

        # Koopman model architecture
        'Koopman': {
            'latent_dim':               args.Koopman_latent_dim,
            'encoder_n_hidden_layers':  args.Koopman_encoder_n_hidden_layers,
            'encoder_activation':       T.nn.Tanh,
        },

        # Koopman training parameters
        'Koopman_loss_weighting': {
            'comb_loss':                args.Koopman_comb_loss,
            'ae_loss':                  args.Koopman_ae_loss,
            'pred_loss':                args.Koopman_pred_loss,
        },

        # Process
        'CSTR1': {
            'n_states':      2,
            'n_controls':    2,
            'state_names':   ['c', 'T'],
            'control_names': ['roh', 'Fc'],
            'state_scaling': {
                'min_unscaled': np.array([0.9*0.1367, 0.8*0.7293]),
                'max_unscaled': np.array([1.1*0.1367, 1.2*0.7293]),
                'min_scaled':   np.array([-1.0, -1.0]),
                'max_scaled':   np.array([1.0, 1.0]),
            },
            'action_scaling': {
                'min_unscaled': np.array([0.8/(60*60), 0.0/(60*60)]),
                'max_unscaled': np.array([1.2/(60*60), 700.0/(60*60)]),
                'min_scaled':   np.array([-1.0, -1.0]),
                'max_scaled':   np.array([1.0, 1.0]),
            },
            'dataset_name': args.dataset_name,
            'testset_name': args.testset_name,
        }
    }

    # Directories (using process and model_type from args)
    process = args.process
    model_type = args.model_type
    settings['models_dir'] = f"./models/{process}/"
    settings['model_name'] = f"{model_type}"
    settings['logdir'] = f"./logs/{process}/{model_type}/"
    settings['data_dir'] = f"./data/{process}/"
    settings['plots_dir'] = f"./plots/{process}/{model_type}/"
    settings['train_data_path'] = f"{settings['data_dir']}{settings['CSTR1']['dataset_name']}.pickle"
    settings['test_data_path'] = f"{settings['data_dir']}{settings['CSTR1']['testset_name']}.pickle"

    os.makedirs(settings['models_dir'], exist_ok=True)
    os.makedirs(settings['logdir'], exist_ok=True)
    os.makedirs(settings['plots_dir'], exist_ok=True)
    return settings

def main():
    parser = argparse.ArgumentParser(description='Process model training settings.')
    # Adding parser arguments corresponding to all settings
    parser.add_argument('--train_new_model', type=bool, default=True, help='Whether to train a new model')
    parser.add_argument('--process', type=str, default='CSTR1', choices=['CSTR1', 'ASU'], help='Process type')
    parser.add_argument('--model_type', type=str, default='Koopman', choices=['MLP', 'Koopman', 'Linear'], help='Type of model to train')
    parser.add_argument('--accuracy', type=str, default='float32', choices=['float32', 'float64'], help='Floating point precision')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--max_epochs', type=int, default=5000, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--validate_every_n_epochs', type=int, default=1, help='Frequency of validation per number of epochs')
    parser.add_argument('--print_every_n_epochs', type=int, default=10, help='Frequency of printing progress per number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=200, help='Patience for early stopping')
    parser.add_argument('--train_val_ratio', type=float, default=0.8, help='Training to validation data ratio')
    parser.add_argument('--plot_test', type=bool, default=True, help='Whether to plot test results')
    # Add more parameters as needed...
    args = parser.parse_args()
    settings = get_settings(args)

    train_dataloader, val_dataloader = data.get_train_val_dataloaders(settings)
    model = getattr(networks, settings['model_type'])(settings).to(settings['device'])
    if settings['train_new_model']:
        trainer = getattr(training, f"{settings['model_type']}Trainer")(
            model, train_dataloader, val_dataloader, settings)
        trainer.train()
    model.load_self(name_suffix='best_val')
    test_model.test_model(model, settings)
    print('Done!')

if __name__ == "__main__":
    main()
