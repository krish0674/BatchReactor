import os
import torch as T
import numpy as np
import argparse

# Local imports
import networks
import test_model
import data
import training

from data import get_train_val_dataloaders 
from data import get_test_dataset,get_test_dataloader1
import torch 
# from cstr import testit

def get_settings(args):
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
            'hidden_layer_sizes':       [int(x) for x in args.MLP_hidden_layer_sizes.split(',')],  # Expecting a comma-separated string
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

        # Process-specific settings
        'CSTR1': {
            'n_states':      2,
            'n_controls':    2,
            'state_names':   ['Tr', 'Tj'],
            'control_names': ['Fc','H'],
            'state_scaling': {
                'min_unscaled': np.array([42.444027359784,35.69237508100001]),
                #Tr,Tj 
                'max_unscaled': np.array([69.39558371041198,48.1877254029999]),
                'min_scaled':   np.array([-1.0, -1.0]),
                'max_scaled':   np.array([1.0, 1.0]),
            },
            'action_scaling': {
                'min_unscaled': np.array([0.23536076041497433,4.0]),
                #H,Fc
                'max_unscaled': np.array([20.0,19.999997463454676]),
                'min_scaled':   np.array([-1.0,-1.0]),
                'max_scaled':   np.array([1.0,1.0]),
            },
            'dataset_name': 'Test_Signal_Train',
            'testset_name': 'Test_Signal_Test',
        }
    }

    process = args.process
    model_type = args.model_type
    settings['models_dir'] = f"./models/{process}/"
    settings['model_name'] = f"{model_type}"
    settings['logdir'] = f"./logs/{process}/{model_type}/"
    settings['data_dir'] = f"./data/{process}/"
    settings['plots_dir'] = f"./plots/{process}/{model_type}/"
    settings['train_data_path'] = args.train_data_path
    settings['test_data_path'] = args.test_data_path

    os.makedirs(settings['models_dir'], exist_ok=True)
    os.makedirs(settings['logdir'], exist_ok=True)
    os.makedirs(settings['plots_dir'], exist_ok=True)
    return settings

def main():
    parser = argparse.ArgumentParser(description='Process model training settings.')
    parser.add_argument('--train_new_model', type=bool, default=True, help='Whether to train a new model')
    parser.add_argument('--process', type=str, default='CSTR1', choices=['CSTR1', 'ASU'], help='Process type')
    parser.add_argument('--model_type', type=str, default='Koopman', choices=['MLP', 'Koopman', 'Linear'], help='Type of model to train')
    parser.add_argument('--accuracy', type=str, default='float32', choices=['float32', 'float64'], help='Floating point precision')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--max_epochs', type=int, default=5000, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--validate_every_n_epochs', type=int, default=1, help='Frequency of validation per number of epochs')
    parser.add_argument('--print_every_n_epochs', type=int, default=10, help='Frequency of printing progress per number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=500, help='Patience for early stopping')
    parser.add_argument('--train_val_ratio', type=float, default=0.8, help='Training to validation data ratio')
    parser.add_argument('--plot_test', type=bool, default=True, help='Whether to plot test results')
    parser.add_argument('--train_data_path', type=str,default='/kaggle/input/variant-h/Opeloop_HFc_TrTj.xlsx', required=True, help='Path to the training data')
    parser.add_argument('--test_data_path', type=str,default='/kaggle/input/si-data/Test_Signal_Data.csv', required=False, help='Path to the testing data')
    parser.add_argument('--MLP_hidden_layer_sizes', type=str, default='10,10', help='Comma-separated list of MLP hidden layer sizes')
    parser.add_argument('--Koopman_latent_dim', type=int, default=8, help='Latent dimension size for Koopman model')
    parser.add_argument('--Koopman_encoder_n_hidden_layers', type=int, default=2, help='Number of hidden layers in Koopman encoder')
    parser.add_argument('--Koopman_comb_loss', type=float, default=0.34, help='Weight for combined loss in Koopman training')
    parser.add_argument('--Koopman_ae_loss', type=float, default=0.33, help='Weight for autoencoder loss in Koopman training')
    parser.add_argument('--Koopman_pred_loss', type=float, default=0.33, help='Weight for prediction loss in Koopman training')

    args = parser.parse_args()
    settings = get_settings(args)

    train_dataloader, val_dataloader = get_train_val_dataloaders(settings)
    
    for X0_batch, U0_batch, X1_batch in train_dataloader:
        print('Train Example:\n')
        print(X0_batch.shape, U0_batch.shape, X1_batch.shape)
        print(X0_batch[3])
        print(X1_batch[3])
        print(U0_batch[3])
        break

    for X0_val, U0_val, X1_val in val_dataloader:
        print('Val Example:\n')
        print(X0_val.shape, U0_val.shape, X1_val.shape)
        print(X0_val[3])
        print(X1_val[3])
        print(U0_val[3])
        break

    # test_dataset = get_test_dataset(settings)

    print('data loading done done')
    print(f"Trained model on {settings['device']} yay")

    model = getattr(networks, settings['model_type'])(settings).to(settings['device'])
    if settings['train_new_model']:
        trainer = getattr(training, f"{settings['model_type']}Trainer")(
            model, train_dataloader, val_dataloader, settings)
        trainer.train()

    # model.load_state_dict(torch.load('/kaggle/working/best_val_model.pth', map_location=settings['device']))

    # test_model.test_model_single(model, settings)
    print('Done!')

    # testit(settings)

if __name__ == "__main__":
    main()
