# 3rd party imports
import numpy as np
import torch as T

class Predictor(T.nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def ensure_batch_shape(self, x=None, u=None):
        if x is not None and u is not None:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            if u.dim() == 1:
                u = u.unsqueeze(0)
            assert x.shape[0] == u.shape[0], "x and u must have same batch size"
            return x, u
        elif x is not None and u is None:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return x
        elif x is None and u is not None:
            if u.dim() == 1:
                u = u.unsqueeze(0)
            return u

    def save_self(self, path=None, name_suffix=None):
        path = self._get_save_path(path=path, name_suffix=name_suffix)
        T.save(self.state_dict(), path)
        return None

    def load_self(self, path=None, name_suffix=None):
        path = self._get_save_path(path=path, name_suffix=name_suffix)
        self.load_state_dict(T.load(path))
        return None

    def _get_save_path(self, path=None, name_suffix=None):
        assert path is None or name_suffix is None, "Cannot specify both path and name_suffix"
        if path is None and name_suffix is None:
            path = f"{self.settings['models_dir']}{self.settings['model_name']}"
        elif path is None and name_suffix is not None:
            path = f"{self.settings['models_dir']}{self.settings['model_name']}_{name_suffix}"
        return path

class Koopman(Predictor):
    def __init__(self, settings):
        super().__init__(settings)

        # extract relevant settings
        self.latent_dim = settings['Koopman']['latent_dim']
        n_states = settings[settings['process']]['n_states']
        n_controls = settings[settings['process']]['n_controls']
        enc_n_hidden_layers = settings['Koopman']['encoder_n_hidden_layers']
        enc_hidden_layer_sizes = np.linspace(
            settings[settings['process']]['n_states'],
            self.latent_dim,
            enc_n_hidden_layers+2,
            dtype=int)[1:-1].tolist()
        enc_activation = settings['Koopman']['encoder_activation']()

        # encoder
        self.encoder = T.nn.Sequential()
        # input layer
        self.encoder.append(T.nn.Linear(
            in_features = n_states,
            out_features = enc_hidden_layer_sizes[0]))
        self.encoder.append(enc_activation)
        # hidden layers
        for i_layer in range(enc_n_hidden_layers-1):
            self.encoder.append(T.nn.Linear(
                in_features = enc_hidden_layer_sizes[i_layer],
                out_features = enc_hidden_layer_sizes[i_layer+1]))
            self.encoder.append(enc_activation)
        # output layer
        self.encoder.append(T.nn.Linear(
            in_features = enc_hidden_layer_sizes[-1],
            out_features = self.latent_dim))
        
        # Koopman operator
        self.Az = T.nn.Linear(
            in_features = self.latent_dim,
            out_features = self.latent_dim,
            bias=False)
        self.Au = T.nn.Linear(
            in_features = n_controls,
            out_features = self.latent_dim,
            bias=False)

        # decoder
        self.decoder = T.nn.Linear(
            in_features = self.latent_dim,
            out_features = n_states,
            bias=False)

    def encode(self, x):
        '''
        x: state
        z: latent state
        '''
        x = self.ensure_batch_shape(x=x)
        z = self.encoder(x)
        return z

    def decode(self, z):
        '''
        z: latent state
        x: state
        '''
        z = self.ensure_batch_shape(x=z)
        x = self.decoder(z)
        return x
    
    def predict(self, z, u):
        '''
        z: latent state
        u: action
        z_: next latent state
        '''
        z, u = self.ensure_batch_shape(x=z, u=u)
        z_ = self.Az(z) + self.Au(u)
        return z_

    def forward(self, x, u):
        '''
        x: state
        u: action
        z: latent state
        z_: next latent state
        x_: next state
        '''
        x, u = self.ensure_batch_shape(x=x, u=u)
        z = self.encode(x)
        z_ = self.predict(z, u)
        x_ = self.decode(z_)
        return x_

    def multi_step_prediction(self, x0, U):
        '''
        x0: initial state
        U: actions
        X_pred: predicted trajectory
        '''
        assert x0.dim() == 1, "x0 must be 1D, function not implemented for batch data"
        assert U.dim() == 2, "U must be 2D, function not implemented for batch data"
        X_pred = T.zeros([U.shape[0]+1, len(x0)])
        X_pred[0, :] = x0

        z = self.encode(x0)
        for t in range(U.shape[0]):
            z = self.predict(z, U[t, :])
            X_pred[t+1, :] = self.decode(z)
        return X_pred



class Linear(Predictor):
    def __init__(self, settings):
        super().__init__(settings)

        # extract relevant settings
        n_states = settings[settings['process']]['n_states']
        n_controls = settings[settings['process']]['n_controls']

        # create model
        self.linear_layer = T.nn.Linear(
            in_features = n_states + n_controls,
            out_features = n_states)
        initial_guess = T.eye(self.linear_layer.weight.shape[0],
                              self.linear_layer.weight.shape[1])
        self.linear_layer.weight = T.nn.Parameter( # initialize close to identity matrix
            self.linear_layer.weight + initial_guess)

    def forward(self, x, u):
        '''
        x: state
        u: action
        x_: next state
        '''
        x, u = self.ensure_batch_shape(x=x, u=u)
        x_ = T.cat([x, u], dim=1)
        x_ = self.linear_layer(x_)
        return x_
    
    def multi_step_prediction(self, x0, U):
        '''
        x0: initial state
        U: actions
        X_pred: predicted trajectory
        '''
        assert x0.dim() == 1, "x0 must be 1D, function not implemented for batch data"
        assert U.dim() == 2, "U must be 2D, function not implemented for batch data"
        X_pred = T.zeros([U.shape[0]+1, len(x0)])
        X_pred[0, :] = x0

        for t in range(U.shape[0]):
            X_pred[t+1, :] = self.forward(X_pred[t, :], U[t, :])
        return X_pred



class MLP(Predictor):
    def __init__(self, settings):
        super().__init__(settings)

        # extract relevant settings
        n_states = settings[settings['process']]['n_states']
        n_controls = settings[settings['process']]['n_controls']
        hidden_layer_sizes = settings['MLP']['hidden_layer_sizes']
        num_hidden_layers = len(hidden_layer_sizes)
        activation = settings['MLP']['activation']()
        output_activation = settings['MLP']['output_activation']()

        # create model
        self.neural_net = T.nn.Sequential()
        # input layer
        self.neural_net.append(T.nn.Linear(
            in_features = n_states + n_controls,
            out_features = hidden_layer_sizes[0]))
        self.neural_net.append(activation)
        # hidden layers
        for i_layer in range(num_hidden_layers-1):
            self.neural_net.append(T.nn.Linear(
                in_features = hidden_layer_sizes[i_layer],
                out_features = hidden_layer_sizes[i_layer+1]))
            self.neural_net.append(activation)
        # output layer
        self.neural_net.append(T.nn.Linear(
            in_features = hidden_layer_sizes[-1],
            out_features = n_states))
        self.neural_net.append(output_activation)

    def forward(self, x, u):
        '''
        x: state
        u: action
        delta_x: change of state
        '''
        x, u = self.ensure_batch_shape(x=x, u=u)
        delta_x = T.cat([x, u], dim=1)
        delta_x = self.neural_net(delta_x)
        return x + delta_x
    
    def multi_step_prediction(self, x0, U):
        '''
        x0: initial state
        U: actions
        X_pred: predicted trajectory
        '''
        assert x0.dim() == 1, "x0 must be 1D, function not implemented for batch data"
        assert U.dim() == 2, "U must be 2D, function not implemented for batch data"
        X_pred = T.zeros([U.shape[0]+1, len(x0)])
        X_pred[0, :] = x0

        for t in range(U.shape[0]):
            X_pred[t+1, :] = self.forward(X_pred[t, :], U[t, :])
        return X_pred

# Testing
if __name__ == '__main__':
    from main import get_settings
    settings = get_settings()

    # iniitialize models
    settings['model_type'] = 'MLP'
    mlp = MLP(settings)
    settings['model_type'] = 'Koopman'
    koopman = Koopman(settings)

    # generate random batch data
    X0 = T.rand([5, settings[settings['process']]['n_states']])
    U = T.rand([5, settings[settings['process']]['n_controls']])

    # test models on batch data
    X1_mlp = mlp.forward(X0, U)
    X1_koopman = koopman.forward(X0, U)

    # test models on single data
    x0 = X0[0,:]
    u = U[0,:]
    x1_mlp = mlp.forward(x0, u)
    x1_koopman = koopman.forward(x0, u)

    print('done')