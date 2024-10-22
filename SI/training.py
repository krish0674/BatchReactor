# 3rd party imports
import torch as T
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self,model, train_dataloader, val_dataloader, settings):
        self.device = settings['device']
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.settings = settings

        self.optimizer = settings['optimizer'](
            self.model.parameters(), lr=settings['learning_rate'])
        self.loss_function = settings['loss_function']()

        self.max_epochs = settings['max_epochs']
        self.validate_every_n_epochs = settings['validate_every_n_epochs']
        self.print_every_n_epochs = settings['print_every_n_epochs']
        self.early_stopping_patience = settings['early_stopping_patience']

        self.logger = SummaryWriter(log_dir=settings['logdir'])

        self.best_val_loss = np.inf
        self.best_val_epoch = 0
        self.epoch = 0
        self.current_train_loss = np.inf
        self.current_val_loss = np.inf

    def train(self):
        while self.epoch < self.max_epochs:
            self.epoch += 1
            self.reset_epoch_info()
            self.train_one_epoch()

            if self.epoch % self.validate_every_n_epochs == 0:
                self.validate()
            self.log_epoch_info()
            self.print_epoch_info()
            if self.epoch - self.best_val_epoch > self.early_stopping_patience:
                print(f"Early stopping after epoch {self.epoch}.")
                break
        self.logger.close()
        print(f"Best validation loss: {self.best_val_loss} at epoch {self.best_val_epoch}.")
        return None

    def train_one_epoch(self):
        self.model.train()
        for X0, U0, X1 in self.train_dataloader:
            X0, U0, X1 = X0.to(self.device), U0.to(self.device), X1.to(self.device)

            batch_loss = self.get_batch_loss(X0, U0, X1, train_or_val='train')

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
        return None

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        
        X0, U0, X1 = next(iter(self.val_dataloader))
        X0, U0, X1 = X0.to(self.device), U0.to(self.device), X1.to(self.device)

        val_loss = self.get_batch_loss(X0, U0, X1, train_or_val='val').item()

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_epoch = self.epoch
            self.model.save_self(name_suffix='best_val')
        return None

    def print_epoch_info(self):
        if self.epoch==1 or self.epoch%(self.print_every_n_epochs*10) == 0:
            print("")
            print(f"{'Epoch':<6} | {'Best Epoch':<11} {'Best Val Loss':<14} | {'Train Loss':<11} {'Val Loss':<9}")

        if self.epoch==1 or self.epoch%(self.print_every_n_epochs) == 0:
            print(f"{self.epoch:<6} | {self.best_val_epoch:<11} {round(self.best_val_loss,7):<14} | {round(self.current_train_loss,7):<11} {round(self.current_val_loss,7):<9}")
        return None

class MLPTrainer(Trainer):
    def __init__(self, model, train_dataloader, val_dataloader, settings):
        super().__init__(model, train_dataloader, val_dataloader, settings)

    def get_batch_loss(self, X0, U0, X1, train_or_val):
        X1_pred = self.model(X0, U0)
        batch_loss = self.loss_function(X1_pred, X1)
        self.log_batch_loss(batch_loss, train_or_val)
        return batch_loss

    def log_batch_loss(self, batch_loss, train_or_val):
        if train_or_val == 'train':
            self.epoch_info['train_batch_losses'].append(batch_loss.item())
        elif train_or_val == 'val':
            self.epoch_info['val_loss'] = batch_loss.item()
        else:
            raise ValueError(f"train_or_val must be 'train' or 'val', not {train_or_val}.")
        return None

    def reset_epoch_info(self):
        self.epoch_info = {
            'train_batch_losses': [],
            'val_loss': None,
        }
        return None

    def log_epoch_info(self):
        self.current_train_loss = np.mean(self.epoch_info['train_batch_losses'])
        self.logger.add_scalar(
            'Train loss',
            self.current_train_loss,
            self.epoch)

        if self.epoch_info['val_loss'] is not None:
            self.current_val_loss = self.epoch_info['val_loss']
            self.logger.add_scalar(
                'Val loss',
                self.current_val_loss,
                self.epoch)
        self.logger.flush()
        return None

class LinearTrainer(MLPTrainer):
    def __init__(self, model, train_dataloader, val_dataloader, settings):
        super().__init__(model, train_dataloader, val_dataloader, settings)

class KoopmanTrainer(Trainer):
    def __init__(self, model, train_dataloader, val_dataloader, settings):
        super().__init__(model, train_dataloader, val_dataloader, settings)
        self.loss_weights = settings['Koopman_loss_weighting']

    def get_batch_loss(self, X0, U0, X1, train_or_val):
        Z0 = self.model.encode(X0)
        Z1_pred = self.model.predict(Z0, U0)
        Z1 = self.model.encode(X1)

        ae_loss = self.loss_function(self.model.decode(Z0), X0)
        pred_loss = self.loss_function(Z1_pred, Z1)
        comb_loss = self.loss_function(self.model.decode(Z1_pred), X1)
        total_loss = self.loss_weights['ae_loss'] * ae_loss\
                   + self.loss_weights['pred_loss'] * pred_loss\
                   + self.loss_weights['comb_loss'] * comb_loss
        
        self.log_batch_loss(ae_loss, pred_loss, comb_loss,
                            total_loss, train_or_val)
        return total_loss

    def log_batch_loss(self, ae_loss, pred_loss,
                       comb_loss, total_loss, train_or_val):
        if train_or_val == 'train':
            self.epoch_info['train_batch_losses']['ae_loss'].append(ae_loss.item())
            self.epoch_info['train_batch_losses']['pred_loss'].append(pred_loss.item())
            self.epoch_info['train_batch_losses']['comb_loss'].append(comb_loss.item())
            self.epoch_info['train_batch_losses']['total_loss'].append(total_loss.item())
        elif train_or_val == 'val':
            self.epoch_info['val_losses']['ae_loss'] = ae_loss.item()
            self.epoch_info['val_losses']['pred_loss'] = pred_loss.item()
            self.epoch_info['val_losses']['comb_loss'] = comb_loss.item()
            self.epoch_info['val_losses']['total_loss'] = total_loss.item()
        else:
            raise ValueError(f"train_or_val must be 'train' or 'val', not {train_or_val}.")
        return None

    def reset_epoch_info(self):
        self.epoch_info = {
            'train_batch_losses': {
                'ae_loss': [],
                'pred_loss': [],
                'comb_loss': [],
                'total_loss': [],
            },
            'val_losses': {
                'ae_loss': None,
                'pred_loss': None,
                'comb_loss': None,
                'total_loss': None,
            },
        }
        return None

    def log_epoch_info(self):
        self.current_train_loss = np.mean(
            self.epoch_info['train_batch_losses']['total_loss'])
        
        self.logger.add_scalar(
            'Training/Autoencoder loss',
            np.mean(self.epoch_info['train_batch_losses']['ae_loss']),
            self.epoch)
        self.logger.add_scalar(
            'Training/Prediction loss',
            np.mean(self.epoch_info['train_batch_losses']['pred_loss']),
            self.epoch)
        self.logger.add_scalar(
            'Training/Combined loss',
            np.mean(self.epoch_info['train_batch_losses']['comb_loss']),
            self.epoch)
        self.logger.add_scalar(
            'Training/Total loss',
            self.current_train_loss,
            self.epoch)

        if self.epoch_info['val_losses']['total_loss'] is not None:
            self.current_val_loss = self.epoch_info['val_losses']['total_loss']
            self.logger.add_scalar(
                'Validation/Autoencoder loss',
                self.epoch_info['val_losses']['ae_loss'],
                self.epoch)
            self.logger.add_scalar(
                'Validation/Prediction loss',
                self.epoch_info['val_losses']['pred_loss'],
                self.epoch)
            self.logger.add_scalar(
                'Validation/Combined loss',
                self.epoch_info['val_losses']['comb_loss'],
                self.epoch)
            self.logger.add_scalar(
                'Validation/Total loss',
                self.current_val_loss,
                self.epoch)
        return None