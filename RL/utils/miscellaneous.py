import torch
import numpy as np


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def save_best_model(current_loss, best_loss, policy, critic, file_name="models/best_model.pt"):
    if current_loss < best_loss:
        best_loss = current_loss
        torch.save({
            'policy_state_dict': policy.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'best_avg_reward': best_loss
        }, file_name)

    return best_loss


def load_best_model(file_name, policy, critic):
    checkpoint = torch.load(file_name)

    policy.load_state_dict(checkpoint['policy_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])

    #best_avg_reward = checkpoint['best_avg_reward']
    #return best_avg_reward
