import matplotlib.pyplot as plt
import numpy as np
from utils.miscellaneous import moving_average

def plot_learning_curve(rewards, window_size):
    plt.plot(rewards)
    plt.plot(moving_average(rewards, window_size))
    plt.title('Learning Curve')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.show()

def plot_gradient_norms(grad_norms_list):
    plt.plot(grad_norms_list)
    #plt.ylim(0.0, 100.0)
    plt.title('Gradient Norms')
    plt.xlabel('Batch')
    plt.ylabel('Norm')
    plt.show()


def plot_cumulative_rewards(cumulative_rewards):
    plt.plot(cumulative_rewards)
    #plt.ylim(-20.0, 10.0)
    plt.title('Cumulative Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.show()

def plot_batch_losses(batch_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(batch_losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Batch Losses')
    plt.show()