import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from matplotlib import pyplot as plt
from timeit import default_timer as timer

# Local imports
from source.critic import Critic
from source.environment import Environment, EnvironmentParameters
from source.memory import Memory

class Agent:
    def __init__(self, num_environments: int, short_horizon_length: int,
                 policy, critic, lr_policy, lr_critic, weight_decay_critic,
                 gamma=0.99, lambda_=0.95, alpha=0.95, exploration_noise=0.1):
        self.policy = policy
        self.critic = critic
        self.target_critic = Critic()
        self.target_critic.load_state_dict(critic.state_dict())
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr_policy)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic,
            weight_decay=weight_decay_critic)
        self.alpha = alpha  # alpha lr i.e. soft update of the target critic network
        self.exploration_noise = exploration_noise
        self.logger = SummaryWriter(log_dir='./logs/', max_queue=1_000, flush_secs=60)

        self.GAMMA  = torch.tensor(gamma)
        self.LAMBDA = torch.tensor(lambda_)

        self.short_horizon_length = short_horizon_length
        self.num_environments = num_environments
        self.initialize_environments()

        tmpObs = self.environments[0].get_observation()
        len_obs     = len(tmpObs)
        len_action  = len(self.policy(tmpObs)[0])
        self.memory = Memory(self.num_environments, self.short_horizon_length, len_obs, len_action)

        self.best_avg_reward = -float('inf')
        self.best_test_reward = -float('inf')
        self.n_steps = 0

    def initialize_environments(self):
        # Initialize multiple instances of an Environment class.
        env_params = EnvironmentParameters()
        self.initial_steps = torch.randint(7200, size=(self.num_environments,))  # Assuming max_step is 7200
        self.environments = [Environment(env_params, i, 7200) for i in self.initial_steps]
        self.val_environment = Environment(env_params, 0, 7200)
        self.test_environment = Environment(env_params, 0, 7200)

    def unroll_environments(self):
        self.memory.clear()

        for ienv, env in enumerate(self.environments):
            env.clear_gradients()
            observation = env.get_observation()

            for istep in range(self.short_horizon_length):
                action = self.policy(observation, noise=self.exploration_noise)
                next_obs, reward = env.step(action[0])

                self.memory.store(ienv, istep, observation, next_obs, action, reward, 0)  # No penalty in this case
                observation = next_obs
        self.n_steps += self.num_environments * self.short_horizon_length
    
        self.logger.add_scalar(f'training/avg_reward',
                               np.mean(self.memory.rewards.detach().numpy()),
                               self.n_steps)

    def calculate_policy_loss(self):
        target_values = self.target_critic(self.memory.next_obs[:, -1, :]).squeeze(dim=1)

        gamma_vec = self.GAMMA ** torch.arange(self.short_horizon_length)
        discounted_rewards = torch.sum(gamma_vec * self.memory.rewards.squeeze(dim=2), dim=1)
        discounted_rewards = discounted_rewards + self.GAMMA ** self.short_horizon_length * target_values

        policy_loss = -torch.sum(discounted_rewards) / (self.short_horizon_length * self.num_environments)
        self.logger.add_scalar(f'policy/loss', policy_loss, self.n_steps)
        return policy_loss

    def calculate_critic_loss(self):
        V = self.critic(self.memory.observations.detach().reshape(-1,self.memory.observations.shape[2]))
        V = V.reshape(self.memory.observations.shape[0], self.memory.observations.shape[1])
        V_est = self.compute_critic_target_values()

        critic_loss = F.mse_loss(V, V_est)
        self.logger.add_scalar(f'critic/loss', critic_loss, self.n_steps)
        return critic_loss

    @torch.no_grad()
    def compute_critic_target_values(self):
        target_values = torch.zeros((self.num_environments, self.short_horizon_length), dtype = torch.float32)
        Ai = torch.zeros(self.num_environments, dtype = torch.float32)
        Bi = torch.zeros(self.num_environments, dtype = torch.float32)
        lam = torch.ones(self.num_environments, dtype = torch.float32)
        done_mask = torch.zeros(self.short_horizon_length, dtype = torch.float32) # adapt if env can terminate
        done_mask[-1] = 1.0
        next_values = self.target_critic(self.memory.next_obs.detach().reshape(-1,self.memory.next_obs.shape[2]))
        next_values = next_values.reshape(self.memory.next_obs.shape[0], self.memory.next_obs.shape[1])
        rewards = self.memory.rewards.detach().squeeze(dim=2)

        for i in reversed(range(self.short_horizon_length)):
            lam = lam * self.LAMBDA * (1. - done_mask[i]) + done_mask[i]
            Ai = (1.0 - done_mask[i]) * (self.LAMBDA * self.GAMMA * Ai + self.GAMMA * next_values[:,i] + (1. - lam) / (1. - the LAMBDA) * rewards[:,i])
            Bi = self.GAMMA * (next_values[:,i] * done_mask[i] + Bi * (1.0 - done_mask[i])) + rewards[:,i]
            target_values[:,i] = (1.0 - self.LAMBDA) * Ai + lam * Bi

        return target_values

    def log_grad_norm(self, parameters, module_name):
        total_norm = 0
        for p in parameters:
            if p.grad is not None and p.requires_grad:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.logger.add_scalar(f'{module_name}/grad_norm', total_norm, self.n_steps)

    def train_policy(self, timesteps_total=10000):
        self.policy.train()
        self.critic.train()

        learning_episode = 0
        while self.n_steps < timesteps_total:
            starttime = timer()

            # Sample N short-horizon trajectories
            self.unroll_environments()

            # Check for best avg training reward and save the best model
            current_avg_reward = np.mean(self.memory.rewards.detach().numpy())
            if current_avg_reward > self.best_avg_reward:
                self.best_avg_reward = current_avg_reward
                self.save_agent(file_name='best_training_episode_agent')

            # Update policy
            policy_loss = self.calculate_policy_loss()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.log_grad_norm(self.policy.parameters(), 'policy')
            if learning_episode >= 0:
                clip_grad_value_(self.policy.parameters(), 1.0)
                self.policy_optimizer.step()

            # Update critic
            critic_loss = self.calculate_critic_loss()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.log_grad_norm(self.critic.parameters(), 'critic')
            clip_grad_value_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            # Soft update target critic
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.alpha * target_param.data + (1.0 - self.alpha) * param.data)
            learning_episode += 1

            time = timer() - starttime
            self.logger.add_scalar(
                'training/fps',
                self.num_environments * self.short_horizon_length / time,
                self.n_steps)
            self.logger.add_scalar(
                'training/updates',
                learning_episode,
                self.n_steps)

            if learning_episode % 100 == 0:
                self.save_agent(file_name='last_agent')

    def save_agent(self, directory='./models/CurrentRun', file_name='agent'):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
        }, f"{directory}/{file_name}.pt")

    def load_agent(self, directory='./models', file_name='agent'):
        checkpoint = torch.load(f"{directory}/{file_name}.pt")
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
