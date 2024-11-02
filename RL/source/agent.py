# 3rd party imports
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
        # initialize multiple instances of an Env. class.
        # Each env. has its own unique initial step but shares the same params and price data.
        env_params = EnvironmentParameters()

        prices, max_step   = Environment.load_prices(
            env_params.price_prediction_horizon,
            train_or_test='train')
        self.initial_steps = torch.randint(max_step, size=(self.num_environments,))
        self.environments  = [Environment(env_params, prices, i, max_step) for i in self.initial_steps]
        self.val_environment = Environment(env_params, prices, 0, max_step)

        prices, max_step   = Environment.load_prices(
            env_params.price_prediction_horizon,
            train_or_test='test')
        self.test_environment = Environment(env_params, prices, 0, max_step)

    def unroll_environments(self):
        self.memory.clear()

        for ienv, env in enumerate(self.environments):
            env.clear_gradients()
            observation = env.get_observation()

            for istep in range(self.short_horizon_length):
                action = self.policy(observation, noise=self.exploration_noise)
                next_obs, reward, penalty = env.step(action[0])

                self.memory.store(ienv, istep, observation, next_obs, action, reward, penalty)
                observation = next_obs
        self.n_steps += self.num_environments * self.short_horizon_length
    
        self.logger.add_scalar(f'training/avg_reward',
                               np.mean(self.memory.rewards.detach().numpy()),
                               self.n_steps)
        self.logger.add_scalar(f'training/avg_penalty',
                               np.mean(self.memory.penalties.detach().numpy()),
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
            Ai = (1.0 - done_mask[i]) * (self.LAMBDA * self.GAMMA * Ai + self.GAMMA * next_values[:,i] + (1. - lam) / (1. - self.LAMBDA) * rewards[:,i])
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
                # Test policy
                test_reward = self.test_policy(
                    num_episodes=1, episode_length=int(24*30),
                    train_or_test_prices='train',
                    plot=False, print_results=True)
                if test_reward > self.best_test_reward:
                    self.best_test_reward = test_reward
                    self.save_agent(file_name='best_val_agent')

    @torch.no_grad()
    def test_policy(self, num_episodes=100, episode_length=32,
                    train_or_test_prices='train',
                    plot=False, print_results=True, plot_name_prefix=None):
        if train_or_test_prices == 'train':
            env = self.val_environment
        elif train_or_test_prices == 'test':
            env = self.test_environment
        else:
            raise ValueError("train_or_test_prices must be either 'train' or 'test'.")

        observations = torch.zeros((num_episodes, episode_length+1, len(env.get_observation())))
        prices       = torch.zeros((num_episodes, episode_length+1))
        actions      = torch.zeros((num_episodes, episode_length, len(self.policy(env.get_observation())[0])))
        rewards      = torch.zeros((num_episodes, episode_length))
        penalties    = torch.zeros((num_episodes, episode_length))

        # loop through episodes
        for i_ep in range(num_episodes):
            env.reset()
            observation = env.get_observation()
            observations[i_ep, 0, :] = observation
            prices[i_ep, 0] = env.current_price
            for i_step in range(episode_length):
                action = self.policy(observation, noise=0.0)
                observation, reward, penalty = env.step(action[0])

                observations[i_ep, i_step+1, :] = observation
                prices[i_ep, i_step+1]          = env.current_price
                actions[i_ep, i_step, :]        = action
                rewards[i_ep, i_step]           = reward
                penalties[i_ep, i_step]         = penalty

        self.log_test_results(observations, prices, actions, rewards, penalties)
        if plot:
            self.plot_test_results(
                observations, prices, actions, rewards, penalties, plot_name_prefix)
        if print_results:
            self.print_test_results(prices, actions, rewards, penalties)

        return rewards.mean().item()

    def log_test_results(self, observations, prices, actions, rewards, penalties):
        penalty_rate = np.count_nonzero(penalties.numpy()) / (penalties.shape[0] * penalties.shape[1])
        prices       = prices.numpy()
        coolant_flow = actions[:, :, 1].numpy()
        costs       = np.sum(coolant_flow * prices[:, :-1], axis=1)
        costs_ss    = np.sum(390.0 / (60*60) * prices[:, :-1], axis=1)
        rel_costs   = costs / costs_ss

        self.logger.add_scalar(f'test/avg_reward', np.mean(rewards.numpy()), self.n_steps)
        self.logger.add_scalar(f'test/avg_penalty', np.mean(penalties.numpy()), self.n_steps)
        self.logger.add_scalar(f'test/penalty_rate', penalty_rate, self.n_steps)
        self.logger.add_scalar(f'test/avg_rel_cost', np.mean(rel_costs), self.n_steps)

    @torch.no_grad()
    def plot_test_results(self, observations, prices, actions, rewards, penalties, plot_name_prefix):
        plot_name_prefix = ''\
            if plot_name_prefix == None\
            else plot_name_prefix + '_'

        n_episodes   = observations.shape[0]
        c            = observations[:, :, 0].numpy()
        T            = observations[:, :, 1].numpy()
        storage      = observations[:, :, 2].numpy()
        prices       = prices.numpy()
        rho          = actions[:, :, 0].numpy()
        coolant_flow = actions[:, :, 1].numpy()
        rewards      = rewards.numpy()
        penalties    = penalties.numpy()
        t            = np.arange(observations.shape[1])

        c_lb, c_ub = 0.9 * 0.1367, 1.1 * 0.1367
        T_lb, T_ub = 0.8*0.7293, 1.2*0.7293
        rho_lb, rho_ub = 0.8 / (60*60), 1.2 / (60*60)
        coolant_flow_lb, coolant_flow_ub = 0.0, 700 / (60*60)
        coolant_flow_ss = 390.0 / (60*60)
        storage_lb, storage_ub = 0.0, 6.0

        for i_ep in range(n_episodes):
            cost = np.sum(coolant_flow[i_ep, :] * prices[i_ep, :-1])
            cost_ss = np.sum(coolant_flow_ss * prices[i_ep, :-1])
            rel_cost = cost / cost_ss
            mean_penalty = np.mean(penalties[i_ep, :])
            penalty_rate = np.count_nonzero(penalties[i_ep, :]) / penalties.shape[1]
            figtitle = f'Relative cost: {rel_cost:.3f}, Mean penalty: {mean_penalty:.3f}, Penalty rate: {penalty_rate*100:.1f} %'

            fig, ax = plt.subplots(7, 1, figsize=(12, 12))
            fig.suptitle(figtitle)

            ax[0].plot(t, c[i_ep, :])
            ax[0].plot(t, c_lb * np.ones_like(t), 'b--')
            ax[0].plot(t, c_ub * np.ones_like(t), 'b--')
            ax[0].set_ylabel('c')
            ax[1].plot(t, T[i_ep, :])
            ax[1].plot(t, T_lb * np.ones_like(t), 'b--')
            ax[1].plot(t, T_ub * np.ones_like(t), 'b--')
            ax[1].set_ylabel('T')
            ax[2].plot(t, storage[i_ep, :])
            ax[2].plot(t, storage_lb * np.ones_like(t), 'b--')
            ax[2].plot(t, storage_ub * np.ones_like(t), 'b--')
            ax[2].set_ylabel('storage')
            ax[3].plot(t[:-1], rho[i_ep, :])
            ax[3].plot(t[:-1], rho_lb * np.ones_like(t[:-1]), 'b--')
            ax[3].plot(t[:-1], rho_ub * np.ones_like(t[:-1]), 'b--')
            ax[3].set_ylabel('rho')
            ax[4].plot(t[:-1], coolant_flow[i_ep, :])
            ax[4].plot(t[:-1], coolant_flow_lb * np.ones_like(t[:-1]), 'b--')
            ax[4].plot(t[:-1], coolant_flow_ub * np.ones_like(t[:-1]), 'b--')
            ax[4].set_ylabel('coolant flow')
            ax41 = ax[4].twinx()
            ax41.plot(t[:-1], prices[i_ep, :-1], 'r')
            ax41.set_ylabel('price')
            ax[5].plot(t[:-1], rewards[i_ep, :])
            ax[5].set_ylabel('reward')
            ax[6].plot(t[:-1], penalties[i_ep, :])
            ax[6].set_ylabel('penalty')
            ax[6].set_xlabel('time step')

            fig.tight_layout()
            fig.savefig(f"./plots/test_trajectories/{plot_name_prefix}{i_ep}.pdf")
            plt.close(fig)

    @torch.no_grad()
    def print_test_results(self, prices, actions, rewards, penalties):
        prices       = prices.numpy()
        coolant_flow = actions[:, :, 1].numpy()
        costs        = np.sum(coolant_flow * prices[:, :-1], axis=1)
        costs_ss     = np.sum(390.0 / (60*60) * prices[:, :-1], axis=1)
        rel_costs    = costs / costs_ss

        print("")
        print(f"Test results after {self.n_steps} timesteps over {rewards.shape[0]} episodes, each of length {rewards.shape[1]}:")
        print(f"Mean reward:        {np.mean(rewards.numpy()):.3f}")
        print(f"Mean penalty:       {np.mean(penalties.numpy()):.3f}")
        penalty_rate = np.count_nonzero(penalties.numpy()) / (penalties.shape[0] * penalties.shape[1])
        print(f"Penalty rate:       {penalty_rate*100:.2f} %")
        print(f"Mean relative cost: {np.mean(rel_costs):.3f}")
        print("")

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