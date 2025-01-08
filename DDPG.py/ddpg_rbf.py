import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import scipy
from scipy.integrate import odeint
import os
import matplotlib.pyplot as plt
import csv
import random
from actor import RBFNN
import math 
# Load data
file_path = r'C:\Users\KRISH DIDWANIA\BatchReactor-5\DDPG.py\Test_Signal_Data.csv'
data = pd.read_csv(file_path)

# Features and Target
X = data[['Tr', 'Tj']]
y = data['Fc']

# Define constants
Ad = 4.4e16
Ed = 140.06e3
Ap = 1.7e11 / 60
Ep = 16.9e3 / 0.239
deltaHp = -82.2e3
UA = 33.3083
Qc = 650
Qs = 12.41e-2
V = 0.5
Tc = 27
Tamb = 27
Cpc = 4.184
R = 8.3145
alpha = 1.212827
beta = 0.000267
epsilon = 0.5
theta = 1.25
m1 = 450
cp1 = 4.184
mjCpj = (18 * 4.184) + (240 * 0.49)
cp2 = 187
cp3 = 110.58
cp4 = 84.95
m5 = 220
cp5 = 0.49
m6 = 7900
cp6 = 0.49
M0 = 0.7034
I0 = 4.5e-3

# Define Batch Reactor model
def br(x, t, u, Ad):
    F = u * 16.667
    Ii = x[0]
    M = x[1]
    Tr = x[2]
    Tj = x[3]

    Ri = Ad * Ii * (np.exp(-Ed / (R * (Tr + 273.15))))
    Rp = Ap * (Ii ** epsilon) * (M ** theta) * (np.exp(-Ep / (R * (Tr + 273.15))))
    mrCpr = m1 * cp1 + Ii * cp2 * V + M * cp3 * V + (M0 - M) * cp4 * V + m5 * cp5 + m6 * cp6
    Qpr = alpha * (Tr - Tc) ** beta

    dy1_dt = -Ri
    dy2_dt = -Rp
    dy3_dt = (Rp * V * (-deltaHp) - UA * (Tr - Tj) + Qc + Qs - Qpr) / mrCpr
    dy4_dt = (UA * (Tr - Tj) - F * Cpc * (Tj - Tc)) / mjCpj

    xdot = np.zeros(4)
    xdot[0] = dy1_dt
    xdot[1] = dy2_dt
    xdot[2] = dy3_dt
    xdot[3] = dy4_dt

    return xdot

class BR3(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=0.25, high=0.75, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        self.t = np.linspace(0, 7200, 7201)
        self.i = 0
        Tr_ref = pd.read_csv(r'C:\Users\KRISH DIDWANIA\BatchReactor-5\DDPG.py\Trajectory2.csv')
        self.v1 = Tr_ref.values
        self.a1 = self.v1.tolist()
        self.sp = self.a1[self.i][0]

        self.I = 4.5e-3
        self.M = 0.7034
        self.Tr = 45.0
        self.Tj = 40.0
        self.Fc=0.5
        self.state = self.Tr, self.sp

        self.y0 = np.empty(4)
        self.y0[0] = self.I
        self.y0[1] = self.M
        self.y0[2] = self.Tr
        self.y0[3] = self.Tj
        self.time_step = 7200

        self.sp_values = []
        self.Tr_values = []

    def step(self, action):
        action = action[0]
        u = action

        ts = [self.t[self.i], self.t[self.i + 1]]
        y = scipy.integrate.odeint(br, self.y0, ts, args=(u, 4.4e16))
        x = np.round(y, decimals=4)

        self.I = x[-1][0]
        self.M = x[-1][1]
        self.Tr = x[-1][2]
        self.Tj = x[-1][3]

        self.y0 = np.empty(4)
        self.y0[0] = self.I
        self.y0[1] = self.M
        self.y0[2] = self.Tr
        self.y0[3] = self.Tj

        data = [self.sp, self.Tr, self.Tj, action]
        with open('data2.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(data)

        self.sp = self.a1[self.i][0]
        self.i = self.i + 1
        #print(f"setpoint:{self.sp}", f"TR:{self.Tr}", f"TJ:{self.Tj}", f"action:{action}", f"step:{self.i}")

        difference = self.sp - self.Tr
        self.reward = 0
        error = abs(difference) #we need to change this
        # if error <= 0.5: #and we need to change how much reward we assign
        #     self.reward = 100
        # elif error <= 1:
        #     self.reward = 50
        # else:
        self.reward = 50 * math.exp(-error)

        self.sp_values.append(self.sp)
        self.Tr_values.append(self.Tr)

        if self.i >= self.time_step:
            done = True
        else:
            done = False

        info = {}
        self.state = self.Tr, self.sp
        return self.state, self.reward, done, info

    def reset(self):
        self.I = 4.5e-3
        self.M = 0.7034
        self.Tr = 45.0
        self.Tj = 400
        self.i = 0
        self.sp = self.a1[self.i][0]
        self.Fc=0.5
        self.state = self.Tr, self.Tj

        self.y0 = np.empty(4)
        self.y0[0] = self.I
        self.y0[1] = self.M
        self.y0[2] = self.Tr
        self.y0[3] = self.Tj

        self.sp_values = []
        self.Tr_values = []

        return self.state,self.Fc
    
# Custom Ornstein-Uhlenbeck noise for exploration
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + self.std_dev * np.sqrt(
            self.dt) * np.random.normal(size=self.mean.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)

# Buffer for experience replay
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, 2))
        self.action_buffer = np.zeros((self.buffer_capacity, 1))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, 2))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def sample(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = torch.tensor(self.state_buffer[batch_indices], dtype=torch.float32)
        action_batch = torch.tensor(self.action_buffer[batch_indices], dtype=torch.float32)
        reward_batch = torch.tensor(self.reward_buffer[batch_indices], dtype=torch.float32)
        next_state_batch = torch.tensor(self.next_state_buffer[batch_indices], dtype=torch.float32)

        return state_batch, action_batch, reward_batch, next_state_batch

    def clear(self):
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, 2))
        self.action_buffer = np.zeros((self.buffer_capacity, 1))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, 2))

# Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_fc = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        self.action_fc = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.ReLU()
        )
        self.concat_fc = nn.Sequential(
            nn.Linear(32 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        state_out = self.state_fc(state)
        action_out = self.action_fc(action)
        concat = torch.cat([state_out, action_out], dim=1)
        return self.concat_fc(concat)

# Define actor
def get_actor():
    centers_1 = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5], [1, 0.5], [0.5, 1], [1, 1]]

    model = RBFNN(centers_1=centers_1, sigma=0.5)

    model.load_state_dict(torch.load(r'C:\Users\KRISH DIDWANIA\BatchReactor-5\DDPG.py\best_model (6).pth'))
    return model

# Define critic
def get_critic():
    return Critic(state_dim=2, action_dim=1)

# Policy function
def policy(state, noise_object, actor_model):
    state = torch.tensor(state, dtype=torch.float32)
    action = actor_model(state)
    noise = torch.tensor(noise_object(), dtype=torch.float32)
    action = action.detach().numpy() + noise.numpy()
    return np.clip(action, 0.25, 0.75)

# Training setup
actor_model = get_actor()
critic_model = get_critic()
target_actor = get_actor()
target_critic = get_critic()

target_actor.load_state_dict(actor_model.state_dict())
target_critic.load_state_dict(critic_model.state_dict())

critic_optimizer = optim.Adam(critic_model.parameters(), lr=0.002)
actor_optimizer = optim.Adam(actor_model.parameters(), lr=0.001)
tau = 0.005
gamma = 0.99

buffer = Buffer(50000, 64)

# Update target networks
def update_target(target_model, model, tau):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

import torch
import numpy as np
import matplotlib.pyplot as plt

total_episodes=5
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))


def ddpg(env):
    ep_reward_list = []
    avg_reward_list = []

    for ep in range(total_episodes):
        prev_state, prev_action = env.reset()
        episodic_reward = 0
        print("hi")
        while True:
            # Convert state to PyTorch tensor
            torch_prev_state = torch.tensor(prev_state, dtype=torch.float32).unsqueeze(0)
            action = policy(torch_prev_state, ou_noise, actor_model)

            # Execute action in environment
            state, reward, done, info = env.step(action)

            # Record experience in buffer
            buffer.record((prev_state, action, reward, state))

            episodic_reward += reward

            prev_state = state
            prev_action = action

            if done:
                break

            # Update models if buffer is sufficiently filled
            if buffer.buffer_counter > buffer.batch_size:
                state_batch, action_batch, reward_batch, next_state_batch = buffer.sample()

                # Critic update
                with torch.no_grad():
                    target_actions = target_actor(next_state_batch)
                    y = reward_batch + gamma * target_critic(next_state_batch, target_actions)

                critic_optimizer.zero_grad()
                critic_value = critic_model(state_batch, action_batch)
                critic_loss = torch.nn.functional.mse_loss(critic_value, y)
                critic_loss.backward()
                critic_optimizer.step()

                # Actor update
                actor_optimizer.zero_grad()
                actions = actor_model(state_batch)
                actor_loss = -target_critic(state_batch, actions).mean()
                actor_loss.backward()
                actor_optimizer.step()

                # Update target networks
                update_target(target_actor, actor_model, tau)
                update_target(target_critic, critic_model, tau)

        ep_reward_list.append(episodic_reward)

        avg_reward = np.mean(ep_reward_list[-40:])
        avg_reward_list.append(avg_reward)
        print(f"Episode * {ep} * Avg Reward is ==> {avg_reward}")

    return ep_reward_list, avg_reward_list

# Run training
env = BR3()
ep_reward_list, avg_reward_list = ddpg(env)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(env.sp_values, label='Setpoint')
plt.plot(env.Tr_values, label='Reactor Temperature (Tr)')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.legend()
plt.title('Setpoint vs. Reactor Temperature')
plt.show()
