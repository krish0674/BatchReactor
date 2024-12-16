import gym
from gym import spaces
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import scipy
from scipy.integrate import odeint
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
from actor import MPC_Policy

file_path = 'Test_Signal_Data.csv'  
data = pd.read_csv(file_path)

# Features and Target
X = data[['Tr', 'Tj']]
y = data['Fc']

os.environ["KERAS_BACKEND"] = "tensorflow"
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random

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


def unscaleaction(self, x):
        x = (x - 0.125) / 0.8732
        x = x * 0.8732 + 0.125 
        return x

# Define Batch Reactor model and returns the rates of change of various state variables
def br(x, t, u, Ad):
    u=unscaleaction(u)
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
        Tr_ref = pd.read_csv('Trajectory2.csv')
        self.v1 = Tr_ref.values
        self.a1 = self.v1.tolist()
        self.sp = self.a1[self.i][0]

        self.I = 4.5e-3
        self.M = 0.7034
        self.Tr = 45.0
        self.Tj = 40.0
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
        if error <= 0.5: #and we need to change how much reward we assign
            self.reward = 100
        elif error <= 1:
            self.reward = 50
        else:
            self.reward = -error

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
        # if network_mode==0: #A2C mode
        #     self.state = self.Tr, self.sp #how do i make this dynamic, so i can use different features for my predictions?
        # if network_mode==1:
        self.state = self.Tr, self.Tj

        self.y0 = np.empty(4)
        self.y0[0] = self.I
        self.y0[1] = self.M
        self.y0[2] = self.Tr
        self.y0[3] = self.Tj

        self.sp_values = []
        self.Tr_values = []

        return self.state


# Define the DDPG agent
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

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        return state_batch, action_batch, reward_batch, next_state_batch

    def clear(self):
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, 2))
        self.action_buffer = np.zeros((self.buffer_capacity, 1))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, 2))


def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def get_actor():
    # last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    # inputs = layers.Input(shape=(2,))
    # out = layers.Dense(256, activation="relu")(inputs)
    # out = layers.Dense(256, activation="relu")(out)
    # outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # model = tf.keras.Model(inputs, outputs)
    model=MPC_Policy()
    return model


def get_critic():
    state_input = layers.Input(shape=(2,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    action_input = layers.Input(shape=(1,))
    action_out = layers.Dense(32, activation="relu")(action_input)

    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    model = tf.keras.Model([state_input, action_input], outputs)
    return model

def scale_state(self, state):
        # scale state from [min_state, max_state] to [-1, 1]
        state = (state - [34.946917, 30.485979]) / ([90.839534,60.022752] -[34.946917, 30.485979])
        state = 2.0 * state - 1.0
        return state

def policy(state, noise_object, actor_model):
    state=scale_state(state)
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    sampled_actions = sampled_actions.numpy() + noise
    legal_action = np.clip(sampled_actions, 0.25, 0.75)
    return [np.squeeze(legal_action)]


# Training hyperparameters
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 10
gamma = 0.99
tau = 0.005
b = 0.5  #action_weight

buffer = Buffer(50000, 64)


def ddpg(env):
    ep_reward_list = []
    avg_reward_list = []

    for ep in range(total_episodes):
        prev_state = env.reset()
        # prev_state_MNN = env.reset(1)
        episodic_reward = 0

        while True:
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            # tf_prev_state_MNN = tf.expand_dims(tf.convert_to_tensor(prev_state_MNN), 0)

            action = np.array(policy(tf_prev_state, ou_noise, actor_model)) #need to add MNN action and weighted sum it here
            # mnn_action = np.array(mnn.predict(tf_prev_state_MNN)) #dont know if this is going to work yet

            # action=b*mnn_action+(1-b)*actor_action

            state, reward, done, info = env.step(action)

            #need to define the error function in step, not here

            buffer.record((prev_state, action, reward, state))

            episodic_reward += reward #reward changes

            buffer.clear()

            prev_state = state

            if done:
                break

            if buffer.buffer_counter > buffer.batch_size:
                state_batch, action_batch, reward_batch, next_state_batch = buffer.sample()

                with tf.GradientTape() as tape:
                    target_actions = target_actor(next_state_batch, training=True)
                    y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True)
                    critic_value = critic_model([state_batch, action_batch], training=True)
                    critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

                critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
                critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

                with tf.GradientTape() as tape:
                    actions = actor_model(state_batch, training=True)
                    critic_value = critic_model([state_batch, actions], training=True)
                    actor_loss = -tf.math.reduce_mean(critic_value)

                actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
                actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

                update_target(target_actor.variables, actor_model.variables, tau)
                update_target(target_critic.variables, critic_model.variables, tau)

        ep_reward_list.append(episodic_reward)

        avg_reward = np.mean(ep_reward_list[-40:])
        avg_reward_list.append(avg_reward)
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))

    return ep_reward_list, avg_reward_list


env = BR3()
ep_reward_list, avg_reward_list = ddpg(env)

plt.figure(figsize=(12, 6))
plt.plot(env.sp_values, label='Setpoint')
plt.plot(env.Tr_values, label='Reactor Temperature (Tr)')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.legend()
plt.title('Setpoint vs. Reactor Temperature')
plt.show()