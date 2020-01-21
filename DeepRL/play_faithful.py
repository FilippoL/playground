import datetime
import os
import random
from collections import deque

import time

import gym
import numpy as np
import tensorflow as tf
import psutil

from utils import preprocess, collect_experience_turtle
from model import create_model_faithful


import linecache
import os
import tracemalloc


collect_experience = collect_experience_turtle

process = psutil.Process(os.getpid())

# env = gym.make('BreakoutDeterministic-v4')
env = gym.make('Assault-v0')

now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# MODEL_PATH = "file:///C:/Users/ohund/workspace/playground/DeepRL/models/20200115-213917/checkpoint"
MODEL_PATH = "models/20200120-053048"
latest = tf.train.latest_checkpoint(MODEL_PATH)
print(f"Loading model from {latest}")

# file_writer_qs = tf.summary.create_file_writer(log_dir + "/qs")
# file_writer.set_as_default()

action_meanings = env.unwrapped.get_action_meanings()
discount_rate = 0.8
action_space = env.action_space.n
time_channels_size = 4
skip_frames = 1
input_shape = list(np.array(env.observation_space.shape) // 2)[:2] + [time_channels_size]
state_shape = list(np.zeros(input_shape).shape)[:2] + [time_channels_size+1]
batch_size = 200
N = batch_size*4
n_episode = 1000
q_mask_shape = (batch_size, action_space)
action_meanings = env.unwrapped.get_action_meanings()

print(f"Pixel space of the game {input_shape}")


# def loss_function(next_qvalues, init_qvalues):
#     init_q = tf.reduce_max(init_qvalues, axis=1)
#     next_qvalues = tf.transpose(next_qvalues)
#     difference = tf.subtract(tf.transpose(init_q), next_qvalues)
#     return tf.square(difference)


approximator_model = create_model_faithful(input_shape, action_space)
approximator_model.load_weights(latest)

# ===== INITIALISATION ======
frame_cnt = 0
prev_lives = 5
acc_nonzeros = []
acc_actions = []
is_done = False
env.reset()


for episode in range(15):
    start_time = time.time()

    print(f"Running episode {episode}")
    initial_observation = env.reset()
    state = np.repeat(preprocess(initial_observation), time_channels_size+1).reshape(state_shape)
    is_done = False

    # next_state = initial_state.copy()  # To remove all the information of the last episode

    episode_rewards = []
    frame_cnt = 0
    while not is_done:
        # https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        init_mask = tf.ones([1, action_space])
        init_state = state[:, :, :-1]
        q_values = approximator_model.predict([tf.reshape(init_state, [1] + input_shape), init_mask])
        action = np.argmax(q_values)
        # action = env.action_space.sample()
        print(f"Executing action: {action_meanings[action]}")

        state, acc_reward, is_done, frames_of_collected = collect_experience(env, action, state_shape, time_channels_size, skip_frames)
        state[:, :, :-1] = state[:, :, 1:]
        frame_cnt += frames_of_collected
        episode_rewards.append(acc_reward)
        # time.sleep(0.05)
        env.render()

    print(f"Total rewards of episode {episode} are {np.sum(episode_rewards)}")
    time_end = np.round(time.time() - start_time, 2)
    print(f"Running at {np.round(frame_cnt/time_end)} frames per second")


# TODO: [x] Simplify the loss function
# TODO: [x] Apply the reward
# TODO: [x] Rethink memory handling
# TODO: [x] Proper memory initialisation
# TODO: [ ] Refactoring and restructuring
# TODO: [ ] Add states to tensorboard for analysis
# TODO: [ ] Write simple model run code
