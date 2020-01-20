import datetime
import os
import random
from collections import deque

import time

import gym
import numpy as np
import tensorflow as tf
# from tensorflow.keras import models, layers
import psutil

from utils import collect_experience_stored_actions, initialize_memory, play_episode, preprocess, image_grid, plot_to_image, exploration_exponential_decay, exploration_linear_decay, exploration_periodic_decay
import utils
from model import create_model, create_model_faithful

import sampling

process = psutil.Process(os.getpid())


# collect_experience = collect_experience_stored_actions
# take_sample = sampling.prioritized_experience_sampling_3
# take_sample = sampling.uniform_sampling
# take_sample = sampling.random_sampling

# env = gym.make('BreakoutDeterministic-v4')
env = gym.make('BreakoutDeterministic-v4', frameskip=8)

now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

checkpoint_path = os.path.join(
    ".",
    "models",
    now,
    "-{epoch:04d}.ckpt"
)

# checkpoint_dir = os.path.dirname(checkpoint_path)

# # Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

log_dir = os.path.join(
    "logs",
    now,
)
tensorflow_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=5, histogram_freq=1)
file_writer_rewards = tf.summary.create_file_writer(log_dir + "/metrics")


# file_writer_qs = tf.summary.create_file_writer(log_dir + "/qs")
# file_writer.set_as_default()

# D = list()
list_size = 60000
D = deque(maxlen=list_size)
# D = RingBuf(list_size)
discount_rate = 0.99
tau = 0
max_tau = 5000
action_space = env.action_space.n
time_channels_size = 4
skip_frames = 1
input_shape = list(np.array(env.observation_space.shape) // 2)[:2] + [time_channels_size]
state_shape = list(np.zeros(input_shape).shape)[:2] + [time_channels_size+1]
batch_size = 256
N = 4*batch_size
n_episode = 1000
q_mask_shape = (batch_size, action_space)
action_meanings = env.unwrapped.get_action_meanings()

# q_writers = [tf.summary.create_file_writer(log_dir + f"/qs/{meaning}") for meaning in action_meanings]

print(f"Pixel space of the game {input_shape}")

approximator_model = create_model_faithful(input_shape, action_space)
target_model = create_model_faithful(input_shape, action_space)

exploration_base = 1.02
exploration_rate = 1
episodes_per_cycle = 50
minimal_exploration_rate = 0.001

# ===== INITIALISATION ======
frame_cnt = 0
prev_lives = 5

is_done = False
env.reset()
td_err_default = 0
acc_nonzeros = []
acc_actions = []

utils.initialize_memory(env, D, N, time_channels_size)

for episode in range(n_episode):
    start_time = time.time()

    acc_actions = []

    if tau >= max_tau:
        tau = 0
        target_model.set_weights(approximator_model.get_weights())
        print("===> Updated weights")

    # exploration_rate = exploration_exponential_decay(episode, exploration_base)
    exploration_rate = exploration_linear_decay(episode, 100)
    # exploration_rate = exploration_periodic_decay(episode, episodes_per_cycle)

    print(f"Running episode {episode} with exploration rate: {exploration_rate}")
    print(f"Number of states in memory {len(D)}")

    # Train on the experience batch
    print(f"Use Priority Experience Replay Sampling")
    ids, importance = sampling.prioritized_experience_sampling_3(D, batch_size)
    max_td_err = max(np.abs([exp[3] for exp in D]))
    experience_batch = [D[idx] for idx in ids]
    history, nonzero_rewards = utils.train_batch(experience_batch, approximator_model, target_model, action_space, discount_rate, tensorflow_callback)

    # Play an entire episode
    print(f"Start playing an episode {episode}")
    if (episode+1) % 5 == 0:
        stats_actions, stats_reward, stats_qs, frame_cnt, stats_frame = utils.play_episode(
            env, D, time_channels_size, max_tau, exploration_rate, approximator_model, max_td_err, True)
    else:
        stats_actions, stats_reward, stats_qs, frame_cnt, stats_frame = utils.play_episode(
            env, D, time_channels_size, max_tau, exploration_rate, approximator_model, max_td_err, False)

    tau += frame_cnt

    # Wrap up
    loss = history.history.get("loss", [0])[0]
    time_end = np.round(time.time() - start_time, 2)
    memory_usage = process.memory_info().rss
    tmp = random.choice(experience_batch)
    print(f"Memory trace {memory_usage}")
    print(f"Loss of episode {episode} is {loss} and took {time_end} seconds")
    print(f"TOTAL REWARD: {np.sum(stats_reward)}")
    utils.write_stats(file_writer_rewards,
                      stats_qs,
                      episode,
                      tmp,
                      action_meanings,
                      loss,
                      time_end,
                      stats_reward,
                      frame_cnt,
                      exploration_rate,
                      memory_usage,
                      nonzero_rewards,
                      stats_actions,
                      stats_frame)
    utils.save_model(approximator_model, episode, checkpoint_path, 50)

# TODO: [x] Simplify the loss function
# TODO: [x] Apply the reward
# TODO: [x] Rethink memory handling
# TODO: [x] Proper memory initialisation
# TODO: [ ] Refactoring and restructuring
# TODO: [ ] Add states to tensorboard for analysis
# TODO: [ ] Write simple model run code
# https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756