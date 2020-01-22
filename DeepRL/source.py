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

from helper import collect_experience_hidden_action, preprocess
from model import create_model, create_model_faithful
import helper as utils
import sampling

process = psutil.Process(os.getpid())


collect_experience = collect_experience_hidden_action
take_sample = sampling.prioritized_experience_sampling
# take_sample = sampling.uniform_sampling
# take_sample = sampling.random_sampling

# env = gym.make('BreakoutDeterministic-v4')
frame_skip = 4
env = gym.make('Assault-v4', frameskip=frame_skip)

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
file_writer_qs = tf.summary.create_file_writer(log_dir + "/metrics")

# file_writer_qs = tf.summary.create_file_writer(log_dir + "/qs")
# file_writer.set_as_default()

# D = list()
list_size = 10000
D = deque(maxlen=list_size)
# D = RingBuf(list_size)
discount_rate = 0.99
tau = 0
max_tau = 2000
action_space = env.action_space.n
action_meanings = env.unwrapped.get_action_meanings()
time_channels_size = 4
skip_frames = 1
input_shape = list(np.array(env.observation_space.shape) // 2)[:2] + [time_channels_size]
state_shape = list(np.zeros(input_shape).shape)[:2] + [time_channels_size+1]
batch_size = 250
N = batch_size
n_episode = 1000
q_mask_shape = (batch_size, action_space)
save_freq = 50
print(f"Pixel space of the game {input_shape}")

approximator_model = create_model_faithful(input_shape, action_space)
target_model = create_model_faithful(input_shape, action_space)

exploration_base = 1.02
exploration_rate = 1
episodes_per_cycle = 50
minimal_exploration_rate = 0.001

# ===== INITIALISATION ======
frame_cnt = 0
prev_lives = env.unwrapped.ale.lives()
is_done = False
env.reset()
td_err_default = 0
acc_nonzeros = []
acc_actions = []

utils.initialize_memory(env, D, N, time_channels_size, state_shape, skip_frames)

for episode in range(n_episode):
    print(" =================== "*3)

    start_time = time.time()

    acc_actions = []

    if tau >= max_tau:
        tau = 0
        target_model.set_weights(approximator_model.get_weights())
        print("===> Updated weights")

    # exploration_rate = np.power(exploration_base, -episode) if exploration_rate > minimal_exploration_rate else minimal_exploration_rate
    exploration_rate = utils.exploration_linear_decay(episode, 500)

    stats_frame_cnt, stats_rewards, stats_actions, stats_qs, stats_frames = utils.play(
        env, approximator_model, D, episode, exploration_rate, state_shape, action_space, time_channels_size, skip_frames)
    tau += stats_frame_cnt
    print(f"Number of frames in memory {len(D)}")
    if take_sample.__name__ == 'prioritized_experience_sampling':
        print("Uses Prioritised Experience Replay Sampling")
        experience_batch, importance = take_sample(D, approximator_model, target_model, batch_size, action_space, gamma=discount_rate, beta=1-(episode/n_episode))
    elif take_sample.__name__ == 'uniform_sampling':
        print("Uses Uniform Experience Replay Sampling")
        experience_batch = take_sample(D, batch_size)
    else:
        print("Uses Random Experience Replay Sampling")
        experience_batch = take_sample(D, batch_size)

    history = utils.train(approximator_model, target_model, experience_batch, importance, batch_size, action_space, discount_rate, tensorflow_callback)
    # Wrap up
    stats_nonzeros = (tf.math.count_nonzero([exp[1] for exp in experience_batch])/batch_size)*100
    stats_loss = history.history.get("loss", [0])[0]
    stats_time_end = np.round(time.time() - start_time, 2)
    stats_memory_usage = np.round(process.memory_info().rss/(1024**3), 2)
    sample_exp = random.choice(experience_batch)

    print(f"Current memory consumption is {stats_memory_usage} GB's")
    print(f"Number of information yielding states: {stats_nonzeros}")
    print(f"Loss of episode {episode} is {stats_loss} and took {stats_time_end} seconds with {stats_frame_cnt}")
    print(f"TOTAL REWARD: {stats_rewards}")

    utils.write_stats(file_writer_rewards, episode, sample_exp, skip_frames, exploration_rate, action_meanings,
                      stats_loss, stats_time_end, np.vstack(stats_qs),
                      stats_rewards, stats_frame_cnt, stats_nonzeros,
                      stats_memory_usage, stats_actions, stats_frames)

    utils.save_model(approximator_model, episode, checkpoint_path, save_freq)

# TODO: [x] Simplify the loss function
# TODO: [x] Apply the reward
# TODO: [x] Rethink memory handling
# TODO: [x] Proper memory initialisation
# TODO: [ ] Refactoring and restructuring
# TODO: [ ] Add states to tensorboard for analysis
# TODO: [ ] Write simple model run code
