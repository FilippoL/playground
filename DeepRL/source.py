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
from model import create_model
import helper as utils
import sampling

process = psutil.Process(os.getpid())


collect_experience = collect_experience_hidden_action
take_sample = sampling.prioritized_experience_sampling
# take_sample = sampling.uniform_sampling
# take_sample = sampling.random_sampling

# env = gym.make('BreakoutDeterministic-v4')
frame_skip = 1
env = gym.make('Assault-v0')

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
list_size = 6000
D = deque(maxlen=list_size)
# D = RingBuf(list_size)
discount_rate = 0.99
tau = 0
max_tau = 2000
action_space = env.action_space.n
action_meanings = env.unwrapped.get_action_meanings()
time_channels_size = 2
skip_frames = 2
input_shape = list(np.array(env.observation_space.shape) // 2)[:2] + [time_channels_size]
state_shape = list(np.zeros(input_shape).shape)[:2] + [time_channels_size+1]
batch_size = 250
N = batch_size
n_episode = 2000
q_mask_shape = (batch_size, action_space)
save_freq = 50
print(f"Pixel space of the game {input_shape}")

approximator_model = create_model(input_shape, action_space)
target_model = create_model(input_shape, action_space)

exploration_base = 1.02
exploration_rate = 1
minimal_exploration_rate = 0.01

# ===== INITIALISATION ======
frame_cnt = 0
prev_lives = env.unwrapped.ale.lives()
is_done = False
env.reset()

lives = prev_lives
for n in range(N):
    if is_done:
        env.reset()

    action = env.action_space.sample()
    state, acc_reward, is_done, frm, lives = collect_experience(env, action, state_shape, time_channels_size, skip_frames)
    is_done = True if lives < prev_lives else is_done
    D.append((state, acc_reward, action, is_done))
    env.render()

for episode in range(n_episode):
    start_time = time.time()
    if tau >= max_tau:
        tau = 0
        target_model.set_weights(approximator_model.get_weights())
        print("===> Updated weights")

    exploration_rate = np.power(exploration_base, -episode) if exploration_rate > minimal_exploration_rate else minimal_exploration_rate
    # exploration_rate = 1-(episode*1/n_episode) if exploration_rate > minimal_exploration_rate else minimal_exploration_rate

    print(f"Running episode {episode} with exploration rate: {exploration_rate}")
    initial_observation = env.reset()
    first_preprocess = preprocess(initial_observation)
    state = np.repeat(first_preprocess, time_channels_size+1).reshape(state_shape)
    is_done = False

    # Initialize stats
    stats_actions = []
    stats_qs = []
    stats_rewards = 0
    stats_frame_cnt = 0
    stats_frames = []

    while not is_done:
        # https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        stats_frame_cnt += 1
        tau += 1
        do_explore = random.choices((True, False), (exploration_rate, 1 - exploration_rate))[0]
        if do_explore:
            q_values = np.zeros([1, action_space])
            action = env.action_space.sample()
        else:
            # Greedy action

            init_mask = tf.ones([1, action_space])
            init_state = state[:, :, :-1]
            q_values = approximator_model.predict([tf.reshape(init_state, [1] + input_shape), init_mask])
            action = np.argmax(q_values)

        # if collect_experience.__name__ == 'collect_experience_hidden_action':
        state, acc_reward, is_done, frames_of_collected, lives = collect_experience(env, action, state_shape, time_channels_size, skip_frames)
        is_done = True if lives < prev_lives else is_done
        stats_frame_cnt += frames_of_collected
        stats_rewards += acc_reward
        stats_actions.append(action)
        stats_qs.append(q_values)
        stats_frames.append(state[:, :, -1])

        D.append((state, acc_reward, action, is_done))
        if (episode % 5) == 0:
            print(f"Reward {acc_reward} with action {action_meanings[action]} which was {'explored' if do_explore else 'greedy'}")
            env.render()

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

    # Gather initial and next state from memory for each batch item
    set_of_batch_initial_states = tf.constant([exp[0][:, :, :-1] for exp in experience_batch])
    # set_of_batch_initial_states = tf.reshape(set_of_batch_initial_states, [-1] + input_shape)
    set_of_batch_next_states = tf.constant([exp[0][:, :, 1:] for exp in experience_batch])
    # set_of_batch_next_states = tf.reshape(set_of_batch_next_states, [-1] + input_shape)

    # Gather actions for each batch item
    set_of_batch_actions = tf.one_hot([exp[2] for exp in experience_batch], action_space)

    next_q_mask = tf.ones([batch_size, action_space])  # Maybe unnecessary - We are using the double q mask instead.
    double_q_mask = tf.one_hot(tf.argmax(approximator_model.predict([set_of_batch_next_states, next_q_mask]), axis=1), action_space)  # http://arxiv.org/abs/1509.06461
    next_q_values = tf.constant(target_model.predict([set_of_batch_next_states, double_q_mask]))

    # Gather rewards for each batch item
    set_of_batch_rewards = tf.constant([exp[1] for exp in experience_batch], dtype=next_q_values.dtype)

    next_q = set_of_batch_rewards + (discount_rate * tf.reduce_max(next_q_values, axis=1))
    history = approximator_model.fit([set_of_batch_initial_states, set_of_batch_actions], next_q, verbose=1, callbacks=[tensorflow_callback], sample_weight=importance)

    # Wrap up
    stats_nonzeros = (tf.math.count_nonzero(set_of_batch_rewards)/batch_size)*100
    stats_loss = history.history.get("loss", [0])[0]
    stats_time_end = np.round(time.time() - start_time, 2)
    stats_memory_usage = np.round(process.memory_info().rss/(1024**3), 2)
    sample_exp = random.choice(experience_batch)

    print(f"Current memory consumption is {stats_memory_usage} GB's")
    print(f"Number of information yielding states: {stats_nonzeros}")
    print(f"Loss of episode {episode} is {stats_loss} and took {stats_time_end} seconds with {stats_frame_cnt}")
    print(f"TOTAL REWARD: {stats_rewards}")

    utils.write_stats(file_writer_rewards, episode, sample_exp, frame_skip, exploration_rate, action_meanings,
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
