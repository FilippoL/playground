import datetime
import os
import random
from collections import deque

import time

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
import psutil
from utils import collect_experience_hidden_action, preprocess, image_grid, plot_to_image, exploration_linear_decay, exploration_exponential_decay
from model import create_model, create_model_faithful

import sampling

process = psutil.Process(os.getpid())


collect_experience = collect_experience_hidden_action
take_sample = sampling.prioritized_experience_sampling
# env = gym.make('BreakoutDeterministic-v4')
env = gym.make('Assault-v0')

now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

checkpoint_path = os.path.join(
    ".",
    "models",
    now,
    "-{epoch:04d}.ckpt"
)
MODEL_PATH = "models/20200119-235832-Better-PER"
latest = tf.train.latest_checkpoint(MODEL_PATH)
print(f"Loading model from {latest}")
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
time_channels_size = 4
skip_frames = 1
input_shape = list(np.array(env.observation_space.shape) // 2)[:2] + [time_channels_size]
state_shape = list(np.zeros(input_shape).shape)[:2] + [time_channels_size+1]
batch_size = 250
N = batch_size
n_episode = 2000
q_mask_shape = (batch_size, action_space)
action_meanings = env.unwrapped.get_action_meanings()

print(f"Pixel space of the game {input_shape}")


# def loss_function(next_qvalues, init_qvalues):
#     init_q = tf.reduce_max(init_qvalues, axis=1)
#     next_qvalues = tf.transpose(next_qvalues)
#     difference = tf.subtract(tf.transpose(init_q), next_qvalues)
#     return tf.square(difference)

# episode_rewards .append()
# reward = reward / np.absolute(reward) if reward != 0 else reward # Reward normalisation
# if reward != 0:
#     print(reward)

approximator_model = create_model_faithful(input_shape, action_space)
target_model = create_model_faithful(input_shape, action_space)

approximator_model.load_weights(latest)
target_model.load_weights(latest)

exploration_base = 1.02
exploration_rate = 1
minimal_exploration_rate = 0.01

# ===== INITIALISATION ======
frame_cnt = 0
prev_lives = 5
is_done = False
env.reset()

td_err_default = 0
acc_nonzeros = []
acc_actions = []

for n in range(N):

    if is_done:
        env.reset()

    action = env.action_space.sample()
    state, acc_reward, is_done, _ = collect_experience(env, action, state_shape, time_channels_size, skip_frames)
    D.append([state, acc_reward, action, acc_reward])
    env.render()

for episode in range(n_episode):
    start_time = time.time()

    acc_actions = []

    if tau >= max_tau:
        tau = 0
        target_model.set_weights(approximator_model.get_weights())
        print("===> Updated weights")

    exploration_rate = exploration_exponential_decay(episode, exploration_base)
    # exploration_rate = exploration_linear_decay(episode, 500)
    # exploration_rate = exploration_periodic_decay(episode, episodes_per_cycle)

    print(f"Running episode {episode} with exploration rate: {exploration_rate}")

    print(f"Memory trace {process.memory_info().rss}")
    print(f"Number of frames in memory {len(D)}")
    if take_sample.__name__ == 'prioritized_experience_sampling':
        print("Uses Prioritised Experience Replay Sampling")
        experience_batch = take_sample(D, approximator_model, target_model, batch_size, action_space)
    elif take_sample.__name__ == 'prioritized_experience_sampling_2':
        print("Uses Prio Experience Replay Sampling")
        experience_batch = take_sample(D, approximator_model, target_model, batch_size, action_space)
    elif take_sample.__name__ == 'prioritized_experience_sampling_3':
        print("Uses Prio Experience Replay Sampling")
        ids = take_sample(D, batch_size)
        experience_batch = [D[idx] for idx in ids]
    elif take_sample.__name__ == 'uniform_sampling':
        print("Uses Uniform Experience Replay Sampling")
        experience_batch = take_sample(D, batch_size)
    elif take_sample.__name__ == 'random_sampling':
        print("Uses Random Experience Replay Sampling")
        experience_batch = take_sample(D, batch_size)

    # next_state = initial_state.copy()  # To remove all the information of the last episode
    episode_rewards = []
    frame_cnt = 0

    # Gather initial and next state from memory for each batch item
    set_of_batch_initial_states = tf.constant([exp[0][:, :, :-1] for exp in experience_batch])
    set_of_batch_next_states = tf.constant([exp[0][:, :, 1:] for exp in experience_batch])

    # Gather actions for each batch item
    set_of_batch_actions = tf.one_hot([exp[2] for exp in experience_batch], action_space)

    next_q_mask = tf.ones([batch_size, action_space])  # Maybe unnecessary - We are using the double q mask instead.
    double_q_mask = tf.one_hot(tf.argmax(approximator_model.predict([set_of_batch_next_states, next_q_mask]), axis=1), action_space)  # http://arxiv.org/abs/1509.06461
    next_q_values = tf.constant(target_model.predict([set_of_batch_next_states, double_q_mask]))

    # Gather rewards for each batch item
    set_of_batch_rewards = tf.constant([exp[1] for exp in experience_batch], dtype=next_q_values.dtype)
    episode_nonzero_reward_states = (tf.math.count_nonzero(set_of_batch_rewards)/batch_size)*100
    print(f"Number of information yielding states: {episode_nonzero_reward_states}")

    next_q = set_of_batch_rewards + (discount_rate * tf.reduce_max(next_q_values, axis=1))

    init_q_values = approximator_model.predict([set_of_batch_initial_states, set_of_batch_actions])
    # init_q_values = approximator_model.predict([set_of_batch_initial_states, next_q_mask])
    init_q = tf.reduce_max(init_q_values, axis=1)
    td_error = (next_q-init_q).numpy()
    td_err_default = max([exp[3] for exp in D])

    history = approximator_model.fit([set_of_batch_initial_states, set_of_batch_actions], next_q, verbose=1, callbacks=[tensorflow_callback])

    for err_value, exp in zip(td_error, experience_batch):
        exp[3] = err_value

    # print(is_done)
    initial_observation = env.reset()
    first_preprocess = preprocess(initial_observation)
    state = np.repeat(first_preprocess, time_channels_size+1).reshape(state_shape)
    is_done = False
    init_mask = tf.ones([1, action_space])
    while not is_done:
        # https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        frame_cnt += 1
        tau += 1
        do_explore = random.choices((True, False), (exploration_rate, 1 - exploration_rate))[0]
        if do_explore:
            action = env.action_space.sample()
        else:
            # Greedy action
            init_state = state[:, :, :-1]
            q_values = approximator_model.predict([tf.reshape(init_state, [1] + input_shape), init_mask])
            action = np.argmax(q_values)

        # if collect_experience.__name__ == 'collect_experience_hidden_action':
        state, acc_reward, is_done, frames_of_collected = collect_experience(env, action, state_shape, time_channels_size, skip_frames)
        frame_cnt += frames_of_collected
        episode_rewards.append(acc_reward)

        # if collect_experience.__name__ == 'collect_experience_stored_actions':
        #     state, acc_reward, is_done, frames_of_collected, all_collected_actions = collect_experience(env, action, state_shape, time_channels_size, skip_frames)
        #     frame_cnt += frames_of_collected
        #     episode_rewards.append(acc_reward)

        acc_actions.append(action)
        D.append([state, acc_reward, action, td_err_default])
        if (episode % 5) == 0:
            with file_writer_rewards.as_default():
                tf.summary.histogram('action_taken', acc_actions, step=episode)
            print(f"Episode {episode}: Reward {acc_reward} with action {action_meanings[action]} which was {'explored' if do_explore else 'greedy'}")
            env.render()

    # Wrap up
    loss = history.history.get("loss", [0])[0]
    time_end = np.round(time.time() - start_time, 2)
    memory_usage = process.memory_info().rss
    tmp = random.choice(experience_batch)
    # print(tmp.shape)
    episode_image = plot_to_image(image_grid(tmp, env.unwrapped.get_action_meanings()))
    print(f"Loss of episode {episode} is {loss} and took {time_end} seconds")
    print(f"TOTAL REWARD: {np.sum(episode_rewards)}")
    with file_writer_rewards.as_default():
        tf.summary.scalar('episode_rewards', np.sum(episode_rewards), step=episode)
        tf.summary.scalar('episode_loss', loss, step=episode)
        tf.summary.scalar('episode_time_in_secs', time_end, step=episode)
        tf.summary.scalar('episode_nr_frames', frame_cnt, step=episode)
        tf.summary.scalar('episode_exploration_rate', exploration_rate, step=episode)
        tf.summary.scalar('episode_mem_usage', memory_usage, step=episode)
        tf.summary.scalar('episode_mem_usage_in_GB', np.round(memory_usage/1024/1024/1024), step=episode)
        tf.summary.scalar('episode_frames_per_sec', np.round(frame_cnt/time_end, 2), step=episode)
        # print(np.shape(experience_batch[0][0][:, :, 0]))
        tf.summary.image('episode_example_state', episode_image, step=episode)
        tf.summary.histogram('q-values', next_q_values, step=episode)
        if (episode+1) % 5 == 0:
            acc_nonzeros.append(episode_nonzero_reward_states)
            tf.summary.histogram('episode_nonzero_reward_states', acc_nonzeros, step=(episode+1)//5)
            acc_nonzeros = []
        else:
            acc_nonzeros.append(episode_nonzero_reward_states)
    if (episode+1) % 50 == 0:
        model_target_dir = checkpoint_path.format(epoch=episode)
        approximator_model.save_weights(model_target_dir)
        print(f"Model was saved under {model_target_dir}")

# TODO: [x] Simplify the loss function
# TODO: [x] Apply the reward
# TODO: [x] Rethink memory handling
# TODO: [x] Proper memory initialisation
# TODO: [ ] Refactoring and restructuring
# TODO: [ ] Add states to tensorboard for analysis
# TODO: [ ] Write simple model run code
