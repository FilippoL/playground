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

from RingBuffer import RingBuf
process = psutil.Process(os.getpid())

# env = gym.make('BreakoutDeterministic-v4')
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
discount_rate = 0.8
tau = 0
max_tau = 2000
action_space = env.action_space.n
time_channels_size = 4
skip_frames = 4
input_shape = list(np.array(env.observation_space.shape) // 2)[:2] + [time_channels_size]
state_shape = list(np.zeros(input_shape).shape)[:2] + [time_channels_size+1]
batch_size = 200
N = batch_size*4
n_episode = 1000
q_mask_shape = (batch_size, action_space)

print(f"Pixel space of the game {input_shape}")


# def loss_function(next_qvalues, init_qvalues):
#     init_q = tf.reduce_max(init_qvalues, axis=1)
#     next_qvalues = tf.transpose(next_qvalues)
#     difference = tf.subtract(tf.transpose(init_q), next_qvalues)
#     return tf.square(difference)


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def standardize(img):
    return img/255


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    return standardize(to_grayscale(downsample(img)))


def collect_experience(env, action, state_shape, time_channels_size, skip_frames):
    next_observation, reward, is_done, _ = env.step(action)
    acc_obs = np.zeros(state_shape)
    acc_obs[:, :, 0] = preprocess(next_observation)
    acc_reward = reward
    frame_cnt = 1
    for i in range(1, (time_channels_size*skip_frames)+1):
        frame_cnt += 1
        next_observation, reward, is_done, _ = env.step(-1)
        acc_reward += reward

        if i % skip_frames == 0:
            acc_obs[:, :, (i//time_channels_size)] = acc_obs[:, :, -1] if is_done else preprocess(next_observation)

    # episode_rewards .append()
    # reward = reward / np.absolute(reward) if reward != 0 else reward # Reward normalisation
    # if reward != 0:
    #     print(reward)

    return acc_obs, acc_reward, is_done, frame_cnt


def create_model(input_shape, action_space):
    input = layers.Input(input_shape, dtype=tf.float32)
    mask = layers.Input(action_space, dtype=tf.float32)

    with tf.name_scope("ConvGroup-1"):
        x = layers.Conv2D(16, (8, 8), strides=4, activation="relu")(input)
        # x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(.4)(x)

    with tf.name_scope("ConvGroup-2"):
        x = layers.Conv2D(32, (4, 4), strides=2, activation="relu")(x)
        # x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(.4)(x)

    with tf.name_scope("ConvGroup-3"):
        x = layers.Conv2D(32, (3, 3), activation="relu")(x)
        # # x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(.4)(x)

    x = layers.Flatten()(x)

    with tf.name_scope("Value-Stream"):
        value_stream = layers.Dense(128, activation="relu")(x)
        value_out = layers.Dense(1)(value_stream)

    with tf.name_scope("Advantage-Stream"):
        advantage_stream = layers.Dense(128, activation="relu")(x)
        advantage_out = layers.Dense(action_space)(advantage_stream)

    with tf.name_scope("Q-Layer"):
        output = value_out + tf.math.subtract(advantage_out, tf.reduce_mean(advantage_out, axis=1, keepdims=True))
        out_q_values = tf.multiply(output, mask)
    # out_q_values = tf.reshape(out_q_values, [1,-1])
    model = models.Model(inputs=[input, mask], outputs=out_q_values)
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    return model


approximator_model = create_model(input_shape, action_space)
target_model = create_model(input_shape, action_space)

exploration_base = 1.02
exploration_rate = 1
minimal_exploration_rate = 0.01

# ===== INITIALISATION ======
frame_cnt = 0
prev_lives = 5
acc_nonzeros = []
acc_actions = []
is_done = False
env.reset()

for n in range(N):

    if is_done:
        env.reset()

    action = env.action_space.sample()
    state, acc_reward, is_done, _ = collect_experience(env, action, state_shape, time_channels_size, skip_frames)

    D.append((state, acc_reward, action))
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
    # print(is_done)
    initial_observation = env.reset()
    state = np.repeat(preprocess(initial_observation), 5).reshape(state_shape)
    is_done = False

    # next_state = initial_state.copy()  # To remove all the information of the last episode

    episode_rewards = []
    frame_cnt = 0
    while not is_done:
        # https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        frame_cnt += 1
        tau += 1

        if random.choices((True, False), (exploration_rate, 1 - exploration_rate))[0]:
            action = env.action_space.sample()
        else:
            # Greedy action
            init_mask = tf.ones([1, action_space])
            init_state = state[:, :, :-1]
            q_values = approximator_model.predict([tf.reshape(init_state, [1] + input_shape), init_mask])
            action = np.argmax(q_values)

        state, acc_reward, is_done, frames_of_collected = collect_experience(env, action, state_shape, time_channels_size, skip_frames)
        frame_cnt += frames_of_collected
        episode_rewards.append(acc_reward)

        acc_actions.append(action)
        D.append((state, acc_reward, action))
        if (episode % 5) == 0:
            with file_writer_rewards.as_default():
                tf.summary.histogram('action_taken', acc_actions, step=episode)
            print(f"Render for episode {episode}")
            env.render()

    print(f"Number of frames in memory {len(D)}")
    experience_batch = random.sample(D, k=batch_size)

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
    episode_nonzero_reward_states = (tf.math.count_nonzero(set_of_batch_rewards)/batch_size)*100
    print(f"Number of information yielding states: {episode_nonzero_reward_states}")

    next_q = set_of_batch_rewards + (discount_rate * tf.reduce_max(next_q_values, axis=1))
    history = approximator_model.fit([set_of_batch_initial_states, set_of_batch_actions], next_q, verbose=1, callbacks=[tensorflow_callback])

    # Wrap up
    loss = history.history.get("loss", [0])[0]
    time_end = np.round(time.time() - start_time, 2)
    memory_usage = process.memory_info().rss
    print(f"Current memory consumption is {memory_usage}")
    print(f"Loss of episode {episode} is {loss} and took {time_end} seconds")
    with file_writer_rewards.as_default():
        tf.summary.scalar('episode_rewards', np.sum(episode_rewards), step=episode)
        tf.summary.scalar('episode_loss', loss, step=episode)
        tf.summary.scalar('episode_time_in_secs', time_end, step=episode)
        tf.summary.scalar('episode_nr_frames', frame_cnt, step=episode)
        tf.summary.scalar('episode_exploration_rate', exploration_rate, step=episode)
        tf.summary.scalar('episode_mem_usage', memory_usage, step=episode)
        tf.summary.scalar('episode_frames_per_sec', np.round(frame_cnt/time_end, 2), step=episode)
        tf.summary.histogram('q-values', next_q_values, step=episode)
        if (episode+1) % 5 == 0:
            acc_nonzeros.append(episode_nonzero_reward_states)
            tf.summary.histogram('episode_nonzero_reward_states', acc_nonzeros, step=(episode+1)//5)
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
