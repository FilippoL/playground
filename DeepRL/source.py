import datetime
import os
import random
from collections import deque

import time

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

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
tensorflow_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
file_writer_rewards = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer_qs = tf.summary.create_file_writer(log_dir + "/metrics")

# file_writer_qs = tf.summary.create_file_writer(log_dir + "/qs")
# file_writer.set_as_default()

# D = list()
list_size = 6000
D = deque(maxlen=list_size)
discount_rate = 0.8
tau = 0
max_tau = 2000
action_space = env.action_space.n
input_shape = list(np.array(env.observation_space.shape) // 2)[:2] + [4]
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
initial_state = deque(maxlen=4)
next_state = deque(maxlen=4)

initial_observation = preprocess(env.reset())
action = env.action_space.sample()
next_observation, reward, is_done, _ = env.step(action)  # Unnecessary

initial_state.append(initial_observation)
initial_state.append(initial_observation)
initial_state.append(initial_observation)
initial_state.append(initial_observation)

next_state = initial_state.copy()
frame_cnt = 0
prev_lives = 5
for n in range(N):
    frame_cnt += 1
    if frame_cnt % 4 != 0:
        env.render()
        continue
    if is_done:
        frame_cnt = 0
        env.reset()
        initial_observation = preprocess(env.reset())
        initial_state.append(initial_observation)
        initial_state.append(initial_observation)
        initial_state.append(initial_observation)
        initial_state.append(initial_observation)
    action = env.action_space.sample()
    next_observation, reward, is_done, _ = env.step(action)
    reward = reward / np.absolute(reward) if reward != 0 else reward
    # reward = -1.0 if _['ale.lives'] < prev_lives else reward
    # reward = -2.0 if is_done else reward
    next_state.append(preprocess(next_observation))
    D.append((initial_state.copy(), reward, action, next_state.copy()))
    initial_state = next_state
    env.render()

for episode in range(n_episode):
    start_time = time.time()
    if tau >= max_tau:
        tau = 0
        target_model.set_weights(approximator_model.get_weights())
        print("===> Updated weights")

    exploration_rate = np.power(exploration_base, -episode) if exploration_rate > minimal_exploration_rate else minimal_exploration_rate
    exploration_rate = 1-(episode*1/n_episode) if exploration_rate > minimal_exploration_rate else minimal_exploration_rate

    print(f"Running episode {episode} with exploration rate: {exploration_rate}")
    # print(is_done)
    initial_observation = preprocess(env.reset())
    is_done = False
    initial_state.append(initial_observation)
    initial_state.append(initial_observation)
    initial_state.append(initial_observation)
    initial_state.append(initial_observation)

    next_state = initial_state.copy()  # To remove all the information of the last episode

    episode_rewards = []
    episode_rewards_normalized = []
    frame_cnt = 0
    # prev_lives = 5
    while not is_done:
        # https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        frame_cnt += 1
        if frame_cnt % 4 != 0:
            continue
        tau += 1
        if random.choices((True, False), (exploration_rate, 1 - exploration_rate))[0]:
            action = env.action_space.sample()
        else:
            # Greedy action
            init_mask = tf.ones([1, action_space])
            q_values = approximator_model.predict([tf.reshape(tf.constant(initial_state), [1] + input_shape), init_mask])
            action = np.argmax(q_values)

        with file_writer_rewards.as_default():
            tf.summary.histogram('action_taken', action, step=frame_cnt)

        next_observation, reward, is_done, _ = env.step(action)

        episode_rewards.append(reward)
        # reward = reward / np.absolute(reward) if reward != 0 else reward # Reward normalisation
        # if reward != 0:
        #     print(reward)

        next_state.append(preprocess(next_observation))

        D.append((initial_state.copy(), reward, action, next_state.copy()))
        if (episode % 5) == 0:
            print(f"Render for episode {episode}")
            env.render()

    print(f"Number of frames in memory {len(D)}")
    experience_batch = random.sample(D, k=batch_size)

    # Gather initial and next state from memory for each batch item
    set_of_batch_initial_states = [exp[0] for exp in experience_batch]
    set_of_batch_initial_states = tf.reshape(set_of_batch_initial_states, [-1] + input_shape)
    set_of_batch_next_states = tf.constant([exp[3] for exp in experience_batch])
    set_of_batch_next_states = tf.reshape(set_of_batch_next_states, [-1] + input_shape)

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
    loss = history.history.get("loss", [0])[0]
    time_end = np.round(time.time() - start_time, 2)

    print(f"Loss of episode {episode} is {loss} and took {time_end} seconds")
    with file_writer_rewards.as_default():
        tf.summary.scalar('episode_rewards', np.sum(episode_rewards), step=episode)
        tf.summary.scalar('episode_loss', loss, step=episode)
        tf.summary.scalar('episode_time_in_secs', time_end, step=episode)
        tf.summary.scalar('episode_exploration_rate', exploration_rate, step=episode)
        tf.summary.histogram('episode_nonzero_reward_states', episode_nonzero_reward_states)
        tf.summary.histogram('q-values', next_q_values)
    if (episode+1) % 50 == 0:
        model_target_dir = checkpoint_path.format(epoch=episode)
        approximator_model.save_weights(model_target_dir)
        print(f"Model was saved under {model_target_dir}")

# TODO: [x] Simplify the loss function
# TODO: [x] Apply the reward
# TODO: [x] Rethink memory handling
# TODO: [x] Proper memory initialisation
# TODO: [ ] Refactoring and restructuring
