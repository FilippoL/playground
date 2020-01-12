import datetime
import os
import random
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

env = gym.make('BreakoutDeterministic-v4')


# frame = env.reset()
# env.render()


def build_function():
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def update_dir(idx):
        log_dir = os.path.join(
            "logs",
            "fit",
            now,
            "episode" + str(idx)
        )
        return log_dir

    return update_dir

time_func = build_function()

N = 10000
D = list()
discount_rate = 0.7

action_space = env.action_space.n
input_shape = list(np.array(env.observation_space.shape) // 2)[:2] + [4]
n_episode = 10
batch_size = 10
q_mask_shape = (batch_size, action_space)


def loss_function(next_qvalues, init_qvalues):
    init_q = tf.reduce_max(init_qvalues, axis=1)
    next_qvalues = tf.transpose(next_qvalues)
    difference = tf.subtract(tf.transpose(init_q), next_qvalues)
    return tf.square(difference)


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    return to_grayscale(downsample(img))


input = layers.Input(input_shape)
mask = layers.Input(action_space, dtype=tf.float32)
x = layers.Conv2D(32, (3, 3), activation="elu")(input)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(32, (3, 3), activation="elu")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(32, (3, 3), activation="elu")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(action_space)(x)
out_q_values = tf.multiply(x, mask)
# out_q_values = tf.reshape(out_q_values, [1,-1])
model = models.Model(inputs=[input, mask], outputs=out_q_values)
model.compile(optimizer='adam', loss=loss_function)

exploration_rate = 0.1

initial_state = deque(maxlen=4)
next_state = deque(maxlen=4)

initial_observation = preprocess(env.reset())
action = env.action_space.sample()
next_observation, reward, is_done, _ = env.step(action)

initial_state.append(initial_observation)
initial_state.append(initial_observation)
initial_state.append(initial_observation)
initial_state.append(initial_observation)

next_state = initial_state.copy()

for n in range(100):
    action = env.action_space.sample()
    next_observation, reward, is_done, _ = env.step(action)
    next_state.append(preprocess(next_observation))
    D.append((initial_state.copy(), reward, action, next_state.copy()))
    initial_state = next_state
    env.render()

for episode in range(n_episode):
    initial_observation = preprocess(env.reset())
    initial_state.append(initial_observation)
    initial_state.append(initial_observation)
    initial_state.append(initial_observation)
    initial_state.append(initial_observation)

    next_state = initial_state.copy()

    tensorflow_callback = tf.keras.callbacks.TensorBoard(log_dir=time_func(episode), histogram_freq=1)

    while not is_done:

        if random.choices((True, False), (exploration_rate, 1 - exploration_rate))[0]:
            action = env.action_space.sample()
        else:
            init_mask = tf.ones([1, action_space])
            q_values = model.predict([tf.reshape(tf.constant(initial_state), [1] + input_shape), init_mask])
            action = np.argmax(q_values)

        next_observation, reward, is_done, _ = env.step(action)
        env.render()
        next_state.append(preprocess(next_observation))
        D.append((initial_state.copy(), reward, action, next_state.copy()))

    experience_batch = random.sample(D, k=batch_size)

    # Gather initial and next state from memory for each batch item
    # for exp in experience_batch:
    #     print(exp[0])
    set_of_batch_initial_states = [exp[0] for exp in experience_batch]
    # print(set_of_batch_initial_states)
    set_of_batch_initial_states = tf.reshape(set_of_batch_initial_states, [-1] + input_shape)

    set_of_batch_next_states = tf.constant([exp[3] for exp in experience_batch])
    set_of_batch_next_states = tf.reshape(set_of_batch_next_states, [-1] + input_shape)

    # Gather actions for each batch item
    set_of_batch_actions = tf.one_hot([exp[2] for exp in experience_batch], action_space)

    # Gather rewards for each batch item
    set_of_batch_rewards = tf.constant([exp[1] for exp in experience_batch])

    next_q_mask = tf.ones([batch_size, action_space])
    next_q_values = model.predict([set_of_batch_next_states, next_q_mask])
    # print(next_q_values)
    # print(tf.reduce_max(next_q_values, axis=1))
    next_q = set_of_batch_rewards + (discount_rate * tf.reduce_max(next_q_values, axis=1))
    # next_q = tf.reshape(next_q, [-1,1])
    # print(f"Next value {next_q}")
    # print(tf.reduce_max(next_q_values, axis=1).shape)
    # print(set_of_batch_rewards.shape)
    # print(set_of_batch_initial_states.shape)
    # print(set_of_batch_actions.shape)

    history = model.fit([set_of_batch_initial_states, set_of_batch_actions], next_q, verbose=1, batch_size=10,
                        callbacks=[tensorflow_callback])

# TODO: [x] Simplify the loss function
# TODO: [x] Apply the reward
# TODO: [x] Rethink memory handling
# TODO: [x] Proper memory initialisation
# TODO: [ ] Refactoring and restructuring
