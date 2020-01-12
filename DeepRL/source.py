import random
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

env = gym.make('BreakoutDeterministic-v4')
# frame = env.reset()
# env.render()

N = 10000
D = list()
discount_rate = 0.5

action_space = env.action_space.n
input_shape = list(np.array(env.observation_space.shape) // 2)[:2] + [4]
n_episode = 1
batch_size = 1
q_mask_shape = (batch_size, action_space)


def loss_function(init_qvalues, td_components):
    # init_q = tf.gather(initial_q, tf.argmax(initial_q, axis=1), axis=1)
    # next_q_indices = tf.math.argmax(next_q_values, axis=1)
    # next_q = tf.gather(next_q_values, next_q_indices, axis=1)

    # gamma = td_components[2]
    # current_rewards = td_components[1]
    init_q = tf.reduce_max(init_qvalues, axis=1)
    next_q = tf.reduce_max(td_components, axis=1)
    # tf.add(tf.multiply(next_q, gamma), current_rewards)

    # return tf.add(init_q, tf.subtract(next_q, init_q))
    return tf.subtract(next_q, init_q)


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
print(x.shape)
print(mask.shape)
out_q_values = tf.multiply(x, mask)
model = models.Model(inputs=[input, mask], outputs=out_q_values)
model.compile(optimizer='adam', loss=loss_function)
print(model)

exploration_rate = 0.5

initial_state = deque(maxlen=4)
next_state = deque(maxlen=4)

initial_observation = preprocess(env.reset())
action = env.action_space.sample()
next_observation, reward, is_done, _ = env.step(action)

initial_state.append(initial_observation)
initial_state.append(initial_observation)
initial_state.append(initial_observation)
initial_state.append(initial_observation)

next_state.append(initial_observation)
next_state.append(initial_observation)
next_state.append(initial_observation)
next_state.append(preprocess(next_observation))

D.append((initial_state, reward, action, next_state))

for episode in range(n_episode):
    experience = random.choice(D)
    initial_state = tf.reshape(tf.constant(experience[0]), [1] + input_shape)
    while not is_done:
        if random.choices((True, False), (exploration_rate, 1 - exploration_rate))[0]:
            action = env.action_space.sample()
        else:
            init_mask = tf.ones([1, action_space])
            q_values = model.predict([initial_state, init_mask])
            action = np.argmax(q_values)

        next_observation, reward, is_done, _ = env.step(action)
        env.render()
        next_state.append(preprocess(next_observation))
        D.append((initial_state, reward, action, next_state))

        experience_batch = random.sample(D, k=batch_size)

        # Gather initial and next state from memory for each batch item
        # for exp in experience_batch:
        #     print(exp[0])
        set_of_batch_initial_states = [exp[0] for exp in experience_batch]
        set_of_batch_initial_states = tf.reshape(set_of_batch_initial_states, [batch_size] + input_shape)
        set_of_batch_next_states = tf.constant([exp[3] for exp in experience_batch])
        set_of_batch_next_states = tf.reshape(set_of_batch_next_states, [batch_size] + input_shape)

        # Gather actions for each batch item
        set_of_batch_actions = tf.one_hot([exp[2] for exp in experience_batch], action_space)

        # Gather rewards for each batch item
        set_of_batch_rewards = tf.constant([exp[1] for exp in experience_batch])

        next_q_mask = tf.ones([1, action_space])
        next_q_values = model.predict([set_of_batch_next_states, next_q_mask])

        print(set_of_batch_actions)

        history = model.fit([set_of_batch_initial_states, set_of_batch_actions],
                            next_q_values, batch_size=batch_size, verbose=1)

# TODO: [ ] Simplify the loss function
# TODO: [ ] Apply the reward
# TODO: [ ] Rethink memory handling
# TODO: [ ] Proper memory initialisation
# TODO: [ ] Refactoring and restructuring
