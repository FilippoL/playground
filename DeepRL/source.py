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
# file_writer_qs = tf.summary.create_file_writer(log_dir + "/qs")
# file_writer.set_as_default()

# D = list()
list_size = 3000
D = deque(maxlen=list_size)
discount_rate = 0.8

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


input = layers.Input(input_shape, dtype=tf.float32)
mask = layers.Input(action_space, dtype=tf.float32)

x = layers.Conv2D(16, (8, 8), strides=4, activation="relu")(input)
# x = layers.MaxPooling2D((2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(.4)(x)

x = layers.Conv2D(32, (4, 4), strides=2, activation="relu")(x)
# x = layers.MaxPooling2D((2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(.4)(x)

x = layers.Conv2D(32, (3, 3), activation="relu")(x)
# # x = layers.MaxPooling2D((2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(.4)(x)

x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(action_space)(x)
out_q_values = tf.multiply(x, mask)
# out_q_values = tf.reshape(out_q_values, [1,-1])
model = models.Model(inputs=[input, mask], outputs=out_q_values)
model.compile(optimizer='rmsprop', loss='mean_squared_error')

exploration_base = 1.02
exploration_rate = 1
minimal_exploration_rate = 0.01

# ===== INITIALISATION ======
initial_state = deque(maxlen=4)
next_state = deque(maxlen=4)

initial_observation = preprocess(env.reset())
action = env.action_space.sample()
next_observation, reward, is_done, _ = env.step(action) # Unnecessary

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
        frame_cnt=0
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

    exploration_rate = np.power(exploration_base, -episode) if exploration_rate > minimal_exploration_rate else minimal_exploration_rate
    # exploration_rate = 1-(episode*0.05) if exploration_rate > minimal_exploration_rate else minimal_exploration_rate
    print(f"Running episode {episode} with exploration rate: {exploration_rate}")
    # print(is_done)
    initial_observation = preprocess(env.reset())
    is_done = False
    initial_state.append(initial_observation)
    initial_state.append(initial_observation)
    initial_state.append(initial_observation)
    initial_state.append(initial_observation)

    next_state = initial_state.copy() # To remove all the information of the last episode

    episode_rewards = []
    episode_rewards_normalized = []
    frame_cnt = 0
    # prev_lives = 5
    while not is_done:
        # https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        frame_cnt += 1
        if frame_cnt % 4 != 0:
            env.render()
            continue
        if random.choices((True, False), (exploration_rate, 1 - exploration_rate))[0]:
            action = env.action_space.sample()
        else:
            # Greedy action
            init_mask = tf.ones([1, action_space])
            # print(init_mask)
            q_values = model.predict([tf.reshape(tf.constant(initial_state), [1] + input_shape), init_mask])
            action = np.argmax(q_values)

        # print(f"Chose action: {action}")

        with file_writer_rewards.as_default():
            tf.summary.histogram('action_taken', action, step=frame_cnt)

        next_observation, reward, is_done, _ = env.step(action)

        episode_rewards.append(reward)
        reward = reward / np.absolute(reward) if reward != 0 else reward # Reward normalisation
        episode_rewards_normalized.append(reward)

        # reward = -1.0 if _['ale.lives'] < prev_lives else reward
        # reward = -2.0 if is_done else reward
        # prev_lives = _['ale.lives']
        # print(f"Current result: {reward}, {is_done}, {_}")
        # if reward>1:
            # print(reward)
        # print("RENDER!")
        next_state.append(preprocess(next_observation))
        # if len(D) < list_size:
        #     D.append((initial_state.copy(), reward , action, next_state.copy()))
        # else:
        #     rm_idx = random.randint(0,len(D)-1)
        #     to_remove = D.pop(rm_idx)
        #     D.append((initial_state.copy(), reward , action, next_state.copy()))

        D.append((initial_state.copy(), reward , action, next_state.copy()))
        env.render()



    print(f"Number of frames in memory {len(D)}")
    # batch_size = len(D)//10
    experience_batch = random.sample(D, k=batch_size)

    # Gather initial and next state from memory for each batch item
    set_of_batch_initial_states = [exp[0] for exp in experience_batch]
    set_of_batch_initial_states = tf.reshape(set_of_batch_initial_states, [-1] + input_shape)
    set_of_batch_next_states = tf.constant([exp[3] for exp in experience_batch])
    set_of_batch_next_states = tf.reshape(set_of_batch_next_states, [-1] + input_shape)

    # Gather actions for each batch item
    set_of_batch_actions = tf.one_hot([exp[2] for exp in experience_batch], action_space)

    # Gather rewards for each batch item

    next_q_mask = tf.ones([batch_size, action_space])
    next_q_values = tf.constant(model.predict([set_of_batch_next_states, next_q_mask]))
    print(next_q_values.dtype)
    set_of_batch_rewards = tf.constant([exp[1] for exp in experience_batch], dtype=next_q_values.dtype)
    print(set_of_batch_rewards.dtype)
    print(sum(set_of_batch_rewards))
    # print(list(zip(set_of_batch_rewards, next_q_values)))
    next_q = set_of_batch_rewards + (discount_rate * tf.reduce_max(next_q_values, axis=1))
    history = model.fit([set_of_batch_initial_states, set_of_batch_actions], next_q, verbose=1, callbacks=[tensorflow_callback])
    
    with file_writer_rewards.as_default():
        tf.summary.scalar('episode_rewards', np.sum(episode_rewards), step=episode)
        tf.summary.scalar('episode_rewards_normalized', np.sum(episode_rewards_normalized), step=episode)
        tf.summary.histogram('qs', next_q_values, step=episode)
    if (episode+1) % 100 == 0:
        model_target_dir = checkpoint_path.format(epoch=episode)
        model.save_weights(model_target_dir)
        print(f"Model was saved under {model_target_dir}")

# TODO: [x] Simplify the loss function
# TODO: [x] Apply the reward
# TODO: [x] Rethink memory handling
# TODO: [x] Proper memory initialisation
# TODO: [ ] Refactoring and restructuring
