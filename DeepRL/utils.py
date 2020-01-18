from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import sys
import tensorflow as tf


class Memory():
    def _init_(self, max_size, time_steps):
        self.full_memory = deque(maxlen=max_size)

    def add(self, experience):
        self.full_memory.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


def collect_experience_turtle(env, action, state_shape, time_channels_size, skip_frames):
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

    return acc_obs, acc_reward, is_done, frame_cnt


def collect_experience_random(env, action, state_shape, time_channels_size, skip_frames):
    action = env.action_space.sample()
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

    return acc_obs, acc_reward, is_done, frame_cnt


def collect_experience_hidden_action(env, action, state_shape, time_channels_size, skip_frames):
    next_observation, reward, is_done, _ = env.step(action)
    acc_obs = np.zeros(state_shape)
    acc_obs[:, :, 0] = preprocess(next_observation)
    acc_reward = reward
    frame_cnt = 1
    obs_cnt = 0
    for i in range(1, (time_channels_size*skip_frames)+1):
        frame_cnt += 1
        if i % skip_frames == 0:
            obs_cnt += 1
            # print(f"Setting observation: {obs_cnt}")
            next_observation, reward, is_done, _ = env.step(env.action_space.sample())
            acc_obs[:, :, obs_cnt] = acc_obs[:, :, -1] if is_done else preprocess(next_observation)
        else:
            next_observation, reward, is_done, _ = env.step(-1)
        acc_reward += reward if not is_done else -25

    return acc_obs, acc_reward, is_done, frame_cnt


def collect_experience_stored_actions(env, action, state_shape, time_channels_size, skip_frames):
    next_observation, reward, is_done, _ = env.step(action)
    acc_obs = np.zeros(state_shape)
    acc_obs[:, :, 0] = preprocess(next_observation)
    acc_actions = []
    acc_reward = reward
    frame_cnt = 1
    for i in range(1, 4+1):
        frame_cnt += 1
        next_observation, reward, is_done, _ = env.step(-1)
        if i % 4 == 0:
            acc_obs[:, :, (i//time_channels_size)] = acc_obs[:, :, -1] if is_done else preprocess(next_observation)
        else:
            next_observation, reward, is_done, _ = env.step(-1)
        acc_reward += reward

    return acc_obs, acc_reward, is_done, frame_cnt, acc_actions


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid(experience, meanings):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    img = experience[0]
    reward = experience[1]
    action = experience[2]
    figure = plt.figure(figsize=(10, 5))
    figure.suptitle(f"Action {meanings[action]} and received {reward}", fontsize=16)
    time_channel = img.shape[2]
    for i in range(time_channel):
        # Start next subplot.
        plt.subplot(1, time_channel, i + 1, title=f"Frame {i}")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img[:, :, i], cmap=plt.cm.gray)

    return figure


def image_grid_pommerman(experience, experience2, meanings):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    img = experience[0]
    img_next = experience2[0]
    reward = experience[1]
    action = experience[2]
    figure = plt.figure(figsize=(10, 5))
    figure.suptitle(f"Action {meanings[action]} and received {reward}", fontsize=16)

    # Start next subplot.
    plt.subplot(1, 2, 1, title=f"Frame initial")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img, cmap=plt.cm.gray)

    plt.subplot(1, 2, 2, title=f"Frame next")
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img_next, cmap=plt.cm.gray)

    return figure

# def collect_experience_elephant(env, action, state_shape, time_channels_size, memory):
#     next_observation, reward, is_done, _ = env.step(action)
#     acc_obs = np.zeros(state_shape)
#     acc_obs[:, :, 0] = preprocess(next_observation)
#     acc_actions = []
#     acc_reward = reward
#     frame_cnt = 1
#     for i in range(1, time_channels_size+1):
#         frame_cnt += 1
#         tmp_action = env.action_space.sample()
#         acc_actions.append(tmp_action)
#         next_observation, reward, is_done, _ = env.step(tmp_action)
#         acc_obs[:, :, (i//time_channels_size)] = acc_obs[:, :, -1] if is_done else preprocess(next_observation)
#         acc_reward += reward

#     return acc_obs, acc_reward, is_done, frame_cnt, acc_actions


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def standardize(img):
    return img/255


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    return standardize(to_grayscale(downsample(img)))
