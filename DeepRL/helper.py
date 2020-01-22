from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import sys
import random
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
        acc_reward += reward

    return acc_obs, acc_reward, is_done, frame_cnt, _['ale.lives']


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


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def standardize(img):
    return img/255


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    return standardize(to_grayscale(downsample(img)))


def initialize_memory(env, memory, N, time_channels_size, state_shape, skip_frames):

    init_lives = env.unwrapped.ale.lives()
    is_done = False
    for n in range(N):
        if is_done:
            env.reset()

        action = env.action_space.sample()
        state, acc_reward, is_done, frm, lives = collect_experience_hidden_action(env, action, state_shape, time_channels_size, skip_frames)
        is_done = True if lives < init_lives else is_done
        memory.append((state, acc_reward, action, is_done))
        env.render()

    return env, memory


def play(env, model, memory, episode, exploration_rate, state_shape, action_space, time_channels_size, skip_frames):
    # Initialize stats
    print(f"Running episode {episode} with exploration rate: {exploration_rate}")
    initial_observation = env.reset()
    first_preprocess = preprocess(initial_observation)
    state = np.repeat(first_preprocess, time_channels_size+1).reshape(state_shape)
    is_done = False
    init_lives = env.unwrapped.ale.lives()
    action_meanings = env.unwrapped.get_action_meanings()
    stats_actions = []
    stats_qs = []
    stats_rewards = 0
    stats_frame_cnt = 0
    stats_frames = []
    tau = 0

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
            init_state = tf.expand_dims(state[:, :, 1:], axis=0)
            q_values = model.predict([init_state, init_mask])
            action = np.argmax(q_values)

        # if collect_experience.__name__ == 'collect_experience_hidden_action':
        state, acc_reward, is_done, frames_of_collected, lives = collect_experience_hidden_action(env, action, state_shape, time_channels_size, skip_frames)
        is_done = True if lives < init_lives else is_done
        stats_frame_cnt += frames_of_collected
        stats_rewards += acc_reward
        stats_actions.append(action)
        stats_qs.append(q_values)
        stats_frames.append(state[:, :, -1])

        memory.append((state, acc_reward, action, is_done))
        if (episode % 5) == 0:
            print(f"Reward {acc_reward} with action {action_meanings[action]} which was {'explored' if do_explore else 'greedy'}")
            env.render()
    return stats_frame_cnt, stats_rewards, stats_actions, stats_qs, stats_frames


def train(approximator_model, target_model, experience_batch, importance, batch_size, action_space, discount_rate, tensorflow_callback):
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
    is_terminal = tf.constant([0 if exp[3] else 1 for exp in experience_batch], dtype=tf.float32)
    next_q = set_of_batch_rewards + ((discount_rate * tf.reduce_max(next_q_values, axis=1)) * is_terminal)
    history = approximator_model.fit([set_of_batch_initial_states, set_of_batch_actions], next_q, verbose=1, callbacks=[tensorflow_callback], sample_weight=importance)
    return history


def write_stats(file_writer_rewards, episode, sample_exp, frame_skip, exploration_rate, action_meanings,
                stats_loss, stats_time_end, stats_qs,
                stats_rewards, stats_frame_cnt, stats_nonzeros,
                stats_memory_usage, stats_actions, stats_frames):

    # with file_writer_rewards.as_default():
    #     episode_image = plot_to_image(image_grid(sample_exp, action_meanings))
    #     tf.summary.image('episode_example_state', episode_image, step=episode)

    with file_writer_rewards.as_default():
        tf.summary.scalar('episode_rewards', np.sum(stats_rewards), step=episode)
        tf.summary.scalar('episode_loss', stats_loss, step=episode)
        tf.summary.scalar('episode_time_in_secs', stats_time_end, step=episode)
        tf.summary.scalar('episode_nr_frames', stats_frame_cnt, step=episode)
        tf.summary.scalar('episode_exploration_rate', exploration_rate, step=episode)
        tf.summary.scalar('episode_mem_usage_in_GB', stats_memory_usage, step=episode)
        tf.summary.scalar('episode_frames_per_sec', np.round(stats_frame_cnt/stats_time_end, 2), step=episode)
        tf.summary.scalar('episode_nonzero_reward_states', stats_nonzeros, step=episode)
        tf.summary.histogram('episode_actions', stats_actions, step=episode)
        tf.summary.histogram('episode_qs', stats_qs, step=episode)
        episode_images = plot_to_image(image_grid_for_all_frames(stats_frames, frame_skip))
        episode_q_image = plot_to_image(plot_q(stats_qs, action_meanings))
        tf.summary.image('episode_all_frames', episode_images, step=episode)
        tf.summary.image('episode_q_image', episode_q_image, step=episode)


def save_model(model, episode, checkpoint_path, save_freq):
    if (episode+1) % save_freq == 0:
        model_target_dir = checkpoint_path.format(epoch=episode)
        model.save_weights(model_target_dir)
        print(f"Model was saved under {model_target_dir}")


def image_grid_for_all_frames(images, skipframes):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10, 10))
    skip_image = skipframes
    tmp_images = images[::skip_image]
    grid_h = 6
    grid_w = 6
    nr_images = len(tmp_images)
    for i in range(nr_images):
        if i >= grid_h * grid_w:
            break
        # Start next subplot.
        plt.subplot(grid_h, grid_w, i + 1, title=f"F:{i*skip_image}")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(tmp_images[i], cmap=plt.cm.gray)
    plt.tight_layout()
    return figure

def show_pommerman_game(images, actions, meanings):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10, 10))
    skip_image = 1
    tmp_images = images[::1]
    grid_h = 6
    grid_w = 6
    nr_images = len(tmp_images)
    for i in range(nr_images):
        if i >= grid_h * grid_w:
            break
        # Start next subplot.
        plt.subplot(grid_h, grid_w, i + 1, title=f"F:{i*skip_image}: {meanings[actions[i]]}")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(tmp_images[i], cmap=plt.cm.gray)
    plt.tight_layout()
    return figure

def plot_q(acc_qs, meanings):

    figure = plt.figure(figsize=(10, 5))
    plt.plot(acc_qs)
    plt.legend(meanings)
    return figure


def exploration_periodic_decay(episode, episodes_per_cycle=10, minimal_exploration_rate=0.1):
    return max(minimal_exploration_rate, np.cos(episode/episodes_per_cycle*(np.pi-(np.pi*0.5))))


def exploration_exponential_decay(episode, exploration_base=1.01, minimal_exploration_rate=0.1):
    return max(minimal_exploration_rate, np.power(exploration_base, -episode))


def exploration_linear_decay(episode, n_episodes=1000, minimal_exploration_rate=0.1):
    return max(minimal_exploration_rate, 1-(episode*1/n_episodes))