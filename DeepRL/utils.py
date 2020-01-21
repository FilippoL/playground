from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import sys
import tensorflow as tf
import random
import tensorboard
import time
from DeepRL.sampling import prioritized_experience_sampling_3

tensorboard.plugins.custom_scalar


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
    acc_reward += -25 if is_done else 0

    return acc_obs, acc_reward, is_done, frame_cnt


def collect_experience_hidden_action_faithful(env, action, state_shape, time_channels_size, skip_frames):
    next_observation, reward, is_done, _ = env.step(action)
    acc_obs = np.zeros(state_shape)
    acc_obs[:, :, 0] = preprocess(next_observation)
    acc_reward = reward/np.abs(reward) if reward != 0 else 0
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
        acc_reward += reward/np.abs(reward) if reward != 0 else 0

    return acc_obs, acc_reward, is_done, frame_cnt


def collect_experience_stored_actions(env, action):
    next_observation, reward, is_done, _ = env.step(action)
    acc_obs = preprocess(next_observation)
    acc_reward = reward/np.abs(reward) if reward != 0 else 0

    return acc_obs, acc_reward, is_done


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


def image_grid_for_all_frames(images):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10, 10))
    skip_image = 4
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


def plot_q(acc_qs, meanings):

    figure = plt.figure(figsize=(10, 5))
    plt.plot(acc_qs)
    plt.legend(meanings)
    return figure


def image_grid_pommerman(experience, experience2, meanings):
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


def exploration_periodic_decay(episode, episodes_per_cycle=10, minimal_exploration_rate=0.1):
    return max(minimal_exploration_rate, np.cos(episode/episodes_per_cycle*(np.pi-(np.pi*0.5))))


def exploration_exponential_decay(episode, exploration_base=1.01, minimal_exploration_rate=0.1):
    return max(minimal_exploration_rate, np.power(exploration_base, -episode))


def exploration_linear_decay(episode, n_episodes=1000, minimal_exploration_rate=0.1):
    return max(minimal_exploration_rate, 1-(episode*1/n_episodes))


def initialize_memory(env, memory, N, time_channels_size):
    stacked_frames = None
    is_done = True
    for n in range(N):
        if is_done:
            initial_observation = env.reset()
            first_preprocess = preprocess(initial_observation)
            # state = np.repeat(first_preprocess, time_channels_size+1).reshape(state_shape)
            stacked_frames = deque([first_preprocess[...] for i in range(time_channels_size)], maxlen=time_channels_size+1)

        action = env.action_space.sample()
        frame, reward, is_done = collect_experience_stored_actions(env, action)
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
        memory.append([stacked_state, reward, action, reward, is_done])
        env.render()

    return env, memory


def initialize_memory_pommerman(env, D, N, TD_ERROR_DEFAULT):
    print("Running the init")
    for n in range(N):
        state_obs = env.reset()
        done = False
        while not done:
            actions_all_agents = env.act(state_obs)
            state_obs, reward, done, info, pixels = env.step2(
                actions_all_agents)

            D.append([standardize(pixels), reward[0], actions_all_agents[0], TD_ERROR_DEFAULT])
        print('Init episode {} finished'.format(n))
    return env, D


def play_episode(env, memory, time_channels_size, max_tau, exploration_rate, model, default_td_err, visualize=False):
    initial_observation = env.reset()
    action_space = env.action_space.n
    action_meanings = env.unwrapped.get_action_meanings()
    first_preprocess = preprocess(initial_observation)
    stacked_frames = deque([first_preprocess[...] for i in range(time_channels_size+1)], maxlen=time_channels_size+1)
    stacked_state = np.stack(stacked_frames, axis=2)
    is_done = False
    frame_cnt = 0
    init_mask = tf.ones([1, action_space])
    stats_rewards = []
    stats_actions = []
    stats_qs = []
    stats_frame = []
    time_delay = 1
    while not is_done:
        # https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        do_explore = random.choices((True, False), (exploration_rate, 1 - exploration_rate))[0]
        if do_explore:
            action = env.action_space.sample()
        else:
            # Greedy action
            init_state = stacked_state[:, :, :-1]
            if init_state.shape[2] == 3:
                print("Some stuff")
            q_values = model.predict([tf.reshape(init_state, (1,) + init_state.shape), init_mask])
            stats_qs.append(q_values[0])
            action = np.argmax(q_values)

        # if collect_experience.__name__ == 'collect_experience_hidden_action':

        frame, reward, is_done = collect_experience_stored_actions(env, action)
        frame_cnt += 1
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
        memory.append([stacked_state, reward, action, default_td_err, is_done])

        stats_rewards.append(reward)
        stats_actions.append(action)
        # stats_frame.append(frame)

        if visualize:
            # with file_writer_rewards.as_default():
            #     tf.summary.histogram('action_taken', acc_actions, step=episode)
            # time.sleep(time_delay)
            print(f"Reward {reward} with action {action_meanings[action]} which was {'explored' if do_explore else 'greedy'}")
            env.render()
    env.close()
    return stats_actions, stats_rewards, stats_qs, frame_cnt, stats_frame


def train_batch(experience_batch, approximator_model, target_model, action_space, discount_rate, tensorflow_callback):
    # Gather initial and next state from memory for each batch item
    batch_size = len(experience_batch)
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

    # init_q_values = approximator_model.predict([set_of_batch_initial_states, set_of_batch_actions])
    # init_q_values = approximator_model.predict([set_of_batch_initial_states, next_q_mask])
    # init_q = tf.reduce_max(init_q_values, axis=1)
    # td_error = (next_q-init_q).numpy()
    # td_err_default = max([exp[3] for exp in D])

    history = approximator_model.fit([set_of_batch_initial_states, set_of_batch_actions], next_q, verbose=1, callbacks=[tensorflow_callback])
    return history, episode_nonzero_reward_states


def train_batch_pommerman(D, approximator_model, target_model, ACTION_SPACE, BATCH_SIZE, tensorflow_callback, DISCOUNT_RATE):
    memory_length = len(D)

    print(f"Number of frames in memory {memory_length}")
    # experience_batch = take_sample(D, approximator_model, target_model, BATCH_SIZE, ACTION_SPACE)
    ids = prioritized_experience_sampling_3(D, BATCH_SIZE)
    experience_batch = [(D[idx], D[idx + 1]) if idx < memory_length - 1 else (D[idx - 1], D[idx]) for idx in ids]

    set_of_batch_states = tf.constant([exp[0][0] for exp in experience_batch])
    set_of_batch_next_states = tf.constant([exp[1][0] for exp in experience_batch])

    # Gather actions for each batch item
    set_of_batch_actions = tf.one_hot(
        [exp[0][2] for exp in experience_batch], ACTION_SPACE)

    # Maybe unnecessary - We are using the double q mask instead.
    next_q_mask = tf.ones([BATCH_SIZE, ACTION_SPACE])

    set_of_batch_states = tf.cast(tf.reshape(
        set_of_batch_states, set_of_batch_states.shape + [1]), dtype=tf.float32)
    double_q_mask = tf.one_hot(tf.argmax(approximator_model.predict(
        [set_of_batch_states, next_q_mask]), axis=1), ACTION_SPACE)  # http://arxiv.org/abs/1509.06461

    set_of_batch_next_states = tf.cast(tf.reshape(set_of_batch_next_states, set_of_batch_next_states.shape + [1]),
                                       dtype=tf.float32)
    next_q_values = tf.constant(target_model.predict([set_of_batch_next_states, double_q_mask]))

    # Gather rewards for each batch item
    set_of_batch_rewards = tf.constant(
        [exp[0][1] for exp in experience_batch], dtype=next_q_values.dtype)
    episode_nonzero_reward_states = (
                                            tf.math.count_nonzero(set_of_batch_rewards) / BATCH_SIZE) * 100
    print(
        f"Number of information yielding states: {episode_nonzero_reward_states}")

    next_q = set_of_batch_rewards + (DISCOUNT_RATE * tf.reduce_max(next_q_values, axis=1))
    init_q_values = approximator_model.predict([set_of_batch_states, set_of_batch_actions])
    init_q = tf.reduce_max(init_q_values, axis=1)
    td_error = (next_q - init_q).numpy()

    history = approximator_model.fit(
        [set_of_batch_states, set_of_batch_actions], next_q, verbose=1, callbacks=[tensorflow_callback])

    return history, episode_nonzero_reward_states, experience_batch, next_q_values, td_error


def write_stats(file_writer_rewards, qs, episode, exp, action_meanings, loss, time_end, rewards, frame_cnt, exploration_rate, memory_usage, episode_nonzero_reward_states, acc_actions, all_frames):

    # print(tmp.shape)
    episode_image = plot_to_image(image_grid(exp, action_meanings))

    with file_writer_rewards.as_default():
        tf.summary.scalar('episode_rewards', np.sum(rewards), step=episode)
        tf.summary.scalar('episode_loss', loss, step=episode)
        tf.summary.scalar('episode_time_in_secs', time_end, step=episode)
        tf.summary.scalar('episode_nr_frames', frame_cnt, step=episode)
        tf.summary.scalar('episode_exploration_rate', exploration_rate, step=episode)
        tf.summary.scalar('episode_mem_usage_in_GB', np.round(memory_usage/1024/1024/1024), step=episode)
        tf.summary.scalar('episode_frames_per_sec', np.round(frame_cnt/time_end, 2), step=episode)
        tf.summary.scalar('episode_nonzero_reward_states', episode_nonzero_reward_states, step=episode)
        tf.summary.image('episode_example_state', episode_image, step=episode)
        tf.summary.histogram('episode_actions', acc_actions, step=episode)
        tf.summary.histogram('episode_qs', qs, step=episode)
    # if len(all_frames) < 130:
        # episode_images = plot_to_image(image_grid_for_all_frames(all_frames))
        episode_q_image = plot_to_image(plot_q(qs, action_meanings))
        # tf.summary.image('episode_all_frames', episode_images, step=episode)
        tf.summary.image('episode_q_image', episode_q_image, step=episode)


def save_model(model, episode, checkpoint_path, save_freq):
    if (episode+1) % save_freq == 0:
        model_target_dir = checkpoint_path.format(epoch=episode)
        model.save_weights(model_target_dir)
        print(f"Model was saved under {model_target_dir}")
