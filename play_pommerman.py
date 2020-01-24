import os
import datetime
import platform
import time
import json

# import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

import pommerman
from DeepRL.sampling import prioritized_experience_sampling_3
from pommerman import agents, constants

# =========== HELPER FUNCTIONS =========== #

take_sample = prioritized_experience_sampling_3
now = ""


def preprocess(img):
    return img / 255


# =========== CREATE THE CNN =========== #


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
        output = value_out + \
            tf.math.subtract(advantage_out, tf.reduce_mean(
                advantage_out, axis=1, keepdims=True))
        out_q_values = tf.multiply(output, mask)
    # out_q_values = tf.reshape(out_q_values, [1,-1])
    model = models.Model(inputs=[input, mask], outputs=out_q_values)
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    return model


def save_json(accumulated_frames, episode_rewards):
    stats_path = f"./stats/{now}_SimpleAgent.json"

    data = {}
    data['frames_per_episode'] = accumulated_frames
    data['rewards_per_episode'] = episode_rewards

    with open(stats_path, 'w') as outfile:
        json.dump(data, outfile)


def main():
    # Create the environment
    agent_list = [
        agents.RandomAgent(),
        agents.SimpleAgent(),
        # agents.SimpleAgent(),
        # agents.SimpleAgent(),
    ]
    env = pommerman.make('OneVsOne-v0',
                         agent_list, render_mode='human')

    if platform.system() == 'Darwin':
        print("MacBook Pro user detected. U rule.")
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # =========== STATS =========== #

    global now
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    accumulated_frames = []
    episode_rewards = []

    # =========== (HYPER)PARAMETERS AND VARIABLES =========== #

    ACTION_SPACE = env.action_space.n
    TIME_CHANNELS_SIZE = 1
    INPUT_SHAPE = list(env.get_observation_space()) + [TIME_CHANNELS_SIZE]
    N_EPISODES = 10000

    MODEL_PATH = "models/20200119-121818"
    latest = tf.train.latest_checkpoint(MODEL_PATH)
    print(f"Loading model from {latest}")

    ## - Comment 2 lines below for running Random/SimpleAgent- ##
    restored_model = create_model(INPUT_SHAPE, ACTION_SPACE)
    restored_model.load_weights(latest)

    actions_available = [str(action).split(".")[1]
                         for action in constants.Action]

    for episode in range(N_EPISODES):
        start_time = time.time()

        print(
            f"Running episode {episode}.")

        # Intial step for the episode
        state_obs = env.reset()
        actions = env.act(state_obs)
        initial_observation, reward, done, info, pixels = env.step2(
            actions, render=True)

        state = preprocess(pixels)

        done = False
        frame_cnt = 0
        accumulated_reward = 0
        action_str = ""
        while not done:
            frame_cnt += 1

            actions_all_agents = env.act(state_obs)

            ## - Comment out from here - ##
            init_mask = tf.ones([1, ACTION_SPACE])
            init_state = state

            q_values = restored_model.predict(
                [tf.reshape(init_state, [1] + INPUT_SHAPE), init_mask])

            action = np.argmax(q_values)
            print(q_values)
            print(
                f"Action taken: {actions_available[action]}") if action_str != f"Action taken: {actions_available[action]}" else None

            actions_all_agents[0] = action
            ## - Until here, when you want to use a Random/SimpleAgent instead - ##

            state_obs, reward, done, info, pixels = env.step2(
                actions_all_agents)
            accumulated_reward += reward[0]
            action_str = f"Action taken: {actions_available[action]}"

        time_end = np.round(time.time() - start_time, 2)

        accumulated_frames.append(frame_cnt)
        episode_rewards.append(accumulated_reward)
        save_json(accumulated_frames=accumulated_frames,
                  episode_rewards=episode_rewards)

        print(f"Running at {np.round(frame_cnt / time_end)} frames per second")


if __name__ == "__main__":
    main()
