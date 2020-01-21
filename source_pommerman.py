import datetime
import os
import platform
import random
import time
from collections import deque

import numpy as np
import psutil
import tensorflow as tf

import pommerman
from DeepRL.model import create_model_faithful
from DeepRL.sampling import prioritized_experience_sampling_3
from DeepRL.utils import exploration_linear_decay
from DeepRL.utils import plot_to_image, standardize, image_grid_pommerman, initialize_memory_pommerman, \
    train_batch_pommerman
from pommerman import agents, constants

# =========== HELPER FUNCTIONS =========== #
take_sample = prioritized_experience_sampling_3


def main():
    if platform.system() == 'Darwin':
        print("MacBook Pro user detected. U rule.")
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    process = psutil.Process(os.getpid())

    # MARK: - Create the environment
    agent_list = [
        agents.RandomAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
    ]

    env = pommerman.make('PommeFFACompetition-v0',
                         agent_list, render_mode='human')

    # MARK: - Allowing to save the model
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(
        ".",
        "models",
        now,
        "-{epoch:04d}.ckpt"
    )
    # MARK: - Log for tensorboard
    log_dir = os.path.join(
        "logs",
        now,
    )
    tensorflow_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, profile_batch=5, histogram_freq=1)
    file_writer_rewards = tf.summary.create_file_writer(log_dir + "/metrics")

    # =========== (HYPER)PARAMETERS AND VARIABLES =========== #
    LIST_SIZE = 60000
    D = deque(maxlen=LIST_SIZE)
    DISCOUNT_RATE = 0.99
    TAU = 0
    MAX_TAU = 5000
    ACTION_SPACE = env.action_space.n
    TIME_CHANNELS_SIZE = 1
    INPUT_SHAPE = list(env.get_observation_space()) + [TIME_CHANNELS_SIZE]
    BATCH_SIZE = 256
    N = BATCH_SIZE * 4
    N_EPISODES = 1000
    TD_ERROR_DEFAULT = 0
    print(f"Pixel space of the game {INPUT_SHAPE}")

    approximator_model = create_model_faithful(INPUT_SHAPE, ACTION_SPACE)
    target_model = create_model_faithful(INPUT_SHAPE, ACTION_SPACE)

    # ================== CONTINUE TRAIN FROM LOADED MODEL ==================== #
    # MODEL_PATH = "models/20200119-121818"
    # latest = tf.train.latest_checkpoint(MODEL_PATH)
    # print(f"Loading model from {latest}")
    # approximator_model.load_weights(latest)
    # target_model.load_weights(latest)
    # ======================================================================== #

    # ===== INITIALISATION ======
    acc_nonzeros = []
    actions_available = [str(action).split(".")[1] for action in constants.Action]

    env, D = initialize_memory_pommerman(env, D, N, TD_ERROR_DEFAULT)

    for episode in range(N_EPISODES):
        start_time = time.time()

        if TAU >= MAX_TAU:
            TAU = 0
            # Copy the weights from policy model to target model
            target_model.set_weights(approximator_model.get_weights())
            print("=" * 35 + "> Updated weights")

        EXPLORATION_RATE = exploration_linear_decay(episode, minimal_exploration_rate=0.01)
        print(
            f"Running episode {episode} with exploration rate: {EXPLORATION_RATE}")

        # Initial step for the episode
        state_obs = env.reset()
        actions = env.act(state_obs)
        initial_observation, reward, done, info, pixels = env.step2(
            actions, render=True)
        state = standardize(pixels)
        done = False
        episode_rewards = []
        frame_cnt = 0
        acc_actions = []
        action_str = ""

        while not done:
            # https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-standardizeing-for-deep-q-networks-on-atari-2600-games/
            frame_cnt += 1
            TAU += 1

            actions_all_agents = env.act(state_obs)
            action = actions_all_agents[0]

            if not random.choices((True, False), (EXPLORATION_RATE, 1 - EXPLORATION_RATE))[0]:
                # Greedy action
                init_mask = tf.ones([1, ACTION_SPACE])
                init_state = state
                q_values = approximator_model.predict(
                    [tf.reshape(init_state, [1] + INPUT_SHAPE), init_mask])
                action = np.argmax(q_values)
                actions_all_agents[0] = action

            print(action_str) if action_str != f"Action taken: {actions_available[action]}" else None

            state_obs, reward, done, info, pixels = env.step2(
                actions_all_agents)

            acc_actions.append(action)
            if done:
                with file_writer_rewards.as_default():
                    tf.summary.histogram('action_taken', acc_actions, step=episode)

            episode_rewards.append(reward[0])
            D.append([standardize(pixels), reward[0], actions_all_agents[0], TD_ERROR_DEFAULT])
            action_str = f"Action taken: {actions_available[action]}"

        history, episode_nonzero_reward_states, experience_batch, next_q_values, td_error = \
            train_batch_pommerman(D,
                                  approximator_model,
                                  target_model,
                                  ACTION_SPACE,
                                  BATCH_SIZE,
                                  tensorflow_callback,
                                  DISCOUNT_RATE)

        for idx, exp in enumerate(experience_batch):
            exp[0][3] = td_error[idx]

        # Wrap up
        loss = history.history.get("loss", [0])[0]
        time_end = np.round(time.time() - start_time, 2)
        memory_usage = process.memory_info().rss
        print(f"Current memory consumption is {memory_usage}")
        print(
            f"Loss of episode {episode} is {loss} and took {time_end} seconds")
        random_experience_idx = random.choice(range(len(experience_batch) - 1))
        random_experience = experience_batch[random_experience_idx][0]
        random_experience_next = experience_batch[random_experience_idx][1]

        # print(tmp.shape)
        episode_image = plot_to_image(
            image_grid_pommerman(random_experience, random_experience_next, [action for action in constants.Action]))
        with file_writer_rewards.as_default():
            tf.summary.scalar('episode_rewards', np.sum(
                episode_rewards), step=episode)
            tf.summary.scalar('episode_loss', loss, step=episode)
            tf.summary.scalar('episode_time_in_secs', time_end, step=episode)
            tf.summary.scalar('episode_nr_frames', frame_cnt, step=episode)
            tf.summary.scalar('episode_exploration_rate',
                              EXPLORATION_RATE, step=episode)
            tf.summary.scalar('episode_mem_usage', memory_usage, step=episode)
            tf.summary.scalar('episode_frames_per_sec', np.round(
                frame_cnt / time_end, 2), step=episode)
            tf.summary.histogram('q-values', next_q_values, step=episode)

            tf.summary.scalar('episode_mem_usage_in_GB', np.round(memory_usage / 1024 / 1024 / 1024), step=episode)
            tf.summary.image('episode_example_state', episode_image, step=episode)
            if (episode + 1) % 5 == 0:
                acc_nonzeros.append(episode_nonzero_reward_states)
                tf.summary.histogram(
                    'episode_nonzero_reward_states', acc_nonzeros, step=(episode + 1) // 5)
            else:
                acc_nonzeros.append(episode_nonzero_reward_states)
        if (episode + 1) % 50 == 0:
            model_target_dir = checkpoint_path.format(epoch=episode)
            approximator_model.save_weights(model_target_dir)
            print(f"Model was saved under {model_target_dir}")


# TODO: [x] Simplify the loss function
# TODO: [x] Apply the reward
# TODO: [x] Rethink memory handling
# TODO: [x] Proper memory initialisation
# TODO: [ ] Refactoring and restructuring


if __name__ == "__main__":
    main()
