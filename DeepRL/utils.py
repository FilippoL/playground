from collections import deque
import numpy as np

class Memory():
    def _init_(self, max_size, time_steps):
        self.full_memory = deque(maxlen = max_size)
    
    def add(self, experience):
        self.full_memory.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index =  np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]

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




def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def standardize(img):
    return img/255


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    return standardize(to_grayscale(downsample(img)))