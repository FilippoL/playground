
import numpy as np
import random
from collections import deque
import tensorflow as tf
import multiprocessing


def random_sampling(memory, K):
    return random.sample(memory, k=K)


def uniform_sampling(memory, K, early_stop=8):
    balance = K//2
    nonrewarding_experiences = deque(maxlen=balance)
    rewarding_experiences = deque(maxlen=balance)
    first_condition = False
    secon_condition = False
    batch = []
    while True:
        first_condition = len(nonrewarding_experiences) >= balance
        secon_condition = len(rewarding_experiences) >= balance
        experience_batch = random.sample(memory, k=K)
        for exp in experience_batch:
            if exp[1] == 0 and not first_condition:
                nonrewarding_experiences.append(exp)

            if exp[1] != 0 and not secon_condition:
                rewarding_experiences.append(exp)

        early_stop -= 1

        if first_condition and secon_condition:
            break

        if early_stop == 0:
            break
    batch.extend(nonrewarding_experiences)
    batch.extend(rewarding_experiences)
    remaining = K-len(batch)
    batch.extend(random.sample(memory, k=remaining))
    return batch


def prioritized_experience_sampling(memory, approximator_model, target_model, batch_size, action_space, gamma=0.99 ,beta=0.4, a=0.7, e=0.1):

    N = len(memory)
    memory_copy = np.array(memory)
    all_states = memory_copy[:, 0]  # extract init_states
    next_states = np.array([exp[:, :, :-1] for exp in all_states])
    init_states = np.array([exp[:, :, 1:] for exp in all_states])
    rewards = np.array(memory_copy[:, 1])
    actions = np.array(memory_copy[:, 2])

    init_mask = tf.keras.utils.to_categorical(actions, action_space)
    next_mask = np.ones([N, action_space])

    q_init = approximator_model.predict([init_states, init_mask])
    q_next = target_model.predict([next_states, next_mask])

    q_init_max = np.sum(q_init, axis=1)
    q_next_max = np.max(q_next, axis=1)

    error_ = np.abs(rewards + gamma*q_next_max - q_init_max) + e
    probality_ = error_ ** a / (np.sum(error_) ** a)

    inversed_probability = (1/(probality_ * N)) ** beta
    indices = random.choices(range(N), probality_, k=batch_size)
    batch = random.choices(memory, probality_, k=batch_size)
    importance = np.array(inversed_probability[indices], dtype=np.float32)
    return batch, importance


def prioritized_experience_sampling_3(memory, batch_size, beta=0.4, a=0.6, e=0.01):

    N = len(memory)

    error_ = np.abs([exp[3] for exp in memory]) 
    for exp, err in zip(memory, error_):
        exp[3] = err

    error_ = error_ + e
    probality_ = error_ ** a / (np.sum(error_) ** a)

    inversed_probability = (1/(probality_ * N)) ** beta
    indices = random.choices(range(N), probality_, k=batch_size)
    importance = inversed_probability[indices]
    max_td_error = max(error_)
    return indices, importance, max_td_error

# def prioritized_experience_sampling_pommerman(memory, approximator_model, target_model, batch_size, action_space, beta=0.4, a=0.6, e=0.01):

#     N = len(memory)

#     error_ = np.abs(q_target - q_values) + e
#     probality_ = np.max(error_ ** a / (np.sum(error_) ** a), axis=1)

#     inversed_probability = (1/(probality_ * N)) ** beta

#     picked_indices = random.choices(range(len(memory)-1), inversed_probability[:-1], k=batch_size)
#     initial_states_result = memory_array[picked_indices]
#     next_states_result = memory_array[np.array(picked_indices)+1]

#     return list(zip(initial_states_result, next_states_result))


def prioritized_experience_sampling_pommerman(memory, approximator_model, target_model, batch_size, action_space, beta=0.4, a=0.6, e=0.01):

    N = len(memory)

    memory_array = np.array(memory)
    # all_states = memory_array[:, 0]  # extract init_states
    init_states = np.array([exp for exp in memory_array[:, 0]])
    next_states = np.roll(init_states, -1)  # i+1
    next_states[-1] = init_states[-1]  # Last one is the same

    init_states = init_states.reshape(init_states.shape+(1,))
    next_states = next_states.reshape(next_states.shape+(1,))

    init_mask = np.ones([N, action_space])

    q_values = approximator_model.predict([init_states, init_mask])
    q_target = target_model.predict([next_states, init_mask])

    error_ = np.abs(q_target - q_values) + e
    probality_ = np.max(error_ ** a / (np.sum(error_) ** a), axis=1)

    inversed_probability = (1/(probality_ * N)) ** beta

    picked_indices = random.choices(range(len(memory)-1), probality_[:-1], k=batch_size)
    initial_states_result = memory_array[picked_indices]
    next_states_result = memory_array[np.array(picked_indices)+1]

    return list(zip(initial_states_result, next_states_result))
