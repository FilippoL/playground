
import numpy as np
import random
from collections import deque
import tensorflow as tf


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


def prioritized_experience_sampling(memory, approximator_model, target_model, batch_size, action_space, beta=0.4, a=0.6, e=0.01):

    N = len(memory)

    all_states = np.array(memory)[:, 0]  # extract init_states
    next_states = np.array([exp[:, :, :-1] for exp in all_states])
    init_states = np.array([exp[:, :, 1:] for exp in all_states])

    init_mask = tf.ones([N, action_space])

    q_values = approximator_model.predict([init_states, init_mask])
    q_target = target_model.predict([next_states, init_mask])

    error_ = np.abs(q_target - q_values) + e
    probality_ = np.max(error_ ** a / (np.sum(error_) ** a), axis=1)

    inversed_probability = (1/(probality_ * N)) ** beta

    return random.choices(memory, inversed_probability, k=batch_size)


def prioritized_experience_sampling_pommerman(memory, approximator_model, target_model, batch_size, action_space, beta=0.4, a=0.6, e=0.01):

    N = len(memory)

    memory_array = np.array(memory)
    # all_states = memory_array[:, 0]  # extract init_states
    init_states = np.array([exp for exp in memory_array[:, 0]])
    next_states = np.roll(init_states, -1) # i+1
    next_states[-1] = init_states[-1] # Last one is the same

    init_states = init_states.reshape(init_states.shape+(1,))
    next_states = next_states.reshape(next_states.shape+(1,))

    init_mask = np.ones([N, action_space])

    q_values = approximator_model.predict([init_states, init_mask])
    q_target = target_model.predict([next_states, init_mask])

    error_ = np.abs(q_target - q_values) + e
    probality_ = np.max(error_ ** a / (np.sum(error_) ** a), axis=1)

    inversed_probability = (1/(probality_ * N)) ** beta

    picked_indices = random.choices(range(len(memory)-1), inversed_probability[:-1], k=batch_size)
    initial_states_result = memory_array[picked_indices]
    next_states_result = memory_array[np.array(picked_indices)+1]
    
    return list(zip(initial_states_result, next_states_result))
