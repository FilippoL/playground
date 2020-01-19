
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


# def provide_init(memory, action_space):
#     def callable_fn():
#         for exp in memory:
#             one_hot = np.zeros(action_space)
#             one_hot[exp[2]] = 1
#             yield exp[0][:, :, :-1], one_hot
#     return callable_fn


# def provide_next(memory, action_space):
#     def callable_fn():
#         for exp in memory:
#             yield exp[0][:, :, :-1], np.ones(action_space)
#     return callable_fn

# def provide_states(memory):
#     def callable_fn():
#         for exp in memory:
#             yield exp[0][:, :, :-1], exp[0][:, :, 1:]
#     return callable_fn


# def provide_mask(memory, action_space):
#     def callable_fn():
#         for exp in memory:
#             one_hot = np.zeros(action_space)
#             one_hot[exp[2]] = 1
#             yield one_hot, np.ones(action_space)
#     return callable_fn

def provide_init_gen(memory):
    def callable_fn():
        for exp in memory:
            yield exp[0][:, :, :-1]
    return callable_fn


def provide_next_gen(memory):
    def callable_fn():
        for exp in memory:
            yield exp[0][:, :, 1:]
    return callable_fn


def provide_action_gen(memory, action_space):
    def callable_fn():
        for exp in memory:
            yield tf.keras.utils.to_categorical(exp[2], action_space)
    return callable_fn


# def provide_ones_gen(action_space):
#     yield np.ones(action_space)


# def combine():
#     return [provide_init_gen, provide_action_gen]

# def combine2():
#     return [provide_next_gen, provide_ones_gen]

def provide_init(memory, action_space, stepsize=10):
    for exp in memory:
        one_hot = np.zeros((1, action_space))
        one_hot[0, exp[2]] = 1
        state = exp[0][:, :, :-1]
        state = state.reshape((1,)+state.shape)
        yield [state, one_hot]


def provide_next(memory, action_space):
    for exp in memory:
        state = exp[0][:, :, :-1]
        state = state.reshape((1,)+state.shape)
        yield [state, np.ones((1, action_space))]


def prioritized_experience_sampling(memory, approximator_model, target_model, batch_size, action_space, beta=0.4, a=0.6, e=0.01):

    N = len(memory)

    # q_values = np.array(list(approximator_model.predict_generator(provide_init(memory, action_space), steps=N)))
    # q_target = np.array(list(target_model.predict_generator(provide_next(memory, action_space), steps=N)))
    pre_err = target_model.predict_generator(provide_next(memory, action_space), steps=N) - approximator_model.predict_generator(provide_init(memory, action_space), steps=N)
    error_ = np.abs(list(pre_err)) + e
    probality_ = np.max(error_ ** a / (np.sum(error_) ** a), axis=1)

    inversed_probability = (1/(probality_ * N)) ** beta
    # print(len(inversed_probability))
    # print(N)
    return random.choices(memory, inversed_probability, k=batch_size)


def prioritized_experience_sampling_2(memory, approximator_model, target_model, batch_size, action_space, beta=0.4, a=0.6, e=0.01):

    N = len(memory)

    all_states = np.array(memory)[:, 0]  # extract init_states
    next_states = np.array([exp[:, :, :-1] for exp in all_states])
    init_states = np.array([exp[:, :, 1:] for exp in all_states])

    del all_states
    init_mask = tf.ones([N, action_space])

    q_values = approximator_model.predict([init_states, init_mask])
    q_target = target_model.predict([next_states, init_mask])

    error_ = np.abs(q_target - q_values) + e
    del next_states
    del init_states
    del q_target
    del q_values
    probality_ = np.max(error_ ** a / (np.sum(error_) ** a), axis=1)

    inversed_probability = (1/(probality_ * N)) ** beta

    return random.choices(memory, inversed_probability, k=batch_size)


def prioritized_experience_sampling_3(memory, batch_size, beta=0.4, a=0.6, e=0.01):

    N = len(memory)

    error_ = np.abs([exp[3] for exp in memory]) + e

    probality_ = error_ ** a / (np.sum(error_) ** a)

    inversed_probability = (1/(probality_ * N)) ** beta

    indices = random.choices(range(N), inversed_probability, k=batch_size)
    return indices

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

    picked_indices = random.choices(range(len(memory)-1), inversed_probability[:-1], k=batch_size)
    initial_states_result = memory_array[picked_indices]
    next_states_result = memory_array[np.array(picked_indices)+1]

    return list(zip(initial_states_result, next_states_result))


