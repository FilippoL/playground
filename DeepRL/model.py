import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

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
        output = value_out + tf.math.subtract(advantage_out, tf.reduce_mean(advantage_out, axis=1, keepdims=True))
        out_q_values = tf.multiply(output, mask)
    # out_q_values = tf.reshape(out_q_values, [1,-1])
    model = models.Model(inputs=[input, mask], outputs=out_q_values)
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    return model