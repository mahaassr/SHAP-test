# Mute Tensorflow
import os

import split_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras as k
from keras import layers as l
from keras.models import load_model
from keras.models import Model
from matplotlib import pyplot
from split_model import *
import numpy as np

# Model name/path
MODEL_PATH = 'BirdNET_2022_3K_V2.2.h5'


# Define custom Layer
class LinearSpecLayer(l.Layer):

    def __init__(self, sample_rate=48000, spec_shape=(64, 384), frame_step=374, frame_length=512, fmin=250, fmax=15000,
                 data_format='channels_last', **kwargs):
        super(LinearSpecLayer, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.spec_shape = spec_shape
        self.data_format = data_format
        self.frame_step = frame_step
        self.frame_length = frame_length
        self.fmin = fmin
        self.fmax = fmax

    def build(self, input_shape):
        print('input_shape', input_shape)
        self.mag_scale = self.add_weight(name='magnitude_scaling',
                                         initializer=k.initializers.Constant(value=1.23),
                                         trainable=True)

        super(LinearSpecLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return tf.TensorShape((None, self.spec_shape[0], self.spec_shape[1], 1))
        else:
            return tf.TensorShape((None, 1, self.spec_shape[0], self.spec_shape[1]))

    def call(self, inputs, training=None):

        # Normalize values between 0 and 1
        print('inputss', inputs.shape)
        inputs = tf.math.subtract(inputs, tf.math.reduce_min(inputs, axis=1, keepdims=True))
        inputs = tf.math.divide(inputs, tf.math.reduce_max(inputs, axis=1, keepdims=True) + 0.000001)

        # Perform STFT
        spec = tf.signal.stft(inputs,
                              self.frame_length,
                              self.frame_step,
                              fft_length=self.frame_length,
                              window_fn=tf.signal.hann_window,
                              pad_end=False,
                              name='stft')
        print('spec', spec.shape)

        # Cast from complex to float
        spec = tf.dtypes.cast(spec, tf.float32)

        # Only keep bottom half of spectrum
        spec = spec[:, :, :self.frame_length // 4]
        # spec = spec[:, :, 2:self.frame_length // 4 + 2]
        print('spec', spec.shape)
        # Convert to power spectrogram
        spec = tf.math.pow(spec, 2.0)
        print('spec', spec.shape)

        # Convert magnitudes using nonlinearity
        spec = tf.math.pow(spec, 1.0 / (1.0 + tf.math.exp(self.mag_scale)))
        print('spec', spec.shape)
        # Swap axes to fit input shape
        spec = tf.transpose(spec, [0, 2, 1])

        # Add channel axis
        if self.data_format == 'channels_last':
            spec = tf.expand_dims(spec, -1)
        else:
            spec = tf.expand_dims(spec, 1)
        print('pectrogram', spec.shape)
        return spec

    '''
    def get_config(self):
        config = {'data_format': self.data_format,
                  'sample_rate': self.sample_rate,
                  'spec_shape': self.spec_shape,
                  'frame_step': self.frame_step,
                  'fmin': self.fmin,
                  'fmax': self.fmax,
                  'frame_length': self.frame_length}

        base_config = super(LinearSpecLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))'''


def layers_output(model):
    layer_names = [layer.name for layer in model.layers]
    layer_outputs = [layer.output for layer in model.layers]
    feature_map_model = tf.keras.models.Model(input=model.input, output=layer_outputs)
    feature_maps = feature_map_model.predict(input)
    for layer_name, feature_map in zip(layer_names, feature_maps):
        if len(feature_map.shape) == 4:
            k = feature_map.shape[-1]
            size = feature_map.shape[1]
            for i in range(k):
                feature_image = feature_map[0, :, :, i]
                feature_image -= feature_image.mean()
                feature_image /= feature_image.std()
                feature_image *= 64
                feature_image += 128

            # Load Keras model


model = load_model(MODEL_PATH,
                   custom_objects={'LinearSpecLayer': LinearSpecLayer})

# Print model summary
# model.summary()

# Create dummy data with input shape (1,144000)
dummy_data = tf.random.uniform(shape=(1, 144000))

# Run inference
p = model.predict(dummy_data)

# Print prediction
print(p, p.shape)

split_model.split_birdnet_keras_model(model)
