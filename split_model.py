import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from matplotlib import pyplot
import numpy as np
import matplotlib
matplotlib.use('Agg')


def split_birdnet_keras_model(model):
    # split birdnet kears model's layer into spectrogram and the rest of con layers.
    # getting the information of layers
    for i in range(len(model.layers)):
        layer = model.layers[i]
        print(i, layer.name, layer.output.shape)

    # print('model.layers[1].output', model.layers[1].output.shape)
    # redefine model to output right after the first hidden layer
    model1 = Model(inputs=model.inputs, outputs=model.layers[1].output)
    model1.summary()
    model2 = Model(inputs=Input(tensor=model.layers[1].output), outputs=model.output)
    model2.summary()
    # Create dummy data with input shape (1,144000)
    dummy_data = tf.random.uniform(shape=(1, 144000))
    p1 = model1.predict(dummy_data)
    print(p1.shape)
    pyplot.imshow(p1[0, :, :, :])
    pyplot.show()
    '''
    outputs = [model.layers[i + 1].output for i in p1]
    model = Model(inputs=model.inputs, outputs=outputs)'''
    # maybe I have to save models again and load them
    #  preparing a pipline for reading the data
    # preparing shap function and visualization
    #
    return model1, model2


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


#def wave_data_loader(path):
