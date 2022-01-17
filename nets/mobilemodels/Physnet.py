import tensorflow as tf
import numpy as np
from typeguard import typechecked
from typing import Union, Callable, Iterable
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend
import math

import keras_utils as conv_utils

import h5py

import json

print(tf.__version__)
backend.set_image_data_format('channels_first')
class AdaptivePooling3D(tf.keras.layers.Layer):
    """Parent class for 3D pooling layers with adaptive kernel size.
    This class only exists for code reuse. It will never be an exposed API.
    Args:
      reduce_function: The reduction method to apply, e.g. `tf.reduce_max`.
      output_size: An integer or tuple/list of 3 integers specifying (pooled_dim1, pooled_dim2, pooled_dim3).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
    """

    @typechecked
    def __init__(
        self,
        reduce_function: Callable,
        output_size: Union[int, Iterable[int]],
        data_format=None,
        **kwargs,
    ):
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.reduce_function = reduce_function
        self.output_size = conv_utils.normalize_tuple(output_size, 3, "output_size")
        super().__init__(**kwargs)

    def call(self, inputs, *args):
        h_bins = self.output_size[0]
        w_bins = self.output_size[1]
        d_bins = self.output_size[2]
        if self.data_format == "channels_last":
            split_cols = tf.split(inputs, h_bins, axis=1)
            split_cols = tf.stack(split_cols, axis=1)
            split_rows = tf.split(split_cols, w_bins, axis=3)
            split_rows = tf.stack(split_rows, axis=3)
            split_depth = tf.split(split_rows, d_bins, axis=5)
            split_depth = tf.stack(split_depth, axis=5)
            out_vect = self.reduce_function(split_depth, axis=[2, 4, 6])
        else:
            split_cols = tf.split(inputs, h_bins, axis=2)
            split_cols = tf.stack(split_cols, axis=2)
            split_rows = tf.split(split_cols, w_bins, axis=4)
            split_rows = tf.stack(split_rows, axis=4)
            split_depth = tf.split(split_rows, d_bins, axis=6)
            split_depth = tf.stack(split_depth, axis=6)
            out_vect = self.reduce_function(split_depth, axis=[3, 5, 7])
        return out_vect

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    self.output_size[0],
                    self.output_size[1],
                    self.output_size[2],
                    input_shape[4],
                ]
            )
        else:
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    input_shape[1],
                    self.output_size[0],
                    self.output_size[1],
                    self.output_size[2],
                ]
            )

        return shape

    def get_config(self):
        config = {
            "output_size": self.output_size,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return {**base_config, **config}

def neg_Pearson_Loss(predictions, targets):
    '''
    :param predictions: inference value of trained model
    :param targets: target label of input data
    :return: negative pearson loss
    '''
    print(predictions,targets)
    rst = 0
    # Pearson correlation can be performed on the premise of normalization of input data
    predictions = (predictions - tf.math.reduce_mean(predictions)) / tf.math.reduce_std(predictions)
    targets = (targets - tf.math.reduce_mean(targets)) / tf.math.reduce_std(targets)

    for i in range(1):
        sum_x = tf.math.reduce_sum(predictions[i])  # x
        sum_y = tf.math.reduce_sum(targets[i])  # y
        sum_xy = tf.math.reduce_sum(predictions[i] * targets[i])  # xy
        sum_x2 = tf.math.reduce_sum(tf.math.pow(predictions[i], 2))  # x^2
        sum_y2 = tf.math.reduce_sum(tf.math.pow(targets[i], 2))  # y^2
        N = predictions.shape[1]
        pearson = (N * sum_xy - sum_x * sum_y) / (
            tf.math.sqrt((N * sum_x2 - tf.math.pow(sum_x, 2)) * (N * sum_y2 - tf.math.pow(sum_y, 2))))

        rst += 1 - pearson

    rst = rst / predictions.shape[0]
    return rst


@tf.keras.utils.register_keras_serializable(package="Addons")
class AdaptiveAveragePooling3D(AdaptivePooling3D):
    """Average Pooling with adaptive kernel size.
    Args:
      output_size: An integer or tuple/list of 3 integers specifying (pooled_depth, pooled_height, pooled_width).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`.
    Input shape:
      - If `data_format='channels_last'`:
        5D tensor with shape `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`.
      - If `data_format='channels_first'`:
        5D tensor with shape `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
    Output shape:
      - If `data_format='channels_last'`:
        5D tensor with shape `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`.
      - If `data_format='channels_first'`:
        5D tensor with shape `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`.
    """

    @typechecked
    def __init__(
        self, output_size: Union[int, Iterable[int]], data_format=None, **kwargs
    ):
        super().__init__(tf.reduce_mean, output_size, data_format, **kwargs)

class Model(tf.Module):

    def __init__(self):
        self.model = tf.keras.Sequential([

            # EncoderBlock

            # ConvBlock1
            tf.keras.layers.ZeroPadding3D(padding=(0,2,2),input_shape=(3,32,128,128)),
            tf.keras.layers.Conv3D(filters=16,kernel_size=(1,5,5),strides=(1,1,1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # #MaxPool3d1r
            tf.keras.layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2)),
            #
            # ConvBlock2
            tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)),
            tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # ConvBlock3
            tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            # MaxPool3d2
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)),

            # ConvBlock4
            tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # ConvBlock5
            tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            # MaxPool3d3
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)),

            # ConvBlock6
            tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # ConvBlock7
            tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            # MaxPool3d4
            tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2)),
            #
            # ConvBlock8
            tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            # ConvBlock9
            tf.keras.layers.ZeroPadding3D(padding=(1, 1, 1)),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            #
            # DecoderBlock
            #
            #DeConvBlock1
            tf.keras.layers.Convolution3DTranspose(filters=64,kernel_size=(4,1,1),strides=(2,1,1),padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ELU(),

            # DeConvBlock2
            tf.keras.layers.Convolution3DTranspose(filters=64, kernel_size=(4, 1, 1), strides=(2, 1, 1),padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ELU(),

            #
            # AdaptivePooling
            #
            AdaptivePooling3D(tf.reduce_max,(32,1,1)),
            #
            #Conv3D
            #
            tf.keras.layers.Conv3D(1,kernel_size=(1,1,1),strides=(1,1,1),padding='same')
        ])

        self.model.compile( optimizer='adam', loss=neg_Pearson_Loss )

        @tf.function(input_signature=[
            tf.TensorSpec([None, 32, 128, 128, 3], tf.float32),
            tf.TensorSpec([None, 1], tf.float32),
        ])
        def train(self, x, y):
            with tf.GradientTape() as tape:
                predictions = self.model(x)
                loss = self.model.loss(y, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            result = {"loss":loss}
            return result

        @tf.function(input_signature=[
            tf.TensorSpec([None,32,128,128,3], tf.float32),
        ])
        def infer(self, x):
            output = self.model(x)
            return {
                "output" : output
            }

        @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
        def save(self, checkpoint_path):
            tensor_names = [weight.name for weight in self.model.weights]
            tensors_to_save = [weight.read_value() for weight in self.model.weights]
            tf.raw_ops.Save(
                filename=checkpoint_path, tensor_names=tensor_names,
                data=tensors_to_save, name='save')
            return {
                "checkpoint_path": checkpoint_path
            }

        @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
        def restore(self, checkpoint_path):
            restored_tensors = {}
            for var in self.model.weights:
                restored = tf.raw_ops.Restore(
                    file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
                    name='restore')
                var.assign(restored)
                restored_tensors[var.name] = restored
            return restored_tensors

class DataLoader(Sequence):
    def __init__(self, x_set,y_set, batch_size, *args, **kwargs):
        super(DataLoader, self).__init__(*args,**kwargs)
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

		# batch 단위로 직접 묶어줘야 함
    def __getitem__(self, idx):
				# sampler의 역할(index를 batch_size만큼 sampling해줌)
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_x = [self.x[i] for i in indices]
        batch_y = [self.y[i] for i in indices]

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))

with open('../../params.json') as f:
    jsonObject = json.load(f)
    params = jsonObject.get("params")
    model_params = jsonObject.get("model_params")
    save_root_path = params["save_root_path"]
    model_name = model_params["name"]
    dataset_name = params["dataset_name"]
    option="train"

hpy_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_" + option + ".hdf5", "r")
video_data = []
label_data = []
for key in hpy_file.keys():
     video_data.extend(hpy_file[key]['preprocessed_video'])
     label_data.extend(hpy_file[key]['preprocessed_label'])
hpy_file.close()

train_loader = DataLoader(video_data,label_data,128)
valid_loader = DataLoader(video_data,label_data,128)
test_loader = DataLoader(video_data,label_data,128)

# trian
model = Model()
# model.model.build(input_shape=(1,32,128,128,3))
# model.model.summary()
model.model.fit(train_loader, validation_data=valid_loader, epochs=10,
					workers=4)# multi로 처리할 개수
model.evaluate(test_loader)