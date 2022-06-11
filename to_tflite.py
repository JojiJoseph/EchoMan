import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Dropout, Reshape
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2

num_labels = 8

input_shape = (124, 10, 1)
input_shape = [input_shape[0], input_shape[1],1]
filters = 64
weight_decay = 1e-4
regularizer = l2(weight_decay)
final_pool_size = (int(input_shape[0]/2), int(input_shape[1]/2))
# Model layers
# Input pure conv2d
inputs = Input(shape=input_shape)
x = Conv2D(filters, (10,4), strides=(2,2), padding='same', kernel_regularizer=regularizer)(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(rate=0.2)(x)

# First layer of separable depthwise conv2d
# Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Second layer of separable depthwise conv2d
x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Third layer of separable depthwise conv2d
x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Fourth layer of separable depthwise conv2d
x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Reduce size and apply final softmax
x = Dropout(rate=0.4)(x)

x = AveragePooling2D(pool_size=final_pool_size)(x)
x = Flatten()(x)
outputs = Dense(num_labels, activation='softmax')(x)

# Instantiate model.
model = Model(inputs=inputs, outputs=outputs)

model.load_weights("./ds_cnn.hdf5")
model.summary()
# convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8
tflite_model = converter.convert()
print(dir(tflite_model))
print(len(tflite_model))

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
