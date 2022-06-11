from cv2 import ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE
from numpy.core.records import record
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import sounddevice as sd
import numpy as np
from model import model_cnn as model
import random

# def get_spectrogram(waveform):
#   # Zero-padding for an audio waveform with less than 16,000 samples.
#   input_len = 16000
#   waveform = waveform[:input_len]
#   zero_padding = tf.zeros(
#       [16000] - tf.shape(waveform),
#       dtype=tf.float32)
#   # Cast the waveform tensors' dtype to float32.
#   waveform = tf.cast(waveform, dtype=tf.float32)
#   # Concatenate the waveform with `zero_padding`, which ensures all audio
#   # clips are of the same length.
#   equal_length = tf.concat([waveform, zero_padding], 0)
#   # Convert the waveform to a spectrogram via a STFT.
#   spectrogram = tf.signal.stft(
#       equal_length, frame_length=255, frame_step=128)
#   # Obtain the magnitude of the STFT.
#   spectrogram = tf.abs(spectrogram)
#   # Add a `channels` dimension, so that the spectrogram can be used
#   # as image-like input data with convolution layers (which expect
#   # shape (`batch_size`, `height`, `width`, `channels`).
#   spectrogram = spectrogram[..., tf.newaxis]
#   return spectrogram

def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128,fft_length=None,window_fn=tf.signal.hann_window)
  # Obtain the magnitude of the STFT.
  spectrograms = tf.abs(spectrogram)
  num_spectrogram_bins = spectrogram.shape[-1]
  lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, 40 
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix( num_mel_bins, num_spectrogram_bins,
                                                                           16000,
                                                                           lower_edge_hertz, upper_edge_hertz)
  mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
  # Compute MFCCs from log_mel_spectrograms and take the first 13.
  mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :10]
  # mfccs = tf.reshape([mfccs,16000, 10, 1])
  # next_element['audio'] = mfccs
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  mfccs = mfccs[..., tf.newaxis]

  return mfccs

input_shape = (124, 129, 1)
input_shape = (124, 10, 1)
num_labels = 8

# model = models.Sequential([
#     layers.Input(shape=input_shape),
#     # Downsample the input.
#     layers.Resizing(32, 32),
#     # Normalize.
#     layers.Normalization(),
#     layers.Conv2D(32, 3, activation='relu'),
#     layers.Conv2D(64, 3, activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Dropout(0.25),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(num_labels),
# ])
# # model = tf.keras.models.load_model("./simple_audio.hdf5")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Dropout, Reshape
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2

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
model.summary()

# model.load_weights("./simple_audio.hdf5")
model.load_weights("ds_cnn.hdf5")


# def predict(interpreter, data):

#   # Get input and output tensors.
#   input_details = interpreter.get_input_details()
#   output_details = interpreter.get_output_details()
  
#   # Test the model on input data.
#   input_shape = input_details[0]['shape']

#   input_data = np.array(data, dtype=np.int8)
#   output_data = np.empty_like(data)
  
#   interpreter.set_tensor(input_details[0]['index'], input_data[i:i+1, :])
#   interpreter.invoke()
#   # The function `get_tensor()` returns a copy of the tensor data.
#   # Use `tensor()` in order to get a pointer to the tensor.
#   output_data[i:i+1, :] = interpreter.get_tensor(output_details[0]['index'])
#   return output_data

interpreter = tf.lite.Interpreter(model_path="./model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)
input_scale, input_zero_point = input_details[0]["quantization"]
# model.predict()

print("Enter any word to start recording. q for quit")
# words = ["DOWN", "UP", "STOP", "YES", "GO", "NO", "RIGHT" , "LEFT"]
words = ['GO', 'LEFT', 'STOP', 'UP', 'RIGHT', 'YES', 'DOWN', 'NO']

LISTENING = 0
RECORDING = 1
thresh_voice = 0.1


thresh_silence = 0.08
output = []
import time
sample_rate = 16_000
import librosa
block_size = 8_000

playback_starttime = None
state = LISTENING
time_elapsed = 0

playing = False
last_detected_text = "Please say 'Yes' to start!"
speed = 200

frame_id = 0
fps = 30

direction = [-1, 0]
running = False

start_time = 0

game_over = False
def callback(indata, frames, time_, status):
    if game_over:
        return
    global state, playback_starttime, output, last_detected_text
    global direction, running, playing, start_time
    global input_scale, input_zero_point, interpreter

    if state == LISTENING:
        # print(np.max(np.abs(indata[:, 0])))
        if np.max(np.abs(indata[:, 0])) > thresh_voice:
            state = RECORDING
            output.extend(indata[:, 0])
    elif state == RECORDING:
        output.extend(indata[:, 0])
        # if np.max(np.abs(indata[:, 0])) < thresh_silence:
        if len(output) > 8_000:# or np.max(np.abs(indata[:, 0])) < thresh_silence:
            playback_starttime = time.time()
            playback_duration = len(output)/sample_rate
            state = LISTENING
            output = np.array(output)
            spect = get_spectrogram(output)
            # print(spect.shape)
            # print(words[np.argmax(model.predict(np.array([spect])))])
            input_data = np.array(np.array([spect]), dtype=np.float32)
            last_detected_text = words[np.argmax(model.predict(np.array([spect])))]
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            word_index = np.argmax(interpreter.get_tensor(output_details[0]['index']))
            last_detected_text = words[word_index]
            if last_detected_text == "GO":
                running = True
            if last_detected_text == "STOP":
                running = False
            if last_detected_text == "LEFT":
                direction = [-1, 0]
            if last_detected_text == "RIGHT":
                direction = [1, 0]
            if last_detected_text == "UP":
                direction = [0, -1]
            if last_detected_text == "DOWN":
                direction = [0, 1]
            if last_detected_text == "YES":
                playing = True
                if not playing:
                    start_time = time.time()
                #print("Started", playing)
            output = []

            # output = librosa.effects.pitch_shift(
            #     np.array(output), sr=sample_rate, n_steps=pitch_shift).tolist()
            # sd.play(output)


input_stream = sd.InputStream(
    channels=1, dtype='float32', callback=callback, blocksize=block_size, samplerate=sample_rate)
input_stream.start()

last_time = time.time()
import cv2

frame1 = cv2.imread("./pac_1.png")
frame1 = cv2.resize(frame1, (64,64))
frame2 = cv2.imread("./pac_2.png")
frame2 = cv2.resize(frame2, (64,64))
frame3 = cv2.imread("./pac_3.png")
frame3 = cv2.resize(frame3, (64,64))
# print(frame)
# cv2.imshow("", frame1)
position = [400, 400]

food_position = [random.randint(200,600), random.randint(200, 600)]
food_color = [random.randint(50,255), random.randint(50, 255), random.randint(50, 255)]

start_time = time.time()
food_count = 0
score = 0
while True:
    img = np.zeros((800,800,3), np.uint8)
    # print(playing)
    if not playing:
        if food_count == 10: # 10
            cv2.putText(img, "You won!", (20,50), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))
            cv2.putText(img, f"Score = {score}", (20,80), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))
            # cv2.putText(img, "Speak 'Yes' to restart the game!", (20,100), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))
        elif  start_time and time.time() - start_time > 300: # 300
            cv2.putText(img, "Game Over!", (20,50), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))
            # cv2.putText(img, "Speak 'Yes' to restart the game!", (20,100), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))
        else:
            cv2.putText(img, "Speak 'Yes' to start the game!", (20,60), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))
        # continue
    else:
        cv2.putText(img, last_detected_text, (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))
        cv2.putText(img, f"Remaining time: {300-int(time.time() - start_time)}", (500, 20), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255))
        cv2.putText(img, f"Food eaten: {food_count}/10", (500, 40), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255))
    delta = 1/fps
    if playing:
        if running:
            position[0] += direction[0]*speed*delta
            position[1] += direction[1]*speed*delta
        if position[0] < -20:
            position[0] = 820
        if position[1] < -20:
            position[1] = 820
        if position[0] > 820:
            position[0] = -20
        if position[1] > 820:
            position[1] = -20

        p = np.int0(position)
        sprite_frame = (frame_id//4) % 3
        # left = max(p[0]-16, 0)
        # right = max(p[0]+15, 0)
        # right = min(p[0]+15, 819)
        # width = right - left + 1
        try:
            if  sprite_frame == 0:
                frame = frame1
                if direction == [-1, 0]:
                    frame = cv2.flip(frame1,1)
                if direction == [0, -1]:
                    frame = cv2.rotate(frame1, ROTATE_90_COUNTERCLOCKWISE)
                if direction == [0, 1]:
                    frame = cv2.rotate(frame1, ROTATE_90_CLOCKWISE)
                img[p[1] - 32:p[1]+32, p[0] - 32: p[0] + 32] = frame

            if sprite_frame == 1:
                frame = frame2
                if direction == [-1, 0]:
                    frame = cv2.flip(frame2,1)
                if direction == [0, -1]:
                    frame = cv2.rotate(frame2, ROTATE_90_COUNTERCLOCKWISE)
                if direction == [0, 1]:
                    frame = cv2.rotate(frame2, ROTATE_90_CLOCKWISE)
                img[p[1]-32:p[1]+32, p[0] - 32: p[0] + 32] = frame
            if sprite_frame == 2:
                frame = frame3
                if direction == [-1, 0]:
                    frame = cv2.flip(frame3,1)
                if direction == [0, -1]:
                    frame = cv2.rotate(frame3, ROTATE_90_COUNTERCLOCKWISE)
                if direction == [0, 1]:
                    frame = cv2.rotate(frame3, ROTATE_90_CLOCKWISE)
                img[p[1]-32:p[1]+32, p[0] - 32: p[0] + 32] = frame
            if 60**2 >= (position[0]-food_position[0]) ** 2 + (position[1]-food_position[1]) ** 2:
                # print(50**2 , (position[0]-food_position[0]) ** 2 + (position[1]-food_position[1]) ** 2)
                food_count += 1
                food_position = [random.randint(200,600), random.randint(200, 600)]
                food_color = [random.randint(50,255), random.randint(50, 255), random.randint(50, 255)]
        except:
            pass
        cv2.circle(img, food_position, 50, food_color, -1)
        if food_count == 10: # Change it to 10
            playing = False
            game_over = True
            position = [400, 400]
            score = 300-int(time.time() - start_time)
        if playing and time.time() - start_time > 300:
            playing = False
            game_over = True
        frame_id += 1
    cv2.imshow("Echo Man", img)
    if cv2.waitKey(1000//fps) & 0xFF == ord('q'):
        break
