import tensorflow as tf
#import tensorflow_hub as hub
import config
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.utils import multi_gpu_model
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])

os.environ["CUDA_VISIBLE_DEVICES"]="0" 
#vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
cnn_model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation = 'relu'),
    Dense(42, activation = 'softmax')
])
#cnn_model = multi_gpu_model(cnn_model, gpus=2)
#print(cnn_model.summary())

if __name__ == '__main__':
    print("TF Version : ", tf.__version__)
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
