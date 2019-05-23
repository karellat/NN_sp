from __future__ import absolute_import, division, print_function

import os

import numpy as np
import argparse
import datetime
import re

import tensorflow as tf
import tensorflow_hub as tfhub

def get_layer(arg, inputs):
  C_args = arg.split('-')
  if arg.startswith('C-'):
      return tf.ke368368ras.layers.Conv2D(
              int(C_args[1]),
              int(C_args[2]),
              int(C_args[3]),
              padding=C_args[4],
              activation="relu")(inputs)
  elif arg.startswith('CB-'):
      new_layer = tf.keras.layers.Conv2D(
              int(C_args[1]),
              int(C_args[2]),
              int(C_args[3]),
              padding=C_args[4],
              use_bias=False)(inputs)
      new_layer = tf.keras.layers.BatchNormalization()(new_layer)
      return tf.keras.layers.Activation("relu")(new_layer)
  elif arg.startswith('AV-'):
      return tf.keras.layers.AveragePooling2D(
              int(C_args[1]),
              int(C_args[2]),
              padding=C_args[3])(inputs)
  elif arg.startswith('M-'):
     return tf.keras.layers.MaxPool2D(
         int(C_args[1]),
         int(C_args[2]))(inputs)
  elif arg.startswith('R-'):
      assert len(arg[3:-1].split(';')) != 0
      new_layer = inputs
      for a in arg[3:-1].split(';'):
          new_layer = get_layer(a, new_layer)
      return tf.keras.layers.Add()([new_layer, inputs])
  elif arg.startswith('D-'):
      return tf.keras.layers.Dense(
         int(C_args[1]),
          activation="relu")(inputs)
  elif arg.startswith('DB-'):
      new_layer = tf.keras.layers.Dense(
              int(C_args[1]), use_bias=False)(inputs)
      new_layer = tf.keras.layers.BatchNormalization()(new_layer)
      return tf.keras.layers.Activation("relu")(new_layer)
  elif arg.startswith('F'):
      return tf.keras.layers.Flatten()(inputs)
  elif arg.startswith('Dr'):
      return tf.keras.layers.Dropout(rate=0.5)(inputs)
  else:
    raise Exception('Unknown cnn argument {}'.format(arg))

def init_transfer_model(args):
  network_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/3"
  input = tf.keras.layers.Input([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
  network = tfhub.KerasLayer(network_url, output_shape=[1280],trainable=False)(input, training=False)
  hidden = network
  for l in filter(None, args.nn.split(",")):
    hidden = get_layer(l, hidden)

  flatten = tf.keras.layers.Flatten()(hidden)
  output = tf.keras.layers.Dense(5, activation='softmax')(flatten)

  model = tf.keras.Model(inputs=[input], outputs=[output])

  return model

def init_my_model(args): 
  input = tf.keras.layers.Input([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
  hidden = input 
  for l in filter(None, args.nn.split(",")):
    hidden = get_layer(l, hidden)
    
  flatten = tf.keras.layers.Flatten()(hidden)
  output = tf.keras.layers.Dense(5, activation='softmax')(flatten)
  
  model = tf.keras.Model(inputs=[input], outputs=[output])
  return model 

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--image_size", default=224, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--nn", default="", type=str)
parser.add_argument("--transfer", default=False, action='store_true')
args = parser.parse_args()

parser = argparse.ArgumentParser()

data_dir = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

IMAGE_SIZE = (args.image_size, args.image_size)
datagen_kwargs = dict(rescale=1./255, validation_split=.20)
#Validation dataset
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)

valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False,
    target_size=IMAGE_SIZE, batch_size=args.batch_size)
# Training dataset

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    horizontal_flip=True,
    width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2,
    **datagen_kwargs)

train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", shuffle=True,
    target_size=IMAGE_SIZE, batch_size=args.batch_size)

if args.transfer:
    model = init_transfer_model(args)
else:
    model = init_my_model(args)

print(model.summary())
model.compile(optimizer=tf.keras.optimizers.Adam(),
             loss=tf.losses.CategoricalCrossentropy(from_logits=True),
             metrics=[tf.metrics.CategoricalAccuracy()])

model.fit(x=train_generator,
          validation_data=valid_generator,
          epochs=args.epochs,)
