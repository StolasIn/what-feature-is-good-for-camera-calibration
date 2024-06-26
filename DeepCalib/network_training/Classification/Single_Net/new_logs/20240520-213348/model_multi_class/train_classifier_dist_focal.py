from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["TF_USE_LEGACY_KERAS"]="1"


import tf_keras as keras
from tf_keras.callbacks import TensorBoard, LearningRateScheduler
from tf_keras.applications.inception_v3 import InceptionV3
from tf_keras.applications.imagenet_utils import preprocess_input
from tf_keras.models import Model
from tf_keras.layers import Dense, Flatten, Input
from utils_focal_distortion import RotNetDataGenerator, angle_error, CustomModelCheckpoint
from tf_keras import optimizers
import numpy as np
import glob, math
from shutil import copyfile
import datetime, random
import tensorflow as tf
import tf_keras.backend as set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

model_name = 'model_multi_class/'
SAVE = "new_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
# Save
output_folder = SAVE + model_name
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_log = output_folder + "Log/"
if not os.path.exists(output_log):
    os.makedirs(output_log)

output_weight = output_folder + "Best/"
if not os.path.exists(output_weight):
    os.makedirs(output_weight)

# training parameters
batch_size = 60
nb_epoch = 10000

IMAGE_FILE_PATH_DISTORTED = "/home/binghong/Computer Vision/FinalProject/DeepCalib/dataset/discret/"

classes_focal = list(np.arange(50, 500 + 1, 10))
classes_distortion = list(np.arange(0, 120 + 1, 2) / 100.)


def get_paths(IMAGE_FILE_PATH_DISTORTED):
    paths_train = glob.glob(IMAGE_FILE_PATH_DISTORTED + 'train/' + "*.jpg")
    paths_train.sort()
    parameters = []
    labels_focal_train = []
    for path in paths_train:
        curr_parameter = float((path.split('_f_'))[1].split('_d_')[0])
        parameters.append(curr_parameter)
        curr_class = classes_focal.index(curr_parameter)
        labels_focal_train.append(curr_class)
    labels_distortion_train = []
    for path in paths_train:
        curr_parameter = float((path.split('_d_'))[1].split('.jpg')[0])
        parameters.append(curr_parameter)
        curr_class = classes_distortion.index(curr_parameter)
        labels_distortion_train.append(curr_class)

    c = list(zip(paths_train, labels_focal_train,labels_distortion_train))
    random.shuffle(c)
    paths_train, labels_focal_train,labels_distortion_train = zip(*c)
    paths_train, labels_focal_train, labels_distortion_train = list(paths_train), list(labels_focal_train), list(labels_distortion_train)
    labels_train = [list(a) for a in zip(labels_focal_train, labels_distortion_train)]


    paths_valid = glob.glob(IMAGE_FILE_PATH_DISTORTED + 'valid/' + "*.jpg")
    paths_valid.sort()
    parameters = []
    labels_focal_valid = []
    for path in paths_valid:
        curr_parameter = float((path.split('_f_'))[1].split('_d_')[0])
        parameters.append(curr_parameter)
        curr_class = classes_focal.index(curr_parameter)
        labels_focal_valid.append(curr_class)
    labels_distortion_valid = []
    for path in paths_valid:
        curr_parameter = float((path.split('_d_'))[1].split('.jpg')[0])
        parameters.append(curr_parameter)
        curr_class = classes_distortion.index(curr_parameter)
        labels_distortion_valid.append(curr_class)

    c = list(zip(paths_valid, labels_focal_valid, labels_distortion_valid))
    random.shuffle(c)
    paths_valid, labels_focal_valid, labels_distortion_valid = zip(*c)
    paths_valid, labels_focal_valid, labels_distortion_valid = list(paths_valid), list(labels_focal_valid), list(labels_distortion_valid)
    labels_valid = [list(a) for a in zip(labels_focal_valid, labels_distortion_valid)]

    return paths_train, labels_train, paths_valid, labels_valid


def step_decay(epoch):
   initial_lrate = 0.001
   drop = 0.1
   epochs_drop = 2.
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate


paths_train, labels_train, paths_valid, labels_valid = get_paths(IMAGE_FILE_PATH_DISTORTED)

print(len(paths_train), 'train samples')
print(len(paths_valid), 'valid samples')

with tf.device('/gpu:1'):
    input_shape = (299, 299, 3)
    main_input = Input(shape=input_shape, dtype='float32', name='main_input')
    phi_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=main_input, input_shape=input_shape)
    phi_features = phi_model.output
    phi_flattened = Flatten(name='phi-flattened')(phi_features)
    final_output_focal = Dense(len(classes_focal), activation='softmax', name='output_focal')(phi_flattened)
    final_output_distortion = Dense(len(classes_distortion), activation='softmax', name='output_distortion')(phi_flattened)

    layer_index = 0
    for layer in phi_model.layers:
        layer._name = layer._name + "_phi"

    model = Model(inputs=main_input, outputs=[final_output_focal, final_output_distortion])

    learning_rate = 0.001
    adam = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss={'output_focal':'categorical_crossentropy', 'output_distortion':'categorical_crossentropy'},
                  optimizer=adam,
                  metrics={'output_focal':'accuracy','output_distortion':'accuracy'}
                  )
    model.summary()
    # model_json = phi_model.to_json()

    # with open(output_folder + "model.json", "w") as json_file:
    #     json_file.write(model_json)

    copyfile(os.path.basename(__file__), output_folder + os.path.basename(__file__))

    tensorboard = TensorBoard(log_dir=output_log)

    checkpointer = CustomModelCheckpoint(
        model_for_saving=model,
        filepath=output_weight + "weights_{epoch:02d}_{val_loss:.2f}.h5",
        save_best_only=True,
        monitor='val_loss',
        save_weights_only=True
    )

    lrate = LearningRateScheduler(step_decay)

    generator_training = RotNetDataGenerator(input_shape=input_shape, batch_size=batch_size, one_hot=True,
                                             preprocess_func=preprocess_input, shuffle=True).generate(paths_train,
                                                                                                      labels_train,
                                                                                                      len(classes_focal),len(classes_distortion))
    generator_valid = RotNetDataGenerator(input_shape=input_shape, batch_size=batch_size, one_hot=True,
                                          preprocess_func=preprocess_input, shuffle=True).generate(paths_valid,
                                                                                                   labels_valid,
                                                                                                      len(classes_focal),len(classes_distortion))

    # training loop
    model.fit_generator(
        generator=generator_training,
        steps_per_epoch=(len(paths_train) // batch_size), # 29977
        epochs=nb_epoch,
        validation_data=generator_valid,
        validation_steps=(len(paths_valid) // batch_size),
        callbacks=[tensorboard, checkpointer, lrate],
        use_multiprocessing=True,
        workers=2,
        #verbose=3
    )
