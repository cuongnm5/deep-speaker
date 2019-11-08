#Training the embeddings with the triplet loss

from time import time

import numpy as np
import sys
import os
import random
from keras.optimizers import Adam
from keras.layers.core import Dense
from keras.models import Model

import config as c
import select_batch
from pre_process import data_catalog, preprocess_and_save
from models import convolutional_model
from random_batch import stochastic_mini_batch
from triplet_loss import deep_speaker_loss
from utils import get_last_checkpoint_if_any, create_dir_and_delete_content

# Config GPU for tf1.x
try:
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    keras.backend.set_session(tf.Session(config=config))
except:
    pass


def create_dict(files,labels,classes):
    train_dict = {}
    for i in range(len(classes)):
        train_dict[classes[i]] = []

    for i in range(len(labels)):
        train_dict[labels[i]].append(files[i])

    for spk in classes:
        if len(train_dict[spk]) < 2:
            train_dict.pop(spk)
    unique_speakers=list(train_dict.keys())
    return train_dict, unique_speakers

def main(libri_dir=c.DATASET_DIR):

    PRE_TRAIN = c.PRE_TRAIN
    print('Looking for fbank features [.npy] files in {}.'.format(libri_dir))
    libri = data_catalog(libri_dir)

    if len(libri) == 0:
        preprocess_and_save(c.WAV_DIR, c.DATASET_DIR)
        libri = data_catalog(libri_dir)

    unique_speakers = libri['speaker_id'].unique()
    speaker_utterance_dict, unique_speakers = create_dict(libri['filename'].values,libri['speaker_id'].values,unique_speakers)
    select_batch.create_data_producer(unique_speakers, speaker_utterance_dict)

    batch = stochastic_mini_batch(libri, batch_size=c.BATCH_SIZE, unique_speakers=unique_speakers)
    batch_size = c.BATCH_SIZE * c.TRIPLET_PER_BATCH
    x, y = batch.to_inputs()
    b = x[0]
    num_frames = b.shape[0]
    train_batch_size = batch_size
    #batch_shape = [batch_size * num_frames] + list(b.shape[1:])  # A triplet has 3 parts.
    input_shape = (num_frames, b.shape[1], b.shape[2])

    print('num_frames = {}'.format(num_frames))
    print('batch size: {}'.format(batch_size))
    print('input shape: {}'.format(input_shape))
    print('x.shape : {}'.format(x.shape))
    orig_time = time()
    model = convolutional_model(input_shape=input_shape, batch_size=batch_size, num_frames=num_frames)
    print(model.summary())
    grad_steps = 0
    if PRE_TRAIN:
        last_checkpoint = get_last_checkpoint_if_any(c.PRE_CHECKPOINT_FOLDER)
        if last_checkpoint is not None:
            print('Found pre-training checkpoint [{}]. Resume from here...'.format(last_checkpoint))
            x = model.output
            x = Dense(len(unique_speakers), activation='softmax', name='softmax_layer')(x)
            pre_model = Model(model.input, x)
            pre_model.load_weights(last_checkpoint)
            grad_steps = int(last_checkpoint.split('_')[-2])
            grad_steps = 0
            print('Successfully loaded pre-training model')

    else:
        last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER)
        if last_checkpoint is not None:
            print('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
            model.load_weights(last_checkpoint)
            grad_steps = int(last_checkpoint.split('_')[-2])
            #grad_steps = 0
            print('[DONE]')


    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer='adam', loss=deep_speaker_loss)

    print("model_build_time",time()-orig_time)
    print('Starting training...')
    last_loss = 10
    while True:
        orig_time = time()
        x, _ = select_batch.best_batch(model, batch_size=c.BATCH_SIZE)
        y = np.random.uniform(size=(x.shape[0], 1))
        print('== Presenting step #{0}'.format(grad_steps))
        orig_time = time()
        loss = model.train_on_batch(x, y)
        print('== Processed in {0:.2f}s by the network, training loss = {1}.'.format(time() - orig_time, loss))

        # record training loss
        with open(c.LOSS_LOG, "a") as f:
            f.write("{0},{1}\n".format(grad_steps, loss))

        # checkpoints are really heavy so let's just keep the last one.
        if (grad_steps) % c.SAVE_PER_EPOCHS == 0:
            create_dir_and_delete_content(c.CHECKPOINT_FOLDER)
            model.save_weights('{0}/model_{1}_{2:.5f}.h5'.format(c.CHECKPOINT_FOLDER, grad_steps, loss))
            if loss < last_loss:
                files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"),
                                      map(lambda f: os.path.join(c.BEST_CHECKPOINT_FOLDER, f), os.listdir(c.BEST_CHECKPOINT_FOLDER))),
                               key=lambda file: file.split('/')[-1].split('.')[-2], reverse=True)
                last_loss = loss
                for file in files[:-4]:
                    print("removing old model: {}".format(file))
                    os.remove(file)
                model.save_weights(c.BEST_CHECKPOINT_FOLDER+'/best_model{0}_{1:.5f}.h5'.format(grad_steps, loss))
                
        grad_steps += 1



if __name__ == '__main__':
    main()
