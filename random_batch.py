#Select random batch for training

import numpy as np
import pandas as pd
import config as c
from pre_process import data_catalog

def clipped_audio(audio):
    if audio.shape[0] > c.NUM_FRAMES:
        bias = np.random.randint(0, audio.shape[0] - c.NUM_FRAMES)
        clipped_audio = audio[bias: c.NUM_FRAMES + bias]
    else:
        clipped_audio = audio

    return clipped_audio

# Pick random 1 anchor, get 2 .npy file from anchor and pick random
# 1 speaker diffenrent, get 1 .npy file. Create to a triplets.
# MiniBatch have batch_size triplets like this.
def random_batch(libri, batch_size):
    unique_speakers = list(libri['speaker'].unique())
    anchor_batch = None
    positive_batch = None 
    negative_batch = None
    for i in range(batch_size):
        two_different_speakers = np.random.choice(unique_speakers, size=2, replace=False)
        anchor_positive_speaker = two_different_speakers[0]
        negative_speaker = two_different_speakers[1]
        anchor_positive_audio = libri[libri['speaker'] == anchor_positive_speaker].sample(n=2, replace=False)
        anchor_df = pd.DataFrame(anchor_positive_audio[0:1])
        #anchor_df['type'] = 'anchor'
        positive_df = pd.DataFrame(anchor_positive_audio[1:2])
        #positive_df['type'] = 'positive'
        negative_df = pd.DataFrame(libri[libri['speaker'] == negative_speaker].sample(n=1))
        #negetive_df['type'] = 'negative'

        if anchor_batch is None:
            anchor_batch = anchor_df.copy()
            positive_batch = positive_df.copy()
            negative_batch = negative_df.copy()
        else :
            anchor_batch = pd.concat([anchor_batch, anchor_df], axis=0)
            positive_batch = pd.concat([positive_batch, positive_df], axis=0)
            negative_batch = pd.concat([negative_batch, negative_df], axis=0)

    libri_batch = pd.DataFrame(pd.concat([anchor_batch, positive_batch, negative_batch], axis=0))

    new_x = []
    for i in range(len(libri_batch)):
        filename = libri_batch[i:i + 1]['filename'].values[0]
        x = np.load(filename)
        new_x.append(clipped_audio(x))
    x = np.array(new_x) # x.shape: (batchsize*3, num_frames, 64, 1)
    y = libri_batch['speaker'].values
    return x, y

def main():
    libri = data_catalog(c.DATASET_DIR)
    x, y = random_batch(libri, c.BATCH_SIZE)
    b = x[0]
    print(b.shape[0])
    print(b.shape[1])
    print(b.shape[2])

if __name__ == '__main__':
    main()
