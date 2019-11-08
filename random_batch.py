#Select random batch for training

import numpy as np
import pandas as pd
import config as c
from pre_process import data_catalog


def clipped_audio(x, num_frames=c.NUM_FRAMES):
    if x.shape[0] > num_frames:
        bias = np.random.randint(0, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    else:
        clipped_x = x

    return clipped_x



class MiniBatch:
    # Pick random 1 anchor, get 2 .npy file from anchor and pick random
    # 1 speaker diffenrent, get 1 .npy file. Create to a triplets.
    # MiniBatch have batch_size triplets like this.
    def __init__(self, libri, batch_size, unique_speakers=None):    
        if unique_speakers is None:
            unique_speakers = list(libri['speaker_id'].unique())
        num_triplets = batch_size

        anchor_batch = None
        positive_batch = None
        negative_batch = None
        for ii in range(num_triplets):

            two_different_speakers = np.random.choice(unique_speakers, size=2, replace=False)
            anchor_positive_speaker = two_different_speakers[0]
            negative_speaker = two_different_speakers[1]
            anchor_positive_audio = libri[libri['speaker_id'] == anchor_positive_speaker].sample(n=2, replace=False)
            anchor_df = pd.DataFrame(anchor_positive_audio[0:1])
            anchor_df['training_type'] = 'anchor'
            positive_df = pd.DataFrame(anchor_positive_audio[1:2])
            positive_df['training_type'] = 'positive'
            negative_df = libri[libri['speaker_id'] == negative_speaker].sample(n=1)
            negative_df['training_type'] = 'negative'

            if anchor_batch is None:
                anchor_batch = anchor_df.copy()
            else:
                anchor_batch = pd.concat([anchor_batch, anchor_df], axis=0)
            if positive_batch is None:
                positive_batch = positive_df.copy()
            else:
                positive_batch = pd.concat([positive_batch, positive_df], axis=0)
            if negative_batch is None:
                negative_batch = negative_df.copy()
            else:
                negative_batch = pd.concat([negative_batch, negative_df], axis=0)

        self.libri_batch = pd.DataFrame(pd.concat([anchor_batch, positive_batch, negative_batch], axis=0))
        self.num_triplets = num_triplets


    #From libri_batch load file npy and label for data trainning
    def to_inputs(self):
        new_x = []
        for i in range(len(self.libri_batch)):
            filename = self.libri_batch[i:i + 1]['filename'].values[0]
            x = np.load(filename)
            new_x.append(clipped_audio(x))
        x = np.array(new_x) #(batchsize, num_frames, 64, 1)
        y = self.libri_batch['speaker_id'].values
        np.testing.assert_array_equal(y[0:self.num_triplets], y[self.num_triplets:2 * self.num_triplets])

        return x, y


def stochastic_mini_batch(libri, batch_size=c.BATCH_SIZE,unique_speakers=None):
    mini_batch = MiniBatch(libri, batch_size,unique_speakers)
    return mini_batch


def main():
    libri = data_catalog(c.DATASET_DIR)
    batch = stochastic_mini_batch(libri, c.BATCH_SIZE)
    x, y = batch.to_inputs()
    # print(x.shape,y.shape)
    # print(x)


if __name__ == '__main__':
    main()
