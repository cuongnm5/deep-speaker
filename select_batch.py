#Select hard negative for training

import pandas as pd
import random
import numpy as np
import config as c
from utils import get_last_checkpoint_if_any
from models import convolutional_model
from triplet_loss import deep_speaker_loss
from pre_process import data_catalog
import heapq
import threading
from time import time, sleep

#Function to computed cosine of 2 embeddings
def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.multiply(x1, x2)
    s = np.sum(mul,axis=1)
    return s

def matrix_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.dot(x1, x2.T)
    return mul

def clipped_audio(x, num_frames=c.NUM_FRAMES):
    if x.shape[0] > num_frames + 20:
        bias = np.random.randint(20, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    elif x.shape[0] > num_frames:
        bias = np.random.randint(0, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    else:
        clipped_x = x

    return clipped_x

speaker_utterance_index = {}

# Get candidates/2 speakers and then get 2 utterance per speaker. Return np.array 
# of extracted utterance and label
def preprocess(unique_speakers, speaker_utterance_dict, candidates=c.CANDIDATES_PER_BATCH):
    files = []
    flag = False if len(unique_speakers) > candidates/2 else True
    speakers = np.random.choice(unique_speakers, size=int(candidates/2), replace=flag)
    for speaker in speakers:
        index=0
        ll = len(speaker_utterance_dict[speaker])
        if speaker in speaker_utterance_index:
            index = speaker_utterance_index[speaker] % ll
        files.append(speaker_utterance_dict[speaker][index])
        files.append(speaker_utterance_dict[speaker][(index+1)%ll])
        speaker_utterance_index[speaker] = (index + 2) % ll

    x = []
    labels = []
    for file in files:
        x_ = np.load(file)
        x_ = clipped_audio(x_)
        if x_.shape != (c.NUM_FRAMES, 64, 1):
            print("Error !!!",file['filename'].values[0])
        x.append(x_)
        labels.append(file.split("/")[-1].split("-")[0])

    return np.array(x),np.array(labels)

stack = []
#Multi thread
def create_data_producer(unique_speakers, speaker_utterance_dict, candidates=c.CANDIDATES_PER_BATCH):
    producer = threading.Thread(target=addstack, args=(unique_speakers, speaker_utterance_dict,candidates))
    producer.setDaemon(True)
    producer.start()

# Get data_stack_size=10 batch, 1 patch have candidates utterance with label
def addstack(unique_speakers, speaker_utterance_dict,candidates=c.CANDIDATES_PER_BATCH):
    data_produce_step = 0
    while True:
        if len(stack) >= c.DATA_STACK_SIZE:
            sleep(0.01)
            continue

        orig_time = time()
        feature, labels = preprocess(unique_speakers, speaker_utterance_dict, candidates)
        #Append patch to stack to run multi thread
        stack.append((feature, labels))

        data_produce_step += 1
        #Shuffle data per 100 steps
        if data_produce_step % 100 == 0:
            for speaker in unique_speakers:
                np.random.shuffle(speaker_utterance_dict[speaker])

def getbatch():
    while True:
        if len(stack) == 0:
            continue
        return stack.pop(0)

history_embeds = None
history_labels = None
history_features = None
history_index = 0
history_table_size = c.HIST_TABLE_SIZE

#Slect best batch to train
def best_batch(model, batch_size=c.BATCH_SIZE,candidates=c.CANDIDATES_PER_BATCH):
    orig_time = time()
    global history_embeds, history_features, history_labels, history_index, history_table_size
    
    features,labels = getbatch() 
    embeds = model.predict_on_batch(features)
    #embeds save features after predict - embeddings
    if history_embeds is None:
        history_features = np.copy(features)
        history_labels = np.copy(labels)
        history_embeds = np.copy(embeds)
    else:
        if len(history_labels) < history_table_size*candidates:
            history_features = np.concatenate((history_features, features), axis=0)
            history_labels = np.concatenate((history_labels, labels), axis=0)
            history_embeds = np.concatenate((history_embeds, embeds), axis=0)
        else:
            history_features[history_index*candidates: (history_index+1)*candidates] = features
            history_labels[history_index*candidates: (history_index+1)*candidates] = labels
            history_embeds[history_index*candidates: (history_index+1)*candidates] = embeds
    print(history_features)
    history_index = (history_index+1) % history_table_size

    anchor_batch = []
    positive_batch = []
    negative_batch = []
    anchor_labs, positive_labs, negative_labs = [], [], []

    orig_time = time()
    #Select random batch_size/2 speaker for anchor from history label
    anchor_speaker = np.random.choice(history_labels, int(batch_size/2), replace=False)
    anchors_index_dict = {}
    # anchors_index_dict[speaker] = [index of all positive file]
    indexs_set = []
    for speaker in anchor_speaker:
        anchor_index = np.argwhere(history_labels==speaker).flatten() # get index anchor
        anchors_index_dict[speaker] = anchor_index # dict index of anchor [label] = index
        indexs_set.extend(anchor_index)
    indexs_set = list(set(indexs_set)) #indexs_set consist index of anchor
    speakers_embeds = history_embeds[indexs_set] 

    #computed cosine of n speaker's embeds with all embeds
    sims = matrix_cosine_similarity(speakers_embeds, history_embeds) 
    print('beginning to select..........')
    for ii in range(int(batch_size/2)):  
        while True:
            speaker = anchor_speaker[ii]
            indexs = anchors_index_dict[speaker]
            np.random.shuffle(indexs)
            anchor_index = indexs[0]
            positive_index = []
            for jj in range(1,len(indexs)):
                if (history_features[anchor_index] == history_features[indexs[jj]]).all():
                    continue
                positive_index.append(indexs[jj])

            if len(positive_index) >= 1:
                break

        sap = sims[ii][positive_index] # a list sap of anchor ii with all anchor's positive features
        min_saps = heapq.nsmallest(2, sap) #return a list with 2 smallest elements
        positive_0_index = positive_index[np.argwhere(sap == min_saps[0]).flatten()[0]] #index of feature have sap min

        if len(positive_index) > 1:
            positive_1_index = positive_index[np.argwhere(sap == min_saps[1]).flatten()[0]]
        else:
            positive_1_index = positive_0_index

        negative_index = np.argwhere(history_labels != speaker).flatten()
        san = sims[ii][negative_index]
        max_sans = heapq.nlargest(2, san)
        negative_0_index = negative_index[np.argwhere(san == max_sans[0]).flatten()[0]]
        negative_1_index = negative_index[np.argwhere(san == max_sans[1]).flatten()[0]]

        anchor_batch.append(history_features[anchor_index]);  anchor_batch.append(history_features[anchor_index])
        positive_batch.append(history_features[positive_0_index]);  positive_batch.append(history_features[positive_1_index])
        negative_batch.append(history_features[negative_0_index]);  negative_batch.append(history_features[negative_1_index])

        anchor_labs.append(history_labels[anchor_index]);  anchor_labs.append(history_labels[anchor_index])
        positive_labs.append(history_labels[positive_0_index]);  positive_labs.append(history_labels[positive_1_index])
        negative_labs.append(history_labels[negative_0_index]);  negative_labs.append(history_labels[negative_1_index])

    batch = np.concatenate([np.array(anchor_batch), np.array(positive_batch), np.array(negative_batch)], axis=0)
    labs = anchor_labs + positive_labs + negative_labs

    print("select best batch time {0:.3}s".format(time() - orig_time))
    return batch, np.array(labs)

if __name__ == '__main__':
    model = convolutional_model()
    model.compile(optimizer='adam', loss=deep_speaker_loss)
    # best_batch(model)

    last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER)
    if last_checkpoint is not None:
        print('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
        model.load_weights(last_checkpoint)
        grad_steps = int(last_checkpoint.split('_')[-2])
        print('[DONE]')
    libri = data_catalog(c.DATASET_DIR)
    unique_speakers = libri['speaker_id'].unique()
    labels = libri['speaker_id'].values
    files = libri['filename'].values
    spk_utt_dict = {}
    for i in range(len(unique_speakers)):
        spk_utt_dict[unique_speakers[i]] = []

    for i in range(len(labels)):
        spk_utt_dict[labels[i]].append(files[i])

    create_data_producer(unique_speakers,spk_utt_dict)
    for i in range(10):
        x, y = best_batch(model)
        # print(x.shape)
        # print(y)