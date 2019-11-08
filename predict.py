import csv 
import os 
import config as c 
from models import convolutional_model
from pre_process import *
import numpy as np
from multiprocessing import Pool
from time import time
from triplet_loss import deep_speaker_loss
from utils import get_last_checkpoint_if_any
import heapq

def matrix_cosine_similarity(x1, x2):
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

old_file_name = []

def extract_test_audio(create_new = False) :
    labels_list = [] 
    files_list = []
    classes_list = []
    global old_file_name

    with open(c.CSV_DIR, 'r') as f:
        reader = csv.reader(f)
        csv_list = list(reader)
    
    indexs_list = {}
    
    for id, speaker in enumerate(csv_list):
        if id!=0:
            files_list.append(speaker[0])
            labels_list.append(speaker[1]) 
            if speaker[1] != "other" :
                classes_list.append(speaker[1])
    classes_list = np.unique(classes_list)
    for speaker in classes_list:
        indexs_list[speaker] = []
        for i in range(len(files_list)):
            if speaker == labels_list[i]:
                indexs_list[speaker].append(i)


    old_file_name = files_list
    org_time = time()
    count = 0
    new_filelist = []
    for i in range(len(files_list)):
        file_name = "../../../storage/AIF3/test/public-test/audio/" + files_list[i]
        target_filename = c.PUBLIC_TEST_DIR +'/'+ labels_list[i]+'-'+str(count) + '.npy'
        if create_new == True:
            raw_audio = read_audio(file_name)
            feature = extract_features(raw_audio,target_sample_rate=c.SAMPLE_RATE)
            np.save(target_filename, feature)
        new_filelist.append(target_filename)
        # print("Extract success {} to npy. Time: {}".format(target_filename, time()-org_time))
        count += 1
    files_list = new_filelist

    return classes_list, labels_list, files_list, indexs_list

def predict():
    global old_file_name
    embeddings_list = []
    features_list = []
    x = []
    classes_list, labels_list, files_list, indexs_list = extract_test_audio(False)
    
    for file in files_list:
        x_ = np.load(file)
        x_ = clipped_audio(x_)
        x.append(x_)
    x = np.array(x)

    last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER)
    model = convolutional_model()
    
    model.compile(optimizer='adam', loss = deep_speaker_loss)
    embeddings_list = model.predict_on_batch(x)

    sims = matrix_cosine_similarity(embeddings_list, embeddings_list)
    L = len(files_list)
    for i in range(L):
        if labels_list[i] == "other":
            cosine_max = -1
            error_label = {}
            for speaker in classes_list:
                average_cosine = 0
                error_label[speaker] = 0
                for index in indexs_list[speaker]:
                    average_cosine += sims [i][index]
                    if sims[i][index] > 0.9:
                        error_label[speaker]+=1
                        # print("file_name: {} - speaker: {}".format(old_file_name[i], old_file_name[index]))
                if average_cosine <= 0:
                    average_cosine = 0
                else:
                    average_cosine/=len(indexs_list[speaker])
                if average_cosine > cosine_max:
                    if labels_list[i] == "other":
                        indexs_list[speaker].append(i)
                    else:
                        del indexs_list[labels_list[i]][-1]
                        indexs_list[speaker].append(i)
                    labels_list[i] = speaker
                    cosine_max = average_cosine
            for speaker in classes_list:
                print("file_name: {} - speaker:{} - err:{}".format(old_file_name[i], speaker, error_label[speaker]))

            if cosine_max < 0.6:
                labels_list[i] = "other" 
            # print("file_name: {} - labels: {} - cs_max: {}".format(old_file_name[i], labels_list[i], cosine_max))       

    # for i in range(L):
    #     if labels_list[i] == "other":
    #         cosine_max = -2
    #         for j in range(L):
    #             if i!=j and sims[i][j] > 0.9:
    #                 print("file_name: {} - speaker: {}".format(old_file_name[i], old_file_name[j]))
    #             if i!=j and labels_list[j] != "other" and sims[i][j] > cosine_max:
    #                 labels_list[i] = labels_list[j]
    #                 cosine_max = sims[i][j]
                    
    #         if cosine_max < 0.5:
    #             labels_list[i] = "other"
        
    write_csv(old_file_name, labels_list)

def write_csv(file_name, label):
    file = open("solve.csv", "a")
    file.write("audio,speaker\n")
    for i in range(len(file_name)):
        file.write(file_name[i])
        file.write(",")
        file.write(label[i])
        file.write("\n")
    
    file.close()

if __name__ == "__main__":
    predict()