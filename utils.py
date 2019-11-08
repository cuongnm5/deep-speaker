import logging
import os
import re
from glob import glob
import matplotlib.pyplot as plt
import config as c
from natsort import natsorted
import csv


def get_last_checkpoint_if_any(checkpoint_folder):
    os.makedirs(checkpoint_folder, exist_ok=True)
    files = glob('{}/*.h5'.format(checkpoint_folder), recursive=True)
    if len(files) == 0:
        return None
    return natsorted(files)[-1]

def create_dir_and_delete_content(directory):
    os.makedirs(directory, exist_ok=True)
    files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"), 
        map(lambda f: os.path.join(directory, f), os.listdir(directory))),
        key=os.path.getmtime)
    # delete all but most current file to assure the latest model is availabel even if process is killed
    for file in files[:-4]:
        logging.info("removing old model: {}".format(file))
        os.remove(file)

def plot_loss(file=c.CHECKPOINT_FOLDER+'/losses.txt'):
    step = []
    loss = []
    mov_loss = []
    ml = 0
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
           step.append(int(line.split(",")[0]))
           loss.append(float(line.split(",")[1]))
           if ml == 0:
               ml = float(line.split(",")[1])
           else:
               ml = 0.01*float(line.split(",")[1]) + 0.99*mov_loss[-1]
           mov_loss.append(ml)


    p1, = plt.plot(step, loss)
    p2, = plt.plot(step, mov_loss)
    plt.legend(handles=[p1, p2], labels = ['loss', 'moving_average_loss'], loc = 'best')
    plt.xlabel("Steps")
    plt.ylabel("Losses")
    plt.show()

def changefilename(path):
    files = os.listdir(path)
    for file in files:
        name=file.replace('-','_')
        lis = name.split('_')
        speaker = '_'.join(lis[:3])
        utt_id = '_'.join(lis[3:])
        newname = speaker + '-' +utt_id
        os.rename(path+'/'+file, path+'/'+newname)
