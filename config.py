
DATASET_DIR = 'audio/train-npy/'
TEST_DIR = 'audio/train-npy/'
WAV_DIR = 'audio/train/'

CSV_DIR = "../data/public-test/public-test/label.csv"
PUBLIC_TEST_DIR = "audio/test"

BATCH_SIZE = 12#32        #must be even
TRIPLET_PER_BATCH = 3

SAVE_PER_EPOCHS = 200
TEST_PER_EPOCHS = 200
CANDIDATES_PER_BATCH = 100#400       # 18s per batch
TEST_NEGATIVE_No = 33


NUM_FRAMES = 60 #160   # 299 - 16*2
SAMPLE_RATE = 16000
TRUNCATE_SOUND_SECONDS = (0.5, 3.5)  # (start_sec, end_sec)
HIST_TABLE_SIZE = 10
NUM_SPEAKERS = 3#200
DATA_STACK_SIZE = 10

CHECKPOINT_FOLDER = 'checkpoints/checkpoint'
BEST_CHECKPOINT_FOLDER = 'checkpoints/best_checkpoint'
PRE_CHECKPOINT_FOLDER = 'checkpoints/pretraining_checkpoints'

LOSS_LOG= CHECKPOINT_FOLDER + '/losses.txt'
TEST_LOG= CHECKPOINT_FOLDER + '/test_eer_f_acc.txt'

PRE_TRAIN = False

