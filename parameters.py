import os
import time
import argparse
import tensorflow as tf
from mrnn import MRNNCell
from tensorflow.contrib.rnn import GRUCell

LAMBDA = 10.0
MAX_N_EXAMPLES = 10000000



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



#Define flags for the Application
tf.logging.set_verbosity(tf.logging.INFO)
parser = argparse.ArgumentParser(description='Define the arguments for the training to run with.')

flags = tf.app.flags

parser.add_argument("-CRITIC_ITERS", type=int, default="2", help="How many iterations for the critic to use")
parser.add_argument("-PICKLE_PATH", default="./pkl", help="Where to save pickled cache files")

parser.add_argument("-LOGS_DIR", default="./logs/", help="Directory for saving the logs")
parser.add_argument("-DATA_DIR", default="./dataset/nips/", help="Location for the training data")
parser.add_argument("-CKPT_PATH", default="./logs/", help="Location for saving checkpoints")

parser.add_argument("-rnn_type", default="./ckpt/", help="What RNN to use for Critic and Generator -- GRU or MRNN")
parser.add_argument("-BATCH_SIZE", type=int, default="32", help="How many examples to process before updating weights")


parser.add_argument("-GEN_ITERS", type=int, default="10", help="How many iterations to use for the Generator")

parser.add_argument("-ITERATIONS_PER_SEQ_LENGTH", type=int, default="1000", help="Number of iterations to run for each value of sequence length in curriculum training")

parser.add_argument('-TRAIN_FROM_CKPT', type=str2bool, default=False, help="Should it restore from checkpoint")

# RNN Settings
parser.add_argument('-GEN_RNN_LAYERS', type=int, default="1", help="Number of layers for the Generator RNN")
parser.add_argument('-DISC_RNN_LAYERS', type=int, default="1", help="Number of layerss for the Discriminator RNN")

parser.add_argument('-DISC_STATE_SIZE', type=int, default="512", help="State size of the Critic State")
parser.add_argument('-GEN_STATE_SIZE', type=int, default="512", help="State size of the Generator State")

parser.add_argument('-START_SEQ', type=int, default="1", help="What length of sequence should we start training")
parser.add_argument('-END_SEQ', type=int, default="32", help="What length of sequence should we end training on")

parser.add_argument('-PADDING_IS_SUFFIX', type=str2bool, default=False, help="")

parser.add_argument('-SAVE_CHECKPOINTS_EVERY', type=int, default="25000", help="")
parser.add_argument('-LIMIT_BATCH', type=str2bool, default=True, help='')
parser.add_argument('-SCHEDULE_ITERATIONS', type=str2bool, default=False, help='')
parser.add_argument('-SCHEDULE_MULT', type=int, default="2000", help='')
parser.add_argument('-DYNAMIC_BATCH', type=str2bool, default=False, help='')

# Print Options
parser.add_argument('-PRINT_ITERATION', type=int, default="1000", help='How many iterations should elapse when a message is displayed')

# Learning Rates
parser.add_argument('-DISC_LR', type=float, default="2e-4", help='Disc learning rate -- should be different than generator')
parser.add_argument('-GEN_LR', type=float, default="1e-4", help="""Gen learning rate""")

# Only for inference mode
parser.add_argument('-INPUT_SAMPLE', default='./output/sample.txt', help="")

FLAGS = parser.parse_args()

# only for backward compatability

LOGS_DIR = os.path.join(FLAGS.LOGS_DIR,
                        "%s-%s-%s-%s-%s-%s-%s-" % ("GEN_CL_VL_TH", "Disc",
                                                        FLAGS.GEN_ITERS, FLAGS.CRITIC_ITERS,
                                                        FLAGS.DISC_STATE_SIZE, FLAGS.GEN_STATE_SIZE,
                                                        time.time()))


class RestoreConfig():
    def __init__(self):
        if FLAGS.TRAIN_FROM_CKPT:
            self.restore_dir = self.set_restore_dir(load_from_curr_session=False)
        else:
            self.restore_dir = self.set_restore_dir(load_from_curr_session=True)

    def set_restore_dir(self, load_from_curr_session=True):
        if load_from_curr_session:
            restore_dir = os.path.join(LOGS_DIR, 'checkpoint')
        else:
            restore_dir = FLAGS.CKPT_PATH
        return restore_dir

    def get_restore_dir(self):
        return self.restore_dir


def create_logs_dir():
    os.makedirs(LOGS_DIR)

restore_config = RestoreConfig()
if (FLAGS.rnn_type== "GRU"):
	rnn_cell = GRUCell
else:
	rnn_cell = MRNNCell	

DATA_DIR = FLAGS.DATA_DIR
CKPT_PATH = FLAGS.CKPT_PATH
GEN_ITERS = FLAGS.GEN_ITERS
CRITIC_ITERS = FLAGS.CRITIC_ITERS
BATCH_SIZE = FLAGS.BATCH_SIZE
PICKLE_PATH = FLAGS.PICKLE_PATH
