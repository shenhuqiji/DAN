import tensorflow as tf
##########################
#  dataset #
##########################
tf.app.flags.DEFINE_string(
    'dataset_dir', 'data/VOCdata/',
    'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train',
    'The name of the train/test/val split.')
tf.app.flags.DEFINE_string(
    'dataset_val_name', 'val',
    'The name of the val split.')
tf.app.flags.DEFINE_integer(
    'batch_size', 16,
    'The number of samples in each batch.')
tf.flags.DEFINE_bool(
    'debug', "False",
    "Debug mode: True/ False")
tf.flags.DEFINE_integer(
    'image_size', 256,
    "the size of image")
######################
# Optimization Flags #
######################
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'momentum', 0.99,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

#######################
# Learning Rate Flags #
#######################
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                          'Initial learning rate.')
tf.flags.DEFINE_string("logs_dir", './output/logs_new/', "path to logs_new directory")

#######################
# Training  Flags #
######################
tf.app.flags.DEFINE_integer(
    'iters', 1024,
    'iteration numbers')
tf.app.flags.DEFINE_integer(
    'epochs', 200,
    'total epochs')

##########################
#                  restore
##########################
tf.app.flags.DEFINE_string(
    'train_dir', './output/densenet_new/',
    'Directory where checkpoints and event logs_new are written to.')

FLAGS = tf.app.flags.FLAGS
