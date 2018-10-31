import os,time
import tensorflow as tf
from  libs.models.nets import computer_losses
from libs.models.nets_factory import get_network
from libs.datasets.minibatch import get_minibatch
from  libs.datasets.dataset import _load_image_list
from  libs.configs import configs as cfg
FLAGS = tf.app.flags.FLAGS
def get_optimizer(optimizer_name,learning_rate,batch_size, global_step):
    if optimizer_name == "momentum":
        decay_rate=FLAGS.learning_rate_decay_factor
        learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                        global_step=global_step,
                                                        decay_steps=batch_size,
                                                        decay_rate=decay_rate,
                                                        staircase=True)
        momentum =  FLAGS.momentum
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_node, momentum=momentum)
    elif optimizer_name== "adam":
        learning_rate = FLAGS.learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    return optimizer

def train():
    dataset_dir1 = os.path.join(FLAGS.dataset_dir, 'train')
    dataset_dir2 = os.path.join(FLAGS.dataset_dir, 'val')
    image_dir1 = os.path.join(FLAGS.dataset_dir,'train', 'JPEGImages')
    image_dir2 = os.path.join(FLAGS.dataset_dir,'val', 'JPEGImages')
    label_img_dir1 = os.path.join(FLAGS.dataset_dir,'train', 'SegmentationClass')
    label_img_dir2 = os.path.join(FLAGS.dataset_dir, 'val', 'SegmentationClass')
    images_list = _load_image_list(dataset_dir1, FLAGS.dataset_split_name)
    val_img_list = _load_image_list(dataset_dir2, FLAGS.dataset_val_name)

    x=tf.placeholder(dtype=tf.float32,shape=(FLAGS.batch_size,FLAGS.image_size,FLAGS.image_size,3),name='image_input')
    label = tf.placeholder(dtype=tf.int32, shape=(FLAGS.batch_size,FLAGS.image_size,FLAGS.image_size,2), name='labels_input')
    training_flag = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    logits=get_network(x,weight_decay = 0.0001,is_training=training_flag)
    total_loss,loss, regular_loss,accuracy=computer_losses(logits,label)
    global_step = tf.Variable(0)
    optimizer = get_optimizer("adam",learning_rate,FLAGS.batch_size, global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([tf.group(*update_ops)]):
        # Ensures that we execute the update_ops before performing the train_step
        train_step = optimizer.minimize(total_loss,global_step=global_step)
    saver = tf.train.Saver(max_to_keep=20)
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ckpt = tf.train.get_checkpoint_state('./output/densenet_new')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./output/logs_new', sess.graph)
        epoch_learning_rate = FLAGS.learning_rate
        for epoch in range(1, FLAGS.epochs + 1):
            if epoch == (FLAGS.epochs * 0.25) :
                epoch_learning_rate = epoch_learning_rate / 2
            elif  epoch == (FLAGS.epochs * 0.5):
                epoch_learning_rate = epoch_learning_rate / 5

            train_acc = 0.0
            train_loss = 0.0
            for step in range(1, FLAGS.iters + 1):
                batch_x, train_ground_truths = get_minibatch(images_list, image_dir1, FLAGS.batch_size,
                                                                 label_img_dir1)
                train_feed_dict = {
                    x: batch_x,
                    label: train_ground_truths,
                    learning_rate: epoch_learning_rate,
                    training_flag: True
                }
                _,total_ls,batch_accuracy = sess.run([train_step,total_loss,accuracy], feed_dict=train_feed_dict)

                train_loss += total_ls
                train_acc += batch_accuracy
                if step == FLAGS.iters:
                    train_loss /= step  # average loss
                    train_acc /= step  # average accuracy

                    train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                                      tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

                    test_batch_x, test_batch_y = get_minibatch(val_img_list, image_dir2, FLAGS.batch_size,
                                                                 label_img_dir2)
                    test_feed_dict = {
                        x: test_batch_x,
                        label: test_batch_y,
                        learning_rate: epoch_learning_rate,
                        training_flag: False
                    }
                    loss_, acc_ = sess.run([total_loss, accuracy], feed_dict=test_feed_dict)

                    test_loss = loss_
                    test_acc = acc_

                    test_summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                            tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

                    summary_writer.add_summary(summary=train_summary, global_step=epoch)
                    summary_writer.add_summary(summary=test_summary, global_step=epoch)
                    summary_writer.flush()

                    line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
                        epoch, FLAGS.epochs, train_loss, train_acc, test_loss, test_acc)
                    print(line)
                    with open('logs_new.txt', 'a') as f:
                        f.write(line)
                    filename = 'ResDense_epoch_{:d}'.format(epoch) + '.ckpt'
                    filename = os.path.join('./output/densenet_new/', filename)
                    saver.save(sess=sess, save_path=filename)
                    
if __name__ == "__main__":
    train()
    
