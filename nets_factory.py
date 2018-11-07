import tensorflow as tf
from libs.models.nets import res_dense_net as model
from libs.models.nets import net_arg_scope
slim = tf.contrib.slim
def get_network(image, weight_decay=0.000005, is_training=False):
    with slim.arg_scope(net_arg_scope(weight_decay=weight_decay)):
            logits = model(image, is_training=is_training)
    return logits
if __name__ == "__main__":
    _input=tf.placeholder(dtype=tf.float32,shape=(6,352,352,3))
    get_network(_input,weight_decay=0.000005, is_training=True)