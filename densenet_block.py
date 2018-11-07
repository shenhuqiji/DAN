import tensorflow as tf
import tensorflow.contrib.slim as slim
def addlayer(input,out_feature):
    comp_out =slim.conv2d(input, out_feature, [3, 3], stride=1,scope='composite_function')
    # concatenate _input with out from composite function
    output = tf.concat(axis=3, values=(input, comp_out))
    return output
def transition_layer(x, filters,dropou_rate,is_training,scope):
    with tf.name_scope(scope):
        x = slim.conv2d(x, filters, [1,1], stride=1,scope=scope+'_conv1')
        x =slim.dropout(x,dropou_rate,is_training=is_training)
        x = slim.avg_pool2d(x,[2,2], stride=2,padding="SAME")
    return x
def up_transition_layer(x, filters,dropou_rate,is_training,scope):
    with tf.name_scope(scope):
        x = slim.conv2d(x, filters, [1,1], stride=1,scope=scope+'_conv1')
        x =slim.dropout(x,dropou_rate,is_training=is_training)
    return x
