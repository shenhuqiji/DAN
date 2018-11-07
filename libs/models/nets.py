import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.models.resnets_block import attention_module
from libs.models.densenet_block import addlayer,transition_layer,up_transition_layer
def res_dense_net(input,is_training):
    conv_1=slim.conv2d(input, 16, [7,7], stride=2,scope='conv0')
    with slim.arg_scope([slim.batch_norm], is_training=is_training):
        with tf.variable_scope('block1') as scope:
            for i in range(2):
                with tf.variable_scope('layer_%d' % (i + 1)):
                    conv_1 = addlayer(conv_1,32)
            pool1 = transition_layer(conv_1,64,0.2,is_training,'trans_1')
            conv_2=pool1
        with tf.variable_scope('block2') as scope:
            for i in range(2):
                with tf.variable_scope('layer_%d' % (i + 1)):
                    conv_2 = addlayer(conv_2, 32)
            pool2 = transition_layer(conv_2,96,0.2,is_training,'trans_2')
            conv_3=pool2
        with tf.variable_scope('block3') as scope:
            for i in range(3):
                with tf.variable_scope('layer_%d' % (i + 1)):
                    conv_3 = addlayer(conv_3, 32)
            pool3 = transition_layer(conv_3,128,0.2,is_training,'trans_3')
            conv_4=pool3
        with tf.variable_scope('block4') as scope:
            for i in range(3):
                with tf.variable_scope('layer_%d' % (i + 1)):
                    conv_4 = addlayer(conv_4, 32)
            pool4 = transition_layer(conv_4,192,0.2,is_training,'trans_4')
            conv_5=pool4
        with tf.variable_scope('block5') as scope:
            for i in range(3):
                with tf.variable_scope('layer_%d' % (i + 1)):
                    conv_5 = addlayer(conv_5, 32)
        with tf.variable_scope('upblock6') as scope:
            ###up6
            conv_t1=slim.conv2d_transpose(conv_5,int(conv_4.get_shape()[-1]),3,stride=2,padding='SAME', activation_fn=tf.nn.relu)
            merge1=attention_module(conv_t1,conv_4)
            upconv_6=merge1
            for i in range(3):
                with tf.variable_scope('layer_%d' % (i + 1)):
                    upconv_6 = addlayer(upconv_6, 32)
            upconv_6=up_transition_layer(upconv_6, 256, 0.2, is_training, 'trans_6')
        with tf.variable_scope('upblock7') as scope:
            ###up7
            conv_t2 = slim.conv2d_transpose(upconv_6, int(conv_3.get_shape()[-1]),3,stride=2,padding='SAME', activation_fn=tf.nn.relu)
            merge2 = attention_module(conv_t2, conv_3)
            upconv_7 = merge2
            for i in range(3):
                with tf.variable_scope('layer_%d' % (i + 1)):
                    upconv_7 = addlayer(upconv_7, 32)
            upconv_7 = up_transition_layer(upconv_7, 256, 0.2, is_training, 'trans_7')
        with tf.variable_scope('upblock8') as scope:
            ###up8
            conv_t3 = slim.conv2d_transpose(upconv_7,  int(conv_2.get_shape()[-1]),3,stride=2,padding='SAME', activation_fn=tf.nn.relu )
            merge3 = attention_module(conv_t3, conv_2)
            upconv_8 = merge3
            for i in range(2):
                with tf.variable_scope('layer_%d' % (i + 1)):
                    upconv_8 = addlayer(upconv_8, 32)
            upconv_8 = up_transition_layer(upconv_8, 128, 0.2, is_training, 'trans_8')
        with tf.variable_scope('upblock9') as scope:
            ###up9
            conv_t4 =slim.conv2d_transpose(upconv_8, int(conv_1.get_shape()[-1]),3,stride=2,  padding='SAME', activation_fn=tf.nn.relu)
            merge4 = attention_module(conv_t4, conv_1)
            upconv_9 = merge4
            for i in range(2):
                with tf.variable_scope('layer_%d' % (i + 1)):
                    upconv_9 = addlayer(upconv_9, 32)
            upconv_9 = up_transition_layer(upconv_9, 128, 0.2, is_training, 'trans_9')
        with tf.variable_scope('upblock10') as scope:
            ###up10
            conv_t5 =slim.conv2d_transpose(upconv_9, int(conv_1.get_shape()[-1]),3,stride=2,  padding='SAME', activation_fn=tf.nn.relu)
            upconv_10 = conv_t5
            for i in range(2):
                with tf.variable_scope('layer_%d' % (i + 1)):
                    upconv_10 = addlayer(upconv_10, 32)
        conv11 = slim.conv2d(upconv_10,2,1,activation_fn=tf.nn.relu)
    return conv11

def computer_losses(logits,labels):
    flat_logits = tf.reshape(logits, [-1, 2])
    flat_labels = tf.reshape(labels, [-1, 2])
    pred_annotation =tf.argmax(logits, dimension=3, name="prediction")
    y_labels=tf.arg_max(labels,dimension=3,name="label")
    ##compute accuracy

    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(pred_annotation,y_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # logits=logits,labels= tf.squeeze(labels,squeeze_dims=[3]
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,labels=flat_labels))
    regular_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regular_loss = tf.add_n(regular_losses)
    total_loss=loss+regular_loss
    return total_loss,loss,regular_loss,accuracy

def net_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  """Defines the default ResNet arg scope.

  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.

  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.avg_pool2d], padding='SAME') as arg_sc:
        return arg_sc

