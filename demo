import os
import sys
import cv2
import os.path
import numpy as np
from osgeo import gdal
import tensorflow as tf
from  libs.models.nets import res_dense_net

from  libs.configs import configs as cfg
import matplotlib.pyplot as plt
FLAGS = tf.app.flags.FLAGS

def encode_img(fliename,image_size):
    img_blobs=[]
    img_data=get_tiff_to_image_array(fliename)
    img_blob=cv2.resize(img_data, (image_size,image_size), interpolation=cv2.INTER_LINEAR)
    img_blobs.append(img_blob)
    img_height=img_data.shape[0]
    img_width=img_data.shape[1]
    return img_blobs,img_height,img_width

def get_tiff_to_image_array(tiff_image_name):
  tiff_data = gdal.Open(tiff_image_name)
  if tiff_data is None:
    print('cannot open ', tiff_image_name)
    sys.exit(1)
  img_width = tiff_data.RasterXSize
  img_height = tiff_data.RasterYSize
  im_data = tiff_data.ReadAsArray(0, 0, img_width, img_height)  # 获取数据
  image_copy = np.zeros([img_height, img_width, 3],dtype=np.float32)
  im_blueBand = im_data[0, 0:im_data.shape[1], 0:im_data.shape[2]]  # 获取蓝波段
  image_copy[:, :, 2] = im_blueBand
  im_greenBand = im_data[1, 0:im_data.shape[1], 0:im_data.shape[2]]  # 获取绿波段
  image_copy[:, :, 1] = im_greenBand
  im_redBand = im_data[2, 0:im_data.shape[1], 0:im_data.shape[2]]  # 获取红波段
  image_copy[:, :, 0] = im_redBand
  resize_image = image_copy
  return np.array(resize_image)

def visualize_mask(mask_array):
    mask_array[np.where(mask_array == mask_array.max())] = 255
    im =mask_array
    cv2.imwrite('mask.jpg', im)
    '''
    plt.figure(0)
    plt.axis('off')
    plt.imshow(im)
    '''
def test(filename):
    image_input=tf.placeholder(dtype=tf.float32,shape=(1,FLAGS.image_size,FLAGS.image_size,3),name='image_input')
    logits=res_dense_net(image_input,is_training=False)
    pred_annotation = tf.argmax(logits, dimension=3, name="prediction")
    saver = tf.train.Saver()
    assert os.path.exists(FLAGS.train_dir) ,'the path of model does not exit!'
    with tf.Session() as sess:
        image_validation,img_height,img_width=encode_img(filename,FLAGS.image_size)
        feed_dict={image_input:image_validation}
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('restored previous model %s from %s' \
                  % (ckpt.model_checkpoint_path, FLAGS.train_dir))
            pred_result=sess.run(pred_annotation,feed_dict=feed_dict)
            resize_pred_result=pred_result[0]
            resize_pred_result=cv2.resize(resize_pred_result,(img_width,img_height),interpolation=cv2.INTER_LINEAR)
            visualize_mask(resize_pred_result)

if __name__ == "__main__":
    test(r'F:\Res-DenseNetSegmentationV2.0\test\000001.tif')
