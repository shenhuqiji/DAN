import sys
import glob
import scipy
import cv2
import random
import numpy as np
import skimage.io
import os.path
def get_minibatch(image_list,image_dir,batch_size,label_img_dir):
  img_blobs=[]
  ground_truths=[]

  for _ in range(batch_size):
    image_index=random.randrange(65536)
    mod_index = image_index % len(image_list)
    img_name=image_list[mod_index]
    image_full_path=os.path.join(image_dir, img_name+ '.tif')
#    img_data=skimage.io.imread(image_full_path)
    img_data = cv2.imread(image_full_path)
    img_blob, window, scale, padding = resize_image(
        img_data,
        min_dim=200,
        max_dim=256,
        padding=True)
    ground_truth__path=os.path.join(label_img_dir, img_name+'.png')
    if len(glob.glob(ground_truth__path))<1:
        ground_truth__path = os.path.join(label_img_dir, img_name + '.tif')
    ground_truth=get_mask(ground_truth__path)
    ground_truth_blob=resize_mask(ground_truth,scale, padding)
    if random.randint(0, 1):
        img_blob = np.fliplr(img_blob)
        ground_truth_blob = np.fliplr(ground_truth_blob)
    img_blobs.append(img_blob)
    ground_truths.append(ground_truth_blob)
  return img_blobs,ground_truths

def get_mask(mask_img):
    instance_masks = []
    img_grey = cv2.imread(mask_img,0)
    #img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(2):
        m=np.zeros([img_grey.shape[0],img_grey.shape[1]],dtype=np.uint8)
        #img_grey[np.where(img_grey ==i)] = 1
        m[np.where(img_grey ==i)]=1
        instance_masks.append(m)
    #instance_masks.append(img_grey)
    mask=np.stack(instance_masks, axis=2)
    return mask
def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask
