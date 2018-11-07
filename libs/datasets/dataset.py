import os.path
def _load_image_list(dataset_dir,image_set_name):
  """
  Load the indexes listed in this dataset's image set file.
  """
  # Example path to image set file:
  image_set_file = os.path.join(dataset_dir,image_set_name + '.txt')
  assert os.path.exists(image_set_file), \
    'Path does not exist: {}'.format(image_set_file)
  with open(image_set_file) as f:
    image_index = [x.strip() for x in f.readlines()]
  return image_index
