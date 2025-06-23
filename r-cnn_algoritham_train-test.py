import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Download model from TF2 Detection Zoo
MODEL_DATE = '20200711'
MODEL_NAME = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8'
PATH_TO_CKPT = f'http://download.tensorflow.org/models/object_detection/tf2/{MODEL_DATE}/{MODEL_NAME}.tar.gz'

import tarfile
import urllib.request

if not os.path.exists(MODEL_NAME):
    urllib.request.urlretrieve(PATH_TO_CKPT, f'{MODEL_NAME}.tar.gz')
    tar = tarfile.open(f'{MODEL_NAME}.tar.gz')
    tar.extractall()
    tar.close()

# Load pipeline config and build model
pipeline_config = f'{MODEL_NAME}/pipeline.config'
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(f'{MODEL_NAME}/checkpoint/ckpt-0').expect_partial()



############### step 2

import cv2

def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Load image
img_path = "your_image.jpg"
img_np = cv2.imread(img_path)
input_tensor = tf.convert_to_tensor([img_np], dtype=tf.uint8)

# Run detection
detections = detect_fn(input_tensor)

# Visualize results
label_map_path = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(label_map_path)

image_np_with_detections = img_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'][0].numpy(),
    (detections['detection_classes'][0].numpy() + 1).astype(int),
    detections['detection_scores'][0].numpy(),
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=5,
    min_score_thresh=0.5,
    agnostic_mode=False
)

plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


######### STep 2 #############################################3



import cv2

import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# === STEP 1: LOAD THE MODEL ===
# Replace with your saved model directory path
model_dir = "models/my_rcnn_model/saved_model"

# Load saved model
detect_fn = tf.saved_model.load(model_dir)

# === STEP 2: LOAD LABEL MAP ===
# This should match the labels used during training (like COCO)
label_map_path = "models/research/object_detection/data/mscoco_label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

# === STEP 3: LOAD IMAGE ===
image_path = "your_image.jpg"  # Replace with your image path
img_np = cv2.imread(image_path)
if img_np is None:
    raise ValueError(f"Could not load image from {image_path}")

# Convert to tensor
input_tensor = tf.convert_to_tensor([img_np], dtype=tf.uint8)

# === STEP 4: RUN DETECTION ===
detections = detect_fn(input_tensor)

# === STEP 5: VISUALIZE RESULTS ===
image_np_with_detections = img_np.copy()

# Extract detection results
boxes = detections['detection_boxes'][0].numpy()
classes = (detections['detection_classes'][0].numpy() + 1).astype(np.int32)
scores = detections['detection_scores'][0].numpy()

# Draw boxes and labels
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    boxes,
    classes,
    scores,
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=10,
    min_score_thresh=0.5,
    agnostic_mode=False
)

# === STEP 6: SHOW IMAGE ===
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Detected Objects")
plt.show()
































