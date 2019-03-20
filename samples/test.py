import cv2
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from mrcnn.model import MaskRCNN
import mrcnn.config
import mrcnn.utils

def load_mask_model():
    # Root directory of the project
    ROOT_DIR = os.path.abspath(".")

    print(ROOT_DIR)

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn import visualize
    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
    # import coco
    # from samples.coco import coco

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")

    class InferenceConfig(mrcnn.config.Config):
        NAME = "coco_pretrained_model_config"
        IMAGES_PER_GPU = 1
        GPU_COUNT = 1
        NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
        DETECTION_MIN_CONFIDENCE = 0.6

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    return model



def get_image_mask(model, input_image , out_image_dir , frame_id):


    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']



    img_dir = out_image_dir
    image = input_image

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]

    boxes = r['rois']
    classes = r['class_ids']
    n = boxes.shape[0]


    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for j in range(n):
        if class_names[classes[j]] == 'person':
            y1, x1, y2, x2 = boxes[j]
            image[y1:y2, x1:x2, :] = 255
            mask[y1:y2, x1:x2] = 255


    plt.imshow(image)
    plt.show()
    plt.imshow(mask)
    plt.show()
    plt.imsave(str(img_dir)+'_' + str(frame_id) + '_input.png', image)
    plt.imsave(str(img_dir)+'_' + str(frame_id) + '_mask.png', mask , cmap=matplotlib.cm.gray, vmin=0, vmax=255)


