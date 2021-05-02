"""
Mask R-CNN
Train on the toy valorant dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 valorant.py train --dataset=/path/to/valorant/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 valorant.py train --dataset=/path/to/valorant/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 valorant.py train --dataset=/path/to/valorant/dataset --weights=imagenet

    # Apply color splash to an image
    python3 valorant.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 valorant.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from skimage.transform import rescale, resize, downscale_local_mean
import cv2 as cv

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

import matplotlib.pyplot as plt

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class ValorantConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "valorant"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 4  # Background + valorant

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class ValorantDataset(utils.Dataset):

    def get_class_id(self, source, name):
        id = next(x["id"] for x in self.class_info if x["source"] == source and x["name"] == name)
        return id

    def load_valorant(self, dataset_dir, subset):
        """Load a subset of the valorant dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, "train")

        resize_width, resize_height = (640, 320)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.

        # {
        #     "project_name": "Project Name,
        #                     "create_date": "2019-10-22 19:04:16Z",
        # "export_format_version": "1.0",
        # "export_date": "2019-11-19 16:21:18Z",
        # "label_classes": [
        #     {
        #         "class_name": "cat",
        #         "color": "#1f78b44d",
        #         "class_type": "object"
        #     },
        #     {
        #         "class_name": "dog",
        #         "color": "#e31a1c4d",
        #         "class_type": "object"
        #     }
        # ],
        # "images": [
        #     {
        #         "image_name": "IMG_000002.jpg",
        #         "dataset_name": "train dataset",
        #         "width": 500,
        #         "height": 430,
        #         "image_status": "TO REVIEW",
        #         "labels": [
        #             {
        #                 "class_name": "cat",
        #                 "bbox": [102, 45, 420, 404],
        #                 "polygon": null,
        #                 "mask": null,
        #                 "z_index": 1
        #             },
        #             {...},
        #             {...}
        #         ]
        #     },
        #     {...},
        #     {...}
        # ]
        # }
        annotations = json.load(open(os.path.join(dataset_dir, "JSON.json")))
        images_path = os.path.join(dataset_dir, "images")
        # annotations = list(annotations.values())  # don't need the dict keys

        # Add classes. We have only one class to add.

        # Add classes
        for i in range(len(annotations["label_classes"])):
            self.add_class("valorant", i + 1, annotations["label_classes"][i]["class_name"])

        annotations = annotations["images"]
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        # annotations = [a for a in annotations if a['regions']]
        # print(json.dumps(annotations[0]))
        # Add images
        for a in annotations:
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            width = a["width"]
            height = a["height"]

            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['labels']) is dict:
                polygons = [{"polygon": r['polygon'], "class_name": r["class_name"]} for r in a['labels'].values()]
            else:
                polygons = [{"polygon": r['polygon'], "class_name": r["class_name"]} for r in a['labels']]

            outPolygons = []
            for poly in polygons:
                outX = []
                outY = []
                if poly["polygon"]:
                    for coord in poly["polygon"]:
                        outX.append(coord[0] / width * resize_width)
                        outY.append(coord[1] / height * resize_height)

                    outPoly = {
                        "all_points_x": outX,
                        "all_points_y": outY,
                        "name": "polygon",
                        "class_id": self.get_class_id("valorant", poly["class_name"])
                    }
                    outPolygons.append(outPoly)

            img_name = a["image_name"].split(".")[0]
            img_ext = a["image_name"].split(".")[1]
            resized_path = os.path.join(images_path, "{}_{}x{}.{}".format(img_name, resize_width, resize_height, img_ext))
            if not os.path.exists(resized_path):
                img = cv.imread(os.path.join(images_path, a["image_name"]))
                resized_img = cv.resize(img, (resize_width, resize_height))
                cv.imwrite(resized_path, resized_img)

            self.add_image(
                "valorant",
                image_id=a['image_name'],  # use file name as a unique image id
                path=resized_path,
                width=resize_width + 1, height=resize_height + 1,
                polygons=outPolygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a valorant dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "valorant":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids = []
        for i, p in enumerate(info["polygons"]):
            class_ids.append(p["class_id"])
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1  # p["class_id"]

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(bool), np.array(class_ids)

    def load_mask_bad(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']

        if info["source"] != "valorant":
            return super(self.__class__, self).load_mask(image_id)

        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(),
                                                  shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(
                occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask, class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        print("getting image path")
        info = self.image_info[image_id]
        if info["source"] == "valorant":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, epochs):
    """Train the model."""
    # Training dataset.
    dataset_train = ValorantDataset()
    dataset_train.load_valorant(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ValorantDataset()
    dataset_val.load_valorant(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30 if not epochs else int(epochs) ,
                layers='heads')
    # Convert the model.
    '''converter = tf.lite.TFLiteConverter.from_keras_model(model)

    optimize="Speed"
    if optimize=='Speed':
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    elif optimize=='Storage':
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #reduce the size of a floating point model by quantizing the weights to float16
    converter.target_spec.supported_types = [tf.float16]
    tflite_quant_model = converter.convert()

    # Save the model.
    with open('/content/drive/MyDrive/ValorantTFLite/model_{:%Y%m%dT%H%M%S}.tflite'.format(datetime.datetime.now()), 'wb') as f:
      f.write(tflite_quant_model)'''


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def color_splash(image, detection):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    print("Detections: {}".format(detection["class_ids"]))
    mask = detection['masks']
    #visualize.display_instances(image, detection['rois'], detection['masks'], detection['class_ids'],
    #                           ["bg", "Enemy", "Friendly", "Self"], detection['scores'], ax=get_ax())
    #plt.show()
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    #gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    gray = image
    # Copy color pixels from the original color image where mask is set
    color = [np.zeros(image.shape, image.dtype), np.zeros(image.shape, image.dtype), np.zeros(image.shape, image.dtype)]
    for x in range(color[0].shape[0]):
      for y in range(color[0].shape[1]):
        img_colors = np.average(image[x][y])/255
        color[0][x][y] = (img_colors*255, img_colors*76, img_colors*76)
        color[1][x][y] = (img_colors*153, img_colors*255, img_colors*51)
        color[2][x][y] = (img_colors*150, img_colors*200, img_colors*255)

    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        N = detection["rois"].shape[0]
        for i in range(N):
            m = mask[:, :, i].astype(int)
            m = np.reshape(m, (320, 640, 1))
            # gray = np.where(m, image, gray).astype(np.uint8)
            #print(detection["class_ids"][i])
            gray = np.where(m, color[detection["class_ids"][i]-1], gray).astype(np.uint8)


            """m = cv.inRange(m, 0.1, 2)
            color = np.zeros(image.shape, image.dtype)
            color[:, :] = (0, 0, 255)
            print("Mask Size: {}; Image Size: {};".format(m.shape, color.shape))
            colorMask = cv.bitwise_and(color, color, mask=m)
            plt.imshow(color)
            plt.show()
            colored_img = cv.addWeighted(colorMask, 1, image, 1, 0, image)"""

        # mask = (np.sum(mask, -1, keepdims=True) >= 1)
        """enemy_mask = (np.sum(mask, -1, keepdims=True) == 1)
        friendly_mask = (np.sum(mask, -1, keepdims=True) == 2)
        self_mask = (np.sum(mask, -1, keepdims=True) == 3)
        splash = np.where(enemy_mask, image, gray).astype(np.uint8)"""
        splash = gray


    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        resized_img = cv.resize(image, (640, 320))

        # Detect objects
        r = model.detect([resized_img], verbose=1)[0]
        # Color splash
        splash = color_splash(resized_img, r)
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        from time import time
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            start = int(round(time() * 1000))
            #print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                resized_img = cv.resize(image, (640, 320))
                # Detect objects
                r = model.detect([resized_img], verbose=0)[0]
                # Color splash
                splash = color_splash(resized_img, r)
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
            tot_time = int(time() * 1000) - start
            current_fps = 1000/tot_time
            print("Frame: {}, Time: {}, FPS: {}".format(count, tot_time, current_fps))
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect valorants.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/valorant/dataset/",
                        help='Directory of the valorant dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--epochs', required=False, metavar="number of epochs for training", help="number of epochs for training")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ValorantConfig()
    else:
        class InferenceConfig(ValorantConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.epochs)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
