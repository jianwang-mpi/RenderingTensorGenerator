"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from hmr.src.RunModel import RunModel
from hmr.src import config as hmr_config
from hmr.src.util import renderer as vis_util
from hmr.src.util import image as img_util

class HMR:
    def __init__(self, batch_size=1):
        config = hmr_config.get_config()
        # Using pre-trained model, change this to use your own.
        config.load_path = config.PRETRAINED_MODEL

        config.batch_size = batch_size
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        self.model = RunModel(config, sess)
        self.config = config

    def preprocess_image(self, img, config):
        img = img[..., ::-1]
        if img.shape[2] == 4:
            img = img[:, :, :3]

        if np.max(img.shape[:2]) != config.img_size:
            # print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]

        crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                                   config.img_size)

        # Normalize image to [-1, 1]
        crop = 2 * ((crop / 255.) - 0.5)

        return crop, proc_param, img

    def visualize(self, img, proc_param, joints, verts, cam):
        """
        Renders the result in original image coordinate frame.
        """
        cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
            proc_param, verts, cam, joints, img_size=img.shape[:2])
        return cam_for_render, vert_shifted

    def predict(self, img_bgr):
        crop, proc_param, img = self.preprocess_image(img_bgr, self.config)
        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(crop, 0)

        joints, verts, cams, joints3d, theta = self.model.predict(
            input_img, get_theta=True)

        cam_for_render, vert_shifted = self.visualize(img, proc_param, joints[0], verts[0], cams[0])

        return vert_shifted, theta[0], cam_for_render


