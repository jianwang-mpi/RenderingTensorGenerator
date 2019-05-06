#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from opendr.camera import ProjectPoints
from opendr.renderer import TexturedRenderer, ColoredRenderer
import sys
from get_body_mesh import get_body_mesh
from hmr.hmr import HMR
from smpl_.serialization import load_model
import scipy.sparse as sp
import os
import tqdm


class PartTextureGenerator:
    def __init__(self, obj_path, model_path, img_size, batch_size=1):
        if type(img_size) is tuple:
            self.width = img_size[0]
            self.height = img_size[1]
        else:
            self.width = img_size
            self.height = img_size
        self.num_cam = 3
        self.num_theta = 72
        self.m = get_body_mesh(obj_path, trans=np.array([0, 0, 4]), rotation=np.array([np.pi / 2, 0, 0]))
        # Load SMPL model (here we load the female model)
        self.body = load_model(model_path)

        self.hmr = HMR(batch_size)

    def set_texture(self, img_bgr):
        """
        set the texture image for the human body
        :param img_bgr: image should be bgr format
        :return:
        """
        # sz = np.sqrt(np.prod(img_bgr.shape[:2]))
        # sz = int(np.round(2 ** np.ceil(np.log(sz) / np.log(2))))
        self.m.texture_image = img_bgr.astype(np.float64) / 255.
        return self.m

    def generate(self, img_bgr, texture_bgr):
        img = img_bgr
        self.set_texture(texture_bgr)
        vert_shifted, theta, cam_for_render = self.hmr.predict(img)
        pose = theta[self.num_cam: (self.num_cam + self.num_theta)]
        beta = theta[(self.num_cam + self.num_theta):]

        self.body.pose[:] = pose
        self.body.betas[:] = beta

        rn_vis = TexturedRenderer()
        rn_vis.camera = ProjectPoints(t=np.zeros(3), rt=np.zeros(3), c=cam_for_render[1:],
                                      f=np.ones(2) * cam_for_render[0], k=np.zeros(5), v=vert_shifted)
        rn_vis.frustum = {'near': 0.1, 'far': 1000., 'width': self.width, 'height': self.height}
        rn_vis.set(v=vert_shifted, f=self.m.f, vc=self.m.vc, texture_image=self.m.texture_image, ft=self.m.ft,
                   vt=self.m.vt, bgcolor=np.zeros(3))
        # rn_vis.background_image = img_bgr / 255. if img_bgr.max() > 1 else img_bgr

        out_img = rn_vis.r
        out_img = (out_img * 255).astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        silhouette_rn = ColoredRenderer()
        silhouette_rn.camera = ProjectPoints(v=self.body, rt=np.zeros(3), t=np.zeros(3),
                                             f=np.ones(2) * cam_for_render[0],
                                             c=cam_for_render[1:],
                                             k=np.zeros(5))
        silhouette_rn.frustum = {'near': 0.1, 'far': 1000., 'width': self.width, 'height': self.height}
        silhouette_rn.set(v=vert_shifted, f=self.m.f, vc=self.m.vc, bgcolor=np.zeros(3))

        return out_img, texture_dr_wrt(rn_vis, silhouette_rn.r), silhouette_rn.r


def texture_dr_wrt(texture_rn, clr_im):
    """
    Change original texture dr_wrt
    use the rendered silhouette to avoid holes in the rendered image
    change the output dr from rgb format to bgr format
    :param texture_rn:
    :param clr_im:
    :return:
    """
    IS = np.nonzero(clr_im[:, :, 0].ravel() != 0)[0]
    JS = texture_rn.texcoord_image_quantized.ravel()[IS]

    # if True:
    #     cv2.imshow('clr_im', clr_im)
    #     # cv2.imshow('texmap', texture_rn.texture_image.r)
    #     cv2.waitKey(0)

    r = clr_im[:, :, 0].ravel()[IS]
    g = clr_im[:, :, 1].ravel()[IS]
    b = clr_im[:, :, 2].ravel()[IS]
    data = np.concatenate((b, g, r))

    IS = np.concatenate((IS * 3, IS * 3 + 1, IS * 3 + 2))
    JS = np.concatenate((JS * 3, JS * 3 + 1, JS * 3 + 2))

    return sp.csc_matrix((data, (IS, JS)), shape=(texture_rn.r.size, texture_rn.texture_image.r.size))


if __name__ == '__main__':
    #
    # originPath = '/unsullied/sharefs/wangjian02/isilon-home/datasets/Market1501/data'
    # outPath = os.path.join(
    #     '/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/market1501_rendering_matrix_new')

    originPath = sys.argv[1]
    outPath = sys.argv[2]
    texture_generator = PartTextureGenerator('models/body.obj', 'models/neutral.pkl',
                                             batch_size=1, img_size=(64, 128))
    uvmapPath = 'texture_example.jpg'

    if not os.path.exists(outPath):
        os.mkdir(outPath)
        os.mkdir(os.path.join(outPath, 'bounding_box_train'))
        os.mkdir(os.path.join(outPath, 'bounding_box_test'))
        os.mkdir(os.path.join(outPath, 'query'))

    for d in ['bounding_box_train', 'bounding_box_test', 'query']:  # 'query' 'bounding_box_train' bounding_box_test
        path = os.path.join(originPath, d)

        print(path)

        for root, dirs, files in os.walk(path, topdown=False):

            for file in tqdm.tqdm(files):
                if '.jpg' not in file:
                    continue

                img_path = os.path.join(root, file)
                print(img_path)
                input_img = cv2.imread(img_path)
                input_img = input_img[:, 40:-40, :]
                input_img = cv2.resize(input_img, (64, 128))

                uvmap_path = uvmapPath
                uvmap_img = cv2.imread(uvmap_path)

                img, matrix, mask = texture_generator.generate(input_img, uvmap_img)

                temp = {
                    'mat': matrix,
                    'mask': mask
                }

                np.save(os.path.join(outPath, d, file) + '.npy', temp)
