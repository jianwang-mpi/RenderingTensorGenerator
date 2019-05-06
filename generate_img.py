#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from opendr.camera import ProjectPoints
from opendr.renderer import TexturedRenderer

from get_body_mesh import get_body_mesh
from hmr.hmr import HMR
from smpl_.serialization import load_model
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

    def generate(self, img_bgr, background,texture_bgr):
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
        
        
        

        
        if background is not None:
            rn_vis.background_image = background / 255. if img_bgr.max() > 1 else img_bgr

        out_img = rn_vis.r
        out_img = (out_img * 255).astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        return out_img

def read_background():
    
        
    data_path = '/unsullied/sharefs/wangjian02/isilon-home/datasets/SURREAL/smpl_data/textures'

    PRW_img_path = '/unsullied/sharefs/wangjian02/isilon-home/datasets/PRW/frames'
    CUHK_SYSU_path = '/unsullied/sharefs/wangjian02/isilon-home/datasets/CUHK-SYSU'

    data_path_list = [PRW_img_path,CUHK_SYSU_path]

    backgrounds = []

    for data_path in data_path_list:
        for root, dirs, files in os.walk(data_path):
            for name in files:
                if name.endswith('.jpg'):
                    backgrounds.append(os.path.join(root, name))



    return backgrounds
    
    
if __name__ == '__main__':
    texture_generator = PartTextureGenerator('models/body.obj', 'models/neutral.pkl',
                                             batch_size=1, img_size=(64, 128))
    
    backgrounds = read_background()
    
    model_names = ['ImageNet_PerLoss2018-10-23_18:14:53.982469/2018-10-24_10:30:39.835040_epoch_120',
                   'NoPCB_PerLoss2018-10-23_18:16:04.651977/2018-10-24_06:20:33.259434_epoch_120',
                   'PCB_2048_256_L12018-10-23_18:13:29.746996/2018-10-24_05:17:39.706192_epoch_120',
                   'PCB_ALLCat_PerLoss2018-10-23_18:17:51.451793/2018-10-24_09:42:22.511739_epoch_120',
                   'PCB_PerLoss2018-10-23_18:16:59.216650/2018-10-24_13:27:16.867817_epoch_120',
                   'PCB_PerLoss_NoPosed2018-10-26_09:27:58.815980/2018-10-26_16:11:01.006521_epoch_120',
                   'PCB_RGB_L12018-10-23_18:12:42.827038/2018-10-23_23:51:33.516745_epoch_120',
                   'PCB_softmax2018-10-23_18:18:39.775789/2018-10-24_05:05:52.977378_epoch_120',
                   'PCB_TripletHard2018-10-23_18:20:48.070572/2018-10-24_04:35:05.054042_epoch_120']
    
    model_names = ['NoPCB_Resnet_correct_size2018-11-09_10:18:35.348586/2018-11-09_21:46:11.996109_epoch_120']
    
    model_root = '/unsullied/sharefs/zhongyunshan/isilon-home/model-parameters/Texture'
    
    for model_name in model_names:
        print(model_name)
        
        # model_path = os.path.join(model_root,model_name)

        model = 'PCB_256_L12018-11-16_17:53:20.894085_epoch_120'
        # model = model_path[model_path.find('/',61)+1:model_path.find('/',69)]+'_'+model_path[model_path.find('epoch'):]
        
    

        originPath = '/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/market-origin-ssim'

        uvmapPath = os.path.join('/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/market-uvmap',model)

        texturePath_ssim = os.path.join('/unsullied/sharefs/zhongyunshan/isilon-home/datasets/Texture/market-textured-ssim',model)

        if not os.path.exists(texturePath_ssim):
            os.mkdir(texturePath_ssim)
            

        path = originPath

        for root, dirs, files in os.walk(path, topdown=False): 

            for file in tqdm.tqdm(files):
                if '.jpg' not in file:
                    continue


                img_path = os.path.join(root,file)
                input_img = cv2.imread(img_path)

                uvmap_path = os.path.join(uvmapPath,file)
                uvmap_img = cv2.imread(uvmap_path)
                
                background = cv2.imread(backgrounds[np.random.randint(len(backgrounds), size=1)[0]])
                background = cv2.resize(background, (64, 128))

                background[:,:,0] = 0
                background[:,:,1] = 0
                background[:,:,2] = 0
                img = texture_generator.generate(input_img,background,uvmap_img)

                cv2.imwrite(os.path.join(texturePath_ssim,file),img)

