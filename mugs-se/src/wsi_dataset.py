import PIL.Image
import torch
import glob
import random
import numpy as np
import os
import sys
import openslide


# class MILdataset(torch.utils.data.Dataset):
#     def __init__(self, path='', transform=None):
#         self.imgs = sorted(glob.glob(path + '/*.png'))
#         self.transform = transform
#     def __len__(self):
#         # print(len(self.imgs))
#         return len(self.imgs)
#     def __getitem__(self, index):
#         img = PIL.Image.open(self.imgs[index])
#         if self.transform is not None:
#             img = self.transform(img)
#         return img


class MILdataset(torch.utils.data.Dataset):
    def __init__(self, libraryfile='', transform=None, max_bag_size=100000, seed=0):
        random.seed(seed)
        lib = torch.load(libraryfile)
        slides = []
        patch_size = []
        targets = lib['targets']

        for i, name in enumerate(lib['slides']):
            sys.stdout.write('==>Opening SVS headers: [{}/{}]\r'.format(i + 1, len(lib['slides'])))
            sys.stdout.flush()
            slides.append(openslide.OpenSlide(name))
            patch_size.append(int(lib['patch_size'][i]))
        print('')
        #         Flatten grid
        grid = []
        slideIDX = []
        label = []
        # slideIDX列表存放每个WSI以及其坐标列表的标记,假设有0,1,2号三个WSI图像,分别于grid中记录4,7,3组提取的坐标,\
        # 返回为[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        for i, g in enumerate(lib['grid']):
            if len(g)>max_bag_size:
                grid.extend(random.sample(g, k=max_bag_size))
                slideIDX.extend([i] * max_bag_size)
                label.extend([targets[i]] * 100000)
            else:
                grid.extend(g)
                slideIDX.extend([i] * len(g))
                label.extend([targets[i]] * len(g))

        print('Number of tiles: {}'.format(len(grid)))
        self.slidenames = lib['slides']
        self.slides = slides
        # self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.size = patch_size
        self.level = lib['level']
        self.label = label


    def __getitem__(self, index):
        slideIDX = self.slideIDX[index]
        coord = self.grid[index]
        lab = int(self.label[index])

        img = self.slides[slideIDX].read_region(coord, self.level, (self.size[slideIDX], self.size[slideIDX])).convert('RGB')

        if img.size != (224, 224):
            img = img.resize((224, 224))

        # ======================
        ret = []
        if self.transform is not None:
            # for t in self.transform:
            #     ret.append(t(img))
            img = self.transform(img)
        else:
            ret.append(img)
        # ======================

        return img  #, lab


    def __len__(self):
        return len(self.grid)

