import os
import torch
import numpy as np
import torch.nn.functional as F
from skimage import io
from image_alignment import ImgAligner

class PreProcess:
    def __init__(self, sample, index, transform):
        self.transform = transform
        self.sample = sample
        self.index = index
        self.path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".") + r'/data'
        self.path_raw = self.path + r'/rawData/{}'.format(self.sample)
        self.path_normalize = self.path + r'/{}/origin/normalize'.format(self.sample)
        self.path_align = self.path + r'/{}/origin/align/{}.tif'.format(self.sample, self.sample)
        self.path_clear = self.path + r'/{}/block/clear/{}.tif'.format(self.sample, self.sample)
        self.path_blur = self.path + r'/{}/block/blur/{}.tif'.format(self.sample, self.sample)
        self.size_img = 2048
        self.channel = len(os.listdir(self.path + r'/rawData/{}'.format(self.sample)))
    def normalization(self):
        sequence_raw = np.zeros([self.channel, self.size_img, self.size_img], dtype=np.float32)
        for i in range(self.channel):
            sequence_raw[i] = io.imread(self.path_raw + r'/Image{}.tif'.format(i)).astype(np.float32)
        sequence_raw = (sequence_raw - sequence_raw.min()) / (sequence_raw.max() - sequence_raw.min())
        for i in range(self.channel):
            io.imsave(self.path_normalize + r'/Image{}.tif'.format(i), sequence_raw[i])
    def aligner(self):
        aligner = ImgAligner(sample=self.sample, tansform=self.transform)
        aligner.exec()
    def cut(self):
        sequence_align = io.imread(self.path_align)
        sequence_clear = sequence_align[:, self.index[0][0]:self.index[1][0], self.index[0][1]:self.index[1][1]]
        io.imsave(self.path_clear, sequence_clear)
    def blur(self):
        kernel = 3
        weight = torch.ones([self.channel, 1, kernel, kernel]) * 1 / kernel ** 2
        sequence_clear = io.imread(self.path_clear)
        sequence_clear = torch.from_numpy(sequence_clear)
        sequence_clear = sequence_clear.unsqueeze(0)
        sequence_blur = F.conv2d(
            input=sequence_clear,
            weight=weight,
            bias=None,
            stride=1,
            padding='same',
            groups=self.channel
        )
        sequence_blur = sequence_blur.numpy()[0]
        io.imsave(self.path_blur, sequence_blur)
    def exec(self):
        print('Image Normalization')
        self.normalization()
        print('Image Alignment')
        self.aligner()
        print('Image Cut')
        self.cut()
        print('Creating PSF')
        self.blur()