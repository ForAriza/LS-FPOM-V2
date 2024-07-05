import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
class PostProcess:
    def __init__(self, index_sample):
        self.index_sample = index_sample
        self.list_sample = ['tail']
        self.sample = self.list_sample[self.index_sample]
        self.parameter = [
            '0.300000_0.030000_0.030000-0.200000_0.200000_0.600000_0.400000'
        ]
        self.path = os.path.abspath(
            os.path.dirname(os.getcwd())
            + os.path.sep + "."
        ) + r'/data'
        self.path_result = self.path + r'/{}/result/{}'.format(
            self.sample,
            self.parameter[self.index_sample]
        )
        self.path_clear = self.path + r'/{}/block/clear/{}.tif'.format(self.sample, self.sample)
        self.path_oucs = self.path_result + r'/OUCs.tif'
        self.path_alphas = self.path_result + r'/Alphas.tif'
        self.path_total = self.path_result + r'/Total.tif'
        self.path_fluorescence = self.path_result + r'/Fluorescence.tif'
        self.path_ouc_alpha_mapping = self.path_result + r'/com_ouc_alpha.jpeg'
        self.path_ouc_mapping = self.path_result + r'/com_ouc.tif'
        self.path_contrast = self.path_result + r'/contrast.tif'
        self.shape = io.imread(self.path_clear).shape
        self.channel = self.shape[0]
    def draw_ouc_alpha_mapping(self, list_local):
        origin = io.imread(self.path_clear).astype(np.float32)[0]
        fluorescence = io.imread(self.path_fluorescence).astype(np.float32)
        alphas = io.imread(self.path_alphas).astype(np.float32)
        oucs = io.imread(self.path_oucs).astype(np.float32)
        alphas = np.where(fluorescence[0] > 0.005, alphas, 0)
        oucs = np.where(fluorescence[0] > 0.005, oucs, 0)
        origin = 255 * (origin - origin.min()) / (origin.max() - origin.min())
        d, u, l, r = list_local[0][1], list_local[1][1], list_local[0][0], list_local[1][0]
        origin = origin[d:u, l:r]
        fluorescence = fluorescence[0, d:u, l:r]
        alpha = alphas[d:u, l:r]
        ouc = oucs[d:u, l:r]
        ex = 0.5
        mul = 61
        shape = origin.shape
        o_temp = np.ones([mul * shape[0], shape[1]], dtype=np.uint8)
        g_temp = np.ones([mul * shape[0], shape[1]], dtype=np.uint8)
        o_ex = np.ones([mul * shape[0], mul * shape[1]], dtype=np.uint8)
        g_ex = np.ones([mul * shape[0], mul * shape[1]], dtype=np.uint8)
        for i in range(shape[0]):
            o_temp[mul * i:mul * (i + 1), :] = origin[i, :]
            g_temp[mul * i:mul * (i + 1), :] = fluorescence[i, :]
        o_ex = o_ex.T
        g_ex = g_ex.T
        o_temp = o_temp.T
        g_temp = g_temp.T
        for j in range(shape[1]):
            o_ex[mul * j:mul * (j + 1), :] = o_temp[j, :]
            g_ex[mul * j:mul * (j + 1), :] = g_temp[j, :]
        o_ex = o_ex.T
        g_ex = g_ex.T
        x = (ex * (mul - 1) / 2 * ouc * np.cos(alpha * np.pi)).astype(np.int32)
        y = (ex * (mul - 1) / 2 * ouc * np.sin(alpha * np.pi)).astype(np.int32)
        x_l = np.ones([2, shape[0], shape[1]], dtype=np.int32)
        y_l = np.ones([2, shape[0], shape[1]], dtype=np.int32)
        x_index = np.ones_like(origin) * np.arange(shape[1])
        y_index = (np.arange(shape[0]) * np.ones_like(origin).T).T
        x_l[0] = mul * x_index + (mul - 1) / 2 + x
        x_l[1] = mul * x_index + (mul - 1) / 2 - x
        y_l[0] = mul * y_index + (mul - 1) / 2 - y
        y_l[1] = mul * y_index + (mul - 1) / 2 + y
        x0 = np.resize(x_l[0].T, shape[0] * shape[1])
        x1 = np.resize(x_l[1].T, shape[0] * shape[1])
        y0 = np.resize(y_l[0].T, shape[0] * shape[1])
        y1 = np.resize(y_l[1].T, shape[0] * shape[1])
        plt.figure(dpi=mul, figsize=(r - l, u - d))
        plt.imshow(
            o_ex,
            cmap=plt.get_cmap('hot'),
            interpolation='none',
            vmin=o_ex.min(),
            vmax=o_ex.max()
        )
        plt.scatter(x0, y0, color='black', s=5)
        plt.scatter(x1, y1, color='black', s=5)
        length = x0.shape[0]
        for i in range(length):
            print(i / length)
            plt.plot(
                [x0[i], x1[i]],
                [y0[i], y1[i]],
                color='black',
                linewidth=5
            )
        plt.savefig(
            self.path_ouc_alpha_mapping,
            dpi=mul
        )
        plt.close()
    def draw_ouc_mapping(self, list_local):
        total = io.imread(self.path_total).astype(np.float32)
        ouc = io.imread(self.path_oucs).astype(np.float32)
        total = (total - total.min()) / (total.max() - total.min()) * 255
        ouc = ouc * total
        ouc = (ouc - ouc.min()) / (ouc.max() - ouc.min()) * 255
        d, u, l, r = list_local[0][1], list_local[1][1], list_local[0][0], list_local[1][0]
        total = total[d:u, l:r]
        oucs = ouc[d:u, l:r]
        com_ouc = np.zeros([3, total.shape[0], total.shape[1]])
        com_ouc[0] = np.minimum(oucs * 1.5, 255)
        com_ouc[1] = total
        com_ouc[2] = total
        io.imsave(
            self.path_ouc_mapping,
            com_ouc.astype(np.uint8)
        )

    def get_line_data(self, p1, p2, list_local, index_peek, index_valley):
        img1 = io.imread(self.path_clear).astype(np.float32)[0]
        img2 = io.imread(self.path_fluorescence).astype(np.float32)[0]
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())
        d, u, l, r = list_local[0][1], list_local[1][1], list_local[0][0], list_local[1][0]
        img1 = img1[d:u, l:r]
        img2 = img2[d:u, l:r]
        if len(list(img1.shape)) == 2:
            shape = img1.shape
        mapx = np.zeros(shape)
        mapy = np.zeros(shape)
        mapx = (np.arange(shape[0]).T * np.ones_like(mapx).T).T
        mapy = np.arange(shape[1]) * np.ones_like(mapy)
        a = p1[0] - p2[0]
        b = p1[1] - p2[1]
        c = p2[0] * p1[1] - p1[0] * p2[1]
        d = np.abs(a * mapx - b * mapy + c) / np.sqrt(a ** 2 + b ** 2)
        mask = np.where(d < 1 / np.sqrt(2), d, 2)
        if np.abs(a) <= np.abs(b):
            indexx = np.argsort(mask, axis=1).T[0]
            indexx = np.delete(indexx, np.s_[p2[1] + 1:], axis=0)
            indexx = np.delete(indexx, np.s_[:p1[1]], axis=0)
            indexy = np.arange(indexx.shape[0]) + min(p1[1], p2[1])
        else:
            indexy = np.argsort(mask, axis=0)[0]
            indexy = np.delete(indexy, np.s_[p2[0] + 1:], axis=0)
            indexy = np.delete(indexy, np.s_[:p1[0]], axis=0)
            indexx = np.arange(indexy.shape[0]) + min(p1[0], p2[0])
        point_num = indexx.shape[0]
        data1 = np.zeros_like(indexx, dtype=np.float32)
        data2 = np.zeros_like(indexx, dtype=np.float32)
        for i in range(point_num):
            data1[i] = img1[indexy[i], indexx[i]]
            data2[i] = img2[indexy[i], indexx[i]]
        data2 = data1.mean() / data2.mean() * data2
        num = len(index_peek)
        peekI = np.zeros(num, dtype=np.float32)
        peekg = np.zeros(num, dtype=np.float32)
        vallyI = np.zeros(num, dtype=np.float32)
        vallyg = np.zeros(num, dtype=np.float32)
        for i in range(num):
            peekI[i] = data1[index_peek[i]]
            peekg[i] = data2[index_peek[i]]
            vallyI[i] = data1[index_valley[i]]
            vallyg[i] = data2[index_valley[i]]
        contrast_origin = np.mean(np.abs(peekI - vallyI) / (peekI + vallyI))
        contrast_fluorescence = np.mean(np.abs(peekg - vallyg) / (peekg + vallyg))
        x = np.arange(point_num)
        plt.plot(x, data1, label='origin', lw=3)
        plt.plot(x, data2, label='fluorescent component', lw=3)
        plt.legend(loc=2, fontsize=15)
        plt.ylim(0, 1)
        plt.xlabel('pixel sequence No.', fontsize=15)
        plt.ylabel('Int.(norm)', fontsize=15)
        plt.title('Michelson contrast ratio = {:.4f}'.format(contrast_fluorescence / contrast_origin), fontsize=15)
        plt.savefig(self.path_contrast)
        print(contrast_fluorescence / contrast_origin,
              contrast_origin,
              contrast_fluorescence)