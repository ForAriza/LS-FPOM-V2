import numpy as np
import matplotlib.pyplot as plt
import cv2
import datetime
import os
from skimage import io
class ImgAligner:
    def __init__(self, sample, tansform):
        self.transform = tansform
        self.sample = sample
        self.path = os.path.abspath(
            os.path.dirname(os.getcwd())
            + os.path.sep + "."
        ) + r'/data'
        self.path_normalize = self.path + r'/{}/origin/normalize/Image.tif'.format(self.sample)
        self.path_match = self.path + r'/{}/origin/match/Image.tif'.format(self.sample)
        self.path_compare = self.path + r'/{}/origin/compare/Image.tif'.format(self.sample)
        self.path_align = self.path + r'/{}/origin/align/{}.tif'.format(self.sample, self.sample)
        self.dirResultSave = self.path + r'/{}/origin/result'.format(self.sample)
        self.size_img = 2048
        self.ratio = 0.8
        self.corDis = 10
        self.channel = len(os.listdir(self.path + r'/rawData/{}'.format(self.sample)))
    def get_sequence(self, path):
        sequence = np.zeros([self.size_img,
                             self.size_img,
                             self.channel],
                            dtype=np.float32)
        for i in range(self.channel):
            sequence[:, :, i] = io.imread(path[:-4] + '{}.tif'.format(i))
        return sequence
    def sift_compare(self, img_query, img_train, save_path):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img_query, None)
        kp2, des2 = sift.detectAndCompute(img_train, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        cor_area = np.array([self.corDis, self.corDis])
        img_temp = np.copy(img_query)
        img_temp = cv2.UMat(img_temp)
        img_train = cv2.UMat(img_train)
        compare = [[], []]
        for m, n in matches:
            if m.distance < self.ratio * n.distance:
                length = min(len(kp1), len(kp2))
                if max(m.trainIdx, m.queryIdx) < length:
                    train_pt2 = np.array(kp2[m.trainIdx].pt)
                    query_pt1 = np.array(kp1[m.queryIdx].pt)
                    check = np.abs(query_pt1 - train_pt2) - cor_area
                    if max(check[0], check[1]) <= 0:
                        compare[0].append(query_pt1)
                        compare[1].append(train_pt2)
                        if save_path != r'xxx':
                            img_query = cv2.circle(
                                img_temp,
                                center=[
                                    int(np.rint(query_pt1[0])),
                                    int(np.rint(query_pt1[1]))
                                ],
                                radius=10,
                                color=(0, 0, 255),
                                thickness=5
                            )
                            img_train = cv2.circle(
                                img_train,
                                center=[
                                    int(np.rint(train_pt2[0])),
                                    int(np.rint(train_pt2[1]))
                                ],
                                radius=10,
                                color=(0, 0, 255),
                                thickness=5
                            )
                        good_matches.append([m])
        if len(good_matches) < 3:
            if len(good_matches) == 0:
                compare[0].append(query_pt1 * 0)
                compare[1].append(train_pt2 * 0)
                for num_add in range(3 - len(good_matches)):
                    compare[0].append(query_pt1 / 2 + num_add)
                    compare[1].append(train_pt2 / 2 + num_add)
            print('Less Key Points {}'.format(3 - len(good_matches)))
        if save_path != r'xxx':
            img_match = cv2.drawMatchesKnn(
                img_query,
                kp1,
                img_train,
                kp2,
                good_matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imwrite(save_path, img_match)
        return np.array(compare)
    def get_list_compare(self, sequence, save_dir):
        time_sift1 = datetime.datetime.now()
        index_middle = int(self.channel / 2)
        list_compare = []
        img_query = sequence[:, :, index_middle]
        for i in range(self.channel):
            if save_dir != r'xxx':
                save_path = save_dir[:-4] + '{}.tif'.format(i)
            else:
                save_path = save_dir
            if i == index_middle:
                list_compare.append(
                    np.array(
                        [[[0, 0],
                          [0, self.size_img - 1],
                          [index_middle, index_middle],
                          [self.size_img - 1, 0],
                          [self.size_img - 1, self.size_img - 1]
                          ],
                         [
                             [0, 0],
                             [0, self.size_img-1],
                             [index_middle, index_middle],
                             [self.size_img - 1, 0],
                             [self.size_img - 1, self.size_img - 1]]]
                    )
                )
                continue
            img_train = sequence[:, :, i]
            compare = self.sift_compare(
                img_query=img_query,
                img_train=img_train,
                save_path=save_path
            )
            list_compare.append(compare)
        time_sift2 = datetime.datetime.now()
        print('time_sift', time_sift2 - time_sift1)
        return list_compare
    def get_bias(self, list_compare):
        biases = np.zeros([self.channel, 4])
        for i in range(self.channel):
            shape = list_compare[i].shape
            bias_ave = np.mean(list_compare[i][0, :] - list_compare[i][1, :], axis=0)
            bias = np.array([i, shape[1], bias_ave[0], bias_ave[1]])
            biases[i, :] = bias
        print(biases)
        for i in range(self.channel):
            num_nan_x = np.count_nonzero(biases[:, 2] != biases[:, 2])
            num_nan_y = np.count_nonzero(biases[:, 3] != biases[:, 3])
            if num_nan_x != 0:
                biases[:, 2][np.isnan(biases[:, 2])] \
                    = np.mean(biases[:, 2][biases[:, 2] == biases[:, 2]])
            if num_nan_y != 0:
                biases[:, 3][np.isnan(biases[:, 3])] \
                    = np.mean(biases[:, 3][biases[:, 3] == biases[:, 3]])
        return biases
    def image_translate(self, list_compare, sequence):
        sequence_compare = np.zeros([self.size_img + 2 * self.corDis,
                                      self.size_img + 2 * self.corDis,
                                      self.channel],
                                     dtype=np.float32)
        biases = self.get_bias(list_compare=list_compare)
        for i in range(self.channel):
            x_init = int(np.rint(self.corDis + biases[i, 2]))
            y_init = int(np.rint(self.corDis + biases[i, 3]))
            sequence_compare[y_init:y_init + self.size_img,
            x_init:x_init + self.size_img, i] = sequence[:, :, i]
        sequence_translate = np.copy(
            sequence_compare[self.corDis:-self.corDis,
            self.corDis:-self.corDis, :]
        )
        return sequence_translate, biases
    @staticmethod
    def get_h_affine(fp, tp):
        m = np.mean(fp[:2], axis=1)
        std_max = max(np.std(fp[:2], axis=1)) + 1e-9
        c1 = np.diag([1 / std_max, 1 / std_max, 1])
        c1[0, 2] = -m[0] / std_max
        c1[1, 2] = -m[1] / std_max
        fp_cond = np.dot(c1, fp)
        m = np.mean(tp[:2], axis=1)
        c2 = c1.copy()
        c2[0, 2] = -m[0] / std_max
        c2[1, 2] = -m[1] / std_max
        tp_cond = np.dot(c2, tp)
        a = np.concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
        u, s, v = np.linalg.svd(a.T)
        tmp = v[:2].T
        b = tmp[:2]
        c = tmp[2:4]
        tmp2 = np.concatenate([np.dot(c, np.linalg.pinv(b)), np.zeros((2, 1))], axis=1)
        h = np.vstack(tmp2)
        h = np.concatenate([h, np.array([[0, 0, 1]])], axis=0)
        h = np.dot(np.linalg.inv(c2), np.dot(h, c1))
        return h / h[2, 2]
    def warp_affine(self, list_compare, sequence):
        shape_sequence = sequence.shape
        index_middle = int(self.channel / 2)
        sequence_warp_affine = np.copy(sequence)
        for i in range(self.channel):
            num_points = int(list_compare[i][0].shape[0])
            fp = np.concatenate([list_compare[i][0, :, :],
                                 np.ones([num_points, 1])], axis=1).T
            tp = np.concatenate([list_compare[i][1, :, :],
                                 np.ones([num_points, 1])], axis=1).T
            h = self.get_h_affine(fp=fp, tp=tp)
            if i == index_middle:
                h = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            w = np.float64(h[:2, :])
            sequence_warp_affine[:, :, i] = cv2.warpAffine(
                sequence[:, :, i], w,
                (shape_sequence[1], shape_sequence[0])
            )
        return sequence_warp_affine
    def rotate_sequence(self, list_compare, sequence):
        shape_sequence = sequence.shape
        index_middle = int(self.channel / 2)
        sequence_rotate = np.copy(sequence)
        thetas = []
        for i in range(self.channel):
            compares_translation = list_compare[i] - int(shape_sequence[0] / 2)
            x_res = compares_translation[0, :, 0]
            x_com = compares_translation[1, :, 0]
            y_res = compares_translation[0, :, 1]
            y_com = compares_translation[1, :, 1]
            k_res = np.dot(x_res, y_res) / np.dot(x_res, x_res)
            k_com = np.dot(x_com, y_com) / np.dot(x_com, x_com)
            theta = np.arctan((k_res - k_com + 1e-11) / (1 + k_res * k_com + 1e-11))
            thetas.append(theta)
            matrix_rotate = cv2.getRotationMatrix2D(
                (index_middle, index_middle), theta, 1.0
            )
            sequence_rotate[:, :, i] = cv2.warpAffine(
                sequence[:, :, i], matrix_rotate,
                (shape_sequence[1], shape_sequence[0])
            )
        return thetas
    def sum_sequence(self, path, save_dir):
        for i in range(len(path)):
            save_path = save_dir + r'/sum_' + path[i].split(r'/')[-2] + ".tif"
            sequence = self.get_sequence(path[i])
            img_sum = sequence.sum(axis=2) / self.channel
            img_sum = img_sum.astype(np.float32)
            io.imsave(save_path, img_sum)
    def draw_biases(self, bias, save_dir):
        biases = [np.zeros([self.channel, 4]), np.zeros([self.channel, 4])]
        biases[0] = bias
        sequence = self.get_sequence(path=self.path_compare)
        sequence_int = (255 * sequence / sequence.max()).astype(np.uint8)
        list_compare = self.get_list_compare(sequence=sequence_int, save_dir=r'xxx')
        biases[1] = self.get_bias(list_compare=list_compare)
        list_x = np.abs(biases[0][:, 2]) + np.abs(biases[0][:, 3])
        list_y = np.abs(biases[1][:, 2]) + np.abs(biases[1][:, 3])
        degree = 4
        x = np.arange(self.channel)
        parameter_x = np.polyfit(x, list_x, 4)
        parameter_y = np.polyfit(x, list_y, 4)
        px = np.poly1d(parameter_x)
        py = np.poly1d(parameter_y)
        plt.plot(x, px(x), label='origin')
        plt.plot(x, py(x), label='compare')
        plt.scatter(x, list_x)
        plt.scatter(x, list_y)
        plt.legend(loc=2)
        plt.xlabel('Polarization Angle')
        plt.ylabel('Deviation(Absolute Distance)')
        plt.title('Image Alignment(SIFT)')
        f = plt.gcf()
        save_path = save_dir + r'/abs_{}.png'.format(degree)
        f.savefig(save_path)
        f.clear()
    def exec(self):
        raw_path = [self.path_normalize, self.path_compare]
        sequence = self.get_sequence(path=self.path_normalize)
        sequence_int = (255 * sequence / sequence.max()).astype(np.uint8)
        sequence_aligned = None
        biases_raw = None
        if self.transform == 'affine':
            list_compare = self.get_list_compare(
                sequence=sequence_int,
                save_dir=self.path_match
            )
            sequence_warp_affine = self.warp_affine(
                list_compare=list_compare,
                sequence=sequence
            )
            sequence_aligned = sequence_warp_affine
        if self.transform == 'rotate':
            list_compare = self.get_list_compare(
                sequence=sequence_int,
                save_dir=self.path_match
            )
            sequence_rotate = self.rotate_sequence(
                list_compare=list_compare,
                sequence=sequence
            )
            sequence_aligned = sequence_rotate
        if self.transform == 'translate':
            list_compare = self.get_list_compare(
                sequence=sequence_int,
                save_dir=self.path_match
            )
            sequence_translate, biases_raw = self.image_translate(
                list_compare=list_compare,
                sequence=sequence
            )
            sequence_aligned = sequence_translate
        for i in range(self.channel):
            io.imsave(
                self.path_compare[:-4]
                + '{}.tif'.format(i),
                sequence_aligned[:, :, i]
            )
        sequence_aligned = np.swapaxes(sequence_aligned, 0, 2)
        sequence_aligned = np.swapaxes(sequence_aligned, 1, 2)
        io.imsave(self.path_align, sequence_aligned)
        self.draw_biases(bias=biases_raw, save_dir=self.dirResultSave)
        self.sum_sequence(path=raw_path, save_dir=self.dirResultSave)