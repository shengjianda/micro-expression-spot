import cv2
import dlib
from matplotlib import pyplot as plt
import multiprocessing
import numpy as np
import os
import pickle
from tqdm import tqdm

from util import *


class FaceDetector4Alignment(object):
    ...


class LmkDetector4Alignment(object):
    ...


class DlibFaceDetector(FaceDetector4Alignment):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def __call__(self, img):
        faces = self.detector(img)
        faces = np.array([((i_face.left(), i_face.top()), (i_face.right(), i_face.bottom())) for i_face in faces])
        return faces


class DlibLmkDetector(LmkDetector4Alignment):
    left_eye_idx = 39
    right_eye_idx = 42
    nose_idx = [29, 30]

    def __init__(self):
        self.detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def __call__(self, img, box):
        box = dlib.rectangle(box[0, 0], box[0, 1], box[1, 0], box[1, 1])
        return np.array([[p.x, p.y] for p in self.detector(img, box).parts()])


class Alignment(object):
    def __init__(self, face_detector, lmk_detector, flow_fun, size=256):
        self.face_detector = face_detector
        self.lmk_detector = lmk_detector
        self.flow_fun = flow_fun
        self.flow_args = {'flow': None, 'pyr_scale': 0.5, 'levels': 3, 'winsize': 15,
                          'iterations': 5, 'poly_n': 7, 'poly_sigma': 1.5, 'flags': 0}
        self.size = (size, size)

        self.left = None
        self.down = None
        self.width = None
        self.height = None
        self.lmk_face = None
        self.lmk_face_gray = None
        self.lmk = None
        self.roi_tip_idx = None

    def reset(self):
        self.left = None
        self.down = None
        self.width = None
        self.height = None
        self.lmk_face = None
        self.lmk = None
        self.roi_tip_idx = None

    @staticmethod
    def get_face_idx(l, d, w, h):
        return slice(int(d), int(d)+h), slice(int(l), int(l)+w)

    def init_face_lmk(self, img):
        """
        根据图片计算原始图片上的人脸框位置
        把人脸扣出来用于对齐，并保存人脸对应的关键点位置
        :param img:
        :return:
        """
        # face box, not stable
        faces = self.face_detector(img)
        face = faces[-1]
        # lmk
        lmk = self.lmk_detector(img, face)
        # stable box by lmk
        left_eye = lmk[self.lmk_detector.left_eye_idx]
        right_eye = lmk[self.lmk_detector.right_eye_idx]
        center_eye = (left_eye + right_eye) / 2
        d_eye = (right_eye - left_eye)[0]
        self.left = max(center_eye[0] - 2 * d_eye, 0)
        self.down = max((center_eye[1] - 1.5 * d_eye), 0)
        self.width = np.round(4 * d_eye)
        self.height = np.round(4 * d_eye)
        # lmk resized, float 2 int
        face_slice_idx = self.get_face_idx(self.left, self.down, self.width, self.height)
        self.lmk_face = img[face_slice_idx]
        self.lmk = lmk - np.array([face_slice_idx[1].start, face_slice_idx[0].start])
        if self.size is not None:
            self.lmk_face = cv2.resize(self.lmk_face, self.size)
            self.lmk = self.lmk / np.array([self.width, self.height]) * np.array(self.size)
        self.lmk_face_gray = cv2.cvtColor(self.lmk_face, cv2.COLOR_RGB2GRAY)
        # tip roi
        roi_tip_min, roi_tip_max = get_roi_bound(self.lmk[self.lmk_detector.nose_idx], expand_pixel=13)
        roi_tip_min = roi_tip_min.astype(np.int32)
        roi_tip_max = roi_tip_max.astype(np.int32)
        self.roi_tip_idx = slice(roi_tip_min[1], roi_tip_max[1]), slice(roi_tip_min[0], roi_tip_max[0])

    def next(self, img):
        face_left_n = self.left
        face_down_n = self.down
        dr = np.zeros(2)
        for _ in range(4):
            face_left_n = max(0, face_left_n + dr[0])
            face_down_n = max(0, face_down_n + dr[1])
            img_face_n = img[self.get_face_idx(face_left_n, face_down_n, self.width, self.height)]
            img_face_n = cv2.resize(img_face_n, self.size)
            img_face_gray_n = cv2.cvtColor(img_face_n, cv2.COLOR_RGB2GRAY)

            flow = self.flow_fun(self.lmk_face_gray, img_face_gray_n, **self.flow_args)
            flow_tip = flow[self.roi_tip_idx]
            dr = top_percent_average(flow_tip[15:-10, 5:-5, :], 0.7)
            if np.sqrt((dr**2).sum()) <= 1:
                break

        return img_face_n, flow, dr


def main0():
    # 可视化连续三帧两两的光流和一三的光流之前的插值
    face_d = DlibFaceDetector()
    lmk_d = DlibLmkDetector()
    alignment = Alignment(face_detector=face_d, lmk_detector=lmk_d)

    video_path = 'C:\\sheng\\casme2\\rawpic\\s15\\15_0101disgustingteeth'
    frames = os.listdir(video_path)
    frames.sort(key=lambda x: int(x[3:-4]))

    show = True

    all_d_flow = []
    all_d_avg = []

    for i_frame in range(len(frames) - 2):
        frame0_path = os.path.join(video_path, frames[i_frame])
        frame1_path = os.path.join(video_path, frames[i_frame+1])
        frame2_path = os.path.join(video_path, frames[i_frame+2])

        img0 = cv2.imread(frame0_path)
        img1 = cv2.imread(frame1_path)
        img2 = cv2.imread(frame2_path)

        alignment.reset()
        alignment.init_face_lmk(img0)
        img_face0, flow0, flow_avg0 = alignment.next(img0)
        img_face1, flow01, flow_avg01 = alignment.next(img1)
        img_face2, flow12, flow_avg12 = alignment.next(img2)
        alignment.reset_base_face()
        img_face0, flow0, flow_avg0 = alignment.next(img0)
        img_face2, flow02, flow_avg02 = alignment.next(img2)
        d_flow = flow02-flow01-flow12
        d_avg = flow_avg02 - flow_avg01 - flow_avg12

        all_d_flow.append(d_flow)
        all_d_avg.append(d_avg)

    if show:
        fig, ax = plt.subplots(2, 3, figsize=(8, 8))
        X = np.arange(256)
        Y = np.arange(256)

        vv = np.abs(np.array(all_d_flow))
        vv_mean = vv.mean(axis=0)
        vv_max = vv.max(axis=0)
        vv_std = vv.std(axis=0)

        ax[0, 0].imshow(img_face0[:, :, ::-1])
        cs = ax[0, 0].contour(X, Y, vv_mean[:, :, 0], levels=np.hstack((np.arange(-5, 0, 1), np.arange(1, 6, 1))))
        ax[0, 0].invert_yaxis()
        ax[0, 0].clabel(cs, cs.levels, inline=True, fontsize=10)

        ax[1, 0].imshow(img_face0[:, :, ::-1])
        cs = ax[1, 0].contour(X, Y, vv_mean[:, :, 1], levels=np.hstack((np.arange(-5, 0, 1), np.arange(1, 6, 1))))
        ax[1, 0].invert_yaxis()
        ax[1, 0].clabel(cs, cs.levels, inline=True, fontsize=10)

        ax[0, 1].imshow(img_face0[:, :, ::-1])
        cs = ax[0, 1].contour(X, Y, vv_max[:, :, 0], levels=np.hstack((np.arange(-5, 0, 1), np.arange(1, 6, 1))))
        ax[0, 1].invert_yaxis()
        ax[0, 1].clabel(cs, cs.levels, inline=True, fontsize=10)

        ax[1, 1].imshow(img_face0[:, :, ::-1])
        cs = ax[1, 1].contour(X, Y, vv_max[:, :, 1], levels=np.hstack((np.arange(-5, 0, 1), np.arange(1, 6, 1))))
        ax[1, 1].invert_yaxis()
        ax[1, 1].clabel(cs, cs.levels, inline=True, fontsize=10)

        ax[0, 2].imshow(img_face0[:, :, ::-1])
        cs = ax[0, 2].contour(X, Y, vv_std[:, :, 0], levels=np.hstack((np.arange(-5, 0, 1), np.arange(1, 6, 1))))
        ax[0, 2].invert_yaxis()
        ax[0, 2].clabel(cs, cs.levels, inline=True, fontsize=10)

        ax[1, 2].imshow(img_face0[:, :, ::-1])
        cs = ax[1, 2].contour(X, Y, vv_std[:, :, 1], levels=np.hstack((np.arange(-5, 0, 1), np.arange(1, 6, 1))))
        ax[1, 2].invert_yaxis()
        ax[1, 2].clabel(cs, cs.levels, inline=True, fontsize=10)


        plt.show()


def worker_flow_every_frame(video_path, result_path, pid):
    face_d = DlibFaceDetector()
    lmk_d = DlibLmkDetector()
    flow_fun = cv2.calcOpticalFlowFarneback

    alignment = Alignment(face_detector=face_d, lmk_detector=lmk_d, flow_fun=flow_fun)

    frames = os.listdir(video_path)
    # warning!!!
    frames.sort(key=lambda x: int(x[3:-4]))

    all_flow = []
    all_avg = []
    for i_frame0, i_frame1 in zip(frames[::-1], frames[1::]):
        frame0_path = os.path.join(video_path, i_frame0)
        frame1_path = os.path.join(video_path, i_frame1)
        img0 = cv2.imread(frame0_path)
        img1 = cv2.imread(frame1_path)

        alignment.reset()
        alignment.init_face_lmk(img0)
        img_face1, flow01, flow_avg01 = alignment.next(img1)

        all_flow.append(flow01)
        all_avg.append(flow_avg01)
    all_flow = np.abs(np.array(all_flow))
    all_avg = np.abs(np.array(all_avg))

    if False:
        fig, ax = plt.subplots()
        X = np.arange(256)
        Y = np.arange(256)
        U = all_flow[..., 0]
        V = all_flow[..., 1]
        R = np.sqrt(U**2+V**2)
        R = (R > 1).sum(axis=0)
        cs = ax.contour(X, Y, R, levels=np.hstack((np.arange(-100, 0, 20), np.arange(20, 120, 20))))
        ax.clabel(cs, cs.levels, inline=True, fontsize=10)
        plt.show()

    with open(result_path, 'wb') as f:
        pickle.dump((all_flow, all_avg, img_face1, alignment.lmk), f)

    print(f'worker {i} finished')


def main_flow_every_frame(data_root, result_root):
    # 可视化逐帧光流的累计
    os.makedirs(result_root, exist_ok=True)
    subjects = os.listdir(data_root)
    n_worker = 4

    pool = multiprocessing.Pool(n_worker)
    pool_run = pool.apply_async
    # pool_run = pool.apply

    i_process = 0
    for i_sub in subjects:
        sub_path = os.path.join(data_root, i_sub)
        videos = os.listdir(sub_path)
        for i_video in videos:
            pool_run(worker_flow_every_frame, kwds={'video_path': os.path.join(sub_path, i_video),
                                                    'result_path': os.path.join(result_root, f'{i_sub}_{i_video}.pkl'),
                                                    'pid': i_process})
            i_process += 1
    pool.close()
    pool.join()


def main_ana_flow_every_frame():
    data_root = 'C:\\sheng\\casme2\\rawpic'
    result_root = 'result'
    ana_root = 'ana'
    os.makedirs(ana_root, exist_ok=True)
    files = filter(lambda x: x[-4:] == '.pkl', os.listdir(result_root))

    for i_file in files:
        with open(os.path.join(result_root, i_file), 'rb') as f:
            flow, flow_avg = pickle.load(f)
        first_bar = i_file.find('_')
        video_path = os.path.join(data_root, i_file[:first_bar], i_file[first_bar+1:])
        frames = list(filter(lambda x: x[-4:] == '.png' or x[-4:] == '.jpg', os.listdir(video_path)))

        fig, ax = plt.subplots(4, 4, figsize=(8, 8))
        my_ax = [ax[i, j] for i in range(4) for j in range(4)]
        for idx, t in enumerate(np.arange(0.5, 2.1, 0.1)):
            r = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            count = (r > t).sum(axis=0)
            cs = my_ax[idx].contour(count, levels=np.arange(100, 1500, 100), vmin=0, vmax=2000)
            my_ax[idx].clabel(cs, cs.levels, inline=True, fontsize=10)
            # my_ax[idx].title(f'{t}')
            # my_ax[idx].colorbar()
        plt.show()


def main_continue_flow():
    face_d = DlibFaceDetector()
    lmk_d = DlibLmkDetector()
    flow_fun = cv2.calcOpticalFlowFarneback
    alignment = Alignment(face_detector=face_d, lmk_detector=lmk_d, flow_fun=flow_fun)

    video_path = 'C:\\sheng\\casme2\\rawpic\\s15\\15_0101disgustingteeth'
    frames = os.listdir(video_path)
    frames.sort(key=lambda x: int(x[3:-4]))

    show = True
    if show:
        fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        X = np.arange(256)
        Y = np.arange(256)

    for idx, i_frame in enumerate(frames):
        frame_path = os.path.join(video_path, i_frame)
        img = cv2.imread(frame_path)
        if idx % 100 == 0:
            alignment.reset()
            alignment.init_face_lmk(img)
            continue
        img, flow, flow_avg = alignment.next(img)
        if show:
            lmk = alignment.lmk.astype(np.int32)
            for i_point in lmk:
                cv2.circle(img, i_point, 2, (0, 0, 255))
            tip_ld, tip_ru = get_roi_bound(roi_points=lmk[29:31], expand_pixel=13)
            cv2.rectangle(img, tip_ld, tip_ru, (0, 255, 0), 1)

        if show:
            img0 = alignment.lmk_face
            flow1 = flow - flow_avg
            U = flow1[:, :, 0]
            V = flow1[:, :, 1]
            plt.ioff()
            ax[0, 0].cla()
            # self.ax[0, 0].imshow(self.img_face[::-1, :, ::-1])
            q = ax[0, 0].quiver(X, Y, U, V, scale=200, headwidth=1, headlength=1,  width=.0015)
            ax[0, 0].invert_yaxis()

            ax[0, 1].cla()
            ax[0, 1].imshow(img0[:, :, ::-1])

            ax[0, 2].cla()
            ax[0, 2].imshow(img[:, :, ::-1])

            ax[1, 0].cla()
            ax[1, 0].imshow(img[:, :, ::-1])
            ax[1, 0].invert_yaxis()
            cs = ax[1, 0].contour(X, Y, np.sqrt(U**2 + V**2), levels=np.arange(0, 5.5, 0.5))
            ax[1, 0].invert_yaxis()
            ax[1, 0].clabel(cs, cs.levels, inline=True, fontsize=10)

            ax[1, 1].cla()
            ax[1, 1].imshow(img[:, :, ::-1])
            ax[1, 1].invert_yaxis()
            cs = ax[1, 1].contour(X, Y, U, levels=np.hstack((np.arange(-3, 0, 0.5), np.arange(0.5, 3.5, 0.5))))
            ax[1, 1].invert_yaxis()
            ax[1, 1].clabel(cs, cs.levels, inline=True, fontsize=10)

            ax[1, 2].cla()
            ax[1, 2].imshow(img[:, :, ::-1])
            ax[1, 2].invert_yaxis()
            cs = ax[1, 2].contour(X, Y, V, levels=np.hstack((np.arange(-3, 0, 0.5), np.arange(0.5, 3.5, 0.5))))
            ax[1, 2].invert_yaxis()
            ax[1, 2].clabel(cs, cs.levels, inline=True, fontsize=10)
            plt.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.ion()

            plt.show()
        # cv2.imshow('a', img)
        # cv2.waitKey(10)


def main():
    data_root = 'C:\\sheng\\casme2\\rawpic'
    result_root = 'result1'
    main_flow_every_frame(data_root, result_root)
    main_ana_flow_every_frame()


def get_casme2_face_size():
    face_d = DlibFaceDetector()
    lmk_d = DlibLmkDetector()

    data_root = 'C:\\sheng\\casme2\\rawpic'
    subs = os.listdir(data_root)
    for i_sub in subs:
        i_sub_path = os.path.join(data_root, i_sub)
        videos = os.listdir(i_sub_path)
        for i_video in videos:
            i_video_path = os.path.join(i_sub_path, i_video)
            frames = os.listdir(i_video_path)
            frames = list(filter(lambda x: '.jpg' in x, frames))
            frame_path = os.path.join(i_video_path, frames[0])
            img = cv2.imread(frame_path)
            faces = face_d(img)
            face = faces[-1]
            # lmk
            lmk = lmk_d(img, face)
            left_eye = lmk[39]
            right_eye = lmk[46]
            d_eye = (right_eye - left_eye)[0]
            print(f'{i_sub}/{i_video}  {d_eye}')


def get_samml_face_size():
    face_d = DlibFaceDetector()
    lmk_d = DlibLmkDetector()

    data_root = 'C:\\sheng\\SAMM\\SAMM_longvideos'
    videos = os.listdir(data_root)
    for i_video in videos:
        i_video_path = os.path.join(data_root, i_video)
        frames = os.listdir(i_video_path)
        frames = list(filter(lambda x: '.jpg' in x, frames))
        frame_path = os.path.join(i_video_path, frames[0])
        img = cv2.imread(frame_path)
        faces = face_d(img)
        face = faces[-1]
        # lmk
        lmk = lmk_d(img, face)
        left_eye = lmk[39]
        right_eye = lmk[46]
        d_eye = (right_eye - left_eye)[0]
        print(f'{i_video}  {d_eye}')


if __name__ == '__main__':
    get_samml_face_size()
    # get_casme2_face_size()
    # main()
