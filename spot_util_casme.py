import os
import cv2
import dlib
import numpy as np

import sim_filter
import try_emd

from util import *


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
font = cv2.FONT_HERSHEY_SIMPLEX


def crop_picture(img_bgr, size, expend_ratio):
    """
    :param img_bgr:
    :param size:
    :param expend_ratio: down/forehead, up/chin, left, right
    :return:
    """
    down_r, up_r, left_r, right_r = expend_ratio

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
    faces = list(detector(img_gray, 0))
    # ？？只取最后一个
    lmk = np.array([[p.x, p.y] for p in predictor(img_bgr, faces[-1]).parts()])
    # 两个眼角的位置
    left_eye = lmk[39]
    right_eye = lmk[42]
    # error!!!注意这边都是取整，不是中心点会比中心点略小
    d_eye_half = int((right_eye[0] - left_eye[0]) / 2)

    center = ((left_eye + right_eye)/2).astype(np.int32)
    face_down = max((center[1] - int(down_r * d_eye_half)), 0)
    face_up = center[1] + int(up_r * d_eye_half)
    face_left = max(center[0] - int(left_r * d_eye_half), 0)
    face_right = center[0] + int(right_r * d_eye_half)

    img_crop = img_bgr[face_down:face_up, face_left:face_right]
    img_resized = cv2.resize(img_crop, (size, size))
    return lmk, img_resized, face_down, face_up, face_left, face_right


def get_landmarks(img_gray, img_rgb, lmk_global, face_down, face_up, face_left, face_right, size):
    faces = detector(img_gray, 0)
    if len(faces) == 0:
        lmk_local = (lmk_global - np.array([face_left, face_down]))\
                    / np.array([face_right-face_left, face_up-face_down]) * size
        lmk_local = lmk_local.astype(np.int32)
    else:
        lmk_local = np.array([[p.x, p.y] for p in predictor(img_rgb, faces[0]).parts()])
    return lmk_local


def uv_2_amplitude(flow):
    return np.sqrt(np.sum(np.array(flow) ** 2, axis=1))


def joint_2_window(point_picked):
    # todo: is 3 relate to fps
    signal_window = []
    if len(point_picked) > 0:
        head = point_picked[0]
        tail = point_picked[0]
        for i in range(len(point_picked)):
            if point_picked[i] >= tail and point_picked[i] - tail < 3:
                tail = point_picked[i]
            else:
                signal_window.append([head, tail])
                head = point_picked[i]
                tail = point_picked[i]
        signal_window.append([head, tail])
    signal_window = np.array(signal_window)
    return signal_window


def check_point_local(flow_r_low, flow_ext_low, idx, t_flow_gap, t_ext_gap, l_expand):
    j_head = max(0, idx - l_expand)
    j_tail = min(len(flow_r_low) - 1, idx + l_expand)
    flow_r_low_min = np.min(flow_r_low[j_head:j_tail])
    flow_ext_low_min = np.min(flow_ext_low[j_head:j_tail])
    return flow_r_low[idx] - flow_r_low_min > t_flow_gap and flow_ext_low[idx] - flow_ext_low_min > t_ext_gap


def find_peak(flow_r_low, flow_ext_low, t_peak_relative_inf, t_flow_gap, t_ext_gap, l_expand):
    """
    :param flow_r_low:
    :param flow_ext_low:
    :param t_peak_relative_inf: 基线阈值
    :param t_flow_gap: 信号1相对高度阈值
    :param t_ext_gap: 信号2相对高度阈值
    :param l_expand: 搜寻边界扩展
    :return:
    """
    # 使用寻找峰的方法
    # 扣除最小值作为基线
    flow_r_low = flow_r_low - np.min(flow_r_low)
    # 找出超出阈值的部分
    point_picked = np.where(flow_r_low >= t_peak_relative_inf)[0]
    signal_window = joint_2_window(point_picked)
    # 再次局部检查
    point_picked = [j for i_head, i_tail in signal_window for j in range(i_head, i_tail)
                    if check_point_local(flow_r_low, flow_ext_low, j, t_flow_gap, t_ext_gap, l_expand)]
    # 分散的峰值可能会存在交叉，因此新增去重排序
    point_picked = sorted(list(set(point_picked)))
    return joint_2_window(point_picked)


def expend(flow_r_low, signal_window, t_peak_valley_ratio, t_valley_gap,  l_expand, l_small_expend):
    # error 对于边界处没有处理好
    for i in range(len(signal_window)):
        head = signal_window[i, 0]
        tail = signal_window[i, 1]
        a1 = max(0, head - l_expand)
        b1 = min(len(flow_r_low) - 1, head + l_expand)
        a2 = max(0, tail - l_expand)
        b2 = min(len(flow_r_low) - 1, tail + l_expand)
        if tail > head:  # 因为有可能end=start
            peak_value = np.max(flow_r_low[head:tail])
        else:
            peak_value = flow_r_low[head]

        start_valley = np.min(flow_r_low[a1:b1])
        start_valley_idx = np.argmin(flow_r_low[a1:b1]) + a1
        end_valley = np.min(flow_r_low[a2:b2])  # end的左右中最小的索引
        end_valley_idx = np.argmin(flow_r_low[a2:b2]) + a2

        # 谷底在左侧
        if start_valley_idx < head:
            for j in range(head - 1, -1, -1):
                # 确认是0.33山腰
                if flow_r_low[j] - start_valley < t_peak_valley_ratio * (peak_value - start_valley):
                    head = j
                    break
                # 确认发现局部峰值取谷底后一点
                elif flow_r_low[j] > flow_r_low[j + 1]:
                    head = j + 2
                    break
        # 谷底在右侧，对左侧取最小值的下标
        else:
            left = max(head - l_small_expend, 0)
            start_left_inf_idx = np.argmin(flow_r_low[left:head + 1]) + left  # 代表了start左侧十个中值最小的索引
            if flow_r_low[head] - flow_r_low[start_left_inf_idx] > t_valley_gap:
                head = start_left_inf_idx + 1

        if end_valley_idx > tail:
            for j in range(tail + 1, end_valley_idx):
                if flow_r_low[j] - end_valley < t_peak_valley_ratio * (peak_value - end_valley):
                    tail = j
                    break
                if flow_r_low[j] > flow_r_low[j - 1]:
                    tail = j - 2
                    break
        else:
            right = min(tail + l_small_expend, len(flow_r_low) - 1)
            start_left_inf_idx = np.argmin(flow_r_low[tail:right + 1]) + tail  # 代表了end右侧十个中值最小的索引
            if flow_r_low[tail] - flow_r_low[start_left_inf_idx] > t_valley_gap:
                tail = start_left_inf_idx - 1  # 用最小值的索引进行替换

        signal_window[i, 0] = head
        signal_window[i, 1] = tail
    return signal_window


def divide(flow_r_low, signal_window, t_peak_ratio, t_peak_inf, l_split, bound_ignore):
    new_window = []
    for head, tail in signal_window:
        # 连续区间找谷底
        if (tail - head) >= l_split:
            valley_idx = np.argmin(flow_r_low[head + bound_ignore:tail - bound_ignore]) + head + bound_ignore
            left_peak = np.max(flow_r_low[head:valley_idx])
            right_peak = np.max(flow_r_low[valley_idx:tail])
            if ((left_peak - flow_r_low[valley_idx] > max(t_peak_inf, t_peak_ratio * left_peak))
                    and (right_peak - flow_r_low[valley_idx] > max(t_peak_inf, t_peak_ratio * right_peak))):
                new_window.append([head, valley_idx - 1])
                new_window.append([valley_idx + 1, tail])
                continue
        new_window.append([head, tail])
    return np.array(new_window)


def proce2(flow_uv, position, xuhao, tail_idx, bound_clip, *,
           fps, l_expand, l_small_expend, l_split, bound_ignore,
           t_peak_valley_ratio, t_peak_ratio, t_valley_gap,
           t_peak_relative_inf, t_peak_inf, t_flow_gap, t_ext_gap,
           frequency_inf, frequency_sup):

    head_idx = tail_idx - len(flow_uv)
    flow_r = uv_2_amplitude(flow_uv)
    position = position + str(xuhao) + "----"
    flow_r_low = sim_filter.filt(flow_r[bound_clip:-bound_clip], frequency_inf, frequency_sup, fps)

    flow_ext_high, flow_ext_low = try_emd.the_emd1(flow_r[bound_clip:-bound_clip], flow_r_low, position, str(head_idx))

    signal_window = find_peak(flow_r_low, flow_ext_low, t_peak_relative_inf, t_flow_gap, t_ext_gap, l_expand)
    signal_window = expend(flow_r_low, signal_window, t_peak_valley_ratio, t_valley_gap,  l_expand, l_small_expend)
    signal_window = divide(flow_r_low, signal_window, t_peak_ratio, t_peak_inf, l_split, bound_ignore)
    signal_window = signal_window + head_idx + bound_clip
    return signal_window


def nms2(signal_window):
    signal_window = np.array(signal_window)
    # 为什么要加这个？？？
    join_window = [[0, 0]]
    if signal_window.shape[0]:
        join_window.append(signal_window[0])
    for i_window in signal_window[1:]:
        new = 1
        for j_window in join_window[1:]:
            # 计算iou
            if i_window[0] > j_window[1] or i_window[1] < j_window[0]:
                # 两个间隔完全不相交
                iou = 0
            else:
                i_left = max(i_window[0], j_window[0])
                i_right = min(i_window[1], j_window[1])
                i_len = i_right - i_left
                if i_len == 0:
                    iou = 0
                else:
                    # 这个IOU和传统的不一样！！！
                    iou = i_len / min(i_window[1] - i_window[0], j_window[1] - j_window[0])
            # 通过iou决定是否合并
            if iou > 0.34:  # SAMM0.34  CASME 0.29   #如果重复率比较高就
                new = 0
                j_window[1] = max(j_window[1], i_window[1])
                j_window[0] = min(j_window[0], i_window[0])
        if new == 1:
            join_window.append(i_window)
    return np.array(join_window)


def draw_roiline19(files, le_p_kwargs, re_p_kwargs, mth_p_kwargs, ns_p_kwargs, ll_p_kwargs, rl_p_kwargs, t_flow_percent,
                   face_size, *, frame_stride=1, show=False):
    c_clr_0 = (0, 255, 0)
    c_clr_1 = (0, 255, 255)
    c_thk = 1

    # 这边设置会影响后面对齐的方式
    alignment_flow_fun = cv2.calcOpticalFlowFarneback
    alignment_flow_args = {'flow': None, 'pyr_scale': 0.5, 'levels': 3, 'winsize': 15,
                           'iterations': 5, 'poly_n': 7, 'poly_sigma': 1.5, 'flags': 0}
    feature_flow_fun = alignment_flow_fun
    feature_flow_args = alignment_flow_args
    # feature_flow_fun = cv2.optflow.DualTVL1OpticalFlow_create
    # feature_flow_args = {}
    #
    #                       -+-
    #                        |  t_flow_gap/t_ext_gap
    #                        |
    #                       ------+ min
    #           |-----le-----|-----le-----|
    #                  |-lse-|-lse-|
    #  y^    |--------ls--------|  to be split
    #   |        peak
    #   |        /|\
    #   |       / | \    /|\           ==pi
    #   |      /  |  \  / | \   _vg /  |
    #   |  \  / r1|=  \/  |  \  |  /   |   ==pri
    #   |   \/____|       |   \ | /    |   |
    #   |  valley       r2|=   \|/_____|___|______ min valley
    #   |                 |            |
    #   |_________________|____________|____________________>x

    # cas(me)^2
    expend_ratio = (3, 5, 4, 4) # down/forehead, up/chin, left, right
    # samm
    # face_size = 320
    # expend_ratio = (3.5, 5.5, 4.5, 4.5)
    # face_size = 284
    # expend_ratio = (3, 5, 4, 4)
    fps = 30
    flow_scale = face_size / 256

    t_tip_p = t_flow_percent['t_tip_p']
    t_le_p = t_flow_percent['t_le_p']
    t_re_p = t_flow_percent['t_re_p']
    t_mth_tp = t_flow_percent['t_mth_tp']
    t_mth_pp = t_flow_percent['t_mth_pp']
    t_ns_p = t_flow_percent['t_ns_p']
    t_ll_p = t_flow_percent['t_ll_p']
    t_rl_p = t_flow_percent['t_rl_p']

    # cas(me)2 256 flow_scale = 1
    # samml 320 flow_scale = 1
    # pixel
    roi_tip_exp = int(13 * flow_scale)
    roi_tip = slice(int(15 * flow_scale), int(-10 * flow_scale)), slice(int(5 * flow_scale), int(-5 * flow_scale))

    roi_le_min_exp = (np.array([5, 15]) * flow_scale).astype(np.int32)
    roi_le_max_exp = (np.array([0, 5]) * flow_scale).astype(np.int32)
    roi_le_range = (np.array([-10, 10]) * flow_scale).astype(np.int32)
    round_xxx = int(10 * flow_scale)
    roi_le_all = slice(round_xxx, -round_xxx), slice(round_xxx, -round_xxx)

    roi_re_min_exp = (np.array([0, 15]) * flow_scale).astype(np.int32)
    roi_re_max_exp = (np.array([0, 5]) * flow_scale).astype(np.int32)
    roi_re_range = (np.array([-10, 10]) * flow_scale).astype(np.int32)
    roi_re_all = slice(round_xxx, -round_xxx), slice(round_xxx, -round_xxx)

    roi_mth_min_exp = (np.array([20, 15]) * flow_scale).astype(np.int32)
    roi_mth_max_exp = (np.array([20, 10]) * flow_scale).astype(np.int32)
    roi_mth_range = (np.array([[[-10, 10], [-10, 20]],
                              [[-10, 10], [-20, 10]],
                              [[-10, 10], [-10, 10]],
                              [[-10, 10], [-10, 10]],
                              [[-10, 10], [-10, 10]]]) * flow_scale).astype(np.int32)
    roi_mth_all = slice(round_xxx, -round_xxx), slice(round_xxx, -round_xxx)

    roi_ns_min_exp = (np.array([30, 20]) * flow_scale).astype(np.int32)
    roi_ns_max_exp = (np.array([30, 5]) * flow_scale).astype(np.int32)
    roi_ns_range = (np.array([[[-20, 5], [-20, 10]], [[-20, 5], [-10, 20]]]) * flow_scale).astype(np.int32)

    round_xxx = int(5 * flow_scale)
    roi_ll_all = slice(round_xxx, -round_xxx), slice(round_xxx, -round_xxx)
    roi_rl_all = slice(round_xxx, -round_xxx), slice(round_xxx, -round_xxx)

    i_frame = 0  # 这里的k代表开始的位置
    start = i_frame - 99  # 每一小段的开始和结束
    end = i_frame + 100
    window_step = 100  # 默认移动是100
    ana_last = True  # 最后一段是否处理过
    all_window = np.array([0, 0])

    files = files[::frame_stride]

    # 左眼/右眼/嘴巴/鼻子
    while i_frame < len(files):
        start += window_step
        end += window_step
        if end > len(files) and ana_last:
            end = len(files) - 2
            ana_last = False
        i_frame = 0
        for i_path in files:
            i_frame = i_frame + 1
            if i_frame >= start:
                if i_frame == start:
                    # flow list of frames
                    ff_le_t = [[0, 0]]
                    ff_le_1 = [[0, 0]]
                    ff_le_2 = [[0, 0]]
                    ff_le_3 = [[0, 0]]
                    ff_re_t = [[0, 0]]
                    ff_re_1 = [[0, 0]]
                    ff_re_2 = [[0, 0]]
                    ff_re_3 = [[0, 0]]
                    ff_mth_t = [[0, 0]]
                    ff_mth_1 = [[0, 0]]
                    ff_mth_2 = [[0, 0]]
                    ff_mth_3 = [[0, 0]]
                    ff_mth_4 = [[0, 0]]
                    ff_mth_5 = [[0, 0]]
                    ff_ns_1 = [[0, 0]]
                    ff_ns_2 = [[0, 0]]

                    ff_tip = [[0, 0]]
                    ff_left_lip = [[0, 0]]
                    ff_right_lip = [[0, 0]]

                    img_bgr = cv2.imread(i_path)
                    lmk, img_bgr, face_down, face_up, face_left, face_right = crop_picture(img_bgr, face_size, expend_ratio)
                    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
                    lmk = get_landmarks(img_gray, img_bgr, lmk, face_down, face_up, face_left, face_right, face_size)
                    if show:
                        for i_point in lmk:
                            cv2.circle(img_bgr, i_point, 2, c_clr_1)

                    # 左眼
                    # 计算边界框
                    roi_le_min, roi_le_max = get_roi_bound(roi_points=lmk[17:22])  # 左眉毛的位置
                    # 计算整体切片
                    roi_le_idx = slice(roi_le_min[1] - roi_le_min_exp[1], roi_le_max[1] + roi_le_max_exp[1]),\
                                 slice(max(0, roi_le_min[0] - roi_le_min_exp[0]), roi_le_max[0] + roi_le_max_exp[0])
                    # 计算三个点局部切片
                    # 1. 计算区域中心
                    # 2. x,y反转
                    # 3. 区域大小
                    # 4. 转化为切片
                    roi_le = (lmk[[20, 19, 18]] - roi_le_min + roi_le_min_exp)[:, ::-1, np.newaxis] + roi_le_range
                    roi_le = [(slice(*i_point[0]), slice(*i_point[1])) for i_point in roi_le]
                    img_le_gray_0 = img_gray[roi_le_idx]
                    if show:
                        cv2.rectangle(img_bgr, roi_le_min-roi_le_min_exp, roi_le_max+roi_le_max_exp, c_clr_0, c_thk)
                        for i_point in lmk[[20, 19, 18]]:
                            cv2.rectangle(img_bgr, i_point + roi_le_range, i_point - roi_le_range, c_clr_1, c_thk)

                    # 右眼
                    roi_re_min, roi_re_max = get_roi_bound(roi_points=lmk[22:27])
                    roi_re_idx = slice(roi_re_min[1] - roi_re_min_exp[1], roi_re_max[1] + roi_re_max_exp[1]),\
                                 slice(roi_re_min[0] - roi_re_min_exp[0], roi_re_max[0] + roi_re_max_exp[0])
                    roi_re = (lmk[[23, 24, 25]] - roi_re_min + roi_re_min_exp)[:, ::-1, np.newaxis] + roi_re_range
                    roi_re = [(slice(*i_point[0]), slice(*i_point[1])) for i_point in roi_re]
                    img_re_gray_0 = img_gray[roi_re_idx]
                    if show:
                        cv2.rectangle(img_bgr, roi_re_min-roi_re_min_exp, roi_re_max+roi_re_max_exp, c_clr_0, c_thk)
                        for i_point in lmk[[23, 24, 25]]:
                            cv2.rectangle(img_bgr, i_point + roi_re_range, i_point - roi_re_range, c_clr_1, c_thk)

                    # 嘴巴
                    roi_mth_min, roi_mth_max = get_roi_bound(roi_points=lmk[48:67])
                    roi_mth_idx = slice(roi_mth_min[1] - roi_mth_min_exp[1], roi_mth_max[1] + roi_mth_max_exp[1]),\
                                  slice(roi_mth_min[0] - roi_mth_min_exp[0], roi_mth_max[0] + roi_mth_max_exp[0])
                    roi_mth = (lmk[[48, 54, 51, 57, 62]] - roi_mth_min + roi_mth_min_exp)[:, ::-1, np.newaxis]\
                              + roi_mth_range
                    roi_mth = [(slice(*i_point[0]), slice(*i_point[1])) for i_point in roi_mth]
                    img_mth_gray_0 = img_gray[roi_mth_idx]
                    if show:
                        cv2.rectangle(img_bgr, roi_mth_min-roi_mth_min_exp, roi_mth_max+roi_mth_max_exp, c_clr_0, c_thk)
                        for i_point, i_range in zip(lmk[[48, 54, 51, 57, 62]], roi_mth_range):
                            cv2.rectangle(img_bgr, i_point+i_range[::-1, 0], i_point+i_range[::-1, 1], c_clr_1, c_thk)

                    # 鼻子两侧
                    roi_ns_min, roi_ns_max = get_roi_bound(roi_points=lmk[30:36])
                    roi_ns_idx = slice(roi_ns_min[1] - roi_ns_min_exp[1], roi_ns_max[1] + roi_ns_max_exp[1]),\
                                 slice(roi_ns_min[0] - roi_ns_min_exp[0], roi_ns_max[0] + roi_ns_max_exp[0])
                    roi_ns = (lmk[[31, 35]] - roi_ns_min + roi_ns_min_exp)[:, ::-1, np.newaxis] + roi_ns_range
                    roi_ns = [(slice(*i_point[0]), slice(*i_point[1])) for i_point in roi_ns]
                    img_ns_gray_0 = img_gray[roi_ns_idx]
                    if show:
                        cv2.rectangle(img_bgr, roi_ns_min - roi_ns_min_exp, roi_ns_max + roi_ns_max_exp, c_clr_0, c_thk)
                        for i_point, i_range in zip(lmk[[31, 35]], roi_ns_range):
                            cv2.rectangle(img_bgr, i_point+i_range[::-1, 0], i_point+i_range[::-1, 1], c_clr_1, c_thk)

                    # 左眼睑部位
                    roi_ll_min, roi_ll_max = get_roi_bound(roi_points=lmk[36:42])
                    width = roi_ll_max[0] - roi_ll_min[0]
                    roi_ll_ymid = (roi_ll_max[1] + roi_ll_min[1]) / 2
                    roi_ll_idx = slice(int(roi_ll_ymid + width / 4), int(roi_ll_ymid + 3 * width / 4)),\
                                 slice(roi_ll_min[0], roi_ll_max[0])
                    img_left_lid_gray_0 = img_gray[roi_ll_idx]
                    if show:
                        cv2.rectangle(img_bgr, (roi_ll_idx[1].start, roi_ll_idx[0].start),
                                      (roi_ll_idx[1].stop, roi_ll_idx[0].stop), c_clr_1, c_thk)

                    # 右眼睑部位
                    roi_rl_min, roi_rl_max = get_roi_bound(roi_points=lmk[42:48])
                    width = roi_rl_max[0] - roi_rl_min[0]
                    roi_rl_ymid = (roi_rl_max[1] + roi_rl_min[1]) / 2
                    roi_rl_idx = slice(int(roi_rl_ymid + width / 4), int(roi_rl_ymid + 3 * width / 4)),\
                                 slice(roi_rl_min[0], roi_rl_max[0])
                    img_right_lid_gray_0 = img_gray[roi_rl_idx]
                    if show:
                        cv2.rectangle(img_bgr, (roi_rl_idx[1].start, roi_rl_idx[0].start),
                                      (roi_rl_idx[1].stop, roi_rl_idx[0].stop), c_clr_1, c_thk)

                    roi_tip_min, roi_tip_max = get_roi_bound(roi_points=lmk[29:31], expand_pixel=roi_tip_exp)
                    roi_tip_idx = slice(roi_tip_min[1], roi_tip_max[1]), slice(roi_tip_min[0], roi_tip_max[0])
                    img_tip_gray_0 = img_gray[roi_tip_idx]
                    if show:
                        cv2.rectangle(img_bgr, (roi_tip_idx[1].start, roi_tip_idx[0].start),
                                      (roi_tip_idx[1].stop, roi_tip_idx[0].stop), c_clr_0, c_thk)
                        cv2.imshow("image1", img_bgr)
                        cv2.waitKey(0)
                else:
                    img_bgr_read_n = cv2.imread(i_path)
                    x_avg = 0
                    y_avg = 0
                    # 整体多次对齐，有多种方式：
                    # 1. 速度快精度低的对齐
                    # 2. 速度慢精度高的对齐
                    # 3. 前面用速度快的，最后一次用精度高的对齐
                    for _ in range(4):
                        face_left = max(0, face_left + int(round(x_avg)))
                        face_down = max(0, face_down + int(round(y_avg)))
                        face_right += int(round(x_avg))
                        face_up += int(round(y_avg))
                        # 按照第一个图的框对齐后切割出一个脸
                        img_crop = img_bgr_read_n[face_down:face_up, face_left:face_right]
                        img_bgr = cv2.resize(img_crop, (face_size, face_size))
                        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2GRAY)
                        # 求全局的光流
                        img_tip_gray_n = img_gray[roi_tip_idx]
                        f_uv_ns = alignment_flow_fun(img_tip_gray_0, img_tip_gray_n, **alignment_flow_args)
                        x_avg, y_avg = top_percent_average(f_uv_ns[roi_tip], t_tip_p)
                        ff_tip.append([x_avg, y_avg]) # 当前没用
                        if (x_avg ** 2 + y_avg ** 2) <= 1:
                            if feature_flow_fun != alignment_flow_fun:
                                f_uv_ns = feature_flow_fun(img_tip_gray_0, img_tip_gray_n, **feature_flow_args)
                                x_avg, y_avg = top_percent_average(f_uv_ns[roi_tip], t_tip_p)
                            break

                    # 对齐完毕
                    flow_avg = np.array([x_avg, y_avg])
                    img_left_eye_gray_n = img_gray[roi_le_idx]
                    f_uv_le = feature_flow_fun(img_le_gray_0, img_left_eye_gray_n, **feature_flow_args)
                    # 去掉光流特征矩阵周边round大小的部分，求均值。一个感兴趣区域处的平均光流
                    ff_le_t.append(top_percent_average(f_uv_le[roi_le_all], t_le_p) - flow_avg)
                    ff_le_1.append(top_percent_average(f_uv_le[roi_le[0]], t_le_p) - flow_avg)
                    ff_le_2.append(top_percent_average(f_uv_le[roi_le[1]], t_le_p) - flow_avg)
                    ff_le_3.append(top_percent_average(f_uv_le[roi_le[2]], t_le_p) - flow_avg)
                    # right eye
                    img_right_eye_gray_n = img_gray[roi_re_idx]
                    f_uv_re = feature_flow_fun(img_re_gray_0, img_right_eye_gray_n, **feature_flow_args)
                    ff_re_t.append(top_percent_average(f_uv_re[roi_re_all], t_re_p) - flow_avg)
                    ff_re_1.append(top_percent_average(f_uv_re[roi_re[0]], t_re_p) - flow_avg)
                    ff_re_2.append(top_percent_average(f_uv_re[roi_re[1]], t_re_p) - flow_avg)
                    ff_re_3.append(top_percent_average(f_uv_re[roi_re[2]], t_re_p) - flow_avg)
                    # mouth
                    img_mouth_gray_n = img_gray[roi_mth_idx]
                    f_uv_mth = feature_flow_fun(img_mth_gray_0, img_mouth_gray_n, **feature_flow_args)
                    ff_mth_t.append(top_percent_average(f_uv_mth[roi_mth_all], t_mth_tp) - flow_avg)
                    ff_mth_1.append(top_percent_average(f_uv_mth[roi_mth[0]], t_mth_pp) - flow_avg)
                    ff_mth_2.append(top_percent_average(f_uv_mth[roi_mth[1]], t_mth_pp) - flow_avg)
                    ff_mth_3.append(top_percent_average(f_uv_mth[roi_mth[2]], t_mth_pp) - flow_avg)
                    ff_mth_4.append(top_percent_average(f_uv_mth[roi_mth[3]], t_mth_pp) - flow_avg)
                    ff_mth_5.append(top_percent_average(f_uv_mth[roi_mth[4]], t_mth_pp) - flow_avg)
                    # 鼻子两侧
                    img_nose_gray_n = img_gray[roi_ns_idx]
                    f_uv_nose = cv2.calcOpticalFlowFarneback(img_ns_gray_0, img_nose_gray_n, **feature_flow_args)
                    ff_ns_1.append(top_percent_average(f_uv_nose[roi_ns[0]], t_ns_p) - flow_avg)
                    ff_ns_2.append(top_percent_average(f_uv_nose[roi_ns[1]], t_ns_p) - flow_avg)
                    # 左眼睑
                    img_left_lid_gray_n = img_gray[roi_ll_idx]
                    f_uv_ll = feature_flow_fun(img_left_lid_gray_0, img_left_lid_gray_n, **feature_flow_args)
                    a1, b1 = top_percent_average(f_uv_ll[roi_ll_all], t_ll_p)
                    ff_left_lip.append([a1 - x_avg, b1 - y_avg])
                    # 右眼睑
                    img_right_lid_gray_n = img_gray[roi_rl_idx]
                    f_uv_rl = feature_flow_fun(img_right_lid_gray_0, img_right_lid_gray_n, **feature_flow_args)
                    a2, b2 = top_percent_average(f_uv_rl[roi_rl_all], t_rl_p)
                    ff_right_lip.append([a2 - x_avg, b2 - y_avg])
            if i_frame == end:
                bound_clip = 1
                head_idx_1 = end - start + bound_clip

                signal_window = []
                signal_window.extend(proce2(ff_le_t, "left_eye", 0, i_frame, bound_clip, **le_p_kwargs))
                signal_window.extend(proce2(ff_le_1, "left_eye", 1, i_frame, bound_clip, **le_p_kwargs))
                signal_window.extend(proce2(ff_le_2, "left_eye", 2, i_frame, bound_clip, **le_p_kwargs))
                signal_window.extend(proce2(ff_le_3, "left_eye", 3, i_frame, bound_clip, **le_p_kwargs))

                signal_window.extend(proce2(ff_re_t, "right_eye", 0, i_frame, bound_clip, **re_p_kwargs))
                signal_window.extend(proce2(ff_re_1, "right_eye", 1, i_frame, bound_clip, **re_p_kwargs))
                signal_window.extend(proce2(ff_re_2, "right_eye", 2, i_frame, bound_clip, **re_p_kwargs))
                signal_window.extend(proce2(ff_re_3, "right_eye", 3, i_frame, bound_clip, **re_p_kwargs))

                signal_window.extend(proce2(ff_mth_t, "mouth", 0, i_frame, bound_clip, **mth_p_kwargs))
                signal_window.extend(proce2(ff_mth_1, "mouth", 1, i_frame, bound_clip, **mth_p_kwargs))
                signal_window.extend(proce2(ff_mth_2, "mouth", 2, i_frame, bound_clip, **mth_p_kwargs))
                signal_window.extend(proce2(ff_mth_3, "mouth", 3, i_frame, bound_clip, **mth_p_kwargs))
                signal_window.extend(proce2(ff_mth_4, "mouth", 4, i_frame, bound_clip, **mth_p_kwargs))
                signal_window.extend(proce2(ff_mth_5, "mouth", 5, i_frame, bound_clip, **mth_p_kwargs))

                signal_window.extend(proce2(ff_ns_1, "nose", 1, i_frame, bound_clip, **ns_p_kwargs))
                signal_window.extend(proce2(ff_ns_2, "nose", 2, i_frame, bound_clip, **ns_p_kwargs))

                signal_window = np.array(nms2(signal_window))
                signal_window = np.array(nms2(signal_window))

                signal_window_local = signal_window - (i_frame - head_idx_1)
                window_step = 100
                for i_window in signal_window_local:
                    if i_window[0] < 100 and i_window[1] > 100:
                        if i_window[1] < 150:
                            window_step = i_window[1] + 20
                        elif i_window[0] > 50:
                            window_step = i_window[0] - 20
                        else:
                            bound_clip = min(189, i_window[1])
                            window_step = bound_clip + 10

                all_window = np.vstack((all_window, signal_window))
                break

    all_window = [i_w for i_w in all_window if 12 <= i_w[1] - i_w[0] <= 200]
    all_window = np.array(nms2(all_window))
    all_window = np.array(nms2(all_window))
    all_window = [i_w for i_w in all_window if i_w[1] != 0]
    all_window = np.array(all_window)
    all_window = all_window*frame_stride
    return all_window
