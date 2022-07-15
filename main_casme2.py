import csv
import multiprocessing
import os
import pickle
import shutil

import numpy as np
import xlrd

import spot_util_casme as fl


def read_casme2_video(video_path):
    start_sub, end_sub = 3, -4
    files = [i for i in os.listdir(video_path) if 'jpg' in i]
    files.sort(key=lambda x: int(x[start_sub:end_sub]))
    files = [os.path.join(video_path, i) for i in files]
    return files


def read_casme2_label(label_path, fps):
    # 宏观表情平均持续39帧（30fps）
    mic_default_d = int(fps * 2 / 3)

    data_xls = xlrd.open_workbook(label_path)
    table_xls = data_xls.sheets()[0]

    labels = {}
    for i_row in table_xls:
        key = i_row[10].value
        if key not in labels:
            labels[key] = []
        labels[key].append([int(i_cell.value) for i_cell in i_row[2:5]])

    labels = {k:  np.array([[i_row[0], i_row[2] if i_row[2] else i_row[1] + mic_default_d] for i_row in v])
              for k, v in labels.items()}
    return labels


def read_samml_video(video_path):
    start_sub, end_sub = 6, -4
    files = [i for i in os.listdir(video_path) if 'jpg' in i]
    files.sort(key=lambda x: int(x[start_sub:end_sub]))
    files = [os.path.join(video_path, i) for i in files]
    return files


def read_samml_label(label_path):
    data_xls = xlrd.open_workbook(label_path)
    table_xls = data_xls.sheets()[0]

    labels = {}
    for i_row in table_xls:
        key = i_row[1].value[:5]
        if '_' not in key:
            continue
        if key not in labels:
            labels[key] = []
        labels[key].append([int(i_cell.value) for i_cell in i_row[3:6]])

    labels = {k:  np.array([[i_row[0], i_row[2]] for i_row in v])
              for k, v in labels.items()}
    return labels


def metrics(label_slice, predict_slice, t_iou, mic_frame, dataset_root_path, debug_message=False):
    n_pre_micro = int(((predict_slice[:, 1] - predict_slice[:, 0]) <= mic_frame).sum()) if predict_slice.shape[0] else 0

    tp = np.zeros(len(t_iou), dtype=np.int32)
    tp_mic = np.zeros(len(t_iou), dtype=np.int32)

    # 修正标注1开始，预测0开始
    predict_slice += 1

    FPPP = []
    for label_start, label_end in label_slice.tolist():
        percent = 0
        for j, (predict_start, predict_end) in enumerate(predict_slice.tolist()):
            if not (predict_end < label_start or predict_start > label_end):
                all_points = sorted([label_start, label_end, predict_start, predict_end])
                percent = (float(all_points[2] - all_points[1])) / (all_points[3] - all_points[0])

                idx = percent >= t_iou
                tp[idx] += 1
                if label_end - label_start <= mic_frame:
                    tp_mic[idx] += 1

                if percent >= 0.5:
                    FPPP.append(j)
                    if debug_message:
                        print(f'lable:{label_start},{label_end}  test:{predict_start},{predict_end} percent={percent}')

                    with open('my_casme1.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([dataset_root_path, label_start, label_end, predict_start, predict_end, 'TP'])

                if percent >= 0.2:
                    break

        if percent < 0.5:
            if debug_message:
                print(f'lable:{label_start},{label_end} 没有正确结果')
            with open('my_casme1.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([dataset_root_path, label_start, label_end, '', '', 'FN'])

    FPPP = list(set(FPPP))
    for j, (predict_start, predict_end) in enumerate(predict_slice):
        if j not in FPPP:
            with open('my_casme1.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([dataset_root_path, '', '', predict_start, predict_end, 'FP'])

    return tp, tp_mic, len(label_slice), len(predict_slice), n_pre_micro


def casme2_worker(dataset_root_path, sub, label_path, pkl_path,
                  le_p_kwargs, re_p_kwargs, mth_p_kwargs, ns_p_kwargs, ll_p_kwargs, rl_p_kwargs, t_flow_percent,
                  face_size, t_iou, i_worker, debug_message=False):
    fps = 30
    mic_frame = fps // 2
    result = []

    sub_path = os.path.join(dataset_root_path, sub)
    videos = os.listdir(sub_path)
    labels = read_casme2_label(label_path, fps)
    for i_video in videos:
        # CAS(ME)^2
        files = read_casme2_video(os.path.join(dataset_root_path, sub, i_video))
        predict = fl.draw_roiline19(files, le_p_kwargs, re_p_kwargs, mth_p_kwargs, ns_p_kwargs,
                                    ll_p_kwargs, rl_p_kwargs, t_flow_percent, face_size)
        # 这是由于标签文件写死了所以必须使用 '/'
        i_video_relative = sub + '/' + i_video
        gt = labels[i_video_relative] if i_video_relative in labels else np.array([])
        result.append(metrics(gt, predict, t_iou, mic_frame, i_video_relative))

    result = [np.array([j[i] for j in result]).sum(axis=0) for i in range(5)]

    with open(f'{pkl_path}{os.path.sep}casme_{i_worker}.pkl', 'wb') as f:
        pickle.dump(result, f)

    if debug_message:
        print(f'{i_worker} finished')


def samml_worker(dataset_root_path, videos, label_path, pkl_path,
                 le_p_kwargs, re_p_kwargs, mth_p_kwargs, ns_p_kwargs, ll_p_kwargs, rl_p_kwargs, t_flow_percent,
                 face_size, t_iou, i_worker, debug_message=False):
    fps = 200
    mic_frame = fps // 2
    result = []

    labels = read_samml_label(label_path)
    for i_video in videos:
        # CAS(ME)^2
        files = read_samml_video(os.path.join(dataset_root_path, i_video))
        predict = fl.draw_roiline19(files,
                                    le_p_kwargs, re_p_kwargs, mth_p_kwargs, ns_p_kwargs,
                                    ll_p_kwargs, rl_p_kwargs, t_flow_percent, face_size,
                                    frame_stride=7)
        # 这是由于标签文件写死了所以必须使用 '/'
        result.append(metrics(labels[i_video], predict, t_iou, mic_frame, i_video))

    result = [np.array([j[i] for j in result]).sum(axis=0) for i in range(5)]

    with open(f'{pkl_path}{os.path.sep}samml_{i_worker}.pkl', 'wb') as f:
        pickle.dump(result, f)

    if debug_message:
        print(f'{i_worker} finished')


def report(t_iou, n_gt, n_pred, tp, prefix='', show_message=True):
    precision = tp / n_pred
    recall = tp / n_gt
    f1 = (2 * precision * recall) / (precision + recall)

    if show_message:
        print('------------------')
        print(f"iou={','.join([f'{i:20}' for i in t_iou])}")
        print(f"{prefix}精准率：{','.join([f'{i:20}' for i in precision])}")
        print(f"{prefix}召回率：{','.join([f'{i:20}' for i in recall])}")
        print(f"{prefix}F1系数：{','.join([f'{i:20}' for i in f1])}")
    return precision, recall, f1


def main_casme2(le_p_kwargs, re_p_kwargs, mth_p_kwargs, ns_p_kwargs, ll_p_kwargs, rl_p_kwargs,
                t_flow_percent, face_size, show_message=False):
    dataset_root_path = 'C:\\sheng\\casme2\\rawpic'
    label_path = 'CAS(ME)^2code_final.xls'
    pkl_path = 'casme2_pkl'
    shutil.rmtree(pkl_path)
    os.makedirs(pkl_path, exist_ok=True)

    sub_list = os.listdir(dataset_root_path)
    n_result = len(sub_list)
    n_worker = min(25, n_result)
    t_iou = np.array([0.2, 0.3, 0.4, 0.5])

    kwds = {
        'dataset_root_path': dataset_root_path,
        'label_path': label_path,
        'pkl_path': pkl_path,
        'le_p_kwargs': le_p_kwargs,
        're_p_kwargs': re_p_kwargs,
        'mth_p_kwargs': mth_p_kwargs,
        'ns_p_kwargs': ns_p_kwargs,
        'll_p_kwargs': ll_p_kwargs,
        'rl_p_kwargs': rl_p_kwargs,
        't_flow_percent': t_flow_percent,
        'face_size': face_size,
        't_iou': t_iou}

    # if False:
    if True:
        pool = multiprocessing.Pool(n_worker)
        for i_worker, i_sub in enumerate(sub_list):
            pool.apply_async(casme2_worker, kwds={**kwds, 'sub': i_sub, 'i_worker': i_worker})
        pool.close()
        pool.join()
    else:
        for i_worker, i_sub in enumerate(sub_list):
            casme2_worker(**kwds, sub=i_sub, i_worker=i_worker)

    gather_result = []
    for i_worker in range(n_result):
        with open(f'{pkl_path}{os.path.sep}casme_{i_worker}.pkl', 'rb') as f:
            gather_result.append(pickle.load(f))

    tp = np.array([i[0] for i in gather_result]).sum(axis=0)
    tp_mic = np.array([i[1] for i in gather_result]).sum(axis=0)
    tp_mac = tp-tp_mic
    n_gt = np.array([i[2] for i in gather_result]).sum(axis=0)
    n_pred = np.array([i[3] for i in gather_result]).sum(axis=0)
    n_pred_mic = np.array([i[4] for i in gather_result]).sum(axis=0)
    n_pred_mac = n_pred - n_pred_mic

    n_gt_mic = 57
    n_gt_mac = 300
    if show_message:
        print('-------------------------')
        for i_t, i_tp, i_tp_mic in zip(t_iou, tp, tp_mic):
            print(f'iou={i_t}，共有{i_tp}个正确的分析，共有{i_tp_mic}个正确的微表情分析')
        print('-------------------------')
        print(f'共有表情{n_gt}个')
        print(f'共测试出{n_pred}个')
        print(f'宏表情有{n_pred_mac}个')
        print(f'微表情有{n_pred_mic}个')

    report(t_iou, n_gt, n_pred, tp, '', show_message=show_message)
    report(t_iou, n_gt_mac, n_pred_mac, tp_mac, '宏表情',  show_message=show_message)
    report(t_iou, n_gt_mic, n_pred_mic, tp_mic, '微表情',  show_message=show_message)


def main_samml(le_p_kwargs, re_p_kwargs, mth_p_kwargs, ns_p_kwargs, ll_p_kwargs, rl_p_kwargs,
               t_flow_percent, face_size, show_message=False):
    dataset_root_path = 'C:\\sheng\\SAMM\\SAMM_longvideos'
    label_path = 'C:\\sheng\\SAMM\\SAMM_LongVideos_V3_Release.xls'
    pkl_path = 'samml_pkl'
    shutil.rmtree(pkl_path)
    os.makedirs(pkl_path, exist_ok=True)

    video_list = os.listdir(dataset_root_path)
    n_result = len(video_list)
    n_worker = min(25, n_result)
    t_iou = np.array([0.2, 0.3, 0.4, 0.5])

    kwds = {
        'dataset_root_path': dataset_root_path,
        'label_path': label_path,
        'pkl_path': pkl_path,
        'le_p_kwargs': le_p_kwargs,
        're_p_kwargs': re_p_kwargs,
        'mth_p_kwargs': mth_p_kwargs,
        'ns_p_kwargs': ns_p_kwargs,
        'll_p_kwargs': ll_p_kwargs,
        'rl_p_kwargs': rl_p_kwargs,
        't_flow_percent': t_flow_percent,
        'face_size': face_size,
        't_iou': t_iou}

    if True:
    # if True:
        pool = multiprocessing.Pool(n_worker)
        for i_worker in range(n_worker):
            pool.apply_async(samml_worker,
                             kwds={**kwds, 'videos': video_list[i_worker::n_worker], 'i_worker': i_worker})
        pool.close()
        pool.join()
    else:
        for i_worker in range(n_worker):
            samml_worker(**kwds, videos=video_list[i_worker::n_worker], i_worker=i_worker)

    gather_result = []
    for i_worker in range(n_worker):
        with open(f'{pkl_path}{os.path.sep}samml_{i_worker}.pkl', 'rb') as f:
            gather_result.append(pickle.load(f))

    tp = np.array([i[0] for i in gather_result]).sum(axis=0)
    tp_mic = np.array([i[1] for i in gather_result]).sum(axis=0)
    tp_mac = tp-tp_mic
    n_gt = np.array([i[2] for i in gather_result]).sum(axis=0)
    n_pred = np.array([i[3] for i in gather_result]).sum(axis=0)
    n_pred_mic = np.array([i[4] for i in gather_result]).sum(axis=0)
    n_pred_mac = n_pred - n_pred_mic

    n_gt_mic = 159
    n_gt_mac = 343
    if show_message:
        print('-------------------------')
        for i_t, i_tp, i_tp_mic in zip(t_iou, tp, tp_mic):
            print(f'iou={i_t}，共有{i_tp}个正确的分析，共有{i_tp_mic}个正确的微表情分析')
        print('-------------------------')
        print(f'共有表情{n_gt}个')
        print(f'共测试出{n_pred}个')
        print(f'宏表情有{n_pred_mac}个')
        print(f'微表情有{n_pred_mic}个')

    acc_all = report(t_iou, n_gt, n_pred, tp, '', show_message=show_message)
    acc_mac = report(t_iou, n_gt_mac, n_pred_mac, tp_mac, '宏表情', show_message=show_message)
    acc_min = report(t_iou, n_gt_mic, n_pred_mic, tp_mic, '微表情', show_message=show_message)
    return acc_all, acc_mac, acc_min


def main():
    t_flow_percent = {
        't_tip_p': 0.68726687,
        't_le_p': 0.180330275,
        't_re_p': 0.180330275,
        't_mth_tp': 0.329587946,
        't_mth_pp': 0.196391502,
        't_ns_p': 0.253253899,
        't_ll_p': 0.3,
        't_rl_p': 0.3,
    }
    if True:
        fps = 30
        face_size = 256
        flow_scale = face_size / 256

        process_kwargs = {
            'fps': fps,                               # frame
            'l_expand': fps,                          # frame le
            'l_small_expend': int(fps / 3),           # frame lse
            'l_split': int(fps * 2 / 3),              # frame ls
            'bound_ignore': int(fps * 4 / 15),        # frame
            't_peak_valley_ratio': 0.244599954,       # ratio r1
            't_peak_ratio': 0.284047166,              # ratio r2
            't_valley_gap': 0.53090293 * flow_scale,  # pixel vg
            't_peak_relative_inf': 2.697395567 * flow_scale,  # pixel pri
            't_peak_inf': 0.75216054 * flow_scale,    # pixel pi
            't_ext_gap': 0.939653349 * flow_scale,    # pixel emd signal gap
            'frequency_inf': 1,                       #
            'frequency_sup': 5,                       #
        }
        le_p_kwargs = process_kwargs.copy()
        le_p_kwargs['t_flow_gap'] = 1.8 * flow_scale
        re_p_kwargs = process_kwargs.copy()
        re_p_kwargs['t_flow_gap'] = 1.8 * flow_scale
        mth_p_kwargs = process_kwargs.copy()
        mth_p_kwargs['t_flow_gap'] = 1.85 * flow_scale
        ns_p_kwargs = process_kwargs.copy()
        ns_p_kwargs['t_flow_gap'] = 2.1 * flow_scale
        ll_p_kwargs = process_kwargs.copy()
        rl_p_kwargs = process_kwargs.copy()
        main_casme2(le_p_kwargs, re_p_kwargs, mth_p_kwargs, ns_p_kwargs, ll_p_kwargs, rl_p_kwargs,
                    t_flow_percent, face_size)
    else:
        fps = 30
        face_size = 284
        flow_scale = face_size / 256

        process_kwargs = {
            'fps': fps,                               # frame
            'l_expand': fps,                          # frame le
            'l_small_expend': int(fps / 3),           # frame lse
            'l_split': int(fps * 2 / 3),              # frame ls
            'bound_ignore': int(fps * 4 / 15),        # frame
            't_peak_valley_ratio': 0.33,              # ratio r1
            't_peak_ratio': 0.33,                     # ratio r2
            't_valley_gap': 0.3 * 256 / 284 * flow_scale,         # pixel vg
            't_peak_relative_inf': 1.4 * 256 / 284 * flow_scale,  # pixel pri
            't_peak_inf': 0.7 * 256 / 284 * flow_scale,           # pixel pi
            't_ext_gap': 0.8 * 256 / 284 * flow_scale,            # pixel emd signal gap
            'frequency_inf': 1,  #
            'frequency_sup': 5,  #
        }
        le_p_kwargs = process_kwargs.copy()
        le_p_kwargs['t_flow_gap'] = 1.65 * 256 / 284 * flow_scale
        re_p_kwargs = process_kwargs.copy()
        re_p_kwargs['t_flow_gap'] = 1.65 * 256 / 284 * flow_scale
        mth_p_kwargs = process_kwargs.copy()
        mth_p_kwargs['t_flow_gap'] = 1.5 * 256 / 284 * flow_scale
        ns_p_kwargs = process_kwargs.copy()
        ns_p_kwargs['t_flow_gap'] = 1.5 * 256 / 284 * flow_scale
        ll_p_kwargs = process_kwargs.copy()
        rl_p_kwargs = process_kwargs.copy()
        main_samml(le_p_kwargs, re_p_kwargs, mth_p_kwargs, ns_p_kwargs, ll_p_kwargs, rl_p_kwargs,
                   t_flow_percent, face_size)


if __name__ == '__main__':
    main()
