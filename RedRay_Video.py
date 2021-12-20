import os
import ctypes
import ctypes as C
import time
import numpy as np
import cv2
from camera_calib import *
from multiprocessing import Process, Queue
from demo import TargetDetector, plot_one_box

CUR_PATH = os.path.dirname(__file__)
dllPath = os.path.join(CUR_PATH, "./data/IRSDK.dll")
pDll = ctypes.WinDLL(dllPath)
# dll = C.cdll.LoadLibrary("draw_circle.dll")


call_calib = True
matrix, back_matrix = None, None
ray_img = None


def get_temper(box, frame, ret_img):
    matrix, back_matrix = load_matrix()
    # for pp in box:
    #     cv2.rectangle(frame, (pp[0], pp[1]), (pp[2], pp[3]), (0, 0, 255), thickness=3)

    points = back_projection(box, back_matrix)

    x = np.array(points, dtype=np.int32)

    # for i, _ in enumerate(box):
    #     print('i:', i)
    #     cv2.polylines(ret_img, [x[i:i*4+4, :]], True, (0, 0, 255), thickness=2)
    #     xy_p = x[i:i*4+4, :]
    #     for xy in xy_p:
    #         cv2.circle(ret_img, (xy[0], xy[1]), 2, (255, 0, 0), 2)
    #     cv2.imshow('tt', ret_img)
    #     cv2.waitKey(3000)
    points = points.reshape(-1).tolist()
    array = (ctypes.c_float * len(points))(*points)
    p = (ctypes.c_float * len(box))()

    ll = pDll.getObjTemper(array, p, len(box))
    # for i in range(len(box)):
    #     print(p[i])
    return np.array(p)


def load_ray_video(ip):
    count = 0
    flag = False
    print('connecting, Please waiting ...')
    while True: # count < 20:
        if pDll.IRSDK_InitConnect(ctypes.c_wchar_p(ip), len(ip)) == 1:
            print('connect success')
            flag = True
            break
        count += 1
    # cv2.namedWindow('src')
    # cv2.moveWindow('src', 100, 200)

    global ray_img
    rows, cols = 288, 384
    ret_img = np.zeros(dtype=np.uint8, shape=(rows, cols, 3))

    while flag:
        rt = pDll.IRSDK_GetRgbFrame(ret_img.ctypes.data_as(C.POINTER(C.c_ubyte)))
        if rt == 1:
            # pDll.IRSDK_GetRgbFrame(ret_img.ctypes.data_as(C.POINTER(C.c_ubyte)))
            # cv2.imshow("src", ret_img)
            # if cv2.waitKey(100) == ord(' '):
            #     break
            ray_img = ret_img
            break
    return


def run(rtsp, ip):
    # threads = [Process(target=load_ray_video, args=(ip,)),
    #            Process(target=start_Video, args=(frame_queue, rtsp,))]
    # [thread.start() for thread in threads]

    # p1 = Process(target=start_Video, args=(frame_queue, rtsp,))
    # p1.start()
    #
    load_ray_video(ip)
    #
    # if ray_img is not None:
    #     cv2.imshow('ray', ray_img)

    TD = TargetDetector(weight='./data/weights/best_swim_1220.pt', ret_drawbox=False)

    cap = cv2.VideoCapture(rtsp)
    ret, frame = cap.read()

    cv2.namedWindow('rgb', cv2.WINDOW_GUI_NORMAL)

    names = ['person']
    colors = [[0, 0, 255]]

    t1 = time.time()
    count = 0
    while ret:
        count += 1
        pred = TD.detect(frame)
        pred = pred[pred[:, -1] == 0]

        if ray_img is not None:
            ray_c = ray_img.copy()
            if len(pred) > 0:
                temper = get_temper(pred[:, :4], frame, ray_c)
                pred = np.c_[pred, temper]
            else:
                pred = np.c_[pred, np.zeros(len(pred))]

            if ray_c is not None:
                cv2.imshow('ray', ray_c)
            for *xyxy, conf, cls, temp in pred:
                label = f'{conf:.2f} {temp:.2f}'
                plot_one_box(xyxy, frame, color=colors[int(cls)], label=label)
        cv2.imshow('rgb', frame)
        ret, frame = cap.read()
        if cv2.waitKey(1) == ord(' '):
            break
    t2 = time.time()
    print((t2-t1)/count)
    cv2.destroyAllWindows()
    pDll.IRSDK_Release()


if __name__ == '__main__':
    rtsp = 'rtsp://admin:123@192.168.1.51:554'
    ip = '192.168.1.88'
    run(rtsp, ip)

    # while True:
    #     print('配准')
    #     # print(ray_queue, frame_queue)
    #     if ray_queue.empty() is False and frame_queue.empty() is False:
    #         img1 = frame_queue.get()
    #         img2 = ray_queue.get()
    #         get_matrix(img1, img2)
