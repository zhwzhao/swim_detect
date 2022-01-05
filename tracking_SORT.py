import math
# from os import tcgetpgrp
import random
from threading import Thread
import threading
from utils.sort import *
from demo import generate_mask
from RedRay_Video import *

# videopath = r'./data/video/20211102195133468.avi'
# videopath = r'./data/video/20211101181037570.avi'
videopath = 'rtsp://admin:123@192.168.1.51:554'

select = input("please select input(1:video, 2:stream, 3:shenzhen video): ")
if select == '2':  # 输入2为流
    videopath = 'rtsp://admin:123@192.168.1.51:554'
elif select == '3':
    videopath = r'./data/video/20211221143005488.avi'
else:  # 默认视频
    videopath = r'./data/video/20211102195133468.avi'
print(f'video_path:{videopath}\n')

e2 = {}
outText = {}
count = {}
LIGHT_GREEN = (204, 232, 207)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)


def calDistance(a, b):
    centerAx = (a[2] - a[0]) / 2 + a[0]
    centerAy = (a[3] - a[1]) / 2 + a[1]
    centerBx = (b[2] - b[0]) / 2 + b[0]
    centerBy = (b[3] - b[1]) / 2 + b[1]
    d = math.hypot(centerAx - centerBx, centerAy - centerBy)
    # print(d)
    return d


def plot_one_box(x, img, color=None, label=None, line_thickness=3, outFlag=False):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    if color == LIGHT_GREEN:
        centerX = int((x[2] + x[0]) / 2)
        centerY = int((x[3] + x[1]) / 2)
        cv2.circle(img, (centerX, centerY), 10, color, -1)
        return

    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    global outText, count
    if outFlag:
        if float(label) in count:
            outText.setdefault(str(color), []).append([label, count[float(label)], x])
        else:
            outText.setdefault(str(color), []).append([label, 0, x])


def tracking():
    cap = cv2.VideoCapture(videopath)
    mot_tracker = Sort()

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Video size", w, h)
    vid_writer = cv2.VideoWriter(videopath.replace('.avi', '_result1.avi'), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                 (w, h))

    frames = 0
    tracked_objects_previous = {}
    clr = (0, 255, 0)
    line_thickness = 3
    label = ''
    D = 1
    T = 30
    Tb = 5
    Tc = 150
    global count
    timer = {}
    boundTimer = {}
    selectPerson = {}
    global e2
    outFlag = False

    # pr = Process(target=load_ray_video, args=(config.ip,))
    # pr.start()

    ret, frame = cap.read()
    mask = generate_mask(frame) if ret else None
    TD = TargetDetector(mask_img=mask)
    starttime = time.time()

    cv2.namedWindow('Stream', cv2.WINDOW_GUI_NORMAL)
    while True:
        outFlag = False
        outText.clear()
        if frames % 24 == 0:
            outFlag = True

        ret, frame = cap.read()
        if not ret:
            break
        detections = TD.detect(frame)
        # 在detections的最后一列加上每个目标的温度，如果连接红外失败则温度为0

        # if ray_img is not None:
        #     ray_c = ray_img.copy()
        #     if len(detections) > 0:
        #         temper = get_temper(detections[:, :4], frame, ray_c)
        #         print(temper)
        #         # detections = np.c_[detections, temper]
        #     # else:
        #         # detections = np.c_[detections, np.zeros(len(detections))]

        img = np.asarray(frame)
        if detections is not None:
            tracked_objects = mot_tracker.update(detections)

            # obj_id:<class 'numpy.float64'>   ==> float()
            for *box, obj_id in tracked_objects:
                clr = LIGHT_GREEN  # light green
                line_thickness = 3
                label = str(obj_id.astype(int))

                for key, value in e2.items():
                    if label == key:
                        if value == "1":  # 值为1，计时
                            if obj_id not in selectPerson:  # 没有：加入, 开始计时
                                selectPerson[obj_id] = box
                            elif obj_id in selectPerson:
                                if selectPerson[obj_id] is None:  # 另一种没有（）
                                    selectPerson[obj_id] = box
                                else:  # 有，不做操作，继续计时
                                    pass
                        else:  # 值为0，停止计时
                            if obj_id in selectPerson:
                                if selectPerson[obj_id] != None:
                                    selectPerson[obj_id] = None
                                    count[obj_id] = 0

                if obj_id in selectPerson and selectPerson[obj_id] != None:  # 特判
                    # 计时？
                    if obj_id not in count:
                        count[obj_id] = 0
                    count[obj_id] = count[obj_id] + 1
                    if count[obj_id] >= Tc:
                        print(f"alert:{obj_id.astype(int)}")
                        clr = RED  # red
                        line_thickness = 6
                    else:
                        clr = YELLOW  # yellow
                    plot_one_box(box, img, label=label, color=clr, line_thickness=line_thickness, outFlag=outFlag)
                    continue

                if obj_id in tracked_objects_previous:  # 与前一帧对比
                    dis = calDistance(box, tracked_objects_previous[obj_id])
                    # print(dis)
                    if dis >= D:  # 速度较快
                        if obj_id in timer:  # 动起来了，计时器复位
                            if obj_id not in boundTimer:  # 弹性帧
                                boundTimer[obj_id] = 0
                            else:
                                boundTimer[obj_id] = boundTimer[obj_id] + 1
                                if boundTimer[obj_id] >= Tb:
                                    timer[obj_id] = 0
                    else:  # 速度慢，加入队列
                        if obj_id not in timer:  # 计时器计数
                            timer[obj_id] = 0
                        timer[obj_id] = timer[obj_id] + 1
                        if timer[obj_id] >= T:  # 累计计数达到阈值
                            print(f"alert:{obj_id.astype(int)}")
                            clr = RED  # red
                            line_thickness = 6
                        else:
                            clr = GREEN  # green
                plot_one_box(box, img, label=label, color=clr, line_thickness=line_thickness, outFlag=outFlag)

            tracked_objects_previous.clear()
            for x1, y1, x2, y2, obj_id in tracked_objects:
                tracked_objects_previous[obj_id] = [x1, y1, x2, y2]

        if outFlag:
            with open("output.sos", "w") as file:
                file.write('[green]:\n')
                if str(GREEN) in outText:  # 绿框
                    for k, t, v in outText[str(GREEN)]:
                        file.write(k)
                        for xy in v:
                            file.write(", ")
                            file.write(str(int(xy)))
                        file.write(', ')
                        file.write(str(int(t / 24)))
                        file.write('\n')
                file.write('\n')
                file.write('[yellow]:\n')
                if str(YELLOW) in outText:  # 黄框
                    for k, t, v in outText[str(YELLOW)]:
                        file.write(k)
                        for xy in v:
                            file.write(", ")
                            file.write(str(int(xy)))
                        file.write(', ')
                        file.write(str(int(t / 24)))
                        file.write('\n')
                file.write('\n')
                file.write('[red]:\n')
                if str(RED) in outText:  # 红框
                    for k, t, v in outText[str(RED)]:
                        file.write(k)
                        for xy in v:
                            file.write(", ")
                            file.write(str(int(xy)))
                        file.write(', ')
                        file.write(str(int(t / 24)))
                        file.write('\n')
                file.write('\n')

        cv2.imshow('Stream', img)
        frames += 1
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27 or cv2.waitKey(1) == ord(' '):
            break

    totaltime = time.time() - starttime
    print(frames, "frames", totaltime / frames, "s/frame")
    cv2.destroyAllWindows()
    vid_writer.release()


# txtInput
def txtInput():
    global e2
    path = "C:/Windows/Temp/68768787.sos"
    if not os.path.exists(path):
        f = open(path, 'w+')
        f.close()
    while True:
        with open(path, 'r+') as f:
            line = f.readline()
            while line:
                tmp = line.strip()
                a = str(tmp.split(',', 1)[0])
                b = str(tmp.split(',', 1)[1])
                e2[a] = b
                line = f.readline()
        time.sleep(1)


# main
t1 = Thread(target=tracking)
t3 = Thread(target=txtInput, daemon=True)

t1.start()
t3.start()
