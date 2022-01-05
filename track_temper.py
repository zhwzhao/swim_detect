import gc
import math
from os import write
import random
from threading import Thread
import threading
import cv2
from sort import *
from demo import TargetDetector, generate_mask
from demo import plot_one_box
from RedRay_Video import *
from utils.util import calDistance
import config


e2 = {}
LIGHT_GREEN = (204, 232, 207)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)



def plot_one_box(x, img, outText, count, color=None, label=None, line_thickness=3, outFlag=False):
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

    if outFlag:
        if float(label) in count:
            outText.setdefault(str(color), []).append([label, count[float(label)], x])
        else:
            outText.setdefault(str(color), []).append([label, 0, x])


def write_ret(outText):
    output_str = '[green]:\n'
    if str(GREEN) in outText:  # 绿框
        for k, t, v in outText[str(GREEN)]:
            output_str += '%-6s' % k
            for xy in v:
                output_str += '%-5d' % xy
            output_str += '%-5d\n' % (t / 24)

    output_str += '\n[yellow]:\n'
    if str(YELLOW) in outText:  # 黄框
        for k, t, v in outText[str(YELLOW)]:
            output_str += '%-6s' % k
            for xy in v:
                output_str += '%-5d' % xy
            output_str += '%-5d\n' % (t / 24)

    output_str += '\n[red]:\n'
    if str(RED) in outText:  # 红框
        for k, t, v in outText[str(RED)]:
            output_str += '%-6s' % k
            for xy in v:
                output_str += '%-5d' % xy
            output_str += '%-5d\n' % (t / 24)

    with open("output.sos", "w") as file:
        file.write(output_str)


def read_frame(q, video_path):
    t3 = Thread(target=txtInput, daemon=True)
    t3.start()

    mot_tracker = Sort()

    frames = 0
    tracked_objects_previous = {}
    D = 1
    T = 30
    Tb = 5
    Tc = 150
    count = {}
    timer = {}
    boundTimer = {}
    selectPerson = {}
    vid_writer = None
    global e2

    # ret, frame = cap.read()
    # mask = generate_mask(frame) if ret else None

    temper_flag = load_ray_video(config.ip)

    TD = TargetDetector()
    starttime = time.time()
    cv2.namedWindow('Stream', cv2.WINDOW_GUI_NORMAL)
    while True:
        if q.empty():
            continue
        frame = q.get()

        outFlag = False
        outText = {}

        frames += 1
        if frames % 24 == 0:
            outFlag = True

        detections = TD.detect(frame)

        # 在detections的最后一列加上每个目标的温度，如果连接红外失败则温度为0
        if temper_flag:
            if len(detections) > 0:
                temper = get_temper(detections[:, :4])
                # print(temper)
                # detections = np.c_[detections, temper]
            # else:
            # detections = np.c_[detections, np.zeros(len(detections))]

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
                                if selectPerson[obj_id] is not None:
                                    selectPerson[obj_id] = None
                                    count[obj_id] = 0

                if obj_id in selectPerson and selectPerson[obj_id] != None:  # 特判
                    # 计时？
                    if obj_id not in count:
                        count[obj_id] = 0
                    count[obj_id] = count[obj_id] + 1
                    if count[obj_id] >= Tc:
                        # print(f"alert:{obj_id.astype(int)}")
                        clr = RED  # red
                        line_thickness = 6
                    else:
                        clr = YELLOW  # yellow
                    plot_one_box(box, img, outText, count, label=label, color=clr, line_thickness=line_thickness,
                                 outFlag=outFlag)
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
                plot_one_box(box, img, outText, count, label=label, color=clr, line_thickness=line_thickness,
                             outFlag=outFlag)

            tracked_objects_previous.clear()
            for x1, y1, x2, y2, obj_id in tracked_objects:
                tracked_objects_previous[obj_id] = [x1, y1, x2, y2]

        cv2.imshow('Stream', img)
        if config.save_video:
            if vid_writer is None:
                # fps = cap.get(cv2.CAP_PROP_FPS)
                # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps, w, h = 30, img.shape[1], img.shape[0]
                vid_writer = cv2.VideoWriter(video_path.replace('.avi', '_result1.avi'), cv2.VideoWriter_fourcc(*'mp4v'),
                                             fps, (w, h))
            vid_writer.write(img)
        if outFlag:
            write_ret(outText)

        if cv2.waitKey(1) == ord(' '):
            if vid_writer is not None:
                vid_writer.release()
            break
    cv2.destroyAllWindows()
    totaltime = time.time() - starttime
    print(frames, "frames", totaltime / frames, "s/frame")



# txtInput
def txtInput():
    global e2
    path = "C:/Windows/Temp/68768787.sos"
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
# t1 = Thread(target=tracking)
# t3 = Thread(target=txtInput, daemon=True)
#
# t1.start()
# t3.start()


def write_frame(q, cam) -> None:
    """
    :param q: 摄像头参数
    :param cam: manager.list对象
    :return: None
    """
    print('Process to write: %s' % os.getpid())
    cap = cv2.VideoCapture(cam)
    ret, img = cap.read()
    while True:
        q.put(img)
        ret, img = cap.read()
        while ret is False:
            print('read failed')
            cap = cv2.VideoCapture(cam)
            ret, img = cap.read()


if __name__ == '__main__':
    select = input("please select input(1:video, 2:stream, 3:shenzhen video): ")
    if select == '2':  # 输入2为流
        video_path = config.rtsp_ip
    elif select == '3':
        video_path = config.test_video
    else:  # 默认视频
        video_path = config.cuiniao_video
    print(f'video_path:{video_path}\n')

    q = Queue(maxsize=10)
    pw = Process(target=write_frame, args=(q, video_path,))
    pr = Process(target=read_frame, args=(q, video_path, ))
    pw.start()
    pr.start()
    pr.join()
    pw.terminate()
    #
    # t3 = Thread(target=txtInput, daemon=True)
    # t3.start()
