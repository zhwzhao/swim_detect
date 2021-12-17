import math
from os import write
# from os import tcgetpgrp
import random
from threading import Thread
import threading
import cv2
from matplotlib.pyplot import draw
from numpy import exp2
from sort import *
from demo import TargetDetector
from demo import plot_one_box

TD = TargetDetector()

# videopath = r'./data/video/20211102195133468.avi'
# videopath = r'./data/video/20211101181037570.avi'
videopath = 'rtsp://admin:123@192.168.1.51:554'

select = input("please select input(1:video, 2:stream): ")
if select == '2': # 输入2为流
    videopath = 'rtsp://admin:123@192.168.1.51:554'
else: # 默认视频
    videopath = r'./data/video/20211102195133468.avi'
print(f'video_path:{videopath}\n')

point1, point2 = None, None
startEnd = False
drawing = False
outText = {}
LIGHT_GREEN = (204, 232, 207)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)


def on_mouse(event, x, y, flags, param):
    global point1, point2, startEnd, drawing
    if event == cv2.EVENT_LBUTTONDOWN:  #左键点击
        startEnd = True
        drawing = True
        point1 = (int(x), int(y))
        point2 = point1
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON): #按住左键拖曳
        startEnd = False
        drawing = True
        point2 = (int(x), int(y))
    elif event == cv2.EVENT_LBUTTONUP:  #左键释放
        startEnd = True
        drawing = False
        point2 = (int(x), int(y))
    elif event == cv2.EVENT_RBUTTONUP:  # 右键取消选择
        point1, point2 = None, None

def calDistance(a, b):
    centerAx = (a[2]-a[0])/2 + a[0]
    centerAy = (a[3]-a[1])/2 + a[1]
    centerBx = (b[2]-b[0])/2 + b[0]
    centerBy = (b[3]-b[1])/2 + b[1]
    d = math.hypot(centerAx-centerBx, centerAy-centerBy)
    #print(d)
    return d

def plot_one_box(x, img, color=None, label=None, line_thickness=3, outFlag=False):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    if color == LIGHT_GREEN:
        centerX = int((x[2]+x[0])/2)
        centerY = int((x[3]+x[1])/2)
        cv2.circle(img,(centerX,centerY),10,color,-1)
        return

    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    global outText
    if outFlag:
        outText.setdefault(str(color), []).append([label, x])

def tracking():
    global point1, point2, startEnd, drawing, e1
    cap = cv2.VideoCapture(videopath)
    cv2.namedWindow('Stream', cv2.WINDOW_GUI_NORMAL)
    mot_tracker = Sort()

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Video size", w,h)
    vid_writer = cv2.VideoWriter(videopath.replace('.avi', '_result1.avi'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frames = 0
    starttime = time.time()

    tracked_objects_previous = {}
    clr = (0, 255, 0)
    line_thickness = 3
    label = ''
    D = 1
    T = 30
    Tb = 5
    Tc = 150
    count = {}
    timer = {}
    boundTimer = {}
    selectPerson = {}
    global e2
    outFlag = False

    cv2.setMouseCallback("Stream", on_mouse)
    while True:
        # if e2 != None:  # 键入选中
        #     print(type(e2))

        outFlag = False
        outText.clear()
        if frames%5 == 0:
            outFlag = True

        ret, frame = cap.read()
        if not ret:
            break
        detections, _ = TD.detect(frame)
        img = np.asarray(frame)
        if detections is not None:
            # print(detections)
            tracked_objects = mot_tracker.update(detections)
            # print(tracked_objects)
            
            for *box, obj_id in tracked_objects:
                clr = LIGHT_GREEN # ligth green
                line_thickness = 3
                label = str(obj_id.astype(int))
                
                if e2 != None:  # 键入选中
                    oid = str(obj_id.astype(int))
                    # print("notNone")
                    if oid == e2:
                        # print("==")
                        if obj_id not in selectPerson:  # 没有：加入，开始计时
                            # print("1")
                            selectPerson[obj_id] = box
                        elif obj_id in selectPerson:
                            # print("2")
                            if selectPerson[obj_id] == None: # 另一种没有（）
                                # print("3")
                                selectPerson[obj_id] = box
                            else:   # 有：清空
                                # print("4")
                                selectPerson[obj_id] = None
                                count[obj_id] = 0
                        e2 = None


                if obj_id in selectPerson and selectPerson[obj_id] != None:  # 特判
                    # 计时？
                    if obj_id not in count:
                        count[obj_id] = 0
                    count[obj_id] = count[obj_id] + 1
                    if count[obj_id] >= Tc:
                        print(f"alert:{obj_id.astype(int)}")
                        clr = RED # red
                        line_thickness = 6
                    else:
                        clr = YELLOW # yellow
                    plot_one_box(box, img, label=label, color=clr, line_thickness=line_thickness, outFlag=outFlag)
                    continue

                if obj_id in tracked_objects_previous: # 与前一帧对比
                    dis = calDistance(box, tracked_objects_previous[obj_id])
                    # print(dis)
                    if dis >= D: # 速度较快
                        if obj_id in timer: # 动起来了，计时器复位
                            if obj_id not in boundTimer:    # 弹性帧
                                boundTimer[obj_id] = 0
                            else:
                                boundTimer[obj_id] = boundTimer[obj_id] + 1
                                if boundTimer[obj_id] >= Tb:
                                    timer[obj_id] = 0

                            
                    else: # 速度慢，加入队列
                        if obj_id not in timer: # 计时器计数
                            timer[obj_id] = 0
                        timer[obj_id] = timer[obj_id] + 1
                        if timer[obj_id] >= T: # 累计计数达到阈值
                            print(f"alert:{obj_id.astype(int)}")
                            clr = RED # red
                            line_thickness = 6
                        else:
                            clr = GREEN # green
                plot_one_box(box, img, label=label, color=clr, line_thickness=line_thickness, outFlag=outFlag)
            
            tracked_objects_previous.clear()
            for x1, y1, x2, y2, obj_id in tracked_objects:
                tracked_objects_previous[obj_id] = [x1, y1, x2, y2]
        
        if outFlag:
            # print(outText)
            with open("output.txt", "w") as file:
                file.write('(green):\n')
                if str(GREEN) in outText: #绿框
                    for k, v in outText[str(GREEN)]:
                        file.write(k)
                        file.write(" ")
                        file.write(str(v))
                        file.write('\n')
                file.write('\n')
                file.write('(yellow):\n')
                if str(YELLOW) in outText: #黄框
                    for k, v in outText[str(YELLOW)]:
                        file.write(k)
                        file.write(" ")
                        file.write(str(v))
                        file.write('\n')
                file.write('\n')
                file.write('(red):\n')
                if str(RED) in outText: #红框
                    for k, v in outText[str(RED)]:
                        file.write(k)
                        file.write(" ")
                        file.write(str(v))
                        file.write('\n')
                file.write('\n')


        if e2 != None:
            e2 = None
        # # 鼠标框选事件
        # if point1 != point2:
        #     cv2.rectangle(img, point1, point2, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        #     if startEnd == True and drawing == False:
        #         for *box, obj_id in tracked_objects: # 找出被选中的人
        #             centerX = (box[2]+box[0])/2
        #             centerY = (box[3]+box[1])/2
        #             if centerX >= point1[0] and centerX <= point2[0] and centerY >= point1[1] and centerY <=point2[1]:
        #                 selectPerson[obj_id] = box
        #                 # print(selectPerson[obj_id])
        #         point1 = point2
        # else:
        #     if point1 == None and point2 == None: # 右键清楚选中
        #         selectPerson.clear()
        #         count.clear()
        
        
        # img = cv2.resize(img, (1080, 720))
        cv2.imshow('Stream', img)
        cv2.waitKey(1)
        # vid_writer.write(frame)



        frames += 1
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

    totaltime = time.time()-starttime
    print(frames, "frames", totaltime/frames, "s/frame")
    cv2.destroyAllWindows()
    vid_writer.release()


# input
import tkinter
from tkinter import *
from tkinter import messagebox

e1 = None
e2 = None
def show():
    global e1, e2
    print("id:%s"% e1.get())  # 获取用户输入的信息
    e2 = e1.get()
    e1.delete(0, END)

def input():
    global e1
    root = Tk()
    root.title("Input")

    Label1 = Label(root,text='id:').grid(row=0,column=0)

    v1 = StringVar()
    e1 = Entry(root,textvariable=v1)    # Entry 是 Tkinter 用来接收字符串等输入的控件.
    e1.grid(row=0,column=1,padx=10,pady=5)  #设置输入框显示的位置，以及长和宽属性

    lock.acquire()
    Button(root,text='confirm',width=10,command=show).grid(row=2,column=0,sticky=W,padx=10,pady=5)
    lock.release()

    Button(root,text='quit',width=10,command=root.quit).grid(row=2,column=1,sticky=E,padx=10,pady=5)

    mainloop()



# main
lock = threading.Lock()
t1 = Thread(target=tracking)
t2 = Thread(target=input)

t1.start()
t2.start()