import argparse
import time
from numpy import random
import os
import numpy as np
import cv2
import torch
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator
from utils.torch_utils import select_device, load_classifier

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    # print(img)
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class TargetDetector:
    def __init__(self, weight='./data/weights/best.pt', draw_label=True, mask_img=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--hide-labels', type=bool, default=False, help='hide labels')
        parser.add_argument('--hide-conf', type=bool, default=False, help='hide conf')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', type=bool, default=False, help='display results')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')

        opt = parser.parse_args()
        self.opt = opt
        self.drawbox = draw_label
        imgsz = opt.img_size
        self.mask_img = mask_img

        with torch.no_grad():
            # Initialize
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            self.model = attempt_load(weight, map_location=self.device)  # load FP32 model
            self.stride = int(self.model.stride.max())  # model stride
            self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
            if self.half:
                self.model.half()  # to FP16

            # Run inference
            if self.device.type != 'cpu':
                self.model(
                    torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

    def detect(self, img):
        opt = self.opt
        with torch.no_grad():
            # Padded resize
            im0 = img
            img = letterbox(img, self.imgsz, stride=self.stride)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = self.model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            det = pred[0]
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            ret = det.cpu().numpy()

            # 去除异常的检测框
            ret = ret[ret[:, -1] == 0]
            if self.mask_img is not None:
                ret = self.remove_outside_border(ret)

            if self.drawbox:
                annotator = Annotator(im0, line_width=3, example=str('abc'))

                names = ['person']
                colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
                colors = [[0, 0, 255]]
                for *xyxy, conf, cls in ret:
                    c = int(cls)  # integer class
                    label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors[int(cls)])
            return ret

    def remove_outside_border(self, pred):
        center = np.c_[(pred[:, 0] + pred[:, 2]) / 2, (pred[:, 1] + pred[:, 3]) / 2].astype(np.int32)
        index = []
        img = self.mask_img.copy()
        for cc in center:
            index.append(True) if self.mask_img[cc[1], cc[0]] != 0 else index.append(False)
            # if self.mask_img[cc[1], cc[0]] != 0:
            #     cv2.circle(img, (cc[0], cc[1]), radius=3, color=(255, 255, 255), thickness=4)
            #     cv2.circle(im0, (cc[0], cc[1]), radius=3, color=(255, 255, 255), thickness=4)

        return pred[index]


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


points = []


def on_EVENT_BUTTONDOWN(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.pop()


def generate_mask(img):
    mask = np.zeros(img.shape[:2])
    cv2.namedWindow('img', cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback("img", on_EVENT_BUTTONDOWN, True)
    # cv2.namedWindow('mask', cv2.WINDOW_GUI_NORMAL)
    while True:
        img_c1 = img.copy()
        x = np.array(points, dtype=np.int32)
        cv2.polylines(img_c1, [x], True, (0, 0, 255), thickness=2)
        for p in x:
            cv2.circle(img_c1, (p[0], p[1]), 2, (0, 0, 255), thickness=3)
        if len(x) > 2:
            cv2.fillConvexPoly(mask, x, (255, 255, 255))

        cv2.imshow("img", img_c1)
        if cv2.getWindowProperty('img', 0) == -1 or cv2.waitKey(10) == ord(' '):  # 当窗口关闭时为-1，显示时为0
            break
    cv2.destroyAllWindows()
    if len(points) == 0:
        return None
    else:
        return mask


if __name__ == '__main__':
    # video_path = 'rtsp://admin:123@192.168.1.51:554'
    video_path = './data/video/20211102195133468.avi'
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    mask = generate_mask(frame)

    TD = TargetDetector('./data/weights/best_swim_1220.pt', mask_img=mask)

    cv2.namedWindow('src', cv2.WINDOW_GUI_NORMAL)
    while ret:
        pred = TD.detect(frame)
        x = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [x], True, (0, 0, 255), thickness=2)
        cv2.imshow("src", frame)
        ret, frame = cap.read()
        if cv2.waitKey(1) == ord(' '):
            break
    cv2.destroyAllWindows()

    # path = r'./data/images/'
    # tt = time.time()
    # for file in os.listdir(path):
    #     img = cv2.imread(path + file)
    #     pred = TD.detect(img)
    #     print(time.time() - tt)
    #     tt = time.time()
    #     print(pred)
