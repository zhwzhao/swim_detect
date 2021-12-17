import os
import ctypes
import ctypes as C
import numpy as np
import cv2

CUR_PATH = os.path.dirname(__file__)
dllPath = os.path.join(CUR_PATH, "./data/IRSDK.dll")
pDll = ctypes.WinDLL(dllPath)
# dll = C.cdll.LoadLibrary("draw_circle.dll")

img = cv2.imread('./data/12.jpg')
(rows, cols) = (img.shape[0], img.shape[1])
rows, cols = 288, 384

ret_img = np.zeros(dtype=np.uint8, shape=(rows, cols, 3))

ip = '192.168.1.88'

flag = False
count = 0
while count < 8:
    print('init...')
    if pDll.IRSDK_InitConnect(ctypes.c_wchar_p(ip), len(ip)) == 1:
        print('连接成功')
        flag = True
        break
    count += 1

box = [20, 40, 40, 40, 40, 60, 20, 40]
box.extend([100, 100, 200, 100, 200, 200, 100, 200])
array = (ctypes.c_float * len(box))(*box)
p = (ctypes.c_float * 5)()

# pDll.draw_circle(rows, cols, img.ctypes.data_as(C.POINTER(C.c_ubyte)), ret_img.ctypes.data_as(C.POINTER(C.c_ubyte)))
# cv2.imshow('circle', circle_img)

cv2.namedWindow('src')
cv2.moveWindow('src', 100, 200)

while flag:
    if pDll.IRSDK_GetRgbFrame(ret_img.ctypes.data_as(C.POINTER(C.c_ubyte))) == 1:
        # pDll.IRSDK_GetRgbFrame(ret_img.ctypes.data_as(C.POINTER(C.c_ubyte)))
        cv2.imshow("src", ret_img)
        ll = pDll.getObjTemper(array, p, 2)

        for i in range(2):
            print(p[i])

        if cv2.waitKey(100) == ord('q'):
            break

pDll.IRSDK_Release()
cv2.destroyAllWindows()


