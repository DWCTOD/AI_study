#　导入必要的包
import numpy as np
import imutils
import cv2


# 导入图片
image = cv2.imread('logo1.png',0)

# 寻找图片中的所有黑色形状
shapeMask = cv2.inRange(image, 0, 200)
shapeMask = cv2.dilate(shapeMask, None, iterations = 10)
# 在mask上找到轮廓
cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("I found {} black shapes".format(len(cnts)))



cv2.imshow("Mask", shapeMask)
#cv2.imshow("")

# 遍历所有轮廓
for c in cnts:
    # 绘制轮廓并显示
    rect = cv2.minAreaRect(c) # 返回的结果是 矩形中心点，矩形长和宽，以及旋转角度
    print(rect)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
