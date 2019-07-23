#　导入必要的包
import numpy as np
import argparse
import imutils
import cv2

# 构造解析参数并分析参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="paht to the image file")
args = vars(ap.parse_args())
# 导入图片
image = cv2.imread(args["image"])

# 寻找图片中的所有黑色形状
lower = np.array([0, 0, 0])
upper = np.array([15, 15, 15])
shapeMask = cv2.inRange(image, lower, upper)
# 在mask上找到轮廓
cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("I found {} black shapes".format(len(cnts)))
cv2.imshow("Mask", shapeMask)

# 遍历所有轮廓
for c in cnts:
    # 绘制轮廓并显示
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
