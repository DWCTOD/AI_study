import cv2

# 1、读取图片
src = cv2.imread('yibiaopan.png')
# 2、识别仪表盘，并二值化后做分析处理
cv2.imshow('src',src)
cv2.waitKey(0)
cv2.destoryWindows()
# 3、显示度数
