import numpy as np
import math
import cv2
img_name = '8.jpeg'
img = cv2.imread(img_name)
cv2.imshow('panoramagram',img)
img = wrapped_img = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
# 准备工作，计算原图像尺寸和变换后的图片大小
x0 = img.shape[0]
y0 = img.shape[1]
print(x0,y0)
# 最大半径计算
radius = int(y0/(2*math.pi))
w=2*radius
h=2*radius
wrapped_img = 255*np.ones((w, h, 3), dtype="u1")

except_count = 0
for j in range(y0):
    # 1. 求极坐标系中对应的角度theta
    theta = 2 * math.pi * (j /y0)  
    # print(theta)
    for i in range(x0):
    	# 2. 计算半径缩放系数
        wrapped_radius = (i-x0)*radius/x0
        # 3. 利用对应关系进行换算
        y = wrapped_radius * math.cos(theta) + radius  
        x = wrapped_radius * math.sin(theta) + radius
        x, y = int(x), int(y)
        try:
            wrapped_img[x, y, :] = img[i, j, :] 
            # 注意点,在数学坐标系中的坐标与数字图像中的坐标表示存在差异需要注意
        except Exception as e:
            except_count = except_count + 1

print(except_count)
# 提取ROI区域进行平滑处理，效果一般
roi = wrapped_img[0:radius,radius-5:radius+5,:]
roi_blur = cv2.blur(roi,(3,3))
wrapped_img[0:radius,radius-5:radius+5,:]=roi_blur
#wrapped_img = cv2.resize(wrapped_img,None,fx=1,fy=1,interpolation=cv2.INTER_CUBIC)
name = 'p_'+img_name
cv2.imwrite(name, wrapped_img)
cv2.imshow("Unwrapped", wrapped_img)
cv2.waitKey(0)