from imutils.perspective import four_point_transform
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def Get_Outline(input_dir):
    image = cv2.imread(input_dir)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5),0)
    edged = cv2.Canny(blurred,75,200)
    return image,gray,edged

def Get_cnt(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = cnts[0] if  imutils.is_cv2()  else   cnts[1]
    docCnt =None

    if len(cnts) > 0:
        cnts =sorted(cnts,key=cv2.contourArea,reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c,True)                   # 轮廓按大小降序排序
            approx = cv2.approxPolyDP(c,0.02 * peri,True)  # 获取近似的轮廓
            if len(approx) ==4:                            # 近似轮廓有四个顶点
                docCnt = approx
                break
        
    return docCnt
def calculate_distance(point1,point2):
    d_x = point1[0]-point2[0]
    d_y = point1[1]-point2[1]
    distance = math.sqrt(d_x**2+d_y**2)
    return distance

if __name__=="__main__":
    input_dir = "gongjiaoka.png"
    image,gray,edged = Get_Outline(input_dir)
    docCnt = Get_cnt(edged)
    print(docCnt.reshape(4,2))
    # result_img = four_point_transform(image, docCnt.reshape(4,2)) # 对原始图像进行四点透视变换
    # 改变变换的模式 公交卡的比例是16:9
    pts1 = np.float32(docCnt.reshape(4,2))
    # 加入一个判断，对不同宽高采用不同的系数
    p = docCnt.reshape(4,2)
    
    if calculate_distance(p[0],p[1])<calculate_distance(p[0],p[3]):
        pts2 = np.float32([[0,0],[0,180],[320,180],[320,0]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(image,M,(320,180))
    else:
        pts2 = np.float32([[0,0],[0,320],[180,320],[180,0]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(image,M,(180,320))
    cv2.imwrite('0.png',dst)
    #print(result_img.shape)
    # -------画点----------
    point_size = 2
    point_color = (0, 255, 0) # BGR
    thickness = 2 # 可以为 0 、4、8
    for point in docCnt.reshape(4,2):
        cv2.circle(image, tuple(point), point_size, point_color, thickness)
    # # --------------
    cv2.imshow("original", image)
    cv2.imshow("gray", gray)
    cv2.imshow("edged", edged)
    cv2.imshow("result_img", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 备注，（1）顺序“左上，右上，右下，左下” 
# （2）变换后图片的大小变成点之间的距离，即缩小了
