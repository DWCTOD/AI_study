#　导入必要的包
import numpy as np
import cv2

def centroid_histogram(clt):
    # 获取不同簇的数量
    # 基于每个簇的像素点数量产生直方图
    numLabels = np.arange(0,len(np.unique(clt.labels_))+1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    
    # 归一化直方图
    hist = hist.astype("float")
    hist /= hist.sum()
    
    # 返回直方图
    return hist

def plot_colors(hist, centroids):
    # 初始化每种颜色相对频率的条形图
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
    
    # 遍历每个簇的百分比和颜色
    for (percent, color) in zip(hist, centroids):
        # 画出每个簇的相对百分比
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), 
                     color.astype("uint8").tolist(), -1)
        startX = endX
        
	# 返回条形图
    return bar

