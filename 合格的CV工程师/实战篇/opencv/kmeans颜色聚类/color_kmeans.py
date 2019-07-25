# 导入必要的包
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import pylab
import cv2

# 构造解析参数并分析这些参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True , type = int,
                help = "# 簇数")
args = vars(ap.parse_args())

# 导入图片并将其从BGR转为RGB
#　这样子我们可以用matplotlib显示
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 显示我们的图片
plt.figure()
plt.axis("off")
plt.imshow(image)
# pylab.show()

# 将图片变成一个像素列表
image = image.reshape((image.shape[0]*image.shape[1], 3))

# 像素点亮度值聚类
clt = KMeans(n_clusters = args["clusters"])
clt.fit(image)

#　创建一个簇的直方图
#  创建一个用不同颜色代表像素点数量的图形
hist = utils.centroid_histogram(clt)
bar = utils.plot_colors(hist, clt.cluster_centers_)

# 显示我们的颜色棒
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
