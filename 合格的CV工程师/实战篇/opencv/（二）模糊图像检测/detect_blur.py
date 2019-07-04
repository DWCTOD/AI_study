# 用法：直接运行下面的操作即可
# python3 detect_blur.py --images images
# 按 q 不断切换到下一张图片

# 导入需要的库
from imutils import paths
import argparse
import cv2

def variance_of_laplacian(image):
	# 对图像进行拉普拉斯变换，并返回拉普拉斯的方差
	return cv2.Laplacian(image, cv2.CV_64F).var()

# 设置需要输入的参数，-i 输入图像的路径（文件夹名）， -t 设置阈值的大小
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
	help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

# 循环输入图片
for imagePath in paths.list_images(args["images"]):
	# 导入图像并转换为灰度图，计算我们关注的拉普拉斯变换后的方差
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	text = "Not Blurry"

	# 当方差小于阈值时，显示为模糊
	if fm < args["threshold"]:
		text = "Blurry"

	# 显示图片
	cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	cv2.imshow("Image", image)
	key = cv2.waitKey(0)
