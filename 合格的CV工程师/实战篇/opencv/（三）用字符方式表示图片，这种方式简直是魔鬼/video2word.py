# -*- coding:utf-8 -*-
#coding:utf-8
import argparse
import os
import cv2
import subprocess
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
from PIL import Image, ImageFont, ImageDraw
 
# 命令行输入参数处理
# aparser = argparse.ArgumentParser()
# aparser.add_argument('file')
# aparser.add_argument('-o','--output')
# aparser.add_argument('-f','--fps',type = float, default = 24)#帧
# aparser.add_argument('-s','--save',type = bool, nargs='?', default = False, const = True)
# 是否保留Cache文件，默认不保存
 
# 获取参数
# args = parser.parse_args()
# INPUT = args.file
# OUTPUT = args.output
# SAVE = args.save
# FPS = args.fps
# 像素对应ascii码
 
 
ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:oa+>!:+. ")
 
 
# ascii_char = list("MNHQ$OC67+>!:-. ")
# ascii_char = list("MNHQ$OC67)oa+>!:+. ")
 
# 将像素转换为ascii码
def get_char(r, g, b, alpha=256):
    if alpha == 0:
        return ''
    length = len(ascii_char)
    gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
    unit = (256.0 + 1) / length
    return ascii_char[int(gray / unit)]
 
 
# 将txt转换为图片
def txt2image(file_name):
    im = Image.open(file_name).convert('RGB')
    # gif拆分后的图像，需要转换，否则报错，由于gif分割后保存的是索引颜色
    raw_width = im.width
    raw_height = im.height
    width = int(raw_width / 6)
    height = int(raw_height / 15)
    im = im.resize((width, height), Image.NEAREST)
 
    txt = ""
    colors = []
    for i in range(height):
        for j in range(width):
            pixel = im.getpixel((j, i))
            colors.append((pixel[0], pixel[1], pixel[2]))
            if (len(pixel) == 4):
                txt += get_char(pixel[0], pixel[1], pixel[2], pixel[3])
            else:
                txt += get_char(pixel[0], pixel[1], pixel[2])
        txt += '\n'
        colors.append((255, 255, 255))
 
    im_txt = Image.new("RGB", (raw_width, raw_height), (255, 255, 255))
    dr = ImageDraw.Draw(im_txt)
    # font = ImageFont.truetype(os.path.join("fonts","汉仪楷体简.ttf"),18)
    font = ImageFont.load_default().font
    x = y = 0
    # 获取字体的宽高
    font_w, font_h = font.getsize(txt[1])
    font_h *= 1.37  # 调整后更佳
    # ImageDraw为每个ascii码进行上色
    for i in range(len(txt)):
        if (txt[i] == '\n'):
            x += font_h
            y = -font_w
           # self, xy, text, fill = None, font = None, anchor = None,
            #*args, ** kwargs
        dr.text((y, x), txt[i],  fill=colors[i])
        #dr.text((y, x), txt[i], font=font, fill=colors[i])
        y += font_w
 
    name = file_name
    #print(name + ' changed')
    im_txt.save(name)
 
 
# 将视频拆分成图片
def video2txt_jpg(file_name):
    vc = cv2.VideoCapture(file_name)
    c = 1
    if vc.isOpened():
        r, frame = vc.read()
        if not os.path.exists('Cache'):
            os.mkdir('Cache')
        os.chdir('Cache')
    else:
        r = False
    while r:
        cv2.imwrite(str(c) + '.jpg', frame)
        txt2image(str(c) + '.jpg')  # 同时转换为ascii图
        r, frame = vc.read()
        c += 1
    os.chdir('..')
    return vc
 
 
# 将图片合成视频
def jpg2video(outfile_name, fps):
    fourcc = VideoWriter_fourcc(*"MJPG")
 
    images = os.listdir('Cache')
    im = Image.open('Cache/' + images[0])
    vw = cv2.VideoWriter(outfile_name + '.avi', fourcc, fps, im.size)
 
    os.chdir('Cache')
    for image in range(len(images)):
        # Image.open(str(image)+'.jpg').convert("RGB").save(str(image)+'.jpg')
        frame = cv2.imread(str(image + 1) + '.jpg')
        vw.write(frame)
        #print(str(image + 1) + '.jpg' + ' finished')
    os.chdir('..')
    vw.release()
 
 
# 递归删除目录
def remove_dir(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            dirs = os.listdir(path)
            for d in dirs:
                if os.path.isdir(path + '/' + d):
                    remove_dir(path + '/' + d)
                elif os.path.isfile(path + '/' + d):
                    os.remove(path + '/' + d)
            os.rmdir(path)
            return
        elif os.path.isfile(path):
            os.remove(path)
        return
 
 
# 调用ffmpeg获取mp3音频文件
def video2mp3(file_name):
    outfile_name = file_name.split('.')[0] + '.mp3'
    subprocess.call('ffmpeg -i ' + file_name + ' -f mp3 ' + outfile_name, shell=True)
 
 
# 合成音频和视频文件
def video_add_mp3(file_name, mp3_file):
    outfile_name = file_name.split('.')[0] + '-txt.mp4'
    subprocess.call('ffmpeg -i ' + file_name + ' -i ' + mp3_file + ' -strict -2 -f mp4 ' + outfile_name, shell=True)
 
 
if __name__ == '__main__':
    INPUT = r"a.mp4"
    OUTPUT = r"a.mp4"
    SAVE = r"a.mp4"
    FPS = "24"
    vc = video2txt_jpg(INPUT)
    FPS = vc.get(cv2.CAP_PROP_FPS)  # 获取帧率
    print(FPS)
 
    vc.release()
 
    jpg2video(INPUT.split('.')[0], FPS)
    print(INPUT, INPUT.split('.')[0] + '.mp3')
    video2mp3(INPUT)
    video_add_mp3(INPUT.split('.')[0] + '.avi', INPUT.split('.')[0] + '.mp3')
 
    if (not SAVE):
        remove_dir("Cache")
        os.remove(INPUT.split('.')[0] + '.mp3')
        os.remove(INPUT.split('.')[0] + '.avi')