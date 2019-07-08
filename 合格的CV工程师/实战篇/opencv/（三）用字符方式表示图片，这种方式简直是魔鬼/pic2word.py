# coding:utf-8 
  
# 为一张图片生成对应的字符集图片 
  
from PIL import Image 
import argparse 
  
# 命令行输入参数处理 
#parser = argparse.ArgumentParser() 
  
# parser.add_argument('file')   # 输入文件 
# parser.add_argument('-o', '--output')  # 输出文件 
# parser.add_argument('--width', type=int, default=80) # 输出字符画宽 
# parser.add_argument('--height', type=int, default=80) # 输出字符画高 
  
# 获取参数 
# args = parser.parse_args() 
  
IMG = 'DDM.png'
WIDTH = 80
HEIGHT = 80 
OUTPUT = 'DDM-w.txt'
  
ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ") 
  
  
# 将256灰度映射到70个字符上 
def get_char(r, b, g, alpha=256): 
  if alpha == 0: 
    return ' '
  length = len(ascii_char) 
  gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b) 
  
  unit = (256.0+1)/length 
  print(unit)
  return ascii_char[int(gray/unit)] 
  
if __name__ == '__main__': 
  
  im = Image.open(IMG) 
  im = im.resize((WIDTH, HEIGHT), Image.NEAREST) 
  
  txt = "" 
  
  for i in range(HEIGHT): 
    for j in range(WIDTH): 
      txt += get_char(*im.getpixel((j, i))) 
    txt += '\n'
  
  print(txt) 
  
  # 字符画输出到文件 
  if OUTPUT: 
    with open(OUTPUT,'w') as f: 
      f.write(txt) 
  else: 
    with open("output.txt", 'w') as f: 
      f.write(txt)