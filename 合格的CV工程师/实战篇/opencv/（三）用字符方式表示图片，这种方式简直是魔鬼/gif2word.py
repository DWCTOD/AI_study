import imageio as igo
import cv2

pics = igo.mimread('a.gif')
#print(pics.shape)
string = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:oa+>!:+."
A=[]
for img in pics:
    u,v,_=img.shape
    c=img*0+255
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    for i in range(0,u,24):
        for j in range(0,v,15):
            pix = gray[i,j]
            b,g,r,_=img[i,j]
            zifu = string[int(((len(string)-1)*pix)/256)]
            cv2.putText(c,zifu,(j,i),
                    cv2.FONT_HERSHEY_COMPLEX,0.3,
                    (int(b),2*int(g),int(r),1))
    A.append(c)
igo.mimsave('a-w.gif',A,'GIF',duration=0.1)
