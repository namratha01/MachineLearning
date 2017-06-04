from PIL import ImageTk as itk
from PIL import Image
import numpy
numpy.random.seed(1)
import Tkinter as tk
import time
import random

UNIT=60     # pixels
MAZE_H=10   # grid height
MAZE_W=10   # grid width

canvas = tk.Canvas(bg='white',height=MAZE_H*UNIT,width=MAZE_W*UNIT)

for c in range(0,MAZE_W*UNIT,UNIT):
    x0,y0,x1,y1=c,0,c,MAZE_H*UNIT
    canvas.create_line(x0,y0,x1,y1)
for r in range(0,MAZE_H*UNIT,UNIT):
    x0,y0,x1,y1=0,r,MAZE_H*UNIT,r
    canvas.create_line(x0,y0,x1,y1)

origin=numpy.array([20,20])

#Can Placement
for j in range(0,10):
    can_center=origin+numpy.array([UNIT*j,UNIT*j])
    can=canvas.create_rectangle(can_center[0]-5,can_center[1]-5,can_center[0]+25,can_center[1]+25,fill='black')

#terminator
t_center=origin+numpy.array([UNIT*5, UNIT*3])
terminator=itk.PhotoImage(Image.open("/home/manas/Projects/Namratha/MachineLearning/terminator.gif"))
t2=canvas.create_image(30,30,image=terminator)

def moveit():
    for i in range(1,10):
        canvas.move(t2,UNIT,UNIT)
        canvas.update_idletasks()
        time.sleep(1)

canvas.pack()
canvas.after(1000,moveit)
canvas.mainloop()
