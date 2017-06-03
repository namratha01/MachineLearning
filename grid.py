import numpy as np
np.random.seed(1)
import Tkinter as tk
import time

UNIT = 40   # pixels
MAZE_H = 10  # grid height
MAZE_W = 10  # grid width

canvas = tk.Canvas(bg='white',height=MAZE_H*UNIT,width=MAZE_W*UNIT)

for c in range(0,MAZE_W*UNIT,UNIT):
    x0,y0,x1,y1=c,0,c,MAZE_H*UNIT
    canvas.create_line(x0,y0,x1,y1)
for r in range(0,MAZE_H*UNIT,UNIT):
    x0,y0,x1,y1=0,r,MAZE_H*UNIT,r
    canvas.create_line(x0,y0,x1,y1)

origin = np.array([20,20])

# hell
hell1_center = origin + np.array([UNIT * 2, UNIT])
hell1 = canvas.create_rectangle(hell1_center[0] - 15, hell1_center[1] - 15,hell1_center[0] + 15, hell1_center[1] + 15,fill='black')
# hell
hell2_center = origin + np.array([UNIT, UNIT * 2])
hell2 = canvas.create_rectangle(hell2_center[0] - 15, hell2_center[1] - 15,hell2_center[0] + 15, hell2_center[1] + 15,fill='black')
# create oval
oval_center = origin + UNIT * 2
oval = canvas.create_oval(oval_center[0] - 15, oval_center[1] - 15,oval_center[0] + 15, oval_center[1] + 15,fill='yellow')
# create red rect
rect = canvas.create_rectangle(origin[0] - 15, origin[1] - 15,origin[0] + 15, origin[1] + 15,fill='red')

def moveit():
    for i in range(1,10):
        canvas.move(rect,40,40)  # move agent
        canvas.update_idletasks()
        time.sleep(1)

canvas.pack()
canvas.after(1000,moveit)
canvas.mainloop()
