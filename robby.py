# -*- coding: utf-8 -*-
"""
@author: Namratha Basavanahalli Manjunatha
@Des   : Q-Learning (Robby the Robot)
"""
#Imports
import sys
from PIL import ImageTk as itk
from PIL import Image
import numpy
import Tkinter as tk
import time
import random
import matplotlib.pyplot as pyplot

#Variables
no_of_episodes=5000
no_of_steps=200
reward=[-5,-1,10]
actions=[0,1,2,3,4]   # actions = ['up', 'down', 'right', 'left', 'pick']
dry_spell=0

UNIT=60     # pixels
MAZE_H=10   # grid height
MAZE_W=10   # grid width

canvas = tk.Canvas(bg='white',height=MAZE_H*UNIT,width=MAZE_W*UNIT)

origin=numpy.array([20,20])

def initialize_grid():
    grid = numpy.zeros((12,10))
    style='field'
    if style=='field': 
	grid[numpy.arange(1,11),:]=numpy.random.choice([1,2],(10,10),replace=True, p=[0.5, 0.5])
    elif style=='maze':
	grid[numpy.arange(1,11), :] = numpy.random.choice([0, 1, 2], (10, 10), replace=True, p=[0.2, 0.4, 0.4])
    grid = numpy.concatenate((numpy.zeros((12, 1)), grid, numpy.zeros((12, 1))), axis=1)
    loc_R = tuple(list((numpy.random.choice(numpy.arange(1, 11), 2))))
    for c in range(0,MAZE_W*UNIT,UNIT):
        x0,y0,x1,y1=c,0,c,MAZE_H*UNIT
        canvas.create_line(x0,y0,x1,y1)
    for r in range(0,MAZE_H*UNIT,UNIT):
        x0,y0,x1,y1=0,r,MAZE_H*UNIT,r
        canvas.create_line(x0,y0,x1,y1)

    origin=numpy.array([20,20])
    for i in range(1,MAZE_H+1):
        for j in range(1,MAZE_W+1):
            if grid[i][j]==2:
                can_center=origin+numpy.array([UNIT*(j-1),UNIT*(i-1)])
                can=canvas.create_rectangle(can_center[0]-5,can_center[1]-5,can_center[0]+25,can_center[1]+25,fill='black')

    return grid, loc_R	

def analyze_state(robby_loc,grid):
    base3=numpy.array([81, 27, 9, 3, 1])
    x,y=robby_loc
    sensor_data=[grid[x-1,y],grid[x+1,y],grid[x,y+1],grid[x,y-1],grid[x,y]]
    state=numpy.sum(base3*sensor_data)
    return state

def select_action(qmatrix,state,epsilon):
    best_action = numpy.random.choice(numpy.nonzero(qmatrix[int(state),:]==numpy.amax(qmatrix[int(state),:]))[0], 1)
    action = numpy.random.choice([best_action,numpy.random.choice(5, 1)[0]],1,p=[(1-epsilon),epsilon]) 
    return action

def commit_action(state,action,gamma,eta,step_cost,relocate,grid,loc_R,qmatrix,mode,t2):
    #time.sleep(0.1)
    if action==4:
        blank_center=origin+numpy.array([UNIT*(loc_R[0]-1),UNIT*(loc_R[1]-1)])
        blank=canvas.create_rectangle(blank_center[0]-5,blank_center[1]-5,blank_center[0]+25,blank_center[1]+25,fill='white')
        #print "Can Location: ",loc_R,"Blank center: ",(blank_center-20)/60
        canvas.move(t2,0,0)
        canvas.update_idletasks()
        r=reward[int(grid[loc_R])] + step_cost
	grid[loc_R] = 1
	if relocate==True and r==-1:
	    global dry_spell
	    dry_spell += 1
	    if dry_spell>=20:
	        loc_R = tuple(list((numpy.random.choice(numpy.arange(1, 11), 2))))
		dry_spell = 0 
	new_state = analyze_state(loc_R, grid)
	if mode=='train':
	    qmatrix[int(state),action[0]]+=eta*(r+gamma*(numpy.amax(qmatrix[int(new_state),:]))-qmatrix[int(state),action[0]])
	    state = new_state
    else:
        if action==0:
            new_loc = tuple(list(numpy.asarray(loc_R)+numpy.array([-1, 0])))
            if loc_R[0]>1:
                canvas.move(t2,-UNIT,0)
            canvas.update_idletasks()
        elif action==1:
            new_loc = tuple(list(numpy.asarray(loc_R)+numpy.array([1, 0])))
            if loc_R[0]<10:
                canvas.move(t2,UNIT,0)
            canvas.update_idletasks()
        elif action==2:
            new_loc = tuple(list(numpy.asarray(loc_R)+numpy.array([0, 1])))
            if loc_R[1]<10:
                canvas.move(t2,0,UNIT)
            canvas.update_idletasks()
        elif action==3:
            new_loc = tuple(list(numpy.asarray(loc_R)+numpy.array([0, -1])))
            if loc_R[1]>1:
                canvas.move(t2,0,-UNIT)
            canvas.update_idletasks()
        if grid[new_loc]==0:
            r=reward[int(grid[new_loc])]+step_cost 
            if mode=='train':
                qmatrix[int(state),action[0]]+=eta*(r+gamma*(numpy.amax(qmatrix[int(state),:]))-qmatrix[int(state),action[0]])
        else:
            r=step_cost
            loc_R=new_loc
            new_state=analyze_state(loc_R,grid)
            if mode=='train':
                qmatrix[int(state),action[0]]+=eta*(r+gamma*(numpy.amax(qmatrix[int(new_state),:]))-qmatrix[int(state),action[0]])
            state=new_state
    #print "Location: ",loc_R,"Action: ",action,"T2 Coords: ",canvas.coords(t2)
    return state,grid,loc_R,qmatrix,r

def plot_rewards(episode_reward_list, num_episodes, exp):
    pyplot.figure(figsize=(12, 7.5), dpi=100)
    pyplot.plot(num_episodes, episode_reward_list, color='blue', label='Sum of Rewards per Episode' ,lw=2)
    pyplot.xlabel('\nEpisodes\n', size=16)
    pyplot.ylabel('\nSum of Rewards\n', size=16)
    pyplot.title('Q-Learning: Training Robby-The Robot\n', size=18)
    pyplot.legend(loc='lower right')
    pyplot.savefig(exp)

def qlearn(qmatrix,mode,exp='Exp',gamma=0.9,eta=0.2,l_mode='const',step_cost=0,relocate=False,epsilon=1,e_mode='var',style='grid'):
    global no_of_episodes
    episode_reward_array=[]
    for episode in range(no_of_episodes):
        sys.stdout.write("\rEpisode: %d"%episode)
        sys.stdout.flush()
        terminator=itk.PhotoImage(Image.open("/home/manas/Projects/Namratha/MachineLearning/terminator.gif"))
        t2=canvas.create_image(30,30,image=terminator)
        
        if episode%50==0 and episode>0:
            if epsilon>0.1 and e_mode=='var':
                epsilon-=0.01
            if eta>0.1 and l_mode=='decay':
                eta-=0.04
        grid,robby_loc=initialize_grid()
        #print "Initial Location: ",robby_loc
        canvas.move(t2,UNIT*(robby_loc[0]-1),UNIT*(robby_loc[1]-1))
        episode_reward=0
        for step in range(no_of_steps):
            #print "Step: ",step
            current_state=analyze_state(robby_loc,grid)
            action=select_action(qmatrix,current_state,epsilon)
            current_state,grid,robby_loc,qmatrix,reward=commit_action(current_state,action,gamma,eta,step_cost,relocate,grid,robby_loc,qmatrix,"train",t2)
            episode_reward+=reward
        
	if mode is 'test'or(mode is 'train'and(episode%100==0 or episode==no_of_episodes-1)):
	    episode_reward_array.append(episode_reward) 
        
        canvas.delete("all")
    #canvas.delete(ALL)

    if mode is 'train':
        num_episodes=numpy.concatenate((numpy.arange(0,no_of_episodes,100),[no_of_episodes-1]),axis=0)
	plot_rewards(episode_reward_array,num_episodes,exp)
    elif mode is 'test':
        print "\tTest Average: ",numpy.mean(episode_reward_array),"\tTest Standard Deviation: ",numpy.std(episode_reward_array)
    

    return qmatrix

def main():
    #Experiment 1
    print "Experiment 1"
    qmatrix=numpy.zeros((3**5,5))
    qmatrix=qlearn(qmatrix,mode='train',exp='Exp_1.png')
    qlearn(qmatrix,mode='test',epsilon=0.1,e_mode='const')

    #Experiment 2
    #print "Experiment 2"


    #Experiment 3
    #print "Experiment 3"


    #Experiment 4
    #print "Experiment 4"


    #Experiment 5
    #print "Experiment 5"

#if __name__ == "__main__":
#    canvas.pack()
#    canvas.after(0,main)
#    canvas.mainloop()

canvas.pack()
canvas.after(0,main)
canvas.mainloop()


