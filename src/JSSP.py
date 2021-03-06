#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import time
import sys
import os


def take_action(s,q_table):
    possible_actions = [i for i in range(len(Job_state)) if Job_state[i] < m_number]
    #Epsilon greedy algorithm to choose action
    if (np.random.uniform() >= EPSILON) or (all(q_value == 0 for q_value in q_table.get(str(s)) )):
        action = np.random.choice(possible_actions) 
    else:
        action = q_table.get(str(s)).index(max(q_table.get(str(s))))
    return action


def pick_one_job(job_pick):     
    machine = sequence[job_pick][Job_state[job_pick]-1] #which machine to process
    if data[job_pick][machine] > 0:
        makespan_each_machine[machine] =max(makespan_each_machine[machine],completedtime[job_pick])+data[job_pick][machine] #update makespan
        #update:
        completedtime[job_pick] = makespan_each_machine[machine]    
    return machine


#data =[[3,2,0],[2,4,4],[2,1,3]] #0,1,2
#sequence =[[0,1,2],[0,2,1],[0,1,2]] #0,1,2

#Import data
link = os.path.join("Data","JSSP_Instance.xlsx")
data = pd.read_excel(link,header=None)

sequence = data.reset_index().iloc[:,1::2].values.tolist()
machines = data.reset_index().iloc[:,2::2].values.tolist()
big_list =[]
l=0
for i in sequence:  
    k=0
    list_ =[0 for number in i]
    for x in i:
        list_[x] = machines[l][k]
        k+=1
    big_list.append(list_)
    l+=1
data = big_list



#Setting parameters
EPSILON=0.5
ALPHA = 0.2
GAMMA = 0.5

#Initialize best makespan with large number
best_makespan=sys.maxsize
j_number = len(data) #Number of jobs to be processed
m_number = len(data[0])
m_sequence = list(range(0,len(data[0]))) # machines sequence from 0 to n

jobs = list(range(0,len(data))) #list of jobs
MAX_EPISODE = j_number*m_number

q_table =dict()

# Start learning
iteration = 0
tic = time.process_time()
best_sequence = 0
for i in range(MAX_EPISODE):
    Job_state =[0 for i in range(j_number)] # [0,0,0] # number of task done of these jobs
    makespan_each_machine = [0 for i in range(m_number)] #[0,0,0] #makespan of each machine
    completedtime = [0 for i in range(j_number) ]#[0,0,0] #time finish of job i
    s = dict()
    for job in range(len(data)):
        s[job] = sequence[job][0]
    sequence__ = []
    while any(job_state <m_number for job_state in Job_state):

        if str(s) not in q_table:
            q_table[str(s)] =  [0 for i in range(j_number)]
        
        #Take action using epsilon greedy algorithm
        action = take_action(s,q_table)
        old_Q = q_table[str(s)][action]
        
        sequence__.append([action,s[action]])
        # move to the next state
        s_new = s.copy()
        Job_state[action]+=1
        if Job_state[action] < m_number:
            s_new[action] = sequence[action][Job_state[action]]
        else: 
            del s_new[action]

        next_state = s_new

        #maximum Q value of next state taking any actions : Q(s',a')
        try:
            maxQ_value = max(q_table[str(next_state)])
        except:
            maxQ_value = 0
        #print("next_state",next_state)

        #Calculate the makespan of recent sequence   
        machine = pick_one_job(action)
      
        v = max(makespan_each_machine)
        if v == 0:
            v=0.00001
        r=1/v
        #Calculate TD and new Q(s,a)
        Temporal_difference = r + GAMMA * maxQ_value - old_Q
        new_Q = old_Q + ALPHA*Temporal_difference

        #Update Q table
        q_table[str(s)][action] = new_Q

        #Move to the next state to end iteration
        s=next_state
    #Choose the best makespan after every episode

    if v<best_makespan:
        best_makespan=v
        best_iteration = iteration
        best_sequence=sequence__
    iteration+=1
toc = time.process_time()
print("Number of episodes",MAX_EPISODE)
print("Best-found makespan",best_makespan)
print("Best-found Sequence", best_sequence)
print("First episode to find the best-found solution",best_iteration)
print("Computational time", toc-tic)
