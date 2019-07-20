#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:54:01 2019

@author: menghao
- code blocks about os operation
- heu building images preprocessing script
- don't run the whole programme.
- only run the relevant block if it is needed
"""
import os

'''----------Hyperparameters----------'''
#current_path = os.path.abspath('preprocessing.py')
root = os.getcwd()
path1 = os.path.join('0_data/1/')
path2 = os.path.join('0_data/2/')

'''----------delete files----------'''
for file in os.listdir(root):
    if '.jpg' in file:
        os.remove(os.path.join(root,file))
        
'''----------file rename----------'''
'''1'''
for filename in os.listdir(path1):
    num = filename.rfind(' ')
    new_name = filename[num+1:]   
    os.rename(path1 + filename, path1 + '1A' + new_name)

for filename in os.listdir(path2):
    num = filename.rfind(' ')
    new_name = filename[num+1:]    
    os.rename(path2 + filename, path2 + '1B' + new_name)

'''2'''
    
 
'''----------assign label to filename----------'''
# 1A/ 13-54, 150-240, 350-356 are nothing, 
# 1A/ 55-120, 241-345 are dormitory building
# 1A/ 61-150   are mainbuilding front

# 1B/ all pics are 1B
# labels:
# C: nothing; A: mainbuilding front; B: mainbuilding back; D: dormitory

 
for i in range (1,400):
    for filename in os.listdir(path1):
        if (i >= 346 and i <= 349):
            if filename.endswith('0' + str(i) + '.jpg'):
                new_name = filename[2:]
                os.rename(path1 + filename, path1 + 'C' + '0' + new_name)
        if (i >= 61 and i <= 120):
            if filename.endswith('0' + str(i) + '.jpg'):
                new_name = filename[2:]
                os.rename(path1 + filename, path1 + 'AD' + new_name)
        if (i>=13 and i<=54) or (i>=150 and i<=240) or (i>=350 and i<=356): # nothing
            if filename.endswith('0' + str(i) + '.jpg'):
                new_name = filename[2:]
                os.rename(path1 + filename, path1 + 'C' + '0' + new_name)
        if (i>=55 and i<=120) or (i>=241 and i<=345): # dormitory
            if filename.endswith('0' + str(i) + '.jpg'):
                new_name = filename[2:]
                os.rename(path1 + filename, path1 + 'D' + '0' + new_name)
        if (i>120 and i<150):
            if filename.endswith('0' + str(i) + '.jpg'):
                new_name = filename[2:]
                os.rename(path1 + filename, path1 + 'A' + '0' + new_name)

for i in range (740, 890):
    for filename in os.listdir(path2):
        new_name = filename[2:]
        os.rename(path2 + filename, path2 + 'B0' + new_name)
            
            

















