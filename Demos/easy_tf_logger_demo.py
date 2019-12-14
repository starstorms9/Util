#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 20:36:14 2019

@author: starstorms
"""

import easy_tf_log as etl

scale = 1.7

etl.set_dir('logs/1')

for i in range(10):
    etl.tflog('foo', i*scale)
for j in range(10, 20):
    etl.tflog('bar', j*scale)

etl.set_dir('logs/2')

for k in range(20, 30):
    etl.tflog('foo', k*scale)
for l in range(5):
    etl.tflog('bar', l*scale, step=(10 * l))
    
etl.set_dir('logs/3 extra stuff')

for k in range(20, 40):
    etl.tflog('foo', k*scale)
for l in range(15):
    etl.tflog('bar', l*scale)
    
etl.set_dir('logs/4 moar stuff')

for k in range(20, 45):
    multlog(['foo', 'bar'], [k*scale, k/scale])


#%% Multilog

def multlog(keys, values):
    if len(keys)==len(values) | len(keys)<=0 : return
    for i in range(len(keys)) :
        etl.tflog(keys[i], values[i])