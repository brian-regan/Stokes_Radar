# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from bresenham import bresenham
from math import *

    
#def decay_function(r, r_max):
#    m = 1/(1 - r_max)
#    return m*(r - r_max)
#    

def decay_function(r, r_max):
    const = -0.05
    
    return exp(const*(r-1))
    
    
def draw_circle(centre, r_max, resolution, vision, topo):
    
    for i in range(0, resolution + 1):
        
        for r in range(1, r_max + 1):
            x = floor(centre[0] + r*cos(2*pi*i/resolution))
            y = floor(centre[1] + r*sin(2*pi*i/resolution))
            if topo[(x,y)] == 1:
                # THERE IS A MOUNTAIN!!
                break
            vision[(x,y)] = max(decay_function(r, r_max), vision[(x,y)])
    
    return vision 
            
    
    
def field_of_vision(centres, r_maxs, topo):
    
    resolution = 400
    
    vision = np.zeros(topo.shape)
    
    for centre, r_max in zip(centres, r_maxs):
        vision = draw_circle(centre, r_max, resolution, vision, topo)
    
    

    print(vision)
    
    plt.imshow(vision, interpolation = 'none')
        
    



# draws plot
#print(topo)
#plt.imshow(topo, interpolation = 'none')
#plt.show()