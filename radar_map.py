# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import *
import matplotlib.image as mpimg
from PIL import Image


def decay_function(r, r_max, style):

    
    if style == 'none':
        if r <= r_max:
            return 1
        else:
            return 0
    
    elif style == 'linear':
    # Linear Decay from strength 1 at radius 1 to
    # stregth 0 and radius r_max
        
        m = 1/(1 - r_max)
        return m*(r - r_max)
        
    # Exponential Decay from strength 1 at radius 1
    elif style == 'exp':
        
        const = -0.05
        
        return exp(const*(r-1))
    


def draw_circle(centre, r_max, resolution, vision, topo, theta, decay):
    
    # theta is the angle of the radar
    # resolution is how many angle iterations you have
    
    # radar on the ground
    
    
    plot_height, plot_width = test.shape
    
    for i in range(0, resolution + 1):
        
        
        current_min_strength = 1
        
        for r in range(1, r_max + 1):
            
            # within this radial direction track the strengths and
            # use the current lowest
            
            
            
            x = floor(centre[0] + r*cos(2*pi*i/resolution))
            y = floor(centre[1] + r*sin(2*pi*i/resolution))
            
            if y > plot_height - 1 or y < 0:
                break
            if x > plot_width - 1 or x < 0:
                break
            
            # MOUNTAINS REDUCE SIGNAL 
            current_height = topo[(y,x)]
            ratio = current_height/(r*tan(theta))
            strength = max(1 - ratio, 0)
            current_min_strength = min(strength, current_min_strength)
            
            # power = strength x decay function
            power = current_min_strength*decay_function(r, r_max, decay)
            
        
            vision[(y,x)] = max(power, vision[(y,x)])
            
    return vision 
            
    
    
def field_of_vision(centres, r_maxs, topo, theta, decay, resolution):
    
    
    vision = np.zeros(topo.shape)
    
    for centre, r_max in zip(centres, r_maxs):
        vision = draw_circle(centre, r_max, resolution, vision, topo, theta*pi/180, decay)
    
    

    print(vision)
    
    
    # Two subplots, unpack the axes array immediately
    f1, ax1 = plt.subplots(1, 1, sharey=True)
    plot1 = ax1.imshow(vision, interpolation = 'none')
    ax1.set_title('Radar')
    col_bar1 = f1.colorbar(plot1)
    
    f2, ax2 = plt.subplots(1, 1, sharey=True)
    plot2 = ax2.imshow(topo, interpolation = 'none')
    ax2.set_title('Mountain')
    col_bar2 = f2.colorbar(plot2)
    

    return vision
    
# main

#resolution = 400
#
#image = Image.open("topography_1.png").convert("L")
#arr = np.asarray(image)
#arr = (255 - arr)*(10/255)
#
#field_of_vision([(50,50)], [20], arr, 30, 'linear', resolution)


#test = np.zeros((100, 100))
#test[70:80, 70:80] = 5
#field_of_vision([(68,68)], [20], test, 45, 'none', resolution)
#    
