# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 18:51:07 2017

@author: Nick Areias, Jeffrey Paquette
"""

import numpy
import random
import scipy

import filters

class SOMNeuron():
    """A single neuron in a SOM"""
    w = numpy.array([])       # weights
    n = numpy.array([])       # neighbors
    wins = 0                  # number of wins since start of competition cycle
    neighborhood = 1.0        # current epoch neighborhood scalar
    
    def __init__(self, weights):
        """Instantiate a neuron with initial weight values
        
        Keyword arguments:
        weights -- an array of initial weight values for this neuron
        """
        self.w = weights
        
    def round_weights(self):
        """round weight values"""
        numpy.round(self.w)
        
        
class UnchainedSOM():
    """Handles running, managing, and monitoring an unchained SOM for 2D image 
       analysis
    """
    #image = []              # blur-> grayscale -> canny edge detection
    #neurons = []            # list of neurons in map
    #win_limit = 1           # number of wins allowed per competition cycle
    #nu_not = 0.01           # initial learning rate
    #tau_nu = 10000          # learning decay rate
    #epochs = 0              # number of epochs
    #snapshot = 0            # epoch interval at which to show the current state
    #contrast = 0            # constrast of snapshot image
    #neuron_size = 1         # the size of each displayed neuron during a snapshot
    image_count = 0         # index of snapshot image saved
    path = 'sompics/pic_'   # the path of saving the image
    
    def __init__(self, image, learnrate, ldecay, winlimit, epochs, snapshot, contrast, neuron_size):
        self.image = image
        self.nu_not = learnrate
        self.tau_nu = ldecay
        self.win_limit = winlimit
        self.epochs = epochs
        self.snapshot = snapshot
        self.contrast = contrast
        self.neuron_size = neuron_size
        return    
    
    def init_neurons_random(self, number):
        """Initialize a specified number of neurons with random weights.
        
        Keyword arguments:
        number -- the number of neurons to generate
        """
        self.neurons = [SOMNeuron([random.randint(0,len(self.image[0])), random.randint(0, len(self.image))]) for q in range(number)]
        return
                        
    def init_neurons_grid(self, xgrid, ygrid):
        """Initialize a grid of neurons with evenly distributed weight values
        
        Keyword arguments:
        xgrid -- the number of neurons in a row
        ygrid -- the number of neurons in a column
        """
        yoffset = len(self.image) / (ygrid+1)
        xoffset = len(self.image[0]) / (xgrid+1)
    
        for i in range(xgrid):
            for j in range(ygrid):
                xi = (i+1) * numpy.floor(xoffset)
                yj = (j+1) * numpy.floor(yoffset)
                self.neurons.append(SOMNeuron([xi,yj]))
        return
                
    def start(self):
        """Start the SOM learning process"""
        # show initial image
        highlighted_image = self.show_highlighted_image()
        
        # iterate through all epochs
        for t in range(1, self.epochs):
            self.epoch(t)
        
            #check a snapshot of the image and neurons every few iterations
            if(t % self.snapshot == 0):
                highlighted_image = self.show_highlighted_image()
           
        highlighted_image = self.highlight_neurons()
        
        return highlighted_image
    
    def crop_image_at_vertical_outliers(self):
        """Returns cropped image coordinates at the highest and lowest neuron 
           positions ([top, bottom]).
        """
        top_neuron = None
        bottom_neuron = None
        
        # cycle through neurons to find highest and lowest
        for n in self.neurons:
            if (top_neuron is None):
                top_neuron = n
            if (bottom_neuron is None):
                bottom_neuron = n
            if (n.w[1] < top_neuron.w[1]):
                top_neuron = n
            if (n.w[1] > bottom_neuron.w[1]):
                bottom_neuron = n
                
        # crop image at top and bottom neuron positions
        cropped_image = [top_neuron.w[1], bottom_neuron.w[1]]

        # cropped_image = self.image[top_neuron.w[1]:bottom_neuron.w[1]][0:len(self.image[0])]

        return cropped_image
        
    def crop_image_at_vertical_averages(self):
        """Returns cropped image coordinates at the average y position of the 
           top and bottom half of neurons ([top, bottom]).
        """
        top_neurons = []
        bottom_neurons = []
        
        mid = len(self.image) / 2
        # cycle through neurons and split into two lists
        for n in self.neurons:
            if (n.w[1] < mid):
                top_neurons.append(n)
            elif (n.w[1] > mid):
                bottom_neurons.append(n)
                
        # cycle through top neurons to find average top position
        top = 0
        for n in top_neurons:
            top += n.w[1]
        
        top = top / len(top_neurons)
        
        # cycle through bottom neurons to find average top position
        bottom = 0
        for n in bottom_neurons:
            bottom += n.w[1]
        
        bottom = bottom / len(bottom_neurons)
        
        # crop image at top and bottom average positions
        # cropped_image = self.image[top:bottom][0:len(self.image[0])]
        cropped_image = [top, bottom]

        return cropped_image
        
    def epoch(self, time):
        """One cycle of the learning process
        
        Keyword arguments:
        time -- current epoch count
        """
        # keep trying random data points until a valid data point is found
        done = False
        while (done == False):
    
            #select random pixel from the input
            x = random.randint(5,(len(self.image[0]) - 5))
            y = random.randint(5,(len(self.image) - 5))
            
            # pixel is valid only if it is not background
            if(self.image[y][x] != 0):
                
                # valid pixel was picked
                done = True
                
                # determine winner
                winner = self.find_winner(x, y)
                
                # calculate neighborhood
                self.calculate_neighborhood(x, y, winner, time)
                
                # adjust weights
                self.adjust_weights(x, y, time)
        return
        
    def find_winner(self, x, y):
        """Find winner (closest neuron to the current pixel)
        
        Keyword arguments:
        x -- x location of selected sample input
        y -- y location of selected sample input
        """
        #set min_distance to be max distance
        min_distance = numpy.sqrt(numpy.exp(len(self.image)) + numpy.exp(len(self.image[0])))
        temp_dist = 0
        
        for n in self.neurons:
            if(n.wins < self.win_limit):
                temp_dist += numpy.square(x - n.w[0])
                temp_dist += numpy.square(y - n.w[1])
                temp_dist = numpy.sqrt(temp_dist)
                
                if(temp_dist < min_distance):
                    winner = n
                    min_distance = temp_dist
                        
            temp_dist = 0
         
        # increment win count
        winner.wins += 1
        
        # reset competition cycle if every neuron has won at least once
        reset = 1
        for n in self.neurons:
            if (n.wins == 0):
                reset = 0
                
        if (reset == 1):
            for n in self.neurons:
                n.wins = 0
                
        return winner
    
    def calculate_neighborhood(self, x, y, winner, time):
        """Calculate the neighborhood value of each neuron.
        
        Keyword arguments:
        x -- x location of selected sample input
        y -- y location of selected sample input
        winner -- winning neuron
        time -- current epoch
        """
        for n in self.neurons:
            #excite distance is lateral distance between winning neuron and current neuron.
            #winner coords
            w_x = winner.w[0]
            w_y = winner.w[1]
            #current neuron coords
            c_x = n.w[0]
            c_y = n.w[1]
            
            if(w_x != c_x or w_y != c_y):
                
                #distance between winner and current neuron
                d1 = numpy.sqrt(numpy.square(w_x - c_x) + numpy.square(w_y - c_y))
                
                #distance between winner and selected point
                d2 = numpy.sqrt(numpy.square(w_x - x) + numpy.square(w_y - y))
                
                #distance between current neuron and selected point
                d3 = numpy.sqrt(numpy.square(c_x - x) + numpy.square(c_y - y))
                
                if(d1 == 0 or d2 == 0):
                    n.neighborhood = 0
                else:
                    value = (numpy.square(d1) + numpy.square(d2) - numpy.square(d3)) / (2*d1*d2)
                    
                    #angle next to winner in the triangle between winner, current neuron, and selected point
                    theta = numpy.arccos(numpy.clip(value,-1,1))
                    
                    #lateral distance between winner and current neuron
                    lateral_distance = d1 * numpy.cos(theta)
                    
                    # shrink the neighborhood based on lateral distance and time
                    n.neighborhood = 1/(lateral_distance * time)
            else:
                n.neighborhood = 1
        return
        
    def adjust_weights(self, x, y, time):
        """Adjust weights of all neurons based on learning rate, neighborhood,
           and distance from selected sample point.
        
        Keyword arguments:
        x -- x location of sample input
        y -- y location of sample input
        time -- current epoch
        """
        learning_rate = self.nu_not / numpy.exp(time/self.tau_nu)

        for n in self.neurons:
                
                dx = learning_rate * n.neighborhood * (x - n.w[0])
                dy = learning_rate * n.neighborhood * (y - n.w[1])
                
                n.w[0] += dx
                n.w[1] += dy
        return

    def round_neurons(self):
        """Round the weight values of each neuron"""
        for n in self.neurons:
            n.round_weights()
        return
    
    def trim_neurons_on_black(self):
        """Remove all neurons on black pixels."""
        remove_list = []
        for n in self.neurons:
            if (self.image[n.w[1]][n.w[0]] == 0):
                remove_list.append(n)
        
        for n in remove_list:
            self.neurons.remove(n)
            
        return
        
    def show_highlighted_image(self):
        """Show the image, highlighting the neurons by a specified size.
        
        Keyword arguments:
        size -- the size in pixels of each neuron
        """
        highlighted_image = self.highlight_neurons()
        # filters.draw(highlighted_image)
        scipy.misc.imsave(self.path + str(self.image_count) + '.jpg', highlighted_image)
        self.image_count += 1
        return highlighted_image
    
    def highlight_neurons(self):
        """Highlight all neuron locations by making pixels in a neuron_size 
           radius the max intensity.
        """
        #highlight neurons to show where they are
        intensity = numpy.max(self.image) * self.contrast
    
        highlighted_image = filters.darken(self.image, self.contrast)
        
        self.round_neurons()
        for n in self.neurons:
            #highlight a 3x3 around the neuron
            for p in range(self.neuron_size):
                for q in range(self.neuron_size):
                    if(n.w[0] - 1 + p >= 0 and n.w[0] - 1 + p < len(highlighted_image[0]) and n.w[1] - 1 + q >= 0 and n.w[1] - 1 + q < len(highlighted_image)):
                        highlighted_image[n.w[1] - 1 + q][n.w[0] - 1 + p] = intensity
    
        return highlighted_image