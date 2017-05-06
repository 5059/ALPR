# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:02:19 2017

@author: Nick Areias, Jeffrey Paquette
"""

import numpy
import scipy.misc
import filters

class Point():
    """Container for a single 2D point"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        return
        
class Rect():
    """A rectangle object that tracks boundaries"""
    #top = 0        # top edge of rect
    #bottom = 0     # bottom edge of rect
    #left = 0       # left edge of rect
    #right = 0      # right edge of rect
    #area = 0       # area of rect
    
    def __init__(self, top, bottom, left, right):
        """Rect init
        
        Keyword arguments:
        top -- the topmost y position of rect
        bottom -- the bottommost y position of rect
        left -- the leftmost x position of rect
        right -- the rightmost x position of rect
        """
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.calculate_area()
        return
    
    def calculate_area(self):
        """Returns the area of a rectangle (min=1)"""
        self.area = (self.bottom - self.top + 1) * (self.right - self.left + 1)
        return
        
    def isAdjacent(self, x, y):
        """Returns true if this point is adjacent to this rect
        
        Keyword arguments:
        x -- point x position
        y -- point y position
        """
        if (x + 1 == self.left and y >= self.top and y <= self.bottom):
            return True
        elif (x - 1 == self.right and y >= self.top and y <= self.bottom):
            return True
        elif (y - 1 == self.bottom and x >= self.left and x <= self.right):
            return True
        elif (y + 1 == self.top and x >= self.left and x <= self.right):
            return True
        else:
            return False
    
    def contains(self, x, y):
        """Returns true if this point is inside this rect
        
        Keyword arguments:
        x -- point x position
        y -- point y position
        """
        if (x >= self.left and x <= self.right and y >= self.top and y <= self.bottom):
            return True
        else:
            return False
        
    def add(self, x, y):
        """Expands the boundaries of this rect to include this point
        
        Keyword arguments:
        x -- point x position
        y -- point y position
        """
        if (x < self.left):
            self.left = x
        elif (x > self.right):
            self.right = x
            
        if (y < self.top):
            self.top = y
        elif (y > self.bottom):
            self.bottom = y
        
        # update area calculation
        self.calculate_area()
        return
                
class Tracer():
    """Finds, analyzes, and either accepts or rejects clusters of like pixels
       in an image and extracts them.
    """
    #rects = []             # list of rectangles
    #img = []               #image to be processed
    #img_area = 0           # area of entire image
    #mid_y = 0              # middle y position of pic
    #width = 0              # width of image
    #height = 0             # height of image
    #visited = []           # array same size as image to mark visited pixels
    take_snapshot = False;  # flag for saving snapshot
    
    def __init__(self, image, area_min, area_max, max_mid_distance):
        """Tracer init
        
        Keyword arguments:
        image -- the image to analyze
        area_min -- the minimum area of a rect in percentage of image size (0-1)
        area_max -- the maximum area of a rect in percentage of image size (0-1) 
        max_mid_distance -- the max dist rect can be from middle row of pixels 
                            in percentage of image size (0-1)
        """
        self.path = 'rectpics/pic_'
        self.image_count = 0
        self.img_state = numpy.array(image)
        
        # list of rectangles
        self.rects = []
        
        # store image in class variable
        self.img = image
        self.width = len(self.img[0])
        self.height = len(self.img)
        
         # total area of the image
        self.img_area = len(image) * len(image[0])
        
        # min and max sizes of rectangles
        self.min_size = area_min * self.img_area
        self.max_size = area_max * self.img_area
        
        # max distance away from middle horizontal line
        self.max_mid_distance = max_mid_distance
        
        self.visited = numpy.zeros_like(self.img)

        # middle y position of image
        self.mid_y = len(image) / 2
        
        # for each row of pixels (exluding 5 pixels on the edges)
        for y in range(10, len(image) - 10):
            # for each column of pixles (excluding 10 pixels on the edges)
            for x in range(10, len(image[0]) - 10):
                    # if the pixel is not black                    
                    if (image[y][x] != 0):
                        # mark pixel as visited
                        if (self.visited[y][x] == 1):
                            continue
                        
                        # always create the first rect
                        if (len(self.rects) == 0):
                            new_rectangle = Rect(y, y, x, x)
                            self.rects.append(new_rectangle)
                            self.trace(new_rectangle, x, y)
                            self.reject_rect(new_rectangle)
                        else:
                            # check if point already belongs to a rect
                            # before creating one
                            in_rect = False
                            for r in self.rects:
                                if (r.contains(x, y)):
                                    in_rect = True
                                    break
                            if (not in_rect):
                                new_rectangle = Rect(y, y, x, x)
                                self.rects.append(new_rectangle)
                                self.trace(new_rectangle, x, y)
                                self.reject_rect(new_rectangle)
        # check for save state
        if (self.take_snapshot):
            self.save_state() 
        return
    
    def trace(self, r, x, y):
        """Traces a single cluster of pixels and returns a rect
        
        Keyword arguments:
        r -- rectangle object to create
        x -- point x position to start trace
        y -- point y position to start trace
        """
        snap = 0 # snap value for save state
        
        to_visit = []   #list of points to visit
        to_visit.append(Point(x, y))
        
        while (len(to_visit) > 0):
            pixel = to_visit.pop()
            self.visited[pixel.y][pixel.x] = 1
                
            if (self.img[pixel.y][pixel.x] != 0):                
                r.add(pixel.x, pixel.y)
                
                snap += 1
                if (snap % 100 == 0):
                    self.save_state()
                    
                if (pixel.x+1 < self.width-10 and self.visited[pixel.y][pixel.x+1] == 0):
                    to_visit.append(Point(pixel.x+1, pixel.y))
                if (pixel.x-1 > 10 and self.visited[pixel.y][pixel.x-1] == 0):
                    to_visit.append(Point(pixel.x-1, pixel.y))
                if (pixel.y+1 < self.height-10 and self.visited[pixel.y+1][pixel.x] == 0):
                    to_visit.append(Point(pixel.x, pixel.y+1))
                if (pixel.y-1 > 10 and self.visited[pixel.y-1][pixel.x] == 0):
                    to_visit.append(Point(pixel.x, pixel.y-1))
        
        # check for save state
        if (self.take_snapshot):
            self.save_state()
        return
    
    def reject_rect(self, rect):
        """Analyze and remove rect from list if neccessary
        
        Keyword arguments:
        rect -- The rectangle to be analyzed
        """
        if (rect.area < self.min_size):
            self.rects.remove(rect)
        elif (rect.area > self.max_size):
            self.rects.remove(rect)
        elif (rect.top - self.mid_y > self.max_mid_distance * self.mid_y):
            self.rects.remove(rect)
        elif (self.mid_y - rect.bottom > self.max_mid_distance * self.mid_y):
            self.rects.remove(rect)
        return
    
    def highlight_rects(self):
        """Creates a new image that darkens background image and draws all 
           existing rectangles in white.
        """
        highlighted_image = filters.darken(numpy.array(self.img), 2)
        for r in self.rects:
            for x in range (r.left, r.right):
                highlighted_image[r.top][x] = 255
                highlighted_image[r.bottom][x] = 255
            for y in range (r.top + 1, r.bottom - 1):
                highlighted_image[y][r.left] = 255
                highlighted_image[y][r.right] = 255
        return highlighted_image
        
    def extract_rects(self, original):
        """Extracts sub images based on rectangle coordinates.
           Returns a sorted list of these images ordered top-down, left-right
        """
        images = []     # list of images cut out from origin
        
        new_rects = []
        for r in self.rects:
            new_rects.append([r.left,r.right,r.top,r.bottom])
            
        #sort rects before slicing
        new_rects = numpy.sort(new_rects, axis=0)
        
        #slice images and place them in the array                    
        for r in new_rects:
            image = numpy.array(original[r[2]:r[3], r[0]:r[1]])
            images.append(image)
        return images
        
    def save_state(self):
        """Saves an image to the path specificed containing a darkened image
           with rectangles and their contents highlighted
        """
        state = filters.darken(numpy.array(self.img), 2)
        for r in self.rects:
            for x in range (r.left, r.right):
                state[r.top][x] = 255
                state[r.bottom][x] = 255
            for y in range (r.top + 1, r.bottom - 1):
                state[y][r.left] = 255
                state[y][r.right] = 255
            for x in range (r.left, r.right):
                for y in range (r.top+1, r.bottom - 1):
                    if (self.img[y][x] != 0):
                        state[y][x] = 255
            
        scipy.misc.imsave(self.path + str(self.image_count) + '.jpg', state)
        self.image_count += 1
        return
        