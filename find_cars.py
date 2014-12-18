'''
load_images.py

convert an mp4 video into PNG images with ffmpeg

'''

import cv2
import ctypes as C
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
import cPickle
import pdb


def movie_to_train_images(file_path, pkl_file, display_flag=0):
    '''
    given a movie, find contours of moving objects and pickle them onto pkl_file

    display_flag:   boolean
                    if set will show the movie 
    '''

    # open file for pickling
    fid = open(pkl_file, 'wb')

    # Create the capture object
    vc = cv2.VideoCapture(file_path)    #vc: Video Capture

    framesN = int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    FPS = vc.get(cv2.cv.CV_CAP_PROP_FPS)

    history=100
    print('file has {0} frames in total'.format(framesN))
    print('movie is recorded at {0} FPS'.format(FPS))
    
    # create background Subtractor object
    bs = cv2.BackgroundSubtractorMOG()

    # generate masks to define ROI (masks cover everything that is not ROI)
    left_mask = np.array([[0,0],[350,0],[0,450],[0,0]])
    right_mask = np.array([[800,0],[1280,0],[1280,450],[800,0]])

    #for i in range(0,13,6):
    for i in range(0,framesN,1):
        #if i%100==0:
        #    print("working on frame: {0}".format(i))
        
        # set vc to work on frame at index i
        #vc.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, i)

        # Read images from vc and add them to bs
        retval, frame = vc.read()

        fgmask = bs.apply(frame, learningRate = 1.0/history)        # fgmask is binary

        # pain pixels to black outside ROI in fgmask
        cv2.fillConvexPoly(fgmask, left_mask, 0)
        cv2.fillConvexPoly(fgmask, right_mask, 0)

        #contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #contours = [contour for contour in contours if accepted_contour(contour, upper=200)]

        # change contours to bounding box
        #boxes = [contour_to_square_box(contour) for contour in contours]

        boxes=[]
        # save each box to output_path
        for l,b,w,h in boxes:
            cPickle.dump((frame[b:b+h, l:l+w, :], -1), fid, protocol=cPickle.HIGHEST_PROTOCOL)

            #cv2.imwrite(path.join("img_{0}.png".format(str(next_image).zfill(6))), frame[b:b+h, l:l+w, :])
            #next_image += 1

        if display_flag:
            #pdb.set_trace()
#            cv2.drawContours(frame, boxes, -1, (0,255,0), 3)
            cv2.imshow('frame', frame)
            cv2.imshow('fgmask', fgmask)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    vc.release()
    cv2.destroyAllWindows()
    fid.close()

def find_cars():
    # Create the capture object
    file_path = 'VID00057.MP4'
    vc = cv2.VideoCapture(file_path)    #vc: Video Capture

    framesN = int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    FPS = np.round(vc.get(cv2.cv.CV_CAP_PROP_FPS))
    print('file has {0} frames in total'.format(framesN))
    print('movie is recorded at {0} FPS'.format(FPS))
    
    frame_step = int(FPS/60)  # I want 10 frames per second. If movie at 60Hz then process one out of 6 frames
    history = 200

    min_area=900

    # create background Subtractor object
    #bs = cv2.BackgroundSubtractorMOG()
    bs = cv2.BackgroundSubtractorMOG2()

    for i in range(1000):
        # Read images from vc and add them to bs
        retval, frame = vc.read()

        if not i%frame_step==0:
            continue
        
        fgmask = bs.apply(frame, learningRate = 1.0/history)        # fgmask is binary

        # initially don't do anything else until we have a decent background model
        if i/frame_step < history/2:
            continue

        fgmask = cv2.medianBlur(fgmask, 15)
        
        retval, fgmask = cv2.threshold(fgmask, 10, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # convert contours to bounding boxes if area > min_area and they don't look like lines
        boxes = [squared_rectangle(cv2.boundingRect(cnt)) for cnt in contours if cv2.contourArea(cnt)>min_area and not isline(cnt)]
        
        cv2.rectangle(frame, (0,0), (300,60), cv2.cv.RGB(0,0,0), thickness=-1)
        cv2.putText(frame, 'Frame: {0}'.format(i), (10,30), cv2.FONT_HERSHEY_PLAIN, 2.0, cv2.cv.RGB(255,0,0), 2)
        cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        drawRectangles(frame, boxes, (255,255,0), thickness=2)
        cv2.imshow('frame', frame)
        #cv2.imshow('fgmask', fgmask)
        #cv2.imshow('fgmaskork', fgmaskOri)
        #cv2.imshow('contours', frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    vc.release()
    cv2.destroyAllWindows()
    

def contour_to_square_box(contour):
    '''
    return the "square" bounding box for contour, keeping the same cm
    '''
    # first get the bounding box
    (l, b, w, h) = cv2.boundingRect(contour)

    if w < h :
        l = l + w/2 - h/2
        w = h
    else:
        b = b + h/2 - w/2
        h = w

    return l, b, w, h


def unpickle(file_path):
    '''
    pickled file has many tuples, each tuple being compossed of a squared image and a label
    '''

    images = []
    labels = []

    fid = open(file_path, 'rb')
    try:
        while 1:
            next_tuple = cPickle.load(fid)
            images.append(next_tuple[0])
            labels.append(next_tuple[1])
    except:
        return images, labels

def accepted_contour(contour,*vargs, **kwargs):
    '''
    Define some logic to accept/reject contours

    flag is set to 1 at the beginning

    each keyword:value in kwargs can switch the flat to 0

    kwards can take the following values:
        'min_y'             keep only contours with mean y value bigger than kwargs['min_y']. 
                            top pixel in image is 0

        'max_y'             keep only contours with mean y value less than kwargs['max_y']. 
                            top pixel in image is 0
        
        'aspect_ratio'  keep only contours with aspect ratio smaller than kwargs['aspect_ratio'] and bigger than 1.0/kwargs['aspect_ratio']

        'min_area'          keep only contours with area > kwargs['min_area']
        'max_area'          keep only contours with area < kwargs['max_area']
    '''

    #pdb.set_trace()
    # Limint contours to only the ROI
    if 'min_y' in kwargs:
        # limit contours to top of image
         if contour[:,0,1].mean()<kwargs['min_y']:
            return 0

    if 'max_y' in kwargs:
        # limit contours to top of image
         if contour[:,0,1].mean()>kwargs['max_y']:
            return 0
    
    if 'aspect_ratio' in kwargs:
        # limit contours to be a given size
        #pdb.set_trace()
        x,y,w,h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        if aspect_ratio > kwargs['aspect_ratio'] or aspect_ratio < 1.0/kwargs['aspect_ratio']:
            return 0

    if 'min_area' in kwargs:
        if cv2.contourArea(contour) < kwargs['min_area']:
            return 0

    if 'max_area' in kwargs:
        if cv2.contourArea(contour) > kwargs['max_area']:
            return 0

    return 1

    # or Limit contours to only those of a given size
    #newContours = [contour for fontour in newContours if cv2.contourArea(contour)>100]

    # or Limit contours according to bounding box dimensions
#        newContours = [contour for contour in newContours if cv2.boundingRect(contour)[2]>10 and cv2.boundingRect(contour)[3]>10]
    #pdb.set_trace()

    #carContours = [contour for contour in newContours if iscar(contour)]
   

def plot_selector(images,n):
    from mpl_toolkits.axes_grid1 import ImageGrid
    
    fig = plt.figure(1, (10.,10.))

    grid = ImageGrid(fig, 111, nrows_ncols = (n, n), axes_pad=0.1)

    for i in range(min(n**2, len(images))):
        grid[i].imshow(images[i+1])


def isline(contour, max_w=-1, max_h=-1, line_thresh = 0.25):
    '''
    compare several contour metrics to decide whether is a line or not

    right now I'm comparing the area of the boundingRect to the contour's area (bounding box area is always bigger). If the ratio is less than line_thresh then the contour is chosen as a line. This algorithm will fail for vertical and horizontal lines (since in those cases the contour area matches the boundingRect area). I'm also setting up two optional parameters (max_w/h) such that if the bounding box is height (or width) less than max_h (max_w) it is still consider a line
    '''
    
    l,b,w,h = cv2.boundingRect(contour)
    
    box_area = w*h
    cnt_area = cv2.contourArea(contour)

    #pdb.set_trace()
    if cnt_area/box_area < line_thresh:
        return 1

    if w < max_w or h < max_h:
        return 1

    return 0

def erode_dilate(image, ker_size):
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(image, kernel)
    return cv2.dilate(eroded, kernel)


def merge_rect_withangles(rect1, rectangles):
    '''
    compare "rect1" with every rectangle in "rectangles". If there is overlap between the two, change the rectangle in "rectangles" to include rect.

    '''


    pdb.set_trace()
    for i in range(len(rectangles)):
        rect2 = rectangles[i]

        # i'm going to check which rectangle is at the left and which one at bototm
        if rect1[0] < rect2[0]:
            left = rect1
            right = rect2
        else:
            left = rect2
            right = rect1

        if rect1[1] < rect2[1]:
            bottom = rect1
            top = rect2
        else:
            bottom = rect2
            top = rect1

        # is there intersection between the two?
        if left[0]+left[2] >= right[0] and bottom[1] + bottom[3] >= top[1]:
            # intersection is True

            # now figure the width and height of the new rectangle
            w = right[2] + right[0] - left[0]
            h = top[3] + top[1] - bottom[1]

            # replace rect2 in position i by newly computed one
            rectangles[i] = (left[0], bottom[1], w, h) 

    return rectangles


def merge_rectangles(rectangles):
    '''
    This is a recursive function to return a set of rectangles with no overlaps.

    if two (or more) rectangles overlap they are replaced by the rectangle that encompaces both and the len(rectangles) decreases by one.
    '''
    # base condition to finish recurssion
    if len(rectangles)==1:
        #pdb.set_trace()
        return rectangles

    # intersectionFlag will be used to know whether rectangles[0] should be returned or not
    intersectionFlag = 0

    # separate the first rectangle, we are going to compare it against all other rectangles 
    rect1 = rectangles[0]

    # loop through all other rectangles and check whether overlaps with rect1 occur
    for i in range(1, len(rectangles)):
        rect2 = rectangles[i]

        # i'm going to check which rectangle is at the left and which one at bototm
        if rect1[0] < rect2[0]:
            left = rect1
            right = rect2
        else:
            left = rect2
            right = rect1

        if rect1[1] < rect2[1]:
            bottom = rect1
            top = rect2
        else:
            bottom = rect2
            top = rect1

        # is there intersection between the two?
        if left[0]+left[2] >= right[0] and bottom[1] + bottom[3] >= top[1]:
            intersectionFlag = True

            # now figure the width and height of the new rectangle
            w = max(right[2] + right[0] - left[0], left[2])
            h = max(top[3] + top[1] - bottom[1], bottom[3])

            # replace rect2 in position i by newly computed one
            rectangles[i] = (left[0], bottom[1], w, h) 
            #print(left[0], left[1], left[0]+left[2], left[1]+left[3])
            #print(right[0], right[1], right[0]+right[2], right[1]+right[3])
            #print(rectangles[i][0], rectangles[i][1], rectangles[i][0]+rectangles[i][2], rectangles[i][1]+rectangles[i][3])
            #pdb.set_trace()
            assert rectangles[i][0]==min(left[0],right[0])
            assert rectangles[i][1]==min(left[1],right[1])
            assert rectangles[i][0]+rectangles[i][2]==max(left[0]+left[2],right[0]+right[2])
            assert rectangles[i][1]+rectangles[i][3]==max(left[1]+left[3],right[1]+right[3])

    if intersectionFlag:
        # don't return rect1, it was merged in other rectangles
        return merge_rectangles(rectangles[1:])
    else:
        return [rect1] + merge_rectangles(rectangles[1:])

def drawRectangles(image, rectangles, color, **kwargs):
    #pdb.set_trace()

    for rect in rectangles:
        p0 = rect[:2]
        p1 = (rect[0]+rect[2], rect[1]+rect[3])

        cv2.rectangle(image, p0, p1, color, **kwargs)


def squared_rectangle(rect):
    '''
    change rectangles to be squared keeping the same cm
    '''
    l,b,w,h = rect
    if rect[2] > rect[3]:
        b -= (w - h)/2
        h = w
    else:
        l -= (h - w)/2
        w = h

    return (l,b,w,h)
