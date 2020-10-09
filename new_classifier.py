#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 11:50:35 2020

@author: ryan
"""

from matplotlib import pyplot as plot
import numpy as np
import scipy
import itertools

class Box:
    def __init__(self, xi, yi, xf, yf, contents):
        self.xi = xi
        self.yi = yi
        self.xf = xf
        self.yf = yf
        self.contents = contents
    def set_parent(self, obj) :
        self.parent = obj
    def get_bounds(self) :
        return self.xi, self.yi, self.xf, self.yf
    def get_relative_bounds(self) :
        xpi,ypi,xpf,ypf = self.parent.get_bounds()
        return self.xi+xpi, self.yi+ypi, self.xf+xpf, self.yf+ypf
    def get_contents(self) :
        return self.contents
    def get_parent(self) :
        return self.parent
    def get_prediction(self, model, threshold):
        #arrin = rescale_factor(self.contents, 0.8)
        
        array = self.contents
        
        #array = rescale_factor(array, scale_factor)
        
        #array = pad_to_fit(array, (20,20), 255)
        
        #print(arrin.shape, "shape1")
        #arrin = rescale_img(self.contents, (18,18))
        #iarr = pad_to_fit(arrin, (20,20), 255)
        
        arrin = rescale_img(array, (20,20))
        #arrin = array
        arrin = contrast(arrin, 50)

        #arrin = cont(arrin)
        #arrin = pad_array(arrin, val=255, exp=1)

        #trim_bkg(self, get_background_val(self.contents), 100)
        #arrin = sharpen(arrin, False)#tolerance)
        #arrin = np.asarray(arrin, dtype='uint8')
        
        #arrin = np.asarray(arrin, dtype='float32')
        #arrin /= 255
        arrin = invert(arrin)
        
        #return ['0'], [[[0]]], arrin
       # arrin = scipy.ndimage.filters.gaussian_filter(arrin, sigma=0.5)
        #arrin = reframe(arrin)
        #arrin = trim_arr(arrin,0,0)
        #arrin = sharpen(arrin, False)
        #arrin *= 255
        #arrin = cont(arrin)
        p = predict_class_arr(arrin,model)
        i = 0
        while(p[1] < threshold) :
            i += 1
            noise = np.random.normal(127,40,(20,20))
            arrin *= noise
            arrin /= 255
            print(i)
            p = predict_class_arr(arrin,model)
            
        return p[0], p[1], arrin, (i + 1)
    
    
    def set_contents(self, arr) :
        self.contents = arr
    def set_bounds(self, xfs, yfs) :
        self.xf = xfs
        self.yf = yfs

def invert(img) :
    nimg = img
    nimg = nimg.astype(dtype='uint8')
    nimg = np.invert(nimg)
    nimg = nimg.astype(dtype='float32')
    nimg /= 255
    return nimg

def pad_array(array, val, exp) :
    return np.pad(array, pad_width=exp, mode='constant',constant_values=[val])

def fit_to_max(array, maxdim) :
    x = array.shape[0]; y = array.shape[1]
    if x < y :
        arr = rescale_factor(array, maxdim/y)
    else :
        arr = rescale_factor(array, maxdim/x)
    return arr
        
def pad_to_fit(array, dim, val) :
    import math
    h = math.ceil((dim[1]-array.shape[1])/2); v = math.ceil((dim[0]-array.shape[0])/2)
    if v > 0 and h > 0 :
        return np.pad(array, ((h,h),(v,v)), mode='constant', constant_values=[val]);
    else :
        return None
 
def sharpen(img, isRandom) :
    import math
    import random
    exp = 0.05
    shift = 0
    arr = img.astype(dtype='float32')
    arr /= 255
    if isRandom :
        exp = random.random()/2
        shift = random.randint(0,50)
    sigma = lambda x : round((1/(1+(math.e**(-x*(exp) + (shift))))),3)
    rfunction = np.vectorize(sigma)
    arr = np.asarray(rfunction(img))
    arr *= 255
    arr = arr.astype(dtype='uint8')
    #arr *= 255
    return arr

def reframe(array) :
    print(array)

def cont(arr) :
    import math
    import random
    exp = 0.5
    shift = 100
    sigma = lambda x : round((1/(1+(math.e**(-x*(exp) + (shift))))),3)
    rfunction = np.vectorize(sigma)
    arr = np.asarray(rfunction(arr))
    arr *= 255
    return arr

def contrast(imge,pivot) :
    for r in range(imge.shape[0]) :
        for c in range(imge.shape[1]) :
            imge[r,c] -= (pivot - imge[r,c])
            if imge[r,c] > 255 :
                imge[r,c] = 255
            if imge[r,c] < 0 :
                imge[r,c] = 0
    return imge

def predict_class_arr(imw, model) :
    imagen = imw#(invert(imw))
    imagen = np.reshape(imagen, (1,20,20,1))
    p = model.predict(imagen, batch_size=1, verbose=0)
    p = np.asarray(p, dtype='float32')
    prob = 0
    out = list()
    for i in range(len(p)) :
        out.append(chr(33+np.argmax(p[i])))
        prob = p[0,np.argmax(p[i])]
        #out.append(chr(33+np.nonzero(p[i])[0]))
    return out, prob

def create_scaled_image(loc, dim) :
    
    import cv2
    from skimage.io import imread
    img = imread(loc, as_gray = True)
    if dim != None :
        img = cv2.resize(img, dim)
    img *= 255
    return img.astype('uint8')

def get_background_val(arr) :
    (values,counts) = np.unique(arr,return_counts=True)
    ind=np.argmax(counts)
    return values[ind]
    
def get_horizontal_divisions(arr) :
    horz = list()
    for col in range(arr.shape[0]) :
        if np.all(arr[col,:]) :
            horz.append(col)
            #arr[col,:] = 0
    minimum = 0
    for i in range(len(horz)-1) :
        if(horz[i+1]-horz[i] > minimum) :
            minimum = horz[i+1]-horz[i]
    minimum /= 5
    hoz = list()
    for i in range(len(horz)-1) :
        if(horz[i+1]-horz[i] > minimum) :
            hoz.append(horz[i])
            hoz.append(horz[i+1])
            #i -= 1
    hoz.append(0); hoz.append(arr.shape[0]-1)
    hoz.sort()
    return hoz

def get_vertical_divisions(arr) :
    verticals = list()
    for col in range(arr.shape[1]) :
        if np.all(arr[:,col]) :
            verticals.append(col)
            #arr[:,col] = 0
    minimum = 0
    for i in range(len(verticals)-1) :
        if(verticals[i+1]-verticals[i] > minimum) :
            minimum = verticals[i+1]-verticals[i]
    minimum /= 5
    verts = list()
    for i in range(len(verticals)-1) :
        if(verticals[i+1]-verticals[i] > minimum) :
            verts.append(verticals[i])
            verts.append(verticals[i+1])
            #i -= 1
    verts.append(0); verts.append(arr.shape[1]-1)
    verts.sort()
    return verts

def trim_bkg(box,bkgval,tol) :
    arr = box.get_contents()
    try :
        hdim = np.trim_zeros(arr[0,:])
    except :
        hdim = arr[0,:]
    try :
        vdim = np.trim_zeros(arr[:,0])
    except :
        vdim = arr[:,0]
    hdim = np.reshape(hdim,(hdim.shape[0],1))
    vdim = np.reshape(vdim,(1,vdim.shape[0]))
    print(hdim,vdim)
    arr = arr[:,~np.all((abs(arr - bkgval) < tol), axis=0)]
    arr = arr[~np.all((abs(arr - bkgval) < tol), axis=1)]
    xfi = arr.shape[0]; yfi = arr.shape[1]
    box.set_bounds(xfi,yfi)
    box.set_contents(arr)
    
def trim_arr(array,bkgval,tol) :
    arr = array
    try :
        hdim = np.trim_zeros(arr[0,:])
    except :
        hdim = arr[0,:]
    try :
        vdim = np.trim_zeros(arr[:,0])
    except :
        vdim = arr[:,0]
    hdim = np.reshape(hdim,(hdim.shape[0],1))
    vdim = np.reshape(vdim,(1,vdim.shape[0]))
    arr = arr[:,~np.all((abs(arr - bkgval) < tol), axis=0)]
    arr = arr[~np.all((abs(arr - bkgval) < tol), axis=1)]
    return arr

def rescale_img(img, dim) :
    import cv2
    newimg = cv2.resize(img, dim)
    return newimg

def rescale_factor(img, factor) :
    import cv2, math
    x = math.ceil(img.shape[0]*factor); y = math.ceil(img.shape[1]*factor);
    newimg = cv2.resize(img, (x,y))
    return newimg

def combine(arr1, arr2) :
    '''
    width = arr1.shape[0] + arr2.shape[0]
    height = arr1.shape[1]
    arr3 = np.empty((width, height), dtype='float32')
    
    ro = arr1.shape[0]
    co = arr1.shape[1]
    
    for r in range(ro) :
        for c in range(co) :
            arr3[r][c] = arr1[r][c]
    for r in range(arr2.shape[0]) :
        for c in range(arr2.shape[1]) :
            arr3[r+ro][c+co] = arr2[r][c]
    '''
    try :
        return np.concatenate((arr1, arr2), axis=1)
    except :
        return np.empty((arr1.shape[0], arr1.shape[1]))
    

def load_model_from_JSON(modelpath, weightpath) :
    from keras.models import model_from_json
    json_file = open(modelpath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weightpath)
    return loaded_model

def get_emnist_image(pred) :
    from emnist import extract_training_samples
    import pandas as pd
    x, y = extract_training_samples('balanced')
    #find inverse of: chr(33+np.argmax(p[i]))
    csv = pd.io.parsers.read_csv('emnist-balanced-mapping.csv')
    print(pred)
    for i in range(len(x)) :
        if chr(csv['Out'][y[i]]) == pred :
            return x[i]
    return x[0]

def superimpose_pred(box, array, pred) :
    
    xi, yi, xf, yf = box.get_relative_bounds()
    
    bounds = box.get_bounds()
    
    width = bounds[2] - bounds[0]; height = bounds[3] - bounds[1]
    
    print(width, height)
    
    scale = 1.2
    
    overlay = get_emnist_image(pred)
    overlay = rescale_img(overlay, (int(height*scale),int(width*scale)))
    print(overlay.shape)
    overlay = overlay.astype(dtype='float32')
    overlay = scipy.ndimage.filters.gaussian_filter(overlay, sigma=0.5)
    overlay = invert(overlay)
    overlay = overlay * 255
    overlay = overlay.astype(dtype='uint8')
    newarr = array
    for x in range(overlay.shape[0]) :
        for y in range(overlay.shape[1]) :
            newarr[x+xi,y+yi] = overlay[x,y]
    return newarr
    
def get_boxes(arr) :
    import itertools
    
    arr = cont(arr)  
    
    verticals = get_vertical_divisions(arr)
    horizontals = get_horizontal_divisions(arr)
    bkg = get_background_val(arr)
    min_val = np.min(arr)
    
    boxes = list()
    
    for hzi in range(len(horizontals)-1) :
        for vti in range(len(verticals)-1) :
            sub = arr[horizontals[hzi]:horizontals[hzi+1],verticals[vti]:verticals[vti+1]]
            #print(sub)
            if sub.__contains__(min_val) :
                #only gets boxes with meaningful contents; pads each by 1 pixel
                boxes.append(Box(horizontals[hzi]-1,verticals[vti]-1,horizontals[hzi+1]+1,verticals[vti+1]+1,sub))
        
    
    return verticals, horizontals, boxes


array = create_scaled_image('imgin.png', None)
#plot.imshow(array, cmap='bone')
v, h, boxes = get_boxes(array)


#plot.imshow(img, cmap='bone')


#plot.imshow(boxes[0].get_contents(), cmap='bone')

model = load_model_from_JSON("./uci-fonts/model[best].json","./uci-fonts/uciweights[0.99964].h5")

sub_boxes = list()

text = ""

for n in range(len(boxes)) :
    box = boxes[n]
    arr = box.get_contents()
    v, h, new_subs = get_boxes(arr)
    for nbox in new_subs :
        nbox.set_parent(box)
        sub_boxes.append(nbox)
        '''
    for sub_box in new_subs :
        text = text + sub_box.get_prediction(model, 0.9)[0][0]
        '''

print(text)
    
index = 1

nr = 2
nc = 2

par = sub_boxes[index - 1].get_parent()

fig, axarr = plot.subplots(nrows=2,ncols=2)
plot.sca(axarr[0][1])
#plot.subplot(211)
plot.imshow(par.get_contents())
plot.sca(axarr[1][0])
plot.imshow(sub_boxes[index - 1].get_contents())

for sub in sub_boxes :
    plot.sca(axarr[0][0])
    p = sub.get_prediction(model, 0)
    plot.imshow(superimpose_pred(sub, array, p[0][0]))
#plot.imshow(boxes[2].get_contents())
def plot_new(i) :
    #print(i)
    global par
    sub = sub_boxes[i]
    if sub != None :
        
        '''superimpose_box(sub, array)'''
        if sub.get_parent() != par :
            plot.sca(axarr[0][1])
            par = sub.get_parent()
            plot.imshow(par.get_contents())
        
        plot.sca(axarr[1][0])
        p = sub_boxes[i].get_prediction(model, 0.9)
        stri = ("sub_box: ",i,"pred: ",p[0:2],"tries: ",p[3])
        plot.suptitle(stri)
        plot.imshow(p[2])
        
        plot.sca(axarr[1][1])
        plot.imshow(get_emnist_image(p[0][0]))
        
        plot.sca(axarr[0][0])
        plot.imshow(superimpose_pred(sub, array, p[0][0]))
        
        fig.canvas.draw()
        fig.canvas.flush_events()
    return i + 1

def onclick(event):
    global index
    if event.x > 500 :
        index = plot_new(index)
    elif index > 1 :
        index = plot_new(index - 2)

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plot_new(0)

'''
for vi in v :
    array[:,vi] = 100
for hi in h :
    array[hi,:] = 100


model = load_model_from_JSON("./uci-fonts/model[best].json","./uci-fonts/uciweights[0.99964].h5")

sentence = list()

for index in range(0,20) :

    bkgval = get_background_val(boxes[index].get_contents())
    trim_bkg(boxes[index], bkgval, 120)
    
    cont = boxes[index].get_prediction(model, threshold=0.99)
    #ix = 37
    #print(boxes[ix].get_bounds())
    #print(boxes[ix].contents)
    plot.imshow(cont[2], cmap='Blues')
    
    print(cont[0], "Probability:",cont[1])
    sentence.append(cont[0][0])

print(sentence)

'''
