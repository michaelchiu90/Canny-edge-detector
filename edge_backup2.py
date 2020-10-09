from PIL import Image , ImageDraw # pillow package
import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt

def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr

def save_array_as_img(arr, file):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)

def show_array_as_img(arr):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.show()

def rgb2gray(arr):
    R = arr[:, :, 0] # red channel
    G = arr[:, :, 1] # green channel
    B = arr[:, :, 2] # blue channel
    gray = 0.2989*R + 0.5870*G + 0.1140*B
    return gray

#########################################
## Please complete following functions ##
#########################################

def sobel(arr):
    '''Apply sobel operator on arr and return the result.'''
    # TODO: Please complete this function.
    # your code here
    Ax = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Ay = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    Gx = ndimage.filters.convolve(arr,Ax)
    Gy = ndimage.filters.convolve(arr,Ay)    
    G = np.hypot(Gx,Gy)
    G *= 255.0 / np.max(G)    
    return G, Gx, Gy

def nonmax_suppress(G, Gx, Gy):
    '''Suppress non-max value along direction perpendicular to the edge.'''
    assert G.shape == Gx.shape
    assert G.shape == Gy.shape
    # TODO: Please complete this function.
    # your code here
    GxA = np.divide(Gx ,G,out=np.zeros_like(Gx), where=G!=0)
    GyA = np.divide(Gy ,G,out=np.zeros_like(Gy), where=G!=0)
    GShape = G.shape

    rowIndex = np.zeros(GShape[1])
    for i in range(GShape[0]-1):
        rowIndex = np.vstack((rowIndex, np.full((1,GShape[1]), i+1)))

    colIndex = np.arange(GShape[1])
    for i in range(GShape[0]-1):
        colIndex = np.vstack((colIndex, np.arange(GShape[1])))

    XA_Plus = rowIndex + GyA
    YA_Plus = colIndex + GxA

    XA_Minus = rowIndex - GyA
    YA_Minus = colIndex - GxA
    
    GPlus = ndimage.map_coordinates(G,[XA_Plus,YA_Plus])
    GMinus = ndimage.map_coordinates(G,[XA_Minus,YA_Minus])
    mask =  (G < GPlus) | (G < GMinus)

    G[mask] = 0
    return G

def thresholding(G, t):
    '''Binarize G according threshold t'''
    # TODO: Please complete this function.
    # your code here
    G[G<t] = 0
    G[G>=t] = 255
    return G

def hough(G):
    '''Return Hough transform of G'''
    # TODO: Please complete this function.
    # your code here
    H = G.shape[0]  #Y
    W = G.shape[1]  #X
    Maxdist = int(np.round(np.sqrt(H**2 +W**2)))
    thetas = np.linspace(0,180.0, num=720, endpoint=False)
    rhos = np.linspace(-Maxdist,Maxdist, 2*Maxdist)
    accumulator = np.zeros((2*Maxdist, len(thetas)))


    ##################
    #index_array = np.asarray(np.where(G>1)).T
    #accumulator

    mask = np.argwhere(G != 0)
    #mask_index = np.asarray(np.where(mask==1)).T
    sin_thetas = np.sin(np.deg2rad(thetas))
    cos_thetas = np.cos(np.deg2rad(thetas))
    
    rho_values = np.matmul(mask,np.array([sin_thetas,cos_thetas]))

    accumulator , _ , rho_values = np.histogram2d(
        np.tile(thetas,rho_values.shape[0]),
        rho_values.ravel(),
        bins=[thetas,rhos]
    )
    accumulator = np.transpose(accumulator)
  
    #######################
    # for y in range(H):
    #     for x in range(W):
    #         if G[y,x] > 0:
    #             for k in range(len(thetas)):
    #                 p = x*np.cos(thetas[k]) + y*np.sin(thetas[k])
    #                 # vote, p has value from -Maxdist to Maxdist
    #                 accumulator[int(p)+Maxdist][k] +=1
    
    return accumulator , thetas ,rhos

def DrawLine(img_name,mask_index,rhos):
    im = Image.open(img_name)
    draw = ImageDraw.Draw(im)
    #For each mask, draw a line
    for index in range(len(mask_index)):
        #Cut 0~180 degree in 720 parts before
        theta = np.deg2rad(mask_index[index][1]/4)
        p = mask_index[index][0]    
        rho = rhos[p]
        a = np.cos(theta)
        b = np.sin(theta) 

        #Cal the x/y-intercept
        x0 = (a*rho)
        y0 = (b*rho)

        #1000 is length of line
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        draw.line([x1,y1,x2,y2], fill = 128, width=1)
    
    im = im.save("detection_result.jpg")

img_name = 'road.jpeg'
img = read_img_as_array(img_name)
gray = rgb2gray(img)

Gaussian = ndimage.gaussian_filter(gray,sigma=1)
save_array_as_img(Gaussian, "gauss.jpg")
G, Gx, Gy = sobel(Gaussian)
save_array_as_img(G, "G.jpg")
save_array_as_img(Gx, "G_x.jpg")
save_array_as_img(Gy, "G_y.jpg")
suppressed_G = nonmax_suppress(G, Gx, Gy)
save_array_as_img(suppressed_G, "supress.jpg")
edgemap_G = thresholding(suppressed_G, 80)
save_array_as_img(edgemap_G, "edgemap.jpg")
accumulator , thetas ,rhos = hough(edgemap_G)

save_array_as_img(accumulator, "hough.jpg")

G_RightShift = np.roll(accumulator,1,axis=1)
G_RightShift[:,0] = 0
G_LeftShift = np.roll(accumulator,-1,axis=1)
G_LeftShift[:,-1] = 0
G_DownShift = np.roll(accumulator,1,axis=0)
G_DownShift[0,:] = 0
G_UpShift = np.roll(accumulator,-1,axis=0)
G_UpShift[-1,:] = 0
G_UpLeftShift = np.roll(G_LeftShift,-1,axis=0)
G_UpLeftShift[-1,:] = 0
G_UpRightShift = np.roll(G_RightShift,-1,axis=0)
G_UpRightShift[-1,:] = 0
G_DownLeftShift = np.roll(G_LeftShift,1,axis=0)
G_DownLeftShift[0,:] = 0
G_DownRightShift = np.roll(G_RightShift,1,axis=0)
G_DownRightShift[0,:] = 0
mask = (accumulator > 130) & (accumulator>=G_RightShift) & (accumulator>=G_LeftShift) & (accumulator>=G_DownShift) & (accumulator>=G_UpShift) & (accumulator>=G_UpLeftShift) & (accumulator>=G_UpRightShift) & (accumulator>=G_DownLeftShift) & (accumulator>=G_DownRightShift) 

#Get all the local_max point (x,y)
mask_index = np.asarray(np.where(mask==1)).T
DrawLine(img_name,mask_index,rhos)
