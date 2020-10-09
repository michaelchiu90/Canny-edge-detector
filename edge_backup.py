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
    #G =  np.sqrt(np.square(Gx)+np.square(Gy))
    #Gx = ndimage.sobel(arr,0)
    #Gy = ndimage.sobel(arr,1)
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
    ## gA = (GxA, GyA)
    print("GxA.shape:"+ str(GxA.shape))
    #print("GyA.shape:"+ str(GyA.shape))
    #suppressed_G = G.copy()
    #(627, 949)

    #IndexM  = np.indices(G)
    #rowIndex = IndexM[0]
    #colIndex = IndexM[1]
    GShape = G.shape
    print(GShape)
    rowIndex = np.zeros(GShape[1])

    for i in range(GShape[0]-1):
        rowIndex = np.vstack((rowIndex, np.full((1,GShape[1]), i+1)))

    print(rowIndex)
    print("rowIndex.shape:"+ str(rowIndex.shape))

    colIndex = np.arange(GShape[1])

    for i in range(GShape[0]-1):
        colIndex = np.vstack((colIndex, np.arange(GShape[1])))

    print(colIndex)
    print("colIndex.shape:"+ str(colIndex.shape))


    #print(len(G))
    #print(len(G[0]))
    #for i in range(len(G)):
        #for j in range(len(G[0])):

    XA_Plus = rowIndex + GyA
    YA_Plus = colIndex + GxA

    XA_Minus = rowIndex - GyA
    YA_Minus = colIndex - GxA
    
    GPlus = ndimage.map_coordinates(G,[XA_Plus,YA_Plus])
            #GPlus = ndimage.map_coordinates(G,[[i + GyA[i][j]],[j + GxA[i][j]]])
            #print("GPlus:" + str(GPlus))
    GMinus = ndimage.map_coordinates(G,[XA_Minus,YA_Minus])
            #GMinus = ndimage.map_coordinates(G,[[i - GyA[i][j]],[j - GxA[i][j]]])
            #print("GMinus:" + str(GMinus))
            #print("G[i][j]:" + str(G[i][j]))
            #if G[i][j] < GPlus or G[i][j] < GMinus:
            #    G[i][j] = 0

    mask =  (G < GPlus) | (G < GMinus)

    G[mask] = 0
    print(mask)
    #mask = np.any([(G < GPlus),(G < GMinus)])
    #print(G.shape)
    #print(GPlus.shape)
    #mask = (G < GPlus[0]) | (G < GMinus[0])
    #print(mask.shape)
    
    #G[mask] = 0
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
    #print("G.shape:"+ str(G.shape))
    Maxdist = int(np.round(np.sqrt(H**2 +W**2)))
    #print ("Maxdist:"+ str(Maxdist))
    #thetas = np.deg2rad(np.arange(0,180.0)) ##TODO: linspace
    #thetas = np.deg2rad(np.arange(0,180.0))
    #print(thetas)
    thetas = np.deg2rad(np.linspace(0,180.0, num=720, endpoint=False)) ##TODO: linspace
    #print(thetas)
    #print("thetas:" +str(thetas.shape))
    rhos = np.linspace(-Maxdist,Maxdist, 2*Maxdist)
    #print("rhos:" +str(rhos.shape))

    accumulator = np.zeros((2*Maxdist, len(thetas)))
    #print("accumulator.shape:" + str(accumulator.shape))
    for y in range(H):
        for x in range(W):
            if G[y,x] > 0:
                for k in range(len(thetas)):
                    p = x*np.cos(thetas[k]) + y*np.sin(thetas[k])
                    #### vote, p has value from -Maxdist to Maxdist
                    ##accumulator[int(p)+Maxdist,k] +=1
                    # print("K:" +str(k))
                    # print("r:" + str(p))
                    accumulator[int(p)+Maxdist][k] +=1
    
    return accumulator , thetas ,rhos

img_name = 'aqua.png'
#img_name = 'road.jpeg'
img = read_img_as_array(img_name)
#print(img.shape)
gray = rgb2gray(img)

Gaussian = ndimage.gaussian_filter(gray,sigma=3)
save_array_as_img(Gaussian, "gauss.jpg")
G, Gx, Gy = sobel(Gaussian)
save_array_as_img(G, "sobel.jpg")
save_array_as_img(Gx, "sobel_X.jpg")
save_array_as_img(Gy, "sobel_Y.jpg")
suppressed_G = nonmax_suppress(G, Gx, Gy)
save_array_as_img(suppressed_G, "supress.jpg")
edgemap_G = thresholding(suppressed_G, 80)
save_array_as_img(edgemap_G, "edgemap.jpg")
accumulator , thetas ,rhos = hough(edgemap_G)

save_array_as_img(accumulator, "detect.jpg")
print("detect.jpg Gen!")
# plt.figure('Original Image')
# plt.imshow(edgemap_G)
# plt.set_cmap('gray')
# plt.figure('Hough Space')
# plt.imshow(accumulator)
# plt.set_cmap('gray')
# plt.show()

############################
def Local_MAX(G,i,j):
    #LocalMAX = True
    row_limit = len(G)
    # print("row_limit:" + str(len(G)) )
    # print("column_limit:" + str(len(G[0])))

    #if row_limit > 0:
    column_limit = len(G[0])
    for x in range(max(0,i-1),min(i+2,row_limit)):
        for y in range(max(0,j-1), min(j+2,column_limit)):
            if x != i or y!= j:
                #G[i][j] is local max and Larger than 100 vote
                if G[x][y] > G[i][j] or 75 > G[i][j]:
                    return False

    return True    

row_limit = len(accumulator)
column_limit = len(accumulator[0])

mask = np.zeros([row_limit,column_limit])
for i in range(row_limit):
    for j in range(column_limit):
        mask[i][j] = Local_MAX(accumulator,i,j)

print(mask)
print("mask.shape" + str(mask.shape))
print(accumulator)
print("accumulator.shape" + str(accumulator.shape))

mask_index = np.asarray(np.where(mask==1)).T
print("Count of mask:" + str(len(mask_index)))
print(mask_index)

im = Image.open(img_name)
draw = ImageDraw.Draw(im)


print("rhos.shape:"+ str(rhos.shape))
print(rhos)

for index in range(len(mask_index)):
    #print("index:" + str(index))
    #print("theta Before :"+ str(mask_index[index][1]))
    # Cut 0~180 degree in 720 parts before
    theta = np.deg2rad(mask_index[index][1]/4)
    #print("theta:"+ str(theta))
    p = mask_index[index][0]    
    #print("p:"+str(p))
    rho = rhos[p]
    #print("rho:" + str(rho))
    a = np.cos(theta)
    b = np.sin(theta) 

    x0 = (a*rho) #+ row_limit/2
    y0 = (b*rho) #+ column_limit/2

    #1000 is length of line
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    #print("a,b,x0,y0" , str(a), str(b), str(x0), str(y0))
    #print("x1,y1,x2,y2" , str(x1),str(y1),str(x2),str(y2))
    draw.line([x1,y1,x2,y2], fill = 128, width=1)

  
im = im.save("Result.jpeg")
print("Finished!")
############################

## helper function 
## draw line need new library 