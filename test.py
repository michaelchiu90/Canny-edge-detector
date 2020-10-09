import numpy as np
# G = np.array([[11,2,3,44,55],[6,77,77,9,10],[11,12,13,144,15]], np.float)
# print(G)

# G_RightShift = np.roll(G,1,axis=1)
# G_RightShift[:,0] = 0
# # print("G_RightShift")
# # print(G_RightShift)

# G_LeftShift = np.roll(G,-1,axis=1)
# G_LeftShift[:,-1] = 0
# # print("G_LeftShift")
# # print(G_LeftShift)

# G_DownShift = np.roll(G,1,axis=0)
# G_DownShift[0,:] = 0
# # print("G_DownShift")
# # print(G_DownShift)

# G_UpShift = np.roll(G,-1,axis=0)
# G_UpShift[-1,:] = 0
# # print("G_UpShift")
# # print(G_UpShift)

# G_UpLeftShift = np.roll(G_LeftShift,-1,axis=0)
# G_UpLeftShift[-1,:] = 0
# # print("G_UpLeftShift")
# # print(G_UpLeftShift)

# G_UpRightShift = np.roll(G_RightShift,-1,axis=0)
# G_UpRightShift[-1,:] = 0
# # print("G_UpRightShift")
# # print(G_UpRightShift)

# G_DownLeftShift = np.roll(G_LeftShift,1,axis=0)
# G_DownLeftShift[0,:] = 0
# # print("G_DownLeftShift")
# # print(G_DownLeftShift)

# G_DownRightShift = np.roll(G_RightShift,1,axis=0)
# G_DownRightShift[0,:] = 0
# # print("G_DownRightShift")
# # print(G_DownRightShift)


# mask_index = (G>=G_RightShift) & (G>=G_LeftShift) & (G>=G_DownShift) & (G>=G_UpShift) & (G>=G_UpLeftShift) & (G>=G_UpRightShift) & (G>=G_DownLeftShift) & (G>=G_DownRightShift)
# print(mask_index)




# def DrawLine(img_name,mask_index,rhos):
#     im = Image.open(img_name)
#     draw = ImageDraw.Draw(im)
    #print(mask_index)

    # theta = np.deg2rad(mask_index[:,1]/4)
    # #print(theta)
    # p = mask_index[:,0]
    # rho = rhos[p]
    # #print(rho)
    # a = np.cos(theta)
    # b = np.sin(theta) 
    # #Cal the x/y-intercept
    # x0 = (a*rho)
    # y0 = (b*rho)

    # #1000 is length of line
    # x1 = x0 + 1000 * (-b)
    # y1 = y0 + 1000 * (a)
    # x2 = x0 - 1000 * (-b)
    # y2 = y0 - 1000 * (a)

    # x1 = [ int(x) for x in x1]
    # y1 = [ int(x) for x in y1]
    # x2 = [ int(x) for x in x2]
    # y2 = [ int(x) for x in y2]
    # #print("x1:" + str(type(x1)))
    # arr = np.vstack((x1, np.vstack((y1, np.vstack((x2, y2)))))).T.reshape(-1).tolist()
    # #print("arr:" + str(type(arr)))
    # #print(arr)
    # print(arr)
    # draw.line(arr, fill = 128, width=1)
    

    # #For each mask, draw a line
    # for index in range(len(mask_index)):
    #     #Cut 0~180 degree in 720 parts before
    #     theta = np.deg2rad(mask_index[index][1]/4)
    #     p = mask_index[index][0]    
    #     rho = rhos[p]
    #     a = np.cos(theta)
    #     b = np.sin(theta) 

    #     #Cal the x/y-intercept
    #     x0 = (a*rho)
    #     y0 = (b*rho)

    #     #1000 is length of line
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))

    #     draw.line([x1,y1,x2,y2], fill = 128, width=1)
    
    # im = im.save("detection_result.jpg")


# rowIndex = np.zeros(10)
# for i in range(5-1):
#     rowIndex = np.vstack((rowIndex, np.full((1,10), i+1)))
# print(rowIndex)

# rowIndex2 = np.zeros(10)
# rowIndex3 = [ np.vstack((rowIndex3, np.full((1,10), i+1))) for i in range(5-1) ]
# print(rowIndex2)
# print(rowIndex3)

xxx = np.indices((200, 300))
print(xxx)