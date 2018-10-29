import cv2
import numpy as np
import scipy as sp
import scipy.signal


def energy_map(img):
    """This function calculates the total energy by adding the absolute of x and y gradient of the image.
    """
    img_new = img.astype(float) #converting image to float
    total_energy = 0.0 # To store the sum of energy for all channels
    r,c,d = img.shape 
    for i in range(d):
        dy = np.zeros([r, c], dtype=float) 
        dx = np.zeros([r, c], dtype=float)
        if r > 1:
            dy = np.gradient(img_new[:,:,i], axis=0) #gradient along rows
        if c > 1:
            dx = np.gradient(img_new[:,:,i], axis=1) #gradient along columns
        total_energy += np.absolute(dy) + np.absolute(dx) 
    return total_energy #Total energy map for entire image

def cumulative_energy_vertical(img):
    """This function calculates the cumulative minimum energy for all possible connected seams for each element.
    """
    e = energy_map(img) # Total energy 
    M = np.zeros((e.shape[0], e.shape[1]), dtype=type(e)) #To store cumulative minimum energy
    row,col = e.shape
    M[0] = e[0] #First row is same as energy_map first row
    for i in range(1,row):
        for j in range(0,col):
            if j == 0:
                 M[i,j] = e[i,j] + min(M[i-1,j],M[i-1,j+1])
            elif j == col-1:
                M[i,j] = e[i,j] + min(M[i-1,j-1],M[i-1,j])
            else:
                M[i,j] = e[i,j] + min(M[i-1,j-1],M[i-1,j],M[i-1,j+1])   
    return M

def seam_path_vertical(img):
    """This function finds the minimum energy vertical seam path by backtracking from the minimum value of M in the last row.
    """
    row, col = img.shape[:2]
    M = cumulative_energy_vertical(img)
    j = np.argmin(M[-1]) #column number of minimum cumulative energy along last row
    seam_path = np.zeros((row),dtype=int) #To store column numbers of minimum cumulative energy for each row
    seam_path[-1] = j # last element is j
    for i in range(row-1,-1,-1):
        if j == 0:
            j = j + np.argmin((M[i-1,j],M[i-1,j+1])) # either j or j+1
        elif j == col-1:
            j = j + np.argmin((M[i-1,j-1],M[i-1,j])) - 1 # either j-1 or j
        else:            
            j = j + np.argmin((M[i-1,j-1],M[i-1,j],M[i-1,j+1])) - 1 # either j-1 or j or j+1
        seam_path[i-1] = j 
    return seam_path, M
    
def remove_col(channel, seam_path):
    """This function removes the optimal seam path from the given channel of the image
    """
    row, col= channel.shape
    mask = np.ones(channel.size, dtype = bool) 
    mask[np.ravel_multi_index([range(row), seam_path], (row, col))] = False #Mask value along seam path marked as False
    img = channel.flatten()
    return img[mask].reshape(row, col-1) 


def seam_removal_vertical(img, seam):
    """This function returns the new image after removing one vertical seam path
    """
    row, col,channels = img.shape
    path = np.zeros((row),dtype=int)
    M = None
    e = 0.0
    if len(seam) == 0: # For seam removal
        path, M = seam_path_vertical(img)
        e = min(M[-1]) #Minimum cost of seam which is neeeded for optimal retargeting
    else: #For seam insertion where seam path already computed
        path = seam
    img_ret = np.zeros((row, col-1, channels), dtype=np.uint8) #To store new image after removing seam path
    for i in range(channels):
        img_ret[:,:,i] = remove_col(img[:,:,i], path) #Removing seam path for each channel
    return img_ret, e

def seam_insertion(img, k):
    """This function returns the new enlarged image with (row,col+k) shape and the image showing the optimal seam paths
    """
    row,col,channels = img.shape
    img_rem = img.copy()
    I = np.zeros(img.shape[:2],dtype = bool) 
    img_new = np.zeros((row,col+k,3),dtype = img.dtype) # To store enlarged image
    kernel = np.array([[0,0,0],[0.5,0,0.5],[0,0,0]]) # Kernel to find average of left and right neighbors
    seams = [] # To store optimal seam paths
    colidx = np.tile(range(col), (row, 1)) # The column index of the original image
    for i in range(k):
        path,e = seam_path_vertical(img_rem) # Finding seam path 
        img_rem,e = seam_removal_vertical(img_rem, path) #Removing vertical seam 
        I[range(row),colidx[range(row), path]] = True # Marking the seam path in original image True
        seams.append(colidx[range(row),path]) # appending optimal seam path
        colidx = remove_col(colidx, path) #Removing the column numbers of seam path from original image
    delta = np.cumsum(I,axis = 1) # Number of shifts required for the columns of the original image
    for i in range(row):
        img_new[i,range(col)+delta[i,range(col)]] = img[i,range(col)] #Storing the orginal image pixels to new position
    img_new1 = cv2.copyMakeBorder(img_new,1,1,1,1,cv2.BORDER_REFLECT_101 )
    for i in range(channels):
        img1 = sp.signal.convolve2d(img_new1[:,:,i],kernel,mode='valid') #Convolving using kernel to find average of left and right neighbors
        img_new[:,:,i] = img1 
    img_color = img_new.copy()
    img_1 = img.copy()
    for i in seams:
        img_1[range(row),i] = [0,0,255] # Seam path as red
    for i in range(row):
        img_new[i,range(col)+delta[i,range(col)]] = img[i,range(col)] #Restoring the values of pixel in original image
        img_color[i,range(col)+delta[i,range(col)]] = img_1[i,range(col)]
    return img_new,img_color

def image_transpose(img):
    """This function returns the transposed image
    """
    channels = img.shape[2]
    v = [0] * channels
    for i in range(channels):
        v[i] = img[:,:,i].T # Transposing image for each channel
    return np.dstack((v[0],v[1],v[2])) #Returing transposed image

def seam_removal_horizontal(img):
    """This function returns image after removing one horizontal seam
    """
    img_T = image_transpose(img)
    img_T, e = seam_removal_vertical(img_T,[])
    return image_transpose(img_T), e

def transport_map(img):
    """This function returns the Transport map (T) and 1-bit map (C) which indicates whether horizontal or vertical seam 
       was removed in each step for the entire image.
    """
    row, col = img.shape[:2]
    I =  [None] * col # To store column number of images
    T = np.zeros((row,col), dtype=float) #Transport map
    C = np.zeros((row,col), dtype=int) #Map with path chosen
    for i in range(row):
        print "row number Transport map:",i
        for j in range(col):
            if i == 0 and j == 0:
                T[i, j] = 0
                I[j] = img 
                continue
            if j==0 and i > 0: 
                img, e = seam_removal_horizontal(I[j]) 
                T[i,j], I[j], C[i,j] = e + T[i-1, j], img, 0 
            elif i == 0 and j > 0:
                img, e = seam_removal_vertical(I[j-1],[]) 
                T[i,j], I[j], C[i,j] = e + T[i, j-1], img, 1
            else:
                img_h, eh = seam_removal_horizontal(I[j]) 
                img_v, ev = seam_removal_vertical(I[j-1],[])
                T[i,j] = min(eh + T[i-1, j], ev + T[i, j-1]) 
                C[i,j] = np.argmin((eh + T[i-1, j], ev + T[i, j-1]))
                if  C[i,j] == 0:
                    I[j] = img_h 
                else:
                    I[j] = img_v
                       
    return T,C

def optimal_path(T, C, r, c):
    """This function returns a list containing the choice made at each step of the dynamic programming. 
    The choice made is stored by backtracking from T[r,c] to T[0,0].
    """
    seam_path = [0] * (r + c)
    k = r + c - 1
    while k >= 0:
        seam_path[k] = C[r,c]
        T[r,c] = None
        k -= 1
        if C[r,c] == 0:
            r = r-1
        else:
            c = c-1
    assert r == 0 and c == 0
    return seam_path

def retarget_image(img, T, C, r, c):
    """This function returns the retargeted image after removing r rows and c columns from image.
    """
    row, col = img.shape[:2]
    seam_path = optimal_path(T, C, r, c)
    img_final = img
    for i in seam_path:
        if i == 0:
            img_final, _ = seam_removal_horizontal(img_final)
        else:
            img_final, _ = seam_removal_vertical(img_final, [])
    return img_final


def main():
    #Reading fig5
    print "Reading fig5"
    img1 = cv2.imread("fig5.png")

    #Reading fig8
    print "Reading fig8"
    img2 = cv2.imread("fig8.png")

    #Reading fig7
    print "Reading fig7"
    img3 = cv2.imread("fig7.png")

    #Seam removal
    print "Removing Vertical Seams"
    img_new = img1.copy()
    n = 300 # number of vertical seams to remove
    for i in range(n):
        img_new, e = seam_removal_vertical(img_new,[])
        
    #Saving Seam removal result
    print "Saving Seam removal result"
    cv2.imwrite('fig5_seam_removal.png',img_new)

    #Seam imsertion
    print "Inserting Vertical seams"
    num_cols_to_insert = int(img2.shape[1] * 0.5)
    I, I_color = seam_insertion(img2, num_cols_to_insert)
    I_2, I_color_2 = seam_insertion(I, num_cols_to_insert)

    #Saving Seam Insertion results
    print "Saving Seam insertion results"
    cv2.imwrite('fig8_c.png',I_color)
    cv2.imwrite('fig8_d.png',I)
    cv2.imwrite('fig8_f.png',I_2)

    #Optimal Order Retargeting
    print "Transport map and Retargeted Image"
    T, C = transport_map(img3)
    T_new = T.copy()
    r = 125 # number of rows to remove
    c = 135 # number of columns to remove
    image = retarget_image(img3, T_new, C, r, c)
    # Applying color map
    T2 = T_new.copy()
    path_mask = np.isnan(T2)
    T2 = T2 / T2[~path_mask].max() * 255
    T2 = T2.astype(np.uint8)
    T_new_colormap = cv2.applyColorMap(T2, cv2.COLORMAP_JET)
    T_new_colormap[path_mask,:] = 255

    #Saving Transport map and retargeted image
    print "Saving Transport map and retargeted image"
    cv2.imwrite('Transport map.png',T_new_colormap)
    cv2.imwrite('fig7_retargeted.png',image)
    
if __name__ == "__main__":
    main()
