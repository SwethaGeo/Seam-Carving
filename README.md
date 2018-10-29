In this project, I implemented seam removal, seam insertion and optimal order retargeting.

**Seam Removal(vertical):** </br>
First, I calculated the total energy e of the image by adding the absolute sum of the x and y gradient for each channel; the gradient function used was np.gradient. Then calculated the cumulative minimum energy M where the first row of M was same as e and rest of the entries (i,j) was determined using dynamic programming M(i,j) = e(i,j) + min(M(i-1,j-1),M(i-1,j),M(i-1,j+1)). Then backtracked from the minimum value of M in the last row and found the minimum energy seam. Then removed the seam path using mask; the value along seam path for each channel was marked as false and by using the mask as index for the image removed the values along the seam path.


**Seam Insertion(vertical):** </br>
To enlarge the image by K seams, first I found the first K optimal vertical seams for removal and duplicated them by averaging the optimal seams with their left and right neighbors.

**Optimal Order Retargeting:** </br>
First found the transport and 1-bit map with the path chosen map for the entire image. Then backtracked from T(r,c) to T(0,0) removing the corresponding vertical or horizontal seams based on the values of 1-bit map.
