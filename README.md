# image-processing
Basics image processing using NumPy and SVD concept in Linear Algebra

""""  
    a. We will learn about the matrix decomposition which is an important concept in Linear algebra. 
    b. We are employing the Singular Value Decomposition (SVD) to generate the compressed approximation of an image.
    c. In the example we are using the face image from scipy.datasets.
    
"""

try:
    from scipy.datasets import face
except ImportError:  # Data was in scipy.misc prior to scipy v1.10
    from scipy.misc import face

img = face()

# let us the data type of img
print(type(img))

# Let us display the image using the some matplotlib.pyplot.imshow function and special iPython command

import matplotlib.pyplot as plt
%matplotlib inline
plt.imshow(img)
plt.show()

# Let us study some basic properties of the above image.
# shape, axis, and array properties
"""
    In linear algebra, the dimensions of a vectorrefers to the number of elements in an array. But in the NumPy terminology,
    the dimensions of a vector is referred to as the number of axes. 
    For exampl, 1D array is a vector such as [1,2,3],
                2D is a matrix and so on...
                
"""

print(img.shape)    # return a tuple with three values meaning a 3D array

# The image isa color image. The data of image is organised in three 2D arrays. The color is represented in the RGB channel

print("The dimension of an image is ", img.ndim)

# Now let us access the data of each axes
for idx in range(img.ndim):
    if idx == 0:
        color = 'RED'
    elif idx == 1:
        color = 'GREEN'
    else: 
        color = 'BLUE'
    print(f"The data in the {color} channel is shown below")
    print(img[:,:,idx])

"""
    Since we are going to perform linear algebra operations on this data, it might be more interesting to have 
    real numbers between 0 and 1 in each entry of the matrices to represent the RGB values. 
    So let us reduce the value of each entry in the img by diving the img with 255 so that the each entry will have 
    between 0 and 1
"""

img_array = img/255

print(img_array.max(), ',', img_array.min())  # print the maximum and minimum value of the img_array
print(type(img_array))   # prints the type of img_array

# Let us see whether the image changes when we divide by 255 

plt.imshow(img_array)
plt.show()

# Now let us assign each color channel to a seperate matrix using slicing operator

for idx in range(img.ndim):
    if idx == 0:
        color = 'RED'
        red_array = img_array[:,:,idx]
    elif idx == 1:
        color = 'GREEN'
        green_array = img_array[:,:,idx]
    else:
        color = 'BLUE'
        blue_array = img_array[:,:,idx]
print(red_array)
print(green_array)
print(blue_array)

# Operations on an Axis
"""
    We can approximate an data of an image by using the methods from Linear algebra. Here we will discuss the Singular 
    Value Decomposition methos (SVD) method.
    Using SVD method, we try to rebuild an image that uses fewer singular values information than the original one, still
    retainging the features of original image.

    Theere are several ways we can us for approximation of image. and several softwares such as Numpy, MatLab we can use 
    for this. Here we are using NumPy software and SVD method from linear algebra.

    We use the 'linalg' module in Numpy in this tutorial. 

    The SVD method decomposes the original matrix into three matrix .i.e A = U @ S @Vt where U, S, and Vt are three matrices
    when multiplied we get the original matrix A.
    U, and Vt matrices are square matrices and the 'S' matrix has the same shape as that of the original matrix.

    By colorimetry, any grayscaleversion of color image can be approximated by the below formula

                    Y = 0.2126R + 0.7152G + 0-0722B
    Where Y is the array representing grayscale image
          R, G, and B are the Red, Green, and Blue channel arrays.
    
"""

from numpy import linalg     # import the linalg module 

img_gray = img_array @ [0.2126, 0.7152, 0.0722]   

print(img_gray.shape)        # The shape of img_gray array    

# Now let us decompose the img_gray array using SVD
# To do so, we are using the linalg.svd() function in the linalg module

U, s, Vt = linalg.svd(img_gray)

print(U.shape, s.shape, Vt.shape)

"""
    ATTENTION:
    --------- 
    a.  The 'S' matrix (i.e. the middle one in the matrix product) has 1-Dimension. i.e. The shape of the 'S' matrix 
        is shown to be 1-Dimension. But as per theory, it's shape must be the same as that of the original matrix.
        This is because of the following fact.
        The middle matrix i.e. the SIGMA matrix is a diagonal matrix. And all entries in the main diagonal are real values
        and the off diagonal entries are zeros. It is not worthwile to store all those off-diagonal elements at the 
        cost of meory. So these diagonal entries are stored in 1-Dimension array.     
    b.  Now the how can we rebuild the SIGMA matrix i.e. How can we convert the one dimension arraay to Diagonal matrix?
        We can construct the Diagonal matrix from the 1-Dimension matrix 
            i.  by filling all the singular value using the fill_diagonal() function along the diagonal 
            ii. and off-diagonal entries by zeros using zeros() function
      
"""

import numpy as np
Sigma = np.zeros((U.shape[1], Vt.shape[0]))
np.fill_diagonal(Sigma, s)
print(Sigma)

"""
    We have approached the final part of this session. It is approximation.
    
    Approximation:
    --------------
    Now let us see is the reconstructed U @ Sigma @ Vt matrix is close to the original matrix
    
    a. In Numpy, the norm of a vector or a matrix is calculated by  'norm' function in the 'linalg' module. 
    b. The norm difference between the original matrix and the reconstructed matrix should be as less as possible if the
        recostructed image is near to the original image.
        If the norm difference is zero, then the reconstructed image is same as the original image

"""
np.linalg.norm(img_gray - U @ Sigma @ Vt)


# The approximation can also be done using the 'allclose()' function in the numpy

np.allclose(img_gray, U @ Sigma @ Vt)

# To see the approximation is acceptable, let us check the singular values in 's' as shown below

plt.plot(s)
plt.show()

# Let us check the total number of singular value in s

print(len(s)) # Returns the number of singular value in the S matrix

"""
    The reconstructed image contain 786 singular values. Reconstruction of images uses these singular values.
    But not all those singular values (here 786) are needed to reconstruct the image. 
    By observing the above graph plot, we can use upto 50 values to build a more economical approximation of the image.
    The idea is simple:
        a. keep 'U' and 'Vt' matrices as they were
        b. But taking the first k singular values in the SIGMA matrix
        c. The image is approximated by taking the product of 'U', 's' with k singular values and 'Vt'  

    Now let us for different values of k, how the approximation of reconstructed image changes
"""

for k in range(2,150,3):
    approx = U @ Sigma[:, :k] @ Vt[:k, :]
    plt.imshow(approx, cmap="gray")
    plt.show()

# Applying to all colors 
"""
    To apply to all colors, we need to perform the same kind of operation as we did above. But this is a tedious effort.
    But Broadcasting feature in NumPy takes care of this at one go.
    If our array has more than two dimensions (i.e more than 2 axex), while applying SVD, the broadcasting will takecare of 
    this internally.
    
    Note: The linear algebra functions in NumPy expects to see an array of the form (n, M, N), where n represents 
    the number of M*N matrices in the stack

     Here the img.shape gives the (768, 1024, 3)
     o we need to permutate the axis on this array to get a shape like (3, 768, 1024).
     TThe permutating of the matrix is done by using the Tranpose function
"""

img_array_transposed = np.transpose(img_array, (2, 0, 1))
print(img_array_transposed.shape)         # prints the shape of the transposed array 
print(img_array_transposed)               # prints the transposed array

# Now, finally we apply SVD  using the ligalg.svd() function

U, s, Vt = linalg.svd(img_array_transposed)

# print the shapes of U. s and Vt 

print(U.shape, s.shape, Vt.shape)

# To build a final approximation, we must understand how matrix multiplication accross different axes 
# or dimensions  works.

"""
    a. In intermediate we learnt how to perform the matrix multiplication with one- and two-dimensional array manually.
    b. In NumPy We can do the same using the numpy.dot and numpy.matmul (@ operator).
    c. For n-dimensional arrays, the matrix multiplication work in different way. 
    d. we need prepare the Sigma array (i.e. the middle array) ready and compatible for matrix multiplication.
    e. The Sigma array must have dimension of the form (3, 768, 1024). Then we need to add zeros on the off diagonal entries
        and singular value on the diagonal entries as we did before. using the linalg.fill_diagonal() function.
"""

Sigma = np.zeros((3, 768, 1024))
for j in range(3):
    np.fill_diagonal(Sigma[j, :, :], s[j, :])


# Now let us rebuild the full SVD with no approximation 

reconstructed = U @ Sigma @ Vt
print(reconstructed.shape)

"""
    The reconstructed image should be indistinguishable from the original one, except for differences due to 
    floating point errors from the reconstruction. 
    Recall that our original image consisted of floating point values in the range [0., 1.]. 
    The accumulation of floating point error from the reconstruction can result in values slightly outside 
    this original range

"""

print(reconstructed.min(), reconstructed.max())

# Since imshow expects values in the range, we can use clip to excise the floating point error:
reconstructed = np.clip(reconstructed, 0, 1)
plt.imshow(np.transpose(reconstructed, (1, 2, 0)))
plt.show()

# Now, to do the approximation, we must choose only the first k singular values for each color channel. 
# This can be done using the following syntax:

approx_img = U @ Sigma[..., :k] @ Vt[..., :k, :]

"""
    We have selected only the first k components of the last axis for Sigma (this means that we have used only the 
    first k columns of each of the three matrices in the stack), and that we have selected only the first k components 
    in the second-to-last axis of Vt (this means we have selected only the first k rows from every matrix in 
    the stack Vt and all columns

"""
print(approx_img.shape)

# Finally, reordering the axes back to our original shape of (768, 1024, 3), we can see our approximation

plt.imshow(np.transpose(approx_img, (1, 2, 0)))
plt.show()
