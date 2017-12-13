import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial.distance import cosine
from math import exp, hypot
import pickle

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def distance(x_i, x_j):
    return hypot(x_i[0] - x_j[0], x_i[1]-x_j[1])

def plot_gray(gray_mat):
    plt.imshow(gray_mat, cmap=plt.get_cmap('gray'))
    plt.show()

def get_distance_matrix(n_rows, n_cols):
    n_points = n_rows*n_cols
    dist_mat = np.zeros((n_points, n_points))
    for i in range(n_points):
        x_i = (i/n_cols, i%n_cols)
        print (i)
        for j in range(i,n_points):
            dist_mat[i,j] = distance(x_i, (j/n_cols, j%n_cols))
            dist_mat[j,i] = dist_mat[i,j]
    return dist_mat

def get_sim_bright_matrix(gray_mat):
    n_rows = gray_mat.shape[0]
    n_cols = gray_mat.shape[1]
    n_points = n_rows * n_cols
    sim_mat = np.zeros((n_points, n_points))
    for i in range(n_points):
        x_i = (i/n_cols, i%n_cols)
        for j in range(i,n_points):
            sim_mat[i,j] = gray_mat[x_i[0], x_i[1]] - gray_mat[j/n_cols, j%n_cols]
            sim_mat[j,i] = sim_mat[i,j]
    return sim_mat

def get_shi_affinity(gray_mat, sigma_i=0.1, sigma_x=0.1, r=5):
    shape_x = gray_mat.shape[0]
    shape_y = gray_mat.shape[1]
    n_pixels = shape_x*shape_y
    affinity_matrix = lil_matrix((n_pixels, n_pixels))
    for i in range(n_pixels-1):
        x_i = (i//shape_y, i%shape_y)
        gray_i = gray_mat[x_i[0], x_i[1]]
        print (i)
        for j in range(i+1, n_pixels):
            x_j = (j//shape_y, j%shape_y)
            dis = distance(x_i, x_j)
            gray_j = gray_mat[x_j[0], x_j[1]]
            if dis < r:
                gray_dis = gray_i-gray_j
                affinity_matrix[i, j] = exp(-(dis**2)/(sigma_x**2)) * \
                    exp(-(gray_dis**2)/(sigma_i**2))
                affinity_matrix[j,i]=affinity_matrix[i,j]
    return affinity_matrix

def image2affinity(image_file, r=5):
    img = mpimg.imread(image_file)
    gray = rgb2gray(img)
    sigm_x = 26.38314
    sig_i = 0.35589
    aff_mat = get_shi_affinity(gray, sigma_i=sig_i, sigma_x=sigm_x, r=r)
    return aff_mat

if __name__=='__main__':
    img = mpimg.imread('test3.png')
    gray = rgb2gray(img)
    # # plot_gray(gray)
    # dist_mat = get_distance_matrix(gray.shape[0], gray.shape[1])
    # sigm_x = np.std(dist_mat)
    sigm_x = 26.38314
    # bri_mat = get_sim_bright_matrix(gray)
    # sig_i = np.std(bri_mat)
    sig_i = 0.35589
    print (sigm_x)
    # sig_i = 0.33
    print (sig_i)
    # pickle.dump(dist_mat, open('dist.mat', 'wb'))
    #
    # # sim_bri_mat = get_sim_bright_matrix(gray)
    # # pickle.dump(sim_bri_mat, open('sim_bri.mat', 'wb'))
    #
    print ('sigx = ' + str(sigm_x))
    print ('sig1 = ' + str(sig_i))
    aff_mat = get_shi_affinity(gray, sigma_i=sig_i, sigma_x=sigm_x)
    pickle.dump(aff_mat, open('aff4.mat', 'wb'))
    # print 'ok'
