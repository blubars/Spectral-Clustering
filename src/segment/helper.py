import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial.distance import cosine, cdist
from math import exp, hypot
import pickle
from sklearn.neighbors import BallTree

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def distance(x_i, x_j):
    return hypot(x_i[0] - x_j[0], x_i[1]-x_j[1])

def plot_gray(gray_mat):
    plt.imshow(gray_mat, cmap=plt.get_cmap('gray'))
    plt.show()

def get_distance_matrix(n_rows, n_cols):
    points = np.array([(i,j) for i in range(n_rows) for j in range(n_cols)])
    return cdist(points, points, 'euclidean')

def get_sim_bright_matrix(gray_mat):
    pix_values = np.ravel(gray_mat).reshape(-1,1)
    return cdist(pix_values, pix_values)

def get_shi_affinity(dist_mat, bri_diff_mat, sigma_x=0.1, sigma_i=0.1, r=5):
    valid_points = dist_mat < r
    aff_mat = np.exp(-dist_mat**2//(sigma_x**2))*np.exp(-bri_diff_mat**2//(sigma_i**2))*valid_points
    print ('Got affinity matrix')
    return aff_mat

    # affinity_matrix = lil_matrix((n_pixels, n_pixels))
    # for i in range(n_pixels-1):
    #     x_i = (i/shape_y, i%shape_y)
    #     gray_i = gray_mat[x_i[0], x_i[1]]
    #     print (i)
    #     for j in range(i+1, n_pixels):
    #         x_j = (j/shape_y, j%shape_y)
    #         dis = distance(x_i, x_j)
    #         gray_j = gray_mat[x_j[0], x_j[1]]
    #         if dis < r:
    #             gray_dis = gray_i-gray_j
    #             affinity_matrix[i, j] = exp(-(dis**2)/(sigma_x**2)) * \
    #                 exp(-(gray_dis**2)/(sigma_i**2))
    #             affinity_matrix[j,i]=affinity_matrix[i,j]
    # return affinity_matrix

def get_shi_nearest_neighbour(gray_mat, sigma_i=0.1, sigma_x=0.1, r=5, k=5):
    n_rows = gray_mat.shape[0]
    n_cols = gray_mat.shape[1]
    points = np.array([[i, j] for i in range(n_rows) for j in range(n_cols)])
    bt = BallTree(points)
    n_pixels = len(points)
    affinity_matrix = lil_matrix((n_pixels, n_pixels))
    for point in points:
        ndis, ninds = bt.query(point.reshape(1,-1), k=k)
        for dis, npoint in zip(ndis.reshape(-1,)[1:], ninds.reshape(-1,)[1:]):
            # if dis < r:
            x_i, y_i = point
            x_j, y_j = points[npoint, :]
            gray_dis = gray_mat[x_i, y_i] - gray_mat[x_j, y_j]
            affinity_matrix[x_i, y_i] = exp(-(dis**2)//(sigma_x**2)) * \
                exp(-(gray_dis**2)//(sigma_i**2))
            affinity_matrix[y_i, x_i] = affinity_matrix[x_i, y_i]
    return affinity_matrix



def image2affinity(image_file, r=5):
    img = mpimg.imread(image_file)
    gray = rgb2gray(img)
    dist_mat = get_distance_matrix(gray.shape[0], gray.shape[1])
    sigm_x = np.std(dist_mat)
    print ("sigma_x = " + str(sigm_x))
    # sigm_x = 26.38314
    bri_mat = get_sim_bright_matrix(gray)
    sig_i = np.std(bri_mat)
    print ("sigma_i = " + str(sig_i))
    aff_mat = get_shi_affinity(dist_mat, bri_mat, sigm_x, sig_i, r=5)
    return aff_mat

if __name__=='__main__':
    img = mpimg.imread('test3.png')
    gray = rgb2gray(img)
    # # plot_gray(gray)
    dist_mat = get_distance_matrix(gray.shape[0], gray.shape[1])
    sigm_x = np.std(dist_mat)
    print sigm_x
    # sigm_x = 26.38314
    bri_mat = get_sim_bright_matrix(gray)
    sig_i = np.std(bri_mat)
    # sig_i = 0.35589
    # sig_i = np.std(gray)
    # print (sigm_x)
    # sig_i = 0.33
    print (sig_i)
    # pickle.dump(dist_mat, open('dist.mat', 'wb'))
    #
    # # sim_bri_mat = get_sim_bright_matrix(gray)
    # # pickle.dump(sim_bri_mat, open('sim_bri.mat', 'wb'))
    #
    print ('sigx = ' + str(sigm_x))
    print ('sig1 = ' + str(sig_i))
    aff_mat = get_shi_affinity(dist_mat, bri_mat, sigm_x, sig_i, r=5)
    print 'got sparse'
    # aff_mat = get_shi_nearest_neighbour(gray, sigma_i=sig_i, sigma_x=sigm_x, r=5, k=1000)
    pickle.dump(aff_mat, open('aff5.mat', 'wb'))
    # print 'ok'