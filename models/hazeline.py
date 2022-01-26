import cv2
import math
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np


def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w, color] = im.shape[:3]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz, 1)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort(axis=1)
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range( numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)

    return t


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res


def Accumarray(ind, radius, n_point):
    row = ind.shape[0]
    radius1 = radius.reshape(row,1)
    K = np.zeros([n_point, 1])
    for i in range(row):
        ind1 = ind[i,:]
        if K[ind1,:] < radius1[i,:]:
            K[ind1, :] = radius1[i,:]
    return K


def update(ind, K, h, w, radius):
    row = h*w
    radius1 = radius.reshape(row, 1)
    radius_new1 = np.zeros([row, 1])
    transmission_estimation= np.zeros([row, 1])
    for i in range(row):
        j = K[ind[i,:],:]
        radius_new1[i,:] = j
    for k in range(row):
        transmission_estimation[k,:] = radius1[k,:]/radius_new1[k,:]
        if transmission_estimation[k,:] < 0.1:
            transmission_estimation[k, :] = 0.1
        elif transmission_estimation[k,:] > 1:
            transmission_estimation[k, :] = 1
        else:
            transmission_estimation[k, :] = transmission_estimation[k, :]
    transmission_estimation = transmission_estimation.reshape(h,w)
    return transmission_estimation


def KNN(radius, point, k: int):
    y = np.arange(k)
    row = radius.shape[0]
    transfer = StandardScaler()
    point1 = transfer.fit_transform(point)
    radius1 = transfer.transform(radius)
    estimator = KNeighborsClassifier()
    '''param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=2)'''
    estimator.fit(point1, y)
    y_predict = estimator.predict(radius1)
    ind = y_predict.reshape(row,1)
    return ind


def TransmissionEstimate_nonlocal(im, A):
    [h, w, color] = im.shape[:3]
    dist_from_airlight = np.zeros([h, w, color])
    for color_idx in range(color):
        A2 = A[:,color_idx]
        dist_from_airlight[:,:,color_idx] = im[:,:,color_idx]-A2
    radius = np.multiply(dist_from_airlight[:,:,0],dist_from_airlight[:,:,0]) \
             + np.multiply(dist_from_airlight[:,:,1],dist_from_airlight[:,:,1]) \
             + np.multiply(dist_from_airlight[:,:,2],dist_from_airlight[:,:,2])
    radius = np.sqrt(radius)
    dist_unit_radius = dist_from_airlight.reshape(h*w,color)
    dist_norm = np.multiply(dist_unit_radius,dist_unit_radius)
    dist_norm = dist_norm.sum(axis=1).reshape(h*w,1)
    dist_norm = np.sqrt(dist_norm)
    for i in range(h*w):
        dist_unit_radius[i,:] = dist_unit_radius[i,:]/dist_norm[i,:]
    n_points = 1000
    point = np.loadtxt("TR1000.txt")
    ind = KNN(dist_unit_radius, point, n_points)
    ind = ind.astype(np.int16)
    K = Accumarray(ind, radius, n_points)
    transmission_estimation = update(ind, K, h, w, radius)
    return transmission_estimation


def Get_transmission(im):
    I1 = im.cpu().detach().numpy()
    size = I1.shape[2]
    I = np.zeros((size,size,3))
    for i in range(3):
        I[:,:,i] = I1[0,i,:,:]
    A = np.array([[245 / 255, 225 / 255, 200 / 255]])
    te = TransmissionEstimate_nonlocal(I, A)
    trans = torch.Tensor(te).cuda()
    return trans




