import numpy as np
import matplotlib.pyplot as plt

bases = [lambda x: np.ones(len(x)),
         lambda x: x,
         lambda x: x**2,
         lambda x: x**3]

def rotate_points(points, inv=False):
    points = points.copy()
    points = np.vstack([[[0,0]], points])
    points = points.T # 2xN
    N = points.shape[1]
    x_, y_= points
    theta = np.arctan2(y_[-1], x_[-1])
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]])
    if inv:
        R = R.T
    points = R.dot(points)
    return points.T, R # Nx2
    
# order is polynomial
# points centered at origin and are in PIL image frame
def approximate(points, order=-1, M=4, plot=False):

    if order >= 0:
        bases = [lambda x: x**i for i in range(order)]
        bases[0] = lambda x: np.ones(len(x))
    bases = [lambda x: np.ones(len(x)),
         lambda x: x,
         lambda x: x**2,
         lambda x: x**3]


    # R is rotmat for angle from x-axis to endpoint
    # here we align the points along the x-axis, thus inverse
    points, Rinv = rotate_points(points, inv=True)
    x, y = points.T

    # compute basis coefficients
    A = [basis(x) for basis in bases]
    A = np.asarray(A).T
    U,Sflat,V = np.linalg.svd(A, full_matrices=True)
    V=V.T
    Sinv = np.zeros((len(U),len(V)))
    for i, val in enumerate(Sflat):
        Sinv[i,i] = 1/val if val != 0 else 0
    Sinv = Sinv.T
    Pinv = np.matmul(V, np.matmul(Sinv, U.T))
    c = np.asarray([np.matmul(Pinv,y)]).T

    # approximated points
    Ac = np.matmul(A,c)
    N = len(x) 
    x_apx = np.zeros((N-1)*M) # M defaults to 4 equally spaced segments between points
    for i in range(N-1):
        diff = x[i+1] - x[i]
        dx = diff / M
        for j in range(M): 
            insert = i*M+j
            inc = dx*j
            x_apx[insert] = x[i] + inc

    y_apx = [basis(x_apx) for basis in bases] #3xM
    y_apx = c.T.dot(y_apx)
    y_apx = y_apx[0]

    if plot:
        plt.scatter(x, y, color='green', s=20)
        plt.plot(x, y, color='green', label='gt')
        plt.scatter(x, Ac, color='blue', s=20)
        #plt.plot(x_apx,y_apx, color='blue', label='apx')
        plt.scatter(x_apx, y_apx, color='blue')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.pause(0.5)
        plt.clf()

    result = np.vstack([x_apx,y_apx])
    result = Rinv.T.dot(result).T
    #print(result)
    return result

if __name__ == '__main__':
    path = 'problem2.txt'
    main(path)
