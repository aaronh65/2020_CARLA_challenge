import numpy as np
import matplotlib.pyplot as plt

# order is polynomial
# points centered at origin and are in PIL image frame
def approximate(points, order=-1):
    bases = [lambda x: np.ones(len(x)),
         lambda x: x,
         lambda x: x**2,]


    if order >= 0:
        bases = [lambda x: x**i for i in range(order)]
        bases[0] = lambda x: np.ones(len(x))

    points = points.copy().T # 2xN
    #points[1] = 256 - points[1]
    dy = points[1][-1] - points[1][0]
    dx = points[0][-1] - points[0][0]
    theta = np.arctan2(dy, dx)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]])
    points = points.copy()
    points = R.dot(points)
    x, y = points

    # fit
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
    #c[abs(c)<1e-3]=0
    print(c)
    Ac = np.matmul(A,c)
    x_apx = np.linspace(0, x[-1], 100)
    y_apx = [basis(x_apx) for basis in bases] #3xM
    y_apx = c.T.dot(y_apx)
    y_apx = y_apx[0]

    if True:
        plt.scatter(x, y, color='green', s=20)
        plt.plot(x, y, color='green', label='gt')
        plt.scatter(x, Ac, color='blue', s=20)
        plt.plot(x_apx,y_apx, color='blue', label='apx')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.pause(0.5)
        plt.clf()



def main(path):
    with open(path, 'r') as f:
        arr = f.readlines()[0].split()
    f = np.asarray([float(num) for num in arr])
    x = np.linspace(0, 1, len(f))
    
    A = [basis(x) for basis in bases]
    A = np.asarray(A).T
    U,Sflat,V = np.linalg.svd(A, full_matrices=True)
    V = V.T
    Sinv = np.zeros((len(U),len(V)))
    for i, val in enumerate(Sflat):
        Sinv[i,i] = 1/val if val != 0 else 0
    Sinv = Sinv.T
    Pinv = np.matmul(V, np.matmul(Sinv, U.T))
    c = np.asarray([np.matmul(Pinv,f)]).T
    #c[abs(c)<1e-3]=0
    print(c)
    Ac = np.matmul(A,c)
    plt.plot(x, f)
    plt.plot(x, Ac)
    plt.show()


if __name__ == '__main__':
    path = 'problem2.txt'
    main(path)
