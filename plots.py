import numpy as np
from numpy.linalg import inv
from numpy.random import uniform
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import griddata
import colormaps as cmaps

## make plot of root-finding
##
## Use Function f1(x) = exp(1/(3x)) - cos(x/2) + (x-1)(x-3)^2-1.5
##              f1'(x)= (x-3)^2 + 2(x-3)(x-1) + exp(1/(3x))/(3x*x) + sin(x/2)/2
f1 = lambda x: np.exp(1/(3*x)) - np.cos(x/2) + (x-1)*(x-3)*(x-3) - 1.5
f1prime = lambda x: (x-3)*(x-3) + 2*(x-3)*(x-1) + np.exp(1/(3*x))/(3*x*x) + np.sin(x/2)/2

def plot_root_find():
    '''
    uses newton's method to find the root 
    of f1 and plots that
    '''
    print('==> Finding Roots of a Function f : R -> R')
    root_L = newtons_method(f1, f1prime, 0.4, iters=28)
    # print progression of roots
    for i,root in enumerate(root_L[:28]):
        print('==> Root {} : {} | D = {}'.format(i, root, np.abs(root-root_L[-1])))
    for i in range(len(root_L[:5])-1):
        roots = root_L[:i+1]
        fig, ax = plt.subplots()
        x, fx = points(f1,0.2, 4, 1000)
        ax.plot(x,fx, color='red', alpha=1)
        for i,root in enumerate(roots):
            ax.annotate('$x_{}$'.format(i), xy=(root,0), 
                    xytext=(-5,10), xycoords='data', 
                    textcoords='offset points')
            ax.plot((root,root), (0,f1(root)), color='grey', alpha=0.9)
        ax.plot((root_L[i],root_L[i+1]),(f1(root_L[i]),0), color='green', alpha=0.8)

        # make axes w/ spines
        ax.grid(True, which='both')
        
        # set x spines
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')

        # turn off right spine/ticks
        ax.spines['right'].set_color('none')
        ax.yaxis.tick_left()

        # turn off top spine/ticks
        ax.spines['top'].set_color('none')
        ax.xaxis.tick_bottom()

        plt.title("Newton's Method for Root Finding")
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.savefig('plots/root_plt_{}.png'.format(i), bbox_inches='tight')

def f2(x):
    '''f2 : R^2 -> R'''
    if len(x.shape) > 1:
        xx = np.sum(x*x, axis=1)
    else:
        xx = x[0]*x[0] + x[1]*x[1]
    return -np.cos(xx) - np.exp(-xx)

def f2_point(x,y):
    '''f2 at a point'''
    return -np.cos(x*x + y*y) - np.exp(-x*x - y*y)

def f2grad(x):
    '''second function gradient'''
    if len(x.shape) > 1:
        xx = np.sum(x*x, axis=1)
    else:
        xx = x[0]*x[0] + x[1]*x[1]
    constant = 2*np.exp(-xx) + 2*np.sin(xx)
    return np.array([x[0]*constant, x[1]*constant])

def f2hessian(x):
    '''second functon inverse hessian
    evaluated at x'''
    if len(x.shape) > 1:
        xx = np.sum(x*x, axis=1)
    else:
        xx = x[0]*x[0] + x[1]*x[1]
    # compute the hessian
    A = np.zeros((2,2))
    A[0,0] = 2*(2*x[0]*x[0]*np.cos(xx) + np.exp(-xx) * (1 - 2*x[0]*x[0] + np.exp(xx) * np.sin(xx)))
    A[0,1] = 4*np.product(x)*(-np.exp(-xx) + np.cos(xx))
    A[1,0] = A[0,1]
    A[1,1] = 2*(2*x[1]*x[1]*np.cos(xx) + np.exp(-xx) * (1 - 2*x[1]*x[1] + np.exp(xx) * np.sin(xx)))
    return inv(A) # returns the inverse of the hessian

def plot_f2(num_points = 20):
    '''makes a single plot of
    the function f2 in R^3 for
    evaluation purposes'''
    x = np.outer(np.linspace(-2, 2, num_points), np.ones(num_points))
    y = x.copy().T
    z = -np.cos(x**2 + y**2) - np.exp(-x**2 - y**2)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(x,y,z, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0, color='red', alpha=0.9, label='$f(x) = -\cos(x^2 + y^2) - e^{-(x^2 + y^2)}$')
    plt.title("Newton's Method for Optimization")
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()

def plot_optimize(num_points=50):
    '''
    uses newton's method to optimize
    a function
        f : R^n -> R
    and plot the progression
    '''
    root_L = newtons_method_opt(f2grad, f2hessian, np.array([1,0.6]), iters=5)
    # print progression of roots
    print('\n\n==> Optimization of a Function f : R^2 -> R')
    for i,root in enumerate(root_L):
        print('==> Root {} : {} | D = {} | f {}'.format(i, root, np.abs(root-root_L[-1]), f2(root)))
    for i in range(len(root_L[:5])-1):
        roots = np.array(root_L[:i])

        x = np.outer(np.linspace(-2, 2, num_points), np.ones(num_points))
        y = x.copy().T
        z = -np.cos(x**2 + y**2) - np.exp(-x**2 - y**2)

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(122, projection='3d')

        ax.plot_surface(x,y,z, cmap=cmaps.viridis, rstride=1, cstride=1, linewidth=0, color='red', alpha=0.5)
        ax.set_xlim3d(-2,2)
        ax.set_ylim3d(-2,2)
        ax.set_zlim3d(-2,1)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')

        if len(roots) > 0:
            ax.plot(roots[:,0], roots[:,1], f2(roots), color='red')
            for root in roots:
                ax.scatter(root[0],root[1],f2(root), color='black')

        ax = fig.add_subplot(121)
        ax.contourf(x,y,z, cmap=cmaps.viridis, levels=np.linspace(-2,1,40), label='$f(x) = -\cos(x^2 + y^2) - e^{-(x^2 + y^2)}$')
        if len(roots) > 0:
            ax.plot(roots[:,0],roots[:,1], color='red')
            for root in roots:
                ax.scatter(root[0],root[1], color='white')
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

        fig.suptitle("Newton's Method for Optimization")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('plots/optim_plt_{}.png'.format(i), bbox_inches='tight')

def makeplots():
    plot_root_find()
    plot_optimize()

############################
## Helper functions
############################

def newtons_method(f, fprime, x0, iters=4):
    '''newtons_method
    approximates finding a root of f
    using Newton's Method:
      x_{n+1} = x_n - f(x_n)/f'(x_n)
    returns the sequence x_i as an
    np array.

    outputs:
      x -> 1xn - d np.array
    '''
    x = [x0]
    for i in range(iters):
        x.append(x[-1] - f(x[-1])/fprime(x[-1]))
    return x

def newtons_method_opt(gradient, hessian_inv, x0, iters=4):
    '''newtons_method_opt
    approximates finding the minimum of f
    with m in n variables
    using Newton's Method:
      x_{n+1} = x_n - H^{-1}\\nabla f(x_n)
    returns the sequence x_i as an
    np array.

    outputs:
      x -> mxn - d np.array
    '''
    x = [x0]
    for i in range(iters):
        correction = np.dot(hessian_inv(x[-1]), gradient(x[-1]))
        x.append(x[-1] - correction)
    return x

def points(f,start,stop,num):
    '''generate
    generates a vector of inputs and
    outputs of a function x for a
    num points across the interval
    [left,right]

    output:
      (x, f(x)) -> (1d, n-d) numpy array

    >>> x,f = points(f1,0.2,4,100)
    >>> len(x)
    100
    >>> len(f)
    100
    '''
    x = np.linspace(start, stop, num)
    return x, f(x)

############################
## Main
############################
if __name__ == '__main__':
    makeplots()
