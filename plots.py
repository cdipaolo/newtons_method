import numpy as np
from matplotlib import pyplot as plt

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
    root_L = newtons_method(f1, f1prime, 0.4, iters=28)
    # print progression of roots
    for i,root in enumerate(root_L[:5]):
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

def makeplots():
    plot_root_find()

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
      x -> 1d np.array
    '''
    x = [x0]
    for i in range(iters):
        x.append(x[-1] - f(x[-1])/fprime(x[-1]))
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
