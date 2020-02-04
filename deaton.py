import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize_scalar

class deaton():
    '''Implementation of the Deaton consumption-savings problem with income.
    '''
    def __init__(self,
                 beta=.9,
                 R=1.05,
                 y=1,
                 Mbar=10,
                 ngrid_state=50,
                 ngrid_choice=100,
                 interpolation='linear',
                 maximization='discretized',
                 maxiter_bellman=100,
                 tol_bellman=1e-8,
                 testing=False):
        self.beta = beta    # Discount factor
        self.Mbar = Mbar    # Upper bound on wealth
        self.R = R          # Gross interest
        self.y = y          # Income level
        self.ngrid_state = ngrid_state    # Number of grid points for wealth
        self.ngrid_choice = ngrid_choice  # Number of grid points for consumption
        self.maximization = maximization  # Type of maximization in Bellman
        self.maxiter_bellman = maxiter_bellman # Maxiter for numerical optimization in Bellman
        self.tol_bellman=1e-8                  # Tolerance for numerical optimization in Bellman
        self.testing = testing # print debug messages
        if testing:
            self.epsilon = 0.0 # for testing and debugging
        else:
            self.epsilon = np.finfo(float).eps # smallest positive float number
        self.grid_state = np.linspace(self.epsilon,Mbar,ngrid_state) # grid for state space
        self.grid_choice = np.linspace(self.epsilon,Mbar,ngrid_choice) # grid for decision space
        self.interpolation = interpolation # interpolation type for Bellman equation

    def u(self,c):
        '''Utility function'''
        return np.log(c)

    def mu(self,c):
        '''Marginal utility function'''
        return 1/c

    def imu(self,u):
        '''Inverse marginal utility function'''
        return 1/u

    def next_period_wealth(self,M,c):
        '''Next period budget, vectorized'''
        assert isinstance(M, float) or M.shape == c.shape, 'Shape of M and c must be the same'
        return self.R*(M-c) + self.y

    def interpolate(self,x,f):
        '''Returns the interpolation function for given data'''
        if self.interpolation=='linear':
            return interpolate.interp1d(x,f,kind='slinear',fill_value="extrapolate")
        elif self.interpolation=='quadratic':
            return interpolate.interp1d(x,f,kind='quadratic',fill_value="extrapolate")
        elif self.interpolation=='cubic':
            return interpolate.interp1d(x,f,kind='cubic',fill_value="extrapolate")
        else:
            print('Unknown interpolation type')
            return None

    def bellman_discretized(self,V0):
        '''Bellman operator with discretized choice,
           V0 is 1-dim vector of values on grid
        '''
        # idea: create maxtix with state grid in columns and choice grid in rows
        # and then take maximum in each column to perform discretized choice
        M = np.repeat(np.reshape(self.grid_state,(1,-1)),self.ngrid_choice,0) # matrix with state space repeated in rows
        c = np.repeat(np.reshape(self.grid_choice,(-1,1)),self.ngrid_state,1) # decisions grid repeated by columns
        # c *= np.reshape(self.grid_state,(1,-1)) /self.Mbar # what does this line do???
        # compute wealth in the next period
        nxM = self.next_period_wealth(M,c)
        mask = c<=M # mask off infeasible choices
        if self.testing:
            print('M=',M,sep='\n') # debugging and testing
            print('c=',c,sep='\n')
            print('nxM=',nxM,sep='\n')
            print('mask=',mask,sep='\n')
        # interpolate values of next period value at next period case sizes
        inter = self.interpolate(self.grid_state,V0)
        nxV = inter(nxM) # value of next period wealth
        # construct the matrix with maximand of the Bellman equation
        preV1 = np.full((self.ngrid_choice,self.ngrid_state),-np.inf) # init V with -inf
        preV1[mask] = self.u(c[mask]) + self.beta*nxV[mask]
        # find optimal choice
        V1 = np.amax(preV1,axis=0,keepdims=False) # maximum in every column
        c1 = c[np.argmax(preV1,axis=0),range(self.ngrid_state)] # choose the max attaining levels of c
        return V1, c1

    def bellman_continuous(self,V0):
        #Bellman operator, V0 is one-dim vector of values on grid
        def maximand(c,M,inter):
            '''Maximand of the Bellman equation'''
            Vnext = inter(self.next_period_wealth(M,c)) # next period value at the size of cake in the next period
            V1 = self.u(c) + self.beta*Vnext
#             V1[np.isneginf(V1)]=np.log(self.epsilon) # replace -inf with low number
            return -V1 # negative because of minimization
        def findC(M,maximand=None,inter=None):
            '''Solves for optimal consumption for given cake size M and interpolated V0'''
            opt = {'maxiter':self.maxiter_bellman, 'xatol':self.tol_bellman}
            if self.testing: opt['disp']=3
            res = minimize_scalar(maximand,args=(M,inter),method='Bounded',bounds=[self.epsilon,M],options=opt)
            if res.success:
                return res.x # if converged successfully
            else:
                return M/2 # return some visibly wrong value
        # interpolation method
        inter = self.interpolate(self.grid_state,V0)
        # loop over states
        c1=np.empty(self.ngrid_state,dtype='float')
        for i in range(self.ngrid_state):
            c1[i] = findC(self.grid_state[i],maximand,inter)
            if self.testing: print('\n STATE POINT',i,': M =',self.grid_state[i],'c =',c1[i])
        V1 = - maximand(c1,self.grid_state,inter) # don't forget the negation!
        return V1, c1
    
    def solve_vfi (self, maxiter=100, tol=1e-4, callback=None):
        '''Solves the model using successive approximations
        '''
        if self.testing: maxiter = 1 # limit output in testing mode            
        V0=self.u(self.grid_state) # on first iteration assume consuming everything
        for iter in range(maxiter):
            if self.maximization == 'discretized':
                V1,c1=self.bellman_discretized(V0)
            else:
                V1,c1=self.bellman_continuous(V0)
            if callback: callback(iter,self.grid_state,V1,c1) # callback for making plots
            if np.all(abs(V1-V0) < tol):
                break
            V0=V1
        else:  # when i went up to maxiter
            if self.testing: 
                print('Stopped after first VFI iteration in testing mode')
            else:
                print('No convergence: maximum number of iterations achieved!')
        return V1,c1

    def solve_egm (self, maxiter=100, tol=1e-4, callback=None):
        '''Solver the model using endogenous gridpoint method
        '''
        A = np.linspace(0,self.Mbar,self.ngrid_state) # grid over savings
        zz = np.zeros(self.ngrid_state) # vector of zeros
        gr1 = np.array([0,self.Mbar]) # grid of two points
        c1 = np.array([0,self.Mbar]) # on first iteration assume consuming everything
        self.y = max(self.y,self.epsilon) # to avoid devision by zero
        for iter in range(maxiter):
            # EGM step
            if self.testing: print('Iteration %3d : '%iter,end='')
            nxM = self.next_period_wealth(A,zz) # next period M
            inter = self.interpolate(gr1,c1) # interpolate current policy function
            nxc = inter(nxM) # consumption next period
            c0 = np.empty(self.ngrid_state+1) # one extra point
            gr0 = np.empty(self.ngrid_state+1) # one extra point
            c0[0] = 0.
            c0[1:] = self.imu(self.beta*self.R*self.mu(nxc)) # consumption this period
            gr0[0] = 0
            gr0[1:] = c0[1:] + A
            if callback: callback(iter,gr0,np.full(gr0.shape,np.nan),c0) # callback for making plots
            # interpolate old policy on new grid
            dev = np.abs( inter(gr0[1:]) - c0[1:] )
            if self.testing:  print('max difference = %1.6e'%np.max(dev))
            if np.max(dev) < tol:
                break
            gr1 = gr0
            c1 = c0
        else:  # when i went up to maxiter
            print('No convergence: maximum number of iterations achieved!')
        # reinterpolate to the state grid for convenience
        inter = self.interpolate(gr0,c0)
        c = inter(self.grid_state)
        return [],c

    def solve_plot(self, solver_name='vfi', **kvarg):
        '''Illustrate solution
           Inputs: solver (string), and any inputs to the solver
        '''
        if solver_name=='egm':
            solver = self.solve_egm
        else:
            solver = self.solve_vfi
        fig1, (ax1,ax2) = plt.subplots(1,2,figsize=(16,8))
        ax1.grid(b=True, which='both', color='0.65', linestyle='-')
        ax2.grid(b=True, which='both', color='0.65', linestyle='-')
        ax1.set_title('Value function convergence with %s'%solver_name)
        ax2.set_title('Policy function convergence with %s'%solver_name)
        ax1.set_xlabel('Wealth, M')
        ax2.set_xlabel('Wealth, M')
        ax1.set_ylabel('Value function')
        ax2.set_ylabel('Policy function')
        def callback(iter,grid,v,c):
            print('.',end='')
            ax1.plot(grid[1:],v[1:],color='k',alpha=0.25)
            ax2.plot(grid,c,color='k',alpha=0.25)
        V,c = solver(callback=callback,**kvarg)
        # add solutions
        if any(V): ax1.plot(self.grid_state[1:],V[1:],color='r',linewidth=2.5)
        if any(c): ax2.plot(self.grid_state,c,color='r',linewidth=2.5)
        plt.show()
        return V,c