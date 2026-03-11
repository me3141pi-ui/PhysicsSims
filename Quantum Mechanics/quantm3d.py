import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import pickle

class quantm3d:
    def __init__(self,n = 50,potential_func = lambda x,y,z: 0,size = 1, reduced_planck = 1,mass = 1):
        #defining dimensions
        self.n = n
        self.grid_n = n**3

        #defining constants of physics
        self.hbar = reduced_planck
        self.mass = mass

        #vectorizing the potential function
        self.potential_func = np.vectorize(potential_func)

        #creating the set of all possible values of any coordinate in our grid
        self.set_of_coordinates = np.linspace(-size,size,n)
        self.delx = self.set_of_coordinates[1] - self.set_of_coordinates[0]
        #creating the x, y, z coordinate meshgrid
        self.x ,self.y,self.z = np.meshgrid(self.set_of_coordinates,self.set_of_coordinates,self.set_of_coordinates)

        #creating the potential vector_coord
        self.potential_vector = self.potential_func(self.x,self.y,self.z).flatten()

        def gen_laplacian():
            #generates the laplacian operator for 3d
            main_diag = np.ones(self.grid_n) * -6
            off_x = np.ones(self.grid_n - 1)
            off_y = np.ones(self.grid_n - n)
            off_z = np.ones(self.grid_n - n ** 2)

            for i in range(len(off_x)):
                if (i + 1) % n == 0:
                    off_x[i] = 0

            for i in range(len(off_y)):
                if (i + n) % (n * n) == 0:
                    off_y[i] = 0

            diagonals = [main_diag, off_x, off_x, off_y, off_y, off_z, off_z]
            offsets = [0, 1, -1, n, -n, n ** 2, -n ** 2]

            mat = sp.diags(diagonals, offsets, shape=(self.grid_n, self.grid_n))
            return mat

        self.kinetic_op = -1*((self.hbar**2)/((2*self.mass)*(self.delx**2))) * gen_laplacian()
        self.potential_op = sp.diags([self.potential_vector],offsets=[0])

        self.hamiltonian = self.kinetic_op+self.potential_op

        self.eigenvals = []
        self.psi = []

    def solve(self,activations = 1):
        from scipy.sparse.linalg import eigsh
        #solving the linear equation self.hamiltonian * psi[i] = E * psi[i]
        self.eigenvals, eigenvectors = eigsh(self.hamiltonian,k=activations+1,which='SA')

        #appending the reshaped,normalized eigenvector (aka state) to the list of states
        for i in range(activations+1):
            v = eigenvectors[:,i]
            v = v/np.linalg.norm(v)
            self.psi.append(v.reshape((self.n, self.n, self.n)))

    def plot_3d(self, state_idx=0, threshold=0.1):
        if state_idx >= len(self.psi):
            return

        #calculating probability vector
        prob = np.abs(self.psi[state_idx])**2
        max_val = np.max(prob)

        #creating threshhold mask to graph only high value points
        mask = prob>(max_val*threshold)

        x_vals = self.x[mask]
        y_vals = self.y[mask]
        z_vals = self.z[mask]
        c_vals = prob[mask]

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')

        img = ax.scatter(x_vals,y_vals,z_vals,c=c_vals, cmap='jet', alpha=(prob[mask]/np.max(prob[mask]))**1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        fig.colorbar(img, ax=ax, shrink=0.5, aspect=5)

        plt.show()

    def save_states(self,filename):
        with open(filename,'wb') as f:
            pickle.dump([self.psi,self.x,self.y,self.z],f)

    @staticmethod
    def plot_saved_state(filename,state_idx = 1,threshold = 0.1,highlight = None):
        with open(filename,'rb') as f:
            loaded_data = pickle.load(f)

        psi,x,y,z = loaded_data
        if state_idx >= len(psi):
            return

        # calculating probability vector
        prob = np.abs(psi[state_idx]) ** 2
        max_val = np.max(prob)

        # creating threshhold mask to graph only high value points
        mask = prob > (max_val * threshold)

        x_vals = x[mask]
        y_vals = y[mask]
        z_vals = z[mask]
        c_vals = prob[mask]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        al = (prob[mask] / np.max(prob[mask])) ** highlight if highlight else 0.2

        img = ax.scatter(x_vals, y_vals, z_vals, c=c_vals, cmap='jet', alpha=al)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        fig.colorbar(img, ax=ax, shrink=0.5, aspect=5)

        plt.show()
def inverse(x,y,z):
    return -10 / ((x**2 + y**2 + z**2))

# Copy-paste this replacement
def soft_potential(x,y,z):
    return -1.5 /np.sqrt( (x**2 + y**2 + z**2 + 0.001)) # The 0.1 is the magic number
def well(x,y,z):
    if -2 <x <2 and -2 <y<2 and -2 < z < 2:
        return 0
    else:
        return 20
#
# x = quantm3d(n=60, potential_func=well, size=3)
# x.solve(activations=20)
# '''function_n_size_activations'''
# x.save_states('States/well_60_3_20.pkl')
quantm3d.plot_saved_state('States/soft_70_4_20.pkl', state_idx=3, threshold=0.2,highlight=1)