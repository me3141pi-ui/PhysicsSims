import matplotlib.pyplot as plt
import numpy as np
import copy
class quantmX:

    #functions to generate the 1,2,-1 matrix for the finite difference method of second derivative wrt x
    def gen121mat(self,n):
        mat = np.zeros((n,n))
        mat[0][0] = -2;mat[0][1] = 1
        for i in range(1,n-1):
            mat[i][i-1] = 1;mat[i][i] = -2; mat[i][i+1] = 1
        mat[n-1][n-2] = 1;mat[n-1][n-1] = -2

        return mat

    #initialization
    def __init__(self,n = 1000,x1 = -5,x2 = 5,
                 potential_func = lambda x: 0,mass = 1,
                 red_plank = 1):
        potential_func = np.vectorize(potential_func)
        #grid size
        self.size = n
        #position array
        self.pos = np.linspace(x1,x2,n)
        #potential array
        self.potential = np.transpose(potential_func(self.pos))
        #delta x
        self.del_x = self.pos[1]-self.pos[0]

        #hamiltonian matrix H in H psi = E psi
        self.hamiltonian = self.potential@np.identity(n) - ((red_plank**2)/(2*mass*(self.del_x**2))) * self.gen121mat(n)

        #list of all possible states
        self.psi = [False] * self.size

    def _thomas_solver(self,d):

        d = np.transpose(d)
        ad = [self.hamiltonian[0][0]]
        dd = [d[0]]

        k = self.hamiltonian[0][1]

        for i in range(1,self.size-1):
            dd.append(d[i] - k*dd[-1] / ad[-1])
            ad.append(self.hamiltonian[i][i] - k*k/ad[-1])

        xs = []
        xs.append((dd[-1] - d[-1]*ad[-1])/(1-self.hamiltonian[-1][-1]*ad[-1]))
        for i in range(self.size-1):
            xs.append((dd[-1 - i] - k*xs[-1])/ad[-1-i])

        return xs[::-1]

    def solve_psi0(self,iter = 100):
        x = np.transpose(np.random.rand(self.size))
        for _ in range(iter):
            ykpo = self._thomas_solver(x)
            x = ykpo/np.linalg.norm(ykpo)
        self.psi[0] = x

    def solve(self,iter = 100,activation = 1):
        self.solve_psi0(iter)
        def cos_angle(v1,v2):
            return np.sum(v1*v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

        assert(activation < self.size)



        for i in range(1,activation+1):
            x = np.transpose(np.random.rand(self.size))*5

            for _ in range(iter):
                for j in range(i):
                    psiJ = self.psi[j]
                    x = x - cos_angle(psiJ,x)*psiJ

                ykpo = self._thomas_solver(x)
                x = ykpo/np.linalg.norm(ykpo)
            self.psi[i] = x

    def plot(self, states=[0], plot_potential=True,plot_probability = False):
        """
        Plots the wave functions and optionally the potential energy.
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))
        lines = []

        # 1. Plot Wave Functions (Left Axis)
        for i in states:
            # Check if state is initialized (not False) and within bounds
            if i >= self.size or self.psi[i] is False:
                print(f'State {i} not initialized')
                continue

            # Plot
            if plot_probability:
                l, = ax1.plot(self.pos, self.psi[i]**2, label=fr'$\psi^2_{{{i}}}$')
            else:
                l, = ax1.plot(self.pos, self.psi[i], label=fr'$\psi_{{{i}}}$')

            lines.append(l)

        ax1.set_xlabel('Position (x)')
        ax1.set_ylabel(r'Wave Function Amplitude $\psi(x)$')
        ax1.axhline(0, color='gray', linestyle=':', linewidth=0.8)

        # 2. Plot Potential (Right Axis)
        if plot_potential and self.potential is not None:
            ax2 = ax1.twinx()

            # Plot potential with a dashed line
            l_pot, = ax2.plot(self.pos, self.potential, 'k--', alpha=0.5, label=r'Potential $V(x)$')
            lines.append(l_pot)

            ax2.set_ylabel(r'Potential Energy $V(x)$')

            # Fill for visibility
            ymin = np.min(self.potential)
            ymax = np.max(self.potential)
            if ymax != ymin:  # Avoid warning on flat potential
                ax2.fill_between(self.pos, self.potential, ymax, color='gray', alpha=0.1)

        # 3. Combined Legend
        if lines:
            labs = [l.get_label() for l in lines]
            ax1.legend(lines, labs, loc='best')

        plt.title('Quantum States and Potential')
        plt.tight_layout()
        plt.show()

def potential_barrier(x):
    if -5 < x < 5:
        return 0
    else:
        return 1


def sho(x):
    return (x**2)/2

def two_nuclei(x):
    return -10/((x-7)**2 + 1) - 10/((x+7)**2 + 1)
def double_well_cnt(x):

    return (-200 / (x+15) **2 ) - (200/(x+19.99)**2)
def assymetric_well(x):

        return 10 * (1 - np.exp(-0.3 * (x + 2))) ** 2 - 10

def step_well(x):
    # Deep well between -5 and 5
    if x < -5:
        return 100 # High wall
    elif x > 5:
        return 100 # High wall
    elif x > 0:
        return -5  # Shallow shelf
    else:
        return -10 # Deep part

def id_x(x):
    return x

x = quantmX(potential_func=potential_barrier,x1 = -10,x2 = 10,n = 2000)
x.solve(iter = 4000,activation=2)
# plt.plot(x.pos,x.psi[2])
# plt.plot(x.pos,x.potential/10)
x.plot(states=[0,2],plot_probability=True)
plt.show()

