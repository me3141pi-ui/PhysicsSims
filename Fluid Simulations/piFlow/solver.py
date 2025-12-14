import numpy as np
import matplotlib.animation as anime
import matplotlib.pyplot as plt
class fluidSolver2d:
    def __init__(self,size = 1,n = 50,viscosity = 1):
        self.size = size
        self.n = n
        self.spacing =size/n

        self.viscosity = viscosity
        #defining x coord grid and y coord grid for our vector space
        coord_range = np.linspace(0,n+1,n+2)
        self.pg_y,self.pg_x = np.meshgrid(coord_range,coord_range)

        #defining velocity field for our grid
        self.vfs = n+2
        self.ux = np.zeros((self.vfs,self.vfs))
        self.uy = np.zeros((self.vfs, self.vfs))

        self.forces = []

    def _bilerp_vector(self,u,ix,iy):
        ixd = np.clip(ix,1,self.vfs-2);iyd = np.clip(iy,1,self.vfs-2)
        i = np.floor(ixd).astype(int);j = np.floor(iyd).astype(int)
        s = ixd - i;t = iyd - j
        sp = 1-s;tp = 1-t

        up = sp*tp*u[i,j] + sp*t*u[i,j+1] + s*tp*u[i+1,j] + s*t*u[i+1,j+1]
        return up

    def _div(self,px,py):
        return (px[2:,1:-1]-px[0:-2,1:-1])/(2*self.spacing) + (py[1:-1,2:] - py[1:-1,0:-2])/(2*self.spacing)
    def _grad(self,p):
        return (p[2:,1:-1] - p[0:-2,1:-1])/(2*self.spacing) , (p[1:-1,2:] - p[1:-1,0:-2])/(2*self.spacing)
    def _curl(self,ux,uy):
        return (uy[2:,1:-1] - uy[0:-2,1:-1])/(2*self.spacing)-(ux[1:-1,2:] - ux[1:-1,0:-2])/(2*self.spacing)

    def swirl_intensity(self):
        return np.abs(self._curl(self.ux,self.uy))
    def speed(self):
        return (self.ux**2 + self.uy**2)**0.5


    ######IMPLEMENTING ADVECTION STEP######
    def advect(self,time_step):
        oldx = self.pg_x - self.ux*time_step/self.spacing
        oldy = self.pg_y - self.uy*time_step/self.spacing

        self.ux = self._bilerp_vector(self.ux,oldx,oldy)
        self.uy = self._bilerp_vector(self.uy,oldx,oldy)

    ######IMPLEMENTING DIFFUSION STEP######
    def diffuse(self,time_step,iter = 20):
        a = self.viscosity*time_step/self.spacing**2
        ux_new = self.ux[:,:];uy_new = self.uy[:,:]
        for _ in range(iter):
            ux_new[1:-1,1:-1] = (self.ux[1:-1,1:-1]
            + a*(
                ux_new[0:-2,1:-1]+
                ux_new[2:,1:-1]+
                ux_new[1:-1,0:-2]+
                ux_new[1:-1,2:]
            ))/(1+4*a)
            uy_new[1:-1,1:-1] = (self.uy[1:-1,1:-1]
            + a*(
                uy_new[0:-2,1:-1]+
                uy_new[2:,1:-1]+
                uy_new[1:-1,0:-2]+
                uy_new[1:-1,2:]
            ))/(1+4*a)
        self.ux = ux_new
        self.uy = uy_new

    ######IMPLEMENTING PROJETION######
    def projection(self, iter=20):
        h = self.spacing
        div = np.zeros_like(self.ux)
        p = np.zeros_like(self.ux)

        div[1:-1, 1:-1] = self._div(self.ux,self.uy) * h* h

        for _ in range(iter):
            p[1:-1,1:-1] =( (
                p[1:-1,0:-2]+p[1:-1,2:]+
                p[0:-2,1:-1]+p[2:,1:-1]
            ) - div[1:-1,1:-1])/4

        dpx,dpy = self._grad(p)

        self.ux[1:-1,1:-1] -= dpx
        self.uy[1:-1,1:-1] -= dpy


    def simulate(self,frames = 100,time_step = 0.001):
        i = 0
        fig,ax = plt.subplots()
        ax.set_xlim(0,self.size);ax.set_ylim(0,self.size)
        while i < frames:
            print(i)
            self.advect(time_step)
            self.diffuse(time_step,20)
            self.projection(20)

            i+=1

