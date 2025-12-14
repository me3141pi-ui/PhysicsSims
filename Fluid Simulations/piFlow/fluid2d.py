import numpy as np
import matplotlib.pyplot as plt
class fluid2d:
    def __init__(self,n = 128,size = 1,viscosity = 1):
        self.n = n;self.size = size;self.viscosity = viscosity
        #defining vector field size (n+2) . the +2 term accounts for boundary cells with velocity 0
        self.vfs = n + 2
        #defining spacing between consecutive cells
        self.spacing = size/n

        #defining the velocity grids
        self.ux = np.zeros((self.vfs,self.vfs));self.uy = np.zeros((self.vfs,self.vfs))

        #defining x coord grid and y coord grid for our vector space
        coord_range = np.linspace(0,n+1,n+2)
        self.pg_y,self.pg_x = np.meshgrid(coord_range,coord_range)

        #defining smoke_density
        self.smoke = np.zeros((self.vfs,self.vfs))

        '''
        NOTE:
        1)In our simulation we will consider the central nxn grid to be our working area. The boundary cells with all parameters 0 are used to
        enforce the boundary conditions.

        '''

    '''given a scalar field p , and set of x coords and y coords ix and iy (where ix and iy may not necessirally be integers, and they have same shape) \t
    the billerp vector function returns the billinear interpolated predicted values of p at ix and iy'''
    def _bilerp_vector(self,u,ix,iy):
        #index values clipped between 1 and n to ensure our indexes dont lie outside our domain of working
        ixd = np.clip(ix,1,self.vfs-2);iyd = np.clip(iy,1,self.vfs-2)
        i = np.floor(ixd).astype(int);j = np.floor(iyd).astype(int)
        s = ixd - i;t = iyd - j
        sp = 1-s;tp = 1-t

        up = sp*tp*u[i,j] + sp*t*u[i,j+1] + s*tp*u[i+1,j] + s*t*u[i+1,j+1]
        return up

    #DEFINING BASIC VECTOR CALCULUS OPERATIONS
    def _div(self,px,py):
        return (px[2:,1:-1]-px[0:-2,1:-1])/(2*self.spacing) + (py[1:-1,2:] - py[1:-1,0:-2])/(2*self.spacing)
    def _grad(self,p):
        return (p[2:,1:-1] - p[0:-2,1:-1])/(2*self.spacing) , (p[1:-1,2:] - p[1:-1,0:-2])/(2*self.spacing)
    def _curl(self,ux,uy):
        return (uy[2:,1:-1] - uy[0:-2,1:-1])/(2*self.spacing)-(ux[1:-1,2:] - ux[1:-1,0:-2])/(2*self.spacing)

    #DEFINING VISUALISATION FUNCTIONS
    def swirl_intensity(self):
        return np.abs(self._curl(self.ux,self.uy))
    def speed(self):
        return (self.ux**2 + self.uy**2)**0.5

    '''DEFINING A GOOD FUNCTION TO SIMPLY DEFINE THE VELOCITIES FOR GIVEN REGIONS'''
    def set_velocity(self,x1=0,x2=0,y1=0,y2=0,ux=0,uy=0):
        self.ux[x1:x2,y1:y2] = ux
        self.uy[x1:x2,y1:y2] = uy

    def set_smoke(self,x1=0,x2=0,y1=0,y2=0,d = 5):
        self.smoke[x1:x2,y1:y2] = d

    '''HANDLING BOUNDARY FOR WALLS'''

    ######IMPLEMENTING WALLS#######
    def wallbounce(self, bounce_factor=1.0):
        self.ux[0, 1:-1] = np.where(self.ux[1, 1:-1] < 0, -self.ux[1, 1:-1] * bounce_factor, self.ux[1, 1:-1])
        self.uy[0, 1:-1] = self.uy[1, 1:-1]

        self.ux[-1, 1:-1] = np.where(self.ux[-2, 1:-1] > 0, -self.ux[-2, 1:-1] * bounce_factor, self.ux[-2, 1:-1])
        self.uy[-1, 1:-1] = self.uy[-2, 1:-1]

        self.uy[1:-1, 0] = np.where(self.uy[1:-1, 1] < 0, -self.uy[1:-1, 1] * bounce_factor, self.uy[1:-1, 1])
        self.ux[1:-1, 0] = self.ux[1:-1, 1]

        self.uy[1:-1, -1] = np.where(self.uy[1:-1, -2] > 0, -self.uy[1:-1, -2] * bounce_factor, self.uy[1:-1, -2])
        self.ux[1:-1, -1] = self.ux[1:-1, -2]

        self.ux[0, 0] = 0.5 * (self.ux[1, 0] + self.ux[0, 1])
        self.uy[0, 0] = 0.5 * (self.uy[1, 0] + self.uy[0, 1])

        self.ux[-1, 0] = 0.5 * (self.ux[-2, 0] + self.ux[-1, 1])
        self.uy[-1, 0] = 0.5 * (self.uy[-2, 0] + self.uy[-1, 1])

        self.ux[0, -1] = 0.5 * (self.ux[1, -1] + self.ux[0, -2])
        self.uy[0, -1] = 0.5 * (self.uy[1, -1] + self.uy[0, -2])

        self.ux[-1, -1] = 0.5 * (self.ux[-2, -1] + self.ux[-1, -2])
        self.uy[-1, -1] = 0.5 * (self.uy[-2, -1] + self.uy[-1, -2])

    '''IMPLEMENTING THE 3 STEPS FROM THE PAPER'''
    ######IMPLEMENTING ADVECTION STEP######
    def advect(self,time_step):
        oldx = self.pg_x - self.ux*time_step/self.spacing
        oldy = self.pg_y - self.uy*time_step/self.spacing

        ux = self._bilerp_vector(self.ux,oldx,oldy)
        uy = self._bilerp_vector(self.uy,oldx,oldy)

        fwd_gridx = oldx + ux*time_step/self.spacing
        fwd_gridy = oldy + uy*time_step/self.spacing
        error1 = self.pg_x - fwd_gridx
        error2 = self.pg_y - fwd_gridy
        oldx += error1 * 0.5
        oldy += error2 * 0.5

        self.ux = self._bilerp_vector(self.ux, oldx, oldy)
        self.uy = self._bilerp_vector(self.uy, oldx, oldy)

    def smoke_advect(self, time_step , decay = 0.98):
        oldx = self.pg_x - self.ux * time_step / self.spacing
        oldy = self.pg_y - self.uy * time_step / self.spacing

        ux_src = self._bilerp_vector(self.ux, oldx, oldy)
        uy_src = self._bilerp_vector(self.uy, oldx, oldy)

        fwd_gridx = oldx + ux_src * time_step / self.spacing
        fwd_gridy = oldy + uy_src * time_step / self.spacing

        error1 = self.pg_x - fwd_gridx
        error2 = self.pg_y - fwd_gridy

        oldx += error1 * 0.5
        oldy += error2 * 0.5

        self.smoke = self._bilerp_vector(self.smoke, oldx, oldy) * decay


    ######IMPLEMENTING DIFFUSION STEP######
    def diffuse(self,time_step,iter = 20):
        a = self.viscosity*time_step/self.spacing**2
        ux_new = self.ux.copy();uy_new = self.uy.copy()
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


    def simulate_visual(self, frames=100, time_step=0.001,cnt_src = [],quiver = False,walls = False,
                        cmap = 'jet'):
        for j in cnt_src:
            self.set_velocity(*j)
        fig, ax = plt.subplots()
        img = ax.imshow(self.speed(), cmap=cmap)
        if quiver:
            q = ax.quiver(self.pg_x[::10,::10],self.pg_y[::10,::10],self.ux[::10,::10],self.uy[::10,::10],color = 'white',scale = 40,pivot = 'mid')
        for i in range(frames):
            print(f"Frame {i+1} | Time {time_step*(i+1)}")
            for j in cnt_src:
                self.set_velocity(*j)
            if quiver:
                q.set_UVC(self.ux[::10,::10],self.uy[::10,::10])
            self.advect(time_step)
            self.diffuse(time_step, 20)
            self.projection(20)
            if walls:
                self.wallbounce()
            img.set_data(self.speed())
            plt.pause(0.001)
        plt.show()

    #simulating smoke
    def simulate_smoke(self, frames=100, time_step=0.001, cnt_src=[], quiver=False, walls=False,
                       cnt_src_smk=[],decay = 0.98,
                       cmap='Greens'):
        for j in cnt_src:
            self.set_velocity(*j)
        for j in cnt_src_smk:
            self.set_smoke(*j)
        fig, ax = plt.subplots()

        img = ax.imshow(self.smoke, cmap=cmap)
        if quiver:
            q = ax.quiver(self.pg_x[::10, ::10], self.pg_y[::10, ::10],
                          self.ux[::10, ::10],
                          self.uy[::10, ::10],
                          color='white', scale=40, pivot='mid')
        for i in range(frames):
            print(f"Frame {i + 1} | Time {time_step * (i + 1)}")
            for j in cnt_src:
                self.set_velocity(*j)

            if quiver:
                q.set_UVC(self.ux[::10, ::10], self.uy[::10, ::10])
            self.advect(time_step)
            self.smoke_advect(time_step,decay  =decay)
            for j in cnt_src_smk:
                self.set_smoke(*j)
            self.diffuse(time_step, 20)
            self.projection(20)
            if walls:
                self.wallbounce()

            img.set_data(self.smoke)
            plt.pause(0.001)
        plt.show()
