import numpy as np
import matplotlib.pyplot as plt

class heat2d:
    def __init__(self,size = 1,n = 128,init_temp = 273,init_cond=1):
        #definign vector field size and spacing between two adjacent grid cells
        self.vfs = n+2
        self.spacing = size/n

        #defining temparature and conductivity matrices
        self.temp = np.ones((self.vfs,self.vfs))*init_temp
        self.conductivity = np.ones((self.vfs,self.vfs))*init_cond

    def set_temp(self,x1,x2,y1,y2,t):
        self.temp[x1:x2,y1:y2] = t
    def set_cond(self,x1,x2,y1,y2,a):
        self.conductivity[x1+1:x2-1,y1+1:y2-1] = a
        self.conductivity[x1+1:x2-1,y1] = self.conductivity[x1+1:x2-1,y1-1]

    def pois(self,temp):
        return ((temp[1:-1,0:-2] + temp [1:-1,2:] + temp[0:-2,1:-1] + temp[2:,1:-1]) - 4*temp[1:-1,1:-1])/self.spacing**2
    def pois_temp(self):
        return self.pois(self.temp)

    def diffuse(self, iter=20, time_step=0.01):
        new = self.temp.copy()
        k = self.conductivity*time_step/self.spacing**2
        den = 1+ 4* k[1:-1,1:-1]

        for _ in range(iter):
            neighbors = (new[0:-2, 1:-1] +
                         new[2:, 1:-1] +
                         new[1:-1, 0:-2] +
                         new[1:-1, 2:])
            new[1:-1,1:-1] = (self.temp[1:-1,1:-1]  + k[1:-1,1:-1]*neighbors)/den

        return new


    def simulate_visual(self,frames =100,time_step=0.01,
                        cnt_src = []):
        fig, ax = plt.subplots()
        for j in cnt_src:
            self.set_temp(*j)
        img = ax.imshow(self.temp, cmap='jet')
        for i in range(frames):
            print(f'Frame {i+1}')
            for j in cnt_src:
                self.set_temp(*j)
            self.temp = self.diffuse(time_step=time_step)
            img.set_data(self.temp)
            plt.pause(0.01)
        plt.show()
    def simulate_krdf4(self,frames = 4,time_step = 0.01,cnt_src = []):
        fig, ax = plt.subplots()
        for j in cnt_src:
            self.set_temp(*j)
        img = ax.imshow(self.temp,cmap = 'jet')
        for i in range(frames):
            print(f'(Frames {i+1}')
            for j in cnt_src:
                self.set_temp(*j)





