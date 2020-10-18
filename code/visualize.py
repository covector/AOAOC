import numpy as np
import matplotlib.pyplot as plt
from fourrooms import Fourrooms
from time import sleep

# 0 - Red
# 1 - Green
# 2 - Blue
# 3 - Black
class Visualization:
    def __init__(self, fRoom, args, nactions, colorList=[[255,0,0],[0,255,0],[0,0,255],[0,0,0]]):
        assert args.noptions <= len(colorList), "Length of color list must match number of options"
        self.colorList = colorList
        self.layout = fRoom.layout
        self.occupancy = fRoom.occupancy
        self.tostate = fRoom.tostate
        self.tocell = fRoom.tocell
        self.screen = np.array([list(map(lambda c: [0,0,0] if c=='w' else [255,255,255], line)) for line in self.layout.splitlines()])
        self.lastphi = None
        self.noptions = args.noptions
        self.nactions = nactions

    def showMap(self, phi, option):
        color = self.colorList[option]
        self._draw(self.lastphi, [255,255,255])
        self._draw(phi, color)
        self.lastphi = phi
        plt.figure(figsize=(5,5))
        plt.subplot(111)
        plt.imshow(self.screen, vmax=255, vmin=0)
        plt.show()
        sleep(0.05)

    def showAttention(self, options):
        x = np.array([i for i in range(self.nactions)])
        plt.plot(x, np.array([int(i != 0) for i in range(self.nactions)]), color=[1,1,1])
        for i in range(self.noptions):
            plt.plot(x, options[i].policy.attention.pmf(), color=np.array(self.colorList[i])/255.)
        plt.show()

    def showPref(self, weight): # policy_over_options.weightsP or options[index].weightsP for weight
        pref = np.zeros((13,13,3), dtype="int")
        for i in range(13):
            for j in range(13):
                if self.occupancy[i,j] == 0:
                    choice = np.argmax(weight[self.tostate[(i,j)],:])
                    pref[i,j] = np.array(self.colorList[choice])
                else:
                    pref[i,j] = np.array([255,255,255]) 
        plt.figure(figsize=(5,5))
        plt.subplot(111)
        plt.imshow(pref, vmax=255, vmin=0)
        plt.show()

    def resetMap(self, phi):
        self.screen = np.array([list(map(lambda c: [0,0,0] if c=='w' else [255,255,255], line)) for line in self.layout.splitlines()])
        self.lastphi = phi
        self._draw([62],[200,200,200])

    def _draw(self, phi, rgb):
        self.screen[self.tocell[phi[0]]] = np.array(rgb)