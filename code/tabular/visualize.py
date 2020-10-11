import numpy as np
import matplotlib.pyplot as plt
from fourrooms import Fourrooms
from time import sleep

class Display:
    def __init__(self):
        self.layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
#         self.layout = """\
# wwwwwwwwwwwww
# w           w
# w     w     w
# w     w     w
# w     w     w
# w     w     w
# w wwwww     w
# w     wwwww w
# w     w     w
# w     w     w
# w     w     w
# w           w
# wwwwwwwwwwwww
# """
        self.fr = Fourrooms(1234)
        self.screen = np.array([list(map(lambda c: [0,0,0] if c=='w' else [255,255,255], line)) for line in self.layout.splitlines()])
        self.lastphi = None

    def render(self, phi, option):
        if option == 0:
            color = [255,0,0]
        if option == 1:
            color = [0,255,0]
        if option == 2:
            color = [0,0,255]
        if option == 3:
            color = [0,0,0]
        self._draw(self.lastphi, [255,255,255])
        self._draw(phi, color)
        self.lastphi = phi
        # plt.figure(figsize=(5,5))
        # plt.subplot(111)
        # plt.imshow(self.screen, vmax=255, vmin=0)
        # plt.ion()
        # plt.show()
        # sleep(1)
        # plt.close()

    def reset(self, phi):
        self.screen = np.array([list(map(lambda c: [0,0,0] if c=='w' else [255,255,255], line)) for line in self.layout.splitlines()])
        self.lastphi = phi
        self._draw([62],[200,200,200])

    def _draw(self, phi, rgb):
        self.screen[self.fr.tocell[phi[0]]] = np.array(rgb)