# -*- coding: utf-8 -*-
# plot.py
import matplotlib.pyplot as plt
import numpy as np


def line_demo():
    x = np.linspace(-1, 1, 50)
    y = 2 * x + 1
    # plt.figure()


plt.plot(x, y)
plt.show()


def histgram_demo(tdata):
    # plt.figure()
    plt.hist(tdata, bins=50, color='steelblue')
    plt.show()


if __name__ == '__main__':
    # histgram_demo()
    line_demo()