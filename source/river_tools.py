import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class CrossSection:
    def __init__(self, df, sta_fld='x', z_fld='z', n_fld='n', panel_fld='panel'):
        self.df = df
        self.x = df[sta_fld].values
        self.z = df[z_fld].values
        self.n = df[n_fld].values
        self.panel = df[panel_fld].values
        self.level = None
        self.wp = None
        self.wet_line = None
        self.area = None

    def set_level(self, level):
        self.level = level
        line = cross_section_level(self.x, self.z, level)
        x = [l[0] for l in line]
        y = [l[1] for l in line]
        self.area = get_area(x, y)
        self.wp = get_length(x, y)
        self.wet_line = line

    def plot(self, level):
        self.set_level(level)
        fig = plt.figure()
        line = self.wet_line
        x = [l[0] for l in line]
        y = [l[1] for l in line]
        # the xs
        plt.plot(self.x, self.z, color='grey')
        # the wet part
        plt.plot(x, y, marker='o', linestyle='dashed', color='blue', markersize=2)

        # water surface
        plt.plot([min(x), max(x)], [level, level],  marker='o', linestyle='dashed', color='blue')
        plt.title('level = %s, area=%.2f, wp=%.2f' % (level, self.area, self.wp))
        return fig


def get_area(x, y):
    """
    get polygon area
    x: a list of x coordinate
    y: a list of y coordinate
    """
    # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_length(x, y):
    """
    get polyline length
    :param x: a list of x of the polyline
    :param y: a list of y of the polyline
    :return:
    """
    # TODO: for W shaped xs, the water is separated in two channels, and need to remove the water surface length.
    points = np.array([[x, y] for x, y in zip(x, y)])
    d = np.diff(points, axis=0)
    return np.sqrt((d ** 2).sum(axis=1)).sum()


def line_intersection(line1, line2):
    """
    get intersection of two lines
    :param line1: x1, y1
    :param line2:  x2, y2
    :return: x, y of the intersection
    """
    # https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines/20679579
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y



def cross_section_level(xs, ys, level):
    """
    given water level, area, wetted perimeter of the cross section
    :param x: list of x
    :param y: list of y
    :param level: water level
    :return: the wetted line of the crossection [[x1, y1], [x2, y2]...]
    """
    area = 0
    perimeter = 0
    line = []
    x = list(xs)
    y = list(ys)
    pt_left = [x[0], y[0]]
    pt_right = [x[-1], y[-1]]

    if level > pt_left[1]:
        x = [pt_left[0]] + x
        y = [level] + y
    if level > pt_right[1]:
        x = x + pt_right[0]
        y = y + [level]

    if level < min(y):
        # level below the crosse section
        pass
    else:
        for i in range(len(x)):
            if i == 0: # first point
                pass
            else:
                x0 = x[i - 1]
                y0 = y[i - 1]
                x1 = x[i]
                y1 = y[i]
                if level == y0:
                    # add point 0
                    line.append([x0, y0])
                elif level == y1:
                    #add point 1
                    line.append([x1, y1])

                elif level < min(y1, y0):
                    pass
                elif (level > y0 and level < y1):#or (level > y1 and level < y0):
                    line.append([x0, y0])
                    line.append(line_intersection([[x0, y0], [x1, y1]], [[pt_left[0], level], [pt_right[0], level]]))
                elif (level > y1 and level < y0):
                    line.append(line_intersection([[x0, y0], [x1, y1]], [[pt_left[0], level], [pt_right[0], level]]))
                    line.append([x1, y1])
                elif (level > y1 and level > y0):
                    line.append([x0, y0])
                    line.append([x1, y1])

    return line


