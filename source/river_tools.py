from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import logging
import seaborn as sns

# logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
#                     datefmt='%Y-%m-%d:%H:%M:%S',
#                     level=logging.DEBUG)
from pandas import DataFrame


def read_icm_cross_section_survey_section_array_csv(csv_path):
    """
    read the ICM csv export for cross section in the following format
ObjectTable	id	X	Y	Z	roughness_N	new_panel
FieldDescription	ID	X coordinate	Y coordinate	Bed level	Roughness Manning's n	New panel
UserUnits		m	m	m AD
hw_cross_section_survey_section_array	1093.1	546618.976	4804330.849	291.4439087	0.013
		546628.1402	4804327.693	291.1368103	0.013

    :param csv_path:
    :return: a dataframe
    """
    xs_name = None
    xs_list = {}
    rows = []
    with open(csv_path) as o:
        i = 0
        for l in csv.reader(o):
            if i == 0:
                header = l
            if i == 2:
                units = l
            if i >= 3:
                # X	Y	Z	roughness_N	new_panel
                for idx in [2, 3, 4, 5, 6]:
                    # try:
                    if True:
                        l[idx] = float(l[idx])
                    # except Exception:
                    #     l[idx] = 0
                    #


                if len(l[1]) > 0:
                    if xs_name is None:
                        # first cross section
                        xs_name = l[1]
                        logging.info('first xs: %s' % xs_name)
                    else:  # start of another xs
                        xs_list[xs_name] = pd.DataFrame(rows)
                        logging.info('xs saved: %s' % xs_name)
                        rows = []
                        xs_name = l[1]
                        logging.info('next xs: %s' % xs_name)
                rows.append(dict(zip(header, l)))
            i += 1
        # add the last one
        xs_list[xs_name] = pd.DataFrame(rows)
        logging.info('xs saved: %s' % xs_name)
    return xs_list


def icm_xs_add_offset(xs_list):
    for xs in xs_list.copy():
        df = xs_list[xs]
        for fld in df.columns:
            if fld in ['X', 'Y', 'Z', 'roughness_N', 'new_panel']:
                df[fld] = pd.to_numeric(df[fld], errors='roughness_N')
        df['length'] = df.loc[:, ['X', 'Y']].diff().apply(lambda x: np.sqrt(x['X'] ** 2 + x['Y'] ** 2), axis=1).fillna(
            0)
        df['offset'] = df['length'].cumsum()
        xs_list[xs] = df
    return xs_list


def plot_xs(df, xs_name, fig=None, axes=None):
    # TODO: add flood level alex
    sns.set()
    if fig:
        pass
    else:
        fig, axes = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.001})
    # sns.lineplot(x='offset', y='Z', data=df, ax=axes[1])
    axes[1].plot(df['offset'], df['Z'])
    axes[1].set_ylabel('Bed Elevation')
    axes[1].set_xlabel('Offset')
    zmax = max(df['Z'])
    zmin = min(df['Z'])
    for x in df.loc[df['new_panel'] == 1, 'offset'].values:
        axes[1].plot([x, x], [zmin, zmax], linestyle='--', color='grey')
    axes[0].step(df['offset'], df['roughness_N'], where='post')
    axes[0].set_ylabel('Roughness')
    fig.suptitle(xs_name)
    return fig


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
        plt.plot([min(x), max(x)], [level, level], marker='o', linestyle='dashed', color='blue')
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


def get_wetted_parameter(df):
    """
    a ordered list of points defining segments of the cross sections
    get polyline length
    :param x: a list of x of the polyline
    :param y: a list of y of the polyline
    :param n: a list of mannin's n
    :return:
    """
    # TODO: write test case
    # TODO: for W shaped xs, the water is separated in two channels, and need to remove the water surface length.
    # df = pd.DataFrame({'offset': x, 'Z': z, 'roughness_N': n})
    df['length'] = df.loc[:, ['X', 'Y']].diff().apply(lambda x: np.sqrt(x['X'] ** 2 + x['Y'] ** 2), axis=1).fillna(0)
    df['offset'] = df['length'].cumsum()

    df_segments = pd.DataFrame({'offset': df['offset'].values[:-1],
                                'n': df['roughness_N'].values[:-1],
                                'length': df['length'].values[1:]})

    n_average = np.sum(df_segments['n'] * df_segments['length']) / np.sum(df_segments['length'])
    wp = np.sum(df_segments.loc[df_segments['n'] > 0, 'length'].values)

    return n_average, wp


def get_conveyance(df, depth):
    """
    df is the cross section dataframe
    X	Y	Z	roughness_N	new_panel
    :param depth: water depth in the cross section
    :return:
    """
    # TODO: write test case
    # TODO: for W shaped xs, the water is separated in two channels, and need to remove the water surface length.
    df = df.copy()
    df['panel_name'] = ''
    df['length'] = df.loc[:, ['X', 'Y']].diff().apply(lambda x: np.sqrt(x['X'] ** 2 + x['Y'] ** 2), axis=1).fillna(0)
    df['offset'] = df['length'].cumsum()

    df_segments = pd.DataFrame({'offset': df['offset'].values[:-1],
                                'dx': df['offset'].diff().values[1:],
                                'Y': df['Y'].values[:-1],
                                'dy': df['Y'].diff().values[1:],
                                'new_panel': df['offset'].values[:-1],
                                'roughness_N': df['roughness_N'].values[:-1],
                                'length': df['length'].values[1:]})
    # set mark panel
    panel_list = []
    for idx, r in df_segments.iterrows():
        offset = r['offset']

        if idx == 0:
            panel = offset
            panel_list.append(panel)
        else:
            if r['new_panel'] == 1:
                panel = offset
                panel_list.append(panel)
        r['panel_name'] = panel
    # wp calculation depends on where the panel is.
    if len(panel_list) == 1:
        # single panel, keep both ends
        pass
    elif len(panel_list) == 2:
        # two panels, left and right
        pass
    else:  # more than two
        # left, middle, right
        pass

    n_average = np.sum(df_segments['n'] * df_segments['length']) / np.sum(df_segments['length'])
    wp = np.sum(df_segments.loc[df_segments['n'] > 0, 'length'].values)

    return n_average, wp


def get_panel_conveyance(df_segments, depth, panel_type):
    """

    :param df_segments:
    :param depth:
    :param panel_type: 'left', 'middle', 'right', 'single'
    :return:
    """

    area = 0
    for idx, r in df_segments.iterrows():
        da = (r['Y'] + r['Y'] + r['dy']) * r['dx'] / 2.0


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
            if i == 0:  # first point
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
                    # add point 1
                    line.append([x1, y1])

                elif level < min(y1, y0):
                    pass
                elif (level > y0 and level < y1):  # or (level > y1 and level < y0):
                    line.append([x0, y0])
                    line.append(line_intersection([[x0, y0], [x1, y1]], [[pt_left[0], level], [pt_right[0], level]]))
                elif (level > y1 and level < y0):
                    line.append(line_intersection([[x0, y0], [x1, y1]], [[pt_left[0], level], [pt_right[0], level]]))
                    line.append([x1, y1])
                elif (level > y1 and level > y0):
                    line.append([x0, y0])
                    line.append([x1, y1])

    return line


def cut_cross_section(xs, ys, ns, ps, level):
    """
    given water level, cut the cross section with only lines touching the water
    :param xs: list of offset
    :param ys: list of z
    :param ns: list of manning's n
    :param ps: list of pannel markers
    :param level: water level
    :return: a list of lines each line is a list of [[x, y, n] ....]
    """
    area = 0
    perimeter = 0
    lines = []
    line = []
    x = list(xs)
    y = list(ys)
    n = list(ns)
    p = list(ps)
    pt_left = [x[0], y[0]]
    pt_right = [x[-1], y[-1]]

    if level > pt_left[1]:  # left side is lower than level
        x = [pt_left[0]] + x
        y = [level] + y
        n = [n[0]] + n
        if p[0] == 1:
            p[0] = 0
            p = [1] + p

    if level > pt_right[1]:  # right side is lower than level
        x = x + [pt_right[0]]
        y = y + [level]
        n = n + [n[-1]]
        if p[-1] == 1:
            p[-1] = 0
            p = p + [1]

    if level < min(y):
        # level below the crosse section
        pass
    else:
        for i in range(len(x)):
            if i == 0:  # first point
                pass
            else:
                x0 = x[i - 1]
                y0 = y[i - 1]
                n0 = n[i - 1]
                p0 = p[i - 1]

                x1 = x[i]
                y1 = y[i]
                n1 = n[i]
                p1 = p[i]
                if level == y0:
                    # add point 0
                    line.append([x0, y0, n0, p0])
                elif level == y1:
                    # add point 1
                    line.append([x1, y1, n1, p1])

                elif level < min(y1, y0):
                    pass
                elif (level > y0 and level < y1):  # xs going uphill
                    line.append([x0, y0, n0, p0])
                    pt = line_intersection([[x0, y0], [x1, y1]], [[pt_left[0], level], [pt_right[0], level]])
                    line.append([pt[0], pt[1], 0, p0])  # set the n to 0 for hitting the bank
                    lines.append(line)
                    line = []
                elif (level > y1 and level < y0):  # xs going downhill
                    pt = line_intersection([[x0, y0], [x1, y1]], [[pt_left[0], level], [pt_right[0], level]])
                    line.append([pt[0], pt[1], n0, p0])
                    line.append([x1, y1, n1, p1])
                elif (level > y1 and level > y0):  # no intersection
                    line.append([x0, y0, n0, p0])
                    line.append([x1, y1, n1, p1])
    if level >= pt_right[1]:
        lines.append(line)
    return lines



def cut_xs(xs, ys, ns, level):
    """
    given water level, cut the cross section with only lines touching the water
    :param xs: list of offset
    :param ys: list of z
    :param ns: list of manning's n
    :param level: water level
    :return: the wetted perimeter line, for lines that is under the ground, the n is set as 0, None if level is lower than the xs
    """
    # make a copy of all the lines
    line = []
    x = list(xs)
    y = list(ys)
    if level < np.min(ys):
        return None
    n = list(ns)
    pt_left = [x[0], y[0]]  # the first point of the xs
    pt_right = [x[-1], y[-1]]  # the last point of the xs

    if level > pt_left[1]:  # left side is lower than level
        x = [pt_left[0]] + x
        y = [level] + y
        n = [0] + n # assume n=0 for the added wall

    if level > pt_right[1]:  # right side is lower than level
        x = x + [pt_right[0]]
        y = y + [level]
        n = n + [0] # assume n=0 for the added wall

    if level < min(y):
        # level below the XS
        logging.warning('level is lower than the bottom of the xs')
    else:
        for i in range(len(x)):
            if i == 0:  # first point
                pass
            else:
                # current point
                x0 = x[i - 1]
                y0 = y[i - 1]
                n0 = n[i - 1]
                logging.debug('process segment %s, %s' % (x0, y0))
                # previous point
                x1 = x[i]
                y1 = y[i]
                n1 = n[i]

                # if level == y0:  # level is the same as current point
                #     # add point 0
                #     line.append([x0, y0, n0])
                #     logging.info('same as point %s, %s' % (x0, y0))
                # elif level == y1:  # level is the same as previous point
                #     # add point 1
                #     line.append([x1, y1, n1])

                if level == y0:
                    line.append([x0, y0, n0])
                elif y0 < level < y1:  # xs going uphill, hitting the bank and line stops
                    line.append([x0, y0, n0])
                    logging.debug('uphill add %s, %s' % (x0, y0))
                    pt = line_intersection([[x0, y0], [x1, y1]], [[pt_left[0], level], [pt_right[0], level]])
                    line.append([pt[0], pt[1], 0])  # set the n to 0 for hitting the bank
                    logging.debug('uphill intersection %s, %s' % (pt[0], pt[1]))
                elif y1 < level < y0:  # xs going downhill, leaving the bank line starts
                    pt = line_intersection([[x0, y0], [x1, y1]], [[pt_left[0], level], [pt_right[0], level]])
                    line.append([pt[0], pt[1], n0])
                    logging.debug('downhill intersection %s, %s' % (pt[0], pt[1]))
                    # line.append([x1, y1, n1])
                elif level >= y1 and level >= y0:  # no intersection
                    line.append([x0, y0, n0])
                    logging.debug('both ends under level add %s, %s' % (x0, y0))
        # check the last point

        if line and (level >= y1):
            line.append([x1, y1, n1])
    return pd.DataFrame(line, columns=['offset', 'Z', 'roughness_N'])


def xs_to_panel(df):
    """
    df is the cross section dataframe
    X	Y	Z	roughness_N	new_panel
    :param df: xs as dataframe
    :return:each panel as a dataframe
    """
    df = df.copy()
    df['panel_name'] = ''
    df['length'] = df.loc[:, ['X', 'Y']].diff().apply(lambda x: np.sqrt(x['X'] ** 2 + x['Y'] ** 2), axis=1).fillna(0)
    df['offset'] = df['length'].cumsum()
    df.iloc[0]['new_panel'] = 0  # no panel markers on start and end of the xs
    df.iloc[-1]['new_panel'] = 0
    results = {}

    # set mark panel
    panel_list = []
    rows = []
    for idx, r in df.iterrows():
        offset = r['offset']

        if idx == 0:  # first point
            panel = '{:.2f}'.format(offset)
            panel_list.append(panel)
            rows.append(r)

        else:
            if r['new_panel'] == 1:
                # save the panel
                rows.append(r)
                results[panel] = {'data': pd.DataFrame(rows)}
                # start a new panel
                panel = '{:.2f}'.format(offset)
                panel_list.append(panel)
                rows = [r]
            else:
                rows.append(r)

    if panel not in results.keys():
        results[panel] = {'data': pd.DataFrame(rows)}


    if len(panel_list) == 1:
        # single panel
        panel = panel_list[0]
        results[panel]['type'] = 'single'
    else:
        results[panel_list[0]]['type'] = 'left'
        results[panel_list[-1]]['type'] = 'right'
        for panel in panel_list[1:-1]:
            results[panel]['type'] = 'middle'

    df_list = []
    for panel in results:
        position = results[panel]['type']
        df = results[panel]['data']
        df['type'] = position
        df['panel_name'] = panel
        df_list.append(df)
    return pd.concat(df_list).reset_index()



def panel_conveyance(df, depth, type):
    """

    :param df: panel list of table
    :param depth: water depth
    :param type: left, right, middle, single
    :return: None if depth < xs, otherwise the results
    """
    xs = df['offset'].values
    ys = df['Z'].values
    if depth < np.min(ys):
        return None
    ns = df['roughness_N'].values
    # set n = 0 for panel points
    if type == 'single':
        # no changes to manning's n
        pass
    elif type == 'left':
        ns[-1] = 0
    elif type == 'right':
        ns[0] = 0
    elif type == 'middle':
        ns[0] = 0
        ns[-1] = 0

    df_wet = cut_xs(xs, ys, ns, depth)
    df_wet['length'] = df_wet.loc[:, ['offset', 'Z']].diff().apply(lambda x: np.sqrt(x['offset'] ** 2 + x['Z'] ** 2), axis=1).fillna(0)

    df_segments = pd.DataFrame({'offset': df_wet['offset'].values[:-1],
                                'n': df_wet['roughness_N'].values[:-1],
                                'doffset':df_wet['offset'].diff().values[1:],
                                'length': df_wet['length'].values[1:]})

    df_n = df_segments.loc[df_segments['n']>0, ] # remove any segments that are not wet
    n_average = np.sum(df_n['n'] * df_n['length']) / np.sum(df_n['length']) # average n
    wp = np.sum(df_segments.loc[df_segments['n'] > 0, 'length'].values) # wetted perimeter
    width = np.max(df_wet['offset']) - np.min(df_wet['offset']) - np.sum(df_segments.loc[df_segments['n'] == 0, 'doffset'].values) # water surface width
    area = get_area(df_wet['offset'].values, df_wet['Z'].values)
    k = 1.49/n_average*area*np.power(area/wp, 2/3.0)
    d = depth - np.min(df_wet['Z'])
    return [d, n_average, wp, width, area, k, df_wet]


def plot_conveyance(df, xs_name, fig=None):

    rows = []

    for depth in sorted(df['Z'].unique()):
        v = xs_conveyance(df, depth)
        rows.append(v)
    # d, n_average, wp, width, area, k
    df_convey = pd.DataFrame(rows, columns=['depth', 'wp', 'width', 'area', 'k']).fillna(0)
    # TODO: add flood level alex
    sns.set()
    if fig:
        pass
    else:
        fig, axes = plt.subplots(2, 2, sharey='row', sharex='col',
                                 gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.001}, figsize=(15,12))
    # sns.lineplot(x='offset', y='Z', data=df, ax=axes[1])
    axes[1][0].plot(df['offset'], df['Z'])
    axes[1][0].set_ylabel('Bed Elevation')
    axes[1][0].set_xlabel('Offset')
    zmax = max(df['Z'])
    zmin = min(df['Z'])
    for x in df.loc[df['new_panel'] == 1, 'offset'].values:
        axes[1][0].plot([x, x], [zmin, zmax],  linestyle='--', color='grey')
    axes[0][0].step(df['offset'], df['roughness_N'], where='post')
    axes[0][0].set_ylabel('Roughness')
    axes[1][1].plot(df_convey['k'], df_convey['depth'], marker='o')
    axes[1][1].set_xlabel('conveyance')
    axes[1][1].set_ylabel('Water Level')
    # plot diff
    dk = df_convey['k'].diff()/df_convey['depth'].diff()
    dk = dk.fillna(0)
    print(dk)
    sf = (np.max(df_convey['k'].values) - np.min(df_convey['k'].values))/(np.max(dk.values) - np.min(dk.values))
    dk = dk*sf
    dk = dk - np.min(dk.values)

    axes[1][1].plot(dk, df_convey['depth'], color='red', marker='x', linestyle='--')
    fig.suptitle(xs_name)
    plt.tight_layout()
    return fig



def cut_xs_new(xs, ys, ns, ps, level):
    """
    given water level, cut the cross section with only lines touching the water
    :param xs: list of offset
    :param ys: list of z
    :param ns: list of manning's n
    :param ps: list of panel markers, 0 no, 1 yes
    :param level: water level
    :return: the wetted perimeter line, for lines that is under the ground, the n is set as 0, None if level is lower than the xs
    """
    # make a copy of all the lines
    panels = []
    line = []
    x = list(xs)
    y = list(ys)
    p = list(ps) # panel markers
    n = list(ns)

    # no panel markers on start and end of the xs
    p[0] = 0
    p[-1] = 0

    logging.debug(p)
    logging.debug(n)

    if level < np.min(ys):
        return None

    pt_left = [x[0], y[0]]  # the first point of the xs
    pt_right = [x[-1], y[-1]]  # the last point of the xs

    if level > pt_left[1]:  # left side is lower than level
        x = [pt_left[0]] + x
        y = [level] + y
        n = [0] + n # assume n=0 for the added wall
        p = [0] + p



    if level < min(y):
        # level below the XS
        logging.warning('level is lower than the bottom of the xs')
    else:
        for i in range(len(x)):
            logging.debug('processing point:%s' % i)
            if i == 0:  # first point
                pass
            else:
                # current point
                x0 = x[i - 1]
                y0 = y[i - 1]
                n0 = n[i - 1]
                p0 = p[i - 1]
                logging.debug('process segment %s, %s' % (x0, y0))
                logging.debug('%s,%s,%s,%s' % (x0, y0, n0, p0))
                # previous point
                x1 = x[i]
                y1 = y[i]
                n1 = n[i]
                p1 = p[i]

                if level == y0:

                    if level <= y0: # uphill
                        line.append([x0, y0, 0])
                        if p0 == 1:  # new marker
                            panels.append(line)
                            line = [[x0, y0, 0]]
                    else:
                        line.append([x0, y0, n0])
                        if p0 == 1:  # new marker
                            panels.append(line)
                            line = [[x0, y0, n0]]

                elif y0 < level <= y1:  # xs going uphill, hitting the bank and line stops
                    line.append([x0, y0, n0])
                    logging.debug('uphill add %s, %s' % (x0, y0))
                    pt = line_intersection([[x0, y0], [x1, y1]], [[pt_left[0], level], [pt_right[0], level]])
                    if p0 == 1:  # new marker
                        line.append([x0, y0, n0])
                        line.append([x0, level, 0])
                        panels.append(line)
                        line = [[x0, level, 0], [x0, y0, n], [pt[0], pt[1], 0]]
                    line.append([pt[0], pt[1], 0])
                    logging.debug('uphill intersection %s, %s' % (pt[0], pt[1]))
                elif y1 <= level < y0:  # xs going downhill, leaving the bank line starts
                    pt = line_intersection([[x0, y0], [x1, y1]], [[pt_left[0], level], [pt_right[0], level]])
                    if p0 == 1:
                        logging.debug('new panel')
                        line.append([x0, level, 0])
                        panels.append(line)
                        line = [[x0, level, 0]]
                    line.append([pt[0], pt[1], n0])
                    logging.debug('downhill intersection %s, %s' % (pt[0], pt[1]))
                elif level >= y1 and level >= y0:  # no intersection
                    line.append([x0, y0, n0])
                    logging.debug('both ends under level add %s, %s' % (x0, y0))
                    if p0 == 1:
                        logging.debug('new panel')
                        line.append([x0, level, 0])
                        panels.append(line)
                        line = [[x0, level, 0], [x0, y0, n0]]
    # check the last point
    if level >= y1:
        line.append([x1, y1, 0])
        line.append([x1, level, 0])

    panels.append(line)
    results = []
    for line in panels:
        results.append(pd.DataFrame(line, columns=['offset', 'Z', 'roughness_N']))
    return results

def conveyance(df_wet, depth):
    """
    :param df: panel list of table
    :param depth: water depth
    :return: None if depth < xs, otherwise the results
    """

    if depth < np.min(df_wet['Z'].values):
        return None

    df_wet['length'] = df_wet.loc[:, ['offset', 'Z']].diff().apply(lambda x: np.sqrt(x['offset'] ** 2 + x['Z'] ** 2), axis=1).fillna(0)
    df_segments = pd.DataFrame({'offset': df_wet['offset'].values[:-1],
                                'n': df_wet['roughness_N'].values[:-1],
                                'doffset':df_wet['offset'].diff().values[1:],
                                'length': df_wet['length'].values[1:]})
    df_n = df_segments.loc[df_segments['n'] > 0, ] # remove any segments that are not wet
    n_average = np.sum(df_n['n'] * df_n['length']) / np.sum(df_n['length']) # average n
    wp = np.sum(df_segments.loc[df_segments['n'] > 0, 'length'].values) # wetted perimeter
    width = np.max(df_wet['offset']) - np.min(df_wet['offset']) - np.sum(df_segments.loc[df_segments['n'] == 0, 'doffset'].values) # water surface width
    area = get_area(df_wet['offset'].values, df_wet['Z'].values)
    k = 1.49/n_average*area*np.power(area/wp, 2/3.0)
    d = depth - np.min(df_wet['Z'])
    return [d, n_average, wp, width, area, k]

def xs_conveyance_new(df, depth):
    xs = df['offset'].values
    ys = df['Z'].values
    ns = df['roughness_N'].values
    ps = df['new_panel'].values
    lines = cut_xs_new(xs, ys, ns, ps, depth)
    rows = []
    for line in lines:
        print(line)
        p = conveyance(line, depth)
        if p:
            rows.append(p)

    # d, n_average, wp, width, area, k, df_we
    df = pd.DataFrame(rows, columns=['d', 'n_average', 'wp', 'width', 'area', 'k'])
    s = df.sum()
    return depth, s['wp'], s['width'], s['area'], s['k']

