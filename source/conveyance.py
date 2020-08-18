import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def conveyance_at_level(xs, ys, ns, ps, level):
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
    p = list(ps)  # panel markers
    n = list(ns)
    results = []
    if level <= np.min(ys):  # water level below the cross section
        return pd.DataFrame(results)

    # add panel marker to the first point
    p[0] = 1
    panel = x[0]

    for i in range(1, len(x)):
        logging.debug('processing line segment(starting from 1):%s' % i)
        pt1 = (x[i - 1], y[i - 1], n[i - 1], p[i - 1])
        pt2 = (x[i], y[i], n[i], p[i])
        if pt1[3] == 1: # panel marker
            panel = pt1[0]

        line_result = calculate_line_segment(level, pt1, pt2, panel)

        if line_result:
            wp, ws, rn, area, panel = line_result
            wp_len = get_length([wp[0][0], wp[1][0]], [wp[0][1], wp[1][1]])
            ws_len = get_length([ws[0][0], ws[1][0]], [ws[0][1], ws[1][1]])
            results.append(dict(zip(['wp', 'ws', 'wp_len', 'ws_len',  'n', 'area', 'panel'], [wp, ws, wp_len, ws_len, rn, area, panel])))

        else:
            logging.info('water below the line')
    return pd.DataFrame(results)


def conveyance_curve(xs, ys, ns, ps):
    level_list = sorted(set(list(ys)))
    con_list = []
    for i, level in enumerate(level_list):
        results = conveyance_at_level(xs, ys, ns, ps, level)
        if i == 0:
            if (results is None) or results.empty:
                summary = {'level': level}
                for v in ['k', 'wp', 'ws', 'area']:
                    summary[v] = 0
                con_list.append(summary)
        else:
            if not results.empty:

                data = get_panel(results)
                summary = {'level': level}
                for v in ['k', 'wp', 'ws', 'area']:
                    summary[v] = data[v].sum()
                con_list.append(summary)
    df = pd.DataFrame(con_list)
    df['el'] = df['level']
    df['level'] = df['level'] - min(ys)
    return df

def get_panel(results):
    rows = []
    for panel in results['panel'].unique():
        data = results.loc[results['panel'] == panel]
        area = data['area'].sum()
        wp = data['wp_len'].sum()
        ws = data['ws_len'].sum()
        n_average = (data['n'] * data['wp_len']).sum() / float(wp)
        k = 1.49 / n_average * area * np.power(area / wp, 2 / 3.0)
        rows.append({'area': area, 'wp': wp, 'n': n_average, 'ws': ws, 'k': k, 'panel': panel})
    df = pd.DataFrame(rows)
    return df







def calculate_line_segment(level, pt1, pt2, panel):
    """
    Given a line segment from the cross secton and the water level, caluclate the parameters
    :param level: water level
    :param pt1: first point
    :param pt2: second point
    :param panel: the panel this line segment is in
    :return: (wp, ws, n, area, panel)
    """

    x0, y0, n0, p0 = pt1
    x1, y1, n1, p1 = pt2
    if p0 == 1:
        panel = x0  # panel is defined as x0
    level_status = ''  # compare level and the line segment
    if level < min(y0, y1):
        level_status = 'below'
    elif level > max(y0, y1):
        level_status = 'above'
    elif min(y0, y1) <= level <= max(y0, y1):
        level_status = 'cross'
    # special case
    # TODO: make sure my assumptions are right
    if y1 == y0:  # flat line, should work the same
        # if y0 == level, should it count as wp or not?
        pass
    if x0 == x1:  # vertical line, should be the same
        pass

    if level_status == 'below':
        return None
    elif level_status == 'cross':
        if x0 == x1:
            pt = [x0, level]
        else:
            if y0 == y1: #flat
                pt = [x1, y1]
            else:
                pt = line_intersection([[x0, y0], [x1, y1]], [[x0, level], [x1, level]])
        if y0 > y1:
            if y1 == level:
                return None
            wp = [(pt[0], pt[1]), (x1, y1)]  # wetted perimeter
            ws = [(pt[0], level), (x1, level)]  # water surface
            n = n0
            area = (x1 - pt[0]) * (level - y1) / 2.0
            return wp, ws, n, area, panel
        else:
            if y0 == level and (y0 != y1):
                return None
            wp = [(x0, y0), (pt[0], pt[1])]  # wetted perimeter
            ws = [(x0, level), (pt[0], level)]  # water surface
            n = n0
            area = (pt[0] - x0) * (level - y0) / 2.0
            return wp, ws, n, area, panel
    elif level_status == 'above':
        wp = [(x0, y0), (x1, y1)]  # wetted perimeter
        ws = [(x0, level), (x1, level)]  # water surface
        n = n0
        area = ((level - y0) + (level - y1)) * (x1 - x0) / 2.0
        return wp, ws, n, area, panel


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


def plot_conveyance_curve(df, title='conveyance curve', xlabel='Conveyance', ylabel='Depth(ft)'):
    sns.set()

    df['dk'] = df['k'].diff()
    df['dlevel'] = df['level'].diff()
    df['dkdlevel'] = df['dk']/df['dlevel']
    fig = plt.plot(df['dk'], df['level'], marker='o')
    fig = plt.plot(df['dkdlevel'], df['level'], marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return fig