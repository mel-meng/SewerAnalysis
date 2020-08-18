from unittest import TestCase
import math
import conveyance
import matplotlib.pyplot as plt
import pandas as pd

pt1 = (0, 0)
pt2 = (2, 2)
pt3 = (2, 0)
pt4 = (0, 2)

go_up = (pt1, pt2)
go_down = (pt4, pt3)
vertical_up = (pt3, pt2)
vertical_down = (pt4, pt1)
flat = (pt1, pt3)

def plot_results(results, df):
    plt.plot(df['offset'].values, df['Z'].values, 'g--')
    for idx, r in results.iterrows():
        ws = r['ws']
        wp = r['wp']
        plt.plot([ws[0][0], ws[1][0]], [ws[0][1], ws[1][1]], 'b-')
        plt.plot([wp[0][0], wp[1][0]], [wp[0][1], wp[1][1]], 'r-')

def plot_line(line, name):
    x = [x[0] for x in line]
    y = [x[1] for x in line]
    plt.plot(x, y)
    plt.title(name)
    # plt.show()

def check_result(level, line, check, msg, show_plot=False):
    pt1, pt2 = line
    panel = 0
    if show_plot:
        plot_line(line, 'go_up')
        plt.plot([pt1[0], pt2[0]], [level, level])
        plt.show()
    pt1 = pt1 + (0.014, 0)
    pt2 = pt2 + (0.014, 0)
    result = conveyance.calculate_line_segment(level, pt1, pt2, panel)
    print(result)
    assert (result == check)


class Test(TestCase):
    def test_calculate_line_segment_go_up(self):
        tests = [
            [1.5, go_up, 'intersect go up', ([(0, 0), (1.5, 1.5)], [(0, 1.5), (1.5, 1.5)], 0.014, 1.125, 0)],
                 [0, go_up, 'touch bottom go up', None],
                 [2, go_up, 'touch top go up', ([(0, 0), (2, 2)], [(0, 2), (2, 2)], 0.014, 2.0, 0)],
                 [2.5, go_up, 'above go up', ([(0, 0), (2, 2)], [(0, 2.5), (2, 2.5)], 0.014, 3.0, 0)]
                 ]
        for level, line, msg, check in tests:
            check_result(level, line, check, msg, show_plot=False)

    def test_calculate_line_segment_go_down(self):
        tests = [
            [1.5, go_down, 'intersect go up', ([(0.5, 1.5), (2, 0)], [(0.5, 1.5), (2, 1.5)], 0.014, 1.125, 0)],
                 [0, go_down, 'touch bottom go up', None],
                 [2, go_down, 'touch top go up', ([(0, 2), (2, 0)], [(0, 2), (2, 2)], 0.014, 2.0, 0)],
                 [2.5, go_down, 'above go up', ([(0, 2), (2, 0)], [(0, 2.5), (2, 2.5)], 0.014, 3.0, 0)]
                 ]
        for level, line, msg, check in tests:
            check_result(level, line, check, msg, show_plot=False)

    def test_calculate_line_segment_vup(self):
        tests = [
            [1.5, vertical_up, 'intersect go up', ([(2, 0), (2, 1.5)], [(2, 1.5), (2, 1.5)], 0.014, 0.0, 0)],
                 [0, vertical_up, 'touch bottom go up', None],
                 [2, vertical_up, 'touch top go up', ([(2, 0), (2, 2)], [(2, 2), (2, 2)], 0.014, 0.0, 0)],
                 [2.5, vertical_up, 'above go up', ([(2, 0), (2, 2)], [(2, 2.5), (2, 2.5)], 0.014, 0.0, 0)]
                 ]
        for level, line, msg, check in tests:
            check_result(level, line, check, msg, show_plot=False)

    def test_calculate_line_segment_vdown(self):
        tests = [
            # [1.5, vertical_down, 'intersect go up', ([(0, 1.5), (0, 0)], [(0, 1.5), (0, 1.5)], 0.014, 0.0, 0)],
                 [0, vertical_down, 'touch bottom go up', None],
                 # [2, vertical_down, 'touch top go up', ([(0, 2), (0, 0)], [(0, 2), (0, 2)], 0.014, 0.0, 0)],
                 # [2.5, vertical_down, 'above go up', ([(0, 2), (0, 0)], [(0, 2.5), (0, 2.5)], 0.014, 0.0, 0)]
                 ]
        for level, line, msg, check in tests:
            check_result(level, line, check, msg, show_plot=False)

    def test_calculate_line_segment_flat(self):
        tests = [

                 [0, flat, 'on top', ([(0, 0), (2, 0)], [(0, 0), (2, 0)], 0.014, 0.0, 0)],
                 [-1, flat, 'below', None],
                 [1.5, flat, 'above go up', ([(0, 0), (2, 0)], [(0, 1.5), (2, 1.5)], 0.014, 3.0, 0)]
                 ]
        for level, line, msg, check in tests:
            check_result(level, line, check, msg, show_plot=False)
    def test_conveyance_curve3(self):
        df = pd.read_excel('./test/river/cross_section3.xlsx', 'xs_single')
        xs = df['offset'].values
        ys = df['Z'].values
        ns = df['roughness_N'].values
        ps = df['new_panel'].values
        curve = conveyance.conveyance_curve(xs, ys, ns, ps)

        # curve['level'] = curve['level'] - min(curve['level'].values)
        curve.to_csv('./test/river/cross_section3_curve.csv')
        check_df = pd.read_excel('./test/river/cross_section3.xlsx', 'icm_check')
        check_df.index = check_df['level']
        curve.index = curve['level']
        for level in curve['level'].unique():
            for fld in ['area', 'wp', 'ws']:
                self.assertAlmostEqual(curve.loc[level, fld], check_df.loc[level, fld], 2, 'level=%s, field=%s' % (level, fld))
                # assert(math.fabs(curve.loc[level, fld] - check_df.loc[level, fld]) < 0.01)

        # FIXME:I cannot get k correct.

    def test_conveyance_curve2(self):
        df = pd.read_excel('./test/river/xs_test2.xlsx', 'xs_single')
        xs = df['offset'].values
        ys = df['Z'].values
        ns = df['roughness_N'].values
        ps = df['new_panel'].values
        curve = conveyance.conveyance_curve(xs, ys, ns, ps)
        curve.to_csv('./test/river/xs_test2_curve.csv')
        check_df = pd.read_excel('./test/river/xs_test2.xlsx', 'icm_check')
        check_df.index = check_df['level_ori'].apply(lambda x: '%.3f' % x)
        curve.index = curve['level'].apply(lambda x: '%.3f' % x)
        for level in curve.index.unique():
            for fld in ['area', 'wp', 'ws']:
                if level in check_df.index:
                    self.assertAlmostEqual(curve.loc[level, fld], check_df.loc[level, fld], 2,
                                       'level=%s, field=%s' % (level, fld))
                    print('good: %s' % level)
                else:
                    print('skipe: %s' % level)

        # FIXME:I cannot get k correct.
    def test_conveyance_curve(self):
        df = pd.read_excel('./test/river/xs_test.xlsx', 'xs_single')
        xs = df['offset'].values
        ys = df['Z'].values
        ns = df['roughness_N'].values
        ps = df['new_panel'].values
        curve = conveyance.conveyance_curve(xs, ys, ns, ps)
        check_df = pd.read_excel('./test/river/xs_test.xlsx', 'icm_check')
        check_df.index = check_df['level']
        curve.index = curve['level']
        for level in [2, 5, 8, 10, 11, 12, 15]:
            for fld in ['area', 'wp', 'ws']:
                assert (math.fabs(curve.loc[level, fld] - check_df.loc[level, fld]) < 0.01)

        # FIXME:I cannot get k correct.

    def test_calculate_conveyance(self):
        # good example for debugging
        df = pd.read_excel('./test/river/xs_test.xlsx', 'xs_single')
        xs = df['offset'].values
        ys = df['Z'].values
        ns = df['roughness_N'].values
        ps = df['new_panel'].values
        con_list = []
        for level in [0,2,5,8,10,11,12,15]:
            results = conveyance.conveyance_at_level(xs, ys, ns, ps, level)
            if not results.empty:
                # plot_results(results, df)
                # plt.show()
                # results.to_csv('./test/river/%s.csv' % level, index=False)
                data = conveyance.get_panel(results)
                # data.to_csv('./test/river/%s_panels.csv' % level, index=False)
                summary = {'level': level}
                for v in ['k', 'wp', 'ws', 'area']:
                    summary[v] = data[v].sum()
                con_list.append(summary)
        curve = pd.DataFrame(con_list) #.to_csv('./test/river/xs_curve.csv')
        check_df = pd.read_excel('./test/river/xs_test.xlsx', 'icm_check')
        check_df.index = check_df['level']
        curve.index = curve['level']
        for level in [2,5,8,10,11,12,15]:
            for fld in ['area', 'wp', 'ws']:
                assert(math.fabs(curve.loc[level, fld] - check_df.loc[level, fld]) < 0.01)

        # FIXME:I cannot get k correct.

    def test_check_lines(self):
        plot_line(go_up, 'go_up')
        plot_line(flat, 'flat')
        plot_line(go_down, 'go_down')
        plot_line(vertical_up, 'vertical_up')
        plot_line(vertical_down, 'vertical_down')

