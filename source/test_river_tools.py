import logging
import math
import os
from unittest import TestCase

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import river_tools

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)


class TestRiverTools(TestCase):
    def setUp(self) -> None:
        f = './test/river/cross_section.xlsx'
        self.f = f

    def test_get_area(self):
        f = self.f
        df = pd.read_excel(f, 'irregular')
        area = river_tools.get_area(df['station'], df['z'])
        assert (area == 350)

        df = pd.read_excel(f, 'triangle')
        area = river_tools.get_area(df['station'], df['z'])
        assert (area == 25)

        df = pd.read_excel(f, 'trapzoid')
        area = river_tools.get_area(df['station'], df['z'])
        assert (area == (10+30)*10/2)

    def test_get_length(self):
        f = self.f
        df = pd.read_excel(f, 'irregular')
        p = river_tools.get_length(df['station'], df['z'])
        assert (math.fabs(p - 53.17424804) < 0.01)

    def test_cross_section_level(self):
        f = self.f
        df = pd.read_excel(f, 'Sheet2')
        ims = []
        fig = plt.figure()
        for level in sorted(df['z']):
            # level = 12
            line = river_tools.cross_section_level(df['station'].values, df['z'].values, level)
            x = [l[0] for l in line]
            y = [l[1] for l in line]
            print(line)

            im = plt.imshow(df['station'].values, df['z'].values, animated=True)
            ims.append([im])
            # im = plt.imshow([min(x),max(x)],[level, level], animated=True)
            # ims.append([im])
            # im = plt.imshow(x,y, animated=True)
            # ims.append([im])
            # plt.plot(x, y)
            # plt.title(level)
            # plt.show()
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=1000)
        plt.show()
        self.fail()

    def test_cross_section_area(self):
        f = self.f
        df = pd.read_excel(f, 'Sheet2')
        xs = river_tools.CrossSection(df=df, sta_fld='station', z_fld='z', n_fld='roughness', panel_fld='pannel')
        xs.set_level(369.33)
        print(xs.area, xs.wp)
        assert (math.fabs(xs.area - 521.2892089281231) < 0.001)
        assert (math.fabs(xs.wp - 163.25403385373312) < 0.001)

    def test_cross_section_plot(self):
        f = self.f
        df = pd.read_excel(f, 'Sheet2')
        xs = river_tools.CrossSection(df=df, sta_fld='station', z_fld='z', n_fld='roughness', panel_fld='pannel')
        level = 369.33
        xs.set_level(level)
        fig = xs.plot(level)
        plt.show()
        self.fail()

    def test_cross_section_plot_animation(self):
        f = self.f
        out_folder = './test/river/tmp'
        df = pd.read_excel(f, 'Sheet2')
        xs = river_tools.CrossSection(df=df, sta_fld='station', z_fld='z', n_fld='roughness', panel_fld='pannel')
        i = 1
        for level in sorted(xs.z):
            xs.set_level(level)
            fig = xs.plot(level)
            fig.savefig(os.path.join(out_folder, '%04d.jpg' % i))
            i += 1
        self.fail()

    def test_read_icm_csv(self):
        csv_path = './test/river/icm/xs_sample_csv_export/xs_sample_cross_section_survey_section_array.csv'
        output_folder = './test/river/icm/tmp'
        xs_list = river_tools.read_icm_cross_section_survey_section_array_csv(csv_path)
        xs_list = river_tools.icm_xs_add_offset(xs_list)
        for xs_name in ['BalzerC_Reach1_182_580', 'BalzerC_Reach1_4_000']:
            df_check = pd.read_excel('./test/river/icm/xs_sample.xlsx', '%s_xs' % xs_name)
            df = xs_list[xs_name]
            np.testing.assert_almost_equal(df_check['X coordinate (m)'].values, df['X'].values, 3)
            np.testing.assert_almost_equal(df_check['Y coordinate (m)'].values, df['Y'].values, 3)
            np.testing.assert_almost_equal(df_check['Bed level (m AD)'].values, df['Z'].values, 3)
            np.testing.assert_almost_equal(df_check["Roughness Manning's n"].values, df['roughness_N'].values, 3)
            np.testing.assert_almost_equal(df_check['New panel'].values, df['new_panel'].values, 3)
            np.testing.assert_almost_equal(df_check['Offset (m)'].values, df['offset'].values, 3)

    def test_batch_print(self):
        csv_path = './test/river/icm/xs_sample_csv_export/xs_sample_cross_section_survey_section_array.csv'
        output_folder = './test/river/icm/tmp'
        xs_list = river_tools.read_icm_cross_section_survey_section_array_csv(csv_path)
        xs_list = river_tools.icm_xs_add_offset(xs_list)

        for k in xs_list:
            out_csv = os.path.join(output_folder, '%s.csv' % str(k).replace('.', '_'))
            xs_list[k].to_csv(out_csv, index_label='no')
            fig = river_tools.plot_xs(xs_list[k], k)
            out_png = os.path.join(output_folder, '%s.png' % str(k).replace('.', '_'))
            fig.savefig(out_png)
        self.fail('check output folder for batch conversions')


    def test_plot_xs(self):
        csv_path = './test/river/icm/xs.csv'
        output_folder = './test/river/icm/tmp'
        df = pd.read_csv(csv_path)
        xs_name = 'BalzerC_Reach1_4_000'
        fig = river_tools.plot_xs(df, xs_name)
        out_png = os.path.join(output_folder, '%s.png' % 'test')
        fig.savefig(out_png)
        self.fail()

    def test_get_wetted_perimeter(self):
        csv_path = './test/river/icm/xs.csv'
        output_folder = './test/river/icm/tmp'
        df = pd.read_csv(csv_path)
        xs_name = 'BalzerC_Reach1_4_000'
        n, wp = river_tools.get_wetted_parameter(df)
        print(n, wp)
        self.fail()

    def test_cut_cross_section_line(self):
        csv_path = './test/river/icm/xs.csv'
        output_folder = './test/river/icm/tmp'
        df = pd.read_csv(csv_path)
        xs = df['offset'].values
        ys = df['Z'].values
        ns = df['roughness_N'].values
        ps = df['new_panel'].values
        all_checks = {311.8: [{'sta': 4.42043478377046, 'width': 105.24753803024845},
                              {'sta': 142.78310982485263, 'width': 10.481538460915516},
                              {'sta': 192.5851427915494, 'width': 3.1768285709524093}],
                      312: [{'sta': 20.44083333254138, 'width': 105.24753803024845},
                            {'sta': 142.78310982485263, 'width': 10.481538460915516},
                            {'sta': 192.5851427915494, 'width': 3.1768285709524093}],
                      311.6: [{'sta': 20.44083333254138, 'width': 105.24753803024845},
                              {'sta': 142.78310982485263, 'width': 10.481538460915516},
                              {'sta': 192.5851427915494, 'width': 3.1768285709524093}]
                      }
        for level in [
            #     311.8,   # higher than righ
            # 312,  # higher than both left and right
            311.6]:  # below than both sides

            checks = all_checks[level]
            lines = river_tools.cut_cross_section(xs, ys, ns, ps, level)
            plt.plot(df['offset'], df['Z'])
            plt.plot([np.min(df['offset']), np.max(df['offset'])], [level, level])

            for idx, line in enumerate(lines):
                df2 = pd.DataFrame({'x': [x for x, y, n, p in line], 'z': [y for x, y, n, p in line]})
                width = line[-1][0] - line[0][0]
                check = checks[idx]
                self.assertAlmostEqual(check['sta'], line[0][0])
                self.assertAlmostEqual(check['width'], width)

                plt.plot(df2['x'], df2['z'])

            plt.show()

    def test_cut_cross_section_line(self):
        csv_path = './test/river/icm/xs.csv'
        output_folder = './test/river/icm/tmp'
        df = pd.read_csv(csv_path)
        river_tools.plot_xs(df, 'line')
        plt.show()

        xs = df['offset'].values
        ys = df['Z'].values
        ns = df['roughness_N'].values
        ps = df['new_panel'].values
        all_checks = {311.8: [{'sta': 4.42043478377046, 'width': 105.24753803024845},
                              {'sta': 142.78310982485263, 'width': 10.481538460915516},
                              {'sta': 192.5851427915494, 'width': 3.1768285709524093}],
                      312: [{'sta': 20.44083333254138, 'width': 105.24753803024845},
                            {'sta': 142.78310982485263, 'width': 10.481538460915516},
                            {'sta': 192.5851427915494, 'width': 3.1768285709524093}],
                      311.6: [{'sta': 20.44083333254138, 'width': 105.24753803024845},
                              {'sta': 142.78310982485263, 'width': 10.481538460915516},
                              {'sta': 192.5851427915494, 'width': 3.1768285709524093}]
                      }
        for level in [
            #     311.8,   # higher than righ
            # 312,  # higher than both left and right
            311.6]:  # below than both sides

            checks = all_checks[level]
            lines = river_tools.cut_cross_section(xs, ys, ns, ps, level)

            fig, axes = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.001})
            axes[1].plot(df['offset'], df['Z'])
            axes[1].plot([np.min(df['offset']), np.max(df['offset'])], [level, level])
            for idx, line in enumerate(lines):
                df2 = pd.DataFrame(line, columns=['offset', 'Z', 'roughness_N', 'new_panel'])
                river_tools.plot_xs(df2, 'line', fig, axes)
            plt.show()

    def test_cut_xs1(self):

        df = pd.read_excel(self.f, 'irregular_single')

        level = 15 # at top

        xs = df['offset'].values
        ys = df['Z'].values
        ns = df['roughness_N'].values

        df2 = river_tools.cut_xs(xs, ys, ns, level)
        df2['new_panel'] = 0

        np.testing.assert_almost_equal(df2.loc[:, ['offset', 'Z']].to_numpy(), df.loc[:, ['offset', 'Z']].to_numpy(), 3)

    def test_cut_xs2(self):
        df = pd.read_excel(self.f, 'irregular_single')
        level = 25  # at top
        xs = df['offset'].values
        ys = df['Z'].values
        ns = df['roughness_N'].values
        df2 = river_tools.cut_xs(xs, ys, ns, level)
        df2['new_panel'] = 0
        df2_check = pd.read_excel(self.f, 'irregular_single_25')
        np.testing.assert_almost_equal(df2.to_numpy(), df2_check.to_numpy(), 3)

    def test_cut_xs3(self):
        df = pd.read_excel(self.f, 'irregular_single')
        level = 10  # at top
        xs = df['offset'].values
        ys = df['Z'].values
        ns = df['roughness_N'].values
        df2 = river_tools.cut_xs(xs, ys, ns, level)
        df2['new_panel'] = 0
        print(df2)
        df2_check = pd.read_excel(self.f, 'irregular_single_10')
        np.testing.assert_almost_equal(df2.to_numpy(), df2_check.to_numpy(), 3)

    def test_cut_xs4(self):
        df = pd.read_excel(self.f, 'irregular_single')
        level = 12  # at top
        xs = df['offset'].values
        ys = df['Z'].values
        ns = df['roughness_N'].values
        df2 = river_tools.cut_xs(xs, ys, ns, level)
        df2['new_panel'] = 0
        print(df2)
        df2_check = pd.read_excel(self.f, 'irregular_single_12')
        np.testing.assert_almost_equal(df2.to_numpy(), df2_check.to_numpy(), 3)
        # fig, axes = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.001})
        # river_tools.plot_xs(df2, 'line', fig, axes)
        #
        # axes[1].plot(df['offset'], df['Z'], linestyle='--')
        # axes[1].plot([np.min(df['offset']), np.max(df['offset'])], [level, level], linestyle='--')
        #
        # plt.show()

    def test_cut_xs5(self):
        df = pd.read_excel(self.f, 'irregular_multiple')
        level = 13  # at top
        level = 10
        level = 6
        xs = df['offset'].values
        ys = df['Z'].values
        ns = df['roughness_N'].values
        df2 = river_tools.cut_xs(xs, ys, ns, level)
        df2['new_panel'] = 0
        # df2.to_csv(r'C:\Users\Mel.Meng\Documents\GitHub\SewerAnalysis\source\test\river\tmp.csv')
        df2_check = pd.read_excel(self.f, 'irregular_multiple_6')
        np.testing.assert_almost_equal(df2.to_numpy(), df2_check.to_numpy(), 2)

    def test_cut_xs6(self):
        df = pd.read_excel(self.f, 'irregular_multiple')
        level = 13  # at top
        level = 10
        xs = df['offset'].values
        ys = df['Z'].values
        ns = df['roughness_N'].values
        df2 = river_tools.cut_xs(xs, ys, ns, level)
        df2['new_panel'] = 0
        # df2.to_csv(r'C:\Users\Mel.Meng\Documents\GitHub\SewerAnalysis\source\test\river\tmp.csv')
        df2_check = pd.read_excel(self.f, 'irregular_multiple_10')
        np.testing.assert_almost_equal(df2.to_numpy(), df2_check.to_numpy(), 2)

    def test_cut_xs7(self):
        df = pd.read_excel(self.f, 'irregular_multiple')
        level = 13  # at top

        xs = df['offset'].values
        ys = df['Z'].values
        ns = df['roughness_N'].values
        df2 = river_tools.cut_xs(xs, ys, ns, level)
        df2['new_panel'] = 0
        # print(df2)
        # df2.to_csv(r'C:\Users\Mel.Meng\Documents\GitHub\SewerAnalysis\source\test\river\tmp.csv')
        df2_check = pd.read_excel(self.f, 'irregular_multiple_13')
        np.testing.assert_almost_equal(df2.to_numpy(), df2_check.to_numpy(), 2)

    def test_cut_xs8(self):
        df = pd.read_excel(self.f, 'irregular_multiple')
        level = 8  # at top

        xs = df['offset'].values
        ys = df['Z'].values
        ns = df['roughness_N'].values
        df2 = river_tools.cut_xs(xs, ys, ns, level)
        df2['new_panel'] = 0
        # df2.to_csv(r'C:\Users\Mel.Meng\Documents\GitHub\SewerAnalysis\source\test\river\tmp.csv')
        df2_check = pd.read_excel(self.f, 'irregular_multiple_8')
        np.testing.assert_almost_equal(df2.to_numpy(), df2_check.to_numpy(), 2)
        # fig, axes = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.001})
        # river_tools.plot_xs(df2, 'line', fig, axes)
        #
        # axes[1].plot(df['offset'], df['Z'], linestyle='--')
        # axes[1].plot([np.min(df['offset']), np.max(df['offset'])], [level, level], linestyle='--')
        #
        # plt.show()

    def test_xs_to_panel(self):
        df = pd.read_excel(self.f, 'irregular_multiple_panels')
        panels = river_tools.xs_to_panel(df)
        # panels.to_csv(r'C:\Users\Mel.Meng\Documents\GitHub\SewerAnalysis\source\test\river\panel.csv')
        panel_check = pd.read_excel(self.f, 'irregular_multiple_panels_check')
        # assert (panel_check.equals(panels))
        for fld in panel_check.columns:
            print(fld)
            if fld =='type':
                # print(panel_check[fld])
                # print(panels[fld])
                assert(panel_check[fld].equals(panels[fld]))
            elif fld == 'panel_name':
                pass
            else:
                np.testing.assert_almost_equal(panel_check[fld].values, panels[fld].values, 3)
        # river_tools.plot_xs(panels, 'test')
        # plt.show()
        # import seaborn as sns
        # sns.lineplot(x='offset', y='Z', style='panel_name', data=panels)
        # plt.show()

    def test_xs_conveyance(self):
        df = pd.read_excel(self.f, 'irregular_multiple_panels')
        rows = []
        for depth in [0,2,5,8,10,11,12,15]:
            v = river_tools.xs_conveyance(df, depth)
            rows.append(v)
        # d, n_average, wp, width, area, k
        df_convey = pd.DataFrame(rows, columns=['wp', 'width', 'area', 'k']).fillna(0)
        df_convey.to_csv(r'C:\Users\Mel.Meng\Documents\GitHub\SewerAnalysis\source\test\river\convey.csv')
        # from ICM results
        df_check = pd.read_excel(self.f, 'irregular_multiple_panels_cv')
        for fld in ['wp', 'area', 'width', 'wp']:
            np.testing.assert_almost_equal(df_convey[fld].values, df_check[fld].values, 3)
        # csv_path = './test/river/icm/xs.csv'
        # output_folder = './test/river/icm/tmp'
        # df = pd.read_csv(csv_path)
        # level = 311.623
        # panels = river_tools.xs_conveyance(df, level)
        # print(panels)
        # print(panels.sum())
        # self.fail()

    def test_panel_conveyance(self):
        # TODO: need to get this one tested using the testing section for single one
        df = pd.read_excel(self.f, 'irregular_multiple_panels')
        rows = []
        for depth in range(-5,20):
            v = river_tools.panel_conveyance(df, depth, 'single')
            if v:
                rows.append(v)
            else:
                logging.warning('no results returned for depth:%s' % depth)
        # d, n_average, wp, width, area, k
        df_convey = pd.DataFrame(rows, columns=['d', 'n_average', 'wp', 'width', 'area', 'k'])
        df_convey.to_csv(r'C:\Users\Mel.Meng\Documents\GitHub\SewerAnalysis\source\test\river\convey.csv')
        self.fail()

    def test_panel_conveyance(self):

        df = pd.read_excel(self.f, 'irregular_multiple_panels')
        rows = []
        for depth in [0,2,5,8,10,11,12,15]:
            v = river_tools.panel_conveyance(df, depth, 'single')
            rows.append(v[:-1])
        # d, n_average, wp, width, area, k
        df_convey = pd.DataFrame(rows, columns=['d', 'n_average', 'wp', 'width', 'area', 'k'])
        df_convey.to_csv(r'C:\Users\Mel.Meng\Documents\GitHub\SewerAnalysis\source\test\river\convey.csv')
        # from ICM results
        df_check = pd.read_excel(self.f, 'irregular_multiple_panels_res')
        for fld in ['area', 'width', 'wp']:
            np.testing.assert_almost_equal(df_convey[fld].values, df_check[fld].values, 3)

    def test_panel_conveyance_xs(self):
        # TODO: still don't get the same conveyance k from ICM
        # When using the same n, it gets close to .99. But with multiple n values, it can get 1.2.
        df = pd.read_excel(self.f, 'xs_single_n')
        rows = []
        for depth in [0,2,5,8,10,11,12,15]:
            v = river_tools.panel_conveyance(df, depth, 'single')
            rows.append(v[:-1])
        # d, n_average, wp, width, area, k
        df_convey = pd.DataFrame(rows, columns=['d', 'n_average', 'wp', 'width', 'area', 'k']).fillna(0)
        df_convey.to_csv(r'C:\Users\Mel.Meng\Documents\GitHub\SewerAnalysis\source\test\river\convey.csv')
        # from ICM results
        df_check = pd.read_excel(self.f, 'xs_single_n_check')
        for fld in ['area', 'width', 'wp']:
            np.testing.assert_almost_equal(df_convey[fld].values, df_check[fld].values, 3)

    def test_panel_conveyance15(self):
        # TODO: need to get this one tested using the testing section for single one
        df = pd.read_excel(self.f, 'irregular_multiple_panels')
        rows = []
        for depth in [15]:
            v = river_tools.panel_conveyance(df, depth, 'single')
            # rows.append(v[:-1])
            df_wet = v[-1]
            df_wet.to_csv(r'C:\Users\Mel.Meng\Documents\GitHub\SewerAnalysis\source\test\river\wet.csv')
            df_wet['new_panel'] = 0
            print('width:', v[3])

            fig, axes = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.001})
            river_tools.plot_xs(df_wet, 'line', fig, axes)

            axes[1].plot(df['offset'], df['Z'], linestyle='--')
            axes[1].plot([np.min(df['offset']), np.max(df['offset'])], [depth, depth], linestyle='--')
            plt.show()

        self.fail()
    def test_plot_conveyance(self):
        df = pd.read_excel(self.f, 'irregular_multiple_panels')
        fig = river_tools.plot_conveyance(df, 'test')
        plt.show()
        self.fail()

    def test_batch_plot_conveyance(self):
        csv_path = './test/river/icm/xs_sample_csv_export/xs_sample_cross_section_survey_section_array.csv'
        output_folder = './test/river/icm/tmp'
        xs_list = river_tools.read_icm_cross_section_survey_section_array_csv(csv_path)
        xs_list = river_tools.icm_xs_add_offset(xs_list)
        for xs in xs_list:
            df = xs_list[xs]
            df['new_panel'] = 0
            print(xs)
            print(df)
            fig = river_tools.plot_conveyance(df, xs)
            fig.savefig(os.path.join(output_folder, '%s.png' % xs))

    def test_cut_xs_new(self):
        ## TODO: need to add a flat segment see how that works. it is not working correctly
        df = pd.read_excel(self.f, 'irregular_multiple_panels')
        level = 15  # at top
        for level in [0,2,8,5,10,11,12,15, 20]:
            xs = df['offset'].values
            ys = df['Z'].values
            ns = df['roughness_N'].values
            ps = df['new_panel'].values
            lines = river_tools.cut_xs_new(xs, ys, ns, ps, level)
            fig, axes = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.001})
            river_tools.plot_xs(df, level, fig, axes)
            # axes[1].plot([np.min(df['offset']), np.max(df['offset'])], [level, level], linestyle='--')
            for line in lines:
                axes[1].plot(line['offset'], line['Z'], linewidth=3, linestyle='--', label='a')
                axes[0].step(line['offset'], line['roughness_N'], where='post')
                # axes[1].plot([np.min(df['offset']), np.max(df['offset'])], [level, level], linestyle='--')
            plt.legend()

            plt.show()

    def test_xs_conveyance(self):
        df = pd.read_excel(self.f, 'irregular_multiple_panels')
        rows = []
        for depth in [0,2,5,8,10,11,12,15]:
            v = river_tools.xs_conveyance_new(df, depth)
            rows.append(v)
        # d, n_average, wp, width, area, k
        df_convey = pd.DataFrame(rows, columns=['depth', 'wp', 'width', 'area', 'k']).fillna(0)
        df_convey.to_csv(r'C:\Users\Mel.Meng\Documents\GitHub\SewerAnalysis\source\test\river\convey.csv')
        # from ICM results
        df_check = pd.read_excel(self.f, 'irregular_multiple_panels_cv')
        for fld in ['wp', 'area', 'width', 'wp']:
            np.testing.assert_almost_equal(df_convey[fld].values, df_check[fld].values, 3)