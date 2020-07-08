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
                    level=logging.INFO)


class TestRiverTools(TestCase):
    def setUp(self) -> None:
        f = './test/river/cross_section.xlsx'
        self.f = f

    def test_get_area(self):
        f = self.f
        df = pd.read_excel(f, 'Sheet1')
        area = river_tools.get_area(df['station'], df['z'])
        assert (area == 350)

    def test_get_length(self):
        f = self.f
        df = pd.read_excel(f, 'Sheet1')
        p = river_tools.get_length(df['station'], df['z'])
        assert (math.fabs(p - 53.17424803612103) < 0.01)

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
        csv_path = './test/river/icm/icm_cross_section_survey_section_array.csv'
        output_folder = './test/river/icm/tmp'
        xs_list = river_tools.read_icm_cross_section_survey_section_array_csv(csv_path)
        xs_list = river_tools.icm_xs_add_offset(xs_list)
        for k in xs_list:
            out_csv = os.path.join(output_folder, '%s.csv' % str(k).replace('.', '_'))
            xs_list[k].to_csv(out_csv, index_label='no')
            fig = river_tools.plot_xs(xs_list[k], k)
            # fig = plt.figure()
            # df = xs_list[k]
            # plt.plot(df['offset'], df['Z'])
            # plt.title(k)
            out_png = os.path.join(output_folder, '%s.png' % str(k).replace('.', '_'))
            fig.savefig(out_png)

        self.fail()

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

    def test_cut_xs(self):
        csv_path = './test/river/icm/xs.csv'
        output_folder = './test/river/icm/tmp'
        df = pd.read_csv(csv_path)

        level = 311.6

        xs = df['offset'].values
        ys = df['Z'].values
        ns = df['roughness_N'].values

        df2 = river_tools.cut_xs(xs, ys, ns, level)
        df2['new_panel'] = 0
        # df2.to_csv(os.path.join(output_folder, 'xs_311_6.csv'))
        df2_check = pd.read_csv(os.path.join(output_folder, './../xs_311_6.csv'))
        np.testing.assert_almost_equal(df2.to_numpy(), df2_check.to_numpy())

        fig, axes = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.001})
        river_tools.plot_xs(df2, 'line', fig, axes)

        axes[1].plot(df['offset'], df['Z'], linestyle='--')
        axes[1].plot([np.min(df['offset']), np.max(df['offset'])], [level, level], linestyle='--')

        plt.show()

    def test_xs_to_panel(self):
        csv_path = './test/river/icm/xs.csv'
        output_folder = './test/river/icm/tmp'
        df = pd.read_csv(csv_path)

        panels = river_tools.xs_to_panel(df)
        fig, axes = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.001})

        for panel in panels:
            # print(panel)
            position = panels[panel]['type']
            panels[panel]['data'].plot(x='offset', y='roughness_N', ax=axes[0], drawstyle="steps-post")
            panels[panel]['data'].plot(x='offset', y='Z', ax=axes[1], label='%s(%s)' % (panel, position), marker='o')
        plt.show()
        self.fail()
