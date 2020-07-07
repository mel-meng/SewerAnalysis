from unittest import TestCase
import river_tools
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import matplotlib.animation as animation
import logging
import seaborn as sns


logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

class TestRiverTools(TestCase):
    def setUp(self) -> None:
        f = './test/river/cross_section.xlsx'
        self.f = f
    def test_get_area(self):
        f = self.f
        df = pd.read_excel(f, 'Sheet1')
        area = river_tools.get_area(df['station'], df['z'])
        assert(area == 350)

    def test_get_length(self):
        f = self.f
        df = pd.read_excel(f, 'Sheet1')
        p = river_tools.get_length(df['station'], df['z'])
        assert(math.fabs(p - 53.17424803612103) < 0.01)

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
        assert(math.fabs(xs.area - 521.2892089281231) < 0.001)
        assert(math.fabs(xs.wp - 163.25403385373312) < 0.001)

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








