from unittest import TestCase
import river_tools
import pandas as pd
import matplotlib.pyplot as plt
import math
import logging
import os
import matplotlib.animation as animation

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
        df = pd.read_excel(f, 'Sheet1')
        xs = river_tools.CrossSection(df=df, sta_fld='station', z_fld='z', n_fld='roughness', panel_fld='pannel')
        for level in sorted(xs.z):
            xs.set_level(level)
            fig = xs.plot(level)
            fig.savefig(os.path.join(out_folder, '%s.jpg' % level))
        self.fail()






