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

    def test_icm_csv_to_coneyance_curve(self):
        csv_path = './test/river/icm/xs_sample_csv_export/xs_sample_cross_section_survey_section_array.csv'
        output_folder = './test/river/icm/tmp'
        xs_list = river_tools.read_icm_cross_section_survey_section_array_csv(csv_path)
        xs_list = river_tools.icm_xs_add_offset(xs_list)
        for k in xs_list:
            xs_name = str(k).replace('.', '_')
            out_csv = os.path.join(output_folder, '%s.csv' % xs_name)
            df = xs_list[k]
            df.to_csv(out_csv, index_label='no')
            fig = river_tools.plot_xs(xs_list[k], k)
            out_png = os.path.join(output_folder, '%s.png' % xs_name)
            fig.savefig(out_png)
            c = river_tools.conveyance_curve(df)



