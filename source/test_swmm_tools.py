from unittest import TestCase
import swmm_tools
import os
import filecmp

class TestSWMM_Tools(TestCase):
    def setUp(self) -> None:
        self.workspace = './test'

    def test_run(self):
        inp_path = os.path.join(self.workspace, 'rkt_sample_all.inp')
        inp_path = os.path.abspath(inp_path)
        swmm_tools.run(inp_path)

    def test_extract_node(self):
        out_path = os.path.join(self.workspace, 'rkt_sample_all.out')
        df = swmm_tools.extract_node(out_path, '1')
        total_flow = df.iloc[0]['total_flow']
        out_path = os.path.join(self.workspace, 'rkt_sample_all.csv')
        df.to_csv(out_path, index_label='datetime')
        self.assertAlmostEqual(total_flow, 0.04260213300585747)


    def test_render_input(self):
        data = {'start_dt': '02/20/2011', 'end_dt': '06/15/2011',
                      # RTK
                      'r1': 0.01, 't1': 1, 'k1': 2,
                      'r2': 0.045, 't2': 3, 'k2': 4,
                      'r3': 0.03, 't3': 7, 'k3': 4,
                      # IA
                      'dmax': 0, 'drecov': 0, 'dinit': 0,
                      # DWF
                      'ssarea': 523, 'weekday_avg': 0.0259, 'gwi': 0.0256
                      }
        tmp_folder = os.path.join(self.workspace, 'tmp')
        tmp_name = 'calibration_tmp.inp'
        out_inp = os.path.join(self.workspace, 'template_test.inp')
        check_out_inp = os.path.join(self.workspace, 'template_test_check.inp')
        swmm_tools.render_input(tmp_folder, tmp_name, data, out_inp)
        assert(filecmp.cmp(out_inp, check_out_inp))




