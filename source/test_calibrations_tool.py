from unittest import TestCase
import calibration_tools
import os
import filecmp
import pandas as pd
import datetime
import matplotlib.pyplot as plt


def read_flow_csv(f):
    df = pd.read_csv(f)
    df.index = df.apply(
        lambda x: datetime.datetime(int(x['Year']), int(x['Month']), int(x['Day']), int(x['Hour']), int(x['Minute'])),
        axis=1)
    df['Flow_mgd'] = df['Flow (MGD)']
    df['Velocity_fps'] = 0
    df['Depth_in'] = 0

    return df.loc[:, ['Flow_mgd', 'Velocity_fps', 'Depth_in']]


def read_rain_csv(f):
    df = pd.read_csv(f)
    df.index = df.apply(
        lambda x: datetime.datetime(int(x['year']), int(x['month']), int(x['day']), int(x['hour']), int(x['minute'])),
        axis=1)
    df['Rain (inch)'] = df['rainfall']
    return df.loc[:, ['Rain (inch)']]

class TestRun_scenarios(TestCase):
    def setUp(self) -> None:
        self.workspace = './test'
        ws = os.path.join(self.workspace, 'flow')
        rain_df = read_rain_csv(os.path.join(ws, 'Post_Rehab_Rainfall_Data.csv'))
        # need to resampel to 5 min time step
        self.rain_df = rain_df.resample('5T').asfreq().fillna(0)
        self.flow_df = read_flow_csv(os.path.join(ws, 'control_subsewershed_post.CSV'))
        df = pd.read_csv(os.path.join(self.workspace, 'sim/rkt_sample_all.csv'), parse_dates=True)
        df.index = pd.to_datetime(df['datetime'])
        self.sim = df

    def test_performance_fn(self):
        sim_df = self.sim
        sim_fld = 'total_flow'
        obs_df = self.flow_df
        obs_fld = 'Flow_mgd'
        events = [['2/27/2011', '3/3/2011 00:00'], ['3/4/2011', '3/8/2011'], ['3/9/2011', '3/13/2011'],
                  ['3/15/2011', '3/17/2011'], ['3/22/2011', '3/25/2011'], ['4/4/2011', '4/6/2011 00:00']]
        pfn = calibration_tools.performance_fn(sim_df, sim_fld, obs_df, obs_fld, events)
        a = os.path.join(self.workspace, 'pfn.csv')
        pfn.to_csv(a)
        b = os.path.join(self.workspace, 'pfn_check.csv')
        assert(filecmp.cmp(a, b))

    def test_sim(self):
        # print(self.sim)
        # print(self.sim.index)
        # print(self.flow_df.index)

        ax = self.flow_df.plot()
        self.sim.plot(ax=ax)
        plt.show()
    def test_obj_fn(self):
        df = self.flow_df
        fld = 'Flow_mgd'
        flow_of = calibration_tools.obj_fn(df, fld)
        print(flow_of)
        check = {'argmax': pd.Timestamp('2011-03-05 03:10:00'), 'max': 1.956, 'mean': 0.14722334452419536,
                'max_hr': 353.4166666666667, 'min': -0.43700000000000006}
        assert(check == flow_of)

    def test_event_obj_fn(self):
        df = self.flow_df
        fld = 'Flow_mgd'
        events = [['2/27/2011', '3/3/2011 00:00'], ['3/4/2011', '3/8/2011'], ['3/9/2011', '3/13/2011'],
                  ['3/15/2011', '3/17/2011'], ['3/22/2011', '3/25/2011'], ['4/4/2011', '4/6/2011 00:00']]
        summary = calibration_tools.event_obj_fn(df, events, fld)
        a = os.path.join(self.workspace, 'event_obj_fn.csv')
        b = os.path.join(self.workspace, 'event_obj_fn_check.csv')
        summary.to_csv(a, index=False)
        assert(filecmp.cmp(a, b))

    def test_plot_event_obj_fn(self):
        df = self.flow_df
        fld = 'Flow_mgd'
        events = [['2/27/2011', '3/3/2011 00:00'], ['3/4/2011', '3/8/2011'], ['3/9/2011', '3/13/2011'],
                  ['3/15/2011', '3/17/2011'], ['3/22/2011', '3/25/2011'], ['4/4/2011', '4/6/2011 00:00']]
        summary = calibration_tools.event_obj_fn(df, events, fld)
        a = os.path.join(self.workspace, 'event_obj_fn.csv')
        b = os.path.join(self.workspace, 'event_obj_fn_check.csv')
        summary.to_csv(a, index=False)
        assert(filecmp.cmp(a, b))
        calibration_tools.plot_events(df, events, fld)
        plt.show()


    def test_plot_obj_fn(self):
        flow_df = self.flow_df
        ax = flow_df.plot()
        flow_df = flow_df.resample('60T').mean()
        flow_df.plot(ax=ax, color='grey')
        peak_time =flow_df['Flow_mgd'].argmax()
        peak = flow_df['Flow_mgd'].max()
        mean = flow_df['Flow_mgd'].mean()
        min =  flow_df['Flow_mgd'].min()
        ax.axvline(peak_time, color='red', linestyle='--')
        ax.axhline(peak, color='red', linestyle='--')
        ax.axhline(mean, color='red', linestyle='--')
        ax.axhline(min, color='red', linestyle='--')
        plt.show()
        self.fail()

    def test_run_scenarios(self):
        data_tmp = {'start_dt': '02/20/2011', 'end_dt': '06/15/2011',
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
        scenarios = []

        for sc in ['all', 'r1', 'r2', 'r3', 'dwf']:

            data = data_tmp.copy()
            if sc == 'all':
                pass
            elif sc == 'r1':
                data['r2'] = 0
                data['r3'] = 0

            elif sc == 'r2':
                data['r1'] = 0
                data['r3'] = 0
            elif sc == 'r3':
                data['r1'] = 0
                data['r2'] = 0
            elif sc == 'dwf':
                data['r1'] = 0
                data['r2'] = 0
                data['r3'] = 0
            scenarios.append([sc, data])

        out_folder = os.path.join(self.workspace, 'sc')
        calibration_tools.run_scenarios(scenarios, tmp_folder, tmp_name, out_folder)
        a = os.path.join(self.workspace, 'sc/r3.inp')
        b = os.path.join(self.workspace, 'sc/r3_check.inp')
        assert(filecmp.cmp(a, b))
