import swmm_tools
import plot_tools
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt


def run_scenarios(scenarios, tmp_folder, tmp_name, out_folder):
    for sc, data in scenarios:
        out_inp = os.path.join(out_folder, '%s.inp' % sc)
        swmm_tools.render_input(tmp_folder, tmp_name, data, out_inp)
    for sc, data in scenarios:
        out_inp = os.path.join(out_folder, '%s.inp' % sc)
        swmm_tools.run(out_inp)


def obj_fn(df, fld, hourly_avg=False):
    result = {'argmax': df[fld].idxmax(),
              'max': df[fld].max(),
              'mean': df[fld].mean(),
              'min': df[fld].min()}
    dt = result['argmax'] - df.index.min()
    result['max_hr'] = dt.days*24+ dt.seconds/3600.0

    if hourly_avg:
        df = df.resample('60T').mean()
        result['hourly_average'] = {
            'argmax': df[fld].idxmax(),
              'max': df[fld].max(),
              'mean': df[fld].mean(),
              'min': df[fld].min()}
    return result

def event_obj_fn(df, events, fld):
    rows = []
    for start_dt, end_dt in events:
        row = obj_fn(df.loc[start_dt:end_dt, ], fld)
        row['start'] = start_dt
        row['end'] = end_dt
        row['event'] = '%s_%s' % (start_dt, end_dt)
        rows.append(row)
    return pd.DataFrame(rows)

def plot_events(df, events, fld):
    event_ct = len(events)
    fig, axes = plt.subplots(event_ct, 1)
    i = 0
    for start_dt, end_dt in events:
        df_event = df.loc[start_dt:end_dt, [fld]]
        row = obj_fn(df_event, fld)
        row['start'] = start_dt
        row['end'] = end_dt
        ax = axes[i]
        i += 1
        df_event.plot(ax=ax)
        ax.axvline(row['argmax'], color='grey', linestyle='--')
        ax.axhline(row['max'], color='grey', linestyle='--')
        ax.axhline(row['mean'], color='grey', linestyle='--')
        ax.axhline(row['min'], color='grey', linestyle='--')
        ax.set_title(row)

    return fig

def performance_fn(sim_df, sim_fld, obs_df, obs_fld, events):
    sim = event_obj_fn(sim_df, events, sim_fld)
    for fld in sim.keys():
        sim['sim_%s' % fld] = sim[fld]
        if fld != 'event':
            del sim[fld]

    obs = event_obj_fn(obs_df, events, obs_fld)
    for fld in obs.keys():
        obs['obs_%s' % fld] = obs[fld]
        if fld != 'event':
            del obs[fld]


    return sim.set_index('event').join(obs.set_index('event'))


