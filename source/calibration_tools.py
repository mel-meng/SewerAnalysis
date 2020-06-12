import swmm_tools
import plot_tools
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_peformance_fn(df_pfn, fn_list=['max', 'max_hr', 'mean', 'min']):

    ct = len(fn_list)
    fig, axes = plt.subplots(ct, 1)
    i = 0
    for fn in fn_list:
        ax = axes[i]
        i += 1
        sns.scatterplot(x='sim_%s' % fn, y='obs_%s' % fn, ax=ax, data=df_pfn, hue='event', style='event')
        min_v = min(df_pfn['sim_%s' % fn].min(), df_pfn['obs_%s' % fn].min())
        max_v = max(df_pfn['sim_%s' % fn].max(), df_pfn['obs_%s' % fn].max())
        ax.plot([min_v, max_v], [min_v, max_v], 'g--', label='1:1')
        ax.plot([min_v, max_v], [min_v*.9, max_v*.9], 'g--', label='-10%')
        ax.plot([min_v, max_v], [min_v * 1.1, max_v *1.1], 'g--', label='+10%')
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        ax.grid(True)
    fig.tight_layout()
    return fig

