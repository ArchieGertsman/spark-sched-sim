import numpy as np
import pandas as pd
import plotly.express as px


def make_gantt(dagsched_state):
    df_gantt = _dagsched_state_to_gantt_df(dagsched_state)

    n_workers = len(dagsched_state.workers)
    n_jobs = len(dagsched_state.jobs)

    fig_gantt = px.timeline(df_gantt, 
        x_start='t_accepted',
        x_end='t_completed', 
        y='worker_id',
        color='job_id', 
        template='seaborn',
        category_orders={
            'worker_id': [str(i) for i in reversed(range(n_workers))],
            'job_id': [str(i) for i in range(n_jobs)]
        })

    _setup_x_axis(fig_gantt, df_gantt)

    fig_gantt.update_traces(width=.7) # set fixed height of gantt boxes
    
    _add_task_labels(fig_gantt, df_gantt)

    _add_job_completion_vlines(fig_gantt, dagsched_state.jobs)

    return fig_gantt, df_gantt


def _dagsched_state_to_gantt_df(dagsched_state):
    tasks = []
    for job in dagsched_state.jobs:
        for stage in job.stages:
            for task in stage.tasks:
                task_dict = {
                    'worker_id': str(task.worker_id),
                    'job_id': str(job.id_),
                    'stage_id': str(stage.id_),
                    't_accepted': task.t_accepted[0],
                    't_completed': task.t_completed[0]
                }
                tasks += [task_dict]

    df_gantt = pd.DataFrame(tasks)
    df_gantt = df_gantt[df_gantt.t_accepted != np.inf]
    return df_gantt


def _add_task_labels(fig_gantt, df_gantt):
    df_labels = pd.DataFrame(columns=['x','y','text','showarrow'])
    df_labels.x = (df_gantt.t_accepted + df_gantt.t_completed) / 2
    df_labels.y = df_gantt.worker_id
    df_labels.text = df_gantt.stage_id
    df_labels.showarrow = False
    labels = df_labels.to_dict(orient='records')
    for label in labels:
        label['font'] = dict(size=8, color='white')
    fig_gantt['layout']['annotations'] = labels


def _add_job_completion_vlines(fig_gantt, jobs):
    for job in jobs:
        fig_gantt.add_vline(
            x=job.t_completed[0], 
            line_width=2, 
            line_color='green')


def _setup_x_axis(fig_gantt, df_gantt):
    fig_gantt.layout.xaxis.type = 'linear'

    df_gantt['delta'] = df_gantt.t_completed - df_gantt.t_accepted
    for d in fig_gantt.data:
        d.x = df_gantt[df_gantt.job_id == d.name].delta.tolist()
