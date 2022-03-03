import numpy as np
import pandas as pd
import plotly.express as px


def make_gantt(dagsched_state):
    df_gantt = _dagsched_state_to_gantt_df(dagsched_state)

    fig_gantt = px.timeline(df_gantt, 
        x_start='t_accepted',
        x_end='t_completed', 
        y='worker_id',
        color='job_id', 
        template='seaborn')

    _setup_x_axis(df_gantt, fig_gantt)

    _add_job_completion_vlines(fig_gantt, dagsched_state.jobs)

    fig_gantt.update_traces(width=0.7) # set fixed height of gantt boxes
    
    fig_gantt['layout']['annotations'] = _create_task_labels(df_gantt)

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
    df_gantt.sort_values('worker_id', inplace=True)
    return df_gantt


def _create_task_labels(df_gantt):
    df_labels = pd.DataFrame(columns=['x','y','text','showarrow'])
    df_labels.x = (df_gantt.t_accepted + df_gantt.t_completed) / 2
    df_labels.y = df_gantt.worker_id
    df_labels.text = df_gantt.stage_id
    df_labels.showarrow = False
    labels = df_labels.to_dict(orient='records')
    for label in labels:
        label['font'] = dict(size=8, color='white')
    return labels


def _add_job_completion_vlines(fig, jobs):
    for job in jobs:
        fig.add_vline(
            x=job.t_completed[0], 
            line_width=2, 
            line_color='green')


def _setup_x_axis(df_gantt, fig):
    fig.layout.xaxis.type = 'linear'

    df_gantt['delta'] = df_gantt.t_completed - df_gantt.t_accepted
    for d in fig.data:
        d.x = df_gantt[df_gantt.job_id == d.name].delta.tolist()
