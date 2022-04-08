import numpy as np
import pandas as pd
import plotly.express as px


def make_gantt(sim, title, x_max):
    df_gantt = _dagsched_state_to_gantt_df(sim)

    sorted_workers = sorted(sim.workers,
        key=lambda worker: (worker.type_, worker.id_))
    sorted_worker_ids = [_worker_display_id(w) for w in sorted_workers]

    fig_gantt = px.timeline(df_gantt, 
        x_start='t_accepted',
        x_end='t_completed', 
        y='worker_id',
        color='job_id', 
        template='ggplot2',
        title=title,
        category_orders={
            'worker_id': sorted_worker_ids#,
            #'job_id': [str(i) for i in range(n_jobs)]
        })

    _setup_x_axis(fig_gantt, df_gantt, x_max)

    fig_gantt.update_traces(width=1.) # set fixed height of gantt boxes

    fig_gantt.update_yaxes(title_text='Workers', showticklabels=False, showgrid=False, ticks="")
    
    # _add_task_labels(fig_gantt, df_gantt)

    _add_job_completion_vlines(fig_gantt, sim.jobs)

    return fig_gantt


def _worker_display_id(worker):
    return f'{worker.type_}_{worker.id_}'


def _dagsched_state_to_gantt_df(sim):
    tasks = []
    for job in sim.jobs:
        for op in job.ops:
            for task in list(op.completed_tasks):
                worker = sim.workers[task.worker_id]
                task_dict = {
                    'worker_id': _worker_display_id(worker),
                    'job_id': job.id_,
                    'op_id': str(op.id_),
                    't_accepted': task.t_accepted,
                    't_completed': task.t_completed
                }
                tasks += [task_dict]

    df_gantt = pd.DataFrame(tasks)
    return df_gantt



def _add_job_completion_vlines(fig_gantt, jobs):
    for job in jobs:
        fig_gantt.add_vline(
            x=job.t_completed, 
            line_width=2, 
            line_color='red')


def _setup_x_axis(fig_gantt, df_gantt, x_max):
    # fig_gantt.layout.xaxis.type = 'linear'
    fig_gantt.update_xaxes(title_text='Time', type='linear', range=(0,x_max), showgrid=False)
    # fig_gantt.update_layout(xaxis_range=(0,x_max))

    df_gantt['delta'] = df_gantt.t_completed - df_gantt.t_accepted
    for d in fig_gantt.data:
        # d.x = df_gantt[df_gantt.job_id == d.name].delta.tolist()
        d.x = df_gantt.delta.tolist()
        d.marker.line.width = 0.

