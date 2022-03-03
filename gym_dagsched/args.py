import argparse


parser = argparse.ArgumentParser(description='gym-dagsched')


parser.add_argument(
    '--n_jobs', 
    type=int, 
    default=10,
    help='number of jobs that will arrive throughout the simulation (default: 10)'
)


parser.add_argument(
    '--max_stages', 
    type=int, 
    default=8,
    help='max number of stages a job can contain (default: 10)'
)


parser.add_argument(
    '--max_tasks', 
    type=int, 
    default=4,
    help='max number of tasks that a stage can be split into (default: 4)'
)


parser.add_argument(
    '--n_worker_types', 
    type=int, 
    default=2,
    help='number of different worker types for heterogenous settings (default: 2)'
)


parser.add_argument(
    '--n_workers', 
    type=int, 
    default=5,
    help='number of workers in the simulation (default: 5)'
)


parser.add_argument(
    '--mjit', 
    type=float, 
    default=10.,
    help='mean job interarrival time (default: 10.0)'
)


args = parser.parse_args('')