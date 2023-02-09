'''This is an example of how to run a job scheduling simulation
according to a reinforcement-learned scheduling policy.
'''
import torch
import gymnasium as gym

from gym_dagsched.wrappers.decima_wrappers import (
    DecimaObsWrapper,
    DecimaActWrapper
)
from gym_dagsched.agents.decima_agent import DecimaAgent
from gym_dagsched.utils.hidden_prints import HiddenPrints
from gym_dagsched.utils import metrics



if __name__ == '__main__':
    # set rng seeds for reproducibility 
    env_seed = 500 
    torch_seed = 42
    torch.manual_seed(torch_seed)

    # select the number of simulated workers
    num_workers = 10

    # load learned agent
    model_dir = 'gym_dagsched/results/models'
    model_name = 'model_1b_20s_10w_500ep.pt'
    decima_agent = \
        DecimaAgent(num_workers,
                    training_mode=False, 
                    state_dict_path=f'{model_dir}/{model_name}')


    # same settings as in training
    env_kwargs = {
        'num_workers': num_workers,
        'num_init_jobs': 1,
        'num_job_arrivals': 20,
        'job_arrival_rate': 1/25000,
        'moving_delay': 2000.,
        'render_mode': 'human' # visualize simulation
    }

    # setup gym environment
    env_id = 'gym_dagsched:gym_dagsched/DagSchedEnv-v0'
    base_env = gym.make(env_id, **env_kwargs)
    env = DecimaActWrapper(DecimaObsWrapper(base_env))

    # run an episode
    with HiddenPrints():
        obs, _ = env.reset(seed=env_seed, options=None)
        done = False
        
        while not done:
            action = decima_agent(obs)

            obs, reward, terminated, truncated, _ = \
                env.step(action)

            done = (terminated or truncated)


    avg_job_duration = int(metrics.avg_job_duration(env) * 1e-3)
    print(f'Average job duration: {avg_job_duration}s')
    
    # cleanup rendering
    env.close()