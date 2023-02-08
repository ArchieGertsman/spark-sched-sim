import torch
import gymnasium as gym

from gym_dagsched.wrappers.decima_wrappers import (
    DecimaObsWrapper,
    DecimaActWrapper
)
from gym_dagsched.agents.decima_agent import DecimaAgent
from gym_dagsched.utils.metrics import avg_job_duration
from gym_dagsched.utils.hidden_prints import HiddenPrints



if __name__ == '__main__':
    torch.manual_seed(42)

    num_workers = 10

    model_dir = 'gym_dagsched/results/models'
    model_name = 'model_1b_20s_10w_500ep.pt'
    decima_agent = \
        DecimaAgent(num_workers,
                    training_mode=False, 
                    state_dict_path=\
                        f'{model_dir}/{model_name}')

    env_kwargs = {
        'num_workers': num_workers,
        'num_init_jobs': 1,
        'num_job_arrivals': 20,
        'job_arrival_rate': 1/25000,
        'moving_delay': 2000.,
        'render_mode': 'human' # visualize simulation
    }

    env_id = 'gym_dagsched:gym_dagsched/DagSchedEnv-v0'
    base_env = gym.make(env_id, **env_kwargs)
    env = DecimaActWrapper(DecimaObsWrapper(base_env))

    # options = options={'max_wall_time': 1e5}
    options = None

    with HiddenPrints():
        obs, _ = env.reset(seed=0, options=options)
        done = False
        
        while not done:
            action = decima_agent(obs)

            obs, reward, terminated, truncated, _ = \
                env.step(action)

            done = (terminated or truncated)

    print(avg_job_duration(env)*1e-3)
    
    env.close()