from gymnasium.envs.registration import register

register(
     id="gym_dagsched/DagSchedEnv-v0",
     entry_point="gym_dagsched.envs:DagSchedEnv"
)