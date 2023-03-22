from gymnasium import Wrapper

class AverageReward(Wrapper):
    def __init__(self, env, buff_cap=10000):
        super().__init__(env)
        self.buff_cap = buff_cap

        self.dt_buff = []
        self.dt_sum = 0.

        self.rew_buff = []
        self.rew_sum = 0.

        self.prev_wall_time = 0.


    @property
    def avg_rew_per_sec(self):
        return self.rew_sum / self.dt_sum if self.dt_sum > 0 else 0


    def step(self, act):
        obs, rew, term, trunc, info = self.env.step(act)
        wall_time = info['wall_time']
        dt = wall_time - self.prev_wall_time
        self._add(dt, rew)
        self.prev_wall_time = wall_time
        avg_rew = rew - self.avg_rew_per_sec * dt
        return obs, avg_rew, term, trunc, info


    def _add(self, dt, rew):
        if dt == 0:
            assert rew == 0
            return
        
        self.dt_buff += [dt]
        self.dt_sum += dt

        self.rew_buff += [rew]
        self.rew_sum += rew

        assert len(self.dt_buff) == len(self.rew_buff)

        if len(self.dt_buff) > self.buff_cap:
            self.dt_sum -= self.dt_buff.pop(0)
            self.rew_sum -= self.rew_buff.pop(0)
