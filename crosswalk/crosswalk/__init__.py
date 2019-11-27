from gym.envs.registration import register

register(
    id='crosswalk-v0',
    entry_point='crosswalk.envs:CrosswalkEnv',
)

