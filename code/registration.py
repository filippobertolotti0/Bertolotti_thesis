from gym.envs.registration import register
from gymnasium.envs.registration import register as gymnasium_register

gymnasium_register(
    id='SimpleHouseRad-v0',
    entry_point='SimpleHouseRad:SimpleHouseRad',
)