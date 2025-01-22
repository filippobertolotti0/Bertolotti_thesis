from gym.envs.registration import register
from gymnasium.envs.registration import register as gymnasium_register

# register(
#     id='SimpleHouseRad-v0',
#     entry_point='swissHouseRSlaW2W:swissHouseRSlaW2W',
# )

gymnasium_register(
    id='SimpleHouseRad-v0',
    entry_point='SimpleHouseRad:SimpleHouseRad',
)