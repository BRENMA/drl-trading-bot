from gymnasium.envs.registration import register

register(
    id="gym_examples/TradingEnv-v0",
    entry_point="gym_examples.envs:TradingEnv",
)
