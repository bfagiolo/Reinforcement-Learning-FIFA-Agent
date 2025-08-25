# attempt_limit_callback.py


from stable_baselines3.common.callbacks import BaseCallback


class AttemptLimitCallback(BaseCallback):
    def __init__(self, env, max_attempts=50, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.max_attempts = max_attempts


    def _on_step(self) -> bool:
        # Unwrap DummyVecEnv → Monitor → FifaRLStableEnv
        current_env = self.env.envs[0].env


        if current_env.attempt_count >= self.max_attempts:
            print(f"🎯 Attempt limit reached: {self.max_attempts}. Stopping training.")
            return False
        return True





