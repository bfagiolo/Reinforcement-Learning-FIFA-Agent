# train_run29_forwardpay500.py

import os
import time

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

from fixed_fifa_RL_stable_env import FifaRLStableEnv
from hybrid_beta_finesse_policy import HybridBetaFinessePolicy  # noqa: F401
from ppo_separate_three_entropy import PPOSeparateThreeEntropy
from attempt_limit_callback import AttemptLimitCallback

PREV_RUN_NAME = "ppo_fifa_run28_forwardpay30"
RUN_NAME = "ppo_fifa_run30_forwardpay30"

LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

LOAD_PATH = os.path.join(MODEL_DIR, f"{PREV_RUN_NAME}_final.zip")
if not os.path.exists(LOAD_PATH):
    raise FileNotFoundError(f"Previous model not found at: {LOAD_PATH}")

def make_env():
    env = FifaRLStableEnv()
    mon_path = os.path.join(LOG_DIR, f"{RUN_NAME}_monitor.csv")
    env = Monitor(env, filename=mon_path)
    return env

env = DummyVecEnv([make_env])

logger = configure(
    folder=os.path.join(LOG_DIR, RUN_NAME),
    format_strings=["stdout", "csv", "tensorboard"]
)

print(f"📦 Loading previous model from: {LOAD_PATH}")
model = PPOSeparateThreeEntropy.load(
    LOAD_PATH,
    env=env,
    device="auto"
)
model.set_logger(logger)

ATTEMPT_LIMIT = 200
callbacks = [AttemptLimitCallback(env=env, max_attempts=ATTEMPT_LIMIT)]
TOTAL_STEPS = int(2e7)

print(f"🔧 Starting training: {RUN_NAME} (attempt limit = {ATTEMPT_LIMIT})")
t0 = time.time()

try:
    model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks, progress_bar=True)
except KeyboardInterrupt:
    print("⛔ Interrupted — saving current weights...")
finally:
    try:
        env.close()
    except Exception as e:
        print(f"⚠️ env.close() raised: {e}")

t1 = time.time()
print(f"✅ Training finished in {(t1 - t0)/60:.1f} min (stopped by attempt limit or interrupt)")

final_path = os.path.join(MODEL_DIR, f"{RUN_NAME}_final.zip")
model.save(final_path)
print(f"💾 Saved final model to: {final_path}")
