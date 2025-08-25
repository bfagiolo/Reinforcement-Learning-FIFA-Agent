import cv2
import numpy as np
import mss
import time
import gymnasium as gym
from gymnasium import spaces
import vgamepad as vg
from ultralytics import YOLO
from byte_tracker import BYTETracker
import easyocr
import math
import os
import csv
import pygetwindow as gw
from collections import deque


class TrackerArgs:
    def __init__(self):
        self.track_thresh = 0.5
        self.track_buffer = 60
        self.match_thresh = 0.8
        self.mot20 = False


def smooth(prev, new, alpha=0.4):
    if not prev:
        return new
    return (
        int(alpha * new[0] + (1 - alpha) * prev[0]),
        int(alpha * new[1] + (1 - alpha) * prev[1])
    )


def ema_point(prev, new, alpha=0.3):
    if prev is None:
        return (int(new[0]), int(new[1]))
    return (
        int(alpha * new[0] + (1 - alpha) * prev[0]),
        int(alpha * new[1] + (1 - alpha) * prev[1])
    )


class FifaRLStableEnv(gym.Env):
    def __init__(self):
        super().__init__()

        print("ENV DEBUG: RUN11_4D v1 — action_space will be set to (4,)")

        self.SAVE_ATTEMPT_VIDEOS = False
        self.VID_DIR = "attempt_videos"
        os.makedirs(self.VID_DIR, exist_ok=True)
        self._vid_writer = None
        self._vid_fps = 20.0
        self._vid_size = None

        self.valid_frame_count = 0
        self.sct = mss.mss()

        print("🎮 Looking for FIFA window...")
        fifa_window = next((w for w in gw.getWindowsWithTitle("FIFA 19") if w.visible), None)
        if not fifa_window:
            print("❌ FIFA window not found.")
            raise RuntimeError("FIFA window not found")

        fifa_window = next((w for w in gw.getWindowsWithTitle("FIFA 19") if w.visible), None)
        raw_left, raw_top = fifa_window.left, fifa_window.top
        raw_width, raw_height = fifa_window.width, fifa_window.height

        mon = {
            "top": raw_top,
            "left": raw_left,
            "width": raw_width,
            "height": raw_height,
        }

        desk = self.sct.monitors[0]
        mon["left"] = max(desk["left"], min(mon["left"], desk["left"] + desk["width"] - 1))
        mon["top"] = max(desk["top"], min(mon["top"], desk["top"] + desk["height"] - 1))
        mon["width"] = max(1, min(mon["width"], desk["left"] + desk["width"] - mon["left"]))
        mon["height"] = max(1, min(mon["height"], desk["top"] + desk["height"] - mon["top"]))

        self.monitor = mon
        cv2.imwrite("monitor_probe_after_start.png", np.array(self.sct.grab(self.monitor))[:, :, :3])

        left_frac = 0.204383562
        top_frac = 0.254192410
        width_frac = 0.791780822
        height_frac = 0.598411297

        self.player_crop = {
            "left": self.monitor["left"] + int(self.monitor["width"] * left_frac),
            "top": self.monitor["top"] + int(self.monitor["height"] * top_frac),
            "width": int(self.monitor["width"] * width_frac),
            "height": int(self.monitor["height"] * height_frac),
        }
        cv2.imwrite("player_crop_probe.png", np.array(self.sct.grab(self.player_crop))[:, :, :3])

        self.obs_w = float(self.player_crop["width"])
        self.obs_h = float(self.player_crop["height"])
        self.CROP_DIAG = float(np.hypot(self.obs_w, self.obs_h))

        self._prev_distance_norm = None
        self.DEADZONE_NORM = 2.0 / max(1.0, self.CROP_DIAG)

        self._recent_d_norm = deque(maxlen=16)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        gx_frac = 0.82
        gy_frac = 0.415
        self.goal_fallback = (
            int(gx_frac * self.player_crop["width"]),
            int(gy_frac * self.player_crop["height"])
        )

        big = {
            "top": mon["top"] + int(mon["height"] * 0.03),
            "left": mon["left"] + int(mon["width"] * 0.80),
            "width": int(mon["width"] * 0.18),
            "height": int(mon["height"] * 0.12),
        }
        big_img = np.array(self.sct.grab(big))[:, :, :3]
        cv2.imwrite("hud_probe_big.png", big_img)
        print("big mean:", big_img.mean())

        self.hud_crop = big
        print("🖼️ Using BIG HUD crop for OCR:", self.hud_crop)

        pad_x = max(2, int(self.hud_crop["width"] * 0.02))
        pad_y = max(2, int(self.hud_crop["height"] * 0.08))
        self.hud_crop = {
            "left": self.hud_crop["left"] + pad_x,
            "top": self.hud_crop["top"] + pad_y,
            "width": self.hud_crop["width"] - 2 * pad_x,
            "height": self.hud_crop["height"] - 2 * pad_y,
        }

        num = {
            "left": self.hud_crop["left"] + int(self.hud_crop["width"] * 0.4),
            "top": self.hud_crop["top"] + int(self.hud_crop["height"] * .85),
            "width": int(self.hud_crop["width"] * 0.35),
            "height": int(self.hud_crop["height"] * 0.42),
        }
        num_img = np.array(self.sct.grab(num))[:, :, :3]
        cv2.imwrite("hud_probe_number.png", num_img)

        self.hud_crop = num
        print("🎯 Number crop:", self.hud_crop)

        self.model = YOLO("yolov8m.pt")
        self.tracker = BYTETracker(args=TrackerArgs())
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.gamepad = vg.VX360Gamepad()
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        self.shots_csv = "fifa_rl_shots_v2.csv"
        self.shoot_hold = 0
        self.SHOOT_HOLD_FRAMES = 2
        self.SHOOT_GATE = 0.10

        self.MIN_FRAMES_BEFORE_SHOT = 3
        self.MAX_FRAMES_BEFORE_SHOT = 18

        self.FREE_FRAMES = 3
        self.TIME_PENALTY_BASE = 40.0
        self.TIME_PENALTY_GROWTH = 1.35

        self.cap_forced_shots = 0
        self._cap_triggered = False

        self.last_attacker_pos = None
        self.last_goalkeeper_pos = None
        self.attacker_id = None
        self.goalkeeper_id = None

        self._ema_att = None
        self._ema_goal = None
        self._prev_distance = None
        self.DEADZONE_PX = 3
        self.LAMBDA_F = 0.0
        self.LAMBDA_B = 0.0

        self.forward_frames = 0
        self.backward_frames = 0
        self._motion_considered = 0

        self.no_person_frames = 0
        self.MAX_NO_PERSON_FRAMES = 90

        self.attempt_count = 0
        self.session_id = 1
        self.total_sessions = 5

        self.session_start_time = None
        self.agent_can_act = False
        self.skill_game_started = False
        self.frame_count = 0
        self.b_pressed = False

        self.csv_path = "fifa_rl_training_data.csv"
        self.reward_csv = "fifa_rl_attempt_rewards.csv"
        self.input_csv = "fifa_rl_inputs_log.csv"
        self.frame_csv = "fifa_rl_per_frame_log.csv"

        self.log_rows = []
        self.reward_log = []
        self.input_log = []
        self.frame_log = []

        self.shot_log = []

        self.previous_score = 0
        self.in_attempt_cooldown = False

        self.TELEPORT_JUMP_PX = 180
        self._tele_jump_confirm = 0
        self._raw_att_center = None
        self._raw_att_prev = None

        self.DEBUG_GOAL_FALLBACK_SNAPSHOT = False
        self._did_goal_fallback_snapshot = False

        self.max_t = 0.21
        self.cap_hits = 0
        self.cap_checks = 0

        self.FWD_SIGN_DEFAULT = -1
        self.FWD_DZ_NORM = 0.0015
        self.K_F = 220.0

        self.PROG_WIN = 8
        self.PROG_THRESH = -0.0020

        self.PS_N = 8
        self.PROJ_MALUS_BIG = -800.0

        self.K_B = 200.0

        self.RAW_MULT = 0.18
        self.RAW_CLIP_MAX = 1500.0

        self.MISS_THRESHOLD = 20.0
        self.MISS_PENALTY_BASE = 140.0
        self.MISS_PENALTY_SLOPE = 1100
        self.MISS_PENALTY_H0 = 0.20

        self.FULLPOWER_HOLD_THRESH = 0.235
        self.FULLPOWER_GOAL_TRIM = 120 * self.RAW_MULT
        self.GOAL_TRIM_MIN_REWARD = 200.0
        self.TRIM_D_CLOSE = 0.22

        self.SWEET_MAX_HOLD = 0.20
        self.EFFICIENT_FINISH_BONUS = 100.0 * self.RAW_MULT

        self.motion_csv = "fifa_rl_attempt_motion.csv"
        self.motion_log = []

        self._start_d_norm = None
        self.PROG_PENALTY = 800.0
        self._proj_hist = deque(maxlen=128)

        self.fwd_sign = self.FWD_SIGN_DEFAULT
        self._prev_ax = None
        self._sum_proj = 0.0

        self.LOG_PROJ_CSV = False
        self._proj_log_fp = None

        self._last_net_movement = None
        self.back_consec = 0
    def _dist_to_goal_norm(self):
        if self._raw_att_center is None:
            return None
        ax, ay = self._raw_att_center
        gx, gy = self.goal_fallback
        return math.hypot(gx - ax, gy - ay) / max(1.0, self.CROP_DIAG)

    def _start_attempt_video(self):
        if not self.SAVE_ATTEMPT_VIDEOS:
            return
        frame = self._get_screen()[:, :, :3]
        h, w = frame.shape[:2]
        self._vid_size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        name = f"session{self.session_id}_attempt{self.attempt_count}.avi"
        path = os.path.join(self.VID_DIR, name)
        self._vid_writer = cv2.VideoWriter(path, fourcc, self._vid_fps, self._vid_size)

    def _write_attempt_frame(self, frame_bgr):
        if self._vid_writer is not None:
            self._vid_writer.write(frame_bgr)

    def _end_attempt_video(self):
        if self._vid_writer is not None:
            self._vid_writer.release()
            self._vid_writer = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        assert getattr(self.action_space, "shape", None) == (4,), \
            f"Env must expose 4-D action space, got {getattr(self.action_space, 'shape', None)}"

        if not self.skill_game_started:
            print("⏳ Starting skill game before first attempt...")
            time.sleep(6)
            self._start_skill_game()
            self.skill_game_started = True

        now = time.time()
        delay = now - getattr(self, "cooldown_start_time", now)
        print(f"🔄 PPO called reset() — starting attempt {self.attempt_count + 1} (⏱️ {delay:.2f}s since last shot)")

        self.shoot_hold = 0
        self.attempt_count += 1
        self.frame_count = 0
        self.valid_frame_count = 0
        self.b_pressed = False
        self.in_attempt_cooldown = False
        self.agent_can_act = True

        self.last_attacker_pos = None
        self.last_goalkeeper_pos = None
        self.attacker_id = None
        self.goalkeeper_id = None
        self.no_person_frames = 0
        self._cap_triggered = False
        self.tracker = BYTETracker(args=TrackerArgs())

        self._ema_att = None
        self._ema_goal = None
        self._prev_distance = None
        self.forward_frames = 0
        self.backward_frames = 0
        self._motion_considered = 0

        self._prev_distance_norm = None
        self._recent_d_norm.clear()
        self._start_d_norm = None

        self._proj_hist.clear()
        self._last_net_movement = None

        self.fwd_sign = self.FWD_SIGN_DEFAULT
        self._prev_ax = None
        self._sum_proj = 0.0

        if self.LOG_PROJ_CSV:
            import os, time as _t
            os.makedirs("proj_logs", exist_ok=True)
            fn = f"proj_logs/attempt_{int(_t.time()*1000)}.csv"
            self._proj_log_fp = open(fn, "w", buffering=1)
            self._proj_log_fp.write("frame,ax,ay,proj_norm\n")

        self._end_attempt_video()
        self._start_attempt_video()

        if (self.DEBUG_GOAL_FALLBACK_SNAPSHOT 
            and not self._did_goal_fallback_snapshot 
            and self.attempt_count == 1):
            frame_bgra = self._get_screen()
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            gx, gy = int(self.goal_fallback[0]), int(self.goal_fallback[1])
            cv2.drawMarker(frame_bgr, (gx, gy), (0, 0, 255),
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=28, thickness=2)
            cv2.circle(frame_bgr, (gx, gy), 6, (0, 255, 255), -1)
            cv2.putText(frame_bgr, f"goal_fallback ({gx},{gy})", 
                        (max(0, gx+10), max(20, gy-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(self.VID_DIR, "goal_fallback_probe_attempt1.png"), frame_bgr)
            self._did_goal_fallback_snapshot = True

        self._tele_jump_confirm = 0
        self._raw_att_center = None
        self._raw_att_prev = None

        return self._observe(), {}
    def _att_goal_px(self):
        if self._raw_att_center is not None:
            ax, ay = self._raw_att_center
        elif self.last_attacker_pos is not None:
            ax, ay = self.last_attacker_pos
        else:
            return None

        if self.last_goalkeeper_pos is not None:
            gx, gy = self.last_goalkeeper_pos
        else:
            gx, gy = self.goal_fallback

        return float(ax), float(ay), float(gx), float(gy)

    def step(self, action):
        current_time = time.time()

        if current_time - self.session_start_time > 45:
            print("🛑 45 seconds passed — restarting skill game...")
            self.session_restart()

        if not self.agent_can_act:
            print(f"⏸️ Agent is frozen — skipping frame {self.frame_count + 1}")
            self.frame_count += 1
            return self._observe(), 0, False, False, {}

        self.frame_count += 1

        ax = float(action[0])
        ay = float(action[1])
        shoot_val = float(action[2])
        f_raw = float(action[3]) if (hasattr(action, "__len__") and len(action) >= 4) else 0.0
        finesse = bool(int(round(np.clip(f_raw, 0.0, 1.0))))

        shaping_reward = 0.0

        joy_x = int(ax * 32767)
        joy_y = int(ay * 32767)

        print(f"🎮 Agent input: joy_x={joy_x}, joy_y={joy_y}, shoot_val={shoot_val:.2f}, finesse={int(finesse)} ({f_raw:.2f})")

        self.gamepad.left_joystick(x_value=joy_x, y_value=joy_y)
        self.gamepad.update()

        self.attacker_tracked_this_frame = False
        obs = self._observe()

        attacker_x, attacker_y = float(obs[0]), float(obs[1])
        goalie_x, goalie_y = float(obs[2]), float(obs[3])
        dx = goalie_x - attacker_x
        dy = goalie_y - attacker_y

        if not self.attacker_tracked_this_frame:
            self.no_person_frames += 1
            if self.no_person_frames >= self.MAX_NO_PERSON_FRAMES:
                print(f"⚠️ No players detected for {self.MAX_NO_PERSON_FRAMES} frames — skipping frame {self.frame_count}")
                self.frame_count += 1
                return self._observe(), 0, False, False, {}
        else:
            self.no_person_frames = 0

        if self.agent_can_act:
            if self.attacker_tracked_this_frame:
                self.valid_frame_count += 1
            valid = 1 if self.attacker_tracked_this_frame else 0
            delta_d_norm_out = "NA"
            curr_for_log = self._dist_to_goal_norm()
            if curr_for_log is not None and self._prev_distance_norm is not None:
                if np.isfinite(curr_for_log) and np.isfinite(self._prev_distance_norm):
                    delta_d_norm_out = float(self._prev_distance_norm - curr_for_log)
            self.frame_log.append([
                self.session_id,
                self.attempt_count,
                self.frame_count,
                attacker_x, attacker_y,
                goalie_x, goalie_y,
                dx, dy,
                joy_x, joy_y,
                valid,
                delta_d_norm_out
            ])
            self.input_log.append([
                self.session_id,
                self.attempt_count,
                self.frame_count,
                attacker_x, attacker_y,
                goalie_x, goalie_y,
                dx, dy,
                joy_x, joy_y
            ])
        if self.agent_can_act and self.attacker_tracked_this_frame:
            if self.frame_count > self.FREE_FRAMES:
                extra_frames = self.frame_count - self.FREE_FRAMES
                penalty = self.TIME_PENALTY_BASE * (self.TIME_PENALTY_GROWTH ** (extra_frames - 1))
                shaping_reward -= penalty
                print(f"⏱️ Frame {self.frame_count}: time_penalty = -{penalty:.1f}")

        proj_norm = None
        if self.attacker_tracked_this_frame and self._raw_att_center is not None:
            ax_raw, ay_raw = self._raw_att_center
            print(f"pos=({ax_raw:.1f}, {ay_raw:.1f})", end="")
            if self._prev_ax is not None:
                dx = ax_raw - self._prev_ax
                proj_norm = (self.fwd_sign * dx) / max(1.0, self.CROP_DIAG)
                self._proj_hist.append(proj_norm)
                self._sum_proj += proj_norm

                if self.LOG_PROJ_CSV and self._proj_log_fp:
                    self._proj_log_fp.write(f"{self.frame_count},{ax_raw:.3f},{ay_raw:.3f},{proj_norm:.7f}\n")

                DZ = self.FWD_DZ_NORM
                if proj_norm > DZ:
                    shaping_reward += self.K_F * (proj_norm - DZ)
                elif proj_norm < -DZ:
                    shaping_reward += self.K_F * (proj_norm + DZ)

                if proj_norm < -DZ:
                    self.back_consec += 1
                else:
                    self.back_consec = 0

                if self.back_consec >= 2:
                    shaped_pen = -self.PROG_PENALTY
                    shaping_reward += shaped_pen
                    print(f"⛔ Backdrift streak ({self.back_consec} frames) → penalty {shaped_pen}")
                    self.back_consec = 0

                print(f"  proj_norm={proj_norm:+.5f}")
            else:
                print("  proj_norm=NA")
            self._prev_ax = ax_raw

        if self.agent_can_act and not self.b_pressed and len(self._proj_hist) >= self.PROG_WIN:
            recent = sum(list(self._proj_hist)[-self.PROG_WIN:])
            if recent < self.PROG_THRESH:
                print(f"🛑 Backdrift detected (proj_sum[{self.PROG_WIN}]={recent:.5f} < {self.PROG_THRESH:.5f})")
                shaped_pen = -self.PROG_PENALTY
                shaping_reward += shaped_pen
                print(f"⚠️ Applied backdrift penalty: {shaped_pen:.1f} (episode continues)")
                self.reward_log.append([
                    self.session_id, self.attempt_count,
                    0.0, shaped_pen,
                    f"{self.valid_frame_count/max(1,self.frame_count):.4f}",
                    self.valid_frame_count, self.frame_count,
                    0,
                    0
                ])

        if self.agent_can_act and not self.b_pressed:
            if self._detect_teleport():
                print("🟡 Teleport detected — ending attempt with shaped penalty check.")
                self.b_pressed = True
                self.agent_can_act = False
                self.in_attempt_cooldown = False

                valid_ratio = self.valid_frame_count / max(1, self.frame_count)
                raw_reward = 0.0

                if self.valid_frame_count >= 4:
                    shaped_reward = -150.0
                    print(f"💥 Teleport penalty applied: {shaped_reward} (valid frames: {self.valid_frame_count})")
                else:
                    shaped_reward = 0.0
                    print(f"⚖️ Teleport leniency: {shaped_reward} (valid frames: {self.valid_frame_count})")

                self.motion_log.append([
                    self.session_id, self.attempt_count,
                    int(self.forward_frames), int(self.backward_frames)
                ])

                self.reward_log.append([
                    self.session_id, self.attempt_count,
                    raw_reward, shaped_reward,
                    f"{valid_ratio:.4f}", self.valid_frame_count, self.frame_count,
                    0,
                    1
                ])

                considered = max(1, self._motion_considered)
                fwd_ratio = self.forward_frames / considered
                print(f"📈 forward_ratio={fwd_ratio:.3f} over {considered} considered frames in attempt {self.attempt_count}")

                attempt_net_proj = float(self._sum_proj)
                if attempt_net_proj < -1e-4:
                    shaped_reward += self.PROJ_MALUS_BIG
                    print(f"🚫 Net backward attempt (proj_sum={attempt_net_proj:.4f}) → {self.PROJ_MALUS_BIG:.1f}")
                else:
                    print(f"✅ Net attempt projection OK (proj_sum={attempt_net_proj:.4f})")

                if self.LOG_PROJ_CSV and self._proj_log_fp:
                    self._proj_log_fp.close()
                    self._proj_log_fp = None

                time.sleep(1)
                self._end_attempt_video()
                return self._observe(), shaped_reward, True, False, {}
        if shoot_val > self.SHOOT_GATE:
            self.shoot_hold += 1
        else:
            self.shoot_hold = 0

        def sum_last(n):
            h = self._proj_hist
            return sum(list(h)[-n:]) if len(h) else 0.0

        DZ = self.FWD_DZ_NORM
        forward_gate_ok = (proj_norm is not None and proj_norm > DZ)
        is_back_now = (proj_norm is not None and proj_norm < -DZ)
        recent_sum = sum_last(self.PS_N)
        half_sum = sum_last(max(1, self.PS_N // 2))
        recent_back_ok = (recent_sum > -DZ)
        streak_ok = (half_sum > 0.0)

        policy_ready = (
            self.frame_count >= self.MIN_FRAMES_BEFORE_SHOT and
            self.shoot_hold >= self.SHOOT_HOLD_FRAMES
        )
        cap_ready = self.frame_count >= self.MAX_FRAMES_BEFORE_SHOT

        should_fire = (
            (not self.b_pressed) and
            (forward_gate_ok and (not is_back_now) and recent_back_ok and streak_ok) and
            (policy_ready or cap_ready)
        )

        if should_fire:
            cap_triggered = cap_ready and (not policy_ready)
            self._cap_triggered = bool(cap_triggered)
            if cap_triggered:
                self.cap_forced_shots += 1

            raw01 = (np.clip(float(shoot_val), -0.98, 0.98) + 1.0) / 2.0
            x = 3.0 * (raw01 ** 2) - 2.0 * (raw01 ** 3)
            MID_BOOST = 0.18
            s = x + MID_BOOST * (0.25 - (x - 0.5) ** 2)
            s = float(np.clip(s, 0.0, 1.0))

            min_t = 0.05
            max_t = 0.21
            hold_s = min_t + (max_t - min_t) * s
            hold_s = float(np.clip(hold_s, min_t, max_t))

            if abs(hold_s - max_t) < 1e-6:
                self.cap_hits += 1
            self.cap_checks += 1

            pts = self._att_goal_px()
            if pts is not None:
                ax_px, ay_px, gx_px, gy_px = pts
                self._shot_dist_norm = math.hypot(gx_px - ax_px, gy_px - ay_px) / max(1.0, self.CROP_DIAG)
            else:
                self._shot_dist_norm = 1.0

            if finesse:
                self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
                self.gamepad.update()

            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            self.gamepad.update()
            time.sleep(hold_s)
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            self.gamepad.update()

            if finesse:
                self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
                self.gamepad.update()

            print(f"🏹 Shot mapping: raw01={raw01:.3f}, hold_s={hold_s:.3f}, max_t={max_t:.2f}")

            cap_forced = 1 if self._cap_triggered else 0
            self.shot_log.append([
                self.session_id,
                self.attempt_count,
                self.frame_count,
                raw01,
                hold_s,
                self.max_t,
                cap_forced,
                float(getattr(self, "_shot_dist_norm", -1.0)),
                int(finesse)
            ])

            self._last_hold_s = float(hold_s)

            self.b_pressed = True
            self.agent_can_act = False
            self.in_attempt_cooldown = True
            self.cooldown_start_time = time.time()

            print(f"🟥 Shot at {self.cooldown_start_time:.2f} — entering cooldown...")
            print("🧊 Freezing agent for reward evaluation...")
            time.sleep(2)

            valid_ratio = self.valid_frame_count / max(1, self.frame_count)
            raw_reward, vr, vcount, fcount = self._log_reward(valid_ratio)
            raw = float(raw_reward)
            raw = min(raw, self.RAW_CLIP_MAX)
            reward = self.RAW_MULT * raw

            time.sleep(1)

            hold_s_used = float(getattr(self, "_last_hold_s", 0.0))
            if reward <= self.MISS_THRESHOLD:
                excess = max(0.0, hold_s_used - self.MISS_PENALTY_H0)
                penalty = (self.MISS_PENALTY_BASE + self.MISS_PENALTY_SLOPE * excess) * self.RAW_MULT
                reward -= penalty
                if hold_s_used >= self.FULLPOWER_HOLD_THRESH:
                    local_pen = 8.0 * max(0.0, hold_s_used - 0.235) ** 2
                    if local_pen > 0:
                        reward -= local_pen

            if reward >= self.GOAL_TRIM_MIN_REWARD and hold_s_used >= self.FULLPOWER_HOLD_THRESH:
                dist_norm = float(getattr(self, "_shot_dist_norm", 1.0))
                if dist_norm <= self.TRIM_D_CLOSE:
                    reward -= self.FULLPOWER_GOAL_TRIM

            if reward >= self.GOAL_TRIM_MIN_REWARD and hold_s_used <= self.SWEET_MAX_HOLD:
                reward += self.EFFICIENT_FINISH_BONUS

            pre_net = sum(list(self._proj_hist)[-self.PS_N:])
            self._last_net_movement = pre_net
            if pre_net < 0:
                shaped_pen = self.K_F * pre_net
                reward += shaped_pen

            attempt_net_proj = float(self._sum_proj)
            if attempt_net_proj < -1e-4:
                reward += self.PROJ_MALUS_BIG

            self.motion_log.append([
                self.session_id, self.attempt_count,
                int(self.forward_frames), int(self.backward_frames)
            ])

            self.reward_log.append([
                self.session_id, self.attempt_count,
                float(raw_reward), float(reward),
                f"{vr:.4f}", int(vcount), int(fcount),
                1 if getattr(self, "_cap_triggered", False) else 0,
                0,
                self._last_net_movement if self._last_net_movement is not None else "NA"
            ])

            considered = max(1, self._motion_considered)
            fwd_ratio = self.forward_frames / considered
            print(f"📈 forward_ratio={fwd_ratio:.3f} over {considered} considered frames in attempt {self.attempt_count}")

            if self.LOG_PROJ_CSV and self._proj_log_fp:
                self._proj_log_fp.close()
                self._proj_log_fp = None

            self._end_attempt_video()
            return self._observe(), reward, True, False, {}
        if self._raw_att_center is not None:
            self._raw_att_prev = self._raw_att_center

        if not self.b_pressed:
            return self._observe(), shaping_reward, False, False, {}
        else:
            return self._observe(), 0.0, False, False, {}

    def session_restart(self):
        self.agent_can_act = False
        self._end_attempt_video()

        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        self.gamepad.update()
        time.sleep(0.5)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        self.gamepad.update()

        time.sleep(10)

        for _ in range(2):
            self.gamepad.left_joystick(x_value=-32767, y_value=0)
            self.gamepad.update()
            time.sleep(0.4)
            self.gamepad.left_joystick(x_value=0, y_value=0)
            self.gamepad.update()
            time.sleep(0.3)

        time.sleep(2)

        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(0.3)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()

        self.session_id += 1
        self.session_start_time = time.time()
        print(f"🔁 Skill game restarted. Session {self.session_id} begins.")
        self.previous_score = 0
        self.agent_can_act = True
    def _detect_teleport(self):
        if self._raw_att_center is None or self._raw_att_prev is None:
            return False

        dist = math.hypot(
            self._raw_att_center[0] - self._raw_att_prev[0],
            self._raw_att_center[1] - self._raw_att_prev[1]
        )

        if dist >= self.TELEPORT_JUMP_PX and self.frame_count > 5:
            self._tele_jump_confirm += 1
        else:
            self._tele_jump_confirm = 0

        return self._tele_jump_confirm >= 1

    def _log_reward(self, valid_ratio):
        hud_bgr = np.array(self.sct.grab(self.hud_crop))[:, :, :3]
        gray = cv2.cvtColor(hud_bgr, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray, detail=0, paragraph=False)

        import re
        digits = [re.sub(r"\D+", "", r) for r in results]
        digits = [d for d in digits if d]
        prev = getattr(self, "previous_score", 0)
        new_score = int(digits[0]) if digits else prev

        raw_reward = max(0, new_score - prev)
        self.previous_score = new_score

        print(f"📷 OCR: {results} -> parsed: {new_score}")
        print(f"🧮 Attempt {self.attempt_count}: {self.valid_frame_count} valid / {self.frame_count} total → {valid_ratio:.2%}")

        if valid_ratio < 0.6:
            raw_reward = 0
            print(f"⚠️ Discarded — only {valid_ratio:.2%} valid frames. raw_reward={raw_reward}")
        else:
            print(f"✅ Passed — valid ratio {valid_ratio:.2%}. raw_reward={raw_reward}")

        return raw_reward, float(valid_ratio), int(self.valid_frame_count), int(self.frame_count)

    def _start_skill_game(self):
        print("🎮 Starting Skill Game sequence...")
        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(0.3)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(0.8)

        self.gamepad.left_trigger(value=255)
        self.gamepad.update()
        time.sleep(0.3)
        self.gamepad.left_trigger(value=0)
        self.gamepad.update()
        time.sleep(0.8)

        self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(0.3)
        self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(4)

        self.session_start_time = time.time()
        print("✅ Skill game started. Beginning session 1.")
    def _observe(self):
        frame = self._get_screen()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        results = self.model(rgb, verbose=False)
        detections = []
        for box in results[0].boxes:
            if int(box.cls[0]) != 0:
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            width = x2 - x1
            expand = width * 0.2
            x1 = max(0, x1 - expand / 2)
            x2 = x2 + expand / 2
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf])

        detections = np.array(detections, dtype=np.float32).reshape(-1, 5)
        tracked = self.tracker.update(detections, rgb.shape[:2], rgb.shape[:2])
        attacker, goalie, attacker_tracked = self._assign_roles(tracked)
        self.attacker_tracked_this_frame = attacker_tracked

        if self._vid_writer is not None:
            frame_draw = rgb.copy()
            for obj in tracked:
                x1, y1, w, h = map(int, obj.tlwh)
                x2, y2 = x1 + w, y1 + h
                tid = obj.track_id
                color = (200, 200, 200)
                label = f"ID {tid}"
                if tid == getattr(self, "attacker_id", None):
                    color = (0, 255, 255)
                    label = "Attacker"
                elif tid == getattr(self, "goalkeeper_id", None):
                    color = (255, 100, 100)
                    label = "Goalkeeper"
                cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_draw, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            self._write_attempt_frame(frame_draw)

        return self._build_observation(attacker, goalie)

    def _assign_roles(self, tracked_objects):
        attacker_pos = None
        goalkeeper_pos = None
        attacker_tracked = False
        goalie_tracked = False

        if len(tracked_objects) < 2:
            return (
                self.last_attacker_pos or (-1, -1),
                self.last_goalkeeper_pos or (-1, -1),
                False
            )

        centers = [(obj.track_id, (obj.tlwh[0] + obj.tlwh[2] / 2, obj.tlwh[1] + obj.tlwh[3] / 2))
                   for obj in tracked_objects]

        if self.last_attacker_pos and self.last_goalkeeper_pos:
            attacker_tid, attacker_center, min_att = None, None, float('inf')
            for tid, c in centers:
                d = np.linalg.norm(np.array(c) - np.array(self.last_attacker_pos))
                if d < min_att:
                    attacker_tid, attacker_center, min_att = tid, c, d

            goalie_tid, goalie_center, min_goal = None, None, float('inf')
            for tid, c in centers:
                if tid == attacker_tid:
                    continue
                d = np.linalg.norm(np.array(c) - np.array(self.last_goalkeeper_pos))
                if d < min_goal:
                    goalie_tid, goalie_center, min_goal = tid, c, d

            self.attacker_id = attacker_tid
            self.goalkeeper_id = goalie_tid
            if attacker_tid is not None:
                attacker_tracked = True
                attacker_pos = attacker_center
            if goalie_tid is not None:
                goalie_tracked = True
                goalkeeper_pos = goalie_center
        else:
            sorted_by_x = sorted(centers, key=lambda c: c[1][0])
            self.attacker_id = sorted_by_x[0][0]
            self.goalkeeper_id = sorted_by_x[1][0]
            attacker_tracked = True
            goalie_tracked = True
            attacker_pos = sorted_by_x[0][1]
            goalkeeper_pos = sorted_by_x[1][1]
            print(f"✅ Initial assignment: Attacker = ID {self.attacker_id}, Goalkeeper = ID {self.goalkeeper_id}")

        self._raw_att_center = attacker_pos if attacker_tracked else None
        if attacker_tracked:
            self.last_attacker_pos = smooth(self.last_attacker_pos, attacker_pos)
        if goalie_tracked:
            self.last_goalkeeper_pos = smooth(self.last_goalkeeper_pos, goalkeeper_pos)

        if not attacker_tracked and self.last_attacker_pos:
            attacker_pos = self.last_attacker_pos
        if not goalie_tracked and self.last_goalkeeper_pos:
            goalkeeper_pos = self.last_goalkeeper_pos

        if attacker_tracked and not goalie_tracked:
            goalkeeper_pos = self.goal_fallback

        JUMP_THRESHOLD = 300
        if attacker_tracked and self.last_attacker_pos and attacker_pos:
            dist_att = np.linalg.norm(np.array(attacker_pos) - np.array(self.last_attacker_pos))
            if dist_att > JUMP_THRESHOLD:
                print(f"⚠️ Attacker jumped {int(dist_att)} px on frame {self.frame_count}")
        if goalie_tracked and self.last_goalkeeper_pos and goalkeeper_pos:
            dist_goal = np.linalg.norm(np.array(goalkeeper_pos) - np.array(self.last_goalkeeper_pos))
            if dist_goal > JUMP_THRESHOLD:
                print(f"⚠️ Goalkeeper jumped {int(dist_goal)} px on frame {self.frame_count}")

        return (
            self.last_attacker_pos or (-1, -1),
            self.last_goalkeeper_pos or (self.goal_fallback if attacker_tracked else (-1, -1)),
            attacker_tracked
        )

    def _build_observation(self, attacker, goalie):
        fallback = self.goal_fallback
        attacker_x, attacker_y = attacker if attacker else (-1, -1)
        goalie_x, goalie_y = goalie if goalie else fallback

        dx = goalie_x - attacker_x
        dy = goalie_y - attacker_y

        ax_n = (attacker_x / max(1.0, self.obs_w)) * 2 - 1
        ay_n = (attacker_y / max(1.0, self.obs_h)) * 2 - 1
        gx_n = (goalie_x / max(1.0, self.obs_w)) * 2 - 1
        gy_n = (goalie_y / max(1.0, self.obs_h)) * 2 - 1

        d_n = float(np.hypot(dx, dy)) / max(1.0, self.CROP_DIAG)
        ang = math.atan2(dy, dx)
        cos_a, sin_a = math.cos(ang), math.sin(ang)

        obs = [ax_n, ay_n, gx_n, gy_n, d_n, cos_a, sin_a]

        self.log_rows.append([
            self.session_id, self.attempt_count,
            attacker_x, attacker_y, goalie_x, goalie_y,
            d_n, ang
        ])

        return np.array(obs, dtype=np.float32)
    def _get_screen(self):
        return np.array(self.sct.grab(self.player_crop))

    def close(self):
        self._end_attempt_video()

        write_header = not os.path.exists(self.input_csv)
        with open(self.input_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "session_id", "attempt_id", "frame_num",
                    "attacker_x", "attacker_y", "goalie_x", "goalie_y",
                    "dx", "dy", "joy_x", "joy_y"
                ])
            writer.writerows(self.input_log)

        write_header = not os.path.exists(self.frame_csv)
        with open(self.frame_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "session_id", "attempt_id", "frame_num",
                    "attacker_x", "attacker_y", "goalie_x", "goalie_y",
                    "dx", "dy", "joy_x", "joy_y", "valid", "delta_d_norm"
                ])
            writer.writerows(self.frame_log)

        write_header = not os.path.exists(self.reward_csv)
        with open(self.reward_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "session_id", "attempt_id",
                    "raw_reward", "shaped_reward",
                    "valid_ratio", "valid_frame_count", "frame_count",
                    "cap_triggered", "teleport_ended",
                    "net_movement"
                ])
            writer.writerows(self.reward_log)

        write_header = not os.path.exists(self.motion_csv)
        with open(self.motion_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["session_id", "attempt_id", "forward_frames", "backward_frames"])
            w.writerows(self.motion_log)

        shots_path = getattr(self, "shots_csv", "fifa_rl_shots_v2.csv")
        write_header = not os.path.exists(shots_path)
        with open(shots_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "session_id", "attempt_id", "frame_num",
                    "shoot_raw", "shoot_hold_s", "shoot_max_t",
                    "cap_forced", "shot_dist_norm", "finesse"
                ])
            writer.writerows(self.shot_log)

        print("🛑 Closing environment.")
