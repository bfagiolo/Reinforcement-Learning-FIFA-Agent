# Reinforcement-Learning-FIFA-Agent with Kylian Mbappe

This project is a reinforcement learning agent that learns to play FIFA 19 entirely from raw gameplay video. It combines real-time computer vision with a custom PPO implementation to train a bot that dribbles, shoots, and scores in EA’s Shooting Skill Challenge. The result is not just a high-scoring agent, but one that has discovered genuine soccer techniques like finesse shots, near-post finishes, and subtle low strikes—learned purely through reinforcement.
The entire training is based on the famous super star Kylian Mbappe, who was a main icon in FIFA 19 and also an integral part of Paris Saint Germain. 

The pipeline begins with computer vision. A YOLO detector identifies the attacker, goalkeeper, and goal in every frame, while ByteTrack links those detections over time so the agent understands continuity and player identity. From these detections, a compact feature vector is created each frame: distance to goal, lateral offset, recent net progress, shot cooldown, and other normalized values scaled to [-1,1]. This transforms a complex game screen into a stable, structured input that reinforcement learning can process.
The control output mirrors how a human would actually play FIFA. Each timestep, the policy chooses a left-stick direction for dribbling, a continuous shot power value, and a binary finesse toggle. By keeping the action space natural and controller-like, the agent learns skills that look and feel authentic on screen.

The reinforcement learning backbone is a custom PPO implementation with carefully designed reward shaping. Small positive rewards are given for forward progress toward the goal, while penalties are applied if an attempt ends farther away than it began—ensuring backward dribbling is discouraged. Shots on target earn partial credit, while goals trigger higher rewards, and an efficiency bonus is applied for clean, well-timed finishes. This balance of dense feedback guides the agent away from random exploits and toward repeatable, high-quality play.
An additional innovation was splitting entropy across the different action heads in PPO. Exploration in continuous joystick control, shot power, and the binary finesse toggle needed different weights to avoid over- or under-exploring. By tuning gamma coefficients separately, the agent explored movement more aggressively early on, experimented with power timing moderately, and sampled finesse shots just enough to discover when they were optimal. This entropy design was crucial for stable convergence.

The most exciting outcome was seeing the agent discover real soccer finishing techniques. It learned to use finesse shots across the keeper when the far-post angle was open, to drive powerfully near-post when space was tight, and to slip low shots under the goalkeeper from close range. None of these were hard-coded; they emerged naturally through PPO, vision, and reward shaping. In other words, the bot didn’t just maximize score—it learned how to finish like a striker.

Full Demo here: https://youtu.be/GUY7TWwpd3A 


<img width="1318" height="601" alt="Screenshot 2025-08-24 at 7 21 02 PM" src="https://github.com/user-attachments/assets/59ee1927-6d8c-48f3-886d-b8384f7a5b34" />

![images](https://github.com/user-attachments/assets/71070ec8-c2d1-4276-8141-9c3f65f9dbcc)

