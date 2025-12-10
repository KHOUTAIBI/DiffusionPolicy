# DiffusionPolicy

<img src="https://github.com/KHOUTAIBI/DiffusionPolicy/blob/main/gifs/pusht_gif.gif" width="250" height="250"/>

Code for experimenting with Diffusion Policy and related generative policies for visuomotor control on standard imitation-learning benchmarks (Kitchen / PushT), plus a small DDPM playground and report material.

The code is **not** a drop-in replacement for the original Stanford `diffusion_policy` repo. It is a self-contained student project used to understand and reimplement the main ideas.

---

## Features

- Minimal DDPM implementation for 1D / low-dimensional trajectories (`ddpm/`).
- Diffusion Policy–style action denoising for:
  - **FrankaKitchen** (Gymnasium Robotics / Robomimic kitchen task).
  - **PushT** (2D pushing task).
- Flow-matching / ODE-based variant for PushT (`pusht_flow/`), to compare against DDPM-style diffusion.
---

## Repository structure

Rough overview of the main directories:

- `ddpm/`  
  Standalone DDPM implementation and utilities:
  - Noise schedule, forward / backward process.
  - Training loop for simple trajectory or toy datasets.
  - Useful for debugging diffusion before plugging it into robotics.

- `kitchen/`  
  Code for Franka Kitchen visuomotor policies:
  - Environment setup using `gymnasium` and `gymnasium_robotics` (e.g. `FrankaKitchen-v1`).
  - Data loading / trajectory format for demonstrations.
  - Training and evaluation code for Diffusion Policy on kitchen tasks.
  - Logging / saving checkpoints and rollouts.

- `pusht/`  
  Diffusion Policy on the PushT environment:
  - Observation / action preprocessing.
  - Diffusion transformer / UNet1D for action sequences.
  - Training + evaluation on offline demonstrations.

- `pusht_flow/`  
  Flow-matching / ODE-based policy:
  - Continuous-time parametrization of trajectories.
  - Training to match score / velocity field instead of discrete DDPM.
  - Same evaluation protocol as `pusht/` for fair comparison.
    
- `useful_papers/`  
  Notes and references on:
  - Diffusion models (DDPM, DDIM, score-based models).
  - Diffusion Policy and follow-up work in robotics.
  - Flow matching / probability flow ODEs.

---
