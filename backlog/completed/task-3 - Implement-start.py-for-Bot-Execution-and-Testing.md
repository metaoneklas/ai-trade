---
id: TASK-3
title: Implement start.py for Bot Execution and Testing
status: Done
assignee: []
created_date: '2026-05-12 05:41'
updated_date: '2026-05-12 05:43'
labels: []
dependencies:
  - TASK-2
modified_files:
  - start.py
priority: medium
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create `start.py` as an entry point to test and execute the RL bot. This script will load the environment defined in `learn.py`, potentially load a trained model, and run evaluation episodes to observe the agent's behavior in simulation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement a script to instantiate the environment from `learn.py`.
- [x] #2 Support loading a trained model (e.g., using `stable-baselines3`).
- [x] #3 Run a test loop (episode) and print performance metrics (total reward, profit/loss).
- [x] #4 Visualize agent actions or portfolio value over time (optional but recommended).
- [x] #5 Ensure the script is type-hinted and handles random seeds for reproducibility.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented `start.py` as the main evaluation and execution script. It leverages the environment setup from `learn.py`, supports loading Stable-Baselines3 models, and executes evaluation episodes with detailed performance reporting (net worth, rewards, fees). The script is CLI-ready with argument parsing and ensures reproducibility via seed management.
<!-- SECTION:FINAL_SUMMARY:END -->
