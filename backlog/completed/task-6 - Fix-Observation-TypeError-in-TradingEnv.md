---
id: TASK-6
title: Fix Observation TypeError in TradingEnv
status: Done
assignee: []
created_date: '2026-05-13 12:37'
updated_date: '2026-05-13 12:38'
labels: []
dependencies: []
priority: high
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the TypeError in the RL environment by dropping rows where orderbook data is missing after joining.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Successfully adds `.drop_nulls(subset=["bids", "asks"])` to `load_aligned_data` in `src/data/loader.py`.
- [x] #2 `learn.py` starts without raising `TypeError` in `_get_observation`.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the `TypeError: Value after * must be an iterable, not NoneType` in `TradingEnv._get_observation` by adding `.drop_nulls(subset=["bids", "asks"])` to the aligned data in `TradingDataLoader.load_aligned_data`. This ensures that any trades occurring before the first orderbook snapshot (which result in `null` orderbook columns after a `join_asof`) are excluded from the training dataset. Verified with `learn.py`.
<!-- SECTION:FINAL_SUMMARY:END -->
