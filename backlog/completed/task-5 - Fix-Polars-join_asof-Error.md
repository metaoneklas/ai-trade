---
id: TASK-5
title: Fix Polars join_asof Error
status: Done
assignee: []
created_date: '2026-05-13 12:29'
updated_date: '2026-05-13 12:30'
labels: []
dependencies: []
priority: high
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the 'module polars has no attribute join_asof' error by updating the code to use the DataFrame method instead of the top-level polars function.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Successfully replaces `pl.join_asof(...)` with `trades_df.join_asof(ob_df, ...)` in `src/data/loader.py`.
- [x] #2 Data loading process completes without `AttributeError`.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the `AttributeError: module 'polars' has no attribute 'join_asof'` by updating `src/data/loader.py` to use the `DataFrame.join_asof()` method instead of the top-level `pl.join_asof()` function. Verified the fix by running `learn.py`, which successfully loaded and aligned the data.
<!-- SECTION:FINAL_SUMMARY:END -->
