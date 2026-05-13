---
id: TASK-7
title: Implement Parquet Utility Tool
status: Done
assignee: []
created_date: '2026-05-13 13:08'
updated_date: '2026-05-13 13:09'
labels: []
dependencies: []
priority: medium
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a utility script to quickly check the record count and basic metadata of Parquet files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create a script `src/utils/parquet_util.py`.
- [x] #2 The script should accept a file path or directory as an argument.
- [x] #3 It should print the number of records in the file(s).
- [x] #4 It should use `polars.scan_parquet` for memory efficiency.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented a new Parquet utility tool in `src/utils/parquet_util.py`. The tool uses `polars.scan_parquet` for memory-efficient metadata access, allowing users to quickly check record counts for individual files or entire directories. Verified functionality on both single files and directories.
<!-- SECTION:FINAL_SUMMARY:END -->
