# implementation research system

## Overview

1. spot sequence of the timestamp of the video
2. extract candidates of additional information by querying from the data (tracking player name, action, previsou comments, etc.) to the database (strings wikipedia.com)
3. generate a comment based on the extracted information corresponding to the each timestamp.

## input data

- spot sequence (json file)
- comment csv file

## usage so far

```bash
# 1 prepare input data and fix paths in the scripts

# 2 construct query for retrieving additional information
scripts/construct_query_comments.sh

# 3 generate candidates of additional information
scripts/addinfo_retrieval.sh
```
