#!/usr/bin/env bash
spark-submit \
  --master local[*] \
  --driver-memory 4G \
  src/fraud_unsup/pipelines/train.py
