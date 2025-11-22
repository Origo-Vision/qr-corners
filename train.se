#!/bin/bash

# Best model this far. Validation error of 2.59.

echo "Epochs=35 batch-size=4"
python qr/train.py --datadir-train datasets/multi-train-15000-116433 --datadir-valid  datasets/multi-valid-1000-691112 --epochs 35 --batch-size 4 --loss bcedice --learning-rate 2e-3
