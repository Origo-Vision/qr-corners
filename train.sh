#!/bin/bash

echo "Epochs=35 batch-size=4"
python qr/train.py --datadir-train datasets/multi-train-30000-116433 --datadir-valid  datasets/multi-valid-3000-691112 --epochs 35 --batch-size 4 --loss bcedice --learning-rate 2e-3

echo "Epochs=35 batch-size=8"
python qr/train.py --datadir-train datasets/multi-train-30000-116433 --datadir-valid  datasets/multi-valid-3000-691112 --epochs 35 --batch-size 8 --loss bcedice --learning-rate 2e-3

echo "Epochs=35 batch-size=8, learning-rate=3e-3"
python qr/train.py --datadir-train datasets/multi-train-30000-116433 --datadir-valid  datasets/multi-valid-3000-691112 --epochs 35 --batch-size 8 --loss bcedice --learning-rate 3e-3

echo "Epochs=35 batch-size=8, learning-rate=3e-4"
python qr/train.py --datadir-train datasets/multi-train-30000-116433 --datadir-valid  datasets/multi-valid-3000-691112 --epochs 35 --batch-size 8 --loss bcedice --learning-rate 3e-4