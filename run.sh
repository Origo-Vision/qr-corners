#!/bin/bash

echo "Epochs=50 scheduler=cosine"
python qr/train.py --datadir-train datasets/train-15000-116433 --datadir-valid  datasets/valid-1000-691112 --epochs 50 --scheduler cosine

echo "Epochs=50 scheduler=linear"
python qr/train.py --datadir-train datasets/train-15000-116433 --datadir-valid  datasets/valid-1000-691112 --epochs 50 --scheduler linear

echo "Epochs=50 scheduler=none"
python qr/train.py --datadir-train datasets/train-15000-116433 --datadir-valid  datasets/valid-1000-691112 --epochs 50 --scheduler none
