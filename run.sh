#!/bin/bash

# Best model this far. Validation error or 2.84.

echo "Epochs=35 batch-size=4"
python qr/train.py --datadir-train datasets/train-15000-116433 --datadir-valid  datasets/valid-1000-691112 --epochs 35 --batch-size 4 --loss bcedice

#echo "Epochs=35 batch-size=8"
#python qr/train.py --datadir-train datasets/train-15000-116433 --datadir-valid  datasets/valid-1000-691112 --epochs 35 --batch-size 8
