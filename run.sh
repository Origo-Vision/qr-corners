#!/bin/bash

echo "Epochs=100 loss=bce batch-size=4"
python qr/train.py --datadir-train datasets/train-15000-116433 --datadir-valid  datasets/valid-1000-691112 --model-size tiny --loss bce --batch-size 4 --epochs 100
