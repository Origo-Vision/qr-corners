#!/bin/bash

echo "Epochs=100 scheduler=cosine"
python qr/train.py --datadir-train datasets/train-15000-116433 --datadir-valid  datasets/valid-1000-691112 --epochs 100 --scheduler cosine
