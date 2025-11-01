#!/bin/bash

echo "Epochs=20 batch-size=2"
python qr/train.py --datadir-train datasets/train-15000-116433 --datadir-valid  datasets/valid-1000-691112 --epochs 20 --batch-size 2

echo "Epochs=20 batch-size=4"
python qr/train.py --datadir-train datasets/train-15000-116433 --datadir-valid  datasets/valid-1000-691112 --epochs 20 --batch-size 4

echo "Epochs=20 batch-size=8"
python qr/train.py --datadir-train datasets/train-15000-116433 --datadir-valid  datasets/valid-1000-691112 --epochs 20 --batch-size 8

echo "Epochs=20 batch-size=16"
python qr/train.py --datadir-train datasets/train-15000-116433 --datadir-valid  datasets/valid-1000-691112 --epochs 20 --batch-size 16

echo "Epochs=20 batch-size=32"
python qr/train.py --datadir-train datasets/train-15000-116433 --datadir-valid  datasets/valid-1000-691112 --epochs 20 --batch-size 32
