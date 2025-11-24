#!/bin/bash

echo "Training dataset"
python qr/data.py generate --datadir datasets/train-30000-116433 --seed 116433 --samples 30000 --multi

echo "Validation dataset"
python qr/data.py generate --datadir datasets/valid-3000-691112 --seed 691112 --samples 3000 --multi