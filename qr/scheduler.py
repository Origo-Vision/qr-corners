from __future__ import annotations

import math


class Scheduler:
    """
    A simple learning rate scheduler.
    """

    def __init__(
        self: Scheduler, lr: float, annealing: str, epochs: int, warmup: int = 3
    ) -> None:
        """
        Construct the scheduler.

        Parameters:
            lr: The maximum learning rate.
            annealing: The annealing function (cosine and linear are implemented).
            epochs: The total number of epochs.
            warmup: The warmup period.
        """
        self.lr = lr
        self.annealing = annealing
        self.epochs = epochs - warmup

        self.warmup = warmup
        self.epoch = 0
        self.scale = 0.1

    def learning_rate(self: Scheduler) -> float:
        """
        Get the current learning rate.

        Returns:
            The learning rate.
        """
        return self.lr * self.scale

    def step(self: Scheduler) -> None:
        """
        Step the scheduler and change it's scaling factor.
        """
        if self.warmup > 1:
            self.warmup -= 1
        else:
            if self.annealing == "cosine":
                self.scale = math.cos((self.epoch / self.epochs) * (math.pi / 2.0))
            elif self.annealing == "linear":
                self.scale = 1.0 - self.epoch / self.epochs
            else:
                self.scale = 1.0

            self.epoch += 1
