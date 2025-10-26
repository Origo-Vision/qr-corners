import argparse
import pathlib

import torch
import torch.nn.functional as F

from models import UNet
from qrdataset import QRDataset
import util


def overfit(options: argparse.Namespace) -> None:
    device = util.find_device(options.force_cpu)
    print(f"Selected device={device}")

    dataset = QRDataset(datadir=options.datadir)
    indices = torch.randint(0, len(dataset), size=(options.batch_size,))

    Xb, Yb, _ = dataset.multi_sample(indices)
    Xb = Xb.to(device)
    Yb = Yb.to(device)

    model = UNet(in_channels=3, out_channels=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)

    model.train()
    for epoch in range(options.epochs):
        Yp = model(Xb)

        loss = F.mse_loss(Yp, Yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"{epoch+1:5d}/{options.epochs:5d} loss={loss.item():.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--datadir", type=pathlib.Path, required=True, help="The data directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=[1, 2, 4, 8, 16, 32],
        default=4,
        help="The batch size for overfitting.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="The number of epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="The learning rate"
    )
    parser.add_argument(
        "--force-cpu", action="store_true", help="Force execution on the CPU"
    )
    parser.add_argument("--seed", type=int, default=1598, help="The random seed")
    options = parser.parse_args()

    util.set_seed(options.seed)

    overfit(options)
