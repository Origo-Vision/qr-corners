import argparse
import pathlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import UNet
from qrdataset import QRDataset
import util


def train(options: argparse.Namespace) -> None:
    device = util.find_device(options.force_cpu)
    print(f"Device={device}")

    train = QRDataset(datadir=options.datadir_train)
    valid = QRDataset(datadir=options.datadir_valid)

    print(f"Images in the training dataset={len(train)}")
    print(f"Images in the validation dataset={len(valid)}")

    train_loader = DataLoader(train, batch_size=options.batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=options.batch_size)

    model = UNet(in_channels=3, out_channels=4).to(device)
    print(f"Number of model parameters={util.count_parameters(model)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(options.epochs):
        print(f"epoch {epoch+1:4d}/{options.epochs:4d}")

        print("Training ...")
        model.train()

        num_train_batches = len(train_loader)
        accum_train_loss = 0.0
        accum_train_accuracy = 0.0
        for batch, data in enumerate(train_loader):
            print(f"\r  batch {batch+1:4d}/{num_train_batches:4d} ... ", end="")

            Xb, Yb, Pb = data
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            Pb = Pb.to(device)

            Ypred = model(Xb)

            loss = loss_fn(Ypred, Yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accum_train_loss += loss.item()
            accum_train_accuracy += util.mean_point_accuracy(
                util.batch_heatmap_points(Ypred), Pb
            ).item()

        avg_train_loss = accum_train_loss / num_train_batches
        avg_train_accuracy = accum_train_accuracy / num_train_batches
        print(
            f"\r  avg loss={avg_train_loss:.5f}, avg accuracy={avg_train_accuracy:.2f}"
        )

        print("Validation ...")
        model.eval()

        num_valid_batches = len(valid_loader)
        accum_valid_loss = 0.0
        accum_valid_accuracy = 0.0
        with torch.no_grad():
            for batch, data in enumerate(valid_loader):
                print(f"\r  batch {batch+1:4d}/{num_valid_batches:4d} ... ", end="")

                Xb, Yb, Pb = data
                Xb = Xb.to(device)
                Yb = Yb.to(device)
                Pb = Pb.to(device)

                Ypred = model(Xb)

                loss = loss_fn(Ypred, Yb)

                accum_valid_loss += loss.item()
                accum_valid_accuracy += util.mean_point_accuracy(
                    util.batch_heatmap_points(Ypred), Pb
                ).item()

            avg_valid_loss = accum_valid_loss / num_valid_batches
            avg_valid_accuracy = accum_valid_accuracy / num_valid_batches
            print(
                f"\r  avg loss={avg_valid_loss:.5f}, avg accuracy={avg_valid_accuracy:.2f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--datadir-train",
        type=pathlib.Path,
        required=True,
        help="The data directory for training",
    )
    parser.add_argument(
        "--datadir-valid",
        type=pathlib.Path,
        required=True,
        help="The data directory for validation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=[1, 2, 4, 8, 16, 32],
        default=16,
        help="The batch size",
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

    train(options)
