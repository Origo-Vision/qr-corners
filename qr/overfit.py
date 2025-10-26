import argparse
import pathlib

import torch

from models import UNet
import util


def overfit(options: argparse.Namespace) -> None:
    device = util.find_device(options.force_cpu)
    print(f"Selected device={device}")

    # model = UNet(in_channels=3, out_channels=4)

    # print(f"model count={util.count_parameters(model)}")

    # # print(model)

    # x = torch.randn((1, 3, 256, 256))
    # y = model(x)
    # print(y.shape)


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
        "--force-cpu", action="store_true", help="Force execution on the CPU"
    )
    parser.add_argument("--seed", type=int, default=1598, help="The random seed")
    options = parser.parse_args()

    util.set_seed(options.seed)

    overfit(options)
