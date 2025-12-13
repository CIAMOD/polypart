import argparse
import sys

from . import cli_batch, cli_single


def main():
    parser = argparse.ArgumentParser(
        prog="polypart.experiments", description="Experiment runners"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sp_single = subparsers.add_parser("single", help="Run a single experiment")
    sp_single.add_argument("-d", type=int, required=True)
    sp_single.add_argument("-p", type=str, required=True)
    sp_single.add_argument("-a", type=str, required=True)
    sp_single.add_argument("-m", type=int, default=None)
    sp_single.add_argument("--degen_ratio", type=float, default=0.0)
    sp_single.add_argument("--n_runs", type=int, default=1)
    sp_single.add_argument("--exclude", type=str, nargs="*", default=[])
    sp_single.add_argument("-o", "--output", type=str, default="./data")

    sp_batch = subparsers.add_parser("batch", help="Run experiments from config")
    sp_batch.add_argument("config", type=str)
    sp_batch.add_argument("-r", "--runs", type=int, default=10)
    sp_batch.add_argument("-w", "--workers", type=int, default=None)
    sp_batch.add_argument("-o", "--output", type=str, default="./data/cluster")
    sp_batch.add_argument("--exclude", type=str, nargs="*", default=[])

    args = parser.parse_args()

    if args.command == "single":
        # Reconstruct argv for cli_single
        sys.argv = ["cli_single"] + [
            "-d",
            str(args.d),
            "-p",
            args.p,
            "-a",
            args.a,
            "-m",
            str(args.m) if args.m is not None else "0",
            "--degen_ratio",
            str(args.degen_ratio),
            "--n_runs",
            str(args.n_runs),
            "-o",
            args.output,
            *(["--exclude", *args.exclude] if args.exclude else []),
        ]
        cli_single.main()
    elif args.command == "batch":
        sys.argv = ["cli_batch"] + [
            args.config,
            "-r",
            str(args.runs),
            *(["-w", str(args.workers)] if args.workers is not None else []),
            "-o",
            args.output,
            *(["--exclude", *args.exclude] if args.exclude else []),
        ]
        cli_batch.main()


if __name__ == "__main__":
    main()
