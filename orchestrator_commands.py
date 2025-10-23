#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pymemcache>=4.0.0",
#   "mosaicml-cli",
# ]
# ///
# Copyright (c) 2025 Emerald AI
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

from pymemcache.client.base import Client
from pymemcache import serde
from mcli import Run, RunConfig, ComputeConfig
import mcli

memcache = Client(os.environ["MEMCACHED_HOST"], serde=serde.pickle_serde)

def dvfs_set_power_cap(run_name: str, power_cap: int, **kwargs) -> None:
    """Request a power cap on GPUs allocated to a given run.

    Args:
      run_name: Name of the target run
      power_cap: Power cap, in watts, to apply to each GPU associated with the run
    """
    settings = memcache.get(run_name) or {}
    settings["power_cap"] = power_cap
    memcache.set(run_name, settings)


def checkpoint(run_name: str, stop: bool, **kwargs) -> None:
    """Request a checkpoint on a given run, optionally stopping the run after checkpointing.

    Args:
      run_name: Name of the target run
      stop: Whether to stop the run after applying the checkpoint
    """
    settings = memcache.get(run_name) or {}
    settings["shutdown"] = stop
    settings["checkpoint_now"] = True
    memcache.set(run_name, settings)


def start_run(name: str, image: str, command: str, cluster_name: str, gpus: int, **kwargs) -> Run:
    """Start a new run.

    Args:
      name: Name to assign to the new run
      image: Docker image to use with the run (e.g. mosaicml/composer)
      command: Command to use for the run
      cluster_name: Which MosaicML cluster to use (see `mcli get clusters` for available clusters)
      gpus: Number of GPUs to allocate to the run
    """
    config = RunConfig(
        name=name,
        image=image,
        command=command,
        compute=ComputeConfig(
            cluster=cluster_name,
            gpus=gpus,
        ),
    )
    return mcli.create_run(config)


def stop(run: str | Run, **kwargs) -> None:
    """Stop an existing run.

    Args:
      run: Name of the run to stop
    """
    mcli.stop_run(run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True, help='Job commands')

    power_cap_parser = subparsers.add_parser("power_cap", help="Power-cap a run")
    power_cap_parser.add_argument('run_name', help='Name of the run to power cap')
    power_cap_parser.add_argument('power_cap', type=int, help='Power cap to apply on each GPU in the run')
    power_cap_parser.set_defaults(func=dvfs_set_power_cap)

    checkpoint_parser = subparsers.add_parser("checkpoint", help="Tell a run to checkpoint and optionally stop")
    checkpoint_parser.add_argument('run_name', help='Name of the run to checkpoint')
    checkpoint_parser.add_argument('--stop', action='store_true', help='Stop the run after checkpointing')
    checkpoint_parser.set_defaults(func=checkpoint)

    start_parser = subparsers.add_parser("start", help="Start a run")
    start_parser.add_argument('name', help='Name of the run to start')
    start_parser.add_argument('image', help='Name of the container image for the run')
    start_parser.add_argument('command', help='Command to run')
    start_parser.add_argument('cluster_name', help='Name of the mcli cluster to use')
    start_parser.add_argument('gpus', type=int, help='How many gpus to allocate for the run')
    start_parser.set_defaults(func=start_run)

    stop_parser = subparsers.add_parser("stop", help="Stop a run")
    stop_parser.add_argument('run', help='Name of the run to stop')
    stop_parser.set_defaults(func=stop)

    args = parser.parse_args()
    args.func(**vars(args))
