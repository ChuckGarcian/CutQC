from typing import Optional
from datetime import timedelta

import torch.distributed as dist
import os
from enum import Enum


class Protocol(Enum):
    """Communication Protocol used
    See following for explination on communication backends: https://docs.pytorch.org/docs/stable/distributed.html
    """

    NCCL = "nccl"
    GLOO = "mpi"
    MPI = "gloo"


class Device(Enum):
    """Computation backend used"""

    GPU = "GPU"
    CPU = "CPU"


# TODO: Fix
# context manager so that the 'with' keyword works. And also fix enter and exit correct


class CutQCDistributed:
    @classmethod
    def init(
        cls,
        computation_device: Device,
        communication_protocol: Protocol,
        world_rank: int,
        world_size: int,
        gpus_per_node: int,
        timeout: Optional[int] = 600,
    ) -> None:
        """
        Sets up to call the distributed kernel. Worker nodes

        Args:
            comm_backend: message passing backend internally used by pytorch for
                        sending data between nodes.
            world_rank:   Global Identifier.
            world_size:   Total number of nodes.
            timeout:      Max amount of time pytorch will let any one node wait on
                        a message before killing it.
        """

        instance = cls()
        instance.__enter__(
            computation_device,
            communication_protocol,
            world_rank,
            world_size,
            gpus_per_node,
            timeout,
        )

    @staticmethod
    def exit():
        """
        Sends signal to workers to finish their execution.
        """
        self._exit__()

    def __enter__(
        self,
        computation_device: Device,
        communication_protocol: Protocol,
        world_rank: int,
        world_size: int,
        gpus_per_node: int,
        timeout: int,
    ) -> None:
        # Start the group
        local_rank = world_rank - gpus_per_node * (
            world_rank // gpus_per_node
        )  # GPU identifer on local compute cluster

        timelimit = timedelta(hours=timeout)  # Bounded wait time to prevent deadlock

        # Initializes pytorch for multiproccessing

        dist.init_process_group(
            backend=communication_protocol.value,
            rank=world_rank,
            world_size=world_size,
            timeout=timelimit,
        )

        # Referenced in distributed_graph_contractor
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["PYTORCH"] = "True"
        os.environ["COMPUTATION_DEVICE"] = computation_device.value

        if world_rank == 0:
            os.environ["HOST"] = "True"

    def __exit__(self) -> None:
        mp_backend = os.environ["COMMUNICATION_PROTOCOl"]

        termination_signal = torch.tensor([-1], dtype=torch.int64).to(mp_backend)
        for rank in range(1, dist.get_world_size()):
            dist.send(termination_signal, dst=rank)

        dist.destroy_process_group()
