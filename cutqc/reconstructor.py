from cutqc.cutqc_model import CutQCModel
from cutqc.dynamic_definition import DynamicDefinition
from typing import Optional

import os


class CircuitReconstructor:
    """
    The main module for CutQC
    cut --> evaluate results --> verify (optional)
    """

    def __init__(
        self,
        cutqc_model: CutQCModel,
        mem_limit: int,
        recursion_depth: int,
        verbose: Optional[bool] = False,
    ):
        """
        --- Distributed Reconstruction Related Arguments ---

        cutqc_model: `CutQCModel` containing subcircuitt output vectors

        """
        self.mem_limit = mem_limit
        self.recursion_depth = recursion_depth
        self.times = {}
        self.cutqc_model = cutqc_model
        self.verbose = verbose

        self.pytorch_distributed = False
        self.local_rank = None
        self.compute_backend = None

        if os.environ["PYTORCH"] == "True":
            self.local_rank = os.environ["LOCAL_RANK"]
            self.compute_backend = os.environ["COMPUTATION_DEVICE"]
            self.pytorch_distributed = True

    def build(self):
        """
        mem_limit: memory limit during post process. 2^mem_limit is the largest vector
        """
        if self.verbose:
            print("--> Build %s" % (self.name))

        ## Resume
        # RESUME: So I need to figure out how to connectthe environment variables for distrbuted, e.g pytrochdistrbuted local rank, comptue backend ect, too where it is being used bellow.

        self.dd = DynamicDefinition(
            cutqc_model=self.cutqc_model,
            mem_limit=self.mem_limit,
            recursion_depth=self.recursion_depth,
            pytorch_distributed=self.pytorch_distributed,
            local_rank=self.local_rank,
            compute_backend=self.compute_backend,
        )
        self.dd.build()

        # self.times = add_times(times_a=self.times, times_b=self.dd.times)
        print(type(self.dd.dd_bins))

        self.cutqc_model.approximation_bins = self.dd.dd_bins
        self.cutqc_model._is_reconstructed = True
        self.num_recursions = len(self.dd.dd_bins)
        self.overhead = self.dd.overhead

        # self.times["build"] = perf_counter() - build_begin
        # self.times["build"] += self.times["cutter"]
        # self.times["build"] -= self.times["merge_states_into_bins"]

        if self.verbose:
            print("Overhead = {}".format(self.overhead))

        return self.dd.graph_contractor.times["compute"]
