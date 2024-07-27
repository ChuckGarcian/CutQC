"""
File: distributed_graph_contraction.py
Original Author: Wei Tang (tangwei13579@gmail.com)
Current Version Author: Charles "Chuck" Garcia (chuckgarcian@utexas.edu)
Description: Distributed implementation of Wei Tang's original TensorFlow CutQC implementation.
"""

import itertools
from time import perf_counter
from typing import List, Optional
import numpy as np
import torch
import torch.distributed as dist
from cutqc.abstract_graph_contractor import AbstractGraphContractor
from cutqc.post_process_helper import ComputeGraph


__host_machine__ = 0


class DistributedGraphContractor(AbstractGraphContractor):
    """
     Distributed Graph Contractor Implementation
    
     Args:
            local_rank (int): Node identifier value
            compute_backend (str): Device used for compute (Default is GPU)
            
    """
    def __init__(self, local_rank: Optional[int] = None, compute_backend: str = 'gpu') -> None:
        self.local_rank = local_rank
        
        # Set up compute devices based on backend
        self.mp_backend = torch.device(f"cuda:{local_rank}" if dist.get_backend() == 'nccl' else "cpu") # Deviced used MP
        self.compute_device = torch.device(f"cuda:{local_rank}") if compute_backend == 'gpu' else self.mp_backend
        self.is_gpu = compute_backend == 'gpu'
        
        print ("Worker {}, compute_device: {}".format (dist.get_rank(), self.compute_device), flush=True)

        if dist.get_rank() != __host_machine__:
            self._initiate_worker_loop()
        
        self.times = {'compute': 0}
        self.compute_graph = None
        self.subcircuit_entry_probs = None
        self.reconstructed_prob = None


    def terminate_distributed_process(self):
        """
        Sends signal to workers to finish their execution.
        """
        termination_signal = torch.tensor([-1], dtype=torch.int64).to(self.mp_backend)
        for rank in range(1, dist.get_world_size()):
            dist.send(termination_signal, dst=rank)
        
        print(f"DESTROYING NOW! {self.times['compute']}", flush=True)
        dist.destroy_process_group()

    def _get_paulibase_probability (self, edge_bases: tuple, edges: list):
        """
        Returns probability contribution for the basis 'edge_bases' in the circuit
        cutting decomposition.
        """
        with torch.no_grad():
            self.compute_graph.assign_bases_to_edges(edge_bases=edge_bases, edges=edges)

            # Create list of kronecker product terms
            flat_size = np.sum(self.subcircuit_entry_lengths)
            flat = torch.empty(flat_size)
            idx = 0
            
            # Store all probability tensors into single flattened tensor
            for size, subcircuit_idx in zip(self.subcircuit_entry_lengths, self.smart_order):
                subcircuit_entry_prob = self._get_subcircuit_entry_prob(subcircuit_idx)
                flat[idx:idx+size] = torch.tensor(subcircuit_entry_prob, dtype=torch.float32)
                idx += size

        return flat

    def _send_distributed(self, dataset: List[torch.Tensor], num_batches: int) -> torch.Tensor:
        """
        Decomposes `dataset` list into 'num_batches' number of batches and distributes
        to worker processes.
        """
        torch.set_default_device(self.mp_backend)

        with torch.no_grad():
            print ("LEN(DATASET): {}".format (len(dataset)), flush=True)
            print ("NUMBER BATCHES: {}".format (num_batches), flush=True)
            if len(dataset) < num_batches:
                raise ValueError("Error 2000: Invalid number of requested batches -- Too many nodes allocated, for dataset length {} and {} number of batches".format (len(dataset), num_batches))
            
            batches = torch.stack(dataset).tensor_split(num_batches)
            tensor_sizes = torch.tensor(self.subcircuit_entry_lengths, dtype=torch.int64)
            tensor_sizes_shape = torch.tensor(tensor_sizes.shape, dtype=torch.int64)

            if dist.get_backend() == 'gloo':
                op_list = []
                # List of sending objects
                for dst, batch in enumerate(batches, start=1):
                    op_list.extend([
                        dist.P2POp(dist.isend, tensor_sizes_shape, dst),
                        dist.P2POp(dist.isend, tensor_sizes, dst),
                        dist.P2POp(dist.isend, torch.tensor(batch.shape, dtype=torch.int64), dst),
                        dist.P2POp(dist.isend, batch, dst),
                    ])
                handles = dist.batch_isend_irecv(op_list)
            else:
                # NCCL backend
                for dst_rank, batch in enumerate(batches, start=1):
                    # Non-Blocking send on NCCL
                    dist.isend(tensor_sizes_shape, dst=dst_rank)
                    dist.isend(tensor_sizes, dst=dst_rank)
                    dist.isend(torch.tensor(batch.shape), dst=dst_rank)
                    dist.isend(batch.to(self.compute_device), dst=dst_rank)
            
            # Receive Results
            output_buff = torch.zeros(self.result_size, dtype=torch.float32)
            dist.reduce(output_buff, dst=0, op=dist.ReduceOp.SUM)
        
        return torch.mul(output_buff, (1/2**self.num_cuts))

    def _compute(self) -> np.ndarray:
        """
        Performs distributed graph contraction. Returns the reconstructed probability.
        """
        edges = self.compute_graph.get_edges(from_node=None, to_node=None)
        summation_terms_sequence = []

        # Assemble sequence of uncomputed kronecker products
        for edge_bases in itertools.product(["I", "X", "Y", "Z"], repeat=len(edges)):
            summation_terms = self._get_paulibase_probability(edge_bases, edges)
            summation_terms_sequence.append(summation_terms)

        self.compute_graph.remove_bases_from_edges(edges=self.compute_graph.edges)
        
        # Distribute and Execute reconstruction on nodes
        num_batches = dist.get_world_size() - 1  # No batch for host
        reconstructed_prob = self._send_distributed(summation_terms_sequence, num_batches)

        return reconstructed_prob.cpu().numpy()
    

    def _receive_from_host(self):
        """
        Receives tensors sent by host. Returns batch and unpadded sizes.
        """
        torch.set_default_device(self.mp_backend)
        torch.cuda.device(self.compute_device)
        if (self.is_gpu): torch.cuda.device(self.compute_device)
        
        with torch.no_grad():
            tensor_sizes_shape = torch.empty([1], dtype=torch.int64)
            dist.recv(tensor=tensor_sizes_shape, src=0)
            
            # Check for termination signal
            if tensor_sizes_shape.item() == -1:
                print(f"WORKER {dist.get_rank()} DYING", flush=True)
                dist.destroy_process_group()
                exit()

            # Used to unflatten
            tensor_sizes = torch.empty(tensor_sizes_shape, dtype=torch.int64)
            dist.recv(tensor=tensor_sizes, src=0)

            # Get shape of the batch we are receiving
            batch_shape = torch.empty([2], dtype=torch.int64)
            dist.recv(tensor=batch_shape, src=0)
            
            # Create an empty batch tensor and receive its data
            batch = torch.empty(tuple(batch_shape), dtype=torch.float32)
            dist.recv(tensor=batch, src=0)

        return batch_shape[0], batch, tensor_sizes

    def _initiate_worker_loop(self):
        """
        Primary worker kernel

        Each worker receives a portion of the workload from the host/master node.
        Once done with computation, all nodes perform a collective reduction
        operation back to the host. Synchronization among nodes is provided via
        barriers and blocked message passing.
        """
        from pprint import pprint
        
        while True:
            with torch.no_grad():  # Disable gradient computation if not needed               
                torch.cuda.device(self.compute_device)
                num_samples, batch, tensor_sizes = self._receive_from_host()                        
                SAFETY_MARGIN = 7e9  # 1 GB safety margin

                # Get available GPU memory
                gpu_free, _ = torch.cuda.mem_get_info()
                gpu_free -= SAFETY_MARGIN  # Leave some safety margin

                # Calculate memory requirements
                element_size = batch.element_size()
                total_N_elements = torch.prod(tensor_sizes)
                single_sample_size = element_size * total_N_elements

                # Calculate maximum number of samples that fit in GPU memory
                max_tile_size = gpu_free // single_sample_size

                # If even a single item doesn't fit, raise an error
                if max_tile_size == 0:
                    raise ValueError("Error 2006: Not enough GPU memory to process even a single kronecker sample")

                # Calculate number of tiles 
                import math
                N_tiles = math.ceil(num_samples / max_tile_size)

                print(f"Processing in {N_tiles} tiles, max batch size: {max_tile_size}")

                # Prepare vectorized function
                vec_fn = torch.func.vmap(lambda x: compute_kronecker_product(x, tensor_sizes))

                # Process in tiles
                result = None
                curr_idx = 0
                
                for i in range(N_tiles):
                    # Stride by tiles
                    print ("index: {}".format (i), flush=True)
                    
                    delta = int(min ((num_samples - curr_idx), max_tile_size))                  
                    tile = batch[:delta]  # Get current tile 
                    batch = batch[delta:] # next tile
                    curr_idx += delta

                    tile_result = vec_fn(tile).sum(dim=0)                    
                    
                    if result is None:
                        result = tile_result
                    else:
                        result.add(tile_result)
                    
                    # Clear cache after each processed tile
                    del tile
                    del tile_result
                    torch.cuda.empty_cache()

            # Send Back to host
            dist.reduce(result.to(self.mp_backend), dst=__host_machine__, op=dist.ReduceOp.SUM)
            

from functools import reduce
def compute_kronecker_product(flattened: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    """
    Micro_kernel:
    Computes sequence of Kronecker products, where operands are tensors in 'components'.
    """
    tensors = torch.split(flattened, tuple(sizes))
    return reduce(torch.kron, tensors)
