# CutQC
CutQC is the backend codes for the paper [CutQC: using small quantum computers for large quantum circuit evaluations](https://dl.acm.org/doi/10.1145/3445814.3446758).
CutQC cuts a large quantum circuits into smaller subcircuits and run on small quantum computers.
By combining classical and quantum computation, CutQC significantly expands the computational reach beyond either platform alone.

## Important note:
There are currently no fault tolerant quantum computers available.
As a result, the perfect fidelity toolchain of CutQC has to rely on classical simulators.
Therefore, using CutQC nowadays will NOT provide better performance than purely classical simulations.
However, with the rapid development of the various hardware vendors,
CutQC is expected to achieve the advantage discussed in the paper over either quantum or classical platforms.

This code repo hence provides two CutQC backends:
1. Using classical simulators as the ``QPU'' backend.
2. Using random number generator as the ``QPU'' backend.
Use this mode if you are just interested in the runtime performance of CutQC.

## Latest Developments
- Added GPU support

## Installation
1. Make a Python virtual environment:
```
conda create cutqc python=3.12
conda activate cutqc
```
2. CutQC uses the [Gurobi](https://www.gurobi.com) solver. Obtain and install a Gurobi license.
Follow the [instructions](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer).
3. Install required packages:
```
pip install -r requirements.txt
```

> Note on Qiskit Version: If you get warnings about conditionals '==' and 'is', switching to qiskit==0.45.2 may fix the issue.

## Example Reconstruction
See `explaning_example.md` for running the example scripts. 

## Citing CutQC
If you use CutQC in your work, we would appreciate it if you cite our paper:

Tang, Wei, Teague Tomesh, Martin Suchara, Jeffrey Larson, and Margaret Martonosi. "CutQC: using small quantum computers for large quantum circuit evaluations." In Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, pp. 473-486. 2021.

## Contact Us
Please open an issue here. Please reach out to [Wei Tang](https://www.linkedin.com/in/weitang39/).

## TODO
- [ ] Qubit reorder function