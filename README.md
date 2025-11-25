# Introduction

This is the code for the experiments in the paper **Qualitative and Quantitative Analysis of Riemannian Optimization Methods for Ground States of Rotating Multicomponent Bose-Einstein Condensates**. The aim of the experiments is to find the ground state, i.e. the state with the minimal energy for systems of coupled Gross-Pitaevskii equations with rotation. For details, we refer to the paper.

# Usage

To reproduce the experiments, you need to run `experiments/harmonic_trap.jl` for the two-component models and `experiments/harmonic_trap_three_component.jl` for the three-component model. The parameters can be set in the `@parameters` section in the beginning. Running these scripts will produce a file in the format `experiment-name_YYYY-MM-DD_HHMMSS` that contains the logging data as well as the final ground state and can be loaded by the `load_experiment` function. The best way to evaluate these results is via the provided files in the experiments folder.

## Environment

You should run the experiments and evaluations in the provided package environment, i.e., first run `using Pkg; Pkg.activate("./Riemannian-coupledGPE-rotation/"); Pkg.instantiate();` inside julia, and when calling the files, use the `--project` flag, e.g. `julia --project=Riemannian-coupledGPE-rotation/ experiments/harmonic_trap.jl`.
