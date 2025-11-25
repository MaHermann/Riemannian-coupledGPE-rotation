module RiemannianCoupledGPERotation

using Ferrite, SparseArrays
using LinearAlgebra
using IncompleteLU
using IterativeSolvers, LinearMaps
using Plots
using ProgressBars
using Serialization
using StaticArrays

import Dates: format, now

export CGSolver, get_A0preconditioner, get_A0preconditioner_real, get_identitypreconditioner

export Experiment, load_experiment, save_experiment, @parameters, @experiment

export GridContext, generate_grid_context, grid_points, get_n_dofs

export VariableStorage

export gradient_descent_energy_adaptive,
        alternating_gradient_descent_energy_adaptive,
        gradient_descent_Lagrangian,
        alternating_gradient_descent_Lagrangian

export GPESystem

export AbsoluteCriterion, SumResidual

export gradient_energy_adaptive, gradient_Lagrangian

export project_energy_adaptive, metric_energy_adaptive, normalization_retraction!

export ConstantStepSize, Adaptive, LineSearch, reset!

export PFrame, density, plot_PFrame, plot_PFrame!

export ConstantPotential, ComponentwisePotential, create_periodic_potential_2D,
        add_potentials, create_random_checkerboard_potential_2D,
        plot_potential_1D, plot_potential_1D!, plot_potential_2D,
        plot_potential_2D!

export energy, residuals, residuals!, random_normed_PFrame, constant_normed_PFrame

export LoggingCallback

include("fem_util.jl")
include("solver_util.jl")
include("pframe.jl")
include("variable_storage.jl")
include("potential.jl")
include("gpe_system.jl")
include("termination_criterion.jl")
include("gradients.jl")
include("retractions.jl")
include("step_size.jl")
include("argument_parsing.jl")
include("gradient_methods.jl")
include("experiment_util.jl")
include("util.jl")
include("callback.jl")

end
