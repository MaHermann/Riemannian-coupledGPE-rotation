using Ferrite

using RiemannianCoupledGPERotation

import Dates: format, now

## Model parameters
params = Dict([])
@parameters params begin
    dimension               = 2
    model                   = :strong
    n_components            = 2
    Ω                       = (1.0, 1.2)
    masses                  = [2.0, 1.0]
    int_12                  = model == :weak ? 20.0 : 60.0
    interactions            = [120.0   int_12;
                               int_12  100.0]
    potential_function      = [
                                x -> 0.75 * ((0.8*x[1])^2 + (1.2*x[2])^2),
                                x -> 0.5  * ((1.2*x[1])^2 + (0.9*x[2])^2),
                            ]
    domain                  = [(-10.0, -10.0), (10.0, 10.0)]
    boundary_condition      = :Dirichlet
end
## Numerical parameters
@parameters params begin
    n_elements              = 2^6     # per direction
    interp_degree           = 2
    quad_degree             = 5
    step_size               = 1#Adaptive(n_components)#LineSearch
    step_size_range         = 0.1:0.1:10
    ω                       = 0.0
    type                    = Ferrite.Quadrilateral
    optimization_algorithm  = gradient_descent_energy_adaptive
    solver_reltol           = 1e-8
    solver_max_iter         = nothing
    start_value             = :constant
    start_residual          = 1e-2
    termination_residual    = 1e-14
    max_iter                = 10_000
    initialization_max_iter = 1_000
    reference_path          = nothing#"path/to/experiment"
end
##
results_folder = "path/to/outputfolder/"
filename = "strong_eaRGD_constant" * format(now(), "yyyy-mm-dd_HHMMSS")
## Setup logging
n = get_n_dofs(dimension, n_elements, interp_degree, type)
grid_context = generate_grid_context(
    dimension, n_elements, interp_degree, quad_degree;
    left = domain[1], right = domain[2], type, boundary_condition,
)
if !isnothing(reference_path)
    reference_solution = load_experiment(reference_path).value 
else
    reference_solution = nothing
end
L = RiemannianCoupledGPERotation.assemble_stiffness_matrix(grid_context)
logging_callback = LoggingCallback(n, n_components, max_iter; reference_solution, L)
n_initialization_steps = [0]
initialization_residuals = zeros(n_components)
count_steps(_) = (n_initialization_steps[1] += 1)
residual_temp = zeros(n_components) # temp array for the termination_criterion
log = Dict(
    "energies" => logging_callback.energies,
    "residual_values" => logging_callback.residual_values,
    "n_inner_iterations" => logging_callback.n_inner_iterations,
    "solver_resnorms" => logging_callback.solver_resnorms,
    "τs" => logging_callback.τs,
    "h_norm_distances" => logging_callback.h_norm_distances,
	"n_initialization_steps" => n_initialization_steps,
	"initialization_residuals" => initialization_residuals,
)
##
@experiment params log begin
    # Setup
    grid_context = generate_grid_context(
        dimension, n_elements, interp_degree, quad_degree;
        left = domain[1], right = domain[2], type, boundary_condition,
    )
    if length(potential_function) == 1
        potential = ConstantPotential(potential_function, grid_context)
    else
        potential = ComponentwisePotential(potential_function, grid_context)
    end

    gpe_system = GPESystem(potential, interactions, masses, Ω, grid_context)

    # LineSearch is moved out to avoid logging the grid_context twice
    if step_size == LineSearch
        _step_size = LineSearch(
            step_size_range, grid_context,
        )
    else
        _step_size = step_size
    end

    if start_value == :constant
        ϕ₀ = constant_normed_PFrame(gpe_system; complex = true)
    elseif start_value == :analytical
        components = [
            zeros(
                RiemannianCoupledGPERotation.COMPLEX_TYPE,
                size(gpe_system.grid_context.M, 1),
            )
            for i in 1:n_components
        ]
        for i in 1:n_components
            Ferrite.apply_analytical!(
                components[i], grid_context.dof_handler, :u,
                x -> (x[1] + 1im*x[2])/(sqrt(π))*exp((x[1]^2 + x[2]^2)/(-2)),
            )
        end
        ϕ₀ = RiemannianCoupledGPERotation.normalized_PFrame(components, gpe_system)
    else
        ϕ₀ = random_normed_PFrame(gpe_system; complex = true)
    end
    preconditioner_initialization = get_A0preconditioner(gpe_system)
    ϕ₀ = gradient_descent_energy_adaptive(
        ϕ₀, gpe_system;
        max_iter = initialization_max_iter,
        termination_criterion = SumResidual(
            start_residual, false, residual_temp,
        ),
        solver = CGSolver(
            eltype(ϕ₀[1]);
            preconditioner = preconditioner_initialization, isLogging = false,
        ),
        verbose = true, callback = count_steps,
    )
    residuals!(initialization_residuals, ϕ₀, gpe_system)
    if optimization_algorithm == gradient_descent_Lagrangian
        preconditioner = get_A0preconditioner_real(gpe_system)
        solver_type = RiemannianCoupledGPERotation.REAL_TYPE
    else
        preconditioner = get_A0preconditioner(gpe_system)
        solver_type = eltype(ϕ₀[1])
    end
    # Optimization
    ϕ = optimization_algorithm(
        ϕ₀, gpe_system;
        max_iter = max_iter, step_size = _step_size, ω = ω,
        solver = CGSolver(
            solver_type;
            preconditioner = preconditioner, reltol = solver_reltol,
            maxiter = solver_max_iter, isLogging = true,
        ),
        termination_criterion = SumResidual(
            termination_residual, false, residual_temp,
        ),
        verbose = true, callback = logging_callback,
    )
end results_folder*filename
