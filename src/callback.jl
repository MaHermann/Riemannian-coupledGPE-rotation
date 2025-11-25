abstract type Callback end

struct LoggingCallback{V, VVI, VV, VS<:VariableStorage} <: Callback
    energies::V
    n_inner_iterations::VVI
    solver_resnorms::VV
    τs::VV
    σs::VV
    h_norm_distances::VV
    residual_values::VV
    variable_storage::VS
    reference_solution
    L
end

function LoggingCallback(n, p, max_iter; T = COMPLEX_TYPE, reference_solution = nothing, L = nothing)
    variable_storage = VariableStorage(n, p, T)
    energies = zeros(max_iter)
    residual_values = ones(max_iter, p)
    solver_resnorms = zeros(max_iter, 2 * p)
    n_inner_iterations = zeros(Int, max_iter, 2 * p)
    τs = zeros(max_iter, p)
    σs = zeros(max_iter, p)
    if !isnothing(reference_solution)
        h_norm_distances = ones(max_iter, p)
    else
        h_norm_distances = ones(max_iter, 0)
    end
    return LoggingCallback(
        energies, n_inner_iterations, solver_resnorms, τs, σs, 
        h_norm_distances, residual_values, variable_storage, reference_solution, L,
    )
end

function call!(callback::LoggingCallback, variable_storage::VariableStorage)
    variables = variable_storage.variables
    n_step = variables["n_step"][1]
    ϕ = variables["ϕ"]
    gpe_system = variables["gpe_system"]
    current_energy = energy(
        ϕ, gpe_system;
        variable_storage = callback.variable_storage,
        update_gpe_system = false,
    )
    callback.energies[n_step] = current_energy
    if size(callback.h_norm_distances, 2) > 0
        for i in 1:size(gpe_system.interactions, 1)
            diff = ϕ[i] .- callback.reference_solution[i]
            callback.h_norm_distances[n_step, i] = sqrt(real(diff ⋅ (callback.L * diff)))
        end
    end
    @views residuals!(
        callback.residual_values[n_step, :], ϕ, gpe_system;
        variable_storage =  callback.variable_storage,
        update_gpe_system = false,
    )
    invalidate_cache!(callback.variable_storage)
    solver = variables["solver"]
    if !isempty(solver.history)
        for (i, history) in enumerate(solver.history)
            callback.n_inner_iterations[n_step, i] = history.iters
            if history.iters > 0
                callback.solver_resnorms[n_step, i] = history[:resnorm][end]
            end
        end
    end
    callback.τs[n_step,:] .= variables["τ"]
    callback.σs[n_step,:] .= variable_storage.σ
    # cleanup
    solver.history = ConvergenceHistory[]
end

function call!(callback::Function, variable_storage)
    callback(variable_storage)
end