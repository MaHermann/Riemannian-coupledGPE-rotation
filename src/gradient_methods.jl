function riemannian_gradient_descent(
    ϕ₀::PFrame{p},
    gpe_system,
    gradient_function!;
    update_diagonal_Bs = false,
    kwargs...,
) where {p}
    arguments = Dict{Symbol, Any}(kwargs)

    retraction! = get_retraction(arguments)
    iteration = get_iteration(arguments)
    termination_criterion = get_termination_criterion(arguments)
    callback = get_callback(arguments)
    step_size = get_step_size(arguments)
    solver = get_solver(arguments, p)
    variable_storage = VariableStorage(ϕ₀)
    gradient_kwargs = get_gradient_kwargs(arguments)

    # preallocate and initialize
    ϕ = deepcopy(ϕ₀)
    if !isnothing(gpe_system.grid_context.constraint_handler)
        for i in 1:p
            apply!(ϕ[i], gpe_system.grid_context.constraint_handler)
        end
    end
    riemannian_gradient = PFrame(SVector{p}(zero(ϕ₀[i]) for i in 1:p))
    n_step = [0]                # array to have fixed memory location
    τ = zeros(p)
    variable_storage.variables["ϕ"] = ϕ
    variable_storage.variables["riemannian_gradient"] = riemannian_gradient
    variable_storage.variables["gpe_system"] = gpe_system
    variable_storage.variables["n_step"] = n_step
    variable_storage.variables["τ"] = τ
    variable_storage.variables["solver"] = solver

    update!(gpe_system, ϕ; update_diagonal_Bs)

    loop_concurrently!(
        ϕ, riemannian_gradient, gradient_function!, gpe_system, iteration,
        callback, termination_criterion, variable_storage, step_size, retraction!,
        solver, n_step, τ, update_diagonal_Bs, gradient_kwargs,
    )

    return ϕ
end

# Function barrier for type stabillity
function loop_concurrently!(
    ϕ::PFrame{p}, riemannian_gradient, gradient_function!, gpe_system, iteration,
    callback, termination_criterion, variable_storage, step_size, retraction!,
    solver, n_step, τ, update_diagonal_Bs, gradient_kwargs
) where {p}
    for step in iteration
        n_step[1] = step

        if is_met(termination_criterion, ϕ, gpe_system; variable_storage)
            !isnothing(callback) && call!(callback, variable_storage)
            break
        end

        riemannian_gradient = gradient_function!(
            riemannian_gradient, ϕ, gpe_system;
            solver, variable_storage, update_gpe_system = false,
            gradient_kwargs...
        )
        for i in 1:p
            τ[i] = determine(
                step_size, ϕ, gpe_system, riemannian_gradient, i;
                retraction!, variable_storage, solver = solver,
            )
            lmul!(-1 * τ[i], riemannian_gradient[i])
        end
        retraction!(
            ϕ, riemannian_gradient, gpe_system;
            variable_storage, update_gpe_system = false,
        )
        update!(gpe_system, ϕ; update_diagonal_Bs)
        !isnothing(callback) && call!(callback, variable_storage)
    end
end


# Aliases

gradient_descent_energy_adaptive(ϕ₀, gpe_system; kwargs...) =
    riemannian_gradient_descent(ϕ₀, gpe_system, gradient_energy_adaptive!; kwargs...)

gradient_descent_Lagrangian(ϕ₀, gpe_system; ω = 1.0, kwargs...) =
    riemannian_gradient_descent(
        ϕ₀, gpe_system, gradient_Lagrangian!;
        ω, update_diagonal_Bs = true, kwargs...,
    )
