function gradient!(
    gradient::PFrame{p}, ϕ::PFrame{p}, gpe_system, gradient_componentwise!;
    solver = nothing, variable_storage = nothing, update_gpe_system = true,
    kwargs...
) where {p}
    isnothing(solver) && (solver = get_default_solver(p))
    isnothing(variable_storage) && (variable_storage = VariableStorage(ϕ))
    update_gpe_system && update!(gpe_system, ϕ)
    for i in 1:p
        gradient[i] .= gradient_componentwise!(
            gradient[i], ϕ[i], gpe_system, i, solver, variable_storage;
            kwargs...
        )
    end
    return gradient
end

# Aliases
function gradient_energy_adaptive!(
    gradient, ϕ, gpe_system;
    solver = nothing, variable_storage = nothing,
    update_gpe_system = true,
    kwargs...,
)
    gradient!(
        gradient, ϕ, gpe_system, gradient_energy_adaptive_componentwise!;
        solver, variable_storage, update_gpe_system,
    )
end

function gradient_energy_adaptive(
    ϕ::PFrame{p}, gpe_system;
    solver = nothing, variable_storage = nothing,
    update_gpe_system = true,
    kwargs...,
) where {p}
    gradient_energy_adaptive!(
        PFrame(SVector{p}(zeros(ComplexF64, size(ϕ[i], 1)) for i in 1:p)), ϕ, gpe_system;
        solver, variable_storage, update_gpe_system,
    )
end

function gradient_Lagrangian!(
    gradient, ϕ, gpe_system;
    solver = nothing, variable_storage = nothing,
    update_gpe_system = true, ω = 1.0, kwargs...,
)
    gradient!(
        gradient, ϕ, gpe_system, gradient_Lagrangian_componentwise!;
        solver, variable_storage, update_gpe_system, ω, kwargs...,
    )
end

function gradient_Lagrangian(
    ϕ::PFrame{p}, gpe_system;
    solver = nothing, variable_storage = nothing,
    update_gpe_system = true, ω = 1.0, kwargs...,
) where {p}
    gradient_Lagrangian!(
        PFrame(SVector{p}(zeros(ComplexF64, size(ϕ[i], 1)) for i in 1:p)), ϕ, gpe_system;
        solver, variable_storage, update_gpe_system, ω, kwargs...,
    )
end

# Componentwise gradients
function gradient_energy_adaptive_componentwise!(
    u, ϕᵢ, gpe_system, i, solver, variable_storage,
    kwargs...,
)
    n = size(ϕᵢ, 1)
    a(y, x) = mul_hamiltonian!(y, x, gpe_system, i)
    # TODO Check again whats happening here with eltype and if we may need to treat
    # this as an r-linear map
    A = LinearMap{eltype(ϕᵢ)}(a, n, issymmetric = false, ismutating = true)

    Mϕ = get_Mϕ!(variable_storage, ϕᵢ, gpe_system, i)
    R  = get_R!( variable_storage, ϕᵢ, gpe_system, i)
    massᵢ = gpe_system.masses[i]

    u .= R
    if !isnothing(gpe_system.grid_context.constraint_handler)
        apply!(u, gpe_system.grid_context.constraint_handler)
    end

    # compute the relative tolerance in dependence of the residual
    abstol_old = solver.abstol
    reltol_old = solver.reltol
    r = get_r!(variable_storage, ϕᵢ, gpe_system, i)
    solver.abstol = r * solver.reltol
    solver.reltol = 0.0

    solve!(u, A, R, solver, i)
    if !isnothing(gpe_system.grid_context.constraint_handler)
        apply!(u, gpe_system.grid_context.constraint_handler)
    end

    solver.abstol = abstol_old
    solver.reltol = reltol_old

    σ_R = real((Mϕ ⋅ u) / massᵢ)
    σ_R = 1 / (1 - σ_R)

    axpy!(-1, ϕᵢ, u)
    lmul!(-σ_R, u)

    axpy!(-1, ϕᵢ, u)
    lmul!(-1, u)
    return u
end

# We have to be careful here, as we use CG as a solver, and we have no guarantee
# that G is positive definite at a random point (but it should be if it is close
# enough to a critical point)
function gradient_Lagrangian_componentwise!(
    u, ϕᵢ, gpe_system, i, solver, variable_storage;
    ω = 1.0, kwargs...,
)
    Mϕ = get_Mϕ!(variable_storage, ϕᵢ, gpe_system, i)
    R  = get_R!( variable_storage, ϕᵢ, gpe_system, i)
    σ  = get_σ!( variable_storage, ϕᵢ, gpe_system, i)
    temp_n = variable_storage.temp_n

    function g_ω_real(y, x)
        x_complex = reinterpret(COMPLEX_TYPE, x)
        y_complex = reinterpret(COMPLEX_TYPE, y)
        mul_hamiltonian!(y_complex, x_complex, gpe_system, i)
        mul_B!(temp_n, x_complex, gpe_system, i, i)
        y_complex .+= temp_n
        mul_M!(temp_n, x_complex, gpe_system)
        axpy!(-ω*σ, temp_n, y_complex)
        return y
    end
    G_ω_real = LinearMap(g_ω_real, size(u, 1)*2, issymmetric = false, ismutating = true)
    
    # compute the relative tolerance in dependence of the residual
    abstol_old = solver.abstol
    reltol_old = solver.reltol
    r = get_r!(variable_storage, ϕᵢ, gpe_system, i)
    solver.abstol = r * solver.reltol
    solver.reltol = 0.0

    u .= R
    u_real = reinterpret(REAL_TYPE, u)
    R_real = reinterpret(REAL_TYPE, R)
    solve!(u_real, G_ω_real, R_real, solver, i)
    if !isnothing(gpe_system.grid_context.constraint_handler)
        apply!(u, gpe_system.grid_context.constraint_handler)
    end

    # renaming for clarity
    v = R
    v_real = reinterpret(REAL_TYPE, v)
    Mϕ_real = reinterpret(REAL_TYPE, Mϕ)
    solve!(v_real, G_ω_real, Mϕ_real, solver, i)
    if !isnothing(gpe_system.grid_context.constraint_handler)
        apply!(v, gpe_system.grid_context.constraint_handler)
    end
   
    solver.abstol = abstol_old
    solver.reltol = reltol_old

    σ_G = real(Mϕ ⋅ u) / real(Mϕ ⋅ v)
    axpy!(-σ_G, v, u)
    return u
end
