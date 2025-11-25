abstract type Solver end

"""
    Mainly a thin wrapper around `IterativeSolvers.jl`s cg!

    Originally intended to be more flexible and e.g. also allow
    functions from LinearSolve.jl which have a slightly different
    interface, but for various reasons this turned out to be not
    necessary.
"""
mutable struct CGSolver{P,R,A,M,S} <: Solver
    preconditioner::Vector{P}
    isLogging::Bool
    abstol::R
    reltol::A
    maxiter::M
    history::Vector{ConvergenceHistory}
    statevariables::S
end

# Set all undefined values to the default values of Iterativesolvers.cg!
# This is not particularly clean as we may break on changes in IterativeSolvers.jl
# but its avoids a lot of dispatch issues
function CGSolver(
    T;
    preconditioner = [Identity()], isLogging = false,
    abstol = zero(real(T)), reltol = sqrt(eps(real(T))),
    maxiter = -1,
)
    isnothing(abstol) && (abstol = zero(real(T)))
    isnothing(reltol) && (reltol = sqrt(eps(real(T))))
    isnothing(maxiter) && (maxiter = -1)
    return CGSolver(
        preconditioner, isLogging, abstol, reltol, maxiter,
        ConvergenceHistory[], CGStateVariables(
            zeros(T, 1), zeros(T, 1), zeros(T, 1)
        ),
    )
end

function solve!(u, A, b, solver::CGSolver, i)
    # slight optimization to reuse statevars
    if size(u, 1) != size(solver.statevariables.u, 1)
        solver.statevariables = CGStateVariables(zero(u), similar(u), similar(u))
    else
        solver.statevariables.u .= 0
        solver.statevariables.r .= 0
        solver.statevariables.c .= 0
    end
    if solver.maxiter < 0
        maxiter = size(A, 2)
    else
        maxiter = solver.maxiter
    end

    if solver.isLogging
        u, ch = cg!(
            u, A, b;
            log = true, abstol = solver.abstol, reltol = solver.reltol,
            Pl = solver.preconditioner[i], maxiter = maxiter,
            statevars = solver.statevariables,
        )
        push!(solver.history, ch)
    else
        u = cg!(
            u, A, b;
            log = false, abstol = solver.abstol, reltol = solver.reltol,
            Pl = solver.preconditioner[i], maxiter = maxiter,
            statevars = solver.statevariables,
        )
        end
    return u
end

function get_A0preconditioner(gpe_system; τ = 0.01)
    preconditioner = []
    for i in 1:size(gpe_system.interactions, 1)
        push!(preconditioner, ilu(gpe_system.fixed_part[i], τ = τ))
    end
    return preconditioner
end

function get_A0preconditioner_real(gpe_system; τ = 0.01)
    preconditioner = []
    for i in 1:size(gpe_system.interactions, 1)
        fixed_part = gpe_system.fixed_part[i]
        F1 = reinterpret(REAL_TYPE, fixed_part)
        F2 = reinterpret(REAL_TYPE, 1im.*fixed_part)
        F = [zero(F1) zero(F2)]
        for i in 1:size(fixed_part, 1)
            F[:,2*i-1] .= @view F1[:,i]
            F[:,2*i] .= @view F2[:,i]
        end
        push!(preconditioner, ilu(F, τ = τ))
    end
    return preconditioner    
end

get_identitypreconditioner(p) = [Identity() for _ in 1:p]
