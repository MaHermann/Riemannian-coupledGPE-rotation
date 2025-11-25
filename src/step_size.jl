abstract type StepSize end

struct ConstantStepSize <: StepSize
    τ::Number
end

function determine(step_size::ConstantStepSize, ϕ, gpe_system, gradient, i; kwargs...)
    return step_size.τ
end

reset!(step_size::ConstantStepSize) = nothing

mutable struct Adaptive{T, P<:PFrame} <: StepSize
    τ₀::T
    τ_min::T
    τ_max::T
    cached_τ::T
    previous_direction::P
    previous_ϕ::P
    initialized::Bool
end

function Adaptive(p; τ₀ = 1.0, τ_max = Inf, τ_min = -Inf)
    return Adaptive(
        τ₀, τ_min, τ_max, τ₀,
        PFrame(SVector{p}([1.0im] for _ in 1:p)),
        PFrame(SVector{p}([1.0im] for _ in 1:p)),
        false,
    )
end

function determine(
    step_size::Adaptive, ϕ::PFrame{p}, gpe_system, direction, i;
    kwargs...,
) where {p}
    if !step_size.initialized
        step_size.previous_direction = deepcopy(direction)
        step_size.previous_ϕ = deepcopy(ϕ)
        step_size.initialized = true
        step_size.cached_τ = step_size.τ₀
        return step_size.τ₀
    end
    if i == 1
        d_x = 0.0
        d_g = 0.0
        for i in 1:p
            step_size.previous_ϕ[i] .-= ϕ[i]
            d_x += real(step_size.previous_ϕ[i] ⋅
                (gpe_system.grid_context.M * step_size.previous_ϕ[i]))

            step_size.previous_direction[i] .-= direction[i]
            d_g += real(step_size.previous_direction[i] ⋅
                (gpe_system.grid_context.M * step_size.previous_direction[i]))
        end

        τ = sqrt(d_x / d_g)

        τ = max(step_size.τ_min, min(step_size.τ_max, τ))
        step_size.cached_τ = τ

        for i in 1:p
            step_size.previous_direction[i] .= direction[i]
            step_size.previous_ϕ[i] .= ϕ[i]
        end
    else
        τ = step_size.cached_τ
    end
    return τ
end

function reset!(step_size::Adaptive)
    for i in 1:step_size.p
        step_size.initialized[i] = false
    end
    return nothing
end

mutable struct LineSearch{T, C, I, CSC_C<:SparseMatrixCSC{C, Int64},
                                CSC_T<:SparseMatrixCSC{T, Int64}} <: StepSize
    interval::I
    direction_matrix::CSC_T
    mixed_matrix::CSC_T
    S::CSC_C
    previous_τ::T
end

function LineSearch(interval, grid_context)
    direction_matrix = allocate_matrix(
        grid_context.dof_handler,
    )
    mixed_matrix = allocate_matrix(
        grid_context.dof_handler,
    )
    S = allocate_matrix(
        SparseMatrixCSC{COMPLEX_TYPE, Int},
        grid_context.dof_handler,
    )
    return LineSearch(interval, direction_matrix, mixed_matrix, S, 1.0)
end

function determine(
    step_size::LineSearch, ϕ, gpe_system, direction, i;
    variable_storage = VariableStorage(ϕ), kwargs...
)
    if i == 1
        p = size(gpe_system.interactions, 1)
        # these could also be preallocated in the step_size object,
        # but it should not be too relevant for performance
        ζ₀ = zeros(p)
        ζ₁ = zeros(p)
        ζ₂ = zeros(p)
        η₁ = zeros(p)
        η₂ = zeros(p)
        ξ₀ = zeros(p,p)
        ξ₁ = zeros(p,p)
        ξ₂ = zeros(p,p)
        ξ₄ = zeros(p,p)
        ξ₅ = zeros(p,p)
        ξ₈ = zeros(p,p)
        for j in 1:p
            assemble_density_weighted_mass_matrix!(
                step_size.direction_matrix, direction[j],
                gpe_system.grid_context, gpe_system.Ne,
            )
            assemble_density_weighted_mass_matrix!(
                step_size.mixed_matrix, real(ϕ[j]), real(direction[j]),
                gpe_system.grid_context, gpe_system.Ne,
            )
            assemble_density_weighted_mass_matrix!(
                step_size.mixed_matrix, imag(direction[j]), imag(ϕ[j]),
                gpe_system.grid_context, gpe_system.Ne;
                fillzero = false,
            )

            M = gpe_system.grid_context.M
            S = step_size.S
            S .= gpe_system.fixed_part[j]

            v = variable_storage.temp_n

            mul!(v, S, ϕ[j])
            ζ₀[j] = real(ϕ[j] ⋅ v)
            ζ₁[j] = real(direction[j] ⋅ v)
            mul!(v, S, direction[j])
            ζ₂[j] = real(direction[j] ⋅ v)

            mul!(v, M, direction[j])
            η₁[j] = real(ϕ[j] ⋅ v)
            η₂[j] = real(direction[j] ⋅ v)

            # This only works because we update after the step size
            M_ϕϕ_j = gpe_system.weighted_mass_matrices[j]
            M_ϕd_j = step_size.mixed_matrix
            M_dd_j = step_size.direction_matrix

            # We do a little extra work here as we compute e.g. both ϕₖ'M_ϕϕ_jϕₖ
            # AND ϕⱼ'M_ϕϕ_kϕⱼ, even though they are the same
            for k in 1:p
                mul!(v, M_ϕϕ_j, ϕ[k])
                ξ₀[k,j] = real(ϕ[k] ⋅ v)

                mul!(v, M_ϕd_j, ϕ[k])
                ξ₁[k,j] = real(v ⋅ ϕ[k])
                ξ₄[k,j] = real(direction[k] ⋅ v)

                mul!(v, M_dd_j, ϕ[k])
                ξ₂[k,j] = real(v ⋅ ϕ[k])
                ξ₅[k,j] = real(v ⋅ direction[k])
                mul!(v, M_dd_j, direction[k])
                ξ₈[k,j] = real(v ⋅ direction[k])

            end
        end

        energy_(τ) = sum(
            gpe_system.masses[j] .* (ζ₀[j] - 2ζ₁[j]*τ + ζ₂[j]*τ^2) /
                (2*gpe_system.masses[j] - 4η₁[j]*τ + 2η₂[j]*τ^2) +
            sum(
                gpe_system.masses[j] .* gpe_system.masses[k] .*
                gpe_system.interactions[j,k] *
                    (
                           ξ₀[j,k]      - 4ξ₁[j,k]*τ    + 2ξ₂[j,k]*τ^2
                                        + 4ξ₄[j,k]*τ^2  - 4ξ₅[j,k]*τ^3
                                                        +  ξ₈[j,k]*τ^4
                    ) /
                    (
                        (2*gpe_system.masses[j] - 4η₁[j]*τ + 2η₂[j]*τ^2) *
                        (2*gpe_system.masses[k] - 4η₁[k]*τ + 2η₂[k]*τ^2)
                    )
                for k in 1:p
            )
            for j in 1:p
        )

        energy_optimal = Inf
        τ_optimal = 0
        for τ in step_size.interval
            energy_current = energy_(τ)
            if energy_current < energy_optimal
                τ_optimal = τ
                energy_optimal = energy_current
            end
        end
        τ = τ_optimal
        step_size.previous_τ = τ
    else
        τ = step_size.previous_τ
    end
    return τ
end

reset!(step_size::LineSearch) = nothing
