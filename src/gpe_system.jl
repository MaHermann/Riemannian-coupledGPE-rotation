"""
    GPESystem{T, dim, refshape, G, CSC_T, CI, V, VV, M, V_T}

    This manages the information and state of the GPE that is used. This includes
    the concrete potential, masses and nonlinearities, but also the Finite Element
    grid and the mass matrices of the current ϕ.

    To interact with this, the preferred interface is via the `update!` function,
    which should be called after changing the current ϕ, and the three functions
    `mul_hamiltonian!`, `mul_M!` and `mul_B!`, that carry out in-place multiplications
    with the hamiltonian, the mass matrix and the second order matrices B_ij,
    respectively.
"""
struct GPESystem{T, C, dim, CV, CH, G, CSC_T<:SparseMatrixCSC{T, Int64},
                CSC_C<:SparseMatrixCSC{C, Int64}, CI, V_CSC_C<:Vector{CSC_C},
                V_CSC_T<:Vector{CSC_T}, VVV_CSC_T<:Vector{Vector{V_CSC_T}},
                M<:Matrix{T}, V_T<:Vector{T}}
    potential::Potential
    fixed_part::V_CSC_C
    hamiltonian::V_CSC_C
    weighted_mass_matrices::V_CSC_T
    B_matrices::VVV_CSC_T
    interactions::M
    masses::V_T
    grid_context::GridContext{T, dim, CV, CH, G, CSC_T, CI}
    Ne::M
end

function GPESystem(potential, interactions, masses, omega, grid_context)
    L = assemble_stiffness_matrix(grid_context)
    R = assemble_rotation_matrix(grid_context)
    fixed_part = [
        allocate_matrix(SparseMatrixCSC{COMPLEX_TYPE, Int}, grid_context.dof_handler)
        for _ in 1:size(masses, 1)
    ]
    for i in 1:size(masses, 1)
        fixed_part[i].nzval .= L.nzval .+  matrix_representation(potential, i).nzval .+
            omega[i] .* 1im .* R.nzval # the main issue is we need the structural zeros
    end
    return GPESystem(
        potential, fixed_part,
        [
            allocate_matrix(
                SparseMatrixCSC{COMPLEX_TYPE, Int},
                grid_context.dof_handler,
            ) for i in 1:size(masses, 1)
        ],
        [allocate_matrix(grid_context.dof_handler) for _ in 1:size(masses, 1)],
        [
            [
                [allocate_matrix(grid_context.dof_handler) for _ in 1:4]
                for _ in 1:size(masses, 1)
            ]
            for _ in 1:size(masses, 1)
        ],
        interactions, masses, grid_context,
        zeros(
            getnbasefunctions(grid_context.cellvalues),
            getnbasefunctions(grid_context.cellvalues),
        ),
    )
end

function update!(gpe_system, ϕ::PFrame{p}; update_diagonal_Bs = false) where{p}
    if update_diagonal_Bs
        for i in 1:p
            assemble_B!(gpe_system, ϕ, i, i)
        end
    end
    update_diagonal_weighted_mass_matrices!(gpe_system, ϕ; use_B = update_diagonal_Bs)
    update_hamiltonian!(gpe_system)
end

function update_diagonal_weighted_mass_matrices!(
    gpe_system, ϕ::PFrame{p};
    use_B = false,
) where {p}
    for i in 1:p
        update_weighted_mass_matrix!(gpe_system, ϕ, i; use_B)
    end
end

function update_weighted_mass_matrix!(gpe_system, ϕ, i; use_B = false)
    if use_B
        gpe_system.weighted_mass_matrices[i].nzval .= gpe_system.B_matrices[i][i][1].nzval .+
            gpe_system.B_matrices[i][i][4].nzval
    else
        assemble_density_weighted_mass_matrix!(
            gpe_system.weighted_mass_matrices[i], ϕ[i],
            gpe_system.grid_context, gpe_system.Ne,
        )
    end
end

function update_hamiltonian!(gpe_system)
    p = size(gpe_system.masses, 1)
    for i in 1:p
        update_hamiltonian!(gpe_system, i)
    end
end

function update_hamiltonian!(gpe_system, i)
    p = size(gpe_system.masses, 1)
    gpe_system.hamiltonian[i] .= gpe_system.fixed_part[i]
    # This can fail if the fixed part contains zeros where the densities do not,
    # but this shouldn't be the case and this is much much faster
    for j in 1:p
        gpe_system.hamiltonian[i].nzval .+= gpe_system.interactions[j,i] .*
            gpe_system.weighted_mass_matrices[j].nzval
    end

end

function mul_hamiltonian!(v, u, gpe_system, i)
    mul!(v, gpe_system.hamiltonian[i], u)
    return v
end

function mul_M!(v, u, gpe_system)
    mul!(v, gpe_system.grid_context.M, u)
    return v
end

function assemble_B!(gpe_system, ϕ, i, j; compute_full = false)
    if i == j && !compute_full
        assemble_density_weighted_mass_matrix!(
            gpe_system.B_matrices[i][j][1], real(ϕ[i]),
            gpe_system.grid_context, gpe_system.Ne,
        )
        assemble_density_weighted_mass_matrix!(
            gpe_system.B_matrices[i][j][2], real(ϕ[i]), imag(ϕ[i]),
            gpe_system.grid_context, gpe_system.Ne,
        )
        gpe_system.B_matrices[i][j][3] .= gpe_system.B_matrices[i][j][2]
        assemble_density_weighted_mass_matrix!(
            gpe_system.B_matrices[i][j][4], imag(ϕ[i]),
            gpe_system.grid_context, gpe_system.Ne,
        )
    else
        assemble_density_weighted_mass_matrix!(
            gpe_system.B_matrices[i][j][1], real(ϕ[i]), real(ϕ[j]),
            gpe_system.grid_context, gpe_system.Ne,
        )
        assemble_density_weighted_mass_matrix!(
            gpe_system.B_matrices[i][j][2], real(ϕ[i]), imag(ϕ[j]),
            gpe_system.grid_context, gpe_system.Ne,
        )
        assemble_density_weighted_mass_matrix!(
            gpe_system.B_matrices[i][j][3], imag(ϕ[i]), real(ϕ[j]),
            gpe_system.grid_context, gpe_system.Ne,
        )
        assemble_density_weighted_mass_matrix!(
            gpe_system.B_matrices[i][j][4], imag(ϕ[i]), imag(ϕ[j]),
            gpe_system.grid_context, gpe_system.Ne,
        )
    end
    nothing
end

function mul_B!(v, u, gpe_system, i, j)
    v_as_r2 = reinterpret(Float64, v)
    v_real = @view v_as_r2[1:2:end-1]
    v_imag = @view v_as_r2[2:2:end]
    u_as_r2 = reinterpret(Float64, u)
    u_real = @view u_as_r2[1:2:end-1]
    u_imag = @view u_as_r2[2:2:end]
    mul!(v_real, gpe_system.B_matrices[i][j][1], u_real)
    v_real .+= gpe_system.B_matrices[i][j][2] * u_imag
    mul!(v_imag, gpe_system.B_matrices[i][j][3], u_real)
    v_imag .+= gpe_system.B_matrices[i][j][4] * u_imag
    lmul!(2 * gpe_system.interactions[i,j], v)
    return v
end
