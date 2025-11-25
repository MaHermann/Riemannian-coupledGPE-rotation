struct GridContext{T<:Real, dim, CV, CH,
                    G<:Ferrite.AbstractGrid{dim}, CSC_T<:SparseMatrixCSC{T, Int64},
                    CI<:CellIterator}
    grid::G
    dof_handler::DofHandler{dim, G}
    constraint_handler::CH
    cellvalues::CV
    celliterator::CI
    M::CSC_T
end

function get_type(dimension, type = Union{Type, Nothing})
    if dimension == 1
        return Line
    elseif dimension == 2
        if isnothing(type)
            return Quadrilateral
        else
            return type
        end
    elseif dimension == 3
        if isnothing(type)
            return Hexahedron
        elseif type == Quadrilateral
            return Hexahedron
        else
            return type
        end
    else
        throw(ArgumentError("Unsupported dimension "*string(dimension)*"!"))
    end
end

function get_endpoint(dimension, endpoint)
    if typeof(endpoint) <: Tuple
        if size(endpoint, 1) == dimension
            return Vec{dimension}(endpoint)
        else
            throw("Expected "*string(dimension)*"D endpoints for "*
                    string(dimension)*"D grid!")
        end
    end
    return Vec{dimension}(ntuple(_ -> endpoint, dimension))
end

function get_ref_shape(dimension, type)
    if dimension == 1
        return RefLine
    elseif dimension == 2
        if type == Triangle
            return RefTriangle
        else
            return RefQuadrilateral
        end
    elseif dimension == 3
        if type == Tetrahedron
            return RefTetrahedron
        else
            return RefCube
        end
    else
        throw(ArgumentError("Unsupported dimension "*string(dimension)*"!"))
    end
end

# Note that this does have some overhead and there is probably a more efficient
# way to handle this, but for the usecases we have, its enough
function get_n_dofs(
    dimension, n_elements_per_direction, interpolation_degree, type,
)
    type = get_type(dimension, type)
    refShape = get_ref_shape(dimension, type)
    left = get_endpoint(dimension, 0.0)
    right = get_endpoint(dimension, 1.0)
    grid = generate_grid(
        type, ntuple(_ -> n_elements_per_direction, dimension), left, right,
    )
    dof_handler = DofHandler(grid)
    interpolation = Lagrange{refShape, interpolation_degree}()
    add!(dof_handler, :u, interpolation)
    close!(dof_handler)
    return ndofs(dof_handler)
end

function generate_grid_context(
    dimension, n_elements_per_direction, interpolation_degree, quadrature_degree;
    left = 0.0, right = 1.0, type = nothing, boundary_condition = nothing,
)
    type = get_type(dimension, type)
    refShape = get_ref_shape(dimension, type)
    left = get_endpoint(dimension, left)
    right = get_endpoint(dimension, right)
    grid = generate_grid(
        type, ntuple(_ -> n_elements_per_direction, dimension), left, right,
    )
    dof_handler = DofHandler(grid)
    interpolation = Lagrange{refShape, interpolation_degree}()
    add!(dof_handler, :u, interpolation)
    close!(dof_handler)
    quadrature = QuadratureRule{refShape}(quadrature_degree)
    cellvalues = CellValues(quadrature, interpolation)
    celliterator = CellIterator(dof_handler)
    if boundary_condition == :Dirichlet        # homogenoeus Dirichlet bc
        if dimension == 1
            ∂Ω = union(
            getfacetset(grid, "left"), getfacetset(grid, "right"),
        )
        elseif dimension == 2
            ∂Ω = union(
            getfacetset(grid, "left"), getfacetset(grid, "right"),
            getfacetset(grid, "top"), getfacetset(grid, "bottom"),
        )
        else
            error("Dirichlet Boundary conditions not yet implemented for 3D!")
        end
       constraint_handler = ConstraintHandler(dof_handler)
       bc = Dirichlet(:u, ∂Ω, x -> 0)
       add!(constraint_handler, bc)
       close!(constraint_handler)
    else
        constraint_handler = nothing
    end
    M = assemble_mass_matrix(dof_handler, celliterator, cellvalues, constraint_handler)
    return GridContext(grid, dof_handler, constraint_handler, cellvalues, celliterator, M)
end

grid_points(grid_context::GridContext) = [node.x for node in grid_context.grid.nodes]

function assemble_stiffness_matrix!(K, grid_context, Ke)
    cellvalues = grid_context.cellvalues
    assembler = start_assemble(K, )
    constraint_handler = grid_context.constraint_handler
    n_basefunctions = getnbasefunctions(cellvalues)
    for cell in grid_context.celliterator
        Ferrite.reinit!(cellvalues, cell)
        fill!(Ke, 0)
        for quadrature_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, quadrature_point)
            for i in 1:n_basefunctions
                ∇φᵢ = shape_gradient(cellvalues, quadrature_point, i)
                for j in 1:n_basefunctions
                    ∇φⱼ = shape_gradient(cellvalues, quadrature_point, j)
                    Ke[i,j] += (∇φᵢ ⋅ ∇φⱼ) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end
    if !isnothing(constraint_handler)
        apply!(K, constraint_handler)
    end
    return K
end

function assemble_stiffness_matrix(grid_context)
    return assemble_stiffness_matrix!(
        allocate_matrix(grid_context.dof_handler),
        grid_context,
        zeros(
            getnbasefunctions(grid_context.cellvalues),
            getnbasefunctions(grid_context.cellvalues),
        ),
    )
end

function assemble_mass_matrix!(M, grid_context, Me)
    cellvalues = grid_context.cellvalues
    assembler = start_assemble(M)
    constraint_handler = grid_context.constraint_handler
    n_basefunctions = getnbasefunctions(cellvalues)
    for cell in grid_context.celliterator
        Ferrite.reinit!(cellvalues, cell)
        fill!(Me, 0)
        for quadrature_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, quadrature_point)
            for i in 1:n_basefunctions
                φᵢ = shape_value(cellvalues, quadrature_point, i)
                for j in 1:n_basefunctions
                    φⱼ = shape_value(cellvalues, quadrature_point, j)
                    Me[i,j] += (φᵢ ⋅ φⱼ) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Me)
    end
    if !isnothing(constraint_handler)
        apply!(M, constraint_handler)
    end
    return M
end

function assemble_mass_matrix(grid_context)
    return assemble_mass_matrix!(
        allocate_matrix(grid_context.dof_handler),
        grid_context,
        zeros(
            getnbasefunctions(grid_context.cellvalues),
            getnbasefunctions(grid_context.cellvalues)
        ),
    )
end

function assemble_mass_matrix(dof_handler, celliterator, cellvalues, constraint_handler)
    n_basefunctions = getnbasefunctions(cellvalues)
    M = allocate_matrix(dof_handler)
    Me = zeros(n_basefunctions, n_basefunctions)
    assembler = start_assemble(M)
    for cell in celliterator
        Ferrite.reinit!(cellvalues, cell)
        fill!(Me, 0)
        for quadrature_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, quadrature_point)
            for i in 1:n_basefunctions
                φᵢ = shape_value(cellvalues, quadrature_point, i)
                for j in 1:n_basefunctions
                    φⱼ = shape_value(cellvalues, quadrature_point, j)
                    Me[i,j] += (φᵢ ⋅ φⱼ) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Me)
    end
    if !isnothing(constraint_handler)
        apply!(M, constraint_handler)
    end
    return M
end

function assemble_weighted_mass_matrix!(VM, V::Function, grid_context, VMe)
    cellvalues = grid_context.cellvalues
    assembler = start_assemble(VM)
    constraint_handler = grid_context.constraint_handler
    n_basefunctions = getnbasefunctions(cellvalues)
    n_quadpoints = getnquadpoints(cellvalues)
    for cell in grid_context.celliterator
        Ferrite.reinit!(cellvalues, cell)
        fill!(VMe, 0)
        for quadrature_point in 1:n_quadpoints
            dΩ = getdetJdV(cellvalues, quadrature_point)
            x = spatial_coordinate(cellvalues, quadrature_point, cell.coords)
            for i in 1:n_basefunctions
                φᵢ = shape_value(cellvalues, quadrature_point, i)
                for j in 1:n_basefunctions
                    φⱼ = shape_value(cellvalues, quadrature_point, j)
                    VMe[i, j] += V(x) * (φᵢ ⋅ φⱼ) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), VMe)
    end
    if !isnothing(constraint_handler)
        apply!(VM, constraint_handler)
    end
    return VM
end

function assemble_weighted_mass_matrix(V::Function, grid_context)
    return assemble_weighted_mass_matrix!(
        allocate_matrix(grid_context.dof_handler),
        V,
        grid_context,
        zeros(
            getnbasefunctions(grid_context.cellvalues),
            getnbasefunctions(grid_context.cellvalues)
        ),
    )
end

function assemble_weighted_mass_matrix!(N, u, grid_context, Ne)
    cellvalues = grid_context.cellvalues
    assembler = start_assemble(N)
    constraint_handler = grid_context.constraint_handler
    n_basefunctions = getnbasefunctions(cellvalues)
    n_quadpoints = getnquadpoints(cellvalues)
    for cell in grid_context.celliterator
        Ferrite.reinit!(cellvalues, cell)
        fill!(Ne, 0)
        dofs = celldofs(cell)
        for quadrature_point in 1:n_quadpoints
            dΩ = getdetJdV(cellvalues, quadrature_point)
            uₓ = 0.0
            # another option would be to use Ferrite.function_value here
            # but it turns out that there is a bit of unneeded overhead there
            # which is not ideal in this performance ciritcal point
            for i in 1:n_basefunctions
                @inbounds φᵢ = shape_value(cellvalues, quadrature_point, i)
                uₓ += u[dofs[i]] * φᵢ
            end
            for i in 1:n_basefunctions
                @inbounds φᵢ = shape_value(cellvalues, quadrature_point, i)
                for j in 1:n_basefunctions
                    @inbounds φⱼ = shape_value(cellvalues, quadrature_point, j)
                    Ne[i, j] += uₓ * (φᵢ ⋅ φⱼ) * dΩ
                end
            end
        end
        assemble!(assembler, dofs, Ne)
    end
    if !isnothing(constraint_handler)
        apply!(N, constraint_handler)
    end
    return N
end

function assemble_weighted_mass_matrix(u, grid_context)
    return assemble_weighted_mass_matrix!(
        allocate_matrix(grid_context.dof_handler),
        u, grid_context,
        zeros(
            getnbasefunctions(grid_context.cellvalues),
            getnbasefunctions(grid_context.cellvalues)
        ),
    )
end

function assemble_density_weighted_mass_matrix!(
    N::SparseMatrixCSC, u, grid_context, Ne::Matrix,
)
    cellvalues = grid_context.cellvalues
    assembler = start_assemble(N)
    constraint_handler = grid_context.constraint_handler
    n_basefunctions = getnbasefunctions(cellvalues)
    n_quadpoints = getnquadpoints(cellvalues)
    @assert size(Ne) == (n_basefunctions, n_basefunctions)
    for cell in grid_context.celliterator
        Ferrite.reinit!(cellvalues, cell)
        fill!(Ne, 0)
        dofs = celldofs(cell)
        for quadrature_point in 1:n_quadpoints
            dΩ = getdetJdV(cellvalues, quadrature_point)
            uₓ = 0.0im
            # another option would be to use Ferrite.function_value here
            # but it turns out that there is a bit of unneeded overhead there
            # which is not ideal in this performance ciritcal point
            for i in 1:n_basefunctions
                @inbounds φᵢ = shape_value(cellvalues, quadrature_point, i)
                @inbounds uₓ += u[dofs[i]] * φᵢ
            end
            uₓ² = abs2(uₓ)
            for i in 1:n_basefunctions
                @inbounds φᵢ = shape_value(cellvalues, quadrature_point, i)
                for j in 1:n_basefunctions
                    @inbounds φⱼ = shape_value(cellvalues, quadrature_point, j)
                    @inbounds Ne[i, j] += uₓ² * (φᵢ ⋅ φⱼ) * dΩ
                end
            end
        end
        assemble!(assembler, dofs, Ne)
    end
    if !isnothing(constraint_handler)
        apply!(N, constraint_handler)
    end
    return N
end

function assemble_density_weighted_mass_matrix(u, grid_context)
    return assemble_density_weighted_mass_matrix!(
        allocate_matrix(grid_context.dof_handler),
        u, grid_context,
        zeros(
            getnbasefunctions(grid_context.cellvalues),
            getnbasefunctions(grid_context.cellvalues)
        ),
    )
end

# This assumes REAL u and v!
function assemble_density_weighted_mass_matrix!(
    N::SparseMatrixCSC, u, v, grid_context, Ne::Matrix;
    fillzero = true,
)
    cellvalues = grid_context.cellvalues
    assembler = start_assemble(N; fillzero)
    constraint_handler = grid_context.constraint_handler
    n_basefunctions = getnbasefunctions(cellvalues)
    n_quadpoints = getnquadpoints(cellvalues)
    for cell in grid_context.celliterator
        Ferrite.reinit!(cellvalues, cell)
        fill!(Ne, 0)
        dofs = celldofs(cell)
        for quadrature_point in 1:n_quadpoints
            dΩ = getdetJdV(cellvalues, quadrature_point)
            uₓ = 0.0
            vₓ = 0.0
            # another option would be to use Ferrite.function_value here
            # but it turns out that there is a bit of unneeded overhead there
            # which is not ideal in this performance ciritcal point
            for i in 1:n_basefunctions
                @inbounds φᵢ = shape_value(cellvalues, quadrature_point, i)
                uₓ += u[dofs[i]] * φᵢ
                vₓ += v[dofs[i]] * φᵢ
            end
            uvₓ = uₓ * vₓ
            for i in 1:n_basefunctions
                @inbounds φᵢ = shape_value(cellvalues, quadrature_point, i)
                for j in 1:n_basefunctions
                    @inbounds φⱼ = shape_value(cellvalues, quadrature_point, j)
                    Ne[i, j] += uvₓ * (φᵢ ⋅ φⱼ) * dΩ
                end
            end
        end
        assemble!(assembler, dofs, Ne)
    end
    if !isnothing(constraint_handler)
        apply!(N, constraint_handler)
    end
    return N
end

function assemble_density_weighted_mass_matrix(u, v, grid_context)
    return assemble_density_weighted_mass_matrix!(
        allocate_matrix(grid_context.dof_handler),
        u, v, grid_context,
        zeros(
            getnbasefunctions(grid_context.cellvalues),
            getnbasefunctions(grid_context.cellvalues)
        ),
    )
end

function assemble_rotation_matrix!(
    L::SparseMatrixCSC, grid_context, Le::Matrix,
)
    cellvalues = grid_context.cellvalues
    assembler = start_assemble(L)
    constraint_handler = grid_context.constraint_handler
    n_basefunctions = getnbasefunctions(cellvalues)
    n_quadpoints = getnquadpoints(cellvalues)
    for cell in CellIterator(grid_context.dof_handler)
        Ferrite.reinit!(cellvalues, cell)
        fill!(Le, 0)
        for quadrature_point in 1:n_quadpoints
            dΩ = getdetJdV(cellvalues, quadrature_point)
            x = spatial_coordinate(cellvalues, quadrature_point, cell.coords)
            for i in 1:n_basefunctions
                φᵢ = shape_value(cellvalues, quadrature_point, i)
                for j in 1:n_basefunctions
                    ∇φⱼ = shape_gradient(cellvalues, quadrature_point, j)
                    Le[i, j] -= ((x[1] * ∇φⱼ[2] - x[2] * ∇φⱼ[1]) * φᵢ * dΩ)
                end
            end
        end
        assemble!(assembler, celldofs(cell), Le)
    end
    if !isnothing(constraint_handler)
        apply!(L, constraint_handler)
    end
    return L
end

function assemble_rotation_matrix(grid_context)
    return assemble_rotation_matrix!(
        allocate_matrix(
            grid_context.dof_handler,
        ),
        grid_context,
        zeros(
            COMPLEX_TYPE,
            getnbasefunctions(grid_context.cellvalues),
            getnbasefunctions(grid_context.cellvalues),
        ),
    )
end
