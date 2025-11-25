struct PFrame{p, T}
    entries::SVector{p, Vector{T}}
end

Base.size(_::PFrame{p}) where {p} = p

Base.getindex(u::PFrame{p}, i::Int) where {p} = u.entries[i]

density(u::PFrame{p}) where {p} =
    PFrame(SVector{p}(abs2.(u[i]) for i in 1:p))

Base.real(u::PFrame{p}) where {p} =
    PFrame(SVector{p}(real(u[i]) for i in 1:p))

Base.imag(u::PFrame{p}) where {p} =
    PFrame(SVector{p}(imag(u[i]) for i in 1:p))

Base.:+(u::PFrame{p}, v::PFrame{p}) where {p} =
    PFrame(SVector{p}(u[i] .+ v[i] for i in 1:p))

Base.:-(u::PFrame{p}, v::PFrame{p}) where {p} =
    PFrame(SVector{p}(u[i] .- v[i] for i in 1:p))

Base.:*(a::Number, u::PFrame{p}) where {p} =
    PFrame(SVector{p}(a .* u[i] for i in 1:p))

function LinearAlgebra.lmul!(a::Number, u::PFrame{p}) where {p}
    for i in 1:p
        u[i] .= a .* u[i]
    end
    return u
end

function plot_PFrame(
    u::PFrame{p}, grid_context::GridContext{T, 2};
    plot_sum = true, n_points = 100, base_size = 200, colormap = :viridis, kwargs...,
) where {p, T}
    points_x = range(minimum([x[1] for x in grid_points(grid_context)]),
        maximum([x[1] for x in grid_points(grid_context)]), n_points)
    points_y = range(minimum([x[2] for x in grid_points(grid_context)]),
        maximum([x[2] for x in grid_points(grid_context)]), n_points)
    point_handler = PointEvalHandler(grid_context.grid,
        [Vec((x, y)) for (x, y) in
            vec(collect(Base.Iterators.product(points_x, points_y)))])
    subplots = []
    if plot_sum
        joint_plot = contourf(
            points_x, points_y,
            evaluate_at_points(point_handler, grid_context.dof_handler, sum(u[i] for i in 1:p));
            colorbar = false, levels = 100, linewidth = 0, axis = ([], false), colormap = colormap,
        )
        for _ in 1:p-1
            push!(subplots, plot(legend=false, grid=false, foreground_color_subplot=:white))
        end
    end
    append!(subplots, [
        contourf(
            points_x, points_y,
            evaluate_at_points(point_handler, grid_context.dof_handler, u[i]);
            colorbar = false, levels = 100, linewidth = 0,
            axis = ([], false), colormap = colormap,
        ) for i in 1:p])
    if plot_sum
        l = @layout [grid(p,1) grid(p, 1)]
        plot(joint_plot, subplots...;
            size = (base_size * 2, base_size * p), layout = l, kwargs...,
        )
    else
        l = @layout grid(1, p)
        plot(subplots...;
            size = (base_size * p, base_size), layout = l, kwargs...,
        )
    end
end
