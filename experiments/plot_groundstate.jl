using Ferrite, GLMakie, IterTools

using RiemannianCoupledGPERotation
##
n_points_evaluated  = 300
px_per_unit         = 3
size_3D             = (400, 400)
colormaps           = [:Blues, :Reds, :Greens]
linewidth_3D        = 1.5
azimuth             = 1.5π
elevation           = 0.5π
alpha_inactive      = 0.2
contour_offset      = 0.6
contour_level       = 0.605
linewidth_contour   = 2
plotting_theme      = merge(theme_minimal(), theme_latexfonts())
experiment_path     = "path/to/experiment"
output_path         = "path/to/outputfolder"
##
experiment = load_experiment(experiment_path)
ϕ = experiment.value
grid_context = generate_grid_context(
    experiment.parameters["dimension"], experiment.parameters["n_elements"],
    experiment.parameters["interp_degree"], experiment.parameters["quad_degree"];
    left = experiment.parameters["domain"][1],
    right = experiment.parameters["domain"][2],
);
n_components = experiment.parameters["n_components"]
##
points_x = range(minimum([x[1] for x in grid_points(grid_context)]),
           maximum([x[1] for x in grid_points(grid_context)]), n_points_evaluated)
points_y = range(minimum([x[2] for x in grid_points(grid_context)]),
           maximum([x[2] for x in grid_points(grid_context)]), n_points_evaluated)
point_handler = Ferrite.PointEvalHandler(
    grid_context.grid,
    [
        Ferrite.Vec((x, y))
        for (x, y) in
        Ferrite.vec(collect(Base.Iterators.product(points_x, points_y)))
    ]
)
points_x_ = [vec(collect(product(points_x, points_y)))[i][1] for i in 1:(n_points_evaluated^2)]
points_y_ = [vec(collect(product(points_x, points_y)))[i][2] for i in 1:(n_points_evaluated^2)]
points_x, points_y = points_x_, points_y_;
##
with_theme(plotting_theme) do
    for i in 1:n_components
        fig = Figure(size = size_3D)

        axis = Axis3(
            fig[1,1], aspect = (1,1,0.1), perspectiveness = 0.0,
            azimuth = azimuth, elevation = elevation,
            yreversed = false, xreversed = false,
            xgridvisible = false, ygridvisible = false, zgridvisible = false,
            xticksvisible = false, yticksvisible = false, zticksvisible = false,
            xspinesvisible = false, yspinesvisible = false, zspinesvisible = false,
            xlabelsize = 0, ylabelsize = 0, zlabelsize = 0,
            xticklabelsize = 0, yticklabelsize = 0, zticklabelsize = 0,
        )
        rowsize!(fig.layout, 1, Aspect(1, 1.0))
        resize_to_layout!(fig)

        Makie.surface!(
            axis, points_x, points_y,
            100 * evaluate_at_points(point_handler, grid_context.dof_handler, abs2.(ϕ[i])),
            colormap = colormaps[i],
            transparency = true,
            shading = NoShading,
            alpha = 0.9,
        )
        for j in 1:n_components
            Makie.contour3d!(
                axis, points_x, points_y,
                evaluate_at_points(
                    point_handler,
                    grid_context.dof_handler,
                    abs2.(ϕ[j])) .+ contour_offset,
                levels = [contour_level],
                colormap = cgrad([cgrad(colormaps[j])[end]]),
                linewidth = linewidth_contour,
            )
        Makie.save(output_path*"_"*string(i)*".png", fig, px_per_unit = px_per_unit)
        end
    end
end
