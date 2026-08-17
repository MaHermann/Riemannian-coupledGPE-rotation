abstract type Potential end

struct ConstantPotential{T} <: Potential
    potential_function
    weighted_mass_matrix::T
end

function ConstantPotential(potential_function, grid_context::GridContext)
    weighted_mass_matrix = assemble_weighted_mass_matrix(potential_function, grid_context)
    return ConstantPotential(potential_function, weighted_mass_matrix)
end

function matrix_representation(potential::ConstantPotential, _::Number)
    return potential.weighted_mass_matrix
end

struct ComponentwisePotential{T} <: Potential
    potential_functions
    weighted_mass_matrices::Vector{T}
end

function ComponentwisePotential(potential_functions, grid_context::GridContext)
    weighted_mass_matrices = [
        assemble_weighted_mass_matrix(potential_functions[i], grid_context)
        for i in eachindex(potential_functions)
    ]
    return ComponentwisePotential(potential_functions, weighted_mass_matrices)
end

function matrix_representation(potential::ComponentwisePotential, i::Number)
    return potential.weighted_mass_matrices[i]
end

function plot_potential_2D(
    x_interval, y_interval, potential::ConstantPotential;
    scaling = 1, kwargs...
)
    V = x -> potential.potential_function(x)
    # flip x and y so the horizontal is x as is common in plotting
    heatmap(
        x_interval, y_interval, [scaling * V([x , y]) for y in y_interval, x in x_interval];
        kwargs...,
    )
end

function plot_potential_2D!(
    x_interval, y_interval, potential::ConstantPotential;
    scaling = 1, kwargs...
)
    V = x -> potential.potential_function(x)
    # flip x and y so the horizontal is x as is common in plotting
    heatmap!(
        x_interval, y_interval, [scaling * V([x , y]) for y in y_interval, x in x_interval];
        kwargs...,
    )
end
