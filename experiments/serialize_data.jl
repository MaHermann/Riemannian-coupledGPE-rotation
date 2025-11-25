using Plots, Statistics

using RiemannianCoupledGPERotation

##
function get_gpe_system(experiment)
    parameters = experiment.parameters
    grid_context = generate_grid_context(
            parameters["dimension"], parameters["n_elements"], parameters["interp_degree"],
            parameters["quad_degree"];
            left = parameters["domain"][1], right = parameters["domain"][2], type = parameters["type"],
            boundary_conditions = parameters["boundary_conditions"],
        )
        if length(parameters["potential_function"]) == 1
            potential = ConstantPotential(parameters["potential_function"], grid_context)
        else
            potential = ComponentwisePotential(parameters["potential_function"], grid_context)
        end
    return GPESystem(
        potential, parameters["interactions"], parameters["masses"], parameters["Ω"], grid_context,
    )
end
##
experiment_path = "path/to/experiment"
output_path = "path/to/outputfolder"
parameter = "residuals"
filename = parameter * "_3_component_LgrRGD_095_Adaptive"
experiment = load_experiment(experiment_path);
smoothing = false
window_size = 100
moving_average(vs,n) = [mean(@view vs[max(1,i-n):min(length(vs),i+n)]) for i in 1:length(vs)]
every_n = 100
##
file = open(output_path * filename * ".txt", "w")
if parameter == "residuals"
    values = [
        sqrt.(sum(experiment.log["residual_values"][i,:].^2))   
        for i in 1:size(experiment.log["residual_values"], 1)
        if experiment.log["residual_values"][i,1] != 1.0
    ]
    smoothing && (values = moving_average(values, window_size))
    values = enumerate(values)
elseif parameter == "contraction_rates"
        distances = experiment.log["h_norm_distances"]
        contraction_rates = sqrt.(sum(distances[2:end,:].^2, dims=2))./ 
                                sqrt.(sum(distances[1:end-1,:].^2,  dims=2))
    values = [
        (i+1, contraction_rates[i])
        for i in 1:(size(experiment.log["residual_values"], 1) - 1)
        if experiment.log["residual_values"][i,1] != 1.0
    ]
else
    values = enumerate(experiment.log[parameter])
end
for (i, value) in values
    if !isnothing(every_n)
        if i % every_n != 1
            continue
        end
    end
    write(file, string(i) * " " * string(value) * "\n")
end
close(file)