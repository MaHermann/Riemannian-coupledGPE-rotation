using Arpack, LinearAlgebra, LinearMaps, ProgressBars

using RiemannianCoupledGPERotation
##
function get_gpe_system(experiment)
    parameters = experiment.parameters
    grid_context = generate_grid_context(
            parameters["dimension"], parameters["n_elements"], parameters["interp_degree"],
            parameters["quad_degree"];
            left = parameters["domain"][1], right = parameters["domain"][2], type = parameters["type"],
            boundary_condition = parameters["boundary_condition"],
        )
        if length(parameters["potential_function"]) == 1
            potential = ConstantPotential(parameters["potential_function"][1], grid_context)
        else
            potential = ComponentwisePotential(parameters["potential_function"], grid_context)
        end
    return GPESystem(
        potential, parameters["interactions"], parameters["masses"], parameters["Ω"], grid_context,
    )
end
##
experiment_path = "path/to/experiment"
experiment = load_experiment(experiment_path);
##
output_path = "path/to/outputfolder"
filename = "condition_20"
##
ϕ = experiment.value
gpe_system = get_gpe_system(experiment)
RiemannianCoupledGPERotation.update!(gpe_system, ϕ; update_diagonal_Bs = true)
variable_storage = VariableStorage(ϕ)
temp_n = variable_storage.temp_n
M = gpe_system.grid_context.M

M1 = reinterpret(RiemannianCoupledGPERotation.REAL_TYPE, M .+ 0im)
M2 = reinterpret(RiemannianCoupledGPERotation.REAL_TYPE, 1im.*M)
M_large = [zero(M1) zero(M2)]
for i in 1:size(M, 1)
    M_large[:,2*i-1] .= @view M1[:,i]
    M_large[:,2*i]   .= @view M2[:,i]
end

##
ωs = collect(0.5:0.01:0.99)
##
i = 1
ϕᵢ = ϕ[i]
σ  = RiemannianCoupledGPERotation.get_σ!(variable_storage, ϕᵢ, gpe_system, i)
file_condition = open(
    output_path * filename * "_" * string(i) * ".txt", "w")
function g_ω_real(y, x, ω)
    x_complex = reinterpret(RiemannianCoupledGPERotation.COMPLEX_TYPE, x)
    y_complex = reinterpret(RiemannianCoupledGPERotation.COMPLEX_TYPE, y)
    RiemannianCoupledGPERotation.mul_hamiltonian!(y_complex, x_complex, gpe_system, i)
    RiemannianCoupledGPERotation.mul_B!(temp_n, x_complex, gpe_system, i, i)
    y_complex .+= temp_n
    RiemannianCoupledGPERotation.mul_M!(temp_n, x_complex, gpe_system)
axpy!(-ω*σ, temp_n, y_complex)
    return y
end
G_ω_real(ω) = LinearMap(
    (y, x) -> g_ω_real(y, x, ω), size(ϕ[i], 1)*2, issymmetric = false, ismutating = true)
for ω in ProgressBar(ωs)
    try
        λ_min, u = eigs(G_ω_real(ω), M_large; which=:SR, nev = 1, maxiter = 10_000)
        λ_max, _ = eigs(G_ω_real(ω), M_large; which=:LR, nev = 1, maxiter = 10_000)
        write(file_condition, string(ω) * " " * string(real(λ_max[1] / λ_min[1])) * "\n")
    catch
    end
end
close(file_condition)
