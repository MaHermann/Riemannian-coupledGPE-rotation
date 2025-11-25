using Arpack, LinearAlgebra, LinearMaps

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
experiment_path     = "path/to/experiment"
##
experiment = load_experiment(experiment_path)
gpe_system = get_gpe_system(experiment)
n_components = experiment.parameters["n_components"]
ϕ = experiment.value
M = gpe_system.grid_context.M .+ 0*im
n = size(experiment.value[1], 1)
RiemannianCoupledGPERotation.update!(gpe_system, ϕ);
##
for i in 1:n_components
    a(y, x) = RiemannianCoupledGPERotation.mul_hamiltonian!(y, x, gpe_system, i)
    A = LinearMap{eltype(ϕ[i])}(a, n, issymmetric = false, ismutating = true)
    λ_A, ev_A = eigs(A, M, nev = 30, which = :SM, maxiter = 5000)
    @assert all(imag.(λ_A) .< 1e-13)
    println(real.(λ_A))

    RiemannianCoupledGPERotation.assemble_B!(gpe_system, ϕ, i, i)
    b(y, x) = RiemannianCoupledGPERotation.mul_B!(y, x, gpe_system, i, i)
    B = LinearMap{ComplexF64}(b, size(ϕ[i], 1), issymmetric = false, ismutating = true)

    C = A + B
    # projectors
    p(y)  = y - (ϕ[i]/ gpe_system.masses[i]) * real(ϕ[i] ⋅ (M * y))
    P = LinearMap{ComplexF64}(p, size(ϕ[i], 1), issymmetric = false, ismutating = false)
    p_T(y) = y - (M * ϕ[i]/ gpe_system.masses[i]) * real(ϕ[i] ⋅ y)
    P_T = LinearMap{ComplexF64}(p_T, size(ϕ[i], 1), issymmetric = false, ismutating = false)
    C_projected = P_T * C * P

    function c_R_linear(x)
        n = size(x, 1)
        y = C_projected(x[1:Int(n/2)] + 1im*x[Int(n/2)+1:end])
        return [real(y); imag(y)]
    end
    C_final = LinearMap(c_R_linear, size(ϕ[i], 1)*2, issymmetric = false, ismutating = false)

    M_final = [M zero(M); zero(M) M]
    println("")
    λ_C, u_C = eigs(C_final, M_final; nev = 6, which = :SM, maxiter = 5000)
    @assert all(imag.(λ_C) .< 1e-13)
    println(real.(λ_C))
    println("")
end

Aϕ = deepcopy(ϕ)
rayleigh = zeros(RiemannianCoupledGPERotation.COMPLEX_TYPE, size(ϕ))
for i in 1:n_components
    RiemannianCoupledGPERotation.mul_hamiltonian!(Aϕ[i], ϕ[i], gpe_system, i)
    rayleigh[i] = (ϕ[i] ⋅ Aϕ[i]) / gpe_system.masses[i]
end

println("rayleigh coefficients:")
for i in 1:n_components
    println(rayleigh[i])
end