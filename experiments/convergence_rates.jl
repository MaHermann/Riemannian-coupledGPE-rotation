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
experiment_path = "experiments/results/weak_Lgr_095_2025-11-17_104654"
##
experiment = load_experiment(experiment_path)
gpe_system = get_gpe_system(experiment)
n_components = experiment.parameters["n_components"]
ϕ = experiment.value
M_small = gpe_system.grid_context.M .+ 0*im
n = size(experiment.value[1], 1)
p = experiment.parameters["n_components"]
variable_storage = VariableStorage(ϕ)
RiemannianCoupledGPERotation.update!(gpe_system, ϕ);
Λ = [RiemannianCoupledGPERotation.get_σ!(variable_storage, ϕ[i], gpe_system, i) for i in 1:p]
for i in 1:p
    for j in 1:p
        RiemannianCoupledGPERotation.assemble_B!(gpe_system, ϕ, i, j)
    end
end
##
M = hcat(
    (
        vcat(
            (
                RiemannianCoupledGPERotation.Ferrite.allocate_matrix(
                    gpe_system.grid_context.dof_handler,
                ) for _ in 1:(i-1)
            )...,
            M_small,
            (
                RiemannianCoupledGPERotation.Ferrite.allocate_matrix(
                    gpe_system.grid_context.dof_handler
                ) for _ in (i+1):p
            )...,
        )
        for i in 1:p
    )...
)
M = [real(M) -1*imag(M); imag(M) real(M)]

A = hcat(
    (
        vcat(
            (
                RiemannianCoupledGPERotation.Ferrite.allocate_matrix(
                    gpe_system.grid_context.dof_handler,
                ) for _ in 1:(i-1)
            )...,
            gpe_system.hamiltonian[i],
            (
                RiemannianCoupledGPERotation.Ferrite.allocate_matrix(
                    gpe_system.grid_context.dof_handler
                ) for _ in (i+1):p
            )...,
        )
        for i in 1:p
    )...
)
A = [real(A) -1*imag(A); imag(A) real(A)]

Bs = gpe_system.B_matrices
κ₁₁ = 2 * gpe_system.interactions[1,1]
κ₂₂ = 2 * gpe_system.interactions[2,2]
κ₁₂ = 2 * gpe_system.interactions[1,2]
κ₂₁ = 2 * gpe_system.interactions[2,1]

if p == 3
    κ₃₃ = 2 * gpe_system.interactions[3,3]
    κ₁₃ = 2 * gpe_system.interactions[1,3]
    κ₂₃ = 2 * gpe_system.interactions[2,3]
    κ₃₁ = 2 * gpe_system.interactions[3,1]
    κ₃₂ = 2 * gpe_system.interactions[3,2]
end

if p == 2
    B_realreal =  [
                    κ₁₁*Bs[1][1][1] κ₁₂*Bs[1][2][1];
                    κ₂₁*Bs[2][1][1] κ₂₂*Bs[2][2][1];
                ]
    B_realimag =  [
                    κ₁₁*Bs[1][1][2] κ₁₂*Bs[1][2][2];
                    κ₂₁*Bs[2][1][2] κ₂₂*Bs[2][2][2];
                ]
    B_imagreal =  [
                    κ₁₁*Bs[1][1][3] κ₁₂*Bs[1][2][3];
                    κ₂₁*Bs[2][1][3] κ₂₂*Bs[2][2][3];
                ]
    B_imagimag =  [
                    κ₁₁*Bs[1][1][4] κ₁₂*Bs[1][2][4];
                    κ₂₁*Bs[2][1][4] κ₂₂*Bs[2][2][4];
                ]

    B_diag_realreal =  [
                    κ₁₁*Bs[1][1][1]  zeros(n,n);
                    zeros(n,n)       κ₂₂*Bs[2][2][1];
                ]
    B_diag_realimag =  [
                    κ₁₁*Bs[1][1][2]  zeros(n,n);
                    zeros(n,n)       κ₂₂*Bs[2][2][2];
                ]
    B_diag_imagreal =  [
                    κ₁₁*Bs[1][1][3]  zeros(n,n);
                    zeros(n,n)       κ₂₂*Bs[2][2][3];
                ]
    B_diag_imagimag =  [
                    κ₁₁*Bs[1][1][4]  zeros(n,n);
                    zeros(n,n)       κ₂₂*Bs[2][2][4];
                ]
elseif p == 3
    B_realreal =  [
                    κ₁₁*Bs[1][1][1] κ₁₂*Bs[1][2][1] κ₁₃*Bs[1][3][1];
                    κ₂₁*Bs[2][1][1] κ₂₂*Bs[2][2][1] κ₂₃*Bs[2][3][1];
                    κ₃₁*Bs[3][1][1] κ₃₂*Bs[3][2][1] κ₃₃*Bs[3][3][1];
                ]
    B_realimag =  [
                    κ₁₁*Bs[1][1][2] κ₁₂*Bs[1][2][2] κ₁₃*Bs[1][3][2];
                    κ₂₁*Bs[2][1][2] κ₂₂*Bs[2][2][2] κ₂₃*Bs[2][3][2];
                    κ₃₁*Bs[3][1][2] κ₃₂*Bs[3][2][2] κ₃₃*Bs[3][3][2];
                ]
    B_imagreal =  [
                    κ₁₁*Bs[1][1][3] κ₁₂*Bs[1][2][3] κ₁₃*Bs[1][3][3];
                    κ₂₁*Bs[2][1][3] κ₂₂*Bs[2][2][3] κ₂₃*Bs[2][3][3];
                    κ₃₁*Bs[3][1][3] κ₃₂*Bs[3][2][3] κ₃₃*Bs[3][3][3];
                ]
    B_imagimag =  [
                    κ₁₁*Bs[1][1][4] κ₁₂*Bs[1][2][4] κ₁₃*Bs[1][3][4];
                    κ₂₁*Bs[2][1][4] κ₂₂*Bs[2][2][4] κ₂₃*Bs[2][3][4];
                    κ₃₁*Bs[3][1][4] κ₃₂*Bs[3][2][4] κ₃₃*Bs[3][3][4];
                ]

    B_diag_realreal =  [
                    κ₁₁*Bs[1][1][1]  zeros(n,n)      zeros(n,n);
                    zeros(n,n)       κ₂₂*Bs[2][2][1] zeros(n,n);
                    zeros(n,n)       zeros(n,n)      κ₃₃*Bs[3][3][1];
                ]
    B_diag_realimag =  [
                    κ₁₁*Bs[1][1][2]  zeros(n,n)      zeros(n,n);
                    zeros(n,n)       κ₂₂*Bs[2][2][2] zeros(n,n);
                    zeros(n,n)       zeros(n,n)      κ₃₃*Bs[3][3][2];
                ]
    B_diag_imagreal =  [
                    κ₁₁*Bs[1][1][3]  zeros(n,n)      zeros(n,n);
                    zeros(n,n)       κ₂₂*Bs[2][2][3] zeros(n,n);
                    zeros(n,n)       zeros(n,n)      κ₃₃*Bs[3][3][3];
                ]
    B_diag_imagimag =  [
                    κ₁₁*Bs[1][1][4]  zeros(n,n)      zeros(n,n);
                    zeros(n,n)       κ₂₂*Bs[2][2][4] zeros(n,n);
                    zeros(n,n)       zeros(n,n)      κ₃₃*Bs[3][3][4];
                ]
end

B = [B_realreal B_realimag; B_imagreal B_imagimag]
B_diagonal = [B_diag_realreal B_diag_realimag; B_diag_imagreal B_diag_imagimag]

function lambda(y, x)
    for i in 1:p
        Λᵢ = Λ[i]
        @views x_real = x[((i-1)*n+1):(i*n)]
        @views x_imag = x[(((i-1)*n+1)+(p*n)):((i*n)+(p*n))]
        @views y_real = y[((i-1)*n+1):(i*n)]
        @views y_imag = y[(((i-1)*n+1)+(p*n)):((i*n)+(p*n))]

        y_real .= Λᵢ .* M_small * x_real
        y_imag .= Λᵢ .* M_small * x_imag
    end
    return y
end
Lambda = LinearMap(lambda, 2*p*n, issymmetric = false, ismutating = true)

C = A + B - Lambda

G = A .+ B_diagonal
ω = experiment.parameters["ω"]
for i in 1:p
    Λᵢ = Λ[i]
    @views G_i_real = G[((i-1)*n+1):(i*n),((i-1)*n+1):(i*n)]
    @views G_i_imag = G[(((i-1)*n+1)+(p*n)):((i*n)+(p*n)),(((i-1)*n+1)+(p*n)):((i*n)+(p*n))]

    G_i_real .-= ω * Λᵢ .* M_small
    G_i_imag .-= ω * Λᵢ .* M_small
end

function projection_real(y, x)
    for i in 1:p
        Nᵢ = gpe_system.masses[i]
        @views ϕ_real = real.(ϕ[i])
        @views ϕ_imag = imag.(ϕ[i])
        @views x_real = x[((i-1)*n+1):(i*n)]
        @views x_imag = x[(((i-1)*n+1)+(p*n)):((i*n)+(p*n))]
        @views y_real = y[((i-1)*n+1):(i*n)]
        @views y_imag = y[(((i-1)*n+1)+(p*n)):((i*n)+(p*n))]

        ϕᵣxᵣ = (ϕ_real ⋅ (M_small * x_real)) ./ Nᵢ
        ϕᵢxᵢ = (ϕ_imag ⋅ (M_small * x_imag)) ./ Nᵢ

        px_real = (ϕᵣxᵣ + ϕᵢxᵢ) .* ϕ_real
        px_imag = (ϕᵣxᵣ + ϕᵢxᵢ) .* ϕ_imag

        y_real .= x_real .- px_real
        y_imag .= x_imag .- px_imag
    end
    return y
end
function projection_real_T(y, x)
    for i in 1:p
        Nᵢ = gpe_system.masses[i]
        @views ϕ_real = real.(ϕ[i])
        @views ϕ_imag = imag.(ϕ[i])
        @views x_real = x[((i-1)*n+1):(i*n)]
        @views x_imag = x[(((i-1)*n+1)+(p*n)):((i*n)+(p*n))]
        @views y_real = y[((i-1)*n+1):(i*n)]
        @views y_imag = y[(((i-1)*n+1)+(p*n)):((i*n)+(p*n))]

        ϕᵣxᵣ = (ϕ_real ⋅ x_real) ./ Nᵢ
        ϕᵢxᵢ = (ϕ_imag ⋅ x_imag) ./ Nᵢ

        px_real = (ϕᵣxᵣ + ϕᵢxᵢ) .* (M_small * ϕ_real)
        px_imag = (ϕᵣxᵣ + ϕᵢxᵢ) .* (M_small * ϕ_imag)

        y_real .= x_real .- px_real
        y_imag .= x_imag .- px_imag
    end
    return y
end
Projection_real   = LinearMap(projection_real,   2*p*n, issymmetric = false, ismutating = true)
Projection_real_T = LinearMap(projection_real_T, 2*p*n, issymmetric = false, ismutating = true)

function projection_imag(y, x)
    for i in 1:p
        Nᵢ = gpe_system.masses[i]
        @views ϕ_real = real.(ϕ[i])
        @views ϕ_imag = imag.(ϕ[i])
        @views x_real = x[((i-1)*n+1):(i*n)]
        @views x_imag = x[(((i-1)*n+1)+(p*n)):((i*n)+(p*n))]
        @views y_real = y[((i-1)*n+1):(i*n)]
        @views y_imag = y[(((i-1)*n+1)+(p*n)):((i*n)+(p*n))]

        ϕᵣxᵢ = (ϕ_real ⋅ (M_small * x_imag)) ./ Nᵢ
        ϕᵢxᵣ = (ϕ_imag ⋅ (M_small * x_real)) ./ Nᵢ

        px_real = (ϕᵢxᵣ - ϕᵣxᵢ) .* ϕ_imag
        px_imag = (ϕᵢxᵣ - ϕᵣxᵢ) .* -1 .* ϕ_real

        y_real .= x_real .- px_real
        y_imag .= x_imag .- px_imag
    end
    return y
end
function projection_imag_T(y, x)
    for i in 1:p
        Nᵢ = gpe_system.masses[i]
        @views ϕ_real = real.(ϕ[i])
        @views ϕ_imag = imag.(ϕ[i])
        @views x_real = x[((i-1)*n+1):(i*n)]
        @views x_imag = x[(((i-1)*n+1)+(p*n)):((i*n)+(p*n))]
        @views y_real = y[((i-1)*n+1):(i*n)]
        @views y_imag = y[(((i-1)*n+1)+(p*n)):((i*n)+(p*n))]

        ϕᵣxᵢ = (ϕ_real ⋅ x_imag) ./ Nᵢ
        ϕᵢxᵣ = (ϕ_imag ⋅ x_real) ./ Nᵢ

        px_real = (ϕᵢxᵣ - ϕᵣxᵢ) .* (M_small * ϕ_imag)
        px_imag = (ϕᵢxᵣ - ϕᵣxᵢ) .* -1 .* (M_small * ϕ_real)

        y_real .= x_real .- px_real
        y_imag .= x_imag .- px_imag
    end
    return y
end
Projection_imag   = LinearMap(projection_imag  , 2*p*n, issymmetric = false, ismutating = true)
Projection_imag_T = LinearMap(projection_imag_T, 2*p*n, issymmetric = false, ismutating = true)

Giϕ = zeros(2*p*n)
G_invMϕ = zeros(2*p*n)
iϕGiϕ = zeros(p)
ϕMG_invMϕ = zeros(p)

for i in 1:p
    Mϕ_long  = [real.(M_small * ϕ[i]); imag.(M_small * ϕ[i])]
    iϕ_long = [imag.(ϕ[i]); -1 * real.(ϕ[i])]
    ind_real = ((i-1)*n+1):(i*n)
    ind_imag = (((i-1)*n+1)+(p*n)):((i*n)+(p*n))
    @views Gi_realreal = G[ind_real,ind_real]
    @views Gi_realimag = G[ind_real,ind_imag]
    @views Gi_imagreal = G[ind_imag,ind_real]
    @views Gi_imagimag = G[ind_imag,ind_imag]
    Gi = [Gi_realreal Gi_realimag; Gi_imagreal Gi_imagimag]
    Giϕ[[ind_real; ind_imag]] = Gi * iϕ_long
    G_invMϕ[[ind_real; ind_imag]] = Gi \ Mϕ_long
    iϕGiϕ[i] = iϕ_long ⋅ Giϕ[[ind_real; ind_imag]]
    ϕMG_invMϕ[i] = Mϕ_long ⋅ G_invMϕ[[ind_real; ind_imag]] 
end


function projection_imag_G(y, x)
    for i in 1:p
        ind_real = ((i-1)*n+1):(i*n)
        ind_imag = (((i-1)*n+1)+(p*n)):((i*n)+(p*n))

        @views ϕ_real = real.(ϕ[i])
        @views ϕ_imag = imag.(ϕ[i])
        @views x_real = x[ind_real]
        @views x_imag = x[ind_imag]
        @views y_real = y[ind_real]
        @views y_imag = y[ind_imag]
        @views Giϕ_real = Giϕ[ind_real]
        @views Giϕ_imag = Giϕ[ind_imag]

        ϕᵣGxᵣ = (Giϕ_real ⋅ x_real) ./ iϕGiϕ[i]
        ϕᵢGxᵢ = (Giϕ_imag ⋅ x_imag) ./ iϕGiϕ[i]

        px_real = (ϕᵣGxᵣ + ϕᵢGxᵢ) .* ϕ_imag
        px_imag = (ϕᵣGxᵣ + ϕᵢGxᵢ) .* -1 .* ϕ_real

        y_real .= x_real .- px_real
        y_imag .= x_imag .- px_imag
    end
    return y
end
function projection_imag_G_T(y, x)
    for i in 1:p
        ind_real = ((i-1)*n+1):(i*n)
        ind_imag = (((i-1)*n+1)+(p*n)):((i*n)+(p*n))

        @views ϕ_real = real.(ϕ[i])
        @views ϕ_imag = imag.(ϕ[i])
        @views x_real = x[ind_real]
        @views x_imag = x[ind_imag]
        @views y_real = y[ind_real]
        @views y_imag = y[ind_imag]
        @views Giϕ_real = Giϕ[ind_real]
        @views Giϕ_imag = Giϕ[ind_imag]

        ϕᵣxᵢ = (ϕ_real ⋅ x_imag) ./ iϕGiϕ[i]
        ϕᵢxᵣ = (ϕ_imag ⋅ x_real) ./ iϕGiϕ[i]

        px_real = (ϕᵢxᵣ - ϕᵣxᵢ) .* Giϕ_real
        px_imag = (ϕᵢxᵣ - ϕᵣxᵢ) .* Giϕ_imag

        y_real .= x_real .- px_real
        y_imag .= x_imag .- px_imag
    end
    return y
end
Projection_imag_G   = LinearMap(projection_imag_G  , 2*p*n, issymmetric = false, ismutating = true)
Projection_imag_G_T = LinearMap(projection_imag_G_T, 2*p*n, issymmetric = false, ismutating = true)

function projection_real_G(y, x)
    for i in 1:p
        ind_real = ((i-1)*n+1):(i*n)
        ind_imag = (((i-1)*n+1)+(p*n)):((i*n)+(p*n))

        @views ϕ_real = real.(ϕ[i])
        @views ϕ_imag = imag.(ϕ[i])
        @views x_real = x[ind_real]
        @views x_imag = x[ind_imag]
        @views y_real = y[ind_real]
        @views y_imag = y[ind_imag]
        @views G_invMϕ_real = G_invMϕ[ind_real]
        @views G_invMϕ_imag = G_invMϕ[ind_imag]

        ϕᵣxᵣ = (ϕ_real ⋅ (M_small * x_real)) ./ ϕMG_invMϕ[i]
        ϕᵢxᵢ = (ϕ_imag ⋅ (M_small * x_imag)) ./ ϕMG_invMϕ[i]

        px_real = (ϕᵣxᵣ + ϕᵢxᵢ) .* G_invMϕ_real
        px_imag = (ϕᵣxᵣ + ϕᵢxᵢ) .* G_invMϕ_imag

        y_real .= x_real .- px_real
        y_imag .= x_imag .- px_imag
    end
    return y
end
function projection_real_G_T(y, x)
    for i in 1:p
        ind_real = ((i-1)*n+1):(i*n)
        ind_imag = (((i-1)*n+1)+(p*n)):((i*n)+(p*n))

        @views ϕ_real = real.(ϕ[i])
        @views ϕ_imag = imag.(ϕ[i])
        @views x_real = x[ind_real]
        @views x_imag = x[ind_imag]
        @views y_real = y[ind_real]
        @views y_imag = y[ind_imag]
        @views G_invMϕ_real = G_invMϕ[ind_real]
        @views G_invMϕ_imag = G_invMϕ[ind_imag]

        ϕᵣxᵣ = (G_invMϕ_real ⋅ x_real) ./ ϕMG_invMϕ[i]
        ϕᵢxᵢ = (G_invMϕ_imag ⋅ x_imag) ./ ϕMG_invMϕ[i]

        px_real = (ϕᵣxᵣ + ϕᵢxᵢ) .* (M_small * ϕ_real)
        px_imag = (ϕᵣxᵣ + ϕᵢxᵢ) .* (M_small * ϕ_imag)

        y_real .= x_real .- px_real
        y_imag .= x_imag .- px_imag
    end
    return y
end
Projection_real_G   = LinearMap(projection_real_G  , 2*p*n, issymmetric = false, ismutating = true)
Projection_real_G_T = LinearMap(projection_real_G_T, 2*p*n, issymmetric = false, ismutating = true)

C_projected = Projection_real_T * Projection_imag_T *  C *  Projection_imag * Projection_real

C_G_projected = Projection_real_G_T * Projection_imag_G_T *  C *  Projection_imag_G * Projection_real_G

λ_A, _ = eigs(C_projected, A; nev=2*p+1, which=:SM, maxiter = 5000);
println(λ_A)
println(1 - real(λ_A[2*p+1]))

λ_G, _ = eigs(C_G_projected, G; nev=2*p+1, which=:SM, maxiter = 5000)
println(λ_G)
println(1 - real(λ_G[2*p+1]))
