println("Loading environment...")
t₀ = time()
using DifferentialEquations, DifferentialEquations.EnsembleAnalysis, Plots
using Distributions, Random, LinearAlgebra, Statistics
using ProgressBars
println("...done (in $(time()-t₀))")

# generate samples
println("Generating samples...")
t₀ = time()
nₛ=100
x=collect(range(0.0,2*pi,length=nₛ)) # generating training dataset
y=cos.(x)
xᵥ=rand(nₛ).*2*pi #generating validation dataset
yᵥ=cos.(xᵥ)
println("...done (in $(time()-t₀))")

# model hyperparameters: EFP
println("Initializing network weights and hyperparameters...")
t₀ = time()
Δt = 0.1 # timestep length for discretization of the EFP
nₜ=100 # this is the number of EFP timesteps, T = Δt*nₜ
N₀=10 # number of neurons
αₗ=1.0 # learning rate
σᵢ=2.0/11.0 # Xavier-Glorot initialization
mv_norm_init = MvNormal(σᵢ*Matrix(1.0*I,N₀,N₀))
α = rand(mv_norm_init)
β = rand(mv_norm_init)
γ = zeros(N₀) # XG dictates that the biases should be initially null
println("...done (in $(time()-t₀))")

# neuron activation and gradients
# the gradients and the activation are fully vectorized in order to process the whole dataset at once
# (more efficient training and lets the compiler do some SIMD tricks)
println("Initializing neuron activations and gradients")
t₀ = time()
n(x,α,β,γ)=β.*tanh.(α.*x.+γ) # sigmoids are better suited for this task than ReLU

ϕ(x,α,β,γ)= mean(
    reduce(
        hcat,
        n.(x,Ref(α),Ref(β),Ref(γ)) # more vectorization magic
    ),
    dims=1)

# type-stability operator used to prevent broadcasting the parameters, only vectorize the arguments.
# this is needed because gradients are not scalars... 
# ...may be improved but it seems that it does not impact performance
# Julia's compiler is quite smart after all
∂n_∂α(x,α,β,γ) = β.*x'.*(sech.(reduce(hcat,Ref(α).*x.+Ref(γ))).^2)
∂n_∂β(x,α,β,γ) = tanh.(reduce(hcat,Ref(α).*x.+Ref(γ)))
∂n_∂γ(x,α,β,γ) = β.*(sech.(reduce(hcat,Ref(α).*x.+Ref(γ))).^2)
println("...done (in $(time()-t₀))")




# model hyperparameters: best response approximation
println("Initializing Langevin SDE...")
t₀ = time()
S=30.0 # time horizon for BRA simulation
λ=0.001 # shrinkage factor
σ = 0.005 # other regularization factor
# ↓ timestep length for discretization of the BRA simulation.
Δs=0.025
# ↑ Not strictly needed if we let the solver uses adaptive discretization with higher order methods
M = 10 # number of particles

# as far as the particle interaction is concerned, the parameters are all components of a single vector
function langevin(du,u,p,t)
    du[1:N₀] = mean((y'-ϕ(x,u[N₀+1:2*N₀],u[1:N₀],u[2*N₀+1:3*N₀])).*∂n_∂β(x,u[N₀+1:2*N₀],u[1:N₀],u[2*N₀+1:3*N₀]),dims=2) - λ.*u[1:N₀]
    du[N₀+1:2*N₀] = mean((y'-ϕ(x,u[N₀+1:2*N₀],u[1:N₀],u[2*N₀+1:3*N₀])).*∂n_∂α(x,u[N₀+1:2*N₀],u[1:N₀],u[2*N₀+1:3*N₀]),dims=2) - λ.*u[N₀+1:2*N₀]
    du[2*N₀+1:3*N₀] = mean((y'-ϕ(x,u[N₀+1:2*N₀],u[1:N₀],u[2*N₀+1:3*N₀])).*∂n_∂γ(x,u[N₀+1:2*N₀],u[1:N₀],u[2*N₀+1:3*N₀]),dims=2) - λ.*u[2*N₀+1:3*N₀]
end

function noise_langevin(du,u,p,t)
    du[1:3*N₀]=σ*ones(3*N₀)
end
println("...done (in $(time()-t₀))")

# Where the magic happens ↓
σᵣ=1/(N₀^2)
mv_norm_randomizer = MvNormal(σᵣ*Matrix(1.0*I,3*N₀,3*N₀))
u=vcat(β,α,γ)
MSEₜ=zeros(nₜ)
MSEᵥ=zeros(nₜ)
layout = @layout [a b; c]
iter = ProgressBar(1:nₜ)
animation = @animate for i in iter
#animation = for i in iter
    local prob_sde = SDEProblem(langevin, noise_langevin, (rand(mv_norm_randomizer).+1).*u, (0.0, S))
    local ensembleprob = EnsembleProblem(prob_sde)
    local sol = solve(ensembleprob, RKMil(), EnsembleThreads(), trajectories = M, dt=Δs)
    global u=αₗ*Δt.*timepoint_mean(sol,S)+(1-αₗ*Δt).*u
    global MSEₜ[i]=mean((y'-ϕ(x,u[N₀+1:2*N₀],u[1:N₀],u[2*N₀+1:3*N₀])).^2)
    global MSEᵥ[i]=mean((yᵥ'-ϕ(xᵥ,u[N₀+1:2*N₀],u[1:N₀],u[2*N₀+1:3*N₀])).^2)
    set_description(iter, "Loss: $(MSEₜ[i])")
    plot(
        plot(x,[y vec(ϕ(x,u[N₀+1:2*N₀],u[1:N₀],u[2*N₀+1:3*N₀]))],ylimits=(-1.25,1.25),xlimits=(0,2*pi),title="cos(x) vs. network output",label=["cos(zₖ)" "ϕ(zₖ|α,β,γ)"]),
        plot(1:i,MSEₜ[1:i],xlimits=(0,nₜ),ylimits=(0,1.25*MSEₜ[1]),title="Training loss",label="MSE"),
        plot(1:i,MSEᵥ[1:i],xlimits=(0,nₜ),ylimits=(0,1.25*MSEᵥ[1]),title="Validation error",label="MSE"),
        layout=layout,
        show=true
    )
end





