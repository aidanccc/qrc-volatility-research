# =============================================================================
# Time_series.jl — Quantum Reservoir Computing Utilities
# =============================================================================
#
# This file is the core Julia library for the quantum reservoir computing
# (QRC) pipeline described in:
#   "Quantum Reservoir Computing for Realized Volatility Forecasting"
#   (arXiv:2505.13933, Physical Review Research 8, 023028, 2026)
#
# PURPOSE:
#   Provides all functions needed by Time_serial_Finance_regression.ipynb
#   to run quantum reservoir simulations, including:
#     - Quantum circuit construction and density matrix operations
#     - Hamiltonian construction (transverse-field Ising model)
#     - Quantum reservoir evolution and measurement
#     - Data normalization/denormalization
#     - Forecasting error metrics (MSE, MAE, MAPE, RMSE, QLIKE)
#     - Ridge regression readout utilities
#
# NOTE ON DEPENDENCIES:
#   This file requires two EXTERNAL packages NOT included in the repository:
#     - QuantumCircuits_demo (at ../package/QuantumCircuits_demo/src)
#     - VQC_demo_cuda (at ../package/VQC_demo_cuda/src)
#   These are custom, unpublished quantum simulation packages. Without them,
#   this code CANNOT be executed. The pre-computed output (predict_result.csv)
#   is shipped with the repo as a fixed artifact.
#
# ALSO REQUIRES:
#   - NVIDIA GPU with CUDA support (for CuArray, CuDensityMatrixBatch)
#   - Julia 1.11+ with packages: Flux, Random, Statistics, StatsBase,
#     LinearAlgebra, CUDA
#
# =============================================================================

# Add custom quantum packages to Julia's load path
# These packages provide: QCircuit, RyGate, DensityMatrix, CuDensityMatrixBatch,
# QubitsTerm, QubitsOperator, expectation, partial_tr, etc.
push!(LOAD_PATH,"../package/QuantumCircuits_demo/src","../package/VQC_demo_cuda/src")
using VQC, VQC.Utilities
using QuantumCircuits, QuantumCircuits.Gates
using Flux:train!
using Flux
using Random
using Statistics
using StatsBase
using LinearAlgebra
using CUDA
import LinearAlgebra: tr


# =============================================================================
# MyModel — Container for trained quantum reservoir model parameters
# =============================================================================
# Stores the ridge regression weights (W) for each rolling window,
# along with the feature list and output dimensions.
#
# Fields:
#   L       — Number of out-of-sample predictions (245)
#   OutLen  — Number of readout features (= number of observables * virtual nodes)
#   W       — Weight matrix (L x OutLen): one set of weights per rolling window
#   Features — List of input feature column names
# =============================================================================
struct MyModel
    L::Int
    OutLen::Int
    W::Matrix{Float64}
    Features::Vector{String}
end
# Convenience constructor: initializes weights to zeros
MyModel(L::Int,OutLen::Int,Features::Vector{String},) = MyModel(L,OutLen,zeros(L,OutLen),Features)


# =============================================================================
# Trace operation for GPU-accelerated density matrix batches
# =============================================================================
# Computes the trace of a CuDensityMatrixBatch (density matrix on GPU).
# The trace should equal 1.0 for a properly normalized quantum state.
# This extends Julia's LinearAlgebra.tr for the custom quantum type.
function tr(m::CuDensityMatrixBatch)
    mat = storage(m)
    remat = reshape(mat,2^m.nqubits,2^m.nqubits,m.nitems)
    x=zeros(eltype(m),m.nitems)
    for i in 1:m.nitems
        x[i]=CUDA.tr(remat[:,:,i])
    end
    return x[1]
end


# =============================================================================
# Normalization Constants
# =============================================================================
# These constants define the mapping between the normalized RV scale
# ([-1, 0] as stored in Data.CSV) and the original log-RV scale.
#
# Original log-RV range: [Min_RV, Max_RV] = [-4.772, -1.254]
#
# Normalization formula (Data.CSV → Julia):
#   normalized = (log_RV - Min_RV) / (Max_RV - Min_RV) - 1
#   This maps Min_RV → -1 and Max_RV → 0
#
# Denormalization formula (Julia → original scale):
#   log_RV = (normalized + 1) * dif + Min_RV
#
# coe = dif^2 is used for MSE in the original scale
# dif is used for MAE in the original scale
# =============================================================================
Max_RV=-1.2543188032019446
Min_RV=-4.7722718186046515
coe = (Max_RV-Min_RV)^2    # Scale factor for MSE: ~12.376
dif = (Max_RV-Min_RV)      # Scale factor for MAE: ~3.518


# =============================================================================
# Denormalization functions
# =============================================================================
# Convert predictions from normalized [-1, 0] scale back to original log-RV.
# Two variants: one with explicit min/max, one that computes from data.

function denormalization(x1,x2,y,a,b)
    # Denormalize y from [a,b] to [x1,x2] (x1=max, x2=min)
    xmax=x1
    xmin=x2
    f(z)=(z-a)*(xmax-xmin)/(b-a)+xmin
    return map(f,y)
end


# =============================================================================
# Tensor Product for Density Matrices
# =============================================================================
# The ⊗ operator computes the tensor (Kronecker) product of two quantum
# states. This is used to combine input qubits with hidden/memory qubits
# at each step of the quantum reservoir evolution.
#
# Example: If input has 7 qubits and hidden has 3 qubits,
#   ρ = ρ_hidden ⊗ ρ_input creates a 10-qubit joint state.

function ⊗(A::DensityMatrix,B::DensityMatrix)
    return DensityMatrix(kron(storage(A),storage(B)),nqubits(A)+nqubits(B))
end

function ⊗(A::CuDensityMatrixBatch,B::CuDensityMatrixBatch)
    return CuDensityMatrixBatch(kron(storage(A),storage(B)),A.nqubits+B.nqubits,1)
end


# =============================================================================
# Matrix * DensityMatrix operators
# =============================================================================
# These allow multiplying unitary evolution operators (matrices) with
# density matrices for time evolution: ρ' = U * ρ * U†
#
# The quantum state evolves under the Hamiltonian H for time τ as:
#   ρ(τ) = e^{-iτH} * ρ(0) * e^{+iτH}

import Base: *
function *(A::Union{Matrix,CuArray}, B::DensityMatrix)
    return DensityMatrix(Array(A*CuArray(storage(B))))
end
function *(B::DensityMatrix,A::Union{CuArray,Adjoint})
    return DensityMatrix(Array(CuArray(storage(B))*A))
end

function *(A::Union{Matrix,CuArray}, B::CuDensityMatrixBatch)
    return CuDensityMatrixBatch(A*storage(B),B.nqubits,B.nitems)
end
function *(B::CuDensityMatrixBatch,A::Union{CuArray,Adjoint})
    return CuDensityMatrixBatch(storage(B)*A,B.nqubits,B.nitems)
end


# =============================================================================
# Qreservoir — Construct the Reservoir Hamiltonian
# =============================================================================
# Builds the transverse-field Ising model Hamiltonian (paper Eq. 1):
#
#   H = Σ_{i<j} J_{ij} * X_i * X_j  +  v * Σ_i Z_i
#
# Where:
#   X_i, Z_i = Pauli operators on qubit i
#   J_{ij}   = coupling strengths (from coeff_10.jld2, random in [0,1])
#   v = 1    = magnetic field strength (energy unit)
#
# This is a fully-connected graph: every pair of qubits interacts.
# The Hamiltonian is FIXED (not trained) — this is the reservoir property.
#
# Parameters:
#   nqubit — Number of qubits (10 in the paper)
#   ps     — Coupling matrix J_{ij} (nqubit × nqubit, symmetric)
# =============================================================================
function Qreservoir(nqubit,ps)
    H=QubitsOperator()
    # XX coupling terms between all pairs of qubits
    for i in 1:nqubit
        for j in i+1:nqubit
            H+=QubitsTerm(i=>"X",j=>"X",coeff=ps[i,j])
        end
    end
    # Transverse field (Z) on each qubit with strength v=1
    for i in 1:nqubit
        H+=QubitsTerm(i=>"Z",coeff=1)
    end
    return H
end


# =============================================================================
# Normalization / Denormalization utilities
# =============================================================================

function normalization(x,a,b)
    # Normalize vector x from its [min,max] range to [a,b]
    xmax=maximum(x)
    xmin=minimum(x)
    f(z)=(b-a)*(z-xmin)/(xmax-xmin)+a
    return map(f,x)
end

function denormalization(x,y,a,b)
    # Denormalize y from [a,b] back to the range of x
    xmax=maximum(x)
    xmin=minimum(x)
    f(z)=(z-a)*(xmax-xmin)/(b-a)+xmin
    return map(f,y)
end


# =============================================================================
# Forecasting Error Metrics
# =============================================================================
# All metrics operate in the ORIGINAL (denormalized) log-RV scale.
# The `dif` and `coe` factors convert from normalized [-1,0] to original scale.
#
# MAPE: Mean Absolute Percentage Error (in %)
# MAE:  Mean Absolute Error
# MSE:  Mean Squared Error
# RMSE: Root Mean Squared Error

MAPE(x,y) = mean(abs.((x-y).*dif./((y.+1)*dif.+Min_RV)))*100
MAPE_std(x,y) = std(abs.((x-y).*dif./((y.+1)*dif.+Min_RV)))*100

MAE(x,y) = mean(abs.((x-y).*dif))
MAE_std(x,y) = std(abs.((x-y).*dif))

MSE(x,y) = mean(((x-y).^2).*coe)
MSE_std(x,y) = std(((x-y).^2).*coe)

RMSE(x,y) = sqrt(mean(((x-y).^2).*coe))


# =============================================================================
# Quantum_Reservoir — Main Quantum Reservoir Computing Function
# =============================================================================
# This is the core function that simulates the quantum reservoir for ALL
# data points. It implements the 3-step iterative protocol from the paper
# (Section II.B, Figure 1):
#
# For each time step l (out-of-sample month):
#   Step I:   Encode x_{t-K} into input qubits via RY gates.
#             Initialize hidden qubits as |0><0|.
#             Evolve joint state under H for time τ.
#             Trace out input qubits → ρ_hidden.
#
#   Step II:  Encode x_{t-K+1} into fresh input qubits.
#             Tensor product with ρ_hidden from Step I.
#             Evolve under H for time τ.
#             Trace out input qubits → updated ρ_hidden.
#
#   Step III: Encode x_{t-1} into fresh input qubits.
#             Tensor product with ρ_hidden from Step II.
#             Evolve under H for time δτ = τ/VirtualNode.
#             Measure <Z_i> for each qubit → readout features.
#
# Parameters:
#   Data        — Full dataset (DataFrame, 816 rows)
#   features    — Input feature column names (e.g., ["RV","MKT",...])
#   QR          — Reservoir Hamiltonian (from Qreservoir())
#   Observable  — List of measurement operators (Pauli-Z on each qubit)
#   K_delay     — Memory depth / number of input steps (K=3)
#   VirtualNode — Number of virtual nodes (1 for QR1, 2 for QR2)
#   τ           — Total evolution time (=1, the energy unit)
#   nqubit      — Total number of qubits (10)
#
# Returns:
#   Output — Matrix (N*VirtualNode × L): readout features for all time steps
# =============================================================================
function Quantum_Reservoir(Data, features, QR, Observable, K_delay, VirtualNode, τ, nqubit)
    N = length(Observable)          # Number of observables (= nqubit = 10)
    L = size(Data,1)                # Total data points (816)

    InputSize = length(features)    # Number of input features (7)

    Output = zeros(N*VirtualNode,L) # Readout feature matrix

    δτ = τ/VirtualNode              # Virtual node evolution time step

    # Pre-compute unitary evolution operators (matrix exponentials)
    # U = e^{-iτH}:  full evolution for intermediate steps
    # δU = e^{-iδτH}: fractional evolution for virtual node readout
    δU = CuArray(exp(-im*δτ*Matrix(matrix(QR))))
    U = CuArray(exp(-im*τ*Matrix(matrix(QR))))

    for l in (K_delay+1):L
        # Build input encoding circuit: one RY gate per input feature
        cir = QCircuit()
        for i in 1:InputSize
            push!(cir,RyGate(i,rand(),isparas=true))
        end

        # Initialize hidden qubits in |0> state
        ρᵣ = CuDensityMatrixBatch{ComplexF32}(nqubit-InputSize,1)

        # Process K time steps (from oldest to newest)
        for k in K_delay:-1:1
            # Prepare fresh input qubits in |0> state
            ρ₁ = CuDensityMatrixBatch{ComplexF32}(InputSize,1)

            # Encode input data via RY rotation gates
            # Each feature value is multiplied by π to map [-1,0] → [-π,0]
            para = [Data[l-k,str] for str in features]
            cir(para.*pi)

            # Combine hidden state with encoded input state
            if (nqubit-InputSize)==0
                ρ = cir(ρ₁)
            else
                ρ = ρᵣ⊗(cir(ρ₁))
            end

            if k!=1
                # Intermediate step: evolve and trace out input qubits
                # This implements ρ' = U * ρ * U†
                ρ = U*ρ*U'
                # Partial trace: discard input qubits, keep hidden state
                ρᵣ=partial_tr(ρ, Vector(1:InputSize))
            else
                # Final step: evolve in small increments, measure at each
                it=1
                for v in 1:VirtualNode
                    ρ = δU*ρ*δU'   # Evolve for δτ
                    for n in 1:N
                        # Measure expectation value <Z_n>
                        Output[it,l] = vec(real(expectation(B[n],ρ)))[1]
                        it+=1
                    end
                end
            end
        end
    end
    return Output
end


# =============================================================================
# QLIKE Loss Function
# =============================================================================
# Quasi-Likelihood loss for volatility forecast evaluation (Patton, 2011).
# Operates in the denormalized (original log-RV) scale.
#
# Formula: QLIKE = Σ( ratio - log(ratio) - 1 )
#   where ratio = |actual_RV| / |forecast_RV|
#
# The denormalization converts from normalized [-1,0] to original log-RV:
#   actual_RV = (y + 1) * dif + Min_RV
# =============================================================================
function compute_qlike(forecasts,  actuals)
    """
    Compute the QLIKE (Quasi-Likelihood) loss function for evaluating forecasting accuracy.
    forecasts: Forecasted variance (sigma squared from a model)
    actuals: Realized variance (actual observed variance)
    """
    # Denormalize both to original log-RV scale
    forecasts =abs.((forecasts.+1)*dif.+Min_RV)
    actuals = abs.((actuals.+1)*dif.+Min_RV)

    ratio = actuals ./ forecasts
    qlike = sum(ratio - log.(ratio).-1)
    return qlike
end

function compute_qlike2(forecasts, actuals)
    # Alternative QLIKE that operates in exp(log-RV) = RV space
    forecasts =(forecasts.+1)*dif.+Min_RV
    actuals = (actuals.+1)*dif.+Min_RV

    ratio = exp.(actuals) ./ exp.(forecasts)
    qlike = sum(ratio -(actuals-forecasts).-1)
    return qlike
end


# =============================================================================
# coeff_matrix — Generate Random Coupling Matrices
# =============================================================================
# Creates a random symmetric coupling matrix J_{ij} for the Ising Hamiltonian.
# The matrix is normalized so that its largest eigenvalue equals J (=1).
# This ensures the Hamiltonian has a controlled energy scale.
#
# Parameters:
#   N — Number of qubits (10)
#   J — Maximum coupling strength (1.0)
#
# Note: The actual matrices used in the paper are pre-generated and stored
# in coeff_10.jld2 (100 random instances, best one selected).
# =============================================================================
function coeff_matrix(N,J)
    m=rand(N,N)
    m=(m+transpose(m))./2   # Symmetrize
    for i in 1:N
        m[i,i]=0.0          # No self-coupling
    end
    return m./max(eigvals(m)...).*J  # Normalize by largest eigenvalue
end


# =============================================================================
# wave / hitrate — Directional Accuracy (Hit Rate)
# =============================================================================
# Measures how often the model correctly predicts the DIRECTION of RV change.
# wave() converts a time series to +1 (up) / -1 (down) direction indicators.
# hitrate() compares predicted vs actual direction changes.

function wave(y)
    L=length(y)
    w=zeros(L-1)
    for i in 1:L-1
        w[i]=sign(y[i+1]-y[i])
    end
    return w
end

function hitrate(x,y)
    L=length(x)
    # Prepend the last training value for the first direction comparison
    x=vcat(-0.5704088242386152,x)
    y=vcat(-0.5704088242386152,y)
    wx=wave(x)
    wy=wave(y)
    return sum(wx.==wy)/L
end


# =============================================================================
# QCircuit callable extensions
# =============================================================================
# Make QCircuit objects callable:
#   cir(parameters) — set circuit parameters (RY rotation angles)
#   cir(ρ)          — apply circuit to a density matrix

function (c::QCircuit)(p::Vector)
    return reset_parameters!(c,p)
end

function (c::QCircuit)(ρ::DensityMatrix)
    return c*ρ
end

function (c::QCircuit)(ρ::CuDensityMatrixBatch)
    return c*ρ
end


# =============================================================================
# shift / rolling — Time Series Manipulation Utilities
# =============================================================================
# shift(): Shift a vector/matrix forward by `step` positions (zero-fill start)
# rolling(): Create a matrix of lagged copies for time-delay embedding

function shift(V::Vector{T},step::Int) where T
    V1=zeros(T, length(V))
    V1[step+1:end] = V[1:end-step]
    return V1
end

function shift(V::Matrix{T},step::Int) where T
    V1=zeros(T, size(V))
    V1[step+1:end,:] = V[1:end-step,:]
    return V1
end

function rolling(V::Vector{T},window::Int) where T
    # Creates a (length(V) × window) matrix where column i contains
    # V shifted by (i-1) positions. Used for time-delay embedding.
    M = zeros(T,length(V),window)
    for i in 1:window
        M[i:end,i]=V[1:end-i+1]
    end
    return M
end


# =============================================================================
# Quantum_Reservoir_util — Variant for Shapley Value Analysis
# =============================================================================
# Similar to Quantum_Reservoir but takes pre-formatted data columns
# instead of named features. Used by the Shapley value computation in
# Time_serial_Finance_regression.ipynb to assess feature importance.
function Quantum_Reservoir_util(Data, features, QR, Observable, K_delay, VirtualNode, τ, nqubit)
    N = length(Observable)
    L = size(Data,1)

    InputSize = length(features)

    Output = zeros(N*VirtualNode,L)

    δτ = τ/VirtualNode

    δU = CuArray(exp(-im*δτ*Matrix(matrix(QR))))
    U = CuArray(exp(-im*τ*Matrix(matrix(QR))))

    for l in (K_delay+1):L
        cir = QCircuit()
        for i in 1:InputSize
            push!(cir,RyGate(i,rand(),isparas=true))
        end
        ρᵣ = CuDensityMatrixBatch{ComplexF32}(nqubit-InputSize,1)
        for k in K_delay:-1:1

            ρ₁ = CuDensityMatrixBatch{ComplexF32}(InputSize,1)
            # Key difference: reads data by column index rather than feature name
            para = Vector(Data[l-1,(k-1)*InputSize+1:k*InputSize])
            cir(para.*pi)
            ρ = ρᵣ⊗(cir(ρ₁))
            if k!=1
                ρ = U*ρ*U'
                ρᵣ=partial_tr(ρ, Vector(1:InputSize))
            else
                it=1
                for v in 1:VirtualNode
                    ρ = δU*ρ*δU'
                    for n in 1:N
                        Output[it,l] = vec(real(expectation(B[n],ρ)))[1]
                        it+=1
                    end
                end
            end
        end
    end
    return Output
end


# =============================================================================
# Quantum_Reservoir_single — Single Time-Step Reservoir (CPU version)
# =============================================================================
# Processes a single input sequence through the quantum reservoir on CPU.
# Used for threaded parallelism (commented-out alternative in the notebook).
#
# Parameters:
#   Input       — (K × InputSize) matrix of input features for one sample
#   U           — Full evolution unitary e^{-iτH}
#   U1          — Fractional evolution unitary e^{-iδτH}
#   Observable  — Measurement operators (Pauli-Z)
#   VirtualNode — Number of virtual nodes
#   nqubit      — Total qubits (10)
#   bias        — Initial parameters for hidden qubit circuit
# =============================================================================
function Quantum_Reservoir_single(Input, U, U1, Observable, VirtualNode, nqubit,bias)
    N = length(Observable)
    K, InputSize = size(Input)

    # Input encoding circuit
    cir = QCircuit()
    for i in 1:InputSize
        push!(cir,RyGate(i,rand(),isparas=true))
    end

    # Hidden qubit initialization circuit (with learnable bias)
    cir2 = QCircuit()
    for i in 1:nqubit-InputSize
        push!(cir2,RyGate(i,rand(),isparas=true))
    end
    cir2(bias.*pi)
    ρᵣ = DensityMatrix(nqubit-InputSize)
    ρᵣ=cir2(ρᵣ)
    Output = zeros(N*VirtualNode)

    for k in 1:K
        ρ₁ = DensityMatrix(InputSize)
        cir(Input[k,:].*pi)
        ρ = ρᵣ⊗(cir(ρ₁))
        if k!=K
            ρ = U*ρ*U'
            ρᵣ=partial_tr(ρ, Vector(1:InputSize))
        else
            it=1
            for v in 1:VirtualNode
                ρ = U1*ρ*U1'
                for n in 1:N
                    Output[it] = real(expectation(Observable[n],ρ))
                    it+=1
                end
            end
        end
    end
    return Output
end

# =============================================================================
# Commented-out alternative: Threaded Quantum_Reservoir
# =============================================================================
# This version uses Julia's Threads.@threads for CPU parallelism.
# Uncommented in the notebook when GPU is not available.
#
# function Quantum_Reservoir(Data, features, QR, Observable, K_delay, VirtualNode, τ, nqubit)
#     N = length(Observable)
#     L = size(Data,1)
#     x_data = zeros(N*VirtualNode,L)
#     Input = Matrix(Data[:,features])
#     δv = τ/VirtualNode
#     U = exp(-im*τ*Matrix(matrix(QR)))
#     U1 = exp(-im*δv*Matrix(matrix(QR)))
#     Threads.@threads for l in K_delay+1:L
#         x_data[:,l] = Quantum_Reservoir_single(Input[l-K_delay:l,:], U, U1, Observable, VirtualNode, nqubit)
#     end
#     return x_data
# end
