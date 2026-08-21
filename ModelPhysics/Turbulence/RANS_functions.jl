# TO DO: These functions needs to be organised in a more sensible manner
function bound!(field, config)
    # Extract hardware configuration
    (; hardware) = config
    (; backend, workgroup) = hardware

    (; values, mesh) = field
    (; cells, cell_neighbours) = mesh

    # set up and launch kernel
    ndrange = length(values)
    kernel! = _bound!(_setup(backend, workgroup, ndrange)...)
    kernel!(values, cells, cell_neighbours)
    # KernelAbstractions.synchronize(backend)
end

@kernel function _bound!(values, cells, cell_neighbours)
    i = @index(Global)

    sum_flux = 0.0
    sum_area = 0
    average = 0.0
    @uniform mzero = eps(eltype(values)) # machine zero

    @inbounds begin
        for fi ∈ cells[i].faces_range
            cID = cell_neighbours[fi]
            sum_flux += max(values[cID], mzero) # bounded sum
            sum_area += 1
        end
        average = sum_flux/sum_area

        values[i] = max(
            max(
                values[i],
                average*signbit(values[i])
            ),
            mzero
        )
    end
end

y_plus_laminar(E, kappa) = begin
    yL = 11.0; for i ∈ 1:10; yL = log(max(yL*E, 1.0))/kappa; end
    yL
end

ω_vis(nu, y, beta1) = 6*nu/(beta1*y^2)

ω_log(k, y, cmu, kappa) = sqrt(k)/(cmu^0.25*kappa*y)

y_plus(k, nu, y, cmu) = cmu^0.25*y*sqrt(k)/nu

sngrad(Ui, Uw, delta, normal) = begin
    Udiff = (Ui - Uw)
    Up = Udiff - (Udiff⋅normal)*normal # parallel velocity difference
    grad = Up/delta
    return grad
end

mag(vector) = sqrt(vector[1]^2 + vector[2]^2 + vector[3]^2) 

nut_wall(nu, yplus, kappa, E::T) where T = begin
    max(nu*(yplus*kappa/log(max(E*yplus, 1.0 + 1e-4)) - 1.0), zero(T))
end

# OpenFOAM's actual nutkWallFunction::calcNut() (STEPWISE blending, the
# default) has no "-1" term in the log-law branch, and returns nu (not 0)
# in the viscous sublayer -- nutVis[facei] = turbModel.nu(patchi), used
# directly as nutw there, not as a correction subtracted from the log-law
# value. Gated behind `fixed` for backward compatibility.
nut_wall_v2(nu, yplus, kappa, E::T, yPlusLam) where T = begin
    yplus > yPlusLam ? nu*yplus*kappa/log(max(E*yplus, 1.0 + 1e-4)) : nu
end

@generated correct_production!(P, fieldBCs, model, gradU, config, wallfn_v2=false) = begin
    BCs = fieldBCs.parameters
    func_calls = Expr[]
    for i ∈ eachindex(BCs)
        call = quote
            set_production!(P, fieldBCs[$i], model, gradU, config, wallCount)
        end
        push!(func_calls, call)
    end
    quote
    wallCount = nothing
    if wallfn_v2
        mesh = model.domain
        (; hardware) = config
        wallCount = KernelAbstractions.zeros(hardware.backend, _get_float(mesh), length(mesh.cells))
        count_all_wallfn_cells!(wallCount, P, fieldBCs, model, config)
    end
    $(func_calls...)
    nothing
    end
end

set_production!(P, BC, model, gradU, config, wallCount=nothing) = nothing

function set_production!(P, BC::KWallFunction, model, gradU, config, wallCount=nothing)
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundary_cellsID, boundaries) = mesh

    # Extract physics models
    (; fluid, momentum, turbulence) = model

    # facesID_range = get_boundaries(BC, boundaries)
    # boundaries_cpu = get_boundaries(boundaries)
    # facesID_range = boundaries_cpu[BC.ID].IDs_range
    facesID_range = BC.IDs_range
    start_ID = facesID_range[1]

    fixed = !isnothing(wallCount)

    # Execute apply boundary conditions kernel
    ndrange = length(facesID_range)
    kernel! = _set_production!(_setup(backend, workgroup, ndrange)...)
    kernel!(
        P.values, BC, fluid, momentum, turbulence, faces, boundary_cellsID, start_ID, gradU,
        wallCount, fixed
    )
end

@kernel function _set_production!(
    values, BC::KWallFunction, fluid, momentum, turbulence, faces, boundary_cellsID, start_ID, gradU,
    wallCount, fixed)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    (; kappa, beta1, cmu, B, E, yPlusLam) = BC.value
    (; nu) = fluid
    (; U) = momentum
    (; k, nut) = turbulence

    Uw = SVector{3}(0.0,0.0,0.0)
    # Uw = boundaries.U[BC.ID].value
    cID = boundary_cellsID[fID]
    face = faces[fID]
    nuc = nu[cID]
    (; delta, normal)= face
    uStar = cmu^0.25*sqrt(k[cID])
    dUdy = uStar/(kappa*delta)
    yplus = y_plus(k[cID], nuc, delta, cmu)
    mag_grad_U = mag(sngrad(U[cID], Uw, delta, normal))
    # mag_grad_U = mag(gradU[cID]*normal)

    if fixed
        # OpenFOAM (omegaWallFunctionFvPatchScalarField::calculate) sums
        # G contributions from every wall-function face touching a cell,
        # each weighted 1/(number of such faces) -- corner/edge cells on
        # motorBike's ~90 adjacent body patches otherwise only see the
        # last-processed face's contribution.
        nutw = nut_wall_v2(nuc, yplus, kappa, E, yPlusLam)
        contribution = yplus > yPlusLam ? (nu[cID] + nutw)*mag_grad_U*dUdy : zero(eltype(values))
        w = one(eltype(wallCount))/wallCount[cID]
        Atomix.@atomic values[cID] += w*contribution
    else
        nutw = nut_wall(nuc, yplus, kappa, E)
        if yplus > yPlusLam
            values[cID] = (nu[cID] + nutw)*mag_grad_U*dUdy
        else
            values[cID] = 0.0
        end
    end
end

# --- Wall function v2 shared infrastructure: count wall-function faces per
# cell (so contributions from cells touching multiple wall patches can be
# averaged instead of the last one silently winning), gated behind
# XCALIBRE_WALLFN_V2=1 for backward compatibility. ---

count_wallfn_cell!(countField, P, BC, model, config) = nothing

function count_wallfn_cell!(countField, P, BC::KWallFunction, model, config)
    mesh = model.domain
    (; boundary_cellsID) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware
    facesID_range = BC.IDs_range
    start_ID = facesID_range[1]
    ndrange = length(facesID_range)
    kernel! = _count_wallfn_cell!(_setup(backend, workgroup, ndrange)...)
    kernel!(countField, P.values, boundary_cellsID, start_ID)
end

@kernel function _count_wallfn_cell!(countField, Pvalues, boundary_cellsID, start_ID)
    i = @index(Global)
    @inbounds begin
        fID = i + start_ID - 1
        cID = boundary_cellsID[fID]
        Atomix.@atomic countField[cID] += one(eltype(countField))
        Pvalues[cID] = zero(eltype(Pvalues)) # idempotent reset before accumulation
    end
end

@generated function count_all_wallfn_cells!(countField, P, fieldBCs, model, config)
    BCs = fieldBCs.parameters
    calls = Expr[]
    for i ∈ eachindex(BCs)
        push!(calls, :(count_wallfn_cell!(countField, P, fieldBCs[$i], model, config)))
    end
    quote
        $(calls...)
        nothing
    end
end

@generated function correct_eddy_viscosity!(νtf, nutBCs, model, config, wallfn_v2=false)
    unpacked_BCs = []
    for i ∈ 1:length(nutBCs.parameters)
        unpack = quote
            correct_nut_wall!(νtf, nutBCs[$i], model, config, wallfn_v2)
        end
        push!(unpacked_BCs, unpack)
    end
    quote
    $(unpacked_BCs...)
    end
end

correct_nut_wall!(nutf, BC, model, config, wallfn_v2=false) = nothing

function correct_nut_wall!(νtf, BC::NutWallFunction, model, config, wallfn_v2=false)
    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundary_cellsID, boundaries) = mesh

    # Extract physics models
    (; fluid, turbulence) = model

    # facesID_range = get_boundaries(BC, boundaries)
    # boundaries_cpu = get_boundaries(boundaries)
    # facesID_range = boundaries_cpu[BC.ID].IDs_range
    facesID_range = BC.IDs_range
    start_ID = facesID_range[1]

    # Execute apply boundary conditions kernel
    fixed = wallfn_v2
    ndrange=length(facesID_range)
    kernel! = _correct_nut_wall!(_setup(backend, workgroup, ndrange)...)
    kernel!(νtf.values, fluid, turbulence, BC, faces, boundary_cellsID, start_ID, fixed)
end

@kernel function _correct_nut_wall!(
    values, fluid, turbulence, BC::NutWallFunction, faces, boundary_cellsID, start_ID, fixed)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    (; kappa, beta1, cmu, B, E, yPlusLam) = BC.value
    (; nu) = fluid
    (; k) = turbulence

    cID = boundary_cellsID[fID]
    face = faces[fID]
    # nuf = nu[fID]
    (; delta)= face
    # yplus = y_plus(k[cID], nuf, delta, cmu)
    nuc = nu[cID]
    yplus = y_plus(k[cID], nuc, delta, cmu)
    if fixed
        values[fID] = nut_wall_v2(nuc, yplus, kappa, E, yPlusLam)
    else
        nutw = nut_wall(nuc, yplus, kappa, E)
        if yplus > yPlusLam
            values[fID] = nutw
        else
            values[fID] = 0.0
        end
    end
end

function correct_nut_wall!(νtf, BC::NutMixingLengthWallFunction, model, config, wallfn_v2=false)
    (; hardware) = config
    (; backend, workgroup) = hardware

    mesh = model.domain
    (; faces, boundary_cellsID) = mesh
    (; fluid, momentum, turbulence) = model

    facesID_range = BC.IDs_range
    start_ID = facesID_range[1]

    ndrange = length(facesID_range)
    kernel! = _correct_nut_wall_mixing_length!(_setup(backend, workgroup, ndrange)...)
    kernel!(νtf.values, fluid, momentum, turbulence, BC, faces, boundary_cellsID, start_ID)
end

@kernel function _correct_nut_wall_mixing_length!(
    values, fluid, momentum, turbulence, BC::NutMixingLengthWallFunction, faces, boundary_cellsID, start_ID)
    i = @index(Global)
    fID = i + start_ID - 1

    (; kappa, E, yPlusLam) = BC.value
    (; nu) = fluid
    (; U) = momentum
    (; nut) = turbulence

    cID = boundary_cellsID[fID]
    face = faces[fID]
    (; delta, normal) = face
    nuc = nu[cID]

    # Tangential velocity magnitude at cell centre (wall velocity = 0)
    Ucell = U[cID]
    U_tang = Ucell - (Ucell ⋅ normal) * normal
    U_tang_mag = mag(U_tang)

    # Newton iteration: solve U_tang_mag = (u_tau/kappa)*ln(E*u_tau*delta/nu) for u_tau
    # Initial guess from viscous sublayer: u_tau ≈ sqrt(nu*|U_t|/delta)
    u_tau = sqrt(nuc * U_tang_mag / delta + eltype(values)(1e-20))
    for _ in 1:10
        yp  = u_tau * delta / nuc
        lv  = log(max(E * yp, eltype(values)(1.0 + 1e-4)))
        f   = U_tang_mag * kappa - u_tau * lv
        df  = -(lv + one(eltype(values)))
        u_tau = max(u_tau - f / df, eltype(values)(1e-20))
    end

    yplus = u_tau * delta / nuc
    nutw  = nut_wall(nuc, yplus, kappa, E)

    if yplus > yPlusLam
        values[fID] = nutw
        nut[cID] = nutw
    else
        values[fID] = zero(eltype(values))
    end
end

@generated constrain_equation!(eqn, fieldBCs, model, config, wallfn_v2=false) = begin
    BCs = fieldBCs.parameters
    old_calls = Expr[]
    count_calls = Expr[]
    accum_calls = Expr[]
    finalize_calls = Expr[]
    for i ∈ eachindex(BCs)
        push!(old_calls, :(constrain!(eqn, fieldBCs[$i], model, config)))
        push!(count_calls, :(count_wallfn_omega_cell!(wallCount, fieldBCs[$i], model, config)))
        push!(accum_calls, :(accumulate_omega_wall!(omega0, wallCount, fieldBCs[$i], model, config)))
        push!(finalize_calls, :(finalize_omega_wall!(omega0, wallCount, eqn, fieldBCs[$i], model, config)))
    end
    quote
    if wallfn_v2
        # OpenFOAM's omegaWallFunctionFvPatchScalarField::calculate sums
        # weighted contributions from every wall-function face touching a
        # cell (weight = 1/count) before constraining the equation once
        # per cell -- otherwise corner/edge cells (common on motorBike's
        # ~90 adjacent body patches) only see whichever face is processed
        # last.
        mesh = model.domain
        (; hardware) = config
        TF = _get_float(mesh)
        n_cells = length(mesh.cells)
        wallCount = KernelAbstractions.zeros(hardware.backend, TF, n_cells)
        omega0 = ScalarField(mesh)
        $(count_calls...)
        $(accum_calls...)
        $(finalize_calls...)
    else
        $(old_calls...)
    end
    nothing
    end
end

constrain!(eqn, BC, model, config) = nothing

function constrain!(eqn, BC::OmegaWallFunction, model, config)

    # backend = _get_backend(mesh)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # Access equation data and deconstruct sparse array
    A = _A(eqn)
    b = _b(eqn, nothing)
    colval = _colval(A)
    rowptr = _rowptr(A)
    nzval = _nzval(A)
    
    # Deconstruct mesh to required fields
    mesh = model.domain
    (; faces, boundaries, boundary_cellsID) = mesh

    fluid = model.fluid 
    # turbFields = model.turbulence.fields
    turbulence = model.turbulence

    # facesID_range = get_boundaries(BC, boundaries)
    # boundaries_cpu = get_boundaries(boundaries)
    # facesID_range = boundaries_cpu[BC.ID].IDs_range
    facesID_range = BC.IDs_range
    start_ID = facesID_range[1]

    # Execute apply boundary conditions kernel
    ndrange = length(facesID_range)
    kernel! = _constrain!(_setup(backend, workgroup, ndrange)...)
    kernel!(
        turbulence, fluid, BC, faces, start_ID, boundary_cellsID, colval, rowptr, nzval, b
    )
end

@kernel function _constrain!(turbulence, fluid, BC::OmegaWallFunction, faces, start_ID, boundary_cellsID, colval, rowptr, nzval, b)
    i = @index(Global)
    fID = i + start_ID - 1 # Redefine thread index to become face ID

    @uniform begin
        nu = fluid.nu
        k = turbulence.k
        (; kappa, beta1, cmu, B, E, yPlusLam) = BC.value
    end
    ωc = zero(eltype(nzval))
    
    @inbounds begin
        cID = boundary_cellsID[fID]
        face = faces[fID]
        y = face.delta
        ωvis = ω_vis(nu[cID], y, beta1)
        ωlog = ω_log(k[cID], y, cmu, kappa)
        yplus = y_plus(k[cID], nu[cID], y, cmu) 

        if yplus > yPlusLam 
            ωc = ωlog
        else
            ωc = ωvis
        end
        # Line below is weird but worked
        # b[cID] = A[cID,cID]*ωc

        
        # Classic approach
        # b[cID] += A[cID,cID]*ωc
        # A[cID,cID] += A[cID,cID]
        
        # nzIndex = spindex(rowptr, colval, cID, cID)
        # Atomix.@atomic b[cID] += nzval[nzIndex]*ωc
        # Atomix.@atomic nzval[nzIndex] += nzval[nzIndex] 

        z = zero(eltype(nzval))
        for nzi ∈ rowptr[cID]:(rowptr[cID+1] - 1)
            nzval[nzi] = z
        end
        cIndex = spindex(rowptr, colval, cID, cID)
        nzval[cIndex] = one(eltype(nzval))
        b[cID] = ωc
    end
end

# --- Wall function v2 for OmegaWallFunction: count -> weighted accumulate
# -> finalize (constrain matrix once per cell using the fully-accumulated
# value), matching OpenFOAM's cornerWeights_ = 1/(wall faces touching that
# cell), summed rather than overwritten. Three separate passes over all
# OmegaWallFunction patches because the matrix constraint must only be
# applied once all patches' contributions have been accumulated.

count_wallfn_omega_cell!(countField, BC, model, config) = nothing

function count_wallfn_omega_cell!(countField, BC::OmegaWallFunction, model, config)
    mesh = model.domain
    (; boundary_cellsID) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware
    facesID_range = BC.IDs_range
    start_ID = facesID_range[1]
    ndrange = length(facesID_range)
    kernel! = _count_wallfn_omega_cell!(_setup(backend, workgroup, ndrange)...)
    kernel!(countField, boundary_cellsID, start_ID)
end

@kernel function _count_wallfn_omega_cell!(countField, boundary_cellsID, start_ID)
    i = @index(Global)
    @inbounds begin
        fID = i + start_ID - 1
        cID = boundary_cellsID[fID]
        Atomix.@atomic countField[cID] += one(eltype(countField))
    end
end

accumulate_omega_wall!(omega0, countField, BC, model, config) = nothing

function accumulate_omega_wall!(omega0, countField, BC::OmegaWallFunction, model, config)
    mesh = model.domain
    (; faces, boundary_cellsID) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware
    fluid = model.fluid
    turbulence = model.turbulence
    facesID_range = BC.IDs_range
    start_ID = facesID_range[1]
    ndrange = length(facesID_range)
    kernel! = _accumulate_omega_wall!(_setup(backend, workgroup, ndrange)...)
    kernel!(omega0.values, countField, turbulence, fluid, BC, faces, boundary_cellsID, start_ID)
end

@kernel function _accumulate_omega_wall!(
    omega0vals, countField, turbulence, fluid, BC::OmegaWallFunction, faces, boundary_cellsID, start_ID)
    i = @index(Global)
    fID = i + start_ID - 1
    @uniform begin
        nu = fluid.nu
        k = turbulence.k
        (; kappa, beta1, cmu, B, E, yPlusLam) = BC.value
    end
    @inbounds begin
        cID = boundary_cellsID[fID]
        face = faces[fID]
        y = face.delta
        ωvis = ω_vis(nu[cID], y, beta1)
        ωlog = ω_log(k[cID], y, cmu, kappa)
        yplus = y_plus(k[cID], nu[cID], y, cmu)
        ωc = yplus > yPlusLam ? ωlog : ωvis
        w = one(eltype(countField))/countField[cID]
        Atomix.@atomic omega0vals[cID] += w*ωc
    end
end

finalize_omega_wall!(omega0, countField, eqn, BC, model, config) = nothing

function finalize_omega_wall!(omega0, countField, eqn, BC::OmegaWallFunction, model, config)
    mesh = model.domain
    (; boundary_cellsID) = mesh
    (; hardware) = config
    (; backend, workgroup) = hardware
    A = _A(eqn)
    b = _b(eqn, nothing)
    colval = _colval(A)
    rowptr = _rowptr(A)
    nzval = _nzval(A)
    facesID_range = BC.IDs_range
    start_ID = facesID_range[1]
    ndrange = length(facesID_range)
    kernel! = _finalize_omega_wall!(_setup(backend, workgroup, ndrange)...)
    kernel!(omega0.values, boundary_cellsID, start_ID, colval, rowptr, nzval, b)
end

@kernel function _finalize_omega_wall!(omega0vals, boundary_cellsID, start_ID, colval, rowptr, nzval, b)
    i = @index(Global)
    @inbounds begin
        fID = i + start_ID - 1
        cID = boundary_cellsID[fID]
        z = zero(eltype(nzval))
        for nzi ∈ rowptr[cID]:(rowptr[cID+1] - 1)
            nzval[nzi] = z
        end
        cIndex = spindex(rowptr, colval, cID, cID)
        nzval[cIndex] = one(eltype(nzval))
        b[cID] = omega0vals[cID]
    end
end

# @generated constrain_boundary!(field, fieldBCs, model, config) = begin
#     BCs = fieldBCs.parameters
#     func_calls = Expr[]
#     for i ∈ eachindex(BCs)
#         call = quote
#             set_cell_value!(field, fieldBCs[$i], model, config)
#         end
#         push!(func_calls, call)
#     end
#     quote
#     $(func_calls...)
#     nothing
#     end 
# end

# set_cell_value!(field, BC, model, config) = nothing

# function set_cell_value!(field, BC::OmegaWallFunction, model, config)
#     # backend = _get_backend(mesh)
#     (; hardware) = config
#     (; backend, workgroup) = hardware
    
#     # Deconstruct mesh to required fields
#     mesh = model.domain
#     (; faces, boundaries, boundary_cellsID) = mesh
#     (; fluid, turbulence) = model
#     # turbFields = turbulence.fields

#     # facesID_range = get_boundaries(BC, boundaries)
#     boundaries_cpu = get_boundaries(boundaries)
#     facesID_range = boundaries_cpu[BC.ID].IDs_range
#     start_ID = facesID_range[1]

#     # Execute apply boundary conditions kernel
        # ndrange=length(facesID_range)
#     kernel! = _set_cell_value!(_setup(backend, workgroup, ndrange)...)
#     kernel!(
#         field, turbulence, fluid, BC, faces, start_ID, boundary_cellsID
#     )
# end

# @kernel function _set_cell_value!(field, turbulence, fluid, BC, faces, start_ID, boundary_cellsID)
#     i = @index(Global)
#     fID = i + start_ID - 1 # Redefine thread index to become face ID

#     @uniform begin
#         (; nu) = fluid
#         (; k) = turbulence
#         (; kappa, beta1, cmu, B, E, yPlusLam) = BC.value
#         (; values) = field
#         ωc = zero(eltype(values))
#     end


#     @inbounds begin
#         cID = boundary_cellsID[fID]
#         face = faces[fID]
#         y = face.delta
#         ωvis = ω_vis(nu[cID], y, beta1)
#         ωlog = ω_log(k[cID], y, cmu, kappa)
#         yplus = y_plus(k[cID], nu[cID], y, cmu) 

#         if yplus > yPlusLam 
#             ωc = ωlog
#         else
#             ωc = ωvis
#         end

#         values[cID] = ωc # needs to be atomic?
#     end
# end

