# moment based 0-form FEs

"""
    struct ModalScalar{F} <: ReferenceFEName

where `F` is either `:P⁻`, `:P`, `:Q⁻` or `:S`.
"""
struct ModalScalar{F} <: ReferenceFEName
  ModalScalar{:P }() = new{:P }()
  ModalScalar{:P⁻}() = new{:P⁻}()
  ModalScalar{:Q⁻}() = new{:Q⁻}()
  ModalScalar{:S }() = new{:S }()
end

"""
    const modal_scalar_Pm = ModalScalar{:P⁻}()
    const modal_scalar_P  = ModalScalar{:P}()
    const modal_scalar_Qm = ModalScalar{:Q⁻}()
    const modal_scalar_S  = ModalScalar{:S}()

Singletons of the [`ModalScalar`](@ref) reference FE name.
"""
const modal_scalar_Pm = ModalScalar{:P⁻}()
const modal_scalar_P  = ModalScalar{:P}()
const modal_scalar_Qm = ModalScalar{:Q⁻}()
const modal_scalar_S  = ModalScalar{:S}()

"""
    ModalScalarRefFE(::Type{T}, p::Polytope{D}, order::Integer; F::Symbol, kwargs...)

Scalar moment-based reference FE, for space FᵣΛ⁰ where `r=order` and `F` is either
`:P⁻`, `:P`, `:Q⁻` or `:S`. `T` is the type of scalar components.

This is a variant of `lagrangian`/`serendipity` elements that is more accurate for high order (≥5).

The `kwargs` are [`sh_is_pb`](@ref "`sh_is_pb` keyword argument"),
[`poly_type`](@ref "`poly_type` keyword argument") and
[`mom_poly_type`](@ref "`mom_poly_type` keyword argument").

For `F=:S`, `mom_poly_type` is changed for `Legendre` when `ModalC0` (the
default) was given because the moment basis need be hierarchical.
"""
function ModalScalarRefFE(::Type{T}, p::Polytope{D}, r::Integer; F::Symbol,
  sh_is_pb=true, poly_type=_mom_reffe_default_PT(p), mom_poly_type=poly_type) where {T,D}

  PT, MPT = poly_type, mom_poly_type
  cart_prod = T <: MultiValue

  # TODO fix Q⁻/S on SEGMENT
  if is_simplex(p)
    if     F==:P⁻
      prebasis = FEEC_poly_basis(Val(D),T,r,    0,:P⁻,PT; cart_prod) # P⁻ᵣΛ⁰(□ᴰ)
      mb = [ (r-d-1 >= 0 ?
                 FEEC_poly_basis(Val(d),T,r-d-1,d,:P, MPT; cart_prod)
                 : nothing) for d in 0:D ]                # PᵨΛᵈ(□ᵈ), ρ = r-d-1
    elseif F==:P
      prebasis = FEEC_poly_basis(Val(D),T,r,  0,:P, PT; cart_prod)   # PᵣΛ⁰(□ᴰ)
      mb = [ (r-d > 0 ?
                 FEEC_poly_basis(Val(d),T,r-d,d,:P⁻,MPT; cart_prod)
                 : nothing) for d in 0:D ]                # P⁻ᵨΛᵈ(□ᵈ), ρ = r-d
    else
      @notimplemented "Only :P⁻ and :P elements are implemented on simplices, got F=$F."
    end
  elseif is_n_cube(p)
    if     F==:Q⁻
      prebasis = FEEC_poly_basis(Val(D),T,r,  0,:Q⁻,PT; cart_prod)   # Q⁻ᵣΛ⁰(□ᴰ)
      mb = [ ((r-1 > 0 || d==0) ?
                 FEEC_poly_basis(Val(d),T,r-1,d,:Q⁻,MPT; cart_prod)
                 : nothing) for d in 0:D ]                # Q⁻ᵨΛᵈ(□ᵈ), ρ = r-1
    elseif F==:S
      prebasis = FEEC_poly_basis(Val(D),T,r,   0,:S,PT; cart_prod)   # SᵣΛ⁰(□ᴰ)
      MPT = MPT == Polynomials.ModalC0 ? Legendre : MPT
      mb = [ (r-2d >= 0 ?
                 FEEC_poly_basis(Val(d),T,r-2d,d,:P,MPT; cart_prod)
                 : nothing) for d in 0:D ]                # PᵨΛᵈ(□ᵈ), ρ = r-2*d
    else
      @notimplemented "Only :Q⁻ and :S elements are implemented on n-cubes, got F=$F."
    end
  else
    @notimplemented "ModalScalar FE is only implemented on simplices and n-cubes."
  end

  # The vector proxies of d-forms that we get in `mb` must be turned back into
  # volume forms by multiplying by |df|
  mom_integrand = if is_simplex(p)
    @assert D<4
    function mom_s(φ,μ,ds) # moment function: σ_K(φ,μ) = ∫(φ*μ)df
      d = num_dims(ds.fpoly)
      face_measure = Gridap.ReferenceFEs._get_dfaces_measure(ds.cpoly, d)
      df = Gridap.Fields.ConstantField(face_measure[ds.face])
      dμ = Broadcasting(Operation(/))(μ,df)
      Broadcasting(Operation(⊙))(φ,dμ) # using inner in case of cartesian product space
    end
  elseif is_n_cube(p)
    # TODO assumes p is a reference n-cube (all d-faces have same d-volume),
    # otherwise face volume needed
    @assert p isa ExtrusionPolytope
    function mom_c(φ,μ,ds) # moment function: σ_K(φ,μ) = ∫(φ*μ)df
      Broadcasting(Operation(⊙))(φ,μ) # using inner in case of cartesian product space
    end
  end

  moments = Tuple[ (get_dimrange(p,d), mom_integrand, mb[d+1]) for d in 0:D if !isnothing(mb[d+1]) ]

  name = ModalScalar{F}()
  conf = GradConformity()
  sh_is_pb = _validate_sh_is_pb(sh_is_pb,prebasis,p,conf)
  MomentBasedReferenceFE(name,p,prebasis,moments,conf; sh_is_pb)
end

function ReferenceFE(p::Polytope,::ModalScalar{F},::Type{T}, order; kwargs...) where {F, T}
  ModalScalarRefFE(T,p,order; F, kwargs...)
end

function Conformity(::GenericRefFE{<:ModalScalar},sym::Symbol)
  hgrad = (:H1,:C0,:Hgrad,:HGrad)
  if sym == :L2
    L2Conformity()
  elseif sym in hgrad
    GradConformity()
  else
    @unreachable """\n
    It is not possible to use conformity = $sym on a Nedelec reference FE.

    Possible values of conformity for this reference fe are $((:L2, hcurl...)).
    """
  end
end

function get_face_own_dofs(reffe::GenericRefFE{<:ModalScalar}, conf::GradConformity)
  get_face_dofs(reffe)
end
#get_face_dofs(::GenericRefFE{<:ModalScalar}, ::GradConformity) = throw(ErrorException("not implemeted"))
