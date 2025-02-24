#################################
# Tensorial nD polynomial bases #
#################################

"""
    struct FEECPolyBasis{r,k,F,D,...} <: PolynomialBasis

where `F` is in  {:P⁻,:P,:Q⁻,:S} and represents the FE family in Arnold et. al. nomenclature.

Finite Element Exterior Calculus polynomial basis for the spaces `Fr`Λ`ᴷ`
in dimension `D`, that is P⁻`r`Λ`ᴷ`(△`ᴰ`), P`r`Λ`ᴷ`(△`ᴰ`), Q⁻`r`Λ`ᴷ`(□`ᴰ`) or S`r`Λ`ᴷ`(□`ᴰ`).

Reference: D. N. Arnold and A. Logg, Periodic Table of the Finite Elements, SIAM News, vol. 47 no. 9, November 2014
"""
struct FEECPolyBasis{r,k,F,D,V,K,PT,B} <: PolynomialBasis{D,V,K,PT}
  _basis::B # <: PolynomialBasis{D,V,K,PT}

  function FEECPolyBasis{D}(::Type{T},r,k,F::Symbol,::Type{PT}) where {D,PT<:Polynomial,T}
    @check T<:Real "T needs to be <:Real since represents the scalar type"
    @check k in 0:D "The form order k must be in 0:D"
    @check r > 0    "The polynomial order r must be positive"

    b = _select_FEEC_basis(r,k,F,Val(D),T,PT)
    V = return_type(b)
    K = get_order(b)
    new{r,k,F,D,V,K,PT,typeof(b)}(b)
  end
end

@inline Base.size(b::FEECPolyBasis) = size(b._basis)

function FEECPolyBasis(::Val{D},::Type{T},r,k,F::Symbol,pt::Type{PT}=Monomial) where {D,T,PT<:Polynomial}
  FEECPolyBasis{D}(T,r,k,F,pt)
end


# Implementation

function _select_FEEC_basis(r,k,F,::Val{D},::Type{T},::Type{PT}) where {D,T,PT}
  @check F in (:P⁻,:P,:Q⁻,:S) "F must be either :P⁻,:P,:Q⁻ or :S"

  if k == 0
    # Scalar function
    @notimplementedif r < 1
    if     F == :P⁻ || F == :P # Lagrange, ℙr space
      CartProdPolyBasis(PT,Val(D),T,r,_p_filter)
    elseif F == :Q⁻            # Lagrange, ℚr space
      CartProdPolyBasis(PT,Val(D),T,r,_q_filter)
    elseif F == :S             # Lagrange, 𝕊r space
      CartProdPolyBasis(PT,Val(D),T,r,_ser_filter)
    end


  elseif k == D
    # Scalar densities
    if     F == :P⁻ # Lagrange, ℙr₋1 space
      CartProdPolyBasis(PT,Val(D),T,r-1,_p_filter)
    elseif F == :P  # Lagrange, ℙr space
      CartProdPolyBasis(PT,Val(D),T,r,_p_filter)
    elseif F == :Q⁻ # Lagrange, ℚr₋1 space
      CartProdPolyBasis(PT,Val(D),T,r-1,_q_filter)
    elseif F == :S  # Serandipity Lagrange ≡ ℙr space
      CartProdPolyBasis(PT,Val(D),T,r,_p_filter)
    end


  elseif k == 1 # D > 1
    @notimplementedif D > 3
    V = VectorValue{D,T}

    if D == 2
      if     F == :P⁻ # Raviart-Thomas
        @notimplementedif PT ≠ Monomial
        PCurlGradBasis(PT,Val(D),T,r-1)
      elseif F == :P  # BDM on simplex
        CartProdPolyBasis(PT,Val(D),V,r,Polynomials._p_filter)
      elseif F == :Q⁻ # Raviart-Thomas
        QCurlGradBasis(PT,Val(D),T,r-1)
      elseif F == :S  # BDM on D-cubes
        @notimplemented
      end

    else # so D == 3
      if     F == :P⁻ # First kind Nedelec
        @notimplementedif PT ≠ Monomial
        PGradBasis(PT,Val(D),T,r-1)
      elseif F == :P  # Second kind Nedelec
        @notimplemented
      elseif F == :Q⁻ # First kind Nedelec
        QGradBasis(PT,Val(D),T,r-1)
      elseif F == :S  # "Serendipity second kind Nedelec" ?
        @notimplemented
      end
    end


  elseif k == 2 # D > 2
    @notimplementedif D > 3
    # D == 3
    if     F == :P⁻ # Raviart-Thomas
      PCurlGradBasis(PT,Val(D),T,r-1)
    elseif F == :P  # "3D BDM" ?
      @notimplemented
    elseif F == :Q⁻ # Raviart-Thomas
      QCurlGradBasis(PT,Val(D),T,r-1)
    elseif F == :S  # "3D Serandipity BDM" ?
      @notimplemented
    end

  else
    @unreachable
  end
end

# API

function return_cache(b::FEECPolyBasis, x::AbstractVector{<:Point})
  return_cache(b._basis,x)
end

function evaluate!(cache, b::FEECPolyBasis, x::AbstractVector{<:Point})
  evaluate!(cache, b._basis, x)
end

# Derivatives, option 2:

function return_cache(
  fg::FieldGradientArray{1,<:FEECPolyBasis},
  x::AbstractVector{<:Point})

  fg_b = FieldGradientArray{1}(fg.fa._basis)
  return (fg_b, return_cache(fg_b,x))
end

function evaluate!(cache,
  fg::FieldGradientArray{1,<:FEECPolyBasis},
  x::AbstractVector{<:Point})

  fg_b, b_cache = cache
  evaluate!(b_cache, fg_b, x)
end

function return_cache(
  fg::FieldGradientArray{2,<:FEECPolyBasis},
  x::AbstractVector{<:Point})

  fg_b = FieldGradientArray{2}(fg.fa._basis)
  return (fg_b, return_cache(fg_b,x))
end

function evaluate!(cache,
  fg::FieldGradientArray{2,<:FEECPolyBasis},
  x::AbstractVector{<:Point})

  fg_b, b_cache = cache
  evaluate!(b_cache, fg_b, x)
end




