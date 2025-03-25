"""
    RaviartThomasPolyBasis{D,V,PT} <: PolynomialBasis{D,V,PT}

Basis of the vector valued (`V<:VectorValue{D}`) space

ℝ𝕋ᴰₙ = (𝕊ₙ)ᴰ ⊕ x (𝕊ₙ\\𝕊₍ₙ₋₁₎)

where 𝕊ₙ is a `D`-multivariate scalar polynomial space of maximum degree n = `K`-1.

This ℝ𝕋ᴰₙ is the polynomial space for Raviart-Thomas elements with divergence in 𝕊ₙ.
Its maximum degree is n+1 = `K`. `get_order` on it returns `K`.

The space 𝕊ₙ, typically ℙᴰₙ or ℚᴰₙ, does not need to have a tensor product
structure of 1D scalar spaces. Thus, the ℝ𝕋ᴰₙ component's scalar spaces are not
tensor products either.

𝕊ₙ is defined like a scalar valued [`CartProdPolyBasis`](@ref) via the `_filter`
argument of the constructor, by default `_p_filter` for ℙᴰₙ.
As a consequence, `PT` must be hierarchical, see [`isHierarchical`](@ref).
"""
struct RaviartThomasPolyBasis{D,V,PT} <: PolynomialBasis{D,V,PT}
  max_order::Int
  pterms::Vector{CartesianIndex{D}}
  sterms::Vector{CartesianIndex{D}}

  """
      RaviartThomasPolyBasis{D}(::Type{PT}, ::Type{T}, order::Int, _filter::Function=_p_filter)

  Where `_filter` defines 𝕊ₙ and `order` = n = K-1 (cf. struct docstring).
  """
  function RaviartThomasPolyBasis{D}(
    ::Type{PT}, ::Type{T}, order::Int,
    _filter::Function=_p_filter
    ) where {PT<:Polynomial,D,T}

    @check T<:Real "T needs to be <:Real since represents the type of the components of the vector value"
    @check D > 1
    @check isconcretetype(PT) "PT needs to be a concrete <:Polynomial type"
    @check isHierarchical(PT) "The polynomial basis must be hierarchical for this space."

    V = VectorValue{D,T}
    indexbase = 1

    # terms defining 𝕊ₙ
    P_k = MonomialBasis(Val(D), T, order, _filter)
    pterms = P_k.terms
    msg =  "Some term defining `𝕊ₙ` contain a higher index than the maximum,
      `order`+1, please fix the `_filter` argument"
    @check all( pterm -> (maximum(Tuple(pterm) .- indexbase, init=0) <= order), pterms) msg

    # terms defining 𝕊ₙ\𝕊ₙ₋₁
    _minus_one_order_filter = term -> _filter(Tuple(term) .- indexbase, order-1)
    sterms = filter(!_minus_one_order_filter, pterms)

    new{D,V,PT}(order+1,pterms,sterms)
  end
end

Base.size(b::RaviartThomasPolyBasis{D}) where {D} = (D*length(b.pterms) + length(b.sterms), )
get_order(b::RaviartThomasPolyBasis) = b.max_order


#################################
# nD evaluations implementation #
#################################

function _evaluate_nd!(
  b::RaviartThomasPolyBasis{D,V,PT}, x,
  r::AbstractMatrix{V}, i,
  c::AbstractMatrix{T}) where {D,V,PT,T}

  K = get_order(b)
  pterms = b.pterms
  sterms = b.sterms

  Kv = Val(K)
  for d in 1:D
    _evaluate_1d!(PT,Kv,c,x,d)
  end

  m = zero(Mutable(V))
  k = 1

  @inbounds begin
    for l in 1:D
      for ci in pterms

        s = one(T)
        for d in 1:D
          s *= c[d,ci[d]]
        end

        k = _comp_wize_set_value!(r,i,s,k,l)
      end
    end

    for ci in sterms
      for i in 1:D
        m[i] = zero(T)
      end

      for l in 1:D

        s = x[l]
        for d in 1:D
          s *= c[d,ci[d]]
        end

        m[l] = s
      end

      r[i,k] = m
      k += 1
    end

  end
end

function _gradient_nd!(
  b::RaviartThomasPolyBasis{D,V,PT}, x,
  r::AbstractMatrix{G}, i,
  c::AbstractMatrix{T},
  g::AbstractMatrix{T},
  s::MVector{D,T}) where {D,V,PT,G,T}

  K = get_order(b)
  pterms = b.pterms
  sterms = b.sterms

  Kv = Val(K)
  for d in 1:D
    _derivatives_1d!(PT,Kv,(c,g),x,d)
  end

  m = zero(Mutable(G))
  k = 1

  @inbounds begin
    for l in 1:D
      for ci in pterms

        for i in eachindex(s)
          s[i] = one(T)
        end

        for q in 1:D
          for d in 1:D
            if d != q
               s[q] *= c[d,ci[d]]
            else
               s[q] *= g[d,ci[d]]
            end
          end
        end

        k = _comp_wize_set_derivative!(r,i,s,k,Val(l),V)
      end
    end

    for ci in sterms

      for i in eachindex(m)
        m[i] = zero(T)
      end

      for l in 1:D

        for i in eachindex(s)
          s[i] = x[l]
        end

        aux = one(T)
        for q in 1:D
          aux *= c[q,ci[q]]
          for d in 1:D
            if d != q
               s[q] *= c[d,ci[d]]
            else
               s[q] *= g[d,ci[d]]
            end
          end
        end
        s[l] += aux

        for i in 1:D
           m[i,l] = s[i]
        end
      end
      r[i,k] = m
      k += 1
    end

  end
end

"""
    PCurlGradBasis(::Type{PT}, ::Val{D}, ::Type{T}, order::Int) :: PolynomialBasis

Return a basis of

ℝ𝕋ᴰₙ(△) = (ℙᴰₙ)ᴰ ⊕ x (ℙᴰₙ \\ ℙᴰₙ₋₁)

with n=`order`, the polynomial space for Raviart-Thomas elements on
`D`-dimensional simplices with scalar type `T`.

The `order`=n argument of this function has the following meaning: the divergence
of the functions in this basis is in ℙᴰₙ.

`PT<:Polynomial` is the choice of the family of the scalar 1D basis polynomials,
it must be hierarchical, see [`isHierarchical`](@ref).

# Example:

```jldoctest
# a basis for Raviart-Thomas on tetrahedra with divergence in ℙ₂
b = PCurlGradBasis(Monomial, Val(3), Float64, 2)
```

For more details, see [`RaviartThomasPolyBasis`](@ref), as `PCurlGradBasis` returns
an instance of\\
`RaviartThomasPolyBasis{D, VectorValue{D,T}, order+1, PT}`  for `D`>1, or\\
`CartProdPolyBasis{1, VectorValue{1,T}, order+1, PT}` for `D`=1.
"""
function PCurlGradBasis(::Type{PT},::Val{D},::Type{T},order::Int) where {PT,D,T}
  RaviartThomasPolyBasis{D}(PT, T, order)
end

function PCurlGradBasis(::Type{PT},::Val{1},::Type{T},order::Int) where {PT,T}
  @check T<:Real "T needs to be <:Real since represents the type of the components of the vector value"

  V = VectorValue{1,T}
  CartProdPolyBasis(PT, Val(1), V, order+1)
end
