"""
    CompWiseTensorPolyBasis{D,V,PT,L} <: PolynomialBasis{D,V,PT}

"Polynomial basis of component wise tensor product polynomial spaces"

Polynomial basis for a `D`-multivariate `V`-valued polynomial space:

`V`(𝕊¹, 𝕊², ..., 𝕊ᴸ)

with `L`>1, where the scalar `D`-multivariate spaces 𝕊ˡ (for 1 ≤ l ≤ `L`) of each
(independent) component of `V` is the tensor product of 1D ℙ spaces of order
α(l,n) for 1 ≤ n ≤ `D`, that is:

𝕊¹ = ℙα(1,1) ⊗ … ⊗ ℙα(1,`D`)\\
⋮\\
𝕊ˡ =     ⊗ₙ  ℙα(l,n)\\
⋮\\
𝕊ᴸ = ℙα(`L`,1) ⊗ … ⊗ ℙα(`L`,`D`)

The `L`×`D` matrix of orders α is given in the constructor, and `K` is the
maximum of α. Any 1D polynomial family `PT<:Polynomial` is usable.
"""
struct CompWiseTensorPolyBasis{D,V,PT,L} <: PolynomialBasis{D,V,PT}
  max_order::Int
  orders::SMatrix{L,D,Int}

  function CompWiseTensorPolyBasis{D}(
    ::Type{PT}, ::Type{V}, orders::SMatrix{L,D,Int}) where {D,PT<:Polynomial,V,L}

    msg1 = "The orders matrix rows number must match the number of independent components of V"
    @check L == num_indep_components(V) msg1
    msg2 = "The Component Wise construction is useless for one component, use CartProdPolyBasis instead"
    @check L > 1 msg2
    @check D > 0
    @check isconcretetype(PT) "PT needs to be a concrete <:Polynomial type"
    K = maximum(orders)

    new{D,V,PT,L}(K,orders)
  end
end

Base.size(a::CompWiseTensorPolyBasis) = ( sum(prod.(eachrow(a.orders .+ 1))), )
get_order(b::CompWiseTensorPolyBasis) = b.max_order

"""
    get_comp_terms(f::CompWiseTensorPolyBasis{D,V})

Return a tuple (terms\\_1, ..., terms\\_l, ..., terms\\_L) containing, for each
component of V, the Cartesian indices iterator over the terms that define 𝕊ˡ,
that is all elements of ⟦1,`o`(l,1)+1⟧ × ⟦1,`o`(l,2)+1⟧ × … × ⟦1,`o`(l,D)+1⟧.

E.g., if `orders=[ 0 1; 1 0]`, then the `comp_terms` are
`( CartesianIndices{2}((1,2)), CartesianIndices{2}((2,1)) )`.
"""
function get_comp_terms(f::CompWiseTensorPolyBasis{D,V,PT,L}) where {D,V,PT,L}
  _terms(l) = CartesianIndices( Tuple(f.orders[l,:] .+ 1) )
  comp_terms = ntuple(l -> _terms(l), Val(L))
  comp_terms::NTuple{L,CartesianIndices{D}}
end


#################################
# nD evaluations implementation #
#################################

function _evaluate_nd!(
  b::CompWiseTensorPolyBasis{D,V,PT,L}, x,
  r::AbstractMatrix{V}, i,
  c::AbstractMatrix{T}) where {D,V,PT,L,T}

  K = get_order(b)
  #orders = b.orders
  comp_terms = get_comp_terms(b)

  Kd = Val(K)
  for d in 1:D
    # The optimization below of fine tuning Kd is a bottlneck if not put in a
    # function due to runtime dispatch and creation of Val(Kd)
    #  # for each coordinate d, the order at which the basis should be evaluated is
    #  # the maximum d-order for any component l
    #  Kd = Val(maximum(orders[:,d]))
    _evaluate_1d!(PT,Kd,c,x,d)
  end

  m = zero(Mutable(V))
  k = 1

  for (l,terms) in enumerate(comp_terms)
    for ci in terms

      s = one(T)
      @inbounds for d in 1:D
        s *= c[d,ci[d]]
      end

      k = _comp_wize_set_value!(r,i,s,k,l)
    end
  end
end

"""
    _comp_wize_set_value!(r::AbstractMatrix{V},i,s::T,k,l)

```
r[i,k]   = V(0, ..., 0, s, 0, ..., 0); return k+1
```

where `s` is at position `l` in `V<:MultiValue`.
"""
function _comp_wize_set_value!(r::AbstractMatrix{V},i,s::T,k,l) where {V,T}
  z = zero(T)
  ncomp = num_indep_components(V)
  r[i,k] = ntuple(i -> ifelse(i == l, s, z),Val(ncomp))
  return k + 1
end

function _gradient_nd!(
  b::CompWiseTensorPolyBasis{D,V,PT,L}, x,
  r::AbstractMatrix{G}, i,
  c::AbstractMatrix{T},
  g::AbstractMatrix{T},
  s::MVector{D,T}) where {D,V,PT,L,G,T}

  K = get_order(b)
  #orders = b.orders
  comp_terms = get_comp_terms(b)

  Kd = Val(K)
  for d in 1:D
    _derivatives_1d!(PT,Kd,(c,g),x,d)
  end

  k = 1

  for (l,terms) in enumerate(comp_terms)
    for ci in terms

      for i in eachindex(s)
        s[i] = one(T)
      end

      for q in 1:D
        for d in 1:D
          if d != q
            @inbounds s[q] *= c[d,ci[d]]
          else
            @inbounds s[q] *= g[d,ci[d]]
          end
        end
      end

      k = _comp_wize_set_derivative!(r,i,s,k,Val(l),V)
    end
  end
end

"""
    _comp_wize_set_derivative!(r::AbstractMatrix{G},i,s,k,::Val{l},::Type{V})

```
z = zero(s)
r[i,k]     = G(z…, ..., z…, s…, z…, ..., z…) = (Dbᵏ)(xi)
return k+1
```

where `s…` is the `l`ᵗʰ set of components. This is the gradient or hessian of
the `k`ᵗʰ basis polynomial, whose nonzero component in `V` is the `l`ᵗʰ.
"""
@generated function _comp_wize_set_derivative!(
  r::AbstractMatrix{G},i,s,k,::Val{l},::Type{V}) where {G,l,V}

  N_val_dims = length(size(V))
  s_size = size(G)[1:end-N_val_dims]

  body = "T = eltype(s); z = zero(T);"
  m = Array{String}(undef, size(G))
  m .= "z"

  for ci in CartesianIndices(s_size)
    m[ci,l] = "(@inbounds s[$ci])"
  end
  body *= "@inbounds r[i,k] = ($(join(tuple(m...), ", ")));"

  body = Meta.parse(string("begin ",body," end"))
  return Expr(:block, body ,:(return k+1))
end

# See _cartprod_set_derivative!(r::AbstractMatrix{G},i,s,k,::Type{V}) where {G,V<:AbstractSymTensorValue{D}} where D
@generated function _comp_wize_set_derivative!(
  r::AbstractMatrix{G},i,s,k,::Type{V}) where {G,V<:AbstractSymTensorValue{D}} where D

  @notimplemented
end

function _hessian_nd!(
  b::CompWiseTensorPolyBasis{D,V,PT,L}, x,
  r::AbstractMatrix{H}, i,
  c::AbstractMatrix{T},
  g::AbstractMatrix{T},
  h::AbstractMatrix{T},
  s::MMatrix{D,D,T}) where {D,V,PT,L,H,T}

  K = get_order(b)
  #orders = b.orders
  comp_terms = get_comp_terms(b)

  Kd = Val(K)
  for d in 1:D
    _derivatives_1d!(PT,Kd,(c,g,h),x,d)
  end

  k = 1

  for (l,terms) in enumerate(comp_terms)
    for ci in terms

      for i in eachindex(s)
        s[i] = one(T)
      end

      for r in 1:D
        for q in 1:D
          for d in 1:D
            if d != q && d != r
              @inbounds s[r,q] *= c[d,ci[d]]
            elseif d == q && d ==r
              @inbounds s[r,q] *= h[d,ci[d]]
            else
              @inbounds s[r,q] *= g[d,ci[d]]
            end
          end
        end
      end

      k = _comp_wize_set_derivative!(r,i,s,k,Val(l),V)
    end
  end
end


################################
# Basis for Nedelec on D-cubes #
################################

"""
    QGradBasis(::Type{PT}, ::Val{D}, ::Type{T}, order::Int) :: PolynomialBasis

Return a basis of

ℕ𝔻ᴰₙ(□) = (ℚᴰₙ)ᴰ ⊕ x × (ℚᴰₙ \\ ℚᴰₙ₋₁)ᴰ

with n=`order`, the polynomial space for Nedelec elements on `D`-dimensional
cubes with scalar type `T`.

The `order`=n argument has the following meaning: the curl of the  functions in
this basis is in (ℚᴰₙ)ᴰ.

`PT<:Polynomial` is the choice of the family of the scalar 1D basis polynomials.

# Example:

```jldoctest
# a basis for Nedelec on hexahedra with divergence in ℚ₂
b = QGradBasis(Monomial, Val(3), Float64, 2)
```

For more details, see [`CompWiseTensorPolyBasis`](@ref), as `QGradBasis` returns
an instance of\\
`CompWiseTensorPolyBasis{D, VectorValue{D,T}, order+1, PT}` for `D`>1, or\\
`CartProdPolyBasis{1, VectorValue{1,T}, order+1, PT}` for `D`=1.
"""
function QGradBasis(::Type{PT},::Val{D},::Type{T},order::Int) where {PT,D,T}
  @check T<:Real "T needs to be <:Real since represents the type of the components of the vector value"

  V = VectorValue{D,T}
  m = [ order + (i==j ? 0 : 1) for i in 1:D, j in 1:D ]
  orders = SMatrix{D,D,Int}(m)
  CompWiseTensorPolyBasis{D}(PT, V, orders)
end

function QGradBasis(::Type{PT},::Val{1},::Type{T},order::Int) where {PT,T}
  @check T<:Real "T needs to be <:Real since represents the type of the components of the vector value"

  V = VectorValue{1,T}
  CartProdPolyBasis(PT, Val(1), V, order+1)
end


#######################################
# Basis for Raviart-Thomas on D-cubes #
#######################################

"""
    QCurlGradBasis(::Type{PT}, ::Val{D}, ::Type{T}, order::Int) :: PolynomialBasis

Return a basis of

ℝ𝕋ᴰₙ(□) = (ℚᴰₙ)ᴰ ⊕ x (ℚᴰₙ \\ ℚᴰₙ₋₁)

with n=`order`, the polynomial space for Raviart-Thomas elements on
`D`-dimensional cubes with scalar type `T`.

The `order`=n argument has the following meaning: the divergence of the functions
in this basis is in ℚᴰₙ.

`PT<:Polynomial` is the choice of the family of the scalar 1D basis polynomials.

# Example:

```jldoctest
# a basis for Raviart-Thomas on rectangles with divergence in ℚ₃
b = QCurlGradBasis(Bernstein, Val(2), Float64, 3)
```

For more details, see [`CompWiseTensorPolyBasis`](@ref), as `QCurlGradBasis`
returns an instance of\\
`CompWiseTensorPolyBasis{D, VectorValue{D,T}, order+1, PT}` for `D`>1, or\\
`CartProdPolyBasis{1, VectorValue{1,T}, order+1, PT}` for `D`=1.
"""
function QCurlGradBasis(::Type{PT},::Val{D},::Type{T},order::Int) where {PT,D,T}
  @check T<:Real "T needs to be <:Real since represents the type of the components of the vector value"

  V = VectorValue{D,T}
  m = [ order + (i==j ? 1 : 0) for i in 1:D, j in 1:D ]
  orders = SMatrix{D,D,Int}(m)
  CompWiseTensorPolyBasis{D}(PT, V, orders)
end

function QCurlGradBasis(::Type{PT},::Val{1},::Type{T},order::Int) where {PT,T}
  @check T<:Real "T needs to be <:Real since represents the type of the components of the vector value"

  V = VectorValue{1,T}
  CartProdPolyBasis(PT, Val(1), V, order+1)
end
