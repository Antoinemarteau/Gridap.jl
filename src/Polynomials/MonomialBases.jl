struct Monomial <: Field end

testvalue(::Type{Monomial}) = Monomial()
function testvalue(::Type{<:AbstractVector{Monomial}})
  @notimplemented
end

"""
    struct MonomialBasis{D,T} <: AbstractVector{Monomial}

Type representing a basis of multivariate scalar-valued, vector-valued, or
tensor-valued, iso- or aniso-tropic monomials. The fields
of this `struct` are not public.
This type fully implements the [`Field`](@ref) interface, with up to second order
derivatives.
"""
struct MonomialBasis{D,T} <: AbstractVector{Monomial}
  orders::NTuple{D,Int}
  terms::Vector{CartesianIndex{D}}
  function MonomialBasis{D}(
    ::Type{T}, orders::NTuple{D,Int}, terms::Vector{CartesianIndex{D}}) where {D,T}
    new{D,T}(orders,terms)
  end
end

Base.size(a::MonomialBasis{D,T}) where {D,T} = (length(a.terms)*num_indep_components(T),)
# @santiagobadia : Not sure we want to create the monomial machinery
Base.getindex(a::MonomialBasis,i::Integer) = Monomial()
Base.IndexStyle(::MonomialBasis) = IndexLinear()

function testvalue(::Type{MonomialBasis{D,T}}) where {D,T}
  MonomialBasis{D}(T,tfill(0,Val{D}()),CartesianIndex{D}[])
end

"""
    MonomialBasis{D}(::Type{T}, orders::Tuple [, filter::Function]) where {D,T}

This version of the constructor allows to pass a tuple `orders` containing the
polynomial order to be used in each of the `D` dimensions in order to  construct
an anisotropic tensor-product space.
"""
function MonomialBasis{D}(
  ::Type{T}, orders::NTuple{D,Int}, filter::Function=_q_filter) where {D,T}

  terms = _define_terms(filter, orders)
  MonomialBasis{D}(T,orders,terms)
end

"""
    MonomialBasis{D}(::Type{T}, order::Int [, filter::Function]) where {D,T}

Returns an instance of `MonomialBasis` representing a multivariate polynomial basis
in `D` dimensions, of polynomial degree `order`, whose value is represented by the type `T`.
The type `T` is typically `<:Number`, e.g., `Float64` for scalar-valued functions and `VectorValue{D,Float64}`
for vector-valued ones.

# Filter function

The `filter` function is used to select which terms of the tensor product space
of order `order` in `D` dimensions are to be used. If the filter is not provided, the full tensor-product
space is used by default leading to a multivariate polynomial space of type Q.
The signature of the filter function is

    (e,order) -> Bool

where `e` is a tuple of `D` integers containing the exponents of a multivariate monomial. The following filters
are used to select well known polynomial spaces

- Q space: `(e,order) -> true`
- P space: `(e,order) -> sum(e) <= order`
- "Serendipity" space: `(e,order) -> sum( [ i for i in e if i>1 ] ) <= order`

"""
function MonomialBasis{D}(
  ::Type{T}, order::Int, filter::Function=_q_filter) where {D,T}

  orders = tfill(order,Val{D}())
  MonomialBasis{D}(T,orders,filter)
end

# API

"""
    get_exponents(b::MonomialBasis)

Get a vector of tuples with the exponents of all the terms in the
monomial basis.

# Examples

```jldoctest
using Gridap.Polynomials

b = MonomialBasis{2}(Float64,2)

exponents = get_exponents(b)

println(exponents)

# output
Tuple{Int,Int}[(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
```
"""
function get_exponents(b::MonomialBasis)
  indexbase = 1
  [Tuple(t) .- indexbase for t in b.terms]
end

"""
    get_order(b::MonomialBasis)
"""
function get_order(b::MonomialBasis)
  maximum(b.orders)
end

"""
    get_orders(b::MonomialBasis)
"""
function get_orders(b::MonomialBasis)
  b.orders
end

"""
"""
return_type(::MonomialBasis{D,T}) where {D,T} = T

# Field implementation
function return_cache(f::MonomialBasis{D,T},x::AbstractVector{<:Point}) where {D,T}
  @check D == length(eltype(x)) "Incorrect number of point components"
  zT = zero(T)
  zxi = zero(eltype(eltype(x)))
  Tp = typeof( zT*zxi*zxi + zT*zxi*zxi  )
  np = length(x)
  ndof = length(f)
  n = 1 + _maximum(f.orders)
  r = CachedArray(zeros(Tp,(np,ndof)))
  v = CachedArray(zeros(Tp,(ndof,)))
  c = CachedArray(zeros(eltype(Tp),(D,n)))
  (r, v, c)
end

function evaluate!(cache,f::MonomialBasis{D,T},x::AbstractVector{<:Point}) where {D,T}
  r, v, c = cache
  np = length(x)
  ndof = length(f)
  n = 1 + _maximum(f.orders)
  setsize!(r,(np,ndof))
  setsize!(v,(ndof,))
  setsize!(c,(D,n))
  for i in 1:np
    @inbounds xi = x[i]
    _evaluate_nd!(v,xi,f.orders,f.terms,c)
    for j in 1:ndof
      @inbounds r[i,j] = v[j]
    end
  end
  r.array
end

function _return_cache(
  fg::FieldGradientArray{1,MonomialBasis{D,V}},
  x::AbstractVector{<:Point},
  ::Type{T},
  TisbitsType::Val{true}) where {D,V,T}

  f = fg.fa
  @check D == length(eltype(x)) "Incorrect number of point components"
  np = length(x)
  ndof = length(f)
  n = 1 + _maximum(f.orders)
  r = CachedArray(zeros(T,(np,ndof)))
  v = CachedArray(zeros(T,(ndof,)))
  c = CachedArray(zeros(eltype(T),(D,n)))
  g = CachedArray(zeros(eltype(T),(D,n)))
  (r,v,c,g)
end

function _return_cache(
  fg::FieldGradientArray{1,MonomialBasis{D,V}},
  x::AbstractVector{<:Point},
  ::Type{T},
  TisbitsType::Val{false}) where {D,V,T}

  cache = _return_cache(fg,x,T,Val{true}())
  z = CachedArray(zeros(eltype(T),D))
  (cache...,z)
end

function return_cache(
  fg::FieldGradientArray{1,MonomialBasis{D,V}},
  x::AbstractVector{<:Point}) where {D,V}

  xi = testitem(x)
  T = gradient_type(V,xi)
  TisbitsType = Val(isbitstype(T))
  _return_cache(fg,x,T,TisbitsType)
end

function _evaluate!(
  cache,
  fg::FieldGradientArray{1,MonomialBasis{D,T}},
  x::AbstractVector{Tp},
  TisbitsType::Val{true}) where {D,T,Tp<:Point}

  f = fg.fa
  r, v, c, g = cache
  Tz = VectorValue{D,eltype(gradient_type(T,zero(Tp)))}
  z = zero(Mutable(Tz))
  np = length(x)
  ndof = length(f)
  n = 1 + _maximum(f.orders)
  setsize!(r,(np,ndof))
  setsize!(v,(ndof,))
  setsize!(c,(D,n))
  setsize!(g,(D,n))
  for i in 1:np
    @inbounds xi = x[i]
    _gradient_nd!(v,xi,f.orders,f.terms,c,g,z,T)
    for j in 1:ndof
      @inbounds r[i,j] = v[j]
    end
  end
  r.array
end

function _evaluate!(
  cache,
  fg::FieldGradientArray{1,MonomialBasis{D,T}},
  x::AbstractVector{<:Point},
  TisbitsType::Val{false}) where {D,T}

  f = fg.fa
  r, v, c, g, z = cache
  np = length(x)
  ndof = length(f)
  n = 1 + _maximum(f.orders)
  setsize!(r,(np,ndof))
  setsize!(v,(ndof,))
  setsize!(c,(D,n))
  setsize!(g,(D,n))
  for i in 1:np
    @inbounds xi = x[i]
    _gradient_nd!(v,xi,f.orders,f.terms,c,g,z,T)
    for j in 1:ndof
      @inbounds r[i,j] = v[j]
    end
  end
  r.array
end

function evaluate!(
  cache,
  fg::FieldGradientArray{1,MonomialBasis{D,T}},
  x::AbstractVector{<:Point}) where {D,T}

  r, v, c, g = cache
  TisbitsType = Val(isbitstype(eltype(c)))
  _evaluate!(cache,fg,x,TisbitsType)
end

function return_cache(
  fg::FieldGradientArray{2,MonomialBasis{D,V}},
  x::AbstractVector{<:Point}) where {D,V}

  f = fg.fa
  @check D == length(eltype(x)) "Incorrect number of point components"
  np = length(x)
  ndof = length(f)
  xi = testitem(x)
  T = gradient_type(gradient_type(V,xi),xi)
  n = 1 + _maximum(f.orders)
  r = CachedArray(zeros(T,(np,ndof)))
  v = CachedArray(zeros(T,(ndof,)))
  c = CachedArray(zeros(eltype(T),(D,n)))
  g = CachedArray(zeros(eltype(T),(D,n)))
  h = CachedArray(zeros(eltype(T),(D,n)))
  (r, v, c, g, h)
end

function evaluate!(
  cache,
  fg::FieldGradientArray{2,MonomialBasis{D,T}},
  x::AbstractVector{<:Point}) where {D,T}

  f = fg.fa
  r, v, c, g, h = cache
  np = length(x)
  ndof = length(f)
  n = 1 + _maximum(f.orders)
  setsize!(r,(np,ndof))
  setsize!(v,(ndof,))
  setsize!(c,(D,n))
  setsize!(g,(D,n))
  setsize!(h,(D,n))
  for i in 1:np
    @inbounds xi = x[i]
    _hessian_nd!(v,xi,f.orders,f.terms,c,g,h,T)
    for j in 1:ndof
      @inbounds r[i,j] = v[j]
    end
  end
  r.array
end

# Optimizing evaluation at a single point

function return_cache(f::AbstractVector{Monomial},x::Point)
  xs = [x]
  cf = return_cache(f,xs)
  v = evaluate!(cf,f,xs)
  r = CachedArray(zeros(eltype(v),(size(v,2),)))
  r, cf, xs
end

function evaluate!(cache,f::AbstractVector{Monomial},x::Point)
  r, cf, xs = cache
  xs[1] = x
  v = evaluate!(cf,f,xs)
  ndof = size(v,2)
  setsize!(r,(ndof,))
  a = r.array
  copyto!(a,v)
  a
end

function return_cache(
  f::FieldGradientArray{N,<:AbstractVector{Monomial}}, x::Point) where {N}
  xs = [x]
  cf = return_cache(f,xs)
  v = evaluate!(cf,f,xs)
  r = CachedArray(zeros(eltype(v),(size(v,2),)))
  r, cf, xs
end

function evaluate!(
  cache, f::FieldGradientArray{N,<:AbstractVector{Monomial}}, x::Point) where {N}
  r, cf, xs = cache
  xs[1] = x
  v = evaluate!(cf,f,xs)
  ndof = size(v,2)
  setsize!(r,(ndof,))
  a = r.array
  copyto!(a,v)
  a
end

# Helpers

_q_filter(e,o) = true

function _define_terms(filter,orders)
  t = orders .+ 1
  g = (0 .* orders) .+ 1
  cis = CartesianIndices(t)
  co = CartesianIndex(g)
  maxorder = _maximum(orders)
  [ ci for ci in cis if filter(Int[Tuple(ci-co)...],maxorder) ]
end

function _evaluate_1d!(v::AbstractMatrix{T},x,order,d) where T
  n = order + 1
  z = one(T)
  @inbounds v[d,1] = z
  @inbounds xd = x[d]
  xn = xd
  for i in 2:n
    @inbounds v[d,i] = xn
    xn *= xd
  end
end

function _gradient_1d!(v::AbstractMatrix{T},x,order,d) where T
  n = order + 1
  z = zero(T)
  @inbounds v[d,1] = z
  @inbounds xd = x[d]
  xn = one(T)
  for i in 2:n
    @inbounds v[d,i] = (i-1)*xn
    xn *= xd
  end
end

function _hessian_1d!(v::AbstractMatrix{T},x,order,d) where T
  n = order + 1
  z = zero(T)
  @inbounds v[d,1] = z
  if n>1
    @inbounds v[d,2] = z
  end
  @inbounds xd = x[d]
  xn = one(T)
  for i in 3:n
    @inbounds v[d,i] = (i-1)*(i-2)*xn
    xn *= xd
  end
end

function _evaluate_nd!(
  v::AbstractVector{V},
  x,
  orders,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T}) where {V,T,D}

  dim = D
  for d in 1:dim
    _evaluate_1d!(c,x,orders[d],d)
  end

  o = one(T)
  k = 1

  for ci in terms

    s = o
    for d in 1:dim
      @inbounds s *= c[d,ci[d]]
    end

    k = _set_value!(v,s,k)

  end

end

function _set_value!(v::AbstractVector{V},s::T,k) where {V,T}
  ncomp = num_indep_components(V)
  z = zero(T)
  @inbounds for j in 1:ncomp
    v[k] = ntuple(i -> ifelse(i == j, s, z),Val(ncomp))
    k += 1
  end
  k
end

function _set_value!(v::AbstractVector{<:Real},s,k)
    @inbounds v[k] = s
    k+1
end

function _gradient_nd!(
  v::AbstractVector{G},
  x,
  orders,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T},
  g::AbstractMatrix{T},
  z::AbstractVector{T},
  ::Type{V}) where {G,T,D,V}

  dim = D
  for d in 1:dim
    _evaluate_1d!(c,x,orders[d],d)
    _gradient_1d!(g,x,orders[d],d)
  end

  o = one(T)
  k = 1

  for ci in terms

    s = z
    for i in eachindex(s)
      @inbounds s[i] = o
    end
    for q in 1:dim
      for d in 1:dim
        if d != q
          @inbounds s[q] *= c[d,ci[d]]
        else
          @inbounds s[q] *= g[d,ci[d]]
        end
      end
    end

    k = _set_gradient!(v,s,k,V)

  end

end

function _set_gradient!(
  v::AbstractVector{G},s,k,::Type{<:Real}) where G

  @inbounds v[k] = s
  k+1
end

@generated function  _set_gradient!(
  v::AbstractVector{G},s,k,::Type{V}) where {V,G}
  # Git blame me for readable non-generated version

  w = zero(V)
  m = Array{String}(undef, size(G))
  N_val_dims = length(size(V))
  s_size = size(G)[1:end-N_val_dims]

  body = "T = eltype(s); z = zero(T);"
  for ci in CartesianIndices(s_size)
    id = join(Tuple(ci))
    body *= "@inbounds s$id = s[$ci];"
  end

  for j in CartesianIndices(w)
    for i in CartesianIndices(m)
      m[i] = "z"
    end
    for ci in CartesianIndices(s_size)
      id = join(Tuple(ci))
      m[ci,j] = "s$id"
    end
    body *= "@inbounds v[k] = ($(join(tuple(m...), ", ")));"
    body *= "k = k + 1;"
  end

  body = Meta.parse(string("begin ",body," end"))
  return Expr(:block, body ,:(return k))
end

# Specialization for SymTensorValue and SymTracelessTensorValue,
# necessary as long as outer(Point, V<:AbstractSymTensorValue)::G does not
# return a tensor type that implements the appropriate symmetries of the
# gradient (and hessian)
@generated function _set_gradient!(
  v::AbstractVector{G},s,k,::Type{V}) where {V<:AbstractSymTensorValue{D},G} where D
  # Git blame me for readable non-generated version
  
  T = eltype(s)
  m = Array{String}(undef, size(G))
  s_length = size(G)[1]

  is_traceless = V <: SymTracelessTensorValue
  skip_last_diagval = is_traceless ? 1 : 0    # Skid V_DD if traceless

  body = "z = $(zero(T));"
  for i in 1:s_length 
    body *= "@inbounds s$i = s[$i];"
  end
  
  for c in 1:(D-skip_last_diagval) # Go over cols
    for r in c:D                   # Go over lower triangle, current col
      for i in eachindex(m)
        m[i] = "z"
      end
      for i in 1:s_length # indices of the Vector s
        m[i,r,c] = "s$i"
        if (r!=c)
          m[i,c,r] = "s$i"
        elseif is_traceless # V_rr contributes negatively to V_DD (tracelessness)
          m[i,D,D] = "-s$i"
        end
      end
      body *= "@inbounds v[k] = ($(join(tuple(m...), ", ")));"
      body *= "k = k + 1;"
    end
  end

  body = Meta.parse(string("begin ",body," end"))
  return Expr(:block, body ,:(return k))
end

function _hessian_nd!(
  v::AbstractVector{G},
  x,
  orders,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T},
  g::AbstractMatrix{T},
  h::AbstractMatrix{T},
  ::Type{V}) where {G,T,D,V}

  dim = D
  for d in 1:dim
    _evaluate_1d!(c,x,orders[d],d)
    _gradient_1d!(g,x,orders[d],d)
    _hessian_1d!(h,x,orders[d],d)
  end

  z = zero(Mutable(TensorValue{D,D,T}))
  o = one(T)
  k = 1

  for ci in terms

    s = z
    for i in eachindex(s)
      @inbounds s[i] = o
    end
    for r in 1:dim
      for q in 1:dim
        for d in 1:dim
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

    k = _set_gradient!(v,s,k,V)

  end

end

_maximum(orders::Tuple{}) = 0
_maximum(orders) = maximum(orders)
