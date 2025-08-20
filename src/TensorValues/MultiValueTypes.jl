###############################################################
# MultiValue Type
###############################################################

"""
    MultiValue{T,L} <: Number

Abstract type representing a multi-dimensional number value.
- `T` is the type of the scalar components, should be subtype of `Number`,
- `L` is the number of components (scalars) of type `T` stored internally.

Concrete instances are immutable. `MultiValue`s implement the `AbstractArray`
interface except of `setindex!` and `similar`.
"""
abstract type MultiValue{T,L} <: Number end

@inline Base.Tuple(arg::MultiValue) = arg.data

# Custom type printing

function show(io::IO,v::MultiValue)
  print(io,v.data)
end

function show(io::IO,::MIME"text/plain",v:: MultiValue)
  print(io,typeof(v))
  print(io,v.data)
end

"""
    abstract type CalculusStyle

Trait that signals if a quantity or field belongs to the formalism of traditional
vector (differential) calculus (trait value [`VectorCalculus()`](@ref VectorCalculus)),
or in the formalism of differential geometry calculus (trait value
[`DiffGeoCalculus()`](@ref DiffGeoCalculus)).
"""
abstract type CalculusStyle end
struct VectorCalculus <: CalculusStyle end
struct DiffGeoCalculus <: CalculusStyle end

CalculusStyle(::Type) = VectorCalculus()


###############################################################
# Introspection
###############################################################

eltype(::Type{<:MultiValue{T}}) where T = T
eltype(::MultiValue{T}) where T = T

length(::Type{V}) where V<:MultiValue = prod(size(V))
length(a::MultiValue)  = length(typeof(a))

size(a::MultiValue) = size(typeof(a))

"""
    num_components(::Type{<:Number})
    num_components(a::Number)

Total number of components of a `Number` or `MultiValue`, that is 1 for scalars
and the product of the size dimensions for a `MultiValue`. This is the same as `length`.
"""
num_components(a::Number) = num_components(typeof(a))
num_components(::Type{<:Number}) = 1
num_components(T::Type{<:MultiValue}) = @unreachable "$T type is too abstract to count its components, the shape (firt parametric argument) is neede."

"""
    num_indep_components(::Type{<:Number})
    num_indep_components(a::Number)

Number of independent components of a `Number`, that is `num_components`
minus the number of components determined from others by symmetries or constraints.

For example, a `TensorValue{3,3}` has 9 independent components, a `SymTensorValue{3}`
has 6 and a `SymTracelessTensorValue{3}` has 5. But they all have 9 (non independent) components.
"""
num_indep_components(::Type{T}) where T<:Number = num_components(T)
num_indep_components(::T) where T<:Number = num_indep_components(T)

"""
!!! warning
    Deprecated in favor on [`num_components`](@ref).
"""
function n_components end
@deprecate n_components num_components


#######################################################################
# Other constructors and conversions implemented for more generic types
#######################################################################

zero(::Type{V}) where V<:MultiValue{T} where T = V(tfill(zero(T),Val(num_indep_components(V))))
zero(a::MultiValue) = zero(typeof(a))

one(::Type{V}) where V<:MultiValue = @unreachable "`one` not defined for $V"
one(a::MultiValue) = one(typeof(a))

function rand(rng::AbstractRNG,::Random.SamplerType{V}) where V<:MultiValue{T} where {T}
  Li = num_indep_components(V)
  vrand = rand(rng, SVector{Li,T})
  V(Tuple(vrand))
end

"""
    change_eltype(m::Number,::Type{T2})
    change_eltype(M::Type{<:Number},::Type{T2})

For multivalues, returns `M` or `typeof(m)` but with the component type (`MultiValue`'s parametric type `T`) changed to `T2`.

For scalars (or any non MultiValue number), `change_eltype` returns T2.
"""
change_eltype(::Type{<:Number}, T::Type) = T
change_eltype(::T,::Type{T2}) where {T<:Number,T2} = change_eltype(T,T2)

change_eltype(a::MultiValue,::Type{T2}) where T2 = change_eltype(typeof(a),T2)

get_array(a::MultiValue{T}) where T = convert(SArray{Tuple{size(a)...},T},a)

"""
    Mutable(T::Type{<:MultiValue}) -> ::Type{<:MArray}
    Mutable(a::MultiValue)

Return the concrete mutable `MArray` type (defined by `StaticArrays.jl`) corresponding
to the `MultiValue` type T or array size and type of `a`.

See also [`mutable`](@ref).
"""
Mutable(::Type{<:MultiValue}) = @abstractmethod
Mutable(a::MultiValue) = Mutable(typeof(a))

"""
    mutable(a::MultiValue)

Converts `a` into a mutable array of type `MArray` defined by `StaticArrays.jl`.

See also [`Mutable`](@ref).
"""
mutable(a::MultiValue) = @abstractmethod

###############################################################
# Conversions
###############################################################

convert(::Type{V}, arg::Tuple) where V<:MultiValue = V(arg)

# Inverse conversion
convert(::Type{<:NTuple{L,T}}, arg::MultiValue) where {L,T} = NTuple{L,T}(Tuple(arg))


###############################################################
# Indexing independant components
###############################################################

# This should probably not be exported, as (accessing) the data field of
# MultiValue is not a public api
"""
Previously used to transform Cartesian indices to linear indices that index `MultiValue`'s private internal storage.

!!! warning
    Deprecated, not all components of all tensors are stored anymore, so this
    index is ill defined. Use `getindex` or [`indep_comp_getindex`](@ref)
    instead of this.
"""
function data_index(::Type{<:MultiValue},i...)
  @abstractmethod
end
@deprecate data_index getindex

# The order of export of components is that of their position in the .data
# field, but the actual method "choosing" the export order is
# Gridap.Visualization._prepare_data(::Multivalue).
"""
    indep_comp_getindex(a::Number,i)

Get the `i`th independent component of `a`. It only differs from `getindex(a,i)`
when the components of `a` are interdependent, see [`num_indep_components`](@ref).
`i` should be in `1:num_indep_components(a)`.
"""
@propagate_inbounds function indep_comp_getindex(a::Number,i)
  @boundscheck @check 1 <= i <= num_indep_components(Number)
  @inbounds a[i]
end

@propagate_inbounds function indep_comp_getindex(a::V,i) where {V<:MultiValue}
  @boundscheck @check 1 <= i <= num_indep_components(V)
  @inbounds _get_data(a,i)
end

# abstraction of Multivalue data access in case subtypes of MultiValue don't
# store its data in a data field
@propagate_inbounds function _get_data(a::MultiValue,i)
  a.data[i]
end

"""
    indep_components_names(::MultiValue)

Return an array of strings containing the component labels in the order they
are exported in VTK file.

If all dimensions of the tensor shape S are smaller than 3, the components
are named with letters "X","Y" and "Z" similarly to the automatic naming
of Paraview. Else, if max(S)>3, they are labeled by integers starting from "1".
"""
function indep_components_names(::Type{<:MultiValue{T,L}}) where {T,L}
  return ["$i" for i in 1:L]
end

"""
    component_basis(V::Type{<:MultiValue}) -> V[ Vᵢ... ]
    component_basis(T::Type{<:Real}) -> [ one(T) ]
    component_basis(a::T<:Number)

Given a `Number` type `V` with N independent components, return a vector of
N values ``\\{ Vᵢ=V(eᵢ) \\}_i`` forming the component basis of ``\\{ u : u\\text{ isa }V\\}``
(where ``\\{eᵢ\\}_i`` is the Cartesian basis of (`eltype(V)`)ᴺ).

The `Vᵢ` verify the property that for any `u::V`,

    u = sum( indep_comp_getindex(u,i)*Vᵢ for i ∈ 1:N )
"""
component_basis(a::Number) = component_basis(typeof(a))
component_basis(T::Type{<:Number}) = [ one(T) ]
function component_basis(V::Type{<:MultiValue})
  T = eltype(V)
  Li = num_indep_components(V)
  return V[ ntuple(i -> T(i == j), Li) for j in 1:Li ]
end

"""
    representatives_of_componentbasis_dual(V::Type{<:MultiValue}) -> V[ Vᵢ... ]
    representatives_of_componentbasis_dual(T::Type{<:Real}) -> [ one(T) ]
    representatives_of_componentbasis_dual(a::V<:Number)

Given a `Number` type `V` with N independent components, return a vector of
N values ``\\{ Vᵢ \\}_i`` that define the form basis ``\\{ Lⁱ := (u -> u ⊙ Vᵢ) \\}_i`` that
is the dual of the component basis ``\\{ V(eᵢ) \\}_i`` (where ``\\{eᵢ\\}_i`` is the
Cartesian basis of (`eltype(V)`)ᴺ).

The `Lⁱ`/`Vᵢ` verify the property that for any `u::V`,

    u = V( [ Lⁱ(u) for i ∈ 1:N ]... )
      = V( [ u⊙Vᵢ  for i ∈ 1:N ]... )

Rq, when `V` has dependent components, the `Vᵢ` are NOT a component basis because
`Vᵢ ≠ V(eᵢ)` and

    u ≠ sum( indep_comp_getindex(u,i)*Vᵢ for i ∈ 1:N )
"""
representatives_of_componentbasis_dual(a::Number) = representatives_of_componentbasis_dual(typeof(a))
representatives_of_componentbasis_dual(T::Type{<:Real}) = [ one(T) ]
function representatives_of_componentbasis_dual(V::Type{<:MultiValue})
  V = typeof(zero(V)) # makes V concrete for type inference
  N = num_indep_components(V)
  T = eltype(V)
  B = component_basis(V)

  M = MMatrix{N,N,T}(undef)
  for ci in CartesianIndices(M)
    M[ci] = B[ci[1]] ⊙ B[ci[2]]
  end
  Minv = inv(M)

  return V[ Tuple(Minv[i,:]) for i in 1:N ]
end

@inline function ForwardDiff.value(x::MultiValue{S,<:ForwardDiff.Dual}) where S
  VT = change_eltype(x,ForwardDiff.valtype(eltype(x)))
  data = map(ForwardDiff.value,x.data)
  return VT(data)
end


###############################################################
# ArrayMultiValue Type
###############################################################

"""
    ArrayMultiValue{S,T,N,L} <: MultiValue{T,L}

Abstract type representing a multi-dimensional number value. The parameters are
analog to that of StaticArrays.jl:
- `S` is a Tuple type holding the size of the tensor, e.g. Tuple{3} for a 3d vector or Tuple{2,4} for a 2 rows and 4 columns tensor,
- `T` is the type of the scalar components, should be subtype of `Number`,
- `N` is the order of the tensor, the length of `S`,
- `L` is the number of components stored internally.

`ArrayMultiValue`s are immutable.
"""
abstract type ArrayMultiValue{S,T,N,L} <: MultiValue{T,L} end

size(::Type{<:ArrayMultiValue{S}}) where S<:Tuple = tuple(S.parameters...)

###############################################################
# AbstractDifferentialTensorValue Type
###############################################################

"""
    AbstractDGTensorValue{D,R,S,T,L} <: MultiValue{T,L}

Abstract type for values of type-(`R`,`S`) tensors
- `D` is the spatial dimension
- `R` is contravariant index
- `S` is covariant index
- `T` is the type of the scalar components, should be subtype of `Number`
- `L` is the number of components (scalars) of type `T` stored internally.

Let A be a differential geometry tensor, that is a multilinear map

    A : (V*)`ᴿ` × V`ˢ`  -> ℝ

where V is a `D`-dimentional vector space of basis {e¹, ..., e`ᴰ`}, instances of
`AbstractDGTensorValue` store the components of A into the canonical basis

    { eᵢ₁ ⊗ ... ⊗ eᵢᵣ ⊗ eʲ¹ ⊗ ... ⊗ eʲˢ |  il,im ∈ ⟦1,`D`⟧, 1 ≤ l ≤`R`, 1 ≤ m ≤ `S`},

then can be accessed via indexing as in `A[i1, ..., iR, j1, ..., jS]`.

Unlike intances of [`ArrayMultiValue`](@ref)s, the differential geometry tensor values
have a fixed number of components along each axes, `D`, and axes have a variance.

The concrete subtypes are [`DGTensorValue`](@ref) and [`FormValue`](@ref).

The main purpose of this type is to discriminate the tensor values between the
vector differential calculus and the differential geometry calculus with zero
runtime cost.
"""
abstract type AbstractDGTensorValue{D,R,S,T,L} <: MultiValue{T,L} end

CalculusStyle(::Type{<:AbstractDGTensorValue}) = DiffGeoCalculus()

size(::AbstractDGTensorValue{D,R,S}) where {D,R,S} = ntuple(_->D, Val(R+S))

num_components(::Type{<:ArrayMultiValue{S}}) where S = length(ArrayMultiValue{S})

# typeof(zero(...)) is for the N and L type parameters to be computed, and get
# e.g. MVector{D,T} == MMarray{Tuple{D},T,1,D} instead of MMarray{Tuple{D},T}
Mutable(::Type{<:ArrayMultiValue{S,T}}) where {S,T} = typeof(zero(MArray{S,T}))
mutable(a::ArrayMultiValue{S}) where S = MArray{S}(Tuple(get_array(a)))

convert(::Type{V}, arg::AbstractArray) where V<:ArrayMultiValue{S,T} where {S,T} = V(arg)

