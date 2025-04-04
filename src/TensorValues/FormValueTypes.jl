###############################################################
# Types
###############################################################
"""
    FormValue{D,k,T,L} <: AbstractDGTensorValue{D,0,k,T,L}

Type representing a `D`-dimensional exterior `k`-form.
Seen as an order `k` covariant tensor, it has `Dᵏ` components, but due to
antisymetry, only `L`=binomial(`D`,`k`) are independent and stored.
"""
struct FormValue{D,k,T,L} <: AbstractDGTensorValue{D,0,k,T,L}
    data::NTuple{L,T}
    function FormValue{D,k,T}(data::NTuple{L,T}) where {D,k,L,T}
        @check 0 ≤ k ≤ D "Invalid dimension or form order, expected 0 ≤ k ≤ D, got  k=$k and D=$D"
        @check L == binomial(D,k) "wrong number of values, expecting L=binomial(D,k), got  L=$L, D=$D and k=$k"
        new{D,k,T,L}(data)
    end
end


###############################################################
# Constructors (FormValue)
###############################################################

# Static binomial coef for static type conversion
@generated _binom(::Val{D},::Val{k}) where {D,k} = :( $(binomial(D,k)) )

# Empty FormValue constructor

FormValue()                   = FormValue{0,0,Int}(NTuple{0,Int}())
FormValue{0}()                = FormValue{0,0,Int}(NTuple{0,Int}())
FormValue{0,0,T}() where T    = FormValue{0,0,T}(NTuple{0,T}())
FormValue(data::NTuple{0})    = FormValue{0,0,Int}(data)
FormValue{0}(data::NTuple{0}) = FormValue{0,0,Int}(data)

# FormValue single NTuple argument constructor

FormValue{D,k}(data::NTuple{L,T})     where {D,k,L,T}     = FormValue{D,k,T}(data)
FormValue{D,k,T1}(data::NTuple{L,T2}) where {D,k,L,T1,T2} = FormValue{D,k,T1}(NTuple{L,T1}(data))

#FormValue{D}(data::NTuple{D2,T2}) where {D,D2,T2} = @unreachable
#FormValue{D1,T1}(data::NTuple{D2,T2}) where {D1,T1,D2,T2} = @unreachable

# FormValue single Tuple argument constructor

FormValue(data::Tuple)                        = FormValue{0}(promote(data...))
FormValue{0}(data::Tuple)                     = FormValue{0}(promote(data...))
FormValue{D,k}(data::Tuple)    where {D,k}    = FormValue{D,k}(promote(data...))
FormValue{D,k,T1}(data::Tuple) where {D,k,T1} = FormValue{D,k,T1}(NTuple{_binom(Val(D),Val(k)),T1}(data))

# FormValue Vararg constructor

FormValue(data::Number...)                        = FormValue{0}(data)
FormValue{0}(data::Number...)                     = FormValue{0}(data)
FormValue{D,k}(data::Number...)    where {D,k}    = FormValue{D,k}(data)
FormValue{D,k,T1}(data::Number...) where {D,k,T1} = FormValue{D,k,T1}(data)

# Fix for julia 1.0.4
FormValue{D,k}(data::T...)    where {D,k,T<:Number}    = FormValue{D,k,T}(data)

# FormValue single AbstractVector argument constructor

#FormValue(data::AbstractArray{T}) where {T}                  = (D=length(data);FormValue(NTuple{D,T}(data)))
FormValue{0}(data::AbstractArray{T}) where {T}               = FormValue{0}(NTuple{length(data),T}(data))
FormValue{D,k}(data::AbstractArray{T}) where {D,k,T}         = FormValue{D,k}(NTuple{_binom(Val(D),Val(k)),T}(data))
FormValue{D,k,T1}(data::AbstractArray{T2}) where {D,k,T1,T2} = FormValue{D,k,T1}(NTuple{_binom(Val(D),Val(k)),T1}(data))

###############################################################
# Conversions (FormValue)
###############################################################

# Direct conversion
convert(::Type{<:FormValue{D,k,T}}, arg:: AbstractArray) where {D,k,T} = FormValue{D,k,T}(NTuple{_binom(Val(D),Val(k)),T}(arg))
convert(::Type{<:FormValue{D,k,T}}, arg:: Tuple) where {D,k,T} = FormValue{D,k,T}(arg)

# Inverse conversion TODO fix
# convert(::Type{<:MMatrix{D,D,T}}, arg::SymTensorValue) where {D,T} = MMatrix{D,D,T}(_SymTensorValue_to_array(arg))
convert(::Type{<:SArray}, arg::FormValue) = @notimplemented
convert(::Type{<:MArray}, arg::FormValue) = @notimplemented
convert(::Type{<:NTuple{L,T}}, arg::FormValue) where {T,L} = NTuple{L,T}(Tuple(arg))

# Internal conversion
convert(::Type{<:FormValue{D,k,T}}, arg::FormValue{D}) where {D,k,T} = FormValue{D,k,T}(Tuple(arg))
convert(::Type{<:FormValue{D,k,T}}, arg::FormValue{D,k,T}) where {D,k,T} = arg

###############################################################
# Other constructors and conversions (FormValue)
###############################################################

zero(::Type{<:FormValue{D,k,T}}) where {D,k,T} = FormValue{D,k,T}(tfill(zero(T),_binom(Val(D),Val(k))))
zero(::FormValue{D,k,T}) where {D,k,T} = zero(FormValue{D,k,T})

function rand(rng::AbstractRNG, ::Random.SamplerType{FormValue{D,k,T}}) where {D,k,T}
  return FormValue{D,k,T}(Tuple(rand(rng, SVector{_binom(Val(D),Val(k)),T})))
end

Mutable(::Type{FormValue{D,k,T,L}}) where {D,k,T,L} = @notimplemented
Mutable(::T) where T<:FormValue = Mutable(T)
mutable(a::FormValue) = MVector(a.data)

change_eltype(::Type{FormValue{D,k}},::Type{T}) where {D,k,T} = FormValue{D,k,T}
change_eltype(::Type{FormValue{D,k,T1}},::Type{T2}) where {D,k,T1,T2} = FormValue{D,k,T2}
change_eltype(::FormValue{D,k,T1},::Type{T2}) where {D,k,T1,T2} = change_eltype(FormValue{D,k,T1},T2)

get_array(arg::FormValue{D,k,T,L}) where {D,k,T,L} = @notimplemented

###############################################################
# Introspection (FormValue)
###############################################################

eltype(::Type{<:FormValue{D,k,T}}) where {D,k,T} = T
eltype(::T) where T<:FormValue = eltype(T)

size(::Type{<:FormValue{D,k}}) where {D,k} = ntuple(_->D,Val(k))
size(::T) where T<:FormValue = size(T)

#length(::Type{<:FormValue{D}}) where {D} = D
#length(::FormValue{D}) where {D} = length(FormValue{D})

num_components(::Type{<:FormValue}) = @unreachable "The dimension D and form order k are needed to count components"
num_components(::Type{<:FormValue{D,k}}) where {D,k} = length(FormValue{D,k})
num_components(::T) where T<:FormValue = num_components(T)

num_indep_components(::Type{<:FormValue}) = @unreachable "The dimension D and form order k are needed to count components"
num_indep_components(::Type{<:FormValue{D,k}}) where {D,k} = _binom(Val(D),Val(k))
num_indep_components(::T) where T<:FormValue = num_indep_components(T)

###############################################################
# VTK export (FormValue)
###############################################################

# TODO
#function indep_components_names(::Type{<:FormValue{A}}) where A
#  [ "$i" for i in 1:A ]
#  if A>3
#    return ["$i" for i in 1:A ]
#  else
#    c_name = ["X", "Y", "Z"]
#    return [c_name[i] for i in 1:A ]
#  end
#end
