###############################################################
# Types
###############################################################
"""
    DGTensorValue{D,R,S,T,L} <: AbstractDGTensorValue{D,R,S,T,L}

Type representing a full rank `D`-dimensional type-(`R`,`S`) differential
geometry tensor. It has `L`=`D`⁽`ᴿ`⁺`ˢ`⁾ independent/stored scalar components
of type `T`.
"""
struct DGTensorValue{D,R,S,T,L} <: AbstractDGTensorValue{D,R,S,T,L}
    data::NTuple{L,T}
    function DGTensorValue{D,R,S,T}(data::NTuple{L,T}) where {D,R,S,L,T}
        @check L == D^(R+S) "wrong number of values, expecting L=D^(R+S), got  L=$L, D=$D, R=$R and S=$S"
        new{D,R,S,T,L}(data)
    end
end


###############################################################
# Constructors (DGTensorValue)
###############################################################

# Empty DGTensorValue constructor

DGTensorValue()                   = DGTensorValue{0,0,0,Int}( () )
DGTensorValue{0}()                = DGTensorValue{0,0,0,Int}( () )
DGTensorValue{0,0,0}()            = DGTensorValue{0,0,0,Int}( () )
DGTensorValue{0,0,0,T}() where T  = DGTensorValue{0,0,0,T  }( () )
DGTensorValue(data::NTuple{0})    = DGTensorValue{0,0,0,Int}(data)
DGTensorValue{0}(data::NTuple{0}) = DGTensorValue{0,0,0,Int}(data) # TODO check if this makes sense
DGTensorValue{0,0}(data::NTuple{0})=DGTensorValue{0,0,0,Int}(data)
DGTensorValue{0,0,0}(data::NTuple{0})=DGTensorValue{0,0,0,Int}(data)

# DGTensorValue single NTuple argument constructor

DGTensorValue{D,R,S}(data::NTuple{L,T})     where {D,R,S,L,T}     = DGTensorValue{D,R,S,T}(data)
DGTensorValue{D,R,S,T1}(data::NTuple{L,T2}) where {D,R,S,L,T1,T2} = DGTensorValue{D,R,S,T1}(NTuple{L,T1}(data))

#DGTensorValue{D}(data::NTuple{D2,T2}) where {D,D2,T2} = @unreachable
#DGTensorValue{D1,T1}(data::NTuple{D2,T2}) where {D1,T1,D2,T2} = @unreachable

# DGTensorValue single Tuple argument constructor

DGTensorValue(data::Tuple)                            = DGTensorValue(promote(data...))
DGTensorValue{D}(data::Tuple)        where {D}        = DGTensorValue{D}(promote(data...))
DGTensorValue{D,R,S}(data::Tuple)    where {D,R,S}    = DGTensorValue{D,R,S}(promote(data...))
DGTensorValue{D,R,S,T1}(data::Tuple) where {D,R,S,T1} = DGTensorValue{D,R,S,T1}(NTuple{D^(R+S),T1}(data))

# DGTensorValue Vararg constructor

DGTensorValue(data::Number...)                            = DGTensorValue{0}(data)
DGTensorValue{0}(data::Number...)                         = DGTensorValue{0}(data)
DGTensorValue{D,R,S}(data::Number...)    where {D,R,S}    = DGTensorValue{D,R,S}(data)
DGTensorValue{D,R,S,T1}(data::Number...) where {D,R,S,T1} = DGTensorValue{D,R,S,T1}(data)

# Fix for julia 1.0.4
DGTensorValue{D,R,S}(data::T...)    where {D,R,S,T<:Number}    = DGTensorValue{D,R,S,T}(data)

# DGTensorValue single AbstractVector argument constructor

DGTensorValue(data::AbstractArray{T}) where T                        = DGTensorValue{0,0,0,T}(data)
DGTensorValue{0}(data::AbstractArray{T}) where T                     = DGTensorValue{0,0,0,T}(data)
DGTensorValue{0,0}(data::AbstractArray{T}) where T                   = DGTensorValue{0,0,0,T}(data)
DGTensorValue{D,R,S}(data::AbstractArray{T}) where {D,R,S,T}         = DGTensorValue{D,R,S}(NTuple{D^(R+S),T}(data))
DGTensorValue{D,R,S,T1}(data::AbstractArray{T2}) where {D,R,S,T1,T2} = DGTensorValue{D,R,S,T1}(NTuple{D^(R+S),T1}(data))

###############################################################
# Conversions (DGTensorValue)
###############################################################

# Direct conversion
convert(::Type{<:DGTensorValue{D,R,S,T}}, arg:: AbstractArray) where {D,R,S,T} = DGTensorValue{D,R,S,T}(NTuple{D^(R+S),T}(arg))
convert(::Type{<:DGTensorValue{D,R,S,T}}, arg:: Tuple) where {D,R,S,T} = DGTensorValue{D,R,S,T}(arg)

# Inverse conversion TODO fix
# convert(::Type{<:MMatrix{D,D,T}}, arg::SymTensorValue) where {D,T} = MMatrix{D,D,T}(_SymTensorValue_to_array(arg))
convert(::Type{<:SArray}, arg::DGTensorValue) = @notimplemented
convert(::Type{<:MArray}, arg::DGTensorValue) = @notimplemented
convert(::Type{<:NTuple{L,T}}, arg::DGTensorValue) where {T,L} = NTuple{L,T}(Tuple(arg))

# Internal conversion
convert(::Type{<:DGTensorValue{D,R,S,T}}, arg::DGTensorValue{D,R}) where {D,R,S,T} = DGTensorValue{D,R,S,T}(Tuple(arg))
convert(::Type{<:DGTensorValue{D,R,S,T}}, arg::DGTensorValue{D,R,S,T}) where {D,R,S,T} = arg

###############################################################
# Other constructors and conversions (DGTensorValue)
###############################################################

zero(::Type{<:DGTensorValue{D,R,S,T}}) where {D,R,S,T} = DGTensorValue{D,R,S,T}(tfill(zero(T),D^(R+S)))
zero(::DGTensorValue{D,R,S,T}) where {D,R,S,T} = zero(DGTensorValue{D,R,S,T})

function rand(rng::AbstractRNG, ::Random.SamplerType{DGTensorValue{D,R,S,T}}) where {D,R,S,T}
  return DGTensorValue{D,R,S,T}(Tuple(rand(rng, SVector{D^(R+S),T})))
end

Mutable(::Type{DGTensorValue{D,R,S,T,L}}) where {D,R,S,T,L} = @notimplemented
Mutable(::T) where T<:DGTensorValue = Mutable(T)
mutable(a::DGTensorValue) = MVector(a.data)

change_eltype(::Type{DGTensorValue{D,R,S}},::Type{T}) where {D,R,S,T} = DGTensorValue{D,R,S,T}
change_eltype(::Type{DGTensorValue{D,R,S,T1}},::Type{T2}) where {D,R,S,T1,T2} = DGTensorValue{D,R,S,T2}
change_eltype(::DGTensorValue{D,R,S,T1},::Type{T2}) where {D,R,S,T1,T2} = change_eltype(DGTensorValue{D,R,S,T1},T2)

get_array(arg::DGTensorValue{D,R,S,T,L}) where {D,R,S,T,L} = @notimplemented

###############################################################
# Introspection (DGTensorValue)
###############################################################

eltype(::Type{<:DGTensorValue{D,R,S,T}}) where {D,R,S,T} = T
eltype(::T) where T<:DGTensorValue = eltype(T)

size(::Type{<:DGTensorValue{D,R,S}}) where {D,R,S} = ntuple(_->D,Val(R+S))
size(::T) where T<:DGTensorValue = size(T)

num_components(::Type{<:DGTensorValue}) = @unreachable "The dimension D and type indices R and S are needed to count components"
num_components(::Type{<:DGTensorValue{D,R,S}}) where {D,R,S} = length(DGTensorValue{D,R,S})
num_components(::T) where T<:DGTensorValue = num_components(T)

num_indep_components(::Type{<:DGTensorValue}) = @unreachable "The dimension D and type indices R and S  are needed to count components"
num_indep_components(::Type{<:DGTensorValue{D,R,S}}) where {D,R,S} = D^(R+S)
num_indep_components(::T) where T<:DGTensorValue = num_indep_components(T)

###############################################################
# VTK export (DGTensorValue)
###############################################################

# TODO
#function indep_components_names(::Type{<:DGTensorValue{A}}) where A
#  [ "$i" for i in 1:A ]
#  if A>3
#    return ["$i" for i in 1:A ]
#  else
#    c_name = ["X", "Y", "Z"]
#    return [c_name[i] for i in 1:A ]
#  end
#end
