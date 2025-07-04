###############################################################
# TensorValue Type
###############################################################

"""
    TensorValue{D1,D2,T,L} <: MultiValue{Tuple{D1,D2},T,2,L}

Type representing a second-order `D1`×`D2` tensor. It must hold `L` = `D1`*`D2`.

If only `D1` or no dimension parameter is given to the constructor, `D1`=`D2` is assumed.
"""
struct TensorValue{D1,D2,T,L} <: MultiValue{Tuple{D1,D2},T,2,L}
    data::NTuple{L,T}
    function TensorValue{D1,D2,T}(data::NTuple{L,T}) where {D1,D2,T,L}
        @check L == D1*D2
        new{D1,D2,T,L}(data)
    end
end

###############################################################
# Constructors
###############################################################

# Empty TensorValue constructor

TensorValue()                     = TensorValue{0,0,Int}(NTuple{0,Int}())
TensorValue{0,0}()                = TensorValue{0,0,Int}(NTuple{0,Int}())
TensorValue{0,0,T}() where {T}    = TensorValue{0,0,T}(NTuple{0,T}())
TensorValue(data::NTuple{0})      = TensorValue{0,0,Int}(data)
TensorValue{0,0}(data::NTuple{0}) = TensorValue{0,0,Int}(data)

# TensorValue single NTuple argument constructor

@generated function TensorValue(data::NTuple{L,T}) where {L,T}
  msg = "The number of scalar arguments in TensorValue has to be a perfect square (e.g., 1, 4, 9, 16, ...)"
  V = sqrt(L)
  @assert floor(Int,V) == ceil(Int,V) msg
  D = Int(V)
  quote
    TensorValue{$D,$D,T}(data)
  end
end
TensorValue{D}(data::NTuple{L,T}) where {D,L,T}                   = TensorValue{D,D,T}(data)
TensorValue{D1,D2}(data::NTuple{L,T}) where {D1,D2,L,T}           = TensorValue{D1,D2,T}(data)
TensorValue{D1,D2,T1}(data::NTuple{L,T2}) where {D1,D2,L,T1,T2}   = TensorValue{D1,D2,T1}(NTuple{L,T1}(data))
TensorValue{D1,D2,T1,L}(data::NTuple{L,T2}) where {D1,D2,L,T1,T2} = TensorValue{D1,D2,T1}(NTuple{L,T1}(data))

# TensorValue single Tuple argument constructor

TensorValue(data::Tuple)                             = TensorValue(promote(data...))
TensorValue{D}(data::Tuple)        where {D}         = TensorValue{D}(promote(data...))
TensorValue{D1,D2}(data::Tuple)    where {D1,D2}     = TensorValue{D1,D2}(promote(data...))
TensorValue{D1,D2,T1}(data::Tuple) where {D1,D2,T1}  = TensorValue{D1,D2,T1}(NTuple{length(data),T1}(data))
TensorValue{D1,D2,T1,L}(data::Tuple) where {D1,D2,T1,L}  = TensorValue{D1,D2,T1}(NTuple{length(data),T1}(data))

# TensorValue Vararg constructor

TensorValue(data::Number...)                            = TensorValue(data)
TensorValue{D}(data::Number...)        where {D}        = TensorValue{D}(data)
TensorValue{D1,D2}(data::Number...)    where {D1,D2}    = TensorValue{D1,D2}(data)
TensorValue{D1,D2,T1}(data::Number...) where {D1,D2,T1} = TensorValue{D1,D2,T1}(data)
TensorValue{D1,D2,T1,L}(data::Number...) where {D1,D2,T1,L} = TensorValue{D1,D2,T1}(data)

# TensorValue single AbstractMatrix argument constructor

TensorValue(data::AbstractMatrix{T}) where {T} = ((D1,D2)=size(data);L=length(data);TensorValue{D1,D2,T}(NTuple{L,T}(data)))
TensorValue{D}(data::AbstractMatrix{T}) where {D,T} = (L=length(data);TensorValue{D,D,T}(NTuple{L,T}(data)))
TensorValue{D1,D2}(data::AbstractMatrix{T}) where {D1,D2,T} = (L=length(data);TensorValue{D1,D2,T}(NTuple{L,T}(data)))
TensorValue{D1,D2,T1}(data::AbstractMatrix{T2}) where {D1,D2,T1,T2} = (L=length(data);TensorValue{D1,D2,T1}(NTuple{L,T1}(data)))
TensorValue{D1,D2,T1,L}(data::AbstractMatrix{T2}) where {D1,D2,T1,T2,L} = TensorValue{D1,D2,T1}(NTuple{L,T1}(data))

###############################################################
# Conversions (TensorValue)
###############################################################

# Direct conversion
convert(::Type{<:TensorValue{D1,D2,T}}, arg::AbstractArray) where {D1,D2,T} = TensorValue{D1,D2,T}(arg)
convert(::Type{<:TensorValue{D1,D2,T}}, arg::Tuple) where {D1,D2,T} = TensorValue{D1,D2,T}(arg)

# Inverse conversion
convert(::Type{<:SMatrix{D1,D2,T}}, arg::TensorValue) where {D1,D2,T} = SMatrix{D1,D2,T}(Tuple(arg))
convert(::Type{<:MMatrix{D1,D2,T}}, arg::TensorValue) where {D1,D2,T} = MMatrix{D1,D2,T}(Tuple(arg))
convert(::Type{<:NTuple{L,T1}}, arg::TensorValue) where {L,T1} = NTuple{L,T1}(Tuple(arg))

# Internal conversion
convert(::Type{<:TensorValue{D1,D2,T}}, arg::TensorValue{D1,D2}) where {D1,D2,T} = TensorValue{D1,D2,T}(Tuple(arg))
convert(::Type{<:TensorValue{D1,D2,T}}, arg::TensorValue{D1,D2,T}) where {D1,D2,T} = arg

###############################################################
# Other constructors and conversions (TensorValue)
###############################################################

zero(::Type{<:TensorValue{D1,D2,T}}) where {D1,D2,T} = TensorValue{D1,D2,T}(tfill(zero(T),Val{D1*D2}()))
zero(::TensorValue{D1,D2,T}) where {D1,D2,T} = zero(TensorValue{D1,D2,T})

@generated function one(::Type{<:TensorValue{D1,D2,T}}) where {D1,D2,T}
  str = join(["$i==$j ? one(T) : zero(T), " for i in 1:D1 for j in 1:D2])
  Meta.parse("TensorValue{D1,D2,T}(($str))")
end
one(::TensorValue{D1,D2,T}) where {D1,D2,T} = one(TensorValue{D1,D2,T})

@generated function rand(rng::AbstractRNG,
                         ::Random.SamplerType{<:TensorValue{D1,D2,T}}) where {D1,D2,T}
  L=D1*D2
  quote
    rand(rng, TensorValue{D1,D2,T,$L})
  end
end
function rand(rng::AbstractRNG,
              ::Random.SamplerType{<:TensorValue{D1,D2,T,L}}) where {D1,D2,T,L}
  return TensorValue{D1,D2,T,L}(Tuple(rand(rng, SVector{L,T})))
end

Mutable(::Type{<:TensorValue{D1,D2,T}}) where {D1,D2,T} = MMatrix{D1,D2,T}
Mutable(::TensorValue{D1,D2,T}) where {D1,D2,T} = Mutable(TensorValue{D1,D2,T})
mutable(a::TensorValue{D1,D2}) where {D1,D2} = MMatrix{D1,D2}(a.data)

change_eltype(::Type{TensorValue{D1,D2,T1}},::Type{T2}) where {D1,D2,T1,T2} = TensorValue{D1,D2,T2}
change_eltype(::Type{TensorValue{D1,D2,T1,L}},::Type{T2}) where {D1,D2,T1,T2,L} = TensorValue{D1,D2,T2,L}
change_eltype(::TensorValue{D1,D2,T1,L},::Type{T2}) where {D1,D2,T1,T2,L} = change_eltype(TensorValue{D1,D2,T1,L},T2)

get_array(arg::TensorValue{D1,D2,T}) where {D1,D2,T} = convert(SMatrix{D1,D2,T},arg)

"""
    diagonal_tensor(v::VectorValue{D,T}) -> ::TensorValue{D,D,T}

Return a diagonal `D`×`D` tensor with diagonal containing the elements of `v`.
"""
@generated function diagonal_tensor(v::VectorValue{D,T}) where {D,T}
  s = ["zero(T), " for i in 1:(D*D)]
  for i in 1:D
    d = D*(i-1)+i
    s[d] = "v.data[$i],"
  end
  str = join(s)
  Meta.parse("TensorValue{D,D,T,$(D*D)}(($str))")
end

"""
    tensor_from_columns(cols::NTuple{D2,VectorValue{D1,T}}) -> ::TensorValue{D1,D2,T}

Return a `D1`×`D2` tensor with columns given by the `D1`-dimensional vectors in `cols`.
"""
@generated function tensor_from_columns(cols::NTuple{D2,VectorValue{D1,T}}) where {D1,D2,T}
  s = ""
  for j in 1:D2
    for i in 1:D1
      s *= "cols[$j].data[$i],"
    end
  end
  Meta.parse("TensorValue{D1,D2,T,$(D1*D2)}(($s))")
end

tensor_from_columns(cols::VectorValue...) = tensor_from_columns(cols)

"""
    tensor_from_rows(rows::NTuple{D1,VectorValue{D2,T}}) -> ::TensorValue{D1,D2,T}

Return a `D1`×`D2` tensor with rows given by the `D2`-dimensional vectors in `rows`.
"""
@generated function tensor_from_rows(rows::NTuple{D1,VectorValue{D2,T}}) where {D1,D2,T}
  s = ""
  for j in 1:D2
    for i in 1:D1
      s *= "rows[$i].data[$j],"
    end
  end
  Meta.parse("TensorValue{D1,D2,T,$(D1*D2)}(($s))")
end

tensor_from_rows(rows::VectorValue...) = tensor_from_rows(rows)

###############################################################
# Introspection (TensorValue)
###############################################################

eltype(::Type{<:TensorValue{D1,D2,T}}) where {D1,D2,T} = T
eltype(::TensorValue{D1,D2,T}) where {D1,D2,T} = eltype(TensorValue{D1,D2,T})

size(::Type{<:TensorValue{D}}) where {D} = (D,D)
size(::Type{<:TensorValue{D1,D2}}) where {D1,D2} = (D1,D2)
size(::TensorValue{D1,D2}) where {D1,D2} = size(TensorValue{D1,D2})

length(::Type{<:TensorValue{D}}) where {D} = length(TensorValue{D,D})
length(::Type{<:TensorValue{D1,D2}}) where {D1,D2} = D1*D2
length(::TensorValue{D1,D2}) where {D1,D2} = length(TensorValue{D1,D2})

num_components(::Type{<:TensorValue}) = @unreachable "All two size dimensions are needed to count components"
num_components(::Type{<:TensorValue{D,D}}) where {D} = length(TensorValue{D,D})
num_components(::Type{<:TensorValue{D1,D2}}) where {D1,D2} = length(TensorValue{D1,D2})
num_components(::TensorValue{D1,D2}) where {D1,D2} = num_components(TensorValue{D1,D2})

###############################################################
# VTK export (TensorValue)
###############################################################

function indep_components_names(::Type{<:TensorValue{D1,D2}}) where {D1,D2}
  if D1>3 || D2>3
    return ["$i$j" for i in 1:D1 for j in 1:D2 ]
  else
    c_name = ["X", "Y", "Z"]
    return [c_name[i]*c_name[j] for i in 1:D1 for j in 1:D2 ]
  end
end
