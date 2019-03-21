export CellField
export evaluate

abstract type CellField{D,T} end

evaluate(::CellField{D,T} where {D,T},::CellPoints{D} where D)::CellFieldValues{T} = @abstractmethod

"""
Returns another CellField object that represents the gradient.
TG is a value whose rank is one order grater than the one of T
"""
gradient(::CellField{D,T} where {D,T})::CellField{D,TG} = @abstractmethod

# Concrete implementations

"""
General implementation
Accepts any kind of `CellBasisValues{TB}` representing
the evaluated shape functions and `CellFieldValues{TN}` representing
the "nodal" values of the field to be interpolated at the points,
where the shape functions are evaluated. This implementation only assumes
that outer(·,·) is defined for instances of TB and TN. The result is of type T
"""
struct CellFieldValuesFromInterpolation{TB,TN,T} <: CellFieldValues{T}
  cellbasisvalues::CellBasisValues{TB}
  cellnodalvalues::CellFieldValues{TN}
end

function Base.length(self::CellFieldValuesFromInterpolation)
  @assert length(self.cellbasisvalues) == length(self.cellnodalvalues)
  length(self.cellbasisvalues)
end

function Base.iterate(
  self::CellFieldValuesFromInterpolation{TB,TN,T}) where {TB,TN,T}
  (maxdofs,maxqpoints) = maxsize(self.cellbasisvalues)
  pointvalues = Array{T,1}(undef,(maxqpoints,))
  basisvalsnext = iterate(self.cellbasisvalues)
  nodalvalsnext = iterate(self.cellnodalvalues)
  state = (pointvalues,basisvalsnext,nodalvalsnext)
  iterate(self,state)
end

function Base.iterate(self::CellFieldValuesFromInterpolation,state)
  (pointvalues,basisvalsnext,nodalvalsnext) = state
  if basisvalsnext == nothing || nodalvalsnext == nothing
    nothing
  else
    (basisvals,basisvalsstate) = basisvalsnext
    (nodalvals,nodalvalsstate) = nodalvalsnext
    interpolate_kernel!(basisvals,nodalvals,pointvalues)
    npoints = size(basisvals,2)
    vals = @view pointvalues[1:npoints]
    basisvalsnext = iterate(self.cellbasisvalues,basisvalsstate)
    nodalvalsnext = iterate(self.cellnodalvalues,nodalvalsstate)
    state = (pointvalues,basisvalsnext,nodalvalsnext)
    (vals, state)
  end
end

function interpolate_kernel!(basisvals,nodalvals,pointvalues)
  @assert size(basisvals,1) == length(nodalvals)
  @assert size(basisvals,2) == length(pointvalues)
  ndofs = size(basisvals,1)
  npoin = size(basisvals,2)
  @inbounds for i = 1:npoin
    pointvalues[i] = zero(pointvalues[i])
    @inbounds for j = 1:ndofs
      # TODO: to think about the best order in this multiplication
      # It will depend in how we define the gradient
      pointvalues[i] += outer(basisvals[j,i],nodalvals[j])
    end
  end
end

struct CellFieldFromInterpolation{D,T,TB,TN} <: CellField{D,T}
  cellbasis::CellBasis{D,TB}
  cellnodalvalues::CellFieldValues{TN}
end

function evaluate(
  self::CellFieldFromInterpolation{D,T,TB,TN},points::CellPoints{D}) where {D,T,TB,TN}
  cellbasisvalues = evaluate(self.cellbasis,points)
  CellFieldValuesFromInterpolation{TB,TN,T}(cellbasisvalues,self.cellnodalvalues)
end

function gradient(self::CellFieldFromInterpolation{D,T,TB,TN}) where {D,T,TB,TN}
  grad_cellbasis = gradient(self.cellbasis)
  TG = gradient(T,Val(D))
  TBG = gradient(TB,Val(D))
  CellFieldFromInterpolation{D,TG,TBG,TN}(grad_cellbasis,self.cellnodalvalues)
end
