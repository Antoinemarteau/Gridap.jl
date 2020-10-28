"""
   abstract type CellDofBasis end

Abstract type that represents a cell array of `Dof`. The main motivation for
its definition is to provide a trait that informs whether the `Dof` entries are
defined for functions in the reference or physical space
"""
abstract type CellDofBasis end

RefStyle(::Type{<:CellDofBasis}) = @abstractmethod
get_array(::CellDofBasis) = @abstractmethod

function test_cell_dof_basis(cf::CellDofBasis,f::CellField)
  ar = get_array(cf)
  @test isa(ar,AbstractArray)
  a = evaluate(cf,f)
  _ = collect(a)
  rs = RefStyle(cf)
  @test isa(get_val_parameter(rs),Bool)
end

"""
    evaluate(dof_array::CellDofBasis,field_array::AbstractArray)

Evaluates the `CellDofBasis` for the `Field` objects
at the array `field` element by element.

The result is numerically equivalent to

    map(evaluate, dof_array.array, field_array)

but it is described with a more memory-friendly lazy type.
"""
function evaluate(cell_dofs::CellDofBasis,cell_field::CellField)
 _evaluate_cell_dofs(cell_dofs,cell_field,RefStyle(cell_dofs))
end

function  _evaluate_cell_dofs(cell_dofs,cell_field,ref_trait::Val{true})
  ReferenceFEs.evaluate_dof_array(get_array(cell_dofs),get_array(to_ref_space(cell_field)))
end

function  _evaluate_cell_dofs(cell_dofs,cell_field,ref_trait::Val{false})
  ReferenceFEs.evaluate_dof_array(get_array(cell_dofs),get_array(to_physical_space(cell_field)))
end

"""
"""
struct GenericCellDofBasis{R} <: CellDofBasis
   ref_trait::Val{R}
   array::AbstractArray{<:Dof}
end

RefStyle(::Type{<:GenericCellDofBasis{R}}) where R = Val{R}()

get_array(a::GenericCellDofBasis) = a.array

function reindex(cf::CellDofBasis,a::AbstractVector)
  similar_object(cf,reindex(get_array(cf),a))
end

function similar_object(cf::CellDofBasis,a::AbstractArray)
  GenericCellDofBasis(RefStyle(cf),a)
end
