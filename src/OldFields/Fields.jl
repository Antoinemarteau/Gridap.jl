"""
This module provides:

- An interface for physical fields, basis of physical fields and related objects.
- Helpers functions to work with fields and arrays of fields.
- Helpers functions to create lazy operation trees from fields and arrays of fields

The exported names are:

$(EXPORTS)
"""
module Fields

using Gridap.Helpers
using Gridap.Inference
using Gridap.TensorValues
using Gridap.Arrays
using Gridap.Arrays: BCasted
using Gridap.Arrays: NumberOrArray
using Gridap.Arrays: LazyArray
using Gridap.Arrays: Contracted
using LinearAlgebra: ⋅
using BlockArrays

using Test
using DocStringExtensions
using FillArrays
import ForwardDiff

export Point
export field_gradient
export evaluate_field!
export evaluate_field
export field_cache
export field_return_type
export evaluate
export evaluate!
export gradient
export ∇
export Field
export test_field
export evaluate_to_field
export lazy_map_to_field_array
export test_array_of_fields
export compose
export compose_fields
export compose_field_arrays
export lincomb
export lazy_map_lincomb
export attachmap
export integrate
export field_caches
export field_return_types
export evaluate_fields
export evaluate_fields!
export field_gradients
export field_array_cache
export evaluate_field_array
export evaluate_field_arrays
export field_array_gradient
export gradient_type
export curl
export grad2curl
export laplacian
export divergence
export Δ
export ε
export symmetric_gradient

export Homothecy
export AffineMap

export VectorOfBlockBasisCoo
export insert_array_of_bases_in_block
export create_array_of_blocked_axes

export operate_fields
export operate_arrays_of_fields
export trialize_basis
export trialize_array_of_bases
export field_operation_axes
export field_operation_metasize

export function_field

import Gridap.Arrays: return_cache
import Gridap.Arrays: evaluate!
import Gridap.Arrays: return_type
import Gridap.Arrays: testitem!
import Gridap.Arrays: lazy_map
import Gridap.Arrays: reindex
import Gridap.TensorValues: outer
import Gridap.TensorValues: inner
import Gridap.TensorValues: symmetric_part
import Base: +, - , *
import LinearAlgebra: cross
import LinearAlgebra: tr
import LinearAlgebra: dot
import Base: transpose
import Base: adjoint

include("FieldInterface.jl")

include("MockFields.jl")

include("FunctionFields.jl")

include("ConstantFields.jl")

include("Homothecies.jl")

include("AffineMaps.jl")

include("FieldApply.jl")

include("FieldArrays.jl")

include("Lincomb.jl")

include("Compose.jl")

include("Attachmap.jl")

include("Integrate.jl")

include("FieldOperations.jl")

include("VectorsOfBlockBasisCoo.jl")

include("DiffOperators.jl")

include("UnimplementedFields.jl")

end # module
