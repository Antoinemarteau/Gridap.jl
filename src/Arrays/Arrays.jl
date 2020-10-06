"""
This module provides:
- An extension of the `AbstractArray` interface in order to properly deal with mutable caches.
- A mechanism to generate lazy arrays resulting from operations between arrays.
- A collection of concrete implementations of `AbstractArray`.

The exported names in this module are:

$(EXPORTS)
"""
module Arrays

using Gridap.Helpers
using Gridap.Inference
using Gridap.Algebra

using DocStringExtensions
using Test
using FillArrays
using LinearAlgebra
using BlockArrays
using Base: @propagate_inbounds
using ForwardDiff

import Base: size
import Base: getindex, setindex!
import Base: similar
import Base: IndexStyle

import Gridap.Algebra: scale_entries!

export BlockArrayCoo
export BlockVectorCoo
export BlockMatrixCoo
export is_zero_block
export is_nonzero_block
export enumerateblocks
export eachblockindex
# export VectorOfBlockArrayCoo
# export VectorOfBlockVectorCoo
# export VectorOfBlockMatrixCoo
export zeros_like
# export TwoLevelBlockedUnitRange

export array_cache
export getindex!
# # export getitems!
# export getitems
export testitem
# export uses_hash
export test_array
# export testitems
# # export array_caches
export get_array
# export get_arrays
# # export add_to_array!

export CachedArray
export CachedMatrix
export CachedVector
export setsize!
export setaxes!

# export CompressedArray
# export LocalToGlobalArray
# export LocalToGlobalPosNegArray
# export FilteredCellArray
# export FilterKernel

# export kernel_cache
# export kernel_caches
# export lazy_map_kernels!
# export lazy_map_kernel!
# export lazy_map_kernel
# export test_kernel
# export bcast
# export elem
# export contract
# export MulKernel
# export MulAddKernel
# export kernel_return_type
# export kernel_return_types
# export kernel_testitem
# export Kernel

# # export lazy_map
# export lazy_map_all

# export Table
# export identity_table
# export empty_table
# export rewind_ptrs!
# export length_to_ptrs!
# export append_ptrs
# export append_ptrs!
# export get_ptrs_eltype
# export get_data_eltype
# export generate_data_and_ptrs
# export find_inverse_index_map
# export find_inverse_index_map!
# export append_tables_globally
# export append_tables_locally
# export flatten_partition
# export collect1d
# export UNSET
# export get_local_item
# export find_local_index

# export reindex
# export identity_vector

# export SubVector
# export pair_arrays
# export unpair_arrays

# export lazy_append
# export lazy_split
# export AppendedArray

# export autodiff_array_gradient
# export autodiff_array_jacobian
# export autodiff_array_hessian

# export VectorWithEntryRemoved
# export VectorWithEntryInserted

# import Gridap.Io: to_dict
# import Gridap.Io: from_dict

include("Interface.jl")

include("BlockArraysCoo.jl")

include("CachedArrays.jl")

# include("Kernels.jl")

# include("Apply.jl")

# include("CompressedArrays.jl")

# include("Tables.jl")

# include("LocalToGlobalArrays.jl")

# include("LocalToGlobalPosNegArrays.jl")

# include("FilteredArrays.jl")

# include("Reindex.jl")

# include("IdentityVectors.jl")

# include("SubVectors.jl")

# include("ArrayPairs.jl")

# include("AppendedArrays.jl")

# include("VectorsOfBlockArrayCoo.jl")

# include("Autodiff.jl")

# include("VectorsWithEntryRemoved.jl")

# include("VectorsWithEntryInserted.jl")


end # module
