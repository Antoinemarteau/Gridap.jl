
"""
    struct ZeroMeanFESpace <: SingleFieldFESpace
      # private fields
    end
"""
struct ZeroMeanFESpace{B} <: SingleFieldFESpace
  space::FESpaceWithConstantFixed
  vol_i::Vector{Float64}
  vol::Float64
  constraint_style::Val{B}
end

"""
    ZeroMeanFESpace(
      space::SingleFieldFESpace,
      trian::Triangulation,
      quad::CellQuadrature)
"""
function ZeroMeanFESpace(
  space::SingleFieldFESpace,trian::Triangulation,quad::CellQuadrature)

  _space = FESpaceWithConstantFixed(space,
                                    true,
                                    num_free_dofs(space))
  vol_i, vol = _setup_vols(space,trian,quad)
  ZeroMeanFESpace(_space,vol_i,vol,constraint_style(_space))
end

function _setup_vols(V,trian,quad)
  U = TrialFESpace(V)
  assem = SparseMatrixAssembler(U,V)
  bh = get_cell_basis(V)
  bh_trian = restrict(bh,trian)
  cellvec = integrate(bh_trian,trian,quad)
  cellids = get_cell_id(trian)
  vecdata = ([cellvec],[cellids])
  vol_i = assemble_vector(assem,vecdata)
  vol = sum(vol_i)
  (vol_i, vol)
end

# Genuine functions

function TrialFESpace(f::ZeroMeanFESpace)
  U = TrialFESpace(f.space)
  ZeroMeanFESpace(U,f.vol_i,f.vol,f.constraint_style)
end

function FEFunction(
  f::ZeroMeanFESpace,
  free_values::AbstractVector,
  dirichlet_values::AbstractVector)
  c = _compute_new_fixedval(free_values,
                            dirichlet_values,
                            f.vol_i,
                            f.vol,
                            f.space.dof_to_fix)
  fv = apply(+,free_values,Fill(c,length(free_values)))
  dv = dirichlet_values .+ c
  FEFunction(f.space,fv,dv)
end

function EvaluationFunction(f::ZeroMeanFESpace,free_values)
  FEFunction(f.space,free_values)
end

function _compute_new_fixedval(fv,dv,vol_i,vol,fixed_dof)
  @assert length(fv) + 1 == length(vol_i)
  @assert length(dv) == 1
  c = zero(eltype(vol_i))
  for i=1:fixed_dof-1
    c += fv[i]*vol_i[i]
  end
  c += first(dv)*vol_i[fixed_dof]
  for i=fixed_dof+1:length(vol_i)
    c += fv[i-1]*vol_i[i]
  end
  c = -c/vol
  c
end

# Delegated functions

constraint_style(::Type{ZeroMeanFESpace{B}}) where B = Val{B}()

get_cell_axes(t::ZeroMeanFESpace)= get_cell_axes(t.space)

get_cell_axes_with_constraints(t::ZeroMeanFESpace)= get_cell_axes_with_constraints(t.space)

CellData.CellField(t::ZeroMeanFESpace,cell_vals) = CellField(t.space,cell_vals)

get_cell_isconstrained(f::ZeroMeanFESpace) = get_cell_isconstrained(f.space)

get_cell_constraints(f::ZeroMeanFESpace) = get_cell_constraints(f.space)

get_dirichlet_values(f::ZeroMeanFESpace) = get_dirichlet_values(f.space)

get_cell_basis(f::ZeroMeanFESpace) = get_cell_basis(f.space)

get_cell_dof_basis(f::ZeroMeanFESpace) = get_cell_dof_basis(f.space)

num_free_dofs(f::ZeroMeanFESpace) = num_free_dofs(f.space)

zero_free_values(f::ZeroMeanFESpace) = zero_free_values(f.space)

get_cell_dofs(f::ZeroMeanFESpace) = get_cell_dofs(f.space)

num_dirichlet_dofs(f::ZeroMeanFESpace) = num_dirichlet_dofs(f.space)

zero_dirichlet_values(f::ZeroMeanFESpace) = zero_dirichlet_values(f.space)

num_dirichlet_tags(f::ZeroMeanFESpace) = num_dirichlet_tags(f.space)

get_dirichlet_dof_tag(f::ZeroMeanFESpace) = get_dirichlet_dof_tag(f.space)

scatter_free_and_dirichlet_values(f::ZeroMeanFESpace,fv,dv) = scatter_free_and_dirichlet_values(f.space,fv,dv)

gather_free_and_dirichlet_values(f::ZeroMeanFESpace,cv) = gather_free_and_dirichlet_values(f.space,cv)

gather_free_and_dirichlet_values!(fv,dv,f::ZeroMeanFESpace,cv) = gather_free_and_dirichlet_values!(fv,dv,f.space,cv)

gather_dirichlet_values(f::ZeroMeanFESpace,cv) = gather_dirichlet_values(f.space,cv)

gather_dirichlet_values!(dv,f::ZeroMeanFESpace,cv) = gather_dirichlet_values!(dv,f.space,cv)

gather_free_values(f::ZeroMeanFESpace,cv) = gather_free_values(f.space,cv)

gather_free_values!(fv,f::ZeroMeanFESpace,cv) = gather_free_values!(fv,f.space,cv)
