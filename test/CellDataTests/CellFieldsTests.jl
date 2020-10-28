module CellFieldsTests

using Test
using BlockArrays
using FillArrays
using Gridap.Arrays
using Gridap.Fields
using Gridap.TensorValues
using Gridap.CellData





#np = 3
#ndofs = 4
#
#p = Point(1,2)
#x = fill(p,np)
#z = 2.0
#
#v = VectorValue(3.0,1.5)
#w = VectorValue(3.4,3.5)
#a = MockBasis{2}(v,ndofs)
#b = MockBasis{2}(w,ndofs)
#c = fill(1.0,ndofs)
#f = OtherMockBasis{2}(ndofs)
#
#g = MockField{2}(v)
#
#l = 10
#xl = Fill(x,l)
#zl = [ z for  i in 1:l]
#cl = fill(c,l)
#fl = Fill(f,l)
#ϕl = lincomb(fl,cl)
#gl = fill(g,l)
#al = Fill(a,l)
#bl = fill(b,l)
#
#gf = GenericCellField(gl,ϕl,Val(true))
#gf_x = evaluate(gf,xl)
#∇gf_x = evaluate(∇(gf),xl)
#test_cell_field(gf,xl,gf_x,grad=∇gf_x)
#
#af = GenericCellField(al,ϕl,Val(true),Fill((Base.OneTo(ndofs),),l),Val((:,)))
#bf = GenericCellField(bl,ϕl,Val(true),Fill((Base.OneTo(ndofs),),l),Val((:,)))
#zf = convert_to_cell_field(zl,ϕl)
#df = af*zf
#dft = trialize_cell_basis(df)
#
#zfr = reindex(zf,[1,4,3])
#@test length(zfr) == 3
#@test length(get_array(zfr)) == 3
#@test length(get_cell_map(zfr)) == 3
#@test length(get_cell_axes(zfr)) == 3
#
## Check memoization
#df_x1 = evaluate(df,xl)
#df_x2 = evaluate(df,xl)
#@test df_x1 === df_x2
#∇gf1 = ∇(gf)
#∇gf2 = ∇(gf)
#@test ∇gf1 === ∇gf2
#@test evaluate(∇gf1,xl) === evaluate(∇gf2,xl)
#εgf1 = ε(gf)
#εgf2 = ε(gf)
#@test εgf1 === εgf2
#@test ∇×gf === ∇×gf
#@test evaluate(∇×gf,xl) === evaluate(∇×gf,xl)
#
#@test is_test(af)
#@test is_trial(dft)
#@test is_basis(af)
#@test is_basis(dft)
#mf = af⋅dft
#@test get_metasize(mf) == (:,:)
#mf_x = evaluate(mf,xl)
#@test size(mf_x[1]) == (np,ndofs,ndofs)
#
#@test get_cell_axes(mf) == Fill((Base.OneTo(ndofs),Base.OneTo(ndofs)),l)
#
#idsL = [ i*collect(1:ndofs)  for i in 1:l]
#idsR = [ 2*i*collect(1:ndofs)  for i in 1:l]
#axesL = Fill((Base.OneTo(ndofs),),l)
#axesR = Fill((Base.OneTo(ndofs),),l)
#
#idsS = merge_cell_dofs_at_skeleton(idsL,idsR,axesL,axesR)
#@test isa(idsS,VectorOfBlockArrayCoo)
#
#afS = merge_cell_fields_at_skeleton(af,2*af)
#@test isa(afS,SkeletonCellField)
#
#afL_x = evaluate(afS.left,xl)
#afR_x = evaluate(afS.right,xl)
#@test isa(afL_x,VectorOfBlockArrayCoo)
#@test isa(afR_x,VectorOfBlockArrayCoo)
#
#@test isa(afS*2,SkeletonCellField)
#@test isa(afS+afS,SkeletonCellField)
#
## Checks associated with trial bases
#df = bf
#dft = trialize_cell_basis(df)
#@test is_trial(dft) == true
#cell_vals = [rand(ndofs) for  i in 1:l]
#cell_field = lincomb(df,cell_vals)
#@test isa(cell_field,CellField)
#@test get_metasize(cell_field) == ()
#cell_field_x = evaluate(cell_field,xl)
#@test isa(cell_field_x[1],AbstractVector)

end # module

