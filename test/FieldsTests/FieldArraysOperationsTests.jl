module FieldOperationsTests

using Gridap.Arrays
using Gridap.Mappings
using Gridap.TensorValues
using FillArrays

using Test
using BenchmarkTools

# Operations

np = 4
d = 2
p = Point(1.0,2.0)
x = fill(p,np)
npp = (np,np)

v = VectorValue{d}(1.0,1.0)
f = MockField{d}(v)
fx = fill(v,np)

nf = 3
nff = (nf,nf)

# Sum

df = f+f

fia, p = test_field_array(f,p,nf,grad=true)
fia, p = test_field_array(f,p,nff,grad=true)

fa, p = test_field_array(df,p,nf,grad=true)
fa, p = test_field_array(df,p,nff,grad=true)

@test fia+fia == fa

bfia, x = test_broadcast_field_array(f,p,nf,np,grad=true)
bfia, x = test_broadcast_field_array(f,p,nff,npp,grad=true)

bfa, x = test_broadcast_field_array(df,p,nf,np,grad=true)
bfa, x = test_broadcast_field_array(df,p,nff,npp,grad=true)

@test bfia+bfia == bfa

@test evaluate(bfia,x)+evaluate(bfia,x) == evaluate(bfa,x)

# c = return_cache(bfia,x)
# @btime evaluate!(c,bfia+bfia,x);
# c = return_cache(bfa,x)
# @btime evaluate!(c,bfa,x);

# Subtraction

df = f-f

fa, p = test_field_array(df,p,nf,grad=true)
fa, p = test_field_array(df,p,nff,grad=true)

@test fia-fia == fa

bfa, x = test_broadcast_field_array(df,p,nf,np,grad=true)
bfa, x = test_broadcast_field_array(df,p,nff,npp,grad=true)

@test bfia-bfia == bfa

@test evaluate(bfia,x)-evaluate(bfia,x) == evaluate(bfa,x)

# Times scalar

df = GenericField(2.0)*f

fa, p = test_field_array(df,p,nf,grad=true)
fa, p = test_field_array(df,p,nff,grad=true)

@test GenericField(2.0).⋅fia == fa

bfa, x = test_broadcast_field_array(df,p,nf,np,grad=true)
bfa, x = test_broadcast_field_array(df,p,nff,npp,grad=true)

@test bfa == BroadcastField(GenericField(2.0)).⋅bfia

c = return_cache(bfa,x)
@btime evaluate!(c,bfa,x);
c = return_cache(df,p)
@btime evaluate!(c,df,p);

# Inner product

df = f⋅f

fa, p = test_field_array(df,p,nf,grad=true)
fa, p = test_field_array(df,p,nff,grad=true)

@test fia.⋅fia == fa

bfa, x = test_broadcast_field_array(df,p,nf,np,grad=true)
bfa, x = test_broadcast_field_array(df,p,nff,npp,grad=true)

@test bfa == bfia.⋅bfia

c = return_cache(bfa,x)
@btime evaluate!(c,bfa,x);
c = return_cache(df,p)
@btime evaluate!(c,df,p);

# Function

q(x) = 2*x

f = GenericField(q)
fia, p = test_field_array(f,p,nff,grad=true)
bfia, x = test_broadcast_field_array(f,p,nff,npp,grad=true)


fa, p = test_field_array(f,p,nf,grad=true)
fa, p = test_field_array(f,p,nff,grad=true)

@test fia == fa

bfa, x = test_broadcast_field_array(f,p,nf,np,grad=true)
bfa, x = test_broadcast_field_array(f,p,nff,npp,grad=true)

@test bfia == bfa

# Algebraic operations with arrays of fields

fia = fill(f,nf)
@test evaluate(transpose(fia)*fia,p) == nf*(evaluate(f,p)⋅evaluate(f,p))

bfia = fill(BroadcastField(f),nf)
@test evaluate(transpose(bfia)*bfia,p) == nf*(evaluate(f,p)⋅evaluate(f,p))

fia = fill(f,nff)
@test evaluate(transpose(fia).*fia,p) == fill(evaluate(f,p)⋅evaluate(f,p),nff)

@test evaluate(fia*fia,p) == fill(60.0,nff)
@test evaluate(transpose(fia)*fia,p) == fill(60.0,nff)

v = VectorValue{d}(1.0,1.0)
# v = 2.0
g = MockField{d}(v)

_nff = (3,3)
fia = fill(g,_nff)
@test evaluate(fia*fia,p) == fill(6.0,_nff)
@test evaluate(transpose(fia)*fia,p) == fill(6.0,_nff)

fia = [f g g; g f g; g g g]
@test evaluate(fia*fia,p) == [24.0 14.0 10.0; 14.0 24.0 10.0; 10.0 10.0 6.0]

####

# This is the Operation-based approach... we must decide whether it pays
# the price

op = Operation(+)
myop = OperationArray(Operation(+),fia,fia)
c = return_cache(myop,p)
@btime evaluate!(c,myop,p)
c = return_cache(myop.res,p)
@btime evaluate!(c,myop.res,p)
fa = fia+fia
c = return_cache(fa,p)
@btime evaluate!(c,fa,p)
@test myop.res == fa

op = Operation(*)
myop = OperationArray(Operation(*),fia,fia)
c = return_cache(myop,p)
@btime evaluate!(c,myop,p)
c = return_cache(myop.res,p)
@btime evaluate!(c,myop.res,p)


fa = fia*fia
c = return_cache(fa,p)
@btime evaluate!(c,fa,p)
@test myop.res == fa

####

# Operations between fields and array of fields

# p = Point(1,2)
# x = fill(p,np)
# z = 2.0

# v = VectorValue(3.0,1.5)
# w = VectorValue(3.4,3.5)
# test_basis = MockBasis(2,v,ndofs)
# test_basis_2 = MockBasis(2,w,ndofs)

# t1x = evaluate(test_basis,x)
# t2x = evaluate(test_basis_2,x)
# ∇t1x = evaluate(∇(test_basis),x)
# ∇t2x = evaluate(∇(test_basis_2),x)


# b = field_composition(BroadcastMapping(+),test_basis,test_basis_2)
# r = broadcast(+,t1x,t2x)
# test_mapping(b,(x,),r)

# ∇b = composition(BroadcastMapping(+),gradient(test_basis),gradient(test_basis_2))
# ∇r = broadcast(+,∇t1x,∇t2x)
# test_mapping(b,(x,),r)

# b = composition(BroadcastMapping(-),test_basis,test_basis_2)
# r = broadcast(-,t1x,t2x)
# test_mapping(b,(x,),r)

# ∇b = composition(BroadcastMapping(-),gradient(test_basis),gradient(test_basis_2))
# ∇r = broadcast(-,∇t1x,∇t2x)
# test_mapping(b,(x,),r)

# b = composition(BroadcastMapping(⋅),test_basis,test_basis_2)
# r = broadcast(⋅,t1x,t2x)
# test_mapping(b,(x,),r)

# ∇b = composition(BroadcastMapping(⋅),gradient(test_basis),gradient(test_basis_2))
# ∇r = broadcast(⋅,∇t1x,∇t2x)
# test_mapping(b,(x,),r)

# b = composition(BroadcastMapping(*),test_basis,ConstantField(z))
# r = broadcast(*,t1x,z)
# test_mapping(b,(x,),r)

# ∇b = composition(BroadcastMapping(*),gradient(test_basis),ConstantField(z))
# ∇r = broadcast(*,∇t1x,z)
# test_mapping(b,(x,),r)

# b = composition(BroadcastMapping(+),test_basis,ConstantField(z))
# r = broadcast(+,t1x,z)
# test_mapping(b,(x,),r)
# test_mapping(b,(x,),fill(v+z,np,ndofs))

# # @santiagobadia : To understand what should I do here
# # trial_basis = trialize_basis(test_basis)
# # r = reshape(evaluate(test_basis,x),(np,1,ndofs))
# # ∇r = reshape(evaluate(∇(test_basis),x),(np,1,ndofs))
# # test_field(trial_basis,x,r,grad=∇r

# xl = Fill(x,l)
# fl = [ ConstantField(z) for  i in 1:l]

# test_basis_array = Fill(test_basis,l)
# test_basis_2_array = fill(test_basis_2,l)

# ∇test_basis_array = Fill(gradient(test_basis),l)
# ∇test_basis_2_array = fill(gradient(test_basis_2),l)

# op = Fill(BroadcastMapping(*),l)

# # @santiagobadia : I am here ... arrays of arrays ...
# bl = apply_mapping(composition,op,test_basis_array,fl)
# test_mapped_array(bl,xl,fill(fill(z*v,np,ndofs),l))

# # @santiagobadia :  Not sure what to check with the Nothing
# # bl = operate_arrays_of_fields(Nothing,*,test_basis_array,fl)

# op = Fill(BroadcastMapping(+),l)
# bl = apply_mapping(composition,op,test_basis_array,test_basis_2_array)
# r = fill(broadcast(+,t1x,t2x),l)
# ∇r = fill(broadcast(+,∇t1x,∇t2x),l)
# test_mapped_array(bl,xl,r)
# ∇bl = apply_mapping(composition,op,∇test_basis_array,∇test_basis_2_array)
# test_mapped_array(∇bl,xl,∇r)

# op = Fill(BroadcastMapping(⋅),l)
# bl = apply_mapping(composition,op,test_basis_array,test_basis_2_array)
# r = fill(broadcast(⋅,t1x,t2x),l)
# ∇r = fill(broadcast(⋅,∇t1x,∇t2x),l)
# test_mapped_array(bl,xl,r)
# ∇bl = apply_mapping(composition,op,∇test_basis_array,∇test_basis_2_array)
# test_mapped_array(∇bl,xl,∇r)

# trial_basis_array = trialize_array_of_bases(operate_arrays_of_fields(*,test_basis_array,fl))

# trial_basis_array_x = evaluate(trial_basis_array,xl)
# @test trial_basis_array_x.g.value === trialize_basis_value

# bl = operate_arrays_of_fields(⋅,trial_basis_array,test_basis_array)
# r = fill(fill(z*v⋅v,np,ndofs,ndofs),l)
# bl_x = evaluate(bl,xl)
# test_array_of_fields(bl,xl,r)
# @test bl_x.g.value == FieldOpKernel(⋅)


# Operations between values

# al = [rand(np,ndofs) for k in 1:l]
# bl = [rand(np) for k in 1:l]
# cl = [rand(np,ndofs) for k in 1:l]

# f(a,b) = 2*a-b*a
# af = fill(FunctionMapping(f),l)
# dl = apply_mapping(af,bl,bl)
# @santiagobadia : Errors in the algebraic operations
# test_array(dl,map((a,b)->f.(a,b),bl,bl))

# f(a,b) = 2*a-b*a
# dl = apply_mapping(FunctionMapping(f),al,bl)
# test_array(dl,map((a,b)->f.(a,b),al,bl))

# f(a,b) = 2*a-b
# dl = apply_mapping(FunctionMapping(f),al,cl)
# test_array(dl,map((a,b)->f.(a,b),al,bl))

# f(a,b,c) = b*(2*a-c)
# dl = apply_mapping(FunctionMapping(f),al,bl,cl)
# test_array(dl,map((a,b,c)->f.(a,b,c),al,bl,cl))

# dl = apply(trialize_basis_value,al)
# test_array(dl,map(a->reshape(a,(size(a,1),1,size(a,2))),al))

# atl = apply(trialize_basis_value,al)
# ctl = apply(trialize_basis_value,cl)

# f(a,b) = 2*a*b
# dl = apply_mapping(FunctionMapping(f),al,atl)
# test_array(dl,map((a,b)->f.(a,b),al,atl))

# f(a,c,at,ct) = 2*(a+c)*(2*at-ct)
# dl = apply(FunctionMapping(f),al,cl,atl,ctl)
# test_array(dl,map((a,c,at,ct)->f.(a,c,at,ct),al,cl,atl,ctl))

# @test size(dl[1]) == (np,ndofs,ndofs)

# Blocks

# blocks = (al,)
# blockids = [(1,1)]
# axs_i = (blockedrange([np]),blockedrange([ndofs,ndofs]))
# axs = Fill(axs_i,l)
# aBl = VectorOfBlockArrayCoo(blocks,blockids,axs)

# blocks = (cl,)
# blockids = [(1,2)]
# cBl = VectorOfBlockArrayCoo(blocks,blockids,axs)

# atBl = apply(trialize_basis_value,aBl)
# ctBl = apply(trialize_basis_value,cBl)
# @test isa(atBl,VectorOfBlockArrayCoo)

# f(a) = 2*a
# dl = apply(FieldOpKernel(f),aBl)
# @test isa(dl,VectorOfBlockArrayCoo)
# test_array(dl,map((a)->f.(a),aBl))

# f(a,b) = 2*a + a*b
# dl = apply(FieldOpKernel(f),aBl,bl)
# @test isa(dl,VectorOfBlockArrayCoo)
# test_array(dl,map((a,b)->f.(a,b),aBl,bl))

# f(b,a) = 2*a + a*b
# dl = apply(FieldOpKernel(f),bl,atBl)
# @test isa(dl,VectorOfBlockArrayCoo)
# test_array(dl,map((a,b)->f.(a,b),bl,atBl))

# f(b,a) = 2*a + b
# dl = apply(FieldOpKernel(f),atBl,ctBl)
# @test isa(dl,VectorOfBlockArrayCoo)
# test_array(dl,map((a,b)->f.(a,b),atBl,ctBl))

# dl = apply(FieldOpKernel(+),atBl,ctBl)
# @test isa(dl,VectorOfBlockArrayCoo)
# test_array(dl,map((a,b)->a+b,atBl,ctBl))

# dl = apply(FieldOpKernel(-),atBl,ctBl)
# @test isa(dl,VectorOfBlockArrayCoo)
# test_array(dl,map((a,b)->a-b,atBl,ctBl))

# f(b,a) = 2*a*b
# dl = apply(FieldOpKernel(f),aBl,ctBl)
# @test isa(dl,VectorOfBlockArrayCoo)
# test_array(dl,map((a,b)->f.(a,b),aBl,ctBl))

# f(b,a) = 2*a*b
# dl = apply(FieldOpKernel(f),atBl,cBl)
# @test isa(dl,VectorOfBlockArrayCoo)
# test_array(dl,map((a,b)->f.(a,b),atBl,cBl))

# BlockWise integration

# cj = [ fill(TensorValue(1,0,0,2),np) for cell in 1:l ]
# cw = [ rand(np) for cell in 1:l ]

# f(a,b) = 2*a + a*b
# vl = apply(FieldOpKernel(f),aBl,bl)
# dl = apply(IntKernel(),vl,cw,cj)
# @test isa(dl,VectorOfBlockArrayCoo)
# test_array(dl,map( (v,w,j) -> reshape(sum( broadcast(*,v,w,det.(j)), dims=1),(2*ndofs,)), vl,cw,cj ))

# vl = apply(FieldOpKernel(*),aBl,ctBl)
# dl = apply(IntKernel(),vl,cw,cj)
# @test isa(dl,VectorOfBlockArrayCoo)
# test_array(dl,map( (v,w,j) -> reshape(sum( broadcast(*,v,w,det.(j)), dims=1),(2*ndofs,2*ndofs)), vl,cw,cj ))

# Blocks of Blocks

# a0Bl = zeros_like(aBl)
# c0Bl = zeros_like(cBl)
# @test isa(a0Bl,VectorOfBlockArrayCoo)
# @test blocksize(a0Bl) == blocksize(aBl)
# @test length(a0Bl.blocks) == 0

# blockids = [(1,1)]
# ax1 = blockedrange([np])
# ax2 = blockedrange([ndofs,ndofs])
# axs_i  = (blockedrange([ax1]),blockedrange([ax2,ax2]))
# axs = Fill(axs_i,l)
# aLl = VectorOfBlockArrayCoo((aBl,),blockids,axs,(a0Bl,))
# cLl = VectorOfBlockArrayCoo((cBl,),blockids,axs,(c0Bl,))

# @test isa(aLl,VectorOfBlockArrayCoo)
# @test aBl === aLl[Block(1,1)]
# @test al === aLl[Block(1,1)][Block(1,1)]
# @test isa(aLl[Block(1,1)],VectorOfBlockArrayCoo)
# @test isa(aLl[Block(1,2)],VectorOfBlockArrayCoo)

# blockids = [(1,2)]
# aRl = VectorOfBlockArrayCoo((aBl,),blockids,axs,(a0Bl,))
# cRl = VectorOfBlockArrayCoo((cBl,),blockids,axs,(c0Bl,))

# a0tBl = zeros_like(atBl)
# c0tBl = zeros_like(ctBl)

# blockids = [(1,1,1)]
# axs_i  = (blockedrange([ax1]),blockedrange([blockedrange([1])]),blockedrange([ax2,ax2]))
# axs = Fill(axs_i,l)
# atLl = VectorOfBlockArrayCoo((atBl,),blockids,axs,(a0tBl,))
# ctLl = VectorOfBlockArrayCoo((ctBl,),blockids,axs,(c0tBl,))

# @test isa(atLl,VectorOfBlockArrayCoo)
# @test atBl === atLl[Block(1,1,1)]
# @test atl === atLl[Block(1,1,1)][Block(1,1,1)]
# @test isa(atLl[Block(1,1,1)],VectorOfBlockArrayCoo)
# @test isa(atLl[Block(1,1,2)],VectorOfBlockArrayCoo)

# blockids = [(1,1,2)]
# atRl = VectorOfBlockArrayCoo((atBl,),blockids,axs,(a0tBl,))
# ctRl = VectorOfBlockArrayCoo((ctBl,),blockids,axs,(c0tBl,))

# f(a) = 2*a
# dl = apply(FieldOpKernel(f),aLl)
# @test isa(dl,VectorOfBlockArrayCoo)
# @test isa(dl[Block(1,1)],VectorOfBlockArrayCoo)
# @test isa(dl[Block(1,2)],VectorOfBlockArrayCoo)
# @test isa(dl[1][Block(1,1)],BlockArrayCoo)
# @test isa(dl[1][Block(1,2)],BlockArrayCoo)
# test_array(dl,map((a)->f.(a),aLl))

# f(a,b) = 2*a + a*b
# dl = apply(FieldOpKernel(f),aRl,bl)
# @test isa(dl,VectorOfBlockArrayCoo)
# @test isa(dl[Block(1,1)],VectorOfBlockArrayCoo)
# @test isa(dl[Block(1,2)],VectorOfBlockArrayCoo)
# @test isa(dl[1][Block(1,1)],BlockArrayCoo)
# @test isa(dl[1][Block(1,2)],BlockArrayCoo)
# test_array(dl,map((a,b)->f.(a,b),aRl,bl))

# f(b,a) = 2*a + b
# dl = apply(FieldOpKernel(f),atRl,ctLl)
# @test isa(dl,VectorOfBlockArrayCoo)
# @test isa(dl[Block(1,1,1)],VectorOfBlockArrayCoo)
# @test isa(dl[Block(1,1,2)],VectorOfBlockArrayCoo)
# @test isa(dl[1][Block(1,1,1)],BlockArrayCoo)
# @test isa(dl[1][Block(1,1,2)],BlockArrayCoo)
# test_array(dl,map((a,b)->f.(a,b),atRl,ctLl))

# f(b,a) = 2*a + b
# dl = apply(FieldOpKernel(f),aRl,cLl)
# @test isa(dl,VectorOfBlockArrayCoo)
# @test isa(dl[Block(1,1)],VectorOfBlockArrayCoo)
# @test isa(dl[Block(1,2)],VectorOfBlockArrayCoo)
# @test isa(dl[1][Block(1,1)],BlockArrayCoo)
# @test isa(dl[1][Block(1,2)],BlockArrayCoo)
# test_array(dl,map((a,b)->f.(a,b),aRl,cLl))

# f(b,a) = 2*a + b
# dl = apply(FieldOpKernel(f),aRl,aRl)
# @test isa(dl,VectorOfBlockArrayCoo)
# @test isa(dl[Block(1,1)],VectorOfBlockArrayCoo)
# @test isa(dl[Block(1,2)],VectorOfBlockArrayCoo)
# @test isa(dl[1][Block(1,1)],BlockArrayCoo)
# @test isa(dl[1][Block(1,2)],BlockArrayCoo)
# test_array(dl,map((a,b)->f.(a,b),aRl,aRl))

# f(b,a) = a * b
# dl = apply(FieldOpKernel(f),aLl,ctRl)
# @test isa(dl,VectorOfBlockArrayCoo)
# @test is_zero_block(dl,Block(1,1,1))
# @test is_nonzero_block(dl,Block(1,1,2))
# @test is_zero_block(dl,Block(1,2,1))
# @test is_zero_block(dl,Block(1,2,2))
# @test isa(dl[Block(1,1,1)],VectorOfBlockArrayCoo)
# @test isa(dl[Block(1,1,2)],VectorOfBlockArrayCoo)
# @test isa(dl[Block(1,2,1)],VectorOfBlockArrayCoo)
# @test isa(dl[Block(1,2,2)],VectorOfBlockArrayCoo)
# @test isa(dl[1][Block(1,1,1)],BlockArrayCoo)
# @test isa(dl[1][Block(1,1,2)],BlockArrayCoo)
# @test isa(dl[1][Block(1,2,1)],BlockArrayCoo)
# @test isa(dl[1][Block(1,2,2)],BlockArrayCoo)
# test_array(dl,map((a,b)->f.(a,b),aLl,ctRl))

# f(b,a) = a * b
# v = apply(FieldOpKernel(-),ctLl,ctRl)
# dl = apply(FieldOpKernel(f),v,aLl)
# @test isa(dl,VectorOfBlockArrayCoo)
# @test is_nonzero_block(dl,Block(1,1,1))
# @test is_nonzero_block(dl,Block(1,1,2))
# @test is_zero_block(dl,Block(1,2,1))
# @test is_zero_block(dl,Block(1,2,2))
# @test isa(dl[Block(1,1,1)],VectorOfBlockArrayCoo)
# @test isa(dl[Block(1,1,2)],VectorOfBlockArrayCoo)
# @test isa(dl[Block(1,2,1)],VectorOfBlockArrayCoo)
# @test isa(dl[Block(1,2,2)],VectorOfBlockArrayCoo)
# @test isa(dl[1][Block(1,1,1)],BlockArrayCoo)
# @test isa(dl[1][Block(1,1,2)],BlockArrayCoo)
# @test isa(dl[1][Block(1,2,1)],BlockArrayCoo)
# @test isa(dl[1][Block(1,2,2)],BlockArrayCoo)
# test_array(dl,map((a,b)->f.(a,b),v,aLl))

# # Integration of Blocks of Blocks

# f(a,b) = 2*a + a*b
# vl = apply(FieldOpKernel(f),aRl,bl)
# dl = apply(IntKernel(),vl,cw,cj)
# test_array(dl,map( (v,w,j) -> reshape(sum( broadcast(*,v,w,det.(j)), dims=1),(4*ndofs,)), vl,cw,cj ))
# @test isa(dl,VectorOfBlockArrayCoo)
# @test is_zero_block(dl,Block(1))
# @test is_nonzero_block(dl,Block(2))
# @test isa(dl[Block(1)],VectorOfBlockArrayCoo)
# @test isa(dl[Block(2)],VectorOfBlockArrayCoo)

# f(b,a) = a * b
# v = apply(FieldOpKernel(-),ctLl,ctRl)
# vl = apply(FieldOpKernel(f),v,aLl)
# dl = apply(IntKernel(),vl,cw,cj)
# test_array(dl,map( (v,w,j) -> reshape(sum( broadcast(*,v,w,det.(j)), dims=1),(4*ndofs,4*ndofs)), vl,cw,cj ))
# @test isa(dl,VectorOfBlockArrayCoo)
# @test is_nonzero_block(dl,Block(1,1))
# @test is_nonzero_block(dl,Block(1,2))
# @test is_zero_block(dl,Block(2,1))
# @test is_zero_block(dl,Block(2,2))
# @test isa(dl[Block(1,1)],VectorOfBlockArrayCoo)
# @test isa(dl[Block(1,2)],VectorOfBlockArrayCoo)
# @test isa(dl[Block(2,1)],VectorOfBlockArrayCoo)
# @test isa(dl[Block(2,2)],VectorOfBlockArrayCoo)

# #using BenchmarkTools
# #cache = array_cache(dl)
# #@btime getindex!($cache,$dl,3)

end # module
