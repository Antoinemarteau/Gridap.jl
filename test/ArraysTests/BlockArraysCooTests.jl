module BlockArraysCooTests

using Test
using Gridap.Arrays
using BlockArrays
using LinearAlgebra

# Check MultiLevelBlockedUnitRange

r1 = Base.OneTo(2)
r2 = Base.OneTo(3)

r = MultiLevelBlockedUnitRange([r1,r2])
@test length(r) == 5
@test r.local_ranges[1] === r1
@test r.local_ranges[2] === r2

s1 = Base.OneTo(1)
s2 = Base.OneTo(2)
s3 = Base.OneTo(0)

s = MultiLevelBlockedUnitRange([s1,s2,s3])
@test length(s) == 3

o = MultiLevelBlockedUnitRange([r,s])
@test length(o) == 8
@test allblocksequal(o,o)

a = MultiLevelBlockedUnitRange([r,s])
@test allblocksequal(o,a)

b = MultiLevelBlockedUnitRange([s,s])
@test !allblocksequal(a,b)

r1_axes = (r1,)
r2_axes = (r2,)
s1_axes = (s1,)
s2_axes = (s2,)

r_axes = (MultiLevelBlockedUnitRange(map(first,[r1_axes,r2_axes])),)
s_axes = (MultiLevelBlockedUnitRange(map(first,[s1_axes,s2_axes])),)
o_axes = (MultiLevelBlockedUnitRange(map(first,[r_axes,s_axes])),)
@test allblocksequal(o_axes,o_axes)

# BlockArrayCoo

# Minimal constructor

blocks = Matrix{Float64}[ [1 2; 3 4], [5 6; 7 8; 9 10] ]
blockids = [(1,1),(2,1)]
ax = (blockedrange([2,3]), blockedrange([2,4]))
a = BlockArrayCoo(ax,blockids,blocks)
a = BlockArrayCoo(ax,blockids,blocks...)

# Specific API

@test is_zero_block(rand(4,5),1,1) == false
@test is_zero_block(a,2,2) == true
@test is_zero_block(a,2,1) == false
@test is_nonzero_block(a,2,1) == true
@test is_zero_block(a,Block(1,2)) == true
@test is_zero_block(a,Block(1),Block(2)) == true

# AbstractBlockArray interface

@test a[Block(1),Block(1)] === blocks[1]
@test a[Block(1,1)] === blocks[1]
@test a[Block(1,2)] === a.zero_blocks[1]

c = copy(a[Block(1,1)])
fill!(c,0)

getblock!(c,a,1,1)
@test c == a[Block(1,1)]
@test a[Block(1,1)] === blocks[1]

c = copy(a)
@test isa(c,BlockArrayCoo)
@test c == a
c[Block(1,1)] = a[Block(1,1)]
@test c[Block(1,1)] === a[Block(1,1)]
@test a[Block(1,1)] === blocks[1]

@test a[BlockIndex((1,1),(2,1))] === blocks[1][2,1]
@test a[BlockIndex(1,2),BlockIndex(1,1)] === blocks[1][2,1]
@test a[2,1] === blocks[1][2,1]
@test a[3,2] === blocks[2][1,2]
@test axes(a) === ax
@test size(a) == (5,6)

for (i,b) in enumerateblocks(a)
  @test a[i] === b
end

b21 = zeros(3,2)
getblock!(b21,a,Block(2,1))
@test b21 == a[Block(2,1)]

b12 = ones(2,4)
getblock!(b12,a,Block(1,2))
@test b12 == a[Block(1,2)]

blocks = Vector{Float64}[ [1,2,3] ]
blockids = [(2,)]
ax = (blockedrange([2,3,4]),)
a = BlockArrayCoo(ax,blockids,blocks)

@test a[Block(1)] === a.zero_blocks[1]
@test a[Block(2)] === blocks[1]
@test a[Block(3)] === a.zero_blocks[2]
@test a[BlockIndex(2,3)] === blocks[1][3]

blocks = Matrix{Float64}[ [1 2; 3 4], [5 6 7 8; 9 10 11 12; 13 14 15 16], [1 2 3 4; 5 6 7 8], [1 2 3; 4 5 6; 7 8 9] ]
blockids = [(1,1),(2,2),(1,2),(3,3)]
ax = (blockedrange([2,3,3]), blockedrange([2,4,3]))
a = BlockArrayCoo(ax,blockids,blocks)

blocks = Vector{Float64}[ 10*[1,2], 20*[1,2,3] ]
blockids = [(1,),(3,)]
ax = (blockedrange([2,4,3]),)
b = BlockArrayCoo(ax,blockids,blocks)

# similar

c = similar(a)
@test typeof(c) == typeof(a)
@test axes(a) == axes(c)
@test a.blockids == c.blockids

c = similar(a,Int)
@test axes(a) == axes(c)
@test a.blockids == c.blockids

ax = (blockedrange([2,3,3]), blockedrange([2,4,3]))
c = similar(a,eltype(a),ax)
@test typeof(c) == typeof(a)
@test axes(a) == axes(c)
@test a.blockids == c.blockids

ax = (blockedrange([2,5,3]), blockedrange([2,4,1]))
c = similar(a,eltype(a),ax)
@test typeof(c) == typeof(a)
@test axes(a) != axes(c)
@test axes(c) == ax
@test a.blockids != c.blockids
@test length(c.blockids) == 9

c = a*b
@test axes(c,1) === axes(a,1)
@test blocksize(c) == (3,)
@test Array(a)*Array(b) == c

display(c)

kk

mul!(c,a,b)
@test axes(c,1) === axes(a,1)
@test blocksize(c) == (3,)
@test Array(a)*Array(b) == c

d = copy(c)
mul!(d,a,b,2,3)
@test axes(d,1) === axes(a,1)
@test blocksize(d) == (3,)
@test 2*Array(a)*Array(b) + 3*Array(c) == d

b = transpose(a)
@test isa(b,BlockArrayCoo)
c = a*b
@test axes(c,1) === axes(a,1)
@test axes(c,2) === axes(a,1)
@test blocksize(c) == (3,3)
@test Array(a)*Array(b) == c

mul!(c,a,b)
@test axes(c,1) === axes(a,1)
@test axes(c,2) === axes(a,1)
@test blocksize(c) == (3,3)
@test Array(a)*Array(b) == c

cc = CachedArray(c)

axs = (blockedrange([2,3,3]), blockedrange([2,3,3]))
setaxes!(cc,axs)
@test map(blocklasts,axes(cc.array)) == map(blocklasts,axs)
@test cc.array === c

axs = (blockedrange([4,5,3]), blockedrange([2,4,3]))
setaxes!(cc,axs)
@test map(blocklasts,axes(cc.array)) == map(blocklasts,axs)
fill!(cc.array,0)

axs = (blockedrange([5,4,3]), blockedrange([4,2,3]))
setaxes!(cc,axs)
@test map(blocklasts,axes(cc.array)) == map(blocklasts,axs)

axs1 = (blockedrange([5,4,3]), blockedrange([4,2,3]))
axs2 = (blockedrange([4,5,3]), blockedrange([2,4,3]))
axs3 = (blockedrange([4,5,3]), blockedrange([2,4,3]))
@test Arrays._same_axes(axs1,axs2) == false
@test Arrays._same_axes(axs2,axs2)
@test Arrays._same_axes(axs2,axs3)

blocks = [ 10*[1,2], 20*[1,2,3] ]
blockids = [(1,),(3,)]
axs = (blockedrange([2,4,3]),)
b = BlockArrayCoo(blocks,blockids,axs)

cb = CachedArray(b)

setaxes!(cb,axs)
@test map(blocklasts,axes(cb.array)) == map(blocklasts,axs)
@test cb.array === b

axs = (blockedrange([3,2,3]),)
setaxes!(cb,axs)
@test map(blocklasts,axes(cb.array)) == map(blocklasts,axs)
@test size(cb) == (8,)

c = copy(a)
@test isa(c,BlockArrayCoo)
@test c == a
@test axes(c) == axes(a)

fill!(c,0)
copyto!(c,a)
@test c == a
@test axes(c) == axes(a)

c = 2*a
@test isa(c,BlockArrayCoo)
@test 2*Array(a) == c

c = a*2
@test isa(c,BlockArrayCoo)
@test 2*Array(a) == c

c = a + a
@test isa(c,BlockArrayCoo)
@test 2*Array(a) == c

blocks = [ [1 2; 3 4], [5 6 7 8; 9 10 11 12; 13 14 15 16], [1 2 3 4; 5 6 7 8], [1 2 3; 4 5 6; 7 8 9] ]
blockids = [(1,1),(2,2),(1,2),(3,3)]
ax = (blockedrange([2,3,3]), blockedrange([2,4,3]))
a = BlockArrayCoo(blocks,blockids,ax)

blocks = [ [1 2; 3 4], [1 2; 4 5; 8 9], [5 6 7 8; 9 10 11 12; 13 14 15 16] ]
blockids = [(1,1),(2,1),(3,2)]
b = BlockArrayCoo(blocks,blockids,ax)

c = a + b
@test isa(c,BlockArrayCoo)
@test Array(a)+Array(b) == c

c = a - 2*b
@test isa(c,BlockArrayCoo)
@test Array(a)-2*Array(b) == c

@test a[1,2,1] == a[1,2]

blocks = [ 10*[1,2], 20*[1,2,3] ]
blockids = [(1,),(3,)]
axs = (blockedrange([2,4,3]),)
b = BlockArrayCoo(blocks,blockids,axs)

@test b[2] == b[2,1]

axs = (blockedrange([2,4,3]),)

# Blocks of Blocks

bL = BlockArrayCoo([[1,2]],[(1,)],axs)
bR = BlockArrayCoo([[2,3],[4,5,6]],[(1,),(3,)],axs)
axs = (blockedrange([axs[1],axs[1]]),)

bS = BlockArrayCoo([bL,bR],[(1,),(2,)],axs)

zS = zeros_like(bS)
@test isa(zS,BlockArrayCoo)
@test isa(zS[Block(1)],BlockArrayCoo)
@test isa(zS[Block(2)],BlockArrayCoo)
@test length(zS.blocks) == 0
@test length(zS[Block(1)].blocks) == 0
@test length(zS[Block(2)].blocks) == 0

cS = 2*bS
@test isa(cS,BlockArrayCoo)
@test isa(cS[Block(1)],BlockArrayCoo)

bZ = zeros_like(bR)

dS = BlockArrayCoo([bZ,bR],axs,[false,true])

r = cS - dS
@test isa(r,BlockArrayCoo)
@test isa(r[Block(1)],BlockArrayCoo)

r = cS + dS
@test isa(r,BlockArrayCoo)
@test isa(r[Block(1)],BlockArrayCoo)

ax = (blockedrange([2,3]), blockedrange([2,4]))
aLL = BlockArrayCoo([[1 2; 3 4],[5 6; 7 8; 9 10] ],[(1,1),(2,1)],ax)
aLR = BlockArrayCoo([[1 2 5 6; 3 4 1 2; 1 2 3 4],[5 6; 7 8; 9 10] ],[(2,2),(2,1)],ax)
aRL = zeros_like(aLL)
aRR = BlockArrayCoo([[1 2; 3 4],[5 6; 7 8; 9 10] ],[(1,1),(2,1)],ax)

allblocks = Matrix{typeof(aLL)}(undef,2,2)
allblocks[1,1] = aLL
allblocks[1,2] = aLR
allblocks[2,1] = aRL
allblocks[2,2] = aRR

mask = [true true; false true]

axs = ( blockedrange([ax[1],ax[1]]) , blockedrange([ax[2],ax[2]]))

aS = BlockArrayCoo(allblocks,axs,mask)
@test is_zero_block(aS,2,1)
@test is_nonzero_block(aS,2,2)

aS2 = copy(aS)
@test isa(aS2,BlockArrayCoo)
@test isa(aS2[Block(2),Block(2)],BlockArrayCoo)

aSt = transpose(aS)
@test Array(aSt) == transpose(Array(aS))
@test isa(aSt,BlockArrayCoo)
@test isa(aSt[Block(2),Block(2)],BlockArrayCoo)

rS = aS*aSt
@test rS == Array(aS)*Array(aSt)
@test isa(rS,BlockArrayCoo)
@test isa(rS[Block(2),Block(2)],BlockArrayCoo)

mul!(rS,aS,aSt)
@test rS == Array(aS)*Array(aSt)
@test isa(rS,BlockArrayCoo)
@test isa(rS[Block(2),Block(2)],BlockArrayCoo)

axs =(blockedrange([2,4]),)
bL = BlockArrayCoo([[1,2]],[(1,)],axs)
bR = BlockArrayCoo([[2,3],[4,5,6,8]],[(1,),(2,)],axs)
bS = BlockArrayCoo([bL,bR],[(1,),(2,)],(blockedrange([6,6]),))

rS = aS*bS
@test rS == Array(aS)*Array(bS)
@test isa(rS,BlockArrayCoo)
@test isa(rS[Block(2)],BlockArrayCoo)

mul!(rS,aS,bS)
@test rS == Array(aS)*Array(bS)
@test isa(rS,BlockArrayCoo)
@test isa(rS[Block(2)],BlockArrayCoo)

cS = copy(rS)
mul!(rS,aS,bS,3,2)
@test rS == 3*Array(aS)*Array(bS) + 2*cS
@test isa(rS,BlockArrayCoo)
@test isa(rS[Block(2)],BlockArrayCoo)

axs1 = (blockedrange([5,4,3]), blockedrange([4,2,3]))
axs2 = (blockedrange([4,5,3]), blockedrange([2,4,3]))
axsA = (blockedrange([axs1[1],axs1[1]]),blockedrange([axs2[2],axs2[2]]))
axsB = (blockedrange([axs2[1],axs2[1]]),blockedrange([axs1[2],axs1[2]]))
axsC = (blockedrange([axs2[1],axs2[1]]),blockedrange([axs1[2],axs1[2]]))
@test Arrays._same_axes(axsA,axsB) == false
@test Arrays._same_axes(axsA,axsA)
@test Arrays._same_axes(axsB,axsC)

#using BenchmarkTools
#@btime Arrays._same_axes($axsA,$axsA)

end # module
