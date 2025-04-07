module PLambdaBasisTests

using Test
using Gridap.TensorValues
using Gridap.Fields
using Gridap.Arrays
using Gridap.Polynomials
using ForwardDiff
using StaticArrays
using BenchmarkTools
using ProfileView

import Gridap.Arrays.evaluate!
T = Float64


#using Gridap.Polynomials: PΛ_bubble_indices, P⁻Λ_bubble_indices, complement, sorted_combinations, _hat_Ψ
#export _bench
#function _bench(D=8,k=4)
#  M = rand(SMatrix{D,D,Float64})
#  nb_perms = binomial(D,k)
#  m = MMatrix{nb_perms,nb_perms,Float64}(undef)
#  Vk = Val(k)
#  @btime all_k_minors!($m,$M,$(Vk))
#end
#
#export P_bubles
#function P_bubles(;r=2,k=2,D=3)
#  for (d, F, dF_bubbles) in PΛ_bubble_indices(r,k,D)
#    #s = Set()
#    println("d = $d, F=$F, F*=$(complement(F))")
#    for (i, α, J) in dF_bubbles
#      println("i=$i, α=$α, J=$J")
#      for I in sorted_combinations(D,k)
#        M = MMatrix{k,k,Float64}(undef)
#        for (i, Ii) in enumerate(I)
#          for (j, Jj) in enumerate(J)
#            @inbounds M[i,j] = Int((Ii+1)==Jj) - Int(Jj==1)
#          end
#        end
#        s = Int(isone(J[1]))
#        n = count(i-> (J[i]-1)∉I, (s+1):length(J))
#        p = _findfirst_or_zero(j-> (I[j]+1)∉J, 1,length(J))
#        m = _findfirst_or_zero(i-> (J[i]-1)∉I, (s+1),length(J))
#        q = _findfirst_or_zero(j-> (I[j]+s)∉J, (p+1),length(J))
#        println("\tI=$I, J=$J, M_IJ=$M, Ψ[I,J] = $(_hat_Ψ(r,α,F,I,J,Float64))")
#        println("\ts=$s, n=$n, p=$p, m=$m, q=$q")
#        println()
#      end
#      #if α in s
#      #  println("  REDUNDANT")
#      #else
#      #  println()
#      #  push!(s,α)
#      #end
#    end
#    println()
#  end
#end
#function _findfirst_or_zero(pred, start, endd)
#  r = findfirst(pred,start:endd)
#  return isnothing(r) ? 0 : r+start-1
#end
#
#export Pm_bubles
#function Pm_bubles(;r=2,k=2,D=3)
#  for (d, F, dF_bubbles) in P⁻Λ_bubble_indices(r,k,D)
#    println("d = $d, F=$F, F*=$(complement(F))")
#    for (i, α, J) in dF_bubbles
#      println("i=$i, α=$α, J=$J")
#      #for (l,J_l) in enumerate(sub_combinations(J))
#        #println("sgn=$(-(-1)^l), J[l]=$(J[l]), J\\l=$(J_l)")
#      #end
#    end
#    println()
#  end
#end

# 0D                                           0D #
D = 0
vertices = (Point{D,T}(),)
x = [vertices[1]]
x1 = x[1]

k = 0
r = 1
b = PLambdaBasis(Val(D),T,r,k)
evaluate(b,x)
evaluate(Broadcasting(∇)(b),x)
evaluate(Broadcasting(∇∇)(b),x)
evaluate(Broadcasting(𝑑)(b),x)

r = 2
b = PLambdaBasis(Val(D),T,r,k)
evaluate(b,x)
evaluate(Broadcasting(∇)(b),x)
evaluate(Broadcasting(∇∇)(b),x)
evaluate(Broadcasting(𝑑)(b),x)

r = 3
b = PLambdaBasis(Val(D),T,r,k)
evaluate(b,x)
evaluate(Broadcasting(∇)(b),x)
evaluate(Broadcasting(∇∇)(b),x)
evaluate(Broadcasting(𝑑)(b),x)

r = 4
b = PLambdaBasis(Val(D),T,r,k)
evaluate(b,x)
evaluate(Broadcasting(∇)(b),x)
evaluate(Broadcasting(∇∇)(b),x)
evaluate(Broadcasting(𝑑)(b),x)

b = PLambdaBasis(Val(D),T,r,k,vertices)
evaluate(b,x)
evaluate(Broadcasting(∇)(b),x)
evaluate(Broadcasting(∇∇)(b),x)
evaluate(Broadcasting(𝑑)(b),x)

# 1D                                           1D #
D = 1
Pt = Point{D,T}
vertices = (Pt(.5),Pt(1.))
x = [xi for xi in vertices]
x1 = x[1]

r = 4
k = 0
b = PLambdaBasis(Val(D),T,r,k)
evaluate(b,x)
evaluate(Broadcasting(∇)(b),x)
evaluate(Broadcasting(∇∇)(b),x)
evaluate(Broadcasting(𝑑)(b),x)

b = PLambdaBasis(Val(D),T,r,k,vertices)
evaluate(b,x)
evaluate(Broadcasting(∇)(b),x)
evaluate(Broadcasting(∇∇)(b),x)
evaluate(Broadcasting(𝑑)(b),x)

k = 1
b = PLambdaBasis(Val(D),T,r,k)
evaluate(b,x)
evaluate(Broadcasting(∇)(b),x)
evaluate(Broadcasting(∇∇)(b),x)
evaluate(Broadcasting(𝑑)(b),x)

b = PLambdaBasis(Val(D),T,r,k,vertices)
evaluate(b,x)
evaluate(Broadcasting(∇)(b),x)
evaluate(Broadcasting(∇∇)(b),x)
evaluate(Broadcasting(𝑑)(b),x)

# 2D                                           2D #
D = 2
Pt = Point{D,T}
vertices = (Pt(0., 0.5),Pt(1.,0),Pt(.5,1.))
x = [xi for xi in vertices]
x1 = x[1]

r = 4
k = 0
b = PLambdaBasis(Val(D),T,r,k)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
dbx = evaluate(Broadcasting(𝑑)(b),x)

V = eltype(dbx)
Gbx = reinterpret(V, Gbx)
@test all(Gbx .≈ dbx)

b = PLambdaBasis(Val(D),T,r,k,vertices)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
dbx = evaluate(Broadcasting(𝑑)(b),x)

V = eltype(dbx)
Gbx = reinterpret(V, Gbx)
@test all(Gbx .≈ dbx)

k = 1
b = PLambdaBasis(Val(D),T,r,k)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
#dbx = evaluate(Broadcasting(𝑑)(b),x)
cbx = evaluate(Broadcasting(curl)(b),x)

#V = eltype(dbx)
#cbx = reinterpret(V, cbx)
#@test all(cbx .≈ dbx)

b = PLambdaBasis(Val(D),T,r,k,vertices)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
#dbx = evaluate(Broadcasting(𝑑)(b),x)
cbx = evaluate(Broadcasting(curl)(b),x)

#V = eltype(dbx)
#cbx = reinterpret(V, cbx)
#@test all(cbx .≈ dbx)

k = 2
b = PLambdaBasis(Val(D),T,r,k)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
dbx = evaluate(Broadcasting(𝑑)(b),x)

b = PLambdaBasis(Val(D),T,r,k,vertices)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
dbx = evaluate(Broadcasting(𝑑)(b),x)


# 3D                                           3D #
D = 3
Pt = Point{D,T}
vertices = (Pt(0., 0., 0.5),Pt(1.,0,0),Pt(0,.5,0),Pt(0,.5,.5))
x = [xi for xi in vertices]
x1 = x[1]

r = 4
k = 0
b = PLambdaBasis(Val(D),T,r,k)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
dbx = evaluate(Broadcasting(𝑑)(b),x)

V = eltype(dbx)
Gbx = reinterpret(V, Gbx)
@test all(Gbx .≈ dbx)

b = PLambdaBasis(Val(D),T,r,k,vertices)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
dbx = evaluate(Broadcasting(𝑑)(b),x)

V = eltype(dbx)
Gbx = reinterpret(V, Gbx)
@test all(Gbx .≈ dbx)

k = 1
function hodge_grad2curl_3D(∇u::ArrayMultiValue{Tuple{3,3},T}) where T
  @inbounds begin
    c1 = ∇u[2,3] - ∇u[3,2]
    c2 = ∇u[3,1] - ∇u[1,3]
    c3 = ∇u[1,2] - ∇u[2,1]
    VectorValue(c3,-c2,c1)
  end
end
hodge_and_curl(f) = Operation(hodge_grad2curl_3D)(∇(f))
function Arrays.evaluate!(cache,::Broadcasting{typeof(hodge_and_curl)},f)
  Broadcasting(Operation(hodge_grad2curl_3D))(Broadcasting(∇)(f))
end
b = PLambdaBasis(Val(D),T,r,k)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
dbx = evaluate(Broadcasting(𝑑)(b),x)
cbx = evaluate(Broadcasting(hodge_and_curl)(b),x)

V = eltype(dbx)
cbx = reinterpret(V, cbx)
@test all(cbx .≈ dbx)

b = PLambdaBasis(Val(D),T,r,k,vertices)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
dbx = evaluate(Broadcasting(𝑑)(b),x)
cbx = evaluate(Broadcasting(hodge_and_curl)(b),x)

V = eltype(dbx)
cbx = reinterpret(V, cbx)
@test all(cbx .≈ dbx)

k = 2
function hodge_tr_3D(v::ArrayMultiValue{Tuple{3,3},T}) where T
  @inbounds v[3,1] - v[2,2] + v[1,3]
end
hodge_and_div(f) = Operation(hodge_tr_3D)(∇(f))
function Arrays.evaluate!(cache,::Broadcasting{typeof(hodge_and_div)},f)
  Broadcasting(Operation(hodge_tr_3D))(Broadcasting(∇)(f))
end

b = PLambdaBasis(Val(D),T,r,k)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)

#rr, s, c = return_cache(b,x)
#np = length(x)
#Polynomials._setsize!(b,np,rr,c)
#parms = Polynomials._get_static_parameters(b)
#Polynomials._evaluate_nd!(b,x1,rr,1,c,parms)
#@code_warntype Polynomials._evaluate_nd!(b,x1,rr,1,c,parms)
#@benchmark Polynomials._evaluate_nd!($b,$x1,$rr,1,$c,$parms)
#
#rr, s, c, g = return_cache(Broadcasting(∇)(b),x)
#np = length(x)
#Polynomials._setsize!(b,np,rr,c,g)
#parms = Polynomials._get_static_parameters(b)
#Polynomials._gradient_nd!(b,x1,rr,1,c,g,s,parms)
#@code_warntype Polynomials._gradient_nd!(b,x1,rr,1,c,g,s,parms)
#@benchmark Polynomials._gradient_nd!($b,$x1,$rr,1,$c,$g,$s,$parms)
#VSCodeServer.@profview for _ in 1:100000 Polynomials._gradient_nd!(b,x1,rr,1,c,g,s,parms) end
#VSCodeServer.@profview_allocs for _ in 1:100000 Polynomials._gradient_nd!(b,x1,rr,1,c,g,s,parms) end
#
#rr, s, c, g = return_cache(Broadcasting(𝑑)(b),x)
#np = length(x)
#Polynomials._setsize!(b,np,rr,c,g)
#parms = Polynomials._get_static_parameters(b)
#Polynomials._exterior_derivative_nd!(b,x1,rr,1,c,g,s,parms)
#@code_warntype Polynomials._exterior_derivative_nd!(b,x1,rr,1,c,g,s,parms)
#@benchmark Polynomials._exterior_derivative_nd!($b,$x1,$rr,1,$c,$g,$s,$parms)
#VSCodeServer.@profview for _ in 1:10000 Polynomials._exterior_derivative_nd!(b,x1,rr,1,c,g,s,parms) end
#VSCodeServer.@profview_allocs for _ in 1:10000 Polynomials._exterior_derivative_nd!(b,x1,rr,1,c,g,s,parms) end
#
#const xx = [rand(vertices) for _ in 1:100]

#cdb = return_cache(Broadcasting(∇)(b),x)
#@profview for _ in 1:1000 evaluate!(cdb,Broadcasting(∇)(b),x) end
#
#r, s, c, g = cdb
#np = length(x)
#Polynomials._setsize!(b,np,r,c,g)
#Polynomials._gradient_nd!(b,x1,r,1,c,g,s)
#@code_warntype Polynomials._gradient_nd!(b,x1,r,1,c,g,s)
#
#const xx = [xi for xi in vertices]
#const bb = PLambdaBasis(Val(D),T,r,k)
#const 𝑑bb = Broadcasting(𝑑)(bb)
#const dbb = Broadcasting(hodge_and_div)(bb)
#const c𝑑bb = return_cache(𝑑bb,xx)
#const cdbb = return_cache(dbb,xx)

#@benchmark evaluate!($c𝑑bb,$𝑑bb,$xx)
#@benchmark evaluate!($cdbb,$dbb,$xx)

#@btime evaluate!($c𝑑bb,$𝑑bb,$xx)
#@btime evaluate!($cdbb,$dbb,$xx)
#
#dbx = reinterpret(Float64, dbx)
#@test all(divbx .≈ dbx)


b = PLambdaBasis(Val(D),T,r,k,vertices)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
#dbx = evaluate(Broadcasting(𝑑)(b),x)
divbx=evaluate(Broadcasting(hodge_and_div)(b),x)

#dbx = reinterpret(Float64, dbx)
#@test all(divbx .≈ dbx)


k = 3
b = PLambdaBasis(Val(D),T,r,k)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
dbx = evaluate(Broadcasting(𝑑)(b),x)

b = PLambdaBasis(Val(D),T,r,k,vertices)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
dbx = evaluate(Broadcasting(𝑑)(b),x)

# 4D                                           4D #
D = 4
Pt = Point{D,T}
vertices = (Pt(0.,0.,0.,0.),Pt(0.,0.,0.,0.5),Pt(0.,1.,0,0),Pt(0.,0,.5,0),Pt(.5,1,1,1))
x = [xi for xi in vertices]
x1 = x[1]

r = 4
k = 0
b = PLambdaBasis(Val(D),T,r,k)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
dbx = evaluate(Broadcasting(𝑑)(b),x)

V = eltype(dbx)
Gbx = reinterpret(V, Gbx)
@test all(Gbx .≈ dbx)

b = PLambdaBasis(Val(D),T,r,k,vertices)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
dbx = evaluate(Broadcasting(𝑑)(b),x)

V = eltype(dbx)
Gbx = reinterpret(V, Gbx)
@test all(Gbx .≈ dbx)

k = 1
b = PLambdaBasis(Val(D),T,r,k)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
dbx = evaluate(Broadcasting(𝑑)(b),x)

b = PLambdaBasis(Val(D),T,r,k,vertices)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
dbx = evaluate(Broadcasting(𝑑)(b),x)

k = 3
#k = 2
#b = PLambdaBasis(Val(D),T,r,k)
#bx  = evaluate(b,x)
#Gbx = evaluate(Broadcasting(∇)(b),x)
#Hbx = evaluate(Broadcasting(∇∇)(b),x)
#dbx = evaluate(Broadcasting(𝑑)(b),x)
#
#
#b = PLambdaBasis(Val(D),T,r,k,vertices)
#bx  = evaluate(b,x)
#Gbx = evaluate(Broadcasting(∇)(b),x)
#Hbx = evaluate(Broadcasting(∇∇)(b),x)
#dbx = evaluate(Broadcasting(𝑑)(b),x)

k = 3
function hodge_tr_4D(∇u::ArrayMultiValue{Tuple{4,4}})
  @inbounds -∇u[4,1] + ∇u[3,2] - ∇u[2,3] + ∇u[1,4]
end
hodge4D_and_div(f) = Operation(hodge_tr_4D)(∇(f))
function Arrays.evaluate!(cache,::Broadcasting{typeof(hodge4D_and_div)},f)
  Broadcasting(Operation(hodge_tr_4D))(Broadcasting(∇)(f))
end

b = PLambdaBasis(Val(D),T,r,k)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
#dbx = evaluate(Broadcasting(𝑑)(b),x)
divbx=evaluate(Broadcasting(hodge4D_and_div)(b),x)

#dbx = reinterpret(Float64, dbx)
#@test all(divbx .≈ dbx)


b = PLambdaBasis(Val(D),T,r,k,vertices)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
#dbx = evaluate(Broadcasting(𝑑)(b),x)
divbx=evaluate(Broadcasting(hodge4D_and_div)(b),x)

#dbx = reinterpret(Float64, dbx)
#@test all(divbx .≈ dbx)

k = 4
b = PLambdaBasis(Val(D),T,r,k)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
dbx = evaluate(Broadcasting(𝑑)(b),x)


b = PLambdaBasis(Val(D),T,r,k,vertices)
bx  = evaluate(b,x)
Gbx = evaluate(Broadcasting(∇)(b),x)
Hbx = evaluate(Broadcasting(∇∇)(b),x)
dbx = evaluate(Broadcasting(𝑑)(b),x)


end # module

#using .PLambdaBasisTests
#
#function α_to_I(::NTuple{0,Int})
#  return Combination{0,0}()
#end
#function α_to_I(α::NTuple{k,Int}) where k
#  r = sum(α)
#  v = zero(MVector{r,Int})
#  i_v = 1
#  i = 1
#  for α_i in α
#    for _ in 1:α_i
#      v[i_v] = i
#      i += 1
#      i_v += 1
#    end
#    i += 1
#  end
#  #println(r,k,r+k-1,v)
#  return Gridap.Polynomials.Combination{r,k+r-1}(v)
#end
