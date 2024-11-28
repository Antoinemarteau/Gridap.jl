using Gridap, Gridap.TensorValues, Gridap.Helpers
using Test
using BenchmarkTools
using StaticArrays

gradient_type = Gridap.Fields.gradient_type

################################################
# src/Polynomials/MonomialBasis.jl: _set_value_!
################################################

_set_value! = Gridap.Polynomials._set_value!

function _set_value_new!(v::AbstractVector{V},s::T,k) where {V,T}
  ncomp = num_indep_components(V)
  z = zero(T)
  @inbounds for j in 1:ncomp
    v[k] = ntuple(i -> ifelse(i == j, s, z),Val(ncomp))
    k += 1
  end
  k
end


function set_value_benchmark(D, T, V, n)
  C = num_indep_components(V)
  x = zeros(V,n*C)

  function set_value_driver(f,T,D,x,n)
    k = 1
    s = one(T)
    for i in 1:n
      #s = T(i)
      k = f(x,s,k)
    end
  end

  b_old = @benchmark $set_value_driver($_set_value!,$T,$D,$x,$n)
  b_new = @benchmark $set_value_driver($_set_value_new!,$T,$D,$x,$n)
  judge(median(b_new), median(b_old))
end

##################################################
# src/Polynomials/ModalC0Bases.jl: _set_value_mc0!
##################################################

_set_value_mc0! = Gridap.Polynomials._set_value_mc0!

@inline function _set_value_mc0_new!(v::AbstractVector{V},s::T,k,l) where {V,T}
  ncomp = num_indep_components(V)
  z = zero(T)
  for j in 1:ncomp
    m = k+l*(j-1)
    @inbounds v[m] = ntuple(i -> ifelse(i == j, s, z),Val(ncomp))
  end
  k+1
end

function set_value_mc0_benchmark(D, T, V, n)
  C = num_indep_components(V)
  x = zeros(V,2*n*C)

  function set_value_mc0_driver(f,T,D,x,n) 
    k = 1
    s = one(T)
    for i in 1:n
      #s = T(i)
      k = f(x,s,k,2)
    end
  end

  b_old = @benchmark $set_value_mc0_driver($_set_value_mc0!,$T,$D,$x,$n)
  b_new = @benchmark $set_value_mc0_driver($_set_value_mc0_new!,$T,$D,$x,$n)
  judge(median(b_new), median(b_old))
end

###################################################
# src/Polynomials/MonomialBasis.jl: _set_gradient!
###################################################

 _set_gradient! = Gridap.Polynomials. _set_gradient!


# V :: Multivalue{D2}
# s :: VectorValue{D}
# G :: TensorValue{D,D2}
@generated function  _set_gradient_new!(v::AbstractVector{G},s,k,::Type{V}) where {V,G}
  w = zero(V)
  m = Array{String}(undef, size(G))
  N_val_dims = length(size(V))
  s_size = size(G)[1:end-N_val_dims]

  body = "T = eltype(s); z = zero(T);"
  for ci in CartesianIndices(s_size)
    id = join(Tuple(ci))
    body *= "@inbounds s$id = s[$ci];"
  end

  for j in CartesianIndices(w)
    for i in CartesianIndices(m)
      m[i] = "z"
    end
    for ci in CartesianIndices(s_size)
      id = join(Tuple(ci))
      m[ci,j] = "s$id"
    end
    body *= "@inbounds v[k] = ($(join(tuple(m...), ", ")));"
    body *= "k = k + 1;"
  end
  #println(body)

  body = Meta.parse(string("begin ",body," end"))
  return Expr(:block, body ,:(return k))
end


@generated function  _set_gradient_new!(
  v::AbstractVector{G},s,k,::Type{V}) where {V<:AbstractSymTensorValue{D},G} where D
  
  m = Array{String}(undef, size(G))
  s_length = size(G)[1]

  is_traceless = V <: SymTracelessTensorValue
  skip_last_diagval = is_traceless ? 1 : 0    # Skid V_DD if traceless
  i = 0

  body = "T = eltype(s); z = zero(T);"
  for i in 1:s_length 
    body *= "@inbounds s$i = s[$i];"
  end
  
  for c in 1:(D-skip_last_diagval) # Go over cols
    for r in c:D                   # Go over lower triangle, current col
      for i in eachindex(m)
        m[i] = "z"
      end
      for i in 1:s_length # indices of the Vector s
        m[i,r,c] = "s$i"
        if (r!=c)
          m[i,c,r] = "s$i"
        elseif is_traceless # V_rr contributes negatively to V_DD (tracelessness)
          m[i,D,D] = "-s$i"
        end
      end
      body *= "@inbounds v[k] = ($(join(tuple(m...), ", ")));"
      body *= "k = k + 1;"
    end
  end

  body = Meta.parse(string("begin ",body," end"))
  return Expr(:block, body ,:(return k))
end

function set_gradient_benchmark(D, T, V, n)
  C = num_indep_components(V)
  G = gradient_type(V, zero(Point{D,T}))
  x = zeros(G,n*C);

  function set_gradient_driver(f,T,D,V,x,n)
    k = 1
    s = VectorValue{D,T}(ntuple(_->one(T),D))
    for i in 1:n
      #s = VectorValue{D,T}(ntuple(k->i+k,D)) # validation
      k = f(x,s,k,V)
    end
  end

  #H = gradient_type(G, zero(Point{D,T}))
  #xH = zeros(H,n*C);
  function set_hessian_driver(f,T,D,V,x,n)
    k = 1
    s = TensorValue{D,D,T}(ntuple(_->one(T),D*D))
    for i in 1:n
      k = f(x,s,k,V)
    end
  end

  #println(x)
  #println();println(x)
  #x = zeros(G,n*C);
  #println();println(x)

  # warmup 
  set_gradient_driver( _set_gradient!,T,D,V,x,n)
  set_gradient_driver( _set_gradient_new!,T,D,V,x,n)
  x = zeros(G,n*C);

  b_old = @benchmark $set_gradient_driver($_set_gradient!,$T,$D,$V,$x,$n)
  b_new = @benchmark $set_gradient_driver($_set_gradient_new!,$T,$D,$V,$x,$n)
  judge(median(b_new), median(b_old))
end

#set_gradient_benchmark(2, Float64, VectorValue{2,Float64}, 5)
#set_gradient_benchmark(2, Float64, VectorValue{3,Float64}, 5)
#set_gradient_benchmark(3, Float64, VectorValue{2,Float64}, 5)
#set_gradient_benchmark(2, Float64, TensorValue{2,2,Float64}, 5)
#set_gradient_benchmark(1, Float64,  SymTensorValue{1,Float64}, 5)
#set_gradient_benchmark(1, Float64,  QTensorValue{1,Float64}, 5)
#set_gradient_benchmark(3, Float64,  SymTensorValue{2,Float64}, 5)
#set_gradient_benchmark(3, Float64,  QTensorValue{2,Float64}, 5)
#set_gradient_benchmark(2, Float64,  SymTensorValue{3,Float64}, 5)
#set_gradient_benchmark(3, Float64,  QTensorValue{2,Float64}, 5)


#####################################################
# src/Polynomials/ModalC0Bases.jl: _set_gradient_mc0!
#####################################################

 _set_gradient_mc0! = Gridap.Polynomials. _set_gradient_mc0!

@inline function _set_gradient_mc0_new!(
  v::AbstractVector{G},s,k,l,::Type{V}) where {V,G}
  @notimplementedif num_indep_components(G) != num_components(G) "Not implemented for symmetric Jacobian or Hessian"

  T = eltype(s)
  z = zero(T)
  w = zero(V)
  ci = CartesianIndices(zero(G))
  n = num_components(G)
  @inbounds for (p, c) in enumerate(CartesianIndices(transpose(w)))
      @inbounds i = c[1]
      m = k+l*(p-1)
      @inbounds v[m] = ntuple(j -> ifelse(ci[j][2] == i, s[ci[j][1]], z), Val(n))
  end
  k+1
end

@inline @generated function _set_gradient_mc0_new!(
  v::AbstractVector{G},s,k,l,::Type{V}) where {V,G}
  @notimplementedif num_indep_components(G) != num_components(G) "Not implemented for symmetric Jacobian or Hessian"
  
  m = Array{String}(undef, size(G))
  N_val_dims = length(size(V))
  s_size = size(G)[1:end-N_val_dims]

  body = "T = eltype(s); z = zero(T);"
  for ci in CartesianIndices(s_size)
    id = join(Tuple(ci))
    body *= "@inbounds s$id = s[$ci];"
  end
  
  V_size = size(V)
  for (ij,j) in enumerate(CartesianIndices(V_size))
    for i in CartesianIndices(m)
      m[i] = "z"
    end
    for ci in CartesianIndices(s_size)
      id = join(Tuple(ci))
      m[ci,j] = "s$id"
    end
    body *= "i = k + l*($ij-1);"
    body *= "@inbounds v[i] = ($(join(tuple(m...), ", ")));"
  end

  body = Meta.parse(string("begin ",body," end"))
  return Expr(:block, body ,:(return k+1))
end

function set_gradient_mc0_benchmark(D, T, V, n)
  C = num_indep_components(V)
  G = gradient_type(V, zero(Point{D,T}))
  x = zeros(G,n*C);

  function set_gradient_mc0_driver(f,T,D,V,x,n)
    k = 1
    s = VectorValue{D,T}(ntuple(_->one(T),D))
    for i in 1:n
      k = f(x,s,k,1,V)
    end
  end

  b_old = @benchmark $set_gradient_mc0_driver($_set_gradient_mc0!,$T,$D,$V,$x,$n)
  b_new = @benchmark $set_gradient_mc0_driver($_set_gradient_mc0_new!,$T,$D,$V,$x,$n)

  judge(median(b_new), median(b_old))
end

#################################################
# src/Polynomials/MonomialBasis.jl: _evaluate_1d!
#################################################

_evaluate_1d! = Gridap.Polynomials._evaluate_1d!

function _evaluate_1d_new!(v::AbstractMatrix{T},x,order,d) where T
  n = order + 1
  z = one(T)
  @inbounds v[d,1] = z
  @inbounds xd = x[d]
  xn = xd
  for i in 2:n
    @inbounds v[d,i] = xn
    xn *= xd
  end
end

function evaluate_1d_benchmark(D, T, V, n)
  n = Integer(n/50)
  order = num_indep_components(V)
  v = zeros(D,order+1);
  x = rand(MVector{n,T})

  function evaluate_1d_driver(f,order,D,v,x_vec)
    for x in x_vec
      f(v,x,order,D)
    end
  end


  b_old = @benchmark $evaluate_1d_driver($_evaluate_1d!,$order,$D,$v,$x)
  b_new = @benchmark $evaluate_1d_driver($_evaluate_1d_new!,$order,$D,$v,$x)

  judge(median(b_new), median(b_old))
end


################################################
# src/Polynomials/MonomialBasis.jl:_gradient_1d!
################################################

_gradient_1d! = Gridap.Polynomials._gradient_1d!

function _gradient_1d_new!(v::AbstractMatrix{T},x,order,d) where T
  n = order + 1
  z = zero(T)
  @inbounds v[d,1] = z
  @inbounds xd = x[d]
  xn = one(T)
  for i in 2:n
    @inbounds v[d,i] = (i-1)*xn
    xn *= xd
  end
end

function gradient_1d_benchmark(D, T, V, n)
  n = Integer(n/10)
  order = num_indep_components(V)
  v = zeros(D,order+1);
  x = rand(MVector{n,T})

  function gradient_1d_driver(f,order,D,v,x_vec)
    for x in x_vec
      f(v,x,order,D)
    end
  end

  b_old = @benchmark $gradient_1d_driver($_gradient_1d!,$order,$D,$v,$x)
  b_new = @benchmark $gradient_1d_driver($_gradient_1d_new!,$order,$D,$v,$x)
  judge(median(b_new), median(b_old))
end


################################################
# src/Polynomials/MonomialBasis.jl:_hessian_1d!
################################################

_hessian_1d! = Gridap.Polynomials._hessian_1d!

function _hessian_1d_new!(v::AbstractMatrix{T},x,order,d) where T
  n = order + 1
  z = zero(T)
  @inbounds v[d,1] = z
  if n>1
    @inbounds v[d,2] = z
  end
  @inbounds xd = x[d]
  xn = one(T)
  for i in 3:n
    @inbounds v[d,i] = (i-1)*(i-2)*xn
    xn *= xd
  end
end

function hessian_1d_benchmark(D, T, V, n)
  n = Integer(n/10)
  order = num_indep_components(V)
  v = zeros(D,order+1);
  x = rand(MVector{n,T})

  function hessian_1d_driver(f,order,D,v,x_vec)
    for x in x_vec
      f(v,x,order,D)
    end
  end

  b_old = @benchmark $hessian_1d_driver($_hessian_1d!,$order,$D,$v,$x)
  b_new = @benchmark $hessian_1d_driver($_hessian_1d_new!,$order,$D,$v,$x)
  judge(median(b_new), median(b_old))
end


################
# Run benchmarks 
################

benchmarks = (                # 1D     2D     3D     5D     
  set_value_benchmark,        # ~67%   85:95% 40:85% 30:80% 
  set_gradient_benchmark,     #  0:3%  5:35%  8:55%  37:100%
  set_value_mc0_benchmark,    #  0%    80:99% 0:90%  0:60%   
  set_gradient_mc0_benchmark, # 16:18% 27:36% 22:60% 38:100%
  evaluate_1d_benchmark,      # ~80%   ~84%   80:90% 90:95%
  gradient_1d_benchmark,      # 10:17% 38:55% 50:75% 70:85%
  hessian_1d_benchmark,       # ~6%    0:60%  40:80% 60:90%
)
benchmarks = (set_gradient_mc0_benchmark, )

function run_all_benchmarks(; dims=(1, 2, 3, 5, 8), n=2000, T=Float64)
  for bench in benchmarks 
    for D in dims
      TV = [
        VectorValue{D,T},
        TensorValue{D,D,T,D*D},
        SymTensorValue{D,T,Integer(D*(D+1)/2)},
        SymTracelessTensorValue{D,T,Integer(D*(D+1)/2)}
      ]
      
      for V in TV
        if V == SymTracelessTensorValue{1,T,1} continue end # no dof
          tj = bench(D, T, V, n)
          println("D: $D, V: $V ", @show bench)
          println(tj)
      end
    end
  end
end

run_all_benchmarks(n=3000, dims=( 1, 2, 3, 5))
