
"""
    divergence(f)
"""
divergence(f) = tr(gradient(f))

function divergence(a::AbstractArray{<:Field})
  ag = gradient(a)
  operate_arrays_of_fields(tr,ag)
end

function symmetric_gradient end

"""
    symmetric_gradient(f)
"""
symmetric_gradient(f) = symmetric_part(gradient(f))

function symmetric_gradient(a::AbstractArray{<:Field})
  ag = gradient(a)
  operate_arrays_of_fields(symmetric_part,ag)
end

"""
    const ε = symmetric_gradient

Alias for the symmetric gradient
"""
const ε = symmetric_gradient

"""
    curl(f)
"""
curl(f) = grad2curl(gradient(f))

function curl(a::AbstractArray{<:Field})
  ag = gradient(a)
  operate_arrays_of_fields(grad2curl,ag)
end

"""
    grad2curl(∇f)
"""
function grad2curl(f)
  @abstractmethod
end

grad2curl(a::GridapType) = operate(grad2curl,a)

grad2curl(f::Field) = operate_fields(grad2curl,f)

@inline function grad2curl(∇u::TensorValue{2})
  ∇u[1,2] - ∇u[2,1]
end

@inline function grad2curl(∇u::TensorValue{3})
  c1 = ∇u[2,3] - ∇u[3,2]
  c2 = ∇u[3,1] - ∇u[1,3]
  c3 = ∇u[1,2] - ∇u[2,1]
  VectorValue(c1,c2,c3)
end

function laplacian end

"""
    const Δ = laplacian

Alias for the `laplacian` function
"""
const Δ = laplacian

"""
    laplacian(f)
"""
function laplacian(f)
  g = gradient(f)
  divergence(g)
end

"""
    ∇⋅f

Equivalent to

    divergence(f)
"""
dot(::typeof(∇),f) = divergence(f)
dot(::typeof(∇),f::GridapType) = divergence(f)

function (*)(::typeof(∇),f)
  msg = "Syntax ∇*f has been removed, use ∇⋅f (\\nabla \\cdot f) instead"
  error(msg)
end

function (*)(::typeof(∇),f::GridapType)
  msg = "Syntax ∇*f has been removed, use ∇⋅f (\\nabla \\cdot f) instead"
  error(msg)
end

"""
    outer(∇,f)

Equivalent to

    gradient(f)
"""
outer(::typeof(∇),f) = gradient(f)
outer(::typeof(∇),f::GridapType) = gradient(f)

"""
    outer(f,∇)

Equivalent to

    transpose(gradient(f))
"""
outer(f,::typeof(∇)) = transpose(gradient(f))
outer(f::GridapType,::typeof(∇)) = transpose(gradient(f))

"""
    cross(∇,f)

Equivalent to
    
    curl(f)
"""
cross(::typeof(∇),f) = curl(f)
cross(::typeof(∇),f::GridapType) = curl(f)

# Automatic differentiation of functions

function gradient(f::Function)
  function grad_f(x)
    _grad_f(f,x,zero(return_type(f,typeof(x))))
  end
end

function _grad_f(f,x,fx)
  VectorValue(ForwardDiff.gradient(f,get_array(x)))
end

function _grad_f(f,x,fx::VectorValue)
  TensorValue(transpose(ForwardDiff.jacobian(y->get_array(f(y)),get_array(x))))
end

function _grad_f(f,x,fx::MultiValue)
  @notimplemented
end

function divergence(f::Function)
  x -> tr(ForwardDiff.jacobian(y->get_array(f(y)),get_array(x)))
end

function curl(f::Function)
  x -> grad2curl(TensorValue(transpose(ForwardDiff.jacobian(y->get_array(f(y)),get_array(x)))))
end

function laplacian(f::Function)
  function lapl_f(x)
    _lapl_f(f,x,zero(return_type(f,typeof(x))))
  end
end

function _lapl_f(f,x,fx)
  tr(ForwardDiff.jacobian(y->ForwardDiff.gradient(f,y), get_array(x)))
end

function _lapl_f(f,x,fx::VectorValue)
  A = length(x)
  B = length(fx)
  a = ForwardDiff.jacobian(y->transpose(ForwardDiff.jacobian(z->get_array(f(z)),y)), get_array(x))
  tr(ThirdOrderTensorValue{A,A,B}(Tuple(transpose(a))))
end

function _lapl_f(f,x,fx::MultiValue)
  @notimplemented
end

function symmetric_gradient(f::Function)
    x -> symmetric_part(_grad_f(f,x,zero(return_type(f,typeof(x)))))
end

