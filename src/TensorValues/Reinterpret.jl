
function reinterpret(a::Array{V}) where V<:ArrayMultiValue{S,T,N,L} where {S,T,N,L}
  b = reinterpret(T,a)
  sa = size(a)
  s = (L,sa...)
  reshape(b,s)
end
