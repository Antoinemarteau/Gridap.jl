###############################################################
# Derivative Tensor Type
###############################################################

"""
"""
struct DerivativeValue{O,D,V,S,T,N,L} <: MultiValue{S,T,N,L}
  data::NTuple{L,T}
  # 1 ≤ O, 1 ≤ D
  # V <: MultiValue{Sv, T, Nv, Lv} # support recursive DerivativeValue ?
  # N = O + Nv
  # S = Tuple{D, ..., D, Sv...}
  #            "O*D"
  # L = binomial(O+D-1,O) * num_indep_components(V)
  #  "nb independent partials"
end

