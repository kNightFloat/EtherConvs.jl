#=
  @ author: ChenyuBao <chenyu.bao@outlook.com>
  @ date: 2026-02-23 21:00:25
  @ license: MIT
  @ language: Julia
  @ declaration: EtherConvs.jl contains some convolution functions.
  @ description: /
 =#

module EtherConvs

export AbstractConv
export SPHConvs

abstract type AbstractConv{N} end

include("SPHConvs.jl")

end # module EtherConvs
