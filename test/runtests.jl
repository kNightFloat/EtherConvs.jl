#=
  @ author: ChenyuBao <chenyu.bao@outlook.com>
  @ date: 2026-02-23 22:34:06
  @ license: MIT
  @ language: Julia
  @ declaration: EtherConvs.jl contains some convolution functions.
  @ description: /
 =#

using Test
using EtherConvs
using Pkg

@testset "EtherConvs.jl" begin
    @testset "SPHConvs.jl" begin
        Pkg.add("SPHKernels")
        using SPHKernels
        # * see reference [url](https://github.com/LudwigBoess/SPHKernels.jl/tree/dccf7fd6e34215428e145639e6c1c456f7c4ef10)
        for dim in [2, 3]
            for h in Float32.(0.05:0.05:0.2)
                for x in Float32.(0.02:0.02:0.1)
                    my_ck = SPHConvs.CubicSpline{Float32,dim}()
                    sp_ck = SPHKernels.Cubic(Float32, dim)
                    @test SPHConvs.value(my_ck, x, h) ≈
                          SPHKernels.kernel_value(sp_ck, x / (2 * h), 1 / (2 * h))
                    @test SPHConvs.gradient(my_ck, x, h) ≈
                          SPHKernels.kernel_deriv(sp_ck, x / (2 * h), 1 / (2 * h))
                    my_w2k = SPHConvs.WendlandC2{Float32,dim}()
                    sp_w2k = SPHKernels.WendlandC2(Float32, dim)
                    @test SPHConvs.value(my_w2k, x, h) ≈
                          SPHKernels.kernel_value(sp_w2k, x / (2 * h), 1 / (2 * h))
                    @test SPHConvs.gradient(my_w2k, x, h) ≈
                          SPHKernels.kernel_deriv(sp_w2k, x / (2 * h), 1 / (2 * h))
                    my_w4k = SPHConvs.WendlandC4{Float32,dim}()
                    sp_w4k = SPHKernels.WendlandC4(Float32, dim)
                    @test SPHConvs.value(my_w4k, x, h) ≈
                          SPHKernels.kernel_value(sp_w4k, x / (2 * h), 1 / (2 * h))
                    # ! seems that `SPHKernels.jl` makes some mistakes
                    # ! see [url](https://github.com/LudwigBoess/SPHKernels.jl/blob/dccf7fd6e34215428e145639e6c1c456f7c4ef10/src/wendland/C4.jl)
                    @test abs(
                        SPHConvs.gradient(my_w4k, x, h) -
                        SPHKernels.kernel_deriv(sp_w4k, x / (2 * h), 1 / (2 * h)),
                    ) / abs(SPHConvs.gradient(my_w4k, h * Float32(0.7), h)) < 2e-2
                end
            end
        end
    end
end
