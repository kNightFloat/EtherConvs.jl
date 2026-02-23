#=
  @ author: ChenyuBao <chenyu.bao@outlook.com>
  @ date: 2026-02-23 21:35:16
  @ license: MIT
  @ language: Julia
  @ declaration: EtherConvs.jl contains some convolution functions.
  @ description: /
 =#

module SPHConvs

using EtherMaths
using EtherConvs

export SPHConv
export W, dW
export WendlandC2, Gaussian, CubicSpline, WendlandC4

abstract type SPHConv{N} <: AbstractConv{N} end

@inline function value(conv::SPHConv{N}, r::Real, h::Real) where {N}
    return __value(conv, r, 1/h)
end

@inline function gradient(conv::SPHConv{N}, r::Real, h::Real) where {N}
    return __gradient(conv, r, 1/h)
end

const W = value
const dW = gradient

# * ========== WendlandC2 ========== * #

struct WendlandC2{T<:AbstractFloat,N} <: SPHConv{N} end

@inline ratio(::WendlandC2{T,N}) where {T,N} = T(2)
@inline sigma(::WendlandC2{T,1}) where {T} = T(0)
@inline sigma(::WendlandC2{T,2}) where {T} = T(7 / (4 * pi))
@inline sigma(::WendlandC2{T,3}) where {T} = T(21 / (16 * pi))

@inline @fastmath function __value(
    conv::WendlandC2{T,N},
    r::Real,
    hinv::Real,
) where {T<:AbstractFloat,N}
    q::T = r * hinv
    val =
        q < ratio(conv) ?
        sigma(conv) * power(2 - q, Val(4)) * power(hinv, Val(N)) * (1 + 2*q) * T(0.0625) :
        T(0)
    return val
end

@inline @fastmath function __gradient(
    conv::WendlandC2{T,N},
    r::Real,
    hinv::Real,
) where {T<:AbstractFloat,N}
    q::T = r * hinv
    grad =
        q < ratio(conv) ?
        -sigma(conv) * power(hinv, Val(N + 1)) * T(0.625) * q * power(2 - q, Val(3)) : T(0)
    return grad
end

# * ========== Gaussian ========== * #

struct Gaussian{T<:AbstractFloat,N} <: SPHConv{N} end

@inline ratio(::Gaussian{T,N}) where {T,N} = T(3)
@inline sigma(::Gaussian{T,1}) where {T} = T(1 / sqrt(pi))
@inline sigma(::Gaussian{T,2}) where {T} = T(1 / pi)
@inline sigma(::Gaussian{T,3}) where {T} = T(1 / sqrt(pi * pi * pi))

@inline @fastmath function __value(
    conv::Gaussian{T,N},
    r::Real,
    hinv::Real,
) where {T<:AbstractFloat,N}
    q::T = r * hinv
    val = q < ratio(conv) ? sigma(conv) * exp(-q*q) * power(hinv, Val(N)) : T(0)
    return val
end

@inline @fastmath function __gradient(
    conv::Gaussian{T,N},
    r::Real,
    hinv::Real,
) where {T<:AbstractFloat,N}
    q::T = r * hinv
    grad =
        q < ratio(conv) ? -2 * sigma(conv) * q * exp(-q*q) * power(hinv, Val(N + 1)) : T(0)
    return grad
end

# * ========== CubicSpline ========== * #

struct CubicSpline{T<:AbstractFloat,N} <: SPHConv{N} end

@inline ratio(::CubicSpline{T,N}) where {T,N} = T(2)
@inline sigma(::CubicSpline{T,1}) where {T} = T(2 / 3)
@inline sigma(::CubicSpline{T,2}) where {T} = T(10 / (7 * pi))
@inline sigma(::CubicSpline{T,3}) where {T} = T(1 / pi)

@inline @fastmath function __value(
    conv::CubicSpline{T,N},
    r::Real,
    hinv::Real,
) where {T<:AbstractFloat,N}
    q::T = r * hinv
    val = T(0)
    val =
        q < T(1) ? sigma(conv) * power(hinv, Val(N)) * (3 * q * q * (q - 2) + 4) * T(0.25) :
        (
            q < ratio(conv) ?
            sigma(conv) * power(hinv, Val(N)) * T(0.25) * power(2 - q, Val(3)) : T(0)
        )
    return val
end

@inline @fastmath function __gradient(
    conv::CubicSpline{T,N},
    r::Real,
    hinv::Real,
) where {T<:AbstractFloat,N}
    q::T = r * hinv
    grad = T(0)
    grad =
        q < T(1) ? sigma(conv) * power(hinv, Val(N + 1)) * q * (3 * q - 4) * T(0.75) :
        (
            q < ratio(conv) ?
            -sigma(conv) * power(hinv, Val(N + 1)) * T(0.75) * power(2 - q, Val(2)) : T(0)
        )
    return grad
end

# * ========== WendlandC4 ========== * #

struct WendlandC4{T<:AbstractFloat,N} <: SPHConv{N} end

@inline ratio(::WendlandC4{T,N}) where {T,N} = T(2)
@inline sigma(::WendlandC4{T,1}) where {T} = T(5 / 8)
@inline sigma(::WendlandC4{T,2}) where {T} = T(9 / (4 * pi))
@inline sigma(::WendlandC4{T,3}) where {T} = T(495.0 / (256.0 * pi))

@inline @fastmath function __value(
    conv::WendlandC4{T,N},
    r::Real,
    hinv::Real,
) where {T<:AbstractFloat,N}
    q::T = r * hinv
    val =
        q < ratio(conv) ?
        sigma(conv) *
        power(2 - q, Val(6)) *
        power(hinv, Val(N)) *
        (q * (35 * q + 36) + 12) *
        T(0.0013020833333333333) : T(0)
    return val
end

@inline function __gradient(
    conv::WendlandC4{T,N},
    r::Real,
    hinv::Real,
) where {T<:AbstractFloat,N}
    q::T = r * hinv
    grad =
        q < ratio(conv) ?
        -sigma(conv) *
        power(hinv, Val(N + 1)) *
        q *
        power(2 - q, Val(5)) *
        (2 + 5 * q) *
        T(0.07291666666666667) : T(0)
    return grad
end

end # module SPHConvs
