#=
  @ author: ChenyuBao <chenyu.bao@outlook.com>
  @ date: 2026-02-23 21:00:48
  @ license: MIT
  @ language: Julia
  @ declaration: EtherConvs.jl contains some convolution functions.
  @ description: /
 =#

using JSON
using OrderedCollections

const kDeclaration = "EtherConvs.jl contains some convolution functions."
const kVSCSettingsPath = joinpath(".vscode", "settings.json")

kAutoHeader::OrderedDict{String,Any} = OrderedDict(
    "format" => OrderedDict(
        "startWith" => "#=",
        "middleWith" => "",
        "endWith" => "=#",
        "headerPrefix" => "@",
    ),
    "header" => OrderedDict(
        "author" => "ChenyuBao <chenyu.bao@outlook.com>",
        "date" =>
            OrderedDict("type" => "createTime", "format" => "YYYY-MM-DD HH:mm:ss"),
        "license" => "MIT",
        "language" => "Julia",
        "declaration" => kDeclaration,
        "description" => "/",
    ),
)

function main()::Nothing
    settings = OrderedDict{String,Any}()
    settings["autoHeader"] = kAutoHeader
    isdir(".vscode") || mkpath(".vscode")
    open(kVSCSettingsPath, "w") do io
        JSON.print(io, settings, 4)
    end
    return nothing
end

main()
