using Test #, Random
import Pkg
using Pluto

@testset "Testing solution to Exercise 1" begin

@testset "Running ex1.jl" begin
   Pluto.activate_notebook_environment("../ex1.jl"); 
   Pkg.instantiate();
   include("../ex1.jl")
end;

@testset "Testing that variables exist" begin
   @test @isdefined(response_1a)
   @test @isdefined(response_1b)
   @test @isdefined(response_1c)
   @test @isdefined(response_1d)
   @test @isdefined(response_1e)
   @test @isdefined(response_1f)
   @test @isdefined(response_1g)
   @test @isdefined(response_1h)
   @test @isdefined(response_1i)
   @test @isdefined(response_1j)
end;

@testset "Testing that variables are not missing" begin
   @test check_type_isa(:response_1a,response_1a,Markdown.MD)
   @test check_type_isa(:response_1b,response_1b,Markdown.MD)
   @test check_type_isa(:response_1c,response_1c,Markdown.MD)
   @test check_type_isa(:response_1d,response_1d,Markdown.MD)
   @test check_type_isa(:response_1e,response_1e,Markdown.MD)
   @test check_type_isa(:response_1f,response_1f,Markdown.MD)
   @test check_type_isa(:response_1g,response_1g,Markdown.MD)
   @test check_type_isa(:response_1h,response_1h,Markdown.MD)
   @test check_type_isa(:response_1i,response_1i,Markdown.MD)
   @test check_type_isa(:response_1j,response_1j,Markdown.MD)
end;

@testset "Testing gradient calculations" begin
   @test maximum(abs.(grad_banana_2d_at_origin.-[0.8, -6.4]))<1e-5
   @test maximum(abs.(grad_banana_2d_at_minimum.-zeros(2)))<1e-5
end;

end; # Exercise 1
