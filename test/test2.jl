using Test #, Random
using Pkg, Pluto

@testset "Testing solution to Exercise 2" begin

@testset "Running ex2.jl" begin
   Pluto.activate_notebook_environment("ex2.jl");
   Pkg.instantiate();
   include("../ex2.jl")
end;

@testset "Testing that variables exist" begin
   @test @isdefined(response_2a)
   @test @isdefined(response_2b)
   @test @isdefined(response_2c)
   @test @isdefined(response_2d)
   #@test @isdefined(response_2e)
   @test @isdefined(response_2f)
   @test @isdefined(response_2g)
   @test @isdefined(response_2h)
end;

@testset "Testing that variables are not missing" begin
   @test check_type_isa(:response_2a,response_2a,Markdown.MD)
   @test check_type_isa(:response_2b,response_2b,Markdown.MD)
   @test check_type_isa(:response_2c,response_2c,Markdown.MD)
   @test check_type_isa(:response_2d,response_2d,Markdown.MD)
   #@test check_type_isa(:response_2e,response_2e,Markdown.MD)
   @test check_type_isa(:response_2f,response_2f,Markdown.MD)
   @test check_type_isa(:response_2g,response_2g,Markdown.MD)
   @test check_type_isa(:response_2h,response_2h,Markdown.MD)
end;

end; # Exercise 2
