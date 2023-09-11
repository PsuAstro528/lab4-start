### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 69787870-0b1e-11ec-3354-a914e59bdad2
begin
	using PlutoUI, PlutoTest, PlutoTeachingTools
	using Plots, LaTeXStrings#,StatsPlots
	using LinearAlgebra, PDMats
	using Optim
	using ForwardDiff,ReverseDiff
	using Distributions
	using BenchmarkTools
	using Random
	using Printf
	Random.seed!(123)
	eval(Meta.parse(code_for_check_type_funcs))
end;

# ╔═╡ 2046af3d-aea5-433c-8b70-4916a8b7ef1a
md"""
# Lab 4, Exercise 1
## Auto-differentiation & Optimization
### (as in minimizing a function)
"""

# ╔═╡ 8b35ac88-aece-4f78-b0e7-0a03ff1fbe26
md"""
In this lab we'll consider the goal of finding the minimum (or maximum) of a mathematical function.  This is a common problem in science, such as when we have astronomical obsevations, a physical model and unknown model parameters that we want to infer from the data.  In this case, the *target function* to be maximized might be the log likelihood or the log posterior density based on the astrophysical model and statistical properties of the observations.   

One common goal is to find the *"best-fit"* model parameters.  The computational expense of a model and finding the best-fit parameters can vary widely.  In this lab, we'll consider functions that can be evaluated very quickly, allowing us interactively explore the efficiency of different algorithms and how they depend on the number of parameters to the function.
"""

# ╔═╡ 3daa9062-8c45-41fd-a52d-d042fd86d1b6
md"This lab has some benchmarks that take a few minutes to run.  Check this box  $(@bind run_benchmarks CheckBox(default=false)) if/when you want to start the benchmarking."

# ╔═╡ 9463b61d-99e1-4542-828f-14d35f059ba8
md"""
# Gaussian Density Target
We'll start with the negative log of a [multivariate Gaussian probability density](https://en.wikipedia.org/wiki/Multivariate_normal_distribution).  Technically, this is a very easy function to optimize, since there is one global minimum, there is a straight downhill path from any point to that minimum, and the derivaties of the target function are constant.  But we'll pretend like we don't know that and we need to use an iterative algorithm to search for the minimum.  

## 2-D Gaussian Target 
First, let's make a 2-dimensional function to be optimized, so it'll be easy to visualize.  Itterative algorithms require an initial guess, so we'll show that, too.
"""

# ╔═╡ c0f97de7-832d-46a0-b82d-a04c47bbafce
@bind draw_new_target_gauss_2d Button("Create a new 2D Gaussian target function and initial guess.")

# ╔═╡ 9278ed8e-eb16-4edb-955d-e6621c405285
protip(md"""Julia makes extensive use of [multiple dispatch](https://en.wikipedia.org/wiki/Multiple_dispatch) which can be thought of as a generalization of single dispatch used by languages that encourage object-oriented programming such as C++, Java and Python.  
	If you're used to object-oriented programming, then you may be interested to see the syntax for creating a [*functor* or function-like object](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects) at the bottom of this notebook.  We're using that programming pattern here, creating a functor that computes the 2D Gaussian target function and stores data about the mean and covariance of the target function.  
	""")

# ╔═╡ 0716f5c7-b682-46ab-9345-a88dd53bb46a
md"""
There are numerous optimization algorithms that are specialized for a wide variety of functions.  If you know the specific properties of your target function, then it may be possible to choose a very efficient algorithm for that problem.  Often, astrophysics involves complex target functions that are best suited for optimization by a more general algorithm.  These can be quite complex and we won't explore the mathematics of why they work.  For the purposes of this lab, we'll simply make use of some powerful algorithms for non-linear optimization, provided by the [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/) package.
"""

# ╔═╡ b013cf33-c7a6-42ad-8842-aa631f86cace
md"""
The first algorithm we'll try is the [downhill simplex method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method) (sometimes referred to as the amoeba method or Nelder-Mead).  Below we call the `optimize` function, specifying the target function to be minimized, an initial guess (as a vector), and the algorithm to be used.  The `optimize` function will return a structure that contains its estimate of the location of the minimum (the `minimizer`), the value of the function at that point (the `minimum`), and additional information such as how many function calls were used and why the algorithm decided to stop its search.
"""

# ╔═╡ 5ef412aa-c13b-447f-9d0a-9c6b37e8b37a
md"""
We can compare it's approximation to the minimum to the true location of the minimum.
"""

# ╔═╡ fe603b3d-ecc7-4259-985e-4ce827347391
md"""
and the value of the target function at that location to the true minimum
"""

# ╔═╡ 0bd886fc-6784-4196-a874-1e20a63683cc
md"""
We'd expect that the algorithm worked pretty well.  

There are other algorithms that are often more efficient, particularly for problems where one can compute (or approximate) the partial derivatives of one's function (i.e., gradient).  The simplest is [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent), which simply takes small steps downhill each time.  There are numerous variations that try to be more clever.  For example, the [BFGS algorithm](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) is many people's go-to algorithm for non-linear optimization.  
Thanks to good software engineering, it's very easy to call `optimize` and swap out the optimization algorithm, just by changing the final arguement.
"""

# ╔═╡ 3192b6ce-e86f-499d-9a17-eb7bcfcd591c
md"""
Notice that using the Gradient Descent and BFGS optimizers resulted in using many fewer evaluations of the target function ("see f(x) calls above").  How is that possible?  It also made use of the gradient of the function ("see ∇f(x) calls").  In this case, we only provided the function to be minimized, so it [estimated the gradient numerically](https://en.wikipedia.org/wiki/Numerical_differentiation).  

Let's check how well the gradient descent and BFGS algorithm did at finding the minimum in comparison to downhill simplex. 
"""

# ╔═╡ cadde19b-0e43-476b-afef-dec549b2fdd0
md"""
1a.  Which algorithm(s) got closest to the true minimum?  If this function represented a likelihood function, would the difference likely to be scientifically significant?  Click the button at the top of the notebook to generate different Gaussian targets and see how the two optimizers perform for each case.  
"""

# ╔═╡ e8124c99-b889-46da-a9aa-5f36c3941efe
response_1a = missing

# ╔═╡ 7e7aa8c8-5573-4a52-800b-ed8fce316d94
display_msg_if_fail(check_type_isa(:response_1a,response_1a,Markdown.MD)) 

# ╔═╡ 463010d8-a66c-4289-a025-cd74e8ab3f08
md"""
Next, we'll benchmark the different optimization algorithms.  

1b. Which do you think will be faster and why?
"""

# ╔═╡ 074ce518-5a0e-4d56-95c6-678967d171e5
response_1b = missing

# ╔═╡ 558c1c42-61cd-4597-b110-19206034cf5d
display_msg_if_fail(check_type_isa(:response_1b,response_1b,Markdown.MD)) 

# ╔═╡ 7047987f-0b45-4224-8e41-fb7953ba3e6d
md"**Nelder Mead**"

# ╔═╡ 0cb98deb-8807-406f-993d-17ddeb0ff686
md"**Gradient Descent**"

# ╔═╡ c6b4377a-5fff-4e8b-a96f-df971a4d74f6
md"**BFGS**"

# ╔═╡ 2b4188b3-8a1c-44a2-acdd-e8a0949e2659
md"""
1c.  How did the benchmarking results compare to your predictions?  If different, try to explain what may have caused the unexpected results.
"""

# ╔═╡ b38e119e-e98f-4922-a027-cc6d66a47d96
response_1c = missing

# ╔═╡ a27e2ba3-f31a-40df-819e-6d46551efae4
display_msg_if_fail(check_type_isa(:response_1c,response_1a,Markdown.MD)) 

# ╔═╡ 1a746366-ce12-43ae-b02e-11c5dd713eb6
md"""
In the example above, the optimize function had to estimate the gradient of our target function numerically.  We could likely improve the performance by providing a function to compute the gradient explicitly.  Alternatively, we could make use of automatic differentiation (or "autodiff"), where we let the compiler do the work of computing derivatives for us.

In this case, we have a simple target function and we could compute the gradient analytically ourselves.  However, that would require human time and we'd have to be careful not to miss a minus sign or factor of 2.  Autodiff is a particularly powerful tool when the target function is more complex and computing gradients analytically would be impractical.  

There are several different strategies for performing automatic differentiation.  Which is more efficient depends on the problem.  For this lab, we'll use [Foward Accumulation](https://en.wikipedia.org/wiki/Automatic_differentiation#Forward_accumulation) which is implemented by Julia's [ForwardDiff package](https://github.com/JuliaDiff/ForwardDiff.jl).  For example, we can compute the gradient with
"""

# ╔═╡ 888e04cd-bf48-4219-bd49-346bfacb860f
md"""
Often, the computational cost to compute the gradient is only slightly more than computing the function itself.  Let's compare the time for our 2-d Gaussian log pdf.
"""

# ╔═╡ e2ba5402-9be8-4bfe-acfc-5acf4ac01ace
md"**Evaluate function only**"

# ╔═╡ a585d799-4c14-45e3-8250-d9a19df075b3
md"**Evaluate gradient only**"

# ╔═╡ 54319c3e-fed4-4184-81ce-767f9a527c1b
md"""
There are many subexpressions that are shared across the function and gradient evaluation.  We can combine the two calculations, by using the *mutating* form of `gradient!` that writes both the gradient and the function to a preallocated results object.
"""

# ╔═╡ b3d5cf28-64bb-4e62-b72b-e6747a668f37
md"**Evaluate function and gradient with Forward Diff**"

# ╔═╡ 61704586-c69f-473f-bfed-cff98767b729
md"""
The [DiffResults.jl](https://juliadiff.org/DiffResults.jl/stable/) package provides an *Application Programming Interface (API)* for retreiving the results of autodifferentiation packages.  For example,
"""

# ╔═╡ f3cedf20-eda8-4bfe-a8fa-497e1cfe5dd4
md"""It's better to use the API (as opposed to accessing the internal variables in the result of type `MutableDiffResult`), since the API is shared by other autodiff packages, such as *Reverse-mode* autodiff packages like [ReveresDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl).    Thus, one can more easily try/swap out different autodiff engines to see which works better for a particular application. Let's compare these two options. 
"""

# ╔═╡ 980e8ae8-7b6c-43cd-a217-b9de54731b6f
md"**Evaluate function and gradient with Reverse Diff**"

# ╔═╡ f9ce690a-fab6-40fd-b4fc-6788840dbe30
protip(md"""While forward-mode autodifferentiation is relatively straight-forward, there are various strategies for reverse-mode autodifferentiation, so there are multiple packages that make different design choices.  Typically, forward-mode is best for functions with few input parameters.  Reverse-mode is often more efficient functions with many inputs and more inputs than outputs.  The latter case is common in some machine learning applications.
""")

# ╔═╡ 32801c51-156f-490d-b980-2d98e8f5be23
md"""
We can tell the Optim package to make use of automatic differentiation via an optional parameter as shown below.
"""

# ╔═╡ 357e0485-6f6c-4885-ba2c-915afc1c7580
md"""
The number of function and gradient evaluations is probably the same as when we estimated gradients numerically.  (This isn't necessarily the case, particularly for more complex functions where numerical estimates may be inaccurate.)  However, the compute time might be significantly different.  Let's check. 
"""

# ╔═╡ 286f9794-d139-4058-9fc2-8588d9040f3f
md"**BFGS w/ autodiff**"

# ╔═╡ f33f8dc4-df49-496b-9da9-503f4beaf1c8
md"""
1d.  How does the compute time compare when autodiff compared to when estimating gradients numerically?  
"""

# ╔═╡ 2f04f9c4-afd3-4cfe-ae7a-56433e91763f
response_1d = missing

# ╔═╡ 29feb38d-4b5c-4fc2-a7f4-2cdbca7c1f57
display_msg_if_fail(check_type_isa(:response_1d,response_1a,Markdown.MD)) 

# ╔═╡ 1b5c212d-aa44-4d4d-bddb-db43280afd6c
md"""
## Higher dimension Gaussian target
Next, we'll compare the time required to find the minimum for a Gaussian log density target, but varying the number of dimensions.

1e.  Which algorithms do you expect to perform best and worst as the dimensionality increases?  Do you expect the difference between algorithms will increase or decrease as the number of dimensions increases? 
"""

# ╔═╡ 21bb794f-bd75-44f5-8131-6ce43b3bab77
response_1e = missing

# ╔═╡ 3a3fd94d-16cb-41e8-a3d4-7ae6774b69b6
display_msg_if_fail(check_type_isa(:response_1e,response_1e,Markdown.MD)) 

# ╔═╡ b9e835b4-039e-4601-a85f-06abd54ff0c0
dimen_to_benchmark_gauss = [2,3,4,5,6,8,12,16,20,24]	

# ╔═╡ d3b0b80d-c25d-4028-bd46-ddd2f3db0cdc
md"I've made my predictions.  $(@bind ready_to_see_plt_vs_ndim CheckBox(default=false))"

# ╔═╡ 7513a015-34fb-4cea-a236-facfd6f89b4a
md"""
When comparing performance of algorithms, it's also important to consider the accuracy. Below, we'll plot of the distance between the estimated minimum and true minimum.  
"""

# ╔═╡ 74b79479-fd13-416c-a6bd-d9967624a5b4
md"""
1f.  What could explain why one algorithm doesn't appear take longer when applied to problems of higher dimensions?  What do you think would happen if you adjusted the parameters passed to `optimize`, so that the different algorithms acheived a similar accuracy?
"""

# ╔═╡ d9b81c82-4939-4b19-b00e-c26f0ef87aa0
response_1f = missing

# ╔═╡ 150c9fdf-4647-4115-b391-d5d391a505c7
display_msg_if_fail(check_type_isa(:response_1f,response_1f,Markdown.MD)) 

# ╔═╡ 6a549339-e4a5-47fa-9f5d-fa5b36a3e302
md"""
# More challenging target functions
## 2-D Warped Gaussian or Banana Target

The previous example was a fairly easy optimization problem.  Next, we'll explore a more challenging function that has a shallow valley that is warped into the shape of a banana.  Again, we'll start with a 2-d version, so that we can visualize it easily.
"""

# ╔═╡ ba8665f6-883d-485d-86c1-7f42fd1df5ed
begin  # Set parameters for warped Gaussian
	min_prewarp_banana_2d = [0.0,4.0] 
	banana_a = 2.0
	banana_b = 0.2
end;

# ╔═╡ 0f06886f-8eb3-4828-b5dd-b84a4749519b
md"As before, it'll be helpful to be able to compare our results to the location of the true minimum."

# ╔═╡ 66e079a4-e14e-4700-adfa-51a05a4cd07e
md"""
Now it's your turn to evaluate the gradient of the `banana_2d` function at a couple of locations.  Try using ForwardDiff's [`gradient`](https://juliadiff.org/ForwardDiff.jl/stable/user/api/#ForwardDiff.gradient) function to compute the gradient of the `banana_2d` function evaluated at the origin and at the location of the true minimum.
"""

# ╔═╡ c0d077f4-ab1d-4a79-b6e2-722ddc33b141
grad_banana_2d_at_origin = missing

# ╔═╡ a1ff09d0-536b-4c1b-94a7-6caebaa45f93
begin
    if !@isdefined(grad_banana_2d_at_origin)
   		var_not_defined(:grad_banana_2d_at_origin)
    elseif ismissing(grad_banana_2d_at_origin)
        still_missing()
    elseif !(maximum(abs.(grad_banana_2d_at_origin.-[0.8, -6.4]))<1e-5) 
        almost(md"Check that you're evaluating the gradient at the origin.")
    else
        correct()
    end
end


# ╔═╡ 450d1d7b-032c-420b-8dd9-34b26daa0c3b
grad_banana_2d_at_minimum = missing 

# ╔═╡ 2ef2e78f-a324-4bc6-b7ef-6cb43e0491f6
begin
    if !@isdefined(grad_banana_2d_at_minimum)
   		var_not_defined(:grad_banana_2d_at_minimum)
    elseif ismissing(grad_banana_2d_at_minimum)
        still_missing()
    elseif !(maximum(abs.(grad_banana_2d_at_minimum.-zeros(2)))<1e-5) 
        almost(md"Hmm, your gradient is larger than expected at the minimum.")
    else
        correct()
    end
end


# ╔═╡ 31a3807f-2075-4e06-a2fc-8ba4a2ae412a
md"""
## Higher dimensional warped target
Now, we'll construct a similar warped Gaussian function that has been generalized to higher dimensions.  
"""

# ╔═╡ c9ee5aea-4d24-43bd-91c5-5efe42344425
function banana_highd(x)
	@assert length(x) >=2
	a = banana_a::Float64   # Type annotations to avoid performance hit
	b = banana_b::Float64   # from using a global variable inside function
	mu = min_prewarp_banana_2d::Vector{Float64}
	dist_2d = MvNormal(mu,PDMat([1.0 0.5; 0.5 1.0]))
	target = zero(eltype(x))
	for i in 1:2:(length(x)-1)
		y = [ x[i]/a, x[i+1]*a + a*b*(x[i]^2+a^2) ]
		target += -logpdf(dist_2d,y)
	end
	if mod(length(x),2) == 1
		target += -logpdf(Normal(),x[end])
	end
	return target
end;

# ╔═╡ 60f83c0f-0177-49a1-af98-fe3359ce1aa1
protip(md"Using global variables like `banana_a` and `banana_b` inside functions can have a very negative effect on performance, since the compiler can't be sure of their type at compile time.  If you are using a global variable inside a function and know the type of a global variable will be fixed, then you can use avoid the performance hit by making a [type annotation](https://docs.julialang.org/en/v1/manual/performance-tips/#Annotate-values-taken-from-untyped-locations).  The function above demonstrates this technique.")

# ╔═╡ bb7c4307-2db1-4447-82a8-1ffa981144c9
md"""
Even though `banana_hd` is a fairly complex function with function calls to other packages (i.e., [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) and [PDMats.jl](https://github.com/JuliaStats/PDMats.jl)), a `for` loop and an `if` statements, we're still able to compute the gradient with small cost via autodiff.  This demonstrates the power of Julia's multiple dispatch for making packages *composable*.   

The plot below should show that it gets more expensive as the number of dimensions increases.  This is expected, since the function complexity grows with the number of dimensions and the number of partial derivatives required to compute the gradient also increases.
"""

# ╔═╡ 08a5515e-12f4-4a37-9a8b-e844cd5eda13
num_dim_to_benchmark_banana = [2, 4, 6, 8, 10, 12, 16] 

# ╔═╡ 737cc421-e0ec-4d00-99f1-0201a6ee0bbc
if run_benchmarks  # Benchmarks for plot below
	runtime_banana = map(nd->(@belapsed banana_highd($randn($nd)) samples = 10),num_dim_to_benchmark_banana)
	runtime_banana_grad = map(nd->(@belapsed ForwardDiff.gradient($banana_highd,$randn($nd)) samples=10) ,num_dim_to_benchmark_banana)
	runtime_banana_grad_rev = map(nd->(@belapsed ReverseDiff.gradient($banana_highd,$randn($nd)) samples=10) ,num_dim_to_benchmark_banana)
end;

# ╔═╡ a8f36cbe-a8ff-49d1-806a-13d2b27b9d8f
if run_benchmarks
	local plt = plot(legend=:topleft,yscale=:log10)
	scatter!(plt,num_dim_to_benchmark_banana,runtime_banana, label="Function")
	scatter!(plt,num_dim_to_benchmark_banana,runtime_banana_grad, label="Gradient (forward-mode)")
	scatter!(plt,num_dim_to_benchmark_banana,runtime_banana_grad_rev, label="Gradient (reverse-mode)")
	xlabel!("Number of dimensions")
	ylabel!("Runtime (s)")
	title!("Cost of function and gradient")
end

# ╔═╡ 0bd2b178-4db5-49bf-b653-6b90dd93f77a
md"""
## Finding the minimum of a warped target
Before we compare the performance of different algorithms, we should check whether the minimization algorithms are able to find the minimum accurately when there are more dimensions.  
The calculations below are a little slow, so we'll try them for just one size at a time.  You can adjust the number of dimensions used via the number box below.
"""

# ╔═╡ 0101c723-4cd5-4d91-9a9e-3c39ea0632ba
md"Number of dimensions for target function: $(@bind banana_nd NumberField(1:1:20, default=2) ) " 

# ╔═╡ 1a2ba5a9-c16f-4a48-b3b1-1bb1ef5caf09
begin 
	init_guess_banana_nd = 5.0.*randn(banana_nd);
end

# ╔═╡ 997042fa-803d-49c4-9ed6-b6e0809d6cf8
md"""
1g.  Try increasing the number of dimensions to 10-20.  How does the accuracy compare?   What could you change in order to improve the accuracy?  What are the implications for your choice of algorithm for problems with many model parameters? 
"""

# ╔═╡ 1e986186-f8a8-4952-84f8-55ce4c684212
response_1g = missing

# ╔═╡ ea69280d-0af3-48a9-90e3-cb94dc0b23cd
display_msg_if_fail(check_type_isa(:response_1g,response_1g,Markdown.MD)) 

# ╔═╡ 6d910e6f-295c-4428-a689-aea3f8b139c7
hint(md"The `optimize` function can takes an optional `Optim.Options` parameter that specifies one or more [configurable options](https://julianlsolvers.github.io/Optim.jl/stable/#user/config/#general-options) to the `optimize` function.  For example, the optional parameter `iterations` specifies the maximum number of function evaluations before the algorithm returns.  The default value is 1,000.  In the code above, I set the maximum number of itterations to be a configurable parameter using the number box below.")

# ╔═╡ 9cf8cd1e-1a17-4245-80e1-c6bbce1c0db1
md"Maximum number of itterations for calling optimize on high-d warped density:  $(@bind max_iter_banana_hd NumberField(100:10000,default=500))"

# ╔═╡ 78b3a239-b04f-43f2-9b48-15ce7ada9d3d
result_opt_banana_nd_nm = optimize(banana_highd,init_guess_banana_nd,NelderMead(),Optim.Options(iterations=max_iter_banana_hd))

# ╔═╡ 9c5540e1-2397-4f53-815d-d1eb5694af56
result_opt_banana_nd_gd = optimize(banana_highd,init_guess_banana_nd,GradientDescent(),Optim.Options(iterations=max_iter_banana_hd))

# ╔═╡ c306d465-5e6c-446a-8dd1-b02399ec42ba
result_opt_banana_nd_bfgs = optimize(banana_highd,init_guess_banana_nd,BFGS(),Optim.Options(iterations=max_iter_banana_hd))

# ╔═╡ f36431c7-3da8-4a65-af05-f6e85b27a5f0
result_opt_banana_nd_bfgs_ad = optimize(banana_highd,init_guess_banana_nd,BFGS(), autodiff=:forward,Optim.Options(iterations=max_iter_banana_hd))

# ╔═╡ a6e1b07c-2c5a-4e3f-9cb6-e7e35e3f619b
md"""
Now let's compare the time required to run each algorithm (with a few choices for the number of dimensions).  

1h.  Which algorithms do you think will successfully find the true minimum the fastest for 4 dimensions?  What about 12 dimensions?  
"""

# ╔═╡ 2f2debce-d281-4c53-8cb7-77cd4159846d
response_1h = missing

# ╔═╡ 518551bc-bdb5-439e-90c1-369d1dd694ff
display_msg_if_fail(check_type_isa(:response_1h,response_1h,Markdown.MD)) 

# ╔═╡ c02cb04f-e757-4963-bfb6-8213fff1c4cf
begin 
	time_banana_nd_nm = @elapsed result_banana_nd_nm = optimize(banana_highd,randn(banana_nd),NelderMead(),Optim.Options(iterations=max_iter_banana_hd))
end

# ╔═╡ d0668a5b-4081-4d53-b90a-2f65b14d8d01
begin 
	time_banana_nd_bfgs = @elapsed result_banana_nd_bfgs = optimize(banana_highd,randn(banana_nd),BFGS(),Optim.Options(iterations=max_iter_banana_hd))
end

# ╔═╡ 6a2629c0-cde8-4d1f-9e0f-10c9762b432f
begin 
	time_banana_nd_bfgs_ad = @elapsed result_banana_nd_bfgs_ad = optimize(banana_highd,randn(banana_nd),BFGS(), autodiff=:forward,Optim.Options(iterations=max_iter_banana_hd))
end

# ╔═╡ a2365327-5843-43e6-9184-b94072f69bac
dimen_to_benchmark_banana = [2,4,6,8,10,12,16]

# ╔═╡ 28e3a97b-7208-48a2-97f3-d1003eb89c58
if run_benchmarks  # Benchmark different optimization algorithsm on warped Gaussian density
	local nd_list = dimen_to_benchmark_banana
	# Create target densities
	target_banana_list = map(n->banana_highd,nd_list)
	# Use same initial guess for each algorithm
	init_guess_list = map(n->5.0.*randn(n),nd_list)
	# Preallocate arrays for results
	runtime_banana_nm_list = zeros(length(nd_list))
	results_banana_nm_list = Vector{Any}(undef,length(nd_list))
	runtime_banana_bfgs_list = zeros(length(nd_list))
	results_banana_bfgs_list = Vector{Any}(undef,length(nd_list))
	runtime_banana_gd_list = zeros(length(nd_list))
	results_banana_gd_list = Vector{Any}(undef,length(nd_list))
	runtime_banana_bfgs_ad_list = zeros(length(nd_list))
	results_banana_bfgs_ad_list = Vector{Any}(undef,length(nd_list))
	# Perform benchmarks
	for (i,n) in enumerate(nd_list)
		GC.gc()
		runtime_banana_nm_list[i]   = @belapsed (results_banana_nm_list[$i] = optimize(target_banana_list[$i],init_guess_list[$i],NelderMead(),Optim.Options(iterations=$max_iter_banana_hd))) samples=1
		GC.gc()
		runtime_banana_gd_list[i]   = @belapsed (results_banana_gd_list[$i] = optimize(target_banana_list[$i],init_guess_list[$i],GradientDescent(),autodiff=:forward,Optim.Options(iterations=$max_iter_banana_hd))) samples=3
		GC.gc()
		runtime_banana_bfgs_list[i] = @belapsed (results_banana_bfgs_list[$i] = optimize(target_banana_list[$i],init_guess_list[$i],BFGS(),Optim.Options(iterations=$max_iter_banana_hd))) samples=10
		GC.gc()
		runtime_banana_bfgs_ad_list[i] = @belapsed (results_banana_bfgs_ad_list[$i] = optimize(target_banana_list[$i],init_guess_list[$i],BFGS(),autodiff=:forward, Optim.Options(iterations=$max_iter_banana_hd))) samples=10
	end
end;

# ╔═╡ 6a3ed3ce-3709-4e29-a45e-e1ac3a73a8c2
if run_benchmarks
	local plt = plot(xscale=:linear,yscale=:log10,legend=:topleft)
	scatter!(plt,dimen_to_benchmark_banana,runtime_banana_nm_list, label="Nelder Mean (Gradient-free)",color=:blue)
	plot!(plt,dimen_to_benchmark_banana,runtime_banana_nm_list, label=:none,color=:blue)
	scatter!(plt,dimen_to_benchmark_banana,runtime_banana_gd_list, label="Using Gradient Descent, Auto-Diff",color=:purple)
	plot!(plt,dimen_to_benchmark_banana,runtime_banana_gd_list, label=:none,color=:purple)
	scatter!(plt,dimen_to_benchmark_banana,runtime_banana_bfgs_list, label="Using BFGS, Numerical Gradients",color=:green)
	plot!(plt,dimen_to_benchmark_banana,runtime_banana_bfgs_list, label=:none,color=:green)
	scatter!(plt,dimen_to_benchmark_banana,runtime_banana_bfgs_ad_list, label="Using BFGS, Auto-Diff Gradients",color=:red)
	plot!(plt,dimen_to_benchmark_banana,runtime_banana_bfgs_ad_list, label=:none,color=:red)
	xlabel!(plt,"Number of dimensions")
	ylabel!(plt,"Runtime (s)")
	title!(plt,"Benchmarking Algorithms for Warped Gaussian target")
end

# ╔═╡ 1f122def-f9fb-41df-8292-c1b0c93cfb4d
md"1i.  How did the results compare to your predictions?"  

# ╔═╡ 598b3abf-8e4c-4244-9b76-93019fb69c41
response_1i = missing

# ╔═╡ 645f1706-a748-492f-bd04-634dfdd6657c
display_msg_if_fail(check_type_isa(:response_1i,response_1i,Markdown.MD)) 

# ╔═╡ 868ca537-1e50-45c6-a8ba-12c3d179d192
md"1j.  Will your class project involve optimizing a function?  If so, what are the implications of this exercise for your project?" 

# ╔═╡ ea8d39ef-3428-4d5f-8520-e20cb8afac92
response_1j = missing

# ╔═╡ f4c43d3d-894b-4bf0-b37d-3c65a4e2e4b8
display_msg_if_fail(check_type_isa(:response_1j,response_1j,Markdown.MD)) 

# ╔═╡ 6d216ba9-ca71-4e92-bf36-c569f0170657
md"# Helper Funtions"

# ╔═╡ e78d0b7d-78d3-4ea7-b7d6-cc275c91ddb5
ChooseDisplayMode()

# ╔═╡ 35ce6db0-1066-4321-9e06-be54378211fd
TableOfContents(aside=true)

# ╔═╡ b7e4acf0-3528-4e5e-8733-02c1275195af
md"""
## Funtors demonstrated using GaussianTarget
"""

# ╔═╡ 40f3c16f-6fa2-4e14-9e0e-c3f608d3f564
begin
	struct GaussianTarget
		dist::FullNormal
	end
	
	# Constructor 
	function GaussianTarget(μ::Vector{Float64}, Σ::AbstractPDMat{Float64})
		@assert length(μ) == size(Σ,1) ==  size(Σ,2)
		dist = MvNormal(μ,Σ)
		GaussianTarget(dist) 
	end
	
	function (target::GaussianTarget)(x::AVN) where { N1<:Number, AVN<:AbstractVector{N1} }
		-logpdf(target.dist,x)
	end
end

# ╔═╡ c09194bf-6c62-454e-86d2-f2531a68d58c
mean(target::GaussianTarget) = target.dist.μ

# ╔═╡ 583b06d1-0957-4fb2-af88-6994415bca81
covar(target::GaussianTarget) = target.dist.Σ

# ╔═╡ ce4c9111-5949-4d92-ba71-10ae0afa8dea
function make_target(num_dim::Integer)
	@assert 1 <= num_dim <= 1000
	target = GaussianTarget(randn(num_dim), PDMat(rand(InverseWishart(num_dim,diagm(ones(num_dim)))) ) )
end

# ╔═╡ cdfa9982-be89-4949-abb2-363de63bdf89
begin
	draw_new_target_gauss_2d             # Trigger new density with click
	gaussian_target_2d = make_target(2)  # If you're curious, you can find the definition of make_target at the bottom of he notebook.
	init_guess_gauss_2d = 6 .* randn(2)  
end;

# ╔═╡ 98c67125-0e38-4a3a-8f3f-b99b6bb16682
let
	n_plt = 100
	max_axis_range = maximum(vcat(6.0,abs.(init_guess_gauss_2d)))
	plt_grid = range(-max_axis_range,stop=max_axis_range,length=n_plt)
	plt_z = [ gaussian_target_2d([plt_grid[i],plt_grid[j]]) for i in 1:n_plt, j in 1:n_plt ]
	plt = contour(plt_grid,plt_grid,plt_z', levels=10)
	scatter!(plt,[init_guess_gauss_2d[1]],[init_guess_gauss_2d[2]], ms=5, color=:red, label="Initial guess")
	scatter!(plt,[mean(gaussian_target_2d)[1]],[mean(gaussian_target_2d)[2]], ms=5, color=:blue, label="True minimum")
	title!("2D Gaussian Target")
	xlabel!(L"x_1")
	ylabel!(L"x_2")
end

# ╔═╡ 3a94ca69-1b30-4664-9178-68de61393525
result_gauss_2d_nm = optimize(gaussian_target_2d,init_guess_gauss_2d,NelderMead())

# ╔═╡ 49177429-beb7-4b22-8587-ecead0e473ab
result_gauss_2d_nm.minimizer .- mean(gaussian_target_2d) 

# ╔═╡ 5e845450-c80b-4391-aa3e-5cbb71ce2c87
result_gauss_2d_nm.minimum .- gaussian_target_2d(mean(gaussian_target_2d))

# ╔═╡ aea18eaf-fb64-41ba-8fa6-082e14419152
result_gauss_2d_gd = optimize(gaussian_target_2d,init_guess_gauss_2d,GradientDescent())

# ╔═╡ 88d31552-6086-42a1-afab-7106f81a16f0
result_gauss_2d_gd.minimizer

# ╔═╡ 4b391f26-c996-4f4b-b0b1-7871380121eb
result_gauss_2d_bfgs = optimize(gaussian_target_2d,init_guess_gauss_2d,BFGS())

# ╔═╡ cf7a8fc3-68c3-4431-abb5-f031d967f65d
result_gauss_2d_bfgs.minimizer

# ╔═╡ 67d6b3be-9b6b-4609-ac51-80aa6edba4a6
result_gauss_2d_gd.minimizer .- mean(gaussian_target_2d) 

# ╔═╡ 1a55adcd-7af1-4024-bea1-62055ac72af6
result_gauss_2d_gd.minimum .- gaussian_target_2d(mean(gaussian_target_2d))

# ╔═╡ 3414e205-e33c-4f14-a329-b63d1ebd411d
result_gauss_2d_bfgs.minimizer .- mean(gaussian_target_2d) 

# ╔═╡ b075963b-79ed-49f5-8bc8-616196de8b36
result_gauss_2d_bfgs.minimum .- gaussian_target_2d(mean(gaussian_target_2d))

# ╔═╡ b6f54005-2d6c-4629-b8f8-49ee7117d2ba
if !ismissing(response_1b)
	@benchmark optimize($gaussian_target_2d,$randn(2),NelderMead()) samples=100
end

# ╔═╡ 8dfc25fc-109d-4026-ac1c-3182d8ac1f81
if !ismissing(response_1b)
	@benchmark optimize($gaussian_target_2d,$randn(2),GradientDescent()) samples=100
end

# ╔═╡ 4c4649cb-8f73-41cb-b7f9-e48f540fda7e
if !ismissing(response_1b)
	@benchmark optimize($gaussian_target_2d,$randn(2),BFGS()) samples=100
end

# ╔═╡ dd3afd35-5508-48e8-b38d-a3342f32ba7c
ForwardDiff.gradient(gaussian_target_2d,init_guess_gauss_2d)

# ╔═╡ b18277bd-e23c-4184-bc10-5a47d83a3df5
if !ismissing(response_1b)
	@benchmark gaussian_target_2d($init_guess_gauss_2d)  samples=20
end

# ╔═╡ 8aa054de-403b-47e8-9f84-73e2ca2fa4e3
if !ismissing(response_1b)
	@benchmark ForwardDiff.gradient($gaussian_target_2d,$init_guess_gauss_2d)  samples=20
end

# ╔═╡ 756389e5-662b-4384-a94e-ed05c8c286f5
begin 
	result = DiffResults.GradientResult(init_guess_gauss_2d)
	result = ForwardDiff.gradient!(result, gaussian_target_2d,init_guess_gauss_2d)
end

# ╔═╡ 13dbca62-fd43-4a8e-a8e7-25e8672b8a60
DiffResults.value(result), DiffResults.gradient(result)

# ╔═╡ 7e751550-75f9-4b2d-9331-6d8ae09ff397
if !ismissing(response_1b)
	result_rev = DiffResults.GradientResult(init_guess_gauss_2d)
	@benchmark ForwardDiff.gradient!($result_rev,$gaussian_target_2d,$init_guess_gauss_2d)  samples=20
end

# ╔═╡ e5140102-36e7-450f-ac6d-7f32c0668b98
if @isdefined gaussian_target_2d
	cost_one_eval = @belapsed gaussian_target_2d(init_guess_gauss_2d) samples=20
	cost_one_eval = (1+2)*cost_one_eval
	cost_num_grad_str = @sprintf "%1.3g" cost_one_eval
end

# ╔═╡ 9b02c64d-bed9-4d4c-bdd8-67648dd7bfae
if @isdefined cost_num_grad_str
	md"""
If you're evaluating the gradient, then the incremental cost to evaluate the function itself is very small.  For comparison, estimating the gradient numerically would require evaluating the function 1+(number of input parameters) times.  That would be ≃$cost_num_grad_str seconds.  For this function, computing the gradient with ForwardDiff is most efficient, so we'll stick with that option for the rest of the lab.
"""
end

# ╔═╡ f318374c-1938-4efd-b540-f3d2afae2c70
if @isdefined cost_num_grad_str
	md"""
For comparison, estimating the gradient numerically would require evaluating the function 1+(number of input parameters) times.  That would be ≃$cost_num_grad_str seconds.  For this function, computing the gradient with ForwardDiff is most efficient, so we'll stick with that option for the rest of the lab.
"""
end

# ╔═╡ cb9de107-7a33-4a59-8296-d373024e81b6
if !ismissing(response_1b)
	result_rev2 = DiffResults.GradientResult(init_guess_gauss_2d)
	@benchmark ReverseDiff.gradient!($result_rev2,$gaussian_target_2d,$init_guess_gauss_2d)  samples=20
end

# ╔═╡ 64c63760-f2d4-482a-8343-7a8d982b0844
if @isdefined result_rev2
	@test DiffResults.value(result_rev2) ≈ DiffResults.value(result)
end

# ╔═╡ cf37e7b2-6b3e-44aa-9665-723019f6a95c
if @isdefined result_rev2
	@test all(DiffResults.gradient(result_rev2).≈ DiffResults.gradient(result))
end

# ╔═╡ 3db6d059-1995-4648-a7e5-eacb40131c8a
result_gauss_2d_bfgs_ad = optimize(gaussian_target_2d,init_guess_gauss_2d,BFGS(),autodiff=:forward)

# ╔═╡ 2cc9be0e-4b91-4d6a-87da-ad70317c4742
if run_benchmarks
	@benchmark optimize($gaussian_target_2d,$randn(2),BFGS(),autodiff=:forward) samples=20
end

# ╔═╡ adcc2804-6cf9-43a9-8291-08671c64188c
if run_benchmarks  # Code to compute benchmarks
	local nd = dimen_to_benchmark_gauss  
	target_gauss_list = map(n->make_target(n),nd)
	init_guess_list_gauss = map(n->5.0.*ones(n),nd)
	# Preallocate arrys for storing results
	runtime_gauss_nm_list = zeros(length(nd))
	results_gauss_nm_list = Vector{Any}(undef,length(nd))
	runtime_gauss_bfgs_list = zeros(length(nd))
	results_gauss_bfgs_list = Vector{Any}(undef,length(nd))
	runtime_gauss_bfgs_ad_list = zeros(length(nd))
	results_gauss_bfgs_ad_list = Vector{Any}(undef,length(nd))
	# Benchmark optimization algorithms as function of number of dimensions
	for (i,n) in enumerate(nd)
		GC.gc()
		runtime_gauss_nm_list[i]   = @belapsed (results_gauss_nm_list[$i] = optimize(target_gauss_list[$i],init_guess_list_gauss[$i],NelderMead()) ) samples=10
		GC.gc()
		runtime_gauss_bfgs_list[i] = @belapsed (results_gauss_bfgs_list[$i] = optimize(target_gauss_list[$i],init_guess_list_gauss[$i],BFGS()) ) samples=10
		GC.gc()
		runtime_gauss_bfgs_ad_list[i] = @belapsed (results_gauss_bfgs_ad_list[$i] = optimize(target_gauss_list[$i],init_guess_list_gauss[$i],BFGS(),autodiff=:forward) ) samples=10
	end
end;

# ╔═╡ 248f546e-d836-4960-95c9-c3bc4380861e
if run_benchmarks	# Compute how far the estimated minimum is from the true minimum
	local nd = dimen_to_benchmark_gauss  
	dist_gauss_nm = map(i-> sum( (results_gauss_nm_list[i].minimizer.-mean(target_gauss_list[i])).^2), 1:length(nd) )
	dist_gauss_bfgs = map(i-> sum( (results_gauss_bfgs_list[i].minimizer.-mean(target_gauss_list[i])).^2), 1:length(nd) )
	dist_gauss_bfgs_ad = map(i-> sum( (results_gauss_bfgs_ad_list[i].minimizer.-mean(target_gauss_list[i])).^2), 1:length(nd) )
end;

# ╔═╡ 4327b095-8c53-436e-9dad-2ebba01ee52e
if run_benchmarks && ready_to_see_plt_vs_ndim
	local plt = plot(yscale=:log10, legend=:right)
	scatter!(plt,dimen_to_benchmark_gauss,dist_gauss_nm, label="Nelder Mean (Gradient-free)",color=:blue)
	scatter!(plt,dimen_to_benchmark_gauss,dist_gauss_bfgs, label="BFGS, Numerical Gradients",color=:green)
	scatter!(plt,dimen_to_benchmark_gauss,dist_gauss_bfgs_ad, label="BFGS, Auto-Diff Gradients",color=:red)
	xlabel!("Number of dimensions")
	ylabel!("Distance to true minimum")
	title!(plt,"Accuracy of Algorithms for Gaussian target")
end

# ╔═╡ 78797a3b-99c5-46ad-9bd7-486b18a0a19c
if run_benchmarks && ready_to_see_plt_vs_ndim
	local plt = plot(xscale=:linear,yscale=:log10,legend=:topleft)
	scatter!(plt,dimen_to_benchmark_gauss,runtime_gauss_nm_list, label="Nelder Mean (Gradient-free)",color=:blue)
	plot!(plt,dimen_to_benchmark_gauss,runtime_gauss_nm_list, label=:none,color=:blue)
	scatter!(plt,dimen_to_benchmark_gauss,runtime_gauss_bfgs_list, label="BFGS, Numerical Gradients",color=:green)
	plot!(plt,dimen_to_benchmark_gauss,runtime_gauss_bfgs_list, label=:none,color=:green)
	scatter!(plt,dimen_to_benchmark_gauss,runtime_gauss_bfgs_ad_list, label="BFGS, Auto-Diff Gradients",color=:red)
	plot!(plt,dimen_to_benchmark_gauss,runtime_gauss_bfgs_ad_list, label=:none,color=:red)
	xlabel!(plt,"Number of dimensions")
	ylabel!(plt,"Runtime (s)")
	title!(plt,"Benchmarking Algorithms for Gaussian target")
end

# ╔═╡ fb435357-b356-43c5-9fa9-ab4f3235f2ac
md"## Banana target function"

# ╔═╡ 979a17de-3a33-44a4-85bb-b7e663b46dfb
function banana_2d(x::Vector)
	@assert length(x) == 2 
	a = banana_a::Float64     # We create local copies of global variables 
	b = banana_b::Float64     # with fixed type to improve performance.
	mu = min_prewarp_banana_2d::Vector{Float64}
	dist = MvNormal(mu,PDMat([1.0 0.5; 0.5 1.0]))
	y = [ x[1]/a, x[2]*a + a*b*(x[1]^2+a^2) ]
	-logpdf(dist,y)
end

# ╔═╡ 691dbb96-1348-4a63-aa2c-36c577a77f93
"Compute location of location of minimum for 2D banana"
function compute_loc_min_banana_2d(a, b, min_prewarp::Vector)
	true_min_postwarp_banana_x = min_prewarp[1]*a 
	true_min_postwarp_banana_y = (min_prewarp[2]-a*b*(min_prewarp[1]^2+a^2))/a
	true_min_postwarp_banana_2d = [true_min_postwarp_banana_x, true_min_postwarp_banana_y]
end;

# ╔═╡ 7d188cda-70c8-44b7-9773-911e5f92bb35
true_min_banana_2d = compute_loc_min_banana_2d(banana_a,banana_b, min_prewarp_banana_2d)

# ╔═╡ f2209d29-1c49-499f-949a-85e680559e22
let
	n_plt = 400
	plt_grid = range(-6,stop=6,length=n_plt)
	plt_z = [ banana_2d([plt_grid[i],plt_grid[j]]) for i in 1:n_plt, j in 1:n_plt ]
	local plt = plot()
	contour!(plt,plt_grid,plt_grid,plt_z', levels=40)
#	scatter!(plt,[min_prewarp_banana_2d[1]],[true_min_prewarp_banana_2d[2]], color=:blue, ms=5)
	scatter!(plt,[true_min_banana_2d[1]],[true_min_banana_2d[2]], color=:blue, ms=5, label="True minimum")
	xlabel!(L"x_1")
	ylabel!(L"x_2")
	title!("Gaussian warped into a 2-D Banana")
end

# ╔═╡ 73d84cba-c5a5-40fd-bc99-15a00570d413
function calc_min_banana_nd(d::Integer)
	@assert d>=2
	true_min_banana_nd = repeat(true_min_banana_2d,floor(Int,d//2))
	if d-length(true_min_banana_nd) == 1
	   true_min_banana_nd = vcat(true_min_banana_nd,0.0)
	end
	true_min_banana_nd
end;

# ╔═╡ ea622981-ebca-4407-826e-c7141562e76d
if run_benchmarks # Evaluate accuracy of minima found 
	local nd_list = dimen_to_benchmark_banana
	local ref = map(d->calc_min_banana_nd(d), nd_list )
	dist_banana_nm = map(i-> sum( (results_banana_nm_list[i].minimizer.-ref[i]).^2), 1:length(nd_list) )
	dist_banana_gd = map(i-> sum( (results_banana_gd_list[i].minimizer.-ref[i]).^2), 1:length(nd_list) )
	dist_banana_bfgs = map(i-> sum( (results_banana_bfgs_list[i].minimizer.-ref[i]).^2), 1:length(nd_list) )
	dist_banana_bfgs_ad = map(i-> sum( (results_banana_bfgs_ad_list[i].minimizer.-ref[i]).^2), 1:length(nd_list) )
end;

# ╔═╡ 140c83c4-bbe8-4ffe-b272-5edf10511456
if run_benchmarks
	local plt = plot(yscale=:log10, legend=:right)
	scatter!(plt,dimen_to_benchmark_banana,dist_banana_nm, label="Nelder Mean (Gradient-free)",color=:blue)
	scatter!(plt,dimen_to_benchmark_banana,dist_banana_gd, label="Gradient Descent",color=:purple)
	scatter!(plt,dimen_to_benchmark_banana,dist_banana_bfgs, label="BFGS, Numerical Gradients",color=:green)
	scatter!(plt,dimen_to_benchmark_banana,dist_banana_bfgs_ad, label="BFGS, Auto-Diff Gradients",color=:red)
	xlabel!("Number of dimensions")
	ylabel!("Distance to true minimum")
	title!(plt,"Accuracy of Algorithms for Banana target")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoTest = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f83ec24f76d4c8f525099b2ac475fc098138ec31"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.4.11"

    [ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "a1296f0fe01a4c3f9bf0dc2934efbf4416f5db31"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.4"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "02aa26a4cf76381be7f66e020a3eddeb27b0a092"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.2"

[[ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "d9a8f86737b665e15a9641ecbac64deef9ce6724"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.23.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "e460f044ca8b99be31d35fe54fc33a5c33dd8ed7"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.9.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "5372dbbf8f0bdb8c700db5367132925c0771ef7e"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.1"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

    [ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "938fe2981db009f531b6332e31c58e9584a2f9bd"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.100"

    [Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "a20eaa3ad64254c61eeb5f230d9306e937405434"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.6.1"
weakdeps = ["SparseArrays", "Statistics"]

    [FillArrays.extensions]
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "c6e4a1fbe73b31a3dea94b1da449503b8830c306"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.21.1"

    [FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "d73afa4a2bb9de56077242d98cf763074ab9a970"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.9"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1596bab77f4f073a14c62424283e7ebff3072eca"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.9+1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "19e974eced1768fb46fd6020171f2cec06b1edb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.15"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "81dc6aefcbe7421bd62cb6ca0e700779330acff8"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.25"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "0d097476b6c381ab7906460ef1ef1638fbce1d91"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.2"

[[LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "bbb5c2115d63c2f1451cb70e5ef75e8fe4707019"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.22+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "963b004d15216f8129f6c0f7d187efa136570be0"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.7"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

    [Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "542de5acb35585afcf202a6d3361b430bc1c3fbd"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.13"

[[PlutoTest]]
deps = ["HypertextLiteral", "InteractiveUtils", "Markdown", "Test"]
git-tree-sha1 = "17aa9b81106e661cffa1c4c36c17ee1c50a86eda"
uuid = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
version = "0.2.2"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

[[PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "364898e8f13f7eaaceec55fd3d08680498c0aa6e"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.4.2+3"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[ReverseDiff]]
deps = ["ChainRulesCore", "DiffResults", "DiffRules", "ForwardDiff", "FunctionWrappers", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "Random", "SpecialFunctions", "StaticArrays", "Statistics"]
git-tree-sha1 = "d1235bdd57a93bd7504225b792b867e9a7df38d5"
uuid = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
version = "1.15.1"

[[Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "1e597b93700fa4045d7189afa7c004e0584ea548"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "51621cca8651d9e334a659443a74ce50a3b6dfab"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.3"
weakdeps = ["Statistics"]

    [StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "75ebe04c5bed70b91614d684259b661c9e6274a4"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.0"

[[StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[URIs]]
git-tree-sha1 = "b7a5e99f24892b6824a954199a45e9ffcc1c70f0"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "a72d22c7e13fe2de562feda8645aa134712a87ee"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.17.0"

    [Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "04a51d15436a572301b5abbb9d099713327e9fc4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.4+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cf2c7de82431ca6f39250d2fc4aacd0daa1675c0"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.4+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─2046af3d-aea5-433c-8b70-4916a8b7ef1a
# ╟─8b35ac88-aece-4f78-b0e7-0a03ff1fbe26
# ╟─3daa9062-8c45-41fd-a52d-d042fd86d1b6
# ╟─9463b61d-99e1-4542-828f-14d35f059ba8
# ╟─c0f97de7-832d-46a0-b82d-a04c47bbafce
# ╠═cdfa9982-be89-4949-abb2-363de63bdf89
# ╟─98c67125-0e38-4a3a-8f3f-b99b6bb16682
# ╟─9278ed8e-eb16-4edb-955d-e6621c405285
# ╟─0716f5c7-b682-46ab-9345-a88dd53bb46a
# ╟─b013cf33-c7a6-42ad-8842-aa631f86cace
# ╠═3a94ca69-1b30-4664-9178-68de61393525
# ╟─5ef412aa-c13b-447f-9d0a-9c6b37e8b37a
# ╠═49177429-beb7-4b22-8587-ecead0e473ab
# ╟─fe603b3d-ecc7-4259-985e-4ce827347391
# ╠═5e845450-c80b-4391-aa3e-5cbb71ce2c87
# ╟─0bd886fc-6784-4196-a874-1e20a63683cc
# ╠═aea18eaf-fb64-41ba-8fa6-082e14419152
# ╠═4b391f26-c996-4f4b-b0b1-7871380121eb
# ╟─3192b6ce-e86f-499d-9a17-eb7bcfcd591c
# ╠═88d31552-6086-42a1-afab-7106f81a16f0
# ╠═67d6b3be-9b6b-4609-ac51-80aa6edba4a6
# ╠═1a55adcd-7af1-4024-bea1-62055ac72af6
# ╠═cf7a8fc3-68c3-4431-abb5-f031d967f65d
# ╠═3414e205-e33c-4f14-a329-b63d1ebd411d
# ╠═b075963b-79ed-49f5-8bc8-616196de8b36
# ╟─cadde19b-0e43-476b-afef-dec549b2fdd0
# ╠═e8124c99-b889-46da-a9aa-5f36c3941efe
# ╟─7e7aa8c8-5573-4a52-800b-ed8fce316d94
# ╟─463010d8-a66c-4289-a025-cd74e8ab3f08
# ╠═074ce518-5a0e-4d56-95c6-678967d171e5
# ╟─558c1c42-61cd-4597-b110-19206034cf5d
# ╟─7047987f-0b45-4224-8e41-fb7953ba3e6d
# ╟─b6f54005-2d6c-4629-b8f8-49ee7117d2ba
# ╟─0cb98deb-8807-406f-993d-17ddeb0ff686
# ╟─8dfc25fc-109d-4026-ac1c-3182d8ac1f81
# ╟─c6b4377a-5fff-4e8b-a96f-df971a4d74f6
# ╟─4c4649cb-8f73-41cb-b7f9-e48f540fda7e
# ╟─2b4188b3-8a1c-44a2-acdd-e8a0949e2659
# ╠═b38e119e-e98f-4922-a027-cc6d66a47d96
# ╟─a27e2ba3-f31a-40df-819e-6d46551efae4
# ╟─1a746366-ce12-43ae-b02e-11c5dd713eb6
# ╠═dd3afd35-5508-48e8-b38d-a3342f32ba7c
# ╟─888e04cd-bf48-4219-bd49-346bfacb860f
# ╟─e2ba5402-9be8-4bfe-acfc-5acf4ac01ace
# ╟─b18277bd-e23c-4184-bc10-5a47d83a3df5
# ╟─a585d799-4c14-45e3-8250-d9a19df075b3
# ╟─8aa054de-403b-47e8-9f84-73e2ca2fa4e3
# ╟─54319c3e-fed4-4184-81ce-767f9a527c1b
# ╠═756389e5-662b-4384-a94e-ed05c8c286f5
# ╟─b3d5cf28-64bb-4e62-b72b-e6747a668f37
# ╠═7e751550-75f9-4b2d-9331-6d8ae09ff397
# ╟─e5140102-36e7-450f-ac6d-7f32c0668b98
# ╟─9b02c64d-bed9-4d4c-bdd8-67648dd7bfae
# ╟─61704586-c69f-473f-bfed-cff98767b729
# ╠═13dbca62-fd43-4a8e-a8e7-25e8672b8a60
# ╟─f3cedf20-eda8-4bfe-a8fa-497e1cfe5dd4
# ╟─980e8ae8-7b6c-43cd-a217-b9de54731b6f
# ╠═cb9de107-7a33-4a59-8296-d373024e81b6
# ╠═64c63760-f2d4-482a-8343-7a8d982b0844
# ╠═cf37e7b2-6b3e-44aa-9665-723019f6a95c
# ╟─f318374c-1938-4efd-b540-f3d2afae2c70
# ╟─f9ce690a-fab6-40fd-b4fc-6788840dbe30
# ╟─32801c51-156f-490d-b980-2d98e8f5be23
# ╠═3db6d059-1995-4648-a7e5-eacb40131c8a
# ╟─357e0485-6f6c-4885-ba2c-915afc1c7580
# ╟─286f9794-d139-4058-9fc2-8588d9040f3f
# ╟─2cc9be0e-4b91-4d6a-87da-ad70317c4742
# ╟─f33f8dc4-df49-496b-9da9-503f4beaf1c8
# ╠═2f04f9c4-afd3-4cfe-ae7a-56433e91763f
# ╟─29feb38d-4b5c-4fc2-a7f4-2cdbca7c1f57
# ╟─1b5c212d-aa44-4d4d-bddb-db43280afd6c
# ╟─21bb794f-bd75-44f5-8131-6ce43b3bab77
# ╟─3a3fd94d-16cb-41e8-a3d4-7ae6774b69b6
# ╠═b9e835b4-039e-4601-a85f-06abd54ff0c0
# ╟─adcc2804-6cf9-43a9-8291-08671c64188c
# ╟─248f546e-d836-4960-95c9-c3bc4380861e
# ╟─d3b0b80d-c25d-4028-bd46-ddd2f3db0cdc
# ╟─78797a3b-99c5-46ad-9bd7-486b18a0a19c
# ╟─7513a015-34fb-4cea-a236-facfd6f89b4a
# ╟─4327b095-8c53-436e-9dad-2ebba01ee52e
# ╟─74b79479-fd13-416c-a6bd-d9967624a5b4
# ╠═d9b81c82-4939-4b19-b00e-c26f0ef87aa0
# ╟─150c9fdf-4647-4115-b391-d5d391a505c7
# ╟─6a549339-e4a5-47fa-9f5d-fa5b36a3e302
# ╟─f2209d29-1c49-499f-949a-85e680559e22
# ╠═ba8665f6-883d-485d-86c1-7f42fd1df5ed
# ╟─0f06886f-8eb3-4828-b5dd-b84a4749519b
# ╠═7d188cda-70c8-44b7-9773-911e5f92bb35
# ╟─66e079a4-e14e-4700-adfa-51a05a4cd07e
# ╠═c0d077f4-ab1d-4a79-b6e2-722ddc33b141
# ╟─a1ff09d0-536b-4c1b-94a7-6caebaa45f93
# ╠═450d1d7b-032c-420b-8dd9-34b26daa0c3b
# ╟─2ef2e78f-a324-4bc6-b7ef-6cb43e0491f6
# ╟─31a3807f-2075-4e06-a2fc-8ba4a2ae412a
# ╠═c9ee5aea-4d24-43bd-91c5-5efe42344425
# ╟─60f83c0f-0177-49a1-af98-fe3359ce1aa1
# ╟─bb7c4307-2db1-4447-82a8-1ffa981144c9
# ╠═08a5515e-12f4-4a37-9a8b-e844cd5eda13
# ╠═737cc421-e0ec-4d00-99f1-0201a6ee0bbc
# ╟─a8f36cbe-a8ff-49d1-806a-13d2b27b9d8f
# ╟─0bd2b178-4db5-49bf-b653-6b90dd93f77a
# ╟─0101c723-4cd5-4d91-9a9e-3c39ea0632ba
# ╠═1a2ba5a9-c16f-4a48-b3b1-1bb1ef5caf09
# ╟─73d84cba-c5a5-40fd-bc99-15a00570d413
# ╠═78b3a239-b04f-43f2-9b48-15ce7ada9d3d
# ╠═9c5540e1-2397-4f53-815d-d1eb5694af56
# ╠═c306d465-5e6c-446a-8dd1-b02399ec42ba
# ╠═f36431c7-3da8-4a65-af05-f6e85b27a5f0
# ╟─997042fa-803d-49c4-9ed6-b6e0809d6cf8
# ╠═1e986186-f8a8-4952-84f8-55ce4c684212
# ╟─ea69280d-0af3-48a9-90e3-cb94dc0b23cd
# ╟─6d910e6f-295c-4428-a689-aea3f8b139c7
# ╟─9cf8cd1e-1a17-4245-80e1-c6bbce1c0db1
# ╟─a6e1b07c-2c5a-4e3f-9cb6-e7e35e3f619b
# ╠═2f2debce-d281-4c53-8cb7-77cd4159846d
# ╟─518551bc-bdb5-439e-90c1-369d1dd694ff
# ╠═c02cb04f-e757-4963-bfb6-8213fff1c4cf
# ╠═d0668a5b-4081-4d53-b90a-2f65b14d8d01
# ╠═6a2629c0-cde8-4d1f-9e0f-10c9762b432f
# ╠═a2365327-5843-43e6-9184-b94072f69bac
# ╟─28e3a97b-7208-48a2-97f3-d1003eb89c58
# ╟─6a3ed3ce-3709-4e29-a45e-e1ac3a73a8c2
# ╟─ea622981-ebca-4407-826e-c7141562e76d
# ╟─140c83c4-bbe8-4ffe-b272-5edf10511456
# ╟─1f122def-f9fb-41df-8292-c1b0c93cfb4d
# ╠═598b3abf-8e4c-4244-9b76-93019fb69c41
# ╟─645f1706-a748-492f-bd04-634dfdd6657c
# ╟─868ca537-1e50-45c6-a8ba-12c3d179d192
# ╠═ea8d39ef-3428-4d5f-8520-e20cb8afac92
# ╟─f4c43d3d-894b-4bf0-b37d-3c65a4e2e4b8
# ╟─6d216ba9-ca71-4e92-bf36-c569f0170657
# ╟─e78d0b7d-78d3-4ea7-b7d6-cc275c91ddb5
# ╠═69787870-0b1e-11ec-3354-a914e59bdad2
# ╠═35ce6db0-1066-4321-9e06-be54378211fd
# ╟─b7e4acf0-3528-4e5e-8733-02c1275195af
# ╠═40f3c16f-6fa2-4e14-9e0e-c3f608d3f564
# ╠═c09194bf-6c62-454e-86d2-f2531a68d58c
# ╠═583b06d1-0957-4fb2-af88-6994415bca81
# ╠═ce4c9111-5949-4d92-ba71-10ae0afa8dea
# ╟─fb435357-b356-43c5-9fa9-ab4f3235f2ac
# ╠═979a17de-3a33-44a4-85bb-b7e663b46dfb
# ╠═691dbb96-1348-4a63-aa2c-36c577a77f93
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
