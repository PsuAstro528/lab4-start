### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 6fd32120-4df1-4f2d-bb6f-c348a6999ad5
begin
	using PlutoUI, PlutoTeachingTools
	using CSV, DataFrames
	using Turing,  MCMCChains
	using Plots, LaTeXStrings,StatsPlots
	using DataFrames
	using LinearAlgebra, PDMats
	using Statistics: mean, std
	using Distributions
	using Random
	Random.seed!(123)
	eval(Meta.parse(code_for_check_type_funcs))
end;

# ╔═╡ 10709dbc-c38b-4f15-8ea8-772db2acfbb3
md"""# Astro 528, Lab 4, Exercise 2
## Probabilistic Programing Language
"""

# ╔═╡ 7c12a4d6-1bf6-41bb-9528-9fe11ff8fb70
md"Traditionally, programmers provide explicit instructions for what steps the program should perform (known as the [imperative programming](https://en.wikipedia.org/wiki/Imperative_programming) model).  There are alternatives.  For example, [probabilistic programming languages](https://en.wikipedia.org/wiki/Probabilistic_programming) allow programmers to specify target probability distributions to be computed (or sampled from), without specifying how to perform inference with that target distribution.  In this exercise, we'll explore how to use the [Turing.jl](https://turing.ml/stable/) probabilistic programming language.  Turing is implemented entirely in Julia and is composable, so that one can perform inference with complex Julia models.  Here we'll use a relatively simple model implemented entirely in this notebook."

# ╔═╡ 73021015-4357-4196-868c-e9564de02ede
md"# Example Problem"

# ╔═╡ 26684357-eba5-4a90-b54e-9ad7e64b05a3
md"""
## Physical Model
We aim to infer the orbital properties of an exoplanet orbiting a star based on radial velocity measurements.  
We will approximate the motion of the star and planet as a Keplerian orbit.  In this approximation, the radial velocity perturbation ($Δrv$) due to the planet ($b$) is given by 
```math
\mathrm{Δrv}_b(t) = \frac{K}{\sqrt{1-e^2}} \left[ \cos(ω+T(t,e)) + e \cos(ω) \right],
```
where $K$ is the velocity amplitude (a function of the planet and star masses, the orbital period and the sine of the orbital inclination relative to the sky plane), the orbital eccentricity ($e$), the arguement of pericenter ($\omega$) which specifies the direction of the pericenter, and $T(t,e)$ the [true anomaly](https://en.wikipedia.org/wiki/True_anomaly) at time $t$ (which also depends on $e$).

The true anomaly ($T$) is related to the eccentric anomaly ($E$) by
```math
\tan\left(\frac{T}{2}\right) = \sqrt{\frac{1+e}{1-e}} \tan\left(\frac{E}{2}\right).
```
The eccentric anomaly is related to the mean anomaly by [Kepler's equation](https://en.wikipedia.org/wiki/Kepler%27s_equation).  We will reuse the code to compute the eccentric anomaly from a [previous exercise](https://psuastro528.github.io/lab2-start/ex3.html).  It and the code to compute the true anomaly from the eccentric anomaly and eccentricitiy are at the bottom of the notebook.

The mean anomaly increases linearly in time.
```math
M(t) = 2π(t-t_0) + M_0.
```
Here $M_0$ is the mean anomaly at the epoch  $t_0$.

High-resolution spectroscopy allows for precision *relative* radial velocity measurements.  Due to the nature of the measurement process there is an arbitrary velocity offset ($C$), a nuisance parameter.
```math
\mathrm{rv(t)} = \mathrm{Δrv}_b(t) + C.
```
"""

# ╔═╡ a85268d5-40a6-4654-a4be-cba380e97d35
md"## Statistical model"

# ╔═╡ cfe8d6ad-2125-4587-af70-875e7c4c4844
md"""
In order to perform inference on the parameters of the physical model, we must specify a statistical model.  In Bayesian statistics, we must specify both a [likelihood](https://en.wikipedia.org/wiki/Likelihood_function) function (the probability distribution for a given set of observations for given values of the model parameters) and a [prior distribution](https://en.wikipedia.org/wiki/Prior_probability) for the model parameters (θ).  
		
We will assume that each observation ($rv_i$) follows a normal distribution centered on the true radial velocity ($\mathrm{rv}(t_i)$) at time $t_i$ and that the measurement errors are independent of each other.
```math
L(θ) \sim \prod_{i=1}^{N_{\mathrm{obs}}} N_{\theta}( \mathrm{rv}_i - \mathrm{rv}_{\mathrm{true}}(t_i | \theta), \sigma_{\mathrm{eff},i}^2).
```
Above, $N_x(\mu,\sigma^2)$ indicates a normal probability distribution for $x$ with mean $\mu$ and variance $\sigma^2$.  
We assume that the variance for each measurement ($\sigma_{\mathrm{eff},i}^2$) is given by 
```math
\sigma_{\mathrm{eff},i}^2 = \sigma_i^2 + \sigma_{\mathrm{jitter}}^2, 
```
where $\sigma_i$ is the estimated measurement uncertainty for the $i$th measurement
and $\sigma_{\mathrm{jitter}}$ parameterizes any additional ``noise'' source .  
"""

# ╔═╡ a2140dbe-8736-4ed9-ae6f-b1b0c7df3bc9
md"""
## Observational Data
"""

# ╔═╡ 22719976-86f5-43d3-b890-d3520f9916d2
md"""
We will read in a DataFrame containing radial observations of the star 16 Cygni B from Keck Observatory.  (Data reduction provided by [Butler et al. 2017](https://ui.adsabs.harvard.edu/link_gateway/2017AJ....153..208B/doi:10.3847/1538-3881/aa66ca) and the [Earthbound Planet Search](https://ebps.carnegiescience.edu/) project.)  The star hosts a giant planet on an eccentric orbit ([Cochran et al. 1097](https://doi.org/10.1086/304245)).  In this exercise, we will construct a statistical model for this dataset.
"""

# ╔═╡ e50bdd14-d855-4043-bbab-f6526a972e31
begin
	data_path = occursin("test",pwd()) ? "../data" : "data" 
	df = CSV.read(joinpath(data_path,"16cygb.txt"),DataFrame,header=[:Target,:bjd,:rv,:σ_rv,:col5,:col6,:col7,:col8],skipto=100,delim=' ',ignorerepeated=true)
end;

# ╔═╡ 87dd11ad-c29e-4a5e-90b1-289863eedd57
md"""
It's often useful to do a quick plot to make sure there aren't any suprises (e.g., read wrong column into wrong variable, unexpected units, outliers, etc.).
"""

# ╔═╡ ddac2663-fd90-4c60-acaa-da2560367706
let 
	plt = plot(legend=:none)
	scatter!(plt,df[!,:bjd],df[!,:rv],yerr=df[!,:σ_rv])
	xlabel!(plt,"Time (BJD)")
	ylabel!(plt,"RV (m/s)")
	title!(plt,"Keck Observations of 16 Cyb B")
end

# ╔═╡ b9efe114-717e-46c8-8225-a4ab4d3df439
md"""
Having learned the importance of not having a large time offset in [lab 1's exercise 2](https://psuastro528.github.io/lab1-start/ex2.html), we will specify a reference epoch near many of our observations.   
"""

# ╔═╡ 2844cd3a-9ed1-47da-ab59-ea33575b4991
bjd_ref = 2456200.0

# ╔═╡ f28d4bc8-53e7-45f9-8126-7339a6f54732
md"# Coding up a Probabilistic Model"

# ╔═╡ 56aa2fd5-fb34-495c-9b9f-0ce0bbbd6b1b
md"""
We'll define a probabilistic model use Turing's `@model` macro applied to a julia function.  Our model function takes the observed data (in this case the observation times, radial velocities and the estimated measurement uncertainties) as function arguements.  
In defining a probabilistic model, we specify the distribution that each random variable is to be drawn from using the `~` symbol.  Inside the model, we can specify transformations, whether simple arithmetic and or complex functions calls based on both random and concrete variables.   
"""

# ╔═╡ 37edd756-e889-491e-8710-a54a862a9cd8
@model rv_kepler_model_v1(t, rv_obs, σ_obs) = begin
	# Specify Priors
	P ~ prior_P                  # orbital period
	K ~ prior_K                  # RV amplitude
	e ~ prior_e                  # orbital eccentricity
	ω ~ Uniform(0, 2π)           # arguement of pericenter
	M0_minus_ω ~ Uniform(0,2π)   # mean anomaly at t=0 minus ω
	C ~ Normal(0,1000.0)         # velocity offset
	σ_j ~ prior_jitter           # magnitude of RV jitter
	
	# Transformations to make sampling easier
	M0 = M0_minus_ω + ω

	# Reject any parameter values that are unphysical, _before_ trying 
	# to calculate the likelihood to avoid errors/assertions
	if !(0.0 <= e < 1.0)      
        Turing.@addlogprob! -Inf
        return
    end
	
    # Calculate the true velocity given model parameters
	rv_true = calc_rv_keplerian_plus_const.(t, P,K,e,ω,M0,C)  
	
	# Specify model likelihood for the observations
	σ_eff = sqrt.(σ_obs.^2 .+ σ_j.^2)
 	rv_obs ~ MvNormal(rv_true, σ_eff )
end

# ╔═╡ 84c6b16b-ff8c-4a05-ac97-aa610f328370
md"""
In the model above, we used some common distributions provided by the [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) package, as well as some less common distributions specified in the cell below.  For the eccentricity prior, we adopt a [Rayleigh distribution](https://en.wikipedia.org/wiki/Rayleigh_distribution) which we truncated to remain less than unity (so the planet remains bound).  For the period and amplitude, we adopt a modified Jeffreys prior for a scale parameter
```math
p(x) ∝ \frac{1}{1+\frac{x}{x_0}}, \qquad 0 \le x \le x_{\max}
```
as suggested in [Ford & Gregory (2007)](https://ui.adsabs.harvard.edu/#abs/2007ASPC..371..189F/abstract). 
Since this is not provided in Distributions, we implement this distribution ourselves near the bottom of the notebook.  This may be a useful example for anyone whose class project involves performing statistical inference on data.    
"""

# ╔═╡ 228bb255-319c-4e80-95b3-8bf333be29e4
md"""
Remember that our model is written as a function of the observational data.  
Therefore, we will specify a posterior probability distribution for a given set of observational data.  
"""

# ╔═╡ 776a96af-2c4f-4d6d-9cec-b5db127fed6c
posterior_1 = rv_kepler_model_v1(df.bjd.-bjd_ref,df.rv,df.σ_rv)

# ╔═╡ 5ab4a787-fe9b-4b2c-b14c-273e0205259d
md"""
## Sampling from the Posterior
"""

# ╔═╡ 5f20e01d-3489-4828-9e4e-119095e9c29c
md"""
Since our model includes non-linear functions, the posterior distribution for the model parameters can not be computed analytically.  Fortunately, there are sophisticated algorithms for sampling from a probability distribution (e.g., [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo); MCMC).  Soon, we will compute a few posterior samples and investigate the results.
"""

# ╔═╡ dfdf843d-98ce-40a1-bd0b-0a11f1cdb5f9
md"""
The next calculation can take seconds to minutes to run, so I've provide boxes to set some parameters below.  The default number of steps per Markov chain is much smaller than you'd want to use for a real scientific study, but is likely enough to illustrate the points for this lab.  Near the end of this exercise (i.e., after you've made a good initial guess for the orbital period below), feel free to dial it up and  wait a few to several minutes to see the improved results for the subsequent calculations.
"""

# ╔═╡ 87a2281b-8827-4eca-89a6-c7d77aa7a00f
md"Number of steps per Markov chain  $(@bind num_steps_per_chain NumberField(1:10_000; default=500)) "

# ╔═╡ 358ab0f4-3423-4728-adfa-44126744ae18
md"""
In the cell above, we called Turing's `sample` function applied to the probability distribution given by `posterior_1`, specified that it should use the [No U-Turn Sampler (NUTS)](https://arxiv.org/abs/1111.4246), asked for the calculation to be parallelized using multiple threads, and specified the number and length of Markov chains to be computed.
"""

# ╔═╡ c1fa4faa-f10c-4ed0-9539-20f5fcde8763
md"## Inspecting the Markov chains"

# ╔═╡ 00c4e783-841c-4d6a-a53d-34a88bbe1f7a
md"""
In the above calculations, we drew the initial model parameters from the prior probability distrirbution.  Sometimes those are very far from the true global mode.  For a simple model, it's possible that all the chains will (eventually) find the global mode.  However, the different Markov chains might have  gotten "stuck" in different local maxima of the posterior density, depending on where each started.  We can visualize how the Markov chains progressed by looking at a trace plot.
"""

# ╔═╡ f7046ee4-9eb7-4324-886e-5a3650b58db7
tip(md"""
Since we know the initial state of each Markov chain are strongly influenced by our initial guess, we usually discard the first portion of each Markov chain.  Normally, Turing does this automatically.  Above, we explicitly passed the optional arguement 'discard_initial=0', so that we could make the above plot and see where each chain started.
""")

# ╔═╡ bb2ff977-2d12-4baa-802c-097a4138a24b
md"""
The [MCMCChains.jl](https://github.com/TuringLang/MCMCChains.jl) package provides a `describe` function that provides a simple summary of Markov chain results.  
"""

# ╔═╡ 1196e7e6-4144-47e8-97cd-6f93530e11e5
md"""
*In theory*, each Markov chains is guaranteed to converge to the same posterior distribution, *if* we were to run it infinitely long.  In practice, we can't run things infinitely long. So we must be careful to check for any indications that the Markov chains might not have converged.  The above table has some statistics that can be useful for recognizing signs of non-convergence.  But since this isn't a statistics class, we'll just use one simple qualitative test.  Since we've computed several Markov chains (potentially in parallel to save our time), we can compare the results.  If they don't agree, then they can't all have converged to the same distribution.
"""

# ╔═╡ 55a3378a-da18-4310-8272-cf11298a118d
md"""
We can inspect the results of each of the Markov chains separately, by taking a 2-D "slice", where we fixed the value of the third axis to `chain_id` set by the slider below.
"""

# ╔═╡ ac237181-b6d2-4968-b6e3-146ba41bcc18
md"""
Below, we'll visualize the results for one important parameter by ploting an estimate of the marginal density for the orbital period from each of the Markov chains we've computed.  
"""

# ╔═╡ a8e547c5-c2b5-4058-a1e2-35397d7e8112
md"2a.  Based on the summary statistics and/or estimated marginal posterior density for the orbital period, did the different Markov chains provide qualitatively similar results for the marginal distribution of the orbital period?"

# ╔═╡ 80d076a8-7fb1-4dc7-8e3b-7579db30b038
response_2a = missing

# ╔═╡ 3759ec08-14f4-4026-916b-6a0bbc818ca7
display_msg_if_fail(check_type_isa(:response_2a,response_2a,[AbstractString,Markdown.MD]))

# ╔═╡ 2fa23729-007e-4dcb-b6f9-67407929c53a
tip(md"""
Since it's a random algorithm, I can't be 100% sure what results you'll get.  My guess is that if you ran several Markov chains for the suggested model and dataset, then some will likely have resulted in substantially different marginal density estimates.
""")

# ╔═╡ 52f4f8bd-41a9-446f-87e7-cdc1e9c178f8
md"## Visualize the model predictions"

# ╔═╡ bf5b40c5-399e-40a0-8c27-335f72015e58
md"Another good strategy to help check whether one is happy with the results of a statistic model is to compare the predictions of the model to the observational data.  Below, we'll draw several random samples from the posterior samples we computed and plot the model RV curve predicted by each set of model parameters." 

# ╔═╡ 98bc7209-a6a1-4c06-8e6f-de5b421b7b8c
t_plt = range(first(df.bjd)-bjd_ref-10, stop=last(df.bjd)-bjd_ref+10, length=400);

# ╔═╡ 207a0f69-30ce-4107-a21d-ba9f6c5a5039
 n_draws = 10;

# ╔═╡ 93115337-ec60-43b9-a0ef-1d3ffc1fe5ff
@bind draw_new_samples_from_posterior_with_init_from_prior Button("Draw new samples")

# ╔═╡ 0d83ebc9-ecb3-4a9d-869b-351b28d5e962
md"2b.  Are any/some/most/all of the predicted RV curves good fits to the data?"

# ╔═╡ cc2dbf2f-27a4-4cd6-81e5-4a9f5a3bc488
response_2b = missing

# ╔═╡ f00539c9-ee40-4128-928c-460fd439fa87
display_msg_if_fail(check_type_isa(:response_2b,response_2b,[AbstractString,Markdown.MD]))

# ╔═╡ b787d7ef-bda6-470d-93b8-954227dec24b
md"# Global search with an approximate model"

# ╔═╡ aaf24ba9-48dd-49ff-b760-2b806f7cf5ba
md"""
It turns out that the posterior distribution that we're trying to sample from is *highly* multimodal.  We can get a sense for this by considering a *periodogram*.  For our purposes, this is simply an indication of the goodness of the best-fit model for each of many possible periods.  To make the calculations faster, we've assumed that the orbits are circular, allowing us to make use of generalized linear least squares fitting which is much faster than fitting a non-linear model.  This allows us to quickly explore the parameter space and identify regions worthy of more detailed investigation.  
"""

# ╔═╡ e62067ce-dbf6-4c50-94e3-fce0b8cbef12
md"""
We can see that there are multiple orbital periods that might be worth exploring, but they are separated by deep valleys, where models would not fit well at all.  Strictly speaking the ideal Markov chain would repeatedly transition between orbital solutions in each of the posterior modes.  In practice, most MCMC algoritms are prone to getting stuck in one mode for difficult target densities like this one.   Therefore, we will compute new Markov chains using the same physical and statistical model, but starting from initial guesses near the posterior mode that we want to explore.
"""

# ╔═╡ dd918d00-fbe7-4bba-ad6a-e77ea0acb43a
md"## Initial guess for model parameters"

# ╔═╡ 9c82e944-d5f2-4d81-9b33-7e618421b715
md"""
Below, I've provided a pretty good guess below that is likely to give good results.  After you're read through the lab, you're welcome to come back to this part and see what happens when you try initializing the Markov chains with different initial states.
"""

# ╔═╡ b76b3b7f-c5f3-4aec-99b5-799479c124d6
begin
	P_guess = 710.0         # d
	K_guess = 40.9          # m/s
	e_guess = 0.70
	ω_guess = 1.35          # rad
	M0_minus_ω_guess = 4.35 # rad
	C_guess = -2.3          # m/s
	jitter_guess = 3.0      # m/s
	param_guess = (;P=P_guess, K=K_guess, e=e_guess, ω=ω_guess, M0_minus_ω=M0_minus_ω_guess, C=C_guess, σ_j=jitter_guess)
end

# ╔═╡ 3459436a-bfeb-45d4-950e-68fd55af76d7
md"## Visualize model predictions"

# ╔═╡ 47e61e3a-466d-40f9-892e-5691eb6a2393
md"""
Next, we'll try computing a new set of Markov chains using our guess above to initialize the Markov chains.
"""

# ╔═╡ ba4e0dc6-5008-4ed8-8eba-3902820cf241
md"# Sampling from the posterior distribution with an initial guess"

# ╔═╡ a247788c-8072-4cc4-8c38-68d0a3445e83
md"""
I'm ready to compute posterior sample with new guess: $(@bind go_sample_posterior1 CheckBox(default=true))

(Uncheck box above if you want to inspect the predicted RV curve using several different sets of orbital parameters before you compute a new set of Markov chains.)
"""

# ╔═╡ 565d68a4-24cd-479a-82ab-21d64a6a01f6
md"### Inspecting the new Markov chains"

# ╔═╡ db2ba5d1-3bd3-4208-9805-9c3fab259377
md"""
Again, we'll check the summary of posterior samples for the  model parameters for the group of Markov chains and for each chain individually.
"""

# ╔═╡ c9433de9-4f48-4bc6-846c-3d684ae6adee
md"We can also compare the marginal posterior density estimated from each Markov chain separately for each of the model parameters." 

# ╔═╡ 7f8394ee-e6a2-4e4f-84b2-10a043b3da35
md"2c.  Based on the summary statistics and estimated marginal posterior densities, did the different Markov chains provide qualitatively similar results for the marginal distributions of the orbital period and other model parameters?"

# ╔═╡ f2ff4d87-db0a-4418-a548-a9f3f04f93cd
response_2c = missing

# ╔═╡ 42fe9282-6a37-45f5-a833-d2f6eb0518fe
display_msg_if_fail(check_type_isa(:response_2c,response_2c,[AbstractString,Markdown.MD]))

# ╔═╡ b38f3767-cd14-497c-9576-22764c53a48d
protip(md"Assessing the performance of Markov chain Monte Carlo algorithm could easily be the topic of a whole lesson.  We've only scratched the surface here.  You can find [slides](https://astrostatistics.psu.edu/su18/18Lectures/AstroStats2018ConvergenceDiagnostics-MCMCv1.pdf) from a lecture on this topic for the [Penn State Center for Astrostatistics](https://astrostatistics.psu.edu) summer school.")  

# ╔═╡ 35562045-abba-4f36-be92-c41f71591b1a
md"### Visualize predictions of new Markov chains"

# ╔═╡ e9530014-1186-4897-b875-d9980e0c3ace
md"Below, we'll draw several random samples from the posterior samples we computed initializing the Markov chains with our guess at the model parameters and plot the model RV curve predicted by each set of model parameters." 

# ╔═╡ 9dc13414-8266-4f4d-94dd-f11c0726c539
@bind draw_new_samples_from_posterior_with_guess Button("Draw new samples")

# ╔═╡ 95844ef7-8730-469f-b69b-d7bbe5fc2607
md"2d.  Are any/some/most/all of the predicted RV curves good fits to the data? "

# ╔═╡ 3699772e-27d4-402e-a669-00f5b22f2ed5
response_2d = missing

# ╔═╡ 9eccb683-d637-4ab9-8af7-70c24a7d8478
display_msg_if_fail(check_type_isa(:response_2d,response_2d,[AbstractString,Markdown.MD]))

# ╔═╡ 9a2952be-89d8-41ef-85ce-310ed90bd0d1
md"# Generalize the Model"

# ╔═╡ c7e9649d-496d-44c2-874c-5f51e311b21d
md"""
One of the benefits of probabilistic programming languages is that they make it relatively easy to compare results using different models.
The starter code below defines a second model identical to the first.  Now, it's your turn to modify the model to explore how robust the planet's orbital parameters are.  

The star 16 Cygni B is part of a wide binary star system.  The gravitational pull of 16 Cygni A is expected to cause an acceleration on the entire 16 Cygni B planetary system.  Since the separation of the two stars is large, the orbital period is very long and the radial velocity perturbation due to the star A over the short time span of observations can be approximated as a constant acceleration,  
```math
\Delta~\mathrm{rv}_A = a\times t,
```
and the observations can be modeled as the linear sum of the perturbations due to the planet and the perturbations due to the star,
```math
\mathrm{rv(t)} = \mathrm{Δrv}_b(t) + \mathrm{Δrv}_A(t) + C.
```

2e. Update the model below to include an extra model parameter ($a$) for the acceleration due to star A and to include that term in the true velocities.
You'll need to choose a reasonable prior distribution for $a$.    
"""

# ╔═╡ 3cfc82d6-3390-4572-b5f0-124503e2e9e0
@model rv_kepler_model_v2(t, rv_obs, σ_obs) = begin
	# Specify Priors
	P ~ prior_P                  # orbital period
	K ~ prior_K                  # RV amplitude
	e ~ prior_e                  # orbital eccentricity
	ω ~ Uniform(0, 2π)           # arguement of pericenter
	M0_minus_ω ~ Uniform(0,2π)   # mean anomaly at t=0 minus ω
	C ~ Normal(0,1000.0)         # velocity offset
	σ_j ~ prior_jitter           # magnitude of RV jitter
	# TODO:  Set prior for a
	
	# Transformations to make sampling easier
	M0 = M0_minus_ω + ω

	# Reject any parameter values that are unphysical, _before_ trying 
	# to calculate the likelihood to avoid errors/assertions
	if !(0.0 <= e < 1.0)      
        Turing.@addlogprob! -Inf
        return
    end
	
    # Calculate the true velocity given model parameters
	# TODO: Update to include an acceleration
	rv_true = calc_rv_keplerian_plus_const.(t, P,K,e,ω,M0,C)  
	
	# Specify model likelihood for the observations
	σ_eff = sqrt.(σ_obs.^2 .+ σ_j.^2)
	rv_obs ~ MvNormal(rv_true, σ_eff )
end

# ╔═╡ 5084b413-1211-490a-89d3-1cc782f1741e
md"## Sampling from the new model"

# ╔═╡ c6939875-0efc-4f1d-a2d3-6b46484328a5
md"""
Since we have a new statistical model, we'll need to define a new posterior based on the new statistical model and our dataset.
"""

# ╔═╡ fb2fe33c-3854-435e-b9e5-60e8531fd1f3
posterior_2 = rv_kepler_model_v2(df.bjd.-bjd_ref,df.rv,df.σ_rv)

# ╔═╡ 43177135-f859-423d-b70c-e755fdd06765
md"""
Since the new model includes an extra variable, we'll need to update our initial guess to include the new acceleration term.  We'll choose a small, but non-zero value.  (Using a to exactly zero could make it hard for the sampling algorithm to find an appropriate scale.)  
"""

# ╔═╡ c33d9e9d-bf00-4fa4-9f90-33415385507e
param_guess_with_acc = merge(param_guess, (;a = 1e-4) )

# ╔═╡ 20393a82-6fb1-4583-b7b7-8d1cda43bd47
md"""
We're you're ready to start sampling from the new posterior, check the box below.

Ready to sample from new posterior using your new model? $(@bind go_sample_posterior2 CheckBox(default=false))
"""

# ╔═╡ 13c56edd-3c1e-496e-8116-fb158dd0f133
md"## Inspect Markov chains for generalized model"

# ╔═╡ c2b7dfcc-6238-4e9a-a2b8-efcc05c17361
md"""
Let's inspect summary statistics and marginal distributions as we did for the previous model.
"""

# ╔═╡ 21fdeaff-0c91-481c-bd8e-1dba27e275a6
md"""
2f.  Based on the the above summary statistics and estimated marginal posterior densities from each of the Markov chains, do you see any reason to be suspicious of the results from the new analysis using the model with a long-term acceleration?
"""

# ╔═╡ a4be55ab-3e8e-423c-b3bc-3f3b88c5d2b7
response_2f = missing

# ╔═╡ 5efb1eac-c1fe-417f-828a-3cfb8978da40
display_msg_if_fail(check_type_isa(:response_2f,response_2f,[AbstractString,Markdown.MD]))

# ╔═╡ bcedc85f-8bf2-49d4-a60a-d6020450fd76
md"## Visualize predictions of generalized model"

# ╔═╡ d751f93c-9ef4-41ad-967b-7ccde6e40afd
@bind draw_new_samples_from_posterior_with_acc Button("Draw new samples")

# ╔═╡ 5171f2f0-e60c-4038-829f-9baf2d5f178e
md"To see if the model better describes our data, we can inspect the histogram of residuals between the observations and the model predictions for each of the models.  "

# ╔═╡ 083e9418-9b64-46d0-8da4-3396fb958862
md"Standardize Residuals? $(@bind standardize_histo CheckBox(default=false))"

# ╔═╡ 8180b43d-81aa-4be0-bdf1-ac93f734331c
md"# Compare results using two different models"

# ╔═╡ d9cd7102-490d-4f35-a254-816c069d3810
md"""
Finally, we'll compare our estimates of the marginal posterior distributions computed using the two models.   
"""

# ╔═╡ 9fd50ada-702f-4ca4-aab2-abfa0f4f597c
md"""
2g.  Did the inferences for the orbital period or velocity amplitude change significantly depending on which model was assumed?  
"""

# ╔═╡ b3f9c7b7-5ed5-47d7-811c-6f4a313de24b
response_2g = missing

# ╔═╡ 461416e4-690f-4ebd-9f07-3e34962c8693
display_msg_if_fail(check_type_isa(:response_2g,response_2g,[AbstractString,Markdown.MD]))

# ╔═╡ e5d0b1cc-ea7b-42a4-bfcd-684337b0f98b
md"""
2h.  Describe how a probabilistic programming language could be a useful tool for you, whether that be for your class project or a current/past/future research project.
"""

# ╔═╡ 5e5d4560-fa1e-48f6-abe4-3b1221d44609
response_2h = missing

# ╔═╡ dd40a3cf-76d3-4eb9-8027-274a065c762c
display_msg_if_fail(check_type_isa(:response_2h,response_2h,[AbstractString,Markdown.MD]))

# ╔═╡ 940f4f42-7bc3-48d4-b9f4-22f9b94c345d
md"""
# Helper Code
"""

# ╔═╡ 40923752-9215-45c9-a444-5a519b64df97
ChooseDisplayMode()

# ╔═╡ dfe71f99-5977-46e5-957d-10a32ce8340e
TableOfContents(aside=true)

# ╔═╡ 16802453-5fdf-4e1f-8e99-51d1fcaa248a
num_threads = Threads.nthreads()

# ╔═╡ 9f27ff9f-c2c4-4bf4-b4cd-bac713de0abe
md"Number of Markov chains  $(@bind num_chains NumberField(1:num_threads; default=num_threads))"

# ╔═╡ 3a838f95-283b-4e06-b54e-400c1ebe94f8
chains_rand_init = sample(posterior_1, NUTS(), MCMCThreads(), num_steps_per_chain, num_chains, discard_initial=0)

# ╔═╡ 6fad4fd9-b99b-44a9-9c45-6e79ffd4a796
traceplot(chains_rand_init,:P)

# ╔═╡ 62c0c2ca-b43b-4c53-a297-08746fae3f6e
describe(chains_rand_init)

# ╔═╡ 1966aff9-ee3d-46f0-be0f-fed723b14f30
md"Chain to calculate summary statistics for: $(@bind chain_id Slider(1:size(chains_rand_init,3);default=1))"

# ╔═╡ 764d36d4-5a3b-48f6-93d8-1e15a44c3ace
md"**Summary statistics for chain $chain_id**"

# ╔═╡ 71962ff1-b769-4601-8b50-7484ca3a0d91
describe(chains_rand_init[:,:,chain_id])

# ╔═╡ c81bb742-a911-4fba-9e85-dfb9a241b290
density(chains_rand_init,:P)

# ╔═╡ 7a664634-c6e3-464b-90a0-6b4cf5015e83
if go_sample_posterior1 && (P_guess > 0)
	chains_with_guess = sample(posterior_1, NUTS(), MCMCThreads(), num_steps_per_chain*2, num_chains; init_params = fill(param_guess, num_chains))
	#chains_with_guess = sample(posterior_1, NUTS(), num_steps_per_chain*2; init_params = param_guess)
end;

# ╔═╡ e9465e3f-f340-4cc5-9fa2-454feaa6bd4d
if @isdefined chains_with_guess
	chain_summary_stats = describe(chains_with_guess)
end

# ╔═╡ 3c3ecf3e-5eba-4bb1-92f7-449680be4edd
if @isdefined chains_with_guess
md"Chain to calculate summary statistics for: $(@bind chain_id2 Slider(1:size(chains_with_guess,3);default=1))"
end

# ╔═╡ b7766545-f1a9-4635-905a-fa3e798f12dc
md"**Summary statistics for chain $chain_id2**"

# ╔═╡ 6441c9cf-f0b1-4229-8b72-1b89e9f0c6f3
if @isdefined chains_with_guess
	describe(chains_with_guess[:,:,chain_id2])
end

# ╔═╡ 3b7580c9-e04e-4768-ac6f-ddb4462dedd8
density(chains_with_guess)

# ╔═╡ 8ffb3e78-515b-492d-bd6f-b24eb08e93d6
if go_sample_posterior2 && (P_guess > 0)
	chains_posterior2 = sample(posterior_2, NUTS(), MCMCThreads(), num_steps_per_chain, num_chains; init_params = fill(param_guess_with_acc,num_chains))
end;

# ╔═╡ cd1ca1b3-a0ec-4d10-90a3-7648fe52f206
if @isdefined chains_posterior2
	describe(chains_posterior2)
end

# ╔═╡ 9d57a595-fad9-4263-baf4-33d4ac5209f7
if @isdefined chains_posterior2
md"Chain to calculate summary statistics for: $(@bind chain_id3 Slider(1:size(chains_posterior2,3);default=1))"
end

# ╔═╡ 4c05803d-3b8d-4c03-9c7a-3c589227a807
if @isdefined chains_posterior2
	describe(chains_posterior2[:,:,chain_id3])
end

# ╔═╡ 8a4ae52b-9cc6-478f-b836-62e59694949e
if @isdefined chains_posterior2
	density(chains_posterior2)
end

# ╔═╡ 9987e752-164f-40df-98ed-073d715ad87b
if @isdefined chains_posterior2
	local plt1 = density(chains_with_guess,:P)
	local plt2 = density(chains_posterior2,:P)
	title!(plt1,"p(P | data, model w/o acceleration)")
	title!(plt2,"p(P | data, model w/ acceleration)")
	plot(plt1, plt2, layout = @layout [a; b] )	
end

# ╔═╡ 2fb25751-a036-4156-9fbd-3aaf4e373b91
if @isdefined chains_posterior2
	local plt1 = density(chains_with_guess,:K)
	local plt2 = density(chains_posterior2,:K)
	title!(plt1,"p(K | data, model w/o acceleration)")
	title!(plt2,"p(K | data, model w/ acceleration)")
	plot(plt1, plt2, layout = @layout [a; b] )	
end

# ╔═╡ 1dfb8d58-b9f3-47a3-a7b1-e8354e7db4e2
if @isdefined chains_posterior2
	local plt1 = density(chains_with_guess,:e)
	local plt2 = density(chains_posterior2,:e)
	title!(plt1,"p(e | data, model w/o acceleration)")
	title!(plt2,"p(e | data, model w/ acceleration)")
	plot(plt1, plt2, layout = @layout [a; b] )	
end

# ╔═╡ 0f1bf89e-c195-4c5f-9cd9-a2982b2e7bf0
if @isdefined chains_posterior2
	local plt1 = density(chains_with_guess,:ω)
	local plt2 = density(chains_posterior2,:ω)
	title!(plt1,"p(ω | data, model w/o acceleration)")
	title!(plt2,"p(ω | data, model w/ acceleration)")
	plot(plt1, plt2, layout = @layout [a; b] )	
end

# ╔═╡ 83485386-90fd-4b2d-bdad-070835d8fb44
md"""
## Solving Kepler's Equation
"""

# ╔═╡ b01b2df0-9cfd-45c8-ba35-cd9ec018af6a
"""
   ecc_anom_init_guess_danby(mean_anomaly, eccentricity)

Returns initial guess for the eccentric anomaly for use by itterative solvers of Kepler's equation for bound orbits.  

Based on "The Solution of Kepler's Equations - Part Three"
Danby, J. M. A. (1987) Journal: Celestial Mechanics, Volume 40, Issue 3-4, pp. 303-312 (1987CeMec..40..303D)
"""
function ecc_anom_init_guess_danby(M::Real, ecc::Real)
	#@assert -2π<= M <= 2π
	@assert 0 <= ecc <= 1.0
    M = mod2pi(M)
	if  M < zero(M)
		M += 2π
	end
    E = (M<π) ? M + 0.85*ecc : M - 0.85*ecc
end;

# ╔═╡ f104183b-ea56-45c3-987e-94e42d687143
"""
   update_ecc_anom_laguerre(eccentric_anomaly_guess, mean_anomaly, eccentricity)

Update the current guess for solution to Kepler's equation
  
Based on "An Improved Algorithm due to Laguerre for the Solution of Kepler's Equation"
   Conway, B. A.  (1986) Celestial Mechanics, Volume 39, Issue 2, pp.199-211 (1986CeMec..39..199C)
"""
function update_ecc_anom_laguerre(E::Real, M::Real, ecc::Real)
  (es, ec) = ecc .* sincos(E)
  F = (E-es)-M
  Fp = one(M)-ec
  Fpp = es
  n = 5
  root = sqrt(abs((n-1)*((n-1)*Fp*Fp-n*F*Fpp)))
  denom = Fp>zero(E) ? Fp+root : Fp-root
  return E-n*F/denom
end;

# ╔═╡ 9c50e9eb-39a0-441a-b03f-6358caa2d0e9
begin
	"""
	   calc_ecc_anom( mean_anomaly, eccentricity )
	   calc_ecc_anom( param::Vector )
	
	Estimates eccentric anomaly for given 'mean_anomaly' and 'eccentricity'.
	If passed a parameter vector, param[1] = mean_anomaly and param[2] = eccentricity. 
	
	Optional parameter `tol` specifies tolerance (default 1e-8)
	"""
	function calc_ecc_anom end
	
	function calc_ecc_anom(mean_anom::Real, ecc::Real; tol::Real = 1.0e-8)
	  	if !(0 <= ecc <= 1.0)
			println("mean_anom = ",mean_anom,"  ecc = ",ecc)
		end
		@assert 0 <= ecc <= 1.0
		@assert 1e-16 <= tol < 1
	  	M = rem2pi(mean_anom,RoundNearest)
	    E = ecc_anom_init_guess_danby(M,ecc)
		local E_old
	    max_its_laguerre = 200
	    for i in 1:max_its_laguerre
	       E_old = E
	       E = update_ecc_anom_laguerre(E_old, M, ecc)
	       if abs(E-E_old) < tol break end
	    end
	    return E
	end
	
	function calc_ecc_anom(param::Vector; tol::Real = 1.0e-8)
		@assert length(param) == 2
		calc_ecc_anom(param[1], param[2], tol=tol)
	end;
	
end

# ╔═╡ b9267ff2-d401-4263-bf25-d52be6260859
md"""
## RV perturbation by planet on a Keplerian orbit
"""

# ╔═╡ 1492e4af-440d-4926-a4b5-b33da77dbee2
function calc_true_anom(ecc_anom::Real, e::Real)
	true_anom = 2*atan(sqrt((1+e)/(1-e))*tan(ecc_anom/2))
end

# ╔═╡ 0ad398fb-9c7e-467d-a932-75db70cd2e86
begin 
	""" Calculate RV from t, P, K, e, ω and M0	"""
	function calc_rv_keplerian end 
	calc_rv_keplerian(t, p::Vector) = calc_rv_keplerian(t, p...)
	function calc_rv_keplerian(t, P,K,e,ω,M0) 
		mean_anom = t*2π/P-M0
		ecc_anom = calc_ecc_anom(mean_anom,e)
		true_anom = calc_true_anom(ecc_anom,e)
		rv = K/sqrt((1-e)*(1+e))*(cos(ω+true_anom)+e*cos(ω))
	end
end

# ╔═╡ d30799d5-6c82-4987-923e-b8beb2aac74a
begin 
	""" Calculate RV from t, P, K, e, ω, M0	and C   """
	function calc_rv_keplerian_plus_const end 
	calc_rv_keplerian_plus_const(t, p::Vector) = calc_rv_keplerian_plus_const(t, p...)
	function calc_rv_keplerian_plus_const(t, P,K,e,ω,M0,C) 
		calc_rv_keplerian(t, P,K,e,ω,M0) + C
	end
end

# ╔═╡ 7ef2f1ca-d542-435b-abdd-03af4b4257f3
if  @isdefined param_guess
	local plt = plot(legend=:none)
	scatter!(plt,df[!,:bjd].-bjd_ref,df[!,:rv],yerr=df[!,:σ_rv])
	#local t_plt = range(first(df.bjd)-bjd_ref-10, stop=last(df.bjd)-bjd_ref+10, length=400)
	rvs_plt = calc_rv_keplerian_plus_const.(t_plt,P_guess,K_guess,e_guess,ω_guess,M0_minus_ω_guess+ω_guess,C_guess)
	plot!(plt,t_plt,rvs_plt) 
	xlabel!(plt,"Time (d)")
	ylabel!(plt,"RV (m/s)")
	title!("RV Predictions using above guess for model parameters") 
end

# ╔═╡ 7330040e-1988-4410-b625-74f71f031d43
function simulate_rvs_from_model_v1(chain, times; sample::Integer, chain_id::Integer=1)
	@assert 1<=sample<=size(chain,1)
	@assert 1<=chain_id<=size(chain,3)
	# Extract parameters from chain
	P = chain[sample,:P,chain_id]
	K = chain[sample,:K,chain_id]
	e = chain[sample,:e,chain_id]
	ω = chain[sample,:ω,chain_id]
	M0_minus_ω = chain[sample,:M0_minus_ω,chain_id]
	C = chain[sample,:C,chain_id]

	M0 = M0_minus_ω + ω
	rvs = calc_rv_keplerian_plus_const.(times, P,K,e,ω,M0,C)
end

# ╔═╡ bdac967d-82e0-4d87-82f7-c771896e1797
begin 
	""" Calculate RV from t, P, K, e, ω, M0, C and a	"""
	function calc_rv_keplerian_plus_acc end 
	calc_rv_keplerian_plus_acc(t, p::Vector) = calc_rv_keplerian_plus_acc(t, p...)
	function calc_rv_keplerian_plus_acc(t, P,K,e,ω,M0,C,a) 
		#t0 = bjd_ref::Float64
		calc_rv_keplerian(t, P,K,e,ω,M0) + C + a*t
	end
end

# ╔═╡ 93adb0c3-5c11-479b-9436-8c7df34bd8fe
function simulate_rvs_from_model_v2(chain, times; sample::Integer, chain_id::Integer=1)
	@assert 1<=sample<=size(chain,1)
	@assert 1<=chain_id<=size(chain,3)
	# Extract parameters from chain
	P = chain[sample,:P,chain_id]
	K = chain[sample,:K,chain_id]
	e = chain[sample,:e,chain_id]
	ω = chain[sample,:ω,chain_id]
	M0_minus_ω = chain[sample,:M0_minus_ω,chain_id]
	C = chain[sample,:C,chain_id]
	a = chain[sample,:a,chain_id]
	M0 = M0_minus_ω + ω
	rvs = calc_rv_keplerian_plus_acc.(times, P,K,e,ω,M0,C,a)
end

# ╔═╡ f3fb26a5-46b5-4ba3-8a30-9a88d6868a24

function make_logp(
    model::Turing.Model,
    sampler=Turing.SampleFromPrior(),
    ctx::Turing.DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
)
    vi = Turing.VarInfo(model)

    # define function to compute log joint.
    function ℓ(θ)
        new_vi = Turing.VarInfo(vi, sampler, θ)
        model(new_vi, sampler, ctx)
        logp = Turing.getlogp(new_vi)
        return logp
    end

end

# ╔═╡ 3fd2ec1a-fed7-43a6-bc0b-ccadb1f711dd
md"""
## A custom prior probability distribution
"""

# ╔═╡ f1547a42-ee3b-44dc-9147-d9c8ec56f1e3
begin
	struct ModifiedJeffreysPriorForScale{T1,T2,T3} <: ContinuousUnivariateDistribution where { T1, T2, T3 }
		scale::T1
		max::T2
		norm::T3
	end
	
	function ModifiedJeffreysPriorForScale(s::T1, m::T2) where { T1, T2 }
		@assert zero(s) < s && !isinf(s)
		@assert zero(m) < m && !isinf(s)
		norm = 1/log1p(m/s)         # Ensure proper normalization
		ModifiedJeffreysPriorForScale{T1,T2,typeof(norm)}(s,m,norm)
	end
	
	function Distributions.rand(rng::AbstractRNG, d::ModifiedJeffreysPriorForScale{T1,T2,T3}) where {T1,T2,T3}
		u = rand(rng)               # sample in [0, 1]
		d.scale*(exp(u/d.norm)-1)   # inverse CDF method for sampling
	end

	function Distributions.logpdf(d::ModifiedJeffreysPriorForScale{T1,T2,T3}, x::Real) where {T1,T2,T3}
		log(d.norm/(1+x/d.scale))
	end
	
	function Distributions.logpdf(d::ModifiedJeffreysPriorForScale{T1,T2,T3}, x::AbstractVector{<:Real})  where {T1,T2,T3}
	    output = zeros(x)
		for (i,z) in enumerate(x)
			output[i] = logpdf(d,z)
		end
		return output
	end
	
	Distributions.minimum(d::ModifiedJeffreysPriorForScale{T1,T2,T3})  where {T1,T2,T3} = zero(T2)
	Distributions.maximum(d::ModifiedJeffreysPriorForScale{T1,T2,T3})  where {T1,T2,T3} = d.max
	
end

# ╔═╡ 83598a97-cf59-4ed9-8c6e-f72a87f4feb6
begin
	P_max = 10*365.25 # 100 years
	K_max = 2129.0     # m/s
	prior_P = ModifiedJeffreysPriorForScale(1.0, P_max)
	prior_K = ModifiedJeffreysPriorForScale(1.0, K_max)
	prior_e = Truncated(Rayleigh(0.3),0.0,0.999)
	prior_jitter = LogNormal(log(3.0),1.0)
end;

# ╔═╡ df503ec7-a9fa-4170-8892-d19e78c32d39
if  @isdefined chains_rand_init
	draw_new_samples_from_posterior_with_init_from_prior
	local plt = plot(legend=:none)
	scatter!(plt,df[!,:bjd].-bjd_ref,df[!,:rv],yerr=df[!,:σ_rv])
	for i in 1:n_draws
		sample_id = rand(floor(Int,size(chains_rand_init,1)//2) :
				 size(chains_rand_init,1))
		chain_id =  rand(1:size(chains_rand_init,3))
		rvs = simulate_rvs_from_model_v1(chains_rand_init,t_plt,
					sample=sample_id, 
					chain_id=chain_id)
		plot!(t_plt,rvs) 
	end
	xlabel!(plt,"Time (d)")
	ylabel!(plt,"RV (m/s)")
	title!(plt,"Predicted RV curves for $n_draws random samples from\nMarkov chains initialized with draws from the prior")
end

# ╔═╡ 6d2e7fd2-3cb8-4e10-ad7c-6d05eb974aa7
if  @isdefined chains_with_guess
	draw_new_samples_from_posterior_with_guess
	local plt = plot(legend=:none)
	scatter!(plt,df[!,:bjd].-bjd_ref,df[!,:rv],yerr=df[!,:σ_rv])
	for i in 1:n_draws
		sample_id = rand(floor(Int,size(chains_with_guess,1)//2) :
				 size(chains_with_guess,1))
		chain_id =  rand(1:size(chains_with_guess,3))
		rvs = simulate_rvs_from_model_v1(chains_with_guess,t_plt,
					sample=sample_id, 
					chain_id=chain_id)
		plot!(t_plt,rvs) 
	end
	xlabel!(plt,"Time (d)")
	ylabel!(plt,"RV (m/s)")
	title!(plt,"Predicted RV curves for $n_draws random samples from\nnew Markov chains")

end

# ╔═╡ 693cae36-613b-4c3d-b6a0-3284b1831520
if  @isdefined chains_posterior2
	draw_new_samples_from_posterior_with_acc
	local plt = plot(legend=:none)
	scatter!(plt,df[!,:bjd].-bjd_ref,df[!,:rv],yerr=df[!,:σ_rv])
	for i in 1:n_draws
		sample_id = rand(floor(Int,size(chains_posterior2,1)//2) :
				 size(chains_posterior2,1))
		local chain_id =  rand(1:size(chains_posterior2,3))
		rvs = simulate_rvs_from_model_v2(chains_posterior2,t_plt,
					sample=sample_id, 
					chain_id=chain_id)
		plot!(t_plt,rvs) 
	end
	xlabel!(plt,"Time (d)")
	ylabel!(plt,"RV (m/s)")
	title!(plt,"Predicted RV curves for $n_draws random samples from\nMarkov chains for model with acceleration term")

end

# ╔═╡ fe8f637d-3721-4a9f-9e6e-f6aee00b7f18
if  @isdefined chains_posterior2
	draw_new_samples_from_posterior_with_guess
	local plt = standardize_histo ? plot(Normal(0,1),legend=:none, color=:black, lw=3) : plot() 
	local resid = zeros(length(df.bjd),n_draws)
	for i in 1:n_draws
		sample_id = rand(floor(Int,size(chains_with_guess,1)//2) :
				 size(chains_with_guess,1))
		local chain_id =  rand(1:size(chains_with_guess,3))
		rvs_pred = simulate_rvs_from_model_v1(chains_with_guess,df.bjd.-bjd_ref,
					sample=sample_id, 
					chain_id=chain_id)
		resid[:,i] .= (df.rv.-rvs_pred)
		if standardize_histo
			resid[:,i] ./= sqrt.(df.σ_rv.^2 .+ chains_with_guess[sample_id,:σ_j,chain_id]^2)
		end
	end
	
	histogram!(vec(resid), bins=32, alpha=0.5, label="w/o acc", normalize=true)
	
	#resid = zeros(length(df.bjd),n_draws)
	for i in 1:n_draws
		sample_id = rand(floor(Int,size(chains_posterior2,1)//2) :
				 size(chains_posterior2,1))
		local chain_id =  rand(1:size(chains_posterior2,3))
		rvs_pred = simulate_rvs_from_model_v2(chains_posterior2,df.bjd.-bjd_ref,
					sample=sample_id, 
					chain_id=chain_id)
		resid[:,i] .= (df.rv.-rvs_pred)
		if standardize_histo
			resid[:,i] ./= sqrt.(df.σ_rv.^2 .+ chains_posterior2[sample_id,:σ_j,chain_id]^2)
		end
	end
	
	histogram!(vec(resid), bins=32, alpha=0.5, normalize=true, label="w/ acc")
	if standardize_histo
		title!(plt,"Histogram of standarized residuals")
		xlabel!(plt,"Standardized Residuals")
	else
		title!(plt,"Histogram of residuals")
		xlabel!(plt,"Residuals (m/s)")
		end
	ylabel!(plt,"Density")
end

# ╔═╡ 0bb0c056-cb48-4ed7-8305-24d2957b595a
md"""
## Compute a Periodorgam
"""

# ╔═╡ bfc03760-17bb-4dac-a233-47dd3827519c
md"### Compute design matrix for periodograms"

# ╔═╡ 8b662b76-79d3-41ff-b5e0-fd06163ad5f8
function calc_design_matrix_circ!(result::AM, period, times::AV) where { R1<:Real, AM<:AbstractMatrix{R1}, AV<:AbstractVector{R1} }
	n = length(times)
	@assert size(result) == (n, 2)
	for i in 1:n
		( result[i,1], result[i,2] ) = sincos(2π/period .* times[i])
	end
	return result
end

# ╔═╡ 61c7e285-6dd4-4811-8016-45c863fdb397
function calc_design_matrix_circ(period, times::AV) where { R1<:Real, AV<:AbstractVector{R1} }
	n = length(times)
	dm = zeros(n,2)
	calc_design_matrix_circ!(dm,period,times)
	return dm
end

# ╔═╡ e887b6ee-9f57-4629-ab31-c74d80cb948a
function calc_design_matrix_lowe!(result::AM, period, times::AV) where { R1<:Real, AM<:AbstractMatrix{R1}, AV<:AbstractVector{R1} }
	n = length(times)
	@assert size(result) == (n, 4)
	for i in 1:n
		arg = 2π/period .* times[i]
		( result[i,1], result[i,2] ) = sincos(arg)
		( result[i,3], result[i,4] ) = sincos(2*arg)
	end
	return result
end

# ╔═╡ 327391cf-864a-4c82-8aa9-d435fe44d0e1
function calc_design_matrix_lowe(period, times::AV) where { R1<:Real, AV<:AbstractVector{R1} }
	n = length(times)
	dm = zeros(n,4)
	calc_design_matrix_lowe!(dm,period,times)
	return dm
end

# ╔═╡ c9687fcb-a1e1-442c-a3c7-f0f60350b059
md"## Generalized Linear Least Squares Fitting" 

# ╔═╡ 6f820e58-b61e-43c0-95dc-6d0e936f71c3
function fit_general_linear_least_squares( design_mat::ADM, covar_mat::APD, obs::AA ) where { ADM<:AbstractMatrix, APD<:AbstractPDMat, AA<:AbstractArray }
	Xt_inv_covar_X = Xt_invA_X(covar_mat,design_mat)
	X_inv_covar_y =  design_mat' * (covar_mat \ obs)
	AB_hat = Xt_inv_covar_X \ X_inv_covar_y                            # standard GLS
end

# ╔═╡ b1bfd41e-3335-46ef-be5a-2aab2532060f
function predict_general_linear_least_squares( design_mat::ADM, covar_mat::APD, obs::AA ) where { ADM<:AbstractMatrix, APD<:AbstractPDMat, AA<:AbstractArray }
	param = fit_general_linear_least_squares(design_mat,covar_mat,obs)
	design_mat * param 
end

# ╔═╡ 1d3d4e92-e21d-43f8-b7f4-5191d8d42821
function calc_χ²_general_linear_least_squares( design_mat::ADM, covar_mat::APD, obs::AA ) where { ADM<:AbstractMatrix, APD<:AbstractPDMat, AA<:AbstractArray }
	pred = predict_general_linear_least_squares(design_mat,covar_mat,obs)
	invquad(covar_mat,obs-pred)
end

# ╔═╡ 42fd9719-26a3-4742-974a-303eb5e810c5
function calc_periodogram(t, y_obs, covar_mat; period_min::Real = 2.0, period_max::Real = 4*(maximum(t)-minimum(t)), num_periods::Integer = 4000)
	period_grid =  1.0 ./ range(1.0/period_max, stop=1.0/period_min, length=num_periods) 
	periodogram = map(p->-0.5*calc_χ²_general_linear_least_squares(calc_design_matrix_circ(p,t),covar_mat,y_obs),period_grid)
	period_fit = period_grid[argmax(periodogram)]
	design_matrix_fit = calc_design_matrix_circ(period_fit,t)
	coeff_fit = fit_general_linear_least_squares(design_matrix_fit,covar_mat,y_obs)
	phase_fit = atan(coeff_fit[1],coeff_fit[2])
	pred = design_matrix_fit * coeff_fit
	rms = sqrt(mean((y_obs.-pred).^2))
	return (;period_grid=period_grid, periodogram=periodogram, period_best_fit = period_fit, coeff_best_fit=coeff_fit, phase_best_fit=phase_fit, predict=pred, rms=rms )
end
	

# ╔═╡ 2ccca815-bd26-4f0f-b966-b2ab2fe02d01
begin
	jitter_for_periodogram = 3.0
	num_period_for_periodogram = 10_000
	periodogram_results = calc_periodogram(df.bjd.-bjd_ref,df.rv,
								PDiagMat(df.σ_rv.+jitter_for_periodogram^2), 
									num_periods=num_period_for_periodogram)
end

# ╔═╡ a5be518a-3c7c-424e-b100-ec8967f4ae27
plot(periodogram_results.period_grid,periodogram_results.periodogram, xscale=:log10, xlabel="Putative Orbital Period (d)", ylabel="-χ²/2", legend=:none)

# ╔═╡ cc51e4bd-896f-479c-8d09-0ce3f07e402c


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.2"
manifest_format = "2.0"
project_hash = "6f5fd0a15ab95fd3911efbca2a6bcdb69972a959"

[[deps.ADTypes]]
git-tree-sha1 = "60665b326b75db6517939d0e1875850bc4a54368"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.17.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "FillArrays", "LogDensityProblems", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers", "UUIDs"]
git-tree-sha1 = "e4b6a25ba2e033c74ea11720daacafbc2ab50a7e"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "5.7.2"

[[deps.AbstractPPL]]
deps = ["AbstractMCMC", "Accessors", "DensityInterface", "JSON", "Random", "StatsBase"]
git-tree-sha1 = "b7a856399119394a573141c553aeb5b674a500b5"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.13.0"
weakdeps = ["Distributions", "LinearAlgebra"]

    [deps.AbstractPPL.extensions]
    AbstractPPLDistributionsExt = ["Distributions", "LinearAlgebra"]

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "ProgressMeter", "Random", "Setfield", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "22496647c061d00217759e95a18d601c959df0c9"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.8.1"

    [deps.AdvancedHMC.extensions]
    AdvancedHMCADTypesExt = "ADTypes"
    AdvancedHMCCUDAExt = "CUDA"
    AdvancedHMCComponentArraysExt = "ComponentArrays"
    AdvancedHMCMCMCChainsExt = "MCMCChains"
    AdvancedHMCOrdinaryDiffEqExt = "OrdinaryDiffEq"

    [deps.AdvancedHMC.weakdeps]
    ADTypes = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "DocStringExtensions", "FillArrays", "LinearAlgebra", "LogDensityProblems", "Random", "Requires"]
git-tree-sha1 = "0205823d612410230d18c421ed6d9d851a5451b9"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.8.8"
weakdeps = ["DiffResults", "ForwardDiff", "MCMCChains", "StructArrays"]

    [deps.AdvancedMH.extensions]
    AdvancedMHForwardDiffExt = ["DiffResults", "ForwardDiff"]
    AdvancedMHMCMCChainsExt = "MCMCChains"
    AdvancedMHStructArraysExt = "StructArrays"

[[deps.AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Random", "Random123", "Requires", "SSMProblems", "StatsFuns"]
git-tree-sha1 = "5d34d826ece67ce790d4a7f3f97d837e52aba7f8"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.7.0"
weakdeps = ["Libtask"]

    [deps.AdvancedPS.extensions]
    AdvancedPSLibtaskExt = "Libtask"

[[deps.AdvancedVI]]
deps = ["ADTypes", "Accessors", "DiffResults", "DifferentiationInterface", "Distributions", "DocStringExtensions", "FillArrays", "Functors", "LinearAlgebra", "LogDensityProblems", "Optimisers", "ProgressMeter", "Random", "StatsBase"]
git-tree-sha1 = "59c9723a71ed815eafec430d4cafa592b5889b96"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.4.1"
weakdeps = ["Bijectors"]

    [deps.AdvancedVI.extensions]
    AdvancedVIBijectorsExt = "Bijectors"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "9606d7832795cbef89e06a550475be300364a8aa"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.19.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "26f41e1df02c330c4fa1e98d4aa2168fdafc9b1f"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.4"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bijectors]]
deps = ["ArgCheck", "ChainRulesCore", "ChangesOfVariables", "Distributions", "DocStringExtensions", "Functors", "InverseFunctions", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "Random", "Reexport", "Roots", "SparseArrays", "Statistics"]
git-tree-sha1 = "35ba30573d4f83242f1e788c87ca792c83553c9e"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.15.9"

    [deps.Bijectors.extensions]
    BijectorsDistributionsADExt = "DistributionsAD"
    BijectorsEnzymeCoreExt = "EnzymeCore"
    BijectorsForwardDiffExt = "ForwardDiff"
    BijectorsLazyArraysExt = "LazyArrays"
    BijectorsMooncakeExt = "Mooncake"
    BijectorsReverseDiffChainRulesExt = ["ChainRules", "ReverseDiff"]
    BijectorsReverseDiffExt = "ReverseDiff"
    BijectorsTrackerExt = "Tracker"

    [deps.Bijectors.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    DistributionsAD = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "224f9dc510986549c8139def08e06f78c562514d"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.72.5"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Chairmarks]]
deps = ["Printf", "Random"]
git-tree-sha1 = "9a49491e67e7a4d6f885c43d00bb101e6e5a434b"
uuid = "0ca39b1e-fe0b-4e98-acfc-b1656634c4de"
version = "1.3.1"
weakdeps = ["Statistics"]

    [deps.Chairmarks.extensions]
    StatisticsChairmarksExt = ["Statistics"]

[[deps.ChangesOfVariables]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "3aa4bf1532aa2e14e0374c4fd72bed9a9d0d0f6c"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.10"
weakdeps = ["InverseFunctions", "Test"]

    [deps.ChangesOfVariables.extensions]
    ChangesOfVariablesInverseFunctionsExt = "InverseFunctions"
    ChangesOfVariablesTestExt = "Test"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "a656525c8b46aa6a1c76891552ed5381bb32ae7b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.30.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Combinatorics]]
git-tree-sha1 = "8010b6bb3388abe68d95743dcbea77650bb2eddf"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.3"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "0037835448781bb46feb39866934e243886d756a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "16946a4d305607c3a4af54ff35d56f0e9444ed0e"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.7"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = "GPUArraysCore"
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3e6d038b77f22791b8e3472b7c633acea1ecac06"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.120"
weakdeps = ["ChainRulesCore", "DensityInterface", "Test"]

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "Distributions", "FillArrays", "LinearAlgebra", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "4acbf909e892ce1f94c39a138541566c1aad5e66"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.58"

    [deps.DistributionsAD.extensions]
    DistributionsADForwardDiffExt = "ForwardDiff"
    DistributionsADLazyArraysExt = "LazyArrays"
    DistributionsADReverseDiffExt = "ReverseDiff"
    DistributionsADTrackerExt = "Tracker"

    [deps.DistributionsAD.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DynamicPPL]]
deps = ["ADTypes", "AbstractMCMC", "AbstractPPL", "Accessors", "BangBang", "Bijectors", "Chairmarks", "Compat", "ConstructionBase", "DifferentiationInterface", "Distributions", "DocStringExtensions", "InteractiveUtils", "LinearAlgebra", "LogDensityProblems", "MacroTools", "OrderedCollections", "Printf", "Random", "Requires", "Statistics", "Test"]
git-tree-sha1 = "0cdb6cb40d00f38d3f2e4a73e16004402465dad7"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.37.1"

    [deps.DynamicPPL.extensions]
    DynamicPPLChainRulesCoreExt = ["ChainRulesCore"]
    DynamicPPLEnzymeCoreExt = ["EnzymeCore"]
    DynamicPPLForwardDiffExt = ["ForwardDiff"]
    DynamicPPLJETExt = ["JET"]
    DynamicPPLMCMCChainsExt = ["MCMCChains"]
    DynamicPPLMooncakeExt = ["Mooncake"]

    [deps.DynamicPPL.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    JET = "c3a54625-cd67-489e-a8e7-0a5a0ff4e31b"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterface", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "e611b7fdfbfb5b18d5e98776c30daede41b44542"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "2.0.0"

[[deps.EnumX]]
git-tree-sha1 = "bddad79635af6aec424f53ed8aad5d7555dc6f00"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.5"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7bb1361afdb33c7f2b085aa49ea8fe1b0fb14e58"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.1+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.ExproniconLite]]
git-tree-sha1 = "c13f0b150373771b0fdc1713c97860f8df12e6c2"
uuid = "55351af7-c7e9-48d6-89ff-24e801d99491"
version = "0.10.14"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "83dc665d0312b41367b7263e8a4d172eac1897f4"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.4"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3a948313e7a41eb1db7a1e733e6335f17b4ab3c4"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "7.1.1+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "797762812ed063b9b94f6cc7742bc8883bb5e69e"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.9.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "31fd32af86234b6b71add76229d53129aa1b87a9"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.28.1"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "910febccb28d493032495b7009dce7d7f7aee554"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.0.1"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "60a0339f28a233601cb74468032b5c302d5067de"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "1828eb7275491981fa5f1752a5e126e8f26f8741"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.17"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "27299071cc29e409488ada41ec7643e0ab19091f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.17+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "35fbd0cefb04a516104b8e183ce0df11b70a3f1a"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.84.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "ed5e9c58612c4e081aecdb6e1a479e18462e041e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "8594fac023c5ce1ef78260f24d1ad18b4327b420"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.4"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalSets]]
git-tree-sha1 = "5fbb102dcb8b1a858111ae81d56682376130517d"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.11"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.Jieko]]
deps = ["ExproniconLite"]
git-tree-sha1 = "2f05ed29618da60c06a87e9c033982d4f71d0b6c"
uuid = "ae98c720-c025-4a4a-838c-29b094483192"
version = "0.2.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e95866623950267c1e4878846f848d94810de475"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.2+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "ba51324b894edaf1df3ab16e2cc6bc3280a2f1a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.10"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LBFGSB]]
deps = ["L_BFGS_B_jll"]
git-tree-sha1 = "e2e6f53ee20605d0ea2be473480b7480bd5091b5"
uuid = "5be7bae1-8223-5378-bac3-9e7378a2f6e6"
version = "0.4.1"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.L_BFGS_B_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "77feda930ed3f04b2b0fbb5bea89e69d3677c6b0"
uuid = "81d17ec3-03a1-5e46-b53e-bddc35a13473"
version = "3.0.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "52e1296ebbde0db845b356abbbe67fb82a0a116c"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.9"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "95ba48564903b43b2462318aa243ee79d81135ff"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.2.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "706dfd3c0dd56ca090e86884db6eda70fa7dd4af"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.1+0"

[[deps.Libtask]]
deps = ["MistyClosures", "Test"]
git-tree-sha1 = "40574644c2baf96ec5d8dbb8c6d038a2e50b775c"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.9.3"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d3c8af829abaeba27181db4acb485b18d15d89c6"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.1+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "4adee99b7262ad2a1a4bbbc59d993d24e55ea96f"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.4.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random"]
git-tree-sha1 = "4e0128c1590d23a50dcdb106c7e2dbca99df85c0"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "2.1.2"

[[deps.LogDensityProblemsAD]]
deps = ["DocStringExtensions", "LogDensityProblems"]
git-tree-sha1 = "7b83f3ad0a8105f79a067cafbfd124827bb398d0"
uuid = "996a588d-648d-4e1f-a8f0-a84b347e47b1"
version = "1.13.1"

    [deps.LogDensityProblemsAD.extensions]
    LogDensityProblemsADADTypesExt = "ADTypes"
    LogDensityProblemsADDifferentiationInterfaceExt = ["ADTypes", "DifferentiationInterface"]
    LogDensityProblemsADEnzymeExt = "Enzyme"
    LogDensityProblemsADFiniteDifferencesExt = "FiniteDifferences"
    LogDensityProblemsADForwardDiffBenchmarkToolsExt = ["BenchmarkTools", "ForwardDiff"]
    LogDensityProblemsADForwardDiffExt = "ForwardDiff"
    LogDensityProblemsADReverseDiffExt = "ReverseDiff"
    LogDensityProblemsADTrackerExt = "Tracker"
    LogDensityProblemsADZygoteExt = "Zygote"

    [deps.LogDensityProblemsAD.weakdeps]
    ADTypes = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
    BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
    DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"
weakdeps = ["ChainRulesCore", "ChangesOfVariables", "InverseFunctions"]

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "DataAPI", "Dates", "Distributions", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "f3993c723865f670102011ef22811e2bbb0ef1a8"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "7.2.0"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "DataStructures", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "526c98cd41028da22c01cb8a203246799ad853a8"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.3.15"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MLJModelInterface]]
deps = ["InteractiveUtils", "REPL", "Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "ccaa3f7938890ee8042cc970ba275115428bd592"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.12.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.MistyClosures]]
git-tree-sha1 = "d1a692e293c2a0dc8fda79c04cad60582f3d4de3"
uuid = "dbe65cb8-6be2-42dd-bbc5-4196aaced4f4"
version = "2.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.Moshi]]
deps = ["ExproniconLite", "Jieko"]
git-tree-sha1 = "53f817d3e84537d84545e0ad749e483412dd6b2a"
uuid = "2e0e35c7-a2e4-4343-998d-7ef72827ed2d"
version = "0.3.7"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "816620e3aac93e5b5359e4fdaf23ca4525b00ddf"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.3"

[[deps.NLSolversBase]]
deps = ["ADTypes", "DifferentiationInterface", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "25a6638571a902ecfb1ae2a18fc1575f86b1d4df"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.10.0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "b0219babbb69e4f0b292a11ad7f33520bdfcea8f"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.10.4"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ca7e18198a166a1f3eb92a3650d53d94ed8ca8a1"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.22"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "f1a7e086c677df53e064e0fdd2c9d0b0833e3f6e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.5.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2ae7d4ddec2e13ad3bddf5c0796f7547cf682391"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.2+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["Compat", "EnumX", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "61942645c38dd2b5b78e2082c9b51ab315315d10"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.13.2"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "ConstructionBase", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "131dc319e7c58317e8c6d5170440f6bdaee0a959"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.4.6"

    [deps.Optimisers.extensions]
    OptimisersAdaptExt = ["Adapt"]
    OptimisersEnzymeCoreExt = "EnzymeCore"
    OptimisersReactantExt = "Reactant"

    [deps.Optimisers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

[[deps.Optimization]]
deps = ["ADTypes", "ArrayInterface", "ConsoleProgressMonitor", "DocStringExtensions", "LBFGSB", "LinearAlgebra", "Logging", "LoggingExtras", "OptimizationBase", "Printf", "ProgressLogging", "Reexport", "SciMLBase", "SparseArrays", "TerminalLoggers"]
git-tree-sha1 = "41902230755effe29a8599ea4b61dc3ffc2c952d"
uuid = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
version = "4.5.0"

[[deps.OptimizationBase]]
deps = ["ADTypes", "ArrayInterface", "DifferentiationInterface", "DocStringExtensions", "FastClosures", "LinearAlgebra", "PDMats", "Reexport", "Requires", "SciMLBase", "SparseArrays", "SparseConnectivityTracer", "SparseMatrixColorings"]
git-tree-sha1 = "474b2fa6de9288d34b8ad42c9c500088132621a7"
uuid = "bca83a33-5cc9-4baa-983d-23429ab6bcbb"
version = "2.10.0"

    [deps.OptimizationBase.extensions]
    OptimizationEnzymeExt = "Enzyme"
    OptimizationFiniteDiffExt = "FiniteDiff"
    OptimizationForwardDiffExt = "ForwardDiff"
    OptimizationMLDataDevicesExt = "MLDataDevices"
    OptimizationMLUtilsExt = "MLUtils"
    OptimizationMTKExt = "ModelingToolkit"
    OptimizationReverseDiffExt = "ReverseDiff"
    OptimizationSymbolicAnalysisExt = "SymbolicAnalysis"
    OptimizationZygoteExt = "Zygote"

    [deps.OptimizationBase.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    MLDataDevices = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    ModelingToolkit = "961ee093-0014-501f-94e3-6117800e7a78"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SymbolicAnalysis = "4297ee4d-0239-47d8-ba5d-195ecdf594fe"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.OptimizationOptimJL]]
deps = ["Optim", "Optimization", "PrecompileTools", "Reexport", "SparseArrays"]
git-tree-sha1 = "6f228118b81ce4e849091ee0d00805f2ecb18f54"
uuid = "36348300-93cb-4f02-beb5-3c3902f8871e"
version = "0.4.3"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c392fc5dd032381919e3b22dd32d6443760ce7ea"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.5.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f07c06228a1c670ae4c87d1276b92c7c597fdda0"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.35"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "275a9a6d85dc86c24d03d1837a0010226a96f540"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.3+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "0c5a5b7e440c008fe31416a3ac9e0d2057c81106"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.19"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "85778cdf2bed372008e6646c64340460764a5b85"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.5"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "8329a3a4f75e178c11c1ce2342778bcbbbfa7e3c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.71"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "PrecompileTools"]
git-tree-sha1 = "520070df581845686c8c488b6dadce6b2c316856"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.4.32"

    [deps.PreallocationTools.extensions]
    PreallocationToolsForwardDiffExt = "ForwardDiff"
    PreallocationToolsReverseDiffExt = "ReverseDiff"
    PreallocationToolsSparseConnectivityTracerExt = "SparseConnectivityTracer"

    [deps.PreallocationTools.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "d95ed0324b0799843ac6f7a6a85e65fe4e5173f0"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.5"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "13c5103482a8ed1536a54c08d0e742ae3dca2d42"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.4"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "eb38d376097f47316fe089fc62cb7c6d85383a52"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "e1d5e16d0f65762396f9ca4644a5f4ddab8d452b"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "dbe5fd0b334694e905cb9fda73cd8554333c46e2"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.1"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "96bef5b9ac123fff1b379acf0303cf914aaabdfd"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.37.1"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsKernelAbstractionsExt = "KernelAbstractions"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsStructArraysExt = "StructArrays"
    RecursiveArrayToolsTablesExt = ["Tables"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "8a433b1ede5e9be9a7ba5b1cc6698daa8d718f1d"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.2.10"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "86a8a8b783481e1ea6b9c91dd949cb32191f8ab4"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.15"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SSMProblems]]
deps = ["AbstractMCMC", "Distributions", "Random"]
git-tree-sha1 = "5dd0431563784b468db06335ce8777653092d621"
uuid = "26aad666-b158-4e64-9d35-0e672562fa48"
version = "0.5.2"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Moshi", "PreallocationTools", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "a06d451a6d0fa6e6da34d047d61af8beb187b0f1"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.112.0"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseDistributionsExt = "Distributions"
    SciMLBaseForwardDiffExt = "ForwardDiff"
    SciMLBaseMLStyleExt = "MLStyle"
    SciMLBaseMakieExt = "Makie"
    SciMLBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    SciMLBaseMooncakeExt = "Mooncake"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseReverseDiffExt = "ReverseDiff"
    SciMLBaseTrackerExt = "Tracker"
    SciMLBaseZygoteExt = ["Zygote", "ChainRulesCore"]

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    MLStyle = "d8e11817-5142-5d16-987a-aa16d5891078"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["Accessors", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "MacroTools"]
git-tree-sha1 = "78ac1b947205b07973321f67f17df8fbe6154ac9"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "1.6.0"
weakdeps = ["SparseArrays", "StaticArraysCore"]

    [deps.SciMLOperators.extensions]
    SciMLOperatorsSparseArraysExt = "SparseArrays"
    SciMLOperatorsStaticArraysCoreExt = "StaticArraysCore"

[[deps.SciMLStructures]]
deps = ["ArrayInterface"]
git-tree-sha1 = "566c4ed301ccb2a44cbd5a27da5f885e0ed1d5df"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.7.0"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SparseConnectivityTracer]]
deps = ["ADTypes", "DocStringExtensions", "FillArrays", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "7bd2b8981cc57adcf5cf1add282aba2713a7058f"
uuid = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
version = "1.0.0"

    [deps.SparseConnectivityTracer.extensions]
    SparseConnectivityTracerLogExpFunctionsExt = "LogExpFunctions"
    SparseConnectivityTracerNNlibExt = "NNlib"
    SparseConnectivityTracerNaNMathExt = "NaNMath"
    SparseConnectivityTracerSpecialFunctionsExt = "SpecialFunctions"

    [deps.SparseConnectivityTracer.weakdeps]
    LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
    NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SparseMatrixColorings]]
deps = ["ADTypes", "DocStringExtensions", "LinearAlgebra", "PrecompileTools", "Random", "SparseArrays"]
git-tree-sha1 = "9de43e0b9b976f1019bf7a879a686c4514520078"
uuid = "0a514795-09f3-496d-8182-132a7b665d35"
version = "0.4.21"

    [deps.SparseMatrixColorings.extensions]
    SparseMatrixColoringsCUDAExt = "CUDA"
    SparseMatrixColoringsCliqueTreesExt = "CliqueTrees"
    SparseMatrixColoringsColorsExt = "Colors"

    [deps.SparseMatrixColorings.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "cbea8a6bd7bed51b1619658dec70035e07b8502f"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.14"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "89f86d9376acd18a1a4fbef66a56335a3a7633b8"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.5.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2c962245732371acd51700dbb268af311bddd719"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.6"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "8e45cecc66f3b42633b8ce14d431e8e57a3e242e"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3b1dcbf62e469a67f6733ae493401e53d92ff543"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "8ad2e38cbb812e29348719cc63580ec1dfeb9de4"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.1"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "93104ca226670c0cb92ba8bc6998852ad55a2d4c"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.43"
weakdeps = ["PrettyTables"]

    [deps.SymbolicIndexingInterface.extensions]
    SymbolicIndexingInterfacePrettyTablesExt = "PrettyTables"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "f133fab380933d042f6796eda4e130272ba520ca"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.7"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

[[deps.Turing]]
deps = ["ADTypes", "AbstractMCMC", "AbstractPPL", "Accessors", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "Compat", "DataStructures", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "LogDensityProblems", "MCMCChains", "NamedArrays", "Optimization", "OptimizationOptimJL", "OrderedCollections", "Printf", "Random", "Reexport", "SciMLBase", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "33f6f6b0844b01bce45eba0311620ab2ddb7d37a"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.40.2"

    [deps.Turing.extensions]
    TuringDynamicHMCExt = "DynamicHMC"
    TuringOptimExt = "Optim"

    [deps.Turing.weakdeps]
    DynamicHMC = "bbc10e6e-7c05-544b-b16e-64fede858acb"
    Optim = "429524aa-4258-5aef-a3af-852621145aeb"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "6258d453843c466d84c17a58732dda5deeb8d3af"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.24.0"
weakdeps = ["ConstructionBase", "ForwardDiff", "InverseFunctions", "Printf"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    PrintfExt = "Printf"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "af305cc62419f9bd61b6644d19170a4d258c7967"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.7.0"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "e9aeb174f95385de31e70bd15fa066a505ea82b9"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.7"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "9caba99d38404b285db8801d5c45ef4f4f425a6d"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.1+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "c5bf2dad6a03dfef57ea0a170a1fe493601603f2"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.5+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "434b3de333c75fc446aa0d19fc394edafd07ab08"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.7"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4bba74fa59ab0755167ad24f98800fe5d727175b"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.12.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "07b6a107d926093898e82b3b1db657ebe33134ec"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.50+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d5a767a3bb77135a99e433afe0eb14cd7f6914c3"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "fbf139bce07a534df0e699dbb5f5cc9346f95cc1"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.9.2+0"
"""

# ╔═╡ Cell order:
# ╟─10709dbc-c38b-4f15-8ea8-772db2acfbb3
# ╟─7c12a4d6-1bf6-41bb-9528-9fe11ff8fb70
# ╟─73021015-4357-4196-868c-e9564de02ede
# ╟─26684357-eba5-4a90-b54e-9ad7e64b05a3
# ╟─a85268d5-40a6-4654-a4be-cba380e97d35
# ╟─cfe8d6ad-2125-4587-af70-875e7c4c4844
# ╟─a2140dbe-8736-4ed9-ae6f-b1b0c7df3bc9
# ╟─22719976-86f5-43d3-b890-d3520f9916d2
# ╠═e50bdd14-d855-4043-bbab-f6526a972e31
# ╟─87dd11ad-c29e-4a5e-90b1-289863eedd57
# ╟─ddac2663-fd90-4c60-acaa-da2560367706
# ╟─b9efe114-717e-46c8-8225-a4ab4d3df439
# ╠═2844cd3a-9ed1-47da-ab59-ea33575b4991
# ╟─f28d4bc8-53e7-45f9-8126-7339a6f54732
# ╟─56aa2fd5-fb34-495c-9b9f-0ce0bbbd6b1b
# ╠═37edd756-e889-491e-8710-a54a862a9cd8
# ╟─84c6b16b-ff8c-4a05-ac97-aa610f328370
# ╠═83598a97-cf59-4ed9-8c6e-f72a87f4feb6
# ╟─228bb255-319c-4e80-95b3-8bf333be29e4
# ╠═776a96af-2c4f-4d6d-9cec-b5db127fed6c
# ╟─5ab4a787-fe9b-4b2c-b14c-273e0205259d
# ╟─5f20e01d-3489-4828-9e4e-119095e9c29c
# ╟─dfdf843d-98ce-40a1-bd0b-0a11f1cdb5f9
# ╟─87a2281b-8827-4eca-89a6-c7d77aa7a00f
# ╟─9f27ff9f-c2c4-4bf4-b4cd-bac713de0abe
# ╠═3a838f95-283b-4e06-b54e-400c1ebe94f8
# ╟─358ab0f4-3423-4728-adfa-44126744ae18
# ╟─c1fa4faa-f10c-4ed0-9539-20f5fcde8763
# ╟─00c4e783-841c-4d6a-a53d-34a88bbe1f7a
# ╟─6fad4fd9-b99b-44a9-9c45-6e79ffd4a796
# ╟─f7046ee4-9eb7-4324-886e-5a3650b58db7
# ╟─bb2ff977-2d12-4baa-802c-097a4138a24b
# ╠═62c0c2ca-b43b-4c53-a297-08746fae3f6e
# ╟─1196e7e6-4144-47e8-97cd-6f93530e11e5
# ╟─55a3378a-da18-4310-8272-cf11298a118d
# ╟─1966aff9-ee3d-46f0-be0f-fed723b14f30
# ╟─764d36d4-5a3b-48f6-93d8-1e15a44c3ace
# ╠═71962ff1-b769-4601-8b50-7484ca3a0d91
# ╟─ac237181-b6d2-4968-b6e3-146ba41bcc18
# ╠═c81bb742-a911-4fba-9e85-dfb9a241b290
# ╟─a8e547c5-c2b5-4058-a1e2-35397d7e8112
# ╠═80d076a8-7fb1-4dc7-8e3b-7579db30b038
# ╟─3759ec08-14f4-4026-916b-6a0bbc818ca7
# ╟─2fa23729-007e-4dcb-b6f9-67407929c53a
# ╟─52f4f8bd-41a9-446f-87e7-cdc1e9c178f8
# ╟─bf5b40c5-399e-40a0-8c27-335f72015e58
# ╟─98bc7209-a6a1-4c06-8e6f-de5b421b7b8c
# ╟─207a0f69-30ce-4107-a21d-ba9f6c5a5039
# ╟─93115337-ec60-43b9-a0ef-1d3ffc1fe5ff
# ╟─df503ec7-a9fa-4170-8892-d19e78c32d39
# ╟─0d83ebc9-ecb3-4a9d-869b-351b28d5e962
# ╠═cc2dbf2f-27a4-4cd6-81e5-4a9f5a3bc488
# ╟─f00539c9-ee40-4128-928c-460fd439fa87
# ╟─b787d7ef-bda6-470d-93b8-954227dec24b
# ╟─aaf24ba9-48dd-49ff-b760-2b806f7cf5ba
# ╠═2ccca815-bd26-4f0f-b966-b2ab2fe02d01
# ╟─a5be518a-3c7c-424e-b100-ec8967f4ae27
# ╟─e62067ce-dbf6-4c50-94e3-fce0b8cbef12
# ╟─dd918d00-fbe7-4bba-ad6a-e77ea0acb43a
# ╟─9c82e944-d5f2-4d81-9b33-7e618421b715
# ╠═b76b3b7f-c5f3-4aec-99b5-799479c124d6
# ╟─3459436a-bfeb-45d4-950e-68fd55af76d7
# ╟─7ef2f1ca-d542-435b-abdd-03af4b4257f3
# ╟─47e61e3a-466d-40f9-892e-5691eb6a2393
# ╟─ba4e0dc6-5008-4ed8-8eba-3902820cf241
# ╟─a247788c-8072-4cc4-8c38-68d0a3445e83
# ╠═7a664634-c6e3-464b-90a0-6b4cf5015e83
# ╟─565d68a4-24cd-479a-82ab-21d64a6a01f6
# ╟─db2ba5d1-3bd3-4208-9805-9c3fab259377
# ╟─e9465e3f-f340-4cc5-9fa2-454feaa6bd4d
# ╟─3c3ecf3e-5eba-4bb1-92f7-449680be4edd
# ╟─b7766545-f1a9-4635-905a-fa3e798f12dc
# ╟─6441c9cf-f0b1-4229-8b72-1b89e9f0c6f3
# ╟─c9433de9-4f48-4bc6-846c-3d684ae6adee
# ╠═3b7580c9-e04e-4768-ac6f-ddb4462dedd8
# ╟─7f8394ee-e6a2-4e4f-84b2-10a043b3da35
# ╠═f2ff4d87-db0a-4418-a548-a9f3f04f93cd
# ╟─42fe9282-6a37-45f5-a833-d2f6eb0518fe
# ╟─b38f3767-cd14-497c-9576-22764c53a48d
# ╟─35562045-abba-4f36-be92-c41f71591b1a
# ╟─e9530014-1186-4897-b875-d9980e0c3ace
# ╟─9dc13414-8266-4f4d-94dd-f11c0726c539
# ╟─6d2e7fd2-3cb8-4e10-ad7c-6d05eb974aa7
# ╟─95844ef7-8730-469f-b69b-d7bbe5fc2607
# ╠═3699772e-27d4-402e-a669-00f5b22f2ed5
# ╟─9eccb683-d637-4ab9-8af7-70c24a7d8478
# ╟─9a2952be-89d8-41ef-85ce-310ed90bd0d1
# ╟─c7e9649d-496d-44c2-874c-5f51e311b21d
# ╠═3cfc82d6-3390-4572-b5f0-124503e2e9e0
# ╟─5084b413-1211-490a-89d3-1cc782f1741e
# ╟─c6939875-0efc-4f1d-a2d3-6b46484328a5
# ╠═fb2fe33c-3854-435e-b9e5-60e8531fd1f3
# ╟─43177135-f859-423d-b70c-e755fdd06765
# ╠═c33d9e9d-bf00-4fa4-9f90-33415385507e
# ╟─20393a82-6fb1-4583-b7b7-8d1cda43bd47
# ╠═8ffb3e78-515b-492d-bd6f-b24eb08e93d6
# ╟─13c56edd-3c1e-496e-8116-fb158dd0f133
# ╟─c2b7dfcc-6238-4e9a-a2b8-efcc05c17361
# ╟─cd1ca1b3-a0ec-4d10-90a3-7648fe52f206
# ╟─9d57a595-fad9-4263-baf4-33d4ac5209f7
# ╟─4c05803d-3b8d-4c03-9c7a-3c589227a807
# ╟─8a4ae52b-9cc6-478f-b836-62e59694949e
# ╟─21fdeaff-0c91-481c-bd8e-1dba27e275a6
# ╠═a4be55ab-3e8e-423c-b3bc-3f3b88c5d2b7
# ╟─5efb1eac-c1fe-417f-828a-3cfb8978da40
# ╟─bcedc85f-8bf2-49d4-a60a-d6020450fd76
# ╟─d751f93c-9ef4-41ad-967b-7ccde6e40afd
# ╟─693cae36-613b-4c3d-b6a0-3284b1831520
# ╟─5171f2f0-e60c-4038-829f-9baf2d5f178e
# ╟─fe8f637d-3721-4a9f-9e6e-f6aee00b7f18
# ╟─083e9418-9b64-46d0-8da4-3396fb958862
# ╟─8180b43d-81aa-4be0-bdf1-ac93f734331c
# ╟─d9cd7102-490d-4f35-a254-816c069d3810
# ╟─9987e752-164f-40df-98ed-073d715ad87b
# ╟─2fb25751-a036-4156-9fbd-3aaf4e373b91
# ╟─1dfb8d58-b9f3-47a3-a7b1-e8354e7db4e2
# ╟─0f1bf89e-c195-4c5f-9cd9-a2982b2e7bf0
# ╟─9fd50ada-702f-4ca4-aab2-abfa0f4f597c
# ╠═b3f9c7b7-5ed5-47d7-811c-6f4a313de24b
# ╟─461416e4-690f-4ebd-9f07-3e34962c8693
# ╟─e5d0b1cc-ea7b-42a4-bfcd-684337b0f98b
# ╠═5e5d4560-fa1e-48f6-abe4-3b1221d44609
# ╟─dd40a3cf-76d3-4eb9-8027-274a065c762c
# ╟─940f4f42-7bc3-48d4-b9f4-22f9b94c345d
# ╟─40923752-9215-45c9-a444-5a519b64df97
# ╠═dfe71f99-5977-46e5-957d-10a32ce8340e
# ╠═6fd32120-4df1-4f2d-bb6f-c348a6999ad5
# ╠═16802453-5fdf-4e1f-8e99-51d1fcaa248a
# ╟─83485386-90fd-4b2d-bdad-070835d8fb44
# ╠═b01b2df0-9cfd-45c8-ba35-cd9ec018af6a
# ╠═f104183b-ea56-45c3-987e-94e42d687143
# ╠═9c50e9eb-39a0-441a-b03f-6358caa2d0e9
# ╟─b9267ff2-d401-4263-bf25-d52be6260859
# ╠═1492e4af-440d-4926-a4b5-b33da77dbee2
# ╠═0ad398fb-9c7e-467d-a932-75db70cd2e86
# ╠═d30799d5-6c82-4987-923e-b8beb2aac74a
# ╠═7330040e-1988-4410-b625-74f71f031d43
# ╠═bdac967d-82e0-4d87-82f7-c771896e1797
# ╠═93adb0c3-5c11-479b-9436-8c7df34bd8fe
# ╠═f3fb26a5-46b5-4ba3-8a30-9a88d6868a24
# ╟─3fd2ec1a-fed7-43a6-bc0b-ccadb1f711dd
# ╠═f1547a42-ee3b-44dc-9147-d9c8ec56f1e3
# ╟─0bb0c056-cb48-4ed7-8305-24d2957b595a
# ╠═42fd9719-26a3-4742-974a-303eb5e810c5
# ╟─bfc03760-17bb-4dac-a233-47dd3827519c
# ╠═8b662b76-79d3-41ff-b5e0-fd06163ad5f8
# ╠═61c7e285-6dd4-4811-8016-45c863fdb397
# ╠═e887b6ee-9f57-4629-ab31-c74d80cb948a
# ╠═327391cf-864a-4c82-8aa9-d435fe44d0e1
# ╟─c9687fcb-a1e1-442c-a3c7-f0f60350b059
# ╠═6f820e58-b61e-43c0-95dc-6d0e936f71c3
# ╠═b1bfd41e-3335-46ef-be5a-2aab2532060f
# ╠═1d3d4e92-e21d-43f8-b7f4-5191d8d42821
# ╠═cc51e4bd-896f-479c-8d09-0ce3f07e402c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
