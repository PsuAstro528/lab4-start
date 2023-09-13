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

# ╔═╡ 5084b413-1211-490a-89d3-1cc782f1741e
md"## Sampling from the new model"

# ╔═╡ c6939875-0efc-4f1d-a2d3-6b46484328a5
md"""
Since we have a new statistical model, we'll need to define a new posterior based on the new statistical model and our dataset.
"""

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

# ╔═╡ 776a96af-2c4f-4d6d-9cec-b5db127fed6c
posterior_1 = rv_kepler_model_v1(df.bjd.-bjd_ref,df.rv,df.σ_rv)

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

# ╔═╡ fb2fe33c-3854-435e-b9e5-60e8531fd1f3
posterior_2 = rv_kepler_model_v2(df.bjd.-bjd_ref,df.rv,df.σ_rv)

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

julia_version = "1.9.2"
manifest_format = "2.0"
project_hash = "6f5fd0a15ab95fd3911efbca2a6bcdb69972a959"

[[deps.ADTypes]]
git-tree-sha1 = "f2b16fe1a3491b295105cae080c2a5f77a842718"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "0.2.3"

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
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "LogDensityProblems", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "87e63dcb990029346b091b170252f3c416568afc"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "4.4.2"

[[deps.AbstractPPL]]
deps = ["AbstractMCMC", "DensityInterface", "Random", "Setfield", "SparseArrays"]
git-tree-sha1 = "caa9b62583577b0d6b222f11f54aa29fabbdb5ca"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.6.2"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "ProgressMeter", "Random", "Requires", "Setfield", "SimpleUnPack", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "acbe805c3078ba0057bb56985248bd66bce016b1"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.5.5"

    [deps.AdvancedHMC.extensions]
    AdvancedHMCCUDAExt = "CUDA"
    AdvancedHMCMCMCChainsExt = "MCMCChains"
    AdvancedHMCOrdinaryDiffEqExt = "OrdinaryDiffEq"

    [deps.AdvancedHMC.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "FillArrays", "LinearAlgebra", "LogDensityProblems", "Random", "Requires"]
git-tree-sha1 = "b2a1602952739e589cf5e2daff1274a49f22c9a4"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.7.5"
weakdeps = ["DiffResults", "ForwardDiff", "MCMCChains", "StructArrays"]

    [deps.AdvancedMH.extensions]
    AdvancedMHForwardDiffExt = ["DiffResults", "ForwardDiff"]
    AdvancedMHMCMCChainsExt = "MCMCChains"
    AdvancedMHStructArraysExt = "StructArrays"

[[deps.AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Libtask", "Random", "Random123", "StatsFuns"]
git-tree-sha1 = "4d73400b3583147b1b639794696c78202a226584"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.4.3"

[[deps.AdvancedVI]]
deps = ["Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "1f919a9c59cf3dfc68b64c22c453a2e356fca473"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.2.4"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

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
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f83ec24f76d4c8f525099b2ac475fc098138ec31"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.4.11"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "e28912ce94077686443433c2800104b061a827ed"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.39"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bijectors]]
deps = ["ArgCheck", "ChainRules", "ChainRulesCore", "ChangesOfVariables", "Compat", "Distributions", "Functors", "InverseFunctions", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "Random", "Reexport", "Requires", "Roots", "SparseArrays", "Statistics"]
git-tree-sha1 = "8eacff457e5b8c13a97848484ad650dabbffa0fc"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.13.7"

    [deps.Bijectors.extensions]
    BijectorsDistributionsADExt = "DistributionsAD"
    BijectorsForwardDiffExt = "ForwardDiff"
    BijectorsLazyArraysExt = "LazyArrays"
    BijectorsReverseDiffExt = "ReverseDiff"
    BijectorsTrackerExt = "Tracker"
    BijectorsZygoteExt = "Zygote"

    [deps.Bijectors.weakdeps]
    DistributionsAD = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "44dbf560808d49041989b8a96cae4cffbeb7966a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.11"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "dbeca245b0680f5393b4e6c40dcead7230ab0b3b"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.54.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "2fba81a302a7be671aefe194f0525ef231104e7f"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.8"
weakdeps = ["InverseFunctions"]

    [deps.ChangesOfVariables.extensions]
    ChangesOfVariablesInverseFunctionsExt = "InverseFunctions"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "b86ac2c5543660d238957dbde5ac04520ae977a7"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.4"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "a1296f0fe01a4c3f9bf0dc2934efbf4416f5db31"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "02aa26a4cf76381be7f66e020a3eddeb27b0a092"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "d9a8f86737b665e15a9641ecbac64deef9ce6724"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.23.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "e460f044ca8b99be31d35fe54fc33a5c33dd8ed7"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.9.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "5372dbbf8f0bdb8c700db5367132925c0771ef7e"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.1"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

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

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "b6def76ffad15143924a2199f72a5cd883a2e8a9"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.9"
weakdeps = ["SparseArrays"]

    [deps.Distances.extensions]
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "938fe2981db009f531b6332e31c58e9584a2f9bd"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.100"
weakdeps = ["ChainRulesCore", "DensityInterface"]

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "Distributions", "FillArrays", "LinearAlgebra", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "975de103eb2175cf54bf14b15ded2c68625eabdf"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.52"

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
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicPPL]]
deps = ["AbstractMCMC", "AbstractPPL", "BangBang", "Bijectors", "ChainRulesCore", "ConstructionBase", "Distributions", "DocStringExtensions", "LinearAlgebra", "LogDensityProblems", "MacroTools", "OrderedCollections", "Random", "Requires", "Setfield", "Test", "ZygoteRules"]
git-tree-sha1 = "c60859c9bb51353a6f4a9769a0fb771cc4de94bc"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.23.17"
weakdeps = ["MCMCChains"]

    [deps.DynamicPPL.extensions]
    DynamicPPLMCMCChainsExt = ["MCMCChains"]

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterface", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "973b4927d112559dc737f55d6bf06503a5b3fc14"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "1.1.0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "b4fbdd20c889804969571cc589900803edda16b7"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "a20eaa3ad64254c61eeb5f230d9306e937405434"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.6.1"
weakdeps = ["SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

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
deps = ["LinearAlgebra"]
git-tree-sha1 = "9a68d75d466ccc1218d0552a8e1631151c569545"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.5"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "d73afa4a2bb9de56077242d98cf763074ab9a970"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.9"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1596bab77f4f073a14c62424283e7ebff3072eca"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.9+1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "19e974eced1768fb46fd6020171f2cec06b1edb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.15"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ad37c091f7d7daf900963171600d7c1c5c3ede32"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalSets]]
deps = ["Dates", "Random"]
git-tree-sha1 = "8e59ea773deee525c99a8018409f64f19fb719e6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.7"
weakdeps = ["Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "68772f49f54b479fa88ace904f6127f0a3bb2e46"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.12"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "81dc6aefcbe7421bd62cb6ca0e700779330acff8"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.25"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "Requires", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "4c5875e4c228247e1c2b087669846941fb6e0118"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.8"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "90442c50e202a5cdf21a7899c66b240fdef14035"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.7"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "a9d2ce1d5007b1e8f6c5b89c5a31ff8bd146db5c"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.2.1"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "7ca6850ae880cc99b59b88517545f91a52020afa"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.25+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LRUCache]]
git-tree-sha1 = "48c10e3cc27e30de82463c27bef0b8bdbd1dc634"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.4.1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "fb6803dafae4a5d62ea5cab204b1e657d9737e7f"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.2.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtask]]
deps = ["FunctionWrappers", "LRUCache", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "345a40c746404dd9cb1bbc368715856838ab96f2"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.8.6"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random"]
git-tree-sha1 = "f9a11237204bc137617194d79d813069838fcf61"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "2.1.1"

[[deps.LogDensityProblemsAD]]
deps = ["DocStringExtensions", "LogDensityProblems", "Requires", "SimpleUnPack"]
git-tree-sha1 = "a0512ad65f849536b5a52e59b05c59c25cdad943"
uuid = "996a588d-648d-4e1f-a8f0-a84b347e47b1"
version = "1.5.0"

    [deps.LogDensityProblemsAD.extensions]
    LogDensityProblemsADEnzymeExt = "Enzyme"
    LogDensityProblemsADFiniteDifferencesExt = "FiniteDifferences"
    LogDensityProblemsADForwardDiffBenchmarkToolsExt = ["BenchmarkTools", "ForwardDiff"]
    LogDensityProblemsADForwardDiffExt = "ForwardDiff"
    LogDensityProblemsADReverseDiffExt = "ReverseDiff"
    LogDensityProblemsADTrackerExt = "Tracker"
    LogDensityProblemsADZygoteExt = "Zygote"

    [deps.LogDensityProblemsAD.weakdeps]
    BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"
weakdeps = ["ChainRulesCore", "ChangesOfVariables", "InverseFunctions"]

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "0d097476b6c381ab7906460ef1ef1638fbce1d91"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.2"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "8778ea7283a0bf0d7e507a0235adfff38071493b"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "6.0.3"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "DataStructures", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "3e6db72c2ab9cadfa3278ff388473a01fc0cfb9d"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.3.5"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "eb006abbd7041c28e0d16260e50a24f8f9104913"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2023.2.0+0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "03ae109be87f460fe3c96b8a0dbbf9c7bf840bd5"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.9.2"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "3b29fafcdfa66d6673306cf116a2dc243933e2c5"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.5"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "6d42eca6c3a27dc79172d6d947ead136d88751bb"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.10.0"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2ac17d29c523ce1cd38e27785a7d23024853a4bb"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.10"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "bbb5c2115d63c2f1451cb70e5ef75e8fe4707019"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.22+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "c1fc26bab5df929a5172f296f25d7d08688fd25b"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.20"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

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

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "542de5acb35585afcf202a6d3361b430bc1c3fbd"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.13"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "ee094908d720185ddbdc58dbe0c1cbe35453ec7a"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "ae36206463b2395804f2787ffe172f44452b538d"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.8.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "364898e8f13f7eaaceec55fd3d08680498c0aa6e"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.4.2+3"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "552f30e847641591ba3f39fd1bed559b9deb0ef3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.1"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

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
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "Requires", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "5f834446731ba29d1d68d91fddf8f9593b68b7a2"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.38.8"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "1e597b93700fa4045d7189afa7c004e0584ea548"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.3"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.Roots]]
deps = ["ChainRulesCore", "CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "ff42754a57bb0d6dcfe302fd0d4272853190421f"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.19"

    [deps.Roots.extensions]
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"

    [deps.Roots.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "6aacc5eefe8415f47b3e34214c1d79d2674a0ba2"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.12"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ADTypes", "ArrayInterface", "ChainRulesCore", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "PrecompileTools", "Preferences", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables", "TruncatedStacktraces", "ZygoteRules"]
git-tree-sha1 = "c0781c7ebb65776e9770d333b5e191d20dd45fcf"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.97.0"

    [deps.SciMLBase.extensions]
    ZygoteExt = "Zygote"

    [deps.SciMLBase.weakdeps]
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["ArrayInterface", "DocStringExtensions", "Lazy", "LinearAlgebra", "Setfield", "SparseArrays", "StaticArraysCore", "Tricks"]
git-tree-sha1 = "65c2e6ced6f62ea796af251eb292a0e131a3613b"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.6"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "04bdff0b09c65ff3e06a05e3eb7b120223da3d39"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleUnPack]]
git-tree-sha1 = "58e6353e72cde29b90a69527e56df1b5c3d8c437"
uuid = "ce78b400-467f-4804-87d8-8f486da07d0a"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "91402087fd5d13b2d97e3ef29bbdf9d7859e678a"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.1"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "51621cca8651d9e334a659443a74ce50a3b6dfab"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.3"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "30b9236691858e13f167ce829490a68e1a597782"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "75ebe04c5bed70b91614d684259b661c9e6274a4"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.0"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "9115a29e6c2cf66cf213ccc17ffd61e27e743b24"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.6"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.StructArrays]]
deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "0a3db38e4cce3c54fe7a71f831cd7b6194a54213"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.16"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.SymbolicIndexingInterface]]
deps = ["DocStringExtensions"]
git-tree-sha1 = "f8ab052bfcbdb9b48fad2c80c873aa0d0344dfe5"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.2.2"

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
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

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

[[deps.Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "Functors", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NNlib", "NaNMath", "Optimisers", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "92364c27aa35c0ee36e6e010b704adaade6c409c"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.26"
weakdeps = ["PDMats"]

    [deps.Tracker.extensions]
    TrackerPDMatsExt = "PDMats"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "53bd5978b182fa7c57577bdb452c35e5b4fb73a5"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.78"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.Turing]]
deps = ["AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "DataStructures", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "LogDensityProblems", "LogDensityProblemsAD", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SciMLBase", "Setfield", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "83bcdf908c241f1ab54abc95af182f7b6550d591"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.29.0"

    [deps.Turing.extensions]
    TuringDynamicHMCExt = "DynamicHMC"
    TuringOptimExt = "Optim"

    [deps.Turing.weakdeps]
    DynamicHMC = "bbc10e6e-7c05-544b-b16e-64fede858acb"
    Optim = "429524aa-4258-5aef-a3af-852621145aeb"

[[deps.URIs]]
git-tree-sha1 = "b7a5e99f24892b6824a954199a45e9ffcc1c70f0"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "a72d22c7e13fe2de562feda8645aa134712a87ee"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.17.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "323e3d0acf5e78a56dfae7bd8928c989b4f3083e"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "04a51d15436a572301b5abbb9d099713327e9fc4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.4+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cf2c7de82431ca6f39250d2fc4aacd0daa1675c0"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "977aed5d006b840e2e40c0b48984f7463109046d"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.3"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
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
