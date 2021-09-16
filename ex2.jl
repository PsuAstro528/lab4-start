### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
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
	df = CSV.read(joinpath(data_path,"16cygb.txt"),DataFrame,header=[:Target,:bjd,:rv,:σ_rv,:col5,:col6,:col7,:col8],datarow=100,delim=' ',ignorerepeated=true)
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

# ╔═╡ 9f27ff9f-c2c4-4bf4-b4cd-bac713de0abe
md"Number of Markov chains  $(@bind num_chains NumberField(1:10; default=5)) "

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
param_guess_with_acc = merge(param_guess, (;a = 0.0) )

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
	@assert -2π<= M <= 2π
	@assert 0 <= ecc <= 1.0
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
  #es = ecc*sin(E)
  #ec = ecc*cos(E)
  (es, ec) = ecc .* sincos(E)  # Does combining them provide any speed benefit?
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
	#calc_rv(t, p::Vector) = calc_rv(t, p[1],p[2],p[3],p[4],p[5])
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

# ╔═╡ 7a664634-c6e3-464b-90a0-6b4cf5015e83
if go_sample_posterior1 && (P_guess > 0)
	chains_with_guess = sample(posterior_1, NUTS(), MCMCThreads(), num_steps_per_chain*2, num_chains; init_params = param_guess)
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
	chains_posterior2 = sample(posterior_2, NUTS(), MCMCThreads(), num_steps_per_chain, num_chains; init_params = param_guess_with_acc)
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

# ╔═╡ 693cae36-613b-4c3d-b6a0-3284b1831520
if  @isdefined chains_posterior2
	draw_new_samples_from_posterior_with_acc
	local plt = plot(legend=:none)
	scatter!(plt,df[!,:bjd].-bjd_ref,df[!,:rv],yerr=df[!,:σ_rv])
	for i in 1:n_draws
		sample_id = rand(floor(Int,size(chains_posterior2,1)//2) :
				 size(chains_posterior2,1))
		chain_id =  rand(1:size(chains_posterior2,3))
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
		chain_id =  rand(1:size(chains_with_guess,3))
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
		chain_id =  rand(1:size(chains_posterior2,3))
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

[compat]
CSV = "~0.8.5"
DataFrames = "~1.2.2"
Distributions = "~0.25.14"
LaTeXStrings = "~1.2.1"
MCMCChains = "~5.0.0"
PDMats = "~0.11.1"
Plots = "~1.21.1"
PlutoTeachingTools = "~0.1.4"
PlutoUI = "~0.7.9"
StatsPlots = "~0.14.26"
Turing = "~0.18.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "db0a7ff3fbd987055c43b4e12d2fa30aaae8749c"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "3.2.1"

[[AbstractPPL]]
deps = ["AbstractMCMC"]
git-tree-sha1 = "15f34cc635546ac072d03fc2cc10083adb4df680"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.2.0"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "Setfield", "Statistics", "StatsBase", "StatsFuns", "UnPack"]
git-tree-sha1 = "c71d9da0b0e5183a3410066e6b85670b0ae2bdbe"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.3.1"

[[AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "Random", "Requires"]
git-tree-sha1 = "6fcaabc5def4dcb20218a12c73a261090182b0c1"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.6.3"

[[AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Libtask", "Random", "StatsFuns"]
git-tree-sha1 = "06da6c283cf17cf0f97ed2c07c29b6333ee83dc9"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.2.4"

[[AdvancedVI]]
deps = ["Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "130d6b17a3a9d420d9a6b37412cae03ffd6a64ff"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.1.3"

[[ArgCheck]]
git-tree-sha1 = "dedbbb2ddb876f899585c4ec4433265e3017215a"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.1.0"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[ArrayInterface]]
deps = ["IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "85d03b60274807181bae7549bb22b2204b6e5a0e"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.30"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "d127d5e4d86c7680b20c35d40b503c74b9a39b5e"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.4"

[[BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "e239020994123f08905052b9603b4ca14f8c5807"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.31"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[Bijectors]]
deps = ["ArgCheck", "ChainRulesCore", "Compat", "Distributions", "Functors", "LinearAlgebra", "MappedArrays", "NNlib", "NonlinearSolve", "Random", "Reexport", "Requires", "SparseArrays", "Statistics", "StatsFuns"]
git-tree-sha1 = "f032f0b27318b0ea5e35fc510759971fbba65179"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.9.7"

[[BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "652aab0fc0d6d4db4cc726425cadf700e9f473f1"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.0"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c3598e525718abcc440f69cc6d5f60dda0a1b61e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.6+5"

[[CPUSummary]]
deps = ["Hwloc", "IfElse", "Static"]
git-tree-sha1 = "ed720e2622820bf584d4ad90e6fcb93d95170b44"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.1.3"

[[CSV]]
deps = ["Dates", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode"]
git-tree-sha1 = "b83aa3f513be680454437a0eee21001607e5d983"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.8.5"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "e2f47f6d8337369411569fd45ae5753ca10394c6"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.0+6"

[[ChainRules]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "0902fc7f416c8f1e3b1e014786bb65d0c2241a9b"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "0.8.24"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f53ca8d41e4753c41cdafa6ec5f7ce914b34be54"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "0.10.13"

[[CloseOpenIntervals]]
deps = ["ArrayInterface", "Static"]
git-tree-sha1 = "ce9c0d07ed6e1a4fecd2df6ace144cbd29ba6f37"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.2"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "9995eb3977fbf67b86d0a0a0508e83017ded03f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.14.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "727e463cfebd0c7b999bbf3e9e7e16f254b94193"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.34.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DefineSingletons]]
git-tree-sha1 = "77b4ca280084423b728662fe040e5ff8819347c5"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.1"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "3ed8fa7178a10d1cd0f1ca524f249ba6937490c0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.3.0"

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "abe4ad222b26af3337262b8afb28fab8d215e9f8"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.3"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "f389cb8974e02d7eaa6ae2ccedbbfb43174cd8e8"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.14"

[[DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "DiffRules", "Distributions", "FillArrays", "LinearAlgebra", "NaNMath", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "f773f784beca655b28ec1b235dbb9f5a6e5e151f"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.29"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[DynamicPPL]]
deps = ["AbstractMCMC", "AbstractPPL", "BangBang", "Bijectors", "ChainRulesCore", "Distributions", "MacroTools", "Random", "ZygoteRules"]
git-tree-sha1 = "1f8047e21fb29df859a7ce38e64a35ec6aeb0211"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.15.0"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "92d8f9f208637e8d2d28c664051a00569c01493d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.1.5+1"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "8041575f021cba5a099a456b4163c9a08b566a02"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.0"

[[EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterface", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "254182080498cce7ae4bc863d23bf27c632688f7"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "0.4.4"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "LibVPX_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3cc57ad0a213808473eafef4845a74766242e05f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.3.1+4"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "f985af3b9f4e278b1d24434cbb546d6092fca661"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.3"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3676abafff7e4ff07bbd2c42b3d8201f31653dcc"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.9+8"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "693210145367e7685d8604aee33d9bfb85db8b31"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.11.9"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "8b3c09b56acaf3c0e581c66638b85c8650ee9dca"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.8.1"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "35895cf184ceaab11fd778b4590144034a167a2f"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.1+14"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "b5e930ac60b613ef3406da6d4f42c35d8dc51419"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.19"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "cbd58c9deb1d304f5a245a0b7eb841a2560cfec6"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.1+5"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Functors]]
git-tree-sha1 = "39007773fd6097164ab537f78d3ac78ad2b8b695"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.4"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "182da592436e287758ded5be6e32c406de3a2e47"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.58.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "d59e8320c2747553788e4fc42231489cc602fa50"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.58.1+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "44e3b40da000eab4ccb1aecdc4801c040026aeb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.13"

[[HostCPUFeatures]]
deps = ["IfElse", "Libdl", "Static"]
git-tree-sha1 = "e86382a874edd4ff47fd1373e03f38302af93345"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.2"

[[Hwloc]]
deps = ["Hwloc_jll"]
git-tree-sha1 = "92d99146066c5c6888d5a3abc871e6a214388b91"
uuid = "0e44f5e4-bd66-52a0-8798-143a42290a1d"
version = "2.0.0"

[[Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3395d4d4aeb3c9d31f5929d32760d8baeee88aaf"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.5.0+0"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InitialValues]]
git-tree-sha1 = "26c8832afd63ac558b98a823265856670d898b6c"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.2.10"

[[InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[InvertedIndices]]
deps = ["Test"]
git-tree-sha1 = "15732c475062348b0165684ffe28e85ea8396afc"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.0.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1a8c6237e78b714e901e406c096fc8a65528af7d"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.1"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static"]
git-tree-sha1 = "560d7dcaf8cf8e5b13f73d90b4c90288f8ad7d14"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.2"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "71be1eb5ad19cb4f61fa8c73395c0338fd092ae0"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.2"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[LibVPX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "12ee7e23fa4d18361e7c2cde8f8337d4c3101bc7"
uuid = "dd192d2f-8180-539f-9fb4-cc70b1dcf69a"
version = "1.10.0+0"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtask]]
deps = ["Libtask_jll", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "90c6ed7f9ac449cddacd80d5c1fca59c97d203e7"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.5.3"

[[Libtask_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "901fc8752bbc527a6006a951716d661baa9d54e9"
uuid = "3ae2931a-708c-5973-9c38-ccf7496fb450"
version = "0.4.3+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "3d682c07e6dd250ed082f883dc88aee7996bf2cc"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.0"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "dfeda1c1130990428720de0024d4516b1902ce98"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.7"

[[LoopVectorization]]
deps = ["ArrayInterface", "DocStringExtensions", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "Polyester", "Requires", "SLEEFPirates", "Static", "StrideArraysCore", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "e04edb516a3314209af22c26954d4285152e185d"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.69"

[[MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Compat", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "PrettyTables", "Random", "RecipesBase", "Serialization", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "49923cab8e63eedf5fbcbe85fde33e878da5792a"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "5.0.0"

[[MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "f6ab1e254e51ddebf9bcb45301625224291ca8e0"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.1.0"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "c253236b0ed414624b083e6b72bfe891fbd2c7af"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+1"

[[MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "0c2bcd5c5b99988bb88552a4408beb3da1f1fd4d"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.2.0"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "0fb723cd8c45858c22169b2e42269e53271a6df7"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.7"

[[ManualMemory]]
git-tree-sha1 = "9cb207b18148b2199db259adfa923b45593fe08e"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.6"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[MicroCollections]]
deps = ["BangBang", "Setfield"]
git-tree-sha1 = "e991b6a9d38091c4a0d7cd051fcb57c05f98ac03"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.0"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "2ca267b08821e86c5ef4376cffed98a46c2cb205"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "3bc876dbff74cc9c4ece84ef9326da8ccd71c98f"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.7.28"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "2fd5787125d1a93fbe30961bd841707b8a80d75b"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.9.6"

[[NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[NonlinearSolve]]
deps = ["ArrayInterface", "FiniteDiff", "ForwardDiff", "IterativeSolvers", "LinearAlgebra", "RecursiveArrayTools", "RecursiveFactorization", "Reexport", "SciMLBase", "Setfield", "StaticArrays", "UnPack"]
git-tree-sha1 = "35585534c0c79c161241f2e65e759a11a79d25d0"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "0.3.10"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c870a0d713b51e4b49be6432eff0e26a4325afee"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.6"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "bfd7d8c7fd87f04543810d9cbd3995972236ba1b"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "c67334c786157d6ef091ce622b365d3d60b1e2c4"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.12"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "0036d433cacff4767ff622be3cb2c281b773a2b4"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.21.1"

[[PlutoTeachingTools]]
deps = ["LaTeXStrings", "Markdown", "PlutoUI", "Random"]
git-tree-sha1 = "e2b63ee022e0b20f43fcd15cda3a9047f449e3b4"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.1.4"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "Requires", "Static", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "114396b925753bb6ab11cf436c5ff854c72a59c2"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.4.2"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "cde4ce9d6f33219465b55162811d8de8139c0414"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.2.1"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "7dff99fbc740e2f8228c6878e2aad6d7c2678098"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.1"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "32efa73dece357e9c834cae8af00265752c80061"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.3.5"

[[RecursiveArrayTools]]
deps = ["ArrayInterface", "ChainRulesCore", "DocStringExtensions", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "00bede2eb099dcc1ddc3f9ec02180c326b420ee2"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.17.2"

[[RecursiveFactorization]]
deps = ["LinearAlgebra", "LoopVectorization", "Polyester", "StrideArraysCore", "TriangularSolve"]
git-tree-sha1 = "b1db8c4f4699d779cb4efe60d02e79b559a62a4d"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.2.3"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "bfdf9532c33db35d2ce9df4828330f0e92344a52"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.25"

[[SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "ff686e0c79dbe91767f4c1e44257621a5455b1c6"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.18.7"

[[ScientificTypesBase]]
git-tree-sha1 = "9c1a0dea3b442024c54ca6a318e8acf842eab06f"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "2.2.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "54f37736d8934a12a200edea2f9206b03bdf3159"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.7"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "fca29e68c5062722b5b4435594c3d1ba557072a3"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.7.1"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a322a9493e49c5f3a10b50df3aedaf1cdb3244b7"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.1"

[[SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "edef25a158db82f4940720ebada14a60ef6c4232"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.13"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "854b024a4a81b05c0792a4b45293b85db228bd27"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.3.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "730732cae4d3135e2f2182bd47f8d8b795ea4439"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "2.1.0"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "fed1ec1e65749c4d96fc20dd13bea72b55457e62"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.9"

[[StatsFuns]]
deps = ["IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "20d1bb720b9b27636280f751746ba4abb465f19d"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.9"

[[StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "e7d1e79232310bd654c7cef46465c537562af4fe"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.26"

[[StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "ManualMemory", "Requires", "SIMDTypes", "Static", "ThreadingUtilities"]
git-tree-sha1 = "6abbf6ed8d2a0135a899619260fb7432e120654d"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.2.1"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "1700b86ad59348c0f9f68ddc95117071f947072d"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.1"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "019acfd5a4a6c5f0f38de69f2ff7ed527f1881da"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.1.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "d0c690d37c73aeb5ca063056283fde5585a41710"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "d620a061cb2a56930b52bdf5cf908a5c4fa8e76a"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.4"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "03013c6ae7f1824131b2ae2fc1d49793b51e8394"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.4.6"

[[Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "LinearAlgebra", "MacroTools", "NNlib", "NaNMath", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "bf4adf36062afc921f251af4db58f06235504eff"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.16"

[[Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "34f27ac221cb53317ab6df196f9ed145077231ff"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.65"

[[TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[TriangularSolve]]
deps = ["CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "LoopVectorization", "Polyester", "Static", "VectorizationBase"]
git-tree-sha1 = "1eed054a58d9332adc731103fe47dad2ad1a0adf"
uuid = "d5829a12-d9aa-46ab-831f-fb7c9ab06edf"
version = "0.1.5"

[[Turing]]
deps = ["AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "DataStructures", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tracker", "ZygoteRules"]
git-tree-sha1 = "e22a11c2029137b35adf00a0e4842707c653938c"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.18.0"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "Hwloc", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static"]
git-tree-sha1 = "22fd0d5214b0423cb18b8cf40988f0606962c724"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.1"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "eae2fbbc34a79ffd57fb4c972b08ce50b8f6a00d"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.3"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "59e2ad8fd1591ea019a5259bd012d7aee15f995c"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.3"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "9e7a1e8ca60b742e508a315c17eef5211e7fbfd7"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.1"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "acc685bcf777b2202a904cdcb49ad34c2fa1880c"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.14.0+4"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7a5780a0d9c6864184b3a2eeeb833a0c871f00ab"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "0.1.6+4"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d713c1ce4deac133e3334ee12f4adff07f81778f"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2020.7.14+2"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "487da2f8f2f0c8ee0e83f39d13037d6bbf0a45ab"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.0.0+3"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
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
# ╠═bfc03760-17bb-4dac-a233-47dd3827519c
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
