### A Pluto.jl notebook ###
# v0.19.40

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

# ╔═╡ 3514a0be-49e7-47d5-bd92-784a4d745745
begin
  	import Pkg
  	Pkg.activate(@__DIR__)
	#Pkg.status()
	using Markdown
	using PlutoUI
	using Plots
	using Plots.PlotMeasures
	using PythonCall
	using PlutoTeachingTools
	using HypertextLiteral: @htl_str, @htl
	using Distances, LinearAlgebra
	using Random, Optim, Statistics, Distributions, ForwardDiff
end

# ╔═╡ d7e723d8-04f4-464b-9572-793f6349b63e
using BifurcationKit

# ╔═╡ 2b621d5e-1d9e-43ee-98db-1d5b5f2a80d7
using OptimalTransport

# ╔═╡ 9c684b79-bf17-4ee2-a2f5-9c5ef6dcae59
	using ApproxFun, GaussianProcesses

# ╔═╡ b2f34b36-6ce4-441f-a007-ae77273291c8
html"""
<script>
	const button = document.createElement("button")

	button.addEventListener("click", () => {
		editor_state_set(old_state => ({
			notebook: {
				...old_state.notebook,
				process_status: "no_process",
			},
		})).then(() => {
			window.requestAnimationFrame(() => {
				document.querySelector("#process_status a").click()
			})
		})
	})
	button.innerText = "Restart process"

	return button
</script>
"""

# ╔═╡ 9a7f6170-46f7-4bdb-8603-e00539a70a82
begin
	F(x, p) = @. p.μ + x - x^3/3
	prob = BifurcationProblem(F, [-2.], (μ = -1.,), (@lens _.μ);
        record_from_solution = (x,p) -> (x = x[1]))
	br = continuation(prob, PALC(), ContinuationPar(p_min = -1., p_max = 1.))
	nothing
end

# ╔═╡ 3f18bda8-9e22-4386-88b4-e1e175f40531
begin
	mtlib = pyimport("matplotlib")
	#plt = mtlib.pyplot
end

# ╔═╡ f49691f0-8a42-4768-ac6e-313053677397
plt = mtlib.pyplot

# ╔═╡ 3cffdfba-f85f-4372-b127-3dc987136a01
function inv_spectral(Q)
    EE_Q = eigen(Q)
    return EE_Q.vectors*diagm(EE_Q.values.^(-1))*EE_Q.vectors'
end

# ╔═╡ 316e1e9b-94c0-46e7-980e-13a65107d559
begin 
	#True "unknown" function to be modelled
	Ec(x) = sin(2*pi*x)
	#Ec(x) = exp(2*x)

	# machine-precision AD derivative (just so that we do not have to bother to compute it by hand for different examples):
	D_Ec(x) = ForwardDiff.derivative(Ec,x)
	E_full = Fun(Ec, Legendre())
	θ = E_full.coefficients # true coefficients
	#scatter(abs.(E_full.coefficients.+1e-16),title="Decay of coefficients",yaxis=:log,legend=false,size = (1000, 500))
	#plot!(xr, E_Pmean.(xr), ribbon=(abs.(aa-E_Pmean.(x)),bb-E_Pmean.(x)),fc=:orange,fa=0.3,label="model", linewidth=1)
	nothing
end

# ╔═╡ 85c82916-15b0-4782-b0f9-30ae1bd74e7c
########## teaching example
begin
	f1(x) = 1 / (1 + x^2)
	trapezoidal_rule(f, N) =  (
	    0.5*π/N * (f(-π) + f(π))
	+       π/N * sum( f(n*π/N) for n = -N+1:N-1 )  );
	NNN = 3:3:30       # number of quadrature points N means 2N+1 points
	I1 = 2 * atan(π)  # exact value of ∫ f₁
	I2 = √2 * π       # exact value of ∫ f₂
	I1N = trapezoidal_rule.(f1, NNN)   # trapezoidal rule approximations
	#I2N = trapezoidal_rule.(f2, NNN)
	E1N = abs.(I1N .- I1)   # errors
	#E2N = abs.(I2N .- I2)
end;

# ╔═╡ e2641ad6-4a54-4715-8669-27c62793a94d
begin
	Px = plot(NNN, E1N, lw=2, m=:o, ms=4, label = raw"$E_N[f_1]$", yaxis = :log10)
	#plot!(NNN, E2N.+1e-16, lw=2, m=:o, ms=4, label = raw"$E_N[f_2]$",
	#      xlabel = raw"$N$")
	P1 = plot!(deepcopy(Px), NNN[3:end], 0.04*NNN[3:end].^(-2), lw=2, ls=:dash, 
				c=:black, label = raw"$N^{-2}$", ylims = [1e-5, 1e-2],xticks = (NNN[1:3:end],string.(NNN[1:3:end])), xaxis = :log10)
	# P2 = plot!(Px, NNN[2:6], 0.1*exp.(- 2 * log(1 + sqrt(2)) * NNN[2:6]), lw=2,
	# 		   c=:black, ls = :dash, label = raw"$e^{- 2 \alpha N}$", 
	# 		   legend = :right, ylims = [1e-16, 1e-1])
	# alpha = log(sqrt(2)+1)
	fig_trap = plot(P1, size = (500, 350))
	nothing
end
#xticks = (1:10, string.(1:10)

# ╔═╡ 18e0a59e-89bb-4db3-aaab-269fbfa3a8a7
begin
	surface_step_video =  "https://i.imgur.com/0hMOElU.mp4"
	pyapd_logo = "https://raw.githubusercontent.com/mbuze/PyAPD/main/logo/logo.png"
	ebsd_scan = "https://i.imgur.com/6kuobQz.png"
	ip_figure = "https://mbuze.github.io/LJ.png"
	nothing
end

# ╔═╡ 25392b9f-a8d2-4609-88d4-c33e78853840
gr(dpi=400)
#gr()

# ╔═╡ 2c08d228-4afa-4d6b-ba8b-df22abb5ee09
html"""
<style>
	#green {
		color: #1b9e77;
	}
	#pink {
		color: #e7298a;
	}
	#magenta {
		color: #7570b3;
	}
	#orange {
		color: #d95f02;
	}
</style>
"""

# ╔═╡ 8702a769-6167-4342-9229-914f1f076a38
ChooseDisplayMode()

# ╔═╡ 930f698e-1827-4086-99d8-e8080ea1be4e
# html"""
# <style>
# 	main {
# 		margin: 0 auto;
# 		max-width: 1700px;
#     	padding-left: max(300px, 10%);
#     	padding-right: max(160px, 10%);
# 	}
# </style>
# """

# ╔═╡ a753156d-88a1-49ca-9223-5cb4077cc626
# html"<button onclick='present()'>present</button>"

# ╔═╡ fa0842af-36f4-46ef-8faf-dc5a6593957b
# struct TwoColumn{A, B}
# 	left::A
# 	right::B
# end

# ╔═╡ 1a0877a6-747b-4c0f-8993-9f5648d3e623
md"""
# Some mathematical aspects of materials modelling (and data science) in the age of AI

#### **Maciej Buze, 30 May 2024**

##### Candidate presentation for the position of **Lecturer in Mathematics and AI** at **Lancaster University**


"""

# ╔═╡ 60cf1098-5887-49fa-b28b-7d14d75d6ef2
md"""
## Hello

* #### My research spans a wide range of topics at the intersection of **applied and computational mathematics** and **mathematical analysis**
* #### I am primarily inspired by **applications in materials science, biology, physics and data science**.
* #### I use and develop tools in:
"""

# ╔═╡ 8b5b4088-060e-45e7-b3ff-1581f27a67df
md"""
| calculus of variations | bifurcation theory | numerical analysis | optimal transport |
|:---------- | ---------- |:------------:|:------------:|
|    |   |  | | 
"""

# ╔═╡ 035f9e75-5255-4798-83e1-e2723f1837f4
md"""
| approximation theory | uncertainty quantification | scientific GPU computing | machine learning |
|:---------- | ---------- |:------------:|:------------:|
|    |   |  | |
"""

# ╔═╡ acec9ef0-9e35-4a31-9a14-2e4114ed6e04
md"""
\

\

* #### I currently have three major ongoing research projects:
!!! note "A"
    #### Discrete modelling of nucleation and migration of defects in materials

!!! warning "B"
    #### Optimal transport: theory and applications

!!! tip "C"
    #### Uncertainty quantification for machine learning interatomic potentials
"""

# ╔═╡ 6a11493b-e486-4dee-a0ba-80fdd9b542da
md"""
## Discrete modelling of nucleation and migration of defects in materials

\


\

"""

# ╔═╡ cd58a897-3e18-4eb2-a81d-89dfb00257d0
md""" $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp $nbsp  $(Resource(surface_step_video, :width => 1300))
	"""

# ╔═╡ d5a8d207-80a6-4d22-b500-9448cbd0c339


# ╔═╡ 103b2f93-ebf0-42c8-92ca-2632a9b1005b


# ╔═╡ c3790891-2336-4108-9285-3ed85de872d7
TwoColumn(md""" $nbsp $nbsp """ , md"""
#### (1)""")

# ╔═╡ 0ed3fa19-b22b-46ca-a090-fcde0383d0d3
md"""
## Discrete modelling of nucleation and migration of defects in materials
"""

# ╔═╡ 3d0a89b5-3733-4029-a908-d31c8a38265b
html"""
	 <ul>
  	<li><h5>Modern high-throughput molecular and atomistic simulations rely on sophisticated optimisation tools to effectively <span id="magenta">explore the severely non-convex energy landscapes.</span></li>
  	<li><h5><span id="magenta">Bifurcation theory-based techniques</span> such as numerical continuation and deflation <span id="magenta">remain underutilised.</span></li>
  	<br>
	<li><h5>My work: mathematical and numerical <span id="green">analysis of idealised atomistic models, development of the NCFlex algorithm</span> (numerical continuation + flexible boundary conditions), <span id="pink">leading a working group on the adoption of numerical continuation tools in atomistic modelling of materials at the IPAM.</li>
	<li><h5><span id="green">5 published papers (SIMA, M3AS, M2AN, PhysRevE, Nonlinearity), one preprint (under review at MMS).</li>
	<br>
	<li><h5>The broader adoption of such tools at atomistic scales has been identified as highly desirable in a recent white paper, a summary document of the <span id="pink">IPAM Long Program New Mathematics for the Exascale: Applications to Materials Science</span>.</li>
	<li><h5><span id="pink">ICMS Workshop: Computational Materials Science and Mathematics at the Particle and Atomistic Scales, November 2025, £24,000 funding.</span></li>
	<br>
	<li><h5>Future funding proposals: <span id="orange">EPSRC Postdoctoral Fellowship -> New Investigator Award, Leverhulme Research Project Grants scheme.</li>
	<br>
	<li><h6><div>Collaborators:</div>C.Ortner (UBC), T.Hudson (Warwick), J.Braun (Heriot-Watt), J.Kermode (Warwick), F.Birks (Warwick, PhD student I help supervise), S.Bagchi (Los Alamos), D.Perez (Los Alamos), T.Swinburne (Marseille), P. van Meurs (Kanazawa)	</ul> 
	"""

# ╔═╡ 1705424a-163d-4ec3-b5d5-c5bd4b0ca7ea
@bind ptt Slider(1:1:length(br), default=1)

# ╔═╡ cd8181a0-53be-40f0-94ac-680e41192e31
TwoColumn(md"""
Find lines belonging to the set 
```math
\{ (x,\mu) \in \mathbb{R}^2 \,\mid\, F(x,\mu)= 0\},\quad\text{ where }\quad F(x,\mu) = \mu + x - \frac13x^3
```
```
F(x, p) = @. p.μ + x - x^3/3
prob = BifurcationProblem(F, [-2.], (μ = -1.,), (@lens _.μ);
	record_from_solution = (x,p) -> (x = x[1]))
br = continuation(prob, PALC(), ContinuationPar(p_min = -1., p_max = 1.))
```
"""	, 
begin
	plot(br,xlabel="", ylabel="")
	scatter!((br.sol[ptt].p,br.sol[ptt].x[1]),markersize=12)
	plot!(size=(1000,300),legend=false,)
	
end
)

# ╔═╡ 6a8aeb0a-a570-4f74-83d3-ab321ab25911
TwoColumn(md""" $nbsp $nbsp """ , md"""
#### (2)""")

# ╔═╡ 2fab6ede-98d3-4783-9bc3-6ae4f95e9c8e
md"""
## Optimal transport: theory and applications
"""

# ╔═╡ 9a6e9a57-8ac1-47f1-b00b-7c3907b2720e
TwoColumnWideLeft(
	html"""
	 <ul>
  	<li><h5>Optimal Transport theory continues to develop at pace as one of the fundamental mathematical theories with an ever-growing list of <span id="magenta">diverse applications in fields such as machine learning, economics, computer
vision, image processing,</span> <span id="green">materials science.</span></li>
  	<li><h5>Recent advances include <span id="magenta">theory of unbalanced OT</span> and <span id="green">GPU-accelerated computational OT.</span></li>
	<br>
	<li><h5> <div> My work: (i) theory of unbalanced OT: <span id="magenta">conic entropic regularisation, barycenters</span>;</div>
	(ii) <span id="green"> GPU-accelerated semi-discrete OT with exotic costs (with materials modelling application).</span></li>
	<li><h5>3 preprints (under review in <span id="magenta">SIMA</span> and <span id="green">Comput. Mater. Sci.</span>, one to be submitted to <span id="magenta">Nonlinearity</a>)</li>
	<br>
	<li><h5><span id="pink">Invitation to Paris in June to present my work.</li>
	<br>
	<li><h5>Future funding proposals: <span id="orange">industry-focused Horizon Europe grant with Tata Steel, EPSRC Small Grant.</li>
	</ul> 
	""",
	md""" $(Resource(pyapd_logo, :width => 1000))
	$nbsp PyAPD: A Python library for computing (optimal) anisotropic power diagrams using GPU acceleration\
	
	######  Collaborators: D.Bourne (Heriot-Watt), S.Roper (Glasgow), J.Feydy (Inria Paris), K. Sedighiani (Tata Steel), H.Duong (Birmingham).
	"""
)

# ╔═╡ 53c0f81d-0019-4799-8725-e4b3420d684e
pyapd = pyimport("PyAPD")

# ╔═╡ c62a52e1-249b-48a6-90ae-f8552bf9a224
@bind NN Slider(10:5:150, default=50)

# ╔═╡ 3ec7601c-2b0f-4c47-8467-4079a338b155
apd = pyapd.apd_system(N=NN, pixel_size_prefactor=4,  ani_thres=0.5)

# ╔═╡ e8496864-b5a1-4a08-b2da-c7f882f35763
apd.find_optimal_W()

# ╔═╡ ed1fa3d5-ef9b-489a-9f73-0067896f1c95
ThreeColumn(md"""
Anistropic Power Diagrams generated by $\{x_i,A_i,w_i\}_{i=1}^N$, where $N=$ $NN.
```math
  L_i = \{{ x} \in \Omega\,\mid\, |{ x}-{ x_i}|^2_{ A_i} - { w_i} \leq |{ x}-{ x_j}|^2_{ A_j} - { w_j},\quad \forall j\},
```
```
apd = pyapd.apd_system(N=NN, pixel_size_prefactor=3,  ani_thres=0.5)
apd.find_optimal_W()
assignment = apd.assemble_apd()
LL = Int(sqrt(length(apd.Y)))
heatmap_data = reshape(convert(Array,PyArray(assignment)),(LL,LL))
heatmap(1:LL,1:LL,heatmap_data)
```
"""	, 
begin
	assignment = apd.assemble_apd()
	LL = Int(sqrt(length(apd.Y)))
	heatmap_data = reshape(convert(Array,PyArray(assignment)),(LL,LL))
	c = cgrad(:cmyk)
	heatmap(1:LL,1:LL,heatmap_data,
	color = c,
	aspect_ratio=:equal,  # 1:1 aspect ratio
    axis=([], false),     # Hide axes
    cbar=false,           # Hide color bar
	margins=0pt,# Hide margins
    size=(650, 355),          # Figure size
)
end
	,
		md""" $(Resource(ebsd_scan, :height => 350))
	"""
)

# ╔═╡ 9b62c56d-4ced-4117-b20f-040a84ce8d3a
TwoColumn(md""" $nbsp $nbsp """ , md"""
#### (3)""")

# ╔═╡ 127a3661-b34a-458d-adea-7d9e98ed3415
md"""
## Optimal transport: theory and applications
"""

# ╔═╡ 7ac043a5-f6f9-4826-a8f1-29fdaa7687b1
TwoColumnWideLeft(
	html"""
	 <ul>
  	<li><h5>Optimal Transport theory continues to develop at pace as one of the fundamental mathematical theories with an ever-growing list of <span id="magenta">diverse applications in fields such as machine learning, economics, computer
vision, image processing,</span> <span id="green">materials science.</span></li>
  	<li><h5>Recent advances include <span id="magenta">theory of unbalanced OT</span> and <span id="green">GPU-accelerated computational OT.</span></li>
	<br>
	<li><h5> <div> My work: (i) theory of unbalanced OT: <span id="magenta">conic entropic regularisation, barycenters</span>;</div>
	(ii) <span id="green"> GPU-accelerated semi-discrete OT with exotic costs (with materials modelling application).</span></li>
	<li><h5>3 preprints (under review in <span id="magenta">SIMA</span> and <span id="green">Comput. Mater. Sci.</span>, one to be submitted to <span id="magenta">Nonlinearity</a>)</li>
	<br>
	<li><h5><span id="pink">Invitation to Paris in June to present my work.</li>
	<br>
	<li><h5>Future funding proposals: <span id="green">industry-focused Horizon Europe grant with Tata Steel</span>, <span id="magenta">EPSRC Small Grant.</li>
	</ul> 
	""",
	md""" $(Resource(pyapd_logo, :width => 1000))
	$nbsp PyAPD: A Python library for computing (optimal) anisotropic power diagrams using GPU acceleration\
	
	######  Collaborators: D.Bourne (Heriot-Watt), S.Roper (Glasgow), J.Feydy (Inria Paris), K. Sedighiani (Tata Steel), H.Duong (Birmingham).
	"""
)

# ╔═╡ 94d185b1-e3b7-46fe-a4ff-60c5956d86b2
@bind λ Slider(0.1:0.1:0.9, default=0.5)

# ╔═╡ f64441d7-e11f-4638-a60c-1c0bdac8bd57
begin
	my_support = range(-2, 2; length=250)
	mu1 = normalize!(exp.(-(my_support .+ 0.75) .^ 2 ./ 0.5^2), 1)
	mu2 = normalize!(exp.(-(my_support .- 0.5) .^ 2 ./ 0.4^2), 1)
	
	plt3 = plot(; size=(700, 350), legend=false,title="\$ \\lambda_1 = $λ\$")
	plot!(plt3, my_support, mu1; label=raw"$\mu_1$",linewidth=4)
	plot!(plt3, my_support, mu2; label=raw"$\mu_2$",linewidth=4)
	
	mu = hcat(mu1, mu2)
	C = pairwise(SqEuclidean(), my_support'; dims=2)
	for λ1 in (λ)
	    λ2 = 1 - λ1
	    a = sinkhorn_barycenter(mu, C, 0.01, [λ1, λ2], SinkhornGibbs())
	    plot!(plt3, my_support, a; label="\$\\mu \\quad (\\lambda_1 = $λ1)\$",linewidth=4)
	end
end

# ╔═╡ 39e98f07-d1d7-4f4b-b052-77dfa4c5d322
TwoColumnWideLeft(md"""
Finding a barycentric description of a set of $N$ probability measures:
```math
\inf_{\nu \in \mathcal{P}(\mathbb{R})} \sum_{i=1}^N \lambda_i{\rm OT}(\mu_i,\nu),\quad  {\rm OT}(\mu,\nu) = \min_{\gamma} \left\{ \int_{\mathbb R \times \mathbb R} |x-y|^2 d\gamma(x,y)\, \Bigm\vert \, \gamma \in \mathcal{P}(\mathbb R \times \mathbb R),\; \gamma_x = \mu,\; \gamma_y = \nu\right\}
```
```
support = range(-2, 2; length=250)
mu1 = normalize!(exp.(-(support .+ 0.75) .^ 2 ./ 0.5^2), 1)
mu2 = normalize!(exp.(-(support .- 0.5) .^ 2 ./ 0.4^2), 1)
plt3 = plot(; size=(800, 400), legend=false)
plot!(plt3, support, mu1; linewidth=4)
plot!(plt3, support, mu2; linewidth=4)
mu = hcat(mu1, mu2)
C = pairwise(SqEuclidean(), support'; dims=2)
for λ1 in (λ)
	λ2 = 1 - λ1
	a = sinkhorn_barycenter(mu, C, 0.01, [λ1, λ2], SinkhornGibbs())
	plot!(plt3, support, a; linewidth=4)
end
```
"""	, 
begin
	plt3
end
)

# ╔═╡ f0face85-80c0-4367-b5e2-a13c4598bcf7
TwoColumn(md""" $nbsp $nbsp """ , md"""
#### (4)""")

# ╔═╡ d4660044-8438-4502-84e1-031813e9ca25
md"""
## Uncertainty Quantification for Machine Learning Interatomic Potentials
"""

# ╔═╡ 7ba63a0e-c205-44df-9df3-50481886c15d
TwoColumnWideLeft(
	html"""
	 <ul>
  	<li><h5><span id="magenta">Interatomic potentials</span> approximate the potential energy of systems of atoms as a function of their positions and are seen <span id="magenta">as a computationally feasible alternative to electronic structure calculations.</span></li>
  	<li><h5><span id="green">Empirical potentials have between 2 and 11 parameters</span>, rising to <span id="orange">1000 for modern machine-learning potentials (MLIPs)</span>. <span id="magenta">Highly nonlinear, UQ essential!</li>
	<br>
	<li><h5>My work: (i) <span id="green">maximum entropy priors for empirical potentials</span>; (ii) <span id="orange">Combining approximation theory and Bayesian inference for MLIPs .</li>
	<li><h5><span id="green">One paper published (SIAP), <span id="orange">one manuscript in preparation.</span></li>
	<br>
	<li><h5><span id="pink">Visiting Fellowship at Newton Institute during the programme: Uncertainty quantification and stochastic modelling of materials</li>
	<br>
	<li><h5>Future funding proposals: <span id="orange">The Higher Education, Research and Innovation Department (HERI) of the French Embassy (funding for UK-France research links), Research-in-Groups at ICMS.</li>
	</ul> 
	""",
	md""" $(Resource(ip_figure, :width => 800))

	###### Collaborators: A.Mihai (Cardiff), T.Woolley (Cardiff), G.Dusson (CNRS Bourgogne Franche-Comté), J.Kermode (Warwick)
	"""
)

# ╔═╡ a4b9fb3a-2a77-4301-b03f-702386375513
TwoColumn(
@bind data_points Slider(5:1:35, default=10)
,
@bind poly_degree Slider(1:1:13, default=9)
	)

# ╔═╡ 3d5b6447-60de-490b-aa62-e6500460eb5e
begin
	n=data_points  #number of training points
	x = range(-1, 1, length=n) #equidistant 
	P = poly_degree # Polynomial degree

	# Observations of values only
	y = Ec.(x)

	# Function space for approximations on the given interval
	S = Legendre(x[begin]..x[end])

	# Build design matrix for linear regression by evaluating each
	# polynomial at the given x values
	A = hcat((Fun(S, [zeros(k-1); 1]).(x) for k in 1:P)...)
	
	# Use linear regression to find the Legendre coefficients
	θ_tilde0 = A\y
	E_P0 = Fun(S, θ_tilde0)
	
	
	# Deterministic polynomial approach, function values and derivatives:
	y1 = Ec.(x)    #a bit of noise here is needed for Optim to be able to optimise over hyparameters
	y2 = D_Ec.(x)
	Y = [y;y2]
		
	# Function space for approximations on the given interval
	S = Legendre(x[begin]..x[end])
	Der = Derivative(Legendre(), 1)
	
	# Build design matrix for linear regression by evaluating each
	# polynomial at the given x values
	A = hcat((Fun(S, [zeros(k-1); 1]).(x) for k in 1:P)...)
	AA = hcat(((Der*Fun(S, [zeros(k-1); 1])).(x) for k in 1:P)...)
	AAA = vcat(A,AA)
	
	# Use linear regression to find the Legendre coefficients
	θ_tilde1 = AAA\Y
	E_P1 = Fun(S, θ_tilde1)
	
	# plotting
	xr = range(-1,1,length=100)
	fig = scatter(x, y,label="data",size = (1000, 500))
	plot!(xr, E_P0.(xr), label="fitted model, f based ")
	plot!(xr, E_P1.(xr), label="fitted model, f and Df based ")
	plot!(xr,Ec.(xr), label= "truth")
	#fig
	nothing
end

# ╔═╡ e46a4af0-e06b-4a5f-98fc-4aa7fee904d9
function find_ribbon(xx,m_t,cov_t)
    d = MvNormal(m_t, cov_t)
    lower_bound = mean(d) - var(d)
    upper_bound = mean(d) + var(d)
    A = hcat((Fun(S, [zeros(k-1); 1]).(xx) for k in 1:P)...)
    lower_envelope = zeros(length(xx))
    upper_envelope = zeros(length(xx))
    for i in 1:length(xx)
        lower_envelope[i] = optimize(θ -> (A*θ)[i], lower_bound, upper_bound, mean(d)).minimum
        upper_envelope[i] = -optimize(θ -> -(A*θ)[i], lower_bound, upper_bound, mean(d)).minimum
    end
    return lower_envelope, upper_envelope
end

# ╔═╡ 8322e1b4-5f97-465d-a31e-c6f3dd2d675d
begin
	# Bayesian approach

	# Use linear regression to find the Legendre coefficients
	c_bar = A\y
	
	# we can the GP kernel to define a covariance matrix
	σ = 0.01 # level of noise
	#with default hyperparameters:
	hp_0 = [0.0,0.0]
	kern = SE(hp_0[1],hp_0[2])                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
	Q0 = Statistics.cov(kern,collect(x)',collect(x)')
	
	# this is actually by itself nearly singular, so we instead work with
	# recall that σ is the level of noise defined above
	Q0 = Q0 + σ*diagm(ones(n))
	
	#with optimised hyperparameters:
	hp_1 = [-11.899785696162567,-0.35137099604755984]
	kern = SE(hp_1[1],hp_1[2])
	Q1 = Statistics.cov(kern,collect(x)',collect(x)')
	Q1 = Q1 + σ*diagm(ones(n))
	
	
	Q = Q0
	
	## Young's formalism:
	λ_UQ = 2.0
	variance_coefficients(j) = factorial(j-1)*λ_UQ^j
	#variance_coefficients(j) = λ
	
	τ = 0.1
	λ_s = [variance_coefficients(i) for i in 1:(P-1)]
	
	C_UQ = diagm(append!([1.0],τ./λ_s))
	

	B = inv_spectral(A'*inv_spectral(Q)*A + inv_spectral(C_UQ))
	b = A'inv_spectral(Q)*y + inv_spectral(C_UQ)*c_bar
	;
end

# ╔═╡ 0f5f332b-80bd-4b60-a2a4-d692e97f4a86
begin
	aa, bb = find_ribbon(x,B*b,Symmetric(B))
	E_Pmean = Fun(S, B*b)
	# plotting
	#xr = range(-1,1,length=100)
	fig2 = scatter(x, y,label="data", size = (900, 300), title="\$ N = $(data_points),\\quad  P = $(poly_degree)\$")
	plot!(xr, E_Pmean.(xr), ribbon=(abs.(aa-E_Pmean.(x)),bb-E_Pmean.(x)),fc=:orange,fa=0.3,label="", linewidth=4)
	plot!(xr, E_P0.(xr), label="model, f based ", linewidth=4)
	plot!(xr, E_P1.(xr), label="model, f and Df based ", linewidth=4)
	plot!(xr,Ec.(xr), label= "truth",linewidth=4)
	nothing
end

# ╔═╡ 99c81597-09df-40a3-bc03-42fa4fbd4802
TwoColumn(md"""
Given an unknown $\mathcal{E}\,\colon\,[-1,1] \to \mathbb{R}$ and data $D_N = \{(x_i,y_i)\}_{i=1}^N, y_i = (\mathcal{E}(x_i),\mathcal{E}'(x_i))$, let us reconstruct it using Legendre polynomials $E_P(x) := \sum_{i=1}^P c_i p_i(x)$.
```
	x = range(-1, 1, length=n)
	Ec(x) = sin(2*pi*x)
	D_Ec(x) = ForwardDiff.derivative(Ec,x)
	Y = [Ec.(x);D_Ec.(x)]
		
	S = Legendre(x[begin]..x[end]) 	# Function space for approximations on the given interval
	Der = Derivative(Legendre(), 1) 

	# Build design matrix for linear regression
	A = hcat((Fun(S, [zeros(k-1); 1]).(x) for k in 1:P)...)
	AA = hcat(((Der*Fun(S, [zeros(k-1); 1])).(x) for k in 1:P)...)
	AAA = vcat(A,AA)
	
	# Use linear regression to find the Legendre coefficients
	θ_tilde1 = AAA\Y
	E_P1 = Fun(S, θ_tilde1)
```
"""	, 
begin
	fig2
end
)

# ╔═╡ 071cdefb-bdd4-4f77-8b63-4e1408aedc6a
TwoColumn(md""" $nbsp $nbsp """ , md"""
#### (5)""")

# ╔═╡ 47589ea7-8723-4375-9a50-4caf0909b18b
md"""
## My vision for the success of MARS
"""

# ╔═╡ b47e6443-c067-4c34-9b86-6f3dca8a27e9
	html"""
	<h3> Mathematical research in the age of AI
	 <ul>
  	<li><h5>Unprecedented access to data and computing power</span></li>
	<li><h5> Mathematicians are best positioned to "tame" the complexities involved with rigorous "pen & paper" analysis (we might be the last profession standing...)</li>
	<li><h5>But for that to happen, especially for Real-world Systems, we need to "get our hands dirty": engage with practitioners, utilise computing resources (such as GPU-computing), embrace the modern "software-engineering" working practices:</li>
	<ul>
		<li><h5>Github/Gitlab integration, version control </li>
		<li><h5>cloud computing, high-performance computing resources</li>
		<li><h5>open-source communication-platforms such as Zulip</li>
		<li><h5>project-management tools such as Quire</li>
	</ul></li>
	</ul>
	<h3> My contribution
	<ul>
		<li><h5>cutting-edge applied analysis research in atomistic modelling of materials leading to reproducible results and usable software aimed at and developed with practitioners </li>
		<li><h5>computationally-oriented data science research in optimal transport, with a materials engineering application and an industrial focus</li>
		<li><h5>high-performance computing simulations using modern machine learning and AI oriented methodologies</li>
		<li><h5><span id="pink">an open mind to new projects and collaborations and a broad research portfolio with many potential applications - vital to ensure the early-stage success of MARS</span></li>
	</ul>
	<h3> Research links with Lancaster		
	"""

# ╔═╡ 58a81bb8-6714-48e4-b1e2-bd8b2fe34675
TwoColumn(html"""

		<h5>Mathematics and Statistics Department:
		<ul>
			<li><span id="green"> Prof. Gordon Blower </span> - optimal transport: <i>Transportation on spheres via an entropy formula</i></li>
			<li><span id="green"> Prof. Chris Sherlock</span> - optimal transport: <i>Bounds on Wasserstein distances between distributions using independent samples</i>
			<li><span id="green"> Prof. Christopher Nemeth</span> - optimal transport: <i>Coin Wasserstein Gradient Descent (Coin-ParVI)</i>
			<li><span id="orange">Prob_AI_Hub</span> - optimal transport: Matthew Thorpe (co-I, Warwick) </li>
			<li><span id="green">Dr Anthony Nixon</span> and <span id="green"> Dr Bernd Schulze </span> - discrete and computational geometry</li>
			<li><span id="green">Prof. Christopher Jewell</span> - GPU computing: <i>Tensorflow Probability</i>
		</ul>
"""
	,
	html"""
<h5>Wider University environment:
		<ul>
			<li><h5><span id="green"> Dr Samuel Murphy (Nuclear Materials and Engineering)</span> - MD simulations, DefAP: A python code for the analysis of point defects in crystalline solids</li>
			<li><h5><span id="green"> Dr Neil Drummond (Physics) </span> - Electronic-structure calculation and quantum Monte Carlo simulation for 2D materials</li>
			<li><h5><span id="green"> Prof. Abbie Trewin (Materials Chemistry) </span> - The Amorphous Builder (Ambuild) code which involves geometry optimisation and running of MD simulations</li>
			<li><h5><span id="green"> Dr Wei Wen (Mechanical Engineering) </span> - multi-scale modelling of Crystal Plasticity (CP).</li>
		</ul>
	"""
)

# ╔═╡ 0f33a0f6-052f-49eb-803a-e108c9509568
TwoColumn(md""" $nbsp $nbsp """ , md"""
#### (6)""")

# ╔═╡ 66890fc9-1872-4c5a-80bd-f4194428b42b
md"""
## My vision for the success of MARS
"""

# ╔═╡ 1fb30bc5-3ddd-4d87-b5b9-9009a674ff29
	html"""
	<h3> Mathematical education in the age of AI
	 <ul>
  	<li><h5>Unprecedented access to data and computing power</span></li>
	<br>
  	<li><h5> Critical, creative and abstract thinking can soon be the last remaining employability asset...</li>
	<br>
	<li><h5>We should double-down on rigorous mathematical education, but embed it into a modern <span id="orange">computationally-native</span> context:</li>
	<ul>
		<li><h5>Github/Gitlab integration, version control </li>
		<li><h5>cloud computing, high-performance computing resources</li>
		<li><h5>open-source communication-platforms such as Zulip</li>
		<li><h5>project-management tools such as Quire</li>
	</ul></li>
	</ul>
	<h3> My contribution
	<ul>
		<li><h5><span id="orange"> developing modern analysis / numerical analysis / numerical methods / scientific computation courses at various levels, in which I will combine my versatile computational experience in Python, Julia, SageMath and Desmos. </li>
		<br>
		<li><h5>bespoke GitLab integration, version control for homework submission </li>
		<br>
		<li><h5>engaging with students using industry standards (Slack/Zulip/Quire)</li>
	"""

# ╔═╡ 3dbfa412-dbd6-4efe-9c81-118905ad4bc0


# ╔═╡ fb474f4d-fc4e-4d4a-b783-65de9a64499b


# ╔═╡ eb8e0263-886a-4ad4-afd7-3174f9756e2a


# ╔═╡ d6b5c960-e9d6-4079-b950-d7c465782b29


# ╔═╡ 6f395ff1-ffd2-46ad-b1b6-538b53a00a98
TwoColumn(md""" $nbsp $nbsp """ , md"""
#### (7)""")

# ╔═╡ 89f8c606-c8a5-4300-9662-8b36e2537f8c
md"""
## "Computationally-native" teaching example
"""

# ╔═╡ 4a87efe7-f1ec-49ba-99d4-75515879d540
md"""
Given some function $f\,\colon\, \mathbb{R} \to \mathbb{R}$, we can approximate the integral $I[f] := \int_{-\pi}^\pi f(x) \,dx$ with a quadrature rule, the [composite trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule)
```math
\begin{aligned}
	I[f] \approx I_N[f] := \sum_{n = -N+1}^N \frac{2\pi}{2N}
		\cdot \frac{f(x_{n-1})) + f(x_n)}{2}
\end{aligned}
```
where $x_n = 2\pi n / (2N) = \pi n/N, n = -N, \dots, N$ are the quadrature nodes.

If $f \in C^1$ then it is not too difficult to show that on each sub-interval $(x_{n-1}, x_n)$ of length ``h \approx 1/N`` approximating ``f`` with a piecewise affine function yields an ``O(h^2)`` error and therefore the total error is expected to also scale like ``h^2 \approx N^{-2}``, i.e., we expect that
```math
  |I[f] - I_N[f]| \lesssim N^{-2}
```
To test this numerically we implement the quadrature rule using the code 
	```
trapezoidal_rule(f, N) =  (
	    0.5*π/N * (f(-π) + f(π))
	+       π/N * sum( f(n*π/N) for n = -N+1:N-1 )  );

f1(x) = 1 / (1 + x^2) # example function
NNN = 3:3:30       # number of quadrature points N means 2N+1 points
I1 = 2 * atan(π)  # exact value of ∫ f
I1N = trapezoidal_rule.(f1, NNN)   # trapezoidal rule approximations
E1N = abs.(I1N .- I1)   # errors
```
"""

# ╔═╡ 2edfa6cb-beb4-4f63-a736-7efdaf3dd07e
md"""
The error plot is on the right, and on the left we have a Desmos interactive plot:
"""

# ╔═╡ f910535b-4746-4845-b5ec-7fecc9b15656
TwoColumnWideLeft(
	@htl("""
<script src="https://www.desmos.com/api/v1.9/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6"></script>

<div id="calculator" style="width: 1200px; height: 350px;"></div>

<script>
    var elt = document.getElementById('calculator');
    var calculator = Desmos.GraphingCalculator(elt);
  	calculator.setExpressions(
	[
	  {
	    "type": "text",
	    "id": "2",
	    "text": "Trapezoidal rule for definite integrals"
	  },
	  {
	    "type": "expression",
	    "id": "3",
	    "color": "#000000",
	    "latex": "f\\\\left(x\\\\right)= 1 / (1 + x^2)"
	  },
	  {
	    "type": "expression",
	    "id": "4",
	    "color": "#388c46",
	    "latex": "a=-3.14",
	    "hidden": true
	  },
	  {
	    "type": "expression",
	    "id": "5",
	    "color": "#6042a6",
	    "latex": "b=3.14",
	    "hidden": true
	  },
	  {
	    "type": "expression",
	    "id": "7",
	    "color": "#c74440",
	    "latex": "n=4",
	    "slider": {
	      "hardMin": true,
	      "hardMax": true,
	      "playDirection": -1,
	      "min": "0",
	      "max": "20",
	      "step": "1"
	    }
	  },
	  {
	    "type": "expression",
	    "id": "15",
	    "color": "#388c46",
	    "latex": "T_n"
	  },
	  {
	    "type": "folder",
	    "id": "9",
	    "title": "Trapezoidal Sum"
	  },
	  {
	    "type": "expression",
	    "id": "17",
	    "folderId": "9",
	    "color": "#000000",
	    "latex": "g\\\\left(x\\\\right)=f\\\\left(x\\\\right)\\\\left\\\\{a\\\\le x\\\\le b\\\\right\\\\}"
	  },
	  {
	    "type": "expression",
	    "id": "16",
	    "folderId": "9",
	    "color": "#000000",
	    "latex": "\\\\min\\\\left(0,g\\\\left(x\\\\right)\\\\right)\\\\le y\\\\le\\\\max\\\\left(0,g\\\\left(x\\\\right)\\\\right)"
	  },
	  {
	    "type": "expression",
	    "id": "11",
	    "folderId": "9",
	    "color": "#6042a6",
	    "latex": "i=\\\\left[1...n\\\\right]"
	  },
	  {
	    "type": "expression",
	    "id": "19",
	    "folderId": "9",
	    "color": "#2d70b3",
	    "latex": "x_{T1}=a+\\\\left(i-1\\\\right)\\\\frac{b-a}{n}"
	  },
	  {
	    "type": "expression",
	    "id": "20",
	    "folderId": "9",
	    "color": "#388c46",
	    "latex": "x_{T2}=a+i\\\\frac{b-a}{n}"
	  },
	  {
	    "type": "expression",
	    "id": "18",
	    "folderId": "9",
	    "color": "#c74440",
	    "latex": "T\\\\left(x\\\\right)=f\\\\left(x_{T1}\\\\right)\\\\cdot\\\\frac{x-x_{T2}}{x_{T1}-x_{T2}}+f\\\\left(x_{T2}\\\\right)\\\\cdot\\\\frac{x-x_{T1}}{x_{T2}-x_{T1}}",
	    "hidden": true
	  },
	  {
	    "type": "expression",
	    "id": "10",
	    "folderId": "9",
	    "color": "#c74440",
	    "latex": "\\\\min\\\\left(0,T\\\\left(x\\\\right)\\\\right)\\\\le y\\\\le\\\\max\\\\left(0,T\\\\left(x\\\\right)\\\\right)\\\\left\\\\{x_{T1}\\\\le x\\\\le x_{T2}\\\\right\\\\}"
	  },
	  {
	    "type": "expression",
	    "id": "12",
	    "folderId": "9",
	    "color": "#c74440",
	    "latex": "x=a+\\\\left(i-1\\\\right)\\\\frac{b-a}{n}\\\\left\\\\{\\\\min\\\\left(0,f\\\\left(a+\\\\left(i-1\\\\right)\\\\frac{b-a}{n}\\\\right)\\\\right)\\\\le y\\\\le\\\\max\\\\left(0,f\\\\left(a+\\\\left(i-1\\\\right)\\\\frac{b-a}{n}\\\\right)\\\\right)\\\\right\\\\}",
	    "lineOpacity": "1.0",
	    "lineWidth": "5.0"
	  },
	  {
	    "type": "expression",
	    "id": "13",
	    "folderId": "9",
	    "color": "#c74440",
	    "latex": "x=b\\\\left\\\\{\\\\min\\\\left(0,f\\\\left(b\\\\right)\\\\right)\\\\le y\\\\le\\\\max\\\\left(0,f\\\\left(b\\\\right)\\\\right)\\\\right\\\\}"
	  },
	  {
	    "type": "expression",
	    "id": "14",
	    "folderId": "9",
	    "color": "#2d70b3",
	    "latex": "T_n=\\\\frac{1}{2}\\\\cdot\\\\frac{b-a}{n}\\\\left(f\\\\left(a\\\\right)+2\\\\sum_{t=1}^{n-1}f\\\\left(a+t\\\\frac{b-a}{n}\\\\right)+f\\\\left(b\\\\right)\\\\right)"
	  }
	]


	)
</script>

""")
	,
	fig_trap
)

# ╔═╡ eadb07d9-9227-4479-890a-f3bea3efc8f5
TwoColumn(md""" $nbsp $nbsp """ , md"""
#### (8)""")

# ╔═╡ Cell order:
# ╟─b2f34b36-6ce4-441f-a007-ae77273291c8
# ╠═3514a0be-49e7-47d5-bd92-784a4d745745
# ╠═9a7f6170-46f7-4bdb-8603-e00539a70a82
# ╠═f64441d7-e11f-4638-a60c-1c0bdac8bd57
# ╠═3f18bda8-9e22-4386-88b4-e1e175f40531
# ╠═f49691f0-8a42-4768-ac6e-313053677397
# ╠═3ec7601c-2b0f-4c47-8467-4079a338b155
# ╠═e8496864-b5a1-4a08-b2da-c7f882f35763
# ╠═0f5f332b-80bd-4b60-a2a4-d692e97f4a86
# ╠═e46a4af0-e06b-4a5f-98fc-4aa7fee904d9
# ╠═8322e1b4-5f97-465d-a31e-c6f3dd2d675d
# ╠═3cffdfba-f85f-4372-b127-3dc987136a01
# ╠═316e1e9b-94c0-46e7-980e-13a65107d559
# ╠═3d5b6447-60de-490b-aa62-e6500460eb5e
# ╠═85c82916-15b0-4782-b0f9-30ae1bd74e7c
# ╠═e2641ad6-4a54-4715-8669-27c62793a94d
# ╠═18e0a59e-89bb-4db3-aaab-269fbfa3a8a7
# ╠═25392b9f-a8d2-4609-88d4-c33e78853840
# ╠═2c08d228-4afa-4d6b-ba8b-df22abb5ee09
# ╠═8702a769-6167-4342-9229-914f1f076a38
# ╟─930f698e-1827-4086-99d8-e8080ea1be4e
# ╟─a753156d-88a1-49ca-9223-5cb4077cc626
# ╟─fa0842af-36f4-46ef-8faf-dc5a6593957b
# ╟─1a0877a6-747b-4c0f-8993-9f5648d3e623
# ╟─60cf1098-5887-49fa-b28b-7d14d75d6ef2
# ╟─8b5b4088-060e-45e7-b3ff-1581f27a67df
# ╟─035f9e75-5255-4798-83e1-e2723f1837f4
# ╟─acec9ef0-9e35-4a31-9a14-2e4114ed6e04
# ╟─6a11493b-e486-4dee-a0ba-80fdd9b542da
# ╟─cd58a897-3e18-4eb2-a81d-89dfb00257d0
# ╟─d5a8d207-80a6-4d22-b500-9448cbd0c339
# ╟─103b2f93-ebf0-42c8-92ca-2632a9b1005b
# ╟─c3790891-2336-4108-9285-3ed85de872d7
# ╟─0ed3fa19-b22b-46ca-a090-fcde0383d0d3
# ╟─3d0a89b5-3733-4029-a908-d31c8a38265b
# ╠═d7e723d8-04f4-464b-9572-793f6349b63e
# ╟─1705424a-163d-4ec3-b5d5-c5bd4b0ca7ea
# ╟─cd8181a0-53be-40f0-94ac-680e41192e31
# ╟─6a8aeb0a-a570-4f74-83d3-ab321ab25911
# ╟─2fab6ede-98d3-4783-9bc3-6ae4f95e9c8e
# ╟─9a6e9a57-8ac1-47f1-b00b-7c3907b2720e
# ╠═53c0f81d-0019-4799-8725-e4b3420d684e
# ╟─c62a52e1-249b-48a6-90ae-f8552bf9a224
# ╟─ed1fa3d5-ef9b-489a-9f73-0067896f1c95
# ╟─9b62c56d-4ced-4117-b20f-040a84ce8d3a
# ╟─127a3661-b34a-458d-adea-7d9e98ed3415
# ╟─7ac043a5-f6f9-4826-a8f1-29fdaa7687b1
# ╠═2b621d5e-1d9e-43ee-98db-1d5b5f2a80d7
# ╟─94d185b1-e3b7-46fe-a4ff-60c5956d86b2
# ╟─39e98f07-d1d7-4f4b-b052-77dfa4c5d322
# ╟─f0face85-80c0-4367-b5e2-a13c4598bcf7
# ╟─d4660044-8438-4502-84e1-031813e9ca25
# ╟─7ba63a0e-c205-44df-9df3-50481886c15d
# ╠═9c684b79-bf17-4ee2-a2f5-9c5ef6dcae59
# ╟─a4b9fb3a-2a77-4301-b03f-702386375513
# ╟─99c81597-09df-40a3-bc03-42fa4fbd4802
# ╟─071cdefb-bdd4-4f77-8b63-4e1408aedc6a
# ╟─47589ea7-8723-4375-9a50-4caf0909b18b
# ╟─b47e6443-c067-4c34-9b86-6f3dca8a27e9
# ╟─58a81bb8-6714-48e4-b1e2-bd8b2fe34675
# ╟─0f33a0f6-052f-49eb-803a-e108c9509568
# ╟─66890fc9-1872-4c5a-80bd-f4194428b42b
# ╟─1fb30bc5-3ddd-4d87-b5b9-9009a674ff29
# ╟─3dbfa412-dbd6-4efe-9c81-118905ad4bc0
# ╟─fb474f4d-fc4e-4d4a-b783-65de9a64499b
# ╟─eb8e0263-886a-4ad4-afd7-3174f9756e2a
# ╟─d6b5c960-e9d6-4079-b950-d7c465782b29
# ╟─6f395ff1-ffd2-46ad-b1b6-538b53a00a98
# ╟─89f8c606-c8a5-4300-9662-8b36e2537f8c
# ╟─4a87efe7-f1ec-49ba-99d4-75515879d540
# ╟─2edfa6cb-beb4-4f63-a736-7efdaf3dd07e
# ╟─f910535b-4746-4845-b5ec-7fecc9b15656
# ╟─eadb07d9-9227-4479-890a-f3bea3efc8f5
