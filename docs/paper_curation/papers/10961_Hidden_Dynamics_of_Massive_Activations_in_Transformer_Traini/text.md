# arXiv:2508.03616v2[cs.AI]24Feb2026

## Hidden Dynamics of Massive Activations in Transformer Training

### Jorge Gallego-Feliciano1, +, S. Aaron McClendon1, +, Juan Morinelli1, Stavros Zervoudakis2, and Antonios Saravanos2, *

1Aimpoint Digital Labs, Atlanta, GA, USA 2New York University, New York, NY, USA

+these authors contributed equally to this work

*please direct correspondence to: Dr. Antonios Saravanos (saravanos@nyu.edu)

### ABSTRACT

We present the first comprehensive analysis of massive activation development throughout transformer training, using the Pythia model family as our testbed, and release our full dataset publicly to support further research. Through systematic analysis of various model sizes across multiple training checkpoints, we demonstrate that massive activation emergence follows highly predictable mathematical patterns that can be accurately modeled using an exponentially-modulated logarithmic function with five key parameters. Additionally, We develop a machine learning framework to predict these mathematical parameters from architectural specifications alone, achieving high accuracy for steady-state behavior and moderate accuracy for emergence timing and magnitude. These findings enable architects to predict and potentially control key aspects of massive activation emergence through design choices, with significant implications for model stability, training cycle length, interpretability, and optimization. Our findings demonstrate that the emergence of massive activations is governed by model design and can be anticipated, and potentially controlled, before training begins. Code is available at https://github.com/ Aimpoint-Digital/massive-activations-fork

[Figure 1]

[Figure 2]

[Figure 3]

(c)

(a)

(b)

Figure 1. Example analysis of massive activation development in Pythia 1B. a) Top 3 activations and median for each layer before and after training, showing that MAs develop during training. b) Evolution of the top three to median activation ratios during training for two example layers. c) 5-parameter model fits the evolution of MA with an R2 > 0.99 over all layers – here shown 2 fits.

### Introduction

Transformers have become the dominant architecture for large-scale language models, particularly through decoder-only designs for generative tasks [1, 2]. A decoder-only transformer comprises a stack of layers that update a d-dimensional hidden state (the “residual stream”) via self-attention and feed-forward sublayers, coupled with residual connections and normalization for stable information flow [3]. Self-attention contextualizes token representations, while feed-forward networks apply position-wise transformations that build the high-dimensional features used for prediction.

A notable phenomenon in these models is the emergence of massive activations: individual neuron activations that exceed typical magnitudes within a layer by factors of 103–104 [4, 5]. Whereas activations are generally content-dependent, these extreme values can remain nearly constant across inputs and act as implicit bias terms that steer attention toward particular tokens. Massive activations have practical consequences for quantization [6, 7], inference optimization [7, 8, 9, 10], and training stability [11]. Prior work shows that removing them can cause model failure, while replacing them with mean values can preserve functionality [5], and selectively amplifying high-impact activations can promote Chain-of-Thought reasoning without reinforcement learning [12]. Recent analyses further trace their roots to specific architectural components [13, 14], but key questions about their emergence during training remain open.

Existing mitigation strategies largely operate after massive activations appear. Proposed approaches include modifying attention nonlinearities (e.g., Softpick, softmax-1, clipped softmax, gated attention) [15, 16, 17], training-time interventions such as MacDrop [18], and explicit outlier handling methods like DuQuant [19]. Normalization placement has also been implicated: schemes such as Peri-LayerNorm, which normalize both before and after sublayers, can stabilize activation variance and gradients relative to standard Pre-LayerNorm designs [20]. Broader evidence suggests that not all outliers are harmful and that combining strategies can suppress damaging extremes while preserving downstream performance [21]. Related phenomena appear in Vision Transformers, where high-norm “artifact” tokens can distort attention; introducing dedicated “register” tokens can absorb these activations and improve feature and attention maps [22, 23].

Despite these advances, most existing interventions remain fundamentally reactive, addressing the symptoms of massive activations only after they have emerged. Comparatively little is known about the temporal dynamics of massive activations: when they first appear, how they evolve across layers and training stages, and whether their properties can be anticipated from architectural choices. A mechanistic account of their development would clarify representation formation [24], information propagation, and implicit biases, while also informing training diagnostics and architectures optimized for quantization and efficiency [8, 25, 26, 27].

To address these gaps, this work studies massive activation development across training in the EleutherAI Pythia suite [2], comprising 9 decoder-only transformers from 14M to 12B parameters with over 150 checkpoints per model. This setting enables a controlled analysis of when and how massive activations arise, how their trajectories depend on scale, and how well they can be predicted from architectural parameters such as depth, hidden size, and attention head count. We then provide a quantitative characterization of these trajectories, introduce a unified mathematical framework for modeling massive activation emergence, and discuss implications for transformer design, interpretability, and optimization.

### Preliminary

This section establishes the mathematical framework and key definitions necessary for analyzing massive activation dynamics during transformer training. We focus on decoder-only transformer architectures, as exemplified by the Pythia model family studied in this work.

Transformer architecture and hidden states We consider decoder-only transformer models composed of L residual blocks. Each layer ℓ ∈ L receives a hidden state hℓ−1 ∈ RS×d and produces an updated hidden state:

hℓ = hℓ−1 +Fℓ(hℓ−1) (1)

where Fℓ includes both multi-head self-attention and MLP submodules. Throughout this paper, we denote by hℓ the postresidual hidden state, i.e., the output after the residual summation. As in [5], we do not consider intermediate computations

within Fℓ unless explicitly stated.

An activation refers to a specific scalar element of a hidden state tensor hℓ. For a model processing a sequence of S tokens with d hidden dimensions, each layer’s output hℓ ∈ RS×d contains S·d scalar activations. In this work, we focus exclusively on the scalar values in hℓ, rather than weights, attention logits, or intermediate MLP states.

Massive activations Following the definition introduced in [5], we refer to certain rare, abnormally large activations as massive activations (MAs). They propose a loose rule of thumb, and consider a scalar activation a ∈ hℓ to be massive if:

|a| > 100 and |a|

median(|hℓ|) ≥ 1000. (2)

These activations have been observed to occur consistently at a small set of fixed feature dimensions and are often associated with the initial token or delimiter tokens in the input sequence (e.g., “.” or “\n”). While small in number, they are disproportionately large—often exceeding the median activation by four or more orders of magnitude—and have been shown to function as implicit bias terms in the model’s computation.

The original definition in [5], which includes a hard threshold of |a| > 100, does not generalize well to smaller models. For instance, in Pythia-14M (Figure 3), the top activation magnitudes at layer 3 clearly dominate all others in the model, exhibiting the characteristic sharp spike associated with massive activations. Yet their absolute values remain well below 100, and their ratio to the median is also far less than 103.

Despite this, the behavioral pattern is qualitatively similar to that observed in larger models: a small number of activations attain disproportionately high values, persist across tokens and inputs, and concentrate in specific feature dimensions. To better capture this effect across model scales, we relax the definition and focus instead on the top largest activation, which in the case of smaller models, even without reaching the thresholds from the previous definition, exhibit the same patterns as those activations in larger models.

[Figure 4]

- Figure 2. Plot of transformer parameter count vs value of the top activation to median ratio per model, in each respective final model checkpoint.

Figure 2 illustrates the scaling relationship between model size and the maximum observed activation magnitude (averaged across samples in X). We observe a steep rise in activation magnitudes from 14M to 1B parameters, followed by a plateau and a secondary increase beyond 6B. This suggests that massive activations emerge gradually with scale and stabilize in prevalence or intensity past a certain model size.

We study a transformer, along with its training checkpoints, denoted as Mt, for 0 ≤ t ≤ T, where each Mt is a transformer model with the GPT-NeoX architecture [2]. The index t represents the training step, ranging from 0 to 143,000. EleutherAI released 154 checkpoints at regular intervals: multiples of 1000 steps for the full training duration, with additional higherresolution checkpoints at powers of 2 up to step 512 for detailed analysis of early training dynamics. Each model Mt consists of L decoder layers. Passing an input sequence x through Mt yields a series of hidden states:

hℓ(Mt,x) ∈ RS×d (3)

where ℓ ∈ {1,...,L} is the layer index, S is the sequence length, and d is the hidden dimension. For brevity, we denote activations as hℓ,t(x) := hℓ(Mt,x).

#### Computing massive activations during training

Massive activations are defined based on the top values in each layer relative to the median. To clarify, we characterize the activations being measured as the final hidden state output from each decoder layer, which represents the post-residual activations after both the self-attention and MLP (feed-forward) components have been applied. These correspond to the hℓ values in our notation and are the inputs to the subsequent layer. Let the following denote scalar quantities:

hmedianℓ,t (x) : the median value of |hℓ,t(x)| (4)

hmaxℓ,t (x) : the largest value in |hℓ,t(x)| (5)

hmaxℓ,t (x) hmedianℓ,t (x)

rℓ,t(x) :=

: ratio of the largest activation to the median (6)

Since activations depend on the input, we evaluate them over a distribution X of realistic inputs. We define X to contain natural language sentences representative of real-world usage, excluding out-of-distribution inputs that would yield unpredictable behavior. We define the expected activation over X as:

1

Hℓ,t(X ) := Ex∼X [hℓ,t(x)] and approximate it with h˜ℓ,t(X) :=

|X| ∑

hℓ,t(x) (7)

x∈X

We similarly define h˜medianl,t , h˜maxl,t , and r˜l,t.

In practice, X is a random sample of 10 sequences from the RedPajama dataset [28]. We measure the variance on 7 corresponding to our sample size, and report on precision and confidence intervals in the results section. Our practical experiments and prior work [5] support that massive activation patterns have low variance across inputs, which justifies the sample size.

Mathematical modeling of massive activation evolution To track the development of MAs over time, we construct a time series:

rℓ := r ˜ℓ,t(x) t∈T =

h ˜maxℓ,t (x) h˜medianℓ,t (x) t∈T

(8)

for each layer l. These series are smooth and exhibit consistent patterns across model sizes and layers, suggesting the potential for a generalizable predictive model. Throughout the paper, we set the variable i in rl to be 1, or, equivalently, we measure the max activation value as in Equation 6.

### Results

This section reports two main findings on MAs. First, we trace how MA magnitudes rise and fall throughout training and fit these curves with an accurate predictive model. Second, we show how key architectural choices—layer depth, hidden width, and head count—shape those trajectories, revealing design-level predictors of when and how large MAs will become.

[Figure 5]

##### Figure 3. Top activation magnitudes per layer in models Pythia-14M, Pythia-1.4B and Pythia-12B at revision step 0 and 143000, which correspond to the start and end of training. Pythia-14M reaches a top 1 to median ratio of 83, Pythia-1.4B reaches 2350, and Pythia-12B reaches 3200.

Evolution of massive activations during training We now focus on the evolution of the ratio of the top 1 activation to the median (Equation 8), a magnitude that characterizes massive activations. For convenience, throughout this section, we refer to this quantity simply as ‘massive activations’.

Massive activations are learned throughout training, as they are not present at model initialization (see Figure 3), which motivates our work in discovering exactly when and how they develop. We thus plot how they evolve during training and analyze the resulting data, see Figure 4. We discover several clear patterns:

- 1. Layer differentiation - The evolution of shallow, middle, and deep layers exhibit starkly different shapes.
- 2. Strong Predictability - Experiments show we can predict MA evolution accurately by fitting an exponentially decaying, log-modulated function, scoring an average coefficient of determination of 0.984.
- 3. Stage-wise Development - MAs often peak early on during training and then monotonically decrease from there, showcasing two clear stages.

Layer differentiation

In terms of MA magnitude, the first few shallow and last few deep layers overall have significantly smaller MAs than the middle layers. Observing Figure 5a, we note that systematically, between 1 and 3 shallow layers, and 1 or 2 deep layers have significantly different MA patterns than the rest of the layers. We observe that bigger models tend to support this pattern more strongly, with the exception of size 2.8B that has noisier data. This is seen in Figure 7b where we observe bigger models have less noisy MA trajectories that can be modeled with Equation 9 with higher confidence.

In terms of MA temporal dynamics, most model sizes and layers display smooth MA trajectories with very similar shapes, with only a reduced number of model sizes and layers displaying noisier time series. Across all model sizes, we observe two broad classes of MA trajectories, largely determined by layer depth:

- • Early peak: Shallow and deep layers exhibit a rapid rise, reach a clear maximum early in training, then decay toward an asymptote.
- • Log increase: Middle layers follow a smooth logarithmic climb with no apparent peak during the training window.

These patterns are exemplified in Figure 4, which shows layers 1, 2, 3 and 16 to follow an “early peak” pattern, with layers

##### 4 to 15 displaying a “log increase” curve. More systematically, Figure 5b shows the stark change in pattern in early and late, that peak during training vs middle layers, that peak at step 143k, which is the end of training. Middle layers peak at the end of training as they are monotonically increasing, do to their logarithmic shape. This pattern is particularly stronger for larger models, with 410M or more parameters.

[Figure 6]

- Figure 4. Evolution of the ratio of top activations to median (Equation 8) during training for Pythia 1B. It is a linear interpolation of 37 data points corresponding to different training checkpoints. Apart from the highest activation which is the focus of our study, we also plot ratios corresponding to the top 2 and 3 for comparison. The plots show the training steps on the x-axis, and the ratio of the top magnitudes to median activations on the y-axis.

Strong Predictability

Based on our observations of the MA trajectories, we sought to find a low-dimensional functional form hypothesis that could describe the two main observed dynamics: a logarithmic shape, and an initial peak with eventual decay. We evaluated a suite of low-parameter functions to eventually find the function we present in this section: an exponentially decaying, log-modulated function. The theoretical justification of this exact mathematical pattern remains an open question that our empirical findings now motivate.

f(t) = Ae−λxt log(xt)+K, where xt = γt +t0 (9)

This unified model fits both the “early peak” and “log increase” regimes across all model sizes and depths with high fidelity, by adjusting the influence of decay with the λ parameter, which can make the curve purely logarithmic when λ = 0, or decaying if λ ≫ 0.

Equation 9 forms the core of our following analysis. The fitting process is described further in the Methodology portion. It has 5 parameters: {A,λ,γ,t0,K}, where A controls the amplitude, λ the decay rate, γ the time scaling, t0 the time offset, and K the asymptotic baseline. The proposed equation was based upon the observation of the massive activation trajectories of each layer, and seeks to unify all models and layers under a single general formula. Note that for each model and each layer, these

- 5 parameters take different values. The dynamics of the MA trajectories are very similar across models and layers, but their magnitudes and curves vary from model to model.

Model size 14M 70M 160M 410M 1B 1.4B 2.8B 6.9B 12B R2 0.9307 0.9681 0.9831 0.9937 0.9922 0.9956 0.9686 0.9954 0.9829

- Table 1. Average layer-wise best R2 scores — quality of fit of Equation 9 to the time series in Equation 8 — for each Pythia model size.

Our proposed log-modulated exponential model predicts the MA evolution with outstanding accuracy, achieving a mean coefficient of determination of 0.984 across 188 layers from nine model sizes1. Table 1 reports the average R2 for each model.

1Average taken over all fitted layers.

[Figure 7]

(a) Middle-depth layers display significantly higher MAs than shallow and deep layers.

[Figure 8]

(b) Note the very stark change from shallow and deep layers to middle ones, particularly in the bigger (>410M) models.

- Figure 5. Heatmaps showing the location and magnitude of peak MAs by layer depth and model size. Training ends at 143k for the Pythia family, so the yellow middle layers in 5b show that MAs would continue to rise monotonically if training continued, where as the darker layers, generally shallow and deep layers, peak and start decreasing before training ends.

[Figure 9]

- Figure 6. Example fits for two model sizes: 1B and 6.9B and an example shallow, middle and deep layer from each. The plot shows the best fit for Equation 9, and data points corresponding to Equation 8. The last training step for the Pythia family is 143k.

Notably, smaller models (14M & 70M) already reach R2 > 0.93, while larger sizes (160M and above) are generally above R2 = 0.98, indicating that the MA pattern becomes even more pronounced and regular as model size grows.

Figure 6 showcases representative fits for shallow, middle, and deep layers in both Pythia 1B and Pythia 6.9B, illustrating how the same exponentially decaying log function captures the full range of training dynamics. In turn, Figure 7b breaks down the full layer-wise R2 distribution across all nine sizes, revealing that—even at the extremes—no layer falls below R2 ≈ 0.93, and most cluster tightly around R2 ≈ 0.99. This robustness underlines the universality of our fitted form across depth and scale.

The strength of these results is two-fold. Firstly, the magnitude we are modeling is a result of LLM training - a very noisy process - so an average R2 value greater than 0.98 is surprising. Secondly, the curve hypothesis is a very simple 5-parameter model, which is able to generalize to 9 model sizes and 188 layers.

The activations we are modeling vary with respect to different input sequences, which is why we construct our data points from an average over 10 input sequences. We report on the noise calculated over the 10 data points, and conclude the sample size is sufficient, giving us an average standard error of the mean percentage of 2 points.

Stage-wise Development

We have found evidence that in early and late layers across all model sizes, MAs quickly develop in the first 60k steps before starting to monotonically decrease. This occurs in every model and layer that is colored anything but yellow in Figure 5b.

[Figure 10]

(a) Example R2 values for various models.

[Figure 11]

(b) Full map of all models and all layers and the strength or ease of fit. A greener color means that massive activation trajectories in that coordinate can be modeled with high confidence with Equation 9. Even the lower scoring locations still show evidence of reasonable fits.

Figure 7. Coefficients of Determination for the MA trajectory fits.

- Table 2. Report of various statistical metrics to quantify variability in the activation ratios we measure, calculated over the 10 samples that were used as input sequences. The metrics are Coefficient of Variation percentage, Standard Error of the Mean percentage, and 95% Confidence Intervals.

Model CV% SEM% 95% CI ±% Pythia-14M 6.95 2.20 4.97 Pythia-70M 7.00 2.21 5.00 Pythia-160M 6.98 2.21 4.99 Pythia-410M 4.64 1.47 3.32 Pythia-1B 6.57 2.08 4.70

- Pythia-1.4B 5.19 1.64 3.71
- Pythia-2.8B 6.46 2.04 4.62 Pythia-6.9B 5.78 1.83 4.13 Pythia-12B 4.22 1.34 3.02

To the best of our knowledge, this is the first time this phenomenon has been recorded, and it opens up an exciting path of deeper understanding of LLMs. The critical point at which MAs start decreasing suggests there is an underlying two-stage development process governing LLM learning that is poorly understood. Proof of the existence and uniqueness of this critical point is discussed in the next section.

We hope this discovery motivates future work to understand these developmental stages in the field of mechanistic interpretability focusing on training-dynamics.

#### Predicting Massive Activation Trajectories from Transformer Architecture

Parameters within Equation 9 are highly influential in the overall shape and size of the curve of massive activation ratios (Equation 8). We now analyze the mathematical behavior of the equation itself. Let us look at steady-state behavior of the equation. In the limit of t −→ ∞, the equation reduces via L’Hôpital’s rule for indeterminate forms:

Ae−λ(γt+t0)log(γt +t0)+K = K (10)

lim

f(t) = lim

t→∞

t→∞

So the parameter K can be seen to be related to the final steady state value of the ratio. Indeed, in the limit as train steps go to infinity it mathematically is the steady state value, however in practice it is also impacted by the value of the exponential term. Clearly the A parameter affects the overall height of the peak, but let’s investigate where the peak occurs. We can do this by

first taking the derivative of Equation 9:

d dt

f(t) = Aγe−λ(γt+t0) 1

γt +t0 −λ log(γt +t0) (11) Setting this derivative to zero gives us:

1 γt +t0

= λ log(γt +t0) (12)

Rearranging this critical point equation yields 1 = λ(γt +t0)log(γt +t0), which can be solved exactly using the Lambert W function. The solution is:

eW(1/λ) −t0 γ

tpeak =

(13)

where W is the Lambert W function. This analytical solution reveals several key insights about peak behavior. For any positive decay rate λ > 0, a real-valued critical point always exists since W(1/λ) is defined for all positive arguments. However, an

observable peak during training requires tpeak > 0, which imposes the constraint λW(11/λ) > t0. When this condition is not satisfied, the peak occurs before training begins (t < 0), and only monotonic decay is observed. Second, the peak location

scales inversely with γ (smaller γ shifts the peak to much larger training steps) and is offset by t0. This mathematical analysis suggests that γ and λ are the most critical parameters for controlling curve shape and location, with γ having particularly strong influence on peak timing (shown in Figure 8) due to its position in the denominator. Figure 8 illustrates the relationship between the number of training steps needed to see a peak in Equation 8 and parameters γ and λ. Note that the Pythia model family did not reach training steps greater than ≈ 143k, so for any fitted equation parameters that predict a peak at location greater than the maximum number of training steps, a peak would not have been seen for a particular layer during the training cycle. As seen in our previous results section, certain layers within each model exhibit monotonically increasing ratio (8) behavior as training progresses, and some layers exhibit a sharp peak early in training and then decay to a steady state value (see Figure 6). Overall, the layer depth (along with training length) is correlated with peak observation, but we notice that there are breaks in this pattern. For example, the layers (or more specifically, the depth of a layer within a model relative to its overall size) in which quantity 8 exhibits a peak is different for Pythia 12B vs Pythia 1B.

[Figure 12]

Figure 8. Surface plot illustrating peak location tpeak as a function of parameters λ and γ from Equation 13, with t0 fixed at a typical value found across the Pythia model family. Observable peaks require tpeak > 0, i.e., λW(11/λ) > t0; regions violating this constraint are masked. The inverse dependence on γ means small changes in this parameter produce large shifts in peak timing.

The predictability analysis presented below uncovers a relationship between transformer architecture and massive activation dynamics. We demonstrate that the complex, seemingly chaotic emergence of massive activations during training is actually governed by predictable mathematical relationships that can be controlled through architectural design. Specifically, we show how architectural parameters play a role in whether peaks in Equation 8 will emerge, their magnitude, and their steady-state behavior—enabling practitioners to architect models with desired massive activation properties from the outset for both further study of massive activations and potentially optimized training dynamics. For instance, controlling γ could theoretically allow peaks to occur sooner in the training cycle, allowing steady states to be reached sooner. However, the relationship between massive activation timing and overall training efficiency requires further investigation. Note that the λ parameter does effectively control whether a peak exists, but in practice within the Pythia model family, λ always takes a value which allows peak existence, therefore γ or t0 would play the largest roles in peak observation.

- Table 3. Machine Learning Model Performance for Predicting Massive Activation Parameters. Values show test set R2 scores. Best performing model for each parameter is shown in bold. Negative R2 values indicate worse-than-baseline performance. Dataset: 188 samples (80% train, 20% test), 5-fold cross-validation.

#### Parameter Transform Ridge Lasso Random Forest Gradient Boosting XGBoost

A (Amplitude) log1p 0.077 0.195 0.476 0.274 0.244 λ (Peak Occurrence) log1p 0.031 -0.015 0.643 0.506 0.664 γ (Peak Location) log1p 0.056 -0.006 0.055 -1.571 -0.089 t0 (Time Offset) log1p 0.100 0.017 0.447 0.266 0.387 K (Asymptotic Baseline) yeo-johnson 0.405 0.316 0.803 0.824 0.847

The fitted models in the predictability analysis showed strong fit characteristics for certain variables (see Table 3), such as λ, and K, while the variables A, γ and t0 showed a slightly weaker performance across all models. Table 3 shows the performance of various standard machine learning algorithms in predicting the values of parameters in Equation 9 using only features constructed using architectural specifications for various models within the Pythia family. In most cases we fit the models to transformed features due to large outliers which can easily skew predictions for most model types. Additionally, target variables were isolated such that other variables in the equation were not used as predictors to help isolate the effects of the architecture from the fitted equation. Below, we offer an in-depth analysis of key features and their fits, as well as key explainability measures the top models (shown in bold in Table 3) were able to determine using a SHAP and PDP feature analysis. This interpretability analysis gives us a way to control the value of the behaviors of our MA ratio (Equation 8) using purely architectural design patterns. We focus on parameters λ and K due to their strong fits.

A and t0 form amplitudes and fixed offsets of the peak and are less impactful and have lower R2 scores, so we omit their discussion, along with γ due to its lower performance, for brevity. Table 4 provides interpretable definitions of various features used in the predictive ML models. Note that raw features are not used to avoid having direct dependency on specific models with the Pythia family; we primarily focus on normalized features to increase applicability more widely across the model family, in general. We also note that the architectural design choices within the Pythia model family do not have high degrees of variation, and often a particular feature scales proportionally to model size. So machine learning models fit using this architectural distribution could have trouble generalizing.

Parameter K

Parameter K represents the steady-state value of massive activation ratios in the limit as training steps approach infinity, making it a critical architectural design target. The analysis reveals that this asymptotic behavior is highly predictable from architectural choices, with the model achieving an R2 of 0.8186 on the test set.

The SHAP feature importance analysis identifies attention density (attention heads per hidden dimension) as the dominant architectural control for steady-state behavior, followed by the layer depth interaction term and layer position (as seen in Figure 9a). This hierarchy indicates that attention architecture design has the strongest influence on long-term activation dynamics, while layer-specific effects provide secondary modulation.

- Table 4. Key Architectural Features Used in Predictive Models. Note: ℓ = layer index, L = total layers, d = hidden dimension, H = attention heads, df f = intermediate (MLP) dimension.

Feature Name Formula Interpretation Layer Position ℓ/L Relative depth within model (0 = first layer,

1 = last layer) Layer Position² (ℓ/L)2 Quadratic position effects for non-linear layer behavior Layer Position³ (ℓ/L)3 Cubic position effects capturing complex depth dependencies Layer Position1/2 ℓ/L Square root position effects for early-layer

emphasis Attn. Heads/Hidden Size H/d Number of attention heads per hidden dimension Intermediate Ratio df f/d MLP expansion factor (in the Pythia family this a fixed value of 4x) Width/Depth Ratio d/L Model width relative to depth (architectural shape) Attn. Heads/Num. Layers H/L Attention head budget distributed across layers log(Hidden Size) log(d) Logarithm of hidden dimension (proxy for model size) Layer DepthxModel Depth ℓL Interaction between layer location and total length

[Figure 13]

(a) SHAP Summary Plot for Parameter K revealing directional relationships between architectural features and steady-state predictions. Colors indicate feature values (red = high, blue = low).

[Figure 14]

(b) SHAP Waterfall Plot for highest Parameter K prediction

showing individual feature contributions. Predicted value f(x) = 1.811 in transformed space corresponds to approximately 5.1 in original scale.

#### Figure 9

The SHAP summary plot in Figure 9a reveals the directional relationships governing steady-state control. Most notably, decreasing attention density—either by reducing the number of attention heads or increasing the hidden dimension—consistently increases Parameter K, leading to higher steady-state massive activation ratios. This relationship suggests that models with fewer, larger attention heads will exhibit elevated baseline activation levels in the long term compared to models with many smaller heads. This finding has practical implications for model architecture design: decisions about head count and hidden

dimension size do not merely affect computational efficiency or representational capacity, but also systematically influence the long-term magnitude of massive activations within the network.

The waterfall plot in Figure 9b demonstrates these effects in practice, showing how a high Parameter K prediction results from specific architectural choices. The largest contribution (+0.81) comes from the layer depth interaction term, indicating this is a deep layer within a deep model. Additional positive contributions from low attention density (+0.33) and high width/depth ratio (+0.45) further elevate the steady-state prediction, while early layer position (+0.01) provides additional upward pressure.

These findings establish that transformer architects can systematically control steady-state massive activation behavior through attention architecture design, with attention density serving as the primary control mechanism and layer depth providing amplification effects for deeper models.

Parameter λ

Lastly, we provide analysis for parameter λ, which influences whether peaks are observable during training. For any λ > 0, a mathematical peak exists, but observability requires tpeak > 0, i.e., λW(11/λ) > t0. Since larger λ values decrease the critical point xt∗ = λW(11/λ), high λ makes it more likely that the peak occurs before training begins (tpeak < 0), rendering it unobservable.

[Figure 15]

Figure 10. SHAP summary plot for parameter λ, indicating directional relationships between architectural choices and values of λ predicted by the top performing machine learning model.

The SHAP analysis in Figure 10 reveals that layer depth characteristics dominate λ predictions, with the layer depth interaction term showing the strongest and most consistent effects. Deeper layers within deeper models (high feature values, red points) consistently push λ toward higher values, effectively suppressing peak behavior. This relationship suggests that peak occurrence follows a systematic architectural pattern, with early layers in shallow models most likely to exhibit peaks, while deeper layers in larger models tend toward monotonic decay. This is exactly what is seen, for example, in 4.

These findings establish a clear architectural hierarchy for controlling peak occurrence: layer depth and model shape provide the primary controls, while attention architecture offers secondary modulation. The high predictive accuracy suggests that architects can systematically design models to either encourage or suppress massive activation peaks across different layers, providing a new dimension of control over internal model dynamics during training.

### Discussion

This study presents a new framework for understanding the emergence and dynamics of MAs in transformer models. Our core contributions can be summarized as follows. First, we introduce a five-parameter, exponentially-decaying log-modulated

function that accurately models how the top-magnitude activations evolve throughout training within each transformer layer. Each parameter: amplitude (A), decay rate (λ), time scaling (γ), time offset (t0), and asymptotic baseline (K), captures an interpretable aspect of the activation emergence curve. This mathematical model provides, for the first time, a quantitative and interpretable description of the temporal development of MAs across model scales and layers.

Second, we develop a machine learning-based predictive framework that can forecast the values of these five parameters solely from architectural features such as the number of layers, hidden dimension size, attention head count, and layer position. By leveraging tree-based ensemble models (Random Forest [29], XGBoost [30]) and modern explainability techniques (SHAP values [31], partial dependence plots [32]), we show that it is possible to predict, with high accuracy, key aspects of massive activation dynamics—notably the steady-state baseline K (R2 = 0.847) and curvature features λ —based purely on a model’s static architecture. This demonstrates that MAs are not random artifacts of training, but follow systematic, architecture-dependent rules.

Third, our interpretability analyses reveal that specific architectural choices serve as master controls for MA dynamics. In particular, the ratio of attention heads to hidden dimension (“Attn. Heads/Hidden Size") and layer position (“Attn. Heads/Num. Layers") within the network are the dominant drivers of MA emergence and amplitude. We identify that changing these ratios within a transformer architecture, for example, changing “Attn. Heads/Layer" from 0.7 to 1.0, can have direct impact in controlling the shape of the development of MA ratio curves during the training cycle, and change the distribution of which layers have a visible peak and which do not. This non-monotonic relationship suggests that careful tuning of attention architecture offers a concrete handle for modulating high-norm activation behavior.

Taken together, these findings establish that the emergence of massive activations is a predictable, quantifiable, and interpretable phenomenon rooted in architectural design—not merely a quirk of optimization or dataset. MAs appear abruptly during training, often at specific layers, hidden dimensions, and token positions (e.g., beginning-of-sequence or delimiter tokens [5, 21]), and then stabilize to input-agnostic, nearly constant values that act as implicit bias terms within the network.

This new understanding of MA dynamics opens several avenues for both theory and practice. For model designers, our predictive framework enables MA-aware architecture by providing precise control over when and where massive activations emerge. Rather than treating MAs as unpredictable training artifacts, architects can now systematically design models to either suppress or accentuate MAs as desired. For example, our findings show that adjusting attention density can create sharp phase transitions in steady-state behavior, while modifying width/depth ratios provides coordinated control over both peak timing. This offers a principled way to navigate architectural tradeoffs while maintaining desired MA properties.

While our study is comprehensive within the EleutherAI Pythia family of decoder-only transformers, several limitations remain. Our results may not fully generalize to encoder-based models (see BERT, [33]), sequence-to-sequence architectures (see orginal transformer variant outlined in [4]), or other architectures such as LLaMA-based models [34]. The temporal resolution of our checkpoint data means we capture the emergence of MAs at coarse granularity; more frequent checkpoints might uncover finer dynamics or more gradual onset. Our input sampling for activation measurement was necessarily limited for computational reasons; broader input distributions may reveal additional or rare MAs not captured here. Finally, while our framework robustly predicts several parameters (especially K, λ), timing-related parameters (γ and t0), remain harder to predict, suggesting that some aspects of MA emergence depend on optimization dynamics or data ordering not captured in architecture alone.

These limitations point toward several promising directions for future research. First, our observation that some layers require extended training beyond 143k steps to reach their predicted peaks raises intriguing questions about the relationship between MA dynamics and grokking phenomena [35, 36]. Since grokking often occurs after hundreds of thousands of training steps, future work could investigate whether this extended training provides sufficient time for “slow-peaking" layers to complete their internal reorganization, potentially revealing MA peak timing as a predictor or correlate of delayed learning transitions. Additionally, our predictive framework suggests the possibility of designing quantization-aware architectures that intentionally delay MA peak emergence well beyond typical training horizons. Since many applications require quantization for deployment, architectures that can maintain performance while keeping MAs suppressed during standard training durations could offer significant practical advantages for efficient inference.

Second, validation across diverse model families would strengthen the generalizability of our framework. While Pythia provides an excellent controlled environment, expanding to encoder-decoder architectures [4], different training objectives [37], and alternative attention mechanisms [38, 39] would test the universality of our architectural control principles. Of particular interest would be investigating whether our predictive relationships hold across models with different positional encoding schemes [40, 41], normalization strategies [42, 43] and non-normalization strategies [44], or activation functions.

Third, our current analysis is constrained by the limited architectural diversity within existing model families. For example, Pythia models consistently use a 4× MLP expansion ratio, making it impossible to predict how variations in this parameter affect MA dynamics. Future work could involve training custom model families with systematic variations in currently fixed ratios—such as MLP expansion factors ranging from 2× to 8×, or models with heterogeneous attention head configurations

across layers. Such experiments would provide the architectural diversity needed to validate and extend our predictive framework to the full space of transformer design choices.

In summary, our results provide the first quantitative, predictive, and interpretable model of massive activation emergence in transformers, with immediate implications for theory, design, and deployment. We hope these findings will serve as a foundation for future MA-aware model development and inspire new techniques for harnessing or controlling this fundamental phenomenon.

### Methods

In this section, we outline the experimental setup used to both fit a mathematical model to the evolution of massive activations during a training cycle, and also to detail the framework used to extract explainable predictive insights from the model.

#### Experimental setup

MAs were estimated by setting X in Equation 7 to be a sample of 10 random sequences from the Red Pajama dataset [45]. This dataset represents data from the training distribution, or more generally, from the target distribution. For each model, we elicited MAs for at least 37 regularly spaced steps.

Parameters for Equation 9 were estimated with SciPy 1.15.2 (scipy.optimize.curve_fit; [46]) using the TrustRegion Reflective algorithm [47], with analytic Jacobians supplied and bounds enforced on the λ parameter to keep it positive, as negative values produce an exponential curve, much different from the observed trajectories. The curve_fit algorithm seeks to find a set of parameters that minimize the fit error, and does so iteratively. To speed up convergence, we provide an analytic Jacobian of our function, and an initial guess. Data points are normalized first, fitted in the normalized space, and then the parameters are scaled back. Each model and layer had a minimum of 27 data points, corresponding to regularly spaced training checkpoints.

For our initial guess, we set A to the response range (maxy−miny) and K to the midrange (miny+A/2), for an appropriate the vertical scale and offset of initial curve. We initialize λ = 1/std(x) and γ = 1/range(x) to keep the effective input (γx+t0) and decay rate on an O(1) scale, which improves conditioning and avoids overly flat or overly stiff starting dynamics. We use t0 = 1 as a safe positive shift so log(γx+t0) is well-defined at initialization.

We reparametrized Eq. (9) to reduce parameter coupling and improve numerical stability under the Trust-Region Reflective solver as

f(t) = e−βt A1log t +τ0 +A2 +K, (14) with the parameter mapping

t0 γ

, A1 = A,e−λt0, A2 = A1log(γ).

β = λγ, τ0 =

This follows from xt = γt +t0 = γ(t +τ0), so that log(xt) = log(γ)+log(t +τ0). Other hypothesis were tested, such as first- and second-order step response functions:

First-order step response

y(t) = K 1−e−t/τ . (15)

Second-order step response

#### Underdamped (0 ≤ ζ < 1):

1 1−ζ2

e−ζωnt sin(ωdt +arccos(ζ)) , ωd = ωn 1−ζ2. (16)

y(t) = K 1−

#### Critically damped (ζ = 1):

y(t) = K 1−e−ωnt (1+ωnt) . (17) Overdamped (ζ > 1):

y(t) = K 1−

r1 r1 −r2

r2 r2 −r1

er1t +

er2t , r1,2 = −ωn ζ ∓ ζ2 −1 . (18)

Model hypothesis 14M 70M 160M 410M 1B 1.4B 2.8B 6.9B 12B

Original 0.9951 0.9829 0.8068 0.9832 0.9922 0.9666 0.9928 0.9947 0.9659 Reparametrized 0.9956 0.9825 0.9307 0.9831 0.9922 0.9630 0.9936 0.9954 0.9681

- Step 1 0.8008 0.6177 -0.6833 0.2592 0.6183 0.3134 0.8202 0.8774 -0.3587
- Step 2 0.9754 0.9682 0.6316 0.9071 0.9760 0.9619 0.9816 0.9774 0.8031

- Table 5. Average R2 over all layers for each Pythia size, for the different hypothesis we tested to model MA evolution behavior.

These were discarded due to lower R2 and Akaike Information Criterion (AIC) score [48]. We opted for AIC score because we sought to compare not only accuracy of the models, but also simplicity, when comparing the 5 parameter hypothesis in Equation 9, versus the 3 parameter second degree step function. The strength of performance from Equation 9 compensated the additional parameters, as we can see in Table 5, which compares the average R2 score per hypothesis. Original and Reparametrized represent the same hypothesis, and strongly dominate over the other choices.

To facilitate further analysis and reproducibility, all processed activation statistics and fitted parameter values from this stage are included in our released dataset (see Data Availability).

#### Parameter analysis and architectural relationships

After fitting our mathematical models to the temporal evolution data, we investigate how the learned parameters relate to architectural properties of the transformer models. This predictive analysis enables us to understand which architectural design choices most strongly influence massive activation emergence patterns, and more importantly, provides us with a way to directly control the training dynamics purely through initial model architectural choices.

Feature engineering and data preparation

We construct a comprehensive feature set from the architectural specifications of each Pythia model variant, including:

- • Core architectural features: hidden dimension size (d), number of layers (L), number of attention heads, feed-forward network width, rotary embedding base, and rotary percentage
- • Derived features: layer position normalized by total depth, ratios between architectural components (e.g., attention heads per hidden dimension), and interaction terms
- • Polynomial features: quadratic and cubic terms for layer position to capture non-linear positional effects
- • Logarithmic transformations: log-scaled versions of large architectural parameters to handle wide dynamic ranges

Given the diverse scales and distributions of our fitted parameters, we apply appropriate transformations to improve model performance. For parameters with positive values and high skewness, we use log transformations. For parameters with mixed signs or extreme outliers, we employ a Yeo-Johnson [49] power transformation to achieve approximate normality. Features are transformed with a standard scaler to allow usage across a range of model types; plots showing relationships between targets and features, therefore, often represent the relationship between the transformed target and transformed feature.

Machine learning model selection

We evaluate multiple regression algorithms to identify the best predictor for each fitted parameter:

- • Linear models: Ridge and Lasso regression with L2 and L1 regularization respectively
- • Tree-based ensembles: Random Forest and Gradient Boosting regressors to capture non-linear relationships
- • Advanced boosting: XGBoost with careful hyperparameter tuning for optimal performance

Model selection is based on 5-fold cross-validation performance, with the final evaluation conducted on a held-out test set (20% of data). We use coefficient of determination (R2) as our primary metric, supplemented by mean absolute error (MAE) and root mean squared error (RMSE).

Model interpretability and feature importance

To understand which architectural factors most strongly influence massive activation dynamics, we employ multiple interpretability techniques:

- • Feature importance analysis: For tree-based models, we extract built-in feature importance scores.
- • SHAP (SHapley Additive exPlanations) analysis: Provides model-agnostic explanations of individual predictions and global feature importance.
- • Partial dependence plots: Visualize the marginal effect of individual features on predicted parameter values
- • Residual analysis: Examine prediction errors to identify potential model limitations or data quality issues.

This comprehensive analysis enables us to develop predictive relationships that can forecast massive activation emergence patterns based solely on architectural specifications, providing insights into how design choices influence these critical phenomena during training.

### Data Availability

The complete processed dataset, including all fitted parameters, statistics, and figures, is publicly released under the MIT License at https://huggingface.co/datasets/Aimpoint-Digital/pythia-massive-activations.

In addition to our processed dataset, the following publicly available datasets and models were used to generate the results in this study:

- • Pythia Scaling Suite – An EleutherAI collection, available at: https://huggingface.co/collections/ EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1
- • Red Pajamas – Available at: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample

### Ethical Approval

This study was reviewed by the Institutional Review Board of New York University and was determined to be exempt from further review (IRB protocol number: IRB-FY2025-10500).

### Funding

This work was not supported by any specific funding.

### References

- 1. Brown, T. et al. Language models are few-shot learners. Adv. neural information processing systems 33, 1877–1901

(2020).

- 2. Biderman, S. et al. Pythia: A suite for analyzing large language models across training and scaling (2023). 2304.01373.
- 3. Zhang, A., Lipton, Z. C., Li, M. & Smola, A. J. Dive into deep learning (Cambridge University Press, 2023).
- 4. Vaswani, A. et al. Attention is all you need. Adv. neural information processing systems 6000–6010 (2017).
- 5. Sun, M., Chen, X., Kolter, J. Z. & Liu, Z. Massive activations in large language models. In First Conference on Language Modeling (2024). ArXiv preprint: https://arxiv.org/abs/2402.17762.
- 6. Nrusimha, A. et al. Mitigating the impact of outlier channels for language model quantization with activation regularization. arXiv preprint arXiv:2404.03605 (2024). Applies kurtosis-based regularization to mitigate activation outliers for W4A4 quantization.
- 7. Dettmers, T., Thoppilan, R. et al. Gptq and llm.int8(): 1-bit quantization for large language models. NeurIPS (2022). Isolates activation outliers into high-precision for 8-bit inference.
- 8. Ma, C. et al. First activations matter: Training-free methods for dynamic activation in large language models (2024). 2408.11393.

- 9. Szatkowski, F., Wójcik, B., Piórczy´nski, M. a. & Scardapane, S. Exploiting Activation Sparsity with Dense to Dynamic-k Mixture-of-Experts Conversion. In Globerson, A. et al. (eds.) Advances in Neural Information Processing Systems, vol. 37, 43245–43273 (Curran Associates, Inc., 2024).
- 10. Yu, M., Wang, D., Shan, Q., Reed, C. J. & Wan, A. The super weight in large language models (2025). 2411.07191.
- 11. Team, D. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 (2024). Details FP8 training for a 671B MoE model, using tile-wise scaling to handle activation outliers and maintain stability.
- 12. Zhao, Z. et al. Activation control for efficiently eliciting long chain-of-thought ability of language models (2025). Under review, 2505.17697.
- 13. An, Y., Zhao, X., Yu, T., Tang, M. & Wang, J. Systematic outliers in large language models. In The Thirteenth International Conference on Learning Representations (2025).
- 14. He, B., Noci, L., Paliotta, D., Schlag, I. & Hofmann, T. Understanding and minimising outlier features in transformer training. Adv. Neural Inf. Process. Syst. 37, 83786–83846 (2024).
- 15. Zuhri, Z. M. K., Fuadi, E. H. & Aji, A. F. Softpick: No attention sink, no massive activations with rectified softmax (2025). 2504.20966.
- 16. Kaul, P., Ma, C., Elezi, I. & Deng, J. From attention to activation: Unravelling the enigmas of large language models. arXiv preprint arXiv:2410.17174 (2024).
- 17. Bondarenko, Y., Nagel, M. & Blankevoort, T. Quantizable transformers: Removing outliers by helping attention heads do nothing. In NeurIPS (2023).
- 18. Oh, J., Shin, S. & Oh, D. House of cards: Massive weights in llms (2025). 2410.01866.
- 19. Lin, H. et al. Duquant: Distributing outliers via dual transformation makes stronger quantized llms. Adv. Neural Inf. Process. Syst. 37, 87766–87800 (2024).
- 20. Kim, J. et al. Peri-ln: Revisiting normalization layer in the transformer architecture. In Forty-second International Conference on Machine Learning (2025).
- 21. Owen, L., Chowdhury, N. R., Kumar, A. & Güra, F. A refined analysis of massive activations in llms (2025). 2503.22329.
- 22. Darcet, T., Oquab, M., Mairal, J. & Bojanowski, P. Vision transformers need registers. arXiv preprint arXiv:2309.16588

(2023).

- 23. Gan, C. et al. Unleashing diffusion transformers for visual correspondence by modulating massive activations (2025). Under Review, 2505.18584.
- 24. Xu, Y., Huang, H., Wang, Y. & Wang, H. Tracking the feature dynamics in llm training: A mechanistic study (2024). 2412.17626.
- 25. Jin, M. et al. Massive values in self-attention modules are the key to contextual knowledge understanding (2025). 2502.01563.
- 26. Yue, Y. et al. Wkvquant: Quantizing weight and key/value cache for large language models (2024). 2402.12065.
- 27. Yang, J., Kim, H. & Kim, Y. Mitigating quantization errors due to activation spikes in glu-based llms (2024). 2405.14428.
- 28. Computer, T. Redpajama: An open-source recipe to reproduce the llama training dataset (redpajama-data-1t-sample)

(2023).

- 29. Breiman, L. Random forests. Mach. learning 45, 5–32 (2001).
- 30. Chen, T. & Guestrin, C. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, 785–794 (2016).
- 31. Lundberg, S. M. & Lee, S.-I. A unified approach to interpreting model predictions. Proc. 31st Int. Conf. on Neural Inf. Process. Syst. 4768–4777 (2017).

- 32. Friedman, J. H. Greedy function approximation: a gradient boosting machine. Annals statistics 1189–1232 (2001).
- 33. Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).
- 34. Touvron, H. et al. Llama 2: Open foundation and fine-tuned chat models (2023). 2307.09288.
- 35. Power, A., Burda, Y., Edwards, H., Babuschkin, I. & Misra, V. Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv preprint arXiv:2201.02177 (2022).
- 36. Liu, Z. et al. Towards understanding grokking: An effective theory of representation learning. Adv. Neural Inf. Process. Syst. 34651–34663 (2022).
- 37. Radford, Narasimhan & Salimans, S. Improving language understanding by generative pre-training. OpenAI Tech. Rep.

(2018).

- 38. Liu, Z. et al. Swin transformer: Hierarchical vision transformer using shifted windows. In IEEE/CVF International Conference on Computer Vision (ICCV), 9992–10002 (2021).
- 39. Choromanski, K. et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794 (2020).
- 40. Su, J. et al. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568, 127063 (2024).
- 41. Press, O., Smith, N. A. & Lewis, M. Train short, test long: Attention with linear biases enables input length extrapolation. arXiv preprint arXiv:2108.12409 (2021).
- 42. Ba, J. L., Kiros, J. R. & Hinton, G. E. Layer normalization. arXiv preprint arXiv:1607.06450 (2016).
- 43. Zhang, B. & Sennrich, R. Root mean square layer normalization (2019). 1910.07467.
- 44. Zhu, J., Chen, X., He, K., LeCun, Y. & Liu, Z. Transformers without normalization (2025). CVPR 2025, 2503.10622.
- 45. Computer, T. RedPajama-Data-1T-Sample: a 1 b-token open corpus for llm pre-training (2023). Accessed 28 Jul 2025.
- 46. Virtanen, P. et al. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nat. Methods 17, 261–272, DOI: 10.1038/s41592-019-0686-2 (2020).
- 47. Coleman, T. F. & Li, Y. An interior, trust region approach for nonlinear minimization subject to bounds. SIAM J. on Optim. 6, 418–445, DOI: 10.1137/0806023 (1996).
- 48. Akaike, H. A new look at the statistical model identification. IEEE Transactions on Autom. Control. 19, 716–723, DOI: 10.1109/TAC.1974.1100705 (1974).
- 49. Yeo, I.-K. & Johnson, R. A. A new family of power transformations to improve normality or symmetry. Biometrika 87, 954–959 (2000).

