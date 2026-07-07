# arXiv:2605.20119v2[cs.LG]4Jun2026

## Toto 2.0: Time Series Forecasting Enters the Scaling Era

###### Emaad Khwaja∗,†,1, Chris Lettieri∗,1, Gerald Woo∗,†,1, Eden Belouadah1, Marc Cenac1, Guillaume Jarry‡, Enguerrand Paquin‡, Xunyi Zhao1, Viktoriya Zhukova1, Othmane Abou-Amal1, Chenghao Liu1, Ameet Talwalkar1,2, David Asker1

1Datadog AI Research, 2Carnegie Mellon University ∗Core Contributor, listed alphabetically †Correspondence: {emaad,gerald.woo}@datadoghq.com ‡Work completed during internship at Datadog

We show that time series foundation models scale: a single training recipe produces reliable forecast-quality improvements from 4m to 2.5B parameters. We release Toto 2.0, a family of five open-weights forecasting models trained under this recipe. The Toto 2.0 family sets a new state of the art on three forecasting benchmarks: BOOM, our observability benchmark; GIFT-Eval, the standard general-purpose benchmark; and the recent contamination-resistant TIME benchmark. This report describes our experimental results and details the design decisions behind Toto 2.0: its architecture and training recipe, training data, and the u-µP hyperparameter transfer pipeline. All five base checkpoints are released under Apache 2.0.

Date: June 5, 2026 Code: https://www.github.com/DataDog/toto Weights: https://www.huggingface.co/collections/Datadog/toto-20

###### BOOM GIFT-Eval

40

Chronos Bolt

13

Moirai-2

TimesFM-2.0

35

Moirai 1.1

Toto 1.0

Moirai-2

11

CRPSRank()

CRPSRank()

Chronos-2-synth

###### Xihe

30

*

9

TimesFM-2.5

TimesFM-2.5

Timer-s1

Chronos-2

7

25

FlowState-r1.1

###### Toto 1.0

TiRex

Chronos-2

5

20

5m 10m 50m 100m 500m 1B 2.5B

5m 10m 50m 100m 500m 1B 2.5B

Parameter Count

Parameter Count

Figure 1 CRPS rank vs. parameter count on BOOM (left) and GIFT-Eval (right) for top foundation models; lower is better. Toto 2.0 is the only family whose performance improves reliably with scale, with every size sitting on or near the Pareto frontier of both benchmarks. Competing model families scale unevenly, with larger versions sometimes underperforming smaller ones.

∗Xihe-ultra parameter count estimated (∼3B); not officially disclosed. †Timer-s1 is an 8.3B mixture-of-experts model (750m active).

### 1 Introduction

Over the past year, time series foundation models (TSFMs) have begun to match or exceed tuned statistical baselines across heterogeneous domains, much as BERT (Devlin et al., 2019) did for language a decade ago (BERT2S Workshop, 2025). What TSFMs have not yet replicated from NLP and vision is reliable scaling: a single recipe applied at successively larger widths and token budgets that produces predictable

returns (Radford et al., 2019; Kaplan et al., 2020).

We present Toto 2.0, a family of five open-weights forecasting models (4m, 22m, 313m, 1B, and 2.5B parameters) designed to answer a simple, open question: can TSFMs improve from scaling? Our results show they do. Every size improves on the one below it (Figure 1). Toto 2.0 takes the top spots on every benchmark we evaluated: BOOM (Cohen et al., 2025), GIFT-Eval (Aksu et al., 2024a), and TIME (Qiao et al., 2026). The family is also a generational jump from Toto 1.0: the 22m matches Toto 1.0’s quality with 7× fewer parameters, and inference is dramatically faster at long horizons. Toto 2.0 sees no public forecasting data during pretraining. It trains exclusively on Datadog observability metrics and synthetic series, yet leads the field on general-purpose benchmarks.

The remainder of this report is organized as follows.

- • Architecture and training recipe (Section 2). Toto 2.0 refines the Toto 1.0 backbone in three key aspects: contiguous patch masking (CPM) replaces autoregressive decoding to enable single-pass parallel forecasting; a quantile output head replaces the Student-T mixture of Toto 1.0 to improve stability at scale; and NorMuon replaces AdamW to better match the new loss function ((2)) used for fitting the quantile head.
- • Training data (Section 3). Unlike other leading TSFMs, we do not pretrain on any public time series data, and instead rely exclusively on a mix of Datadog’s internal observability metrics and synthetic data. Public data enters the recipe only during finetuning, where it makes up 45% of the mix (Section 5.3). This makes Toto 2.0’s public-benchmark performance a stronger test of cross-domain generalization than for models pretrained directly on public time-series corpora: the base models have never seen any public evaluation domains, yet generalize to them.
- • Hyperparameter transfer pipeline (Section 4). We built a structured search procedure that tunes hyperparameters once on a 10m proxy and transfers the same configuration to all five target sizes, modifying width, depth, and head count. The transfer is enabled by u-µP, which makes learning dynamics width-independent.
- • Results and scaling behavior (Section 5). Toto 2.0 sets a new state of the art on BOOM, GIFT-Eval, and TIME, with every size on or near the Pareto frontier. Finetuned and ensembled variants additionally top the full GIFT-Eval leaderboard outright. Inference is dramatically faster than Toto 1.0 at long horizons, and we show larger models notably produce coherent forecasts well past their training context on synthetic multi-scale signals.
- • Where TSFMs go next (Section 6). We share our view of the next set of bottlenecks and opportunities: closing the long-horizon gap with classical baselines, data curation, evaluation that tracks downstream value, and multimodality.

Releases. Model weights for all five sizes are available at https://huggingface.co/collections/Datadog/toto20, and our distributed training library is released at dd_unit_scaling under Apache 2.0.

### 2 Architecture

The Toto 2.0 backbone is largely retained from Toto 1.0 (Cohen et al., 2024): a decoder-only patched transformer whose attention layers alternate between time-axis (causal) and variate-axis (full) views of the input. The main changes include: contiguous patch masking for parallel decoding (Section 2.1), a quantile output head replacing the Student-T mixture (Section 2.2), NorMuon replacing AdamW (Section 2.3), amongst others (Section 2.4).

###### 2.1 Contiguous patch masking

Toto 2.0 (Figure 2) replaces Toto 1.0’s autoregressive decoding with contiguous patch masking, an elegant single-pass parallel scheme adapted from Auer et al. (2025). In Toto 1.0, the model fθ extends N context patches p1:N of size P one patch at a time via pˆi = fθ(p1:i−1). A H-step horizon takes K = H/P sequential

Inputs

###### Contiguous Patch Masking (CPM)

###### Robust Causal Scaler

zt = asinh((xt−µt)/σt)

##### Toto 2.0

Training

y

Robust Causal Scaler

###### Toto 2.0

t

Patch Embedding + CPM

raw xt

scaled zt

Input Residual MLP

target shifted by one patch

Variate-Time Transformer Decoder

###### Quantile Output Head

Inference

Output Residual MLP

0.9

###### Toto 2.0

0.5

Quantile Output Head

0.1

Forecasts

full horizon in one forward pass

observed context forecast horizon

- Figure 2 Toto 2.0 architecture. Left: training and inference protocol: CPM training applies variable-length contiguous masked spans to the input; at inference the horizon is filled with mask tokens and decoded in a single forward pass. Center: forward pass: a decoder-only transformer with alternating time-axis (causal) and variate-axis (full) attention, retained from Toto 1.0. The input scaler, patch projections, masking strategy, and output head are all improvements on the Toto 1.0 backbone. Right: input and output heads: a robust causal scaler (arcsinh normalization) on the input side, and a quantile output head producing nine quantile levels.

calls, which is both slow and fragile to errors compounding across the K steps. CPM addresses both: train with variable-length masked spans so the model learns to predict multiple future patches at once. Each patch carries a binary mask channel bi ∈ {0,1}P with bi,k = 1 at unobserved entries and 0 elsewhere. For CPM-masked positions M ⊆ {1, . . . , N}:

pˆi = fθ(p1:N, b1:N) i, i ∈ M, (1) with the loss (Equation (3)) averaged over all N positions. CPM pays off more with a transformer than on the xLSTM (Beck et al., 2024) it was designed for: Equation (1) is one call to fθ with a transformer, |M| on a SSM. At train time, M is sampled as random contiguous spans length c ∼ U{1:cmax} with probability p ∼ U(0, pmax). At inference, M = {N +1, . . . , N + K}. Either way, the model commits to a coherent forecast all at once, mitigating the compounding error of autoregressive decoding.

For horizon lengths where single-pass decoding may lose coherence, Toto 2.0 also supports block decoding: apply Equation (1) round by round in blocks of B patches, committing pi ← median(pˆi) and bi ← 0 for i ∈ M after each round (KV cache is reused). This incurs B − 1 more forward passes but mitigates overall drift. We find single-pass generally remains stable up to a ∼768-step horizon (on synthetic multi-scale signals). We use block decoding for the long-horizon study in Section 5.6.

Our sweeps (Section 4) found optimal settings of cmax = 16 and pmax = 0.4, versus TiRex’s cmax = 5 and pmax = 0.25, suggesting Toto 2.0 can handle longer masked spans than the recurrent schema TiRex was originally designed with.

###### 2.2 Quantile output head

- Toto 1.0 used a Student-T mixture model (SMM) to produce probabilistic forecasts. The SMM worked well at the size of Toto 1.0, but as we scaled beyond the original recipe, we encountered practical limits: the SMM becomes numerically unstable at large activations and diverges when predictions approach zero due to the variance term in its normalization. These issues surfaced during training as we pushed toward larger models and broader data mixes.
- Toto 2.0 replaces SMM with a quantile output head: for each future timestep, the model predicts nine quantile

levels at T = {0.1,0.2, . . . ,0.9}, trained with the pinball loss (Koenker and Bassett, 1978). For a target value y and predicted quantile qˆτ, the pinball loss at level τ is

ρτ(y − qˆτ) = (y − qˆτ) τ − 1[y < qˆτ] , (2) and the head loss averages over the nine levels:

1

#### |T | ∑

ρτ(y − qˆτ). (3)

Lquantile =

τ∈T

Quantile heads are now standard among leading TSFMs (Ansari et al., 2025; Google, 2025; Liu et al., 2025a) for their stability and calibration. We sort the predicted quantiles during inference to prevent crossing.

###### 2.3 Optimizer

Toto 2.0 uses NorMuon (Li et al., 2025) to optimize all matrix-shaped parameters. We argue this choice particularly well-suited to pinball training; the rest of this section develops the reasoning.

- Toto 1.0 trained with AdamW (Loshchilov and Hutter, 2019) on the negative log-likelihood (NLL) of its SMM. The pairing was natural: NLL provides smooth, magnitude-bearing gradients, and AdamW is the default optimizer for nearly all foundation models. With Toto 2.0’s switch to pinball, that pairing becomes less effective: pinball’s sign-valued gradients narrow the dynamic range over which AdamW’s variance-driven step-size mechanism operates. Differentiating Equation (2) gives

 

−τ y > qˆ,

∂ρτ(y − qˆ) ∂qˆ

(4)

= gτ =

- 0 y = qˆ,
- 1 − τ y < qˆ,



2

which takes only three values regardless of |y − qˆ|. Contrast this with the MSE gradient, ∂(y−qˆ)

∂qˆ = −2(y − qˆ), whose magnitude scales linearly with the error. Two residuals differing by an order of magnitude produce gradients differing by an order of magnitude under MSE, but identical-magnitude gradients under pinball. With sign-valued gradients, the loss provides a direction to refine the model towards, but not how wrong it is, so the optimizer has to infer step size from its own internal states.

One possible explanation for AdamW’s weaker performance in this setting comes from Balles and Hennig (2020), who decompose Adam (Kingma and Ba, 2017) into two aspects: “for each weight, the update direction is determined by the sign of stochastic gradients, whereas the update magnitude is determined by an estimate of their relative variance (vt).” Under the sign-valued gradients of Equation (4), this is the only step-size mechanism Adam has: the per-step gradient carries no magnitude information, so all per-weight scale adaptation comes from vt. Adam trains successfully in this regime, but with limited dynamic range.

Muon (Jordan et al., 2024) has emerged as the leading post-AdamW candidate for large-scale training, with roughly 2× compute-efficiency gains over AdamW in scaling-law experiments and adoption at trillionparameter scale by Moonshot AI’s Kimi K2 (Liu et al., 2025b). For a 2D weight W with matrix gradient Gt, Muon maintains a momentum buffer Bt = µBt−1 + Gt, orthogonalizes it via a Newton–Schulz iteration Ot = NS(Bt) that drives the singular values of Bt toward unity, and applies Wt ← Wt−1 − η Ot.

Muon contains no second-moment EMA, discarding Adam’s β2 variance mechanism by design. On smooth losses, this is part of what gives Muon its compute-efficiency advantage over AdamW, and is part of why the broader community has adopted it. In our pinball-loss setting, this tradeoff appears less favorable: removing the variance mechanism entirely also removes the limited step-size adaptation that remained.

Although Newton–Schulz drives the singular values of Bt toward unity, the per-row L2 norms of Ot can still vary by orders of magnitude, so a handful of neurons dominate each update. NorMuon1 balances per-neuron

1NorMuon has also been gaining traction more broadly: Andrej Karpathy’s nanochat uses it to train GPT-2 for under $100 (Karpathy, 2026).

contributions by normalizing each row of Ot against an EMA of its own squared magnitude:

vt = β2 vt−1 + (1 − β2) · mean_cols(Ot ⊙ Ot), Wt ← Wt−1 − η Ot √vt + ϵ,

(5)

where ⊙ denotes the Hadamard product, mean_cols reduces each row of Ot ⊙ Ot to its column-mean (yielding a per-row scalar), and the division and square root in the update are applied row-wise via broad-

casting. NorMuon’s row normalization, motivated by per-neuron balancing, also reinstates the β2 variance mechanism—now applied per neuron rather than per parameter. This contrasts with Adam, whose parameter-

wise vt never leaves the single weight it indexes and has no view of how weights within a neuron relate to each other.

We use NorMuon for all internal matrix-shaped parameters and AdamW for input/output projections, biases, and norms. We use Nesterov momentum and replace the standard Newton–Schulz orthogonalization with Polar Express (Amsel et al., 2026), a quintic iteration with coefficients optimized for faster convergence of the singular values to unity at low precision. Following µP++ (Ren et al., 2025), we do not apply weight decay to biases, norms, or input/output projection weights. For other parameters, we apply cautious weight decay (Chen et al., 2025), which applies decay only to parameters whose signs align with the optimizer update.

###### 2.4 Additional architectural changes Four more changes round out the redesign:

Patch size. Toto 2.0 uses a patch size of 32, down from 64 in Toto 1.0. This doubles the sequence length the transformer sees for a given input window, allowing the model to learn finer-grained representations of within-patch dynamics at the cost of longer attention computations.

Robust input normalization. Observability metrics routinely span many orders of magnitude. Request rates can move from tens to millions per second, latencies from microseconds to seconds. Toto 1.0 handled this with a novel causal normalization mechanism. Toto 2.0 enhances this by adding a robust arcsinh(z) = log z +

√z2 + 1 transformation (Ansari et al., 2025), which behaves as z for |z| ≪ 1 and as sign(z) log(2|z|) for |z| ≫ 1. The model predicts in this scaled space, and predictions are unscaled to compute the final forecast. Small fluctuations near zero are thus preserved at full resolution while large excursions are compressed logarithmically, all without discarding sign information.

Residual MLP patch projections. Toto 1.0 used linear layers for both patch embedding (mapping raw patches to model-dimension vectors) and output projection (mapping model-dimension vectors to distribution parameters). Toto 2.0 replaces both with two-layer SiLU networks with residual connections, giving the model nonlinear patch representations at both ends of the transformer.

Attention changes. We add PerDimScale (learned per-dimension query scaling, also used in TimesFM 2.5 (Google, 2025)) with 1/dk attention scaling for µP (Yang et al., 2021) compatibility. Patches with entirely missing observations are masked out of attention computation. Bias terms are enabled on attention projections but not on MLPs, and dropout is not used during training.

### 3 Training data

- Toto 2.0 trains exclusively on a mix of Datadog’s internal telemetry and synthetic data. Our larger models (313m, 1B, 2.5B) see 5.04T data points and our smaller ones (4m, 22m) see 3.40T, up from 2.36T in Toto 1.0 (Figure 3).

We made two structural changes from Toto 1.0. First, we removed all public data from pretraining. Our hyperparameter sweep (Section 4.2) found that public time series data was suboptimal at proxy model scale; the best mixtures the sweep found excluded it entirely. Public data does, however, enter the finetuning recipe

INTERNAL

SYNTHETIC

PUBLIC

- Toto 1.0

2.36T

total points

INTERNAL

2.14T

SYNTHETIC

2.90T

- Toto 2.0

5.04T

total points

1.0

2.0

10 sec 78.5%

- 60 sec 16.5% 5 min+ 5.0%

10 sec 47.1%

- 60 sec 17.6% 5 min+ 35.3%

Composition Interval

Toto 2.0 Training Data

Figure 3 Training data composition for Toto 1.0 (2.36T points) and Toto 2.0 (5.04T points for the 313m, 1B, and 2.5B;

- 3.40T points for the 4m and 22m). Left: Toto 2.0 composition shown is for the 5.04T mix used by the three largest models; the 3.40T mix used by the 4m and 22m holds the relative proportions constant. Toto 2.0 drops public data entirely; internal observability metrics roughly double, and synthetic data nearly quadruples compared to Toto 1.0. Right: sampling-interval breakdown of the internal observability portion only (2.14T points of Toto 2.0, vs. the corresponding

1.00T

0.78T

0.58T

- Toto 1.0 subset); percentages are within this subset rather than the full training mix. Toto 2.0 rebalances away from high-frequency intervals: 5m+ data rises from 5% to 35%, while 10s data drops from 78.5% to 47.1%.

of Toto 2.0 2.5B-FT, where is makes up 45% of the mix (Section 5.3). Second, we more than doubled our synthetic data using newer generation methods that produce more diverse regimes.

We also rebalanced the internal Datadog telemetry data. Toto 1.0’s mix skewed heavily toward high-frequency (10s) intervals. For Toto 2.0 we parameterized the sampling interval and overweighted longer intervals, so the model sees a more diverse, higher-signal view of the same underlying telemetry.

3.1 Observability time series from Datadog

- Toto 2.0’s real-world training data comes exclusively from Datadog’s own internal observability metrics: CPU utilization, memory usage, request latency, error rates, and similar infrastructure signals. Compared to

- Toto 1.0, the dataset is larger, draws from a broader set of data sources, and covers more recent time periods. No customer data is used at any point.

- 3.2 Synthetic data

- Toto 1.0’s synthetic training data used generic stochastic processes similar to Das et al. (2024). Toto 2.0 uses the synthetic data generation method from TempoPFN (Moroshan et al., 2025), built on the prior-data fitted network (PFN) framework (Müller et al., 2022) in which a transformer is trained on samples drawn from a hand-crafted prior. The TempoPFN prior is rich with nonstationary trends, abrupt changepoints, and long-range dependencies. The final training mix for base models is 42.5% observability data and 57.5% synthetic data, with the observability portion further split across sampling intervals as detailed in Section 4.2.

### 4 Hyperparameter transfer pipeline

Scaling models to multiple sizes lets users trade off inference cost against forecast quality, but this is only useful if each size is reliably better than the last. Achieving this kind of scaling behavior efficiently is notoriously difficult, and for TSFMs in particular it has been a recurring gap. Critical hyperparameters such as the learning rate are not stable across model widths under standard parametrization—empirically, the

Proxy Model

Target Models

u-μP

- Figure 4 u-µP makes optimal hyperparameters independent of model width. We tune parameters on a small proxy model, select the best configuration (depicted with a black outline here) and directly transfer the same configuration to any larger target model with no retuning required.

optimal learning rate can shift by an order of magnitude across width sweeps (Yang et al., 2021). The naive approach, tuning hyperparameters independently for each of the five target sizes, would be inefficient: each target model requires days of training, making a large hyperparameter search computationally expensive at that scale. To turn the architectural improvements into a reliable scaling recipe, we sought a way to transfer hyperparameters across widths. For that, we turned to u-µP (Blake et al., 2025).

u-µP combines Maximal Update Parametrization (µP) (Yang et al., 2021; Yang and Hu, 2021) with unit scaling (Blake et al., 2023) to make the optimal learning rate independent of model width. We selected the unit-scaled variant because of its simplicity and improved transfer for decoder-only models. This approach allowed us to sweep hyperparameters on a cheap 10m proxy, then transfer the configuration directly to all five target sizes (Figure 4) in a largely automated fashion. To our knowledge, this is the first application of µP to time series forecasting.

###### 4.1 The proxy model

The proxy is a 10m-parameter model (L = 12, dmodel = 256, h = 4). We chose a dmodel = 256 because Blake et al. (2025) demonstrates this as a floor to prevent optimal parameter drift. Each sweep trial trains the proxy for 30,000 steps at the same batch size used for the target models, under a warmup-stable-decay (WSD) (Hu et al., 2024) learning-rate schedule. At this scale, each training run completes in a few hours rather than days, enabling a configuration search several orders of magnitude broader than would be tractable at the target sizes.

###### 4.2 Structured hyperparameter search

Even at proxy scale, the joint search space spans 17 continuous and several categorical dimensions (∼ 1019 configurations under a modest grid discretization), making exhaustive search intractable. We split the search into four sequential rounds, each one selecting the empirical optimum for a different group of decisions on top of the previous round’s best configuration. The order follows the natural dependency chain: architecture and data shape the loss landscape, the optimizer must adapt to that landscape, and the decay schedule is tuned downstream of the optimized stable regime. All four rounds use Optuna (Akiba et al., 2019) with TreeStructured Parzen Estimator (TPE) (Watanabe, 2023) sampling, optimizing against seasonal-naive-normalized MASE and CRPS on the GIFT-Eval validation set.

- Round 1: Architecture. We swept attention normalization (PerDimScale, QK-Norm (Henry et al., 2020), or neither), how often the variate-axis attention layer appears in the layer stack, which transformer layers carry bias terms, and the contiguous-patch-masking parameters. The proxy’s twelve layers allowed clean exploration of several variate-attention cadences (every 2, 3, 4, 6, or 12 layers).

The best configuration uses PerDimScale (over QK-Norm), places the variate-axis attention layer last in the stack, and sets the contiguous-patch-masking parameters to cmax = 16 and pmax = 0.4 (longer masked spans than TiRex’s defaults).

- Round 2: Data mixture. We parameterized the training mix as a constrained probability simplex over five sources, with each lower bound set to 0 so TPE could remove a source entirely if optimal:

sweep: dd_10s: 0.0 - 1.0 # Datadog 10-second metrics dd_60s: 0.0 - 0.7 # Datadog 60-second metrics dd_long: 0.0 - 0.2 # Datadog 5+ minute metrics synthetic: 0.0 - 1.0 # TempoPFN data public: 0.0 - 0.05 # GIFT-Eval Pretrain

constraint: sum = 1.0

Upper bounds on the smaller corpora are set to cap repetition during training.

The optimal mixture excluded public data and settled at 42.5% Datadog observability data and 57.5% synthetic, with the Datadog portion split across 10s (20%), 60s (7.5%), and 5+m (15%) intervals. This is the mix used for all base models.

- Round 3: Optimizer. Starting from Round 2’s best configuration, we swept the learning rate, weight decay,

and first- and second-moment exponential decay rates (µ and β2 for NorMuon; β1 and β2 for AdamW), along with shared warmup steps and gradient clipping threshold.

The best configuration for NorMuon is η = 0.652, µ = 0.96, β2 = 0.999, weight decay = 2 × 10−8, and for AdamW is η = 0.012, β1 = 0.91, β2 = 0.972. Warmup is 6,000 steps with gradient clipping at 7.0.

- Round 4: Decay schedule. Starting from a checkpoint inside the stable portion of Round 3’s best run, we swept the length and shape (linear vs. 1-sqrt) of the learning-rate decay.

Linear decay won; the final schedule decays linearly over 10,500 steps—a short tail relative to the total training budget (1.7–2.6% of the 400,000 and 600,500 total steps in Table 1). We maintain 10,500 decay steps for all base models.

###### 4.3 Zero-shot transfer to target sizes

Scaling up is straightforward: take the proxy’s best configuration and apply it to every target size. The main architectural changes between sizes are embedding dimension dmodel, depth L, and head count h (we fix the head dimension at dhead = 64).

Under u-µP, each hidden weight is reparametrized as W = AW · w with w0 ∼ N (0,1), and updated as wt+1 = wt + CW · Φt, where Φt is the optimizer’s step direction on the gradient history. For hidden weights, the multipliers scale as AW ∝ 1/√

fan_in and CW ∝ η/√

fan_in (see Table 2 of Blake et al. (2025) for the input/output and depth-dependent variants), which makes the optimal learning rate η invariant across widths. Weight decay is selected at proxy scale and held fixed; it is not guaranteed by u-µP to transfer.

Table 1 lists the five resulting model configurations:

2The NorMuon learning rate looks large at first glance, but is in the expected range under u-µP: unit scaling absorbs the 1/√

fan_in

factor into the parametrization itself, so the user-facing η is the per-tensor update size at unit scale rather than the unnormalized step that an unconstrained optimizer would take.

Model dmodel h L Training steps Norm ε 4m 256 4 4 400,000 1 × 10−4 22m 512 8 6 400,000 1 × 10−4 313m 1024 16 24 600,500 1 × 10−4 1B 1536 24 36 600,500 5 × 10−4 2.5B 2048 32 48 600,500 5 × 10−4

Table 1 Toto 2.0 model sizes. dmodel is the embedding (hidden) dimension, h the number of attention heads, and L the depth (number of transformer blocks); the head dimension is fixed at dhead = 64 for all sizes. All five sizes train on 4,096-timestep contexts with patch size 32 and 32 variates per sample, at a global batch size of 64. The 4m and 22m converged at 400,000 steps; the larger sizes were still improving past that point and trained for 600,500.

###### 4.4 Making u-µP work in production

The upstream unit_scaling library (Graphcore Research, 2023) used for implementing u-µP targets singleGPU eager-mode. Training large models at scale often requires torch.compile, model sharding, and distributed parallelism strategies for optimal speed and memory utilization. u-µP works by attaching scaling metadata (fan_in, fan_out, scaling type) to each parameter tensor, and each of these infrastructure layers either destroys or invalidates that metadata. Through our distributed u-µP training wrapper, dd_unit_scaling, we address the following:

torch.compile compatibility. We rewrote the autograd scaling functions to eliminate graph breaks and cache distributed state before compilation.

FSDP2. FSDP2 replaces parameter tensors with DTensors, which destroys any attached metadata. We cache all µP metadata by parameter name before sharding so it survives the replacement.

Data/Tensor parallelism. All batch-dependent scale factors are computed from the global effective batch: local_batch × world_size × accumulation_steps. Loss is multiplied by world_size to undo DDP’s gradient averaging.

Sequence-length invariance. Unit-scaled attention has scale factors that depend on sequence length, which breaks KV caching (vital for production inference) since the effective length changes between decoding steps. We disable unit scaling in attention and the MLP activations. However, we still use the µP-standard 1/dk scale for scaled dot-product attention. The resulting variance mismatch between residual branches is mitigated by setting αres-attn-ratio = S/ log S, where S = context_length/patch_size, and setting αres = 0.75. We provide dd_unit_scaling to the community as an open-source, general-purpose library. We built it for Toto, but it is useful for anyone training under u-µP at scale beyond what the upstream library was designed for.

### 5 Results

We evaluate Toto 2.0 on three forecasting benchmarks: BOOM, our observability benchmark; GIFT-Eval, the standard general-purpose benchmark; and TIME, a recent contamination-resistant zero-shot benchmark constructed from fresh datasets specifically chosen to mitigate the test-set contamination that affects established benchmarks.

- Toto 2.0 sets a new state of the art on all three. Every Toto 2.0 size leads on BOOM. The three largest Toto 2.0 sizes lead foundation models on GIFT-Eval, and 2.5B-FT and Toto 2.0 FnF ensemble take the top two spots outright. On TIME, the same larger sizes take the top three spots on every metric, ahead of every external foundation model evaluated (Figure 8).

###### BOOM

###### CRPS RANK

###### CRPS

###### MASE

.70

20

.90

.881

17.5 17.6

CRPSRank()

.639 .643

CRPS()

MASE()

.60

15

.796

.80

12.3

11.9

11.3

10.7

.720 .725 .726

.50

10

.70

.447 .451

.672

.436

6.94 7.17 7.39 7.70

.427

.641

.632

.617 .624

.392

.40

5.53

.375 .377 .382

.601

5

.363

.60

3.88 3.96 4.26

.581 .582 .585

.349 .349 .351

Toto2.02.5BToto2.01BToto2.0313mToto2.022mToto1.0Toto2.04mChronos2TimesFM2.5Moirai1.1Moirai2TimesFM2.0ChronosBoltTime-MoETimers1

Toto2.02.5BToto2.01BToto2.0313mToto2.022mToto1.0Toto2.04mTimesFM2.5Chronos2Moirai2Moirai1.1TimesFM2.0ChronosBoltTimers1Time-MoE

Toto2.02.5BToto2.01BToto2.0313mToto2.022mToto1.0Toto2.04mTimesFM2.5Chronos2Moirai2Moirai1.1TimesFM2.0ChronosBoltTimers1Time-MoE

- Figure 5 BOOM results across CRPS rank, CRPS, and MASE; lower is better. All five Toto 2.0 sizes outrank every other foundation model on every metric. Toto 2.0 22m matches or beats Toto 1.0 across all three with roughly 7× fewer parameters. Toto 2.0 models are shaded in purple.

Beyond accuracy, Section 5.5 examines inference latency, where every Toto 2.0 size beats Toto 1.0 at long horizons, and Section 5.6 probes long-horizon stability, showing how larger sizes retain coherent multi-scale structure well past their training context.

Benchmark setup. All three benchmarks report results across several metrics. CRPS (Continuous Ranked Probability Score) measures the quality of a probabilistic forecast, scoring how well a predicted distribution over future values aligns with observed outcomes; it is the metric most directly relevant to production forecasting use cases. MASE (Mean Absolute Scaled Error) measures point forecast accuracy normalized against a naive seasonal baseline. Where metrics are reported as ranks, scores are averaged across all benchmark datasets to enable comparison across heterogeneous data.

We use a context length of 2,048 on BOOM and 4,096 on GIFT-Eval; TIME prescribes a per-task context length aligned with each task’s horizon, which we use as specified. Internal missing values in the context gaps are forward-filled, and the causal scaler’s location and scale are backfilled on leading patches with fewer than 8 observations. At decode time, each real-space output quantile is clamped to the observed context’s min and max, each extended by 104 times the anchor scale at the final context position.

###### 5.1 BOOM

BOOM evaluates forecasting on observability metrics like CPU utilization, memory, request latency, and error rates. These are the signals production monitoring systems care about.

Every Toto 2.0 size sits on the Pareto frontier of BOOM (Figure 5): at any given parameter count, no other foundation model produces better forecasts. The three largest sizes lead the chart with CRPS ranks of 3.88 (2.5B), 3.96 (1B), and 4.26 (313m). Behind them, the 22m at 5.53 already clears Toto 1.0 (6.94), establishing a ∼ 7× parameter-efficiency improvement over Toto 1.0 (which has 151m parameters). The 4m, at 7.17, is competitive with Toto 1.0 and Chronos-2 (7.39) despite being ∼ 38× smaller, making it a strong option for edge deployment.

###### 5.2 GIFT-Eval – foundation models

GIFT-Eval spans 97 evaluation tasks (combinations of dataset, frequency, and prediction horizon) drawn from 23 base datasets across domains like energy, retail, weather, and finance.

While most models train on a large collection of public domain data, Toto 2.0 ranks first among foundation models on GIFT-Eval (Figure 6) despite training only on synthetic and observability data (Section 3). The three largest sizes score 20.3 (2.5B), 21.1 (1B), and 21.4 (313m) on CRPS rank, with a 1.7-point gap separating the 313m from the next best foundation model, PatchTST-FM r1 (Nie et al., 2023) at 23.1. Chronos-2, a strong competitor, sits at 23.5. The 22m at 26.8 beats Toto 1.0 (35.1) by more than 8 points. On GIFT-Eval, each successive Toto 2.0 size improves over the one below it on the rank metrics.

###### GIFT-Eval

###### CRPS RANK

###### MASE RANK

CRPSRank()

MASERank()

37.6

41.4

36.5

40.1

35.1

38.3

35

35.2 35.4 35.6

35

31.5

33.9

30.3 30.6

29.5

30.6 30.6 30.8 31.5 31.9

30

28.1

29.5

30

26.5 26.7 26.8

28.2

25.6 25.8 25.8

26.3 26.4 26.5 27.0 27.0

25

23.1 23.5 23.9

25

23.1 23.3

21.1 21.4

22.2

21.0 21.3

20.3

20

20

Toto2.02.5BMigas1.0Toto2.01BGraniteFlowStater1.1Toto2.0313mChronos2PatchTST-FMr1Timers1TimesFM2.5FlowStater1.1LongSeerv1.0GranitePatchTST-FMr1TiRexReversoToto2.022mChronos2SynthXihe-ultraXihe-maxReversoSmallTTM-R3FlowState9.1MMoirai2Toto1.0Toto2.04m

Toto2.02.5BToto2.01BToto2.0313mPatchTST-FMr1Chronos2FlowStater1.1GranitePatchTST-FMr1TiRexTimesFM2.5GraniteFlowStater1.1Timers1Toto2.022mLongSeerv1.0Xihe-ultraFlowState9.1MChronos2SynthXihe-maxToto1.0Moirai2Toto2.04m

###### CRPS

###### MASE

.54

.76

.757

CRPS()

MASE()

.750

###### .524

.520

.52

.516 .517

.74

.724 .726 .726 .728

.502

.716 .717 .719 .720

.50

.72

.496 .496

.710 .711 .711

.485 .485 .487 .488 .488 .488 .490 .490 .490 .491

.693 .696 .697 .698 .699 .701 .701 .701 .703 .705 .707

.481 .483

.70

.48

.478

.476

Toto2.02.5BTimers1Migas1.0Chronos2GraniteFlowStater1.1Toto2.01BXihe-ultraFlowStater1.1Toto2.0313mTimesFM2.5PatchTST-FMr1LongSeerv1.0GranitePatchTST-FMr1Xihe-maxReversoTiRexToto2.022mChronos2SynthFlowState9.1MTTM-R3ReversoSmallMoirai2Toto1.0Toto2.04m

Toto2.02.5BToto2.01BToto2.0313mPatchTST-FMr1GranitePatchTST-FMr1Timers1FlowStater1.1Chronos2GraniteFlowStater1.1Xihe-ultraTiRexTimesFM2.5LongSeerv1.0Xihe-maxChronos2SynthToto2.022mFlowState9.1MMoirai2Toto1.0Toto2.04mTTM-R3

- Figure 6 GIFT-Eval results, filtered to foundation models only (i.e., excluding finetuned, ensemble, and agentic systems), across CRPS rank, MASE rank, CRPS, and MASE; lower is better. Toto 2.0 sizes are highlighted in purple. Toto 2.0 sizes claim the top three spots on CRPS rank; the 2.5B alone leads on MASE rank.

###### 5.3 GIFT-Eval – finetuned and ensemble models

The results in this section are not used to support the zero-shot scaling claim; they show that the Toto 2.0 base family is a strong starting point for downstream adaptation. The GIFT-Eval leaderboard includes entries for finetuned foundation models (tuned on the benchmark’s official training split), as well as agentic and ensembling methods that combine multiple foundation models. We explored both: finetuning a single model on a mix that includes the GIFT-Eval train split (Toto 2.0 2.5B-FT), and ensembling multiple models with a learned per-window weighting scheme (Toto 2.0 FnF).

Finetuning. GIFT-Eval ships with two separate public datasets, both of which we use here: GIFT-Eval Pretrain (Aksu et al., 2024c), a large companion pretraining corpus curated to not overlap with the benchmark’s evaluation datasets; and the official train splits of those evaluation datasets themselves (Aksu et al., 2024b), which we refer to as GIFT-Eval train. Only the latter places a submission in the leaderboard’s finetuned tier. We finetuned the 2.5B Toto 2.0 base model for 10,000 steps from a fully-decayed base checkpoint on a mix of these two sources plus Datadog observability data. The full mix was: GIFT-Eval Pretrain (45%), Datadog 5+minute metrics (25%), GIFT-Eval train (15%), synthetic (10%), and Datadog 10s and 60s metrics (2.5% each), with the GIFT-Eval Pretrain portion drawn from the Toto 1.0 public-data pool of GIFT-Eval Pretrain and the Chronos pretraining corpus (Ansari et al., 2024) (non-leaking). We also reduced the NorMuon and AdamW learning rates by roughly an order of magnitude from pretraining, to 0.05 and 0.001, respectively.

Ensembling. Forecasting datasets reward different model strengths: some favor strong short-horizon priors, others broad pretraining coverage Toto 2.0 FnF is an ensemble approach that picks per-window weights over a pool of ten foundation models: all five Toto 2.0 sizes plus Chronos-2 (Ansari et al., 2025), TimesFM 2.5 (Google, 2025), TiRex (Auer et al., 2025), FlowState (Graf et al., 2025), and PatchTST-FM r1 (Nie et al., 2023).

- Toto 2.0 FnF follows the FFORMA (Feature-based FORecast Model Averaging) framework (Montero-Manso

###### GIFT-Eval

###### CRPS RANK

###### MASE RANK

CRPSRank()

MASERank()

26.3 26.4

25.6

25

25

23.1 23.5 23.9

23.1 23.3

22.2

21.1 21.2 21.4

20.6 21.0 21.3 21.4 21.4

20.3

20.0

20

19.4 19.4

20

18.0

17.8 17.9

17.3

15.6

14.7 14.9 15.3

15

15

13.4 13.5 13.8

13.5

12.8

10

10

Toto2.0FnFToto2.02.5B-FTTSOrchestraDeOSAlphaCredenceSamaySynapseMoirai-AgentTimeCopilotToto2.02.5BToto2.01BToto2.0313mZooCastPatchTST-FMr1Chronos2FlowStater1.1TiRex

Toto2.0FnFToto2.02.5B-FTTSOrchestraDeOSAlphaMoirai-AgentCredenceSamayToto2.02.5BZooCast Migas1.0TimeCopilotSynapseToto2.01BToto2.0313mGranite-FlowState-r1.1Chronos2 Timers1

###### CRPS

###### MASE

- .45
- .46
- .47
- .48
- .49

.72

.487

.485 .485

.481 .482 .483 .483

CRPS()

MASE()

.478 .479

.701 .701 .701

.477

.476 .476

.696 .697 .698 .698 .699 .699

.70

.472

.691 .692 .693

.689

.468

.466

.682

.463 .463

.679

.68

.676 .677

.66

Toto2.0FnFToto2.02.5B-FTDeOSAlphaTSOrchestraCredenceMoirai-AgentToto2.02.5BToto2.01BSamay Toto2.0313mSynapse PatchTST-FMr1ZooCastTimeCopilotTimers1Chronos2FlowStater1.1

Toto2.0FnFTSOrchestraToto2.02.5B-FTDeOSAlphaMoirai-AgentCredenceZooCastTimers1Toto2.02.5BMigas1.0Chronos2SamaySynapseToto2.01BGranite-FlowState-r1.1Xihe-ultraFlowStater1.1

- Figure 7 GIFT-Eval leaderboard showing all submission types: foundation models, finetuned models, ensembles, and agentic systems together. On this leaderboard, “finetuned” is used as an umbrella term for any model that uses the GIFT-Eval training split, including ensemble and agentic systems. Our finetuned and ensemble models are highlighted in pink. The Toto 2.0 FnF ensemble ranks first on every metric (tied on raw CRPS), and the finetuned Toto 2.0 2.5B ranks second on the rank metrics and third on the raw metrics.

et al., 2020), with an XGBoost regressor (Chen and Guestrin, 2016) as the meta-learner. The regressor consumes lightweight summary features extracted from each input window – statistical moments, autocorrelation, seasonality, frequency, and horizon, extracted with the tsfeatures library (Garza et al., 2023) – and emits softmax-normalized weights over the model pool. We train one head per (frequency, horizon-term) bucket, twenty in total, to handle GIFT-Eval’s heterogeneity. We then adapt the overall weighted average (OWA) metric (Makridakis et al., 2020) for the GIFT-Eval leaderboard. For a model f in the candidate pool, and window j of a dataset, the OWA is defined as

- 1

- 2

OWAf,j =

MASEf,j MASEsNaive

CRPSf,j CRPSsNaive

+

where MASEsNaive and CRPSsNaive are computed from the seasonal naive baseline, across train windows in the dataset.

Both place at the top of the GIFT-Eval leaderboard (Figure 7): Toto 2.0 FnF ranks first on every metric (tied with TSOrchestra on raw CRPS), and the finetuned 2.5B ranks second on the rank metrics and third on the raw metrics.

But the more interesting finding is what is inside the ensemble. The meta-learner’s softmax weights reveal what each candidate actually contributes to each prediction. Averaged across all predictions, the Toto 2.0 family accounts for 39% of the assigned weight, more than any other model in the pool, ahead of Chronos-2 (32%) and more than the four remaining external models combined. The ensemble does not replace Toto 2.0; instead it confirms that, when the meta-learner is free to weight everything available to it, the learner consistently spends more on the Toto 2.0 family than on any other source.

###### TIME

###### CRPS RANK

###### MASE RANK

CRPSRank()

MASERank()

16

16

14.6

14.4 14.6

14.1

13.1

12.8 13.0

12.4

12.1

11.7

11.6

12

12

11.2

11.0

10.7

9.02 9.31

8.82

8.29

8.20

7.89

7.43 7.69

8

8

6.75

5.89 6.09 6.30

5.99 6.00

5.31 5.53

5.04 5.09

3.43 3.51 3.86 4.03 4.38

3.54 3.65 3.71 3.98 4.09

4

4

Toto2.02.5BToto2.0313mToto2.01BPatchTST-FMTsMix10MChronos2PatchTST-FMR1Toto2.022mTimesFM2.5TimerS1Toto2.04mTiRex EidosToto1.0Moirai2TimesFM2.0ChronosBoltVisionTS-PPBaseMoiraiBaseKairosSundialBaseTimesFM1.0

Toto2.02.5BToto2.0313mPatchTST-FMTsMix10MToto2.01BChronos2TimesFM2.5PatchTST-FMR1TimerS1Toto2.022mToto2.04mTiRex EidosToto1.0Moirai2TimesFM2.0ChronosBoltSundialBaseVisionTS-PPBaseKairos TimesFM1.0MoiraiBase

###### CRPS

###### MASE

.70

.679

.823

CRPS()

MASE()

.664

.803

.654 .657

.80

.787

.65

.635

.755

.75

.612 .613

.738

.731

.60

.717

.696 .698 .701

.574 .576 .579 .580

.70

.689

.565 .565

.681

.667 .668 .669 .674

.549 .549 .554 .556 .559

.660 .660

.55

.532 .535 .537

.65

.640 .642 .643

Toto2.02.5BToto2.0313mPatchTST-FMTsMix10MToto2.01BChronos2Toto2.022mPatchTST-FMR1TimesFM2.5TimerS1Toto2.04mTiRexToto1.0EidosMoirai2ChronosBoltTimesFM2.0SundialBaseKairosVisionTS-PPBaseMoiraiBaseTimesFM1.0

Toto2.02.5BToto2.0313mPatchTST-FMTsMix10MToto2.01BChronos2TimesFM2.5Toto2.022mPatchTST-FMR1TimerS1Toto2.04mTiRexToto1.0EidosMoirai2TimesFM2.0ChronosBoltSundialBaseKairosVisionTS-PPBaseTimesFM1.0MoiraiBase

- Figure 8 Results on TIME across CRPS rank, MASE rank, CRPS, and MASE; lower is better. Toto 2.0 sizes are highlighted in purple. Toto 2.0 sizes take the top three slots on every metric. The 2.5B leads on CRPS rank, MASE rank, and CRPS; the 313m leads on MASE and edges out the 1B on the two rank metrics, the one place the family departs from the otherwise clear size-vs.-quality trend.

###### 5.4 TIME

We additionally evaluate on TIME (Qiao et al., 2026), comprising 98 forecasting tasks drawn from 50 “fresh” (never/rarely been explored by existing TSF benchmarks) datasets curated under a human-inthe-loop pipeline, with horizons aligned to real-world operational requirements rather than mechanical short/medium/long buckets. The benchmark deliberately avoids legacy datasets such as ETTh1, Electricity, Traffic, and Weather that have circulated through TSFM pretraining corpora for years, replacing them with recent data unlikely to have been seen during pretraining.

Toto 2.0 takes the top three spots on every TIME metric (Figure 8). The 2.5B leads on CRPS rank (3.43), MASE rank (3.54), and CRPS (0.532). The strongest external foundation models, Chronos-2 (Ansari et al., 2025) and PatchTST-FM r1 (Nie et al., 2023), trail the Toto 2.0 top three on every metric, with Chronos-2 fourth on CRPS rank (4.03) and PatchTST-FM r1 fifth (5.04). Scaling on TIME is not strictly monotonic within the Toto 2.0 family: the 313m leads on MASE and edges out the 1B on both rank metrics—the only point at which the family departs from a clear size-vs.-quality trend (Figure 8). Every Toto 2.0 size, including the 4m, still outperforms Toto 1.0.

###### 5.5 Inference latency

CPM does not just improve forecast quality; it makes Toto 2.0 dramatically faster. The two decoding modes, single-pass and block decoding (Section 2.1), trade off speed for long-horizon stability. Single-pass runs the entire horizon in one forward pass and is what we use for the leaderboard submissions above. Block decoding generates the horizon in segments, conditioning each on the previous segment’s median, with KV caching for efficiency.

We evaluate forward pass latency against Toto 1.0 and Chronos-2, the previous state of the art on GIFT-Eval. A 1,024-step forecast takes Toto 1.0 up to 16 autoregressive steps and single-pass Toto 2.0 a single forward

###### Forward Pass Latency

single forward pass block decoding

Toto 1.0

- 101
- 102
- 103

- 101
- 102
- 103

###### Toto 1.0

Latency(ms)()

Latency(ms)()

Toto 2.0 block decoding

Chronos 2

- Toto 2.0 1B
- Toto 2.0 2.5B

Chronos 2

Toto 2.0 313m

Toto 2.0 single forward pass

Toto 2.0 22m

Toto 2.0 4m

3m 10m 30m 100m 300m 1B 3B

32 64 128 256 512 1024 2048

Parameter Count

Prediction Length

- Figure 9 Left: forward pass latency vs. parameter count at forecast length=1,024. Every Toto 2.0 size is significantly faster than Toto 1.0. Right: forward pass latency vs. forecast horizon (log scale). Toto 2.0 stays flat in single-pass mode up to a 768-point forecast length, which we found best on synthetic signals. At a forecast horizon of 4,096 steps, 2.5B in single-pass mode remains faster than Chronos-2.

- pass. Every Toto 2.0 size is significantly faster than Toto 1.0 at this horizon, and the 313m runs at roughly the same latency as Chronos-2 (120m parameters) (Ansari et al., 2025) (Figure 9).

5.6 Long-horizon stability

Forecast quality on benchmarks like BOOM and GIFT-Eval reflects how a model performs within or near its training context. But many practical tasks want both long horizons and fine resolution. Downsampling buys horizon at the cost of the high-frequency structure (e.g. spikes, transient anomalies, sub-period dynamics, etc.) that the forecast is meant to capture.

To understand how Toto 2.0 behaves when asked to forecast much further than it was trained on, we evaluated all five sizes on randomly-generated sinusoidal mixtures at horizons of 2,048, 4,096, and 8,192 timesteps (well

- past the 4,096-step training context used for Toto). This is an illustrative stability test: it measures behavior beyond the training horizon, not extrapolation to genuinely novel dynamics.

We compare all five Toto 2.0 sizes to Toto 1.0 and Chronos-2 across the three horizons (Figure 10). 4m captures short-range patterns but collapses past its training context, producing flat or noisy forecasts. 22m holds longer but degrades by a 4,096-step forecast horizon. 313m is stable through 4,096 but loses structure beyond. 1B maintains the underlying pattern across all three horizons; 2.5B is more accurate still. Toto 1.0 and Chronos-2, despite Chronos-2 being trained on longer sequences, both lose coherence well before the 1B does.

### 6 Discussion

Toto 2.0 is the first TSFM family for which simply making the model bigger reliably makes it better. A single recipe applied across widths produces smooth improvements on BOOM, GIFT-Eval, and TIME from 4m up to 2.5B parameters, with only minor inversions inside the family on TIME’s rank metrics (Section 5.4). If

- Toto 1.0 and its contemporaries were the field’s BERT (Devlin et al., 2019) moment (BERT2S Workshop, 2025),
- Toto 2.0 is similar in some respects to a GPT-2 moment (Radford et al., 2019): scaling TSFMs is no longer a research question but a tool. Continuing to scale—more data, larger models—is a natural direction for future work. Below we outline what we see as the other major open questions for TSFM research:

Closing the gap with classical baselines. Foundation models capture dynamics classical statistical methods largely cannot: multivariate interactions, long context, and transfer across domains. But classical methods still have properties foundation models lack: clean extrapolation on simple signals, appropriate

###### Multi-Scale Decomposition

slow 500 / medium 100 / fast 20 - period in timesteps

HORIZON 2048

###### HORIZON 4096

###### HORIZON 8192

Toto2.02.5B

| | | | | |
|---|---|---|---|---|
| | | | | |

r = 0.990

r = 0.979

r = 0.818

###### Toto2.01B

| | | | | |
|---|---|---|---|---|
| | | | | |

r = 0.984 r = 0.945

r = 0.643

###### Toto2.0313M

r = 0.986 r = 0.947 r = 0.681

###### Toto2.022M

| | | | | |
|---|---|---|---|---|
| | | | | |

r = 0.805 r = 0.484

r = 0.315

###### Toto2.04M

| | | | | |
|---|---|---|---|---|
| | | | | |

r = 0.538 r = 0.371

r = 0.173

###### Toto1.0

r = 0.816 r = 0.627 r = 0.333

###### Chronos2

| | | | | |
|---|---|---|---|---|
| | | | | |

r = 0.663

r = 0.457

r = 0.310

0 500 1000 1500 2000

0 1000 2000 3000 4000

0 2000 4000 6000 8000

Timesteps Ahead

Timesteps Ahead

Timesteps Ahead

- Figure 10 Forecasts on a synthetic multi-scale signal (superimposed periods of 500, 100, and 20 timesteps) at three forecast horizons (2,048, 4,096, and 8,192 steps). Each row is a model, each column a horizon. Ground truth is plotted in gray; model forecasts in color. Larger Toto 2.0 sizes maintain coherent multi-scale structure at 8,192 steps; smaller sizes and prior-generation models lose structure progressively. Pearson correlation against ground truth is shown in each panel. Toto 2.0 forecasts use block decoding (Section 2.1). This experiment is illustrative: it measures stability beyond the training horizon on synthetic signals, not extrapolation to genuinely novel dynamics.

growth of prediction intervals with horizon under well-specified models, and predictable behavior on outof-distribution samples. The long-horizon study in Section 5.6 (Figure 10) is one window into this. Even the 2.5B loses some structure at a forecast horizon of 8,192 steps where a properly-fitted seasonal model would extrapolate cleanly. The gap shows up in many places: tail behavior, regime shifts, and forecasts on signals far outside any plausible training context. Closing it will likely require several things in combination: targeted architectural changes, continued scaling, and novel post-training objectives.

Improved data curation. Data curation in TSFMs has been ad hoc. Models typically mix synthetic series and a few public (or private) datasets, sample frequencies in proportions chosen by hand or by sweep, and stop there. In language modeling, data curation is treated as a first-class research problem: quality filtering, deduplication, annotation, mixing, curriculum. TSFM research has not gotten there yet, partly because scaling itself was still the open question: curation is a luxury you can only afford once data is abundant. In our own hyperparameter sweep, the optimal mix for pretraining excluded public data entirely (Section 4.2), while the optimal mix for finetuning was 45% public (Section 5.3). These are not intuitive results, and we arrived at them empirically rather than through principled selection. With scaling now reliable, it is time to take curation more seriously.

Metrics as a distinct modality. With Toto 1.0 and 2.0, we have built TSFMs suited for generic time series found commonly in the open. However, here at Datadog, we are interested in modeling the massive amounts of metrics data3 that we collect. While we have been able to cast Datadog metrics as basic time series, they are in fact a distinct data modality with unique properties. By compressing them into the mold of generic time series data, we lose significant amounts of embedded information and structure. In future work, we aim to prioritize the unique challenges of modeling Datadog metrics. Firstly, our architecture should be able to cater to the various metric types found on the Datadog platform, including histogram and distribution type data. Secondly, we deal with real world time series which have complex seasonality, such as multiple seasonality across long contexts, as well as non-integer and uneven periods. Thirdly, our data contains complex multivariate structure, including heterogeneous frequency where multivariate series can be sampled at different frequencies, as well as a context selection problem, where we have extremely high-dimensional series and we face the problem of selecting the relevant variates for the task at hand.

Multimodality and world models for observability. While multimodality for time series models has become an increasingly hot topic, it predominantly focuses on time series + text with limited datasets and evaluations (Liu et al., 2024, 2026; Xu et al., 2025; Chang et al., 2025). At Datadog, we care about models that understand how distributed systems behave. Our observability data is diverse and comprehensive, meaning we can develop models that deal not just with metrics, but also traces, logs, topology, code changes, events, alerts, text, etc. Our first step in this direction has been our recently released ARFBench (Xie et al., 2026), which focused on evaluating incident-grounded multimodal reasoning. Our longer-term goal is to develop a full-fledged world model for observability, extending to all telemetry types, unlocking capabilities such as proactive incident detection, root cause analysis, counterfactual analysis, simulation, and agent training.

### Acknowledgements

We thank Clement Acher, Askar Aitzan, Taha Aksu, Bogna Blaszczyck, Etienne Brodu, Ben Cohen, Antonin Couturier, Walid Elbouchikhi, Quentin Gendre Robin, Howard Huang, Sarra Kazdaghli, Mikhail Khodak, Shridhar Kumar, Rohan Kulkarni, Salahidine Lemaachi, Gael Magnan, Savita Manghnani, Hugo Miccinilli, Samuel Mueller, Ali Naeimi, Matthieu Neau, Sergey Pastukhov, Qiqi Ren, Afshin Rostamizadeh, AnnaMonica Toon, Lucas Verdonk, Kan Wang, and Stephan Xie for valuable discussions, infrastructure support, and contributions to the broader Toto effort.

### References

Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2623–2631, 2019. doi: 10.1145/3292500.3330701.

Taha Aksu, Gerald Woo, Juncheng Liu, Xu Liu, Chenghao Liu, Silvio Savarese, Caiming Xiong, and Doyen Sahoo. GIFT-Eval: A benchmark for general time series forecasting model evaluation. arXiv preprint arXiv:2410.10393, 2024a. URL https://arxiv.org/abs/2410.10393.

Taha Aksu, Gerald Woo, Juncheng Liu, Xu Liu, Chenghao Liu, Silvio Savarese, Caiming Xiong, and Doyen Sahoo. GIFT-Eval dataset. https://huggingface.co/datasets/Salesforce/GiftEval, 2024b.

Taha Aksu, Gerald Woo, Juncheng Liu, Xu Liu, Chenghao Liu, Silvio Savarese, Caiming Xiong, and Doyen Sahoo. GIFT-Eval pre-training datasets. https://huggingface.co/datasets/Salesforce/GiftEvalPretrain, 2024c.

Noah Amsel, David Persson, Christopher Musco, and Robert M. Gower. The polar express: Optimal matrix sign methods and their application to the Muon algorithm. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=yRtgZ1K8hO.

Abdul Fatir Ansari, Lorenzo Stella, Ali Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Hao Shen, Oleksandr Shchur, Syama S. Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Yuyang Wang. Chronos:

3https://docs.datadoghq.com/metrics/

Learning the language of time series. Transactions on Machine Learning Research, 2024. URL https://arxiv.org/abs/ 2403.07815.

Abdul Fatir Ansari, Oleksandr Shchur, Jasper Küken, Andreas Auer, Boran Han, Pedro Mercado, Syama S. Rangapuram, Hao Shen, Lorenzo Stella, Xiyuan Zhang, Mononito Goswami, Sanyam Kapoor, Danielle C. Maddix, Pablo Guerron, Tony Hu, Junming Yin, Nick Erickson, Prateek M. Desai, Hao Wang, Huzefa Rangwala, George Karypis, Yuyang Wang, and Michael Bohlke-Schneider. Chronos-2: From univariate to universal forecasting. arXiv preprint arXiv:2510.15821, 2025. URL https://arxiv.org/abs/2510.15821.

Andreas Auer, Patrick Podest, Daniel Klotz, Sebastian Böck, Günter Klambauer, and Sepp Hochreiter. TiRex: Zero-shot forecasting across long and short horizons with enhanced in-context learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=v7UqniC9pF.

Lukas Balles and Philipp Hennig. Dissecting Adam: The sign, magnitude and variance of stochastic gradients. arXiv preprint arXiv:1705.07774, 2020. URL https://arxiv.org/abs/1705.07774.

Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael K Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xLSTM: Extended long short-term memory. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=ARAxPPIAhq.

BERT2S Workshop. Recent advances in time series foundation models: Have we reached the ‘BERT moment’?, 2025. URL https://berts-workshop.github.io/. Workshop at NeurIPS 2025, San Diego.

Charlie Blake, Douglas Orr, and Carlo Luschi. Unit scaling: Out-of-the-box low-precision training. In Proceedings of the 40th International Conference on Machine Learning, pages 2548–2576. PMLR, 2023.

Charlie Blake, Constantin Eichenberg, Josef Dean, Lukas Balles, Luke Y. Prince, Björn Deiseroth, Andres F. Cruz-Salinas, Carlo Luschi, Samuel Weinbach, and Douglas Orr. u-µP: The unit-scaled maximal update parametrization. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= P7KRIiLM8T.

Ching Chang, Jeehyun Hwang, Yidan Shi, Haixin Wang, Wen-Chih Peng, Tien-Fu Chen, and Wei Wang. Time-IMM: A dataset and benchmark for irregular multimodal multivariate time series. In Advances in Neural Information Processing Systems (NeurIPS 2025 Datasets and Benchmarks Track), 2025. URL https://arxiv.org/abs/2506.10412.

Lizhang Chen, Jonathan Li, Kaizhao Liang, Baiyu Su, Cong Xie, Nuo Wang Pierse, Chen Liang, Ni Lao, and Qiang Liu. Cautious weight decay. arXiv preprint arXiv:2510.12402, 2025. URL https://arxiv.org/abs/2510.12402.

Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 785–794, 2016. doi: 10.1145/2939672.2939785.

Ben Cohen, Emaad Khwaja, Kan Wang, Clément Masson, Elise Ramé, Youssef Doubli, and Othmane Abou-Amal. Toto: Time series optimized transformer for observability. arXiv preprint arXiv:2407.07874, 2024. URL https://arxiv.org/ abs/2407.07874.

Ben Cohen, Emaad Khwaja, Youssef Doubli, Salahidine Lemaachi, Chris Lettieri, Charles Masson, Hugo Miccinilli, Elise Ramé, Qiqi Ren, Afshin Rostamizadeh, Jean Ogier du Terrail, Anna-Monica Toon, Kan Wang, Stephan Xie, Zongzhe Xu, Viktoriya Zhukova, David Asker, Ameet Talwalkar, and Othmane Abou-Amal. This time is different: An observability perspective on time series foundation models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=1jDAYXfcS2.

Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting. arXiv preprint arXiv:2310.10688, 2024. URL https://arxiv.org/abs/2310.10688.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://aclanthology.org/N19-1423/.

Federico Garza, Kin Gutiérrez, Cristian Challu, Jose Moralez, Ricardo Olivares, and Max Mergenthaler. tsfeatures: Calculates various features from time series data. python implementation of the r package tsfeatures, 2023. URL https://github.com/Nixtla/tsfeatures.

Google. TimesFM 2.5 200m. Hugging Face, 2025. URL https://huggingface.co/google/timesfm-2.5-200m-pytorch.

Lars Graf, Thomas Ortner, Stanisław Wo´zniak, and Angeliki Pantazi. FlowState: Sampling-rate invariant time series foundation model with dynamic forecasting horizons. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=R50AT6nAsM.

Graphcore Research. unit-scaling. GitHub, 2023. URL https://github.com/graphcore-research/unit-scaling. Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, and Yuxuan Chen. Query-key normalization for transformers.

arXiv preprint arXiv:2010.04245, 2020. URL https://arxiv.org/abs/2010.04245.

Shengding Hu, Yuge Tu, Xu Han, Ganqu Cui, Chaoqun He, Weilin Zhao, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Xinrong Zhang, Zhen Leng Thai, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, dahai li, Zhiyuan Liu, and Maosong Sun. MiniCPM: Unveiling the potential of small language models with scalable training strategies. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=3X2L2TFr0f.

Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https://kellerjordan.github.io/posts/muon/.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. URL https://arxiv.org/abs/2001.08361.

Andrej Karpathy. Beating GPT-2 for <<$100: the nanochat journey. GitHub Discussions, 2026. URL https://github. com/karpathy/nanochat/discussions/481.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2017.

URL https://arxiv.org/abs/1412.6980. Roger Koenker and Gilbert Bassett. Regression quantiles. Econometrica, 46(1):33–50, 1978. Zichong Li, Liming Liu, Chen Liang, Weizhu Chen, and Tuo Zhao. NorMuon: Making muon more efficient and scalable.

arXiv preprint arXiv:2510.05491, 2025. URL https://arxiv.org/abs/2510.05491.

Chenghao Liu, Taha Aksu, Juncheng Liu, Xu Liu, Hanshu Yan, Quang Pham, Silvio Savarese, Doyen Sahoo, Caiming Xiong, and Junnan Li. Moirai 2.0: When less is more for time series forecasting. arXiv preprint arXiv:2511.11698, 2025a. URL https://arxiv.org/abs/2511.11698.

Haoxin Liu, Shangqing Xu, Zhiyuan Zhao, Lingkai Kong, Harshavardhan Kamarthi, Aditya B. Sasanur, Megha Sharma, Jiaming Cui, Qingsong Wen, Chao Zhang, and B. Aditya Prakash. Time-MMD: Multi-domain multimodal dataset for time series analysis. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum?id=fuD0h4R1IL.

Haoxin Liu, Yichen Zhou, Rajat Sen, B. Aditya Prakash, and Abhimanyu Das. Rethinking multimodal time-series forecasting evaluation. OpenReview, 2026. URL https://openreview.net/forum?id=Z1TMV4bGuu.

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for LLM training. arXiv preprint arXiv:2502.16982, 2025b. URL https://arxiv.org/abs/ 2502.16982.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.

Spyros Makridakis, Evangelos Spiliotis, and Vassilios Assimakopoulos. The m4 competition: 100,000 time series and 61 forecasting methods. International Journal of Forecasting, 36(1):54–74, 2020. ISSN 0169-2070. doi: https://doi.org/10. 1016/j.ijforecast.2019.04.014. URL https://www.sciencedirect.com/science/article/pii/S0169207019301128. M4 Competition.

Pablo Montero-Manso, George Athanasopoulos, Rob J. Hyndman, and Thiyanga S. Talagala. FFORMA: Feature-based forecast model averaging. International Journal of Forecasting, 36(1):86–92, 2020. doi: 10.1016/j.ijforecast.2019.02.011.

Vladyslav Moroshan, Julien Siems, Arber Zela, Timur Carstensen, and Frank Hutter. TempoPFN: Synthetic pretraining of linear RNNs for zero-shot time series forecasting. arXiv preprint arXiv:2510.25502, 2025. URL https: //arxiv.org/abs/2510.25502.

Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. Transformers can do Bayesian inference. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=KSugKcbNf9.

Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations, 2023. URL https://arxiv.org/ abs/2211.14730.

Zhongzheng Qiao, Sheng Pan, Anni Wang, Viktoriya Zhukova, Yong Liu, Xudong Jiang, Qingsong Wen, Mingsheng Long, Ming Jin, and Chenghao Liu. It’s TIME: Towards the next generation of time series forecasting benchmarks. arXiv preprint arXiv:2602.12147, 2026. URL https://arxiv.org/abs/2602.12147.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners, 2019. URL https://cdn.openai.com/better-language-models/language_models_ are_unsupervised_multitask_learners.pdf. OpenAI Technical Report.

Liliang Ren, Congcong Chen, Haoran Xu, Young Jin Kim, Adam Atkinson, Zheng Zhan, Jiankai Sun, Baolin Peng, Liyuan Liu, Shuohang Wang, Hao Cheng, Jianfeng Gao, Weizhu Chen, and Yelong Shen. Decoder-hybrid-decoder architecture for efficient reasoning with long generation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=twSE0WA1vh.

Shuhei Watanabe. Tree-structured Parzen estimator: Understanding its algorithm components and their roles for better empirical performance. arXiv preprint arXiv:2304.11127, 2023. URL https://arxiv.org/abs/2304.11127.

Stephan Xie, Ben Cohen, Mononito Goswami, Junhong Shen, Emaad Khwaja, Chenghao Liu, David Asker, Othmane Abou-Amal, and Ameet Talwalkar. ARFBench: Benchmarking time series question answering ability for software incident response. arXiv preprint arXiv:2604.21199, 2026. URL https://arxiv.org/abs/2604.21199.

Zhijian Xu, Wanxu Cai, Xilin Dai, Zhaorong Deng, and Qiang Xu. Fidel-TS: A high-fidelity multimodal benchmark for time series forecasting. arXiv preprint arXiv:2509.24789, 2025. URL https://arxiv.org/abs/2509.24789.

Greg Yang and Edward J. Hu. Tensor programs IV: Feature learning in infinite-width neural networks. In Proceedings of the 38th International Conference on Machine Learning, volume 139, pages 11727–11737. PMLR, 2021.

Greg Yang, Edward Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tuning large neural networks via zero-shot hyperparameter transfer. In Advances in Neural Information Processing Systems, volume 34, pages 17084–17097, 2021.

