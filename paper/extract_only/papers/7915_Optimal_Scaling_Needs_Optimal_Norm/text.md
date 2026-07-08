# arXiv:2510.03871v2[cs.LG]27Jan2026

## OPTIMAL SCALING NEEDS OPTIMAL NORM

Oleg Filatov, Jiangtao Wang, Jan Ebert, Stefan Kesselheim J¨ulich Supercomputing Centre Forschungszentrum J¨ulich {o.filatov,jian.wang,ja.ebert,s.kesselheim}@fz-juelich.de

ABSTRACT

Despite recent progress in optimal hyperparameter transfer under model and dataset scaling, no unifying explanatory principle has been established. For Adam and Scion optimizers, we discover that joint optimal scaling across model and dataset sizes is conditioned on a single invariant: the operator norm of the output layer. Across models with up to 1.3B parameters trained on up to 138B tokens, the optimal learning rate/batch size pair (η∗,B∗) consistently has the same operator norm value — a phenomenon we term norm transfer. This constant norm condition is necessary but not sufficient: while for each dataset size, multiple (η,B) reach the optimal norm, only a unique (η∗,B∗) achieves the best loss. As a sufficient condition, we provide the first measurement of (η∗,B∗) scaling with dataset size for Scion, and find that the scaling rules are consistent with those of Adam. Tuning per-layer-group learning rates also improves model performance, with the output layer being the most sensitive and hidden layers benefiting from lower learning rates. We provide practical insights on norm-guided optimal scaling and release our Distributed Scion (Disco) implementation with logs from over two thousand runs to support research on LLM training dynamics at scale.

[Figure 1]

[Figure 2]

SDLAML/disco sdlaml-llm/norm-transfer

1 INTRODUCTION

Recent advancements in the domain of Large Language Models (LLMs) have been largely driven by the principle of scale. Increasing model size and training dataset volume consistently yields more capable systems (Hoffmann et al., 2022; Kaplan et al., 2020), yet at an increasing computational cost. Consequently, achieving optimal scaling — a training regime where hyperparameters are optimally configured with growing scale — becomes a necessary step to push the model frontier further.

To address the challenge of hyperparameter tuning, several powerful yet disparate methods have emerged. Theoretically grounded frameworks like Maximum Update Parametrization (µP) (Yang et al., 2022) help transfer optimal hyperparameters with model scaling. Meanwhile, empirical scaling laws (Li et al., 2025) provide rules of thumb for setting hyperparameters optimally when theory is absent, such as with dataset size scaling. Yet, these approaches often feel like pieces of a puzzle, with a unifying principle for scaling across both model and dataset dimensions remaining elusive.

Recently, an emerging paradigm of norm-based optimization (Bernstein & Newhouse, 2024a; Pethick et al., 2025a) has offered a new lens through which to view training dynamics: it reframes optimization as a process that controls the operator norms of the model’s weight matrices and gradient updates. This perspective enables monitoring of model properties during training, potentially revealing insights deeper than the loss curve alone. This raises a natural question: can the normbased perspective shed light on how to unify optimal model and dataset size scaling?

In this work, we argue that the answer is yes. By tracking and analyzing layer norms across thousands of experiments, we have made several discoveries, summarized below:

• Unifying invariant for optimal scaling. The operator norm of the output layer ∥Wout∥RMS→∞ (see Definition 2) for the optimal learning rate (η) and batch size (B) configuration has the same value — in other words, is invariant or “transfers” — with both model scaling (in width and depth) and dataset scaling (Fig. 2), as observed for both Scion

and Adam optimizers (Appendix A.12). We refer to this phenomenon as norm transfer, and it provides a necessary condition for optimality. However, it is not sufficient, as multiple non-optimal (η,B) pairs can reach the same optimal norm value (Fig. 3a).

- • Scaling rules for the Scion optimizer. As a sufficient condition for optimality, we empirically measure the relationship between optimal learning rate η∗, batch size B, and dataset size D. The result is η∗(B,D) ∝ B0.62 · D−0.56, matching the known square-root scaling rules for the Adam optimizer. We further find that the optimal batch size scales as B∗(D) ∝ D0.45±0.07, leading to η∗(D) ∝ D−0.28±0.07. For fixed D, one can trade off

η∗ ↔ B∗ via the η ∝

√

B rule within a low-sensitivity region around the optimal norm (Fig. 3b). While the model performance is insensitive to this change, this freedom can be of computational advantage, allowing for training with larger batch sizes.

- • Optimal per-layer-group learning rate. Performance can be improved by up to 6% in relative loss through additional per-layer-group tuning. We observe that a learning rate

ratio ηinput : ηhidden : ηoutput = 1 : 1/8 : 1 is consistently optimal across dataset sizes and batch sizes (Fig. 4). We also find the uniform 1 : 1 : 1 layout to be close to the optimal one. Among layer groups, the output layer is the most sensitive to tuning, with sensitivity decreasing gradually for the hidden layers and then the input layer.

- • Distributed Scion/Muon and experimental logs. To facilitate further research on large-scale training dynamics, we release Disco1, a distributed implementation of the Scion/Muon optimizer compatible with modern parallelization strategies, along with norm logs2 from over two thousand training runs conducted for this study.

2 METHODOLOGY

- 2.1 BACKGROUND & TERMINOLOGY

Recently, a fundamental shift in the field of optimal scaling occurred with the work of Yang et al. (2024). It changed the focus from model parametrizations towards the norm perspective by showing that Maximum Update Parametrization (µP) (Yang et al., 2022) can be derived from a more fundamental principle: enforcing a spectral condition on the model weights and their updates during the training. We briefly explain the idea behind these concepts below.

µP introduces theoretically grounded scaling rules for hyperparameters as a function of model width in order to ensure “maximal” feature learning in the infinite width limit. This way, the model is guaranteed to learn meaningful features while remaining stable as one scales up its size. As an important by-product, it was found that models with different widths, once parameterized within µP, all share the same optimal hyperparameters (e.g. learning rate) — therefore allowing for what is known as zero-shot hyperparameter transfer. This property has been extensively used for the past years to ensure optimal model scaling by tuning hyperparameters for a small (proxy) model, and then effortlessly transferring them to a larger one (Gunter et al., 2024; Dey et al., 2024; Meta AI, 2025; Zuo et al., 2025).

In turn, the spectral condition specifies bounds on the norms of weights and weight updates that are necessary to ensure feature learning. More formally:

out×dℓin to the ℓth weight matrix Wℓ ∈ Rd

ℓ

- Definition 1 (Spectral condition). Consider applying a gradient update ∆Wℓ ∈ Rd

out×dℓin for a layer ℓ = 1,...,L. The spectral norms of these matrices should satisfy

ℓ

∥Wℓ∥∗ = Θ

dℓout dℓin

and ∥∆Wℓ∥∗ = Θ

dℓout dℓin

, (1)

where ∥W∥∗ is the spectral norm, also equal to the largest singular value of W, and ∥x∥RMS = ∥x∥2/

#### √

d. The symbol Θ is employed following the ”Big-O” notation, indicating scaling behaviour

- 1https://github.com/SDLAML/disco
- 2https://wandb.ai/sdlaml-llm/norm-transfer/reports/

Norm-Transfer--VmlldzoxNDYwNjE2Mw

(in this case, “constant”3) w.r.t. infinite width limit d → +∞. If conditions in Definition 1 are met, the zero-shot hyperparameter transfer is guaranteed and the model is being trained in the µP regime.

Let us rewrite Definition 1 in a more “natural” way as:

∥Wℓ∥RMS→RMS = Θ(1) and ∥∆Wℓ∥RMS→RMS = Θ(1), (2) where we follow Large et al. (2024) and introduce the core concept of this work:

- Definition 2 (Induced operator norm4). Given a matrix W ∈ Rd

out×din and two normed vector spaces (Rd

#### ,∥·∥α) and (Rd

,∥·∥β), the “α to β” induced operator norm is given by:

in

out

∥Wx∥β ∥x∥α

. (3)

∥W∥α→β = max

x∈Rdin

The operator norms we are most interested in will be:

∥W∥1→RMS := maxj ∥colj(W)∥RMS , (4) ∥W∥RMS→RMS := din/dout ∥W∥∗ , (5)

∥W∥RMS→∞ := maxi din ∥rowi(W)∥RMS , (6) where rowi(.) and colj(.) denote the i-th row and j-th column of a matrix. In order to control the operator norms, Bernstein & Newhouse (2024a) derived duality maps, i.e. transformation rules of the gradients induced by a given norm. Applying these transformations not only keeps the gradient updates within the required bound (e.g. Eq. 2), but also ensures the steepest descent under the chosen norm (Bernstein & Newhouse, 2024b). For the norms in Eq. 4–6, the corresponding duality maps for the gradient G with singular value decomposition (SVD) G = UΣV ⊤ are:

colj(G) ∥colj(G)∥RMS

(7)

∥.∥1→RMS : colj(G)  →

din × UV ⊤ (8) ∥.∥RMS→∞ : rowi(G)  →

#### ∥.∥RMS→RMS : G  → d

out

1 din

rowi(G) ∥rowi(G)∥RMS

(9)

where the ∥.∥RMS→∞ norm was added by Pethick et al. (2025a). Moreover, they wrapped the normbased approach outlined above into a Scion optimizer.

Within Scion, one has to assign an operator norm to each layer, e.g. out of those in Eq. 4–6. The corresponding duality maps determine how raw gradients should be transformed for those layers before the optimizer updates the weights. For simplicity, layers are typically grouped as input, hidden, and output, and norms are assigned to these groups. Importantly, model weights are not explicitly transformed within Scion; only the weight updates are, via duality maps.

One prominent example of the norm-based view on model optimization is the Muon optimizer (Jordan et al., 2024), which proved to outperform Adam at scale (Liu et al., 2025; Wang et al., 2025) and showed great performance for models up to 1T parameters (Team et al., 2025). Muon can be viewed as a specific instantiation of Scion: it optimizes hidden layers under ∥.∥RMS→RMS assumption, and uses Adam for the remaining parameters. However, only in the case with no exponential moving average does Adam coincide with the steepest descent in “max-of-max norm” (Bernstein & Newhouse, 2024b). Since this is uncommon in practice, no “natural” norm applies, making Muon hard to analyze through the norm lens. By contrast, Scion naturally incorporates the norm perspective, updating every layer with an assigned, layer-specific norm, making it easy to interpret.

In practice, using norm-based optimizers as of now looks like a free lunch: they require only one momentum buffer5 (compared to two for Adam), result in better performance with almost no computational overhead in large-scale distributed scenarios, and by design have zero-shot hyperparameter transfer built in. Moreover, the norm-based approach provides more insights into the dynamics of the model training: optimizer-assigned norms can be used naturally to monitor the training dynamics on a per-layer basis. This observation leads us to discoveries that we describe in Sec. 3.

3Formally, f(x) = Θ(g(x)) if there are constants A, B > 0 such that A · g(x) ≤ f(x) ≤ B · g(x). 4In the following we will omit “induced operator” for simplicity. 5Or even none, see ScionLight (Pethick et al., 2025a).

- 2.2 TRAINING SETUP

In all experiments, we use the Llama 3 architecture (Grattafiori et al., 2024) and torchtitan training framework (Liang et al., 2025). Most of the experiments are performed on the model with a total size of 69M trainable parameters (including input/output embedding layers), hereafter referred to as proxy model. For additional ablations in Sec. 3.2, we scale up the model up to ×12 in width (to

- 1.3B parameters) and up to ×32 in depth (to 168M). Notably, we employ a norm-everywhere approach, inspired by the concept of well-normedness in Large et al. (2024) and the recent line of work (Loshchilov et al., 2025; Kim et al., 2025). Effectively, we ensure that the input x to every

Linear layer is normalized to ||x||RMS = 1 by a preceding RMSNorm layer without learnable parameters. More details on model configurations are provided in Appendix A.2 and Appendix A.3.

As optimizer, we use Scion without weight decay (i.e. its unconstrained version) (Pethick et al., 2025a) without momentum and with the norm assumptions ∥.∥1→RMS ⇒ ∥.∥RMS→RMS ⇒ ∥.∥RMS→∞ for input ⇒ hidden ⇒ output layers. Furthermore, we developed its distributed version, which natively integrates into torchtitan, supports FSDP/DDP/TP/EP/CP/PP strategies, and greatly speeds up the training at scale compared to the standard implementation. We make it openly available and provide more details in Appendix A.5.

For pretraining, we use a high-quality partition of the Nemotron-CC dataset (Su et al., 2025), Llama 3 tokenizer (Grattafiori et al., 2024) with a vocabulary size of 128,256 (after padding) and a context window of 4096. All the models are pretrained with the causal language modelling task. Unless stated otherwise, a constant learning rate schedule without warmup and without decay is used. This allows us, for a given set of hyperparameters, to perform a single long run and evaluate progressively larger dataset sizes, rather than conducting several runs for each dataset individually, thereby substantially reducing computational costs (Hu et al., 2024; H¨agele et al., 2024).

- 2.3 OPTIMAL NORM MEASUREMENT

Our initial intuition was that for a given model and data scale, there is always some optimal norm value, corresponding to some optimal hyperparameter choice. To establish this, we focus on the output layer with the Scion-assigned ∥.∥RMS→∞ norm (hereafter referred to as output norm) as being the most natural layer to study.6 The choice of ∥.∥RMS→∞ norm is motivated by Bernstein & Newhouse (2024a) as mapping from a “natural” continuous RMS norm semantics for hidden model representations onto a discrete vocabulary, although we also ablate this in Appendix A.15.2. Since by default we disable momentum and any regularization, we are only left with learning rate (η) and batch size (B) as hyperparameters to tune for optimality.

To extract the optimal hyperparameter configuration and the corresponding optimal norm, we run an (η,B) grid search for a given model and a given pretraining dataset size (hereafter referred to

- as horizon D, measured in tokens), and evaluate the model performance with training loss (crossentropy of the next token prediction). Since we train in a non-repeating “infinite-data” regime, training loss faithfully reflects model performance and its generalization. First, we examine how the optimal norm, associated to a (η∗,B∗) configuration optimal for a given horizon, changes as the horizon increases. Then, we fix the horizon and scale up the model in width and depth, repeating the same optimal norm measurement. This way, we study both model and dataset scaling directions.

Practically, for every batch size we are interested in “marginalising” or “profiling” across learning rates, i.e. picking the optimal one and the corresponding output norm (see Appendix A.2 for details on the grid and random seed variations). However, an empirically lowest-loss point across the learning rate grid turned out to be a statistically noisy estimate; therefore, for each batch size, we perform a fit to the distribution of training loss vs. output norm across learning rates. Finally, we extract the optimal norm value from the fitted curve and the corresponding learning rate from the nearest data point to the fitted optimum. We provide more details on the fitting procedure in Appendix A.4.

6The output layer is invariant to both width and depth scaling, it is the most sensitive to learning rate tuning (Sec. 3.4), and it can be viewed as a linear classifier on the learned hidden representations. These considerations make us believe that the output layer plays a “representative” role for the entire model, thus making it a distinct layer to analyse.

- 3 RESULTS

- 3.1 OUTPUT NORM DYNAMICS

First, we describe how the output layer norm evolves depending on the hyperparameter settings. From learning rate scans, we observe that indeed there is an optimal norm value for a given batch size and horizon (Fig. 1a). Furthermore, learning rate is positively correlated with the output norm: the higher the learning rate, the higher the norm. Since we use an unconstrained version of Scion, the norms generally grow with the number of gradient steps (Fig. 1b and Appendix A.6). However, we note that norm values can also be constrained during training with weight decay (see Appendix A.9) or with various spectral clipping techniques (Newhouse et al., 2025). Intriguingly, the norm growth is not linear in log-log scale but piecewise linear: with the slope abruptly changing for all batch sizes

- at the norm value of 26 − 27 and then at 29 − 210, where for the latter the dynamics enters the “turbulence” region. This slope change may have the same nature as a recently observed phenomenon in the loss curve dynamics (Mircea et al., 2025). Last but not least, we observe that learning rate controls the “ offset” of norm curves, and batch size controls the “decoupling degree” of curves: while early in training the curves of same η but different B are identical, the slope change at 26 −27 norm is more pronounced for larger batch sizes. Interestingly, after decoupling the curves seem to converge again to the same slope, that is lower than the initial one.

Output norm vs step for different ( , B) combinations

B = 128 samples, D = 233 tokens

2

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

= 7.8e-03, B=32 = 3.1e-02, B=32 = 1.2e-01, B=32 = 5.0e-01, B=32

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |

|[Figure 3]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

210

0

6.0

= 7.8e-03, B=128 = 3.1e-02, B=128 = 1.2e-01, B=128 = 5.0e-01, B=128 = 7.8e-03, B=512 = 3.1e-02, B=512 = 1.2e-01, B=512 = 5.0e-01, B=512

2

28

5.5

log||||WoutRMS

TrainLoss

log()2

4

26

5.0

6

= 7.8e-03, B=2048 = 3.1e-02, B=2048 = 1.2e-01, B=2048 = 5.0e-01, B=2048

24

8

4.5

22

10

23 24 25 26 27 28 29 210

102 103 104 105 106 Step

||Wout||RMS

(a)

(b)

- Figure 1: (a) Interplay of training loss, output layer norm ∥Wout∥RMS→∞ and learning rate. Results are for the proxy model (69M parameters), batch size B = 128 samples and horizon D =

233 tokens. Points are colored by log2(η) where η is the learning rate. Black dashed lines mark the optimal configuration with minimum training loss. (b) Growth of the output layer norm vs. gradient steps. Each curve corresponds to a (learning rate η, batch size B) pair, with B measured in samples; colour encodes batch size and line style encodes learning rate. See also the same plot vs. token horizons in Appendix A.6.

- 3.2 OPTIMAL NORM TRANSFER

After analysing learning rate scans across batch sizes, horizons and models of varying width/depth, we visualise results in Fig. 2, with an extended set of plots in Appendix A.7 and Appendix A.17. Each data point corresponds to optimally tuned learning rate η∗ for a given batch size, minimising training loss for that horizon and model. We report our observations below, separately for each direction of scaling, as well as additional ablations.

Data scaling: After profiling across learning rates and plotting optimal norm against batch size, we observe that for a given horizon there is a single optimal batch size with the corresponding optimal output norm ∥Wout∥RMS→∞ = 27.0±0.2. Intriguingly, this norm value transfers across horizons. We refer to this phenomenon as norm transfer: the optimal (η,B) configuration for a given horizon must result in the optimal norm of ≈ 27. Also note that the optimal batch size grows with horizon scaling, which we discuss in Sec. 3.3. Interestingly, we observe the same norm transfer behavior when switching to a different dataset (Appendix A.11): specifically, Thai and Russian language partitions of Fineweb-2 (Penedo et al., 2025).

Scaling horizon for proxy model (69M)

5.4

Horizon [tokens]

231 (2.1 B) 233 (8.6 B)

235 (34 B)

5.2

237 (137 B)

5.0

4.8

Trainloss

4.6

4.4

4.2

Batch size [samples]

4.0

32 2048

3.8

5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 log ||Wout||RMS

(a)

Scaling width/depth for 233 (8.6 B) token horizon

4.6

d=256, L=4 (69M) d=1024, L=4 (314M)

d=256, L=32 (91M)

d=3072, L=4 (1.3B)

d=256, L=128 (168M)

4.4

4.2

Trainloss

4.0

3.8

Batch size [samples]

32 1024

3.6

5.0 5.5 6.0 6.5 7.0 7.5 8.0 log ||Wout||RMS

(b)

- Figure 2: Training loss vs. output layer norm across batch sizes. (a) Fixed proxy model (69M parameters) while increasing token horizon from 231 to 237. (b) Fixed token horizon 233 while scaling width/depth of the proxy model as indicated in the legend. Each batch size point (increasing from 32 in ×2 steps, reflected by marker size) has its learning rate optimally tuned. The optimal batch size per horizon/model configuration is indicated by the filled marker. All curves share optimal norm at 7.0 ± 0.2 across horizons and 7.4 ± 0.2 across models (grey band).

Model width scaling: It is expected to preserve the optimal norm by the design of our optimizer via the spectral condition (Eq. 1). Indeed, in Fig. 2b we observe that scaling up in width by a factor of ×12 while keeping the horizon fixed results in the nested “µP-style” curves, sharing the same optimal norm while resulting in lower loss as we scale up.

Model depth scaling: Although not obvious a priori, we observe experimentally that scaling up in the number of layers by a factor of ×32 results in norm transfer. This is quite surprising, since we do not employ any of the established depth-transfer techniques (Bordelon et al., 2023; Yang et al., 2023; Dey et al., 2025). We ablate them in Appendix A.10 and find that in our setup they all induce learning rate transfer, but our strategy (no residual scaling factors, initialization rescaling of layers prior to residuals by 1/ 2Nlayers) results in the lowest loss. We speculate that this may be related to our norm-everywhere approach (Sec. 2.2) and uniformity in norm treatment by the optimizer and weight initialization.

Momentum & learning rate decay: In practice, one is more interested in Scion with non-zero momentum and with a decaying learning rate schedule as resulting in better performance. We study the impact of these two options in Appendix A.15 and observe that they both show norm transfer. Notably, the addition of momentum largely reduces sensitivity to batch size choice with multiple values resulting in the same optimal norm and loss (Fig. 14). The same is applicable to learning rate decay, which reduces sensitivity to learning rate choice (Fig. 18b).

Adam optimizer: As the optimizer commonly used in practice, we study its data scaling for the proxy model with two configurations: with momenta (β1 = 0.9,β2 = 0.95) and without (β1 = β2 = 0), where the latter case corresponds to a sign gradient descent. Intriguingly, for both we observe a clear norm transfer (Appendix A.12): the case without momentum exhibits the same optimal norm value (≈ 27) as the analogous without-momentum Scion, while the case with momentum has a noticeably higher optimal norm (≈ 211). We believe that this shared norm transfer pattern further supports a common norm-based view on optimization by Bernstein & Newhouse (2024b), and therefore makes our observations for Scion also transferable to Adam.

Normalization layers: Since we are interested in the norm structure of model scaling, our choice of norm-everywhere strategy may play a key role in the observed phenomena. We ablate various ways to place normalization layers (RMSNorm without trainable parameters) within the architecture in Appendix A.14. For the proxy model, fixed data horizon of 43B tokens, and fixed batch size B = 256 sequences we perform a learning rate scan while removing specific normalization layers

(see Appendix A.14 for a detailed description). In Fig. 13b for the case of the Scion optimizer with momentum α = 0.1 we observe that removing normalization in residual connections results in significant training divergences. The setup with QK-norm + residuals + output layer normalization results in the same learning rate profile as our default norm-everywhere, indicating redundancy of MLP- and VO-normalization. Furthermore, residuals + output configuration results in slightly better performance, albeit with higher learning rate sensitivity. Likewise, addition of QK-norm largely reduces learning rate sensitivity, as also shown by Wortsman et al. (2023), thus making the training less sensitive to learning rate choice.

Finally, we selected the residuals-only configuration as the best trade-off between the minimalist usage of normalization layers, learning rate sensitivity, and model performance, and studied its norm scaling properties. Fig. 12 shows that norm transfer is present also in this scenario, interestingly with significantly lower optimal norm comparing to the norm-everywhere setting (≈ 23, one order of magnitude lower). One can also notice from Fig. 13b (see norm values in the legend) that it is the removal of the output layer norm that induces this reduction. This finding poses an interesting question: can it be in any way advantageous to prefer the model with lower weight norms? Intuitively, we would answer positively, referring to discussions in Newhouse et al. (2025), but leave detailed studies for future work.

|↬ Summary I: Within the Scion framework, optimal norm transfers in both model (width and depth) and data scaling directions: it is necessary to choose the hyperparameter configuration so that the model output norm ∥Wout∥RMS→∞ falls into the optimal region. Substituting alternative norms (∥Wout∥RMS→RMS or ∥Win∥1→RMS) maintains the transfer consistency. The same behaviour holds with non-zero momentum, learning rate decay, and residuals-only layer normalization, as well as for the Adam optimizer. The optimal norm value decreases by a factor of 10 with the removal of the normalization layer before the model output layer.|
|---|

- 3.3 OPTIMAL (η,B) SCALING RULE

2 1

2 3

2 5

2 7

2 9

Batch size [samples]

32 64 128

256 512 1024

2 11

Free fit

Heuristic fit

231 232 233 234 235 236 237 First horizon to reach optimal norm [tokens]

(a)

Horizon [tokens]

231 (2.1 B) 233 (8.6 B) 235 (34 B)

2 1

| |
|---|

237 (137 B)

2 3

optimal

2 5

2 7

log2 = log2 B + log2 D +

2 9

= 0.62 ± 0.05, = -0.56 ± 0.05

= 9.8 ± 1.6; R2 = 0.92

25 26 27 28 29 210 211 B [samples]

(b)

- Figure 3: (a) (η,B) combinations that reach the optimal norm ∥Wout∥RMS→∞ = 27.0±0.2 for a given token horizon. Colours denote batch size (B); the y-axis is learning rate (η). Solid and dashed lines denote free and heuristic fits (described in text). (b) Optimal learning rate per batch size across horizons. Circled markers indicate optimal (η∗,B∗) with the lowest loss. Within a horizon, marker transparency linearly interpolates between the lowest- and highest-loss runs, with higher transparency indicating higher training loss. Error bars show systematic variation from the fitting method (Appendix A.4). Dashed lines are a joint linear regression with log2 η∗ ∼ log2 B + log2 D.

Despite the discovered norm guidance, it is still not obvious how to select the corresponding optimal combination of learning rate and batch size for a given horizon. Or more generally, what is the sufficient condition for optimality? In this Section, we explore this question.

- Fig. 3a illustrates that the optimal norm condition observed in Fig. 2 is necessary but not sufficient. For each token horizon (x-axis), we plot the learning rates (y-axis) and batch sizes (colour) that

reach7 the optimal-norm region ∥Wout∥RMS→∞ ∈ [26.8, 27.2]. One can observe that for a given horizon, every batch size will reach optimal norm with a sufficiently high learning rate. We fit the

data with linear models log2 η = αfirst log2 B + βfirst log2 Dfirst + γfirst (free fit) and log2 η = 1.5log2 B − log2 Dfirst + γfirst (heuristic fit). For the free fit, we find the exponents αfirst = 1.32 ± 0.03 and βfirst = 0.96 ± 0.03, which are close to the values from the heuristic fit.

Hence, we cannot rely on the output norm as a guide to selecting optimal hyperparameters; it is only a necessary and not a sufficient condition. Let us now study sufficient conditions by first unfolding Fig. 2a and including optimal learning rate information that was profiled away. Specifically, we are interested in how the optimal learning rate η∗ changes within a fixed horizon D with the batch size B change, and then with horizons D scaled up. Fig. 3b shows the corresponding data points along with a linear regression fit log2 η∗(B,D) = α log2 B+β log2 D+γ. Note that only circled markers are per-horizon optima with the lowest loss. We observe several things:

- • The coefficients of the fit α = 0.62 ± 0.05, β = −0.56 ± 0.05 are consistent with a well-established square-root scaling with batch size (Malladi et al., 2024) and data horizon (Bjorck et al., 2025) for Adam, respectively. Similar to AI et al. (2025); Sato et al. (2025) we√ observe no surge phenomenon (Li et al., 2024), i.e. transition for a fixed D from η∗ ∝

B to η∗ ∝ 1/

√

B scaling rules for batch sizes higher than the critical one (Zhang et al., 2025). Theoretically, Jianlin (2025) explains this from the mean field theory perspective.

- • Different batch sizes B result in different losses, and for each horizon D there is an optimal one B∗(D), as emphasized in Fig. 3b with circled markers and marker transparency for relative loss difference. The optimal batch size increases with horizon scaling: in Appendix A.8 we measure with extended set of horizons B∗(D) ∝ D0.45±0.07, which is con-

sistent with Adam (Li et al., 2025; Bergsma et al., 2025) and intriguingly with B∗ ∝

√

D.

- • Using B∗(D) ∝ D0.45 and log2 η∗(B,D) ∝ 0.62log2 B − 0.56log2 D with the corresponding uncertainties, we obtain for the optimal learning rate scaling η∗(D) ∝ D−0.28±0.07. This observation is consistent with Li et al. (2025) but appears to be in tension with Shen et al. (2024); Bergsma et al. (2025), albeit our methodologies are not fully comparable.8 Again, this is interestingly close to η∗(D) ∝ D−1/4.
- • Since there exists a single optimal batch size for each data horizon, the number of devices usable for training is fundamentally capped: beyond a point, increasing the number of devices either hurts throughput (small per-device microbatch size to keep the optimal global batch size) or degrades loss (leaving the optimal batch size region to keep throughput). This hints towards an interesting research direction: if this limit can be bypassed.
- • In fact, for a fixed horizon, it is not a single optimal (η∗,B∗) but an optimal region (η∗ ± ∆η, B∗ ± ∆B) that results in near-optimality (opacity in Fig. 3b). We relate this to the notion of learning rate sensitivity (Wortsman et al., 2023) that we rephrase as norm sensitivity. We think this region is defined by the “flatness” of the horizon curve (Fig. 2a) around the optimal norm value. Within this region, one can “exchange” learning rate for

#### √

B rule, thus allowing for some flexibility in optimal hyperparameter choice, e.g. training with larger batch sizes.

batch size via the η ∝

|↬ Summary II: For Scion, we measure the following hyperparameter scaling rules inducing the sufficient optimal scaling condition:<br><br>η∗(D) ∝ D−0.28±0.07 and B∗(D) ∝ D0.45±0.07, (10)<br><br>consistent with the Adam’s scaling exponents. For a fixed horizon D, one can trade off η∗ ↔ B∗ via the η ∝<br><br>√<br><br>B rule within the region of low norm sensitivity, without loss in performance. By Scion’s design, these observations hold true with model width scaling.|
|---|

- 7Optimal norm will most likely be reached at some point (provided learning rate sweep resolution in Fig. 3a is too small), since in unconstrained Scion the weight norms are growing in time (see Sec. 3.1 and Fig. 1b).
- 8For example, because of weight decay usage in Bergsma et al. (2025), which significantly affects norm dynamics by constraining it, as we discuss in Sec. 5.

##### 3.4 OPTIMAL PER-LAYER-GROUP LEARNING RATE

20 Best input = output configuration per horizon, B=512

Best configurations for horizon=233 (8.6 B), B=128

Top 10% loss: [4.11, 4.18]

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

229 (0.5 B): loss = 5.10 | equal @ 2 1: 5.45 231 (2.1 B): loss = 4.38 | equal @ 2 1: 4.66 233 (8.6 B): loss = 4.06 | equal @ 2 2: 4.22 235 (34 B): loss = 3.93 | equal @ 2 3: 4.00

Others: [4.19, 4.76]

20

2 1

- Top 1: 4.11 ± 0.01

- Top 2: 4.12 ± 0.01

- Top 3: 4.12 ± 0.02 Equal LRs at 2 4: 4.13 ± 0.00

2 2

2 2

Top-10% loss counts per layer group

8

2 3

2 4

6

2 4

count

4

2 6

2

2 5

0

2 8 2 6 2 4 2 2 20

2 6

input hidden output

2 8

| |
|---|

| |
|---|

input hidden output

input hidden output

(a)

(b)

- Figure 4: (a) Parallel-coordinates view of per-layer-group learning rate tuning. Results are for the proxy model (69M parameters) and batch size B = 128 samples, averaged across random seeds as described in Appendix A.2. Dark gray lines are the top 10% runs (loss 4.11–4.18); light gray lines are the remainder (loss 4.19–4.76). Orange traces highlight the three best settings. The inset histogram shows the distribution of top 10% counts for each layer group. (b) Best learning rate

layouts per training horizon under the constraint ηinput = ηoutput. Results are for the proxy model (69M parameters) and batch size B = 512 samples. All horizons favor a V-shaped layout

with ηhidden smaller than the input/output learning rates by the same ×1/8 factor. In the legend we also report loss for the optimal ηinput = ηhidden = ηoutput ≡ η layout (“equal @η”).

So far, we approached scaling from a “global” learning rate point of view. However, this may not be the case, and intricate dynamics can emerge where various layers require different learning rates at different scales to be trained optimally, thus questioning our conclusions so far. In this Section, we explore if this is the case.

- Fig. 4a presents results for a proxy model (69M parameters), fixed data horizon (8.6B tokens) and fixed batch size (B = 128 samples, optimal for this horizon) where we run grid search over learning rate values η ∈ {2−8,2−7,...,20} for input (token embedding), output (linear projection onto vocabulary) and hidden (all the other) layers, averaged across random seeds (Appendix A.2). We observe that there is little optimal learning rate imbalance across layer groups, and uniform learning rate assignment results in the same loss as the optimal configurations within uncertainties. Furthermore, from the width of the optimal nodes count histograms per layer groups, we conclude that the output layer is the most sensitive to learning rate mistuning, with the sensitivity progressively decreasing for hidden and then input layers.

From analysing Fig. 4a and additional ones for different batch sizes (Appendix A.16) we found that the configuration ηinput : ηoutput : ηhidden = 1 : 1/8 : 1 is always among the top 10%. This symmetry simplifies the learning scan and notably contradicts the optimal configurations suggested in Pethick et al. (2025a) and Riabinin et al. (2025). To study dynamics with horizon scaling, we perform the learning rate grid scan same as in Fig. 4a but with constraining ηinput = ηoutput 9, for the proxy model with B = 512. Fig. 4b illustrates the results, where we see the optimal hidden ratio (ηinput/ηhidden = 1/8) transfer across horizons, as well as that it brings loss improvement w.r.t. a constant learning rate baseline. Lastly, we note that again, due to the optimizer design, we expect these observations to hold true under model width scaling.

|↬ Summary III: Uniform learning rate configuration across layers is a strong baseline, which still can be improved with additional hidden layer group tuning: ηinput : ηoutput : ηhidden = 1 : 1/8 : 1 yields a relative loss improvement of up to 6% and is transferable across dataset sizes.|
|---|

9In terminology of Bernstein & Newhouse (2024a) this corresponds to mass tuning.

- 4 RELATED WORK

Hyperparameters with model scaling Yang et al. (2022) showed how to transfer optimal hyperparameters from a small to a large model in a principled way via Maximal Update Parametrization (µP). Everett et al. (2024) later showed that such transfer is also possible in other parametrizations. Yang et al. (2023); Dey et al. (2025) extended the method towards model scaling in depth. Empirically, scaling laws on how to set optimal hyperparameters as a function of compute (DeepSeek-AI et al., 2024), loss (Hu et al., 2024) or model size (Porian et al., 2025) were measured.

Hyperparameters with data scaling Remains poorly understood theoretically: Smith & Le (2018) showed for SGD how to adjust learning rate and batch size by modelling optimization trajectory as a stochastic differential equation (SDE). Largely, the problem has been approached experimentally by measuring hyperparameter scaling rules as a function of the dataset size (Shen et al., 2024; Hu et al., 2024; Filatov et al., 2025; Bergsma et al., 2025; Li et al., 2025).

(η,B) scaling rules Historically, studies of interaction between learning rate and batch size emerged as an experimental effort to scale batch size without losing performance (Keskar et al., 2017; Goyal et al., 2018; Hilton et al., 2022). Later, a deeper understanding has been built from various theoretical angles: SDE (Malladi et al., 2024; Compagnoni et al., 2024), loss curvature (McCandlish et al., 2018), random matrix theory (Granziol et al., 2021).

Norm-based optimization Starting from the spectral condition (Yang et al., 2024), the approach of transforming gradient updates based on norm assumptions was fully established in Large et al. (2024); Bernstein & Newhouse (2024a), and recently explored in constraining weights themselves (Newhouse et al., 2025). The steepest descent view allowed for connections with manifold learning (Cesista, 2025) and optimizer design (Riabinin et al., 2025). This line of work has led to Muon (Jordan et al., 2024) and Scion (Pethick et al., 2025a;b), along with improvements (Ahn et al., 2025; Amsel et al., 2025), and benchmarks (Wen et al., 2025; Semenov et al., 2025) thereof.

- 5 CONCLUSION AND DISCUSSION

In this work, we demonstrate that the operator norm of the output layer is a powerful measure that guides joint optimal scaling across both model and dataset dimensions. Informally, we show:

- 1. (η, B, D) choice ===affects⇒ layer operator norm (Sec. 3.1)
- 2. optimal loss ====requires⇒ optimal norm (Sec. 3.2)
- 3. optimal η∗(D),B∗(D) scaling rules ==yield⇒ optimal loss (Sec. 3.3)

In words, we empirically (1) study how norms evolve with hyperparameter change and how to tune them to desired values; (2) demonstrate that the optimal hyperparameter configuration must have a predefined (output) layer norm in order to be transferable across data and model scales; (3) derive optimal hyperparameter scaling rules resulting in optimal loss.

While we are confident that the scaling rules in Sec. 3.3 hold at even larger scales, we still don’t know why they are induced in this form, very much resembling square-root and 1/4-power laws. Moreover, how do these rules connect with our main finding, a necessary condition of scaling trajectory in (data, model) axes to have the same constant value — or one might say, to remain on a manifold (Bernstein, 2025). At this point more new questions arise:

- • Why does optimal norm transfer? It is puzzling what makes the optimal scaling trajectory remain on the constant norm manifold, as well as what defines its structure.
- • What is the reason behind optimal scaling rules? While we show how to set hyperparameters optimally, there is something missing in the norm perspective to explain it.
- • Which norm is exactly optimal? We paid most of our attention to ∥Wout∥RMS→∞, but are the observed phenomena really specific to this one only?
- • How can the constant norm condition be leveraged? It looks like a naturally emerging inductive bias that one can take advantage of to optimize the training process.

We don’t yet have answers to those questions, but we believe our study scratches the surface of exciting phenomena to be further understood.

ACKNOWLEDGMENTS

We thank Kaiyue Wen and Ismail Khalfaoui Hassani for helpful discussions and feedback on the manuscript. This research was supported by TrustLLM funded by Horizon Europe GA 101135671, and by the Helmholtz Foundation Model Initiative as a part of the Synergy Unit. The authors gratefully acknowledge the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) for funding this project by providing computing time on the GCS Supercomputer JUWELS at J¨ulich Supercomputing Centre (JSC). Parts of computational resources were provided by the German AI service center WestAI.

REFERENCES

Kwangjun Ahn, Byron Xu, Natalie Abreu, Ying Fan, Gagik Magakyan, Pratyusha Sharma, Zheng Zhan, and John Langford. Dion: Distributed orthonormalized updates. arXiv preprint arXiv:2504.05295, 2025.

Essential AI, :, Ishaan Shah, Anthony M. Polloreno, Karl Stratos, Philip Monk, Adarsh Chaluvaraju, Andrew Hojel, Andrew Ma, Anil Thomas, Ashish Tanwer, Darsh J Shah, Khoi Nguyen, Kurt Smith, Michael Callahan, Michael Pust, Mohit Parmar, Peter Rushton, Platon Mazarakis, Ritvik Kapila, Saurabh Srivastava, Somanshu Singla, Tim Romanski, Yash Vanjani, and Ashish Vaswani. Practical efficiency of muon for pretraining. arXiv preprint arXiv:2505.02222, 2025.

Noah Amsel, David Persson, Christopher Musco, and Robert M. Gower. The polar express: Optimal matrix sign methods and their application to the muon algorithm, 2025. URL https: //arxiv.org/abs/2505.16932.

Shane Bergsma, Nolan Dey, Gurpreet Gosal, Gavia Gray, Daria Soboleva, and Joel Hestness. Power lines: Scaling laws for weight decay and batch size in llm pre-training. arXiv preprint arXiv:2505.13738, 2025.

Jeremy Bernstein. Modular manifolds. Thinking Machines Lab: Connectionism, 2025. doi: 10. 64434/tml.20250926. https://thinkingmachines.ai/blog/modular-manifolds/.

Jeremy Bernstein and Laker Newhouse. Modular duality in deep learning. arXiv preprint arXiv:2410.21265, 2024a.

Jeremy Bernstein and Laker Newhouse. Old optimizer, new norm: An anthology. arXiv preprint arXiv:2409.20325, 2024b.

Johan Bjorck, Alon Benhaim, Vishrav Chaudhary, Furu Wei, and Xia Song. Scaling optimal lr across token horizons. arXiv preprint arXiv:2409.19913, 2025.

Blake Bordelon, Lorenzo Noci, Mufan Bill Li, Boris Hanin, and Cengiz Pehlevan. Depthwise hyperparameter transfer in residual networks: Dynamics and scaling limit. arXiv preprint arXiv:2309.16620, 2023.

Franz Louis Cesista. Muon and a selective survey on Steepest Descent in Riemannian and non-Riemannian Manifolds, April 2025. URL http://leloykun.github.io/ponder/ steepest-descent-non-riemannian/.

Enea Monzio Compagnoni, Tianlin Liu, Rustem Islamov, Frank Norbert Proske, Antonio Orvieto, and Aurelien Lucchi. Adaptive methods through the lens of sdes: Theoretical insights on the role of noise. arXiv preprint arXiv:2411.15958, 2024.

Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.

DeepSeek-AI, :, Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, Huazuo Gao, Kaige Gao, Wenjun Gao, Ruiqi Ge, Kang Guan, Daya Guo, Jianzhong Guo, Guangbo Hao, Zhewen Hao, Ying He, Wenjie Hu, Panpan Huang, Erhang Li, Guowei Li, Jiashi Li, Yao Li, Y. K. Li, Wenfeng Liang, Fangyun Lin, A. X. Liu, Bo Liu, Wen Liu, Xiaodong Liu, Xin Liu, Yiyuan Liu, Haoyu Lu, Shanghao Lu, Fuli Luo,

Shirong Ma, Xiaotao Nie, Tian Pei, Yishi Piao, Junjie Qiu, Hui Qu, Tongzheng Ren, Zehui Ren, Chong Ruan, Zhangli Sha, Zhihong Shao, Junxiao Song, Xuecheng Su, Jingxiang Sun, Yaofeng Sun, Minghui Tang, Bingxuan Wang, Peiyi Wang, Shiyu Wang, Yaohui Wang, Yongji Wang, Tong Wu, Y. Wu, Xin Xie, Zhenda Xie, Ziwei Xie, Yiliang Xiong, Hanwei Xu, R. X. Xu, Yanhong Xu, Dejian Yang, Yuxiang You, Shuiping Yu, Xingkai Yu, B. Zhang, Haowei Zhang, Lecong Zhang, Liyue Zhang, Mingchuan Zhang, Minghua Zhang, Wentao Zhang, Yichao Zhang, Chenggang Zhao, Yao Zhao, Shangyan Zhou, Shunfeng Zhou, Qihao Zhu, and Yuheng Zou. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954, 2024.

Nolan Dey, Quentin Anthony, and Joel Hestness. The practitioner’s guide to the maximal update parameterization, Sep 2024. URL https://www.cerebras.ai/blog/ the-practitioners-guide-to-the-maximal-update-parameterization.

Nolan Dey, Bin Claire Zhang, Lorenzo Noci, Mufan Li, Blake Bordelon, Shane Bergsma, Cengiz Pehlevan, Boris Hanin, and Joel Hestness. Don’t be lazy: Completep enables compute-efficient deep transformers. arXiv preprint arXiv:2505.01618, 2025.

Katie Everett, Lechao Xiao, Mitchell Wortsman, Alexander A. Alemi, Roman Novak, Peter J. Liu, Izzeddin Gur, Jascha Sohl-Dickstein, Leslie Pack Kaelbling, Jaehoon Lee, and Jeffrey Pennington. Scaling exponents across parameterizations and optimizers. arXiv preprint arXiv:2407.05872, 2024.

Wei Feng, Will Constable, and Yifan Mao. Getting started with fully sharded data parallel (fsdp2), 2025. URL https://docs.pytorch.org/tutorials/intermediate/ FSDP_tutorial.html.

Oleg Filatov, Jan Ebert, Jiangtao Wang, and Stefan Kesselheim. Time transfer: On optimal learning rate and batch size in the infinite data limit. arXiv preprint arXiv:2410.05838, 2025.

Priya Goyal, Piotr Doll´ar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2018.

Diego Granziol, Stefan Zohren, and Stephen Roberts. Learning rates as a function of batch size: A random matrix theory approach to neural network training. arXiv preprint arXiv:2006.09092, 2021.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzm´an, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur ¸Celebi, Patrick Alrassy, Pengchuan

Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, V´ıtor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj

Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Tom Gunter, Zirui Wang, Chong Wang, Ruoming Pang, Andy Narayanan, Aonan Zhang, Bowen Zhang, Chen Chen, Chung-Cheng Chiu, David Qiu, Deepak Gopinath, Dian Ang Yap, Dong Yin, Feng Nan, Floris Weers, Guoli Yin, Haoshuo Huang, Jianyu Wang, Jiarui Lu, John Peebles, Ke Ye, Mark Lee, Nan Du, Qibin Chen, Quentin Keunebroek, Sam Wiseman, Syd Evans, Tao Lei, Vivek Rathod, Xiang Kong, Xianzhi Du, Yanghao Li, Yongqiang Wang, Yuan Gao, Zaid Ahmed, Zhaoyang Xu, Zhiyun Lu, Al Rashid, Albin Madappally Jose, Alec Doane, Alfredo Bencomo, Allison Vanderby, Andrew Hansen, Ankur Jain, Anupama Mann Anupama, Areeba Kamal, Bugu Wu, Carolina Brum, Charlie Maalouf, Chinguun Erdenebileg, Chris Dulhanty, Dominik Moritz, Doug Kang, Eduardo Jimenez, Evan Ladd, Fangping Shi, Felix Bai, Frank Chu, Fred Hohman, Hadas Kotek, Hannah Gillis Coleman, Jane Li, Jeffrey Bigham, Jeffery Cao, Jeff Lai, Jessica Cheung, Jiulong Shan, Joe Zhou, John Li, Jun Qin, Karanjeet Singh, Karla Vega, Kelvin Zou, Laura Heckman, Lauren Gardiner, Margit Bowler, Maria Cordell, Meng Cao, Nicole Hay, Nilesh Shahdadpuri, Otto Godwin, Pranay Dighe, Pushyami Rachapudi, Ramsey Tantawi, Roman Frigg, Sam Davarnia, Sanskruti Shah, Saptarshi Guha, Sasha Sirovica, Shen Ma, Shuang Ma, Simon Wang, Sulgi Kim, Suma Jayaram, Vaishaal Shankar, Varsha Paidi, Vivek Kumar, Xin Wang, Xin Zheng, Walker Cheng, Yael Shrager, Yang Ye, Yasu Tanaka, Yihao Guo, Yunsong Meng, Zhao Tang Luo, Zhi Ouyang, Alp Aygar, Alvin Wan, Andrew Walkingshaw, Andy Narayanan, Antonie Lin, Arsalan Farooq, Brent Ramerth, Colorado Reed, Chris Bartels, Chris Chaney, David Riazati, Eric Liang Yang, Erin Feldman, Gabriel Hochstrasser, Guillaume Seguin, Irina Belousova, Joris Pelemans, Karen Yang, Keivan Alizadeh Vahid, Liangliang Cao, Mahyar Najibi, Marco Zuliani, Max Horton, Minsik Cho, Nikhil Bhendawade, Patrick Dong, Piotr Maj, Pulkit Agrawal, Qi Shan, Qichen Fu, Regan Poston, Sam Xu, Shuangning Liu, Sushma Rao, Tashweena Heeramun, Thomas Merth, Uday Rayala, Victor Cui, Vivek Rangarajan Sridhar, Wencong Zhang, Wenqi Zhang, Wentao Wu, Xingyu Zhou, Xinwen Liu, Yang Zhao, Yin Xia, Zhile Ren, and Zhongzheng Ren. Apple intelligence foundation language models. arXiv preprint arXiv:2407.21075, 2024.

Jacob Hilton, Karl Cobbe, and John Schulman. Batch size-invariance for policy optimization. arXiv preprint arXiv:2110.00641, 2022.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, Xinrong Zhang, Zheng Leng Thai, Kaihuo Zhang, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.

Alexander H¨agele, Elie Bakouch, Atli Kosson, Loubna Ben Allal, Leandro Von Werra, and Martin Jaggi. Scaling laws and compute-optimal training beyond fixed training durations. arXiv preprint arXiv:2405.18392, 2024.

Su Jianlin. Rethinking learning rate and batch size (part 3): Muon, Sep 2025. URL https: //kexue.fm/archives/11285.

Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https: //kellerjordan.github.io/posts/muon/.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. arXiv preprint arXiv:1609.04836, 2017.

Jeonghoon Kim, Byeongchan Lee, Cheonbok Park, Yeontaek Oh, Beomjun Kim, Taehwan Yoo, Seongjin Shin, Dongyoon Han, Jinwoo Shin, and Kang Min Yoo. Peri-ln: Revisiting normalization layer in the transformer architecture. arXiv preprint arXiv:2502.02732, 2025.

Tim Large, Yang Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola, and Jeremy Bernstein. Scalable optimization in the modular norm. arXiv preprint arXiv:2405.14813, 2024.

Houyi Li, Wenzhen Zheng, Qiufeng Wang, Hanshan Zhang, Zili Wang, Shijie Xuyang, Yuantao Fan, Zhenyu Ding, Haoying Wang, Ning Ding, Shuigeng Zhou, Xiangyu Zhang, and Daxin Jiang. Predictable scale: Part i – optimal hyperparameter scaling law in large language model pretraining. arXiv preprint arXiv:2503.04715, 2025.

Shuaipeng Li, Penghao Zhao, Hailin Zhang, Xingwu Sun, Hao Wu, Dian Jiao, Weiyan Wang, Chengjun Liu, Zheng Fang, Jinbao Xue, Yangyu Tao, Bin Cui, and Di Wang. Surge phenomenon in optimal learning rate and batch size scaling. arXiv preprint arXiv:2405.14578, 2024.

Wanchao Liang, Tianyu Liu, Less Wright, Will Constable, Andrew Gu, Chien-Chin Huang, Iris Zhang, Wei Feng, Howard Huang, Junjie Wang, et al. Torchtitan: One-stop pytorch native solution for production ready llm pretraining. In The Thirteenth International Conference on Learning Representations, 2025.

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, and Zhilin Yang. Muon is scalable for llm training. arXiv preprint arXiv:2502.16982, 2025.

Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun, and Boris Ginsburg. ngpt: Normalized transformer with representation learning on the hypersphere. arXiv preprint arXiv:2410.01131, 2025.

Sadhika Malladi, Kaifeng Lyu, Abhishek Panigrahi, and Sanjeev Arora. On the sdes and scaling rules for adaptive gradient algorithms. arXiv preprint arXiv:2205.10287, 2024.

Sam McCandlish, Jared Kaplan, Dario Amodei, and OpenAI Dota Team. An empirical model of large-batch training. arXiv preprint arXiv:1812.06162, 2018.

Meta AI. Introducing llama 4: Advancing multimodal intelligence, 2025. URL https://ai. meta.com/blog/llama-4-multimodal-intelligence/.

Andrei Mircea, Supriyo Chakraborty, Nima Chitsazan, Milind Naphade, Sambit Sahu, Irina Rish, and Ekaterina Lobacheva. Training dynamics underlying language model scaling laws: Loss deceleration and zero-sum learning. arXiv preprint arXiv:2506.05447, 2025.

Laker Newhouse, R. Preston Hess, Franz Cesista, Andrii Zahorodnii, Jeremy Bernstein, and Phillip Isola. Training transformers with enforced lipschitz constants. arXiv preprint arXiv:2507.13338, 2025.

Guilherme Penedo, Hynek Kydl´ıˇcek, Vinko Sabolˇcec, Bettina Messmer, Negar Foroutan, Amir Hossein Kargaran, Colin Raffel, Martin Jaggi, Leandro Von Werra, and Thomas Wolf. Fineweb2: One pipeline to scale them all–adapting pre-training data processing to every language. arXiv preprint arXiv:2506.20920, 2025.

Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, and Volkan Cevher. Training deep learning models with norm-constrained lmos. arXiv preprint arXiv:2502.07529, 2025a.

Thomas Pethick, Wanyun Xie, Mete Erdogan, Kimon Antonakopoulos, Tony Silveti-Falls, and

Volkan Cevher. Generalized gradient norm clipping & non-euclidean (l0,l1)-smoothness. arXiv preprint arXiv:2506.01913, 2025b.

Tomer Porian, Mitchell Wortsman, Jenia Jitsev, Ludwig Schmidt, and Yair Carmon. Resolving discrepancies in compute-optimal scaling of language models. arXiv preprint arXiv:2406.19146, 2025.

Artem Riabinin, Egor Shulgin, Kaja Gruntkowska, and Peter Richt´arik. Gluon: Making muon & scion great again! (bridging theory and practice of lmo-based optimizers for llms). arXiv preprint arXiv:2505.13416, 2025.

Naoki Sato, Hiroki Naganuma, and Hideaki Iiduka. Convergence bound and critical batch size of muon optimizer. arXiv preprint arXiv:2507.01598, 2025.

Andrei Semenov, Matteo Pagliardini, and Martin Jaggi. Benchmarking optimizers for large language model pretraining. arXiv preprint arXiv:2509.01440, 2025.

Yikang Shen, Matthew Stallone, Mayank Mishra, Gaoyuan Zhang, Shawn Tan, Aditya Prasad, Adriana Meza Soria, David D. Cox, and Rameswar Panda. Power scheduler: A batch size and token number agnostic learning rate scheduler. arXiv preprint arXiv:2408.13359, 2024.

Samuel L. Smith and Quoc V. Le. A bayesian perspective on generalization and stochastic gradient descent. arXiv preprint arXiv:1710.06451, 2018.

Dan Su, Kezhi Kong, Ying Lin, Joseph Jennings, Brandon Norick, Markus Kliegl, Mostofa Patwary, Mohammad Shoeybi, and Bryan Catanzaro. Nemotron-cc: Transforming common crawl into a refined long-horizon pretraining dataset. arXiv preprint arXiv:2412.02595, 2025.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, Yichen Feng, Kelin Fu, Bofei Gao, Hongcheng Gao, Peizhong Gao, Tong Gao, Xinran Gu, Longyu Guan, Haiqing Guo, Jianhang Guo, Hao Hu, Xiaoru Hao, Tianhong He, Weiran He, Wenyang He, Chao Hong, Yangyang Hu, Zhenxing Hu, Weixiao Huang, Zhiqi Huang, Zihao Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yongsheng Kang, Guokun Lai, Cheng Li, Fang Li, Haoyang Li, Ming Li, Wentao Li, Yanhao Li, Yiwei Li, Zhaowei Li, Zheming Li, Hongzhan Lin, Xiaohan Lin, Zongyu Lin, Chengyin Liu, Chenyu Liu, Hongzhang Liu, Jingyuan Liu, Junqi Liu, Liang Liu, Shaowei Liu, T. Y. Liu, Tianwei Liu, Weizhou Liu, Yangyang Liu, Yibo Liu, Yiping Liu, Yue Liu, Zhengying Liu, Enzhe Lu, Lijun Lu, Shengling Ma, Xinyu Ma, Yingwei Ma, Shaoguang Mao, Jie Mei, Xin Men, Yibo Miao, Siyuan Pan, Yebo Peng, Ruoyu Qin, Bowen Qu, Zeyu Shang, Lidong Shi, Shengyuan Shi, Feifan Song, Jianlin Su, Zhengyuan Su, Xinjie Sun, Flood Sung, Heyi Tang, Jiawen Tao, Qifeng Teng, Chensi Wang, Dinglu Wang, Feng Wang, Haiming Wang, Jianzhou Wang, Jiaxing Wang, Jinhong Wang, Shengjie Wang, Shuyi Wang, Yao Wang, Yejie Wang, Yiqin Wang, Yuxin Wang, Yuzhi Wang, Zhaoji Wang, Zhengtao Wang, Zhexu Wang, Chu Wei, Qianqian Wei, Wenhao Wu, Xingzhe Wu, Yuxin Wu, Chenjun Xiao, Xiaotong Xie, Weimin Xiong, Boyu Xu, Jing Xu, Jinjing Xu, L. H. Xu, Lin Xu, Suting Xu, Weixin Xu, Xinran Xu, Yangchuan Xu, Ziyao Xu, Junjie Yan, Yuzi Yan, Xiaofei Yang, Ying Yang, Zhen Yang, Zhilin Yang, Zonghan Yang, Haotian Yao, Xingcheng Yao, Wenjie Ye, Zhuorui Ye, Bohong Yin, Longhui Yu, Enming Yuan, Hongbang Yuan, Mengjie Yuan, Haobing Zhan, Dehao Zhang, Hao Zhang, Wanlu Zhang, Xiaobin Zhang, Yangkun Zhang, Yizhi Zhang, Yongting Zhang, Yu Zhang, Yutao Zhang, Yutong Zhang, Zheng Zhang, Haotian Zhao, Yikai Zhao, Huabin Zheng, Shaojie Zheng, Jianren Zhou, Xinyu Zhou, Zaida Zhou, Zhen Zhu, Weiyu Zhuang, and Xinxing Zu. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

Shuche Wang, Fengzhuo Zhang, Jiaxiang Li, Cunxiao Du, Chao Du, Tianyu Pang, Zhuoran Yang, Mingyi Hong, and Vincent YF Tan. Muon outperforms adam in tail-end associative memory learning. arXiv preprint arXiv:2509.26030, 2025.

Kaiyue Wen, David Hall, Tengyu Ma, and Percy Liang. Fantastic pretraining optimizers and where to find them. arXiv preprint arXiv:2509.02046, 2025.

Mitchell Wortsman, Peter J. Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D. CoReyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-dickstein, Kelvin Xu, Jaehoon Lee, Justin Gilmer, and Simon Kornblith. Small-scale proxies for large-scale transformer training instabilities. arXiv preprint arXiv:2309.14322, 2023.

Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer. arXiv preprint arXiv:2203.03466, 2022.

Greg Yang, Dingli Yu, Chen Zhu, and Soufiane Hayou. Tensor programs vi: Feature learning in infinite-depth neural networks. arXiv preprint arXiv:2310.02244, 2023.

Greg Yang, James B. Simon, and Jeremy Bernstein. A spectral condition for feature learning. arXiv preprint arXiv:2310.17813, 2024.

Hanlin Zhang, Depen Morwani, Nikhil Vyas, Jingfeng Wu, Difan Zou, Udaya Ghai, Dean Foster, and Sham Kakade. How does critical batch size scale in pre-training? arXiv preprint arXiv:2410.21676, 2025.

Jingwei Zuo, Maksim Velikanov, Ilyas Chahed, Younes Belkada, Dhia Eddine Rhayem, Guillaume Kunsch, Hakim Hacid, Hamza Yous, Brahim Farhat, Ibrahim Khadraoui, Mugariya Farooq, Giulia Campesan, Ruxandra Cojocaru, Yasser Djilali, Shi Hu, Iheb Chaabane, Puneesh Khanna, Mohamed El Amine Seddik, Ngoc Dung Huynh, Phuc Le Khac, Leen AlQadi, Billel Mokeddem, Mohamed Chami, Abdalgader Abubaker, Mikhail Lubinets, Kacper Piskorski, and Slim Frikha. Falcon-h1: A family of hybrid-head language models redefining efficiency and performance. arXiv preprint arXiv:2507.22448, 2025.

A APPENDIX

- A.1 LLM USAGE

LLMs were used solely to aid in polishing the writing and improving the clarity of exposition. In addition, code-assistant tools were occasionally used for minor programming support, such as code completion and syntax suggestions; they were not employed to design algorithms, generate experiments, or implement the proposed methods from scratch.

- A.2 MODEL TRAINING CONFIGURATION

- • Proxy model, 69M parameters: 4 hidden layers with dmodel = 256, Multi-Head Attention with nheads = 4 and nkv−heads = 4, SwiGLU activation function with MLP expansion factor fMLP = 2.75, RoPE with θ = 10000 (Su et al., 2024), Llama 3 tokenizer with vocabulary size of 128256 (after padding) (Grattafiori et al., 2024), input and output embedding layers are not tied.
- • ×4(12) wider model, 314M (1.3B) parameters: same as proxy, except dmodel = 1024 (3072). In width scaling, we keep fixed dhead = 64 and scale the number of heads accordingly.
- • ×8(32) deeper model, 91M (168M) parameters: same as proxy, except 32 (128) hidden layers.
- • Semi-orthogonal initialization for hidden linear layers and row-wise normalized Gaussian initialization for input/output embedding layers (Pethick et al., 2025a). Initialisation of the last layer of both MLP and attention blocks (those with the output being added with the residual stream) is multiplied by 1/ 2Nlayers.

- • Dropout disabled, no biases in all Linear layers, no weight sharing between input and output embedding layers.
- • norm-everywhere: normalise input to every Linear layer via RMSNorm without learnable parameters with ϵ = 1e−20. Effectively, this corresponds to Pre-LN setup with QK-norm plus three additional normalisation layers: V-norm, O-norm (before output projection matrix in Attention block), and MLP-norm (after SwiGLU and before the last MLP layer). Residual connections, including the ones injecting the input embedding layer information, remain intact.
- • Random seeds:

- – For all proxy model runs in Sec. 3.2 and Sec. 3.3: 30
- – For all width/depth-scaled-up model runs: interleaved 30 + 3034 (every 22 step is 30, every other 22 step is 3034)
- – For layout scans in Fig. 4a and Fig. 16: averaging over 30 + 3034 + 303409 for the three “core” learning rate values ({2−4,2−6,2−8} for B = 32, {2−2,2−4,2−6} for B = 128, {2−1,2−3,2−5} for B = 512), 3034 + 303409 for the rest
- – For layout scans in Fig. 4b: 30

- • torchtitan codebase, (Liang et al., 2025), FSDP2 (Feng et al., 2025), FlashAttention-2 (Dao, 2023)

- A.3 OPTIMIZER CONFIGURATION Except dedicated ablations, we use the following set of hyperparameters:

- • Unconstrained version (without weight decay),
- • Learning rate η: grid with 20.5 step for the proxy model, and 21 step for the width/depthscaled-up models,
- • momentum µ = 0, without Nesterov momentum,
- • no warmup, constant learning rate schedule,
- • ϵ = 1e−20 (used in gradient normalisation),

- • orthogonalization of gradients for hidden layers (∥.∥RMS→RMS norm assumption) with Newton-Schulz algorithm for niter = 5 with original Muon coefficients a,b,c = (3.4445,−4.7750,2.0315) (Jordan et al., 2024).

- A.4 OPTIMAL NORM FITTING & LOSS SMOOTHING

After na¨ıvely taking the empirical optimum across the learning rate grid (e.g. as the one emphasized with dashed black lines in Fig. 1a), we found that the corresponding norm scans, although still indicating norm transfer, are quite noisy (e.g. compare Fig. 2a vs. Fig. 6a). From Fig. 1a we noted that data points in loss vs. norm plot resemble parabola if plotted in log-log scale. Furthermore, we know by design that at initialization (step 0) the output norm equals to 1, and train loss equals to 11.765. With this, we chose to perform a constrained fit with a second-order polynomial function in log-log scale log(loss) = alog(norm)2 + blog(norm) + c, where the free term c is fixed at precisely the loss value at initialization. We do this using weighted least squares fitting with np.linalg.lstsq, where the weighting is done with inverse uncertainties coming from loss smoothing, described below. For robustness, only seven data points around the empirical optimum are used in the fit. The optimal loss and norm values are then extracted as the parabola optimum coordinates. Optimal learning rate is taken from the data point closest to the fitted optimum. Results of such fits for Fig. 2 can be found in Fig. 17.

Since running several random seeds is computationally intensive, we perform loss smoothing to estimate the loss variance and make loss estimates more robust. Essentially, for a given horizon point, instead of taking its loss value, we average it with the previous and next evaluated points (67M tokens away, or e.g. 128 steps from each other with B = 128). Empirically estimated standard deviation is then used in the fits as described above. We employ loss smoothing only for small batch sizes B ≤ 128, as those having large loss variance, and for large token horizons D ≥ 233, to stay in the region of largely converged loss (which can be locally linearly approximated) and therefore not bias the estimate.

In order to get variance estimate in Fig. 3b without running several random seed runs, we vary the fitting procedure outlined above (with/without fitting, with/without loss smoothing, with/without constraint to loss at initialization), thus resulting in 6 total variations. For each of those we track how optimal norm/loss/learning rate changes, and propagate this variance to plotting and downstream analysis.

- A.5 DISTRIBUTED SCION

We implemented a distributed version of Scion/Muon. In this section, we briefly describe the implementation. We assume that the vectorized momentum buffer update is performed before applying the actual weight update.

- A.5.1 DDP-DISCO

As a warm-up, we first consider the DDP case (note that a DDP-based version of Muon has already been implemented in modded-nanogpt10). Our implementation differs slightly from theirs, as we do not explicitly apply communication–computation overlap for DDP.

- Algorithm 1: Disco step ddp

Input: Parameters {pi}Pi=0−1 with P = |{p}|, world size M, local rank r bucket size ← M ; total buckets ← ⌈P/M⌉ ; global updates ← array of length P ;

- /* Step 1: Compute local updates */ for i = 0 to P − 1 do

if i mod M = r then

gi ← GETMOMENTUM(pi) ; ui ← LMO(gi) ; global updates[i] ← ui ;

- /* Step 2: Communicate updates in buckets */ for b = 0 to total buckets −1 do

start idx ← b · M; end idx ← min(start idx + M, P); my idx ← start idx +r; if my idx < end idx then

usend ← global updates[my idx] ; else

usend ← 0

{uj}Mj=0−1 ← ALLGATHER(usend) ; for j = 0 to end idx − start idx −1 do

global updates[start idx +j] ← uj ;

- /* Step 3: Apply updates vectorized */ APPLYUPDATES({pi}Pi=0−1, global updates) ;

### Helper functions:

- • GETMOMENTUM(p): returns the momentum of p from the momentum buffer.
- • LMO(g): runs the LMO based on the chosen norm of p.
- • ALLGATHER(u): gathers one tensor u from each rank in the data-parallel group.
- • APPLYUPDATES({p},{u}): applies the global updates {u} to the parameters {p} in a single vectorized operation.

Notice this version works out-of-the-box for PP+DDP, as we could let each PP(Pipeline Parallelism) stage only manage the parts of the model that the current PP stages needed for forward and backward.

To make it work with TP, one needs to do an extra all-gather in the local update loop.

10https://github.com/KellerJordan/modded-nanogpt

- A.5.2 FSDP-DISCO

Here, “FSDP” refers to a combination of FSDP2 with arbitrary parallelisms, including Data Parallelism (DP), Context Parallelism (CP), Expert Parallelism (EP), Tensor Parallelism (TP), and Pipeline Parallelism (PP). In this section, we restrict our discussion to FSDP and EP (via DP2EP). In principle, there is no need to treat DP and PP separately: one only needs to all-gather the full gradient before communication in the FSDP case to ensure compatibility with TP.

We assume the design of this work, which applies an ∥.∥1→RMS norm for the LLM’s embedding layer and an ∥.∥RMS→∞ norm for the output linear layer. (SignNorm is also acceptable and remains compatible if one strictly follows Scion’s design.)

The FSDP2 implementation in PyTorch shards weights and gradients along the tensor’s first dimension. We discuss Disco under this assumption and further assume that each tensor or matrix corresponds to a single layer. Consequently, fused tensors such as fused QKV in attention layers or fused W13 in SwiGLU are not supported.

Under these hypotheses, we can classify parameters into three groups: embedding, experts, and (pure-)fsdp. For updates, no extra communication is required for embedding and experts parameters, thanks to the Shard(0) strategy in FSDP2.

- Algorithm 2: Disco step embedding

Input: Embedding parameters {pi}Pi=0−1 /* Initialise updates storage */ updates ← array of length P ; /* get momentum and compute LMO update on local shards */ for i = 0 to P − 1 do

gi ← GETMOMENTUM(pi) ; ui ← LMO(gi) ; updates[i] ← ui ;

/* Apply updates vectorized */ APPLYUPDATES({pi}Pi=0−1, updates) ;

- Algorithm 3: Disco step experts

Input: Expert parameters {pi}Pi=0−1, transpose flag transpose /* Initialise updates storage */ updates ← array of length P ; /* get momentum and compute LMO update on local shards */ for i = 0 to P − 1 do

gi ← GETMOMENTUM(pi) ; ui ← BATCHEDLMO(gi; transpose experts = transpose) ; updates[i] ← ui ;

/* Apply updates vectorized */ APPLYUPDATES({pi}Pi=0−1, updates) ;

Noting that MoE expert weights are typically laid out as either (total experts,dout,din) or (total experts,din,dout), we apply a transpose in the latter case to ensure that the output dimension comes first. In an FSDP + DP2EP setting, each gradient passed to LMO is therefore a 3D tensor with layout (local experts,dout,din). Accordingly, SVD or Newton-Schulz-based algorithms must correctly handle batched inputs.

And below is the algorithm for purely fsdp-shard parameters.

- Algorithm 4: Disco step fsdp

Input: FSDP-sharded parameters {pi}Pi=0−1, world size M over fsdp, local rank r bucket size ← M; total buckets ← ⌈P/M⌉; global updates ← array of length P;

for b = 0 to total buckets −1 do start ← b · M; end ← min(start + M, P); my idx ← start + r;

for j = 0 to M − 1 do i ← start + j; if i < end then

gi ← GETMOMENTUM(pi) // row-sharded by FSDP; send list[j] ← gi;

### else

send list[j] ← 0 // zero padding

recv list ← ALLTOALL(send list) g⋆ ← CONCATROWS(recv list) // reconstruct full gradient for pi⋆ u⋆ ← LMO(g⋆)

updates send list ← SPLITROWS(u⋆, M) // split u⋆ by rows; updates recv list ← ALLTOALL(updates send list);

for j = 0 to end − start −1 do

global updates[start + j] ← updates recv list[j];

/* Single vectorized apply */ APPLYUPDATES({pi}Pi=0−1, global updates);

### Helper functions:

- • ALLTOALL(list): list-based ALLTOALL over dp shard cp.

- • CONCATROWS(list): concatenates row-shards into a full tensor.
- • SPLITROWS(u,M): splits u into M contiguous row blocks.

- A.6 OUTPUT NORM EVOLUTION WITH DIFFERENT (η,B)

Output norm vs horizon for different ( , B) combinations

= 7.8e-03, B=32 = 3.1e-02, B=32 = 1.2e-01, B=32 = 5.0e-01, B=32

210

= 7.8e-03, B=128 = 3.1e-02, B=128 = 1.2e-01, B=128 = 5.0e-01, B=128 = 7.8e-03, B=512 = 3.1e-02, B=512 = 1.2e-01, B=512 = 5.0e-01, B=512

28

log||||WoutRMS

26

= 7.8e-03, B=2048 = 3.1e-02, B=2048 = 1.2e-01, B=2048 = 5.0e-01, B=2048

24

22

227 229 231 233 235 237 Horizon

(a)

Output norm vs step for different ( , B) combinations

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

= 7.8e-03, B=32 = 3.1e-02, B=32 = 1.2e-01, B=32 = 5.0e-01, B=32

210

= 7.8e-03, B=128 = 3.1e-02, B=128 = 1.2e-01, B=128 = 5.0e-01, B=128 = 7.8e-03, B=512 = 3.1e-02, B=512 = 1.2e-01, B=512 = 5.0e-01, B=512

28

log||||WoutRMS

26

= 7.8e-03, B=2048 = 3.1e-02, B=2048 = 1.2e-01, B=2048 = 5.0e-01, B=2048

24

22

102 103 104 105 106 Step

(b)

- Figure 5: Growth of the output layer norm ∥Wout∥RMS→∞ vs. horizon, in tokens (a) and number of steps (b). Results are for the proxy model (69M parameters). Each curve is a (learning rate η, batch size B) pair, with B measured in samples: colour encodes batch size and line style encodes learning rate, as described in the legend.

##### A.7 SUPPLEMENTARY PLOTS TO FIG. 2

Scaling horizon for proxy model (69M)

Scaling horizon for proxy model (69M)

5.4

5.4

Horizon [tokens]

Horizon [tokens]

- 229 (0.5 B)

- 230 (1.1 B)

- 231 (2.1 B)

- 232 (4.3 B)

- 233 (8.6 B)

- 234 (17 B)

- 235 (34 B)

- 236 (69 B)

- 237 (137 B)

- 229 (0.5 B)

- 230 (1.1 B)

- 231 (2.1 B)

- 232 (4.3 B)

- 233 (8.6 B)

- 234 (17 B)

- 235 (34 B)

- 236 (69 B)

- 237 (137 B)

5.2

5.2

5.0

5.0

4.8

4.8

Trainloss

Trainloss

4.6

4.6

4.4

4.4

4.2

4.2

Batch size [samples]

Batch size [samples]

4.0

4.0

32 2048

32 2048

3.8

3.8

5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 log ||Wout||RMS

5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 log ||Wout||RMS

(a)

(b)

Scaling horizon for proxy model (69M)

Scaling horizon for proxy model (69M)

5.4

5.4

Horizon [tokens]

Horizon [tokens]

- 229 (0.5 B)

- 230 (1.1 B)

- 231 (2.1 B)

- 232 (4.3 B)

- 233 (8.6 B)

- 234 (17 B)

- 235 (34 B)

- 236 (69 B)

- 237 (137 B)

- 229 (0.5 B)

- 230 (1.1 B)

- 231 (2.1 B)

- 232 (4.3 B)

- 233 (8.6 B)

- 234 (17 B)

- 235 (34 B)

- 236 (69 B)

- 237 (137 B)

5.2

5.2

5.0

5.0

4.8

4.8

Trainloss

Trainloss

4.6

4.6

4.4

4.4

4.2

4.2

Batch size [samples]

Batch size [samples]

4.0

4.0

32 2048

32 2048

3.8

3.8

5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 log ||Wout||RMS

5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 log ||Wout||RMS

(c)

(d)

Scaling width/depth for 233 (8.6 B) token horizon

Scaling width/depth for 233 (8.6 B) token horizon

d=256, L=4 (69M) d=1024, L=4 (314M)

d=256, L=4 (69M) d=1024, L=4 (314M)

d=256, L=32 (91M)

d=256, L=32 (91M)

4.4

4.4

d=3072, L=4 (1.3B)

d=256, L=128 (168M)

d=3072, L=4 (1.3B)

d=256, L=128 (168M)

4.2

4.2

Trainloss

Trainloss

4.0

4.0

3.8

3.8

Batch size [samples]

Batch size [samples]

32 1024

32 1024

3.6

3.6

5 6 7 8 log ||Wout||RMS

5 6 7 8 log ||Wout||RMS

(e)

(f)

- Figure 6: (a) Fig. 2a with an extended set of horizons, raw data (i.e. no loss smoothing, no fitting, see Appendix A.4 for details on fitting and loss smoothing). (b) Same as (a) + loss smoothing. (c) Same as (a) + fitting. (d) Same as (a) + fitting + loss smoothing. (e) Fig. 2b, raw data (no loss smoothing, no fitting). (f) Same as (e) + loss smoothing.

- A.8 OPTIMAL B∗(D) MEASUREMENT

25 26 27 28 29 210 211 B [samples]

2 8

2 7

2 6

2 5

2 4

2 3

2 2

2 1

optimal

Horizon [tokens]

- 229 (0.5 B)

| |
|---|

- 230 (1.1 B)

- 231 (2.1 B)

- 232 (4.3 B)

- 233 (8.6 B)

- 234 (17 B)

- 235 (34 B)

- 236 (69 B)

- 237 (137 B)

log2 = log2 B + log2 D +

= 0.55 ± 0.03, = -0.53 ± 0.02

= 9.1 ± 0.77; R2 = 0.94

(a)

229 230 231 232 233 234 235 236 237 Horizon [tokens]

- 26

- 27

- 28

- 29

210

[samples]B

Fit: B = a * Horizon^b b (exponent) = 0.450 ± 0.066 a (scale) = 0.009 ± 0.013 R² = 0.868

(b)

- Figure 7: (a) Same as Fig. 3b, but with extended set of horizons. (b) Optimal batch size B∗ vs. horizon, as extracted from (a)). The line is a power-law fit (described in legend).

Fig. 3b, for the sake of clarity and simplicity, illustrates only four horizons. This is not really sufficient to extract precisely the scaling of optimal (η∗,B∗) (circled markers) with D, as it would mean fitting of four data points. We therefore perform the ordinary least squares (OLS) fit on the extended set of 9 horizons from Fig. 7a, effectively fitting the x-coordinate of the circled markers with a line. We model optimal batch size dependency on horizon D as a power law B∗(D) = aDb and present results on Fig. 7b. We extract B∗ ∝ D0.45±0.07, consistent with the square-root scaling.

A.9 NORM CONSTRAINT WITH WEIGHT DECAY

21 23 25 27 29 211 213

step

- 20

- 21

- 22

- 23

- 24

- 25

- 26

- 27

||Win1RMS

= 1.0 = 0.1

= 1.0 = 0.01

= 0.0625 = 0.1

= 0.0625 = 0.01

(a)

21 23 25 27 29 211 213

step

- 20

- 21

- 22

- 23

- 24

- 25

- 26

- 27

||WoutRMS

= 1.0 = 0.1

= 1.0 = 0.01

= 0.0625 = 0.1

= 0.0625 = 0.01

(b)

- Figure 8: Operator norm against number of gradient update steps. Fixed batch size B = 32, momentum µ = 0.1, two values of learning rate η = {0.0625,1.} and two values of weight decay λ = {0.01,0.1} (applied as in Pethick et al. (2025a)), for a proxy model (69M parameters). (a)

∥Win∥1→RMS norm (b) ∥Wout∥RMS→∞. We see for λ = 0.1 both norms converging to 1/λ, while for λ = 0.01 asymptotic values are not conclusive.

- A.10 ABLATION OF DEPTH TRANSFER TECHNIQUES

Model: same as our proxy model (Appendix A.2), with the only difference in the head configuration: nquery heads = 2, nkv heads = 1. We run a combination of two ablations: (i) weight initialisation depth-wise scaling (via gains/variance), and (ii) residual branch summation ratios.

For weight initialisation, the depth-wise scaling factors are applied to only the output linear projection of attention and SwiGLU. We compare three flavours of depth init scaling: identity (baseline), total-depth, and relative-depth, defined by multiplying the gain σ by

 

1/ 2Nlayers scale by total-depth, 1/√2li scale by relative-depth, 1 scale by identity.

(11)

σ∗ =



where Nlayers is the total number of Transformer blocks, and li ∈ {1,...,2Nlayers} is the relative depth of the current block; σ is the scaled orthogonal gain, σ = d

din , for hidden weights W ∈ Rd

out

out×din.

Each transformer block is assigned depth 2, since attention and FFN sub-blocks each count as depth 1. When using relative-depth, the depth of all FFN blocks can be offset by 1.

For depth-wise residual scaling, we write the residual connection in transformer as:

#### Y = α · X + β · Block Norm(X) , (12)

where X is the block input and Block(·) denotes either self-attention or a FFN, and Norm is RMSNorm in our setup.

We consider three depth-wise residual scaling schemes:

 

2Nlayers−1

scale by depth-normalized, (1, 2N1

2Nlayers , 2N1

layers

(13)

) scale by completeP, (1,1) scale by identity.

(α,β) =



layers

depth-normalized Large et al. (2024) scales both the residual and block contributions proportionally to depth. completeP Dey et al. (2025) preserves the residual branch while scaling down the block contribution by depth. identity corresponds to the conventional unscaled residual formulation.

We fixed batch size (B) to 32 samples, the sequence length to 4096, and the number of training steps to 2048. Experiments were conducted using proxy models with depths Nlayers ∈ {2,16,64}. For all models, we performed a sweep over the learning rate {2−4,2−3,2−2,2−1,20}.

We report the final-step losses in Table 1, Table 2, and Table 3 for the three depths, respectively, with the two lowest losses highlighted. From the perspective of learning rate transfer, we find that with our optimizer, the optimal learning rate consistently remains around 2−2, regardless of weight initialisation or residual scaling. We also observe that combining total-depth weight initialisation with identity residual scaling yields a negligible improvement compared to using identity weight initialisation.

Table 1: 2 layers (B = 32, steps=2048)

Learning rate η 2−4 2−3 2−2 2−1 20

Residual init Residual multiplier

total-depth identity 4.20 4.11 4.09 4.13 4.22 total-depth depth-normalized 4.20 4.12 4.11 4.17 4.21 total-depth completeP 4.22 4.15 4.16 4.17 4.28 identity identity 4.19 4.10 4.10 4.13 4.23 identity depth-normalized 4.22 4.12 4.12 4.13 4.21 identity completeP 4.21 4.15 4.13 4.16 4.24 relative-depth identity 4.20 4.11 4.09 4.13 4.23 relative-depth depth-normalized 4.20 4.13 4.11 4.16 4.25 relative-depth completeP 4.21 4.16 4.14 4.18 4.24

Table 2: 16 layers (B = 32, steps=2048)

Learning rate η 2−4 2−3 2−2 2−1 20

Residual init Residual multiplier

total-depth identity 3.81 3.75 3.73 3.77 3.88 total-depth depth-normalized 3.85 3.79 3.80 3.84 3.92 total-depth completeP 3.87 3.82 3.82 3.85 3.94 identity identity 3.81 3.74 3.75 3.79 3.89 identity depth-normalized 3.83 3.78 3.78 3.83 3.92 identity completeP 3.86 3.81 3.81 3.85 3.94 relative-depth identity 3.82 3.79 3.74 3.80 3.90 relative-depth depth-normalized 3.84 3.79 3.80 3.83 3.95 relative-depth completeP 3.88 3.82 3.82 3.85 3.95

Table 3: 64 layers (B=32, steps=2048)

Learning rate η 2−4 2−3 2−2 2−1 20

Residual init Residual multiplier

total-depth identity 3.67 3.60 3.60 3.65 3.79 total-depth depth-normalized 3.71 3.65 3.65 3.69 3.80 total-depth completeP 3.72 3.67 3.67 3.72 3.82 identity identity 3.70 3.63 3.62 3.66 3.78 identity depth-normalized 3.70 3.64 3.64 3.69 3.80 identity completeP 3.70 3.70 3.67 3.72 3.82 relative-depth identity 3.70 3.61 3.61 3.67 3.82 relative-depth depth-normalized 3.71 3.65 3.65 3.69 3.80 relative-depth completeP 3.72 3.68 3.67 3.73 3.83

##### A.11 ABLATION WITH FINEWEB-2 DATASET

Proxy model (69M), Scion, FineWeb-2 (th)

Proxy model (69M), Scion, FineWeb-2 (ru)

4.4

3.4

Horizon [tokens]

Horizon [tokens]

231 (2.1 B) 233 (8.6 B)

235 (34 B)

231 (2.1 B) 233 (8.6 B)

235 (34 B)

4.2

3.2

4.0

3.0

Batch size [samples]

Batch size [samples]

3.8

Trainloss

Trainloss

32 1024

8 1024

2.8

3.6

3.4

2.6

3.2

2.4

3.0

2.2

2.8

5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 log ||Wout||RMS

5 6 7 8 9 10 log ||Wout||RMS

(a)

(b)

- Figure 9: Same as Fig. 2a but using the FineWeb-2 dataset: (a) Thai partition, (b) Russian partition. We note that while for the Russian language the three horizon curves are nested inside of each other and share the same optimal norm, for the Thai partition the first horizon is off. This may be due to being an early phase of training or statistical fluctuation. For completeness, we provide the individual learning rate scans and fits used to produce this plot in Fig. 10.

23 24 25 26 27 28

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

221B

bs=32

a=0.121, b=-1.702, c=8.811

min: log2 x 7.02, y 2.84

23 24 25 26 27 28

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

bs=64

a=0.087, b=-1.215, c=6.906

min: log2 x 6.95, y 2.68

23 24 25 26 27 28

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

bs=128

a=0.067, b=-0.874, c=5.462

min: log2 x 6.55, y 2.6

23 24 25 26 27 28

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

bs=256

a=0.128, b=-1.614, c=7.824

min: log2 x 6.33, y 2.72

23 24 25 26 27 28

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

bs=512

a=0.208, b=-2.377, c=9.635

min: log2 x 5.72, y 2.83

23 24 25 26 27 28

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

bs=1024

a=0.230, b=-2.466, c=9.755

min: log2 x 5.35, y 3.16

24 25 26 27 28 29

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

286B

a=0.058, b=-0.899, c=5.940 (WLS)

min: log2 x 7.69, y 2.48

24 25 26 27 28 29

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

a=0.077, b=-1.169, c=6.890 (WLS)

min: log2 x 7.62, y 2.43

24 25 26 27 28 29

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

a=0.075, b=-1.077, c=6.326 (WLS)

min: log2 x 7.21, y 2.44

24 25 26 27 28 29

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

a=0.089, b=-1.227, c=6.733 (WLS)

min: log2 x 6.93, y 2.48

24 25 26 27 28 29

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

a=0.135, b=-1.768, c=8.317 (WLS)

min: log2 x 6.54, y 2.53

24 25 26 27 28 29

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

a=0.222, b=-2.698, c=10.832 (WLS)

min: log2 x 6.09, y 2.62

26 27 28 29

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

234B

a=0.031, b=-0.534, c=4.651 (WLS)

min: log2 x 8.52, y 2.37

26 27 28 29

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

a=0.119, b=-1.981, c=10.620 (WLS)

min: log2 x 8.34, y 2.36

26 27 28 29

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

a=0.083, b=-1.323, c=7.656 (WLS)

min: log2 x 8.01, y 2.35

26 27 28 29

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

a=0.081, b=-1.236, c=7.045 (WLS)

min: log2 x 7.59, y 2.35

26 27 28 29

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

a=0.106, b=-1.520, c=7.848 (WLS)

min: log2 x 7.2, y 2.38

26 27 28 29

2.25

2.50

2.75

3.00

3.25

3.50

3.75

4.00

a=0.145, b=-1.959, c=9.040 (WLS)

min: log2 x 6.75, y 2.43

(a)

21 23 25 27 29

3.0

3.5

4.0

4.5

5.0

5.5

B

bs=8

a=0.138, b=-2.279, c=13.098

min: log2 x 8.23, y 3.72

21 23 25 27 29

3.0

3.5

4.0

4.5

5.0

5.5

bs=16

a=0.134, b=-2.069, c=11.234

min: log2 x 7.74, y 3.22

21 23 25 27 29

3.0

3.5

4.0

4.5

5.0

5.5

bs=32

a=0.097, b=-1.411, c=8.335

min: log2 x 7.27, y 3.2

21 23 25 27 29

3.0

3.5

4.0

4.5

5.0

5.5

bs=64

a=0.118, b=-1.660, c=9.051

min: log2 x 7.04, y 3.21

21 23 25 27 29

3.0

3.5

4.0

4.5

5.0

5.5

bs=128

a=0.129, b=-1.767, c=9.493

min: log2 x 6.84, y 3.45

21 23 25 27 29

3.0

3.5

4.0

4.5

5.0

5.5

bs=256

a=0.202, b=-2.593, c=11.792

min: log2 x 6.43, y 3.45

21 23 25 27 29

3.0

3.5

4.0

4.5

5.0

5.5

bs=512

a=0.247, b=-2.986, c=12.706

min: log2 x 6.05, y 3.67

21 23 25 27 29

3.0

3.5

4.0

4.5

5.0

5.5

bs=1024

a=0.292, b=-3.233, c=13.020

min: log2 x 5.54, y 4.06

21 22 23 24 25 26 27 28 29

3.0

3.5

4.0

4.5

5.0

5.5

B

min: log2 x 9.26, y 3.35

21 22 23 24 25 26 27 28 29

3.0

3.5

4.0

4.5

5.0

5.5

a=0.130, b=-2.200, c=12.451 (WLS)

min: log2 x 8.44, y 3.16

21 22 23 24 25 26 27 28 29

3.0

3.5

4.0

4.5

5.0

5.5

a=0.091, b=-1.439, c=8.791 (WLS)

min: log2 x 7.95, y 3.07

21 22 23 24 25 26 27 28 29

3.0

3.5

4.0

4.5

5.0

5.5

a=0.084, b=-1.268, c=7.869 (WLS)

min: log2 x 7.55, y 3.08

21 22 23 24 25 26 27 28 29

3.0

3.5

4.0

4.5

5.0

5.5

a=0.089, b=-1.286, c=7.786 (WLS)

min: log2 x 7.25, y 3.12

21 22 23 24 25 26 27 28 29

3.0

3.5

4.0

4.5

5.0

5.5

a=0.107, b=-1.483, c=8.311 (WLS)

min: log2 x 6.91, y 3.19

21 22 23 24 25 26 27 28 29

3.0

3.5

4.0

4.5

5.0

5.5

a=0.159, b=-2.113, c=10.276 (WLS)

min: log2 x 6.64, y 3.26

21 22 23 24 25 26 27 28 29

3.0

3.5

4.0

4.5

5.0

5.5

a=0.301, b=-3.850, c=15.706 (WLS)

min: log2 x 6.4, y 3.39

22 23 24 25 26 27 28 29 210

3.0

3.5

4.0

4.5

5.0

5.5

B

min: log2 x 9.67, y 3.39

22 23 24 25 26 27 28 29 210

3.0

3.5

4.0

4.5

5.0

5.5

min: log2 x 9.44, y 3.14

22 23 24 25 26 27 28 29 210

3.0

3.5

4.0

4.5

5.0

5.5

a=0.162, b=-2.620, c=13.589 (WLS)

min: log2 x 8.07, y 3.02

22 23 24 25 26 27 28 29 210

3.0

3.5

4.0

4.5

5.0

5.5

a=0.060, b=-0.987, c=7.079 (WLS)

min: log2 x 8.21, y 3.03

22 23 24 25 26 27 28 29 210

3.0

3.5

4.0

4.5

5.0

5.5

a=0.055, b=-0.862, c=6.399 (WLS)

min: log2 x 7.8, y 3.04

22 23 24 25 26 27 28 29 210

3.0

3.5

4.0

4.5

5.0

5.5

a=0.095, b=-1.422, c=8.330 (WLS)

min: log2 x 7.49, y 3.01

22 23 24 25 26 27 28 29 210

3.0

3.5

4.0

4.5

5.0

5.5

a=0.138, b=-1.965, c=10.068 (WLS)

min: log2 x 7.13, y 3.07

22 23 24 25 26 27 28 29 210

3.0

3.5

4.0

4.5

5.0

5.5

a=0.177, b=-2.394, c=11.208 (WLS)

min: log2 x 6.75, y 3.13

(b)

- Figure 10: Individual ∥Wout∥RMS→∞ norm scans for various batch sizes B (columns) across various horizons D (rows), for the FineWeb-2 dataset: (a) Thai partition, (b) Russian partition.

##### A.12 ABLATION WITH ADAM OPTIMIZER

Proxy model (69M), Adam ( 1 = 2 = 0)

Proxy model (69M), Adam ( 1 = 0.9, 2 = 0.95)

4.5

6.5

Horizon [tokens]

Horizon [tokens]

4.4

231 (2.1 B) 233 (8.6 B)

235 (34 B)

231 (2.1 B) 233 (8.6 B)

235 (34 B)

235+decay (43 B)

235+decay (43 B)

6.0

4.3

4.2

5.5

Trainloss

Trainloss

4.1

5.0

4.0

3.9

4.5

3.8

Batch size [samples]

Batch size [samples]

4.0

32 4096

32 4096

3.7

5 6 7 8 9 log ||Wout||RMS

6 7 8 9 10 11 log ||Wout||RMS

(a)

(b)

- Figure 11: Same as Fig. 2a but using the Adam optimizer (no weight decay, warmup for 229 ≈ 537M tokens across all batch sizes, followed by constant learning rate schedule with linear decay to

0 for 20% of the total schedule): (a) β1 = β2 = 0, (b) β1 = 0.9, β2 = 0.95. For the no-momentum version (a) we observe norm transfer at the same optimal norm value as our main experiment with Scion (Fig. 2a). For the version with momentum (b), norm transfer is also present (with reduced sensitivity to batch size choice similarly to Scion), but notably at a higher optimal norm value.

A.13 RESIDUALS-ONLY DATA SCALING

- 2.0 2.5 3.0 3.5 4.0 log ||Wout||RMS

- 3.6

3.8

4.0

- 4.2

4.4

Trainloss

Proxy model (69M), Scion, residuals-only norm

Horizon [tokens]

231 (2.1 B) 233 (8.6 B)

235 (34 B)

235+decay (43 B)

Batch size [samples]

32 4096

- Figure 12: Same as Fig. 2a but leaving in the proxy model only residuals normalization layers (RMSNorm without trainable parameters normalizing inputs to every attention and MLP blocks).

##### A.14 NORMALIZATION LAYERS

8.0

Everywhere [A: QKVO; MLP: +, R: +, O: +] Everywhere w/o out [A: QKVO; MLP: +, R: +, O: -] Residuals + out + QK [A: QK, MLP: -, R: +, O: +] Residuals + out [A: -, MLP: -, R: +, O: +]

| | | |
|---|---|---|
| | | |

4.15

7.5

4.10

| |
|---|

Residuals only [A: -, MLP: -, R: +, O: -]

7.0

QK only [A: QK, MLP: -, R: -, O: -] Out only [A: -, MLP: -, R: -, O: +] None [A: -, MLP: -, R: -, O: -]

4.05

6.5

Trainingloss

4.00

Everywhere

2 8 2 6 2 4 2 2

6.0

- LR=2^-5: norm=162.5, loss=3.98 LR=2^-4: norm=213.1, loss=4.01
- LR=2^-6: norm=113.1, loss=4.03

Everywhere w/o out

- LR=2^-7: norm=13.8, loss=4.10
- LR=2^-8: norm=12.6, loss=4.11 LR=2^-5: norm=7.7, loss=4.11

5.5

5.0

###### Residuals + out

4.5

- LR=2^-5: norm=nan, loss=3.99 LR=2^-4: norm=nan, loss=3.99
- LR=2^-6: norm=nan, loss=4.03

| |
|---|

| |
|---|

| |
|---|

4.0

| |
|---|

###### Residuals only

- LR=2^-6: norm=3.4, loss=4.17
- LR=2^-7: norm=2.3, loss=4.19 LR=2^-4: norm=18.6, loss=4.23

2 10 2 8 2 6 2 4 2 2

Learning rate

(a)

4.6

Everywhere [A: QKVO; MLP: +, R: +, O: +] Everywhere w/o out [A: QKVO; MLP: +, R: +, O: -] Residuals + out + QK [A: QK, MLP: -, R: +, O: +] Residuals + out [A: -, MLP: -, R: +, O: +]

3.90

| | | |
|---|---|---|
| | | |

3.85

3.80

Residuals only [A: -, MLP: -, R: +, O: -]

4.4

QK only [A: QK, MLP: -, R: -, O: -] Out only [A: -, MLP: -, R: -, O: +]

3.75

| |
|---|

Trainingloss

| |
|---|

3.70

| |
|---|

Everywhere

4.2

LR=2^-2: norm=201.3, loss=3.70

2 8 2 5 2 2

- LR=2^-4: norm=124.0, loss=3.71
- LR=2^-5: norm=95.6, loss=3.71

Everywhere w/o out

- LR=2^-6: norm=14.9, loss=3.71
- LR=2^-7: norm=9.1, loss=3.72
- LR=2^-8: norm=6.6, loss=3.73

4.0

###### Residuals + out

- LR=2^-5: norm=94.9, loss=3.69
- LR=2^-6: norm=77.3, loss=3.69
- LR=2^-7: norm=77.5, loss=3.71

Residuals only

- LR=2^-8: norm=4.6, loss=3.69 LR=2^-7: norm=7.7, loss=3.70
- LR=2^-9: norm=3.1, loss=3.74

| |
|---|

3.8

| |
|---|

| |
|---|

| |
|---|

2 9 2 7 2 5 2 3

Learning rate

(b)

- Figure 13: Learning rate scan for various layer normalization strategies for the proxy model and the Scion optimizer: (a) without momentum α = 1.0, (b) with momentum α = 0.1. All runs share the same batch size B = 256 [sequences] and the same learning rate schedule: no warmup, constant phase (235 tokens), linear decay to 0 for 20% of the total horizon (0.25 × 235 tokens). Learning rate plotted on the X axis corresponds to the learning rate of the constant phase. Markers which are not present on the plot mean that the training diverged. Notations in the legend corresponds to normalising 1) for the attention block (A) QKV: outputs of the QKV projection matrices, O: inputs to the output projection matrix; 2) MLP: inputs to the last linear layer; 3) residuals (R): inputs to both attention and MLP blocks; 4) Output layer (O): inputs to the final layer projecting the model dimension onto the vocabulary. Additional legend block shows

the final norm (∥Wout∥RMS→∞) value and training loss for the top-3 optimal learning rate runs within each normalization strategy.

- A.15 ABLATIONS ON FIG. 2A

- A.15.1 MOMENTUM & LEARNING RATE DECAY

In this set of experiments we set momentum to 0.1 (which is by default disabled in the main text) and firstly run the same horizon scaling experiment for the proxy model (69M parameters) with the constant learning rate schedule and evaluate at the same horizons D = {231,233,235,237} as Fig. 2a. The results are presented in Fig. 14a. Here we perform loss smoothing in the same way as for the no-momentum scenario, but do not perform the fitting, i.e. for each batch size we take the optimal norm from the empirically best performing learning rate run. We find that the curves look more like “blobs”, where multiple batch sizes give almost the same performance and are centered around the optimal norm (which also transfers across horizons). Also the loss difference between horizons is not well-pronounced as in the no-momentum scenario.

Then, we add learning rate decay, where we start from checkpoints of the horizons specified above, assume that that constitutes 75% of the total horizon, and linearly decay learning rate to 0 for the rest 25%. Likewise, we smooth loss values and take optimum value per batch size across empirical ones on the learning rate grid. In Fig. 14b we see that there is potentially a slight drift of the optimal norm with horizon scaling. However, after examining individual scans (Fig. 18) we surprisingly found that for long horizons the learning rate decay smooths out the norm optimum: a broad range (factor ×4 − 8 in norm) of learning rates results in the same loss. Hence, there is no longer a single optimal norm, but rather a sizeable range, indicating that learning rate decay significantly reduces norm sensitivity. Therefore, we conclude that the seaming drift in Fig. 14b is not significant.

Scaling horizon for proxy model (69M)

4.2

Horizon [tokens]

231 (2.1 B) 233 (8.6 B)

235 (34 B)

4.1

237 (137 B)

4.0

3.9

Trainloss

3.8

3.7

Batch size [samples]

3.6

32 1024

3.5

5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 log ||Wout||RMS

(a)

Scaling horizon for proxy model (69M)

4.2

Horizon [tokens]

231 + decay (2.9 B)

235 + decay (46 B)

4.1

233 + decay (11 B)

237 + decay (184 B)

4.0

3.9

Trainloss

3.8

3.7

Batch size [samples]

3.6

32 1024

3.5

5 6 7 8 9 log ||Wout||RMS

(b)

- Figure 14: Same as Fig. 2a but with momentum = 0.1. (a) Without learning rate decay. (b) With linear learning rate decay to 0 for extra 25% of the total horizon.

- A.15.2 NORM CHOICE

In this Section, we ablate if it is only the output layer norm that induces norm transfer. We replot Fig. 2a, with loss smoothing and without fitting, but now where we use ∥.∥RMS→RMS norm of the output or ∥.∥1→RMS of the input layers instead default ∥.∥RMS→∞ of the output layer. We observe in Fig. 15 (see also individual norm scans in Fig. 19) that interestingly both norms induce norm transfer.

Scaling horizon for proxy model (69M)

5.4

Horizon [tokens]

231 (2.1 B) 233 (8.6 B)

235 (34 B)

5.2

237 (137 B)

5.0

4.8

Trainloss

4.6

4.4

4.2

Batch size [samples]

4.0

32 2048

3.8

4 5 6 7 8 log ||Wout||RMS RMS

(a)

Scaling horizon for proxy model (69M)

5.4

Horizon [tokens]

231 (2.1 B) 233 (8.6 B)

235 (34 B)

5.2

237 (137 B)

5.0

4.8

Trainloss

4.6

4.4

4.2

Batch size [samples]

4.0

32 2048

3.8

1 2 3 4 5 6 7 8 log ||Win||1 RMS

(b)

### Figure 15: Same as Fig. 2a but with ∥Wout∥RMS→∞ norm for the X-axis changed to: (a) ∥Wout∥RMS→RMS (output layer). (b) ∥Win∥1→RMS (input layer).

A.16 LEARNING RATE LAYOUT FOR ADDITIONAL BATCH SIZES AND HORIZONS

Best configurations for horizon=231 (2.1 B), B=32

Top 10% loss: [4.49, 4.66]

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

Others: [4.68, 5.89]

2 2

- Top 1: 4.49 ± 0.01

- Top 2: 4.54 ± 0.04

- Top 3: 4.56 ± 0.01 Equal LRs at 2 4: 4.54 ± 0.04

2 4

Top-10% loss counts per layer group

6

2 6

4

count

2

2 8

0

2 6 2 4 2 2

input hidden output

2 10

| |
|---|

| |
|---|

input hidden output

(a)

Best configurations for horizon=233 (8.6 B), B=32

Top 10% loss: [4.53, 4.60]

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

Others: [4.60, 5.64]

2 2

- Top 1: 4.53 ± 0.01

- Top 2: 4.55 ± 0.02

- Top 3: 4.57 ± 0.04 Equal LRs at 2 6: 4.61 ± 0.00

2 4

Top-10% loss counts per layer group

6

2 6

4

count

2

2 8

0

2 8 2 6 2 4 2 2

input hidden output

2 10

| |
|---|

| |
|---|

input hidden output

(d)

Best configurations for horizon=231 (2.1 B), B=128

Top 10% loss: [4.32, 4.43]

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

Others: [4.43, 5.69]

20

- Top 1: 4.32 ± 0.01

- Top 2: 4.33 ± 0.01

- Top 3: 4.37 ± 0.01 Equal LRs at 2 2: 4.43 ± 0.01

2 2

Top-10% loss counts per layer group

8

2 4

6

count

4

2 6

2

0

2 6 2 4 2 2 20

input hidden output

2 8

| |
|---|

| |
|---|

input hidden output

(b)

Best configurations for horizon=233 (8.6 B), B=128

Top 10% loss: [4.11, 4.18]

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

Others: [4.19, 4.76]

20

- Top 1: 4.11 ± 0.01

- Top 2: 4.12 ± 0.01

- Top 3: 4.12 ± 0.02 Equal LRs at 2 4: 4.13 ± 0.00

2 2

Top-10% loss counts per layer group

8

2 4

6

count

4

2 6

2

0

2 8 2 6 2 4 2 2 20

input hidden output

2 8

| |
|---|

| |
|---|

input hidden output

(e)

Best configurations for horizon=231 (2.1 B), B=512

Top 10% loss: [4.40, 4.59]

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

Others: [4.59, 6.45]

21

- Top 1: 4.40 ± 0.02

- Top 2: 4.42 ± 0.02

- Top 3: 4.43 ± 0.03 Equal LRs at 2 1: 4.67 ± 0.06

2 1

Top-10% loss counts per layer group

6

2 3

count

4

2

2 5

0

2 5 2 3 2 1 21

input hidden output

2 7

| |
|---|

| |
|---|

input hidden output

(c)

Best configurations for horizon=233 (8.6 B), B=512

Top 10% loss: [4.08, 4.16]

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

Others: [4.17, 5.16]

21

- Top 1: 4.08 ± 0.01

- Top 2: 4.09 ± 0.01

- Top 3: 4.10 ± 0.00 Equal LRs at 2 1: 4.22 ± 0.02

2 1

Top-10% loss counts per layer group

6

2 3

count

4

2

2 5

0

2 7 2 5 2 3 2 1 21

input hidden output

2 7

| |
|---|

| |
|---|

input hidden output

(f)

##### Figure 16: Extended version of Fig. 4a with additional batch sizes and horizons. Top (bottom) row: D = 231(233) token horizons. Batch sizes, in samples: B = 32 (left), B = 128 (middle), B = 512 (right). Performance is averaged across random seeds as described in Appendix A.2. Note that the optimal B∗ is 128 for both D = 231 and D = 233 according to Fig. 3b.

##### A.17 INDIVIDUAL NORM SCANS AND FITS

bs=32

bs=64

bs=128

bs=256

bs=512

bs=1024

bs=2048

6.00

6.00

6.00

6.00

6.00

6.00

6.00

| | |a=0.160 min: log|, b=-2. 2 x 6.7<br><br>|155, c= 3, y<br><br>|11.765 4.51| | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

5.75

5.75

5.75

5.75

5.75

5.75

5.75

5.50

5.50

5.50

5.50

5.50

5.50

5.50

5.25

5.25

5.25

5.25

5.25

5.25

5.25

221B

5.00

5.00

5.00

5.00

5.00

5.00

5.00

4.75

4.75

4.75

4.75

4.75

4.75

4.75

4.50

4.50

4.50

4.50

4.50

4.50

4.50

4.25

4.25

4.25

4.25

4.25

4.25

4.25

4.00

4.00

4.00

4.00

4.00

4.00

4.00

a=0.123, b=-1.879, c=11.765

a=0.140, b=-2.026, c=11.765

a=0.154, b=-2.128, c=11.765

a=0.185, b=-2.286, c=11.765

a=0.194, b=-2.285, c=11.765

a=0.215, b=-2.342, c=11.765

min: log2 x 7.65, y 4.58

min: log2 x 7.26, y 4.41

min: log2 x 6.92, y 4.4

min: log2 x 6.18, y 4.7

min: log2 x 5.88, y 5.05

min: log2 x 5.45, y 5.39

3.75

3.75

3.75

3.75

3.75

3.75

3.75

21 22 23 24 25 26 27 28 29

21 22 23 24 25 26 27 28 29

21 22 23 24 25 26 27 28 29

21 22 23 24 25 26 27 28 29

21 22 23 24 25 26 27 28 29

21 22 23 24 25 26 27 28 29

21 22 23 24 25 26 27 28 29

6.00

6.00

6.00

6.00

6.00

6.00

6.00

| |a= mi<br><br>|0.159, b=-2.19 n: log2 x 6.91,<br><br>|3, c=11.765 y 4.19<br><br>| | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| |a= mi<br><br>|0.186, b=-2.31 n: log2 x 6.23,<br><br>|5, c=11.765 y 4.55<br><br>| | | | |
| | | | | | | | |
| | | | | | | | |

a=0.116, b=-1.823, c=11.765 (WLS)

min: log2 x 7.87, y 4.59

5.75

5.75

5.75

5.75

5.75

5.75

5.75

5.50

5.50

5.50

5.50

5.50

5.50

5.50

5.25

5.25

5.25

5.25

5.25

5.25

5.25

286B

5.00

5.00

5.00

5.00

5.00

5.00

5.00

4.75

4.75

4.75

4.75

4.75

4.75

4.75

4.50

4.50

4.50

4.50

4.50

4.50

4.50

4.25

4.25

4.25

4.25

4.25

4.25

4.25

4.00

4.00

4.00

4.00

4.00

4.00

4.00

a=0.128, b=-1.960, c=11.765 (WLS)

a=0.143, b=-2.088, c=11.765 (WLS)

a=0.181, b=-2.334, c=11.765

a=0.194, b=-2.406, c=11.765

min: log2 x 7.64, y 4.28

min: log2 x 7.29, y 4.15

min: log2 x 6.44, y 4.24

min: log2 x 6.21, y 4.29

3.75

3.75

3.75

3.75

3.75

3.75

3.75

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

6.00

6.00

6.00

6.00

6.00

6.00

6.00

| | |a=0.113, min: log2|b=-1.80 x 8, y<br><br>|2, c=11. 4.56|765 (WLS|)| | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | |a=0.127, min: log2|b=-1.97 x 7.75,<br><br>|2, c=11. y 4.12<br><br>|765 (W|LS)| | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | |a=0.146, min: log2|b=-2.12 x 7.28,<br><br>|7, c=11 y 4.02<br><br>|.765| | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

| | |a=0.164, min: log2|b=-2.26 x 6.89,<br><br>|3, c=11. y 3.97<br><br>|765| | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | | | | |m|a=0.187, in: log2 x|b=-2.407, 6.43, y<br><br>|c=11.76 4.03<br><br>|5|
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | | | | | |a= mi|0.181, n: log2 x|b=-2.355, 6.52, y<br><br>|c=11.76 4.09<br><br>|5|
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

a=0.118, b=-1.876, c=11.765 (WLS)

min: log2 x 7.97, y 4.28

5.75

5.75

5.75

5.75

5.75

5.75

5.75

5.50

5.50

5.50

5.50

5.50

5.50

5.50

5.25

5.25

5.25

5.25

5.25

5.25

5.25

5.00

5.00

5.00

5.00

5.00

5.00

5.00

###### 234B

4.75

4.75

4.75

4.75

4.75

4.75

4.75

4.50

4.50

4.50

4.50

4.50

4.50

4.50

4.25

4.25

4.25

4.25

4.25

4.25

4.25

4.00

4.00

4.00

4.00

4.00

4.00

4.00

3.75

3.75

3.75

3.75

3.75

3.75

3.75

23 24 25 26 27 28 29 210

23 24 25 26 27 28 29 210

23 24 25 26 27 28 29 210

23 24 25 26 27 28 29 210

23 24 25 26 27 28 29 210

23 24 25 26 27 28 29 210

23 24 25 26 27 28 29 210

6.00

6.00

6.00

6.00

6.00

6.00

6.00

| | | | | | | |a=0.147, b=in: log2 x<br><br>|2.153, c=1 7.33, y 3.8<br><br>|1.765 8|
|---|---|---|---|---|---|---|---|---|---|
| | | | | | |m| | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | | | |a=0 min:<br><br>|.164, b=-2 log2 x 6<br><br>|.275, c=1 .95, y 3.8<br><br>|1.765 6|
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | |a=0 min:<br><br>|.172, b=-2 log2 x 6<br><br>|.324, c=1 .75, y 3.9<br><br>|1.765<br>2<br>|
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

a=0.102, b=-1.697, c=11.765 (WLS)

a=0.117, b=-1.849, c=11.765 (WLS)

a=0.132, b=-1.998, c=11.765 (WLS)

a=0.133, b=-2.033, c=11.765

min: log2 x 8.36, y 4.67

min: log2 x 7.88, y 4.47

min: log2 x 7.58, y 4.19

min: log2 x 7.67, y 3.97

5.75

5.75

5.75

5.75

5.75

5.75

5.75

5.50

5.50

5.50

5.50

5.50

5.50

5.50

5.25

5.25

5.25

5.25

5.25

5.25

5.25

2137B

5.00

5.00

5.00

5.00

5.00

5.00

5.00

4.75

4.75

4.75

4.75

4.75

4.75

4.75

4.50

4.50

4.50

4.50

4.50

4.50

4.50

4.25

4.25

4.25

4.25

4.25

4.25

4.25

4.00

4.00

4.00

4.00

4.00

4.00

4.00

3.75

3.75

3.75

3.75

3.75

3.75

3.75

24 25 26 27 28 29 210

24 25 26 27 28 29 210

24 25 26 27 28 29 210

24 25 26 27 28 29 210

24 25 26 27 28 29 210

24 25 26 27 28 29 210

24 25 26 27 28 29 210

(a)

bs=32

bs=64

bs=128

bs=256

bs=512

bs=1024

5.50

5.50

5.50

5.50

5.50

5.50

a=0.159, b=-2.193, c=11.765

min: log2 x 6.91, y 4.19

5.25

5.25

5.25

5.25

5.25

5.25

5.00

5.00

5.00

5.00

5.00

5.00

4.75

4.75

4.75

4.75

4.75

4.75

286B

4.50

4.50

4.50

4.50

4.50

4.50

4.25

4.25

4.25

4.25

4.25

4.25

4.00

4.00

4.00

4.00

4.00

4.00

3.75

3.75

3.75

3.75

3.75

3.75

a=0.116, b=-1.823, c=11.765 (WLS)

a=0.128, b=-1.960, c=11.765 (WLS)

a=0.143, b=-2.088, c=11.765 (WLS)

a=0.181, b=-2.334, c=11.765

a=0.194, b=-2.406, c=11.765

3.50

3.50

3.50

3.50

3.50

3.50

min: log2 x 7.87, y 4.59

min: log2 x 7.64, y 4.28

min: log2 x 7.29, y 4.15

min: log2 x 6.44, y 4.24

min: log2 x 6.21, y 4.29

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

bs=32

bs=64

bs=128

bs=256

bs=512

bs=1024

5.50

5.50

5.50

5.50

5.50

5.50

a=0.165, b=-2.292, c=11.765

min: log2 x 6.95, y 3.8

5.25

5.25

5.25

5.25

5.25

5.25

5.00

5.00

5.00

5.00

5.00

5.00

4.75

4.75

4.75

4.75

4.75

4.75

286B

4.50

4.50

4.50

4.50

4.50

4.50

4.25

4.25

4.25

4.25

4.25

4.25

4.00

4.00

4.00

4.00

4.00

4.00

3.75

3.75

3.75

3.75

3.75

3.75

a=0.118, b=-1.922, c=11.765 (WLS)

a=0.134, b=-2.072, c=11.765 (WLS)

a=0.140, b=-2.127, c=11.765 (WLS)

a=0.151, b=-2.205, c=11.765

a=0.184, b=-2.392, c=11.765

3.50

3.50

3.50

3.50

3.50

3.50

min: log2 x 8.16, y 3.92

min: log2 x 7.71, y 3.77

min: log2 x 7.58, y 3.71

min: log2 x 7.29, y 3.73

min: log2 x 6.48, y 4.01

21 23 25 27 29 211 213

21 23 25 27 29 211 213

21 23 25 27 29 211 213

21 23 25 27 29 211 213

21 23 25 27 29 211 213

21 23 25 27 29 211 213

bs=32

bs=64

bs=128

bs=256

bs=512

bs=1024

5.50

5.50

5.50

5.50

5.50

5.50

a=0.121, b=-1.975, c=11.765 (WLS)

a=0.140, b=-2.145, c=11.765 (WLS)

a=0.154, b=-2.250, c=11.765 (WLS)

a=0.161, b=-2.292, c=11.765

min: log2 x 8.18, y 3.69

min: log2 x 7.66, y 3.55

min: log2 x 7.29, y 3.56

min: log2 x 7.13, y 3.6

5.25

5.25

5.25

5.25

5.25

5.25

5.00

5.00

5.00

5.00

5.00

5.00

4.75

4.75

4.75

4.75

4.75

4.75

286B

4.50

4.50

4.50

4.50

4.50

4.50

4.25

4.25

4.25

4.25

4.25

4.25

4.00

4.00

4.00

4.00

4.00

4.00

3.75

3.75

3.75

3.75

3.75

3.75

a=0.177, b=-2.392, c=11.765

a=0.186, b=-2.417, c=11.765

3.50

3.50

3.50

3.50

3.50

3.50

min: log2 x 6.74, y 3.7

min: log2 x 6.48, y 3.93

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

bs=32

bs=64

bs=128

bs=256

bs=512

bs=1024

5.50

5.50

5.50

5.50

5.50

5.50

5.25

5.25

5.25

5.25

5.25

5.25

5.00

5.00

5.00

5.00

5.00

5.00

4.75

4.75

4.75

4.75

4.75

4.75

286B

4.50

4.50

4.50

4.50

4.50

4.50

4.25

4.25

4.25

4.25

4.25

4.25

4.00

4.00

4.00

4.00

4.00

4.00

3.75

3.75

3.75

3.75

3.75

3.75

a=0.123, b=-1.937, c=11.765 (WLS)

a=0.137, b=-2.065, c=11.765 (WLS)

a=0.151, b=-2.178, c=11.765 (WLS)

a=0.170, b=-2.308, c=11.765

a=0.193, b=-2.442, c=11.765

a=0.230, b=-2.620, c=11.765

3.50

3.50

3.50

3.50

3.50

3.50

min: log2 x 7.9, y 4.11

min: log2 x 7.55, y 3.97

min: log2 x 7.21, y 3.91

min: log2 x 6.77, y 3.95

min: log2 x 6.33, y 4.03

min: log2 x 5.68, y 4.32

21 23 25 27 29 211 213

21 23 25 27 29 211 213

21 23 25 27 29 211 213

21 23 25 27 29 211 213

21 23 25 27 29 211 213

21 23 25 27 29 211 213

bs=32

bs=64

bs=128

bs=256

bs=512

bs=1024

5.50

5.50

5.50

5.50

5.50

5.50

a=0.129, b=-2.006, c=11.765 (WLS)

a=0.229, b=-2.663, c=11.765

min: log2 x 7.8, y 3.94

min: log2 x 5.82, y 4.01

5.25

5.25

5.25

5.25

5.25

5.25

5.00

5.00

5.00

5.00

5.00

5.00

4.75

4.75

4.75

4.75

4.75

4.75

286B

4.50

4.50

4.50

4.50

4.50

4.50

4.25

4.25

4.25

4.25

4.25

4.25

4.00

4.00

4.00

4.00

4.00

4.00

3.75

3.75

3.75

3.75

3.75

3.75

a=0.139, b=-2.096, c=11.765 (WLS)

a=0.157, b=-2.229, c=11.765 (WLS)

a=0.200, b=-2.510, c=11.765

a=0.290, b=-2.952, c=11.765

3.50

3.50

3.50

3.50

3.50

3.50

min: log2 x 7.52, y 3.88

min: log2 x 7.09, y 3.86

min: log2 x 6.26, y 3.91

min: log2 x 5.08, y 4.26

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

(b)

Figure 17: Individual norm scans for various batch sizes B (columns), across various horizons D in (a), across various models in (b) (rows). We plot train loss (Y-axis) against the output layer operator norm ∥Wout∥RMS→∞, where each point corresponds to a different learning rate run and error bars correspond to loss smoothing variance (see Appendix A.4). The best-loss point for each (B,D) is pinpointed with the blue dashed line, fitted curves are shown with blue solid lines. These fit results are used for: (a) Fig. 2a and Fig. 3b, (b) Fig. 2b, from top to bottom rows: proxy, ×4width, ×12-width, ×8-depth, ×32-depth.

bs=32

bs=64

bs=128

bs=256

bs=512

bs=1024

4.2

4.2

4.2

4.2

4.2

4.2

min: log2 x 7.19, y 3.81

min: log2 x 6.63, y 3.82

min: log2 x 6.32, y 3.87

min: log2 x 6.23, y 3.92

min: log2 x 6.23, y 3.95

min: log2 x 6.3, y 4.04

4.1

4.1

4.1

4.1

4.1

4.1

4.0

4.0

4.0

4.0

4.0

4.0

3.9

3.9

3.9

3.9

3.9

3.9

221B

3.8

3.8

3.8

3.8

3.8

3.8

3.7

3.7

3.7

3.7

3.7

3.7

3.6

3.6

3.6

3.6

3.6

3.6

3.5

3.5

3.5

3.5

3.5

3.5

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

4.2

4.2

4.2

4.2

4.2

4.2

min: log2 x 7.47, y 3.8

min: log2 x 7.15, y 3.75

min: log2 x 6.29, y 3.74

min: log2 x 6.67, y 3.82

min: log2 x 6.36, y 3.8

min: log2 x 6.61, y 3.79

4.1

4.1

4.1

4.1

4.1

4.1

4.0

4.0

4.0

4.0

4.0

4.0

3.9

3.9

3.9

3.9

3.9

3.9

286B

3.8

3.8

3.8

3.8

3.8

3.8

3.7

3.7

3.7

3.7

3.7

3.7

3.6

3.6

3.6

3.6

3.6

3.6

3.5

3.5

3.5

3.5

3.5

3.5

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

4.2

4.2

4.2

4.2

4.2

4.2

min: log2 x 7.42, y 3.8

min: log2 x 7.09, y 3.76

min: log2 x 6.51, y 3.74

min: log2 x 6.53, y 3.7

min: log2 x 6.63, y 3.69

min: log2 x 7.09, y 3.72

4.1

4.1

4.1

4.1

4.1

4.1

4.0

4.0

4.0

4.0

4.0

4.0

3.9

3.9

3.9

3.9

3.9

3.9

234B

3.8

3.8

3.8

3.8

3.8

3.8

3.7

3.7

3.7

3.7

3.7

3.7

3.6

3.6

3.6

3.6

3.6

3.6

3.5

3.5

3.5

3.5

3.5

3.5

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

4.2

4.2

4.2

4.2

4.2

4.2

min: log2 x 8.3, y 3.85

min: log2 x 7.93, y 3.84

min: log2 x 6.38, y 3.65

4.1

4.1

4.1

4.1

4.1

4.1

4.0

4.0

4.0

4.0

4.0

4.0

3.9

3.9

3.9

3.9

3.9

3.9

2137B

3.8

3.8

3.8

3.8

3.8

3.8

3.7

3.7

3.7

3.7

3.7

3.7

3.6

3.6

3.6

3.6

3.6

3.6

min: log2 x 7.19, y 3.81

min: log2 x 7.27, y 3.72

min: log2 x 6.94, y 3.65

3.5

3.5

3.5

3.5

3.5

3.5

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

(a)

bs=32

bs=64

bs=128

bs=256

bs=512

bs=1024

4.2

4.2

4.2

4.2

4.2

4.2

| | |m|in: log2|x 6.91<br><br>|, y 3.8<br><br>| | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

| | |m|in: log2|x 6.4,<br><br>|y 3.85<br><br>| | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

min: log2 x 7.66, y 3.89

min: log2 x 6.38, y 3.76

min: log2 x 6.67, y 3.87

min: log2 x 6.15, y 3.89

4.1

4.1

4.1

4.1

4.1

4.1

4.0

4.0

4.0

4.0

4.0

4.0

3.9

3.9

3.9

3.9

3.9

3.9

229B

3.8

3.8

3.8

3.8

3.8

3.8

3.7

3.7

3.7

3.7

3.7

3.7

3.6

3.6

3.6

3.6

3.6

3.6

3.5

3.5

3.5

3.5

3.5

3.5

20 21 22 23 24 25 26 27 28

20 21 22 23 24 25 26 27 28

20 21 22 23 24 25 26 27 28

20 21 22 23 24 25 26 27 28

20 21 22 23 24 25 26 27 28

20 21 22 23 24 25 26 27 28

4.2

4.2

4.2

4.2

4.2

4.2

min: log2 x 7.99, y 3.75

min: log2 x 7.23, y 3.74

min: log2 x 6.97, y 3.77

min: log2 x 6.95, y 3.77

4.1

4.1

4.1

4.1

4.1

4.1

4.0

4.0

4.0

4.0

4.0

4.0

3.9

3.9

3.9

3.9

3.9

3.9

211B

3.8

3.8

3.8

3.8

3.8

3.8

3.7

3.7

3.7

3.7

3.7

3.7

3.6

3.6

3.6

3.6

3.6

3.6

min: log2 x 6.74, y 3.76

min: log2 x 6.91, y 3.75

3.5

3.5

3.5

3.5

3.5

3.5

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

4.2

4.2

4.2

4.2

4.2

4.2

min: log2 x 8.81, y 3.82

min: log2 x 7.98, y 3.71

min: log2 x 7.71, y 3.68

min: log2 x 7.61, y 3.71

min: log2 x 7.39, y 3.71

min: log2 x 7.14, y 3.68

4.1

4.1

4.1

4.1

4.1

4.1

4.0

4.0

4.0

4.0

4.0

4.0

3.9

3.9

3.9

3.9

3.9

3.9

246B

3.8

3.8

3.8

3.8

3.8

3.8

3.7

3.7

3.7

3.7

3.7

3.7

3.6

3.6

3.6

3.6

3.6

3.6

3.5

3.5

3.5

3.5

3.5

3.5

25 26 27 28 29 210

25 26 27 28 29 210

25 26 27 28 29 210

25 26 27 28 29 210

25 26 27 28 29 210

25 26 27 28 29 210

4.2

4.2

4.2

4.2

4.2

4.2

min: log2 x 9.46, y 3.73

min: log2 x 8.44, y 3.69

min: log2 x 8.52, y 3.69

min: log2 x 8.44, y 3.65

min: log2 x 8.11, y 3.66

min: log2 x 7.81, y 3.64

4.1

4.1

4.1

4.1

4.1

4.1

4.0

4.0

4.0

4.0

4.0

4.0

3.9

3.9

3.9

3.9

3.9

3.9

2184B

3.8

3.8

3.8

3.8

3.8

3.8

3.7

3.7

3.7

3.7

3.7

3.7

3.6

3.6

3.6

3.6

3.6

3.6

3.5

3.5

3.5

3.5

3.5

3.5

26 27 28 29 210

26 27 28 29 210

26 27 28 29 210

26 27 28 29 210

26 27 28 29 210

26 27 28 29 210

(b)

Figure 18: Individual output norm ∥Wout∥RMS→∞ scans for various batch sizes B (columns) across various horizons D (rows). (a) with momentum = 0.1, no learning rate decay. (b) with momentum = 0.1, with learning rate decay linearly to 0 for 25% of total horizon.

bs=32

bs=64

bs=128

bs=256

bs=512

bs=1024

bs=2048

6.00

6.00

6.00

6.00

6.00

6.00

6.00

min: log2 x 7.5, y 4.57

min: log2 x 6.87, y 4.41

min: log2 x 6.52, y 4.4

min: log2 x 6.84, y 4.52

min: log2 x 4.96, y 5.4

5.75

5.75

5.75

5.75

5.75

5.75

5.75

5.50

5.50

5.50

5.50

5.50

5.50

5.50

5.25

5.25

5.25

5.25

5.25

5.25

5.25

221B

5.00

5.00

5.00

5.00

5.00

5.00

5.00

4.75

4.75

4.75

4.75

4.75

4.75

4.75

4.50

4.50

4.50

4.50

4.50

4.50

4.50

4.25

4.25

4.25

4.25

4.25

4.25

4.25

4.00

4.00

4.00

4.00

4.00

4.00

4.00

min: log2 x 5.67, y 4.67

min: log2 x 5.61, y 5.03

3.75

3.75

3.75

3.75

3.75

3.75

3.75

2 1 21 23 25 27 29

2 1 21 23 25 27 29

2 1 21 23 25 27 29

2 1 21 23 25 27 29

2 1 21 23 25 27 29

2 1 21 23 25 27 29

2 1 21 23 25 27 29

6.00

6.00

6.00

6.00

6.00

6.00

6.00

min: log2 x 8.21, y 4.54

min: log2 x 7.31, y 4.3

min: log2 x 6.27, y 4.19

5.75

5.75

5.75

5.75

5.75

5.75

5.75

5.50

5.50

5.50

5.50

5.50

5.50

5.50

5.25

5.25

5.25

5.25

5.25

5.25

5.25

286B

5.00

5.00

5.00

5.00

5.00

5.00

5.00

4.75

4.75

4.75

4.75

4.75

4.75

4.75

4.50

4.50

4.50

4.50

4.50

4.50

4.50

4.25

4.25

4.25

4.25

4.25

4.25

4.25

4.00

4.00

4.00

4.00

4.00

4.00

4.00

min: log2 x 6.94, y 4.15

min: log2 x 6, y 4.23

min: log2 x 5.51, y 4.28

min: log2 x 5.5, y 4.49

3.75

3.75

3.75

3.75

3.75

3.75

3.75

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

6.00

6.00

6.00

6.00

6.00

6.00

6.00

min: log2 x 7.94, y 4.49

min: log2 x 7.98, y 4.27

min: log2 x 7.84, y 4.12

min: log2 x 6.93, y 4.01

min: log2 x 6.44, y 3.97

min: log2 x 5.87, y 4.03

min: log2 x 5.48, y 4.09

5.75

5.75

5.75

5.75

5.75

5.75

5.75

5.50

5.50

5.50

5.50

5.50

5.50

5.50

5.25

5.25

5.25

5.25

5.25

5.25

5.25

5.00

5.00

5.00

5.00

5.00

5.00

5.00

234B

4.75

4.75

4.75

4.75

4.75

4.75

4.75

4.50

4.50

4.50

4.50

4.50

4.50

4.50

4.25

4.25

4.25

4.25

4.25

4.25

4.25

4.00

4.00

4.00

4.00

4.00

4.00

4.00

3.75

3.75

3.75

3.75

3.75

3.75

3.75

22 23 24 25 26 27 28 29 210

22 23 24 25 26 27 28 29 210

22 23 24 25 26 27 28 29 210

22 23 24 25 26 27 28 29 210

22 23 24 25 26 27 28 29 210

22 23 24 25 26 27 28 29 210

22 23 24 25 26 27 28 29 210

6.00

6.00

6.00

6.00

6.00

6.00

6.00

min: log2 x 7.91, y 4.65

min: log2 x 7.97, y 4.46

min: log2 x 8.24, y 4.22

min: log2 x 7.78, y 3.97

min: log2 x 6.83, y 3.88

min: log2 x 6.23, y 3.86

min: log2 x 5.94, y 3.92

5.75

5.75

5.75

5.75

5.75

5.75

5.75

5.50

5.50

5.50

5.50

5.50

5.50

5.50

5.25

5.25

5.25

5.25

5.25

5.25

5.25

2137B

5.00

5.00

5.00

5.00

5.00

5.00

5.00

4.75

4.75

4.75

4.75

4.75

4.75

4.75

4.50

4.50

4.50

4.50

4.50

4.50

4.50

4.25

4.25

4.25

4.25

4.25

4.25

4.25

4.00

4.00

4.00

4.00

4.00

4.00

4.00

3.75

3.75

3.75

3.75

3.75

3.75

3.75

24 25 26 27 28 29 210

24 25 26 27 28 29 210

24 25 26 27 28 29 210

24 25 26 27 28 29 210

24 25 26 27 28 29 210

24 25 26 27 28 29 210

24 25 26 27 28 29 210

(a)

bs=32

bs=64

bs=128

bs=256

bs=512

bs=1024

bs=2048

6.00

6.00

6.00

6.00

6.00

6.00

6.00

min: log2 x 4.57, y 4.41

min: log2 x 4.17, y 4.4

min: log2 x 5.44, y 4.52

min: log2 x 3.82, y 4.67

min: log2 x 4.53, y 5.4

5.75

5.75

5.75

5.75

5.75

5.75

5.75

5.50

5.50

5.50

5.50

5.50

5.50

5.50

5.25

5.25

5.25

5.25

5.25

5.25

5.25

221B

5.00

5.00

5.00

5.00

5.00

5.00

5.00

4.75

4.75

4.75

4.75

4.75

4.75

4.75

4.50

4.50

4.50

4.50

4.50

4.50

4.50

4.25

4.25

4.25

4.25

4.25

4.25

4.25

4.00

4.00

4.00

4.00

4.00

4.00

4.00

min: log2 x 5.48, y 4.57

min: log2 x 5.44, y 5.03

3.75

3.75

3.75

3.75

3.75

3.75

3.75

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

21 23 25 27 29

6.00

6.00

6.00

6.00

6.00

6.00

6.00

min: log2 x 6.68, y 4.54

min: log2 x 5.24, y 4.3

min: log2 x 4.84, y 4.15

min: log2 x 4, y 4.19

min: log2 x 3.94, y 4.23

min: log2 x 3.85, y 4.28

5.75

5.75

5.75

5.75

5.75

5.75

5.75

5.50

5.50

5.50

5.50

5.50

5.50

5.50

5.25

5.25

5.25

5.25

5.25

5.25

5.25

286B

5.00

5.00

5.00

5.00

5.00

5.00

5.00

4.75

4.75

4.75

4.75

4.75

4.75

4.75

4.50

4.50

4.50

4.50

4.50

4.50

4.50

4.25

4.25

4.25

4.25

4.25

4.25

4.25

4.00

4.00

4.00

4.00

4.00

4.00

4.00

min: log2 x 6.42, y 4.49

3.75

3.75

3.75

3.75

3.75

3.75

3.75

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

6.00

6.00

6.00

6.00

6.00

6.00

6.00

min: log2 x 6.16, y 4.49

min: log2 x 6.37, y 4.27

min: log2 x 6.14, y 4.12

min: log2 x 4.71, y 4.01

min: log2 x 4.19, y 3.97

min: log2 x 4.39, y 4.03

min: log2 x 4.52, y 4.09

5.75

5.75

5.75

5.75

5.75

5.75

5.75

5.50

5.50

5.50

5.50

5.50

5.50

5.50

5.25

5.25

5.25

5.25

5.25

5.25

5.25

5.00

5.00

5.00

5.00

5.00

5.00

5.00

234B

4.75

4.75

4.75

4.75

4.75

4.75

4.75

4.50

4.50

4.50

4.50

4.50

4.50

4.50

4.25

4.25

4.25

4.25

4.25

4.25

4.25

4.00

4.00

4.00

4.00

4.00

4.00

4.00

3.75

3.75

3.75

3.75

3.75

3.75

3.75

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

6.00

6.00

6.00

6.00

6.00

6.00

6.00

min: log2 x 6, y 4.65

min: log2 x 6.54, y 4.46

min: log2 x 7.33, y 4.22

min: log2 x 6.02, y 3.97

min: log2 x 4.71, y 3.88

min: log2 x 4.41, y 3.86

min: log2 x 4.98, y 3.92

5.75

5.75

5.75

5.75

5.75

5.75

5.75

5.50

5.50

5.50

5.50

5.50

5.50

5.50

5.25

5.25

5.25

5.25

5.25

5.25

5.25

2137B

5.00

5.00

5.00

5.00

5.00

5.00

5.00

4.75

4.75

4.75

4.75

4.75

4.75

4.75

4.50

4.50

4.50

4.50

4.50

4.50

4.50

4.25

4.25

4.25

4.25

4.25

4.25

4.25

4.00

4.00

4.00

4.00

4.00

4.00

4.00

3.75

3.75

3.75

3.75

3.75

3.75

3.75

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

22 24 26 28 210

(b)

### Figure 19: Individual norm scans for various batch sizes B (columns) across various horizons D (rows). (a) For ∥Wout∥RMS→RMS (output layer). (b) For ∥Win∥1→RMS (input layer).

