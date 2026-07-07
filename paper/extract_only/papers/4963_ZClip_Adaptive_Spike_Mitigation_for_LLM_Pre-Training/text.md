# arXiv:2504.02507v1[cs.LG]3Apr2025

## ZClip: Adaptive Spike Mitigation for LLM Pre-Training

Abhay Kumar, Louis Owen, Nilabhra Roy Chowdhury, Fabian Güra BluOrion {abhay.kumar, louis.owen, nilabhra.chowdhury, fabian.guera}@bluorion.com

April 2, 2025

### Abstract

Training large language models (LLMs) presents numerous challenges, including gradient instability and loss spikes. These phenomena can lead to catastrophic divergence, requiring costly checkpoint restoration and data batch skipping. Traditional gradient clipping techniques, such as constant or norm-based methods, fail to address these issues effectively due to their reliance on fixed thresholds or heuristics, leading to inefficient learning and requiring frequent manual intervention. In this work, we propose ZClip, an adaptive gradient clipping algorithm that dynamically adjusts the clipping threshold based on statistical properties of gradient norms over time. Unlike prior reactive strategies, ZClip proactively adapts to training dynamics without making any prior assumptions on the scale and the temporal evolution of gradient norms. At its core, it leverages z-score-based anomaly detection to identify and mitigate large gradient spikes, preventing malignant loss spikes while not interfering with convergence otherwise. Our code is available at: https://github.com/bluorion-com/ZClip.

### 1 Introduction

Large language models (LLMs) have revolutionized the field of natural language processing [1, 2, 3]. However, the training of these models presents significant challenges, among the most critical ones is the occurrence of loss spikes–sudden, extreme increases in the training loss. These spikes do not just degrade model performance temporarily, but can also lead to catastrophic divergence, forcing manual intervention to resume training from earlier checkpoints [4, 5].

- [6], for example, highlights the severity of instability during large-scale language model pre-training. In their 540B parameter model training run, the authors observed more than 20 loss spikes throughout the training, each requiring checkpoint rewinds and skipping several hundred batches. Interestingly, they note that replaying the same batch that triggered a spike did not consistently reproduce the issue. This led them to hypothesize that loss spikes emerge from rare interactions between the optimizer state and specific input batches—a fragile and hard-to-predict combination.
- [7], as another example, highlights how spikes also increase the environmental impact of LLM pre-training: An additional 30 days were required to address loss spikes while training their 65B model, resulting in the consumption of 129.3 MWh of additional energy. To maintain stability, the authors reported resorting to checkpoint rewinds, batch skipping, and learning rate adjustments—manual interventions that increase engineering complexity and compute overhead. Notably, they distinguish between two types of loss spikes: benign spikes, from which training can recover naturally, and malignant spikes, which lead to irreversible divergence.

Theoretical insights into loss spikes were presented in [8], which attributed the issue to the sudden growth of gradient norms. By analyzing the spectral norms of Jacobian matrices in Transformer sub-layers, the authors proposed stabilizing training through carefully scaled embeddings and small initialization values. While their findings contribute to an early theoretical understanding of loss spikes, the suggested methods

[Figure 1]

- Figure 1: Training loss graph comparing 1) training without clipping, 2) clipping with fixed threshold 1.0, and 3) ZClip for a LLaMA 1B model. The learning rate for all three experiments is 3.0 × 10−3. While both “no clipping” and “constant clipping” exhibit spiky behavior and diverge early,

ZClip (with zthres=2.5 and α = 0.97) remains stable and continues to optimize effectively throughout training. Details on the model configuration and other training hyperparameters are presented in Appendix Section 6.1.

apply only statically at initialization time, i.e. once training has started, they can no longer influence training dynamics.

Gradient clipping has historically been used to address the issue of exploding gradients, particularly in the context of recurrent neural networks (RNNs), where long-term dependencies exacerbate gradient instability [9]. Traditional gradient clipping methods—used in seminal works such as the original Transformer paper [10], PaLM [6], and LLaMA [2, 11]—typically apply fixed thresholds to control gradient norms. While effective in limiting gradient explosion, these approaches lack adaptability to evolving training dynamics and may under-clip in later stages of training (see, for example, Figure 2).

To address these limitations, we propose ZClip, an adaptive gradient clipping algorithm designed to dynamically regulate the gradient norm using recent statistics. ZClip employs z-score-based anomaly detection and leverages exponential moving averages (EMA) for robust gradient norm statistics estimation, enabling it to effectively mitigate loss spikes without manual intervention. Unlike fixed-threshold gradient norm clipping, ZClip hence adapts to evolving training dynamics. Through empirical evaluations, we demonstrate that ZClip eliminates loss spikes in all but the most extreme training regimes we tested, and can enable LLMs to achieve baseline performance with a significantly smaller computational and token budget.

- 2 Gradient Clipping Methods

- 2.1 Gradient Clipping with Fixed Threshold

The core idea of gradient clipping is to limit the magnitude of gradients during backpropagation to ensure numerical stability and prevent gradients from becoming excessively large. The most common form is fixedthreshold norm clipping, where the gradient norm is constrained to a predefined threshold. Mathematically, let gt be the gradient vector of a loss function Ldata(θt) with respect to the model parameters θt at training step t. Then the total gradient norm is defined as:

∥gt∥2 =

N

gti2 , (1)

i=1

[Figure 2]

#### Figure 2: Training loss graph for a LLaMA 1B model trained with fixed-threshold clipping c =

- 1.0. Gradient norm spikes persist due to a mismatch between the static threshold and the running distribution. This reveals a key limitation of fixed-threshold clipping in dynamically changing training regimes.

where N is the total number of model parameters. If the gradient norm ∥gt∥2 exceeds a constant threshold c, the gradient is scaled down proportionally:

gt, if ∥gt∥2 ≤ c,

gt∗ =

(2)

gt

∥gt∥2 · c, if ∥gt∥2 > c.

This ensures that the updated gradients remain within a controlled range, thus avoiding large parameter updates that could destabilize training. While fixed-threshold norm-based gradient clipping has proven effective in traditional deep learning tasks, it exhibits several limitations when applied to modern LLMs.

Inflexibility Under Dynamic Training Regimes. One major limitation is the use of a static threshold. In fixed-threshold gradient clipping, the threshold c remains constant throughout training, despite the general observation that gradient magnitudes vary significantly across iterations and training stages (see, for example, Figure 2). The distribution of gradient norms evolves over time—often decreasing in central tendency as the model converges. This makes it difficult to select a single, optimal threshold upfront. Furthermore, modern LLMs are trained using varying learning rates, curriculum schedules, model depths, batch sizes, and data mixtures. These factors induce non-stationary gradient behavior. A static threshold cannot in itself account for such evolving dynamics, leading to either unnecessarily slow or unstable training [3, 11, 12, 13].

Illustrative Example. Consider an experiment setup in which the maximum gradient norm is clipped to the threshold c = 1.0. Let us assume that during the early stages of training, the distribution of gradient norms has a mean value of µ ≈ 0.8. At a time step t, if the gradient norm reaches ∥gt∥2 = 1.2, the gradient is clipped to 1.0 as ∥gt∥2 > c. This ensures that the gradient does not deviate too far from the mean µ. Now, as training progresses, the magnitude of the loss decreases (and potentially also the learning rate). This will cause the mean to drift towards a lower value, for example, µ = 0.2. At this stage of training, if ∥gt∥2 = 0.9 is encountered, no clipping will take place as ∥gt∥2 < c. However, this can be detrimental for training as ∥gt∥2 is much higher than the current mean µ and the high gradient norm can adversely affect the model weights.

Such a deviation suggests that the distribution’s tail is being reached or even surpassed more frequently, potentially leading to instabilities (see, for example, Figure 2). This example highlights the limitation

of fixed-threshold clipping: it fails to adjust for shifts in the underlying gradient norm distribution over time.

Sensitivity to Hyperparameters. The optimal threshold c furthermore depends on the particular model and the many hyperparameters in the training recipe. If c is too small, gradients are over-clipped and learning slows down. If c is too large, the gradients may not be clipped sufficiently, resulting in instability. All of these dependencies make tuning non-trivial, especially for large-scale models.

In summary, fixed-threshold clipping suffers from two key limitations: (a) the threshold c does not in itself evolve with training, and (b) it is sensitive to model-specific factors. These shortcomings motivate the need for adaptive methods, as discussed in the following.

#### 2.2 AutoClip

AutoClip [14] computes an adaptive clipping threshold by selecting a user-defined percentile from the (preclipping) gradient norm history. It was originally proposed in the context of an audio source separation task using a bi-directional LSTM model. While applicable in principle in the context of LLMs, storing every gradient norm over millions of steps incurs some memory and compute overhead. Furthermore, the use of the pre-clipping gradient norm history still makes it sensitive to outliers accumulating over the course of long LLM pre-training runs.

- 2.3 ZClip

To address the aforementioned challenges related to the temporal evolution of gradient norms during training, model-specific hyperparameters, stability in extremely long training regimes, and the memory and compute overhead of maintaining a full gradient norm history, we propose ZClip. At its core, the approach relies on the z-score statistics of the gradient norm to detect abnormal spikes (see Section 2.3.1), and to adjust the clipping dynamically (see Section 2.3.2). This allows ZClip to respond effectively to the evolving training dynamics. Since we are using z-scores, we assume that gradient norms over a short window are approximately normally distributed (see Appendix 6.4 for an empirical analysis). To capture this local behavior of the gradient norms, ZClip tracks the exponential moving average (EMA) of both mean and variance of the gradient norm. In contrast to methods such as AutoClip, ZClip only maintains a lightweight summary of the recent history. As a result, it is more memory- and compute-efficient, making it particularly suitable for large-scale training scenarios.

##### 2.3.1 Z-score based Spike Detection The ZClip process begins by computing the norm of the gradient gt at training step t as:

gt = ∥gt∥2. (3)

ZClip then iteratively updates the mean and standard deviation with exponential smoothing. Specifically, the mean µt and the standard deviation σt are updated with a smoothing factor α ∈ (0,1) such that:

µt = αµt−1 + (1 − α)gt, (4)

σt = ασt2−1 + (1 − α) gt − µt 2. (5) We then evaluate the deviation of the current gradient norm from the running mean in terms of standard deviations. This is achieved through the z-score calculation:

(gt − µt) σt

. (6)

zt =

The z-score zt then represents how many standard deviations the current gradient norm lies away from the mean. A spike is detected if zt > zthres.

- 2.3.2 Z-score based Gradient Adjustment If, based on the z-score, a spike is detected, we once again rely on the z-score to adjust the gradient norm. We define the adjusted gradient as:

where

gt∗ =

gt if zt ≤ zthres

gt

∥gt∥2 (µt + zt∗ · σt) if zt > zthres,

(7)

zt∗ = ξ(zt(gt)). (8)

The function ξ(zt) maps the original z-score zt to an adjusted, “clipped” z-score zt∗. To reduce the gradient norm for spikes, ξ must satisfy ξ(zt) ≤ zt for zt ≥ zthres. If ξ(zt) ≤ zthres for zt ≥ zthres, then the gradient norm is strictly clipped to gt∗ ≤ µt + zthres · σt. For continuity at zt = zthres, one may want to impose ξ(zt = zthres) = zthres in addition.

- Figure 3 illustrates three possible choices for ξ:

- • Clipping to mean: zt∗ = 0 (red line) clips the gradient norm to the mean µt, creating a discontinuity at zt = zthres.
- • Clipping to max: zt∗ = zthres (green line) clips the gradient norm to the maximum value gt∗ = µt + zthresσt. It is continuous at zt = zthres.
- • Reciprocal clipping: zt∗ = zthres2 /zt (purple line) is a possible compromise between the two previous strategies. It is continuous at zt = zthres, but clips extreme outliers more aggressively and eventually

converges to “clip to mean”, i.e. limz

t→∞ gt∗ = µt.

[Figure 3]

- Figure 3: Three possible choices for the z-score adjustment function ξ(zt) illustrated for zthres = 2.5. Note the discontinuity for ξ(zt) = 0, and the reciprocal nature of ξ(zt) = zthres2 /zt leading to more aggressive clipping for more extreme outliers.

We experimented with all three options presented for ξ(zt) and observed the best results for “reciprocal clipping” (see Appendix 6.3). In the following, we therefore assume ξ(zt) = zthres2 /zt unless otherwise specified. We hypothesize that adjusting the clipping strength dynamically based on the severity of the anomaly helps mitigate the potentially destabilizing effect of discontinuous gradient norm adjustments, as well as the algorithm’s sensitivity towards a particular (potentially slightly suboptimal) choice of zthres. If one interprets η = zt/zthres as a signal-to-noise ratio (SNR), then scaling by the severity η−1 can intuitively be interpreted as follows: When the gradient norm is very high, it generally indicates the presence of significant noise, thus requiring stronger clipping to filter out this disruptive component; in contrast, when the gradient norm is only slightly elevated, it implies lower noise levels, so less aggressive clipping suffices to preserve the informative signal.

#### 2.3.3 Initialization via Warm-up

The initialization of the mean and variance estimates plays a critical role in the performance of ZClip. We experimented with several schemes, including fixed constants, early exponential moving average (EMA) bootstrapping, and dynamic ramp-up strategies. However, we found that using a short warm-up period yielded the most stable results.

During the first Nw steps (e.g., Nw = 25), we collect the unmodified gradient norms without applying ZClip. At the end of this warm-up phase, we compute the initial mean and variance as follows:

µN

w

1 Nw

Nw

=

=

gt, σN

w

t=1

1 Nw

Nw

(gt − µN

)2. (9)

w

t=1

These values serve as the starting point for the EMA-based updates that continue throughout the remainder of the training. This approach ensures that the statistics are grounded in the actual behavior of the gradients at the beginning of training, providing a stable foundation for subsequent anomaly detection and clipping.

#### 2.3.4 Handling Spikes in Statistics Updates

A key challenge in maintaining accurate statistics arises when the current gradient norm gt is classified as a spike, that is, zt > zthres. Including such extreme values in the updates can skew the moving averages and inflate future thresholds, leading to reduced sensitivity towards anomalies. On the other hand, skipping the update entirely biases the statistics toward lower values, which can make clipping overly aggressive over time.

To strike a balance, we use the clipped gradient norm gt∗ to update the statistics during such events. Specifically, we define the value used to update the statistics as:

gtupdate = gt, if zt ≤ zthres, gt∗ if zt > zthres.

(10)

This update strategy ensures that the statistics remain representative of the stable training regime while still adapting to changing dynamics.

#### 2.3.5 Algorithm & Implementation

Algorithm 1 summarizes the mathematical formulations presented above and illustrates ZClip from a practical implementation perspective. It incorporates the key steps outlined above, namely warm-up, anomaly detection, gradient norm adjustment, and the statistics update rule in case of spikes. A concrete implementation for use with PyTorch (Lightning) can be found at https://github.com/bluorion-com/ZClip.

- 3 Experiment Setup In the following, we describe the setup of the experiments we ran to answer three core questions:

- 1. Does ZClip effectively mitigate spikes in pre-training, especially in “aggressive”, spike-prone training regimes such as when running with high learning rates?
- 2. If so, does ZClip stabilize these “aggressive” regimes enough such that training converges towards a particular loss or benchmark milestone earlier than in “standard” training regimes, consequently reducing the total computational cost?
- 3. In “standard” training regimes, i.e. regimes with learning rates that are widely used in literature and are observed to spike occasionally, are there any negative consequences of activating ZClip in terms of loss convergence, downstream task performance, or throughput?

#### 3.1 Model Setup

Unless otherwise specified, all experiments were performed using a 1B (16 layers) LLaMA [15] model. The training dataset consisted of the SmolLM [16] corpus, which in turn is comprised of three sources: FineWebEduDeduplicated, Cosmopedia-V2, and Python-Edu. From this corpus we randomly sampled 50 billion tokens (50BT) and employed packing to obtain examples with a fixed context length of 2048. All experiments were conducted in a distributed training setup using the Fully Sharded Data Parallelism (FSDP) strategy across four GPU nodes, each equipped with eight H100 GPUs. Unless explicitly stated otherwise, ZClip was configured with default hyperparameters α = 0.97 for the smoothing factor, and threshold zthres = 2.5. The results of the hyperparameter sweeps used to determine these values are presented in Appendix 6.6. Other training hyperparameters are presented in Table 1. More details regarding the model configuration and tokenizer are provided in Appendix 6.1.

Algorithm 1 Training Loop with ZClip (EMA-based updates; Reciprocal clipping) Require:

α ∈ (0,1) ▷ EMA smoothing factor (e.g., 0.97) zthres ∈ R+ ▷ Z-score threshold (e.g., 2.5) ϵ ← 10−6 ▷ Small positive constant lr > 0 ▷ Learning rate (e.g., 0.001) f(θ) ▷ Objective function µt,vt ∈ R ▷ EMA estimates of gradient norm mean and variance Nw ∈ N ▷ Number of warm-up steps (e.g., 25) tmax ∈ N ▷ Total number of training steps

Initialize via warm-up: Collect Nw gradient norms to compute µN

and vN

w

w

t ← Nw while t < tmax do

gt ← ∇ft(θt) ▷ Gradient computation gt ← ∥gt ∥2 zt ←

gt − µt √vt + ϵ

gt∗ ← gt if zt > zthres then

zthres2 zt

√vt

gt∗ ← µt +

end if clip_grad_norm_(gt∗) ▷ PyTorch in-place gradient clipping θt+1 ← optimizer_update(θt,lr,gt)

µt+1 ← αµt + (1 − α)gt∗ ▷ EMA update for mean vt+1 ← αvt + (1 − α)(gt∗ − µt+1)2 ▷ EMA update for variance t ← t + 1

end while return θt

max

|Hyperparameter<br><br>|Value|
|---|---|
|Optimizer Learning Rate Schedule Max. Learning Rate End Learning Rate Warm-up Tokens Weight Decay Global Batch Size Sequence Length Precision|Fused AdamW (β1 = 0.9, β2 = 0.999, ϵ = 1 × 10−7)<br><br>Linear warm-up followed by cosine decay<br><br>1 × 10−4 to 5 × 10−3<br><br>10% of Maximum Learning Rate<br><br>2 billion tokens (2BT)<br><br><br>0.01 (AdamW implementation)<br><br>2048 2048 Mixed Precision BFloat16<br><br>|

- Table 1: Default training hyperparameters for the presented 1B LLaMA pre-training experiments.

- 3.2 Evaluation Metrics

To assess the effectiveness of ZClip, we employed a combination of qualitative and quantitative metrics that capture various aspects of training stability, model quality, and gradient behavior:

- • Training stability was qualitatively evaluated by examining the smoothness of the loss curve and the number of (or absence of) loss spikes during training. A stable training process should exhibit consistent convergence without abrupt fluctuations, which can disrupt optimization and lead to suboptimal performance.
- • The quality of the model was measured by evaluating its performance on downstream benchmark tasks, specifically HellaSwag [17] and WinoGrande [18]. These benchmarks provide insight into how well the model generalizes and performs on real-world tasks, serving as a proxy for the impact of gradient clipping on overall model effectiveness.

- • To analyze gradient behavior, we tracked in particular the mean and standard deviation of the gradient norm over time. This analysis helped us understand the behavior of the clipping mechanism and its ability to adapt to the dynamic nature of training.

- 3.3 Learning Rate Regimes We evaluated ZClip across the spectrum of feasible learning rates:

High learning rates (in our scenario 5.0 × 10−3 - 3.0 × 10−3) often lead to unstable training and, if not properly managed, can result in irrecoverable divergence. However, training with high learning rates has the potential to enable faster convergence, which can significantly reduce training time and token budget requirements.

At lower learning rates (in our scenario 1.0 × 10−3 - 1.0 × 10−4), training becomes more stable, but small gradients may still cause subtle instabilities that accumulate over time. This scenario evaluates whether ZClip can effectively handle minor fluctuations without overly regularizing benign updates, ensuring that the method remains adaptive and efficient even in more stable regimes.

4 Results and Analysis

- 4.1 Performance at High Learning Rates

At higher learning rates, such as 3.0 × 10−3, enabling ZClip led to noticeable improvements in training stability, loss convergence, and downstream benchmark performance (see Table 2 and Figure 4). In the particular scenario presented, the model trained with ZClip and a learning rate of 3.0 × 10−3 reached the best baseline validation loss more than 35% faster in terms of training steps compared to a baseline model trained with fixed-threshold gradient norm clipping and the best learning rate of 5.0 × 10−4. For a learning rate of 3.0 × 10−3 the baseline did not converge. By supporting stable optimization at more aggressive learning rates, ZClip expands the viable hyperparameter space and accelerates convergence.

|Learning Rate<br><br>|Train Loss Fixed Threshold ZClip|HellaSwag Fixed Threshold ZClip|WinoGrande Fixed Threshold ZClip|
|---|---|---|---|
| | | | |
|5e-3<br><br>3e-3<br><br>1e-3 7e-4<br><br>5e-4 3e-4<br><br>1e-4<br><br><br><br><br><br><br>|7.28 5.91 5.14 2.14 2.34 2.18 2.29 2.20 2.27 2.24 2.34 2.33 2.46 2.46|25.89 25.27 25.60 50.82<br><br>43.01 49.30<br><br>44.27 46.94<br><br>45.27 45.62<br><br><br>42.34 42.56 33.92 34.12<br><br>|49.32 48.46<br><br>49.03 53.27 52.32 54.85 54.14 52.72 52.32 52.48<br>50.35 51.53<br>51.22 53.35<br>|

- Table 2: Comparison of ZClip to fixed-threshold clipping across a range of learning rates on train loss and downstream task performance. For learning rates of 3 × 10−3 and 5 × 10−3, we set

zthres = 2.0. For the other experiments, the clipping parameters were zthres = 2.5 for ZClip and c = 1.0 for fixed-threshold clipping, respectively.

That being said, we caution that ZClip is not a complete substitute for principled learning rate tuning—extremely high learning rates can still lead to divergence. At an even higher learning rate of 5.0 × 10−3, both fixed-threshold clipping and ZClip diverged. As shown in Figure 5, the gradient norm remained persistently high, causing continuous clipping for both strategies and ultimately leading to divergence.

#### 4.2 Performance at Lower Learning Rates

At lower learning rates, ZClip demonstrated its ability to handle minor gradient fluctuations without overregularizing benign updates, resulting in slightly better performance compared to fixed-threshold gradient norm clipping method (see Table 2 and Figure 6). We hypothesize that this is due to the ability of ZClip to suppress high-frequency gradient noise more effectively in advanced training stages. In the bigger picture, ZClip does not seem to degrade performance in already stable settings, preserving the natural convergence behavior of the model.

In the following, we focus on the learning rate of 1.0 × 10−3, as it strikes a balance between being high enough to challenge stability, and low enough to allow for meaningful gradient dynamics when comparing ZClip against other clipping methods. Table 3 provides the number of loss spikes observed during training for

[Figure 4]

- Figure 4: Comparison of Test Loss between ZClip (lr=3e-3) and a baseline model (lr=5e-4) on a 50B token corpus. ZClip achieved the same final loss as the baseline while requiring 18.6B fewer tokens to get there. The plot uses log-scaled training loss for visibility, and smoothing has been applied to reduce noise. ZClip allows for faster convergence without compromising on final loss value.

[Figure 5]

[Figure 6]

[Figure 7]

Learning Rate = 5.0 × 10−3

[Figure 8]

[Figure 9]

[Figure 10]

Learning Rate = 3.0 × 10−3

- Figure 5: ZClip and fixed-threshold clipping at higher learning rates. Each row shows training loss (left), gradient norm before clipping (middle), and after clipping (right). For 3.0 × 10−3 ZClip stabilized gradients and reduces post-clipping spikes, unlike fixed-threshold clipping which accumulates instability. For

- 5.0 × 10−3 both clipping methods saturated.

[Figure 11]

[Figure 12]

[Figure 13]

#### Learning Rate = 1.0 × 10−3

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Learning Rate = 7.0 × 10−4

[Figure 19]

Learning Rate = 5.0 × 10−4

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Learning Rate = 3.0 × 10−4

[Figure 24]

[Figure 25]

#### Learning Rate = 1.0 × 10−4

- Figure 6: ZClip and fixed-threshold clipping at lower learning rates. Each row shows training loss (left), gradient norm before clipping (middle), and after clipping (right). ZClip preserves stability and convergence also at smaller learning rates, while fixed-threshold still struggles with (benign) spikes.

|Method<br><br>|Spike Count|Test Loss<br><br>|HellaSwag (%)<br><br>|WinoGrande (%)|
|---|---|---|---|---|
|No Clipping Clipping=1.0 Autoclip ZClip<br><br>|12 6 0 0|2.90 2.33 2.20 2.18<br><br>|31.01 43.01 48.10 49.30|51.01 52.32 53.67 54.85<br><br>|

#### Table 3: Performance comparison of gradient clipping methods on HellaSwag and Winogrande after 50BT with a learning rate of 1e-3.

the four scenarios under investigation; both AutoClip and ZClip managed to suppress all spikes, but ZClip outperformed in terms of downstream benchmarks.

[Figure 26]

[Figure 27]

(a) (b)

- Figure 7: Comparison of gradient norms for ZClip and fixed-threshold clipping with threshold value of 1.0, trained with a learning rate of 1e-3. (a) Before clipping: ZClip exhibits small, transient spikes early in training, while fixed-threshold clipping shows larger and more frequent deviations. (b) After clipping: ZClip effectively suppresses these fluctuations, maintaining smooth and stable norms. In contrast, fixed-threshold clipping fails to adapt to the evolving distribution, resulting in persistent post-clipping spikes and instability.

- Figure 7 provides a visual comparison of the gradient norm distribution as training progresses. Unlike fixed-threshold clipping, which suffered from wide variance and persistent post-clipping spikes due to its static threshold, ZClip produced a smoother and more controlled distribution. This demonstrated ZClip’s ability to adaptively manage extreme outliers without suppressing smaller, informative updates, striking a balance that enhances both stability and performance.

### 5 Conclusion

In this work, we introduced ZClip, an adaptive gradient clipping algorithm designed to address the limitations of fixed-threshold gradient norm clipping in training large-scale language models (LLMs). ZClip leverages z-scores for anomaly detection and tracks the evolution of mean and standard deviation using exponential moving averages (EMA). By dynamically adapting to the changing dynamics of gradient norms, ZClip effectively mitigates loss spikes and enhances training stability, making it particularly well suited for the challenges of modern LLM training.

In our experiments on 1B parameter LlaMA models, ZClip consistently outperformed both fixed-threshold clipping and percentile-based approaches. It expanded the space of feasible learning rates, enabling faster overall convergence. In traditionally stable learning rate regimes, it mitigated spikes completely, and provided an uptick in downstream task performance. In future work, we plan to present experiments across a wider set of architectures and larger model sizes, in particular at the 7B to 70B scale. Furthermore, we are interested in evaluating ZClip in other traditionally noisy training scenarios such as reinforcement learning (RL) and multimodality.

### References

- [1] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020.
- [2] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, and Brian Fuller. Llama 2: Open foundation and fine-tuned chat models, 2023.
- [3] DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, and Dongjie Ji. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024.
- [4] Gemini Team. Gemini: A family of highly capable multimodal models, 2024.
- [5] Quentin Malartic, Nilabhra Roy Chowdhury, Ruxandra Cojocaru, Mugariya Farooq, Giulia Campesan, Yasser Abdelaziz Dahou Djilali, Sanath Narayan, Ankit Singh, Maksim Velikanov, Basma El Amel Boussaha, Mohammed Al-Yafeai, Hamza Alobeidli, Leen Al Qadi, Mohamed El Amine Seddik, Kirill Fedyanin, Reda Alami, and Hakim Hacid. Falcon2-11b technical report, 2024.
- [6] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, and Ben Hutchinson. Palm: Scaling language modeling with pathways, 2022.
- [7] Zhengzhong Liu, Bowen Tan, Hongyi Wang, Willie Neiswanger, Tianhua Tao, Haonan Li, Fajri Koto, Yuqi Wang, Suqi Sun, Omkar Pangarkar, Richard Fan, Yi Gu, Victor Miller, Liqun Ma, Liping Tang, Nikhil Ranjan, Yonghao Zhuang, Guowei He, Renxi Wang, Mingkai Deng, Robin Algayres, Yuanzhi Li, Zhiqiang Shen, Preslav Nakov, and Eric Xing. Llm360 k2: Scaling up 360-open-source large language models, 2025.
- [8] Sho Takase, Shun Kiyono, Sosuke Kobayashi, and Jun Suzuki. Spike no more: Stabilizing the pre-training of large language models, 2024.
- [9] Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks, 2013.
- [10] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2017.
- [11] Abhimanyu Dubey, Aaron Grattafiori, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and Arun Rao. The llama 3 herd of models, 2024.
- [12] Qian Liu, Xiaosen Zheng, Niklas Muennighoff, Guangtao Zeng, Longxu Dou, Tianyu Pang, Jing Jiang, and Min Lin. Regmix: Data mixture as regression for language model pre-training, 2025.
- [13] E. Almazrouei and et al. Smollllm: Efficient scaling strategies. ArXiv, 2023.
- [14] Prem Seetharaman, Gordon Wichern, Bryan Pardo, and Jonathan Le Roux. Autoclip: Adaptive gradient clipping for source separation networks, 2020.
- [15] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023.
- [16] Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlíček, Agustín Piqueres Lajarín, Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, Xuan-Son Nguyen, Clémentine Fourrier, Ben Burtenshaw, Hugo Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, Colin Raffel, Leandro von Werra, and Thomas Wolf. Smollm2: When smol goes big – data-centric training of a small language model, 2025.

- [17] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence?, 2019.
- [18] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale, 2019.
- [19] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online, October 2020. Association for Computational Linguistics.

- 6 Appendix

- 6.1 Model and Tokenizer Details Model Architecture

Our model configuration is derived from a 1B parameter model class based on the LLaMA design as implemented in Hugging Face Transformers [19] version 4.47.1. It consists of 16 Transformer decoder layers with a hidden size of 2048 and intermediate size of 5440. The model uses RMSNorm, SwiGLU activation, and supports rotary positional embeddings up to 2048 tokens. The configuration parameters are summarized in Table 4.

|Parameter<br><br>|Value|
|---|---|
|Hidden Size Intermediate Size (FFN) Number of Hidden Layers Number of Attention Heads Number of Key-Value Heads Activation Function Normalization Type RMSNorm Epsilon Vocabulary Size Max Position Embeddings<br><br>|2048 5440 16 16 16 SwiGLU RMSNorm 1 × 10−5 65,536 2048|

#### Table 4: Model Configuration used for ZClip experiments.

The use of SwiGLU activation improves gradient flow during training and complements the adaptive clipping properties of ZClip. RMSNorm provides a stable normalization baseline without introducing additional learnable parameters.

#### Tokenizer

In our experiments, we used a custom tokenizer derived from the LLaMA 3 tokenizer. The tokenizer is truncated to include only the 65536 = 216 most common tokens as computed on a random sample of 90% FineWeb-Edu and 10% Python-Edu. Other than that, it preserves key properties such as Unicode normalization, whitespace collapsing, and byte-pair encoding. This tokenizer was selected for its increased computational efficiency and compatibility with both code and natural language text.

#### 6.2 Parameterizing ZClip as a Percentile-based Approach

Rather than defining a threshold zthres in z-space, analogously to AutoClip one can define a maximum target percentile p and derive the corresponding z-value zp from the standard normal distribution for use in ZClip. For example, for p = 0.99 one obtains z0.99 ≈ 2.32635. In contrast to AutoClip which computes a clipping threshold by extracting an empirical percentile from the full history of gradient norms, the ZClip percentile approach still leverages exponentially weighted estimates of the gradient distribution to better account for the intrinsic shift in distribution as training progresses.

#### 6.3 ZClip - “Clipping to mean and max”

For clipping to max, i.e. zt∗ = µt + zthres · σt, we still observed small spikes (see Figure 8). clipping to mean, on the other hand, is the most aggressive form of clipping. It eliminated spikes, but performed worse in downstream tasks (see Table 5).

|Clipping Strategy|Spike Count<br><br>|Test Loss<br><br>|HellaSwag (%)<br><br>|WinoGrande (%)|
|---|---|---|---|---|
|max mean reciprocal|1 0 0<br><br>|2.19 2.18 2.18<br><br>|48.02 48.90 49.30<br><br>|53.35 54.22 54.85|

#### Table 5: Downstream task performance on HellaSwag and Winogrande benchmarks compraing three ZClip variants—max, reciprocal, and mean. The learning rate for all experiments is 1.0 × 10−3. The token budget was 50BT.

[Figure 28]

#### Figure 8: Training loss for three ZClip variants—max, reciprocal, and mean. The learning rate for all experiments is 1.0 × 10−3.

- 6.4 Normality Assumption of Gradient Norms

ZClip relies on the assumption that gradient norms follow an approximately normal distribution over short temporal windows. This assumption underpins the use of z-scores for detecting statistical anomalies and guiding adaptive clipping thresholds.

- 6.4.1 Empirical Validation

To assess the validity of this assumption, we collected global gradient norms at various phases of training and fitted Gaussian curves to short-range sliding windows (e.g., 135 steps). Figure 9 shows two representative examples.

[Figure 29]

- (a)

[Figure 30]

- (b)

##### Figure 9: Distribution of gradient norms in (a) Early training (steps 500–635) and (b) Mid training (steps 3050–3185). The black curve shows the best-fit normal distribution. Early training exhibits mild skewness and heavier tails, while mid training displays improved symmetry and tighter variance.

#### 6.4.2 Interpretation and Relevance to ZClip

In the early stages of training, we observe heavier tails and a mild right skew, reflecting initial instability and a learning rate ramp-up. In fact, when testing the distribution for training steps 500–635, the statistical tests did not support normality, indicating that the distribution did not yet conform to a normal curve. Despite this, the central mass of the distribution still closely approximated a normal curve, making z-score–based anomaly detection somewhat reliable even in imperfect settings. As training progresses and gradients stabilize, the distribution becomes tighter and more symmetric, reinforcing the validity of the normality assumption during mid- to late-stage training. To validate these assumptions, we conducted the Shapiro–Wilk test during training steps 3050–3185, which yielded a statistic of 0.9792 with a p-value of 0.0508.

We emphasize that ZClip does not require exact normality — it only requires that the distribution is sufficiently well behaved so that the mean and standard deviation can be used to derive a meaningful threshold. Although gradient norms are not perfectly Gaussian—particularly early in training—they seem to be sufficiently regular and unimodal to support robust, statistics-informed clipping using the z-score formulation. In practice, the exponential moving average estimates of these statistics effectively smooth out noise and adapt to gradual distributional drift.

- 6.5 Throughput Analysis

Throughput is a critical factor when training large language models. We evaluated the impact of enabling ZClip on training throughput in our experiments and observed that its lightweight, EMA-based computations introduced negligible overhead (see Figure 10).

[Figure 31]

Figure 10: Training throughput comparison between fixed-threshold gradient clipping, AutoClip, and ZClip methods.

- 6.6 Hyperparameter Tuning

- 6.6.1 Z-Score Threshold

The z-score threshold determines the degree of sensitivity towards anomalies in the gradient norms. Table 6 presents the impact of varying the z-score threshold on training stability and convergence. Thresholds between 2.0 and 3.0 yielded the best overall trade-off, fully suppressing gradient spikes while preserving loss convergence and downstream performance. Lower thresholds introduce over-clipping, which can hinder learning, whereas higher thresholds fail to prevent occasional loss spikes.

|Z-Score Threshold|Train Loss<br><br>|Spike Count<br><br>|HellaSwag|Winogrande|
|---|---|---|---|---|

- 1.5 2.182 0 48.70 52.23

- 2.0 2.180 0 48.75 54.23

- 2.5 2.180 0 49.30 54.85

- 3.0 2.194 0 48.94 53.27

- 3.5 2.202 1 48.30 53.67

- 4.0 2.201 1 48.58 55.40

- Table 6: Impact of Z-Score threshold on downstream task performance. All models trained with a learning rate of 1.0 × 10−3 and a token budget of 50B tokens.

Clipping Percentage Analysis Figure 11 illustrates how the proportion of clipped gradients evolves as training progresses for various z-score thresholds. When the threshold is lower (e.g., 1.5), a larger fraction of gradients are clipped, reflecting a more aggressive approach to suppressing potential outliers. Conversely, higher thresholds (e.g., 4.0) are more permissive and result in fewer clipped updates.

In our experiments, we tested thresholds ranging from 1.5 to 4.0 in increments of 0.5 and found that a threshold of 2.5 provides the most balanced trade-off. Specifically, at zthres = 2.5, the fraction of clipped gradients remains moderate, effectively removing harmful spikes without overly constraining gradients. This leads to more stable training and faster convergence compared to either excessively low or high thresholds.

[Figure 32]

- Figure 11: Clipping percentage vs. training step for different z-score thresholds. Lower thresholds yield higher clipping percentages, indicating more aggressive clipping. The learning rate for all experiments is 1.0 × 10−3. The token budget was 50BT.

#### 6.6.2 Alpha Value

We experimented with α values ranging from 0.9 to 0.99 for ZClip’s exponential moving average (EMA). A higher α assigns more weight to historical gradients, leading to smoother updates but slower adaptation to changes in distribution over time. Conversely, lower α values allow quicker adjustments but may introduce instability. Our results indicate that tuning α appropriately helps balance stability and responsiveness in gradient clipping. Figure 12 shows the effect of different α values on the EMA-estimated mean, while Figure 13 illustrates the corresponding standard deviation estimates. Table 7 further shows that all configurations successfully suppressed gradient spikes, but the choice of alpha did affect downstream task performance. An intermediate setting of α = 0.97 achieved the best trade-off.

|Alpha<br><br>|Train Loss|Spike Count|HellaSwag|Winogrande|
|---|---|---|---|---|

0.90 2.190 0 48.04 54.01 0.93 2.185 0 48.99 54.38 0.95 2.186 0 48.91 53.98

- 0.97 2.180 0 49.30 54.85

- 0.98 2.189 0 48.51 52.09

- 0.99 2.189 0 48.45 53.74

- Table 7: Impact of different values for α on stability and downstream performance. All models

trained with a learning rate of 1.0 × 10−3 and a z-score threshold zthres = 2.5. The token budget was 50B tokens.

[Figure 33]

- Figure 12: EMA-estimated mean for different α values. The learning rate for all experiments is 1.0 × 10−3, and the threshold is zthres = 2.5. The token budget was 50BT.

[Figure 34]

- Figure 13: EMA-estimated standard deviation for different α values. The learning rate for all experiments is 1.0 × 10−3, and the threshold is zthres = 2.5. The token budget was 50BT.

