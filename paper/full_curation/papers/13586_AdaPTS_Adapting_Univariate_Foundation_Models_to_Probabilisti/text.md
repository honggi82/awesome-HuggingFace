### AdaPTS: Adapting Univariate Foundation Models to Probabilistic Multivariate Time Series Forecasting

Abdelhakim Benechehab12 Vasilii Feofanov1 Giuseppe Paolo1 Albert Thomas1 Maurizio Filippone3 Bal´azs K´egl1

# arXiv:2502.10235v1[stat.ML]14Feb2025

#### Abstract

Pre-trained foundation models (FMs) have shown exceptional performance in univariate time series forecasting tasks. However, several practical challenges persist, including managing intricate dependencies among features and quantifying uncertainty in predictions. This study aims to tackle these critical limitations by introducing adapters—feature-space transformations that facilitate the effective use of pre-trained univariate time series FMs for multivariate tasks. Adapters operate by projecting multivariate inputs into a suitable latent space and applying the FM independently to each dimension. Inspired by the literature on representation learning and partially stochastic Bayesian neural networks, we present a range of adapters and optimization/inference strategies. Experiments conducted on both synthetic and real-world datasets confirm the efficacy of adapters, demonstrating substantial enhancements in forecasting accuracy and uncertainty quantification compared to baseline methods. Our framework, AdaPTS, positions adapters as a modular, scalable, and effective solution for leveraging time series FMs in multivariate contexts, thereby promoting their wider adoption in real-world applications. We release the code at https://github.com/abenechehab/AdaPTS.

#### 1. Introduction

Time series forecasting is a well-established machine learning problem that involves analyzing sequential data to predict future trends based on historical patterns. Two key challenges frequently arise in this context: (a) time series are often multivariate, incorporating multiple descriptive

1Huawei Noah’s Ark Lab, Paris, France 2Department of Data Science, EURECOM 3Statistics Program, KAUST. Correspondence to: Abdelhakim Benechehab <abdelhakim.benechehab@gmail.com>.

Preprint, under review. Copyright 2025 by the author(s).

ETTh1 (H = 96)

460 480 500 520 540 560 580 600

Time Steps

Ground Truth Moment Moment+AdaPTS

±1, 3, 5 std

| |
|---|

- (a)

[Figure 1]

- (b)

Figure 1: (a) Augmenting Moment time series foundation model with the AdaPTS framework provides probabilistic and more accurate predictions. (b) The AdaPTS framework: The input time series is transformed through a feature space transformation φ that maps into a stochastic latent space. The prediction is then conducted using a pre-trained FM before transforming back the predicted, now distribution, to the original feature space. The fire symbol indicate trainable weights while the snowflake implicates that the parameters of the FM are kept frozen.

features (Wei, 2019), and (b) estimating the uncertainty of a forecast is equally important, requiring probabilistic model outputs (Gneiting & Katzfuss, 2014). These challenges are particularly relevant in real-world applications where risk assessment depends on reliable forecasts, such as healthcare (Jones & Spiegelhalter, 2012), finance (Groen et al., 2013), energy management (Zhang et al., 2014; Nowotarski & Weron, 2018), and weather prediction (Palmer, 2012; Bi et al., 2023).

Existing foundation models (FMs) for time series forecasting, such as Chronos (Ansari et al., 2024), are typically trained for univariate forecasting tasks due to tractability constraints, as the wide range of real world time series problems typically have different numbers of features. Even

without discretization, handling multivariate time series directly within these models (Moment (Goswami et al., 2024), Moirai (Liu et al., 2024)) remains computationally challenging due to the high-dimensional dependencies among features. This limitation raises a fundamental question: how can we leverage existing pre-trained univariate FMs to enable probabilistic forecasting for multivariate time series?

To address this, we introduce AdaPTS, a novel framework designed to augment FMs with probabilistic adapters. As illustrated in Figure 1, AdaPTS applies a stochastic feature transformation that maps the input time series into a latent space, where predictions are made using the frozen FM. Our framework sets itself apart from existing literature by enforcing an invertibility constraint on the adapter, allowing predictions to be transformed back into the original feature space. Beyond enhancing forecasting accuracy, the integration of stochasticity into the adapter’s latent representation ensures that the model captures uncertainty, thereby improving both calibration and robustness.

Our approach leads to several novel insights and contributions, which we summarize as follows:

- 1. Multivariate FM adaptation. We introduce a principled methodology for adapting existing pre-trained univariate FMs to multivariate probabilistic forecasting, resulting in the AdaPTS framework.
- 2. Theoretical foundations of adapters. We provide a theoretical analysis to support the necessity of adapters, starting with the analytically tractable case of linear adapters and linear FMs. We then build on the literature on partially stochastic Bayesian neural networks to introduce probabilistic adapters.
- 3. Empirical validation. We conduct extensive experiments on multivariate time series forecasting benchmarks, demonstrating that our approach improves forecasting accuracy baseline methods. We also analyze the interpretability of the learned latent representation and show that adapters enable cost-effective adaptation by reducing the dimensionality of the feature space.

The rest of this paper is organized as follows: Section 2 discusses related work, Section 3 details the problem setup and the theoretical analysis on linear adapters. Section 4 extends our framework to probabilistic adapters, and Section 5 showcases experimental results. Finally, we conclude with limitations and future directions in Section 6.

#### 2. Related work

Time Series Foundational Models. Over the past two years, a plethora of foundation models have been pro-

posed with a particular focus on time series forecasting. Some of these models like GPT4TS (Tian et al., 2023) and Time-LLM (Jin et al., 2024) “reprogram” a Large Language Model to the forecasting setting by freezing most of its layers and fine-tuning additional time series-specific modules to a new downstream task. The majority of these time series FMs including Lag-Llama (Rasul et al., 2024), Chronos (Ansari et al., 2024), Moirai (Liu et al., 2024), TimesFM (Das et al., 2024) and Moment (Goswami et al., 2024) are trained from scratch on a large volume of time series data.

Adapters. The multivariate setting presents a significant challenge for time series FMs, as different tasks involve varying numbers of channels1. To the best of our knowledge, the only model that naturally accommodates any number of channels is Moirai (Liu et al., 2024), which, however, suffers from high computational demand due to processing all channels flattened in the transformer simultaneously, leading to a quadratic memory complexity w.r.t. to the number of channels. Most foundation models, instead, treat each one of these independently, which, as noted by Feofanov et al. (2024), remains computationally expensive when full finetuning is required. For classification tasks with hundreds or thousands of features, they demonstrated that simple adapters like the rotation matrix obtained through Principal Components Analysis (PCA) mitigate this issue. At the same time, Benechehab et al. (2025) showed that PCA preserves channel interactions by learning new disentangled components. However, in both cases, PCA provided little improvement over independent processing, leaving room for further enhancements. In the context of tabular regression, foundation models such as (Ma et al., 2024, TabDPT) also use PCA to adapt to a variable number of features.

Less related to our work, Li & Marlin (2016) use a Gaussian process adapter in the context of irregular time series classification. In other domains, adapters have been used for multimodal (text-time series) representation learning (Zhang et al., 2024) and computer vision (Li et al., 2025; Yin et al., 2023; Pan et al., 2022).

#### 3. Adapters

##### 3.1. Problem setup

Consider a multivariate long-term time series forecasting task, represented by: a data matrix X ∈ RL×D where L is the context window size and D is the multivariate time series dimensionality, and a target matrix Y ∈ RH×D, where H is the forecasting horizon. We denote by xd ∈ RL×1 (respectively yd ∈ RH×1) the d-th component of the input

1Throughout the paper, the words features, channels, and components are used interchangeably to refer to the number of variates in a multivariate time series, represented as D in Section 3.

(respectively target) multivariate time series.

Our goal is to use a frozen pre-trained univariate time series foundation model denoted as fFM : RL×1 → RH×1 (Fig. 1b) and exploit the information stored in its weights to achieve the best forecasting performance, measured by the mean squared error (MSE) loss:

L = ∥Y − fFM(X)∥2F (1)

On multivariate time series, for simplicity, we denote by fFM(X) the application of fFM to each channel independently, in which case the loss can be written as:

D d=1 ∥yd − fFM(xd)∥22.

1 D

We now formally define an adapter, a tool by means of which we aim to best use the foundation model fFM for multivariate forecasting:

Definition 3.1 (adapter). An adapter is a feature-space transformation φ : RD → RD

′

that is applied to the data prior to the foundation model2. The forecast is then obtained by transforming the predictions back to the original feature space:

Yˆ (X;φ) = φ−1 fFM(φ(X))

##### 3.2. Theoretical analysis

The purpose of this section is to study the optimization problem that the adapter φ is aiming to solve:

φ∗ = arg min

∥Y − φ−1 fFM(φ(X)) ∥2F (2)

φ

Under mild assumptions on the adapter function class and the backbone foundation model fFM, we aim at characterizing the optimal solution φ∗ and prove that it realizes the optimality condition: L(φ∗) ≤ L.

We first consider the linear case where we constrain the adapter φ to the class of linear transformations, parametrized by a matrix Wφ ∈ RD×D: φ(X) = XWφ.

- Assumption 3.2. Wφ has full rank: rank(Wφ) = D, insuring its invertibility.
- Assumption 3.3. For ease of derivation, we consider a similar linear parametrization for the foundation model:

fFM(X) = WFM⊤ X + bFM1⊤ where WFM ∈ RL×H, bFM ∈ RH, and 1 a vector of ones of dimension D.

Proposition 3.4 (Optimal linear adapter). Under Assumption 3.2 and Assumption 3.3, the closed-form solution of the problem:

According to this definition, an adapter is valid only if the inverse transformation φ−1 : RD

′

→ RD, such that ∀x ∈ RD, φ−1 ◦ φ(x) = x, is well-defined on RD

′

. In the rest of the paper, we relax this condition by naming the direct transformation as encoder (φ ≜ enc), and respectively, the inverse transformation as decoder (φ−1 ≜ dec). In this case, the predictions obtained after the application of the adapter become: Yˆ (X;enc,dec) = dec fFM(enc(X)) .

We note that in the literature, there exist alternatives to adapt to the multivariate setting (Zhang & Yan, 2023; Tian et al., 2023), but we have chosen this family of adapters due to their high flexibility as: (a) any foundation model can be plugged-in, (b) no requirement of fine-tuning due to feature-level transformations (Feofanov et al., 2024), (c) adaptation to the computation budget by defining the number of encoded channels.

Optimality of an adapter. In order for an adapter to be useful, it has to achieve a lower forecasting error than the identity baseline. In fact, the loss defined in Eq. (1) corresponds to the forecasting loss obtained by using an adapter implementing the identity matrix I. Therefore, we define the optimality of the adapter based on improving the forecasting error of the identity baseline:

###### L ≥ L(φ) = ∥Y − φ−1 fFM(φ(X)) ∥2F

2In practice, φ is applied on matrices X in RL×D. This denotes the application of φ on each row of X.

L(Wφ) = ∥Y − WFM⊤ XWφ + bFM1⊤ Wφ−1∥2F (3) writes as:

Wφ∗ = (B⊤A)+B⊤B (4) where Wφ∗ = arg minW

φ∈GLD(R) L(Wφ), A = Y −

WFM⊤ X, B = bFM1⊤, and (B⊤A)+ denoting the pseudo-inverse operator.

Proof. The result follows by differentiating L(Wφ) with respect to Wφ, and solving the Euler equation: ∇WφL = 0. The detailed proof is deferred to Appendix A.

| |
|---|

Remark 3.5. In this case, the fact that the matrix B = bFM1⊤ have identical columns renders the matrix B⊤A degenerate (with rank(B⊤A) = 1). In practice, we add a positive constant to the diagonal in order to numerically stabilize the matrix inversion: Wφ∗ = (B⊤A+λI)−1B⊤B, with λ > 0. In Section 3.3 we show that we are able to reach an optimal solution regardless of this added regularization.

##### 3.3. Working example

Synthetic data. Our synthetic dataset comprises two multivariate time series, one with several independent components and the other with linearly correlated ones (Fig. 2), designed to evaluate a linear feature-space transformation. The

data generation process creates five (uncorrelated) base signals—sinusoids with distinct frequencies, amplitudes, and iid noise— and derives eight additional channels through linear combinations of these bases with additive Gaussian noise of different magnitude (σ ∈ (0.1,0.2,0.5)). This construction provides a controlled environment where the ground truth relationship between channels is known: the underlying data manifold is effectively five-dimensional, but in the correlated case the observed eight-dimensional multivariate time series includes varying levels of noise and linear mixing.

Randomly generated linear FMs. The experimental setup in Fig. 2 consists in randomly sampling the linear parameters of a toy foundation model: WFM and bFM. To simulate a realistic scenario, we use Glorot-uniform initialization distribution as it would be the case in neural network-based architectures. We then compute the closed-form solution Wφ∗ (Eq. (4)) on raw data X, and compare the resulting loss value with the baseline (using the identity matrix I as adapter) and the PCA-only adapter.

Independent

Correlated

| |
|---|

1e1

- 2

- 3

- 4

MSE

I WPCA W*

###### I WPCA W*

Figure 2: Optimality of Wφ∗. Comparing the MSE obtained with Wφ∗ against the baseline, for 1000 randomly generated linear FM.

- Fig. 2 shows that in the case of uncorrelated data (left column) PCA is equivalent to the identity matrix, while the

solution Wφ∗ to the problem L(Wφ) reaches an order of magnitude better forecasting loss. In the correlated case, we observe that PCA has a similar performance to the optimal solution. This example motivates the adapter idea through the existence of better linear transformations than the identity matrix in the case of linear foundation models.

#### 4. AdaPTS: Adapters for Probabilistic Multivariate Time Series Forecasting

In this section, we introduce families of adapters that verify the conditions stated in Definition 3.1. Furthermore, we extend this definition to include probabilistic variants of adapters, useful for uncertainty quantification on the FM

predictions.

##### 4.1. Families of adapters

Our framework accommodates various families of transformations that can serve as adapters. Initially, we define linear AutoEncoders and subsequently extend them to their deep non-linear counterparts. Ultimately, we introduce the probabilistic adapter framework, encompassing Variational AutoEncoders (VAE) and Dropout-based families of adapters.

Linear AutoEncoders. In addition to the linear setup introduced in Section 3.2, we extend Linear AutoEncoders to provide a simple yet effective method for dimensionality reduction while preserving the temporal relationships within time series data. In this more general case, the encoder compresses the multivariate time series X into a potentially lower-dimensional representation Z = XWθ

, where W ∈ RD×D

enc

′

is the linear transformation matrix, and D′ ≤ D. The decoder reconstructs the forecast to the original feature space after prediction as Yˆ = fFM(Z)Wθ

. Finally, the parameters of the encoder θenc and the decoder θdec are jointly optimized to minimize the objective in Eq. (2).

dec

Deep non-linear AutoEncoders. Deep non-linear AutoEncoders extend their linear counterparts by employing multiple layers of non-linear transformations. The encoder maps the input X to a latent space Z = enc(X;θenc), where enc is parameterized by a deep neural network. Similarly, the decoder reconstructs the predictions of the foundation model in the latent space: Yˆ = dec(fFM(Z);θdec).

Besides AutoEncoders, Normalizing Flows (Kobyzev et al., 2021) such as RealNVP (Dinh et al., 2017) are a valid choice in the context of adapters, thanks to their inherently invertible nature. However, their training may be challenging due to various optimization related concerns. We defer a discussion on Normalizing Flows as adapters to Appendix B.

##### 4.2. Probabilistic Adapters

We now discuss an alternative to the optimization of adapters, which is based on a Bayesian treatment of their parameters. There are many options on how to carry out inference over these parameters, and we can draw from the literature on Bayesian inference for neural networks (Papamarkou et al., 2024).

Considering a FM which yields point predictions, the appeal of Bayesian adapters is that they enable probabilistic predictions, which can be used for uncertainty quantification. Note that this is the case for models such as Chronos and Moirai, which output a distribution over the time series

continuous values3. For deterministic FMs such as Moment (Goswami et al., 2024), a Bayesian treatment of adapters yields an ensemble of such predictions, which is key for accounting for the predictive uncertainty.

Inference. Recalling that θ represents the set of parameters of encoder (encθ) and decoder (decθ), we can attempt to obtain the posterior distribution over these parameters through Bayes theorem (Gelman et al., 2013):

p(θ|Y,X) ∝ p(Y|X,θ)p(θ), where p(θ) is the prior distribution over the parameters and

- p(Y|X,θ) the likelihood, with Y,X representing a training dataset in this context. Alternatively, we can rather treat the latent representation Z as stochastic, where the interest is now to characterize the following posterior:

p(Z|Y,X) ∝ p(Y|X,Z)p(Z).

In these two formulations, the posterior distribution over the parameters, is instrumental in obtaining predictive distributions useful for uncertainty quantification. For instance, in the case of inference over θ, for new test data Y∗,X∗ we obtain:

p(Y∗|X∗,Y,X) = p(Y∗|X∗,θ)p(θ|Y,X)dθ.

Characterizing the posterior analytically, however, is intractable and we need to resort to approximations. The literature on Bayesian inference offers various strategies, which can be adapted to neural networks (Papamarkou et al., 2024), including variational inference (Graves, 2011), Laplace approximations (Yang et al., 2024), and Markov chain Monte Carlo (MCMC) (Chen et al., 2014; Tran et al., 2022).

Within the AdaPTS framework, we focus in particular on variational inference (VI) for VAE adapters and on Monte Carlo dropout (Gal & Ghahramani, 2016) as an approximate form of VI for carrying out inference over θ.

Variational AutoEncoders. Following the Bayesian perspective on adapters, VAE assume a prior distribution over the latent representation Z, typically N(0,I). The encoder then outputs parameters of the posterior distribution

- qϕ(Z|X), and in our context, the decoder generates recon-

structions of predictions Yˆ ∼ pθ(Y|X,fFM(Z)) where θ parametrize a likelihood model p. We then define the training objective of the VAE, which brings together the forecasting loss and a regularization term, in a similar way to the evidence lower bound (ELBO) (Kingma & Welling, 2013) objective:

3In the case of Chronos, this distribution is obtained through a categorical distribution (with softmax probabilities) over a tokenized space of the time series values.

Proposition 4.1 (VAE adapter training objective). The training objective for the VAE adapter is the maximization of an ELBO-like lower bound on the marginal likelihood of the target Y:

log pθ(Y|X,fFM) ≥Eq

ϕ(Z|X) [log pθ(Y|X,fFM(Z))] − KL(qϕ(Z|X)∥p(Z)),

where KL denotes the Kullback-Leibler divergence.

The derivation of this lower bound and a discussion on the implications of each term of the loss are deferred to Appendix A.2.

- Remark 4.2. In practice, we use the Gaussian likelihood as our likelihood model: pθ(Y|X,fFM(Z)) =

N(Y;Yˆ ,σ2I), with Yˆ = decθ(fFM(Z)). In this case the forecasting loss term boils down to the MSE objective in Eq. (2) up to a multiplicative and additive noise-related constants: log N(Y;Yˆ ,σ2I) = −2σ12 ∥Y − Yˆ ∥2F − HD2 log(2πσ2). Notice that one can also learn a model of the noise where decθ(fFM(Z)) = [µθ(Y|X,fFM(Z)),σθ(Y|X,fFM(Z))].

- Remark 4.3. The KL divergence regularization term can be multiplied by a scaling factor β to control the disentanglement—independence of the latent representation components. This results in β-VAE (Higgins et al., 2017), which is what we use in practice while referring to it as the VAE adapter throughout the paper.

Dropout as approximate VI. Dropout (Srivastava et al., 2014) can be interpreted as a form of variational inference, where a variational distribution is imposed over the weights of a neural network (Gal & Ghahramani, 2016). Specifically, applying dropout during training corresponds to approximating a posterior over the weights using a Bernoulli distribution. This perspective allows the deterministic models introduced in Section 4.1, such as Linear AutoEncoders, to be transformed into probabilistic models by introducing stochasticity through dropout.

Treating adapters in a Bayesian manner while keeping the FM fixed aligns with the concept of partially stochastic Bayesian neural networks, which provides theoretical guarantees on universal conditional density estimation (Sharma et al., 2023). This framework ensures that the model can approximate any conditional density, provided that stochasticity is introduced early enough in the architecture and that the number of stochastic units matches or exceeds the output dimension. Using probabilistic adapters, We comply with these conditions by making the encoder stochastic, allowing the learned latent space to capture uncertainty while leveraging the FM’s fixed parameters.

No adapter with adapter

Dataset H

Moment PCA LinearAE dropoutLAE LinearVAE VAE ETTh1

96 0.411±0.012 0.433±0.001 0.402±0.002 0.395±0.003 0.400±0.001 0.404±0.001 192 0.431±0.001 0.440±0.000 0.452±0.002 0.446±0.001 0.448±0.002 0.431±0.001 Illness

24 2.902±0.023 2.98±0.001 2.624±0.035 2.76±0.061 2.542±0.036 2.461±0.008 60 3.000±0.004 3.079±0.000 3.110±0.127 2.794±0.015 2.752±0.040 2.960±0.092

96 0.177±0.010 0.176±0.000 0.169±0.000 0.156±0.001 0.161±0.001 0.187±0.001 192 0.202±0.000 0.208±0.001 0.198±0.001 0.200±0.001 0.204±0.000 0.226±0.000

Weather

96 0.130±0.011 0.147±0.000 0.167±0.013 0.130±0.011 0.243±0.039 0.455±0.010 192 0.210±0.002 0.222±0.000 0.304±0.005 0.305±0.013 0.457±0.020 0.607±0.021

ExchangeRate

Table 1: Performance comparison between the baseline Moment model without adapters against different adapter architectures (PCA, LinearAE, dropoutLinearAE, LinearVAE, and VAE), for multivariate long-term forecasting with different horizons H. We display the average test MSE ± standard error obtained on 3 runs with different seeds. Best results are in bold, with lower values indicating better performance.

#### 5. Experiments & Results

In this section, we empirically demonstrate the quantitative and qualitative superiority of AdaPTS in multivariate longterm time series forecasting on common benchmarks. We show that in most of the considered tasks (datasets and forecasting horizons,) our framework improves the performance of Moment , a commonly used time series forecasting foundation model. The implementation details are provided in Appendix C.2.

##### 5.1. Time series forecasting

Datasets. Our experiments are conducted on four publicly available real-world multivariate time series datasets, commonly used for long-term forecasting (Ilbert et al., 2024; Wu et al., 2021; Chen et al., 2023; Nie et al., 2023; Zeng et al., 2023). These datasets include the Electricity Transformer Temperature dataset (ETTh1) (Zhou et al., 2021), ExchangeRate (Lai et al., 2018), Weather (Institute, 2021), and Influenza-like Illness (for Disease Control & Prevention, 2024). All time series are segmented with an input length of L = 512, prediction horizons H ∈ [96,192] and H ∈ [24,60] for the Illness dataset, and a stride of 1, meaning each subsequent window is shifted by one step. These datasets (detailed in Appendix C.1) originate from various application domains, enabling a comprehensive evaluation of our framework across diverse real-world scenarios.

Baseline. We compare our method against the vanilla application of the foundation model Moment small from the Moment family of models (Goswami et al., 2024). This means that for each dataset, we apply Moment small independently to each feature. Additionally, we compare our learning-based adapters against PCA, an adapter that has been used in the literature for model-based reinforcement

learning (Benechehab et al., 2025) and time series classification (Feofanov et al., 2024).

ETTh1 (H=96)

0.50

MSE

0.45

0.40

Illness (H=24)

4.0

3.5

MSE

3.0

2.5

2 5 7 9 14

Number of Components

pca

linearAE

dropoutLinearAE

linearVAE

VAE baseline

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 3: Impact of the number of components on model performance. The dashed line indicates Moment performance without adapters, the shaded area its standard deviation, and the vertical line the number of original features.

AdaPTS improves the performance of Moment . We present the forecasting error measured by the Mean Squared Error (MSE) in Table 1 and the Mean Absolute Error (MAE) in Appendix D. On the ETTh1 dataset with a prediction horizon H = 96, all adapter-based variants outperform the baseline Moment model, with dropoutLinearAE achieving the best performance, showing an 8% improvement. Similar results are observed for the Illness dataset, where all adapters improve over the baseline. Notably, the VAE achieves a significant 15% improvement, reducing the MSE from 2.902 to 2.461 at H = 24. In the Weather dataset, the dropoutLinearAE adapter shows the best improvement across all adapter architectures for H = 96, while its deterministic counterpart, LinearAE, takes the lead at H = 192. The results on the ExchangeRate dataset

PCA (Explained variance: 95.6%)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

4

2

PC2

0

2

4

0 5 10 15

PC 1

dropoutLinearAE

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

5

Latentdim2

0

5

10

15

10 5 0 5

Latent dim 1

LinearVAE

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

4

2

Latentdim2

0

2

4

6 4 2 0 2 4

Latent dim 1

Latentdim2

Training Test

VAE

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- 0
- 1
- 2

1

2

3

4 2 0 2

Latent dim 1

Figure 4: Visualization of the latent representation obtained by different adapters (with number of components equal to 2) on Illness(H = 24). Shaded colors indicate the time dimension, with lighter colors representing earlier timesteps.

are mixed, with some adapters matching the baseline performance (dropoutLinearAE at H = 96) while others show degraded performance, particularly at a longer prediction horizon (H = 192), which is also observed for the ETTh1 dataset. Overall, AdaPTS improves the forecasting accuracy of Moment in 5 out of the 8 considered tasks, matches its performance in 2, and degrades performance in 1 task.

##### 5.2. Dimensionality Reduction

Fig. 3 illustrates the impact of varying latent space dimensions on forecasting performance across different adapters. For the ETTh1 dataset with a 96-step horizon, all adapter architectures achieve optimal performance at 7 components (matching the original feature count), with MSE values consistently lower than the baseline. Notably, at just 5 components, all adapters (except the PCA baseline) match the baseline score, demonstrating the suitability of our framework for low-resource setups through dimensionality reduction. The Illness dataset (H = 24) presents more compelling results, as the VAE adapter achieves significantly optimal performance with only 2 components, underscoring the potential of our approach for cost-effective adaptation of time series foundation models. Ultimately, we find that expanding dimensionality beyond the original feature count does not yield further improvements, as no adapter shows notable enhancements past this point.

##### 5.3. Interpretability of the latent representations

- Fig. 4 compares the representation learning capabilities of different adapters on the Illness(H = 24) dataset, focusing on their ability to distinguish between training and test data. To visualize the raw dataset, we employ PCA for dimensionality reduction, retaining only two principal components, which is justified by the 95.6% explained variance. When representing the training and test datasets in the space of the first two principal components, we observe a clear distribu-

tion shift, potentially complicating the forecasting task for the baseline foundational model. In contrast, using AdaPTS results in well-overlapping Gaussian distributions for the training and test data in the latent space. This demonstrates our framework’s ability to enforce a structured, isotropic representation that mitigates distribution shift. This effect is particularly pronounced with the VAE adapter and, to a lesser extent, with LinearVAE and dropoutLinearAE.

The findings emphasize the advantages of VAE in managing distribution shift, a critical challenge in time series representation learning. By modeling uncertainty and enforcing a continuous latent space, VAE enhance generalization, making them especially valuable for real-world applications where test distributions differ from training data. This aligns with the objective of utilizing adapters in foundational models to optimize zero-shot performance, ensuring robustness across various tasks without extensive fine-tuning.

##### 5.4. On the calibration of the probabilistic adapters

To evaluate the calibration of our adapter-based probabilistic forecasters, we use quantile calibration as depicted in the reliability diagram in Fig. 5. In an ideal scenario, a well-calibrated probabilistic forecast should align with the red dashed diagonal, indicating that the empirical proportion of observations falls within the predicted quantiles at the expected rate. The overall conclusion is that we observe a gradual deviation from ideal calibration as the prediction horizon increases (darker shades). While early prediction horizons display reasonably well-calibrated predictions, longer-horizon forecasts

1.0

| | | | |
|---|---|---|---|
| | | | |

0.8

proportion

0.6

0.4

0.2

0.0

0.0 0.5 1.0

quantile

[Figure 2]

0 25 50 75

Prediction Horizon

Figure 5: Reliability diagram for the first feature of the ETTh1 (H = 96) dataset using LinearVAE.

systematically underestimate uncertainty, as shown by the curve falling below the diagonal. This indicates that observed values exceed predicted quantiles more frequently than expected, suggesting that the predictive distribution becomes too narrow, resulting in overconfident forecasts.

##### 5.5. Ablation studies

Influence of σ and β in the VAE Adapter. Fig. 6 illustrates an ablation study examining the β parameter in β-VAE and the noise scale σ of the likelihood model applied to the prediction Yˆ , assessing their effects on MSE and Expected Calibration Error (ECE). The MSE heatmap (left) demonstrates that increasing β generally diminishes MSE, with the lowest values observed at β = 2.0 and β = 4.0, particularly for higher log σ2. This indicates that stronger regularization through β can enhance forecasting accuracy, possibly due to the disentangling effect of regularization towards a prior distribution with statistically independent components. Conversely, the ECE heatmap (right) shows that higher β and log σ2 values result in lower calibration error, with optimal results at β = 4.0 and log σ2 = 3.0. This outcome is anticipated, as larger values of β and σ mitigate overfitting, where the model tends to exhibit overconfidence in its predictions. Additionally, it is observed that maintaining a fixed σ during training generally outperforms including it in the optimization loop, a configuration denoted as auto in Fig. 6.

###### MSE

ECE

0.28

|2.4|6 2.4|3 2.4|3 2.4|5 2.4|3 2.5|8|
|---|---|---|---|---|---|---|
|2.4|5 2.4|2 2.4|4 2.4|1 2.4|0 2.5|9|
|2.4|2 2.4|4 2.4|1 2.4|0 2.3|9 2.5|6|
|2.4|4 2.3|9 2.4|2 2.4|3 2.4|1 2.6|8|
| | | | | | | |

|0.2|8 0.2|8 0.2|8 0.2|8 0.2|7 0.2|8|
|---|---|---|---|---|---|---|
|0.2|8 0.2|8 0.2|8 0.2|7 0.2|7 0.2|8|
|0.2|8 0.2|8 0.2|7 0.2|7 0.2|7 0.2|8|
|0.2|7 0.2|7 0.2|7 0.2|7 0.2|6 0.2|8|
| | | | | | | |

[Figure 3]

[Figure 4]

0.51.02.04.0

2.64

0.28

2.56

0.27

2.48

0.26

2.40

0.5 1.0 1.5 2.0 3.0 auto

0.5 1.0 1.5 2.0 3.0 auto

log 2

log 2

Figure 6: β and log σ2 VAE hyperparameters ablation on the Illness(H = 24) dataset. For reference, the Moment baseline score on this task is 2.902±0.023.

LinearAE components. The ablation study presented in Fig. 7 examines the performance of different components of the linear autoencoder adapter (LinearAE) across three datasets: ETTh1, Weather, and ExchangeRate. The figure compares the full linear autoencoder with its encoder-only (LinearEncoder) and decoder-only (LinearDecoder) variants. Overall, the results reveal that the decoder component of the linear autoencoder plays the most important role in minimizing the forecasting error across all datasets. The encoder-only variant’s contribution varies, being more impactful in the Weather dataset compared to ETTh1 and ExchangeRate. These findings highlight the significance of the decoder in the LinearAE

adapter and suggest that, in the deterministic case, a decoder might be sufficient to capture feature dependencies.

0.5

###### MSE

0.0

ETTh1 Weather ExchangeRate

linearEncoder

linearAE

linearDecoder

| |
|---|

| |
|---|

Figure 7: LinearAE components ablation.

Nevertheless, as shown in our previous experiments, particularly Table 1, probabilistic adapters generally outperformed the deterministic ones. This underscores the importance of the encoder as well, which is responsible for approximating the posterior distribution in the latent space—a mechanism inherent to our probabilistic framework.

#### 6. Conclusion

In this paper, we investigate how pre-trained univariate time series foundation models can be adapted for probabilistic multivariate forecasting. To address this challenge, we introduce the AdaPTS framework. Our method offers a novel approach to training feature space transformations that facilitate uncertainty quantification and enhance the performance of baseline foundation models. Through a series of experiments, we demonstrate that our framework improves forecasting accuracy, provides reasonably well-calibrated uncertainty estimates, reduces inference cost through dimensionality reduction, and offers interpretable feature space latent representations.

Limitations & Future directions. Our work establishes a principled framework for adapting pre-trained univariate foundation models to multivariate probabilistic time series forecasting. While we focus on Moment , our approach can be applied on other univariate deterministic FMs; we leave this direction to future work. Additionally, while variational inference is chosen for its efficiency, exploring alternative methods like Markov Chain Monte Carlo could improve uncertainty estimation, albeit at a higher computational cost.

Another promising direction is refining the calibration of our uncertainty estimates. While our framework flexibly extracts predictive uncertainty, investigating theoretical guarantees and recalibration techniques could further enhance reliability.

Overall, our work lays a strong foundation for efficiently adapting FMs, extending their applicability to broader forecasting challenges while preserving their expressive power.

#### Reproducibility Statement

In order to ensure reproducibility we will release the code at https://github.com/abenechehab/AdaPTS, once the paper has been accepted. The implementation details and hyperparameters are listed in Appendix C.2.

#### References

Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Arango, S. P., Kapoor, S., et al. Chronos: Learning the language of time series. arXiv preprint arXiv:2403.07815, 2024.

Benechehab, A., Hili, Y. A. E., Odonnat, A., Zekri, O., Thomas, A., Paolo, G., Filippone, M., Redko, I., and K´egl, B. Zero-shot model-based reinforcement learning using large language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=uZFXpPrwSh.

Bi, K., Xie, L., Zhang, H., Chen, X., Gu, X., and Tian, Q. Accurate medium-range global weather forecasting with 3d neural networks. Nature, 619(7970):533–538, 2023.

- Chen, S.-A., Li, C.-L., Arik, S. O., Yoder, N. C., and Pfister, T. TSMixer: An all-MLP architecture for time series forecast-ing. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https: //openreview.net/forum?id=wbpxTuXgm0.
- Chen, T., Fox, E., and Guestrin, C. Stochastic Gradient Hamiltonian Monte Carlo. In Xing, E. P. and Jebara, T. (eds.), Proceedings of the 31st International Conference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pp. 1683–1691, Bejing, China, 22–24 Jun 2014. PMLR. URL https://proceedings.mlr.press/v32/ cheni14.html.

Cowen-Rivers, A., Lyu, W., Tutunov, R., Wang, Z., Grosnit, A., Griffiths, R.-R., Maravel, A., Hao, J., Wang, J., Peters, J., and Bou Ammar, H. Hebo: Pushing the limits of sample-efficient hyperparameter optimisation. Journal of Artificial Intelligence Research, 74, 07 2022.

Das, A., Kong, W., Sen, R., and Zhou, Y. A decoder-only foundation model for time-series forecasting. In Fortyfirst International Conference on Machine Learning, 2024. URL https://openreview.net/forum? id=jn2iTJas6h.

Dinh, L., Sohl-Dickstein, J., and Bengio, S. Density estimation using real nvp, 2017. URL https://arxiv. org/abs/1605.08803.

Feofanov, V., Ilbert, R., Tiomoko, M., Palpanas, T., and Redko, I. User-friendly foundation model adapters for multivariate time series classification. arXiv preprint arXiv:2409.12264, 2024.

for Disease Control, C. and Prevention. In Fluview: Flu activity & surveillance, 2024. URL https://gis.cdc.gov/grasp/fluview/ fluportaldashboard.html.

Gal, Y. and Ghahramani, Z. Dropout As a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. In Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48, ICML’16, pp. 1050– 1059. JMLR.org, 2016. URL http://dl.acm.org/ citation.cfm?id=3045390.3045502.

Gelman, A., Carlin, J., Stern, H., Dunson, D., Vehtari, A., and Rubin, D. Bayesian Data Analysis. Chapman and Hall/CRC, United States, 3rd ed edition, 2013. ISBN 9781439840955.

Gneiting, T. and Katzfuss, M. Probabilistic forecasting. Annual Review of Statistics and Its Application, 1(1):125– 151, 2014.

Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., and Dubrawski, A. Moment: A family of open timeseries foundation models. In International Conference on Machine Learning, 2024.

Graves, A. Practical variational inference for neural networks. In Shawe-Taylor, J., Zemel, R., Bartlett, P., Pereira, F., and Weinberger, K. (eds.), Advances in Neural Information Processing Systems, volume 24. Curran Associates, Inc., 2011. URL https://proceedings.neurips.

cc/paper_files/paper/2011/file/ 7eb3c8be3d411e8ebfab08eba5f49632-Paper. pdf.

Groen, J. J., Paap, R., and Ravazzolo, F. Real-time inflation forecasting in a changing world. Journal of Business & Economic Statistics, 31(1):29–44, 2013.

Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., and Lerchner, A. beta-VAE: Learning basic visual concepts with a constrained variational framework. In International Conference on Learning Representations, 2017. URL https: //openreview.net/forum?id=Sy2fzU9gl.

Ilbert, R., Odonnat, A., Feofanov, V., Virmaux, A., Paolo, G., Palpanas, T., and Redko, I. Samformer: Unlocking the potential of transformers in time series forecasting with sharpness-aware minimization and channelwise attention. In Proceedings of the 41st International

Conference on Machine Learning, volume 235. PMLR, 2024. URL https://proceedings.mlr.press/ v235/ilbert24a.html.

Institute, M. P. In Weather dataset, 2021. URL https: //www.bgc-jena.mpg.de/wetter/.

Jin, M., Wang, S., Ma, L., Chu, Z., Zhang, J. Y., Shi, X., Chen, P.-Y., Liang, Y., Li, Y.-F., Pan, S., and Wen, Q. Time-LLM: Time series forecasting by reprogramming large language models. In International Conference on Learning Representations (ICLR), 2024.

Jones, H. E. and Spiegelhalter, D. J. Improved probabilistic prediction of healthcare performance indicators using bidirectional smoothing models. Journal of the Royal Statistical Society Series A: Statistics in Society, 175(3): 729–747, 2012.

Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., and Choo, J. Reversible instance normalization for accurate time-series forecasting against distribution shift. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=cGDAkQo1C0p.

Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization, 2017. URL https://arxiv.org/abs/ 1412.6980.

Kingma, D. P. and Welling, M. Auto-encoding variational bayes. CoRR, abs/1312.6114, 2013. URL https: //api.semanticscholar.org/CorpusID: 216078090.

Kobyzev, I., Prince, S. J., and Brubaker, M. A. Normalizing flows: An introduction and review of current methods. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(11):3964–3979, November 2021. ISSN 1939-3539. doi: 10.1109/tpami.2020. 2992934. URL http://dx.doi.org/10.1109/ TPAMI.2020.2992934.

Lai, G., Chang, W.-C., Yang, Y., and Liu, H. Modeling long- and short-term temporal patterns with deep neural networks, 2018. URL https://arxiv.org/abs/ 1703.07015.

Li, S. C.-X. and Marlin, B. M. A scalable endto-end gaussian process adapter for irregularly sampled time series classification. In Lee, D., Sugiyama, M., Luxburg, U., Guyon, I., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc., 2016. URL https://proceedings.neurips.

cc/paper_files/paper/2016/file/ 9c01802ddb981e6bcfbec0f0516b8e35-Paper. pdf.

Li, Y., Ye, J., Wen, X., Xu, G., Wang, J., and Liu, X. Padapter: Adapter combined with prompt for image and video classification. Image and Vision Computing, 154:105395, 2025. ISSN 0262-8856. doi: https://doi.org/10.1016/j.imavis.2024.105395. URL https://www.sciencedirect.com/ science/article/pii/S0262885624005006.

Liaw, R., Liang, E., Nishihara, R., Moritz, P., Gonzalez, J. E., and Stoica, I. Tune: A research platform for distributed model selection and training, 2018. URL https://arxiv.org/abs/1807.05118.

Liu, X., Liu, J., Woo, G., Aksu, T., Liang, Y., Zimmermann, R., Liu, C., Savarese, S., Xiong, C., and Sahoo, D. Moirai-moe: Empowering time series foundation models with sparse mixture of experts. arXiv preprint arXiv:2410.10469, 2024.

Ma, J., Thomas, V., Hosseinzadeh, R., Kamkari, H., Labach, A., Cresswell, J. C., Golestan, K., Yu, G., Volkovs, M., and Caterini, A. L. Tabdpt: Scaling tabular foundation models, 2024. URL https://arxiv.org/abs/ 2410.18164.

Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. A time series is worth 64 words: Long-term forecasting with transformers. In The Eleventh International Conference on Learning Representations, 2023. URL https:// openreview.net/forum?id=Jbdc0vTOcol.

Nowotarski, J. and Weron, R. Recent advances in electricity price forecasting: A review of probabilistic forecasting. Renewable and Sustainable Energy Reviews, 81:1548– 1568, 2018.

Palmer, T. Towards the probabilistic earth-system simulator: A vision for the future of climate and weather prediction. Quarterly Journal of the Royal Meteorological Society, 138(665):841–861, 2012.

Pan, J., Lin, Z., Zhu, X., Shao, J., and Li, H. St-adapter: Parameter-efficient image-to-video transfer learning. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 26462–26477. Curran Associates, Inc., 2022.

Papamarkou, T., Skoularidou, M., Palla, K., Aitchison, L., Arbel, J., Dunson, D., Filippone, M., Fortuin, V., Hennig, P., Hern´andez-Lobato, J. M., Hubin, A., Immer, A., Karaletsos, T., Khan, M. E., Kristiadi, A., Li, Y., Mandt, S., Nemeth, C., Osborne, M. A., Rudner, T. G. J., R¨ugamer, D., Teh, Y. W., Welling, M., Wilson, A. G., and Zhang, R. Position: Bayesian deep learning is needed in the age of large-scale ai. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

Rasul, K., Ashok, A., Williams, A. R., Ghonia, H., Bhagwatkar, R., Khorasani, A., Bayazi, M. J. D., Adamopoulos, G., Riachi, R., Hassen, N., Biloˇs, M., Garg, S., Schneider, A., Chapados, N., Drouin, A., Zantedeschi, V., Nevmyvaka, Y., and Rish, I. Lag-llama: Towards foundation models for probabilistic time series forecasting, 2024.

Sharma, M., Farquhar, S., Nalisnick, E., and Rainforth, T. Do Bayesian neural networks need to be fully stochastic? In Ruiz, F., Dy, J., and van de Meent, J.-W. (eds.), Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pp. 7694–7722. PMLR, 25–27 Apr

- 2023. URL https://proceedings.mlr.press/ v206/sharma23a.html.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(56):1929–1958, 2014. URL http://jmlr.org/papers/v15/ srivastava14a.html.

Zhang, W., Ye, J., Li, Z., Li, J., and Tsung, F. Dualtime: A dual-adapter multimodal language model for time series representation, 2024. URL https://arxiv.org/ abs/2406.06620.

Zhang, Y. and Yan, J. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In The eleventh international conference on learning representations, 2023.

Zhang, Y., Wang, J., and Wang, X. Review on probabilistic forecasting of wind power generation. Renewable and Sustainable Energy Reviews, 32:255–270, 2014.

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., and Zhang, W. Informer: Beyond efficient transformer for long sequence time-series forecasting. In The ThirtyFifth AAAI Conference on Artificial Intelligence, AAAI 2021, Virtual Conference, volume 35, pp. 11106–11115. AAAI Press, 2021.

Tian, Z., Peisong, N., Xue, W., Liang, S., and Rong, J. One Fits All: Power general time series analysis by pretrained lm. In NeurIPS, 2023.

Tran, B.-H., Rossi, S., Milios, D., and Filippone, M. All you need is a good functional prior for Bayesian deep learning. 23(1), 2022. ISSN 1532-4435.

Wei, W. W. Multivariate time series analysis and applications. John Wiley & Sons, 2019.

Wu, H., Xu, J., Wang, J., and Long, M. Autoformer: Decomposition transformers with auto-correlation for longterm series forecasting. In Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), Advances in Neural Information Processing Systems, 2021. URL https:

//openreview.net/forum?id=J4gRj6d5Qm.

Yang, A. X., Robeyns, M., Wang, X., and Aitchison, L. Bayesian low-rank adaptation for large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.

net/forum?id=FJiUyzOF1m.

Yin, D., Hu, L., Li, B., and Zhang, Y. Adapter is all you need for tuning visual tasks, 2023. URL https://arxiv. org/abs/2311.15010.

Zeng, A., Chen, M., Zhang, L., and Xu, Q. Are transformers effective for time series forecasting? 2023.

## Appendix

Outline. In Appendix A, we provide the proofs and a discussion on Proposition 3.4 and Proposition 4.1. We then provide a perspective on Normalizing Flows as adapters in Appendix B. The experimental setup is presented in Appendix C, including all the implementation details in Appendix C.2. Finally we showcase some additional results in Appendix D.

### Table of Contents

- A Theoretical analysis 13

- A.1 Proof of Proposition 3.4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- A.2 Proof of Proposition 4.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- B Normalizing Flows 15
- C Experimental setup 15

- C.1 Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- C.2 Implementation details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

- D Additional results 16

- D.1 Moment applied to synthetic data. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- D.2 Mean Absolute Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

#### A. Theoretical analysis

- A.1. Proof of Proposition 3.4 We begin by restating the proposition, and its underlying assumptions:

- Assumption A.1. Wφ has full rank: rank(Wφ) = D, insuring its invertibility.
- Assumption A.2. For ease of derivation, we consider a similar linear parametrization for the foundation model: fFM(X) = WFM⊤ X + bFM1⊤ where WFM ∈ RL×H, bFM ∈ RH, and 1 a vector of ones of dimension D. Proposition A.3 (Optimal linear adapter). Under Assumption 3.2 and Assumption 3.3, the closed-form solution of the problem:

L(Wφ) = ∥Y − WFM⊤ XWφ + bFM1⊤ Wφ−1∥2F (5) writes as:

Wφ∗ = (B⊤A)+B⊤B (6) where Wφ∗ = arg minW

φ∈GLD(R) L(Wφ), A = Y − WFM⊤ X, B = bFM1⊤, and (B⊤A)+ denoting the pseudo-inverse operator.

Proof. We begin by expanding the loss function:

L(Wφ) = ∥Y − (WFM⊤ XWφ + bFM1⊤)Wφ−1∥2F

= ∥A − BWφ−1∥2F

where A = Y − WFM⊤ X and B = bFM1⊤. Expanding the Frobenius norm:

L(Wφ) = Tr (A − BWφ−1)⊤(A − BWφ−1)

Taking the gradient with respect to Wφ−1 yields:

∂L ∂Wφ−1

= −2B⊤A + 2B⊤BWφ−1

∂Wφ = −Wφ−⊤∂W∂L−1

Knowing that Wφ is invertible, We have that: ∂L

Wφ−⊤ hence

φ

∂L ∂Wφ

= −2Wφ−T B⊤A − B⊤BWφ−1 Wφ−T.

Setting ∂L

∂Wφ = 0 and multiplying both sides by Wφ⊤, we obtain:

B⊤A = B⊤BWφ−1. Multiplying both sides by Wφ:

##### B⊤AWφ = B⊤B.

Finally applying the pseudo-inverse to solve for Wφ gives our final result:

###### Wφ∗ = (B⊤A)+B⊤B.

Given the convexity of L(Wφ) (which follows from the convexity of the Frobenius norm ∥·∥2F, the inverse operation, and an affine transformation), we conclude that Wφ∗ is a global solution for Eq. (3).

Remark A.4. We make use of the pseudo-inverse due to the current construction of the matrix B (with identical rows) which implies that the product B⊤A is degenerate. To bypass this limitation and further ensure the invertibility of Wφ∗, we can revisit the definition of the foundation model in Assumption 3.3 to include channel dependent biases and ensure a full rank matrix B.

| |
|---|

##### A.2. Proof of Proposition 4.1

To derive the evidence lower bound (ELBO) used in the training objective of the VAE adapter, we start from the marginal likelihood of the observed data Y given the inputs X and foundation model fFM. The marginal likelihood is expressed as:

###### log pθ(Y|X,fFM) = log pθ(Y|X,fFM(Z))p(Z)dZ, (7)

where Z is the latent variable, pθ(Y|X,fFM(Z)) is the likelihood model parameterized by θ, and p(Z) is the prior distribution over the latent variable Z.

Direct optimization of this marginal likelihood is generally intractable due to the integration over Z. To make this optimization feasible, we introduce a variational distribution qϕ(Z|X), parameterized by ϕ, as an approximation to the true posterior pθ(Z|X,Y,fFM). Using qϕ(Z|X), we can reformulate the log-marginal likelihood as follows:

pθ(Y|X,fFM(Z))p(Z) qϕ(Z|X)

dZ (8)

log pθ(Y|X,fFM) = log qϕ(Z|X)

pθ(Y|X,fFM(Z))p(Z) qϕ(Z|X)

. (9)

= log Eq

ϕ(Z|X)

Using Jensen’s inequality, we can derive a lower bound on this log-marginal likelihood:

pθ(Y|X,fFM(Z))p(Z) qϕ(Z|X)

log pθ(Y|X,fFM) ≥ Eq

ϕ(Z|X) log

qϕ(Z|X) p(Z)

= Eq

ϕ(Z|X) [log pθ(Y|X,fFM(Z))] − Eq

ϕ(Z|X) log

(10)

. (11)

The second term can be rewritten as the Kullback-Leibler (KL) divergence between the variational posterior qϕ(Z|X) and the prior p(Z):

qϕ(Z|X) p(Z)

. (12)

KL(qϕ(Z|X)∥p(Z)) = Eq

ϕ(Z|X) log

Substituting this into the inequality, we obtain the evidence lower bound (ELBO):

ϕ(Z|X) [log pθ(Y|X,fFM(Z))] − KL(qϕ(Z|X)∥p(Z)). (13)

log pθ(Y|X,fFM) ≥ Eq

The ELBO consists of two terms:

- • The forecasting term, Eq

ϕ(Z|X) [log pθ(Y|X,fFM(Z))], which measures how well the model can reconstruct Y given the latent variable Z.

- • The regularization term, KL(qϕ(Z|X)∥p(Z)), which encourages the variational posterior to stay close to the prior distribution p(Z).

Thus, maximizing the ELBO provides a tractable way to train the parameters θ and ϕ by optimizing the balance between forecasting accuracy and latent space regularization.

| |
|---|

#### B. Normalizing Flows

Normalizing Flows make use of invertible transformations to map a simple base distribution (e.g. Gaussian) to a complex data distribution. Each transformation T is designed to maintain invertibility and efficient Jacobian computation. The transformation is applied iteratively: Z = Tk ◦Tk−1◦···◦T1(X). Current Normalizing Flow instantiations (e.g. RealNVP) make use of generic invertible transformations such as coupling flows; the latters can be parametrized using a neural network leading to powerful non-linear generative models that are trained to maximize the data log-likelihood:

k

∂Ti(·;θ) ∂Zi−1

log p(X) = log p(Z) +

log det

i=1

where θ denote the parameters of the non-linear parametrization of the invertible transformations Ti, and Zi−1 is the output of the transformation Ti−1. In the context of time series adapters, we directly optimize the parameters of the transformations based on their direct and inverse application on the time series forecasting problem:

Lflow = ∥Y−T1−1 ◦ T2−1 ◦ ··· ◦ Tk−1( fFM Tk ◦ Tk−1 ◦ ··· ◦ T1(X;θ) ;θ)∥2F

where the encoder is represented by the series of direct transformations: enc(·) = Tk ◦Tk−1 ◦···◦T1(·;θ), and respectively the decoder by the series of inverse transformations dec(·) = T1−1 ◦ T2−1 ◦ ··· ◦ Tk−1(·;θ).

As defined here, Normalizing Flows suffer from the constraint of keeping the same dimension in both original and learned representation space. For this purpose, we investigate coupling a normalizing flow with a linear encoder-decoder type of architecture to enable dimensionality reduction prior to applying the transformation Ti. The parameters of the additional encoder and decoder are then jointly trained to optimize the learning objective Lflow.

Given that the parameters of the encoder and the decoder are shared in Normalizing Flows, the gradient-based optimization within our framework receives conflicting directions due to gradient flow from both the direct and inverse transformations simultaneously. We discovered that this adapter construction was challenging to optimize in practice, and we defer the exploration of this direction to future research endeavors.

#### C. Experimental setup

- C.1. Datasets Table 2: Characteristics of the multivariate time series datasets used in our experiments with various sizes and dimensions.

Dataset ETTh1 Illness ExchangeRate Weather # features 7 7 8 21 # time steps 13603 169 6791 51899 Granularity 1 hour 1 week 1 day 10 minutes (Train, Val, Test) (8033, 2785, 2785) (69, 2, 98) (4704, 665, 1422) (36280, 5175, 10444)

- C.2. Implementation details

In this section, we describe the full AdaPTS framework, starting from the data preprocessing, the training algorithm, and the hyperparameters optimization.

Preprocessing. Given that the adapter as defined in Definition 3.1 is a feature space transformation, we start by rescaling (StandardScaler and MinMaxScaler) the data where all the timesteps are regarded as data points. To account for the temporal specificities in each batch, we use Reversible Instance Normalization (RevIn) (Kim et al., 2022) that has been proven to mitigate time-related distribution shifts in time series problems. Finally, and following the observation that PCA when composed with a linear adapter showed the best result in the case of correlated data (Fig. 2), we include the possibility of applying full-component PCA as part of our data pre-processing pipeline.

1e 1

MeanSquaredError

7.25

7.00

6.75

6.50

I Wpca * * Wpca

Adapter Type

(a) Independent

1e 1

MeanSquaredError

5.8

5.6

5.4

I Wpca * * Wpca

Adapter Type

(b) Correlated

Figure 8: Moment on simulated independent data.

Training parameters. After the pre-processing phase, we proceed to split the data into a train-validation-test sets, where the validation set serves as a tool to select the best hyperparameters for the adapter. The resulting adapter that is instantiated with the optimal hyperparameters is then tested against the unseen test dataset. For all of our experiments, we first train the linear forecasting head of Moment (referred to as Linear Probing in (Goswami et al., 2024)) with the Adam optimizer (Kingma & Ba, 2017), a batch size of 32, a one cycle scheduler starting with 0.001 as learning rate. Once the forecasting linear head is trained, we freeze its parameters and proceed to training the adapter. This is done using the Adam optimizer, a batch size of 32, a reduce on plateau scheduler starting with 0.001 as learning rate.

Hyperparameter optimization. In order to select the best hyperparameters for the adapter architecture we use Ray tune (Liaw et al., 2018) with the Heteroscedastic and Evolutionary Bayesian Optimisation solver (HEBO) (Cowen-Rivers et al., 2022) engine, reporting the average mean squred error (MSE) from k-fold cross validation. Table 3 shows the default hyperparameters for each considered adapter.

Table 3: Adapters hyperparameters.

adapter LinearAE DropoutLinearAE LinearVAE VAE p dropout − 0.1 − − Number of layers − − − 2 Hidden dimension − − − 128

β − − 0.5 0.5 σ − − 1.0 1.0

#### D. Additional results

##### D.1. Moment applied to synthetic data.

To validate the adapter optimality condition with large non-linear foundation models, we use Moment (Goswami et al.,

- 2024). The optimal linear adapter in this case minimizes the following intractable objective:

L(Wφ) = ∥Y − fMoment XWφ Wφ−1∥2F (14)

To approximately solve this optimization problem, we instantiate Wφ as a single-linear-layer encoder denoted encθ, and respectively the inverse transformation Wφ−1 as a single-linear-layer decoder denoted decθ. We then use gradient-based optimization of the parameters θ using the Adam optimizer, aiming at solving the following optimization problem:

θ∗ = arg min

∥Y − decθ fMoment(encθ(X)) ∥2F (15)

θ

Fig. 8 shows the performance gain obtained by optimizing a linear adapter on Moment-small foundation model. Unlike the tractable case, we observe that in both data modalities (independent and correlated data), PCA has little to no improvement over the identity baseline, while φθ∗ reaches an order of magnitude better solution. This confirms our intuition about the existence of a better solution than the identity matrix, even in the case of real-world complex foundation models.

##### D.2. Mean Absolute Error

No adapter with adapter

Dataset H

Momentsmall pca linear dropout linear VAE VAE ETTh1

96 0.422±0.006 0.440±0.000 0.423±0.003 0.415±0.002 0.420±0.001 0.426±0.001 192 0.436±0.000 0.445±0.000 0.449±0.003 0.450±0.001 0.451±0.001 0.444±0.001

24 1.143±0.007 1.163±0.001 2.624±0.035 1.156±0.016 1.074±0.011 1.057±0.012 60 1.149±0.001 1.161±0.001 1.227±0.030 1.173±0.015 1.112±0.021 1.105±0.021

Illness

96 0.232±0.010 0.235±0.000 0.226±0.000 0.212±0.001 0.218±0.001 0.243±0.001 192 0.251±0.001 0.260±0.001 0.251±0.001 0.251±0.000 0.255±0.000 0.274±0.000

Weather

96 0.252±0.010 0.264±0.000 0.308±0.010 0.269±0.012 0.376±0.031 0.488±0.003 192 0.329±0.001 0.335±0.000 0.415±0.002 0.419±0.010 0.513±0.010 0.585±0.008

ExchangeRate

Table 4: Performance comparison between the baseline Moment model without adapters against different adapter architectures (PCA, LinearAE, dropoutLAE, LinearVAE, and VAE), for multivariate long-term forecasting with different horizons H. We display the average test MAE ± standard error obtained on 3 runs with different seeds. Best results are in bold, with lower values indicating better performance.

