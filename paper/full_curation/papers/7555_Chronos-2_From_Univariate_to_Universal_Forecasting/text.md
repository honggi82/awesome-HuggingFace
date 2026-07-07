# arXiv:2510.15821v1[cs.LG]17Oct2025

## Chronos-2: From Univariate to Universal Forecasting

#### Abdul Fatir Ansari1∗, Oleksandr Shchur1∗, Jaris Küken1,3∗†, Andreas Auer1,4†, Boran Han1, Pedro Mercado1, Syama Sundar Rangapuram1, Huibin Shen1, Lorenzo Stella1, Xiyuan Zhang1, Mononito Goswami1, Shubham Kapoor1, Danielle C. Maddix1, Pablo Guerron2,5†, Tony Hu1, Junming Yin1, Nick Erickson1, Prateek Mutalik Desai1, Hao Wang1,6†, Huzefa Rangwala1, George Karypis1, Yuyang Wang1‡, Michael Bohlke-Schneider1‡ ansarnd@amazon.de

1Amazon Web Services 2Amazon 3University of Freiburg 4Johannes Kepler University Linz 5Boston College 6Rutgers University Code: github.com/amazon-science/chronos-forecasting

### Abstract

Pretrained time series models have enabled inference-only forecasting systems that produce accurate predictions without task-specific training. However, existing approaches largely focus on univariate forecasting, limiting their applicability in real-world scenarios where multivariate data and covariates play a crucial role. We present Chronos-2, a pretrained model capable of handling univariate, multivariate, and covariate-informed forecasting tasks in a zero-shot manner. Chronos-2 employs a group attention mechanism that facilitates in-context learning (ICL) through efficient information sharing across multiple time series within a group, which may represent sets of related series, variates of a multivariate series, or targets and covariates in a forecasting task. These general capabilities are achieved through training on synthetic datasets that impose diverse multivariate structures on univariate series. Chronos-2 delivers state-of-the-art performance across three comprehensive benchmarks: fev-bench, GIFT-Eval, and Chronos Benchmark II. On fev-bench, which emphasizes multivariate and covariate-informed forecasting, Chronos-2’s universal ICL capabilities lead to substantial improvements over existing models. On tasks involving covariates, it consistently outperforms baselines by a wide margin. Case studies in the energy and retail domains further highlight its practical advantages. The in-context learning capabilities of Chronos-2 establish it as a general-purpose forecasting model that can be used “as is” in real-world forecasting pipelines.

### 1 Introduction

The advent of pretrained models (also referred to as foundation models) has led to a paradigm shift in time series forecasting. Instead of training a model for each time series (local models) (Hyndman & Athanasopoulos, 2018) or dataset (task-specific models) (Lim et al., 2021; Challu et al., 2023), a single model can be trained once on large-scale time series data and then applied across different forecasting problems (Ansari et al., 2024; Das et al., 2024b). Pretrained models greatly simplify the forecasting pipeline by eliminating the need for training from scratch for each use case. More remarkably, they often match or exceed the forecast accuracy of task-specific models (Aksu et al., 2024).

Despite these advances, a fundamental limitation persists: most pretrained models operate only on univariate data, considering solely the historical observations of a single time series to generate forecasts. Although univariate forecasting is important, the class of real-world forecasting tasks spans far beyond it. In practice, one may encounter tasks where multiple co-evolving time series need to be predicted simultaneously (multivariate forecasting) (Bańbura et al., 2010; Cohen et al., 2025) or where forecasts depend on various external factors (covariate-informed forecasting). For example, cloud infrastructure metrics such as CPU usage, memory consumption, and storage I/O evolve together and benefit from joint modeling (Cohen et al., 2025). Likewise, retail demand is heavily influenced by promotional

∗Equal contribution. †Jaris Küken and Andreas Auer contributed to this work during their internships at AWS. Hao Wang and Pablo Guerron hold concurrent

appointments at Amazon and their corresponding universities, and this report describes work performed at Amazon.

‡Equal advisory contribution.

- TARGET(S) #1

- KNOWN COVARIATE #1

TOKENIZATION TRANSFORMER STACK

TAIMETTENTION

GAROUPTTENTION

FFEEDORWARD

RSOBUSTCALING

MFETAEATURES

PATCHING

PEATCHMBEDDING

QHUANTILEEAD

UNSCALING

attention along time axis

TIME ATTENTION

attentionwithingroups

GROUP ATTENTION

INPUT

Time Series

Time Index

Mask

META FEATURES PATCHING

TimeSeries[ ]

Time Index

Mask

[ ] [ ]

PATCH EMBEDDING

x N

FORECAST

TOKENIZATION

TARGET(S) #2

- KNOWN COVARIATE #2

HISTORY FUTURE

- Figure 1: The complete Chronos-2 pipeline. Input time series (targets and covariates) are first normalized using a robust scaling scheme, after which time index and mask meta features are added. The resulting sequences are split into non-overlapping patches and mapped to high-dimensional embeddings via a residual network. The core transformer stack operates on these patch embeddings and produces multi-patch quantile outputs corresponding to the masked future patches provided as input. Each transformer block alternates between time and group attention layers: the time attention layer aggregates information across patches within a single time series, while the group attention layer aggregates information across all series within a group at each patch index. A group is a flexible notion of relatedness and may correspond to a single time series, multiple series sharing a source or metadata, variates of a multivariate series, or targets along with associated covariates. The figure illustrates two multivariate time series with one known covariate each, with corresponding groups highlighted in blue and red. This example is for illustration only; Chronos-2 supports arbitrary numbers of targets and optional covariates.

activities, while energy consumption patterns are driven by weather conditions (Petropoulos et al., 2022). The lack of multivariate and covariate-informed forecasting capabilities hinders the widespread adoption of pretrained models in real-world production systems.

Developing universal pretrained models that can handle both multivariate dependencies and covariates remains challenging due to two factors. First, the heterogeneity of forecasting problems requires rethinking the model architecture. Each downstream task differs in the number of dimensions and their semantics. Since it is impossible to know a priori how the variables will interact in an unseen task, the model must infer these interactions from the available context. Second, high-quality pretraining data with multivariate dependencies and informative covariates is scarce.

In this work, we present Chronos-2, a pretrained model designed to handle arbitrary forecasting tasks — univariate, multivariate, and covariate-informed — in a zero-shot manner. Chronos-2 leverages in-context learning (ICL) to support multivariate forecasting and arbitrary covariates, whether past-only or with known future values, real-valued or categorical. Its enhanced ICL capabilities also improve univariate forecasting by enabling cross learning, where the model shares information across univariate time series in the batch, leading to more accurate predictions.

At the core of Chronos-2 ’s ICL capabilities is the group attention mechanism. It enables information exchange within groups of time series, which may represent arbitrary sets of related series, variates of a multivariate series, or targets and covariates (both past-only and known) in a forecasting task. Rather than extending the context by concatenating targets and covariates, the group attention layer shares information within groups across the batch axis, allowing it to scale gracefully with the number of variates. A key innovation of Chronos-2 lies in our training approach: to enable its ICL capabilities, we rely on synthetic time series data generated by imposing multivariate structure on time series sampled from base univariate generators. The complete inference pipeline of Chronos-2 including tokenization and modeling is shown in Figure 1.

Empirical evaluation on comprehensive forecasting benchmarks, including fev-bench (Shchur et al., 2025), GIFT-Eval (Aksu et al., 2024), and Chronos Benchmark II (Ansari et al., 2024), shows that Chronos-2 achieves state-of-the-art performance. On fev-bench, which spans a wide range of forecasting tasks — univariate, multivariate,

Univariate Forecasting

Multivariate Forecasting

Past-Only Covariates

Known Covariates

Categorical Covariates

Cross Learning

Memory Scaling

Model

Chronos-2 ✓ ✓ ✓ ✓ ✓ ✓ O(V ) Toto-1.0 ✓ ✓ ✓ ✗ ✗ ✗ O(V ) TabPFN-TS ✓ ✗ ✗ ✓ ✓ ✗ O(V ) COSMIC ✓ ✗ ✓ ✓ ✗ ✗ O(V 2)

- Moirai-1.0 ✓ ✓ ✓ ✓ ✗ ✗ O(V 2) Chronos-Bolt ✓ ✗ ✗ ✗ ✗ ✗ -

- Moirai-2.0 ✓ ✗ ✗ ✗ ✗ ✗ Sundial ✓ ✗ ✗ ✗ ✗ ✗ TimesFM-2.5 ✓ ✗ ✗ ✗ ✗ ✗ TiRex ✓ ✗ ✗ ✗ ✗ ✗ -

- Table 1: Comparison of capabilities of pretrained forecasting models. Past-Only Covariates: support for covariates only observed in the past; Known Covariates: support for covariates whose future values are known; Categorical Covariates: support for nominal features in the covariates; Cross Learning: support for in-context learning across related time series; Memory Scaling: inference memory requirements with respect to the total number of variates V (including both targets and covariates).

and covariate-informed — Chronos-2 outperforms baselines across all categories. The largest gains are observed on covariate-informed tasks, demonstrating Chronos-2’s strength in this practically important setting. Chronos-2 offers these new capabilities while maintaining high computational efficiency, running on a single mid-range GPU (NVIDIA A10G) with a throughput of 300 time series per second.1

The rest of the technical report is organized as follows. Section 2 introduces the background on time series forecasting and existing forecasting methods with a special focus on pretrained models. In Section 3, we describe the architecture of Chronos-2 and discuss its training and inference pipelines. Section 4 briefly discusses the training corpus of Chronos-2. In Section 5, we present our main results on three forecasting benchmarks, case studies on energy and retail domains, and ablations. We conclude the report and discuss potential future work in Section 6.

### 2 Background and Related Work

Time series forecasting aims to predict future values of a temporal sequence given historical observations. Formally, let Y1:T = [y1,...,yT] denote a historical time series of length T, where each observation yt ∈ RD can either be univariate (D = 1) or multivariate (D > 1). Given this historical context, the goal is to predict the next H time steps YT+1:T+H, where H defines the forecast horizon. Forecasts may be supported by covariates (also known as exogenous variables) X1:T+H = [x1,...,xT+H], where xt ∈ RM represents additional information that can span both historical (t ≤ T) and future (t > T) time steps. The task itself can be defined as either point forecasting, where the objective is to predict a single future value at each time step, or probabilistic forecasting, where the objective is to estimate the conditional distribution P(YT+1:T+H | Y1:T,X1:T+H) in order to capture forecast uncertainty. Zero-shot forecasting refers to the setting in which a model generates forecasts for a previously unseen time series datasets without requiring any additional training, adaptation, or fine-tuning.

Forecasting methods preceding the pretrained model paradigm can be broadly divided into local and global models. Local models fit one set of parameters for each time series in the dataset. These include classical approaches such as ARIMA, Exponential Smoothing (Hyndman & Athanasopoulos, 2018), and Theta (Assimakopoulos & Nikolopoulos, 2000). In contrast, global models share their parameters across all time series within a specific dataset. Deep learning approaches in this category have become increasingly common over the last decade. Notable examples of global models include recurrent neural networks (RNN) like DeepState (Rangapuram et al., 2018), DeepAR (Salinas et al., 2020), and TimeGrad (Rasul et al., 2021); stacked architectures such as N-BEATS (Oreshkin et al., 2020) and N-HITS (Challu et al., 2023); and transformer-based architectures like TFT (Lim et al., 2021) and PatchTST (Nie et al., 2023).

Pretrained forecasting models have recently emerged as a new paradigm in time series forecasting. While earlier work already demonstrated limited transfer learning capabilities for forecasting (Orozco & Roberts, 2020; Oreshkin et al.,

1Based on inference time for a batch of 1,024 time series with a context length of 2048 and prediction length of 64 times teps.

2021; Jin et al., 2022; Nie et al., 2023), pretrained models adopt principles similar to large language models (LLMs) and enable zero-shot generalization on diverse datasets. Initial attempts focused on directly adapting language models to time series tasks (Gruver et al., 2023; Jin et al., 2024), whereas more recent approaches primarily borrow architectural concepts from LLMs but pretrain them on time series data (Das et al., 2024b; Garza et al., 2024; Ansari et al., 2024).

The majority of pretrained models are limited to univariate forecasting (Rasul et al., 2023; Das et al., 2024b; Ansari et al., 2024; Liu et al., 2025; Auer et al., 2025b), treating each dimension independently in multivariate scenarios and ignoring covariates. Notable exceptions include Moirai-1 (Woo et al., 2024) and Toto (Cohen et al., 2025), which incorporate multivariate structure into their architectures. Moirai-1 supports multivariate inputs but flattens them internally, which limits scalability to high-dimensional cases. Toto introduces a cross-variate attention mechanism but does not support known or categorical covariates. COSMIC (Auer et al., 2025a) advances covariate utilization through synthetic augmentations but remains restricted to univariate targets. TabPFN-TS (Hoo et al., 2025), a tabular foundation model adapted for time series, can incorporate known covariates but it does not model past-only covariates or multivariate targets. Despite these advances, empirical analyses show that most approaches provide only marginal benefits over univariate models (Żukowska et al., 2024; Auer et al., 2025a), indicating that jointly modeling multiple variates and integrating covariates effectively in a zero-shot setting remains an open challenge.

Our approach addresses this gap with a group attention mechanism, which generalizes ideas from cross-attention architectures for multivariate forecasting (Zhang & Yan, 2023; Rao et al., 2021; Arnab et al., 2021) and cross-learning across multiple univariate series (Das et al., 2024a). Unlike prior approaches, group attention operates over groups of related time series and naturally accommodates diverse forecasting setups, including univariate, multivariate, and covariate-informed tasks, within a unified framework without requiring architectural changes or task-specific adaptations. Table 1 compares the capabilities of Chronos-2 with those of existing pretrained models.

### 3 The Chronos-2 Model

In this section, we introduce the Chronos-2 model. We begin with scaling and tokenization, followed by the model’s architecture including the group attention mechanism which enables Chronos-2’s in-context learning capabilities. Subsequently, we discuss the training and inference pipelines of Chronos-2. The complete inference pipeline of Chronos-2 is visualized in Figure 1.

##### 3.1 Scaling and Tokenization

Input Construction. The model operates on two inputs derived from the target Y1:T and covariates X1:T+H. We concatenate all historical values into V = [v1,...,vT], where each vt ∈ RD+M consists of the target observation yt and the corresponding covariate vector xt. Similarly, we define the future values as W = [wT+1,...,wT+H], where wt ∈ RD+M contains known future covariate values xt when available, while the entries corresponding to targets and past-only covariates are set to missing values.

Categorical covariates in X1:T+H are transformed into real-valued representations before being concatenated into V and W. For univariate targets, we apply target encoding (Pedregosa et al., 2011; Micci-Barreca, 2001), which maps each category to a numerical value based on its relationship with the target. For multivariate targets, the model falls back to ordinal encoding, assigning a unique integer to each category.

Robust Scaling. The input values, V and W, may be at an arbitrary scale, so our tokenization pipeline begins by normalizing the series. We adopt standardization, a widely used normalization method in the literature, and introduce an additional step: applying the sinh−1 transformation to the standardized values. This log-like transformation further stabilizes variance and reduces the influence of outliers on the objective function. It has been used in econometrics (Burbidge et al., 1988) and energy price forecasting (Uniejewski & Weron, 2018) literature for handling extreme values. Formally, each historical value vt,d and the future value wt,d are normalized as

- v˜t,d = sinh−1

vt,d − µd σd

for t ∈ {1,...,T}, (1)

- w˜t,d = sinh−1

wt,d − µd σd

for t ∈ {T + 1,...,T + H}, (2)

where µd and σd are the mean and standard deviation of the historical values [v1,d,...,vT,d], respectively. Any missing values in V are excluded when computing µd and σd. The normalized historical values V˜ and future values W˜ are concatenated to construct the input matrix U = [V˜ ,W˜ ] ∈ R(T+H)×(D+M).

Meta Features. During tokenization, each dimension of U is processed independently by the model. To describe the tokenization procedure, consider a single column ud = [u1,d,...,uT+H,d]⊤ corresponding to one target or covariate dimension d. Two additional meta features are appended to each column: a time index and a mask. The time index j = −CT ,−TC−1,...,0,..., HC−1 encodes the relative position of each time step, where C is the maximum context length supported by the model. It provides explicit information about temporal ordering to the model which is beneficial when using patch-based inputs. The mask md is a binary indicator equal to 1 when the value is observed, and 0 otherwise. It serves two purposes: indicating which values are missing in the historical context and specifying which input dimensions correspond to future-known covariates. After construction of the mask, all missing values in ud are replaced with zeros.

Patching and Embedding. The input ud with the corresponding meta features, j and md, are split into nonoverlapping patches of length P (Nie et al., 2023). The context and future sections of the time series and meta features are split into patches separately. When T and H are not multiples of P, zero padding is applied on the left (context) or right (future). Let up, jp, and mp denote the p-th patches of the input, time index, and mask, respectively. These are concatenated and mapped into the embedding space using a residual network, fϕin : R3P → RD

model,

hp = fϕin up,jp,mp , (3)

where ϕ denotes parameters of the residual network and Dmodel is the hidden dimension of the transformer model. Between the patch embeddings of the context and future, we include a special REG token which serves both as a separator token and an attention sink (Xiao et al., 2024).

##### 3.2 Architecture

Chronos-2 is an encoder-only transformer (Vaswani et al., 2017) model which closely follows the design of the T5 encoder (Raffel et al., 2020). In the following, we discuss the key architectural components of Chronos-2.

Time Attention. The time attention layer is the usual attention layer found in typical sequence models. It applies self-attention along the temporal axis and aggregates information across patches of the same input dimension. We replace relative position embeddings used in the self-attention layers of the original T5 model with rotary position embeddings (RoPE) (Su et al., 2024) which have become the de-facto standard for position embeddings in modern transformer-based models (Touvron et al., 2023).

Group Attention. We introduce a group attention layer into the transformer stack, which is central to enabling the in-context learning capabilities of Chronos-2. This layer aggregates information across time series that belong to the same group at a given patch index. A group refers to a set of related time series and may refer to different things depending on the forecasting task. For example, a group may consist of:

- • a single time series: the minimal grouping where the model makes univariate predictions without referring to other time series in the batch.
- • a set of time series with shared source or metadata: this grouping enables the model to perform cross learning across items by making joint predictions for related time series (also referred to as few-shot learning) instead of generating univariate forecasts by solely taking the histories of individual time series into account. Sharing information between related time series could be especially helpful when all or some (cold start scenario) time series have short histories and when the characteristics of the downstream dataset differ considerably from the training data distribution.
- • a set of variates with shared dynamics: this grouping enables multivariate forecasting where the model jointly predicts all variates with shared dynamics.
- • a set of target(s), past-only covariates and known covariates: the most general case where the model forecasts targets while taking covariates into account.

Within a batch of size B, multiple groups of varying sizes are possible, each identified by group IDs g, a vector of length B. Internally, the group attention layer maps these IDs to a two-dimensional attention mask, ensuring that aggregation occurs only within groups and not across them. Since time series within a group lack a natural ordering, the group attention layer omits positional embeddings.

Quantile Head. After a sequence of alternating time and group attention layers, the embeddings of future patches of the D target dimensions are passed through a residual block to produce the direct multi-step quantile forecast Zˆ ∈ RH×D×|Q|. By producing forecasts for multiple target patches within a single forward pass, the model can efficiently generate predictions over long forecast horizons. Chronos-2 predicts a set of 21 quantiles Q = {0.01,0.05,0.1,...,0.9,0.95,0.99}. This results in a richer representation of the predictive distribution compared to the 9-quantile grid {0.1,0.2,...,0.9} commonly used in existing pretrained models. The inclusion of extreme quantiles (0.01 and 0.99) improves coverage of rare events and enhances the model’s applicability to tasks such as anomaly detection and risk-aware forecasting.

##### 3.3 Training

During training, batches are constructed to include heterogeneous forecasting tasks: univariate forecasting, multivariate forecasting (which also covers tasks with past-only covariates), and multivariate forecasting with known covariates. Each task is characterized by the number of target dimensions D, the number of covariates M, and the role of each dimension (target, past-only covariate, or known covariate). A unique group ID is assigned to each task, and the combination of group IDs g with whether the future input W is observed allows the model to infer the specific forecasting setup.

The model is trained using the quantile regression objective

q · max(z − zˆq,0) + (1 − q) · max(ˆzq − z,0) , (4)

q∈Q

where zˆq is the forecast at quantile level q, and z is the corresponding target value normalized as in Eq. (1). The loss is averaged over all forecast steps and items in the batch and is computed only on target dimensions, with entries corresponding to known covariates or missing target values excluded from the objective. The number of output patches is randomly sampled for each batch during training.

Training proceeds in two stages. First, the model is pretrained with a maximum context length of 2048 and a low number of maximum output patches. In the second stage, the context length is extended to 8192, and the maximum number of sampled output patches is increased. Longer contexts enable the model to capture long-term seasonalities in high-frequency time series, while multi-patch outputs allow for long-horizon forecasts without relying on heuristics.

##### 3.4 Inference

Forecasts are generated by de-normalizing the model predictions zˆt,dq and inverting Eq. (1). Formally, the quantile head output zˆt,dq is transformed as

yˆt,dq = µd + σd · sinh(ˆzt,dq ), (5)

to obtain the prediction yˆt,dq of the quantile level q at time step t along the target dimension d. During inference, multiple time series in a batch can be grouped to solve different forecasting tasks:

- • univariate forecasting: each item in the batch is assigned a unique group ID. This ensures that the model makes independent predictions for each time series in the batch.
- • multivariate forecasting: each variate which belongs to the same multivariate series is assigned the same group ID with variates from different multivariate series having distinct group IDs. This allows the model to share dynamics information between different variates of a multivariate time series.
- • forecasting with covariates: all target(s), past-only and known covariates belonging to the same task are assigned the same group ID. The future inputs W corresponding to known covariates contain their known future values. The predictions generated by the model for covariates are ignored.

#### Task Type Group IDs g Future Inputs W

 

  ∈ R3×H

∗ ... ∗ ∗ ... ∗ ∗ ... ∗

Univariate Forecasting (3 independent series)

g = (1,2,3) W =

 

  ∈ R3×H

∗ ... ∗ ∗ ... ∗ ∗ ... ∗

Multivariate Forecasting (3 targets)

g = (1,1,1) W =

  

   ∈ R4×H

∗ ... ∗ ∗ ... ∗

Forecasting with Covariates (1 target, 1 past-only covariate, 2 known covariates)

g = (1,1,1,1) W =

- xT+1,3 ... xT+H,3
- xT+1,4 ... xT+H,4

- Table 2: Diverse forecasting tasks can be solved by specifying group IDs and future inputs appropriately. Here, g and W denote the group IDs and future values provided to the model. Future inputs for targets and past-only covariates are masked as missing values, denoted as ∗. The examples use fixed numbers of variates for clarity, but Chronos-2 can handle arbitrary dimensions.

- Table 2 summarizes how group IDs and future inputs must be specified to solve different forecasting tasks. In addition to these, Chronos-2 can also be used in the full cross learning mode where each item in the batch is assigned the same group ID regardless of whether the item is a target, a past-only covariate or a known covariate. Since each item belongs to the same group, the model shares information across items in the batch and makes joint predictions for the entire batch.

### 4 Training Data

For a generalist pretrained model such as Chronos-2, the training data often plays a more decisive role than the model’s specific architecture. Although recent efforts have expanded the availability of large-scale time series datasets (Woo et al., 2024; Ansari et al., 2024; Aksu et al., 2024), they primarily contain univariate data. To overcome this limitation and endow Chronos-2 with in-context learning capabilities, we relied extensively on synthetic data.

##### 4.1 Univariate Data

We incorporated select datasets from the Chronos (Ansari et al., 2024) and GIFT-Eval (Aksu et al., 2024) pretraining corpora into Chronos-2’s training corpus. The full list of datasets is provided in Table 6 (Appendix). To further enhance data diversity, we generated synthetic data using two approaches:

- • TSI (Trend, Seasonality, and Irregularity): based on Bahrpeyma et al. (2021), this generator produces diverse synthetic series by randomly constructing and combining different trend, seasonality, and irregularity components.
- • TCM (Temporal Causal Model): this generator samples random causal graphs from a temporal causal model (Runge et al., 2023), from which time series are generated via autoregression.

##### 4.2 Multivariate Data

For multivariate and covariate-informed tasks, we relied entirely on synthetic data. To enable a broad class of multivariate structures, we introduce the concept of multivariatizers. A multivariatizer samples multiple time series from base univariate generators and imposes dependencies among them to create multivariate dynamics. As base univariate generators, we employed a diverse set including autoregressive (AR) models, exponential smoothing (ETS) models, TSI, and KernelSynth (Ansari et al., 2024).

We used two broad classes of multivariatizers:

- • Cotemporaneous multivariatizers apply linear or nonlinear transformations at the same time step across time series sampled from the base univariate generators. This introduces instantaneous correlations between the time series resulting in a multivariate time series.
- • Sequential multivariatizers induce dependencies across time, generating richer multivariate properties such as lead–lag effects and cointegration.

The multivariate time series generated from the multivariatizers were used to construct both multivariate tasks (where all variates must be predicted) and covariate-informed tasks, where a subset of variates was randomly designated as known covariates.

### 5 Experiments

In this section, we present empirical results, beginning with an evaluation of Chronos-2 against state-of-the-art approaches across three comprehensive benchmarks (Section 5.1). We then demonstrate the gains achieved through in-context learning on univariate, multivariate, and covariate-informed forecasting tasks (Section 5.2). Next, we examine Chronos-2’s performance on tasks from the energy and retail domains, where covariates are often important for accurate forecasting (Section 5.3). Finally, we report results for ablated variants of Chronos-2 (Section 5.4), including a smaller model, a version trained only on synthetic data, and the model prior to long-context post-training.

- 5.1 Benchmark Results

Model Avg. Win Rate (%) Skill Score (%) Median runtime (s) Leakage (%) #Failures Chronos-2 90.7 47.3 3.6 0 0 TiRex 80.8 42.6 1.4 1 0 TimesFM-2.5 75.9 42.3 16.9 8 0 Toto-1.0 66.6 40.7 90.7 8 0 COSMIC 65.6 39.0 34.4 0 0 Moirai-2.0 61.1 39.3 2.5 28 0 Chronos-Bolt 60.3 38.9 1.0 0 0 TabPFN-TS 59.3 39.6 305.5 0 2 Sundial 41.0 33.4 35.6 1 0 Stat. Ensemble 40.4 20.2 690.6 0 11 AutoARIMA 35.2 20.6 186.8 0 10 AutoETS 29.1 -26.8 17.0 0 3 AutoTheta 21.8 5.5 9.3 0 0 SeasonalNaive 14.5 0.0 2.3 0 0 Naive 7.8 -45.4 2.2 0 0

- Table 3: fev-bench results. The average win rate and skill score are computed with respect to the scaled quantile loss (SQL) metric. Higher values are better for both. Chronos-2 outperforms all existing pretrained models by a substantial margin on this benchmark that includes univariate, multivariate, and covariate-informed forecasting tasks. Baseline results and the imputation strategy for handling data leakage in certain tasks are both taken from Shchur et al. (2025). Results for additional forecasting metrics are provided in Tables 7 to 9 (Appendix).

We evaluated the base Chronos-2 model with 120M parameters on three comprehensive forecasting benchmarks: fev-bench (Shchur et al., 2025), GIFT-Eval (Aksu et al., 2024), and Chronos Benchmark II (Ansari et al., 2024). To contextualize its performance, we compared it against state-of-the-art time series foundation models that achieved the strongest results on these benchmarks. These include TiRex (Auer et al., 2025b), TimesFM-2.5 (Das et al., 2024b), Toto-1.0 (Cohen et al., 2025), Moirai-2.0 (Woo et al., 2024), TabPFN-TS (Hoo et al., 2025), COSMIC (Auer et al., 2025a), Sundial (Liu et al., 2025), and Chronos-Bolt (Ansari et al., 2024), the latest publicly released version of Chronos. As additional baselines, we also included AutoARIMA, AutoETS, AutoTheta, and their ensemble (Petropoulos & Svetunkov, 2020), representing well-established methods from the statistical forecasting literature (Hyndman & Athanasopoulos, 2018). We compare Chronos-2 only with the aforementioned models and exclude task-specific deep learning models from our evaluation, as prior studies (Aksu et al., 2024; Ansari et al., 2024) — which include

###### GIFT-Eval and Chronos Benchmark II, two of the three benchmarks considered in our work — have shown that pretrained models perform comparably to or better than task-specific models on average.

###### Pairwise Win Rate (SQL) with 95% CIs

###### Model 2

Stat.Ensemble

SeasonalNaive

TimesFM-2.5

Chronos-Bolt

AutoARIMA

TabPFN-TS

AutoTheta

Chronos-2

Moirai-2.0

AutoETS

COSMIC

Toto-1.0

Sundial

Naive

TiRex

Win Rate (%)

100

50 72 74 78 91 93 94 88 96 95 96 94 99 100 100

Chronos-2

(50.0, 50.0) (64.0, 81.0) (65.0, 82.0) (70.0, 86.0) (85.0, 96.0) (88.0, 98.0) (89.0, 98.0) (81.0, 94.0) (92.0, 99.0) (90.0, 99.0) (92.0, 99.0) (89.0, 98.0) (97.0, 100) (100, 100) (100, 100)

80

28 50 54.5 68.5 74 82.5 83.5 72 90.5 92 95 92 99 100 99

TiRex

###### Model1

(19.0, 36.0) (50.0, 50.0) (44.0, 64.0) (59.0, 77.5) (65.0, 82.0) (75.0, 89.5) (76.0, 90.5) (63.0, 80.0) (84.5, 95.5) (86.0, 97.0) (90.0, 99.0) (87.0, 96.0) (97.0, 100) (100, 100) (97.0, 100)

60

26 45.5 50 57 66 75 73 70 92.5 84 92 87 96 99 99

TimesFM-2.5

40

(18.0, 35.0) (36.0, 56.0) (50.0, 50.0) (48.0, 66.0) (57.0, 75.0) (67.5, 83.0) (65.0, 81.0) (60.0, 79.0) (87.0, 97.0) (76.0, 91.0) (86.0, 96.0) (80.0, 93.0) (91.0, 99.0) (97.0, 100) (97.0, 100)

20

22 31.5 43 50 47 60 57 57 82.5 80 86 83 90 96 98

Toto-1.0

(14.0, 30.0) (22.5, 41.0) (34.0, 52.0) (50.0, 50.0) (37.0, 58.0) (51.5, 69.0) (48.0, 66.0) (48.0, 66.0) (74.0, 89.5) (72.0, 87.0) (79.0, 92.0) (75.0, 90.0) (84.0, 95.0) (92.0, 99.0) (95.0, 100)

0

###### (a)

###### Pairwise Skill Score (SQL) with 95% CIs

###### Model 2

Stat.Ensemble

SeasonalNaive

TimesFM-2.5

Chronos-Bolt

AutoARIMA

TabPFN-TS

AutoTheta

Chronos-2

Moirai-2.0

AutoETS

COSMIC

Toto-1.0

Sundial

Naive

TiRex

Skill Score (%)

0 8.2 8.6 11 13.6 13.1 13.7 12.7 20.8 34 33.6 56.2 44.2 47.3 63.7

Chronos-2

(0.0, 0.0) (5.6, 11.2) (5.7, 11.6) (8.2, 14.2) (10.3, 17.1) (10.3, 16.3) (11.0, 16.7) (9.9, 15.8) (17.8, 23.6) (28.7, 39.0) (28.9, 37.9) (45.8, 65.6) (39.1, 49.9) (42.9, 52.2) (58.6, 68.3)

20

- −8.9 0 0.5 3.1 5.9 5.3 6 4.9 13.7 28.1 27.7 52.4 39.3 42.6 60.5
- −9.4 −0.5 0 2.6 5.5 4.9 5.6 4.5 13.3 27.7 27.4 52.2 39 42.3 60.3

TiRex

###### Model1

- (-12.6, -5.9) (0.0, 0.0) (-1.2, 2.0) (1.0, 5.3) (3.6, 8.6) (3.7, 7.1) (4.2, 8.0) (0.6, 9.0) (10.4, 16.7) (22.9, 32.9) (23.4, 31.8) (40.7, 62.6) (33.6, 44.8) (38.1, 47.5) (55.1, 65.4)
- (-13.1, -6.1) (-2.1, 1.2) (0.0, 0.0) (0.4, 5.1) (2.8, 8.6) (2.9, 6.9) (3.5, 7.8) (0.2, 9.0) (10.0, 16.4) (22.3, 32.9) (22.9, 31.8) (40.2, 62.4) (33.4, 45.2) (37.9, 47.4) (54.8, 65.2)

0

TimesFM-2.5

−20

−12.4 −3.2 −2.7 0 2.9 2.3 3 1.9 11 25.8 25.4 51 37.3 40.7 59.2

Toto-1.0

(-16.6, -8.9) (-5.6, -1.0) (-5.4, -0.4) (0.0, 0.0) (-0.6, 6.4) (0.3, 4.4) (0.2, 5.9) (-3.3, 6.7) (6.9, 14.6) (20.0, 31.2) (20.3, 30.1) (39.0, 61.7) (31.3, 43.5) (35.8, 46.3) (53.5, 64.6)

(b)

- Figure 2: The pairwise win rates (a) and skill scores (b) of the top-4 pretrained models on fev-bench with 95% confidence intervals (CIs) obtained through bootstrapping. Chronos-2 outperforms the next best models (TiRex and TimesFM) by a statistically significant margin on both metrics. The complete plot and results for other forecasting metrics can be found in Figures 12 to 19 (Appendix).

Following Shchur et al. (2025), we report both average win rates (W) and skill scores (S) for all models. These metrics are mathematically equivalent to the average rank (R) and geometric mean relative error ( G) metrics used in prior work (Ansari et al., 2024; Aksu et al., 2024). Specifically, R = 1 + (1 − 100W )(N − 1) and G = 1 − 100S , where N is the number of evaluated models. However, win rates and skill scores provide more interpretable summaries. The win rate measures the proportion of pairwise comparisons in which a model outperforms other models, while the skill score reflects the average percentage improvement over a baseline — in our case, the Seasonal Naive model. For a detailed discussion, we refer the reader to Shchur et al. (2025).

fev-bench. This benchmark consists of 100 forecasting tasks and offers the most comprehensive coverage of diverse real-world scenarios, including tasks with covariates. None of these datasets or tasks were seen by Chronos-2 during training. Table 3 reports results on fev-bench with respect to the scaled quantile loss (SQL) metric which evaluates the probabilistic forecasting performance. Chronos-2 outperforms existing time series foundation models by a significant margin, both in win rate and skill score. fev-bench also provides tooling to answer questions like: “Does Model A outperform Model B in a statistically significant way?”. These pairwise comparisons with 95% confidence intervals (CIs), shown in Figure 2, further confirm that Chronos-2 surpasses the next best models (TiRex and TimesFM-2.5) by a statistically significant margin. Specifically, the CIs of the pairwise win rates and skill scores of Chronos-2 against any baseline do not include 50% and 0%, respectively.

Model Avg. Win Rate (%) Skill Score (%) Chronos-2 81.9 51.4 TimesFM-2.5 77.5 51.0 TiRex 76.5 50.2 Toto-1.0 67.4 48.6 Moirai-2.0 64.4 48.4 COSMIC 56.4 44.5 Chronos-Bolt 53.8 42.6 TabPFN-TS 53.5 43.1 Sundial 49.1 44.1 AutoARIMA 21.8 8.8 Seasonal Naive 16.6 0.0 AutoTheta 16.0 -24.4 AutoETS 15.2 -648.9

Model Avg. Win Rate (%) Skill Score (%) Chronos-2 83.8 30.2 TimesFM-2.5 77.7 29.5 TiRex 71.9 27.6 Moirai-2.0 64.3 27.2 Toto-1.0 61.3 25.2 Chronos-Bolt 58.4 19.2 Sundial 53.4 25.0 COSMIC 51.9 20.8 TabPFN-TS 45.4 16.6 AutoARIMA 24.4 -7.4 AutoETS 19.5 -21.2 Seasonal Naive 19.4 0.0 AutoTheta 18.5 -9.0

(a)

(b)

- Table 4: GIFT-Eval results. The average win rate and skill score with respect to the (a) weighted quantile loss (WQL) and (b) mean absolute scaled error (MASE) metrics. Higher values are better for both. Chronos-2 outperforms previous best models, TimesFM-2.5 and TiRex. Baseline results have been taken from the GIFT-Eval leaderboard (Aksu et al., 2024).

GIFT-Eval. The GIFT-Eval benchmark comprises 97 tasks derived from 55 datasets, with a particular emphasis on high-frequency time series and long-horizon forecasting. The results in Table 4 show that Chronos-2 surpasses the previously leading models (TiRex and TimesFM-2.5) in win rate and skill score under both the weighted quantile loss (WQL) and mean absolute scaled error (MASE) metrics. When constructing the pretraining corpus for Chronos-2, we carefully ensured that it did not overlap with the test portions of any GIFT-Eval task at any sampling frequency. Nonetheless, the corpus does include partial overlap with the training portions of some GIFT-Eval datasets. For strictly zero-shot results, we refer the reader to Section 5.4, where we evaluate a variant of Chronos-2 trained exclusively on synthetic data.

Model Avg. Win Rate (%) Skill Score (%) Chronos-2 79.8 46.6 TiRex 70.4 41.7 TimesFM-2.5 70.0 42.4 Toto-1.0 60.9 41.9 Moirai-2.0 56.0 40.9 Chronos-Bolt 49.4 39.3 TabPFN-TS 46.3 32.6 COSMIC 42.8 36.7 Sundial 14.4 24.1 Seasonal Naive 10.1 0.0

(a)

Model Avg. Win Rate (%) Skill Score (%) Chronos-2 81.5 26.5 TimesFM-2.5 71.6 23.3 TiRex 67.1 22.2 Toto-1.0 58.0 22.3 Moirai-2.0 53.5 19.8 Chronos-Bolt 50.6 20.4 COSMIC 42.0 18.1 TabPFN-TS 40.1 10.5 Sundial 21.8 9.5 Seasonal Naive 13.8 0.0

(b)

- Table 5: Chronos Benchmark II results. The average win rate and skill score with respect to the (a) weighted quantile loss (WQL) and (b) mean absolute scaled error (MASE) metrics. Higher values are better for both. Chronos-2 achieves the best results across all metrics.

Chronos Benchmark II. Originally proposed in Ansari et al. (2024) to evaluate the first Chronos models, this benchmark comprises 27 tasks, the majority of which involve short histories (fewer than 300 time steps on average). None of these datasets were included in the training corpus of Chronos-2. On this benchmark, Chronos-2 consistently outperforms existing models in terms of the win rate and skill score under both probabilistic (WQL) and point (MASE) forecasting metrics, as shown in Table 5.

Taken together, these results show that Chronos-2 not only outperforms all competing models across the three benchmarks but also substantially improves over Chronos-Bolt, its predecessor, highlighting the impact of the architectural and training improvements in Chronos-2.

- 5.2 Improvements with In-context Learning

The results in Section 5.1 correspond to Chronos-2 with in-context learning (ICL) enabled, specifically in the full cross learning mode described in Section 3.4. In this section, we disentangle the gains from ICL compared to univariate inference. To this end, we split fev-bench into three subsets: the univariate subset with 32 tasks involving a single target time series without covariates, the multivariate subset with 26 tasks containing multiple targets but no covariates, and the covariates subset with 42 tasks that include at least one past-only or known covariate. We compare Chronos-2 with ICL to its univariate inference mode on these three subsets, as well as on GIFT-Eval and Chronos Benchmark II. In the univariate mode, each time series in the batch is forecast independently, and covariates, if present, are ignored.

###### fev-bench (univariate subset)

###### GIFT-Eval

###### Chronos Benchmark II

whCL

whCL

whCL

360 349350

370

509510 502

514

440 417419424

466

una

una

una

Chronos-2 TiRex TimesFM-2.5

Chronos-2 TimesFM-2.5

Chronos-2 TimesFM-2.5

TiRex

Toto-1.0 TiRex

314316 298301

484486

Toto-1.0 TabPFN-TS

Toto-1.0 Moirai-2.0

409 393

Moirai-2.0 Chronos-Bolt

441445 426431

Chronos-Bolt COSMIC Moirai-2.0 Sundial

COSMIC

367

Sundial TabPFN-TS

COSMIC TabPFN-TS Sundial

290

326

229

241

Chronos-Bolt

Average Skill Score (%) w.r.t. SQL

Average Skill Score (%) w.r.t. WQL

Average Skill Score (%) w.r.t. WQL

(a)

(b)

(c)

- Figure 3: Chronos-2’s probabilistic forecasting results in univariate mode and the corresponding improvements from in-context learning (ICL), shown as stacked bars on (a) the univariate subset of fev-bench, (b) GIFT-Eval, and (c) Chronos Benchmark II. For these univariate benchmarks, ICL enables cross-learning, allowing the model to share information across items within a batch and thereby generate more accurate forecasts than univariate inference alone. Results for point forecasting metrics are available in Figure 9 (Appendix).

Univariate Tasks. ICL provides improvements in skill score on univariate tasks, as shown in Figure 3. The effect is especially strong on Chronos Benchmark II (Figure 3 (b)), which contains many tasks with short contexts. This demonstrates that Chronos-2 can leverage information from related time series to improve predictions when ICL is enabled, particularly when limited time series history is available.

Average Skill Score (%) w.r.t. SQL

Chronos-2

Toto-1.0 TimesFM-2.5

TiRex Moirai-2.0 COSMIC Chronos-Bolt

Sundial TabPFN-TS

57.157.6 55.756.1

53.4 52.152.3

51.1

47.7

57.9

unva

whCL

fev-bench (multivariate subset)

(a)

Average Skill Score (%) w.r.t. SQL

Chronos-2 TabPFN-TS

TiRex TimesFM-2.5 Moirai-2.0 COSMIC Chronos-Bolt Toto-1.0 Sundial

39.940.0 38.7

37.6 36.6

35.936.0 35.1

28.0

47.0

unva

whCL

fev-bench (covariates subset)

(b)

- Figure 4: Chronos-2’s probabilistic forecasting results in univariate mode and the corresponding gains from in-context learning (ICL), shown as stacked bars on the multivariate and covariates subsets of fev-bench. On multivariate tasks, ICL provides only modest improvements, though Chronos-2 in univariate mode already surpasses the multivariate Toto-1.0 model. On the covariates subset, however, ICL delivers the largest gains, demonstrating Chronos-2’s ability to effectively use covariates. Besides Chronos-2, only TabPFN-TS and COSMIC support covariates, and Chronos-2 outperforms all baselines (including TabPFN-TS and COSMIC) by a wide margin. Results for point forecasting metrics are available in Figures 10a and 10b (Appendix).

Multivariate Tasks. On the multivariate subset of fev-bench, ICL yields only modest gains over univariate inference (Figure 4a (a)). Interestingly, in univariate mode, Chronos-2 even outperforms Toto-1.0, a model which natively supports multivariate forecasting. This suggests that while these tasks involve multiple variates with potentially shared dynamics, the benefits of explicit multivariate modeling can be limited. One possible intuition comes from Takens’s Embedding Theorem (Takens, 2006), which implies that the dynamics of a system can often

be reconstructed from delayed observations of a single variable. In practice, this means that with sufficiently long histories, a strong univariate model may capture much of the same structure as a multivariate model. Similar empirical findings have been reported elsewhere; for example, Nie et al. (2023) observed that univariate (“channel-independent”) models often perform on par with multivariate (“channel-dependent”) models, albeit on a different benchmark.

Tasks with Covariates. The largest gains with ICL are observed on tasks with covariates (Figure 4a (b)). Here, the performance margin clearly demonstrates that Chronos-2 with ICL can effectively exploit covariates to improve predictions compared to univariate inference, which ignores them. Chronos-2 outperforms baselines by a large margin on this subset. Unsurprisingly, the second spot is taken by TabPFN-TS, another model which supports (known) covariates. These results underscore both the strength of Chronos-2 and the limitations of existing pretrained models, most of which lack covariate support — a capability of immense practical importance.

###### Energy Tasks with Covariates

###### Retail Tasks with Covariates

whCL

whCL

40.9 38.6

51.3

unvar

Chronos-2

44.3 42.1

48.6

unva

Chronos-2

46.5

TabPFN-TS

TabPFN-TS

39.8

TiRex

TiRex

36.7

35.3

Chronos-Bolt

Chronos-Bolt

Average Skill Score (%) w.r.t. SQL

Average Skill Score (%) w.r.t. WQL

(a)

(b)

- Figure 5: Comparison of Chronos-2 against baselines on tasks which include dynamic covariates from the energy and retail domains. Chronos-2 outperforms all baselines by a wide margin, including TabPFN-TS and TiRex, the strongest baselines on the covariates subset of fev-bench (Figure 4b). For retail, we consider the domain-appropriate WQL metric. Results for point forecasting metrics are available in Figures 11a and 11b (Appendix).

25

50

75

Price

Forecast (Univariate)

25

50 Price

Forecast (ICL)

20000

25000 Ampirion Load

2017-12-07 2017-12-08 2017-12-09 2017-12-10 2017-12-11 2017-12-12

10000

20000

30000 Solar + Wind

- Figure 6: Forecasts generated by Chronos-2 in univariate mode (top), i.e., without covariates, and with in-context learning (second from top) on the energy price forecasting task. The dashed vertical gray line indicates the forecast start date and the shaded region represents 80% prediction interval around the median forecast. With ICL, Chronos-2 leverages Ampirion Load and Solar + Wind covariates to produce a more accurate prediction.

- 5.3 Domain Case Studies

We conducted further analysis on tasks from the energy and retail domains, where covariates often provide crucial information for accurate forecasting. For both domains, we selected all tasks with dynamic covariates from fev-bench resulting in 16 and 17 tasks for energy and retail, respectively (see Tables 10 and 11 in the Appendix for details). As baselines, we used TabPFN-TS and TiRex, the two strongest models on the covariates subset of fev-bench, as shown in Figure 4b. The results in Figures 5a and 5b demonstrate that Chronos-2 consistently outperforms these baselines by a wide margin. Incorporating covariates provides a substantial boost in performance for Chronos-2, reinforcing their critical role in real-world forecasting tasks. Consistent with Figure 4b, the second-best results are achieved by TabPFN-TS, another model capable of leveraging covariates.

To illustrate how Chronos-2 with ICL uses covariates, we compared forecasts produced in univariate mode versus with ICL. We selected one task from each domain where ICL delivers the largest gains.

| |Sales<br><br>Forecast (Univariate)| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

50000

25000

| |Sales<br><br>Forecast (ICL)| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

50000

25000

0.8 Open

0.6

| |Promo| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

0.5

0.0

5

| |SchoolHoliday| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |

0

2

| |StateHoliday| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |

0

6000 Customers

4000

2013-04 2013-07 2013-10 2014-01 2014-04 2014-07 2014-10 2015-01 2015-04 2015-07

- Figure 7: Forecasts generated by Chronos-2 in univariate mode (top), i.e., without covariates, and with in-context learning (second from top) on the Rossmann sales forecasting task. The dashed vertical gray line indicates the forecast start date and the shaded region represents 80% prediction interval around the median forecast. With ICL, Chronos-2 produces a substantially more accurate forecast by capturing the influence of promotion and holiday covariates on future sales.

Figure 6 shows forecasts on the energy price forecasting task for Germany (EPF-DE), where the goal is to predict the hourly energy price for the next day using historical prices, day-ahead forecasts of the load and renewable (solar and wind) energy generation. In the univariate mode, Chronos-2 makes reasonable but imprecise predictions. However, with ICL, Chronos-2 effectively uses the covariates, producing significantly more accurate predictions.

The retail task in Figure 7 involves predicting next quarter’s weekly store sales of Rossmann, a European drug store chain, using historical sales and covariates: historical customer footfall plus known covariates indicating store operation, promotion periods, and holidays. Chronos-2’s univariate forecast is nearly flat with high uncertainty. In contrast, the ICL forecast leverages covariates — particularly promotion and holiday information — to capture the true sales dynamics over the forecast horizon.

- 5.4 Ablation Studies

fev-bench GIFT-Eval Chronos Bench. II

AverageSkillScore(%)

47.3

45.3

51.4 50.4

46.6

44.1

Chronos-2 Chronos-2-Small

Model

Model Size

(a)

fev-bench GIFT-Eval Chronos Bench. II

AverageSkillScore(%)

47.3

45.9

51.4 50.4

46.6 46.4

Chronos-2 Chronos-2-Synth

Model

Synthetic Data Only

(b)

fev-bench GIFT-Eval Chronos Bench. II

AverageSkillScore(%)

47.3 46.9

51.4

50.1

46.6 45.8

Chronos-2 Chronos-2-2K

Model

Long Context Post-training

(c)

- Figure 8: Comparison of the main Chronos-2 model (120M parameters) with (a) a smaller 28M-parameter model, (b) a model trained exclusively on synthetic data, and (c) the main model prior to long-context post-training.

In this section, we present additional experiments and ablations that disentangle the impact of different design choices. We investigate the performance of Chronos-2 across different parameter counts, evaluate models trained exclusively on synthetic data, and demonstrate the importance of post-training on long-context scenarios.

Model Size. We trained a small model with 28M parameters to understand the impact of model size on forecasting performance. As shown in Figure 8a, the small model delivers strong performance despite its reduced size. On GIFT-Eval, for instance, its skill score lags the base model by as little as 1% points, while offering nearly 2× faster inference. This makes it particularly suitable for low-resource environments, such as CPU-only settings, or applications where inference speed is prioritized over maximum forecast accuracy.

Synthetic Data Only. Synthetic time series data has played a pivotal role in advancing pretrained forecasting models (Ansari et al., 2024; Das et al., 2024b). TabPFN-TS (Hoo et al., 2025) demonstrated that strong performance is achievable even when training relies exclusively on synthetic data. To examine the limits of this approach, we trained a version of Chronos-2 using only synthetic data. On Chronos Benchmark II and GIFT-Eval, this model (Chronos-2-Synth) performs only slightly below the version with real data in its pretraining corpus (Figure 8b). It also delivers strong results on fev-bench, though with a larger performance gap. These results underscore the importance of synthetic data, suggesting that with further research, real data may not even be required for effective pretraining.

Long context Post-training. As described in Section 3.3, Chronos-2 is initially trained with a context length of 2,048 time steps and then post-trained with an extended context of 8,192 steps. Figure 8c compares the base model (denoted Chronos-2-2K) with the post-trained variant. Extending the context length yields gains, particularly on the GIFT-Eval benchmark, which contains many high-frequency datasets with long seasonal periods.

### 6 Discussion

We introduced Chronos-2, a pretrained time series model designed to handle a wide range of forecasting scenarios including univariate, multivariate, and covariate-informed tasks — in a zero-shot manner. Across three comprehensive benchmarks, Chronos-2 consistently outperforms existing foundation models, demonstrating that in-context learning enhances forecasting performance across diverse task types.

A particularly large performance gap appears on covariate-informed tasks, where Chronos-2 substantially surpasses prior foundation models. This highlights both the limitations of existing models and the critical role contextual information (e.g., covariates) plays in accurate forecasting. While Chronos-2 supports only numeric and categorical covariates, extending pretrained models to incorporate multimodal inputs, such as text, represents a promising direction for future research (Zhang et al., 2025).

Our results further emphasize the importance of synthetic data in enabling generalist forecasting. The abilities of Chronos-2 beyond univariate forecasting rely entirely on synthetic data, and ablation studies show that models trained solely on synthetic data perform only slightly worse than those trained on a mixture of real and synthetic datasets. We expect synthetic data to play an increasingly central role in advancing pretrained time series models.

Finally, the flexible group attention mechanism in Chronos-2 opens opportunities for further applications. For instance, time series could be grouped using sparse metadata or dense embeddings to enable retrieval-augmented forecasting, potentially improving performance in small-data or cold-start scenarios.

### Acknowledgements

We thank the developers of open-source libraries used in the development of Chronos-2, including but not limited to torch (Paszke et al., 2019), numpy (Harris et al., 2020), pandas (pandas development team, 2020; Wes McKinney, 2010), statsmodels (Seabold & Perktold, 2010), transformers (Wolf et al., 2020), gluonts (Alexandrov et al., 2020), autogluon (Shchur et al., 2023), statsforecast (Garza et al., 2022), einops (Rogozhnikov, 2022) and scikit-learn (Pedregosa et al., 2011). We also thank our colleagues at Amazon for their invaluable support in releasing Chronos-2: Kevin Ormiston, Jenna Larson, Larry Hardesty, Divya Sukumar, Lahari Chowtoori and Henri Yandell. Finally, we are grateful to our fellow researchers for insightful discussions and their contributions to the field: Andrew Gordon Wilson, Michael Mahoney, Dmitry Efimov, Christoph Bergmeir, Valentin Flunkert, David Salinas,

Imry Kissos, Devamanyu Hazarika, Tim Januschowski, Jan Gasthaus, William Gilpin, Annan Yu, Zelin He, Kashif Rasul, Rajat Sen, Yichen Zhou, Chenghao Liu, Taha Aksu, Gerald Woo, Emaad Khwaja and Ben Cohen.

### References

Taha Aksu, Gerald Woo, Juncheng Liu, Xu Liu, Chenghao Liu, Silvio Savarese, Caiming Xiong, and Doyen Sahoo. Gift-Eval: A benchmark for general time series forecasting model evaluation. arXiv preprint arXiv:2410.10393, 2024. 1, 2, 7, 8, 9, 10

Alexander Alexandrov, Konstantinos Benidis, Michael Bohlke-Schneider, Valentin Flunkert, Jan Gasthaus, Tim Januschowski, Danielle C Maddix, Syama Rangapuram, David Salinas, Jasper Schulz, et al. GluonTS: Probabilistic and Neural Time Series Modeling in Python. The Journal of Machine Learning Research, 21(1):4629–4634, 2020. 14

Abdul Fatir Ansari, Lorenzo Stella, Ali Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Bernie Wang. Chronos: Learning the language of time series. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. 1, 2, 4, 7, 8, 9, 10, 14, 20

Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, and Cordelia Schmid. Vivit: A video vision transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 6836–6846, 2021. 4

V. Assimakopoulos and K. Nikolopoulos. The theta model: a decomposition approach to forecasting. International Journal of Forecasting, 16(4):521–530, 2000. 3

Andreas Auer, Raghul Parthipan, Pedro Mercado, Abdul Fatir Ansari, Lorenzo Stella, Bernie Wang, Michael BohlkeSchneider, and Syama Sundar Rangapuram. Zero-shot time series forecasting with covariates via in-context learning. arXiv preprint arXiv:2506.03128, 2025a. 4, 8

Andreas Auer, Patrick Podest, Daniel Klotz, Sebastian Böck, Günter Klambauer, and Sepp Hochreiter. TiRex: Zero-shot forecasting across long and short horizons with enhanced in-context learning. In Advances in Neural Information Processing Systems, 2025b. 4, 8

Fouad Bahrpeyma, Mark Roantree, Paolo Cappellari, Michael Scriney, and Andrew McCarren. A methodology for validating diversity in synthetic time series generation. MethodsX, 8:101459, 2021. 7

Marta Bańbura, Domenico Giannone, and Lucrezia Reichlin. Large bayesian vector auto regressions. Journal of applied Econometrics, 25(1):71–92, 2010. 1

John B Burbidge, Lonnie Magee, and A Leslie Robb. Alternative transformations to handle extreme values of the dependent variable. Journal of the American statistical Association, 83(401):123–127, 1988. 4

Cristian Challu, Kin G Olivares, Boris N Oreshkin, Federico Garza Ramirez, Max Mergenthaler Canseco, and Artur Dubrawski. N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, 2023. 1, 3

Ben Cohen, Emaad Khwaja, Youssef Doubli, Salahidine Lemaachi, Chris Lettieri, Charles Masson, Hugo Miccinilli, Elise Ramé, Qiqi Ren, Afshin Rostamizadeh, et al. This time is different: An observability perspective on time series foundation models. In Advances in Neural Information Processing Systems, 2025. 1, 4, 8

Abhimanyu Das, Matthew Faw, Rajat Sen, and Yichen Zhou. In-context fine-tuning for time-series foundation models. arXiv preprint arXiv:2410.24087, 2024a. 4

Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting. In International Conference on Machine Learning, 2024b. 1, 4, 8, 14

Patrick Emami, Abhijeet Sahu, and Peter Graf. BuildingsBench: A Large-Scale Dataset of 900K Buildings and Benchmark for Short-Term Load Forecasting. arXiv:2307.00142, 2023. 20

FiveThirtyEight. uber-tlc-foil-response: Uber trip data from a freedom of information request to NYC’s Taxi & Limousine Commission. https://github.com/fivethirtyeight/uber-tlc-foil-response, 2025. Accessed: 2025-09-26. 20

Azul Garza, Cristian Challu, and Max Mergenthaler-Canseco. Timegpt-1, 2024. 4 Federico Garza, Max Mergenthaler Canseco, Cristian Challú, and Kin G. Olivares. StatsForecast: Lightning fast

forecasting with statistical and econometric models. PyCon Salt Lake City, Utah, US 2022, 2022. URL https: //github.com/Nixtla/statsforecast. 14

Rakshitha Godahewa, Christoph Bergmeir, Geoffrey I. Webb, Rob J. Hyndman, and Pablo Montero-Manso. Monash Time Series Forecasting Archive. In Neural Information Processing Systems Track on Datasets and Benchmarks, 2021. 20

Nate Gruver, Marc Finzi, Shikai Qiu, and Andrew Gordon Wilson. Large Language Models Are Zero-Shot Time Series Forecasters. In Advances in Neural Information Processing Systems, 2023. 4

Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, and Travis E. Oliphant. Array programming with NumPy. Nature, 585(7825):357–362, September 2020. doi: 10.1038/ s41586-020-2649-2. URL https://doi.org/10.1038/s41586-020-2649-2. 14

Shi Bin Hoo, Samuel Müller, David Salinas, and Frank Hutter. From tables to time: How TabPFN-v2 outperforms

specialized time series forecasting models. arXiv preprint arXiv:2501.02945, 2025. 4, 8, 14 Rob J Hyndman and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2018. 1, 3, 8 Jiawei Jiang, Chengkai Han, Wenjun Jiang, Wayne Xin Zhao, and Jingyuan Wang. Libcity: A unified library towards

efficient and comprehensive urban spatial-temporal prediction. arXiv preprint arXiv:2304.14343, 2023. 20

Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, YuanFang Li, Shirui Pan, and Qingsong Wen. Time-LLM: Time series forecasting by reprogramming large language models. In The Twelfth International Conference on Learning Representations, 2024. 4

Xiaoyong Jin, Youngsuk Park, Danielle Maddix, Hao Wang, and Yuyang Wang. Domain adaptation for time series forecasting via attention sharing. In International Conference on Machine Learning, pp. 10280–10297. PMLR, 2022. 4

Bryan Lim, Sercan Ö Arık, Nicolas Loeff, and Tomas Pfister. Temporal fusion transformers for interpretable multihorizon time series forecasting. International Journal of Forecasting, 37(4):1748–1764, 2021. 1, 3

Xu Liu, Yutong Xia, Yuxuan Liang, Junfeng Hu, Yiwei Wang, Lei Bai, Chao Huang, Zhenguang Liu, Bryan Hooi, and Roger Zimmermann. Largest: A benchmark dataset for large-scale traffic forecasting. arXiv:2306.08259, 2023. 20

Yong Liu, Guo Qin, Zhiyuan Shi, Zhi Chen, Caiyin Yang, Xiangdong Huang, Jianmin Wang, and Mingsheng Long. Sundial: A family of highly capable time series foundation models. In International Conference on Machine Learning,

2025. 4, 8 Spyros Makridakis, Evangelos Spiliotis, and Vassilios Assimakopoulos. The M4 Competition: 100,000 time series and 61 forecasting methods. International Journal of Forecasting, 36(1):54–74, 2020. 20 Daniele Micci-Barreca. A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. ACM SIGKDD explorations newsletter, 3(1):27–32, 2001. 4 Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations, 2023. 3, 4, 5, 12 Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. In International Conference on Learning Representations, 2020. 3

Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. Meta-learning framework with applications to zero-shot time-series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2021. 3

Bernardo Pérez Orozco and Stephen J. Roberts. Zero-shot and few-shot time series forecasting with ordinal regression recurrent neural networks. In 28th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, pp. 503–508, 2020. 3

The pandas development team. pandas-dev/pandas: Pandas, February 2020. URL https://doi.org/10.5281/ zenodo.3509134. 14

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019. 14

Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. Scikit-learn: Machine learning in python. the Journal of machine Learning research, 12:2825–2830, 2011. 4, 14

Fotios Petropoulos and Ivan Svetunkov. A simple combination of univariate models. International journal of forecasting, 36(1):110–115, 2020. 8

Fotios Petropoulos, Daniele Apiletti, Vassilios Assimakopoulos, Mohamed Zied Babai, Devon K Barrow, Souhaib Ben Taieb, Christoph Bergmeir, Ricardo J Bessa, Jakub Bijak, John E Boylan, et al. Forecasting: theory and practice. International Journal of forecasting, 38(3):705–871, 2022. 2

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551, 2020. 5

Syama Sundar Rangapuram, Matthias W Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang, and Tim Januschowski. Deep state space models for time series forecasting. Advances in neural information processing systems, 31, 2018. 3

Roshan M Rao, Jason Liu, Robert Verkuil, Joshua Meier, John Canny, Pieter Abbeel, Tom Sercu, and Alexander Rives. Msa transformer. In International conference on machine learning, pp. 8844–8856. PMLR, 2021. 4

Stephan Rasp, Peter D Dueben, Sebastian Scher, Jonathan A Weyn, Soukayna Mouatadid, and Nils Thuerey. Weatherbench: a benchmark data set for data-driven weather forecasting. Journal of Advances in Modeling Earth Systems, 12(11):e2020MS002203, 2020. 20

Kashif Rasul, Calvin Seward, Ingmar Schuster, and Roland Vollgraf. Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting. In International Conference on Machine Learning, pp. 8857–8868. PMLR, 2021. 3

Kashif Rasul, Arjun Ashok, Andrew Robert Williams, Arian Khorasani, George Adamopoulos, Rishika Bhagwatkar, Marin Biloš, Hena Ghonia, Nadhir Vincent Hassen, Anderson Schneider, Sahil Garg, Alexandre Drouin, Nicolas Chapados, Yuriy Nevmyvaka, and Irina Rish. Lag-llama: Towards foundation models for time series forecasting, 2023. 4

Alex Rogozhnikov. Einops: Clear and reliable tensor manipulations with einstein-like notation. In International Conference on Learning Representations, 2022. 14

Jakob Runge, Andreas Gerhardus, Gherardo Varando, Veronika Eyring, and Gustau Camps-Valls. Causal inference for time series. Nature Reviews Earth & Environment, 4(7):487–505, 2023. 7

David Salinas, Michael Bohlke-Schneider, Laurent Callot, Roberto Medico, and Jan Gasthaus. High-dimensional multivariate forecasting with low-rank gaussian copula processes. Advances in neural information processing systems, 32, 2019. 20

David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting, 36(3):1181–1191, 2020. 3

Skipper Seabold and Josef Perktold. statsmodels: Econometric and statistical modeling with python. In 9th Python in Science Conference, 2010. 14

Oleksandr Shchur, Ali Caner Turkmen, Nick Erickson, Huibin Shen, Alexander Shirkov, Tony Hu, and Bernie Wang. Autogluon–timeseries: Automl for probabilistic time series forecasting. In International Conference on Automated Machine Learning, pp. 9–1. PMLR, 2023. 14

Oleksandr Shchur, Abdul Fatir Ansari, Caner Turkmen, Lorenzo Stella, Nick Erickson, Pablo Guerron, Michael Bohlke-Schneider, and Bernie Wang. fev-bench: A realistic benchmark for time series forecasting models. arXiv preprint arXiv:, 2025. 2, 8, 9

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 5

Floris Takens. Detecting strange attractors in turbulence. In Dynamical Systems and Turbulence, Warwick 1980: proceedings of a symposium held at the University of Warwick 1979/80, pp. 366–381. Springer, 2006. 11

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open Foundation and Fine-Tuned Chat Models, 2023. 5

Bartosz Uniejewski and Rafał Weron. Efficient forecasting of electricity spot prices with expert and lasso models. Energies, 11(8):2039, 2018. 4

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. In Advances in Neural Information Processing Systems, 2017. 5

Wes McKinney. Data Structures for Statistical Computing in Python. In Stéfan van der Walt and Jarrod Millman (eds.), Proceedings of the 9th Python in Science Conference, pp. 56 – 61, 2010. doi: 10.25080/Majora-92bf1922-00a. 14

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38–45. Association for Computational Linguistics, 2020. 14

Gerald Woo, Chenghao Liu, Akshat Kumar, and Doyen Sahoo. Pushing the limits of pre-training for time series forecasting in the cloudops domain. arXiv preprint arXiv:2310.05063, 2023. 20

Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, and Doyen Sahoo. Unified training of universal time series forecasting transformers. In International Conference on Machine Learning, 2024. 4, 7, 8

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In International Conference on Learning Representations, 2024. 5

Xiyuan Zhang, Boran Han, Haoyang Fang, Abdul Fatir Ansari, Shuai Zhang, Danielle C Maddix, Cuixiong Hu, Andrew Gordon Wilson, Michael W Mahoney, Hao Wang, et al. Does multimodality lead to better time series forecasting? arXiv preprint arXiv:2506.21611, 2025. 14

Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In The eleventh international conference on learning representations, 2023. 4

Nina Żukowska, Mononito Goswami, Michał Wiliński, Willa Potosnak, and Artur Dubrawski. Towards long-context time series foundation models. arXiv preprint arXiv:2409.13530, 2024. 4

### A Training Data

Dataset Name Frequencies # Time Series Domain Source

Electricity 15min, 1H, 1W, 1D 370 Energy Godahewa et al. (2021) KDD Cup (2018) 1H, 1D 270 Nature Godahewa et al. (2021) M4 (Daily) 1D 4227 Various Makridakis et al. (2020) M4 (Hourly) 1H 414 Various Makridakis et al. (2020) M4 (Monthly) 1M 48000 Various Makridakis et al. (2020) M4 (Weekly) 1W 359 Various Makridakis et al. (2020) Mexico City Bikes 1H, 1D, 1W 494 Transport Ansari et al. (2024) Pedestrian Counts 1H, 1D, 1W 66 Transport Godahewa et al. (2021) Solar 5min, 10min, 1H 5166 Energy Ansari et al. (2024) Taxi 30min, 1H 2428 Transport Salinas et al. (2019) Uber TLC 1H, 1D 262 Transport FiveThirtyEight (2025) USHCN 1D, 1W 225280 Nature Ansari et al. (2024) Weatherbench 1H, 1D, 1W 225280 Nature Rasp et al. (2020) Wiki 1H, 1D, 1W 100000 Web Ansari et al. (2024) Wind Farms 1H, 1D 337 Energy Godahewa et al. (2021) Temperature-Rain 1D 32072 Nature Godahewa et al. (2021) London Smart Meters 30min, 1D 5560 Energy Godahewa et al. (2021) Alibaba Cluster Trace (2018) 5min, 1H 100000 Cloud Ops Woo et al. (2023) Azure VM Traces (2017) 5min, 1H 100000 Cloud Ops Woo et al. (2023) Borg Cluster Data (2011) 5min, 1H 100000 Cloud Ops Woo et al. (2023) LargeST (2017) 1H, 1D 8196 Transport Liu et al. (2023) Q-Traffic 15min, 1H 45148 Transport Jiang et al. (2023) Buildings 900K 1H, 1D 100000 Energy Emami et al. (2023)

Table 6: Real univariate datasets used for pretraining Chronos-2.

### B Additional Results

###### fev-bench (univariate subset)

###### GIFT-Eval

###### Chronos Benchmark II

whCL

whCL

whCL

269 263

280

295296

302

241 233

265

una

una

una

Chronos-2 TimesFM-2.5

Chronos-2 TimesFM-2.5

Chronos-2 TimesFM-2.5

257

276 272

222223

TiRex

TiRex Moirai-2.0 Toto-1.0

Toto-1.0 TiRex Chronos-Bolt Moirai-2.0 COSMIC TabPFN-TS Sundial

221 208210

Toto-1.0 TabPFN-TS

250252

204 198

Chronos-Bolt Moirai-2.0 COSMIC Sundial

Sundial COSMIC

200203 184

208 192

181

105 95

Chronos-Bolt TabPFN-TS

166

Average Skill Score (%) w.r.t. MASE

Average Skill Score (%) w.r.t. MASE

Average Skill Score (%) w.r.t. MASE

(a)

(b)

- Figure 9: Chronos-2’s point forecasting results in univariate mode and the corresponding improvements from in-context learning (ICL), shown as stacked bars on (a) the univariate subset of fev-bench, (b) GIFT-Eval, and (c) Chronos Benchmark II.

Model Avg. Win Rate (%) Skill Score (%) Median runtime (s) Leakage (%) #Failures Chronos-2 87.9 35.5 3.6 0 0 TiRex 75.1 30.0 1.4 1 0 TimesFM-2.5 74.4 30.3 16.9 8 0 Toto-1.0 64.3 28.2 90.7 8 0 Moirai-2.0 58.7 27.3 2.5 28 0 COSMIC 58.6 25.7 34.4 0 0 Chronos-Bolt 57.9 26.5 1.0 0 0 TabPFN-TS 55.7 27.6 305.5 0 2 Sundial 49.8 24.7 35.6 1 0 Stat. Ensemble 44.2 15.7 690.6 0 11 AutoARIMA 32.1 11.2 186.8 0 10 AutoTheta 30.3 11.0 9.3 0 0 AutoETS 30.2 2.3 17.0 0 3 SeasonalNaive 16.7 0.0 2.3 0 0 Naive 14.0 -16.7 2.2 0 0

- Table 7: fev-bench results. The average win rate and skill score are computed with respect to the mean absolute scaled error (MASE) metric on fev-bench. Higher values are better for both.

Model Avg. Win Rate (%) Skill Score (%) Median runtime (s) Leakage (%) #Failures Chronos-2 88.5 51.5 3.6 0 0 TiRex 79.0 46.7 1.4 1 0 TimesFM-2.5 76.8 46.8 16.9 8 0 Toto-1.0 67.6 45.0 90.7 8 0 COSMIC 65.2 43.7 34.4 0 0 TabPFN-TS 64.8 45.8 305.5 0 2 Moirai-2.0 62.8 43.9 2.5 28 0 Chronos-Bolt 60.5 43.2 1.0 0 0 Sundial 41.9 37.4 35.6 1 0 Stat. Ensemble 38.3 21.8 690.6 0 11 AutoARIMA 34.6 23.4 186.8 0 10 AutoETS 26.8 -27.0 17.0 0 3 AutoTheta 21.3 7.8 9.3 0 0 SeasonalNaive 14.1 0.0 2.3 0 0 Naive 7.8 -39.1 2.2 0 0

- Table 8: fev-bench results. The average win rate and skill score are computed with respect to the weighted quantile loss (WQL) metric on fev-bench. Higher values are better for both.

Model Avg. Win Rate (%) Skill Score (%) Median runtime (s) Leakage (%) #Failures Chronos-2 85.4 39.4 3.6 0 0 TimesFM-2.5 74.1 33.8 16.9 8 0 TiRex 73.7 33.6 1.4 1 0 Toto-1.0 65.1 31.5 90.7 8 0 TabPFN-TS 61.5 33.4 305.5 0 2 COSMIC 60.5 30.1 34.4 0 0 Moirai-2.0 59.6 30.7 2.5 28 0 Chronos-Bolt 58.0 29.8 1.0 0 0 Sundial 47.7 27.3 35.6 1 0 Stat. Ensemble 43.0 17.7 690.6 0 11 AutoETS 30.8 4.3 17.0 0 3 AutoARIMA 30.8 13.3 186.8 0 10 AutoTheta 27.2 13.8 9.3 0 0 Naive 17.5 -6.1 2.2 0 0 SeasonalNaive 15.2 0.0 2.3 0 0

- Table 9: fev-bench results. The average win rate and skill score are computed with respect to the weighted absolute percentage error (WAPE) metric on fev-bench. Higher values are better for both.

###### fev-bench (multivariate subset)

###### fev-bench (covariates subset)

whCL

whCL

39.539.8 38.2

40.2

30.2 29.1

37.9

unva

unva

Chronos-2

Chronos-2 TabPFN-TS

Toto-1.0 TimesFM-2.5

28.128.5 27.1

TiRex TimesFM-2.5 Moirai-2.0

37.2 35.2

TiRex

Moirai-2.0 Chronos-Bolt

34.334.3 33.2

25.6 25.025.0

Chronos-Bolt Toto-1.0 COSMIC

Sundial COSMIC

31.2

23.1

TabPFN-TS

Sundial

Average Skill Score (%) w.r.t. MASE

Average Skill Score (%) w.r.t. MASE

(a)

(b)

- Figure 10: Chronos-2’s point forecasting results in univariate mode and the corresponding gains from in-context learning (ICL), shown as stacked bars on the multivariate and covariates subsets of fev-bench.

Average Skill Score (%) w.r.t. MASE

Chronos-2

TabPFN-TS

TiRex

Chronos-Bolt

36.3

29.2 27.3

26.5

41.8

unvar

whCL

Energy Tasks with Covariates

(a)

Average Skill Score (%) w.r.t. WAPE

Chronos-2

TabPFN-TS

TiRex

Chronos-Bolt

31.6

27.4 25.5

19.3

37.0

unvar

whCL

Retail Tasks with Covariates

(b)

- Figure 11: Comparison of Chronos-2 against baselines on tasks which include dynamic covariates from the energy and retail domains. For retail, we consider the domain-appropriate WAPE metric.

###### Pairwise Win Rate (SQL) with 95% CIs

###### Model 2

Stat.Ensemble

SeasonalNaive

TimesFM-2.5

Chronos-Bolt

AutoARIMA

TabPFN-TS

AutoTheta

Chronos-2

Moirai-2.0

AutoETS

COSMIC

Toto-1.0

Sundial

Naive

TiRex

Win Rate (%)

100

50 72 74 78 91 93 94 88 96 95 96 94 99 100 100

Chronos-2

(50.0, 50.0) (64.0, 81.0) (65.0, 82.0) (70.0, 86.0) (85.0, 96.0) (88.0, 98.0) (89.0, 98.0) (81.0, 94.0) (92.0, 99.0) (90.0, 99.0) (92.0, 99.0) (89.0, 98.0) (97.0, 100) (100, 100) (100, 100)

80

28 50 54.5 68.5 74 82.5 83.5 72 90.5 92 95 92 99 100 99

TiRex

(19.0, 36.0) (50.0, 50.0) (44.0, 64.0) (59.0, 77.5) (65.0, 82.0) (75.0, 89.5) (76.0, 90.5) (63.0, 80.0) (84.5, 95.5) (86.0, 97.0) (90.0, 99.0) (87.0, 96.0) (97.0, 100) (100, 100) (97.0, 100)

60

26 45.5 50 57 66 75 73 70 92.5 84 92 87 96 99 99

TimesFM-2.5

40

(18.0, 35.0) (36.0, 56.0) (50.0, 50.0) (48.0, 66.0) (57.0, 75.0) (67.5, 83.0) (65.0, 81.0) (60.0, 79.0) (87.0, 97.0) (76.0, 91.0) (86.0, 96.0) (80.0, 93.0) (91.0, 99.0) (97.0, 100) (97.0, 100)

20

22 31.5 43 50 47 60 57 57 82.5 80 86 83 90 96 98

Toto-1.0

(14.0, 30.0) (22.5, 41.0) (34.0, 52.0) (50.0, 50.0) (37.0, 58.0) (51.5, 69.0) (48.0, 66.0) (48.0, 66.0) (74.0, 89.5) (72.0, 87.0) (79.0, 92.0) (75.0, 90.0) (84.0, 95.0) (92.0, 99.0) (95.0, 100)

0

9 26 34 53 50 52 60 59 85 77 88 83 95 98 99

COSMIC

(4.0, 15.0) (18.0, 35.0) (25.0, 43.0) (42.0, 63.0) (50.0, 50.0) (43.0, 62.0) (50.0, 70.0) (49.0, 68.0) (77.0, 91.0) (69.0, 85.0) (81.0, 94.0) (76.0, 90.0) (90.0, 99.0) (95.0, 100) (97.0, 100)

7 17.5 25 40 48 50 54 53 85.5 79 87 82 88 94 95

Moirai-2.0

(2.0, 12.0) (10.5, 25.0) (17.0, 32.5) (31.0, 48.5) (38.0, 57.0) (50.0, 50.0) (45.5, 62.5) (44.0, 62.0) (78.5, 92.0) (71.0, 86.0) (80.0, 93.0) (74.0, 89.0) (81.0, 94.0) (89.0, 98.0) (90.0, 99.0)

6 16.5 27 43 40 46 50 53 81.5 78 87 80 92 96 98

Chronos-Bolt

(2.0, 11.0) (9.5, 24.0) (19.0, 35.0) (34.0, 52.0) (30.0, 50.0) (37.5, 54.5) (50.0, 50.0) (43.0, 62.0) (73.5, 88.5) (70.0, 86.0) (80.0, 93.0) (72.0, 87.0) (87.0, 97.0) (92.0, 99.0) (95.0, 100)

###### Model1

12 28 30 43 41 47 47 50 72 73.5 81 75 89 96 96

TabPFN-TS

(6.0, 19.0) (20.0, 37.0) (21.0, 40.0) (34.0, 52.0) (32.0, 51.0) (38.0, 56.0) (38.0, 57.0) (50.0, 50.0) (63.0, 80.0) (65.5, 82.0) (74.0, 89.0) (67.0, 83.0) (83.0, 95.0) (92.0, 99.0) (92.0, 99.0)

- 4 9.5 7.5 17.5 15 14.5 18.5 28 50 64 71 70 74 91 90
- 5 8 16 20 23 21 22 26.5 36 50 51 76.5 88 79.5 93

4 5 8 14 12 13 13 19 29 49 50 69.5 77 88 92

- 6 8 13 17 17 18 20 25 30 23.5 30.5 50 59 63.5 77

Sundial

(1.0, 8.0) (4.5, 15.5) (3.0, 13.0) (10.5, 26.0) (9.0, 23.0) (8.0, 21.5) (11.5, 26.5) (20.0, 37.0) (50.0, 50.0) (54.0, 74.0) (62.0, 79.0) (61.0, 79.0) (66.0, 83.0) (85.0, 96.0) (84.0, 96.0)

Stat. Ensemble

(1.0, 10.0) (3.0, 14.0) (9.0, 24.0) (13.0, 28.0) (15.0, 31.0) (14.0, 29.0) (14.0, 30.0) (18.0, 34.5) (26.0, 46.0) (50.0, 50.0) (41.5, 60.0) (68.0, 85.0) (80.0, 95.0) (72.0, 86.0) (87.0, 98.0)

AutoARIMA

- (1.0, 8.0) (1.0, 10.0) (4.0, 14.0) (8.0, 21.0) (6.0, 19.0) (7.0, 20.0) (7.0, 20.0) (11.0, 26.0) (21.0, 38.0) (40.0, 58.5) (50.0, 50.0) (60.5, 79.0) (68.0, 85.0) (82.0, 93.0) (86.0, 97.0)
- (2.0, 11.0) (4.0, 13.0) (7.0, 20.0) (10.0, 25.0) (10.0, 24.0) (11.0, 26.0) (13.0, 28.0) (17.0, 33.0) (21.0, 39.0) (15.0, 32.0) (21.0, 39.5) (50.0, 50.0) (49.0, 68.0) (54.0, 73.0) (69.0, 86.0)

AutoETS

1 1 4 10 5 12 8 11 26 12 23 41 50 68 83

AutoTheta

(0.0, 3.0) (0.0, 3.0) (1.0, 9.0) (5.0, 16.0) (1.0, 10.0) (6.0, 19.0) (3.0, 13.0) (5.0, 17.0) (17.0, 34.0) (5.0, 20.0) (15.0, 32.0) (32.0, 51.0) (50.0, 50.0) (59.0, 77.0) (75.0, 90.0)

- 0 0 1 4 2 6 4 4 9 20.5 12 36.5 32 50 71.5
- 0 1 1 2 1 5 2 4 10 7 8 23 17 28.5 50

SeasonalNaive

(0.0, 0.0) (0.0, 0.0) (0.0, 3.0) (1.0, 8.0) (0.0, 5.0) (2.0, 11.0) (1.0, 8.0) (1.0, 8.0) (4.0, 15.0) (14.0, 28.0) (7.0, 18.0) (27.0, 46.0) (23.0, 41.0) (50.0, 50.0) (64.5, 78.5)

Naive

(0.0, 0.0) (0.0, 3.0) (0.0, 3.0) (0.0, 5.0) (0.0, 3.0) (1.0, 10.0) (0.0, 5.0) (1.0, 8.0) (4.0, 16.0) (2.0, 13.0) (3.0, 14.0) (14.0, 31.0) (10.0, 25.0) (21.5, 35.5) (50.0, 50.0)

- Figure 12: The pairwise win rates for all models on fev-bench with 95% confidence intervals (CIs) with respect to SQL metric.

###### Pairwise Skill Score (SQL) with 95% CIs

###### Model 2

Stat.Ensemble

SeasonalNaive

TimesFM-2.5

Chronos-Bolt

AutoARIMA

TabPFN-TS

AutoTheta

Chronos-2

Moirai-2.0

AutoETS

COSMIC

Toto-1.0

Sundial

Naive

TiRex

Skill Score (%)

0 8.2 8.6 11 12.7 13.1 13.6 13.7 20.8 33.6 34 44.2 47.3 56.2 63.7

Chronos-2

(0.0, 0.0) (5.6, 11.2) (5.7, 11.6) (8.2, 14.2) (9.9, 15.8) (10.3, 16.3) (10.3, 17.1) (11.0, 16.7) (17.8, 23.6) (28.9, 37.9) (28.7, 39.0) (39.1, 49.9) (42.9, 52.2) (45.8, 65.6) (58.6, 68.3)

20

- −8.9 0 0.5 3.1 4.9 5.3 5.9 6 13.7 27.7 28.1 39.3 42.6 52.4 60.5
- −9.4 −0.5 0 2.6 4.5 4.9 5.5 5.6 13.3 27.4 27.7 39 42.3 52.2 60.3

TiRex

- (-12.6, -5.9) (0.0, 0.0) (-1.2, 2.0) (1.0, 5.3) (0.6, 9.0) (3.7, 7.1) (3.6, 8.6) (4.2, 8.0) (10.4, 16.7) (23.4, 31.8) (22.9, 32.9) (33.6, 44.8) (38.1, 47.5) (40.7, 62.6) (55.1, 65.4)
- (-13.1, -6.1) (-2.1, 1.2) (0.0, 0.0) (0.4, 5.1) (0.2, 9.0) (2.9, 6.9) (2.8, 8.6) (3.5, 7.8) (10.0, 16.4) (22.9, 31.8) (22.3, 32.9) (33.4, 45.2) (37.9, 47.4) (40.2, 62.4) (54.8, 65.2)

0

TimesFM-2.5

−20

−12.4 −3.2 −2.7 0 1.9 2.3 2.9 3 11 25.4 25.8 37.3 40.7 51 59.2

Toto-1.0

(-16.6, -8.9) (-5.6, -1.0) (-5.4, -0.4) (0.0, 0.0) (-3.3, 6.7) (0.3, 4.4) (-0.6, 6.4) (0.2, 5.9) (6.9, 14.6) (20.3, 30.1) (20.0, 31.2) (31.3, 43.5) (35.8, 46.3) (39.0, 61.7) (53.5, 64.6)

−14.6 −5.2 −4.7 −2 0 0.4 1 1.1 9.3 23.9 24.3 36.1 39.6 50.3 58.4

TabPFN-TS

(-18.7, -10.9) (-9.9, -0.6) (-9.9, -0.2) (-7.2, 3.2) (0.0, 0.0) (-4.2, 5.1) (-3.6, 5.9) (-3.1, 5.7) (4.7, 13.2) (18.7, 29.3) (18.8, 30.1) (30.4, 42.1) (34.6, 44.7) (38.1, 61.3) (53.0, 63.6)

Moirai-2.0

- (-19.4, -11.5) (-7.7, -3.8) (-7.4, -3.0) (-4.6, -0.3) (-5.4, 4.0) (0.0, 0.0) (-2.1, 3.4) (-1.7, 2.8) (5.0, 12.4) (18.6, 28.4) (18.5, 29.1) (29.9, 41.8) (34.4, 44.5) (37.7, 60.8) (52.6, 63.5)
- (-20.0, -12.3) (-8.7, -4.3) (-8.4, -3.7) (-6.3, -0.2) (-6.0, 3.0) (-2.9, 1.7) (-1.8, 1.9) (0.0, 0.0) (4.2, 11.6) (18.2, 27.4) (18.2, 28.6) (30.1, 41.3) (34.2, 44.1) (37.0, 60.5) (52.6, 62.9)

- −15.8 −6.3 −5.8 −3 −1.1 −0.6 0 0.1 8.3 23.1 23.5 35.4 39 49.6 58

−15.1 −5.6 −5.2 −2.4 −0.4 0 0.6 0.7 8.9 23.6 24 35.8 39.3 50 58.3

- −15.9 −6.4 −5.9 −3.1 −1.1 −0.7 −0.1 0 8.2 23.1 23.5 35.4 38.9 49.6 58

COSMIC

(-20.6, -11.5) (-9.4, -3.8) (-9.4, -2.9) (-6.8, 0.6) (-6.2, 3.5) (-3.5, 2.1) (0.0, 0.0) (-1.9, 1.8) (4.5, 11.7) (18.2, 27.7) (17.9, 28.6) (30.1, 41.0) (34.3, 44.1) (36.6, 60.6) (52.8, 62.7)

###### Model1

Chronos-Bolt

−26.3 −15.9 −15.4 −12.4 −10.2 −9.7 −9.1 −9 0 16.2 16.6 29.6 33.4 45.3 54.2

Sundial

(-30.8, -21.7) (-20.0, -11.6) (-19.7, -11.2) (-17.1, -7.5) (-15.2, -4.9) (-14.1, -5.3) (-13.3, -4.7) (-13.2, -4.4) (0.0, 0.0) (9.9, 21.7) (9.7, 23.2) (22.9, 36.5) (28.2, 39.0) (31.8, 57.4) (48.1, 59.9)

−50.7 −38.3 −37.7 −34.1 −31.5 −30.9 −30.1 −30 −19.3 0 0.5 16 20.6 35.9 45.4

AutoARIMA

(-60.9, -40.6) (-46.7, -30.6) (-46.6, -29.7) (-43.0, -25.5) (-41.5, -23.0) (-39.7, -22.9) (-38.3, -22.3) (-37.8, -22.3) (-27.7, -11.0) (0.0, 0.0) (-4.4, 5.1) (10.1, 22.5) (14.8, 25.9) (19.3, 50.0) (39.2, 51.4)

−51.4 −39 −38.4 −34.7 −32.2 −31.6 −30.8 −30.7 −19.9 −0.5 0 15.6 20.2 36.5 45.1

Stat. Ensemble

(-63.9, -40.3) (-49.0, -29.7) (-49.0, -28.6) (-45.4, -25.0) (-43.1, -23.2) (-41.0, -22.8) (-40.1, -21.8) (-40.1, -22.2) (-30.3, -10.7) (-5.4, 4.2) (0.0, 0.0) (10.9, 20.9) (14.3, 24.8) (20.7, 50.6) (39.6, 50.6)

−79.3 −64.6 −63.9 −59.6 −56.5 −55.8 −54.9 −54.7 −42 −19 −18.4 0 5.5 25.7 35

AutoTheta

(-99.6, -64.1) (-81.3, -50.7) (-82.3, -50.0) (-76.8, -45.5) (-72.6, -43.6) (-71.9, -42.7) (-69.4, -43.1) (-70.2, -43.1) (-57.4, -29.6) (-29.0, -11.2) (-26.4, -12.2) (0.0, 0.0) (-3.2, 12.3) (6.7, 41.7) (29.6, 40.0)

−89.7 −74.1 −73.3 −68.8 −65.5 −64.8 −63.8 −63.6 −50.2 −25.9 −25.3 −5.8 0 21.1 31.2

SeasonalNaive

(-109, -75.0) (-90.6, -61.6) (-90.1, -61.0) (-86.2, -55.9) (-80.9, -52.9) (-80.1, -52.5) (-78.7, -52.1) (-78.7, -51.9) (-63.9, -39.4) (-34.9, -17.4) (-33.0, -16.7) (-14.0, 3.1) (0.0, 0.0) (0.6, 39.3) (23.2, 38.6)

−128.5 −110.2 −109.1 −104.1 −101.1 −99.9 −98.2 −98.3 −82.8 −56 −57.5 −34.5 −26.8 0 10.4

AutoETS

(-191, -84.5) (-168, -68.7) (-166, -67.3) (-161, -64.0) (-159, -61.6) (-155, -60.5) (-154, -57.6) (-153, -58.7) (-135, -46.6) (-99.8, -24.0) (-102, -26.1) (-71.6, -7.2) (-64.8, -0.6) (0.0, 0.0) (-17.9, 31.7)

−175.8 −153.2 −152 −145.4 −140.7 −139.7 −138.2 −137.9 −118.4 −83 −82.1 −53.8 −45.4 −11.6 0

Naive

(-216, -141) (-189, -123) (-187, -121) (-182, -115) (-174, -113) (-174, -111) (-168, -112) (-170, -111) (-150, -92.6) (-106, -64.4) (-102, -65.5) (-66.6, -42.1) (-62.9, -30.3) (-46.4, 15.2) (0.0, 0.0)

- Figure 13: The pairwise skill scores for all models on fev-bench with 95% confidence intervals (CIs) with respect to SQL metric.

###### Pairwise Win Rate (WQL) with 95% CIs

###### Model 2

Stat.Ensemble

SeasonalNaive

TimesFM-2.5

Chronos-Bolt

AutoARIMA

TabPFN-TS

AutoTheta

Chronos-2

Moirai-2.0

AutoETS

COSMIC

Toto-1.0

Sundial

Naive

TiRex

Win Rate (%)

100

50 74 71 70 90 81 86 90 97 95 96 95 98 98 98

Chronos-2

(50.0, 50.0) (65.0, 82.0) (62.0, 80.0) (61.0, 79.0) (83.0, 95.0) (73.0, 88.0) (80.0, 93.0) (83.0, 95.0) (93.0, 100) (90.0, 99.0) (92.0, 99.0) (91.0, 99.0) (95.0, 100) (95.0, 100) (95.0, 100)

80

26 50 53.5 68.5 77 64 79.5 81.5 89.5 89 93 92 95 99 99

TiRex

(18.0, 35.0) (50.0, 50.0) (42.5, 63.0) (59.5, 77.0) (68.0, 85.0) (55.0, 73.0) (72.0, 87.0) (73.5, 88.5) (83.0, 95.0) (82.0, 95.0) (88.0, 98.0) (86.0, 97.0) (90.0, 99.0) (96.0, 100) (96.0, 100)

60

- 29 46.5 50 57 70 64 74 77 90.5 90 93 90 96 99 99
- 30 31.5 43 50 52 54 59 61 79.5 81 85 84 93 96 97

TimesFM-2.5

40

- (20.0, 38.0) (37.0, 57.5) (50.0, 50.0) (47.5, 66.0) (61.0, 79.0) (55.0, 73.0) (65.5, 82.5) (68.5, 85.0) (84.0, 96.0) (83.0, 95.0) (88.0, 97.0) (84.0, 95.0) (91.0, 99.0) (97.0, 100) (97.0, 100)
- (21.0, 39.0) (23.0, 40.5) (34.0, 52.5) (50.0, 50.0) (42.0, 62.0) (44.0, 63.0) (50.0, 68.5) (52.0, 70.0) (71.0, 87.0) (72.0, 89.0) (77.0, 92.0) (76.0, 91.0) (88.0, 98.0) (92.0, 99.0) (93.0, 100)

20

Toto-1.0

0

10 23 30 48 50 52 52 62 84 85 88 85 97 98 99

COSMIC

(5.0, 17.0) (15.0, 32.0) (21.0, 39.0) (38.0, 58.0) (50.0, 50.0) (42.0, 61.0) (43.0, 62.0) (53.0, 71.0) (76.0, 90.0) (78.0, 91.0) (81.0, 94.0) (78.0, 91.0) (93.0, 100) (95.0, 100) (97.0, 100)

19 36 36 46 48 50 55 52 79 79.5 83 85 94 97 97

TabPFN-TS

(12.0, 27.0) (27.0, 45.0) (27.0, 45.0) (37.0, 56.0) (39.0, 58.0) (50.0, 50.0) (46.0, 65.0) (43.0, 61.0) (71.0, 86.0) (71.5, 87.0) (76.0, 90.0) (78.0, 91.0) (89.0, 98.0) (93.5, 99.5) (93.0, 100)

14 20.5 26 41 48 45 50 59 86.5 84 89 87 91 93 95

Moirai-2.0

(7.0, 20.0) (13.0, 28.0) (17.5, 34.5) (31.5, 50.0) (38.0, 57.0) (35.0, 54.0) (50.0, 50.0) (51.0, 67.0) (79.5, 92.0) (76.0, 91.0) (82.0, 95.0) (80.0, 93.0) (84.0, 96.0) (88.0, 97.0) (90.0, 99.0)

###### Model1

10 18.5 23 39 38 48 41 50 78.5 85 88 87 96 97 98

Chronos-Bolt

(5.0, 17.0) (11.5, 26.5) (15.0, 31.5) (30.0, 48.0) (29.0, 47.0) (39.0, 57.0) (33.0, 49.0) (50.0, 50.0) (70.0, 86.0) (77.0, 91.0) (81.0, 94.0) (80.0, 93.0) (92.0, 99.0) (93.0, 100) (95.0, 100)

- 3 10.5 9.5 20.5 16 21 13.5 21.5 50 68 73 71 77 91 91

5 11 10 19 15 20.5 16 15 32 50 50 75.5 87 83.5 97

- 4 7 7 15 12 17 11 12 27 50 50 68.5 75 88 91
- 5 8 10 16 15 15 13 13 29 24.5 31.5 50 57 62.5 76

Sundial

- (0.0, 7.0) (5.0, 17.0) (4.0, 16.0) (13.0, 29.0) (10.0, 24.0) (14.0, 29.0) (8.0, 20.5) (14.0, 30.0) (50.0, 50.0) (58.0, 77.0) (64.0, 81.0) (62.0, 80.0) (69.0, 85.0) (85.0, 96.0) (84.0, 96.0)
- (1.0, 10.0) (5.0, 18.0) (5.0, 17.0) (11.0, 28.0) (9.0, 22.0) (13.0, 28.5) (9.0, 24.0) (9.0, 23.0) (23.0, 42.0) (50.0, 50.0) (40.5, 59.5) (67.0, 84.5) (80.0, 94.0) (76.5, 89.0) (93.0, 100)

Stat. Ensemble

AutoARIMA

- (1.0, 8.0) (2.0, 12.0) (3.0, 12.0) (8.0, 23.0) (6.0, 19.0) (10.0, 24.0) (5.0, 18.0) (6.0, 19.0) (19.0, 36.0) (40.5, 59.5) (50.0, 50.0) (58.5, 77.0) (66.0, 84.0) (81.5, 93.0) (85.0, 96.0)
- (1.0, 9.0) (3.0, 14.0) (5.0, 16.0) (9.0, 24.0) (9.0, 22.0) (9.0, 22.0) (7.0, 20.0) (7.0, 20.0) (20.0, 38.0) (15.5, 33.0) (23.0, 41.5) (50.0, 50.0) (47.0, 66.0) (53.5, 72.0) (68.0, 85.0)

AutoETS

2 5 4 7 3 6 9 4 23 13 25 43 50 69 85

AutoTheta

(0.0, 5.0) (1.0, 10.0) (1.0, 9.0) (2.0, 12.0) (0.0, 7.0) (2.0, 11.0) (4.0, 16.0) (1.0, 8.0) (15.0, 31.0) (6.0, 20.0) (16.0, 34.0) (34.0, 53.0) (50.0, 50.0) (59.0, 77.0) (78.0, 92.0)

2 1 1 4 2 3 7 3 9 16.5 12 37.5 31 50 68.5

SeasonalNaive

(0.0, 5.0) (0.0, 4.0) (0.0, 3.0) (1.0, 8.0) (0.0, 5.0) (0.5, 6.5) (3.0, 12.0) (0.0, 7.0) (4.0, 15.0) (11.0, 23.5) (7.0, 18.5) (28.0, 46.5) (23.0, 41.0) (50.0, 50.0) (61.5, 76.0)

2 1 1 3 1 3 5 2 9 3 9 24 15 31.5 50

Naive

(0.0, 5.0) (0.0, 4.0) (0.0, 3.0) (0.0, 7.0) (0.0, 3.0) (0.0, 7.0) (1.0, 10.0) (0.0, 5.0) (4.0, 16.0) (0.0, 7.0) (4.0, 15.0) (15.0, 32.0) (8.0, 22.0) (24.0, 38.5) (50.0, 50.0)

- Figure 14: The pairwise win rates for all models on fev-bench with 95% confidence intervals (CIs) with respect to WQL metric.

###### Pairwise Skill Score (WQL) with 95% CIs

###### Model 2

Stat.Ensemble

SeasonalNaive

TimesFM-2.5

Chronos-Bolt

AutoARIMA

TabPFN-TS

AutoTheta

Chronos-2

Moirai-2.0

AutoETS

COSMIC

Toto-1.0

Sundial

Naive

TiRex

Skill Score (%)

0 8.9 9.1 10.5 11.9 13.6 13.9 14.7 22.5 36.7 38 47.4 51.5 59.7 65.2

Chronos-2

(0.0, 0.0) (5.7, 12.3) (6.0, 12.5) (7.3, 13.5) (8.4, 15.7) (10.4, 17.5) (10.5, 17.6) (11.4, 18.1) (19.3, 25.5) (31.8, 41.4) (32.2, 43.3) (42.2, 52.8) (46.8, 56.1) (50.1, 68.7) (60.8, 69.4)

20

−9.7 0 0.3 1.8 3.3 5.2 5.6 6.4 15 30.5 32 42.3 46.8 55.8 61.8

TimesFM-2.5

- (-14.1, -6.1) (0.0, 0.0) (-1.4, 2.0) (-2.8, 5.8) (0.7, 6.0) (2.7, 7.8) (3.1, 7.9) (4.3, 8.5) (11.2, 18.6) (25.2, 35.5) (25.6, 37.6) (36.2, 48.1) (41.8, 51.7) (44.7, 65.6) (57.0, 66.4)

(-18.6, -9.2) (-6.4, -0.7) (-5.5, -0.9) (-7.4, 3.3) (0.0, 0.0) (-0.4, 4.2) (-0.6, 5.4) (0.2, 6.0) (7.4, 16.4) (22.5, 33.3) (22.8, 35.8) (34.1, 46.7) (39.4, 50.0) (43.0, 64.4) (55.2, 65.2)

(-21.4, -11.8) (-8.6, -3.2) (-8.1, -3.3) (-9.1, 0.5) (-5.7, 0.6) (-3.1, 2.5) (0.0, 0.0) (-1.2, 2.7) (5.7, 13.7) (21.3, 31.6) (21.8, 33.5) (33.2, 44.9) (38.8, 48.2) (41.7, 63.9) (54.8, 64.0)

- (-15.6, -7.9) (-6.2, 2.8) (-6.0, 3.3) (0.0, 0.0) (-3.4, 6.9) (-1.2, 8.6) (-0.5, 8.4) (0.3, 9.1) (8.8, 17.9) (24.1, 34.7) (25.2, 36.4) (35.9, 47.0) (41.1, 50.7) (44.4, 65.6) (56.5, 65.7)

0

−10 −0.3 0 1.6 3 5 5.3 6.1 14.8 30.4 31.8 42.1 46.7 55.8 61.7

TiRex

(-14.2, -6.4) (-2.0, 1.4) (0.0, 0.0) (-3.5, 5.7) (0.9, 5.2) (2.9, 7.2) (3.2, 7.5) (4.1, 8.2) (10.6, 18.6) (25.4, 35.0) (25.3, 37.4) (36.4, 47.9) (41.6, 51.3) (44.6, 65.6) (56.8, 66.3)

−20

TabPFN-TS

−13.4 −3.4 −3.1 −1.5 0 2 2.4 3.2 12.1 28.2 29.7 40.3 45 54.5 60.5

Toto-1.0

Moirai-2.0

- (-21.2, -11.6) (-8.5, -2.8) (-7.7, -3.0) (-9.4, 1.2) (-4.4, 0.4) (0.0, 0.0) (-2.6, 3.1) (-1.5, 3.6) (5.9, 14.2) (20.9, 32.0) (21.6, 34.1) (32.9, 45.2) (38.5, 48.5) (41.9, 63.7) (54.5, 64.4)
- (-22.1, -12.9) (-9.3, -4.4) (-9.0, -4.3) (-10.0, -0.3) (-6.4, -0.2) (-3.8, 1.5) (-2.8, 1.2) (0.0, 0.0) (4.7, 13.2) (20.2, 31.0) (21.0, 33.0) (32.3, 44.4) (37.9, 47.9) (41.0, 63.5) (54.3, 63.8)

- −16.2 −5.9 −5.6 −4 −2.4 −0.4 0 0.8 10 26.5 28 38.9 43.7 53.4 59.5

−11.7 −1.8 −1.6 0 1.5 3.5 3.8 4.6 13.4 29.3 30.7 41.2 45.8 55.4 61.1

−15.8 −5.5 −5.3 −3.6 −2.1 0 0.3 1.2 10.3 26.7 28.2 39.1 43.9 53.7 59.7

- −17.2 −6.8 −6.5 −4.9 −3.3 −1.2 −0.9 0 9.2 25.8 27.4 38.4 43.2 53.1 59.2

COSMIC

###### Model1

Chronos-Bolt

−29 −17.6 −17.3 −15.5 −13.8 −11.4 −11.1 −10.1 0 18.3 20 32.1 37.4 48.6 55

Sundial

(-34.2, -23.9) (-22.9, -12.6) (-22.8, -11.9) (-21.8, -9.6) (-19.6, -8.0) (-16.6, -6.2) (-15.8, -6.1) (-15.2, -4.9) (0.0, 0.0) (11.3, 24.6) (12.1, 27.2) (25.3, 39.4) (31.7, 42.9) (35.5, 60.3) (49.2, 60.5)

−58 −44 −43.6 −41.4 −39.3 −36.5 −36 −34.8 −22.4 0 2.1 16.9 23.4 38.1 44.9

AutoARIMA

(-70.6, -46.6) (-54.9, -33.6) (-53.9, -34.0) (-53.2, -31.8) (-49.9, -29.1) (-47.0, -26.4) (-46.2, -27.1) (-44.9, -25.4) (-32.6, -12.8) (0.0, 0.0) (-1.9, 6.0) (11.9, 22.2) (18.1, 28.1) (22.3, 52.1) (39.6, 50.3)

−61.3 −47 −46.6 −44.3 −42.2 −39.3 −38.8 −37.7 −25 −2.1 0 15.1 21.8 38 43.8

Stat. Ensemble

(-76.3, -47.5) (-60.3, -34.5) (-59.6, -34.0) (-57.2, -33.7) (-55.7, -29.6) (-51.6, -27.5) (-50.4, -28.0) (-49.3, -26.6) (-37.4, -13.8) (-6.4, 1.9) (0.0, 0.0) (11.2, 19.8) (16.0, 26.2) (22.5, 51.8) (39.3, 48.4)

−90.1 −73.2 −72.8 −70.1 −67.6 −64.2 −63.6 −62.2 −47.3 −20.3 −17.8 0 7.8 27.8 33.8

AutoTheta

(-112, -73.0) (-92.8, -56.8) (-92.0, -57.2) (-88.8, -56.0) (-87.4, -51.7) (-82.4, -49.1) (-81.4, -49.8) (-79.9, -47.7) (-64.9, -33.8) (-28.5, -13.5) (-24.7, -12.6) (0.0, 0.0) (0.2, 13.7) (9.2, 44.3) (28.5, 39.0)

−106.3 −88 −87.5 −84.6 −81.8 −78.1 −77.5 −76 −59.8 −30.6 −27.9 −8.5 0 21.3 28.1

SeasonalNaive

(-128, -88.1) (-107, -71.7) (-105, -71.1) (-103, -69.9) (-100, -65.0) (-94.3, -62.5) (-93.2, -63.3) (-92.1, -61.0) (-75.0, -46.4) (-39.1, -22.1) (-35.6, -19.1) (-15.9, -0.2) (0.0, 0.0) (0.3, 40.0) (21.3, 34.9)

−148 −126.4 −126.1 −124.2 −119.8 −115.8 −114.8 −113.3 −94.5 −61.6 −61.2 −38.4 −27 0 5.9

AutoETS

(-219, -100) (-190, -81.0) (-191, -80.5) (-191, -79.7) (-181, -75.4) (-175, -72.2) (-177, -71.4) (-174, -69.6) (-152, -54.9) (-109, -28.6) (-107, -29.0) (-79.4, -10.2) (-66.7, -0.3) (0.0, 0.0) (-24.0, 26.4)

−187 −161.5 −160.8 −156.8 −153 −147.8 −147 −144.9 −122.4 −81.6 −77.9 −51 −39.1 −6.3 0

Naive

(-227, -155) (-198, -133) (-196, -132) (-192, -130) (-187, -123) (-181, -120) (-177, -121) (-176, -119) (-153, -96.8) (-101, -65.6) (-93.9, -64.8) (-63.9, -40.0) (-53.7, -27.0) (-35.9, 19.3) (0.0, 0.0)

- Figure 15: The pairwise skill scores for all models on fev-bench with 95% confidence intervals (CIs) with respect to WQL metric.

###### Pairwise Win Rate (MASE) with 95% CIs

###### Model 2

Stat.Ensemble

SeasonalNaive

TimesFM-2.5

Chronos-Bolt

AutoARIMA

TabPFN-TS

AutoTheta

Chronos-2

Moirai-2.0

AutoETS

COSMIC

Toto-1.0

Sundial

Naive

TiRex

Win Rate (%)

100

50 72 69 74 92 94 94 85 91 86 95 94 89 98 98

Chronos-2

(50.0, 50.0) (64.0, 80.0) (60.0, 77.0) (65.0, 82.0) (87.0, 97.0) (89.0, 98.0) (89.0, 98.0) (77.0, 91.0) (85.0, 96.0) (79.0, 92.0) (90.0, 99.0) (89.0, 98.0) (83.0, 95.0) (95.0, 100) (95.0, 100)

80

28 50 46.5 61.5 76.5 71 76.5 69 80.5 81 90 94 84 97 96

TiRex

(20.0, 36.0) (50.0, 50.0) (36.5, 56.0) (52.0, 71.0) (67.5, 85.0) (62.0, 80.0) (68.0, 84.5) (60.0, 78.0) (72.0, 88.0) (73.0, 88.0) (84.0, 95.0) (89.0, 98.0) (77.0, 91.0) (93.0, 100) (92.0, 99.0)

60

31 53.5 50 58 73 66 72 68 81.5 81 88 91 85 97 97

TimesFM-2.5

40

(23.0, 40.0) (44.0, 63.5) (50.0, 50.0) (49.0, 66.5) (65.5, 81.5) (57.0, 75.0) (63.5, 80.5) (58.0, 76.0) (74.0, 88.5) (72.0, 88.0) (81.0, 94.0) (85.0, 96.0) (78.0, 92.0) (93.0, 100) (93.0, 100)

20

26 38.5 42 50 58 54 56 58 67.5 73 81 82 79 91 94

Toto-1.0

(18.0, 35.0) (29.0, 48.0) (33.5, 51.0) (50.0, 50.0) (49.0, 67.0) (45.0, 64.0) (47.0, 65.5) (48.0, 67.0) (58.0, 76.0) (64.0, 82.0) (73.0, 88.0) (74.0, 89.0) (71.0, 86.0) (85.0, 96.0) (89.0, 98.0)

0

- 8 23.5 27 42 50 53 55 52 68.5 73 81 85 76 87 91

6 29 34 46 47 50 49 54 66 69 83 82 74 89 92

6 23.5 28 44 45 51 50 52 60.5 70 84 86 77 90 94

15 31 32 42 48 46 48 50 53 64.5 75 75 72 91 87

- 9 19.5 18.5 32.5 31.5 34 39.5 47 50 66 76 77 71 88 88

Moirai-2.0

- (3.0, 13.0) (15.0, 32.5) (18.5, 34.5) (33.0, 51.0) (50.0, 50.0) (43.0, 63.0) (47.0, 63.0) (42.0, 61.0) (59.5, 77.0) (64.0, 81.0) (72.0, 88.0) (78.0, 92.0) (67.0, 84.0) (80.0, 93.0) (85.0, 96.0)

(2.0, 11.0) (20.0, 38.0) (25.0, 43.0) (36.0, 55.0) (37.0, 57.0) (50.0, 50.0) (39.0, 58.0) (44.0, 63.0) (56.0, 74.0) (60.0, 78.0) (75.0, 90.0) (74.0, 89.0) (65.0, 83.0) (83.0, 95.0) (86.0, 97.0)

(2.0, 11.0) (15.5, 32.0) (19.5, 36.5) (34.5, 53.0) (37.0, 53.0) (42.0, 61.0) (50.0, 50.0) (42.0, 61.0) (50.5, 69.5) (61.0, 79.0) (76.0, 91.0) (79.0, 92.0) (69.0, 85.0) (84.0, 95.0) (89.0, 98.0)

(9.0, 23.0) (22.0, 40.0) (24.0, 42.0) (33.0, 52.0) (39.0, 58.0) (37.0, 56.0) (39.0, 58.0) (50.0, 50.0) (44.0, 62.0) (55.0, 73.5) (67.0, 83.0) (67.0, 84.0) (63.0, 81.0) (85.0, 96.5) (80.0, 93.0)

- (4.0, 15.0) (12.0, 28.0) (11.5, 26.0) (24.0, 42.0) (23.0, 40.5) (26.0, 44.0) (30.5, 49.5) (38.0, 56.0) (50.0, 50.0) (56.0, 75.0) (68.0, 84.0) (69.0, 85.0) (62.0, 80.0) (81.0, 94.0) (81.0, 94.0)

(8.0, 21.0) (12.0, 27.0) (12.0, 28.0) (18.0, 36.0) (19.0, 36.0) (22.0, 40.0) (21.0, 39.0) (26.5, 45.0) (25.0, 44.0) (50.0, 50.0) (55.0, 73.0) (66.0, 84.0) (71.5, 86.5) (72.5, 86.5) (78.0, 91.0)

- (1.0, 10.0) (5.0, 16.0) (6.0, 19.0) (12.0, 27.0) (12.0, 28.0) (10.0, 25.0) (9.0, 24.0) (17.0, 33.0) (16.0, 32.0) (27.0, 45.0) (50.0, 50.0) (45.0, 64.0) (50.0, 68.5) (67.5, 82.5) (69.0, 85.0)
- (2.0, 11.0) (2.0, 11.0) (4.0, 15.0) (11.0, 26.0) (8.0, 22.0) (11.0, 26.0) (8.0, 21.0) (16.0, 33.0) (15.0, 31.0) (16.0, 34.0) (36.0, 55.0) (50.0, 50.0) (49.0, 68.0) (70.0, 86.0) (76.0, 90.0)

- (5.0, 17.0) (9.0, 23.0) (8.0, 22.0) (14.0, 29.0) (16.0, 33.0) (17.0, 35.0) (15.0, 31.0) (19.0, 37.0) (20.0, 38.0) (13.5, 28.5) (31.5, 50.0) (32.0, 51.0) (50.0, 50.0) (54.0, 72.5) (55.0, 74.0)

COSMIC

Chronos-Bolt

###### Model1

TabPFN-TS

Sundial

14 19 19 27 27 31 30 35.5 34 50 64 75 79.5 79.5 85

Stat. Ensemble

- 5 10 12 19 19 17 16 25 24 36 50 55 59.5 75 77
- 6 6 9 18 15 18 14 25 23 25 45 50 59 78 83

AutoARIMA

AutoTheta

11 16 15 21 24 26 23 28 29 20.5 40.5 41 50 63.5 64

AutoETS

- 2 3 3 9 13 11 10 9 12 20.5 25 22 36.5 50 57.5
- 2 4 3 6 9 8 6 13 12 15 23 17 36 42.5 50

SeasonalNaive

- (0.0, 5.0) (0.0, 7.0) (0.0, 7.0) (4.0, 15.0) (7.0, 20.0) (5.0, 17.0) (5.0, 16.0) (3.5, 15.0) (6.0, 19.0) (13.5, 27.5) (17.5, 32.5) (14.0, 30.0) (27.5, 46.0) (50.0, 50.0) (49.5, 66.0)
- (0.0, 5.0) (1.0, 8.0) (0.0, 7.0) (2.0, 11.0) (4.0, 15.0) (3.0, 14.0) (2.0, 11.0) (7.0, 20.0) (6.0, 19.0) (9.0, 22.0) (15.0, 31.0) (10.0, 24.0) (26.0, 45.0) (34.0, 50.5) (50.0, 50.0)

Naive

- Figure 16: The pairwise win rates for all models on fev-bench with 95% confidence intervals (CIs) with respect to MASE metric.

###### Pairwise Skill Score (MASE) with 95% CIs

###### Model 2

Stat.Ensemble

SeasonalNaive

TimesFM-2.5

Chronos-Bolt

AutoARIMA

TabPFN-TS

AutoTheta

Chronos-2

Moirai-2.0

AutoETS

COSMIC

Toto-1.0

Sundial

Naive

TiRex

Skill Score (%)

0 7.4 7.8 10.1 10.8 11.3 12.2 13.1 14.3 23.5 27.3 27.5 34 35.5 44.7

Chronos-2

(0.0, 0.0) (4.5, 10.4) (5.2, 10.8) (7.2, 13.2) (7.7, 13.9) (8.6, 14.3) (9.5, 15.1) (9.7, 16.6) (11.2, 17.2) (19.0, 28.2) (22.8, 31.6) (23.6, 31.5) (28.3, 40.4) (31.6, 39.5) (39.2, 50.3)

20

−8 0 0.5 3 3.7 4.2 5.2 6.2 7.4 17.4 21.5 21.7 28.7 30.3 40.3

TimesFM-2.5

(-11.7, -4.7) (0.0, 0.0) (-1.1, 2.1) (0.8, 5.3) (-0.7, 8.1) (2.3, 6.2) (3.2, 7.5) (3.4, 9.5) (3.8, 10.8) (12.6, 21.9) (17.2, 25.6) (18.0, 25.9) (22.7, 35.4) (26.7, 34.2) (35.0, 45.8)

0

−8.5 −0.5 0 2.5 3.3 3.8 4.8 5.8 7 17 21.1 21.4 28.4 30 40

TiRex

(-12.2, -5.5) (-2.2, 1.1) (0.0, 0.0) (0.3, 4.7) (-0.9, 7.1) (2.2, 5.5) (2.9, 6.8) (3.4, 8.6) (3.3, 10.3) (13.0, 21.0) (17.1, 24.9) (18.0, 25.1) (22.6, 34.9) (26.4, 33.8) (34.7, 45.5)

−20

- −11.3 −3.1 −2.6 0 0.8 1.3 2.3 3.3 4.6 14.9 19.1 19.4 26.6 28.2 38.5
- −12.8 −4.4 −3.9 −1.3 −0.5 0 1 2.1 3.3 13.8 18 18.3 25.6 27.3 37.7

- −15.1 −6.6 −6.1 −3.5 −2.7 −2.1 −1.1 0 1.3 11.9 16.3 16.6 24 25.7 36.3

−13.9 −5.5 −5 −2.4 −1.6 −1 0 1.1 2.4 12.9 17.2 17.4 24.8 26.5 37

−12.2 −3.9 −3.4 −0.8 0 0.5 1.5 2.6 3.9 14.2 18.5 18.7 26 27.6 38

- −16.7 −8 −7.5 −4.8 −4 −3.5 −2.4 −1.3 0 10.8 15.2 15.5 23 24.7 35.5

Toto-1.0

- (-15.3, -7.8) (-5.6, -0.8) (-5.0, -0.3) (0.0, 0.0) (-4.4, 5.4) (-0.7, 3.4) (-0.3, 5.1) (-0.3, 6.9) (0.2, 8.5) (10.0, 19.5) (14.4, 23.5) (15.2, 23.8) (20.3, 33.5) (23.9, 32.7) (32.8, 44.3)
- (-16.7, -9.4) (-6.6, -2.4) (-5.9, -2.2) (-3.5, 0.7) (-5.3, 3.8) (0.0, 0.0) (-1.4, 3.2) (-0.8, 5.1) (-0.6, 7.2) (9.1, 18.0) (13.4, 22.3) (14.3, 22.3) (19.3, 32.4) (23.3, 31.3) (31.9, 43.5)

- (-19.9, -10.8) (-10.5, -3.5) (-9.4, -3.5) (-7.4, 0.3) (-7.5, 1.6) (-5.3, 0.8) (-3.3, 0.7) (0.0, 0.0) (-2.6, 4.7) (6.7, 16.7) (11.1, 20.7) (12.8, 20.6) (17.1, 31.5) (21.2, 29.9) (31.6, 41.5)

(-17.7, -10.5) (-8.1, -3.3) (-7.4, -3.0) (-5.4, 0.3) (-6.0, 2.8) (-3.3, 1.3) (0.0, 0.0) (-0.7, 3.2) (-1.7, 5.8) (8.4, 17.5) (12.5, 21.3) (13.9, 21.5) (18.2, 32.0) (22.5, 30.5) (31.8, 42.6)

(-16.2, -8.4) (-8.8, 0.7) (-7.6, 0.9) (-5.7, 4.3) (0.0, 0.0) (-3.9, 5.0) (-2.9, 5.7) (-1.6, 6.9) (-0.5, 7.8) (8.7, 19.6) (12.9, 23.9) (14.0, 23.7) (19.1, 33.8) (23.0, 32.0) (32.2, 44.0)

- (-20.8, -12.6) (-12.1, -3.9) (-11.5, -3.4) (-9.3, -0.2) (-8.5, 0.5) (-7.8, 0.6) (-6.1, 1.7) (-5.0, 2.6) (0.0, 0.0) (5.0, 16.4) (9.5, 20.4) (11.2, 20.2) (16.0, 30.6) (20.6, 28.9) (29.7, 41.8)

TabPFN-TS

Moirai-2.0

Chronos-Bolt

###### Model1

COSMIC

Sundial

−30.8 −21.1 −20.5 −17.5 −16.6 −16 −14.8 −13.6 −12.1 0 5 5.2 13.7 15.7 27.7

Stat. Ensemble

(-39.3, -23.5) (-28.0, -14.5) (-26.6, -14.9) (-24.3, -11.2) (-24.4, -9.5) (-22.0, -10.0) (-21.2, -9.1) (-20.1, -7.2) (-19.6, -5.3) (0.0, 0.0) (1.8, 8.9) (2.4, 8.3) (8.4, 19.9) (11.4, 19.8) (21.7, 34.7)

−37.6 −27.4 −26.8 −23.6 −22.7 −22 −20.8 −19.5 −17.9 −5.2 0 0.3 9.2 11.2 23.9

AutoARIMA

(-46.1, -29.6) (-34.4, -20.8) (-33.1, -20.6) (-30.8, -16.8) (-31.4, -14.8) (-28.7, -15.4) (-27.0, -14.3) (-26.1, -12.5) (-25.6, -10.5) (-9.7, -1.8) (0.0, 0.0) (-3.7, 4.1) (2.2, 16.6) (6.9, 15.4) (17.3, 31.0)

−38 −27.8 −27.2 −24 −23 −22.4 −21.1 −19.8 −18.3 −5.5 −0.3 0 8.9 11 23.7

AutoTheta

(-45.9, -30.9) (-35.0, -22.0) (-33.5, -21.9) (-31.2, -18.0) (-31.0, -16.3) (-28.8, -16.7) (-27.4, -16.1) (-26.0, -14.6) (-25.2, -12.6) (-9.0, -2.5) (-4.3, 3.6) (0.0, 0.0) (2.3, 16.0) (7.1, 14.5) (18.3, 29.9)

−51.5 −40.3 −39.7 −36.2 −35.1 −34.4 −33 −31.6 −29.9 −15.9 −10.1 −9.8 0 2.3 16.2

AutoETS

(-67.8, -39.5) (-54.9, -29.4) (-53.6, -29.2) (-50.5, -25.5) (-51.0, -23.7) (-47.9, -23.8) (-47.0, -22.3) (-46.0, -20.6) (-44.2, -19.0) (-24.9, -9.2) (-19.9, -2.3) (-19.0, -2.4) (0.0, 0.0) (-6.8, 9.4) (7.8, 25.2)

−55 −43.6 −42.9 −39.3 −38.2 −37.5 −36.1 −34.6 −32.9 −18.6 −12.7 −12.3 −2.3 0 14.3

SeasonalNaive

(-65.3, -46.1) (-52.1, -36.5) (-51.0, -35.8) (-48.6, -31.4) (-47.1, -29.8) (-45.6, -30.4) (-43.9, -29.0) (-42.6, -26.8) (-40.6, -25.9) (-24.7, -12.9) (-18.2, -7.4) (-17.0, -7.7) (-10.4, 6.4) (0.0, 0.0) (7.0, 22.6)

−80.9 −67.5 −66.7 −62.5 −61.3 −60.4 −58.8 −57.1 −55 −38.3 −31.4 −31.1 −19.4 −16.7 0

Naive

(-101, -64.6) (-84.7, -53.8) (-83.5, -53.2) (-79.6, -48.7) (-78.6, -47.5) (-77.0, -46.9) (-74.1, -46.7) (-71.1, -46.1) (-71.8, -42.3) (-53.1, -27.7) (-44.9, -20.9) (-42.6, -22.4) (-33.8, -8.4) (-29.2, -7.5) (0.0, 0.0)

- Figure 17: The pairwise skill scores for all models on fev-bench with 95% confidence intervals (CIs) with respect to MASE metric.

###### Pairwise Win Rate (WAPE) with 95% CIs

###### Model 2

Stat.Ensemble

SeasonalNaive

TimesFM-2.5

Chronos-Bolt

AutoARIMA

TabPFN-TS

AutoTheta

Chronos-2

Moirai-2.0

AutoETS

COSMIC

Toto-1.0

Sundial

Naive

TiRex

Win Rate (%)

100

50 66 73 68 82 87 84 89 93 91 88 92 94 93 95

Chronos-2

(50.0, 50.0) (57.0, 75.0) (65.0, 82.0) (59.0, 77.0) (74.0, 89.0) (80.0, 93.0) (77.0, 91.0) (83.0, 95.0) (88.0, 97.0) (85.0, 96.0) (82.0, 94.0) (86.0, 97.0) (89.0, 98.0) (88.0, 97.0) (90.0, 99.0)

80

34 50 50.5 54 63 67 71 74 83.5 84 85 91 92 93 95

TimesFM-2.5

(25.0, 43.0) (50.0, 50.0) (40.0, 61.5) (45.0, 63.5) (54.0, 72.0) (57.0, 77.0) (62.5, 79.5) (65.5, 81.5) (76.0, 90.5) (76.0, 91.0) (77.0, 92.0) (85.0, 96.0) (86.0, 97.0) (87.0, 98.0) (90.0, 99.0)

60

27 49.5 50 62.5 65 71 69.5 74.5 77.5 82 84 88 93 93 95

TiRex

40

(18.0, 35.0) (38.5, 60.0) (50.0, 50.0) (53.0, 72.0) (55.0, 74.0) (62.0, 80.0) (60.5, 78.0) (66.0, 82.0) (69.0, 85.0) (74.0, 89.0) (76.0, 91.0) (81.0, 94.0) (87.0, 98.0) (88.0, 98.0) (90.0, 99.0)

20

32 46 37.5 50 56 56 60 60 66.5 69 77 82 87 91 92

Toto-1.0

(23.0, 41.0) (36.5, 55.0) (28.0, 47.0) (50.0, 50.0) (47.0, 65.0) (46.0, 66.0) (51.0, 69.0) (51.0, 68.5) (57.0, 75.0) (59.0, 78.0) (68.0, 85.0) (73.0, 89.0) (80.0, 93.0) (86.0, 96.0) (86.0, 97.0)

0

18 37 35 44 50 50 52 53 66 77.5 77 83 84 91 94

TabPFN-TS

(11.0, 26.0) (28.0, 46.0) (26.0, 45.0) (35.0, 53.0) (50.0, 50.0) (40.0, 60.0) (44.0, 62.0) (44.0, 63.0) (56.0, 75.0) (70.0, 85.0) (70.0, 84.0) (76.0, 90.0) (77.0, 91.0) (85.0, 96.0) (89.5, 98.0)

13 33 29 44 50 50 48 53 68 75 79 84 89 90 92

COSMIC

(7.0, 20.0) (23.0, 43.0) (20.0, 38.0) (34.0, 54.0) (40.0, 60.0) (50.0, 50.0) (39.0, 58.0) (44.0, 63.0) (59.0, 76.0) (66.0, 83.0) (70.0, 86.0) (77.0, 91.0) (83.0, 95.0) (84.0, 95.0) (86.0, 97.0)

16 29 30.5 40 48 52 50 56 66.5 74 77 83 88 87 87

Moirai-2.0

(9.0, 23.0) (20.5, 37.5) (22.0, 39.5) (31.0, 49.0) (38.0, 56.0) (42.0, 61.0) (50.0, 50.0) (47.0, 64.5) (57.5, 75.0) (65.0, 82.0) (68.0, 85.0) (74.0, 90.0) (81.0, 94.0) (80.0, 93.0) (80.0, 93.0)

###### Model1

- 11 26 25.5 40 47 47 44 50 63.5 74 77 85 92 89 91

- 7 16.5 22.5 33.5 34 32 33.5 36.5 50 64 68 76 74 82 88

9 16 18 31 22.5 25 26 26 36 50 78.5 65 81 84 83.5

12 15 16 23 23 21 23 23 32 21.5 50 44.5 47 62 68.5

- 8 9 12 18 17 16 17 15 24 35 55.5 50 52 72 80

Chronos-Bolt

- (5.0, 17.0) (18.5, 34.5) (18.0, 34.0) (31.5, 49.0) (37.0, 56.0) (37.0, 56.0) (35.5, 53.0) (50.0, 50.0) (54.0, 72.0) (65.0, 82.0) (69.0, 84.0) (78.0, 92.0) (86.0, 97.0) (82.0, 95.0) (85.0, 96.0)

- (3.0, 12.0) (9.5, 24.0) (15.0, 31.0) (25.0, 43.0) (25.0, 44.0) (24.0, 41.0) (25.0, 42.5) (28.0, 46.0) (50.0, 50.0) (54.0, 73.0) (58.0, 77.0) (67.0, 84.0) (65.0, 82.0) (74.0, 89.0) (81.0, 94.0)
- (4.0, 15.0) (9.0, 24.0) (11.0, 26.0) (22.0, 41.0) (15.0, 30.0) (17.0, 34.0) (18.0, 35.0) (18.0, 35.0) (27.0, 46.0) (50.0, 50.0) (70.0, 86.0) (57.0, 73.5) (73.0, 89.0) (76.0, 91.0) (77.0, 89.5)

- (6.0, 18.0) (8.0, 23.0) (9.0, 24.0) (15.0, 32.0) (16.0, 30.0) (14.0, 30.0) (15.0, 32.0) (16.0, 31.0) (23.0, 42.0) (14.0, 30.0) (50.0, 50.0) (35.0, 54.0) (38.0, 57.0) (52.0, 72.0) (59.0, 77.0)

Sundial

Stat. Ensemble

AutoETS

AutoARIMA

(3.0, 14.0) (4.0, 15.0) (6.0, 19.0) (11.0, 27.0) (10.0, 24.0) (9.0, 23.0) (10.0, 26.0) (8.0, 22.0) (16.0, 33.0) (26.5, 43.0) (46.0, 65.0) (50.0, 50.0) (42.0, 61.0) (63.0, 80.0) (72.5, 86.5)

- 6 8 7 13 16 11 12 8 26 19 53 48 50 76 78
- 7 7 7 9 9 10 13 11 18 16 38 28 24 50 47.5

AutoTheta

- (2.0, 11.0) (3.0, 14.0) (2.0, 13.0) (7.0, 20.0) (9.0, 23.0) (5.0, 17.0) (6.0, 19.0) (3.0, 14.0) (18.0, 35.0) (11.0, 27.0) (43.0, 62.0) (39.0, 58.0) (50.0, 50.0) (68.0, 84.0) (70.0, 85.0)
- (3.0, 12.0) (2.0, 13.0) (2.0, 12.0) (4.0, 14.0) (4.0, 15.0) (5.0, 16.0) (7.0, 20.0) (5.0, 18.0) (11.0, 26.0) (9.0, 24.0) (28.0, 48.0) (20.0, 37.0) (16.0, 32.0) (50.0, 50.0) (39.0, 55.5)

Naive

5 5 5 8 6 8 13 9 12 16.5 31.5 20 22 52.5 50

SeasonalNaive

(1.0, 10.0) (1.0, 10.0) (1.0, 10.0) (3.0, 14.0) (2.0, 10.5) (3.0, 14.0) (7.0, 20.0) (4.0, 15.0) (6.0, 19.0) (10.5, 23.0) (23.0, 41.0) (13.5, 27.5) (15.0, 30.0) (44.5, 61.0) (50.0, 50.0)

Figure 18: The pairwise win rates for all models on fev-bench with 95% (CIs) with respect to WAPE metric.

###### Pairwise Skill Score (WAPE) with 95% CIs

###### Model 2

Stat.Ensemble

SeasonalNaive

TimesFM-2.5

Chronos-Bolt

AutoARIMA

TabPFN-TS

AutoTheta

Chronos-2

Moirai-2.0

AutoETS

COSMIC

Toto-1.0

Sundial

Naive

TiRex

Skill Score (%)

0 8.4 8.8 9.1 11.6 12.6 13.3 13.7 16.7 26.4 29.7 30.1 36.7 39.4 42.9

Chronos-2

(0.0, 0.0) (5.3, 11.7) (5.7, 12.2) (5.7, 12.2) (8.2, 15.2) (9.4, 16.3) (9.9, 16.9) (10.4, 17.1) (13.2, 20.0) (21.2, 31.3) (24.8, 34.3) (25.3, 34.8) (30.1, 43.8) (34.3, 43.9) (37.6, 47.8)

20

−9.2 0 0.4 0.7 3.4 4.6 5.3 5.8 9 19.6 23.2 23.7 30.8 33.8 37.6

TimesFM-2.5

(-13.2, -5.6) (0.0, 0.0) (-1.2, 2.1) (-3.9, 5.0) (0.8, 6.2) (2.1, 7.2) (3.0, 7.7) (3.7, 8.1) (5.0, 12.8) (14.3, 25.0) (18.8, 27.9) (18.4, 28.5) (24.1, 37.8) (29.3, 38.5) (32.3, 42.8)

0

−9.7 −0.4 0 0.3 3 4.2 4.9 5.4 8.7 19.3 22.9 23.4 30.6 33.6 37.4

TiRex

(-13.9, -6.1) (-2.1, 1.2) (0.0, 0.0) (-4.6, 4.8) (0.9, 5.2) (2.2, 6.4) (2.9, 7.1) (3.4, 7.5) (4.2, 12.6) (14.2, 24.3) (18.8, 27.5) (18.5, 28.2) (23.8, 37.6) (29.1, 38.2) (32.3, 42.3)

−20

TabPFN-TS

(-13.9, -6.1) (-5.3, 3.7) (-5.0, 4.4) (0.0, 0.0) (-2.1, 7.6) (-0.9, 8.8) (0.3, 9.0) (0.6, 9.2) (3.9, 12.9) (13.5, 24.2) (18.1, 27.5) (17.8, 28.4) (23.1, 37.9) (29.0, 38.1) (31.8, 42.4)

- −13.1 −3.5 −3.1 −2.8 0 1.2 2 2.4 5.8 16.8 20.5 21 28.4 31.5 35.4

−10 −0.7 −0.3 0 2.7 3.9 4.7 5.1 8.4 19 22.7 23.2 30.3 33.4 37.2

−15.3 −5.6 −5.2 −4.9 −2 −0.8 0 0.5 3.9 15.1 18.9 19.4 26.9 30.1 34.1

- −14.5 −4.8 −4.4 −4.1 −1.2 0 0.8 1.3 4.7 15.7 19.6 20 27.5 30.7 34.6
- −15.9 −6.1 −5.7 −5.4 −2.5 −1.3 −0.5 0 3.4 14.7 18.5 19 26.6 29.8 33.8

Toto-1.0

(-17.9, -9.0) (-6.7, -0.8) (-5.5, -0.9) (-8.2, 2.1) (0.0, 0.0) (-1.2, 3.6) (-1.2, 4.9) (-0.4, 5.2) (0.8, 10.2) (11.3, 22.0) (15.9, 25.5) (15.5, 26.0) (21.0, 35.9) (26.3, 36.5) (30.2, 40.7)

Moirai-2.0

- (-19.5, -10.3) (-7.8, -2.1) (-6.8, -2.2) (-9.7, 0.9) (-3.8, 1.2) (0.0, 0.0) (-2.0, 3.3) (-1.3, 3.7) (-0.0, 9.0) (10.4, 21.1) (14.9, 24.4) (14.2, 25.2) (20.1, 35.2) (25.8, 35.4) (29.3, 39.7)
- (-20.7, -11.6) (-8.8, -3.8) (-8.1, -3.5) (-10.1, -0.7) (-5.5, 0.4) (-3.9, 1.3) (-2.4, 1.4) (0.0, 0.0) (-1.1, 7.3) (9.2, 20.0) (14.2, 23.4) (13.5, 24.3) (19.2, 34.5) (25.1, 34.4) (28.4, 39.1)

COSMIC

(-20.4, -11.0) (-8.4, -3.1) (-7.7, -3.0) (-9.9, -0.3) (-5.2, 1.2) (-3.4, 2.0) (0.0, 0.0) (-1.4, 2.4) (-0.6, 7.8) (10.5, 19.9) (15.0, 23.2) (14.4, 24.5) (19.8, 34.7) (25.9, 34.6) (29.2, 39.3)

###### Model1

Chronos-Bolt

−20 −9.9 −9.5 −9.2 −6.2 −4.9 −4.1 −3.6 0 11.6 15.6 16.1 24 27.3 31.5

Sundial

(-25.0, -15.2) (-14.7, -5.2) (-14.4, -4.4) (-14.7, -4.1) (-11.4, -0.8) (-9.9, 0.0) (-8.5, 0.6) (-7.9, 1.1) (0.0, 0.0) (4.6, 18.1) (10.4, 21.1) (9.9, 21.9) (15.8, 32.4) (22.5, 32.1) (25.6, 37.7)

−35.8 −24.4 −23.9 −23.5 −20.1 −18.7 −17.8 −17.2 −13.2 0 4.5 5.1 14 17.7 22.4

Stat. Ensemble

(-45.6, -26.8) (-33.3, -16.7) (-32.1, -16.5) (-32.0, -15.6) (-28.3, -12.7) (-26.8, -11.6) (-24.8, -11.7) (-25.0, -10.1) (-22.1, -4.8) (0.0, 0.0) (0.7, 7.9) (1.9, 8.8) (7.4, 21.3) (13.5, 21.8) (17.5, 27.9)

−42.3 −30.3 −29.7 −29.4 −25.8 −24.3 −23.3 −22.7 −18.5 −4.7 0 0.6 9.9 13.8 18.8

AutoTheta

(-52.3, -33.0) (-38.7, -23.1) (-38.0, -23.2) (-37.9, -22.0) (-34.2, -18.9) (-32.3, -17.5) (-30.1, -17.6) (-30.6, -16.6) (-26.8, -11.6) (-8.6, -0.7) (0.0, 0.0) (-3.9, 5.4) (2.6, 17.6) (10.0, 17.4) (14.2, 23.4)

−43.1 −31 −30.5 −30.1 −26.6 −25.1 −24.1 −23.5 −19.2 −5.4 −0.6 0 9.3 13.3 18.3

AutoARIMA

(-53.3, -33.8) (-39.8, -22.6) (-39.2, -22.7) (-39.6, -21.6) (-35.2, -18.3) (-33.7, -16.6) (-32.5, -16.9) (-32.1, -15.6) (-28.1, -11.0) (-9.7, -2.0) (-5.7, 3.8) (0.0, 0.0) (1.6, 17.9) (9.2, 17.5) (12.5, 24.0)

−57.9 −44.6 −44 −43.6 −39.6 −38 −36.9 −36.2 −31.5 −16.2 −11 −10.3 0 4.3 9.8

AutoETS

(-78.0, -43.1) (-60.7, -31.8) (-60.3, -31.2) (-60.9, -30.1) (-56.1, -26.5) (-54.4, -25.2) (-53.2, -24.6) (-52.7, -23.8) (-48.0, -18.8) (-27.1, -8.0) (-21.4, -2.7) (-21.7, -1.6) (0.0, 0.0) (-6.2, 12.7) (0.7, 17.1)

−65 −51.1 −50.5 −50.1 −46 −44.2 −43.1 −42.4 −37.5 −21.5 −16 −15.3 −4.5 0 5.8

SeasonalNaive

(-78.3, -52.2) (-62.7, -41.4) (-61.8, -41.1) (-61.6, -40.9) (-57.6, -35.6) (-54.9, -34.8) (-52.9, -35.0) (-52.5, -33.5) (-47.3, -29.0) (-27.9, -15.6) (-21.1, -11.1) (-21.2, -10.1) (-14.6, 5.9) (0.0, 0.0) (-0.7, 12.7)

−75.1 −60.3 −59.7 −59.2 −54.9 −53 −51.8 −51.1 −45.9 −28.9 −23.1 −22.3 −10.9 −6.1 0

Naive

(-91.5, -60.2) (-74.7, -47.7) (-73.4, -47.7) (-73.7, -46.7) (-68.6, -43.2) (-65.9, -41.4) (-64.6, -41.3) (-64.3, -39.6) (-60.5, -34.4) (-38.7, -21.2) (-30.5, -16.5) (-31.5, -14.3) (-20.7, -0.7) (-14.5, 0.7) (0.0, 0.0)

Figure 19: The pairwise skill scores for all models on fev-bench with 95% confidence intervals (CIs) with respect to WAPE metric.

Task Freq. H W Median length # series # targets # past cov. # known cov. # static cov. ENTSO-e Load 15T 96 20 175,292 6 1 0 3 0 ENTSO-e Load 30T 96 20 87,645 6 1 0 3 0 ENTSO-e Load H 168 20 43,822 6 1 0 3 0 EPF-BE H 24 20 52,416 1 1 0 2 0 EPF-DE H 24 20 52,416 1 1 0 2 0 EPF-FR H 24 20 52,416 1 1 0 2 0 EPF-NP H 24 20 52,416 1 1 0 2 0 EPF-PJM H 24 20 52,416 1 1 0 2 0 GFC12 H 168 10 39,414 11 1 0 1 0 GFC14 H 168 20 17,520 1 1 0 1 0 GFC17 H 168 20 17,544 8 1 0 1 0 Solar with Weather 15T 96 20 198,600 1 1 2 7 0 Solar with Weather H 24 20 49,648 1 1 2 7 0 KDD Cup 2022 D 14 10 243 134 1 9 0 0 KDD Cup 2022 10T 288 10 35,279 134 1 9 0 0 KDD Cup 2022 30T 96 10 11,758 134 1 9 0 0

- Table 10: Subset of datasets from fev-bench with dynamic covariates for the energy domain case study.

Task Freq. H W Median length # series # targets # past cov. # known cov. # static cov. Favorita Store Sales M 12 2 54 1,579 1 1 1 6 Favorita Store Sales W 13 10 240 1,579 1 1 1 6 Favorita Store Sales D 28 10 1,688 1,579 1 1 2 6 Favorita Transactions M 12 2 54 51 1 1 0 5 Favorita Transactions W 13 10 240 51 1 1 0 5 Favorita Transactions D 28 10 1,688 51 1 1 1 5 M5 M 12 1 58 30,490 1 0 8 5 M5 W 13 1 257 30,490 1 0 8 5 M5 D 28 1 1,810 30,490 1 0 8 5 Rohlik Orders W 8 5 170 7 1 9 4 0 Rohlik Orders D 61 5 1,197 7 1 9 4 0 Rohlik Sales W 8 1 150 5,243 1 1 13 7 Rohlik Sales D 14 1 1,046 5,390 1 1 13 7 Rossmann W 13 8 133 1,115 1 1 4 10 Rossmann D 48 10 942 1,115 1 1 5 10 Walmart W 39 1 143 2,936 1 0 10 4 Hermes W 52 1 261 10,000 1 0 1 2

- Table 11: Subset of datasets from fev-bench with dynamic covariates for the retail domain case study.

