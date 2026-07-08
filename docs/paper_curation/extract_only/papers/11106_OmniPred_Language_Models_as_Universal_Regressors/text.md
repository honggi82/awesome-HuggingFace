[Figure 1]

# OmniPred: Language Models as Universal Regressors

###### Xingyou Song∗1, Oscar Li∗†2, Chansoo Lee1, Bangding (Jeffrey) Yang3, Daiyi Peng1, Sagi Perel1 and Yutian Chen1 1Google DeepMind, 2Carnegie Mellon University, 3Google

∗Equal Contribution. †Work performed as a student researcher at Google DeepMind.

Regression is a powerful tool to accurately predict the outcome metric of a system given a set of parameters, but has traditionally been restricted to methods which are only applicable to a specific task. In this paper, we propose OmniPred, a framework for training language models as universal end-to-end regressors over (𝑥, 𝑦) data from arbitrary formats. Using data sourced from Google Vizier, one of the largest proprietary blackbox optimization databases in the world, our extensive experiments demonstrate that language models are capable of very precise numerical regression using only textual representations of mathematical parameters and values, and if given the opportunity to train at scale over multiple tasks, can significantly outperform traditional regression models.

## arXiv:2402.14547v6[cs.LG]30Jan2025

##### 1. Introduction

Regression is a fundamental task for experimental design, in many domains such as hyperparameter tuning, computer software, industrial engineering, and chemical discovery. The goal of regression is to predict a metric 𝑦 of a general system given a set of input features 𝑥. Such regressors can later be used for various applications, such as offline optimization (Kumar et al., 2022; Trabucco et al., 2022), online optimization (Cai et al., 2020), low-cost benchmarking (Eggensperger et al., 2015; Zela et al., 2022) and simulation (Hashemi et al., 2018; Kaufman et al., 2021; Mendis et al., 2019).

[Figure 2]

[Figure 3]

X: {lr=1e-3, opt=”SGD”} Name: convnet on cifar10 Metric: accuracy

Y = ?

[Figure 4]

Users

Y = 0.9

LM

[Figure 5]

Offline Database

[Figure 6]

X: {tiles=5, windows=10} Name: tpu design Metric: latency

Y = ?

Y = 0.00015

Automated Systems

Y = ?

###### Offline

Train

Test

Figure 1 | Overview of our method. Using heterogenous offline (𝑥, 𝑦) evaluation data collected from a variety of sources, we train a LM-based regressor.

In recent years, large language models (LLMs) have emerged as powerful tools for processing textual representations at scale over massive heterogeneous datasets to represent complex relationships between input features and output labels. Given that LLMs have been shown to be effective for a variety of tasks beyond natural language processing, such as coding (Li et al., 2022), symbolic mathematics (Lewkowycz et al., 2022), and scientific reasoning (Singhal et al., 2022), it is reasonable to wonder: Can language models be used for numeric regression?

© 2025 Google DeepMind. All rights reserved

Answering this question is highly important not only for the traditional field of experimental design, but also for the ever-changing field of LLM research, especially due to recent interest in the ability to forecast outcomes of complex systems (Gruver et al., 2023) and reward modeling in reinforcement learning fine-tuning (Ziegler et al., 2019). The textual processing abilities of LLMs are particularly attractive, as they can potentially bypass the need to tediously featurize inputs (i.e. the 𝑥’s) into raw numerical tensors. Prior to our work, there has been no such research specifically addressing the feasibility and utility of training a “universal” metric predictor over a large heterogeneous offline dataset.

Our core contributions in summary, are as follows:

- • We propose OmniPred, the first scalable yet simple metric prediction framework based on free-form textual representations, applicable to general input spaces.
- • Through only these text and token-based representations, OmniPred is capable of very accurate metric predictions over experimental design data.
- • By leveraging multi-task learning across vastly different input spaces and objectives, in many cases OmniPred can outperform traditional regression models such as MLPs and boosted trees.
- • These transfer learning benefits persist even on unseen tasks after online finetuning OmniPred on small amounts of new evaluation data.

##### 2. Related Work and Motivation

Traditional regression methods have widely used statistical techniques such as Gaussian Processes (GPs), tree-based methods, and multilayer perceptrons (MLPs), to predict a scalar objective given a fixed-length feature vector, commonly seen in tabular data settings. Multitask (Bonilla et al., 2007) and contextual (Krause & Ong, 2011) variants have been further proposed for transfer learning purposes, but still require fixed-length tensor representations of 𝑥, and can thus only use previous 𝑥 from the same input space. Additional recent works utilizing deep learning-based regressors include Transformers (Garg et al., 2022; Hollmann et al., 2023; Huang et al., 2020), recurrent neural networks (Hashemi et al., 2018), graph neural networks (Gao et al., 2023; Lukasik et al., 2020), and deep-hierarchical GPs (Fan et al., 2024), which allow length-independence. Even so, a frequent issue is still the reliance on tensor representations of (𝑥, 𝑦).

Tensor representations are commonly problem-dependent, where each tensor element may need to be in a reasonable numerical range (e.g. in [−1, 1]) as inputs to a model. Thus to represent 𝑥, every categorical feature must be one-hot embedded against user-provided choices, and scalar features may need to be rescaled against user-provided bounds. Dynamic yet minor input space changes such as new bounds or additional categories, are incompatible with this static representation. To represent 𝑦, a raw objective in ℝ may also need to be rescaled, which can be problematic at test-time when encountering outlier 𝑦-values. Dealing with this issue leads to implementing complicated nonlinear warpings (Daimon, 2011; Yeo & Johnson, 2000), many of which are also data-dependent (e.g. require storing min/max values from training data).

|Regressor<br><br>|Dynamic Input Spaces?|Can Multitask?<br><br>|Tensorize 𝑥?<br><br>|Rescale 𝑦?|
|---|---|---|---|---|
|MLP|No|Only fixed spaces<br><br>|Yes|Yes|
|Tree-based<br><br>|No|Only fixed spaces<br><br>|Yes<br><br>|Optional|
|Gaussian Process (GP)|No|Only fixed spaces<br><br>|Yes|Yes|
|GNN / Transformer / RNN|No<br><br>|Only fixed domains<br><br>|Yes|Yes|
|OmniPred (Ours)<br><br>|Yes<br><br>|Yes|No|No|

- Table 1 | Comparisons between the flexibilties of different typical regressors.

These issues are summarized in Table 1. In principle, an ideal regressor should process 𝑥 and output 𝑦, both in absolute terms, independent of changing external statistics or search constraints. For example, if the underlying relationship is 𝑦 = 𝑥2, then the regressor’s prediction at 𝑥 = 2 should be the same regardless if the constraint is 𝑥 ∈ [1, 5] or 𝑥 ∈ [0, 100]. One way to accomplish this is to represent 𝑥 with an independent universal tokenizer without problem-specific numeric transformations (Zhou et al., 2023). This immediately unlocks a large amount of transferrability when dealing with variable-length inputs and additional contextual metadata.

Previous works in the token-based, or effectively text-to-text paradigm, have mostly been in reinforcement learning from human feedback (Ziegler et al., 2019), where regression over textual responses (the “𝑥”), also known as reward modeling, has been crucial to the success of recent interactive LLMs such as ChatGPT (OpenAI, 2022) and Gemini (Google, 2024). While such works have shown that LLMs are able to imitate human ratings in the form of ordinal variables and their softmax probabilities (Bradley & Terry, 1952), they have not shown whether LLMs are capable of regression over precise numeric-based data where 𝑦 ∈ ℝ.

This is because the overwhelming current focus has been on subjective human-based feedback needed for determining aspects such as creativity, safety, and personality, which contain high aleatoric uncertainty and do not require high-precision measurements. Much less attention has been given towards feedback from complex and natural systems common to experimental design. Given multiple works (Hendrycks et al., 2021; Nogueira et al., 2021) demonstrating their brittle and unreliable numeric abilities, it is non-obvious that language models are capable of high-precision numerical prediction over token-based representations. This is a crucial technical challenge which our paper resolves in the quest for a general-purpose predictor.

##### 3. Methodology

###### 3.1. Preliminaries and Problem Definition

Since regression is commonly used in blackbox optimization settings, we adopt standard terminology (Golovin et al., 2017; Liaw et al., 2018) from the field. For a given task T = (X, 𝑓, D, 𝑚), we assume there is a mapping 𝑓 : X → ℝ for which we obtain trials (𝑥, 𝑦) from evaluating an input 𝑥, selected from a (possibly implicit) input space X. We define a study as an offline collection of trials D = {𝑥𝑠, 𝑦𝑠}𝑇𝑠=1. To distinguish between different tasks, there may be observable task-level metadata 𝑚, which can additionally characterize the task and potentially even describes the behavior of the corresponding mapping 𝑓 (·).

The goal of standard regression is to obtain a distribution mapping 𝑠 : X → P(ℝ) such that 𝑠(𝑥) accurately approximates 𝑓 (𝑥) over some distribution of inputs 𝑥 ∈ X, provided that a training set D𝑡𝑟𝑎𝑖𝑛 is given. In our particular case, we also provide our language model with multi-task training data ∪{D1𝑡𝑟𝑎𝑖𝑛, D2𝑡𝑟𝑎𝑖𝑛, ...} from other tasks {T1, T2, ...}. While these extraneous tasks contain different objectives 𝑓1, 𝑓2, . . . and may even have different input spaces from each other, training on such additional extraneous data may still lead to transferrability, especially for similar tasks.

A common way to measure the accuracy of predictors (deterministic or probabilistic) is to compute the gap between a final pointwise prediction against the true objective value 𝑦. For probabilistic regressors 𝑠 : X → P(ℝ), we may aggregate by e.g. taking the median or mean of the distribution. Since different studies can have vastly different objective scales (e.g. CIFAR10 accuracies are within [0, 1] while synthetic objectives are within [102, 109]), we must therefore normalize the difference based on per-study statistics, i.e. for a specific task, we define the study error as a normalized mean

absolute error (MAE):

∑︁

1 𝑦max − 𝑦min

1 |D𝑡𝑒𝑠𝑡|

|Aggregate(𝑠(𝑥)) − 𝑦| (1)

(𝑥,𝑦)∈D𝑡𝑒𝑠𝑡

To prevent outlier predictions from significantly swaying average errors, we further clip error maximums to 1.0, equivalent to when the regressor simply outputs boundary values from {𝑦min, 𝑦max}.

###### 3.2. Language Model

We use a standard language model in which the model observes a prompt and decodes a response. For the regression setting, naturally these correspond to 𝑥 and 𝑦 respectively. However, in order to allow multi-task training, the task-specific metadata 𝑚 must be appended to the prompt in order to distinguish between different tasks, and thus for a given task, the prompt is actually (𝑥, 𝑚).

For simplicity, we train a relatively small 200M parameter T5 encoder-decoder (Raffel et al., 2020) from scratch to avoid any confounding effects from typical generative language pre-training. We wish to learn a single set of weights 𝜃, which can be used to form a stochastic regressor 𝑠𝜃 : X → P(ℝ) given any arbitrary task T. In contrast to settings such as (1) traditional regression requiring training a separate model 𝜃𝑖 for each task T𝑖 or (2) requiring completely evaluated trajectories over specialized 𝑥tokenizations for in-context learning (Chen et al., 2022; Hollmann et al., 2023), our setting maximizes the usage of training data, much of which may contain unfinished trajectories or free-form 𝑥-formats.

Representation: As mentioned, since input spaces X and 𝑦-scales may vary wildly across different tasks, multi-task training requires 𝑥 and 𝑦 to be represented in absolute formats independent of the input space and numeric scaling of the specific study. Thus, we express 𝑥 in a key-value format, directly mapping parameter names to values, but do not represent1 the input space X, allowing generalizability to conditional parameters and dynamic constraints. We represent 𝑦 with custom tokens to guarantee proper decoding via constrained decoding, using specific tokens to express sign, exponent, and significant digits, although more sophisticated alternatives (Koo et al., 2024) are available. An example is illustrated in Table 2. Ablations over different tokenizations are conducted in Appendix A.1.

| |Language Model Textual Representation|
|---|---|
|𝑥<br><br>|batch_size:128,kernel:‘rbf’,learning_rate:0.5,model:‘svm’,optimizer:‘sgd’|
|𝑚<br><br>|title:‘classification’,user:‘some-person’,description:‘spam detection’, objective:‘accuracy (%)’|
|𝑦<br><br>|<+><7><2><5><E-1>|

- Table 2 | Textual representations used for OmniPred. <∗> represents a single custom token. Input space and 𝑥 is the same as in Figure 2. Example 𝑦-tokenization represents a value of 725 × 10−1 = 72.5.

Training: We apply standard Prefix-LM training (Raffel et al., 2020), in which for a given (prompt, response) pair, cross-entropy losses are only computed over response tokens. Pairs are sampled from training data over multiple tasks. One could additionally make the loss more metric-aware by weighting specific tokens or even reinforce with non-differentiable scores, although we maintain simplicity in this paper for now by using uniform cross-entropy. Thus the model will implicitly learn numeric distances from training data.

Sampling and Decoding: Through regular temperature decoding, we can repeatedly sample 𝑦 ∼ 𝑠𝜃(𝑥), to approximate the prediction distribution over ℝ. To remain robust to strong outliers, instead of

1In applications such as code search, it is even infeasible to express the space of all possible programs.

using empirical mean, our Aggregate(·) function is defined as the empirical median, since we found it leads to lower error from ablations in Section 6.1. Since the model may need to predict over unseen regions of the input space, we can also assess the model’s uncertainty by observing the concentration of sampled predictions 𝑦 and additionally specific log probabilities across every decoded token.

Online Finetuning: To adapt to an unseen task T𝑢 common for Bayesian optimization settings (Wistuba & Grabocka, 2021), the model can further be quickly finetuned online over the tasks’s

corresponding training data D𝑢𝑡𝑟𝑎𝑖𝑛, optionally using LoRA (Hu et al., 2022). Finetuning may also help to refocus over seen data, when the model is not fully optimized against a specific study, e.g. if the pretraining dataset was too large.

Appendix B contains additional implementation details such as the specific architecture and vocabulary used.

##### 4. Data

Many industries possess large collections of metric data from experiments or systems. However, such data is typically not open-sourced as official training datasets for research. A natural dataset to use may come from multiple hyperparameter optimization trajectories, which tend to have (𝑥, 𝑦) evaluations from expensive experiments, expressed as blackbox objectives 𝑦 = 𝑓 (𝑥).

###### 4.1. OSS Vizier Format

The abstractions in Section 3.1 above are concretely implemented in Open Source (OSS) Vizier (Song et al., 2022), a research interface for blackbox and hyperparameter optimization. Every space X is a Cartesian product of parameters, each of type DOUBLE (bounded continuous interval), INTEGER (bounded integer set), DISCRETE (ordered set of real numbers), or CATEGORICAL (unordered set of strings). Every parameter may also have child parameters, only active when the corresponding parent parameter is a specific value (e.g. in Figure 2, “beta” is active only if a parent categorical parameter selects “Adam”, but not “SGD”). An 𝑥 ∈ X can be expressed as a tabular feature, allowing baseline comparisons against traditional regression models.

###### Search Space

|Name: model CATEGORIES: [‘svm’, ‘mlp’]|
|---|

|Name: optimizer CATEGORIES: [‘sgd’, ‘adam’]| |
|---|---|
| | |

|Name: learning_rate DOUBLE: [0, 1.0]|
|---|

model = ‘svm’ model = ‘mlp’

optimizer = ‘adam’

|Name: batch_size INT: [1, 256]|
|---|

|Name: beta DOUBLE: (0.0, 1.0]|
|---|

|Name: kernel CATEGORIES: [‘rbf’, ‘poly’]|
|---|

|Name: num_layers INT: [1, 5]|
|---|

|learning_rate:0.5|batch_size:128|model:‘svm’, kernel:‘rbf’|optimizer:‘sgd’|
|---|---|---|---|

###### Trial 1:

###### Trial 2:

|learning_rate:0.2|
|---|

|batch_size:14|
|---|

|model:‘mlp’, num_layers:2|
|---|

|optimizer:‘adam’, beta:0.6|
|---|

- Figure 2 | Common example of a (possibly nested) space and suggestions 𝑥 in OSS Vizier.

Task-level metadata 𝑚 consists of a title, username, description, objective name, and optional freeform text. Since the OSS Vizier API is meant to provide an optimization service for users, there can be many sources of transferrability due to user-specific settings. These include:

- • A single user or team regularly tuning similar experiments.
- • Multiple different users tuning similar experiments (e.g. training ResNets on CIFAR10).
- • Similar parameters used across different experiments (e.g. “learning rate”).
- • Metadata 𝑚 describing the nature of the objective function.

###### 4.2. Datasets

BBOB (Shifted): For precise controlled experiments where we can generate synthetic datasets and perform online evaluations, we create a multi-task version of the BBOB benchmark (ElHara et al., 2019) containing 24 different synthetic functions, by applying random domain shifts 𝑐 to transform a vanilla synthetic 𝑓 (𝑥) into 𝑓 (𝑥 − 𝑐), and ranging the dimension over [2, 6]. Thus each task T is parameterized by 𝑚 = (function class, dimension, shift), and the corresponding objective can be seen as 𝑓 (𝑥, 𝑚), allowing evaluation over unseen 𝑚. For a specific task T𝑖, we minimize the in-study training data size D𝑖𝑡𝑟𝑎𝑖𝑛 but freely vary inter-study training data {D𝑡𝑟𝑎𝑖𝑛𝑗 }𝑗≠𝑖 from different tasks {T𝑗}≠𝑖. Thus traditional regressors (e.g. MLPs) which can only train from a single D𝑖𝑡𝑟𝑎𝑖𝑛 will struggle to regress the corresponding 𝑓𝑖 under this limited data condition. In contrast, the LM may perform better as it will have seen trials from other tasks whose functions share similarities with 𝑓𝑖.

Real World Data: To investigate metric prediction over real world data which contain a rich variety of tasks, we will use data collected by Google Vizier (Golovin et al., 2017). Because we are not limited to training on fully completed trajectories over flat input spaces, we can train on more data than just the 750K studies used in the OptFormer (Chen et al., 2022), as seen from Table 3.

|Property<br><br>|Statistic|
|---|---|
|# Studies # Trials # Distinct Users<br><br>|O(70M+) O(120B+) O(14K)|

Table 3 | Relevant statistics on the real world database. We provide order estimates as there may be numerous ways to define e.g. “legitimate” studies or trials. See Appendix D for further details and data breakdown.

Since the offline dataset is collected from users’ blackbox optimization trajectories, we for the most part do not have online access to an actual objective 𝑓 (𝑥), rather only data samples D, and thus we must evaluate our predictor’s accuracy via a test set D𝑡𝑒𝑠𝑡 ⊂ D. Thus, we need to take into account how much D𝑡𝑟𝑎𝑖𝑛 sufficiently covers the space X, which affects the difficulty of achieving high accuracy on the task. Influencing factors include:

- • Trial count: Users can decide when to stop tuning, and thus the size of a study can be on the order of 100 to 105.
- • Diversity of trials: By default, a study’s trials {(𝑥1, 𝑦1), ..., (𝑥𝑇, 𝑦𝑇)} form the trajectory of an optimization loop, and thus later trials may converge towards a single local optimum.
- • Space size: Approximate cardinality of a space X is exponential with respect to parameter count, and thus large input spaces will naturally be less explored.

While we apply practical processing steps such as (1) setting a maximum initial trial limit per study and (2) randomly shuffling the trials and then (3) deciding on a fixed train/validation/test splitting ratio (default 0.8/0.1/0.1), we cannot fully control whether each D saturates its space X, or essentially how “easy” the task is. Instead, we use a baseline regressor trained only on D𝑡𝑟𝑎𝑖𝑛 and evaluated on corresponding D𝑡𝑒𝑠𝑡 as a proxy metric of the difficulty of the task.

- 5. Experiments We answer the following key questions:

- 1. Can we simultaneously regress on multiple tasks of different input spaces and objective scales?
- 2. Are there benefits to multi-task training and are textual signals useful for transfer learning?
- 3. Does online finetuning improve accuracy over unseen studies outside of the pretraining set?

###### 5.1. Simultaneous Regression

- In Figure 3, we visually present how a BBOB-trained model captures the overall shape of analytical functions of vastly different objective scales with high precision. Furthermore, the model is capable of expressing uncertainty estimates via i.i.d. prediction samples.

4 2 0 2 4

500

510

520

530

540

StepEllipsoidal-4D Shifts: [-0.50,1.04,-1.79,3.38] x1=-3.78, x2=3.29, x3=-1.29

4 2 0 2 4

1440

1460

1480

StepEllipsoidal-4D Shifts: [-0.50,1.04,-1.79,3.38] x0=3.65, x2=2.57, x3=0.30

2 0 2 4

100

200

300

StepEllipsoidal-4D Shifts: [-0.50,1.04,-1.79,3.38] x0=-0.04, x1=-0.99, x3=-4.18

4 2 0

0

1000

2000

StepEllipsoidal-4D Shifts: [-0.50,1.04,-1.79,3.38] x0=-1.16, x1=2.87, x2=3.00

0 2 4

50

100

Weierstrass-4D Shifts: [-3.79,2.68,-3.78,-1.62] x1=0.12, x2=4.11, x3=0.22

4 2 0 2

50

100

Weierstrass-4D Shifts: [-3.79,2.68,-3.78,-1.62] x0=4.02, x2=4.38, x3=2.48

0 2 4

100

200

Weierstrass-4D Shifts: [-3.79,2.68,-3.78,-1.62] x0=-0.68, x1=-0.69, x3=4.20

2 0 2 4

0

25

50

75

Weierstrass-4D Shifts: [-3.79,2.68,-3.78,-1.62] x0=2.24, x1=1.61, x2=4.11

4 2 0

x0

25000

50000

75000

RosenbrockRotated-4D Shifts: [3.54,3.74,2.63,2.57] x1=-1.76, x2=-0.40, x3=-1.61

4 2 0

x1

25000

50000

75000

RosenbrockRotated-4D Shifts: [3.54,3.74,2.63,2.57] x0=-0.92, x2=-1.58, x3=0.79

4 2 0 2

x2

0

25000

50000

75000

RosenbrockRotated-4D Shifts: [3.54,3.74,2.63,2.57] x0=-0.74, x1=-1.29, x3=1.13

4 2 0 2

x3

90000

100000

110000

120000

RosenbrockRotated-4D Shifts: [3.54,3.74,2.63,2.57] x0=0.45, x1=0.77, x2=1.59

Target

Prediction

ObjectiveValue()y

Figure 3 | Model prediction samples over selected 4D BBOB functions with unseen shifts. Empirical mode (bolded) and min/max are shown from 10 samples. Over all BBOB functions, we vary the coordinate value 𝑥𝑖 while keeping others 𝑥𝑗≠𝑖 fixed.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

2.5 2.0 1.5 1.0 0.5 0.0

2.5

2.0

1.5

1.0

0.5

0.0 CIFAR10 (Valid. Loss)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

8 7 6 5 4

8

7

6

5

4

LM1B LM (Valid. Loss)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | |
|---|---|---|
| | | |
| | | |

2.5 2.0 1.5 1.0 0.5

2.5

2.0

1.5

1.0

0.5

Bid Simulation

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

30 40 50 60 70 80

30

40

50

60

70

80 Protein Design

4.60 4.55 4.50 4.45

4.60

4.55

4.50

4.45 LLM (XLA Latency)

0.2 0.4 0.6 0.8 1.0

0.2

0.4

0.6

0.8

1.0

AutoML (Valid. AUC)

0.0000 0.0005 0.0010

0.0000

0.0005

0.0010 MobileNet (TPU Cost)

0.60 0.62 0.64 0.66 0.68

0.60

0.62

0.64

0.66

0.68

Spam Filter (Valid. Acc.)

Target

Prediction

|Name|Space<br><br>|
|---|---|
|CIFAR10 LM1B LM Bid Simulation Protein Design LLM Latency AutoML MobileNet Spam Filter|4 Double 4 Double 4 Double 60 Categories 31 Hybrid 3-H, 42-T 10 Discrete 13-H, 15-T<br><br>|

Figure 4 | Left: Diagonal fit (/) is better. Model’s 𝑦-prediction vs. ground truth over varying studies. Corporatespecific objective names are redacted. Right: Corresponding input spaces. “#-H, $-T” is shorthand for a conditional hybrid input space with # root parameters and $ total possible parameters.

- In Figure 4 for a model trained over real world data, we present an analogous visualization over handselected studies with drastically different input spaces, representative of objectives tuned commonly in real world settings. These include standard machine learning (e.g. image classification and language

modeling), production systems (e.g. bid simulation, LLM inference latency), and scientific research (e.g. protein and hardware design).

###### 5.2. Multi-task Transferrability

In this subsection, we demonstrate the model’s ability to transfer learn, i.e. improve accuracy over a specific task using knowledge gained from other similar but non-equivalent tasks, in contrast to “single-task” regressors (described in Appendix C) which only observe training data from the task being evaluated. Note that single-task baselines such as MLPs are incapable of multi-task training when tasks have different dimensionalities.

- In Figure 5, we clearly see that the model’s accuracy improves with more tasks seen in training and eventually outperforms all traditional baselines. For AutoML studies, the error is averaged from a fixed subset of encountered studies. For BBOB, we can further demonstrate the model’s inter-study generalization capabilities over metadata 𝑚 (as opposed to 𝑥) by evaluating on unseen tasks with new shifts not encountered during training.

| |
|---|
| |
| |
| |
|GP Random Forest Tree MLP LM<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|

103 104 105

# of Training Studies

0.15

0.20

0.25

0.30

0.35

MeanStudyError

AutoML Eval

| |
|---|
| |
| |
|GP Random Forest Tree MLP LM<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|

103 104 105

# of Training Studies

0.025

0.050

0.075

0.100

0.125

0.150

0.175

0.200

MeanStudyError

BBOB Eval (Unseen Shifts)

Figure 5 | Lower (↓) is better. Mean study prediction error of the model when varying the amount of different studies used in training (log scale). Colored horizontal lines display single-task baseline errors.

|Datasets (# Training Studies)<br><br>|Mean Study Error (↓) Original Anonymized|
|---|---|
|BBOB (50K) BBOB (Full 1M) AutoML (26.3K) AutoML (Full 540K)|0.03 0.46 0.01 FAIL 0.19 0.44 0.15 0.43<br><br>|

Table 4 | Lower (↓) is better. Comparisons between models trained on original vs anonymized data, across BBOB-Shifted and AutoML test trials. “FAIL” means the model failed to even train.

To verify whether the model is performing transfer learning by reading textual cues, in Table 4 we compare results against the case when data is “anonymized” using a study-dependent hash function. For BBOB, we hash metadata 𝑚 which originally displayed (function class, dimension, shift). For AutoML, we hash parameter names and string values. Each study can still be uniquely identified and trained on, but the model can no longer observe useful correlations from common textual clues. Interestingly, the model fails to train over the full anonymized BBOB dataset, a case when the data is too large and heterogeneous.

- In Figure 6, we further see that for the model, multi-task training consistently improves over single-task training, and in regimes with relatively lower input space saturation (i.e. low trial to parameter count ratio) from training data, multi-task models outperform traditional baselines over several different domains. Interestingly, a single-task model trained from scratch remains a competitive choice and for

certain domains such as AutoML, can even outperform all other single-task baselines. We hypothesize this is due to language-based representations being more appropriate for the conditional structures of these domains (e.g. AutoML).

###### Multi-task LM vs Single-task Baselines

| |GP Random Forest Tree MLP<br><br>| |
|---|
<br><br>| |
|---|
| |
|---|---|---|
| | | |

- 0.0
- 0.1
- 0.2
- 0.3
- 0.4
- 0.5

MeanStudyError

Single-task LM Multi-task LM

BBOB Bid Simulation AutoML Init2Winit Protein Design V-AI (Tab) V-AI (Text)

Data Type

|Name<br><br>|# Studies|Avg. TpS<br><br>|Avg. SS|
|---|---|---|---|
|BBOB Bid Simulation AutoML Init2winit Protein Design Vertex AI (Tabular) Vertex AI (Text)|1M 22K 540K 2K 54K 1.4M 544K<br><br>|30 698 250 176 584 88 118<br><br>|4.0 4.6<br><br>(3.3, 29.9) 3.6<br><br>125.6<br><br>(4.6, 42.4) 56.0<br>|

- Figure 6 | Left: Lower (↓) is better. Aggregate error across different domains. Right: Statistics on domains. Shorthand notation: “TpS” = Trials per Study, “SS” = Space Size, with brackets (#, $) denoting conditional space with # root parameters and $ total possible parameters. Note that all baselines are single-task regressors.

- 5.3. Online Finetuning Analysis

|Pretraining Dataset<br><br>|Mean Study Error (↓) on AutoML Before Finetuning After Finetuning|
|---|---|
|None (Single-Task) BBOB AutoML All RealWorldData<br><br>|0.98 0.20 0.98 0.45 0.15 0.15 0.31 0.15|

Table 5 | Lower (↓) is better. Mean study errors of pretrained models and their corresponding finetuned versions.

We first examine the conditions in which finetuning may be beneficial. In Table 5, we finetune various pretrained models over AutoML studies. While there is negligible benefit in finetuning the AutoML model on its data again, we see that a model pretrained over the entire real world dataset is able to finetune to the same level of accuracy as a pretrained AutoML model, while a BBOB-pretrained model leads to significantly worse results than even a single-task model. This suggests that knowledge obtained from pretraining can have a large (positive or negative) influence on transferrability over specific domains such as AutoML.

| |Single-t Pretrain|ask<br><br>+ Finetune|
|---|---|---|
| | | |

0 20 40 Unique User Study (Sorted by Improvement)

0.0

0.2

0.4

0.6

0.8

1.0

StudyError

Finetuning Comparison (Unseen RealWorld Studies)

|Method|Mean Study Err. (↓)<br><br>|
|---|---|
|Single-task (LM) Pretrain (LM) Pretrain + Finetune (LM)|0.28 0.68 0.21<br><br>|
|MLP (Baseline) Tree (Baseline) Random Forest (Baseline) Gaussian Process (Baseline)|0.25 0.32 0.32 0.42<br><br>|

- Figure 7 | Left: Lower (↓) is better. Example LM study errors over unseen studies filtered over random distinct users. Right: Aggregate comparisons across different methods over 1000 unseen studies.

We further examine this effect by evaluating over unseen tasks, i.e. those which were newly created after the original training set was scraped, and can contain studies from new users and objectives. In

- Figure 7, we compare initialization from scratch (leading to single-task training) against a pretrained model on older real world data. We see that knowledge obtained from pretraining can significantly transfer over and help predictions over new tasks, although as seen on the left with three studies, there are few cases of negative transfer.

6. Ablations

We further ablate certain important settings and scenarios which affect the model’s prediction accuracy below. Appendix A contains additional ablations, specifically (1) around our choice of 𝑦-tokenization and (2) experimental outcomes when using ranking-based regression metrics.

6.1. Effect of Sampling

The LM can output extreme outliers in its 𝑦-prediction, usually due to an inaccurate prediction on the exponent token or significant digits. While such issues do not occur once the model has nearly perfectly regressed on a task (e.g. BBOB), they occur frequently on realistic tasks with nontrivial error (e.g. AutoML) and thus require techniques for correction. One obvious method to reduce error is to increase sample count, as demonstrated in Figure 8.

| |
|---|
|MLP<br><br>LM|

1 2 4 8 16 32 64

# of Samples

0.00

0.05

0.10

0.15

0.20

0.25

0.30

MeanStudyError

AutoML Eval

| |
|---|
|MLP<br><br>LM|

1 2 4 8 16 32 64

# of Samples

0.00

0.02

0.04

0.06

0.08

0.10

MeanStudyError

BBOB Eval

- Figure 8 | Lower (↓) is better. Mean study error when varying the samples used during inference (log scale).

An additional method is to change the method of aggregation across samples. In Table 6, we see that using the median considerably outperforms both max-likelihood and mean. We hypothesize this is due to the median’s robustness to hallucinated outlier samples which can occur with relatively high probability and can also skew the mean.

|Empirical Aggregation Method|Mean Study Error (↓) AutoML (Full 540K) BBOB (Full 1M)<br><br>|
|---|---|
|Median (default) Max-likelihood Mean<br><br>|0.15 0.01<br><br>0.22 0.01<br>0.23 0.01<br>|

- Table 6 | Lower (↓) is better. Comparisons between different aggregation methods when using 64 samples when using full datasets for pretraining.

###### 6.2. Uncertainty Calibration

Although the main metric used throughout our work is based on pointwise prediction, an important ability for regressors is to express uncertainty when they are unable to provide an accurate prediction. This is particularly useful in applications such as Bayesian optimization, where uncertainty can be used as an exploration proxy. In this section, we examine whether the model can quantify uncertainty even if we did not calibrate or tune any of the models for such purposes.

StepEllipsoidal-4D Shifts: [-0.50,1.04,-1.79,3.38] x1=-0.79, x2=0.47, x3=-4.18

StepEllipsoidal-4D Shifts: [-0.50,1.04,-1.79,3.38] x0=-1.16, x2=4.00, x3=0.00

StepEllipsoidal-4D Shifts: [-0.50,1.04,-1.79,3.38] x0=-1.33, x1=-0.73, x3=-4.22

StepEllipsoidal-4D Shifts: [-0.50,1.04,-1.79,3.38] x0=3.15, x1=-0.84, x2=0.53

1000

2000

200

50

0

0

0

0

50

200

1000

2000

4 2 0 2 4

4 2 0 2 4

2 0 2 4

4 2 0

DifferentPowers-4D Shifts: [1.88,-0.94,1.74,-3.89] x1=0.19, x2=-1.86, x3=2.87

DifferentPowers-4D Shifts: [1.88,-0.94,1.74,-3.89] x0=-2.48, x2=-2.07, x3=-1.04

DifferentPowers-4D Shifts: [1.88,-0.94,1.74,-3.89] x0=0.68, x1=3.71, x3=1.52

DifferentPowers-4D Shifts: [1.88,-0.94,1.74,-3.89] x0=1.84, x1=4.10, x2=0.78

ObjectiveValue()y

200

10

100

100

100

0

0

0

0

10

100

100

100

20

200

4 2 0 2

4 2 0 2 4

4 2 0 2

1 0 1 2 3 4 5

Weierstrass-4D Shifts: [-3.79,2.68,-3.78,-1.62] x1=-4.38, x2=2.44, x3=4.20

Weierstrass-4D Shifts: [-3.79,2.68,-3.78,-1.62] x0=2.24, x2=4.40, x3=3.80

Weierstrass-4D Shifts: [-3.79,2.68,-3.78,-1.62] x0=0.24, x1=-4.55, x3=4.87

Weierstrass-4D Shifts: [-3.79,2.68,-3.78,-1.62] x0=2.65, x1=-0.94, x2=1.54

200

200

200

100

100

100

0

0

0

0

100

100

100

200

200

200

1 0 1 2 3 4 5

4 2 0 2

1 0 1 2 3 4 5

2 0 2 4

Figure 9 | Setting similar to Figure 3, but with bimodality.

We begin by demonstrating the LM’s ability to nonparametrically express distributions with multiple modes in Figure 9 when we train the model against randomly sign-flipped BBOB objectives. In contrast, traditional methods such as ensembled MLPs (Havasi et al., 2021) and Gaussian Process mixtures (Bonilla et al., 2007) must specify the mixture count a priori.

Pearson, Kendall-Tau, Spearman Regressor Uncertainty Metric AutoML BBOB

Gaussian Process Predicted SD 0.254, 0.230, 0.307 0.018, 0.068, 0.048 LM w/ mean aggregation Sample SD 0.560, 0.487, 0.625 0.360, 0.366, 0.454

LM w/ median aggregation Harrell-Davis SE 0.525, 0.412, 0.539 0.360, 0.293, 0.380

Table 7 | Higher (↑) is better. Rank correlation between quantified uncertainty (SD = standard deviation, SE = standard error) and actual error over studies with at least 10 test trials (all BBOB studies and 641 AutoML studies)

Furthermore, we measured the correlation between uncertainty and error on each study. In Table 7, we report the average correlation across studies. Interestingly, although Table 6 demonstrated that mean aggregation over LM samples is worse for prediction than median aggregation, the errors are well correlated with the standard deviation of the samples.

###### 6.3. Study Size vs. Multi-task Gain

Intuitively, as a task’s space becomes more saturated with trials, single-task training becomes more sufficient for accurate prediction. In Figure 10, we plot the gains from multi-task LM training over the

single-task MLP baseline to validate this hypothesis. Gains are maximized at roughly ≈50 training trials and diminish as the number of training trials increases. Note that maximal gains do not occur with ≈0 trials, as presumably some training trials are still needed to identify the structure of a task.

#### AutoML (MLP vs Multi-Task LM)

1.00

(MLPMultitaskLM)Error

0.75

0.50

0.25

0.00

0.25

50-percentile 75-percentile 90-percentile

0.50

0.75

0 100 200 300 400 500 600 700 800

# Training Trials

- Figure 10 | Higher (↑) is better. Study error differences between MLP and multi-task LM over individual AutoML tasks. Percentiles are computed after binning the x-axis appropriately.

##### 7. Discussion: Limitations and Extensions

In this work, our emphasis was to demonstrate the promise of applying language modeling to generalpurpose regression, and thus our design choices remained relatively simple to avoid confounding factors. We list some limitations of our specific design, which opens many more potential areas of exploration.

In-Context Learning (ICL): By design, we maximized the allowed prompt length to allow flexibility in representing (𝑥, 𝑚) and thus did not use in-context regression, since a single (𝑥, 𝑚) prompt could be 10K+ tokens long for some applications. While online finetuning in principle allows infinite amounts of data to be absorbed at inference time, it is worth investigating ICL methods which allow arbitrarily long prompts as well. Current methods (Chen et al., 2022) which require significant input compression are only applicable to tabular-like 𝑥’s. Additional use of ChatGPT and other service-based chat APIs (Vacareanu et al., 2024) rely on regression as an emergent property of expensive LLM-based training, making their serious use difficult, especially as such services cannot absorb large amount of user-defined offline (𝑥, 𝑦) data which often contain the most relevant information for finetuning.

Hallucinations: By giving the model the freedom to sample 𝑦-values over approximately all of ℝ, wildly inaccurate outlier predictions are now possible. This can be exacerbated by a wrong prediction over a significant float token (e.g. leading digit or exponent). Although for convenience, we used an unweighted cross-entropy loss in which all float tokens are of equal importance, prediction accuracy can be improved by weighting more significant tokens, making the training loss more aware of numerical distances over ℝ.

Prompt-Side Numeric Tokenization: In this work, we directly represented numeric parameter values from 𝑥 into the default human readable format (e.g. 1234.5 is serialized simply to ‘1234.5’) to be consistent with LLM literature. This may be suboptimal, as the corresponding tokens may not exactly be digit-by-digit (e.g. T5’s tokenization leads to tokens {‘12’, ‘3’, ‘4.5’}). One may instead potentially reuse the custom tokenization for 𝑦-values (e.g. <+><1><2><3><4><E0>) or in text-space, represent using other serializations which emphasize digits atomically, e.g. ‘[1 10e2 2

10e1 3 10e0 4 10e-1]’) as in (Nogueira et al., 2021).

Pretrained Language Encoder: Since 𝑥 includes parameter names and metadata which contain English words, warm-starting from a model pretrained on English text may improve accuracy. However, most checkpoints comparable to our model’s size (<1B params) are not pretrained over experimental data and are unlikely to understand the numerical meaning of e.g. ‘learning_rate’. Furthermore, using a pretrained English model introduces numerous confounding technical choices to consider (e.g. whether to freeze the encoder, tune the learning rate, embed additional custom float tokens, and use more English-based representations of 𝑥 and 𝑚), but this topic is worth pursuing in the future. In this work, it is already surprising that a relatively small model trained from scratch can still achieve regression, thus suggesting our technique’s broad applicability even without English understanding.

Computational Costs: Compared to traditional baselines, a language model requires accelerator usage and has a relatively higher computational cost for both training and finetuning, in addition to higher inference times. In this work, we purposely designed the model to minimize costs by using ≈200M params which only requires at most 8 GPUs for training and 1 GPU for inference (see Appendix

- B).

Other Input Spaces: The OSS Vizier API primarily focuses on hyperparameter tuning spaces. Traditionally, more complex spaces such as combinatorics and graphs require sophisticated modeling techniques to form regressors, largely in part to difficulties in representing the 𝑥’s as tensors. In addition, many applications with non-expressible spaces such program synthesis are impossible to traditionally regress over. We believe that text and token-based representations are highly promising and widely applicable to domains previously unexplored in the field of experimental design.

Other Metadata: While we performed ablations which anonymized 𝑚 and parameter names, one can further investigate which types of metadata are particularly useful for prediction. Such metadata could contain proxy metrics introduced by previous domain-specific works, such as Jacobian Covariance for neural architecture search (Mellor et al., 2021) and neural-network norms (Jiang et al., 2020) for the study of generalization. The relevant code implementing machine learning or programming tasks may be especially important.

##### 8. Impact Statement

This research addresses the ability to regress metrics against textual data. Since any textual metadata may be collected, user-specific information may be used, which raises privacy concerns. This would be particularly true on sensitive topics (e.g. predicting metrics related to personal protected characteristics).

Our research does not involve such sensitive topics for regression, as it is performed over blackbox optimization data. We followed ethical guidelines as all users in our proprietary dataset have consented to have their tuning data saved and used for offline analysis. The proprietary real world dataset does not contain any sensitive personal information other than usernames. Furthermore, we do not release any of the trained checkpoints, as it may be possible to reverse-engineer parts of the training data, which can lead to privacy violations and data leakage.

##### 9. Conclusion

Our OmniPred framework is a first step towards a universal regressor, capable of performing highprecision predictions over objectives of any scale from vastly different input spaces and applications. Its simple and scalable design allows transfer learning from large amounts of offline diverse evaluations,

while its single-task variant can still perform competitively against a wide variety of gold-standard baselines. Furthermore, it is capable of adapting to unseen data through finetuning, while still transferring knowledge from previous data. This research lays the groundwork for exciting new potential expansions in the field of experimental design.

##### Acknowledgements

We would like to thank Olivier Bachem, Hado van Hasselt, John Jumper, Aviral Kumar, Yingjie Miao, Sebastian Nowozin, Mangpo Phothilimthana, Zi Wang, Scott Yak, and Amir Yazdanbakhsh for useful discussions and Daniel Golovin for continuing support.

##### References

Edwin V Bonilla, Kian Chai, and Christopher Williams. Multi-task gaussian process prediction. In Advances in Neural Information Processing Systems, volume 20, 2007.

Ralph Allan Bradley and Milton E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952. ISSN 00063444.

Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. Once-for-all: Train one network and specialize it for efficient deployment. In International Conference on Learning Representations, ICLR, 2020.

François Charton. Linear algebra with transformers. Trans. Mach. Learn. Res., 2022, 2022.

Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In Balaji Krishnapuram, Mohak Shah, Alexander J. Smola, Charu C. Aggarwal, Dou Shen, and Rajeev Rastogi (eds.), Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794. ACM, 2016.

Yutian Chen, Xingyou Song, Chansoo Lee, Zi Wang, Richard Zhang, David Dohan, Kazuya Kawakami, Greg Kochanski, Arnaud Doucet, Marc’Aurelio Ranzato, Sagi Perel, and Nando de Freitas. Towards learning universal hyperparameter optimizers with transformers. In NeurIPS, 2022.

Takashi Daimon. Box-cox transformation. In Miodrag Lovric (ed.), International Encyclopedia of Statistical Science, pp. 176–178. Springer, 2011.

Stéphane d’Ascoli, Pierre-Alexandre Kamienny, Guillaume Lample, and François Charton. Deep symbolic regression for recurrent sequences. CoRR, abs/2201.04600, 2022.

Katharina Eggensperger, Frank Hutter, Holger H. Hoos, and Kevin Leyton-Brown. Efficient benchmarking of hyperparameter optimizers via surrogates. In Blai Bonet and Sven Koenig (eds.), Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, pp. 1114–1120. AAAI Press, 2015.

Ouassim Ait ElHara, Konstantinos Varelas, Duc Manh Nguyen, Tea Tusar, Dimo Brockhoff, Nikolaus Hansen, and Anne Auger. COCO: The large scale black-box optimization benchmarking (bboblargescale) test suite. ArXiv, abs/1903.06396, 2019.

Zhou Fan, Xinran Han, and Zi Wang. Transfer learning for bayesian optimization on heterogeneous search spaces. Transactions on Machine Learning Research, 2024. ISSN 2835-8856.

Yanjie Gao, Xianyu Gu, Hongyu Zhang, Haoxiang Lin, and Mao Yang. Runtime performance prediction for deep learning models with graph neural network. In 45th IEEE/ACM International Conference on Software Engineering: Software Engineering in Practice, SEIP@ICSE 2023, Melbourne, Australia, May 14-20, 2023, pp. 368–380. IEEE, 2023.

Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. What can transformers learn in-context? A case study of simple function classes. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), Neural Information Processing Systems, 2022.

Daniel Golovin, Benjamin Solnik, Subhodeep Moitra, Greg Kochanski, John Karro, and D. Sculley. Google vizier: A service for black-box optimization. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1487–1495. ACM, 2017.

Google. Gemini: A family of highly capable multimodal models, 2024.

Nate Gruver, Marc Finzi, Shikai Qiu, and Andrew Gordon Wilson. Large language models are zero-shot time series forecasters. CoRR, abs/2310.07820, 2023.

Milad Hashemi, Kevin Swersky, Jamie A. Smith, Grant Ayers, Heiner Litz, Jichuan Chang, Christos Kozyrakis, and Parthasarathy Ranganathan. Learning memory access patterns. In Jennifer G. Dy and Andreas Krause (eds.), International Conference on Machine Learning, ICML 2018, volume 80 of Proceedings of Machine Learning Research, pp. 1924–1933. PMLR, 2018.

Marton Havasi, Rodolphe Jenatton, Stanislav Fort, Jeremiah Zhe Liu, Jasper Snoek, Balaji Lakshminarayanan, Andrew Mingbo Dai, and Dustin Tran. Training independent subnetworks for robust prediction. In International Conference on Learning Representations, ICLR, 2021.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Joaquin Vanschoren and Sai-Kit Yeung (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021.

Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. Tabpfn: A transformer that solves small tabular classification problems in a second. In International Conference on Learning Representations, ICLR, 2023.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, ICLR, 2022.

Xin Huang, Ashish Khetan, Milan Cvitkovic, and Zohar S. Karnin. Tabtransformer: Tabular data modeling using contextual embeddings. CoRR, abs/2012.06678, 2020.

Yiding Jiang, Behnam Neyshabur, Hossein Mobahi, Dilip Krishnan, and Samy Bengio. Fantastic generalization measures and where to find them. In International Conference on Learning Representations, ICLR, 2020.

Samuel J. Kaufman, Phitchaya Mangpo Phothilimthana, Yanqi Zhou, Charith Mendis, Sudip Roy, Amit Sabne, and Mike Burrows. A learned performance model for tensor processing units. In Alex Smola, Alex Dimakis, and Ion Stoica (eds.), Proceedings of Machine Learning and Systems (MLSys). mlsys.org, 2021.

Terry Koo, Frederick Liu, and Luheng He. Automata-based constraints for language model decoding, 2024.

Andreas Krause and Cheng Ong. Contextual gaussian process bandit optimization. In Advances in Neural Information Processing Systems, 2011.

Taku Kudo and John Richardson. Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 66–71, 2018.

Aviral Kumar, Amir Yazdanbakhsh, Milad Hashemi, Kevin Swersky, and Sergey Levine. Data-driven offline optimization for architecting hardware accelerators. In International Conference on Learning Representations, ICLR, 2022.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay V. Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language

models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, 2022.

Yujia Li, David H. Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d’Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level code generation with alphacode. CoRR, abs/2203.07814, 2022.

Richard Liaw, Eric Liang, Robert Nishihara, Philipp Moritz, Joseph E. Gonzalez, and Ion Stoica. Tune: A research platform for distributed model selection and training. CoRR, abs/1807.05118, 2018.

Jovita Lukasik, David Friede, Heiner Stuckenschmidt, and Margret Keuper. Neural architecture performance prediction using graph neural networks. In Zeynep Akata, Andreas Geiger, and Torsten Sattler (eds.), Pattern Recognition - 42nd DAGM German Conference, DAGM GCPR, volume 12544 of Lecture Notes in Computer Science, pp. 188–201. Springer, 2020.

Joe Mellor, Jack Turner, Amos J. Storkey, and Elliot J. Crowley. Neural architecture search without training. In Marina Meila and Tong Zhang (eds.), International Conference on Machine Learning, ICML, volume 139, pp. 7588–7598. PMLR, 2021.

Charith Mendis, Alex Renda, Saman P. Amarasinghe, and Michael Carbin. Ithemal: Accurate, portable and fast basic block throughput estimation using deep neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), International Conference on Machine Learning, ICML, volume 97 of Proceedings of Machine Learning Research, pp. 4505–4515. PMLR, 2019.

Rodrigo Frassetto Nogueira, Zhiying Jiang, and Jimmy Lin. Investigating the limitations of the

transformers with simple arithmetic tasks. CoRR, abs/2102.13019, 2021. OpenAI. Introducing chatgpt. 2022. Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi

Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21:140:1–140:67, 2020.

Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Kumar Tanwani, Heather Cole-Lewis, Stephen Pfohl, Perry Payne, Martin Seneviratne, Paul Gamble, Chris Kelly, Nathaneal Schärli, Aakanksha Chowdhery, Philip Andrew Mansfield, Blaise Agüera y Arcas, Dale R. Webster, Gregory S. Corrado, Yossi Matias, Katherine Chou, Juraj Gottweis, Nenad Tomasev, Yun Liu, Alvin Rajkomar, Joelle K. Barral, Christopher Semturs, Alan Karthikesalingam, and Vivek Natarajan. Large language models encode clinical knowledge. CoRR, abs/2212.13138, 2022.

Xingyou Song, Sagi Perel, Chansoo Lee, Greg Kochanski, and Daniel Golovin. Open source vizier: Distributed infrastructure and API for reliable and flexible blackbox optimization. In Isabelle Guyon, Marius Lindauer, Mihaela van der Schaar, Frank Hutter, and Roman Garnett (eds.), International Conference on Automated Machine Learning, AutoML, volume 188 of Proceedings of Machine Learning Research, pp. 8/1–17. PMLR, 2022.

Xingyou Song, Qiuyi Zhang, Chansoo Lee, Emily Fertig, Tzu-Kuo Huang, Lior Belenki, Greg Kochanski, Setareh Ariafar, Srinivas Vasudevan, Sagi Perel, and Daniel Golovin. The vizier gaussian process bandit algorithm, 2024.

Brandon Trabucco, Xinyang Geng, Aviral Kumar, and Sergey Levine. Design-bench: Benchmarks for data-driven offline model-based optimization. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (eds.), International Conference on Machine Learning, ICML, volume 162 of Proceedings of Machine Learning Research, pp. 21658–21676. PMLR, 2022.

Robert Vacareanu, Vlad-Andrei Negru, Vasile Suciu, and Mihai Surdeanu. From words to numbers: Your large language model is secretly A capable regressor when given in-context examples. CoRR, abs/2404.07544, 2024.

Martin Wistuba and Josif Grabocka. Few-shot bayesian optimization with deep kernel surrogates. arXiv preprint arXiv:2101.07667, 2021.

In-Kwon Yeo and Richard A. Johnson. A new family of power transformations to improve normality or symmetry. Biometrika, 87(4):954–959, 2000. ISSN 00063444.

Arber Zela, Julien Niklas Siems, Lucas Zimmer, Jovita Lukasik, Margret Keuper, and Frank Hutter. Surrogate NAS benchmarks: Going beyond the limited search spaces of tabular NAS benchmarks. In International Conference on Learning Representations, ICLR, 2022.

Qi-Le Zhou, Han-Jia Ye, Le-Ye Wang, and De-Chuan Zhan. Unlocking the transferability of tokens in deep models for tabular data. CoRR, abs/2310.15149, 2023.

Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul F. Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. CoRR, abs/1909.08593, 2019.

### Appendix

- A. Additional Ablations We further ablate certain settings and scenarios which affect the model’s prediction accuracy below.

###### A.1. 𝑦-Tokenization

There are multiple possible ways to represent a float (e.g. 123.4) using custom tokens. Using examples from (Charton, 2022; d’Ascoli et al., 2022; Nogueira et al., 2021), the following are all possible representations:

- • (Default) Separate Sign and Digit-by-Digit: <+><1><2><3><4><E-2>
- • Merged Mantissa: <+1234><E-2>
- • Exponent Before Mantissa: <+><E-2><1><2><3><4>

In Table 8, we see that these tokenization differences do not matter with large training data (e.g. multi-task), but matter very much in low data regimes (e.g. single-task). The poor accuracy using “Merged Mantissa” is especially apparent as it requires large amounts of data to learn differences between 18K possible mantissa tokens.

AutoML BBOB Tokenization Method Single-Task Multi-Task Single-Task Multi-Task Default 0.21 0.15 0.17 0.01

Merged Mantissa 0.73 0.15 0.41 0.01 Exponent Before Mantissa 0.24 0.15 0.17 0.01

Table 8 | Mean Study Error (↓) comparisons between different tokenization methods.

###### A.2. Ranking and Correlation Metrics

Although our paper focuses on pointwise predictions which are maximally informative, we can trivially bootstrap our predictions into ranking-based metrics, which may be of downstream use for evolutionary algorithms which are agnostic to 𝑦-scaling. We see that in general, the multi-task LM generally maintains competitive ranking metrics.

|Regressor|Kendall-Tau, Spearman Correlation (↑)<br><br>BBOB Bid Simulation AutoML Init2Winit Protein Design V-AI (Tab) V-AI (Text)<br><br>|
|---|---|
|Gaussian Process Random Forest Tree MLP Single-task LM Multi-task LM|0.69, 0.80 0.80, 0.91 0.04, 0.06 0.15, 0.81 0.35, 0.43 -0.03, -0.05 0.30, 0.39<br><br>0.59, 0.75 0.71, 0.84 0.45, 0.57 0.55, 0.67 0.40, 0.52 0.56, 0.71 0.29, 0.38<br><br>0.60, 0.74 0.82, 0.93 0.37, 0.48 0.59, 0.71 0.44, 0.57 0.55, 0.70 0.28, 0.36<br><br><br>0.63, 0.76 0.73, 0.85 0.37, 0.49 0.53, 0.63 0.47, 0.60 0.50, 0.64 0.25, 0.34 0.01, 0.01 0.19, 0.28 0.21, 0.28 0.05, 0.08 0.15, 0.20 0.18, 0.24 0.11, 0.16 0.92, 0.96 0.70, 0.84 0.61, 0.73 0.65, 0.74 0.72, 0.81 0.57, 0.72 0.49, 0.58<br><br>|

- Table 9 | Higher (↑) is better. Ranking metrics across different regressors and tasks.

##### B. Model Details

###### B.1. Pretraining

We pretrained our model using T5X (Raffel et al., 2020), which can be found in the open-source codebase https://github.com/google-research/t5x. Important hyperparameters, most of which are defaulted, include:

- • Architecture size: 12 encoder layers, 12 decoder layers, 12 heads, 64 head dimension, 768 embedding dimension, 2048 MLP dimension.
- • Optimizer: Adafactor with base learning rate 0.01 and square root decay. Batch size 256.
- • Vocabulary and Tokenizer: SentencePiece tokenizer (Kudo & Richardson, 2018) with a vocabulary of 32000 subword tokens, in addition to the custom tokens for representing 𝑦-objectives.
- • Early stopping: We train for a maximum of 1 million steps, but early stop based on validation loss if overfitting is detected.

The model (≈ 200M parameters) was pretrained using a 4x4 TPU V3.

###### B.2. Local Training

During local training, data comes from a single study’s limited trials (at most 1000). The training set size can be lower than the batch size (256), and thus we must define one epoch as seeing the training data once, i.e. only one gradient step if training size ≤ batch size, but multiple gradient steps otherwise.

We use the same settings from pretraining for consistency, but allow a maximum of 30 epochs. For early stopping, validation loss is now measured over the entire validation set instead of sampled batches. Further specific changes:

- • Single-task training: Since the model is initialized randomly, we use a larger constant learning rate of 103, consistent with early learning rates encountered during pretraining.
- • Finetuning: We reload the weights in addition to the optimizer state (containing e.g. momentum parameters) from a checkpoint. We use a smaller fixed learning rate of 10−5, which is 10x lower than the O(10−4) learning rate normally encountered during late stages of training.

Due to the small training set and relatively low finetuning steps, we used a single 1x1 TPU V3.

###### B.3. Inference

At inference time, we perform temperature sampling with a temperature of 1.0. We restrict the logits to only decode the custom floating point tokens for representing 𝑦-values. To maximize batch size for a 1x1 TPU V3, we generate 64 samples and select the empirical median of these floating point samples as our final prediction when computing prediction error.

##### C. Baseline Details

- C.1. OSS Vizier Input Space The space is defined as a list of ParameterConfigs, each of which is one of the four primitives:

- • DOUBLE: Specifies the search range [𝑙, 𝑢].
- • DISCRETE: Specifies a finite subset of ℝ.
- • INTEGER: Specifies an integer range [𝑙, 𝑢].
- • CATEGORICAL: Specifies a set of strings.

Numeric (DOUBLE, DISCRETE and INTEGER) parameters may specify optional log or reverse-log scaling. The log scaling is most commonly used for tuning learning rates.

###### C.2. Data Processing for Flat Space

A flat space is where every trial in the study specifies every parameter configured in the space. In this case, we convert the parameters into the unit hypercube [0, 1]𝑑. For numeric parameters, we scale all values into the range [0, 1], by default using linear scaling unless (reverse)-log scaling was configured. For CATEGORICAL parameters, we use a one-hot encoding.

###### C.3. Data Processing for Conditional Space

A conditional space occurs when one parameter may be unused, depending on its parent parameter’s value. Conditional spaces commonly appear in AutoML settings where different model classes require a different set of parameters to be tuned. Another common use case is when we wish to optimize a numeric hyperparameter in the log scale but include 0 in the search (e.g. dropout rate, regularization coefficient), i.e. {UNUSED} ∪ [𝑙, 𝑢] where 𝑙 > 0.

For a categorical parameter, we simply add an extra out-of-vocabulary dimension for the one hot encoding.

For a numeric parameter, we first convert parameter values to NaN ∪ [0,1], using the same scaling as in the flat space but mapping all UNUSED to NaN. We then add a custom layer (one per parameter) which is defined as:

𝑣𝑝 if 𝑥 is NaN, 𝑥 otherwise

𝑥 ↦→

where 𝑣𝑝 is a parameter that is trained together with the rest of the model.

###### C.4. Regressor Baselines

Below, we list out the specific implementation details of our regressor baselines. One nuanced issue is of hyperparameter tuning the regressors themselves, which could affect results. In order to be reasonably fair to all regressors (including our own OmniPred which has its own hyperparameters), for each regressor, we used a reasonable fixed set of hyperparameters for consistency throughout all experiments.

We emphasize that our paper’s contributions are mostly on regression using flexible string representations and large-scale multi-task training, and do not claim to replace widely accepted baselines in single-task, apples-to-apples comparisons.

Gaussian Process: The GP regressor model is from the GP-Bandit implementation (Song et al.,

2024) found in Open Source Vizier at https://github.com/google/vizier and consists of the following:

- • 𝛼 ∼ TruncatedLogNormal controls the amplitude of Matern-5/2 kernel.
- • 𝜆𝑖 ∼ TruncatedLogNormal (i.i.d. for each dimension 𝑖) controls the length scale for the 𝑖-th dimension.
- • 𝜎 ∼ TruncatedLogNormal controls the Gaussian noise.
- • 𝑧 ∼ Normal(0, 𝜎) is the observation noise.
- • 𝑓 ∼ GP(𝜆, 𝛼) is the function.
- • 𝑦 ∼ 𝑓 (𝑥) + 𝑧 is the noisy function.

The algorithm then uses L-BFGS to obtain the MAP estimate of 𝛼, 𝜆 and 𝜎.

One caveat here is that this model requires a non-linear preprocessing on the observations and thus predicts 𝑦 in the preprocessed space. This preprocessing is found to be critical to achieving stable regression across Vizier studies, which have a wide variety of value ranges. Since the preprocessing is non-linear, we cannot obtain the predictive distribution over the raw observation in a closed form. Instead, we take 1000 samples from the GP, apply the inverse of the preprocessor, and then take the mean.

Tree and Random Forest: We use the standard API (XGBRegressor, XGBRFRegressor in https: //github.com/dmlc/xgboost) found in XGBoost (Chen & Guestrin, 2016), with the following grid-search hyperparameter sweeps for each study:

- • “min_child_weight": [1, 5, 10]
- • “learning_rate": [0.001, 0.01, 0.1]
- • “gamma": [0.0, 0.3, 0.5]
- • “subsample": [0.6, 0.8, 1.0]
- • “colsample_bytree": [0.6, 0.8, 1.0]
- • “max_depth": [3, 5, 7]

Although tree-based methods do not generally require rescaling, we still applied consistent 𝑥preprocessing (in particular to deal with optional log or reverse-log scaling).

Multilayer Perceptron: The base architecture consists of a 2-layer ReLU dense network of hidden size 256 with a final scalar output. 𝑦-values are normalized using tf.keras.layers.Normalization which subtracts mean and divides by standard deviation computed empirically from the training data. Training was performed with an Adam optimizer using learning rate 10−2, full batch training over 100 epochs, and mean squared error.

##### D. Google Vizier Data

###### D.1. Study Preprocessing

Since Google Vizier is a service in which users control evaluations, much of the raw study data can be quite chaotic. We apply certain preprocessing techniques to make the data more conducive to training and evaluation.

Removing bad trials: Users may ignore or fail to evaluate a proposed 𝑥. Furthermore, during some trials, the 𝑦-objective may be denoted with a special “infeasible” value (e.g. if a high batch size led to GPU out-of-memory errors, or if training encountered NaNs). We remove such trials from consideration in our work, although one can extend our 𝑦-tokenization to support infeasibility in later works.

Trial count hard limit: Some raw studies can contain trials in upwards of 105 trials, which could dominate the data distribution. We therefore apply a hard limit and only consider the first 103 trials per study.

Filtering specific users: There are specific human and automated “power users” which produce orders of magnitude more studies than the average user. Some automated users in particular are simply automatic unit tests involving the use of OSS Vizier. We disregard studies from these users to prevent them from dominating the data distribution.

###### D.2. Real World Data Descriptions

(Overall) Entire Database: No filters were applied. All studies from the database were exported on March 31, 2023. Finetuning experiments involving unseen studies consist of studies created after this date.

Bid Simulation: Contains hyperparameters for proprietary bid simulation. The simulator estimates how advertisements might have performed in terms of key metrics such as cost, impressions, clicks, and conversion volume.

AutoML: Collection of proprietary AutoML data for tuning user objectives.

Init2Winit: Data from running deterministic, scalable, and well-documented deep learning experiments, with a particular emphasis on optimization and tuning experiments (e.g. ResNets on CIFAR10, Transformers on LM1B). Public codebase can be found in https://github.com/google/ init2winit.

Protein Design: Each space consists of 50+ parameters, each of which denotes a categorical protein building block.

Vertex AI (Tabular and Text): A Vertex AI platform for automated ML model selection and training for tabular or text data. For tabular data, Vertex AI searches over a tree of model and optimizer types, their hyperparameters, data transformation, and other components in the ML pipeline. For text, Vertex AI trains an ML model to classify text data, extract information, or understand the sentiment of the authors. For more information, see:

https://cloud.google.com/vertex-ai?#train-models-with-minimal-ml-expertise.

###### D.3. Serialization Examples

For transparency, we provide examples of text representations seen by the model. Disclaimer: Due to privacy policies, we redacted (in red) some parameter names and values.

|Dataset<br><br>|Example 𝑥|Example 𝑚<br><br>|
|---|---|---|
|AutoML|batch_size: 128 model_type: REDACTED activation_fn: "tanh" batch_norm: "True" dropout: 0.143 embedding_combiner: "mean" gradient_clip_norm: 1.63e+03 num_hidden_layers: 1 hidden_units[0]: 359 optimizer_type: "AdamOptimizer"<br><br>beta1: 0.9<br>beta2: 0.999 learning_rate: 0.926 nlp_vocabulary_strategy:<br><br><br>"adjusted_mutual_info" vocabulary_strategy:<br><br>"adjusted_mutual_info"<br><br>|title: "n-w597ng99i7lj0-q40zcboi1ea7l" user: REDACTED description: "" objective: "val_categorical_cross_entropy" amc_model_version: REDACTED task_type: "multi_class_classification"|
|Init2Winit|dropout_rate: 0.6 decay_factor: 0.0379 label_smoothing: 0.378 lr_hparams.base_lr: 0.00285 lr_hparams.decay_steps_factor: 0.854 lr_hparams.power: 1.94 opt_hparams.one_minus_momentum: 0.0557<br><br>|title: "d_sp1-lm1b_trfmr-b1024-2021aug20" user: REDACTED description: "" objective: "valid/ce_loss"|
|Protein Design<br><br>|p00000000:"9"<br>p00000001:"16"<br>p00000002:"1"<br>p00000003:"11"<br>p00000004:"16" p00000006:"9"<br><br><br>p00000006:"0"<br>p00000007:"14"<br><br><br>... p00000047:"13"<br><br>|title: "871cac30956711eab5bef371aa1bb25a" user: REDACTED description:"" objective:""|
|Vertex AI Text<br><br>|universal_model_type:<br><br>"single_dense_feature" token_model_type: "cnn" token_bow_combiner: "sqrtn" token_model_type: "bow" rand:0 batch_size: 4 convnet: "2:3:4*100pa" dropout_keep_prob: 1 hidden_layer_dims: 50 max_length: 1.54e+03 max_num_classes_for_per_class_metric: 0 max_token_vocab_size: 1e+05 merge_word_embeddings_vocab_size: 1e+05 token_freq_cutoff: 1 tokenizer_spec: "delimiter" word_embedding: REDACTED word_embedding_dim: 100<br><br>|title: "20220228-621c9aea-0000-2c94" user: REDACTED description: REDACTED objective:"micro-auc-pr-label0_label" atc_study_dataset_tag:"" atc_study_notes:""|

y"

"

"

