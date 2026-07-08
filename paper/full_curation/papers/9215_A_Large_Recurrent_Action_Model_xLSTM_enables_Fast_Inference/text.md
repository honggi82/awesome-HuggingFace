# A Large Recurrent Action Model: xLSTM Enables Fast Inference for Robotics Tasks

Thomas Schmied1 Thomas Adler1 Vihang Patil1 Maximilian Beck12 Korbinian P¨oppel12 Johannes Brandstetter12 G¨unter Klambauer12 Razvan Pascanu34 Sepp Hochreiter12

arXiv:2410.22391v3[cs.LG]4Jun2025

## Abstract

In recent years, there has been a trend in the field of Reinforcement Learning (RL) towards large action models trained offline on large-scale datasets via sequence modeling. Existing models are primarily based on the Transformer architecture, which results in powerful agents. However, due to slow inference times, Transformerbased approaches are impractical for real-time applications, such as robotics. Recently, modern recurrent architectures, such as xLSTM and Mamba, have been proposed that exhibit parallelization benefits during training similar to the Transformer architecture while offering fast inference. In this work, we study the aptitude of these modern recurrent architectures for large action models. Consequently, we propose a Large Recurrent Action Model (LRAM) with an xLSTM at its core that comes with linear-time inference complexity and natural sequence length extrapolation abilities. Experiments on 432 tasks from 6 domains show that LRAM compares favorably to Transformers in terms of performance and speed.

## 1. Introduction

Reinforcement Learning (RL) has been responsible for impressive success stories such as game-playing (Silver et al., 2016; Vinyals et al., 2019; Berner et al., 2019; Patil et al., 2022), plasma control for fusion (Degrave et al., 2022), or navigation of stratospheric balloons (Bellemare et al., 2020). While these successes were based on classical RL approaches, in which agents have been trained online with RL objectives, recently there has been a trend towards offline RL settings (Levine et al., 2020; Schweighofer et al., 2022)

1ELLIS Unit, LIT AI Lab, Institute for Machine Learning, JKU Linz, Austria 2NXAI GmbH, Linz, Austria 3Google DeepMind 4Mila - Qu´ebec AI Institute.

Proceedings of the 42nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

and sequence models trained via behavior cloning (Chen et al., 2021; Janner et al., 2021). Such approaches, in which agents are trained on large-scale offline datasets with causal sequence modeling objectives, have been driven by the proliferation of Transformer-based architectures and gave rise to what we refer to as Large Action Models (LAMs) to highlight their similarity to large language models (LLMs) (Radford et al., 2018). LAM approaches can also be used in multi-task settings to develop generalist agents such as Gato (Reed et al., 2022).

Existing LAMs are primarily based on the Transformer (Vaswani et al., 2017) architecture. Because of their powerful predictive performance, robotics has become an emergent application area for large models (Brohan et al., 2023b;a; Octo Model Team et al., 2024; Gu et al., 2023; Wang et al., 2023), and a number of large multi-task datasets were collected (Jia et al., 2024; Embodiment Collaboration et al., 2024; Jiang et al., 2023; Mandlekar et al., 2023). This development bears the potential to produce robotics agents that learn to master complex tasks in a wide range of environments and even different embodiments. For example, recently it has been demonstrated, albeit in restricted settings, that sequence models trained on multi-episodic contexts can perform in-context learning (ICL) (Laskin et al., 2020; Lee et al., 2023). One potential application of ICL can be to learn new related tasks in robotics without the need for re-training or fine-tuning.

One of the key reasons for the success of Transformer-based models is their ability to scale to large datasets through their efficient parallelization during training. However, despite numerous success stories in RL, language modeling (Brown et al., 2020) or computer vision (Dosovitskiy et al., 2021; He et al., 2022), a persistent drawback of Transformer-based architectures is their high inference cost in terms of both speed and memory (Kim et al., 2023). Consequently, deploying Transformer-based models in resource-constrained scenarios, such as on devices with limited hardware capacity and/or real-time constraints, e.g., robots or smartphones, is prohibitive because of the required fast inference times (Firoozi et al., 2023; Hu et al., 2023). A basic principle of control theory is that the controller sample rate should be in the order of magnitude of the sample rate of the sensors

Figure 1. Illustration of our Large Recurrent Action Model (LRAM) with an xLSTM (Beck et al., 2024) at its core.

(Franklin et al., 1998, Ch. 11). To illustrate this, for typical robots such as drones or industrial robot arms, rates of 100Hz-1000Hz are required to keep the system stable (Salzmann et al., 2023; El-Hussieny, 2024; Hu et al., 2023; Chignoli et al., 2021). This implies inference times of less than 10ms. At 1000Hz, a 15-second movement of the agent corresponds to a sequence of 15K steps (El-Hussieny, 2024), resulting in long context lengths even without ICL. While there exists a range of techniques to make large models faster, such as quantization (Frantar et al., 2023), distillation (Hinton et al., 2015), or pruning (LeCun et al., 1989), the quadratic-time complexity of self attention still remains.

Recently, modern recurrent architectures have been proposed, which exhibit similar parallelization properties during training as the Transformer architecture while offering linear-time inference complexity. These modern recurrent architectures include xLSTM (Beck et al., 2024) and statespace models (SSMs), such as Mamba (Gu & Dao, 2023; Dao & Gu, 2024) and Griffin/Hawk (De et al., 2024), and have challenged the dominance of the Transformer in language modeling but also in other domains such as computer vision (Alkin et al., 2024; Zhu et al., 2024), and biomedicine (Schmidinger et al., 2024). More importantly, their lineartime inference makes them suitable for deployment in scenarios with limited compute, large context sizes, and realtime requirements, such as robotics.

In this work, we assess the aptitude of modern recurrent architectures, such as xLSTM and Mamba, as large action models. To this end, we introduce a Large Recurrent Action Model (LRAM) with an xLSTM at its core (see Figure 1). We train our agents on 432 tasks from 6 domains using a supervised learning setting similar to that of the Decision Transformer (Chen et al., 2021, DT). We use data collected during online-RL training of single-task specialist agents and compile these trajectories alongside other expert demonstrations into a large-scale multi-domain dataset comprising 894M transitions. Due to their parallelization properties,

the modern recurrent architectures considered in this work can process this large-scale training set as efficiently as the Transformer, while being faster at inference. Experiments across 4 model sizes with our multi-task models indicate that LRAM compares favorably to Transformers in terms of both performance and speed. In addition, we study the effect of modern recurrent architectures on fine-tuning performance and in-context learning abilities, and find that they exhibit strong performance in both dimensions.

The main purpose of this paper is to test the hypothesis that modern recurrent model architectures are better suited for building LAMs than Transformers. Hereby, we make the following contributions.

- • We propose a Large Recurrent Action Model (LRAM) with an xLSTM at its core that enables efficient inference.
- • We assess the aptitude of modern recurrent architectures as backbones for large-action models with respect to their efficiency at inference time and overall performance in multi-task, fine-tuning, and in-context learning settings.
- • To foster further research on large action models, we release our data preparation pipeline and our datasets.1

## 2. Related work

Sequence Models in RL. LSTM (Hochreiter & Schmidhuber, 1997) is the dominant backbone architecture for partially observable online RL problems and has been behind achievements such as mastering Starcraft II (Vinyals et al., 2019), Dota 2 (Berner et al., 2019), and Atari (Espeholt et al., 2018; Kapturowski et al., 2019). After the success of the Transformer in NLP (Devlin et al., 2019; Radford et al., 2019; Brown et al., 2020), computer vision (Dosovitskiy

1GitHub: https://github.com/ml-jku/LRAM

- et al., 2021; He et al., 2022; Radford et al., 2021; F¨urst et al.,

- 2022) and speech recognition (Radford et al., 2022; Baevski et al., 2020), the architecture has found its way into RL. Chen et al. (2021) proposed the Decision Transformer (DT), a GPT-style model (Radford et al., 2018), that learns to predict actions from offline trajectories via behavior cloning. Trajectory Transformer (Janner et al., 2021) predicts actions along with states and rewards, which allows for dynamics modeling. Other follow-up works build on the DT (Zheng

et al., 2022; Wang et al., 2022; Shang et al., 2022; Meng et al., 2021; Siebenborn et al., 2022; Schmied et al., 2024a) or replace the Transformer with Mamba (Ota, 2024; Dai et al., 2024). Furthermore, sequence models trained to predict the next action were found to exhibit ICL if conditioned on previous trajectories (Laskin et al., 2022; Lee et al., 2022; Kirsch et al., 2023), albeit in limited scenarios.

Large Action Models (LAMs). LAMs, such as the Decision Transformer, are well-suited for multi-task settings. Lee et al. (2022) found that a multi-game DT can learn to play 46 Atari games. Reed et al. (2022) introduced a generalist agent trained on over 600 tasks from different domains, ranging from Atari to manipulation of a robot arm. Jiang et al. (2022) a Transformer for robot manipulation based on multi-modal prompts, that allow to steer the model to perform new tasks. Recently, Raad et al. (2024) introduced an agent instructable via language to play a variety of commercial video games. Since then, robotics has become an emergent area for developing LAMs (Brohan et al., 2023b;a; Octo Model Team et al., 2024; Gu et al., 2023; Wang et al., 2023; Kim et al., 2024), also due to the availability of largescale datasets (Jia et al., 2024; Embodiment Collaboration et al., 2024; Jiang et al., 2023; Mandlekar et al., 2023).

Next-generation Sequence Modeling Architectures. Linear recurrent models, such as state-space models (SSM, Gu et al., 2021; 2022b; Smith et al., 2023; Orvieto et al.,

- 2023) have challenged the dominance of the Transformer (Vaswani et al., 2017) architecture on long-range tasks (Tay et al., 2020). The key insight of those linear RNNs was to diagonalize the recurrent state matrix and enforce stable training via an exponential parameterization (Gu et al., 2022a; Orvieto et al., 2023). Since then, there have been efforts to include features such as gating from RNNs (Elman, 1990; Jordan, 1990; Hochreiter & Schmidhuber, 1997; Cho et al., 2014). Non-linear gates are believed to have higher expressivity, but are harder to train. Griffin (De et al.,
- 2024) mixes gated linear recurrences with local attention to achieve more training data efficiency than Llama-2 (Touvron et al., 2023) and better sequence extrapolation. Mamba (Gu & Dao, 2023) introduces a selection mechanism similar to gating into SSMs, which makes its state and input matrix time-dependent. This is similar to the gating mechanism of RNNs but also bears resemblance to approaches like fast weights (Schmidhuber, 1992) and Linear Atten-

tion (Katharopoulos et al., 2020). Mamba-2 (Dao & Gu, 2024) highlights the connection between SSMs with inputdependent state and input matrices and (Gated) Linear attention variants. Most recently, the xLSTM (Beck et al., 2024) was proposed as an improvement over the classic LSTM (Hochreiter & Schmidhuber, 1997) that combines gating, linear recurrences, and recurrent weights into a single architecture for language modeling. First, xLSTM leverages exponential gating with stabilization to RNNs for stronger emphasis on important inputs. Second, xLSTM is composed of two variants, the mLSTM variant with an emphasis on memory that proves important in language modeling, and the sLSTM variant that keeps the non-diagonalized recurrent matrix to enable state-tracking (Merrill et al., 2024). State tracking is important in logic tasks and cannot be modeled fundamentally by linearized recurrent or state-space models like Mamba, Griffin, or Transformers.

## 3. Large Recurrent Action Models

### 3.1. Background

Reinforcement Learning. We assume the standard RL formulation via a Markov Decision Process (MDP) represented by a tuple of (S,A,P,R), where S and A denote state and action spaces, respectively. At every timestep t, the agent observes state st ∈ S, predicts action at ∈ A, and receives a scalar reward rt. The reward is determined by the reward function R(rt | st,at). P(st+1 | st,at) defines the transition dynamics and constitutes a probability distribution over next states st+1 when executing action at in state st. The goal of RL is to learn a policy π(at | st) that predicts an action at in state st that maximizes rt.

Decision Transformer (Chen et al., 2021) casts the RL problem setting as next action prediction task via causal sequence modeling. At training time, DT aims to learn a policy πθ that maps future rewards to actions, which is often referred to as upside-down RL (Schmidhuber, 2019). At inference time, the DT is conditioned via a target return to emit high-reward actions. Consequently, we assume access to a dataset D = {τi}Ni=1 containing N trajectories τi consisting of quadruplets τi = (s1,Rˆ1,a1,r1,...,sT,RˆT,aT,rT) of state st, return-to-go (RTG) Rˆt = Tt′=t rt′, action at, and reward rt. Here, T refers to the length of the trajectory. The DT πθ is trained to predict the ground-truth action at conditioned on sub-trajectories from the dataset:

aˆt ∼ πθ(ˆat |st−C,Rˆt−C,at−C,rt−C,..., st−1,Rˆt−1,at−1,rt−1,st,Rˆt)

(1)

where C ≤ T is the size of the context window. In fact, Equation 1 describes the setting of the multi-game DT (Lee et al., 2022), which also includes rewards in the sequence representation.

Table 1. Dataset statistics for all 432 training tasks.

### Dataset Tasks Trajectories Mean Trj. Length Total Transitions Repetitions

Atari 41 136K 2733 205M 1.03× Composuite 240 480K 500 240M 0.87× DMControl 11 110K 1000 110M 1.92× Meta-World 45 450K 200 90M 2.34× Mimicgen 83 83K 300 25M 8.5× Procgen 12 2185K 144 224M 0.94×

Total 432 3.4M - 894M -

### 3.2. Large Recurrent Action Models (LRAMs)

Our LRAM has a modern recurrent architecture at its core (see Figure 1), which comes with a parallel training and a recurrent inference mode. We instantiate LRAM with three different variants, two different xLSTM configurations, and Mamba. We use a training protocol similar to that of Lee et al. (2022) and Reed et al. (2022) with important differences that aim to speed up inference across backbones.

Multi-modal sequence representation. To encode input from different environments with varying state and action spaces, we use separate encoders per modality that are shared across tasks and domains. For encoding images, we use a CNN similar to Espeholt et al. (2018), whereas for low-dimensional inputs we use a fully connected network. We refrain from patchifying images and tokenizing continuous states to avoid unnecessarily long sequences. Similarly, we use linear layers to encode rewards and RTGs. We omit actions in our sequence formulation, as we found that this can be detrimental to performance, in particular for continuous control tasks with smoothly changing actions (see Section 4.3). Consequently, our trajectories have the form τi = (s1,Rˆ1,r1,...,sT,RˆT,rT) and we train our policy πρ to predict the ground-truth action at as:

aˆt ∼ πρ(ˆat |st−C,Rˆt−C,rt−C,..., st−1,Rˆt−1,rt−1,st,Rˆt).

(2)

Shared action head. Action spaces in RL typically vary across environments. For example, in the environments we consider, there are 18 discrete actions and a maximum of 8 continuous dimensions for continuous control environments. Therefore, we employ discretization of continuous action dimensions into 256 uniformly-spaced bins, similar to Reed et al. (2022) and Brohan et al. (2023b). Unlike prior work, we leverage a shared action head to predict all discrete actions or continuous action dimensions jointly. We found that this setup significantly reduces inference time compared to using autoregressive action prediction of continuous actions.

Recurrent inference mode. At inference time, we leverage the recurrent backbone and maintain the hidden states of

the last timestep. This enables fast inference with lineartime complexity along the sequence length. In addition, the recurrent-style inference is well-suited for online fine-tuning via RL objectives, similar to LSTM-based policies in online RL. To speed up inference, we leverage custom kernels for the xLSTM backbone (see Appendix 21).

Our unified discrete action representation enables consistent training of our agents via the cross-entropy loss as training objective across all tasks and domains, similar to Reed et al. (2022). We use separate reward scales per domain and target returns per task. Furthermore, we do not make use of timestep encodings as used by Chen et al. (2021), which are detrimental when episode lengths vary. We provide additional implementation details in Appendix C.

## 4. Experiments

We study the aptitude of modern recurrent architectures as LAMs on 432 tasks from 6 domains: Atari (Bellemare et al., 2013), Composuite (Mendez et al., 2022), DMControl (Tassa et al., 2018), Meta-World (Yu et al., 2020b), Mimicgen (Mandlekar et al., 2023), and Procgen (Cobbe et al., 2020b). To this end, we compile a large-scale dataset containing 894 million transitions (see Section 4.1). Across all experiments, we compare four backbone variants: xLSTM [7:1], xLSTM [1:0] (Beck et al., 2024), Mamba (Gu & Dao, 2023), and the GPT-2 style Transformer employed in the DT (Chen et al., 2021). Following (Beck et al., 2024), we use the bracket notation for xLSTM, which indicates the ratio of mLSTM to sLSTM blocks. For example, xLSTM [1:0] contains only mLSTM blocks.

In Section 4.2, we conduct a scaling comparison for four model sizes ranging from 16M to 206M parameters that shows that modern recurrent architectures achieve performance comparable or favorable to the Transformer baseline across different model sizes. In Section 4.3, we study the impact of the recurrent backbones on fine-tuning performance, ICL abilities, and further analyze our trained recurrent backbones. Finally, in Section 4.4, we empirically examine the differences at inference time in terms of latency and through-

[Figure 2]

[Figure 3]

(a) Sequence prediction (b) Environment interaction

- Figure 2. Scaling comparison. We compare xLSTM, Mamba, DT in four model sizes: 16M, 48M, 110M, and 206M parameters. We show the (a) validation perplexity on the hold-out datasets, and (b) normalized scores obtained from evaluating in the training task environments, averaged over all 6 domains.

put between xLSTM and Transformer-based agents, which indicate advantages for the recurrent backbone.

### 4.1. Datasets & Environments

Datasets. We compile a large-scale dataset comprising 432 tasks from six domains. We leverage datasets from prior works if available, and generate our own data otherwise. For Atari, we extract 5M transitions per task from the DQN-Replay dataset released by Agarwal et al. (2020). For Composuite, we leverage the datasets released by (Hussing

- et al., 2023). For Meta-World, we use 2M transitions per task released by (Schmied et al., 2024a). For DMControl, we generate 10M transitions per task using task-specific RL agents. For Mimicgen, we use the datasets for the 21 tasks released by (Mandlekar et al., 2023) and generate trajectories for the remaining 62 tasks. Finally, for Procgen, we extract 20M transitions from the datasets released by (Schmied
- et al., 2024b). Our final dataset contains 3.4M trajectories and in total 894M transitions (see Table 2). We reserve an additional 37 tasks from the same domains for zero-shot evaluation. To foster future research, we release our datapreparation pipeline and generated data. We provide the rationales for our specific dataset selection in Appendix B.1.

Environments. Atari and Procgen come with image observations and discrete actions. In contrast, the remaining four domains exhibit state-based observations and continuous actions. Consequently, our experiments involve a mixture of state and action spaces as well as varying episode lengths (see Table 2). Periodically evaluating the trained agents on all 432 tasks sequentially is time-consuming, and we, therefore, distributed the evaluation across GPUs and parallel processes (see Appendix C). Additional details on our datasets and environments are available in Appendix B.

### 4.2. Scaling comparison

To conduct our main comparisons, we train our four backbone variants on the full training task mixture of 432 tasks. For each architecture backbone, we report performance scores for four model sizes: 16M, 48M, 108M, and 206M parameters. We train all models for 200K updates with a batch size of 128 and a context length of 50 timesteps. All domains are represented with approximately equal proportion, resulting in 33K updates per domain. Additional implementation details and hyperparameters for every backbone variant and model size are available in Appendix C.

Sequence prediction performance. In Figure 2a, we report the validation set perplexity for all backbones and model sizes averaged over the individual scores from all domains. To achieve this, we maintain a hold-out set of trajectories for each training task (2.5%) and compute the perplexities after every 50K steps (see Figure 12 for training perplexities). Both recurrent backbones outperform the Transformer baseline considerably, especially as the model sizes increase.

Evaluation performance. During training, we evaluate our agents after every 50K step in all 432 training environments. In Figure 2b, we report the resulting normalized performances averaged across all six domains. The recurrent backbones outperform the Transformer one across model sizes. While xLSTM and Mamba perform similarly at smaller scales, xLSTM tends to outperform Mamba at larger scales (206M). This is an important advantage of xLSTM, as LRAM agents can strongly benefit from more data and consequently larger models. Note that Mamba has a significantly higher number of parameters than competitors. For the zero-shot evaluation performances on the 37 hold-out tasks, we refer to Figure 14 in Appendix D.2.

[Figure 5]

- Figure 3. Normalized scores per domain for model size 206M. For Meta-World, DMControl, Mimicgen, Composuite, and Procgen, we report data-normalized scores, for Atari we report human-normalized scores.

Performance per domain. In Figure 3, we report the normalized scores for the 206M models attained on all six domains. For Meta-World, DMControl, Mimicgen, Composuite, and Procgen, we use data-normalized scores, as suggested by (Levine et al., 2020). For Atari, we report human-normalized scores. We observe that xLSTM outperforms competitors on three of the six domains, while they perform similarly on the remaining domains.

- 4.3. Analyses & Ablations

episodic context, which results in a large context length (see Appendix D.4 for details on the experiment setup). In particular, we adopt the Algorithm Distillation (AD, Laskin et al., 2022) framework and exchange the Transformer backbone architecture with modern recurrent architectures. In Figure 4, we report the ICL performance on the 20 hold-out tasks (see Figure 16 for training tasks). We find that xLSTM [7:1] attains the highest overall scores both on the 80 training and 20 hold-out tasks, which we attribute to the state-tracking abilities (Merrill et al., 2024) of sLSTM blocks.

Embedding space analysis. In Figure 5, we analyze the representations learned by our model. We sample 32 subtrajectories from every task, extract the sequence representation at the last layer, cluster them using UMAP (McInnes et al., 2018), and color every point by its domain (see Appendix F for more details). We find that tasks from the same domain cluster together. Furthermore, xLSTM exhibits a more refined domain separation compared to DT, which may further contribute to the better downstream performance. See Appendix F for a more detailed discussion on the embedding space analysis and a comparison to Mamba.

Fine-tuning. To assess the effect of the recurrent backbones on fine-tuning performance, we fine-tune our models on 37 held-out environments from all 6 domains. We evaluate the fine-tuning performance of the xLSTM architecture for the 16M pretrained models and compare it against an xLSTM trained from scratch. The pretrained LRAM outperforms the randomly initialized xLSTM model in most domains (see Figure 15). This suggests that fine-tuning performance is not affected negatively by switching the backbone.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Figure 4. In-context Learning with modern recurrent architectures on 20 hold-out tasks for Dark-Room 10 × 10.

(a) DT (b) xLSTM

In-context Learning. Next, we study the ICL abilities of our recurrent backbones on the Dark-Room environment considered in prior work on in-context RL (Laskin et al., 2022; Lee et al., 2023; Schmied et al., 2024b). To study ICL in isolation, we train models from scratch with a multi-

Figure 5. Embedding space comparison. UMAP clustering of hidden states for all tasks for 16M, colored by domain. xLSTM exhibits a better domain separation than DT.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

(a) Latency, B = 1 (b) Latency, B = 16 (c) Memory, B = 1

Figure 6. Latency comparison on A100. We report latency for varying context lengths (in timesteps) with batch sizes (a) B = 1 and (b)

- B = 16. In (c), we show the memory consumption in % of GPU memory with B = 1. We compare DT to xLSTM and Mamba with the same number of layer blocks and parameters on Atari Freeway. Missing bars for DT indicate out-of-memory (OOM).

Removing Actions & Effect of Context Length. We found that removing actions from the context results in better performance across backbones. While context lengths beyond 1 hurt performance on Meta-World and DMControl, and when training with actions, the reverse is true when training without actions (see Figures 23, 24, 26). This is in contrast to recent works, which did not benefit from longer contexts (Octo Model Team et al., 2024). While removing actions improves performance on Meta-World/DMControl, it does not affect performance on discrete control environments. For Meta-World/DMControl, we observed that the models become overly confident, which is problematic if poor initial actions are produced. This is because many robotics environments exhibit smoothly changing actions, and by observing previous actions, the agent can learn shortcuts. A similar issue has been observed by Wen et al. (2020) and termed the copycat problem. Removing actions from the input prevents the agent from using shortcuts and, therefore, alleviates the copycat problem. Importantly, the evaluation performance improves across domains as the sequence length increases, which indicates that the history helps to predict the next action (e.g., by observing mistakes made in the past, see Figures 25, 27).

Return-conditioning vs. Behavior Cloning. Across our experiments, we utilized a sequence representation that includes return-to-go tokens, as commonly used in DTs (Chen

- et al., 2021; Lee et al., 2022). However, many recent works focus on behavior cloning without return conditioning (Reed
- et al., 2022; Brohan et al., 2023a). Therefore, we study the effect of excluding the RTG/reward tokens from the sequence at the 206M parameter scale, to validate that our findings transfer to the behavior cloning setting. Indeed, we find that the same trends hold (see Figures 28 and 29).

mLSTM-to-sLSTM Ratio. Throughout experiments, we compare two xLSTM variants: xLSTM [7:1] and xLSTM [1:0]. These ratios were proposed by Beck et al. (2024) and

we maintain the same ratios for consistency (see Appendix C.3). While mLSTM is parallelizable, sLSTM enables statetracking (Merrill et al., 2024). To better understand the effect of the ratio, we conduct ablation studies both on the 432 tasks and on Dark-Room (see Appendix E.3), similar to Beck et al. (2024). We find that other ratios, such as [3:1], can be effective, and highlight the importance of placing sLSTMs at lower-level layers (Figure 31). However, the effectiveness of sLSTM layers is dependent on the task at hand. Complex tasks with long horizons or partial observability, as are common in real-world applications, may benefit from the state-tracking abilities provided by sLSTM.

We present additional ablations on the effect of reducing the number of layers in xLSTM and disabling Dropout on DT in Appendix E.5 and E.4, respectively.

### 4.4. Inference Time Comparison

Finally, we empirically examine the difference between recurrent and Transformer-based agents at inference time. Similar to De et al. (2024), we report both latency and throughput. We focus our analysis on latency, as it is the more important dimension for real-time applications.

Setup. We conduct all inference time tests on A100s with 40GB of RAM using 206M models. For the Transformer, we use KV-caching and FlashAttention (Dao, 2023) as supported by PyTorch (Paszke et al., 2019). For xLSTM, we use recurrent-style inference using custom kernels to accelerate computations (see Figure 21 for the impact of kernel acceleration). For Mamba, we make use of the kernels introduced by Gu & Dao (2023). For DT and xLSTM, we use torch.compile, but not for Mamba because we found the kernels to be incompatible with compilation. The Transformer with KV-caching has a linear time complexity per step and quadratic in the sequence length. In contrast, the xLSTM and Mamba have a constant time complexity per

step and are linear in the sequence length. Therefore, we expect speed-ups especially for longer sequences and larger batch sizes, as observed by De et al. (2024). To ensure a fair comparison, we compare all backbones with the same number of layer blocks and increase the hidden size of xLSTM and Mamba to match the number of parameters of DT (see Appendix E.5 for evaluation performance of these models). We provide further details on our inference time tests in Appendix D.5.

Environment. We conduct all inference time tests on the environment that exhibited the longest average episode lengths in our experiments, the Atari game Freeway. Every episode in Freeway lasts for 8192 steps, which is equivalent to 24576 tokens (s/rtg/r). We evaluate all models for 5 episodes and preserve the KV-cache/hidden state across episode boundaries. The reported latencies and throughputs are averaged across all evaluation episodes, except for the first episode, which we discard to exclude compilation times and prefilling. We opted for measuring the inference times during environment interaction, i.e., including simulator latency, rather than mere token generation.

Latency. Similar to De et al. (2024), we measure latency by the average time (in seconds) taken to perform a single inference step with a fixed batch size B (lower is better). In Figure, 6, we report the latencies for varying context lengths,

- C ∈ [50,25600] and two batch sizes B ∈ {1,16}. Note that C is in time steps, and every time step contains 3 tokens (state, reward-to-go, reward). Hence, the effective sequence length for the largest C is 76800. As expected, we find that the recurrent backbones attain lower inference latencies than the Transformer one, especially for longer sequences and with a larger batch size. For B = 1, we find that Mamba is slower than the Transformer and xLSTM, which we believe is because of the incompatibility with torch.compile. Note that we expect the gap to xLSTM to be closed with compatible kernels. As the sequence length increases, DT runs out of memory due to the increasing size of the KV cache (see Figure 6c). In contrast, the inference speeds for Mamba/xLSTM are independent of the context length and therefore, enable significantly longer context lengths. This property is particularly interesting for in-context RL, which requires keeping multiple episodes in the context (Laskin et al., 2022). Nevertheless, our experiments highlight that the materialization of the complexity advantage depends on the device, model size, batch size, and the context length, which is similar to findings by De et al. (2024).

Throughput. Throughput is measured by the total number of inference steps performed per second for a model with a fixed context length. In Figure 7, we report the throughputs for varying batch sizes, B ∈ [1,128] for a fixed context length of C = 1600. Here, the batch size can be interpreted as the number of parallel environments the agent interacts

[Figure 15]

[Figure 16]

Figure 7. Throughput comparison on A100 for varying batch sizes with C = 1600 timesteps on the Atari Freeway environment. We compare DT, xLSTM with 4 and 16 heads, and Mamba. Missing bars for DT indicate OOM.

with. For xLSTM, we report numbers for two variants with

- 4 and 16 heads, respectively. We found that decreasing the head dimension (more heads, same total hidden dim) is important for xLSTM to enable high throughput. This is because a higher head dimension incurs more FLOPS (see Figure 22 in Appendix D.5.4 for an ablation on the impact of the head dimension). As expected, we find that both Mamba and xLSTM attain considerably higher throughputs than the DT. These benefits increase with larger batch sizes. While the DT with quadratic complexity in the sequence length goes OOM for batch sizes above 64, the recurrent backbones with linear complexity can easily handle larger batch sizes. This throughput advantage may be particularly relevant for online fine-tuning of agents in many parallel environments.
- 5. Conclusion

In this work, we study the aptitude of modern recurrent architectures as alternatives to Transformers for building LAMs. We found that our LRAM with an xLSTM or Mamba at its core compares favorably to the Transformer in terms of evaluation performance across model scales ranging from 16M to 206M parameters (see Section 4.2). Moreover, we demonstrated that LRAM exhibits higher inference speeds, especially at large context sizes (see Section 4.4). Thus, the empirical evidence suggests that recurrent backbones can be attractive alternatives for LAMs. Notably, the linear-time inference complexity of xLSTM and Mamba may enable applications that require long context lengths (e.g., ICL) and facilitate the application of large-scale agents for real-time applications, such as robotics.

Modern recurrent architectures and Transformers come with different advantages and disadvantages. xLSTM and Mamba, on the one hand, exhibit a fundamental complexity advantage over Transformers. Their linear complexity

ensures that the computational requirements increase more slowly with the sequence length, which enables more efficient inference and is particularly relevant for edge applications. While we conduct our inference time comparisons on a high-end data center GPU, applications on edge devices may have to deal with less powerful accelerators. Importantly, we found that LAMs strongly benefit from longer sequences (see Section 4.3). Their ability to efficiently handle long sequences can be beneficial for applications in real-world environments, which often exhibit long-term dependencies. Similarly, longer context can be relevant for ICL applications, which benefit from keeping multiple episodes (such as demonstrations or previous trials) in the context. Transformers, on the other hand, are effective for applications that require exact recall of tokens (such as particular locations in a grid, signs in an image) in a sequence, which can be important for decision-making (Ni et al., 2024). Finally, xLSTM in particular enables state-tracking via sLSTM blocks, which Transformers and Mamba cannot perform (Merrill et al., 2024). State tracking can be important for logic tasks or dealing with partial observability and may be a useful tool for practitioners. Given these differences, different backbones should be considered depending on the task at hand.

Limitations & Future Work. The primary target application of LAMs is robotics. While the majority of our experiments involve robotic simulations, we do not yet provide experiments for real robots. We do, however, believe that our findings translate to real-world scenarios and aim to provide further evidence in future work. Moreover, our finetuning experiments are limited to offline RL. We envision that an agent pre-trained on large-scale datasets can be successfully fine-tuned via online RL to explore new strategies that do not appear in the training data. Modern recurrent architectures offer both parallel and recurrent training modes, which might be the key to success for such applications. While we provide evidence for improved ICL abilities of LRAM, we only consider a grid-world setting. We aim to further investigate the ICL abilities of LRAM in more complex environments.

## Impact Statement

While we conduct all our experiments in simulated environments, the primary target application of our method is robotics. We believe that our work can positively impact applications in the near future that require efficient inference, on-device processing, or have real-time constraints. However, robotics applications in the real world are not without risks. In particular, in areas where humans are involved, such as factory settings, special care is required. LAMs are trained via next-action prediction similar to LLMs. Consequently, LAMs may also suffer from hallucinations in

unknown scenarios. We therefore strongly discourage users from blindly following the predictions made by real-world LAMs without appropriate precautions regarding safety and robustness. It is essential to ensure the responsible deployment of such future technologies, and we believe that more research on the robustness of LAMs is necessary.

## Acknowledgements

We acknowledge EuroHPC Joint Undertaking for awarding us access to Karolina at IT4Innovations, Czech Republic, MeluXina at LuxProvide, Luxembourg, and Leonardo at CINECA, Italy. The ELLIS Unit Linz, the LIT AI Lab, the Institute for Machine Learning, are supported by the Federal State Upper Austria. We thank the projects FWF AIRI FG 9N (10.55776/FG9), AI4GreenHeatingGrids (FFG- 899943), Stars4Waters (HORIZON-CL6-2021-CLIMATE-01-01), FWF Bilateral Artificial Intelligence (10.55776/COE12). We thank NXAI GmbH, Audi AG, Silicon Austria Labs (SAL), Merck Healthcare KGaA, GLS (Univ. Waterloo), TUV¨ Holding GmbH, Software Competence Center Hagenberg GmbH, dSPACE GmbH, TRUMPF SE + Co. KG.

## References

Agarwal, R., Schuurmans, D., and Norouzi, M. An optimistic perspective on offline reinforcement learning. In International Conference on Machine Learning, pp. 104– 114. PMLR, 2020.

Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., and Bellemare, M. Deep reinforcement learning at the edge of the statistical precipice. Advances in neural information processing systems, 34:29304–29320, 2021.

Alkin, B., Beck, M., P¨oppel, K., Hochreiter, S., and Brandstetter, J. Vision-lstm: xlstm as generic vision backbone. CoRR, abs/2406.04303, 2024. doi: 10.48550/ARXIV. 2406.04303. URL https://doi.org/10.48550/ arXiv.2406.04303.

Baevski, A., Zhou, Y., Mohamed, A., and Auli, M. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in Neural Information Processing Systems, 33:12449–12460, 2020.

Beck, M., P¨oppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., and Hochreiter, S. xlstm: Extended long short-term memory. CoRR, abs/2405.04517, 2024. doi: 10.48550/ARXIV. 2405.04517. URL https://doi.org/10.48550/ arXiv.2405.04517.

Bellemare, M. G., Naddaf, Y., Veness, J., and Bowling, M. The Arcade learning environment: An evaluation plat-

form for general agents. Journal of Artificial Intelligence Research, 47:253–279, 2013.

Bellemare, M. G., Candido, S., Castro, P. S., Gong, J., Machado, M. C., Moitra, S., Ponda, S. S., and Wang, Z. Autonomous navigation of stratospheric balloons using reinforcement learning. Nature, 588(7836):77–82, 2020.

Berner, C., Brockman, G., Chan, B., Cheung, V., Dkebiak, P., Dennison, C., Farhi, D., Fischer, Q., Hashme, S., Hesse, C., et al. Dota 2 with large scale deep reinforcement learning. arXiv preprint arXiv:1912.06680, 2019.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Chen, X., Choromanski, K., Ding, T., Driess, D., Dubey, A., Finn, C., et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023a.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., Ibarz, J., Ichter, B., Irpan, A., Jackson, T., Jesmonth, S., Joshi, N. J., Julian, R., Kalashnikov, D., Kuang, Y., Leal, I., Lee, K., Levine, S., Lu, Y., Malla, U., Manjunath, D., Mordatch, I., Nachum, O., Parada, C., Peralta, J., Perez, E., Pertsch, K., Quiambao, J., Rao, K., Ryoo, M. S., Salazar, G., Sanketi, P. R., Sayed, K., Singh, J., Sontakke, S., Stone, A., Tan, C., Tran, H. T., Vanhoucke, V., Vega, S., Vuong, Q., Xia, F., Xiao, T., Xu, P., Xu, S., Yu, T., and Zitkovich, B. RT-1: robotics transformer for real-world control at scale. In Bekris, K. E., Hauser, K., Herbert, S. L., and Yu, J. (eds.), Robotics: Science and Systems XIX, Daegu, Republic of Korea, July 10-14, 2023, 2023b. doi: 10.15607/RSS.2023.XIX.025. URL https: //doi.org/10.15607/RSS.2023.XIX.025.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.

cc/paper_files/paper/2020/file/ 1457c0d6bfcb4967418bfb8ac142f64a-Paper. pdf.

Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., and Mordatch, I. Decision transformer: Reinforcement learning via sequence modeling. Advances in neural information processing systems, 34:15084–15097, 2021.

Chignoli, M., Kim, D., Stanger-Jones, E., and Kim, S. The mit humanoid robot: Design, motion planning, and control for acrobatic behaviors. In 2020 IEEE-RAS 20th International Conference on Humanoid Robots (Humanoids), pp. 1–8. IEEE, 2021.

Cho, K., van Merrienboer, B., G¨ulc¸ehre, C¸., Bahdanau, D., Bougares, F., Schwenk, H., and Bengio, Y. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In Moschitti, A., Pang, B., and Daelemans, W. (eds.), Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, EMNLP 2014, October 25-29, 2014, Doha, Qatar, A meeting of SIGDAT, a Special Interest Group of the ACL, pp. 1724–1734. ACL, 2014. doi: 10.3115/ V1/D14-1179. URL https://doi.org/10.3115/ v1/d14-1179.

Cobbe, K., Hesse, C., Hilton, J., and Schulman, J. Leveraging procedural generation to benchmark reinforcement learning. In International conference on machine learning, pp. 2048–2056. PMLR, 2020a.

Cobbe, K., Hesse, C., Hilton, J., and Schulman, J. Leveraging procedural generation to benchmark reinforcement learning. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pp. 2048–2056. PMLR, 2020b. URL http://proceedings.mlr.press/v119/ cobbe20a.html.

Dai, Y., Ma, O., Zhang, L., Liang, X., Hu, S., Wang, M., Ji, S., Huang, J., and Shen, L. Is mamba compatible with trajectory optimization in offline reinforcement learning? arXiv preprint arXiv:2405.12094, 2024.

Dao, T. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.

Dao, T. and Gu, A. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060, 2024.

De, S., Smith, S. L., Fernando, A., Botev, A., CristianMuraru, G., Gu, A., Haroun, R., Berrada, L., Chen, Y., Srinivasan, S., et al. Griffin: Mixing gated linear recurrences with local attention for efficient language models. arXiv preprint arXiv:2402.19427, 2024.

Degrave, J., Felici, F., Buchli, J., Neunert, M., Tracey, B., Carpanese, F., Ewalds, T., Hafner, R., Abdolmaleki, A., de Las Casas, D., et al. Magnetic control of tokamak plasmas through deep reinforcement learning. Nature, 602(7897):414–419, 2022.

Devlin, J., Chang, M., Lee, K., and Toutanova, K. BERT: pre-training of deep bidirectional transformers for language understanding. In Burstein, J., Doran, C., and Solorio, T. (eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pp. 4171– 4186. Association for Computational Linguistics, 2019. doi: 10.18653/v1/n19-1423.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

El-Hussieny, H. Real-time deep learning-based model predictive control of a 3-dof biped robot leg. Scientific Reports, 14(1):16243, 2024.

Elman, J. L. Finding structure in time. Cogn. Sci., 14(2): 179–211, 1990. doi: 10.1207/S15516709COG1402\ 1. URL https://doi.org/10.1207/ s15516709cog1402_1.

Embodiment Collaboration, O’Neill, A., Rehman, A., Maddukuri, A., Gupta, A., Padalkar, A., Lee, A., Pooley, A., Gupta, A., Mandlekar, A., Jain, A., Tung, A., Bewley, A., Herzog, A., Irpan, A., Khazatsky, A., Rai, A., Gupta, A., Wang, A., Singh, A., Garg, A., Kembhavi, A., Xie, A., Brohan, A., Raffin, A., Sharma, A., Yavary, A., Jain, A., Balakrishna, A., Wahid, A., Burgess-Limerick, B., Kim, B., Sch¨olkopf, B., Wulfe, B., Ichter, B., Lu, C., Xu, C., Le, C., Finn, C., Wang, C., Xu, C., Chi, C., Huang, C., Chan, C., Agia, C., Pan, C., Fu, C., Devin, C., Xu, D., Morton, D., Driess, D., Chen, D., Pathak, D., Shah, D., B¨uchler, D., Jayaraman, D., Kalashnikov, D., Sadigh, D., Johns, E., Foster, E., Liu, F., Ceola, F., Xia, F., Zhao, F., Stulp, F., Zhou, G., Sukhatme, G. S., Salhotra, G., Yan, G., Feng, G., Schiavi, G., Berseth, G., Kahn, G., Wang, G., Su, H., Fang, H., Shi, H., Bao, H., Amor, H. B., Christensen, H. I., Furuta, H., Walke, H., Fang, H., Ha, H., Mordatch, I., Radosavovic, I., Leal, I., Liang, J., AbouChakra, J., Kim, J., Drake, J., Peters, J., Schneider, J., Hsu, J., Bohg, J., Bingham, J., Wu, J., Gao, J., Hu, J., Wu, J., Wu, J., Tan, J., Oh, J., Wu, J., Lu, J., Yang, J., Salvador, J., Lim, J. J., Han, J., Wang, K., Rao, K., Pertsch, K., Hausman, K., Go, K., Gopalakrishnan, K., Goldberg, K., Byrne, K., Kawaharazuka, K., Black, K., Lin, K., Zhang, K., Ehsani, K., Lekkala, K., Ellis, K., Rana, K., Fang, K., Singh, K., Zeng, K., Hatch, K., Hsu, K., Itti, L., Chen, L. Y., Pinto, L., Fei-Fei, L., Tan, L., Fan, L., Ott, L., Lee, L., Weihs, L., Chen, M., Lepert, M., Memmel, M.,

Tomizuka, M., Itkina, M., Castro, M. G., Spero, M., Du,

- M., Ahn, M., Yip, M. C., Zhang, M., Ding, M., Heo, M., Srirama, M. K., Sharma, M., Kim, M. J., Kanazawa, M., Hansen, N., Heess, N., Joshi, N. J., Suenderhauf, N., Liu,
- N., Palo, N. D., Shafiullah, N., Mees, O., Kroemer, O., Bastani, O., Sanketi, P. R., Miller, P., Yin, P., Wohlhart, P., Xu, P., Fagan, P., Mitrano, P., Sermanet, P., Abbeel, P., Sundaresan, P., Chen, Q., Vuong, Q., Rafailov, R., Tian,

- R., Doshi, R., Mart´ın-Mart´ın, R., Baijal, R., Scalise, R., Hendrix, R., Lin, R., Qian, R., Zhang, R., Mendonca, R., Shah, R., Hoque, R., Julian, R., Bustamante, S., Kirmani,
- S., Levine, S., Lin, S., Moore, S., Bahl, S., Dass, S., Sonawani, S., Song, S., Xu, S., Haldar, S., Karamcheti, S., Adebola, S., Guist, S., Nasiriany, S., Schaal, S., Welker,

- S., Tian, S., Ramamoorthy, S., Dasari, S., Belkhale, S., Park, S., Nair, S., Mirchandani, S., Osa, T., Gupta, T., Harada, T., Matsushima, T., Xiao, T., Kollar, T., Yu, T., Ding, T., Davchev, T., Zhao, T. Z., Armstrong, T., Darrell,
- T., Chung, T., Jain, V., Vanhoucke, V., Zhan, W., Zhou, W., Burgard, W., Chen, X., Wang, X., Zhu, X., Geng, X., Liu, X., Liangwei, X., Li, X., Lu, Y., Ma, Y., Kim, Y., Chebotar, Y., Zhou, Y., Zhu, Y., Wu, Y., Xu, Y., Wang, Y., Bisk, Y., Cho, Y., Lee, Y., Cui, Y., Cao, Y., Wu, Y., Tang, Y., Zhu, Y., Zhang, Y., Jiang, Y., Li, Y., Li, Y., Iwasawa,

- Y., Matsuo, Y., Ma, Z., Xu, Z., Cui, Z., Zhang, Z., Fu,
- Z., and Lin, Z. Open x-embodiment: Robotic learning datasets and rt-x models, 2024.

Espeholt, L., Soyer, H., Munos, R., Simonyan, K., Mnih, V., Ward, T., Doron, Y., Firoiu, V., Harley, T., Dunning, I., et al. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. In International conference on machine learning, pp. 1407–1416. PMLR, 2018.

Firoozi, R., Tucker, J., Tian, S., Majumdar, A., Sun, J., Liu, W., Zhu, Y., Song, S., Kapoor, A., Hausman, K., et al. Foundation models in robotics: Applications, challenges, and the future. The International Journal of Robotics Research, pp. 02783649241281508, 2023.

Franklin, G. F., Powell, J. D., Workman, M. L., et al. Digital control of dynamic systems, volume 3. Addison-wesley Menlo Park, 1998.

Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. OPTQ: accurate quantization for generative pretrained transformers. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum? id=tcbBPnfwxS.

F¨urst, A., Rumetshofer, E., Lehner, J., Tran, V., Tang, F., Ramsauer, H., Kreil, D., Kopp, M., Klambauer, G., Bitto-

He, K., Chen, X., Xie, S., Li, Y., Doll´ar, P., and Girshick, R. B. Masked autoencoders are scalable vision learners. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pp. 15979–15988. IEEE, 2022. doi: 10.1109/CVPR52688.2022.01553.

Nemling, A., and Hochreiter, S. Cloob: Modern hopfield networks with infoloob outperform clip, 2022.

Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. CoRR, abs/2312.00752, 2023. doi: 10.48550/ARXIV.2312.00752. URL https:// doi.org/10.48550/arXiv.2312.00752.

Hessel, M., Modayil, J., van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., Horgan, D., Piot, B., Azar, M. G., and Silver, D. Rainbow: Combining improvements in deep reinforcement learning. ArXiv, 2017.

Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra,

- A., and R´e, C. Combining recurrent, convolutional, and continuous-time models with linear state space layers. In Ranzato, M., Beygelzimer, A., Dauphin, Y. N., Liang, P., and Vaughan, J. W. (eds.), Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pp. 572–585, 2021. URL https://proceedings. neurips.cc/paper/2021/hash/ 05546b0e38ab9175cd905eebcc6ebb76-Abstract. html.

Hinton, G. E., Vinyals, O., and Dean, J. Distilling the knowledge in a neural network. CoRR, abs/1503.02531, 2015. URL http://arxiv.org/abs/1503.02531.

Hochreiter, S. and Schmidhuber, J. Long short-term memory. Neural Comput., 9(8):1735–1780, 1997.

Hu, Y., Xie, Q., Jain, V., Francis, J., Patrikar, J., Keetha, N., Kim, S., Xie, Y., Zhang, T., Zhao, Z., et al. Toward general-purpose robots via foundation models: A survey and meta-analysis. arXiv preprint arXiv:2312.08782, 2023.

Gu, A., Goel, K., Gupta, A., and R´e, C. On the parameterization and initialization of diagonal state space models. Advances in Neural Information Processing Systems, 35: 35971–35983, 2022a.

Hussing, M., Mendez, J. A., Singrodia, A., Kent, C., and Eaton, E. Robotic manipulation datasets for offline compositional reinforcement learning. arXiv preprint arXiv:2307.07091, 2023.

Gu, A., Goel, K., and R´e, C. Efficiently modeling long sequences with structured state spaces. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022b. URL https://openreview.net/ forum?id=uYLFoz1vlAC.

Janner, M., Li, Q., and Levine, S. Offline reinforcement learning as one big sequence modeling problem. Advances in neural information processing systems, 34: 1273–1286, 2021.

Gu, J., Kirmani, S., Wohlhart, P., Lu, Y., Arenas, M. G., Rao, K., Yu, W., Fu, C., Gopalakrishnan, K., Xu, Z., Sundaresan, P., Xu, P., Su, H., Hausman, K., Finn, C., Vuong, Q., and Xiao, T. Rt-trajectory: Robotic task generalization via hindsight trajectory sketches, 2023.

Jia, X., Blessing, D., Jiang, X., Reuss, M., Donat, A., Lioutikov, R., and Neumann, G. Towards diverse behaviors: A benchmark for imitation learning with human demonstrations. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=6pPYRXKPpw.

Gu, X., Wang, Y.-J., and Chen, J. Humanoid-gym: Reinforcement learning for humanoid robot with zero-shot sim2real transfer, 2024.

Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In Dy, J. G. and Krause, A. (eds.), Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsm¨assan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pp. 1856–1865. PMLR, 2018.

Hafner, D., Lillicrap, T., Fischer, I., Villegas, R., Ha, D., Lee, H., and Davidson, J. Learning latent dynamics for planning from pixels. In International conference on machine learning, pp. 2555–2565. PMLR, 2019.

Jiang, Y., Gupta, A., Zhang, Z., Wang, G., Dou, Y., Chen, Y., Fei-Fei, L., Anandkumar, A., Zhu, Y., and Fan, L. Vima: General robot manipulation with multimodal prompts. arXiv preprint arXiv:2210.03094, 2022.

Jiang, Y., Gupta, A., Zhang, Z., Wang, G., Dou, Y., Chen, Y., Fei-Fei, L., Anandkumar, A., Zhu, Y., and Fan, L. Vima: General robot manipulation with multimodal prompts, 2023.

Jordan, M. I. Attractor dynamics and parallelism in a connectionist sequential machine, pp. 112–127. IEEE Press,

1990. ISBN 0818620153.

Kapturowski, S., Ostrovski, G., Dabney, W., Quan, J., and Munos, R. Recurrent experience replay in distributed reinforcement learning. In International Conference on Learning Representations, 2019. URL https: //openreview.net/forum?id=r1lyTjAqYX.

Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pp. 5156–5165. PMLR, 2020.

Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., et al. Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246, 2024.

Kim, S., Hooper, C., Wattanawong, T., Kang, M., Yan, R., Genc, H., Dinh, G., Huang, Q., Keutzer, K., Mahoney, M. W., et al. Full stack optimization of transformer inference: a survey. arXiv preprint arXiv:2302.14017, 2023.

Kirsch, L., Harrison, J., Freeman, C., Sohl-Dickstein, J., and Schmidhuber, J. Towards general-purpose in-context learning agents. In NeurIPS 2023 Workshop on Generalization in Planning, 2023.

Laskin, M., Lee, K., Stooke, A., Pinto, L., Abbeel, P., and Srinivas, A. Reinforcement learning with augmented data. ArXiv, 2004.14990, 2020.

Laskin, M., Wang, L., Oh, J., Parisotto, E., Spencer, S., Steigerwald, R., Strouse, D., Hansen, S., Filos, A., Brooks, E., et al. In-context reinforcement learning with algorithm distillation. arXiv preprint arXiv:2210.14215, 2022.

LeCun, Y., Denker, J. S., and Solla, S. A. Optimal brain damage. In Touretzky, D. S. (ed.), Advances in Neural Information Processing Systems 2, [NIPS Conference, Denver, Colorado, USA, November 27-30, 1989], pp. 598–605. Morgan Kaufmann, 1989. URL http://papers.nips.cc/paper/ 250-optimal-brain-damage.

- Lee, J. N., Xie, A., Pacchiano, A., Chandak, Y., Finn, C., Nachum, O., and Brunskill, E. Supervised pretraining can learn in-context reinforcement learning. arXiv preprint arXiv:2306.14892, 2023.
- Lee, K.-H., Nachum, O., Yang, M., Lee, L., Freeman, D., Xu, W., Guadarrama, S., Fischer, I., Jang, E., Michalewski, H., et al. Multi-game decision transformers. arXiv preprint arXiv:2205.15241, 2022.

Levine, S., Kumar, A., Tucker, G., and Fu, J. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In International Conference on Learning Representations, 2018.

Mandlekar, A., Nasiriany, S., Wen, B., Akinola, I., Narang, Y., Fan, L., Zhu, Y., and Fox, D. Mimicgen: A data generation system for scalable robot learning using human demonstrations, 2023.

McInnes, L., Healy, J., and Melville, J. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426, 2018.

Mendez, J. A., Hussing, M., Gummadi, M., and Eaton, E. Composuite: A compositional reinforcement learning benchmark. In Chandar, S., Pascanu, R., and Precup, D. (eds.), Conference on Lifelong Learning Agents, CoLLAs 2022, 22-24 August 2022, McGill University, Montr´eal, Qu´ebec, Canada, volume 199 of Proceedings of Machine Learning Research, pp. 982–1003. PMLR, 2022. URL https://proceedings.mlr.press/ v199/mendez22a.html.

Meng, L., Wen, M., Yang, Y., Le, C., Li, X., Zhang, W., Wen, Y., Zhang, H., Wang, J., and Xu, B. Offline pretrained multi-agent decision transformer: One big sequence model conquers all starcraftii tasks. arXiv preprint arXiv:2112.02845, 2021.

Merrill, W., Petty, J., and Sabharwal, A. The illusion of state in state-space models. CoRR, abs/2404.08819, 2024. doi: 10.48550/ARXIV.2404.08819. URL https:// doi.org/10.48550/arXiv.2404.08819.

Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., et al. Mixed precision training. arXiv preprint arXiv:1710.03740, 2017.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., , and Hassabis, D. Human-level control through deep reinforcement learning. Nature, 518(7540): 529–533, 2015. doi: 10.1038/nature14236.

Ni, T., Ma, M., Eysenbach, B., and Bacon, P.-L. When do transformers shine in rl? decoupling memory from credit assignment. Advances in Neural Information Processing Systems, 36, 2024.

Octo Model Team, Ghosh, D., Walke, H., Pertsch, K., Black, K., Mees, O., Dasari, S., Hejna, J., Kreiman, T., Xu, C., Luo, J., Tan, Y. L., Sanketi, P., Vuong, Q., Xiao, T., Sadigh, D., Finn, C., and Levine, S. Octo: An opensource generalist robot policy, 2024.

Orvieto, A., Smith, S. L., Gu, A., Fernando, A., G¨ulc¸ehre, C¸., Pascanu, R., and De, S. Resurrecting recurrent neural networks for long sequences. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pp. 26670–26698. PMLR, 2023. URL https://proceedings.mlr.press/ v202/orvieto23a.html.

Ota, T. Decision mamba: Reinforcement learning via sequence modeling with selective state spaces. arXiv preprint arXiv:2403.19925, 2024.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.

Patil, V., Hofmarcher, M., Dinu, M., Dorfer, M., Blies, P. M., Brandstetter, J., Arjona-Medina, J. A., and Hochreiter, S. Align-rudder: Learning from few demonstrations by reward redistribution. In Chaudhuri, K., Jegelka, S., Song, L., Szepesv´ari, C., Niu, G., and Sabato, S. (eds.), International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pp. 17531–17572. PMLR, 2022.

Raad, M. A., Ahuja, A., Barros, C., Besse, F., Bolt, A., Bolton, A., Brownfield, B., Buttimore, G., Cant, M., Chakera, S., et al. Scaling instructable agents across many simulated worlds. arXiv preprint arXiv:2404.10179, 2024.

Radford, A., Narasimhan, K., Salimans, T., Sutskever, I., et al. Improving language understanding by generative pre-training. 2018.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In Meila, M. and Zhang, T. (eds.), Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pp. 8748– 8763. PMLR, 2021.

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., and Sutskever, I. Robust speech recognition via large-

scale weak supervision. arXiv preprint arXiv:2212.04356, 2022.

Raparthy, S. C., Hambro, E., Kirk, R., Henaff, M., and Raileanu, R. Generalization to new sequential decision making tasks with in-context learning, 2023.

Reed, S. E., Zolna, K., Parisotto, E., Colmenarejo, S. G., Novikov, A., Barth-Maron, G., Gimenez, M., Sulsky, Y., Kay, J., Springenberg, J. T., Eccles, T., Bruce, J., Razavi, A., Edwards, A., Heess, N., Chen, Y., Hadsell, R., Vinyals, O., Bordbar, M., and de Freitas, N. A generalist agent. CoRR, abs/2205.06175, 2022. doi: 10.48550/arXiv.2205. 06175.

Salzmann, T., Kaufmann, E., Arrizabalaga, J., Pavone, M., Scaramuzza, D., and Ryll, M. Real-time neural mpc: Deep learning model predictive control for quadrotors and agile robotic platforms. IEEE Robotics and Automation Letters, 8(4):2397–2404, 2023.

Schmidhuber, J. Learning to control fast-weight memories: An alternative to dynamic recurrent networks. Neural Comput., 4(1):131–139, 1992. doi: 10.1162/NECO. 1992.4.1.131. URL https://doi.org/10.1162/ neco.1992.4.1.131.

Schmidhuber, J. Reinforcement learning upside down: Don’t predict rewards–just map them to actions. arXiv preprint arXiv:1912.02875, 2019.

Schmidinger, N., Schneckenreiter, L., Seidl, P., Schimunek, J., Luukkonen, S., Hoedt, P.-J., Brandstetter, J., Mayr, A., Hochreiter, S., and Klambauer, G. Bio-xlstm: Generative modeling, representation and in-context learning of biological and chemical sequences. Under reveiw, 2024.

Schmidt, D. and Schmied, T. Fast and data-efficient training of rainbow: an experimental study on atari. arXiv preprint arXiv:2111.10247, 2021.

Schmied, T., Hofmarcher, M., Paischer, F., Pascanu, R., and Hochreiter, S. Learning to modulate pre-trained models in rl. Advances in Neural Information Processing Systems, 36, 2024a.

Schmied, T., Paischer, F., Patil, V., Hofmarcher, M., Pascanu, R., and Hochreiter, S. Retrieval-augmented decision transformer: External memory for in-context rl. arXiv preprint arXiv:2410.07071, 2024b.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. ArXiv, 2018.

Schwarzer, M., Ceron, J. S. O., Courville, A., Bellemare, M. G., Agarwal, R., and Castro, P. S. Bigger, better, faster: Human-level atari with human-level efficiency.

In International Conference on Machine Learning, pp. 30365–30380. PMLR, 2023.

Schweighofer, K., Dinu, M.-c., Radler, A., Hofmarcher, M., Patil, V. P., Bitto-Nemling, A., Eghbal-zadeh, H., and Hochreiter, S. A dataset perspective on offline reinforcement learning. In Conference on Lifelong Learning Agents, pp. 470–517. PMLR, 2022.

Shang, J., Kahatapitiya, K., Li, X., and Ryoo, M. S. Starformer: Transformer with state-action-reward representations for visual reinforcement learning. In European Conference on Computer Vision, pp. 462–479. Springer, 2022.

Siebenborn, M., Belousov, B., Huang, J., and Peters, J. How crucial is transformer in decision transformer? arXiv preprint arXiv:2211.14655, 2022.

Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., van den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., Dieleman, S., Grewe, D., Nham, J., Kalchbrenner, N., Sutskever, I., Lillicrap, T. P., Leach, M., Kavukcuoglu, K., Graepel, T., and Hassabis, D. Mastering the game of Go with deep neural networks and tree search. Nature, 529(7587):484–489, 2016. doi: 10.1038/nature16961.

Smith, J. T. H., Warrington, A., and Linderman, S. W. Simplified state space layers for sequence modeling. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.

net/forum?id=Ai8Hw3AXqks.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1):1929–1958, 2014.

Tassa, Y., Doron, Y., Muldal, A., Erez, T., Li, Y., de Las Casas, D., Budden, D., Abdolmaleki, A., Merel, J., Lefrancq, A., Lillicrap, T. P., and Riedmiller, M. A. Deepmind control suite. CoRR, abs/1801.00690, 2018.

Tay, Y., Dehghani, M., Abnar, S., Shen, Y., Bahri, D., Pham, P., Rao, J., Yang, L., Ruder, S., and Metzler, D. Long range arena: A benchmark for efficient transformers. arXiv preprint arXiv:2011.04006, 2020.

Todorov, E., Erez, T., and Tassa, Y. MuJoCo: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 5026–5033, October 2012a. doi: 10.1109/IROS.2012. 6386109.

Todorov, E., Erez, T., and Tassa, Y. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 5026–5033. IEEE, 2012b.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., Bikel, D., Blecher, L., Canton-Ferrer, C., Chen, M., Cucurull, G., Esiobu, D., Fernandes, J., Fu, J., Fu, W., Fuller, B., Gao, C., Goswami, V., Goyal, N., Hartshorn, A., Hosseini, S., Hou, R., Inan, H., Kardas, M., Kerkez, V., Khabsa, M., Kloumann, I., Korenev, A., Koura, P. S., Lachaux, M., Lavril, T., Lee, J., Liskovich, D., Lu, Y., Mao, Y., Martinet, X., Mihaylov, T., Mishra, P., Molybog, I., Nie, Y., Poulton, A., Reizenstein, J., Rungta, R., Saladi, K., Schelten, A., Silva, R., Smith, E. M., Subramanian, R., Tan, X. E., Tang, B., Taylor, R., Williams, A., Kuan, J. X., Xu, P., Yan, Z., Zarov, I., Zhang, Y., Fan, A., Kambadur, M., Narang, S., Rodriguez, A., Stojnic, R., Edunov, S., and Scialom, T. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288, 2023. doi: 10.48550/ARXIV.2307.09288. URL https: //doi.org/10.48550/arXiv.2307.09288.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, l., and Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Vinyals, O., Babuschkin, I., Czarnecki, W. M., Mathieu, M., Dudzik, A., Chung, J., Choi, D. H., Powell, R., Ewalds, T., Georgiev, P., Oh, J., Horgan, D., Kroiss, M., Danihelka, I., Huang, A., Sifre, L., Cai, T., Agapiou, J. P., Jaderberg, M., Vezhnevets, A. S., Leblond, R., Pohlen, T., Dalibard, V., Budden, D., Sulsky, Y., Molloy, J., Paine, T. L., G¨ul¸cehre, C¸., Wang, Z., Pfaff, T., Wu, Y., Ring, R., Yogatama, D., W¨unsch, D., McKinney, K., Smith, O., Schaul, T., Lillicrap, T. P., Kavukcuoglu, K., Hassabis, D., Apps, C., and Silver, D. Grandmaster level in starcraft II using multi-agent reinforcement learning. Nat., 575(7782):350– 354, 2019. doi: 10.1038/s41586-019-1724-z.

Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., and Anandkumar, A. Voyager: An openended embodied agent with large language models, 2023.

Wang, K., Zhao, H., Luo, X., Ren, K., Zhang, W., and Li, D. Bootstrapped transformer for offline reinforcement learning. arXiv preprint arXiv:2206.08569, 2022.

Wen, C., Lin, J., Darrell, T., Jayaraman, D., and Gao, Y. Fighting copycat agents in behavioral cloning from observation histories. Advances in Neural Information Processing Systems, 33:2564–2575, 2020.

Wolczyk, M., Zajkac, M., Pascanu, R., Kuci´nski, L., and Milo´s, P. Continual world: A robotic benchmark for

continual reinforcement learning. Advances in Neural Information Processing Systems, 34:28496–28510, 2021.

Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., and Finn, C. Gradient surgery for multi-task learning. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020a.

Yu, T., Quillen, D., He, Z., Julian, R., Hausman, K., Finn, C., and Levine, S. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on robot learning, pp. 1094–1100. PMLR, 2020b.

Zheng, Q., Zhang, A., and Grover, A. Online decision transformer. In Chaudhuri, K., Jegelka, S., Song, L., Szepesv´ari, C., Niu, G., and Sabato, S. (eds.), International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pp. 27042–27059. PMLR, 2022.

Zhu, G., Lin, Z., Yang, G., and Zhang, C. Episodic reinforcement learning with associative memory. In International Conference on Learning Representations, 2020.

Zhu, L., Liao, B., Zhang, Q., Wang, X., Liu, W., and Wang, X. Vision mamba: Efficient visual representation learning with bidirectional state space model. CoRR, abs/2401.09417, 2024. doi: 10.48550/ARXIV. 2401.09417. URL https://doi.org/10.48550/ arXiv.2401.09417.

Appendix

- A. Reproducibility Statement

We make the code base used for our experiments publicly available and release the datasets we generated. Both are available at: https://github.com/ml-jku/LRAM. We describe the environments we use for our experiments and provide dataset statistics in Appendix B. Furthermore, in Appendix C, we provide implementation details for all methods and a list of hyperparameters used for our experiments. In Appendix D, we present additional figures that accompany our results in the main text (e.g., all model sizes). Finally, in Appendices E and F, we provide further details on the conducted ablation studies and the embedding space analysis, respectively.

- B. Environments & Datasets

- B.1. General

We compile a large-scale dataset comprising 432 tasks from six domains, 3.4M trajectories, and 894M transitions in total (see Table 2). A key motivation behind our dataset compilation is the scarcity of suitable datasets that span many simulated tasks. To address this and to enable a robust comparison of different sequence model architectures, we aimed to assemble a collection of datasets that span as many tasks as possible. In particular, we focused on trajectories in simulated environments rather than real-world trajectories (Embodiment Collaboration et al., 2024), to enable faster iteration cycles. To facilitate usability for future works, we consider standard benchmarks that are widely adopted by the community (e.g., Atari, Meta-World).

We release our data pipeline and generated dataset, and hope that they can serve as a solid basis for future research on multi-task agents. To enable fast and targeted data-loading, every trajectory is stored in a separate hdf5 file. We trade off some data-loading speed for disk space efficiency by compressing trajectories that contain image-based observations.

### B.2. Atari

The Arcade Learning Environment (ALE) (Bellemare et al., 2013) is the standard benchmark for evaluating RL agents and consists of 57 Atari games. Input observations in Atari are RGB images, but as is standard practice, we gray-scale and crop frames (|S| = 1 × 64 × 64). There are 18 discrete actions across all 57 Atari games (|A| = 18), but individual games may use only a subset of these actions. Furthermore, we adopt the standard Atari recipe as used in prior works, including a frame skip of 4, maximum number of no-ops of 30, resetting on life loss, and reward clipping to [−1,1] (Mnih et al., 2015; Hessel et al., 2017).

Tasks. Similar to Lee et al. (2022), we assign 41 games to the training set and 5 additional tasks to the hold-out set. The 41 training tasks include:

amidar, assault, asterix, atlantis, bank-heist, battle-zone, beam-rider, boxing, breakout, carnival, centipede, chopper-command, crazy-climber, demon-attack, double-dunk, enduro, fishing-derby, freeway, frostbite, gopher, gravitar, hero, ice-hockey, jamesbond, kangaroo, krull, kung-fu-master, name-this-game, phoenix, pooyan, qbert, riverraid, road-runner, robotank, seaquest, time-pilot, up-n-down, video-pinball, wizard-of-wor, yars-revenge, zaxxon

The 5 hold-out tasks include: alien, pong, ms-pacman, space-invaders, star-gunner

Dataset. For Atari, we leverage the DQN-Replay dataset released by Agarwal et al. (2020). The dataset contains the trajectories seen over the entire training of the DQN agent (50M frames). We extract a subset of the last 5M transitions for every task, amounting to 205M transitions in total for the 41 training tasks. The number of episodes, the episode lengths, and total achieved rewards vary across tasks, as shown in Table 2.

### B.3. Meta-World

The Meta-World benchmark (Yu et al., 2020a) consists of 50 manipulation tasks using a Sawyer robotic arm, ranging from opening or closing windows to pressing buttons. Meta-World is based on the MuJoCo physics engine (Todorov et al., 2012a). Observations in Meta-World are 39-dimensional continuous vectors (|S| = 1 × 64 × 39), and actions are represented by 6

Table 2. Atari Dataset Statistics.

Task # of Trajectories Mean Length Mean Return

amidar 1813 2753 145 pooyan 2773 1800 176 frostbite 5218 766 18 video-pinball 1023 3902 266 wizard-of-wor 3059 1314 15 chopper-command 5452 738 18 breakout 3780 1300 39 phoenix 3307 1509 49 asterix 5250 951 55 enduro 571 8720 636 kung-fu-master 1775 2812 131 hero 3022 1345 168 assault 3782 1170 77 demon-attack 1649 2431 116 qbert 3939 1138 155 jamesbond 2841 1758 11 bank-heist 4146 1204 62 up-n-down 3246 1538 99 centipede 6879 582 81 boxing 4796 1041 63 battle-zone 1933 2134 15 name-this-game 988 5049 389 zaxxon 2561 1950 12 beam-rider 1232 3248 77 time-pilot 3886 1029 11 ice-hockey 1465 3407 -6 riverraid 2645 1512 143 krull 3032 1319 528 gopher 1817 2338 185 freeway 2438 2048 33 seaquest 2807 1779 150 double-dunk 1774 2815 0 road-runner 3308 1217 135 atlantis 186 26349 1394 gravitar 6187 646 1 yars-revenge 4094 1036 96 crazy-climber 1105 3954 572 kangaroo 1787 2792 50 fishing-derby 2737 1825 0 carnival 21131 194 37 robotank 747 6652 56

Average 3321 2734 153

continuous dimensions (|A| = 18) in range [−1,1]. All tasks share a common action and state space. Following Wolczyk et al. (2021) and Schmied et al. (2024a), we limit the episode lengths to 200 interactions.

Tasks. We follow Yu et al. (2020a) and split the 50 Meta-World tasks into 45 training tasks (MT45) and 5 evaluation tasks (MT5). The 45 training tasks are:

reach, push, pick-place, door-open, drawer-open, drawer-close, button-press-topdown, peg-insert-side, window-open, window-close, door-close, reach-wall, pick-place-wall, push-wall, button-press, button-press-topdown-wall, button-press-wall, peg-unplug-side, disassemble, hammer, plate-slide, plate-slide-side, plate-slide-back, plate-slide-back-side, handle-press, handle-pull, handle-press-side, handle-pull-side, stick-push, stick-pull, basketball,soccer, faucet-open, faucet-close, coffee-push, coffee-pull, coffee-button, sweep, sweep-into, pick-out-of-hole, assembly, shelf-place, push-back, lever-pull, dial-turn

The 5 evaluation tasks are: bin-picking, box-close, door-lock, door-unlock, hand-insert

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

(a) IIWA (b) Panda (c) Jaco (d) Gen3

Figure 8. Illustration of the four supported robot arms in Composuite (Mendez et al., 2022).

Dataset. For Meta-World, we use the datasets released by (Schmied et al., 2024a), which contain 2M transitions per task and consequently 90M transitions in total for the training set. All episodes last for 200 environment interaction steps, and consequently, there are 10K episodes for every task. For detailed dataset statistics per task, we refer to their publication.

### B.4. DMControl

The DMControl benchmark (Tassa et al., 2018) consists of 30 different robotic tasks. Unlike Meta-World, the benchmark contains robots with different morphologies instead of a single common Sawyer arm. Due to the different robot morphologies, the state and action spaces vary across tasks (3 ≤ |S| ≤ 24, 1 ≤ |A| ≤ 6), with all actions in the range [−1,1].

Tasks. We do not use all 30 tasks contained in the DMControl benchmark, but select 16 of the 30 tasks that have been used in prior works (Hafner et al., 2019; Schmied et al., 2024a;b), which we refer to as DMC11 and DMC5, respectively. The 11 training tasks are:

finger-turn easy, fish-upright, hopper-stand, point mass-easy, walker-stand, walker-run, ball in cup-catch, cartpole-swingup, cheetah-run, finger-spin, reacher-easy

The 5 evaluation tasks are: cartpole-balance, finger-turn hard, pendulum-swingup, reacher-hard, walker-walk

Dataset. For DMControl, we generate 10M transitions per task by training task-specific SAC (Haarnoja et al., 2018) agents, using the same setup as Schmied et al. (2024a). Episodes in all DMControl tasks last for 1000 environment steps, and per time-step a maximum reward of +1 can be achieved, which results in a maximum reward of 1000 per episode. Consequently, our training set contains 10K episodes per task, amounting to 110K episodes and 110M transitions in total across all tasks. We list the dataset statistics for all 11 tasks in Table 3.

Table 3. DMControl Data statistics.

### Task # of Trajectories Mean Length Mean Return

point mass easy 10K 1K 851 cheetah run 10K 1K 385 walker run 10K 1K 230 ball in cup catch 10K 1K 969 hopper stand 10K 1K 460 walker stand 10K 1K 939 finger turn easy 10K 1K 954 reacher easy 10K 1K 938 cartpole swingup 10K 1K 817 fish upright 10K 1K 815 finger spin 10K 1K 966

Average 19628 152 8.2

### B.5. Composuite

The Composuite benchmark (Mendez et al., 2022) is a robotics benchmark for grasping and object manipulation. The benchmark is implemented on top of robotsuite (Zhu et al., 2020), which in turn leverages the MuJoCo simulator under the hood (Todorov et al., 2012b). Composuite contains a mix of 4 simulated robot arms: IIWA, Jaco, Gen3, and Panda (see Figure 8). All arms share a common state and action space containing 93 continuous state dimensions and 8 continuous action dimensions, respectively (|S| = 93, |A| = 8).

Tasks. CompoSuite is designed as a compositional multi-task benchmark for RL, in which a particular robot manipulates a particular object given an objective, while avoiding obstacles. Overall, there are 4 robot arms, 4 objects, 4 obstacles, and 4 task objectives. This results in 256 possible robot/object/objective/obstacle combinations. For our experiments, we assign 240 tasks to the training set and use the remaining 16 tasks as a hold-out set (Panda and Object Wall) combinations. For a list of all 256 tasks, we refer to Mendez et al. (2022).

Dataset. For Composuite, we leverage the datasets released by Hussing et al. (2023). For every task, we select 2000 episodes, which last on average for 500 steps. This amounts to 1M transitions per task, and 240M transitions across all 240 training tasks. For dataset statistics, we refer to Hussing et al. (2023).

### B.6. Mimicgen

Similar to Composuite, Mimicgen (Mandlekar et al., 2023) is based on robosuite and the MuJoCo simulator. Mimicgen is designed for automatically synthesizing large-scale datasets from only a handful of human demonstrations. Observations in Mimicgen can be represented as images (from multiple cameras) or low-dimensional continuous states. For our experiments, we opt for the low-dimensional state representation to simplify learning. Therefore, observations and actions are represented by 37-dimensional and 7-dimensional continuous vectors, respectively (|S| = 37, |A| = 7). Similar to Composuite, Mimicgen supports 4 different robot arms: Panda, IIWA, Sawyer, and UR5e (see Figure 9).

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

(a) IIWA (b) Panda (c) Sawyer (d) UR5e

- Figure 9. Illustration of the four supported robot arms in Mimicgen (Mandlekar et al., 2023) solving the stack-three task.

Tasks. Mimicgen consists of 24 diverse tasks, including stacking blocks, reassembling objects, and even long-horizon tasks like coffee preparation. These 24 tasks can be performed with the four supported robot arms, amounting to 96 tasks in total.

Dataset. Mandlekar et al. (2023) released datasets for the 24 tasks using the default robot arm Panda. To increase the dataset diversity, we additionally generated data for the remaining 3 robot arms. However, not all data generation runs produce successful trajectories, and we discard the ones with too few successful trajectories. Our final dataset for Mimicgen contains 83 training and 2 evaluation tasks. For each task, we collect 1000 successful demonstrations (we do not include unsuccessful trajectories). Episode lengths vary across tasks, ranging from 260 to 850 environment steps.

### B.7. Procgen

The Procgen benchmark consists of 16 procedurally-generated video games (Cobbe et al., 2020a). Observations in Procgen are RGB images of dimension 3 × 64 × 64. However, for training efficiency, we apply gray-scaling to image observations (|S| = 1 × 64 × 64). All 16 environments share a common action space of 15 discrete actions (|A| = 16). Procgen is designed to test the generalization abilities of RL agents. Consequently, procedural generation is employed to randomize background and colors, while retaining the game dynamics.

Tasks. Following prior works (Raparthy et al., 2023; Schmied et al., 2024b), we assign 12 and 4 tasks to the training and hold-out sets, respectively. The 12 training tasks are:

bigfish, bossfight, caveflyer, chaser, coinrun, dodgeball, fruitbot, heist, leaper, maze, miner, starpilot

The 4 hold-out tasks are: climber, ninja, plunder, jumper

Dataset. We leverage the datasets released by (Schmied et al., 2024b), which contain 20M transitions per task. The datasets were generated by recording all transitions observed by training RL agents for 25M steps, followed by uniform subsampling to 20M transitions. Consequently, the dataset contains mixed quality trajectories ranging from random (beginning of training) to expert (end of training). We list the dataset statistics for all 16 tasks in Table 4.

Table 4. Procgen Data statistics.

### Task # of Trajectories Mean Length Mean Return

bigfish 82835 230 6.251 bossfight 112459 141 1.946 caveflyer 151694 105 7.745 chaser 93612 212 3.248 coinrun 261117 51 9.473 dodgeball 144364 137 2.884 fruitbot 73653 270 16.094 heist 101361 196 8.405 leaper 296084 67 4.446 maze 482245 41 9.432 miner 288818 68 11.8 starpilot 96468 206 17.3

Average 182059 144 8.3

## C. Experimental & Implementation Details

### C.1. Training & Evaluation

In our experiments, we compare two variants of xLSTM, Mamba and DT. For our main experiments in Section 4.2, we train all models for 200K updates, and evaluate after every 50K update steps. We report the mean and 95% confidence intervals over three seeds in our experiments, as suggested by Agarwal et al. (2021). For every evaluation task, we take the average of 3 evaluation seeds.

We train our agents with a batch size of 128 and gradient accumulation across the 6 domains, such that every domain is represented with the same proportion. This is to compare Consequently, the effective batch size is 768. We use a learning rate of 1e−4, 4000 linear warm-up steps followed by a cosine decay to 1e−6, and train using the AdamW optimizer (Loshchilov & Hutter, 2018). In addition, we employ gradient clipping of 0.25, weight decay of 0.01 for all models. We do not employ Dropout, as is standard practice in DTs, as we found that it negatively affects performance (see Section 4.3). We use separate reward scales of 200, 100, and 20 for Meta-World, DMControl, and Atari, respectively. Furthermore, for all domains, we set the target return to the maximum return achieved for a particular task in the training datasets. This is particularly useful for domains where the maximum returns differ heavily across tasks (e.g., Atari). We list all hyperparameters in Table 5.

We want to highlight that we opt to represent every domain with approximately equal proportion in every update step. This is, because we aim to study how the different backbones perform across domains, rather than optimizing performance on specific domains. However, to better understand the impact of the data ratios on multitask capabilities, we believe it would be interesting to study other data ratios in future work. Varying the data ratios would, for example, allow studying potential interferences between the 432 tasks.

### C.2. Context Lengths

By default, we train all models with a context length C = 50 timesteps. For every timestep, there are three tokens (s/rt/r), and consequently, the effective context length is 150. We found that performance improves for longer context lengths (see Section E.1), but limit our experiments to C = 50 to reduce the computational cost.

Table 5. Hyperparameters for LRAM.

Parameter Value Gradient steps 200K Evaluation frequency 50K Evaluation episodes 5 Optimizer AdamW Batch size 128 Gradient accumulation 6 Lr schedule Linear warm-up + Cosine Warm-up steps 4000 Learning rate 1e-4 → 1e-6 Weight decay 0.01 Gradient clipping 0.25 Dropout 0.2 Context len (timesteps) 50 Reward scale per-domain Target return per-task

### C.3. Model Architectures

We train models across 4 model sizes: 16M, 48M, 110M, and 206M. We follow Lee et al. (2022) in selecting the number of layers and hidden dimensions. For xLSTM and Mamba, we use twice the number of layers blocks to match the number of parameters of the Transformer (Beck et al., 2024; Gu et al., 2024) (see Table 6) For our xLSTM [7:1] variant, which contains sLSTM blocks, we strive to maintain the same ratio as proposed by Beck et al. (2024). Not all our model sizes are divisible by 8, and only the 16M and 110M models exhibit the exact 7:1 ratio of mLSTM to sLSTM blocks. For consistency, however, we maintain the same notation as (Beck et al., 2024). We place sLSTM blocks at positions [1], [1, 3], [1, 3], and [1, 3, 5] for the 16M, 48M, 110M, 206M, respectively.

Across backbones, we use linear layers to encode continuous states, reward returns-to-go, similar to Chen et al. (2021). The maximal state dimension across continuous control environments is 204 in our experiments. To use a shared linear embedding layer for continuous states, we pad states that have a lower number of dimensions to 204 dimensions using zeros. To encode image inputs on visual domains, we use the IMPALA-CNN proposed by Espeholt et al. (2018) and adopted by previous works on Procgen (Cobbe et al., 2020a) and Atari (Schmidt & Schmied, 2021; Schwarzer et al., 2023). Consequently, we do not make use of discretization of continuous states or patchification of images. This design choice significantly reduces the sequence length to only three tokens per time-step (see Appendix C.2) and consequently results in faster inference.

For continuous actions, we make use of discretization and discretize of every action dimension into 256 uniformly-spaced bins, similar to Reed et al. (2022) and Brohan et al. (2023b). We experimented with lower/higher numbers of bins, but did not observe a benefit beyond 256 bins. Consequently, this resolution is sufficient for the environments we consider. We use a shared action head to predict the action bins of all continuous dimensions jointly. The maximum number of continuous action dimensions is 8 in our experiments, and consequently, the number of discrete action classes is 2048. In addition, there are 18 discrete actions originating from Atari and Procgen. Therefore, our action head learns to predict the correct action among the 2066 discrete classes. While different environments may have different action dimensions, the model predicts all action dimensions jointly. At inference time, the number of action dimensions of the current environment is known, and we extract the respective dimensions from the joint predictions. We opt for the shared action head representation, as this further speeds up inference and does not require autoregressive action prediction.

For the Transformer baseline, we use global positional embeddings similar to Chen et al. (2021). For the recurrent backbones, we do not make use of positional encodings.

Table 6. Model Sizes.

### Model Layers Hidden Dim Heads Parameters

Transformer 4 512 8 16M Transformer 6 768 12 48M Transformer 8 1024 16 110M Transformer 10 1280 20 206M

Mamba 8 512 - 16M Mamba 12 768 - 48M Mamba 16 1024 - 110M Mamba 20 1280 - 206M

xLSTM 8 512 4 16M xLSTM 12 768 4 48M xLSTM 16 1024 4 110M xLSTM 20 1280 4 206M

### C.4. Hardware & Training Times

We train all our models on a server equipped with 4 A100 GPUs. We use distributed data parallel to distribute the workload, as supported in PyTorch (Paszke et al., 2019). Training times range from 5 hours for the smallest DT model to 30 hours for the largest Mamba model. Throughout all our experiments, we use mixed precision training (Micikevicius et al., 2017) as supported in PyTorch to speed up training time.

We evaluate our models after every 50K steps. However, periodically evaluating the trained agents on all 432 tasks sequentially is time-consuming. Therefore, we perform parallel evaluation with 4 processes at a time. For multi-GPU setups, we distribute the evaluation workload among the available GPUs. For example, with 4 available GPUs and 4 evaluation processes per GPU, 16 environments are evaluated simultaneously. Consequently, the total evaluation time for all 432 tasks ranges from 18 minutes for the smallest DT model to roughly 2 hours for the largest Mamba model.

## D. Additional Results

### D.1. Training Tasks

In Figures 10 and 11, we report the normalized scores obtained per domain and the average learning curves across tasks for all four model sizes.

[Figure 25]

[Figure 26]

- (a) 16M

[Figure 27]

- (b) 48M

[Figure 28]

- (c) 110M

[Figure 29]

- (d) 206M

- Figure 10. Normalized scores per-domain all four model sizes: 16M, 48M, 110M, and 206M. For Meta-World, DMControl, Mimicgen, Composuite, and Procgen we report data-normalized scores, for Atari we report human-normalized scores.

In Figure 12, we report the training perplexity on the 432 training tasks over 200K updates. Here, we observe that the training perplexity behaves similarly to the validation perplexity. This is expected, as our models see most transitions only a single time (see Table 2 for the number of repetitions per domain).

Furthermore, we report the scaling curves with an additional model size of 408M parameters in Figure 13. Due to the high computational cost of the 408M models, we were currently only able to conduct a single run for this size. However, we aim to provide further empirical evidence for these model sizes in future work.

[Figure 31]

[Figure 32]

(a) 16M (b) 48M

[Figure 33]

[Figure 34]

(c) 110M (d) 206M

Figure 11. Learning curves for all four model sizes, 16M, 48M, 110M, and 206M, on the training tasks.

### D.2. Hold-out Tasks

- In Figure 14, we show the zero-shot evaluation performance on the hold-out tasks 14. We want to highlight that the performance declines for all methods and model sizes compared to performance on training tasks. This is because hold-out tasks exhibit severe shifts in state-spaces, action-spaces, and reward functions.

D.3. Fine-Tuning

- In Figure 15, we present the fine-tuning evaluation performance on the held-out tasks. We compare xLSTMs trained from scratch against xLSTMs initialized with the pre-trained weights. We do observe consistent improvement of the pre-trained models over the models trained from scratch. While we train on a substantial number of environments, the total amount of data used is still only a fraction of that employed in training other large-scale models, such as LLMs. Consequently, we do not observe comparable few-shot generalization. However, we anticipate that few-shot generalization capabilities will emerge as we increase both data volume and model size.

### D.4. In-context Learning

We assess the ICL abilities of modern recurrent architectures on the Dark-Room environment considered in prior works on in-context RL (Laskin et al., 2022; Lee et al., 2023; Schmied et al., 2024b). In Dark-Room, the agent is located in a dark room. The task is to navigate to an invisible goal location in that dark room. The state is partially observable, as the agent only observes its own x-y position on the grid (|S| = 2). The action space consists of 5 discrete actions: move up, move down, move left, move right, stay (|A| = 5). Upon reaching the goal location, the agent receives a reward of +1 for every step in the episode it resides in the goal location. Consequently, the agent first has to explore the room to find the goal. Once the goal location is found (as indicated by the positive reward), the agent can exploit this knowledge. Given a multi-episodic context, the agent should be able to exploit information contained in the previous trials (e.g., exploiting one path vs. avoiding another).

In our experiments, the Dark-Room is a 10 × 10 grid and episodes last for 100 steps, starting in the top left corner of the

[Figure 36]

(a) Training Perplexity

- Figure 12. Scaling comparison. We compare xLSTM, Mamba, DT in four model sizes: 16M, 48M, 110M, and 206M parameters. We show the training perplexity on the training dataset to evaluate the sequence prediction performance.

[Figure 37]

[Figure 38]

(a) Sequence prediction (b) Environment interaction

[Figure 39]

- Figure 13. Scaling comparison with additional 408M parameter models. We show the (a) validation perplexity on the hold-out datasets, and (b) normalized scores obtained from evaluating in the training task environments, averaged over all 6 domains.

grid. We adopt the same experiment setup as Schmied et al. (2024b) and leverage their datasets. We train 16M parameter agents on datasets from 80 randomly selected goal locations in the grid. The datasets contain 100K transitions per task and are obtained by training task-specific PPO (Schulman et al., 2018) agents. Then, we evaluate the in-context abilities of our agents on 20 hold-out goal locations. During evaluation, the agent is given 40 episodes to interact with the environment, which we refer to as ICL-trials. Furthermore, we adopt the AD (Laskin et al., 2022) framework for training our agents with a multi-episodic context. We use the same sequence representation as used in our main experiments, consisting of states, returns-to-go (target return set to 80 during evaluation), and rewards. Note that this differs from the sequence representation used by Laskin et al. (2022). We set the context length for all agents to the equivalent of two episodes, which amounts to 200 timesteps in total.

- In Figure 16, we report the ICL performance over the 40 ICL trials on (a) 80 training locations and (b) 20 hold-out locations for the 4 different backbones considered in this work. We observe that the recurrent backbones attain considerably higher scores than the Transformer backbone. Furthermore, we find that xLSTM [7:1] attains the highest overall scores, which we attribute to the state-tracking abilities (Merrill et al., 2024) of sLSTM blocks. We aim to explore the ICL abilities of modern recurrent backbones more in future work.

[Figure 41]

- Figure 14. Scaling comparison. Zero-shot performance on hold-out tasks at four model sizes, 16M, 48M, 110M, and 206M. Note that performance declines for all methods and model sizes compared to performance on training tasks. This is because hold-out tasks exhibit severe shifts in state-spaces, action-spaces, and reward functions.

[Figure 42]

[Figure 43]

- Figure 15. Fine-tune performance on hold-out tasks. We compare the performance of a pretrained xLSTM against an xLSTM trained from scratch, both with 16 million parameters. We select the top 5% of trajectories from our held-out tasks based on performance and use this subset to fine-tune the models. We perform 25K update steps during fine-tuning and show the normalized scores, averaged across held-out tasks from each domain.

### D.5. Inference Time Comparisons

We empirically examine the difference in inference speed between of our models. Similar to De et al. (2024), we report both latency and throughput. For real-time applications, latency is the more important dimension, and therefore, we focus our analysis on latency.

- D.5.1. LATENCY

In Figures 17 and 18, we report the latencies for DT and xLSTM with the same number of layer blocks as DT, and twice the number of layer blocks as DT, respectively. We conduct our comparison for two different batch sizes and across varying sequence lengths.

- D.5.2. THROUGHPUT In Figures 19 and 20, we similarly report the attained throughput for DT and xLSTM with the same number of layer blocks

- as DT, and twice the number of layer blocks as DT, respectively. We conduct our comparison for two fixed context lengths and varying batch sizes.

[Figure 44]

[Figure 45]

[Figure 46]

(a) 80 training tasks (b) 20 hold-out tasks

Figure 16. In-context Learning on Dark-Room 10 × 10.

[Figure 47]

[Figure 48]

[Figure 49]

###### (a) B = 1 (b) B = 16

- Figure 17. Latency. We report latency with (a) batch size of 1 and (b) batch size of 16 for DT and xLSTM with 206M parameters. For xLSTM, we use the same number of layer blocks as DT and a higher hidden dimension to match parameters.

- D.5.3. XLSTM: KERNEL COMPARISONS

We leverage custom kernels for xLSTM to conduct our inference-speed comparisons. In particular, we compare 4 variants: recurrent-style inference with and without kernel acceleration, and chunkwise inference with and without kernel acceleration. In our experiments, every timestep contains 3 individual tokens. Consequently, regular recurrent-style inference requires iterating over the token sequence of length 3 in a loop, given the hidden state of the previous timestep. This requires 3 forward passes. In contrast, the chunkwise implementation operates on chunks of timesteps given a hidden state. Consequently, this only requires a single forward pass. In Figure 21, we illustrate the impact of kernel acceleration. We find that our chunkwise kernels result in considerably lower latencies. Interestingly, we find that for B = 1, our chunkwise implementation without kernel acceleration is faster than the recurrent-style inference with kernel acceleration. However, as the batch size increases, this trend reverses. This highlights the importance of kernel acceleration for efficient inference.

- D.5.4. XLSTM: IMPACT OF HEAD DIMENSION

In our experiments, we found that choosing the appropriate head dimension is critical to enable high throughput for xLSTM. Therefore, we conduct an inference ablation with xLSTM 206M in which we vary the number of heads between 4 and 32, while keeping the total hidden dimension constant, resulting in different head dimensions. We find that throughput increases considerably when increasing the number of heads (see Figure 22). For 4 heads, and therefore the highest head dimension, the total throughput saturates at batch size 96. In contrast, when increasing the number of heads to 32 (i.e., decreasing the head dimension), the total throughput continues to increase. This is because a higher head dimension incurs more FLOPS.

[Figure 51]

[Figure 52]

###### (a) B = 1 (b) B = 16

- Figure 18. Latency. We report latency with (a) batch size of 1 and (b) batch size of 16 for DT and xLSTM with 206M parameters. For xLSTM, we use twice the number of layer blocks and the same hidden dimension as the Transformer.

[Figure 53]

[Figure 54]

(a) C = 800 (b) C = 1600

[Figure 55]

- Figure 19. Throughput. We report throughput with (a) context size of 800, and (b) context size of 1600 timesteps for DT and xLSTM with 206M parameters. For xLSTM, we use the same number of layer blocks as DT and a higher hidden dimension to match parameters.

## E. Ablations

- E.1. Removing action condition

- E.1.1. DT ON META-WORLD

We found that removing actions from the context results in better performance across backbones. In Figure 23, we report the learning curves over 200K updates for DT with varying context lengths on Meta-World, both with and without actions in the context. While context lengths beyond 1 hurt performance when training with actions, the reverse is true when training without actions. This is in contrast to recent works, which did not benefit from longer contexts (Octo Model Team et al., 2024). However, while removing actions improves performance on Meta-World, it does not affect performance on discrete control. On Meta-World, we observed that the models become overly confident (high action logits), which is problematic if poor initial actions are produced. We assume this is because in robotics, actions change smoothly, and by observing previous actions, the agent learns shortcuts. A similar issue has been identified by Wen et al. (2020) and termed the copycat problem, because the agent is incentivized to copy previous actions. Our solution is to remove actions from the input sequence. This prevents the agent from learning shortcuts and alleviates the copycat problem.

- E.1.2. DT ON ALL 432 TASKS.

To further investigate the effect of removing actions from the context, we repeat this ablation on the full 432 tasks and 6 domains at the 206M model scale. In Figure 24, we report the learning curves for a DT with varying sequence lengths

[Figure 57]

[Figure 58]

###### (a) C = 800 (b) C = 1600

- Figure 20. Throughput. We report throughput with (a) context size of 800, and (b) context size of 1600 timesteps for DT and xLSTM with 206M parameters. For xLSTM, we use twice the number of layer blocks and the same hidden dimension as the Transformer.

[Figure 59]

[Figure 60]

(a) batch size = 1 (b) batch size = 16

[Figure 61]

- Figure 21. Impact of kernel acceleration. We report latency with (a) batch size of 1 and (b) batch size of 32 for DT and xLSTM with 206M parameters. For xLSTM, we use the same number of layer blocks as DT and a higher hidden dimension to match parameters.

trained (a) with and (b) without actions in the agent’s context. Similar to the single-domain study on Meta-World with smaller models, we find that providing a longer context does not improve performance, resulting in a normalized score of around 0.3 across domains. In contrast, without action in the context, we observe a consistent improvement in the evaluation performance as the sequence length increases. In fact, the normalized score increases from around 0.3 with C = 1 to 0.7 with C = 50. For computational reasons, we only report one seed per sequence length in this experiment, but we believe that the overall trends are clear.

To better understand on which domains the longer context benefits or hurts our agents, we also present the normalized score per domain in Figure 25. Without actions in the context, we find that longer context consistently benefits the performance across domains. With actions in the context, we observe that on Meta-World and DMControl, the performance deteriorates for C > 1. In contrast, on the discrete control domains Atari and Procgen, but also on the continuous control domain Composuite, performance tends to improve with C > 1. This suggests that the copycat problem is particularly present on Meta-World and DMControl. However, note that the final performances on Atari, Procgen, and Mimicgen are considerably worse when actions are present in the context compared to when they are not.

To further investigate this, we compute the MSE between subsequent actions in the training dataset (similar to Wen et al. (2020)) for the continuous control domains and report them in Table 7. Indeed, we find that Meta-World and DMControl exhibit significantly lower MSEs between subsequent actions than Composuite. While Mimicgen also exhibits a low MSE between consecutive actions, all backbones perform poorly on this challenging benchmark. Consequently, we conclude that removing actions from the agent’s context is particularly effective for domains where actions change smoothly.

[Figure 62]

[Figure 63]

- Figure 22. Throughput comparison for xLSTM 206M with varying numbers of heads but fixed total hidden size. By default, we used 4 heads for our experiments. Increasing the number of heads results in higher throughput.

[Figure 64]

[Figure 65]

(a) w/ actions (b) w/o actions

[Figure 66]

- Figure 23. Ablation on removing the action condition for varying context lengths C. Performance of DT (a) with, and (b) without action condition on Meta-World. With action in the context, C > 1 harms performance due to overconfidence in action predictions. Without actions in the context, the performance of DT improves with increasing C.

This result highlights the fact that large action models can strongly benefit from increased context length, even on the simulated environments we consider in this work. Furthermore, we believe that this effect can be even bigger in complex real-world environments that require longer-term interactions.

- E.1.3. XLSTM ON ALL 432 TASKS.

To validate that modern recurrent backbones also benefit from training with longer sequence lengths, we repeat the same ablation as presented in Appendix E.1.2 using xLSTM [1:0]. We report the learning curves, validation perplexities, and evaluation performance across all 432 tasks for varying context lengths in Figure 26. Note that the validation perplexity curves in Figure 26a, start at step 50K for readability. Again, we observe considerable improvements in the validation perplexities and the normalized scores (0.4 for C = 1 to 0.8 for C = 50) as the context length increases.

In addition, we provide the normalized scores per domain for xLSTM with varying sequence lengths in Figure 27. Across domains, we observe increasing performance with increasing C.

### E.2. Return-conditioning vs. Behavior Cloning

Across experiments presented in the main text, except for the ICL experiments, we utilized a sequence representation that includes return-to-go tokens (RTG) as commonly used in the DT literature (Chen et al., 2021; Lee et al., 2022). At inference time, the RTG allows to condition the model on a high target return to produce high-quality actions. This is particularly

[Figure 68]

[Figure 69]

(a) w/ actions (b) w/o actions

- Figure 24. Ablation on removing the action condition for varying context lengths C. Performance of DT (a) with, and (b) without action condition on all 432 tasks. Without actions in the context, the performance of DT improves with increasing C.

Table 7. Average MSE (± standard deviation) between subsequent actions in robotics datasets.

### Meta-World DMControl Composuite Mimicgen Avg. MSE 0.08±0.09 0.2±0.22 2.1±0.3 0.015±0.007

useful when the datasets contain a mixture of optimal and suboptimal trajectories. However, many recent works focus on behavior cloning without return conditioning (Brohan et al., 2023b;a; Octo Model Team et al., 2024).

To better understand whether our findings transfer to the behavior cloning setting, we conduct an ablation study in which we exclude the RTG tokens and the reward tokens from the sequence representation. This means that the sequence consists of state and reward tokens, or state-tokens only. In Figures 28 and 28, we report the (a) validation perplexities and (b) evaluation performance on the 432 task for the four considered backbones when removing RTG or RTG and reward, respectively. We retain the same training settings and datasets as reported in Appendix C (200K updates, evaluation after every 50K steps). We observe similar learning dynamics as for the 206M models that include RTG/reward tokens in the sequence representation (see Figure 2 and Figure 11). Consequently, we conclude that the same performance trends hold for training the considered backbones with and without RTG/reward condition. Note that the final performances are lower compared to the models that include the RTG condition, and that can be conditioned on a high return at inference time.

### E.3. Effect of mLSTM-to-sLSTM ratio.

Throughout our experiments, we compare two xLSTM variants: xLSTM [7:1] and xLSTM [1:0]. The bracket notation was introduced by (Beck et al., 2024) and denotes the ratio of mLSTM to sLSTM blocks. For example, xLSTM [7:1] contains 1 sLSTM block for every 7 mLSTM blocks. As described in Appendix C, we aim to maintain the same ratio as proposed by Beck et al. (2024). While mLSTM blocks are fully parallelizable, sLSTM blocks are not. However, sLSTM preserves the non-diagonalized recurrent matrix to enable state-tracking (Merrill et al., 2024). As such, sLSTM can be attractive for tasks that require state-tracking (see Figure 4 in Beck et al. (2024)).

We first conduct an ablation study on the effect of the mLSTM-to-sLSTM ratio on the evaluation performance across all 432 tasks. For this experiment, we use the 16M parameter model that contains 8 xLSTM blocks in total. Consequently, we compare the following ratios [1:0] (only mLSTM), [0:1] (only sLSTM), [1:1], [1:3], [7:1]. In addition, we investigate the placement of sLSTMs across all 8 blocks. To indicate the placement, we use @ followed by the layer index (starting at 0). For example, [3:1] @ 1,3 indicates that the second and fourth layers are sLSTMs. In Figure 30, we report the validation perplexities and evaluation performance for different ratios and layer placements across the 432 tasks. For computational reasons, we conduct this experiment with only 1 seed per ratio. We find that at the 16M parameter scale, xLSTM [1:0] on average outperforms the variants that leverage sLSTM blocks. This indicates that these domains do not strongly benefit from the state tracking abilities of sLSTM.

[Figure 71]

- (a) w/ actions

[Figure 72]

- (b) w/o actions

- Figure 25. Ablation on removing the action condition for varying context lengths C. We show the normalized score per domain for all context lengths (a) with and (b) without actions.

Next, conduct the same analysis on Dark-Room 10 × 10 ICL environment as used in Appendix D.4. Unlike most of the 432 tasks used in our main experiments, Dark-Room exhibits a partially observable observation space and sparse rewards. Consequently, Dark-Room is more likely to require state tracking abilities. In fact, we already observed better performance for xLSTM [7:1] than for xLSTM [1:0] in Appendix 16. In Figure 31, we report the ICL curves for the 80 train tasks and 20 hold-out tasks. We observe that xLSTM variants that contain sLSTM blocks at lower-level positions, such as [7:1] @ 1 and [3:1] @ 1,3 outperform xLSTM [1:0]. In contrast, xLSTM variants that contain sLSTM blocks at deeper-level positions, such as [0:1] and 3:1 @ 5,7, perform poorly. This is similar to findings by Beck et al. (2024) who also place sLSTM layers

- at lower-level positions.

We conclude that sLSTM layers can be important building blocks for tasks that require state-tracking, such as Dark-Room. Most of the 432 tasks we consider in the main experiments of this work contain fully observable observation spaces and may not require state-tracking. However, we believe that more complex tasks with longer horizons or partial observability, as is common in real-world applications, could greatly benefit from the state-tracking abilities provided by sLSTM blocks. As such, equipping an agent with the ability to perform state-tracking by including sLSTM blocks may be a valuable option for practitioners. This is a distinguishing factor of xLSTM from Mamba, which does not exhibit state-tracking.

### E.4. Effect of Dropout in DT

DTs use by default a Dropout (Srivastava et al., 2014) rate of 0.1. However, during our experiments, we found that Dropout has detrimental effects on the evaluation performance, particularly on continuous control domains like Composuite. In Figure 32, we show the validation perplexities and evaluation performance for a DT trained with and without Dropout. Consequently, we remove Dropout from our DT variant.

### E.5. Effect of reducing number of layers in xLSTM

In prior works, xLSTM and Mamba use twice the number of layers blocks as the Transformer baseline, while maintaining the same hidden dimension (Gu & Dao, 2023; Beck et al., 2024). For our inference-time comparisons, we therefore reduce

[Figure 74]

[Figure 75]

(a) Sequence Prediction Performance (b) Evaluation Performance

- Figure 26. Ablation on the effect of varying the context length C for xLSTM. We report (a) validation perplexity and (b) evaluation performance across the 432 training tasks for xLSTM [1:0]. Without actions in the context, the performance of DT improves with increasing C.

[Figure 76]

[Figure 77]

(a) w/o actions

- Figure 27. Ablation on the effect of varying the context length C for xLSTM. We show the normalized scores per domain for all context lengths.

the number of layer blocks in xLSTM by half. To ensure a fair comparison, we consequently adjust the hidden size of xLSTM to match the number of parameters of the Transformer baseline. In this section, we investigate the effect of these modifications of the xLSTM architecture on the model performance.

In Figure 33, report the validation perplexities and evaluation performance for the regular xLSTM with twice the number of layer blocks as DT, and an xLSTM with half the number of blocks. Reducing the number of layer blocks results in a slight decrease in performance on both metrics. However, xLSTM still outperforms the Transformer baseline (see Figure 2).

## F. Embedding Space Analysis

In Figure 5, we analyze the representations learned by our models using UMAP (McInnes et al., 2018). Here, we explain the clustering procedure in more detail. For every task, we sample 32 sub-trajectories containing 50 timesteps (150 tokens) and encode them using our sequence models. Then, we extract the hidden states at the last layer of our model and aggregate them via mean pooling. We cluster all vectors using the default hyperparameters of UMAP into a two-dimensional space. Finally, we color the resulting points by their domain.

The purpose of this analysis is to examine how the models organize their representations of different environments. In general, tasks within the same domain tend to share similar input characteristics, such as visual inputs (e.g., image frames), possible actions to perform, and reward structures. Therefore, they are more likely to be “grouped” together in the embedding

[Figure 78]

[Figure 79]

[Figure 80]

(a) Sequence Prediction Performance (b) Evaluation Performance

- Figure 28. Ablation on the effect of omitting the RTG condition. We report the learning curves for (a) validation perplexity and (b) evaluation performance across the 432 training tasks for 206M parameter models. We observe similar performance trends as when including the RTG in the sequence.

[Figure 81]

[Figure 82]

(a) Sequence Prediction Performance (b) Evaluation Performance

[Figure 83]

- Figure 29. Ablation on the effect of omitting the RTG condition and the reward condition. We report the learning curves for (a) validation perplexity and (b) evaluation performance across the 432 training tasks for 206M parameter models. We observe similar performance trends as when including the RTG in the sequence.

space. For example, when embeddings of Atari games are closer to each other than to Procgen games, it indicates that Atari games share more similar underlying dynamics or input structures compared to Procgen. We indeed find that tasks from the same domain cluster together. A more refined and better-separated embedding space may result in better final performance, potentially because it facilitates task identification at inference time. This may, however, be specific to the mixture of training tasks at hand. Therefore, we believe that studying the learned embedding spaces of multi-task agents in a wide range of environments is interesting for future work.

Analogous to Figure 5 for DT and xLSTM, we show the UMAP clustering for Mamba 16M in Figure 34. In comparison to DT, Mamba exhibits a slightly stronger grouping of the embedding space.

## G. Raw Scores

In this section, we report the raw scores for all 432 training tasks for the 206M parameter scale. See Tables 8, 9, 10, 11, 12 for Procgen, Atari, Meta-World, DMControl, and Mimicgen, respectively. The raw scores for Composuite are available in Tables 13, 14, 15, and 16.

[Figure 84]

[Figure 85]

[Figure 86]

(a) Sequence Prediction Performance (b) Evaluation Performance

- Figure 30. Ablation on the effect of the mLSTM-to-sLSTM ratio. We report the learning curves for (a) validation perplexity and (b) evaluation performance across the 432 training tasks for 206M parameter models with varying ratios.

[Figure 87]

[Figure 88]

[Figure 89]

(a) 80 training tasks (b) 20 hold-out tasks

Figure 31. In-context Learning on Dark-Room 10 × 10 for varying mLSTM-to-sLSTM ratios.

[Figure 90]

[Figure 91]

[Figure 92]

(a) Sequence Prediction Performance (b) Evaluation Performance

- Figure 32. Ablation on the effect of dropout on DT performance. We show the (a) validation perplexity and (b) evaluation performance on the training tasks. DT performance drops considerably if training with dropout.

[Figure 93]

[Figure 94]

[Figure 95]

(a) Sequence Prediction Performance (b) Evaluation Performance

- Figure 33. Ablation on the effect of reducing the number of layer blocks in xLSTM. We show the (a) validation perplexity and (b) evaluation performance on the training tasks for the layer regular and layer-matched xLSTM models. Reducing the number of layer blocks in xLSTM results in a slight performance decrease.

[Figure 96]

[Figure 97]

(a) DT (b) Mamba (c) xLSTM

[Figure 98]

[Figure 99]

- Figure 34. UMAP clustering of hidden states for 432 tasks produced by (a) DT, (b) Mamba, and (c) xLSTM with 16M parameters, colored by domain. We again depict the embedding spaces for DT and xLSTM from Figure 5 for better readability.

- Table 8. Raw Scores for Procgen.

### Task DT Mamba xLSTM [1:0] xLSTM [7:1]

bigfish 2.53 2.0 4.6 5.13 bossfight 6.73 4.1 9.27 2.0 caveflyer 6.67 6.3 6.67 4.87 chaser 3.41 3.91 4.92 4.2 coinrun 10.0 9.0 10.0 10.0 dodgeball 2.8 3.4 4.27 3.87 fruitbot 13.33 19.8 19.73 19.27 heist 7.33 7.0 6.67 6.67 leaper 5.33 4.0 8.67 5.33 maze 8.67 10.0 7.33 7.33 miner 8.07 11.0 9.0 8.27 starpilot 24.93 10.1 21.8 28.2

Avg. Reward 8.32 7.55 8.73 8.76

- Table 9. Raw Scores for Atari.

### Task DT Mamba xLSTM [1:0] xLSTM [7:1]

Amidar 82.27 30.8 71.07 26.73 Assault 438.2 224.7 410.2 494.13 Asterix 573.33 540.0 763.33 583.33 Atlantis 42573.33 97240.0 83760.0 76973.33 BankHeist 2.67 9.0 0.0 8.67 BattleZone 2000.0 2400.0 2600.0 1733.33 BeamRider 126.13 61.6 176.0 243.47 Boxing 80.8 77.7 83.8 84.93 Breakout 68.13 136.6 92.93 93.73 Carnival 618.67 424.0 697.33 484.0 Centipede 1802.13 1238.2 2416.73 1806.6 ChopperCommand 813.33 800.0 813.33 766.67 CrazyClimber 96853.33 65960.0 106606.67 79873.33 DemonAttack 100.0 65.0 181.33 130.67 DoubleDunk -2.53 -3.0 -2.93 -3.87 Enduro 34.53 65.5 98.73 48.53 FishingDerby -72.47 -68.2 -72.07 -71.0 Freeway 29.0 29.8 30.0 28.6 Frostbite 774.67 1248.0 1162.67 1049.33 Gopher 314.67 34.0 132.0 12.0 Gravitar 116.67 175.0 176.67 136.67 Hero 14004.67 11381.0 14688.67 16522.0 IceHockey -4.8 -6.3 -7.6 -5.93 Jamesbond 490.0 540.0 603.33 510.0 Kangaroo 1426.67 2880.0 2620.0 2653.33 Krull 8880.67 10090.0 8918.0 9569.33 KungFuMaster 8866.67 12700.0 8120.0 11233.33 NameThisGame 7976.67 7967.0 7789.33 7232.0 Phoenix 592.0 1600.0 1807.33 1052.67 Pooyan 283.33 87.5 371.67 406.67 Qbert 4306.67 1700.0 805.0 2613.33 Riverraid 2888.67 6923.0 6688.0 7446.67 RoadRunner 1320.0 350.0 1340.0 213.33 Robotank 18.67 13.2 23.07 25.13 Seaquest 182.67 396.0 448.0 209.33 TimePilot 2533.33 3520.0 3200.0 2966.67 UpNDown 10598.0 12043.0 15340.67 12815.33 VideoPinball 1669.07 0.0 220.4 140.6 WizardOfWor 113.33 160.0 160.0 206.67 YarsRevenge 14356.27 14499.0 16815.0 21403.67 Zaxxon 0.0 0.0 20.0 0.0

Avg. Reward 5556.81 6281.27 6705.61 6383.35

- Table 10. Raw Scores for Meta-World.

Task DT Mamba xLSTM [1:0] xLSTM [7:1]

reach 1860.69 ± 12.51 1859.3 ± 5.79 1859.17 ± 12.62 1864.37 ± 6.57 push 1588.19 ± 207.0 1605.03 ± 107.81 1493.31 ± 238.01 1759.33 ± 3.89 pick-place 137.85 ± 99.18 161.74 ± 153.95 389.81 ± 37.36 296.21 ± 43.77 door-open 1552.95 ± 6.51 1562.39 ± 6.79 1569.35 ± 6.71 1570.16 ± 14.83 drawer-open 1735.13 ± 21.76 1714.4 ± 19.3 1740.48 ± 9.2 1747.33 ± 3.88 drawer-close 1856.67 ± 3.06 1858.05 ± 2.75 1858.7 ± 2.34 1859.33 ± 1.15 button-press-topdown 1322.3 ± 3.12 1326.55 ± 19.93 1341.5 ± 3.15 1322.83 ± 7.25 peg-insert-side 1557.59 ± 98.52 1607.59 ± 9.1 1640.43 ± 13.1 1574.75 ± 90.34 window-open 1594.16 ± 34.13 1568.55 ± 14.38 1576.82 ± 10.21 1578.18 ± 70.3 window-close 1474.26 ± 16.88 1443.94 ± 18.99 1459.83 ± 18.79 1452.21 ± 26.56 door-close 1538.02 ± 14.64 1544.31 ± 3.63 1546.0 ± 9.69 1541.64 ± 10.5 reach-wall 1837.64 ± 1.6 1845.12 ± 3.06 1837.76 ± 3.39 1777.17 ± 94.47 pick-place-wall 1041.54 ± 219.67 843.51 ± 224.6 206.88 ± 184.28 385.57 ± 151.52 push-wall 1689.67 ± 12.74 1701.7 ± 1.54 1599.63 ± 189.06 1487.69 ± 195.8 button-press 1512.08 ± 9.54 1488.1 ± 38.83 1541.77 ± 5.48 1527.3 ± 10.16 button-press-topdown-wall 1314.49 ± 62.73 1295.2 ± 6.62 1321.26 ± 17.59 1328.74 ± 24.16 button-press-wall 1359.83 ± 173.51 1547.14 ± 13.84 1326.57 ± 109.09 1267.11 ± 8.78 peg-unplug-side 1415.68 ± 162.54 1517.49 ± 25.27 1393.98 ± 173.0 1422.64 ± 192.05 disassemble 1452.0 ± 44.54 1441.18 ± 29.15 1220.27 ± 441.51 1072.31 ± 374.95 hammer 1446.68 ± 169.03 1683.04 ± 4.82 1669.54 ± 32.0 1642.34 ± 72.23 plate-slide 1673.66 ± 1.72 1676.83 ± 3.0 1682.41 ± 5.02 1677.52 ± 5.46 plate-slide-side 1719.4 ± 7.85 1694.35 ± 46.29 1686.38 ± 61.27 1690.72 ± 12.97 plate-slide-back 1790.96 ± 6.39 1787.65 ± 5.99 1797.78 ± 1.17 1797.17 ± 0.43 plate-slide-back-side 1773.26 ± 9.72 1763.24 ± 5.59 1785.11 ± 7.42 1788.61 ± 6.67 handle-press 1734.75 ± 220.82 1829.07 ± 29.91 1881.23 ± 15.62 1881.92 ± 10.56 handle-pull 1590.74 ± 35.98 1627.4 ± 34.18 1616.62 ± 52.0 1627.6 ± 21.86 handle-press-side 1852.25 ± 7.0 1857.4 ± 10.13 1847.95 ± 5.61 1857.36 ± 5.57 handle-pull-side 1651.05 ± 3.48 1607.3 ± 22.56 1655.75 ± 4.6 1651.77 ± 7.53 stick-push 1595.45 ± 6.88 1585.22 ± 5.17 1595.35 ± 3.29 1595.21 ± 0.88 stick-pull 1377.41 ± 108.31 1401.91 ± 32.79 1460.27 ± 57.13 1442.68 ± 43.23 basketball 1529.79 ± 11.41 1528.22 ± 18.23 1543.02 ± 2.49 1542.8 ± 17.81 soccer 649.69 ± 160.32 929.06 ± 64.35 792.21 ± 139.63 732.44 ± 290.49 faucet-open 1676.95 ± 121.6 1703.83 ± 41.97 1727.05 ± 45.15 1744.83 ± 15.93 faucet-close 1772.91 ± 9.23 1772.13 ± 2.35 1778.25 ± 3.96 1775.25 ± 0.79 coffee-push 340.21 ± 276.9 232.01 ± 225.2 61.35 ± 51.79 41.79 ± 40.9 coffee-pull 1346.29 ± 101.93 1261.39 ± 195.18 1409.68 ± 34.66 1293.92 ± 129.94 coffee-button 1595.94 ± 16.57 1592.77 ± 2.23 1593.15 ± 49.98 1562.92 ± 36.79 sweep 1485.79 ± 12.17 1452.38 ± 13.74 1508.58 ± 14.96 1471.73 ± 29.08 sweep-into 1796.25 ± 7.64 1472.64 ± 455.9 1804.27 ± 2.38 1786.27 ± 14.64 pick-out-of-hole 1437.38 ± 181.15 1499.35 ± 35.73 1529.83 ± 8.09 1415.91 ± 176.44 assembly 1229.39 ± 16.96 1216.34 ± 22.21 1236.68 ± 21.77 1227.81 ± 7.67 shelf-place 1446.07 ± 30.41 1448.75 ± 39.73 1485.4 ± 12.31 1463.53 ± 9.04 push-back 1226.32 ± 172.59 1022.98 ± 158.35 1011.25 ± 396.65 1027.48 ± 303.73 lever-pull 1604.74 ± 3.32 1634.06 ± 6.08 1639.31 ± 10.11 1626.09 ± 23.72 dial-turn 1688.33 ± 22.94 1667.37 ± 41.45 1713.38 ± 35.16 1686.59 ± 55.09

Avg. Reward 1486.05 1486.18 1455.15 1464.16

- Table 11. Raw Scores for DMControl.

##### Task DT Mamba xLSTM [1:0] xLSTM [7:1]

finger-turn-easy 121.27 ± 104.6 396.4 ± 122.47 449.8 ± 186.65 640.13 ± 82.48 fish-upright 181.14 ± 70.82 154.59 ± 34.64 277.23 ± 105.37 241.73 ± 257.01 hopper-stand 296.15 ± 141.83 304.78 ± 32.65 413.95 ± 35.83 392.34 ± 152.75 point mass-easy 342.26 ± 37.42 720.11 ± 42.95 734.95 ± 114.17 823.74 ± 57.3 walker-stand 911.72 ± 38.16 785.21 ± 23.53 947.31 ± 22.13 864.14 ± 181.56 walker-run 155.91 ± 73.84 274.83 ± 0.44 201.34 ± 34.77 145.01 ± 31.71 ball in cup-catch 976.93 ± 0.83 970.9 ± 4.67 977.33 ± 0.5 975.93 ± 0.42 cartpole-swingup 688.5 ± 42.6 762.4 ± 63.93 800.14 ± 13.64 591.08 ± 86.49 cheetah-run 81.21 ± 96.85 482.39 ± 17.23 358.52 ± 127.92 389.04 ± 4.11 finger-spin 209.27 ± 20.57 430.8 ± 61.66 673.47 ± 94.37 626.93 ± 29.21 reacher-easy 45.4 ± 5.21 180.7 ± 133.64 78.73 ± 20.59 58.0 ± 13.91

Avg. Reward 364.52 496.65 505.06 522.55

- Table 12. Raw Scores for Mimicgen.

Task DT Mamba xLSTM [1:0] xLSTM [7:1]

Panda CoffeePreparation D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.13 ± 0.12 Panda CoffeePreparation D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Panda Coffee D0 0.4 ± 0.2 0.0 ± 0.0 0.2 ± 0.2 0.07 ± 0.12 Panda Coffee D1 0.2 ± 0.2 0.0 ± 0.0 0.2 ± 0.2 0.07 ± 0.12 Panda Coffee D2 0.07 ± 0.12 0.0 ± 0.0 0.07 ± 0.12 0.0 ± 0.0 Panda HammerCleanup D0 1.0 ± 0.0 0.9 ± 0.14 1.0 ± 0.0 1.0 ± 0.0 Panda HammerCleanup D1 0.47 ± 0.5 0.1 ± 0.14 0.47 ± 0.23 0.47 ± 0.31 Panda Kitchen D0 0.87 ± 0.23 0.6 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 Panda Kitchen D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Panda MugCleanup D0 0.13 ± 0.12 0.1 ± 0.14 0.6 ± 0.2 0.27 ± 0.12 Panda MugCleanup D1 0.07 ± 0.12 0.0 ± 0.0 0.2 ± 0.2 0.07 ± 0.12 Sawyer NutAssembly D0 0.07 ± 0.12 0.0 ± 0.0 0.0 ± 0.0 0.07 ± 0.12 Sawyer PickPlace D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Panda Square D0 0.2 ± 0.2 0.0 ± 0.0 0.53 ± 0.12 0.53 ± 0.12 Panda Square D1 0.0 ± 0.0 0.0 ± 0.0 0.2 ± 0.2 0.07 ± 0.12 Panda Square D2 0.13 ± 0.12 0.0 ± 0.0 0.07 ± 0.12 0.07 ± 0.12 Panda StackThree D0 0.0 ± 0.0 0.0 ± 0.0 0.07 ± 0.12 0.0 ± 0.0 Panda StackThree D1 0.0 ± 0.0 0.0 ± 0.0 0.07 ± 0.12 0.0 ± 0.0 Panda Stack D0 0.47 ± 0.12 0.2 ± 0.0 0.67 ± 0.31 0.73 ± 0.12 Panda Stack D1 0.4 ± 0.2 0.0 ± 0.0 0.27 ± 0.12 0.4 ± 0.2 Panda Threading D0 0.27 ± 0.12 0.2 ± 0.0 0.27 ± 0.12 0.2 ± 0.2 Panda Threading D1 0.2 ± 0.35 0.0 ± 0.0 0.07 ± 0.12 0.07 ± 0.12 Panda ThreePieceAssembly D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Panda ThreePieceAssembly D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 IIWA Coffee D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Sawyer Coffee D0 0.27 ± 0.31 0.0 ± 0.0 0.13 ± 0.12 0.2 ± 0.2 UR5e Coffee D0 0.33 ± 0.12 0.2 ± 0.0 0.47 ± 0.31 0.4 ± 0.2 IIWA Coffee D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Sawyer Coffee D1 0.07 ± 0.12 0.0 ± 0.0 0.07 ± 0.12 0.0 ± 0.0 UR5e Coffee D1 0.13 ± 0.12 0.0 ± 0.0 0.2 ± 0.2 0.33 ± 0.31 IIWA Coffee D2 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 UR5e Coffee D2 0.0 ± 0.0 0.1 ± 0.14 0.2 ± 0.0 0.07 ± 0.12 IIWA HammerCleanup D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Sawyer HammerCleanup D0 0.73 ± 0.12 0.9 ± 0.14 0.93 ± 0.12 0.87 ± 0.23 UR5e HammerCleanup D0 1.0 ± 0.0 0.9 ± 0.14 1.0 ± 0.0 0.93 ± 0.12 IIWA HammerCleanup D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Sawyer HammerCleanup D1 0.2 ± 0.2 0.2 ± 0.0 0.27 ± 0.23 0.4 ± 0.35 UR5e HammerCleanup D1 0.47 ± 0.12 0.4 ± 0.28 0.8 ± 0.2 0.6 ± 0.0 IIWA Kitchen D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 UR5e Kitchen D0 0.93 ± 0.12 0.8 ± 0.0 1.0 ± 0.0 1.0 ± 0.0 UR5e Kitchen D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.07 ± 0.12 IIWA MugCleanup D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 IIWA MugCleanup D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 UR5e MugCleanup D1 0.07 ± 0.12 0.0 ± 0.0 0.13 ± 0.12 0.13 ± 0.12 IIWA NutAssembly D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Sawyer NutAssembly D0 0.0 ± 0.0 0.0 ± 0.0 0.07 ± 0.12 0.0 ± 0.0 UR5e NutAssembly D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.07 ± 0.12 IIWA PickPlace D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Sawyer PickPlace D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 UR5e PickPlace D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 IIWA Square D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Sawyer Square D0 0.2 ± 0.2 0.4 ± 0.28 0.33 ± 0.12 0.53 ± 0.23 UR5e Square D0 0.13 ± 0.23 0.3 ± 0.42 0.27 ± 0.12 0.53 ± 0.23 IIWA Square D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Sawyer Square D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 UR5e Square D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 IIWA StackThree D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Sawyer StackThree D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 UR5e StackThree D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 IIWA StackThree D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Sawyer StackThree D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.07 ± 0.12 UR5e StackThree D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 IIWA Stack D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Sawyer Stack D0 0.47 ± 0.31 0.2 ± 0.0 0.6 ± 0.2 0.4 ± 0.2 UR5e Stack D0 0.4 ± 0.2 0.3 ± 0.14 0.87 ± 0.12 0.67 ± 0.12 IIWA Stack D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Sawyer Stack D1 0.2 ± 0.2 0.0 ± 0.0 0.4 ± 0.2 0.27 ± 0.12 UR5e Stack D1 0.6 ± 0.0 0.1 ± 0.14 0.73 ± 0.12 0.4 ± 0.2 IIWA Threading D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Sawyer Threading D0 0.13 ± 0.12 0.0 ± 0.0 0.07 ± 0.12 0.13 ± 0.12 UR5e Threading D0 0.27 ± 0.31 0.1 ± 0.14 0.4 ± 0.2 0.4 ± 0.2 IIWA Threading D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Sawyer Threading D1 0.0 ± 0.0 0.0 ± 0.0 0.13 ± 0.12 0.0 ± 0.0 UR5e Threading D1 0.07 ± 0.12 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 IIWA ThreePieceAssembly D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Sawyer ThreePieceAssembly D0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 UR5e ThreePieceAssembly D0 0.0 ± 0.0 0.0 ± 0.0 0.13 ± 0.12 0.0 ± 0.0 IIWA ThreePieceAssembly D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Sawyer ThreePieceAssembly D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 UR5e ThreePieceAssembly D1 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 IIWA ThreePieceAssembly D2 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 Sawyer ThreePieceAssembly D2 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 UR5e ThreePieceAssembly D2 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0

Table 13. Raw Scores for Composuite, Part1.

Task DT Mamba xLSTM [1:0] xLSTM [7:1]

IIWA Box None PickPlace 402.74 ± 14.4 414.73 ± 10.49 424.35 ± 12.95 421.33 ± 11.39 IIWA Box None Push 388.61 ± 35.63 427.0 ± 2.03 424.4 ± 4.63 427.0 ± 0.68 IIWA Box None Shelf 370.3 ± 80.53 417.61 ± 1.44 417.78 ± 0.96 416.41 ± 1.87 IIWA Box None Trashcan 329.27 ± 113.43 424.39 ± 1.04 429.54 ± 1.57 426.07 ± 3.98 IIWA Box GoalWall PickPlace 367.68 ± 81.93 428.6 ± 4.11 428.0 ± 2.32 429.29 ± 1.97 IIWA Box GoalWall Push 299.69 ± 77.03 337.81 ± 88.42 344.59 ± 28.19 318.19 ± 50.76 IIWA Box GoalWall Shelf 360.92 ± 48.29 405.81 ± 9.82 408.1 ± 5.92 402.31 ± 3.08 IIWA Box GoalWall Trashcan 376.45 ± 83.64 422.34 ± 3.61 429.15 ± 2.72 425.64 ± 3.88 IIWA Box ObjectDoor PickPlace 389.21 ± 47.22 417.89 ± 0.92 413.82 ± 4.06 414.08 ± 3.83 IIWA Box ObjectDoor Push 406.51 ± 0.32 403.59 ± 5.82 373.61 ± 40.95 397.45 ± 1.89 IIWA Box ObjectDoor Shelf 329.42 ± 67.73 353.67 ± 56.2 367.47 ± 43.7 396.33 ± 2.67 IIWA Box ObjectDoor Trashcan 325.45 ± 72.77 372.51 ± 41.55 358.72 ± 76.22 391.58 ± 16.76 IIWA Box ObjectWall PickPlace 393.52 ± 51.47 425.76 ± 2.29 420.61 ± 2.99 421.61 ± 1.06 IIWA Box ObjectWall Push 420.21 ± 3.5 412.76 ± 1.67 410.19 ± 1.62 411.5 ± 3.13 IIWA Box ObjectWall Shelf 400.86 ± 3.66 408.22 ± 1.63 401.42 ± 3.93 396.64 ± 10.55 IIWA Box ObjectWall Trashcan 414.43 ± 2.93 413.71 ± 3.47 417.11 ± 1.69 414.46 ± 0.8 IIWA Dumbbell None PickPlace 386.95 ± 51.87 422.35 ± 2.94 421.32 ± 2.03 421.94 ± 1.48 IIWA Dumbbell None Push 360.62 ± 90.94 413.39 ± 6.13 414.23 ± 6.04 393.34 ± 36.66 IIWA Dumbbell None Shelf 310.45 ± 73.45 344.81 ± 53.72 380.51 ± 5.34 350.8 ± 52.16 IIWA Dumbbell None Trashcan 386.09 ± 40.69 396.08 ± 0.7 414.03 ± 3.78 412.34 ± 3.36 IIWA Dumbbell GoalWall PickPlace 413.6 ± 1.16 415.64 ± 3.28 410.7 ± 7.64 413.51 ± 1.23 IIWA Dumbbell GoalWall Push 316.49 ± 38.69 367.45 ± 4.81 336.67 ± 82.13 371.92 ± 5.91 IIWA Dumbbell GoalWall Shelf 395.63 ± 3.19 372.77 ± 30.32 376.75 ± 8.62 372.77 ± 4.25 IIWA Dumbbell GoalWall Trashcan 379.45 ± 58.51 374.31 ± 55.11 412.22 ± 4.09 406.03 ± 5.03 IIWA Dumbbell ObjectDoor PickPlace 358.13 ± 26.76 364.62 ± 40.18 393.83 ± 2.05 347.28 ± 39.81 IIWA Dumbbell ObjectDoor Push 400.9 ± 8.95 383.81 ± 8.46 382.93 ± 0.7 364.06 ± 35.78 IIWA Dumbbell ObjectDoor Shelf 369.75 ± 14.29 325.7 ± 30.94 350.7 ± 21.76 335.84 ± 40.36 IIWA Dumbbell ObjectDoor Trashcan 393.05 ± 3.92 358.77 ± 36.88 397.23 ± 1.73 389.54 ± 9.14 IIWA Dumbbell ObjectWall PickPlace 403.51 ± 12.08 407.37 ± 0.09 404.28 ± 1.23 401.15 ± 10.64 IIWA Dumbbell ObjectWall Push 330.77 ± 30.29 296.98 ± 68.18 334.41 ± 22.28 307.4 ± 33.85 IIWA Dumbbell ObjectWall Shelf 353.9 ± 29.5 374.39 ± 6.58 358.29 ± 33.75 358.76 ± 18.87 IIWA Dumbbell ObjectWall Trashcan 394.48 ± 4.39 361.99 ± 39.17 398.06 ± 0.59 383.43 ± 32.4 IIWA Plate None PickPlace 427.3 ± 0.59 424.44 ± 1.82 424.59 ± 2.01 425.99 ± 1.2 IIWA Plate None Push 424.25 ± 1.13 419.86 ± 3.96 418.13 ± 3.55 418.42 ± 1.3 IIWA Plate None Shelf 408.07 ± 0.95 397.02 ± 6.49 396.55 ± 10.03 394.93 ± 10.81 IIWA Plate None Trashcan 419.62 ± 1.81 420.24 ± 0.33 420.37 ± 0.91 419.42 ± 2.61 IIWA Plate GoalWall PickPlace 424.69 ± 2.67 423.93 ± 1.77 421.83 ± 1.01 420.13 ± 8.21 IIWA Plate GoalWall Push 409.69 ± 3.55 397.97 ± 13.41 390.46 ± 14.79 388.89 ± 3.01 IIWA Plate GoalWall Shelf 404.92 ± 0.82 396.09 ± 4.6 393.01 ± 5.77 401.81 ± 8.93 IIWA Plate GoalWall Trashcan 420.47 ± 1.88 420.68 ± 2.82 420.29 ± 1.48 421.31 ± 1.93 IIWA Plate ObjectDoor PickPlace 408.48 ± 1.12 403.23 ± 7.83 397.51 ± 1.65 401.53 ± 1.76 IIWA Plate ObjectDoor Push 404.34 ± 4.45 395.97 ± 16.84 389.33 ± 7.78 385.77 ± 1.21 IIWA Plate ObjectDoor Shelf 377.91 ± 21.42 373.43 ± 5.34 369.41 ± 4.97 374.16 ± 13.75 IIWA Plate ObjectDoor Trashcan 400.27 ± 3.16 400.74 ± 0.53 399.28 ± 1.63 400.23 ± 0.63 IIWA Plate ObjectWall PickPlace 417.35 ± 3.15 416.76 ± 6.18 409.31 ± 1.26 411.62 ± 0.97 IIWA Plate ObjectWall Push 413.47 ± 3.92 408.16 ± 6.53 405.51 ± 3.71 405.27 ± 1.34 IIWA Plate ObjectWall Shelf 393.23 ± 1.39 376.64 ± 12.49 386.41 ± 8.65 382.81 ± 6.78 IIWA Plate ObjectWall Trashcan 410.85 ± 1.07 408.87 ± 3.95 408.98 ± 0.82 409.35 ± 2.6 IIWA Hollowbox None PickPlace 378.13 ± 94.18 427.5 ± 6.93 428.62 ± 3.62 426.38 ± 3.26 IIWA Hollowbox None Push 386.22 ± 36.15 422.49 ± 8.01 427.73 ± 1.97 426.12 ± 2.3 IIWA Hollowbox None Shelf 416.65 ± 6.66 419.89 ± 11.03 418.34 ± 6.49 415.11 ± 0.89 IIWA Hollowbox None Trashcan 424.38 ± 2.77 421.62 ± 1.4 426.9 ± 2.35 425.99 ± 1.81 IIWA Hollowbox GoalWall PickPlace 430.17 ± 3.37 427.76 ± 0.48 427.91 ± 0.76 426.47 ± 1.62 IIWA Hollowbox GoalWall Push 401.33 ± 3.96 373.0 ± 41.02 390.09 ± 9.46 394.35 ± 14.43 IIWA Hollowbox GoalWall Shelf 424.55 ± 2.3 379.05 ± 64.32 423.51 ± 1.31 419.69 ± 3.38 IIWA Hollowbox GoalWall Trashcan 425.95 ± 0.73 425.27 ± 0.66 424.8 ± 1.0 420.68 ± 3.33 IIWA Hollowbox ObjectDoor PickPlace 276.87 ± 109.64 369.45 ± 57.47 374.76 ± 45.83 301.41 ± 112.33 IIWA Hollowbox ObjectDoor Push 326.56 ± 109.6 352.22 ± 53.97 390.78 ± 6.35 324.09 ± 55.59 IIWA Hollowbox ObjectDoor Shelf 339.03 ± 43.75 370.75 ± 8.36 362.72 ± 30.31 353.98 ± 38.19 IIWA Hollowbox ObjectDoor Trashcan 395.18 ± 8.7 370.39 ± 35.98 387.21 ± 14.61 387.99 ± 21.95 IIWA Hollowbox ObjectWall PickPlace 364.95 ± 27.07 355.61 ± 76.66 356.01 ± 8.3 369.47 ± 24.62 IIWA Hollowbox ObjectWall Push 422.04 ± 2.08 414.47 ± 8.08 414.39 ± 5.5 408.53 ± 8.05 IIWA Hollowbox ObjectWall Shelf 400.82 ± 2.4 400.31 ± 1.28 403.69 ± 2.06 401.27 ± 1.97 IIWA Hollowbox ObjectWall Trashcan 415.82 ± 0.9 416.68 ± 0.14 392.79 ± 44.13 417.34 ± 0.77

Table 14. Raw Scores for Composuite, Part 2.

Task DT Mamba xLSTM [1:0] xLSTM [7:1] Jaco Box None PickPlace 401.38 ± 3.88 400.41 ± 0.63 399.74 ± 5.35 396.54 ± 4.99 Jaco Box None Push 399.84 ± 3.29 397.79 ± 1.71 392.77 ± 1.12 397.31 ± 1.39 Jaco Box None Shelf 383.53 ± 0.31 384.65 ± 5.31 385.85 ± 1.1 386.34 ± 3.47 Jaco Box None Trashcan 374.88 ± 43.66 398.46 ± 2.69 397.66 ± 4.99 398.21 ± 0.91 Jaco Box GoalWall PickPlace 394.75 ± 2.52 395.12 ± 0.38 392.3 ± 5.3 389.93 ± 3.83 Jaco Box GoalWall Push 317.78 ± 67.67 343.43 ± 7.49 351.67 ± 20.65 336.02 ± 8.59 Jaco Box GoalWall Shelf 374.62 ± 20.35 387.0 ± 1.42 387.73 ± 2.11 384.74 ± 1.19 Jaco Box GoalWall Trashcan 374.07 ± 30.72 393.81 ± 0.68 395.49 ± 1.23 392.53 ± 3.46 Jaco Box ObjectDoor PickPlace 396.05 ± 1.12 391.81 ± 4.67 388.37 ± 1.26 383.39 ± 9.07 Jaco Box ObjectDoor Push 364.64 ± 38.39 383.07 ± 5.73 366.91 ± 33.04 387.51 ± 2.93 Jaco Box ObjectDoor Shelf 373.8 ± 2.81 379.75 ± 1.45 375.38 ± 6.27 376.86 ± 1.37 Jaco Box ObjectDoor Trashcan 388.4 ± 1.28 353.97 ± 52.06 389.38 ± 2.0 389.81 ± 2.89 Jaco Box ObjectWall PickPlace 394.31 ± 2.66 385.33 ± 5.43 388.54 ± 7.62 387.82 ± 2.26 Jaco Box ObjectWall Push 387.4 ± 9.34 384.75 ± 4.29 383.61 ± 7.58 383.32 ± 7.73 Jaco Box ObjectWall Shelf 364.38 ± 2.57 361.28 ± 8.2 367.38 ± 2.04 369.22 ± 2.79 Jaco Box ObjectWall Trashcan 385.73 ± 6.85 385.9 ± 1.13 385.34 ± 0.74 380.01 ± 5.08 Jaco Dumbbell None PickPlace 319.87 ± 1.83 334.2 ± 1.93 376.46 ± 9.19 334.95 ± 68.5 Jaco Dumbbell None Push 388.29 ± 1.98 372.13 ± 5.46 373.3 ± 6.88 369.49 ± 4.36 Jaco Dumbbell None Shelf 300.81 ± 61.26 344.47 ± 15.49 361.77 ± 6.21 362.88 ± 8.22 Jaco Dumbbell None Trashcan 369.52 ± 11.5 369.83 ± 13.39 387.28 ± 1.88 377.27 ± 9.7 Jaco Dumbbell GoalWall PickPlace 306.12 ± 40.29 306.26 ± 32.85 349.04 ± 18.3 348.42 ± 37.3 Jaco Dumbbell GoalWall Push 107.91 ± 29.9 136.11 ± 9.04 245.71 ± 30.15 188.19 ± 58.09 Jaco Dumbbell GoalWall Shelf 300.97 ± 114.65 368.99 ± 0.5 363.58 ± 9.74 346.57 ± 27.41 Jaco Dumbbell GoalWall Trashcan 321.81 ± 87.58 317.94 ± 23.15 376.09 ± 2.22 378.49 ± 4.52 Jaco Dumbbell ObjectDoor PickPlace 382.35 ± 1.62 380.2 ± 5.17 349.1 ± 32.92 372.44 ± 7.6 Jaco Dumbbell ObjectDoor Push 382.32 ± 1.08 353.42 ± 7.17 353.85 ± 6.83 338.66 ± 19.03 Jaco Dumbbell ObjectDoor Shelf 312.14 ± 64.22 330.22 ± 47.38 343.51 ± 30.97 331.5 ± 37.18 Jaco Dumbbell ObjectDoor Trashcan 371.06 ± 8.48 375.34 ± 4.07 373.78 ± 6.05 370.06 ± 8.94 Jaco Dumbbell ObjectWall PickPlace 279.55 ± 111.58 314.05 ± 21.02 360.29 ± 15.75 360.38 ± 12.02 Jaco Dumbbell ObjectWall Push 381.11 ± 3.7 351.38 ± 1.82 349.16 ± 2.93 352.64 ± 11.94 Jaco Dumbbell ObjectWall Shelf 354.95 ± 1.59 316.33 ± 42.6 342.43 ± 7.94 332.97 ± 15.33 Jaco Dumbbell ObjectWall Trashcan 367.01 ± 8.38 354.32 ± 22.23 365.47 ± 7.45 363.25 ± 3.18 Jaco Plate None PickPlace 397.25 ± 0.77 389.99 ± 6.44 384.38 ± 5.92 380.69 ± 2.55 Jaco Plate None Push 395.18 ± 1.01 390.69 ± 9.12 381.68 ± 6.86 380.2 ± 3.48 Jaco Plate None Shelf 380.49 ± 0.75 381.62 ± 0.09 356.49 ± 41.25 380.99 ± 2.43 Jaco Plate None Trashcan 391.97 ± 0.76 390.62 ± 0.57 391.2 ± 1.38 390.3 ± 1.83 Jaco Plate GoalWall PickPlace 379.45 ± 24.14 378.13 ± 6.34 377.33 ± 11.32 376.12 ± 4.31 Jaco Plate GoalWall Push 293.6 ± 38.38 319.4 ± 24.13 320.49 ± 24.25 320.5 ± 31.85 Jaco Plate GoalWall Shelf 358.04 ± 22.32 369.8 ± 15.11 367.73 ± 12.97 362.35 ± 3.32 Jaco Plate GoalWall Trashcan 383.53 ± 7.45 387.55 ± 1.56 389.51 ± 2.03 388.57 ± 1.98 Jaco Plate ObjectDoor PickPlace 390.4 ± 1.3 381.92 ± 15.09 376.2 ± 7.51 380.34 ± 9.73 Jaco Plate ObjectDoor Push 372.01 ± 4.07 366.41 ± 16.51 359.43 ± 10.46 355.71 ± 3.99 Jaco Plate ObjectDoor Shelf 366.15 ± 6.61 357.96 ± 8.35 368.82 ± 4.35 362.39 ± 7.11 Jaco Plate ObjectDoor Trashcan 382.66 ± 0.58 384.3 ± 0.38 384.0 ± 1.92 383.57 ± 1.1 Jaco Plate ObjectWall PickPlace 390.73 ± 1.55 378.98 ± 6.95 376.76 ± 8.54 373.98 ± 5.41 Jaco Plate ObjectWall Push 378.3 ± 4.49 372.47 ± 10.13 364.42 ± 8.12 360.69 ± 3.82 Jaco Plate ObjectWall Shelf 364.2 ± 3.52 364.64 ± 3.01 368.33 ± 1.95 360.73 ± 6.42 Jaco Plate ObjectWall Trashcan 374.17 ± 3.76 375.68 ± 1.54 382.5 ± 2.76 373.86 ± 4.91 Jaco Hollowbox None PickPlace 402.23 ± 2.04 386.75 ± 25.35 396.5 ± 1.04 398.48 ± 3.76 Jaco Hollowbox None Push 392.65 ± 9.62 396.56 ± 4.13 397.09 ± 7.5 396.63 ± 0.38 Jaco Hollowbox None Shelf 377.5 ± 2.78 382.06 ± 6.3 384.26 ± 5.2 381.68 ± 4.82 Jaco Hollowbox None Trashcan 394.85 ± 1.28 394.82 ± 3.27 393.68 ± 3.67 392.87 ± 1.71 Jaco Hollowbox GoalWall PickPlace 395.2 ± 1.44 385.82 ± 13.41 378.92 ± 9.41 379.34 ± 7.17 Jaco Hollowbox GoalWall Push 349.5 ± 34.56 337.43 ± 15.64 348.44 ± 11.76 340.9 ± 2.77 Jaco Hollowbox GoalWall Shelf 357.89 ± 19.58 349.29 ± 10.1 344.53 ± 6.27 333.97 ± 12.22 Jaco Hollowbox GoalWall Trashcan 385.01 ± 1.04 385.4 ± 1.7 386.58 ± 0.37 384.52 ± 0.05 Jaco Hollowbox ObjectDoor PickPlace 335.16 ± 76.71 387.66 ± 8.98 375.68 ± 4.01 344.62 ± 44.5 Jaco Hollowbox ObjectDoor Push 356.64 ± 41.54 386.82 ± 11.07 383.4 ± 9.21 385.73 ± 7.74 Jaco Hollowbox ObjectDoor Shelf 371.32 ± 0.65 362.29 ± 13.12 366.72 ± 4.12 360.22 ± 15.51 Jaco Hollowbox ObjectDoor Trashcan 358.07 ± 46.79 385.01 ± 1.12 383.6 ± 2.35 385.17 ± 0.42 Jaco Hollowbox ObjectWall PickPlace 393.5 ± 2.63 377.85 ± 3.53 378.61 ± 8.16 375.96 ± 5.55 Jaco Hollowbox ObjectWall Push 391.74 ± 4.74 382.69 ± 12.26 387.67 ± 9.52 379.01 ± 6.44 Jaco Hollowbox ObjectWall Shelf 371.33 ± 3.41 367.26 ± 11.73 365.73 ± 7.59 356.39 ± 16.14 Jaco Hollowbox ObjectWall Trashcan 382.6 ± 1.63 385.72 ± 2.03 382.62 ± 1.19 382.01 ± 4.22

Table 15. Raw Scores for Composuite, Part 3.

Task DT Mamba xLSTM [1:0] xLSTM [7:1] Kinova3 Box None PickPlace 432.49 ± 3.69 432.11 ± 7.68 432.28 ± 3.45 431.06 ± 2.67 Kinova3 Box None Push 398.81 ± 44.71 416.96 ± 17.33 428.52 ± 1.83 416.41 ± 18.69 Kinova3 Box None Shelf 411.22 ± 3.9 413.65 ± 0.42 415.58 ± 4.21 411.67 ± 3.98 Kinova3 Box None Trashcan 378.21 ± 81.97 426.67 ± 2.1 431.01 ± 0.89 427.82 ± 1.12 Kinova3 Box GoalWall PickPlace 347.29 ± 145.33 430.92 ± 1.73 431.3 ± 2.19 408.26 ± 40.64 Kinova3 Box GoalWall Push 325.78 ± 131.68 390.05 ± 6.59 382.78 ± 2.17 388.29 ± 6.07 Kinova3 Box GoalWall Shelf 357.79 ± 96.22 395.77 ± 28.11 418.95 ± 2.7 417.37 ± 1.02 Kinova3 Box GoalWall Trashcan 373.8 ± 80.27 424.09 ± 0.02 428.12 ± 3.66 427.05 ± 0.87 Kinova3 Box ObjectDoor PickPlace 425.72 ± 1.7 427.38 ± 0.43 424.25 ± 2.86 424.5 ± 3.45 Kinova3 Box ObjectDoor Push 395.44 ± 30.77 414.0 ± 5.47 406.02 ± 0.61 410.58 ± 8.15 Kinova3 Box ObjectDoor Shelf 381.62 ± 37.98 326.93 ± 2.6 408.55 ± 2.3 381.75 ± 45.62 Kinova3 Box ObjectDoor Trashcan 392.17 ± 40.87 415.87 ± 2.48 419.24 ± 0.61 416.46 ± 1.78 Kinova3 Box ObjectWall PickPlace 405.45 ± 21.25 387.27 ± 50.08 425.83 ± 2.68 423.06 ± 3.66 Kinova3 Box ObjectWall Push 419.98 ± 2.8 414.6 ± 1.04 412.82 ± 1.07 415.16 ± 7.28 Kinova3 Box ObjectWall Shelf 399.47 ± 4.56 399.51 ± 1.29 402.37 ± 2.66 402.42 ± 1.48 Kinova3 Box ObjectWall Trashcan 416.15 ± 4.57 412.41 ± 0.4 399.87 ± 31.99 394.97 ± 36.15 Kinova3 Dumbbell None PickPlace 380.36 ± 55.46 418.88 ± 5.8 419.3 ± 7.37 416.89 ± 2.86 Kinova3 Dumbbell None Push 394.84 ± 25.64 396.29 ± 13.63 367.03 ± 53.29 390.74 ± 22.17 Kinova3 Dumbbell None Shelf 290.98 ± 123.89 394.73 ± 4.82 386.09 ± 19.99 397.38 ± 2.93 Kinova3 Dumbbell None Trashcan 358.26 ± 43.32 377.36 ± 53.06 413.01 ± 6.02 414.39 ± 1.97 Kinova3 Dumbbell GoalWall PickPlace 408.52 ± 19.13 392.63 ± 23.38 404.51 ± 4.31 412.68 ± 11.05 Kinova3 Dumbbell GoalWall Push 294.63 ± 35.99 358.66 ± 10.09 321.72 ± 41.37 310.79 ± 67.84 Kinova3 Dumbbell GoalWall Shelf 384.01 ± 20.53 383.06 ± 15.17 395.02 ± 0.83 377.15 ± 28.52 Kinova3 Dumbbell GoalWall Trashcan 377.28 ± 51.33 370.59 ± 31.83 413.63 ± 2.06 378.76 ± 27.34 Kinova3 Dumbbell ObjectDoor PickPlace 415.58 ± 5.38 404.89 ± 11.83 405.77 ± 7.4 410.95 ± 8.75 Kinova3 Dumbbell ObjectDoor Push 359.17 ± 15.53 265.44 ± 62.94 367.39 ± 23.91 311.57 ± 45.56 Kinova3 Dumbbell ObjectDoor Shelf 360.34 ± 28.19 379.36 ± 6.7 385.26 ± 2.74 363.99 ± 37.65 Kinova3 Dumbbell ObjectDoor Trashcan 409.92 ± 1.78 407.09 ± 1.26 407.79 ± 0.71 407.57 ± 2.85 Kinova3 Dumbbell ObjectWall PickPlace 404.63 ± 16.95 409.29 ± 4.6 406.14 ± 2.11 411.69 ± 6.71 Kinova3 Dumbbell ObjectWall Push 311.79 ± 94.94 285.81 ± 62.32 342.04 ± 22.98 244.56 ± 16.32 Kinova3 Dumbbell ObjectWall Shelf 378.68 ± 3.03 378.63 ± 0.91 376.92 ± 0.76 361.79 ± 25.06 Kinova3 Dumbbell ObjectWall Trashcan 400.98 ± 4.19 398.65 ± 3.89 401.96 ± 1.45 395.81 ± 3.51 Kinova3 Plate None PickPlace 424.09 ± 4.78 427.36 ± 4.29 424.82 ± 1.31 425.02 ± 2.92 Kinova3 Plate None Push 412.25 ± 19.8 422.75 ± 2.79 417.63 ± 6.13 416.41 ± 4.33 Kinova3 Plate None Shelf 409.96 ± 0.2 409.11 ± 0.52 410.28 ± 0.65 409.52 ± 1.61 Kinova3 Plate None Trashcan 422.54 ± 2.13 422.07 ± 1.15 421.73 ± 1.36 422.97 ± 0.74 Kinova3 Plate GoalWall PickPlace 427.74 ± 0.81 421.23 ± 6.67 416.44 ± 1.6 416.35 ± 15.86 Kinova3 Plate GoalWall Push 401.46 ± 2.17 385.01 ± 15.39 377.6 ± 3.14 386.87 ± 12.31 Kinova3 Plate GoalWall Shelf 410.49 ± 0.77 409.46 ± 0.15 409.63 ± 0.65 407.67 ± 3.33 Kinova3 Plate GoalWall Trashcan 421.05 ± 0.88 421.19 ± 0.48 422.63 ± 0.81 423.21 ± 1.16 Kinova3 Plate ObjectDoor PickPlace 423.26 ± 0.3 407.55 ± 0.81 406.43 ± 2.07 414.11 ± 7.32 Kinova3 Plate ObjectDoor Push 258.58 ± 18.57 278.08 ± 34.02 300.72 ± 90.5 257.79 ± 48.13 Kinova3 Plate ObjectDoor Shelf 404.4 ± 0.95 403.82 ± 0.86 405.9 ± 0.31 401.09 ± 2.61 Kinova3 Plate ObjectDoor Trashcan 415.34 ± 1.08 415.81 ± 0.35 416.09 ± 0.31 414.34 ± 1.85 Kinova3 Plate ObjectWall PickPlace 420.16 ± 2.07 413.68 ± 5.5 408.0 ± 2.29 411.83 ± 4.11 Kinova3 Plate ObjectWall Push 400.11 ± 16.39 403.95 ± 3.67 406.48 ± 5.73 403.65 ± 6.23 Kinova3 Plate ObjectWall Shelf 391.09 ± 3.65 391.99 ± 6.62 386.25 ± 16.53 391.7 ± 5.14 Kinova3 Plate ObjectWall Trashcan 413.36 ± 1.11 413.44 ± 3.93 413.82 ± 2.45 415.14 ± 1.46 Kinova3 Hollowbox None PickPlace 424.86 ± 6.23 433.78 ± 0.13 430.43 ± 1.11 430.84 ± 1.55 Kinova3 Hollowbox None Push 361.99 ± 40.33 369.17 ± 8.0 396.28 ± 28.04 380.94 ± 28.74 Kinova3 Hollowbox None Shelf 417.73 ± 13.43 417.46 ± 0.36 423.26 ± 3.53 424.02 ± 2.62 Kinova3 Hollowbox None Trashcan 424.65 ± 1.15 409.34 ± 12.4 425.0 ± 2.72 416.0 ± 15.33 Kinova3 Hollowbox GoalWall PickPlace 386.68 ± 49.29 425.24 ± 0.83 421.85 ± 8.69 420.32 ± 9.71 Kinova3 Hollowbox GoalWall Push 403.57 ± 0.96 383.09 ± 8.37 384.13 ± 10.01 381.43 ± 8.58 Kinova3 Hollowbox GoalWall Shelf 385.7 ± 36.06 395.01 ± 4.51 423.93 ± 5.1 417.05 ± 13.43 Kinova3 Hollowbox GoalWall Trashcan 406.37 ± 27.44 404.11 ± 3.64 405.09 ± 22.54 389.36 ± 32.05 Kinova3 Hollowbox ObjectDoor PickPlace 344.01 ± 63.38 364.3 ± 13.82 387.53 ± 20.66 324.36 ± 55.48 Kinova3 Hollowbox ObjectDoor Push 390.98 ± 46.38 416.05 ± 8.96 405.41 ± 5.34 406.76 ± 16.92 Kinova3 Hollowbox ObjectDoor Shelf 359.0 ± 25.63 381.87 ± 12.39 390.42 ± 6.21 357.94 ± 48.51 Kinova3 Hollowbox ObjectDoor Trashcan 405.87 ± 4.17 411.24 ± 1.26 414.92 ± 3.6 408.73 ± 5.66 Kinova3 Hollowbox ObjectWall PickPlace 424.57 ± 0.92 408.98 ± 6.4 417.83 ± 5.67 419.63 ± 9.2 Kinova3 Hollowbox ObjectWall Push 249.37 ± 176.18 319.13 ± 111.09 324.39 ± 76.09 335.61 ± 74.98 Kinova3 Hollowbox ObjectWall Shelf 394.7 ± 9.3 328.52 ± 61.08 357.89 ± 37.75 362.16 ± 40.05 Kinova3 Hollowbox ObjectWall Trashcan 354.65 ± 48.89 353.43 ± 78.59 407.99 ± 1.96 408.29 ± 4.94

Table 16. Raw Scores for Composuite, Part 4.

Task DT Mamba xLSTM [1:0] xLSTM [7:1] Panda Box None PickPlace 409.21 ± 5.27 408.66 ± 7.81 409.83 ± 1.87 405.46 ± 3.84 Panda Box None Push 402.52 ± 2.55 373.74 ± 49.95 400.35 ± 2.32 399.37 ± 9.95 Panda Box None Shelf 383.69 ± 4.34 381.42 ± 3.66 383.55 ± 5.74 386.01 ± 1.29 Panda Box None Trashcan 400.37 ± 5.64 395.77 ± 2.77 407.95 ± 1.92 406.17 ± 3.36 Panda Box GoalWall PickPlace 401.53 ± 6.39 389.57 ± 18.4 397.12 ± 4.39 401.64 ± 9.81 Panda Box GoalWall Push 272.61 ± 79.58 257.61 ± 57.4 263.72 ± 45.71 281.71 ± 31.21 Panda Box GoalWall Shelf 384.43 ± 1.66 389.06 ± 3.69 388.59 ± 3.9 383.94 ± 2.0 Panda Box GoalWall Trashcan 400.68 ± 4.51 400.18 ± 6.03 403.24 ± 5.65 392.28 ± 16.82 Panda Box ObjectDoor PickPlace 359.01 ± 12.2 365.3 ± 5.97 359.63 ± 0.79 359.27 ± 10.88 Panda Box ObjectDoor Push 363.07 ± 3.13 352.85 ± 13.71 340.37 ± 6.06 340.5 ± 4.97 Panda Box ObjectDoor Shelf 346.29 ± 2.53 345.8 ± 4.91 349.82 ± 6.46 341.44 ± 11.05 Panda Box ObjectDoor Trashcan 361.19 ± 1.65 356.77 ± 3.24 356.66 ± 5.73 337.69 ± 32.63 Panda Dumbbell None PickPlace 342.62 ± 39.18 310.15 ± 24.64 318.76 ± 2.7 342.02 ± 31.28 Panda Dumbbell None Push 299.34 ± 78.28 341.64 ± 42.57 359.06 ± 42.88 263.35 ± 154.81 Panda Dumbbell None Shelf 264.01 ± 101.29 362.15 ± 0.87 319.71 ± 33.9 297.54 ± 67.67 Panda Dumbbell None Trashcan 174.45 ± 64.43 329.06 ± 43.08 373.77 ± 16.73 327.93 ± 68.84 Panda Dumbbell GoalWall PickPlace 310.61 ± 42.65 268.34 ± 147.91 329.02 ± 62.28 360.39 ± 5.25 Panda Dumbbell GoalWall Push 249.21 ± 43.29 282.01 ± 4.89 270.81 ± 11.98 285.28 ± 5.25 Panda Dumbbell GoalWall Shelf 319.5 ± 68.89 347.34 ± 20.01 364.15 ± 2.6 318.6 ± 33.85 Panda Dumbbell GoalWall Trashcan 377.5 ± 5.27 360.98 ± 9.73 379.05 ± 7.52 337.19 ± 40.73 Panda Dumbbell ObjectDoor PickPlace 344.54 ± 5.77 346.57 ± 0.33 340.15 ± 8.5 338.46 ± 10.42 Panda Dumbbell ObjectDoor Push 289.31 ± 11.14 308.25 ± 9.24 309.4 ± 5.02 304.1 ± 8.06 Panda Dumbbell ObjectDoor Shelf 323.26 ± 3.52 279.85 ± 18.84 313.19 ± 17.79 323.49 ± 0.27 Panda Dumbbell ObjectDoor Trashcan 334.05 ± 5.55 337.49 ± 0.68 341.0 ± 3.14 333.06 ± 7.77 Panda Plate None PickPlace 384.37 ± 30.37 404.77 ± 5.27 397.34 ± 1.3 398.41 ± 2.51 Panda Plate None Push 397.95 ± 1.05 398.1 ± 4.91 397.42 ± 3.32 397.64 ± 2.7 Panda Plate None Shelf 352.29 ± 37.8 372.12 ± 13.92 370.46 ± 3.11 367.5 ± 6.03 Panda Plate None Trashcan 392.99 ± 1.41 393.63 ± 2.91 394.05 ± 3.74 393.71 ± 1.27 Panda Plate GoalWall PickPlace 398.36 ± 3.95 398.24 ± 4.51 393.0 ± 1.9 399.02 ± 4.53 Panda Plate GoalWall Push 387.68 ± 0.49 377.79 ± 11.92 355.01 ± 34.01 350.1 ± 22.72 Panda Plate GoalWall Shelf 380.05 ± 0.52 367.67 ± 22.6 339.46 ± 40.63 359.76 ± 5.67 Panda Plate GoalWall Trashcan 391.41 ± 3.83 389.44 ± 3.8 395.4 ± 2.49 393.96 ± 2.68 Panda Plate ObjectDoor PickPlace 350.33 ± 18.2 348.67 ± 8.14 329.35 ± 4.62 336.64 ± 16.61 Panda Plate ObjectDoor Push 346.4 ± 9.33 337.36 ± 17.06 326.32 ± 7.92 323.51 ± 2.24 Panda Plate ObjectDoor Shelf 290.68 ± 11.21 321.54 ± 17.89 326.04 ± 18.76 305.25 ± 20.96 Panda Plate ObjectDoor Trashcan 348.09 ± 3.63 349.43 ± 4.05 351.8 ± 0.25 349.29 ± 1.91 Panda Hollowbox None PickPlace 410.32 ± 6.76 412.25 ± 3.0 408.01 ± 1.93 405.29 ± 5.3 Panda Hollowbox None Push 404.95 ± 1.07 406.74 ± 4.03 401.61 ± 6.16 402.46 ± 4.04 Panda Hollowbox None Shelf 387.59 ± 5.19 380.86 ± 10.45 369.22 ± 14.85 369.57 ± 4.84 Panda Hollowbox None Trashcan 399.09 ± 2.01 400.52 ± 5.27 401.03 ± 5.27 392.82 ± 7.37 Panda Hollowbox GoalWall PickPlace 406.02 ± 10.18 403.47 ± 0.97 405.96 ± 0.39 407.16 ± 3.77 Panda Hollowbox GoalWall Push 259.87 ± 75.12 293.02 ± 117.06 341.55 ± 23.29 281.79 ± 42.98 Panda Hollowbox GoalWall Shelf 387.38 ± 3.45 369.01 ± 6.14 365.26 ± 6.74 316.46 ± 81.46 Panda Hollowbox GoalWall Trashcan 377.54 ± 44.77 395.3 ± 4.85 396.82 ± 4.17 401.54 ± 5.21 Panda Hollowbox ObjectDoor PickPlace 334.94 ± 35.48 341.18 ± 32.31 342.71 ± 7.54 353.64 ± 2.45 Panda Hollowbox ObjectDoor Push 192.69 ± 6.49 294.01 ± 57.68 257.48 ± 13.16 230.54 ± 8.56 Panda Hollowbox ObjectDoor Shelf 343.92 ± 10.22 202.17 ± 4.87 328.01 ± 42.52 285.35 ± 64.92 Panda Hollowbox ObjectDoor Trashcan 338.02 ± 36.48 363.04 ± 2.59 360.88 ± 2.45 363.04 ± 1.29

