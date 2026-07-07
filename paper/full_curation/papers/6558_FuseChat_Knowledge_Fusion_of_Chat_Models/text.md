# arXiv:2402.16107v6[cs.CL]29May2024

## Knowledge Fusion of Chat LLMs: A Preliminary Technical Report

Fanqi Wan, Ziyi Yang, Longguang Zhong, Xiaojun Quan∗, Xinting Huang, Wei Bi FuseAI Research Team

### Abstract

Recently, FUSELLM introduced the concept of knowledge fusion to transfer the collective knowledge of multiple structurally varied LLMs into a target LLM through lightweight continual training. In this report, we extend the scalability and flexibility of the FUSELLM framework to realize the fusion of chat LLMs, resulting in FUSIONCHAT. FUSIONCHAT comprises two main stages. Firstly, we undertake knowledge fusion for structurally and scale-varied source LLMs to derive multiple target LLMs of identical structure and size via lightweight finetuning. Then, these target LLMs are merged within the parameter space, wherein we propose a novel method for determining the merging weights based on the variation ratio of parameter matrices before and after fine-tuning. We validate our approach using three prominent chat LLMs with diverse architectures and scales, namely NH2-Mixtral-8x7B, NH2-Solar-10.7B, and OpenChat-3.5-7B. Experimental results spanning various chat domains demonstrate the superiority of FusionChat-7B across a broad spectrum of chat LLMs at 7B and 34B scales, even surpassing GPT-3.5 (March) and approaching Mixtral-8x7B-Instruct.

### 1 Introduction

Large language models (LLMs) such as GPT (Brown et al., 2020) and LLaMA (Touvron et al., 2023) series have demonstrated remarkable success across a wide range of natural language processing (NLP) tasks. However, the computational resources and time costs associated with LLM development remain prohibitively high for most entities. Despite the structural and functional differences among LLMs, they often exhibit similar capabilities across various tasks. Therefore, moving beyond the traditional approach of training a LLM from scratch, an alternative option is to combine existing LLMs into a new, more powerful one, which is termed knowledge fusion of LLMs by Wan et al. (2024). If successful, this fusion not only reduces the initial training costs but also enables the combined model to leverage the strengths of multiple LLMs.

The endeavor to integrate the capabilities of multiple models has been a long-standing pursuit. For example, ensemble methods (Littlestone and Warmuth, 1994; Jiang et al., 2023) directly aggregate the outputs of different models to enhance prediction performance and robustness. However, this approach requires maintaining multiple trained models and executing each during inference, which is inefficient for LLMs due to their substantial memory and inference time requirements. Another approach is to directly merge several neural networks into a single network through arithmetic operations in the parameter space (Gupta et al., 2020). This approach typically assumes uniform network architectures and seeks to merge the parameters of different neural networks either through manual merging weights (Wortsman et al., 2022; Yadav et al., 2023) or by automatically obtaining merging weights based on model gradients or representations of additional data (Matena and Raffel, 2022; Jin et al., 2022). Recently, FUSELLM (Wan et al., 2024) introduced a new paradigm for

∗ Corresponding authors.

Technical Report, work in progress.

integrating the capabilities of multiple LLMs. This approach externalizes the knowledge of multiple source LLMs using their generated probability distribution matrices and transfers their collective knowledge into a target LLM through lightweight continual training. Consequently, FUSELLM facilitates the fusion of multiple pre-trained LLMs with distinct architectures into a unified LLM.

In this study, we extend the framework of FUSELLM to fuse multiple chat LLMs with diverse architectures and scales, leading to the development of FUSIONCHAT, which comprises two main stages. Firstly, it conducts knowledge fusion for source LLMs with varying structures and scales to derive multiple target LLMs of identical structure and size. To this end, FUSIONCHAT follows the idea of FUSELLM but adopts a pairwise knowledge fusion strategy. Secondly, these target LLMs are merged within the parameter space to incorporate the collective knowledge and respective advantages from source LLMs. For merging, we introduce VARM (Variation Ratio Merge), a novel method for determining the merging weights based on the variation ratio of parameter matrices before and after fine-tuning. In contrast to previous approaches, VARM enables the automatic allocation of distinct weights to each parameter matrix based on the variation ratio of updates during fine-tuning. This facilitates merging LLMs with fine-grained weights without requiring additional training efforts.

FUSIONCHAT offers superior scalability compared to FUSELLM. Firstly, while FUSELLM limits its exploration to LLMs of the same size as the target LLM, FUSIONCHAT delves into the fusion of source chat LLMs with varying sizes. This broader scope allows for greater adaptability to diverse model configurations and requirements. Secondly, the framework of FUSELLM does not seamlessly support the inclusion of new source LLMs as it requires the combination of distribution matrices from all source LLMs during continual training. In contrast, integrating a new source LLM at any scale in FUSIONCHAT is plug-and-play, requiring only obtaining a target LLM from the new source LLM and merging it with the existing version of FUSIONCHAT. Given the frequent updates of chat LLMs in the open-source community2, FUSIONCHAT appears to be more promising for the fusion of chat models.

To empirically demonstrate the effectiveness of FUSIONCHAT, we implement FUSIONCHAT using three representative open-source chat LLMs for fusion: NH2-Mixtral-8x7B (Jiang et al., 2024), NH2-Solar-10.7B (Kim et al., 2023), and OpenChat-3.5-7B (Wang et al., 2023). Experimental results on MT-Bench (Zheng et al., 2023), a cutting-edge benchmark consisting of eight different domains to assess chat LLMs’ multi-turn dialogue ability, confirm that FUSIONCHAT outperforms all the source LLMs and fine-tuned baselines at 7B and 10.7B scales, even approaching the 8x7B MoE source LLM. Moreover, among all the merging methods, the proposed VARM achieves the best performance, indicating the efficacy of merging weights based on the variation ratio of updates.

### 2 Related Work

Model Fusion The fusion of capabilities from diverse models has been a long-standing objective, with existing approaches mainly falling into three categories. Firstly, the traditional technique of model ensemble combines the outputs of multiple models to enhance overall system performance (Littlestone and Warmuth, 1994; Sagi and Rokach, 2018). Note that this technique doesn’t involve the explicit merging of multiple models into a new one. Common methods for model ensemble typically employ weighted averaging (Littlestone and Warmuth, 1994) or majority voting (Monteith et al., 2011) to consolidate predictions from various models. Recently, Jiang et al. (2023) introduced an ensemble framework designed to leverage the diverse strengths of multiple open-source LLMs. This framework first employs a pairwise comparison method to detect subtle distinctions among candidate outputs. Then, it combines the top-ranked candidates to produce an enhanced output.

Secondly, model merging presents another approach that facilitates model fusion within the parameter space. Wortsman et al. (2022) combined multiple models, obtained through different strategies or configurations, through a linear weighted average of parameters, resulting in enhanced overall performance. Likewise, Shoemake (1985) and Ilharco et al. (2022) integrated the capabilities of distinct models by employing spherical linear interpolation and task arithmetic to merge model parameters. To avoid redundant parameter interference, Yadav et al. (2023) and Yu et al. (2023b) suggested pruning low-amplitude varying parameter values before model merging. Furthermore, Matena and Raffel (2022) and Jin et al. (2022) incorporated supplementary data to compute merging weights based on model gradients or representations, eliminating the need for hyperparameter tuning.

2There are 7300+ chat LLMs available on HuggingFace as of drafting this report.

Lastly, FUSELLM (Wan et al., 2024) presents a new paradigm for knowledge fusion of multiple LLMs, which leverages the probabilities distribution matrices generated from source LLMs to transfer the collective knowledge and respective advantages into a target LLM. In comparison to the model ensemble method, which requires the parallel deployment of multiple models, and the model merging approach, which is generally limited to models with identical architectures, FUSELLM supports the fusion of multiple source LLMs with different architectures into a target LLM.

Knowledge Distillation Knowledge distillation (Hinton et al., 2015), initially proposed for model compression, involves training a student model under the guidance of one or more teacher models. In the NLP community, knowledge distillation has been widely applied to text classification tasks. These applications include training the student model to replicate the teacher’s output distribution (Sanh et al., 2019; Turc et al., 2019), as well as features (Sun et al., 2019; Jiao et al., 2020) and relations (Wang et al., 2020) derived from intermediate layers of the teacher model. In the realm of text generation, the conventional approach focuses on minimizing the KL divergence between the student and teacher generation distributions. This is achieved by using the teacher’s probability distributions at each time step as supervision (Khanuja et al., 2021; Gu et al., 2023; Agarwal et al., 2023) or by directly training on the teacher’s generated texts (Peng et al., 2023; Xu et al., 2023).

### 3 Knowledge Fusion of Chat Models

The core concept of FUSIONCHAT comprises two stages. Firstly, it externalizes and transfers the knowledge and capabilities inherent in source chat LLMs to multiple target LLMs of the same structure and size. Secondly, these target LLMs are incorporated into a final fused LLM through model merging.

Specifically, considering K source chat LLMs {Msi}Ki=1 with varying architectures and scales, FUSIONCHAT first specifies a source LLM Msv as the pivot and then applies pairwise knowledge fusion for the pivot and each of the rest LLMs, obtaining (K−1) target LLMs {Mtj}Kj=1−1 which share the same architecture and initial parameters as the pivot LLM. To perform the pairwise knowledge fusion, FUSIONCHAT prompts these source LLMs using a compact and representative training dataset D to showcase their inherent knowledge by predicting the next token. The resulting probabilistic distribution matrices are then utilized to perform pairwise knowledge fusion through lightweight fine-tuning as FUSELLM (Wan et al., 2024). After that, the (K − 1) target LLMs are combined in the parameter space using a specific merging method to yield the fused LLM Mf. To incorporate fine-grained advantages of source LLMs, we introduce VARM (Variation Ratio Merge) to determine the merging weights based on the variation ratio of parameter matrices before and after fine-tuning. In the following sections, we provide a brief introduction to the preliminaries, followed by a detailed description of the pairwise knowledge fusion and model merging in FUSIONCHAT.

#### 3.1 Preliminaries

Let us consider a text sequence q of length N, which is sampled from the training dataset D. The sequence preceding the ith token is represented by t<i = (t1,t2,...,ti−1). The causal language modeling (CLM) objective for training a language model parameterized by θ is defined as minimizing the negative log-likelihood:

LCLM = −Eq∼D

i

log pθ(ti|t<i) , (1)

where pθ(ti|t<i) is the model’s predicted probability for the ith token given the preceding tokens.

To facilitate the fine-tuning of chat LLMs, wherein the text sequence q often consists of a multipleturn dialogue between a user and an assistant, we follow previous works (Chiang et al., 2023; Wan et al., 2023) and mask the user instructions when calculating the training loss LCLM.

The above objective decomposes sequence likelihood into token-level cross-entropy losses, comparing each token’s predicted distribution to its one-hot representation. To provide a more generalized perspective, we reframe this token-level view into a sequential distribution format. Specifically, for the text sequence q, we aggregate token-level predictions to form a probabilistic distribution

matrix, Pθq ∈ RN×V , where the i-th row represents the distribution predicted by the model for the ith token over the vocabulary of size V . The CLM objective can then be interpreted as reducing the

discrepancy between Pθq and the one-hot label matrix, Oq ∈ {0,1}N×V , where each row is a one-hot representation of the corresponding gold token. Formally, the CLM objective is transformed into the following representation:

LCLM = −Eq∼D D(Pθq,Oq) , (2) where D(·,·) represents the discrepancy function between two matrices, and it is equivalent to Eq. 1 when implemented using the KL divergence.

#### 3.2 Pairwise Knowledge Fusion

Taking this perspective on a language model, we follow Wan et al. (2024) and assume that the probabilistic distribution matrix reflects certain inherent knowledge of the language model in understanding the text. Consequently, different probabilistic distribution matrices for the same text, originating from various LLMs, can be used to represent the diverse knowledge embedded within these models. Based on this assumption, the proposed FUSIONCHAT externalizes the knowledge of source LLMs through probabilistic modeling and performs pairwise knowledge fusion by fine-tuning target LLMs using the generated distribution matrices of the source LLMs.

Specifically, for each text sample q in the training dataset D, we first apply the provided K source LLMs to obtain a set of probabilistic distribution matrices, denoted as {Pθqj}Kj=1, where θj represents the parameters of the jth source LLM. Note that these source LLMs may employ different tokenizers, and token alignment is often necessary for proper mapping of probabilistic distribution matrices (Fu et al., 2023; Wan et al., 2024). Utilizing these matrices, we externalize the knowledge from individual models into a unified space, essentially creating unified probabilistic representations over the text.

Then, pairwise knowledge fusion is conducted between the pivot LLM and each of the rest source LLMs. To achieve this, we denote the probabilistic distribution matrix generated by the pivot LLM as Pθ

qv and obtain a set {Pjq}Kj=1−1 of fused matrices as follows:

Pjq = Fusion(Pθ

q ,Pθ

q )|v̸=j, (3)

v

j

where Fusion(·) represents the function that fuses two matrices, and the resulting matrix Pjq is seen as a representation of the collective knowledge and distinctive strengths of two source LLMs. Among various fusion strategies, this work employs minimum cross-entropy (MinCE) (Wan et al., 2024), which empirically performs the best in both FUSELLM and FUSIONCHAT.

After that, we enforce alignment between the prediction of each target LLM Mtj and the corresponding fused representation matrices Pjq. We use Qϕqj to represent the output distribution matrix of the target LLM Mtj for text q, and then define the fusion objective for each target LLM as follows:

q ,Pjq) . (4)

LFusion = −Eq∼D D(Qϕ

j

The overall training objective for each target LLM consists of a weighted combination of the causal language modeling objective LCLM and the fusion objective LFusion as follows:

L = λLCLM + (1 − λ)LFusion. (5)

#### 3.3 Model Merging

Given that the fused target LLMs {Mtj}jK=1−1 share identical architecture and scale while possessing diverse advantages and capabilities learned from the source LLMs, which can be further integrated in

the parameters space (Wortsman et al., 2022) to obtain the final fused LLM Mf:

Mf = Merge({Mt1,Mt2,...,MtK−1}), (6)

where Merge(·) denotes the function that merges multiple target LLMs into a final LLM that combines collective knowledge and distinctive strengths of these target LLMs.

To enhance the adaptability of FUSIONCHAT, it is essential to maintain the simplicity of the Merge function. Firstly, it should be capable of automatically computing the merging weights, eliminating

the need for intricate hyperparameter tuning. Secondly, the merging procedure should not require the incorporation of additional data for the calculation of model gradients or representations.

Since the parameters of the target LLMs continuously evolve to align their generated distribution matrices with the corresponding source LLMs, we propose Variation Ratio Merge (VARM) to utilize the variation ratio of parameters before and after fine-tuning each target LLM as an indicator of knowledge updates, determining its importance in the Merge function:

Em∆θj,m2 K−1 j=1 Em∆θj,m2

. (7)

Wj,m =

where Wj,m represents the merging weight for the parameter unit θj,m (e.g., a matrix) in the target LLM Mtj, while Em∆θj,m2 denotes the average squared variation of parameters in the unit θj,m.

In our preliminary explorations, we have investigated several alternative approaches to determining the weights. These include replacing the square operation with the absolute operation or using softmax. However, the results indicate that none of these alternatives outperforms the current method.

In this work, we define the parameter unit for model merging at the matrix level3. This approach enables the automatic allocation of distinct merging weights to each parameter matrix, thereby facilitating the integration of fine-grained advantages from multiple target LLMs into the fused LLM.

#### 3.4 Discussions

The reasons why FUSIONCHAT does not directly follow FUSELLM to fuse multiple source LLMs of different structures and scales are twofold. Firstly, directly fusing all the source LLMs proves to be difficult, as evidenced by the results of OpenChat-3.5-7B Multi in Table 1. Instead, FUSIONCHAT adopts a fuse-then-merge strategy, wherein the fusing stage employs pairwise knowledge fusion between the pivot LLM and other source LLMs, reducing the difficulty of model fusion. Secondly, FUSIONCHAT offers superior scalability compared to FUSELLM. The framework of FUSELLM requires the combination of distribution matrices from all source LLMs during continual training, which does not seamlessly support the inclusion of new source LLMs. In contrast, FUSIONCHAT supports plug-and-play integration of a new source LLM at any scale, requiring only obtaining a target LLM by fusing the new source LLM and the pivot, and then merging it with the existing version of FUSIONCHAT.

Moreover, the concept of knowledge fusion adopted by both FUSIONCHAT and FUSELLM shares a fundamentally similar purpose with other related topics, such as traditional model ensemble and merging techniques, as well as the recently prominent topic of mixture of experts (MoEs), because they all aim to leverage the strengths of multiple models (experts). While model ensemble and MoEs require loading multiple models (experts) during inference, which have higher memory requirements, weight merging is limited to models with identical architectures. In contrast, knowledge fusion supports the integration of multiple LLMs with diverse architectures into a single LLM without any additional memory requirement, making it appealing in terms of both flexibility and efficiency.

### 4 Experiments

In our experiments, we consider a challenging scenario for the fusion of chat LLMs, where the source LLMs exhibit minimal similarities in architectures and scales. Specifically, we conduct experiments with three representative open-source chat LLMs as the source LLMs, including NH2-Mixtral-8x7B4 (Jiang et al., 2024), NH2-Solar-10.7B5 (Kim et al., 2023), and OpenChat-3.5-7B6 (Wang et al., 2023). As for the pivot LLM, which also serves as the starting point for target LLMs, we opt for OpenChat-3.5-7B due to its balanced scale and performance. We then apply pairwise knowledge fusion as introduced in Section 3.2 to obtain two target LLMs OpenChat-3.5-7B Mixtral and OpenChat-3.5-7B Solar. Finally, we merge

3We discuss the influence of different merging granularities in Section 4.3.

- 4https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO
- 5https://huggingface.co/NousResearch/Nous-Hermes-2-SOLAR-10.7B
- 6https://huggingface.co/openchat/openchat_3.5

OpenChat-3.5-7B Mixtral and OpenChat-3.5-7B Solar by our VARM method (Section 3.3) to obtain the final FusionChat-7B. To assess the performance of FUSIONCHAT-7B, we conduct experiments on MT-Bench7 (Zheng et al., 2023), a benchmark specifically designed to evaluate chat LLMs’ capabilities in multi-turn dialogues across various domains.

#### 4.1 Experimental Setup

Training Dataset To acquire the advantages of source LLMs during knowledge fusion, while mitigating catastrophic forgetting, we curated a high-quality training dataset named FUSIONCHAT MIXTURE from two sources. Firstly, 50% of our training data is sampled from the dataset used by OpenChat8. Secondly, we collected the remaining training samples, unseen by OpenChat, from open-source communities. These two sources resulted in a collection of around 95,000 dialogues across various domains. Further details of FUSIONCHAT MIXTURE can be found in Appendix A.

Training Details In all experiments, we train the OpenChat-3.5-7B using a batch size of 128 and a maximum length of 2048 on a single node with 8x40GB NVIDIA A100 GPUs for three epochs, which takes approximately 7 hours. The model is optimized using the AdamW (Loshchilov and Hutter, 2017) optimizer with β1 = 0.9 and β2 = 0.999, with gradient clipping set to 1.0 and weight decay to 0.0. A cosine learning rate schedule is employed, with a maximum learning rate of 5e-6 and a warmup ratio of 0.03. We empirically set the combination weight λ in Eq. 5 to 0.9. Our training framework is implemented based on the HuggingFace Transformers (Wolf et al., 2020).

Evaluation We evaluate FUSIONCHAT on MT-Bench, which comprises 80 multi-turn dialogues spanning writing, roleplay, reasoning, math, coding, stem, and humanities domains. We adhere to the default configuration of Zheng et al. (2023) and use GPT-49 (gpt-4-0613) as the evaluator for the generated responses, setting the temperature to 0.0 to ensure replicability. The evaluation score ranges from 1 to 10, with 1 denoting the poorest quality and 10 denoting the best.

Baselines In our experiments, we compare our FUSIONCHAT with three categories of baselines. (i) Closed-source LLMs: GPT-4 (March), GPT-3.5 (March), and Claude-1.0. (ii) Source LLMs: NH2-Mixtral-8x7B, NH2-Solar-10.7B, and OpenChat-3.5-7B. (iii) Fine-tuned target LLMs: OpenChat-3.5-7B CLM, which is fine-tuned using only the casual language modeling objective; OpenChat-3.5-7B Multi, which is fine-tuned using the fusion of distributions generated from multiple source LLMs (Wan et al., 2024); OpenChat-3.5-7B Mixtral, which is the corresponding target LLM obtained by fusing OpenChat-3.5-7B and NH2-Mixtral-8x7B; OpenChat-3.5-7B Solar, which is the corresponding target LLM obtained by fusing OpenChat-3.5-7B and NH2-Solar-10.7B. We also evaluate the performance of FUSIONCHAT by comparing different merging methods to obtain the fused LLMs, including FusionChat-7B Linear (Wortsman et al., 2022), FusionChat-7B SLERP (Shoemake, 1985), FusionChat-7B TA (Ilharco et al., 2022), FusionChat-7B TIES (Yadav et al., 2023), FusionChat-7B DARE (Yu et al., 2023b), and our FusionChat-7B VaRM.

#### 4.2 Overall Results

- In Table 1, we present the overall results of FUSIONCHAT compared to baselines of different scales and categories across various domains of MT-Bench. Our observations are as follows. First, we note distinct performance among the three source LLMs across all domains, with OpenChat-3.5-7B exhibiting balanced performance despite its smaller scale. Second, after fine-tuning using the casual language model objective on our high-quality training dataset, the resulting model (OpenChat-3.5-7B CLM) achieves an increased average performance from 7.79 to 7.95, although this improvement is relatively modest and inconsistent across distinct domains. Third, in the category of fine-tuned target LLMs, OpenChat-3.5-7B Multi achieves a relative performance gain of 1.38% over OpenChat-3.5-7B CLM. Notably, OpenChat-3.5-7B Mixtral and OpenChat-3.5-7B Solar, two target LLMs obtained by pairwise knowledge fusion, outperform OpenChat-3.5-7B Multi. Moreover, these target LLMs demonstrate individual strengths in different domains, providing a foundation for

- 7https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
- 8https://huggingface.co/openchat/openchat_3.5#dataset-details
- 9https://platform.openai.com/docs/models

subsequent integration into a more powerful LLM. For instance, OpenChat-3.5-7B Mixtral excels in the reasoning domain, surpassing OpenChat-3.5-7B CLM by an average of 12.58%, while OpenChat-3.5-7B Solar achieves the highest scores in both the extraction and STEM domains, with 8.70% and 9.53% relative performance enhancements, respectively.

Models Writing Roleplay Reasoning Math Coding Extraction STEM Humanities Avg.

Closed-source LLMs

GPT-4 (March) 9.65 8.90 9.00 6.80 8.55 9.38 9.70 9.95 8.96 GPT-3.5 (March) 9.20 8.40 5.65 6.30 6.90 8.85 8.70 9.55 7.94 Claude-1.0 9.50 8.50 5.95 4.80 6.25 8.80 9.70 9.70 7.90

Source LLMs

NH2-Mixtral-8x7B 9.70 9.05 6.65 6.75 6.00 8.80 9.70 9.95 8.33 NH2-Solar-10.7B 9.50 8.40 7.35 5.55 4.00 7.85 8.80 9.80 7.66 OpenChat-3.5-7B 9.00 7.85 7.75 5.30 5.50 8.45 8.55 9.90 7.79

Fine-tuned target LLMs

|OpenChat-3.5-7B CLM|8.80 8.60 7.55 5.88 5.45 8.05 9.45 9.85|7.95<br><br>|
|---|---|---|
|OpenChat-3.5-7B Multi|9.35 8.38 8.40 5.35 5.15 8.70 9.28 9.90 (+6.25%) (-2.56%) (+11.26%) (-9.01%) (-5.50%) (+8.07%) (-1.80%) (+0.51%)<br><br>|8.06 (+1.38%)|
|OpenChat-3.5-7B Mixtral|9.35 8.35 8.50 5.80 5.10 8.25 9.38 9.90 (+6.25%) (-2.91%) (+12.58%) (-1.36%) (-6.42%) (+2.48%) (-0.74%) (+0.51%)<br><br>|8.08 (+1.64%)|
|OpenChat-3.5-7B Solar<br><br>|9.35 8.43 8.20 5.70 5.70 8.75 9.53 9.80 (+6.25%) (-1.98%) (+8.61%) (-3.06%) (+4.59%) (+8.70%) (+0.85%) (-0.51%)|8.18 (+2.89%)<br><br>|

Fused LLMs

|FUSIONCHAT-7B Linear|9.18 8.23 8.00 5.78 5.55 8.50 9.43 9.78 (+4.32%) (-4.30%) (+5.96%) (-1.70%) (+1.83%) (+5.59%) (-0.21%) (-0.71%)<br><br>|8.05 (+1.26%)|
|---|---|---|
|FUSIONCHAT-7B SLERP<br><br>|9.45 8.73 8.05 6.20 5.30 8.45 9.43 9.93 (+7.39%) (+1.51%) (+6.62%) (+5.44%) (-2.75%) (+4.97%) (-0.21%) (+0.81%)|8.19 (+3.02%)<br><br>|
|FUSIONCHAT-7B TA<br><br>|9.35 8.33 8.15 5.85 6.05 8.45 9.50 9.90 (+6.25%) (-3.14%) (+7.95%) (-0.51%) (+11.01%) (+4.97%) (+0.53%) (+0.51%)<br><br>|8.20 (+3.14%)|
|FUSIONCHAT-7B TIES<br><br>|9.38 8.58 8.15 5.80 5.85 8.40 9.45 9.88 (+6.59%) (-0.23%) (+7.95%) (-1.36%) (+7.34%) (+4.35%) (+0.00%) (+0.30%)<br><br>|8.18 (+2.89%)|
|FUSIONCHAT-7B DARE<br><br>|9.45 8.78 7.75 5.80 5.80 8.45 9.45 9.65 (+7.39%) (+2.09%) (+2.65%) (-1.36%) (+6.42%) (+4.97%) (+0.00%) (-2.03%)<br><br>|8.14 (+2.39%)|
|FUSIONCHAT-7B VARM|9.20 8.63 8.00 6.15 6.15 8.50 9.30 9.85 (+4.55%) (+0.35%) (+5.96%) (+4.59%) (+12.84%) (+5.59%) (-1.59%) (+0.00%)<br><br>|8.22 (+3.40%)|

- Table 1: Overall results of the proposed FUSIONCHAT compared to baselines of different scales and categories across various domains of MT-Bench. Percentages indicate the rate of improvement (in blue)/decrease (in red) compared to OpenChat-3.5-7B CLM.

The final fused LLM FusionChat-7B is obtained by merging OpenChat-3.5-7B Mixtral and OpenChat-3.5-7B Solar in the parameter space, where various merging methods are explored. It is observed that FusionChat-7B with SLERP, TA, and our VaRM outperform all the fine-tuned target LLMs, showcasing FUSIONCHAT’s ability to integrate the unique strengths and collective capabilities of different target LLMs. In contrast, merging methods such as Linear and DARE tend to result in degraded performance. Since the target LLMs exhibit varying parameter variations, designing finegrained merging weights is crucial for effectively combining their respective advantages. Therefore, methods like Linear, which involves manual weight assignment, and DARE, which eliminates a subset of model parameters before merging, are deemed inappropriate for FUSIONCHAT.

We further demonstrate that FusionChat-7B with VaRM consistently outperforms all other merging methods, achieving an average evaluation score of 8.22. This score not only surpasses GPT-3.5 (March)’s score of 7.94, but also approaches the score of the current state-of-the-art (SOTA) opensource chat LLM, NH2-Mixtral-8X7B, which stands at 8.33. This confirms the effectiveness of the proposed VARM method in utilizing the variation ratio of each parameter matrix to allocate different merging weights, thereby blending updated knowledge at a fine-grained matrix level.

Granularity Writing Roleplay Reasoning Math Coding Extraction STEM Humanities Avg. Model 8.93 8.58 8.20 5.73 5.45 8.55 9.45 9.90 8.10 Layer 9.43 8.73 8.00 6.15 5.25 8.55 9.18 9.90 8.15 Matrix 9.20 8.63 8.00 6.15 6.15 8.50 9.30 9.85 8.22 Parameter 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00

- Table 2: Results of FusionChat-7B VaRM with VARM adopting different merging granularities of parameter units across various domains of MT-Bench.

#### 4.3 Merging Granularities in VARM

Since the merging granularity of the parameter unit θj,m in Eq. 7 can be adaptively adjusted, we investigate its influence on the final performance of FusionChat-7B VaRM.

- In Table 2, we observe a consistent improvement in average performance when transitioning the granularity of merging weights from model level to layer level, and then to matrix level. This suggests that the assignment of fine-grained merging weights is effective for integrating knowledge from multiple target LLMs. However, when the granularity is reduced to the parameter level, we observe a notable decline in performance. This may be attributed to extreme merging weights assigned to specific parameters, which disrupts correlations among other parameters.

8.81

First Second Avg.

8.76

8.8

| |
|---|

8.63

| |
|---|

8.6

8.4

8.22

Performance

8.2

8.15

8.10

8.0

7.8

7.69

7.56

7.6

7.48

7.4

Model Layer Matrix

We further investigate the impact of varying merging granularities on the performance of different dialogue turns. Figure 1 illustrates that as the merging granularity progressively decreases from the model level to the layer level and then to the matrix level, the performance of the first turn first experiences enhancement and then declines, while the performance of the second turn exhibits the opposite trend. Despite this fluctuation, there is a consistent improvement in overall performance. These findings suggest that VARM at the matrix granularity adeptly captures the complex dynamics among multiple dialogue turns, thereby leading to optimal overall performance.

Figure 1: Performance of FusionChat-7B VaRM by using varying merging granularities of parameter groups on different dialogue turns in MTBench.

### 5 Conclusion

In this work, we propose an extended framework of FUSELLM to integrate the collective knowledge and individual strengths of multiple structure and scale-varied chat LLMs into a more powerful chat LLM, resulting in FUSIONCHAT. FUSIONCHAT adopts a fuse-then-merge strategy with two main stages. Firstly, it undertakes pairwise knowledge fusion for source LLMs to derive multiple target LLMs of identical structure and size via lightweight fine-tuning. Then, these target LLMs are merged within the parameter space, wherein we propose a novel method VARM for determining the merging weights based on the variation ratio of parameter matrices before and after fine-tuning. Experimental results spanning various chat domains demonstrate the superiority of FUSIONCHAT across different model scales, even surpassing GPT-3.5 (March) and approaching Mixtral-8x7B-Instruct.

Moreover, we argue that the concept of knowledge fusion adopted by both FUSIONCHAT and FUSELLM shares a fundamentally similar purpose with other related topics, such as the recently popular topic of mixture of experts (MoEs), because they all aim to leverage the strengths of multiple models (experts). However, while MoEs require loading multiple experts during inference, which has higher memory requirements, knowledge fusion supports the integration of multiple LLMs with diverse architectures into a single LLM without any additional memory requirement, making it more memory-efficient. To the best of our knowledge, MoEs typically employ more than six experts, while FUSIONCHAT and FUSELLM only fuse three source LLMs. In future work, we will further explore fusing more source LLMs to fully harness the potential of knowledge fusion for LLMs.

### References

Agarwal, R., Vieillard, N., Stanczyk, P., Ramos, S., Geist, M., and Bachem, O. (2023). Gkd: Generalized knowledge distillation for auto-regressive sequence models. arXiv preprint arXiv:2306.13649.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. (2020). Language models are few-shot learners. Advances in Neural Information Processing Systems, 33:1877–1901.

Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., Stoica, I., and Xing, E. P. (2023). Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. (2021). Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Fu, Y., Peng, H., Ou, L., Sabharwal, A., and Khot, T. (2023). Specializing smaller language models towards multi-step reasoning. arXiv preprint arXiv:2301.12726.

Gu, Y., Dong, L., Wei, F., and Huang, M. (2023). Knowledge distillation of large language models. arXiv preprint arXiv:2306.08543.

Gupta, V., Serrano, S. A., and DeCoste, D. (2020). Stochastic weight averaging in parallel: Largebatch training that generalizes well. International Conference on Learning Representations.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. (2021). Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Hinton, G., Vinyals, O., and Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S., Schmidt, L., Hajishirzi, H., and Farhadi, A. (2022). Editing models with task arithmetic. arXiv preprint arXiv:2212.04089.

Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., Casas, D. d. l., Hanna, E. B., Bressand, F., et al. (2024). Mixtral of experts. arXiv preprint arXiv:2401.04088.

Jiang, D., Ren, X., and Lin, B. Y. (2023). Llm-blender: Ensembling large language models with pairwise ranking and generative fusion. arXiv preprint arXiv:2306.02561.

Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., and Liu, Q. (2020). Tinybert: Distilling bert for natural language understanding. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4163–4174.

Jin, X., Ren, X., Preotiuc-Pietro, D., and Cheng, P. (2022). Dataless knowledge fusion by merging weights of language models. In The Eleventh International Conference on Learning Representations.

Khanuja, S., Johnson, M., and Talukdar, P. (2021). Mergedistill: Merging language models using pre-trained distillation. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 2874–2887.

Kim, D., Park, C., Kim, S., Lee, W., Song, W., Kim, Y., Kim, H., Kim, Y., Lee, H., Kim, J., et al.

(2023). Solar 10.7 b: Scaling large language models with simple yet effective depth up-scaling. arXiv preprint arXiv:2312.15166.

Köpf, A., Kilcher, Y., von Rütte, D., Anagnostidis, S., Tam, Z.-R., Stevens, K., Barhoum, A., Duc, N. M., Stanley, O., Nagyfi, R., et al. (2023). Openassistant conversations–democratizing large language model alignment. arXiv preprint arXiv:2304.07327.

Littlestone, N. and Warmuth, M. K. (1994). The weighted majority algorithm. Information and Computation, 108(2):212–261.

Loshchilov, I. and Hutter, F. (2017). Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Luo, Z., Xu, C., Zhao, P., Sun, Q., Geng, X., Hu, W., Tao, C., Ma, J., Lin, Q., and Jiang, D.

(2023). Wizardcoder: Empowering code large language models with evol-instruct. arXiv preprint arXiv:2306.08568.

Matena, M. S. and Raffel, C. A. (2022). Merging models with fisher-weighted averaging. Advances in Neural Information Processing Systems, 35:17703–17716.

Monteith, K., Carroll, J. L., Seppi, K., and Martinez, T. (2011). Turning bayesian model averaging into bayesian model combination. In The 2011 International Joint Conference on Neural Networks, pages 2657–2663. IEEE.

Mukherjee, S., Mitra, A., Jawahar, G., Agarwal, S., Palangi, H., and Awadallah, A. (2023). Orca: Progressive learning from complex explanation traces of gpt-4. arXiv preprint arXiv:2306.02707.

Peng, B., Li, C., He, P., Galley, M., and Gao, J. (2023). Instruction tuning with gpt-4. arXiv preprint arXiv:2304.03277.

Sagi, O. and Rokach, L. (2018). Ensemble learning: A survey. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 8(4):e1249.

Sanh, V., Debut, L., Chaumond, J., and Wolf, T. (2019). Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

Shoemake, K. (1985). Animating rotation with quaternion curves. In Proceedings of the 12th annual conference on Computer graphics and interactive techniques, pages 245–254.

Sun, S., Cheng, Y., Gan, Z., and Liu, J. (2019). Patient knowledge distillation for bert model compression. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLPIJCNLP), pages 4323–4332.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Turc, I., Chang, M.-W., Lee, K., and Toutanova, K. (2019). Well-read students learn better: On the importance of pre-training compact models. arXiv preprint arXiv:1908.08962.

Wan, F., Huang, X., Cai, D., Quan, X., Bi, W., and Shi, S. (2024). Knowledge fusion of large language models. arXiv preprint arXiv:2401.10491.

Wan, F., Huang, X., Yang, T., Quan, X., Bi, W., and Shi, S. (2023). Explore-instruct: Enhancing domain-specific instruction coverage through active exploration. arXiv preprint arXiv:2310.09168.

Wang, G., Cheng, S., Zhan, X., Li, X., Song, S., and Liu, Y. (2023). Openchat: Advancing opensource language models with mixed-quality data. arXiv preprint arXiv:2309.11235.

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., and Zhou, M. (2020). Minilm: Deep selfattention distillation for task-agnostic compression of pre-trained transformers. Advances in Neural Information Processing Systems, 33:5776–5788.

Wei, Y., Wang, Z., Liu, J., Ding, Y., and Zhang, L. (2023). Magicoder: Source code is all you need. arXiv preprint arXiv:2312.02120.

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., et al. (2020). Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Dyiemonstrations, pages 38–45.

Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A. S., Namkoong, H., Farhadi, A., Carmon, Y., Kornblith, S., et al. (2022). Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International Conference on Machine Learning, pages 23965–23998. PMLR.

Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., and Jiang, D. (2023). Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244.

Yadav, P., Tam, D., Choshen, L., Raffel, C., and Bansal, M. (2023). Ties-merging: Resolving interference when merging models. In Thirty-seventh Conference on Neural Information Processing Systems.

Yu, L., Jiang, W., Shi, H., Yu, J., Liu, Z., Zhang, Y., Kwok, J. T., Li, Z., Weller, A., and Liu, W. (2023a). Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284.

Yu, L., Yu, B., Yu, H., Huang, F., and Li, Y. (2023b). Language models are super mario: Absorbing abilities from homologous models as a free lunch. arXiv preprint arXiv:2311.03099.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al. (2023). Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685.

### A Details of Training Dataset

We curated a comprehensive training dataset, FUSIONCHAT-MIXTURE, from various sources. This dataset covers different styles and capabilities, featuring both human-written and model-generated, and spanning general instruction-following and specific skills. These sources include:

- • Orca-Best10: We sampled 20,000 examples from Orca-Best, which is filtered from the original GPT-4 (1M) partition of Orca (Mukherjee et al., 2023) based on maximum length and embedding clustering of instructions.
- • Capybara11: We incorporated all the 16,000 examples of Capybara, which is a high-quality collection of multi-turn synthetic conversations.
- • No-Robots12: We included all the 9,500 examples of No-Robots, which is a dataset created by skilled human annotators for supervised fine-tuning.
- • ShareGPT-GPT413: We utilized all 6,200 examples from ShareGPT-GPT4, which exclusively uses dialogues generated by GPT-4 in ShareGPT.
- • Oasst-Top114: We selected 5,000 examples from Oasst-Top1, which is a refined version of Oasst1 (Köpf et al., 2023), a human-annotated assistant-style conversation dataset.
- • MetaMathQA 15: We sampled 10,000 examples from MetaMathQA (Yu et al., 2023a), which is augmented from the GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021) datasets for mathematics problem-solving.
- • OSS-Instruct 16: We chose 10,000 examples from OSS-Instruct (Wei et al., 2023), which contains code instruction data synthesized from open-source code snippets.
- • Evol-Alpaca 17: We sampled 10,000 examples from Evol-Alpaca, which is a code instruction dataset generated by GPT-4 with evol-instruct proposed by WizardCoder (Luo et al., 2023).
- • Python-Code 18: We selected 10,000 examples from Python-Code, which comprises instructions and responses generated by GPT-3.5 and GPT-4 for python code generation.

We followed the data processing code in Vicuna (Chiang et al., 2023) to clean instances containing non-English or special characters. Then, we split long conversations into blocks with a maximum length of 2048 tokens, resulting in the final FUSIONCHAT MIXTURE with 95,000 examples.

### B Case Studies

We present case studies to demonstrate the individual strengths of target LLMs (OpenChat-3.5-7B Mixtral and OpenChat-3.5-7B Solar) obtained from knowledge fusion of source LLMs, and show the collective knowledge and strengths of FUSIONCHAT (FusionChat-7B VaRM) obtained by further merging target LLMs. OpenChat-3.5-7B CLM is used as the baseline for comparison.

- 10https://huggingface.co/datasets/shahules786/orca-best
- 11https://huggingface.co/datasets/LDJnr/Capybara
- 12https://huggingface.co/datasets/HuggingFaceH4/no_robots
- 13https://huggingface.co/datasets/shibing624/sharegpt_gpt4
- 14https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25
- 15https://huggingface.co/datasets/meta-math/MetaMathQA
- 16https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K
- 17https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1
- 18https://huggingface.co/datasets/ajibawa-2023/Python-Code-23k-ShareGPT

###### OpenChat-3.5-7B CLM; Roleplay; ID 93;

Turn 1 Question: Imagine yourself as a doctor tasked with devising innovative remedies for various ailments and maladies. Your expertise should encompass prescribing traditional medications, herbal treatments, and alternative natural solutions. Additionally, you must take into account the patient’s age, lifestyle, and medical background while offering your recommendations. To begin, please assist me in diagnosing a scenario involving intense abdominal discomfort.

- Turn 1 Answer: As a doctor, I would first gather information about the patient’s symptoms, medical history, and lifestyle. In the case of intense abdominal discomfort, I would consider several potential causes, such as indigestion, gas, food poisoning, or an inflammatory bowel disease. I would then recommend a combination of traditional medications, herbal treatments, and alternative natural solutions, taking into account the patient’s age, lifestyle, and medical background. For a patient with intense abdominal discomfort due to indigestion or gas, I would recommend the following remedies: 1. Traditional medications: Over-the-counter antacids such as Maalox or Mylanta can help alleviate indigestion and gas. 2. Herbal treatments: Chamomile tea or peppermint tea can help soothe the stomach and reduce abdominal discomfort. 3. Alternative natural solutions: Encourage the patient to avoid trigger foods, such as spicy or fatty foods, and to consume smaller meals more frequently. Additionally, they can try gentle exercises like walking or yoga to improve digestion. For a patient with food poisoning, I would recommend the following remedies: 1. Traditional medications: Depending on the severity of the symptoms, the patient may need prescription antibiotics. Over-the-counter medications such as loperamide can help control diarrhea. 2. Herbal treatments: Ginger tea or lemon water can help soothe the stomach and reduce nausea. 3. Alternative natural solutions: Encourage the patient to stay hydrated by consuming clear fluids like water and electrolyte drinks. Once symptoms subside, probiotics can be taken to restore gut flora balance. For a patient with an inflammatory bowel disease, I would recommend the following remedies: 1. Traditional medications: Prescription medications such as aminosalicylates or corticosteroids may be necessary to reduce inflammation and control symptoms. 2. Herbal treatments: Some herbs, such as turmeric and slippery elm, have anti-inflammatory properties that may help alleviate symptoms. However, it is essential to consult with a healthcare professional before using these herbs, as they can interact with other medications or have side effects. 3. Alternative natural solutions: Encourage the patient to maintain a healthy diet, rich in fiber, and low in processed foods. Stress management techniques, such as meditation or yoga, can also help manage symptoms. In all cases, it is essential to monitor the patient’s progress and adjust the treatment plan as needed. If symptoms worsen or do not improve within a reasonable timeframe, further testing and specialist referral may be necessary.

- Turn 1 Judgment: The assistant’s response is highly detailed, accurate, and relevant to the user’s question. It provides a comprehensive approach to diagnosing and treating intense abdominal discomfort, considering various potential causes. The assistant also takes into account the need for traditional medications, herbal treatments, and alternative natural solutions, as requested by the user. The assistant’s response is also creative in its approach to providing a variety of treatment options for each potential cause of abdominal discomfort. The assistant’s response is also helpful in that it provides clear and actionable advice for each potential cause of the discomfort. Therefore, the assistant’s response is of high quality. Rating: [[9]]

- Turn 2 Question: But I have been pregnant for 20 weeks and I am allergic to many medicines.

- Turn 2 Answer: I am not a doctor, but I can try to provide some general guidance. If you are pregnant and experiencing abdominal discomfort, it’s essential to consult with your healthcare provider before starting any treatment. They will consider your pregnancy and medication allergies when recommending a course of action. In the meantime, here are some general suggestions for managing abdominal discomfort during pregnancy: 1. Eat smaller meals throughout the day to avoid overfilling your stomach. 2. Avoid spicy, fatty, or greasy foods that may worsen indigestion. 3. Stay hydrated by drinking plenty of water. 4. Practice gentle exercises, such as walking or prenatal yoga, to aid digestion. However, it’s crucial to consult with your healthcare provider for personalized advice tailored to your specific situation. Turn2 Judgment: The assistant’s response is relevant, accurate, and detailed. It provides general advice for managing abdominal discomfort during pregnancy, while emphasizing the importance of consulting with a healthcare provider due to the user’s specific circumstances (pregnancy and allergies to many medicines). The assistant also acknowledges its limitations, stating that it is not a doctor and can only provide general guidance. The response is helpful and takes into account the user’s specific situation, but it could have been more creative by suggesting more specific natural remedies that are safe during pregnancy. Overall, the assistant’s response is of high quality. Rating: [[8]]

- Table 3: Case studies on roleplay. The answers are generated by OpenChat-3.5-7B CLM.

###### OpenChat-3.5-7B Mixtral; Roleplay; ID 93;

Turn 1 Question: Imagine yourself as a doctor tasked with devising innovative remedies for various ailments and maladies. Your expertise should encompass prescribing traditional medications, herbal treatments, and alternative natural solutions. Additionally, you must take into account the patient’s age, lifestyle, and medical background while offering your recommendations. To begin, please assist me in diagnosing a scenario involving intense abdominal discomfort.

- Turn 1 Answer: As a doctor, I would first gather information about the patient’s medical history, the duration and severity of the abdominal discomfort, and any other symptoms they might be experiencing. After obtaining a thorough understanding of the patient’s situation, I would consider various factors, such as age, lifestyle, and medical background, to determine the most appropriate treatment plan. 1. Traditional Medications: If the abdominal discomfort is due to a condition such as indigestion or irritable bowel syndrome, over-the-counter medications like antacids or anti-spasmodics may be prescribed to alleviate symptoms. For more severe cases, prescription medications may be necessary, depending on the underlying cause. It is essential to monitor the patient’s response to the medication and adjust the dosage or treatment plan accordingly. 2. Herbal Treatments: Herbal remedies can be a valuable addition to a treatment plan for abdominal discomfort. Chamomile, for example, has anti-inflammatory and calming properties that can help soothe the digestive system. Peppermint oil may also be beneficial for relieving abdominal pain and reducing bloating. However, patients should consult with a healthcare professional before incorporating herbal treatments into their regimen, as some herbs may interact with other medications or have contraindications for certain individuals. 3. Alternative Natural Solutions: In addition to herbal remedies, there are other natural approaches to managing abdominal discomfort. Probiotics, for instance, can help restore balance to the gut microbiome, which may alleviate symptoms in cases of indigestion or irritable bowel syndrome. Dietary modifications, such as reducing intake of high-fat, high-sugar, or processed foods, and increasing consumption of fiber-rich foods, can also help improve digestive health. 4. Lifestyle Recommendations: Encouraging the patient to adopt a healthier lifestyle may also contribute to alleviating abdominal discomfort. Regular exercise, stress reduction techniques (such as yoga or meditation), and maintaining a healthy weight can all support optimal digestive function. In summary, the appropriate treatment for abdominal discomfort depends on the underlying cause and the patient’s unique circumstances. A combination of traditional medications, herbal treatments, alternative natural solutions, and lifestyle adjustments may be recommended to effectively manage the condition. As a doctor, I would continually assess the patient’s response to the treatment plan and make any necessary adjustments to ensure their comfort and well-being.

- Turn 1 Judgment: The assistant’s response is comprehensive, accurate, and highly relevant to the user’s question. It provides a detailed approach to diagnosing and treating abdominal discomfort, considering the patient’s age, lifestyle, and medical background. The assistant suggests traditional medications, herbal treatments, alternative natural solutions, and lifestyle adjustments, which shows a deep understanding of the topic. The assistant also emphasizes the importance of monitoring the patient’s response to the treatment plan, which is a crucial aspect of patient care. The response is creative in its approach to combining different treatment methods and is detailed in its explanation of each method. Therefore, the assistant’s response is of high quality. Rating: [[10]]

- Turn 2 Question: But I have been pregnant for 20 weeks and I am allergic to many medicines.

- Turn 2 Answer: I’m not a doctor, but I can try to provide you with some general guidance based on the information you’ve provided. Pregnancy can indeed present unique challenges when it comes to managing pain or discomfort, especially if you have allergies to certain medications. 1. Consideration of Pregnancy: During pregnancy, it’s crucial to take into account the safety of treatments and medications for both the mother and the developing baby. Always consult with your healthcare provider before starting any new treatment, as they will be able to recommend options that are safe for you and your baby. 2. Allergy Considerations: If you are allergic to many medications, it’s important to be cautious when trying new treatments. Make sure to inform your healthcare provider about your allergies, as they can help identify alternative options that are less likely to cause adverse reactions. 3. Herbal Remedies and Natural Solutions: As mentioned earlier, herbal remedies and natural solutions can be a valuable addition to a treatment plan during pregnancy. However, it’s essential to consult with your healthcare provider before using any herbal products, as some can be harmful during pregnancy. 4. Alternative Treatments: In cases where traditional medications and herbal remedies are not suitable, alternative treatments such as acupuncture, massage, or physical therapy may be considered, provided they are approved by your healthcare provider. 5. Communication with Your Healthcare Provider: It’s vital to maintain open and honest communication with your healthcare provider throughout your pregnancy. Discuss your symptoms, concerns, and any allergies you may have, so they can tailor a treatment plan that addresses your needs while ensuring the safety of both you and your baby. Remember, it’s important to consult with a qualified healthcare professional before making any decisions about your treatment plan during pregnancy. They will be able to provide personalized guidance based on your specific circumstances and medical history. Turn2 Judgment: The assistant’s response is highly relevant, accurate, and detailed. It provides a comprehensive answer to the user’s question, taking into account the user’s pregnancy and allergies to medications. The assistant emphasizes the importance of consulting with a healthcare provider before starting any new treatment, which is crucial advice for a pregnant woman. The assistant also suggests alternative treatments such as acupuncture, massage, or physical therapy, which could be helpful for the user. The assistant’s response is also creative, as it suggests a variety of potential solutions for the user’s problem. Overall, the assistant’s response is very helpful and informative. Rating: [[9]]

- Table 4: Case studies on roleplay. The answers are generated by OpenChat-3.5-7B Mixtral.

###### OpenChat-3.5-7B Solar; Roleplay; ID 93;

Turn 1 Question: Imagine yourself as a doctor tasked with devising innovative remedies for various ailments and maladies. Your expertise should encompass prescribing traditional medications, herbal treatments, and alternative natural solutions. Additionally, you must take into account the patient’s age, lifestyle, and medical background while offering your recommendations. To begin, please assist me in diagnosing a scenario involving intense abdominal discomfort.

- Turn 1 Answer: As a doctor, I would first gather information about the patient’s medical history, the duration and severity of the abdominal discomfort, and any other symptoms they may be experiencing. After obtaining a thorough understanding of the patient’s situation, I would consider various factors, such as age, lifestyle, and medical background, to determine the most appropriate treatment plan. 1. Traditional Medications: If the abdominal discomfort is due to a condition such as indigestion or irritable bowel syndrome, over-thecounter medications like antacids or anti-spasmodics may be prescribed to alleviate symptoms. 2. Herbal Treatments: Chamomile tea is a well-known herbal remedy for digestive issues. It has anti-inflammatory and antispasmodic properties that can help soothe the gut and alleviate abdominal discomfort. Peppermint oil can also be effective in reducing symptoms of irritable bowel syndrome. 3. Alternative Natural Solutions: Probiotics can help restore balance to the gut microbiome, which may alleviate abdominal discomfort. Additionally, consuming ginger, which has anti-inflammatory and antioxidant properties, may help reduce symptoms of indigestion and nausea. Considering the patient’s age, lifestyle, and medical background is crucial for determining the best course of action. For example, older individuals or those with severe abdominal pain may require additional testing or prescription medications. Those with a history of certain medical conditions may need to avoid specific herbal treatments or natural solutions. In conclusion, a multifaceted approach that combines traditional medications, herbal treatments, and alternative natural solutions, tailored to the patient’s unique needs, can be an effective way to address abdominal discomfort. However, it is essential to consult with a healthcare professional before starting any new treatment plan.

- Turn 1 Judgment: The assistant’s response is comprehensive, accurate, and relevant to the user’s question. It provides a detailed approach to diagnosing and treating abdominal discomfort, considering the patient’s age, lifestyle, and medical background. The assistant suggests traditional medications, herbal treatments, and alternative natural solutions, explaining how each could be beneficial. The assistant also emphasizes the importance of consulting with a healthcare professional before starting any new treatment plan, which is a crucial point. The response is well-structured and informative, demonstrating a good understanding of the topic. Therefore, I would rate this response highly. Rating: [[9]]

- Turn 2 Question: But I have been pregnant for 20 weeks and I am allergic to many medicines.

- Turn 2 Answer: I’m not a doctor, but I can try to provide some general guidance. It’s important to consult with your healthcare provider when dealing with pregnancy-related symptoms, as some treatments may not be safe for the mother or fetus. 1. Traditional Medications: During pregnancy, certain medications are considered safe, while others are not. Be sure to inform your healthcare provider about your allergies and any medications you’ve taken during pregnancy. They will be able to recommend safe options based on your specific situation. 2. Herbal Treatments: Some herbal remedies may be safe to use during pregnancy, while others should be avoided. Ginger, for example, is often used to treat nausea and morning sickness and is generally considered safe in moderate amounts. However, it’s important to consult with your healthcare provider before using any herbal treatments during pregnancy. 3. Alternative Natural Solutions: Pregnant women should be cautious with alternative natural solutions, as some may not be safe during pregnancy. For example, certain essential oils should not be used during pregnancy. Always consult with your healthcare provider before trying any new treatments. In summary, it’s essential to consult with your healthcare provider when dealing with pregnancy-related symptoms, especially if you have allergies to medications or other substances. They will be able to recommend safe and effective treatment options tailored to your specific needs. Turn2 Judgment: The assistant’s response is relevant, accurate, and detailed. It provides a comprehensive answer to the user’s question, taking into account the user’s pregnancy and allergies. The assistant correctly advises the user to consult with a healthcare provider before starting any new treatment, which is crucial given the user’s condition. The assistant also provides examples of treatments that might be safe during pregnancy, but again emphasizes the importance of consulting with a healthcare provider. The assistant’s response is also creative, as it provides a variety of potential solutions, including traditional medications, herbal treatments, and alternative natural solutions. Rating: [[9]]

- Table 5: Case studies on roleplay. The answers are generated by OpenChat-3.5-7B Solar.

###### FUSIONCHAT-7B VARM; Roleplay; ID 93;

Turn 1 Question: Imagine yourself as a doctor tasked with devising innovative remedies for various ailments and maladies. Your expertise should encompass prescribing traditional medications, herbal treatments, and alternative natural solutions. Additionally, you must take into account the patient’s age, lifestyle, and medical background while offering your recommendations. To begin, please assist me in diagnosing a scenario involving intense abdominal discomfort.

- Turn 1 Answer: As a doctor, I would first gather information about the patient’s medical history, the duration and severity of the abdominal discomfort, and any other symptoms they may be experiencing. After obtaining a thorough understanding of the patient’s situation, I would consider various factors, such as age, lifestyle, and medical background, to determine the most appropriate treatment plan. 1. Traditional Medications: If the abdominal discomfort is due to a condition such as indigestion or irritable bowel syndrome, over-the-counter medications like antacids or anti-spasmodics may be prescribed to alleviate symptoms. For more severe cases, prescription medications may be necessary, depending on the underlying cause. It is essential to monitor the patient’s response to the medication and adjust the dosage or treatment plan accordingly. 2. Herbal Treatments: Herbal remedies can be a valuable addition to a treatment plan for abdominal discomfort. Chamomile, for example, has anti-inflammatory and calming properties that can help soothe the digestive system. Peppermint oil may also be beneficial for relieving abdominal pain and reducing bloating. However, patients should consult with a healthcare professional before incorporating herbal treatments into their regimen, as some herbs may interact with other medications or have contraindications for certain individuals.

3. Alternative Natural Solutions: In addition to herbal remedies, there are other natural solutions that can help alleviate abdominal discomfort. These may include dietary modifications, such as avoiding trigger foods, increasing water intake, and consuming more fiber. Probiotics, which are beneficial bacteria found in fermented foods like yogurt, can also support gut health and alleviate symptoms. Stress management techniques, such as deep breathing exercises or yoga, may also help reduce abdominal discomfort in some cases. In conclusion, the treatment plan for intense abdominal discomfort will depend on the underlying cause and the patient’s individual circumstances. A combination of traditional medications, herbal treatments, and alternative natural solutions may be recommended to provide the most effective relief. Regular follow-ups with the patient will be crucial to assess their progress and make any necessary adjustments to their treatment plan.

- Turn 1 Judgment: The assistant’s response is comprehensive, accurate, and highly relevant to the user’s question. It provides a detailed approach to diagnosing and treating abdominal discomfort, considering the patient’s medical history, age, and lifestyle. The assistant also offers a variety of treatment options, including traditional medications, herbal treatments, and alternative natural solutions, explaining each in depth. The assistant emphasizes the importance of consulting with a healthcare professional before starting any treatment, which is a crucial point. The response is well-structured and informative, providing a balanced view of different treatment options. Rating: [[10]]

- Turn 2 Question: But I have been pregnant for 20 weeks and I am allergic to many medicines.

- Turn 2 Answer: I’m not a doctor, but I can try to help you with some general suggestions. If you’re pregnant and experiencing abdominal discomfort, it’s crucial to consult with your healthcare provider before trying any remedies. They will be able to take into account your specific situation, including your allergies and pregnancy status, and recommend the safest and most effective treatment options. 1. Dietary modifications: Pregnant women should avoid trigger foods that may cause abdominal discomfort, such as spicy or fatty foods, beans, and carbonated drinks. Eating smaller meals more frequently, avoiding lying down after eating, and chewing food thoroughly may also help. 2. Hydration: Drinking plenty of water is essential during pregnancy, as it can help prevent dehydration, which may contribute to abdominal discomfort. Aim for at least eight 8-ounce glasses of water per day. 3. Gentle exercise: Light exercises like walking or prenatal yoga can help improve blood circulation and alleviate abdominal discomfort. However, it is essential to consult with your healthcare provider before starting any exercise routine during pregnancy. 4. Heat or cold therapy: Applying a heating pad or warm compress to your abdomen may help relieve muscle cramps or discomfort. Alternatively, a cold compress can help reduce inflammation and soothe the area. Ensure the temperature is not too hot, and do not apply it directly to the skin – use a towel or cloth as a barrier. 5. Herbal remedies: Some herbal remedies, such as ginger, may be safe during pregnancy and can help alleviate nausea, indigestion, and abdominal discomfort. However, it is essential to consult with your healthcare provider before using any herbal treatments. 6. Probiotics: Probiotics may help improve digestion and reduce abdominal discomfort during pregnancy. Look for prenatal probiotics specifically designed for pregnant women. Remember that it is essential to consult with your healthcare provider before trying any remedies, as they can provide personalized advice based on your specific situation and pregnancy stage. Turn2 Judgment: The assistant’s response is very helpful, relevant, and detailed. It provides a comprehensive list of potential remedies for the user’s abdominal discomfort during pregnancy, taking into account the user’s allergies to many medicines. The assistant also emphasizes the importance of consulting with a healthcare provider before trying any remedies, which is crucial advice for a pregnant woman. The assistant’s response is also creative, as it suggests a variety of solutions, including dietary modifications, hydration, gentle exercise, heat or cold therapy, herbal remedies, and probiotics. The assistant’s response is accurate and shows a deep understanding of the user’s situation. Rating: [[10]]

Table 6: Case studies on roleplay. The answers are generated by FusionChat-7B VaRM.

