# arXiv:2509.09372v2[cs.RO]22Sep2025

## VLA-ADAPTER: AN EFFECTIVE PARADIGM FOR TINY-SCALE VISION-LANGUAGE-ACTION MODEL

[Figure 1]

Yihao Wang1,2,4,∗,♢ Pengxiang Ding2,3,4,∗,† Lingxiao Li1,4,5 Can Cui2,4 Zirui Ge3,4 Xinyang Tong2,4 Wenxuan Song4,6 Han Zhao2,3,4 Wei Zhao2,4 Pengxu Hou6 Siteng Huang2 Yifan Tang1 Wenhui Wang1 Ru Zhang1, Jianyi Liu1 Donglin Wang2, 1Beijing University of Posts and Telecommunications 2Westlake University 3Zhejiang University 4OpenHelix Team 5State Key Laboratory of Networking and Switching Technology 6The Hong Kong University of Science and Technology (Guangzhou) ∗Equal contribution: yh-wang@bupt.edu.cn; dingpx2015@gmail.com

Corresponding Author †Project Lead ♢Work done during interning at Westlake University

ABSTRACT

Vision-Language-Action (VLA) models typically bridge the gap between perceptual and action spaces by pre-training a large-scale Vision-Language Model (VLM) on robotic data. While this approach greatly enhances performance, it also incurs significant training costs. In this paper, we investigate how to effectively bridge vision-language (VL) representations to action (A). We introduce VLA-Adapter, a novel paradigm designed to reduce the reliance of VLA models on large-scale VLMs and extensive pre-training. To this end, we first systematically analyze the effectiveness of various VL conditions and present key findings on which conditions are essential for bridging perception and action spaces. Based on these insights, we propose a lightweight Policy module with Bridge Attention, which autonomously injects the optimal condition into the action space. In this way, our method achieves high performance using only a 0.5B-parameter backbone, without any robotic data pre-training. Extensive experiments on both simulated and real-world robotic benchmarks demonstrate that VLA-Adapter not only achieves state-of-the-art level performance, but also offers the fast inference speed reported to date. Furthermore, thanks to the proposed advanced bridging paradigm, VLA-Adapter enables the training of a powerful VLA model in just 8 hours on a single consumer-grade GPU, greatly lowering the barrier to deploying the VLA model. Project page: https://vla-adapter.github.io/.

1 INTRODUCTION

In the past two years, with significant breakthroughs in multimodal LLMs (Karamcheti et al., 2024; Steiner et al., 2024; Liu et al., 2023b; Li et al., 2025b), developing robot systems with general perception, understanding, and behavior capabilities has become a key research direction in artificial intelligence. In particular, the emergence of the Vision-Language-Action (VLA) model offers a new solution for enabling robot operations driven by instructions (Kim et al., 2024; Cui et al., 2025; Kim et al., 2025; Song et al., 2025b; Cen et al., 2025; Zhang et al., 2025b; Shi et al., 2025). Research on VLA primarily focuses on extracting multimodal information and aligning it with the action space to generate the high-quality actions (Team et al., 2024; Liu et al., 2024b; Zhong et al., 2025; Fan et al., 2025).

OpenVLA-OFT (SOTA)

VLA-Adapter(Ours)

0.5B 24.7GB 97.3%

7B 62GB 97.1%

Backbone ↓ Training VRAM (8 batch) ↓

1/14×

0.4× Maintain

71.4Hz 219.2Hz 3×

Throughput (8-dim chunk) ↑

Performance(LIBERO) ↑

Frozen Trainable

/ Bridge

Policy

V RGB L Instuction A Action

ActionQuery

Figure 1: Characteristics of VLA-Adapter. “↓” is that smaller values are better, and vice versa. Our paradigm can effectively obtain the SOTA-level VLA model using a tiny-scale backbone.

Current VLA models typically require large-scale embodied data (e.g., Open X-Embodiment (Collaboration et al., 2024) and DROID (Khazatsky et al., 2024)) to pre-train Multimodal Large Language Models (MLLMs) (Especially, VLMs) for task adaptability (Cheang et al., 2024), which is then passed to the designed Policy network (Bu et al., 2024a; Li et al., 2024) to decode or generate actions for handling the tasks in the diverse environments (Liu et al., 2023a; Mees et al., 2022).

However, when confronted with high-dimensional control environments, VLA models still face several bottlenecks, including reliance on large-scale VLMs, slow fine-tuning speed, high GPU memory (VRAM) consumption, and low inference efficiency (throughput), as shown in Figure 1. To this end, it is necessary to explore the most essential but rarely discussed question in the VLA field: How to bridge the gap of VL (vision-language representations) to A (action) more effectively?

To answer this question, we propose VLA-Adapter, a novel bridging paradigm for VLA. We systematically explore how different conditions influence action generation and give some key findings for VLA design. On this basis, we built a Policy network with Bridge Attention to autonomously inject the optimal condition into the action space. Experiments show that VLA-Adapter has superior performance, high inference efficiency, and fast throughput with a tiny-scale backbone. It significantly lowers the barrier to VLA deployment. The main contributions are summarized as follows.

- • To our knowledge, this work is the first systematic analysis of bridging paradigms’ effects on action generation. And we also give some key findings of the VLA model design.
- • VLA-Adapter transfers the sufficient multimodal information to the proposed Policy Network for action generation, effectively bridging the modality gap from VL to A.
- • Rich experiments show that VLA-Adapter has a higher success rate, smaller scale, lower tuning cost, and faster inference in diverse simulated and real-world robotic tasks.

2 RELATED WORK

- 2.1 VISION-LANGUAGE-ACTION (VLA) MODELS

Recently, leveraging pre-trained Vision-Language Models (VLMs) (Karamcheti et al., 2024; Steiner et al., 2024; Liu et al., 2023b; Li et al., 2025b) to control robots for performing various daily tasks has substantially accelerated research in embodied intelligence. This has emerged as a prominent research focus (Black et al., 2025b;a; Shukor et al., 2025; NVIDIA et al., 2025; Liu et al., 2024b; Luo et al., 2025; Cheang et al., 2025; Jiang et al., 2025; Ding et al., 2024; Bu et al., 2024b; Fan et al., 2025; Tong et al., 2024). These models are referred to as the VLA models.

Typically, VLA models require large-scale embodied datasets, such as Open X-Embodiment (Collaboration et al., 2024), for pre-training (Liu et al., 2024b; Cheang et al., 2025). This process integrates VLMs with a dedicated Policy network (Song et al., 2025a; Li et al., 2024), allowing the system to decode or generate action sequences for diverse tasks in an end-to-end manner. Moreover, dual-system VLA architectures (Shentu et al., 2024; Bu et al., 2024a; Cui et al., 2025) have recently garnered attention. These methods typically introduce an intermediate latent token to connect the VLMs and the Policy, using an asynchronous mechanism to enhance coordination between the two systems (Zhang et al., 2024). This design mitigates latency issues during action generation.

Consequently, how to effectively and efficiently bridge the gap from the vision-language perception space to the action space has become a key challenge in the design of VLA models.

- 2.2 BRIDGING FROM PERCEPTION TO ACTION SPACE

Earlier studies (Kim et al., 2024; Brohan et al., 2023a;b) attempted to directly align perception and action spaces by discretizing actions into tokens. However, this discretization inevitably introduces inherent loss. Recent studies have shifted their focus toward continuous action spaces (Liu et al., 2024a; NVIDIA et al., 2025; Black et al., 2025b; Shukor et al., 2025; Kim et al., 2025). Based on the types of perceptual features utilized to bridge to the action space, they can be categorized:

- 1) Raw Features from VLMs. Raw features (refer to vision and language representations) are extracted directly from the VLM. Early methods extract representations from the final-layer VLM,

- operating under the assumption that it encodes the most task-relevant semantic information (Liu et al., 2024a; Zhang et al., 2024). More recent methods leverage the intermediate-layer features within the VLM (Black et al., 2025b). They believe that such representations may retain richer multimodal information, thereby benefiting Policy in tasks that demand fine-grained perception or complex reasoning. For example, some studies use features from a middle layer (NVIDIA et al., 2025), the first-half layers (Shukor et al., 2025), or all intermediate-layer features (Black et al., 2025b).
- 2) Additional Query as Interface. Furthermore, recent studies (Kim et al., 2025; Cui et al., 2025) have introduced a novel interface that employs additional queries as bridges between VLMs and Policy, rather than transmitting Raw features. The query is learnable and can incorporate multimodal information, showing superior performance. The existing bridge paradigms are shown in Figure 2.

Representative work Feature type

RoboVLMs Raw features Raw features Raw features Additional Query

GR00T N1

Which layer Last

π0 OpenVLA-OFT

Intermidiate

Type

- (1)

- (2)

- (3)

- (4)

All Last

[Figure 2]

VLM Policy

(1) Action

V L

A

RGB Instuction

[Figure 3]

VLM

Query

Policy

[Figure 4]

VLM

V L

>>>>

(3)

Instuction

Policy

RGB

>>

Action A

Action A

>>>> >>>> >>>>

[Figure 5]

VLM Policy

(2) Action

V L

A

RGB Instuction

>>>>

L Instuction

(4)

V RGB

Figure 2: Existing representative bridge paradigms from VL to A.

- 3 VLA-ADAPTER METHODOLOGY

- 3.1 PRELIMINARY

We present the VLA-Adapter framework, as illustrated in Figure 3. This VLM follows the PrismaticVLMs architecture (Karamcheti et al., 2024). It has M layers. At timestep t, the input into VLM consists of {Xtv,Xtg,Lt,AQt}: the 3rd-view image Xtv, the gripper image Xtg, the instruction Lt, and additional ActionQuery AQt. After inputting Xtv and Xtg, the DINOv2 (Oquab et al., 2024) and SigLIP (Zhai et al., 2023) extract vision embeddings. Lt is tokenized. The outputs are the specified-layer Raw latent CtR and ActionQuery latent CtAQ. They serve as the conditions for Policy.

Backbone. To build a solid basis for research, we perform experiments of VLA-Adapter on different-scale backbones. The backbones select the Prismatic VLM trained on Qwen2.5-0.5B (Team, 2024), the Prismatic VLM trained on LLaMA2-7B (Touvron et al., 2023), and OpenVLA-7B

- (Kim et al., 2024) pre-trained on robotic data. The benefit gained from increasing backbone scale is limited in VLA-Adapter. The results are shown in Table 2 of Section 4.1. Therefore, to ensure efficiency, Qwen2.5-0.5B is our default backbone unless otherwise specified.

- 3.2 WHICH CONDITION IS ESSENTIAL FOR BRIDGING FROM VL TO A?

Although existing methods have adopted various bridging paradigms from VL to A, their relative effectiveness remains inconclusive. This is mainly due to the differences in the design of the VLM and the Policy. To address this gap, we explore which type of perception information is essential for action generation in the Policy network. In summary, we mainly focus on the following questions:

- Question 1.1. Which layer of features within the VLM is more effective for the Policy network?
- Question 1.2. Are the ActionQuery features a better choice than the Raw features?

###### Unified VLA-Adapter Framework Four Kinds of Conditions

Action

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

| |KV<br><br>KV|
|---|---|
| | |

Attention

Attention

VLM Policy

LN & MLP

KV

QKV

QKV

[Figure 10]

[Figure 11]

FFN

M×

M×

M×

M×

###### Attention

M layer

KV

Attention

Attention

QKV

QKV

###### M×

Condition

a) Single-layer Raw features as condition

b) Single-layer ActionQuery features as condition

FFN

Attention

2 layer

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

| |KV|
|---|---|
| |KV|
| | |

KV

Attention

Attention

QKV

QKV

FFN

ActionQuery features

M×

M×

Raw features

M×

M×

Attention

1 layer

KV

Attention

Attention

| | | |
|---|---|---|
| | | |

QKV

QKV

LN & MLP

[Figure 16]

[Figure 17]

V L

c) All-layer Raw features as condition

d) All-layer ActionQuery features as condition

Initial action

Instruction ActionQuery

RGB

Figure 3: The proposed VLA framework. The key components are the effective condition exploration and Attention design. “Attention” specifically includes cross attention with conditions and self attention with itself. In the “Unified VLA-Adapter Framework”, “Attention” is the Bridge Attention as shown in Section 3.3. Four conditions about “layer” and “type” are given on the right.

To ensure compatibility with existing experimental protocols for representative work (e.g., π0 (Black et al., 2025b)), we let the number of Policy layers be equal to that of VLM. At each layer of Policy, the action latent undergoes cross-attention with conditions and self-attention with itself. This iterative process ultimately yields the action output. Details of the Policy can be seen in Section 3.3.

- 8 4

- 8 8
- 9 2

- 9 6

- 8 4

- 8 8
- 9 2

- 9 6

R a w la te n t c n Q u e r y n t

9 2 .6

Experimental Setting. We evaluate four conditions in our framework. For Question 1.1, to evaluate the effectiveness of the individual-layer information, we employ the single-layer latent as the conditions for the all-layer Policy, as shown in Figure 3a) and 3c). To evaluate the effectiveness of all-layer information, we feed each-layer latent into the corresponding-layer Policy, as shown in Figure 3b) and 3d). For Question 1.2, to compare the effectiveness of the feature types,

9 0 .2 9 0 .6

|A tio la te<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

8 9 .8

- 7 6
- 8 0

8 0

9 / 2 4 2 4 / 2 4 7 6

S i n g l e l a y e r A l l l a y e r s

1 2 4

Figure 4: Comparison of four conditions in the VLA-Adapter framework on the LIBERO-Long. Blue and Green lines are

single-layer CtR and single-layer CtAQ, as in Figure 3a) and 3b). Blue and Green columns are all-layer CtR and all-layer CtAQ, as in Figure 3c) and 3d). The detailed results are shown in Appendix C. Please note: the number of ActionQuery is 64 here. Its number is variable, similar to MetaQueries (Pan et al., 2025) in MLLM research; we will explore it in Section 4.5.

we use the CtR or CtAQ as conditions. The comparison on the LIBERO-Long (Liu et al., 2023a), which is the long-horizon and complex benchmark, the results are as shown in Figure 4. We give the following key findings.

- Key Finding 1. Regarding CtR, the middle-layer latent performs better than the deep-layer latent. Deep-layer CtR is biased towards semantic information and less effective in action genera-

tion. The middle-layer CtR effectively integrates image and text information, retains richer multimodal details, and facilitates action generation.

- Key Finding 2. Regarding CtAQ, deep-layer latent performs better than other-layer latent. Since

ActionQuery is trained from scratch, and deep-layer CtAQ aggregates richer multimodal details and is more effectively promoting action generation than the shallow layers.

- Key Finding 3. Multi-layer features perform better. We observed that using all-layer features generally outperforms a single layer. Not only does it improve performance, but it also saves time on best layer selection during design. This design can be more universal.

Condition Determination. Does VLA-Adapter rely exclusively CtAQ as conditions? The answer is no. While all-layer CtAQ outperforms CtR, middle-layer CtR excels in some hard tasks. Comparison is shown in Table 1. So, we aim to enhance performance by using certain knowledge from CtR.

Table 1: Comparison of the ith-layer CtR and CtAQ in subtasks of LIBERO-Long.

CtR 9 13 CtAQ 1 13 17 21 23 24 All

Subtask 7 90 82 Subtask 7 76 66 74 70 70 74 76 Subtask 9 74 84 Subtask 9 78 62 58 72 72 84 78

- 3.3 POLICY WITH BRIDGE ATTENTION

Overall. For the simplicity of the model, we designed an L1-based Policy network. At t-th timestep, the input to Policy includes: {CtR,CtAQ,Aτt=0,Pt}. τ is the layer of Policy, and it has τ ∈ Z+, 0 ≤ τ ≤ M − 1. A0t is the H-step initial action of all zeros, it is processed by LayerNorm (LN) and Multi Layer Perceptron (MLP) to obtain A0t = a0t, a0t+1,..., a0t+H−1 . Pt is the proprioceptive state, and it is mapped through a two-layer MLP to obtain the proprio embedding σ0(Pt). The output is the H-step action chunk AMt −1. Each layer is composed of a Bridge Attention module and a Feed-Forward Network (FFN). The Bridge Attention architecture is shown in Figure 5.

... ...

###### ...

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

V

[Figure 23]

Gripper+3rd-person

ActionQuery

[Figure 24]

L

Instruction

Task latent

[Figure 25]

P

ActionQuery latent

Proprioceptive

###### Action

VLM

LN & MLP

Policy

[Figure 26]

[Figure 27]

FFN

###### Bridge Attention

KV

Bridge Attention

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

QKV

Concat

###### M×

###### M×

Ratio

|Q|
|---|

MultiHead Self Attention

Multi-Head Cross Attention

FFN

Multi-Head Cross Attention

VLM Policy

KV

Bridge Attention

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

QKV

|QKV|
|---|

|KV(Cond)|
|---|

|KV(Cond)|
|---|

FFN

MLP

KV

Bridge Attention QKV

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

MLP

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

| |[Figure 47]| |
|---|---|---|
| | | |

LN & MLP

[Figure 48]

[Figure 49]

V L

Instruction ActionQuery

RGB

Initial action

Figure 5: The Policy with Bridge Attention. The Policy parameters are only 97M when the backbone is Qwen2.5-0.5B. Each-layer CtR and CtAQ are integrated in Bridge Attention with the corresponding-layer action latent. Bridge Attention maps VL to Action to the greatest extent. The degree of CtR injection is learnable, ensuring the performance and stability of training.

VLM

[Figure 50]

[Figure 51]

K

Bridge Attention. The proposed Bridge Attention hopes to guide action generation to the greatest extent possible through the conditions CtR and CtAQ. Each Bridge Attention consists of two cross attentions and one self attention. In the first cross attention, CtR is processed through an MLP σ1 to obtain K1,V1. The action latent Aτt is used as the Q1, and perform attention to get CA1 Aτt ,σ1(CtR) . In the second cross attention, CtAQ needs to be concatenated with the σ0(Pt) and passed through an MLP σ2 to obtain K2,V2. Aτt is used as the Q2 to get CA2 Aτt ,σ2 CtAQ,σ0(Pt) . In the self attention, Aτt is as Q,K,V , and there is SA( Aτt , Aτt ).

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

To selectively inject certain CtR into the action space of the Policy, we introduce a learning parameter Ratio g to modulate the influence of CA1 Aτt ,σ1(CtR) . g is initialized to 0 value, and the tanh activation function is utilized tanh(g) ∈ [−1,1] to prevent extreme values from destabilizing the distribution (Zhang et al., 2023). And then, the three attentions are concatenated to obtain Aτt :

###### Aτt = [CA1 Aτt ,σ1(CtR) · tanh(g),CA2( Aτt ,σ2[CtAQ,σ0(Pt)]),SA Aτt , Aτt ]. (1)

After Bridge Attention, Aτt passes through a residual FFN to obtain Aτt+1. Repeating the above process, we finally obtain AMt −1. The action chunk AMt −1 is yielded by an LN and MLP layer.

Additionally, we also design a DiT-based (Diffusion Transformer (Peebles & Xie, 2023)) Policy. Since the diversity of Policy is not the focus of this paper, we put its details and the brief results in Appendix B. The results show that L1-based performance and inference speed are generally superior to those of the DiT-based approach. Therefore, VLA-Adapter chose the L1 architecture as the Policy.

- 3.4 TRAINING

The training is conducted end-to-end, with the Policy trained from scratch. Given a ground truth action trajectory At and action latent Aτt . We train VLA-Adapter model πθ(·) with the objective:

t,CtR,CtAQ,σ0(Pt),τ πθ(Aτt ,CtR,CtAQ,σ0(Pt),τ) − At 1 . (2) For more details of training, please see Appendix F.1.

J (θ) = EA

min

θ

- 4 EXPERIMENTS

All experiments are run on 4 NVIDIA H100 GPUs. For more details of the hyperparameters, please see Appendix F.2. We perform rich experiments to answer the following questions:

- Question 2.1. What are the advantages of the VLA-Adapter compared to other bridge paradigms?
- Question 2.2. How does VLA-Adapter perform compared to existing methods?
- Question 2.3. What else key components in the VLA-Adapter paradigm are worth exploring?

Experiment Overview. In Section 4.1, we use the long-horizon and complex LIBERO-Long (Liu

- et al., 2023a), which typically has a low success rate, to investigate the necessity of VLA-Adapter. From Section 4.2 to Section 4.4, we use LIBERO (Liu et al., 2023a) and CALVIN (Mees et al., 2022), which are widely used in VLA, as well as real-world data, to compare the performance comprehensively. In Section 4.5, we use LIBERO-Long to explore key parts of VLA-Adapter.

4.1 NECESSITY OF VLA-ADAPTER

Effectiveness of our bridge paradigm. To validate the effectiveness, we compare three kinds of backbones: · B1: The Prismatic VLM (Karamcheti et al., 2024) trained on Qwen2.5-0.5B (Team,

- 2024). · B2: The Prismatic VLM trained on LLaMA2-7B (Touvron et al., 2023). The first two are different-scale backbones without pre-training on robotic data. · B3: The OpenVLA-7B (Kim

- et al., 2024) pre-trained on robotic data. We adopted the OpenVLA-OFT bridging paradigm (Kim
- et al., 2025) for comparison. It is the existing state-of-the-art level method on major benchmarks, including LIBERO-Long (Liu et al., 2023a). The comparison results are shown in Table 2.

- Table 2: Effectiveness comparison with OpenVLA-OFT (Kim et al., 2025) on the LIBERO-Long (Liu et al., 2023a). “Fine-tuned” is by LoRA fine-tuning (Hu et al., 2022). Bold represents the best

performance. Please note, comparison with the bridge paradigms of π0 (Black et al., 2025b) and GR00T N1 (NVIDIA et al., 2025) has been included in Section 3, so we will not compare it here.

Fine-tuned B1 +OFT B1 +Ours B2 +OFT B2 +Ours B3 +OFT B3 +Ours Success Rate (%) ↑ 85.8 95.0 (9.2% ↑) 87.5 95.2 (7.7% ↑) 94.5 95.4 (0.9% ↑)

Fortunately, VLA-Adapter remains effective when the backbone is frozen. Only the ActionQuery and Policy are trained from scratch. SmolVLA (Shukor et al., 2025) is the VLA dedicated to studying frozen VLMs. So, we compare with OpenVLA-OFT and SmolVLA. The results are shown in Table 3. Since the results of GR00T N1 come from (Song et al., 2025a), it did a full-params tuning, so we will not compare with it here. Based on Tables 2 and 3, we summarize two conclusions:

- Table 3: Effectiveness comparison when the backbone is frozen. Benchmark is the same as Table 2. For a detailed analysis of OpenVLA-OFT (Kim et al., 2025) does not work, please see Appendix H.

Frozen OpenVLA-OFT SmolVLA VLA-Adapter Success Rate (%) ↑ 0.0 77.0 86.4

- Conclusion 1. VLA-Adapter improvement is obvious when VLMs without robotic pre-training.
- Conclusion 2. Even if the backbone freezes, VLA-Adapter still performs strongly.

This can be attributed to the fact that, after pre-training on robotic data, the last-layer features are already adapted to the action domain, enabling efficient fine-tuning with a simple MLP. However, when VLMs without pre-training, relying solely on the last-layer latents, are insufficient for effective action mapping. So, adopting the VLA-Adapter becomes crucial to achieve efficient fine-tuning. These insights highlight a key advantage: VLA-Adapter facilitates efficient fine-tuning of VLMs without robotic pre-training, achieving performance that surpasses baselines using a tiny backbone.

Efficiency. VLA-Adapter attains a faster inference speed. The comparison is shown in Table 4.

- Table 4: Inference efficiency comparison with OpenVLA (Kim et al., 2024) and OpenVLA-OFT

- (Kim et al., 2025). The action chunk is 8 dimensions, consistent with most VLA. “OpenVLA-OFT

(wo Xtg, P)” is the L1-based version where the input is without the gripper image and proprioceptive state. It is the fastest version of OpenVLA-OFT. Benchmark is the same as Table 2.

Efficiency OpenVLA OpenVLA-OFT (wo Xtg, P) OpenVLA-OFT VLA-Adapter Throughput (Hz) ↑ 4.2 109.7 71.4 219.2

Latency (Sec) ↓ 0.2396 0.0729 0.1120 0.0365

- 4.2 OVERALL PERFORMANCE ON VARIOUS TASKS

Benchmark. We selected the widely adopted LIBERO benchmark (Liu et al., 2023a) to evaluate performance across various types of tasks. LIBERO1 provides multiple suites, including Spatial, Object, Goal, and Long. For detailed settings and examples of LIBERO, please see Appendix A.

Baselines. We selected recently published, comprehensive, and high-performance VLA works as comparison baselines. They are Large: 1. FlowVLA (Zhong et al., 2025), 2. UnifiedVLA (Li et al.,

- 2025a), 3. OpenVLA (Kim et al., 2024), 4. OpenVLA-OFT (Kim et al., 2025), 5. UniVLA (Bu et al., 2025), 6. CoT-VLA (Zhao et al., 2025a), 7. WorldVLA (Cen et al., 2025), 8. TraceVLA (Zheng et al., 2024), 9. MolmoAct (Lee et al., 2025), 10. ThinkAct (Huang et al., 2025), and 11. PD-VLA (Song et al., 2025b); Small: 12. 4D-VLA (Zhang et al., 2025a), 13. SpatialVLA (Qu

et al., 2025), 14. π0 (Black et al., 2025b), 15. π0-FAST (Pertsch et al., 2025), 16. NORA (Hung et al., 2025), 17. SmolVLA (Shukor et al., 2025), 18. GR00T N1 (NVIDIA et al., 2025), and 19. GraspVLA (Deng et al., 2025); Tiny: 20. Seer (Tian et al., 2025), 21. VLA-OS (Gao et al., 2025), and 22. Diffusion Policy (Chi et al., 2023). Their performances are all derived from original references or the reproduction of other published works, ensuring objectivity and accuracy.

Metrics. Each subtask is repeated 50× times to evaluate. We use the commonly used metric “Success Rate”, reported as ranging from 0 to 100, with higher values meaning better performance.

Results. Comparison on the LIBERO is shown in Table 5. The results in Table 5 demonstrate that VLA-Adapter, using only a tiny-scale backbone, can achieve performance comparable to OpenVLAOFT with 14× larger. It surpasses representative works such as π0, SmolVLA, and GR00T N1. In addition, VLA-Adapter has a notable advantage of 29.0% over VLA-OS with the same-scale backbone on LIBERO-Long. These demonstrate the VLA-Adapter superiority on various tasks.

1https://libero-project.github.io/datasets

- Table 5: Comparison on the LIBERO benchmark. Bold* is the best performance, Bold is the suboptimal performance, and Italics is the third best performance. † represents that the non-based-VLM baselines. “Scratch” is the work without pre-training on robotic data. “Params” is the backbone scale, and its unit is “Billion”. We give the performance on subtasks. It is shown in Table D1 of Appendix D. Recently, we have updated the VLA-Adapter-Pro model. Its Policy architecture is the same as Figure 5, and we optimized the implementation. For its details, please see Appendix I.

LIBERO Params Spatial Object Goal Long Avg.

FlowVLA (Zhong et al., 2025) (ArXiv) 8.5 93.2 95.0 91.6 72.6 88.1 UnifiedVLA (Li et al., 2025a) (ArXiv) 8.5 95.4 98.8 93.6 94.0 95.5 OpenVLA (Kim et al., 2024) (CoRL) 7 84.7 88.4 79.2 53.7 76.5

OpenVLA-OFT (Kim et al., 2025) (RSS) 7 97.6 98.4 97.9 94.5 97.1

UniVLA (Bu et al., 2025) (RSS) 7 96.5 96.8 95.6 92.0 95.2 CoT-VLA (Zhao et al., 2025a) (CVPR) 7 87.5 91.6 87.6 69.0 81.1

Large

WorldVLA (Cen et al., 2025) (ArXiv) 7 87.6 96.2 83.4 60.0 81.8 TraceVLA (Zheng et al., 2024) (ArXiv) 7 84.6 85.2 75.1 54.1 74.8

MolmoAct (Lee et al., 2025) (ArXiv) 7 87.0 95.4 87.6 77.2 86.6 ThinkAct (Huang et al., 2025) (ArXiv) 7 88.3 91.4 87.1 70.9 84.4 PD-VLA (Song et al., 2025b) (ArXiv) 7 95.5 96.7 94.9 91.7 94.7

4D-VLA (Zhang et al., 2025a) (ArXiv) 4 88.9 95.2 90.9 79.1 88.6 SpatialVLA (Qu et al., 2025) (RSS) 4 88.2 89.9 78.6 55.5 78.1

π0 (Black et al., 2025b) (RSS) 3 96.8 98.8 95.8 85.2 94.2 π0-FAST (Pertsch et al., 2025) (RSS) 3 96.4 96.8 88.6 60.2 85.5

Small

NORA (Hung et al., 2025) (ArXiv) 3 92.2 95.4 89.4 74.6 87.9 SmolVLA (Shukor et al., 2025) (ArXiv) 2.2 93.0 94.0 91.0 77.0 88.8

GR00T N1 (NVIDIA et al., 2025) (ArXiv) 2 94.4 97.6 93.0 90.6 93.9 GraspVLA (Deng et al., 2025) (ArXiv) 1.8 - 94.1 91.2 82.0 89.1

Seer† (Tian et al., 2025) (Scratch) (ICLR) 0.57 - - - 78.7 78.7

VLA-OS (Gao et al., 2025) (ArXiv) 0.5 87.0 96.5 92.7 66.0 85.6 Diffusion Policy† (Chi et al., 2023) (RSS) - 78.3 92.5 68.3 50.5 72.4

Tiny

VLA-Adapter (Ours) 0.5 97.8 99.2 97.2 95.0 97.3 VLA-Adapter-Pro (Ours) 0.5 99.6* 99.6* 98.2* 96.4* 98.5*

- 4.3 PERFORMANCE ON GENERALIZATION TASKS

We used the CALVIN ABC→D (Mees et al., 2022) to evaluate the performance on the zero-shot generalization tasks. CALVIN consists of four environments (Env A, B, C, and D)2 . “ABC→D” means it trains on Env A, B, and C and evaluates on Env D. VLA needs to execute a preset sequence of 1,000 tasks in sequence. Each task row consists of five subtasks. The model can only proceed to the next subtask after completing the current one. Please see Appendix E for more settings.

Baselines. We selected recently published works as baselines. They are Large: 1. UniVLA (Bu et al., 2025), 2. OpenVLA (Kim et al., 2024), 3. OpenVLA-OFT (Kim et al., 2025), 4. VLAS (Zhao et al., 2025b), 5. LCB (Shentu et al., 2024), 6. RoboDual (Bu et al., 2024a), 7. OpenHelix (Cui et al., 2025), and 8. ReconVLA (Song et al., 2025c); Small: 9. DeeR (Yue et al., 2024), 10. RoboFlamingo (Li et al., 2024), 11. VPP (Hu et al., 2025), and 12. SuSIE (Black et al., 2024); Tiny: 13. MoDE (Reuss et al., 2025) and 14. Seer (Tian et al., 2025). The results of these baselines are based on original references or other published works, ensuring objectivity and correctness. Since the original OpenVLA-OFT paper (Kim et al., 2025) did not perform experiments on CALVIN ABC→D, we used its source codes to run 150,000 steps and took the best performance.

Metrics. We use the widely used “Success Rate” (the same in LIBERO (Liu et al., 2023a)) and “Avg. len” of completed tasks (the larger the better, with values between 0-5) as metrics.

Results. Comparison on the CALVIN is shown in Table 6. The results in Table 6 show that VLAAdapter has strong generalization ability, and its average length is better than SOTA baselines.

2http://calvin.cs.uni-freiburg.de/

- Table 6: Comparison on the CALVIN ABC→D benchmark. Bold* is the best performance, Bold is the suboptimal performance, and Italics is the third best performance. † represents that the nonbased-VLM method. Recently, we have updated the VLA-Adapter-Pro. Its Policy architecture is the same as Figure 5, and we optimized the implementation. For its details, please see Appendix I.

1 Task2completed3 in a4row ↑ 5 Avg. len ↑

CALVIN ABC→D Params

UniVLA (Bu et al., 2025) (RSS) 7 95.5 85.8 75.4 66.9 56.5 3.80 OpenVLA (Kim et al., 2024) (CoRL) 7 91.3 77.8 62.0 52.1 43.5 3.27

OpenVLA-OFT (Kim et al., 2025) (RSS) 7 96.3 89.1 82.4 75.8 66.5 4.10 VLAS (Zhao et al., 2025b) (ICLR) 7 87.2 64.2 40.9 28.1 19.6 2.40

Large

LCB (Shentu et al., 2024) (IROS) 7 73.6 50.2 28.5 16.0 9.9 1.78 RoboDual (Bu et al., 2024a) (ArXiv) 7 94.4 82.7 72.1 62.4 54.4 3.66 OpenHelix (Cui et al., 2025) (ArXiv) 7 97.1 91.4 82.8 72.6 64.1 4.08

ReconVLA (Song et al., 2025c) (ArXiv) 7 95.6 87.6 76.9 69.3 64.1 3.95

DeeR (Yue et al., 2024) (NeurIPS) 3 86.2 70.1 51.8 41.5 30.4 2.82 RoboFlamingo (Li et al., 2024) (ICLR) 3 82.4 61.9 46.6 33.1 23.5 2.48

Small

VPP† (Hu et al., 2025) (ICML) 1.5 95.7 91.2 86.3 81.0 75.0 4.33 SuSIE (Black et al., 2024) (ICLR) 1.3 87.0 69.0 49.0 38.0 26.0 2.69

SeerLarge† (Tian et al., 2025) (ICLR) 0.57 96.3 91.6 86.1 80.3 74.0 4.28 MoDE† (Reuss et al., 2025) (ICLR) 0.44 96.2 88.9 81.1 71.8 63.5 4.01

Tiny

Seer† (Tian et al., 2025) (ICLR) 0.32 94.4 87.2 79.9 72.2 64.3 3.98 VLA-Adapter (Ours) 0.5 99.1* 94.6 88.8 82.8 76.5 4.42

###### VLA-Adapter-Pro (Ours) 0.5 98.5 95.0* 90.5* 85.3* 80.0* 4.50*

- 4.4 PERFORMANCE ON REAL-WORLD TASKS

Experimental settings. We use a robotic system to perform real-world tasks. A 6-DOF Synria Alicia-D equipped with a 1-DOF gripper is employed, and it uses Logitech C920e and RealSense D405 cameras to capture the third-view and gripper images. The real-world robotic system is shown in Figure 6. We evaluate the VLA-Adapter method across four experimental categories:

- 1) Simple pick-and-place tasks with objects spanning diverse materials and geometries.
- 2) CALVIN-inspired challenging task II: lateral block relocation (e.g. “Move <obj> left/right”).
- 3) CALVIN-inspired challenging manipulation task I: “Block stacking”.
- 4) LIBERO-inspired complex and long-horizon task: (e.g. “Pick up the spoon and place it on the cup, then place the cup on the plate”).

To strengthen evaluation rigor and assess generalization performance, we randomize the object positions at test time to induce distribution shift and increase task difficulty.

###### Real-World Settings Task Examples

|[Figure 56]<br><br>6-DoF Synria Alicia-D (Follower Arm)<br><br>Synria Alicia-D (Lead Arm)<br><br>Operational Workspace<br><br>Diverse Objects<br><br>Third-view Camera (Logitech C920e)<br><br>Gripper Camera (RealSense D405)|
|---|

|[Figure 57]|
|---|

|[Figure 58]|
|---|

|[Figure 59]|
|---|

e.g.: Pick up the spoon and place it on the cup, then place the cup on the plate

|[Figure 60]|
|---|

|[Figure 61]|
|---|

|[Figure 62]|
|---|

Figure 6: Real-world system Synria Alicia-D and the task examples.

Baselines. ACT (Zhao et al., 2023) and OFT-style variant (Kim et al., 2025) are as baselines.

Results. The comparison results are shown in Figure 7. Each result is obtained by averaging the results of 10 executions. Experimental results show that VLA-Adapter has better generalization capabilities in various scenarios. Therefore, VLA-Adapter greatly lowers the barrier to adopting VLA in practical applications. More real-world experiments are detailed in Appendix G.

ACT 0.5B+OFT VLA-Adapter

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

90

[Figure 67]

###### Success Rate (%)

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

60

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

30

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

###### 0

[Figure 98]

[Figure 99]

[Figure 100]

Pick Move Stack Long Avg.

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

1) Pick

- 3) Stack

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

| |
|---|

- 4) Long

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

| |
|---|

[Figure 119]

| |
|---|

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

2) Move

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

| |
|---|

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Figure 7: Comparison on real-world tasks.

- 4.5 ABLATION EXPERIMENTS

We explore three key components in the VLA-Adapter: 1. Number of ActionQuery, 2. Condition type, and 3. Injection degree for Policy. The benchmark is LIBERO-Long (Liu et al., 2023a).

Number of ActionQuery. In our paradigm, the number of ActionQuery is not fixed. To explore the impact of this number on performance, we conducted the following experiments by varying the number of ActionQuery to 1, 4, 8, 16, 64, 128, 256, and 512. The results are shown in Figure 8. Thus, using too few ActionQuery tokens weakens multimodal aggregation and makes it challenging to condition the Policy. Conversely, employing too many ActionQuery tokens introduces redundancy, interfering with the performance. Therefore, we selected 64 ActionQuery tokens. This number provides the optimal balance between performance and efficiency.

- 8 4

- 8 8
- 9 2

- 9 6 9 5 9 3

L a s t- la y e r A c tio n Q u e r y

F u ll V L A - A d a p te r

- 7 6
- 8 0 N u m b e r o f A c tio n Q u e r y

1 4 8 1 6 6 4 1 2 8 2 5 6 5 1 2

Figure 8: Comparison of the different numbers of ActionQuery. The blue line shows the result of using only the last-layer ActionQuery. The red star shows the result of the full VLA-Adapter.

Condition Type. In Section 3, we analyzed the overall effects of different conditions on action generation. Here, we present the complete comparison results based on the four classic paradigms

- in Section 2, as shown in Table 7. This result demonstrates that using both all-layer Raw and ActionQuery achieves superior performance, indirectly validating the superiority of our bridge paradigm.

Injection Degree for Policy. In the Bridge Attention, we use learnable parameters to control the injection degree of Raw features CtR and set the injection degree of ActionQuery features CtAQ to 1. Here, we explore other injection degrees, and the comparison results are shown in Table 8. Two conclusions can be drawn from the results in Table 8: From 1) and 2), the performance of CtR is inferior to CtAQ, so CtR should inject some effective information into Policy through learning. From 1) and 4), CtAQ aggregates multimodal information, which is beneficial for action generation; it needs to be injected fully into Policy. This result confirms that the Bridge Attention is effective.

- Table 7: Comparison with different condition types. The style can be summarized as representative works in Figure 2 of Section 1. “N/A” represents no such method. “Bold” is the best performance.

Layer Raw ActionQuery Style SR ↑ Last

✓ ✗ RoboVLMs (Liu et al., 2024a) 85.8 ✗ ✓ OpenVLA-OFT (Kim et al., 2025) 90.2

Intermidiate ✓ ✗ GR00T N1 (NVIDIA et al., 2025) 88.4

✓ ✗ π0 (Black et al., 2025b) 90.6 ✗ ✓ N/A 92.6 ✓ ✓ VLA-Adapter (Ours) 95.0

All

Table 8: Ablation of other injection degrees. Raw ActionQuery Success Rate (%)

##### 1) (VLA-Adapter) tanh(g) 1 95.0

- 2) 1 1 91.4

- 3) 1 tanh(g) 91.0

- 4) tanh(g) tanh(g) 92.6

- 5 CONCLUSION

We propose VLA-Adapter, a novel and efficient bridging paradigm for VLA. By leveraging Raw and ActionQuery latent, this method effectively transfers multimodal knowledge to the Policy to generate action. Experiments show that VLA-Adapter achieves SOTA performance using a tinyscale backbone. Even when the VLM is frozen, it has strong performance. In addition, our method has low VRAM usage and high inference speed. These results suggest that VLA-Adapter alleviates VLA’s reliance on large-scale VLMs and huge training costs, lowering the barrier to deploying VLA.

Ultimately, we hope the VLA-Adapter method and key findings of this study can provide a solid basis for future research in the VLA and inspire the development of more advanced VLA methods!

- 6 LIMITATIONS

While VLA-Adapter achieves lightweight and excellent performance, it also has some limitations. First, because VLA-Adapter is not pre-trained on a large amount of embodied data and the scale is tiny, its generalization in real-world systems needs to be improved. Secondly, the quality of the actions generated by the Policy networks depends on the conditions provided by the VLM and how they are used. Therefore, future work can further explore these conditions to improve its representation and ensure its efficient use. Finally, the fundamental training process of the VLA-Adapter is still relatively simple, and the complex processes, such as reinforcement learning, can be explored.

ACKNOWLEDGMENTS

This work was supported in part by the National Natural Science Foundation of China under Grant U21B2020, and the BUPT Excellent Ph.D. Students Foundation under Grant CX20241055.

REFERENCES

Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya, Homer Walke, Chelsea Finn, Aviral Kumar, and Sergey Levine. Zero-shot robotic manipulation with pretrained image-editing diffusion models. arXiv preprint arXiv:2310.10639, 2024. 8, 9

Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom,

Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π0. 5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025a. 2

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. corr, abs/2410.24164, 2024. doi: 10.48550. arXiv preprint arXiv:2410.24164, 2025b. 2, 3, 4, 6, 7, 8, 11

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pp. 2165–2183. PMLR, 2023a. 2

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez, Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2023b. 2

Qingwen Bu, Hongyang Li, Li Chen, Jisong Cai, Jia Zeng, Heming Cui, Maoqing Yao, and Yu Qiao. Towards synergistic, generalized, and efficient dual-system for robotic manipulation. arXiv preprint arXiv:2410.08001, 2024a. 2, 8, 9

Qingwen Bu, Jia Zeng, Li Chen, Yanchao Yang, Guyue Zhou, Junchi Yan, Ping Luo, Heming Cui, Yi Ma, and Hongyang Li. Closed-loop visuomotor control with generative expectation for robotic manipulation. Advances in Neural Information Processing Systems, 37:139002–139029, 2024b. 2

Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. Univla: Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2505.06111, 2025. 7, 8, 9

Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, Deli Zhao, and Hao Chen. Worldvla: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025. 1, 7, 8

Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, Hanbo Zhang, and Minzhao Zhu. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024. 2

Chilam Cheang, Sijin Chen, Zhongren Cui, Yingdong Hu, Liqun Huang, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Xiao Ma, Hao Niu, Wenxuan Ou, Wanli Peng, Zeyu Ren, Haixin Shi, Jiawen Tian, Hongtao Wu, Xin Xiao, Yuyang Xiao, Jiafeng Xu, and Yichu Yang. Gr-3 technical report. arXiv preprint arXiv:2507.15493, 2025. 2

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, pp. 02783649241273668, 2023. 7, 8

Open X-Embodiment Collaboration, Abhishek Padalkar, Acorn Pooley, Ajay Mandlekar, Ajinkya Jain, Albert Tung, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khazatsky, Anant Rai, Anikait Singh, Animesh Garg, Anthony Brohan, Antonin Raffin, Ayzaan Wahid, Ben BurgessLimerick, Beomjoon Kim, Bernhard Sch¨olkopf, Brian Ichter, Cewu Lu, Charles Xu, Chelsea Finn, Chenfeng Xu, Cheng Chi, Chenguang Huang, Christine Chan, Chuer Pan, Chuyuan Fu, Coline Devin, Danny Driess, Deepak Pathak, Dhruv Shah, Dieter B¨uchler, Dmitry Kalashnikov, Dorsa Sadigh, Edward Johns, Federico Ceola, Fei Xia, Freek Stulp, Gaoyue Zhou, Gaurav S. Sukhatme, Gautam Salhotra, Ge Yan, Giulio Schiavi, Gregory Kahn, Hao Su, Hao-Shu Fang, Haochen Shi, Heni Ben Amor, Henrik I. Christensen, Hiroki Furuta, Homer Walke, Hongjie Fang, Igor Mordatch, Ilija Radosavovic, Isabel Leal, Jacky Liang, Jad Abou-Chakra, Jaehyung Kim, Jan Peters, Jan Schneider, Jasmine Hsu, Jeannette Bohg, Jeffrey Bingham, Jiajun Wu, Jialin Wu, Jianlan Luo, Jiayuan Gu, Jie Tan, Jihoon Oh, Jitendra Malik, Jonathan Booher, Jonathan Tompson, Jonathan Yang, Joseph J. Lim, Jo˜ao Silv´erio, Junhyek Han, Kanishka Rao, Karl Pertsch, Karol Hausman, Keegan Go, Keerthana Gopalakrishnan, Ken Goldberg, Kendra Byrne, Kenneth Oslund, Kento Kawaharazuka, Kevin Zhang, Krishan Rana, Krishnan Srinivasan, Lawrence Yunliang Chen, Lerrel Pinto, Li FeiFei, Liam Tan, Lionel Ott, Lisa Lee, Masayoshi Tomizuka, Max Spero, Maximilian Du, Michael Ahn, Mingtong Zhang, Mingyu Ding, Mohan Kumar Srirama, Mohit Sharma, Moo Jin Kim, Naoaki Kanazawa, Nicklas Hansen, Nicolas Heess, Nikhil J. Joshi, Niko Suenderhauf, Norman Di Palo, Nur Muhammad Mahi Shafiullah, Oier Mees, Oliver Kroemer, Pannag R. Sanketi, Paul Wohlhart, Peng Xu, Pierre Sermanet, Priya Sundaresan, Quan Vuong, Rafael Rafailov, Ran Tian, Ria Doshi, Roberto Mart´ın-Mart´ın, Russell Mendonca, Rutav Shah, Ryan Hoque, Ryan Julian, Samuel Bustamante, Sean Kirmani, Sergey Levine, Sherry Moore, Shikhar Bahl, Shivin Dass, Shubham Sonawani, Shuran Song, Sichun Xu, Siddhant Haldar, Simeon Adebola, Simon Guist, Soroush Nasiriany, Stefan Schaal, Stefan Welker, Stephen Tian, Sudeep Dasari, Suneel Belkhale, Takayuki Osa, Tatsuya Harada, Tatsuya Matsushima, Ted Xiao, Tianhe Yu, Tianli Ding, Todor Davchev, Tony Z. Zhao, Travis Armstrong, Trevor Darrell, Vidhi Jain, Vincent Vanhoucke, Wei Zhan, Wenxuan Zhou, Wolfram Burgard, Xi Chen, Xiaolong Wang, Xinghao Zhu, Xuanlin Li, Yao Lu, Yevgen Chebotar, Yifan Zhou, Yifeng Zhu, Ying Xu, Yixuan Wang, Yonatan Bisk, Yoonyoung Cho, Youngwoon Lee, Yuchen Cui, Yueh-Hua Wu, Yujin Tang, Yuke Zhu, Yunzhu Li, Yusuke Iwasawa, Yutaka Matsuo, Zhuo Xu, and Zichen Jeff Cui. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 6892–6903. IEEE, 2024. 2

Can Cui, Pengxiang Ding, Wenxuan Song, Shuanghao Bai, Xinyang Tong, Zirui Ge, Runze Suo, Wanqi Zhou, Yang Liu, Bofang Jia, Han Zhao, Siteng Huang, and Donglin Wang. Openhelix: A short survey, empirical analysis, and open-source dual-system vla model for robotic manipulation. arXiv preprint arXiv:2505.03912, 2025. 1, 2, 3, 8, 9

Shengliang Deng, Mi Yan, Songlin Wei, Haixin Ma, Yuxin Yang, Jiayi Chen, Zhiqi Zhang, Taoyu Yang, Xuheng Zhang, Wenhao Zhang, Heming Cui, Zhizheng Zhang, and He Wang. Graspvla: a grasping foundation model pre-trained on billion-scale synthetic action data. arXiv preprint arXiv:2505.03233, 2025. 7, 8

Pengxiang Ding, Han Zhao, Wenjie Zhang, Wenxuan Song, Min Zhang, Siteng Huang, Ningxi Yang, and Donglin Wang. Quar-vla: Vision-language-action model for quadruped robots. In European Conference on Computer Vision, pp. 352–367. Springer, 2024. 2

Yiguo Fan, Pengxiang Ding, Xinyang Tong Shuanghao Bai, Yuyang Zhu, Hongchao Lu, Fengqi Dai, Wei Zhao, Yang Liu, Zhaoxin Fan Siteng Huang, Badong Chen, and Donglin Wang. Longvla: Unleashing long-horizon capability of vision language action model for robot manipulation. arXiv preprint arXiv:2508.19958, 2025. 1, 2

Chongkai Gao, Zixuan Liu, Zhenghao Chi, Junshan Huang, Xin Fei, Yiwen Hou, Yuxuan Zhang, Yudi Lin, Zhirui Fang, Zeyu Jiang, and Lin Shao. Vla-os: Structuring and dissecting planning representations and paradigms in vision-language-action models. arXiv preprint arXiv:2506.17561, 2025. 7, 8

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 6, 21

Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations. arXiv preprint arXiv:2412.14803, 2025. 8, 9

Chi-Pin Huang, Yueh-Hua Wu, Min-Hung Chen, Yu-Chiang Frank Wang, and Fu-En Yang. Thinkact: Vision-language-action reasoning via reinforced visual latent planning. arXiv preprint arXiv:2507.16815, 2025. 7, 8

Chia-Yu Hung, Qi Sun, Pengfei Hong, Amir Zadeh, Chuan Li, U Tan, Navonil Majumder, Soujanya Poria, et al. Nora: A small open-sourced generalist vision language action model for embodied tasks. arXiv preprint arXiv:2504.19854, 2025. 7, 8

Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, and Hang Zhao. Galaxea open-world dataset and g0 dual-system vla model.

2025. 2

Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. In Forty-first International Conference on Machine Learning (ICML), 2024. 1, 2, 3, 6

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe, Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z. Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul Foster, Jensen Gao, Vitor Guizilini, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Muhammad Zubair Irshad, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen, Abigail O’Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek Gupta, Dinesh Jayaraman, Joseph J Lim, Jitendra Malik, Roberto Mart´ın-Mart´ın, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu, Thomas Kollar, Sergey Levine, and Chelsea Finn. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024. 2

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model. In The Conference on Robot Learning (CoRL), 2024. 1, 2, 3, 6, 7, 8, 9

Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025. 1, 2, 3, 6, 7, 8, 9, 10, 11, 20

Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, Winson Han, Wilbert Pumacay, Angelica Wu, Rose Hendrix, Karen Farley, Eli VanderBilt, Ali Farhadi, Dieter Fox, and Ranjay Krishna. Molmoact: Action reasoning models that can reason in space. arXiv preprint arXiv:2508.07917, 2025. 7, 8

Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model. arXiv preprint arXiv:2503.00200, 2025a. 7, 8

Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, et al. Vision-language foundation models as effective robot imitators. arXiv preprint arXiv:2311.01378, 2024. 2, 8, 9

Zhiqi Li, Guo Chen, Shilong Liu, Shihao Wang, Vibashan VS, Yishen Ji, Shiyi Lan, Hao Zhang, Yilin Zhao, Subhashree Radhakrishnan, Nadine Chang, Karan Sapra, Amala Sanjay Deshmukh, Tuomas Rintamaki, Matthieu Le, Ilia Karmanov, Lukas Voegtle, Philipp Fischer, De-An Huang, Timo Roman, Tong Lu, Jose M. Alvarez, Bryan Catanzaro, Jan Kautz, Andrew Tao, Guilin Liu, and Zhiding Yu. Eagle 2: Building post-training data strategies from scratch for frontier visionlanguage models. arXiv preprint arXiv:2501.14818, 2025b. 1, 2

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023a. 2, 4, 6, 7, 8, 10, 18, 20, 21, 28

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023b. 1, 2

Huaping Liu, Xinghang Li, Peiyan Li, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Tao Kong, and Hanbo Zhang. Towards generalist robot policies: What matters in building vision-language-action models. 2024a. 2, 3, 11

Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024b. 1, 2

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2019. 21

Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, and Zongqing Lu. Being-h0: vision-language-action pretraining from largescale human videos. arXiv preprint arXiv:2507.15597, 2025. 2

Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3):7327–7334, 2022. 2, 6, 8, 21

NVIDIA, Johan Bjorck, Fernando Casta˜neda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Jim Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025. 2, 3, 6, 7, 8, 11

Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2024. 3

Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, and Saining Xie. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025. 4

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023. 6, 18

Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025. 7, 8

Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visuallanguage-action model. arXiv preprint arXiv:2501.15830, 2025. 7, 8

Moritz Reuss, Jyothish Pari, Pulkit Agrawal, and Rudolf Lioutikov. Efficient diffusion transformer policies with mixture of expert denoisers for multitask learning. arXiv preprint arXiv:2412.12953,

2025. 8, 9

Yide Shentu, Philipp Wu, Aravind Rajeswaran, and Pieter Abbeel. From llms to actions: Latent codes as bridges in hierarchical robot control. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 8539–8546. IEEE, 2024. 2, 8, 9

Hao Shi, Bin Xie, Yingfei Liu, Lin Sun, Fengrong Liu, Tiancai Wang, Erjin Zhou, Haoqiang Fan, Xiangyu Zhang, and Gao Huang. Memoryvla: Perceptual-cognitive memory in vision-languageaction models for robotic manipulation. 2025. 1

Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Michel Aractingi Adil Zouitine, Caroline Pascal, Martino Russi, Andres Marafioti, Simon Alibert, Matthieu Cord, Thomas Wolf, and Remi Cadene. Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025. 2, 3, 6, 7, 8

Haoming Song, Delin Qu, Yuanqi Yao, Qizhi Chen, Qi Lv, Yiwen Tang, Modi Shi, Guanghui Ren, Maoqing Yao, Bin Zhao, Dong Wang, and Xuelong Li. Hume: Introducing system-2 thinking in visual-language-action model. arXiv preprint arXiv:2505.21432, 2025a. 2, 6

Wenxuan Song, Jiayi Chen, Pengxiang Ding, Han Zhao, Wei Zhao, Zhide Zhong, Zongyuan Ge, Jun Ma, and Haoang Li. Accelerating vision-language-action model integrated with action chunking via parallel decoding. arXiv preprint arXiv:2503.02310, 2025b. 1, 7, 8

Wenxuan Song, Ziyang Zhou, Han Zhao, Jiayi Chen, Pengxiang Ding, Haodong Yan, Yuxin Huang, Feilong Tang, Donglin Wang, and Haoang Li. Reconvla: Reconstructive vision-language-action model as effective robot perceiver. arXiv preprint arXiv:2508.10333, 2025c. 8, 9

Andreas Steiner, Andr´e Susano Pinto, Michael Tschannen, Daniel Keysers, Xiao Wang, Yonatan Bitton, Alexey Gritsenko, Matthias Minderer, Anthony Sherbondy, Shangbang Long, Siyang Qin, Reeve Ingle, Emanuele Bugliarello, Sahar Kazemzadeh, Thomas Mesnard, Ibrahim Alabdulmohsin, Lucas Beyer, and Xiaohua Zhai. Paligemma 2: A family of versatile vlms for transfer. arXiv preprint arXiv:2412.03555, 2024. 1, 2

Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024. 1

Qwen Team. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2, 2024. 3, 6 Yang Tian, Sizhe Yang, Jia Zeng, Ping Wang, Dahua Lin, Hao Dong, and Jiangmiao Pang. Pre-

dictive inverse dynamics models are scalable learners for robotic manipulation. arXiv preprint arXiv:2412.15109, 2025. 7, 8, 9

Xinyang Tong, Pengxiang Ding, Yiguo Fan, Donglin Wang, Wenjie Zhang, Can Cui, Mingyang Sun, Han Zhao, Hongyin Zhang, Yonghao Dang, Siteng Huang, and Shangke Lyu. Quart-online: Latency-free large multimodal language model for quadruped robot learning. arXiv preprint arXiv:2412.15576, 2024. 2

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 3, 6

Yang Yue, Yulin Wang, Bingyi Kang, Yizeng Han, Shenzhi Wang, Shiji Song, Jiashi Feng, and Gao Huang. Deer-vla: Dynamic inference of multimodal large language models for efficient robot execution. Advances in Neural Information Processing Systems, 37:56619–56643, 2024. 8, 9

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 11975–11986, 2023. 3

Jiahui Zhang, Yurui Chen, Yueming Xu, Ze Huang, Yanpeng Zhou, Yu-Jie Yuan, Xinyue Cai, Guowei Huang, Xingyue Quan, Hang Xu, and Li Zhang. 4d-vla: Spatiotemporal vision-languageaction pretraining with cross-scene calibration. arXiv preprint arXiv:2506.22242, 2025a. 7, 8

Jianke Zhang, Yanjiang Guo, Xiaoyu Chen, Yen-Jen Wang, Yucheng Hu, Chengming Shi, and Jianyu Chen. Hirt: Enhancing robotic control with hierarchical robot transformers. arXiv preprint arXiv:2410.05273, 2024. 2, 3

Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zeroinit attention. arXiv preprint arXiv:2303.16199, 2023. 5

Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, Fan Lu, He Wang, Zhizheng Zhang, Li Yi, Wenjun Zeng, and Xin Jin. Dreamvla: A vision-language-action model dreamed with comprehensive world knowledge. arXiv preprint arXiv:2507.04447, 2025b. 1

Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Ming-Yu Liu, Donglai Xiang, Gordon Wetzstein, and Tsung-Yi Lin. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1702– 1713, 2025a. 7, 8

Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023. 10

Wei Zhao, Pengxiang Ding, Min Zhang, Zhefei Gong, Shuanghao Bai, Han Zhao, and Donglin Wang. Vlas: Vision-language-action model with speech instructions for customized robot manipulation. arXiv preprint arXiv:2502.13508, 2025b. 8, 9

Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daum´e III, Andrey Kolobov, Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. arXiv preprint arXiv:2412.10345, 2024. 7, 8

Zhide Zhong, Haodong Yan, Junfeng Li, Xiangchen Liu, Xin Gong, Wenxuan Song, Jiayi Chen, and Haoang Li. Flowvla: Thinking in motion with a visual chain of thought. 2025. 1, 7, 8

### Appendix of VLA-Adapter

- A SETUP DETAILS OF LIBERO SIMULATION BENCHMARKS

The LIBERO benchmark (Liu et al., 2023a) comprises four distinct task suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-100. The first three suites each contain 10 tasks, and LIBERO-100 contains 90 short-term tasks (LIBERO-90) and 10 long-horizon tasks (LIBEROLong). The strategy for each task depends solely on the current instructions provided. Each task is repeated multiple times (50 repetitions in this paper) to obtain the average success rate for each subtask. The examples and the instructions in the LIBERO benchmark are shown in Figure A1.

Put both the alphabet soup and the tomato sauce in the basket Put both the cream cheese box and the butter in the basket Turn on the stove and put the moka pot on it Put the black bowl in the bottom drawer of the cabinet and close it Put the white mug on the left plate and put the yellow and white mug on the right plate Pick up the book and place it in the back compartment of the caddy Put the white mug on the plate and put the chocolate pudding to the right of the plate Put both the alphabet soup and the cream cheese box in the basket Put both moka pots on the stove Put the yellow and white mug in the microwave and close it

Long Task instruction

- 1.
- 2.
- 3.
- 4.
- 5.
- 6.
- 7.
- 8.
- 9.
- 10.

LIBERO-Object

LIBERO-Goal

LIBERO-Spatial

LIBERO-Long

|[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]|
|---|

[Figure 170]

[Figure 171]

|[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]<br><br>[Figure 179]<br><br>[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]<br><br>[Figure 189]<br><br>[Figure 190]<br><br>[Figure 191]<br><br>[Figure 192]<br><br>[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]<br><br>[Figure 205]<br><br>[Figure 206]<br><br>[Figure 207]|
|---|

[Figure 208]

[Figure 209]

|[Figure 210]<br><br>[Figure 211]<br><br>[Figure 212]<br><br>[Figure 213]<br><br>[Figure 214]<br><br>[Figure 215]<br><br>[Figure 216]<br><br>[Figure 217]<br><br>[Figure 218]<br><br>[Figure 219]<br><br>[Figure 220]<br><br>[Figure 221]<br><br>[Figure 222]<br><br>[Figure 223]<br><br>[Figure 224]<br><br>[Figure 225]<br><br>[Figure 226]<br><br>[Figure 227]<br><br>[Figure 228]<br><br>[Figure 229]<br><br>[Figure 230]<br><br>[Figure 231]<br><br>[Figure 232]<br><br>[Figure 233]<br><br>[Figure 234]<br><br>[Figure 235]<br><br>[Figure 236]<br><br>[Figure 237]<br><br>[Figure 238]<br><br>[Figure 239]<br><br>[Figure 240]<br><br>[Figure 241]<br><br>[Figure 242]<br><br>[Figure 243]<br><br>[Figure 244]<br><br>[Figure 245]|
|---|

[Figure 246]

[Figure 247]

|[Figure 248]<br><br>[Figure 249]<br><br>[Figure 250]<br><br>[Figure 251]<br><br>[Figure 252]<br><br>[Figure 253]<br><br>[Figure 254]<br><br>[Figure 255]<br><br>[Figure 256]<br><br>[Figure 257]<br><br>[Figure 258]<br><br>[Figure 259]<br><br>[Figure 260]<br><br>[Figure 261]<br><br>[Figure 262]<br><br>[Figure 263]<br><br>[Figure 264]<br><br>[Figure 265]<br><br>[Figure 266]<br><br>[Figure 267]<br><br>[Figure 268]<br><br>[Figure 269]<br><br>[Figure 270]<br><br>[Figure 271]<br><br>[Figure 272]<br><br>[Figure 273]<br><br>[Figure 274]<br><br>[Figure 275]<br><br>[Figure 276]<br><br>[Figure 277]<br><br>[Figure 278]<br><br>[Figure 279]<br><br>[Figure 280]<br><br>[Figure 281]<br><br>[Figure 282]<br><br>[Figure 283]|
|---|

[Figure 284]

[Figure 285]

Pick up the orange juice and place it in the basket Pick up the cream cheese and place it in the basket Pick up the salad dressing and place it in the basket Pick up the bbq sauce and place it in the basket Pick up the ketchup and place it in the basket Pick up the tomato sauce and place it in the basket Pick up the butter and place it in the basket Pick up the milk and place it in the basket Pick up the chocolate pudding and place it in the basket Pick up the orange juice and place it in the basket

Object Task instruction

- 1.
- 2.
- 3.
- 4.
- 5.
- 6.
- 7.
- 8.
- 9.
- 10.

Open the middle drawer of the cabinet Put the bowl on the stove Put the wine bottle on top of the cabinet Open the top drawer and put the bowl inside Put the bowl on top of the cabinet

Goal Task instruction

- 1.
- 2.
- 3.
- 4.
- 5.

Pick up the black bowl between the plate and the ramekin and place it on the Plate Pick up the black bowl next to the ramekin and place it on the plate Pick up the black bowl from table center and place it on the plate Pick up the black bowl on the cookie box and place it on the plate Pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate Pick up the black bowl on the ramekin and place it on the plate Pick up the black bowl next to the cookie box and place it on the plate Pick up the black bowl on the stove and place it on the plate Pick up the black bowl next to the plate and place it on the plate Pick up the black bowl on the wooden cabinet and place it on the plate

Spatial Task instruction

- 1.
- 2.
- 3.
- 4.
- 5.
- 6.
- 7.
- 8.
- 9.
- 10.

Push the plate to the front of the stove Put the cream cheese in the bowl Turn on the stove Put the bowl on the plate Put the wine bottle on the rack

- 6.
- 7.
- 8.
- 9.
- 10.

|[Figure 286]|
|---|

|[Figure 287]|
|---|

|[Figure 288]|
|---|

|[Figure 289]|
|---|

Figure A1: The examples and the task instructions on the LIBERO benchmark.

In the LIBERO benchmark, we use third-person images (resolution 224×224×3, RGB) and wrist images (resolution 224×224×3, RGB) as visual input. The task instruction is first constructed as a prompt in a specific format: “In: What action should the robot take to instruction.lower()?\nOut:”, and then input into the VLM module together with the image information. Its output is a 7-dimensional action vector, which is used to control the 7-DOF Franka Emika Panda simulated robot arm to perform the corresponding action sequence.

- B DIT-BASED POLICY NETWORK

- B.1 OVERALL ARCHITECTURE

This architecture is shown in Figure B1. It consists of τ-DiT blocks, 1 ≤ τ ≤ M. It has the same number of layers as VLM. Each DiT block consists of three components: conditional modulation, conditional attention, and a conditional feedforward network. At timestep t, the input chunk to the first-DiT block, A1t, is a noisy action sequence. Since the input contains random noise, to facilitate the transition from “pure noise → fine-grained prediction”, we adopt the AdaLN-Zero layer (Peebles & Xie, 2023) to modulate the activation amplitude at each layer. The AdaLN-Zero consists of LayerNorm, modulation, and gated residual. Specifically, CtM will be obtained by Pt and CtR, i.e., CtM = σ1′ (CtR) + σ0(Pt). It is used to generate “Scale” and “Shift” vectors, which guide the activation direction of intermediate features and inject automatic modulation amplitude via gated residual control. After modulation, A1t = a1t, a1t+1,..., a1t+H−1 is obtained:

A1t = A1t+ατ⊙ε(γτLN(A1t)+βτ) = A1t+σ2′ (CtM)⊙ε(σ1′ (2)(CtM)LN(A1t)+σ1′ (1)(CtM)), [βτ; γτ] = σ1′ (CtM),

(B-1)

[Figure 290]

| | |
|---|---|
| | |

#### V

[Figure 291]

Figure B1: The DiT-based policy network. M

where, βτ and γτ are scaling factors and offset factors, which are dynamically generated by CtM through a projection new with the SiLU. ατ is the gated residual coefficient, used to adjust the injection amplitude. It is dynamically generated by CtM through σ0 with the SiLU, and has ατ = σ3′ (CtM). ⊙ is element-wise multiplication, and ε(·) is self-attention and projection modules.

2 l

After conditional modulation, A1t is used as the QKV vector, and CtR,CtAQ are used as the KV vectors for Bridge Attention. The details of Bridge Attention are shown in Section 3.3. And then, the attention latent A1t will be obtained. A1t is input into the conditional feedforward network. The first-DiT block output A2t is obtained. After passing through M DiT blocks, we get the AMt , which is passed by a LayerNorm and MLP layer to generate the current action chunk AMt .

- B.2 TRAINING OF DIT-BASED POLICY

This Policy is also trained from scratch. Given a ground truth action trajectory At, a noisy action in DiT-based Policy Aτt = √ατAt + √1 − ατϵ, where, √ατ is the cumulative product of noise coefficients, and has √ατ = Ti=1 αi = Ti=1 (1 − βi), βi is the variances used at each step, and ϵ ∼ N(0,I) is Gaussian noise. We train DiT-based model πθ(·) with the training objectives:

min

θ

J (θ) = EA

t,ϵ∼N(0,I),CtAQ,σ1(Pt),τ πθ √ατAt +√1 − ατ,CtAQ,σ1(Pt),τ −ϵ 22 . (B-2)

- B.3 BRIEF COMPARISON WITH L1-BASED POLICY

As exploring the Policy architecture is not the primary focus of this paper, we briefly compare the performance of the L1-based and DiT-based Policy networks on the LIBERO-Long benchmark.

- Table B1: Comparison of the L1-based and DiT-based Policy networks of VLA-Adapter. Bold represents the best results. For detailed task instructions, please see Figure A1 in Appendix A.

Task instructions 1 2 3 4 5 6 7 8 9 10 Avg. ↑

L1-based 96.0 96.0 100.0 98.0 100.0 100.0 84.0 96.0 84.0 96.0 95.0 DiT-based 96.0 92.0 98.0 96.0 90.0 98.0 82.0 100.0 74.0 90.0 91.6

Results presented in Table B1 indicate that the L1-based Policy achieves superior performance compared to the DiT-based Policy. This phenomenon coincides with the conclusions in OpenVLA-OFT (Kim et al., 2025): Diffusion-type Policy performs better during pre-training, but L1-based Policy outperforms Diffusion-type Policy during fine-tuning because their actions are less redundant. Furthermore, consistent with findings from OpenVLA-OFT, the L1-based Policy achieves higher throughput compared to the Diffusion-type Policy. Therefore, we chose the L1-based Policy.

- C DETAILED COMPARISON RESULTS OF DIFFERENT CODITIONS

In this section, we give the specific performance of the ten subtasks on the LIBERO-Long benchmark in Section 3 to explore different conditions. The specific results are shown in Tables C1 and C2.

- Table C1: The specific performance of different layers of Raw features on 10 subtasks. For detailed task instructions, please see Figure A1 in Appendix A. Bold represents the best performance of the average success rate. Italics* represents the suboptimal performance of the average success rate.

Raw feature

Subtasks

Avg.

1 2 3 4 5 6 7 8 9 10

Single-layer

1 78 96 94 100 96 98 62 90 68 88 87.6 5 82 94 84 98 94 96 68 94 66 90 86.6 9 94 94 84 94 90 98 90 90 74 90 89.8*

13 90 94 86 92 86 100 82 96 84 74 88.4* 17 82 92 92 96 92 90 66 72 62 86 84.4 21 78 94 98 90 68 92 66 94 78 88 83.2 24 84 96 94 94 94 100 64 88 56 88 85.8

All-layer 1–24 92 98 96 100 84 94 76 96 84 86 90.6

Table C2: The specific performance of different layers of ActionQuery features on subtasks. For task instructions, please see Figure A1 in Appendix A. Bold represents the best performance of the average success rate. Italics* represents the suboptimal performance of the average success rate.

ActionQuery feature

Subtasks

Avg.

1 2 3 4 5 6 7 8 9 10

Single-layer

1 28 50 98 96 80 92 76 94 78 90 78.2 13 16 52 98 94 94 100 66 98 62 86 76.6 17 90 94 86 88 82 100 74 100 58 96 86.8 21 88 92 94 98 92 92 70 96 72 94 88.8

- 23 92 98 94 96 96 100 70 98 72 82 89.6*

- 24 92 88 100 98 90 96 74 98 84 82 90.2*

All-layer 1–24 92 94 96 98 100 98 76 98 78 96 92.6

In Table C1, the middle-layer Raw features generally outperform other-layer Raw features. In Table C2, deep-layer ActionQuery features generally perform better than shallow-layer ActionQuery’s. In addition, in Table C1 and Table C2, the performance of all layers is the best. Therefore, in VLAAdapter, we use the Raw feature and ActionQuery feature of all layers as conditions for Policy.

- D PERFORMANCE ON LIBERO SUBTASKS

In this section, we demonstrate the performance of VLA-Adapter on 40 (4×10) subtasks on the LIBERO benchmark (Liu et al., 2023a). The detailed performance is shown in Table D1.

- Table D1: The specific performance of VLA-Adapter on the 40 subtasks of four LIBERO (Liu et al., 2023a) suites. For detailed task instructions, please see Figure A1 in Appendix A.

LIBERO 1 2 3 4 5 6 7 8 9 10 Avg. ↑

Spatial 98.0 100.0 100.0 90.0 96.0 100.0 100.0 100.0 98.0 96.0 97.8 Object 98.0 98.0 100.0 100.0 98.0 100.0 98.0 100.0 100.0 100.0 99.2

Goal 92.0 100.0 98.0 96.0 100.0 98.0 94.0 100.0 98.0 96.0 97.2 Long 96.0 96.0 100.0 98.0 100.0 100.0 84.0 96.0 84.0 96.0 95.0

- E SETUP DETAILS OF CALVIN SIMULATION BENCHMARK

Benchmark. We used the CALVIN ABC→D (Mees et al., 2022) to evaluate the performance on the zero-shot generalization tasks. CALVIN consists of four environments (Env A, B, C, and D). “ABC→D” means it trains on Env A, B, and C and evaluates on Env D. These environments collectively include over two million human demonstration trajectories totaling approximately six hours. The CALVIN benchmark contains 34 different subtasks. By screening the combination of five consecutive subtasks, 1,000 unique instruction chains with rationality and diversity are finally generated. In each instruction chain, the agent needs to complete five subtasks in sequence, and can only proceed to the next subtask after successfully completing the current subtask. The benchmark aims to evaluate generalization capabilities and task execution performance under diverse conditions. Examples of each environment and the task instructions are shown in Figure E1.

Move slider right/left Open/Close drawer Lift {object} table/drawer Lift {object} slider Place in slider/drawer Stack/Unstack block Turn on/off lightbulb/led Push into drawer

[move_door_rel, 'base_slide', ±0.23] [move_door_rel, 'base_drawer', ±0.12] [lift_object, '{object}', 0.05, 'table', 'base_link/drawer_link'] [lift_object, '{object}', 0.03, 'table', 'plank_link'] [stack_objects/unstack_objects] [place_object, 'table', 'plank_link/drawer_link'] [toggle_light, 'lightbulb/led', 0/1, 1/0] [push_object_into, [{object}], 'table', 'base_link', 'table', 'drawer_link']

- 3.
- 4.
- 5.
- 6.
- 7.
- 8.

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

Env A Env B

Env D

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

Env C

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

Training

Evaluation

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

Env A Env B Env C Env D

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

###### Task instruction Condition

[rotate_object, '{object}', ±60] [push_object, '{object}', ±0.1, 0] [move_door_rel, 'base_slide', ±0.23] [move_door_rel, 'base_drawer', ±0.12] [lift_object, '{object}', 0.05, 'table', 'base_link/drawer_link'] [lift_object, '{object}', 0.03, 'table', 'plank_link'] [stack_objects/unstack_objects] [place_object, 'table', 'plank_link/drawer_link'] [toggle_light, 'lightbulb/led', 0/1, 1/0] [push_object_into, [{object}], 'table', 'base_link', 'table', 'drawer_link']

Rotate {object} right/left Push {object} right/left Move slider right/left Open/Close drawer Lift {object} table/drawer Lift {object} slider Place in slider/drawer Stack/Unstack block Turn on/off lightbulb/led Push into drawer

- 1.
- 2.
- 3.
- 4.
- 5.
- 6.
- 7.
- 8.

Figure E1: The example and task completion conditions on the CALVIN ABC→D.

In the CALVIN ABC→D benchmark, we use third-person images (resolution 224×224×3, RGB) and Gripper images (resolution 84×84×3, RGB) as visual input. The task instruction is first constructed as a prompt in a specific format: “In: What action should the robot take

LIBERO-Object

LIBERO-Goal

instruction nOut: then input into the VLM module together with the inform ou 7-d action vector, which is used to control the 7-DOF

###### Task instruction

|[Figure 475]<br><br>}?\ output is a<br><br>simulated robot|
|---|

|[Figure 476]<br><br>”, and 7-dimensional arm to|
|---|

|[Figure 477]<br><br>to {Task image Franka|
|---|

|[Figure 478]<br><br>information. Its Panda|
|---|

- 1.
- 2.
- 3.
- 4.
- 5.
- 6.
- 7.
- 8.
- 9.
- 10.

Put both the alphabet soup and the tomato sauce in the basket Put both the cream cheese box and the butter in the basket Turn on the stove and put the moka pot on it Put the black bowl in the bottom drawer of the cabinet and close it Put the white mug on the left plate and put the yellow and white mug on the right plate Pick up the book and place it in the back compartment of the caddy Put the white mug on the plate and put the chocolate pudding to the right of the plate Put both the alphabet soup and the cream cheese box in the basket Put both moka pots on the stove Put the yellow and white mug in the microwave and close it

Emika sim perform the corresponding action sequence.

LIBERO-Spatial LIBERO-Long

|[Figure 479]<br><br>F S|
|---|

|[Figure 480]|
|---|

|[Figure 481]<br><br>D|
|---|

|[Figure 482]<br><br>T|
|---|

UPPLEMENTARY ETAILS OF RAINING AND HYPERPARAMETERS

- F.1 TRAINING DETAILS

During VLA-Adapter training, we use the AdamW (Loshchilov & Hutter, 2019) optimizer and LoRA scheme (Hu et al., 2022). To ensure the stability of training, the learning rate is set to 1e-4, and the cosine-annealing scheduler with warm-up steps is used. Our batch size is set to 16.

Table F1: The detail settings of Training.

Setting value Batch size 16 Max training step 150,000 Learning rate 1e-4 Warmup step 10%

- F.2 HYPERPARAMETER DETAILS We list the hyperparameters of VLA-Adapter. Their corresponding values are shown in Table F2.

Table F2: Specific hyperparameters of VLA-Adapter and their corresponding values.

Backbone Qwen2.5-0.5B Layer (τ / M) 24

Number of ActionQuery 64

Hidden size 896 Attention head 8

Action chunk (H) 8 Intermediate layers of VLM 1–24

Total trainable parameters of Policy 97.3M Total trainable parameters of VLA-Adapter 197.2M

- G EXECUTION EXAMPLES We provide some execution examples, please see Figure G1 and Figure G2 for details.

- G.1 REAL-WORLD EXAMPLES

These include long-horizon tasks: “Pick up the spoon and place it on the cup, then place the cup on the plate” and short-horizon tasks: “Stack red blocks on top of blue blocks”, “Move the blue block to the right”, and “Pick up the duck and place it on a plate”. The settings of real-world experiments are shown in Section 4.4.

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

Real-World: Pick up the spoon and place it on the cup, then place the cup on the plate

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

Real-World: Stack red blocks on top of blue blocks

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

Real-World: Move the blue block to the right

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

Real-World: Pick up the duck and place it on a plate

Figure G1: Execution example on the real-world tasks.

###### G.2 SIMULATION EXAMPLES

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

LIBERO-Spatial: Pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

LIBERO-Object: Pick up the milk and place it in the basket

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

LIBERO-Goal: Put the wine bottle on top of the cabinet

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

LIBERO-Long: Put both moka pots on the stove

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

CALVIN ABC➝D: Turn off lightbulb➝ Move slider left➝Push blue block left➝ Lift pink block slider ➝ Stack block

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

Figure G2: Execution example on the LIBERO and CALVIN ABC→D tasks.

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

- H EFFECTIVENESS ANALYSIS OF FROZEN BACKBONE

Section 4.1 compares the effectiveness of the frozen backbone. The results show that OpenVLAOFT does not work. Although it also uses learnable tokens, it is implemented (line 620 in 3):

CALVIN ABC➝D: Open drawer ➝Lift pink block table➝ Place in slider ➝ Turn on lightbulb➝ Rotate blue block left

# === Handle Multimodal Forward === elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0]

== pixel_values.shape[0]):

... # Process action embeddings if noisy_actions is not None:

...

else: # Replace the embeddings of the action tokens with zeros # (Later on, the positional embeddings will be added to them) all_actions_mask = all_actions_mask.unsqueeze(-1) input_embeddings = input_embeddings * ˜all_actions_mask

The tokens added in the “else:” (L1 architecture) are input to the VLM in the form of a mask. It is initially all zeros, and when the VLM backbone is frozen, it is not trained. Our ActionQuery is:

3https://github.com/moojink/openvla-oft/blob/main/prismatic/extern/hf/ modeling_prismatic.py

# === Handle Multimodal Forward === elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0]

== pixel_values.shape[0]):

... # Process action embeddings if noisy_actions is not None:

...

else: action_queries = self.action_queries.weight # (1, h) action_queries = action_queries.view(1, action_queries.shape[0],

action_queries.shape[1]).repeat(input_embeddings.shape[0], 1, 1)

all_actions_mask = self._process_action_masks(labels) input_embeddings = self._replace_input_embeddings(input_embeddings,

all_actions_mask, action_queries)

Instead of inputting the VLM in the form of a mask, essentially learnable tokens (or multiple tokens) that is inserted into the specified position in the sequence and participate in attention. Therefore, when the VLM is frozen, the VL information is indeed not trained. Still, ActionQuery is not an original part of the VLM, and its parameters can be learned from scratch, so the VLA-Adapter still works. Below, we give examples of OpenVLA-OFT and VLA-Adapter, as shown in Figure H1.

Instruction: Put both the alphabet soup and the tomato sauce in the basket

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

OpenVLA-OFT (False)

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

Ours (True)

Figure H1: Execution example when the backbone is frozen.

- I DESIGN DETAILS AND PERFORMANCE OF VLA-ADAPTER-PRO

To achieve extreme lightweightness in VLA-Adapter, we shared the projection layer, using the same projection layer for all three attention matrices. This resulted in a low parameter count of 97MB.

In the VLA-Adapter-Pro version, we separated the projection layers for the three attention matrices, allowing different channels to learn differentiated representations. In this case, the parameter count was 207MB. Furthermore, VLA-Adapter-Pro adds Rotary Position Embedding to QK, making the cross-attention more sensitive to position information and more suitable for action generation.

Next, we give the key architecture codes of VLA-Adapter-Pro, as shown below:

class MLPResNetBlock_Pro(nn.Module): """One MLP ResNet block with separate projections for self, adapter, task

+ RoPE, now without FiLM modulation.""" def __init__(self, dim, num_heads=8):

super().__init__() self.dim = dim self.num_heads = num_heads self.head_dim = dim // num_heads

self.ffn = nn.Sequential( nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), )

# Q (from x only) self.q_proj = nn.Linear(dim, dim)

# Self-Attention: K, V self.k_self = nn.Linear(dim, dim) self.v_self = nn.Linear(dim, dim)

# Adapter cross-attention: K, V self.k_adapter = nn.Linear(dim, dim) self.v_adapter = nn.Linear(dim, dim)

# Task cross-attention: K, V self.k_task = nn.Linear(dim, dim) self.v_task = nn.Linear(dim, dim)

self.o_proj = nn.Linear(dim, dim) # gating self.gating_factor = nn.Parameter(torch.zeros(1)) # RoPE self.rope = RotaryPositionEmbedding(self.head_dim) # ---- FiLM ---# # FiLM is useless; to avoid conflict with chkpt, it can be kept as is for now. self.film_gen = nn.Sequential(

nn.Linear(dim, dim * 2), )

def apply_film(self, x, gamma, beta): """FiLM: per-channel modulation""" return gamma.unsqueeze(1) * x + beta.unsqueeze(1)

def forward(self, x, h_a=None, h_t=None, p=None): """ h_a: adapter tokens h_t: task tokens p: possible conditioning vector (for FiLM) """ g = self.gating_factor ratio_g = torch.tanh(g)

# concat h_a and p h_adapter = torch.cat((h_a, p),dim=1)

h_task = h_t B, T, C = x.shape K_a = h_adapter.size(1) if h_a is not None else 0 K_t = h_task.size(1) if h_task is not None else 0

# Q q_1 = self.q_proj(x)

# self tokens k_tokens = self.k_self(x) v_tokens = self.v_self(x)

# adapter tokens k_adapter = self.k_adapter(h_adapter) v_adapter = self.v_adapter(h_adapter)

# task tokens k_task = self.k_task(h_task) v_task = self.v_task(h_task)

# reshape -> multi-head def reshape_heads(t, B, L):

return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

q_1 = reshape_heads(q_1, B, T) k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T) k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a) k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

# RoPE cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype) q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main) cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype) _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a) cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype) _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

# attention scores attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))] attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1))) attn_scores.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g) attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim) attn_weights = torch.softmax(attn_scores, dim=-1)

# combine V v_list = [v_tokens,v_adapter,v_task] v_combined = torch.cat(v_list, dim=2)

output = torch.matmul(attn_weights, v_combined) output = output.transpose(1, 2).contiguous().view(B, T, C) output = self.o_proj(output)

# # ---- FiLM ---# gamma_beta = self.film_gen(p) # [B, 2C] # gamma, beta = gamma_beta.chunk(2, dim=-1) # [B, C], [B, C] # output = self.apply_film(output, gamma, beta)

# residual + FFN x = self.ffn(output + x) return x

We also present the performance comparison between VLA-Adapter and VLA-Adapter-Pro on 40 subtasks on LIBERO, as shown in Table I1.

Table I1: The specific performance of VLA-Adapter and VLA-Adapter-Pro on the 40 subtasks of four LIBERO (Liu et al., 2023a) suites.

Spatial 1 2 3 4 5 6 7 8 9 10 Avg. ↑ VLA-Adapter 98.0 100.0 100.0 90.0 96.0 100.0 100.0 100.0 98.0 96.0 97.8

VLA-Adapter-Pro 100.0 98.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0 98.0 99.6

Object 1 2 3 4 5 6 7 8 9 10 Avg. ↑ VLA-Adapter 98.0 98.0 100.0 100.0 98.0 100.0 98.0 100.0 100.0 100.0 99.2

VLA-Adapter-Pro 100.0 100.0 100.0 100.0 98.0 100.0 98.0 100.0 100.0 100.0 99.6

Goal 1 2 3 4 5 6 7 8 9 10 Avg. ↑ VLA-Adapter 92.0 100.0 98.0 96.0 100.0 98.0 94.0 100.0 98.0 96.0 97.2

VLA-Adapter-Pro 98.0 100.0 94.0 96.0 100.0 98.0 96.0 100.0 100.0 100.0 98.2

Long 1 2 3 4 5 6 7 8 9 10 Avg. ↑ VLA-Adapter 96.0 96.0 100.0 98.0 100.0 100.0 84.0 96.0 84.0 96.0 95.0

VLA-Adapter-Pro 92.0 100.0 98.0 96.0 94.0 100.0 94.0 100.0 90.0 100.0 96.4

