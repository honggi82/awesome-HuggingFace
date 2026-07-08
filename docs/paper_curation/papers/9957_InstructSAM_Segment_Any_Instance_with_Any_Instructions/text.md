# arXiv:2605.26102v2[cs.CV]31May2026

## InstructSAM: Segment Any Instance with Any Instructions

Yuqian Yuan1,∗, Wentong Li2,∗,†, Zhaocheng Li1,∗, Yutong Lin1,∗, Juncheng Li1, Siliang Tang1, Jun Xiao1, Yueting Zhuang1, Wenqiao Zhang1,‡

1Zhejiang University, 2Nanjing University of Aeronautics and Astronautics ∗Equal contribution, †Project lead, ‡Corresponding author

In this paper, we introduce InstructSAM, a unified and streamlined framework designed for robust multi-instance segmentation under arbitrary instructions. We formulate instruction-driven instance segmentation as a set-structured query prediction problem and propose an explicit reasoning-to-instance query interface that elegantly bridges a vision-language model (VLM) and SAM3. Specifically, a bank of learnable instance queries is injected into the VLM and contextualized with instruction and visual information, enabling each query to serve as an instance-aware slot. A hybrid-attention mechanism further promotes interaction among these queries, visual tokens, and instruction tokens, improving instance enumeration and reducing duplicate predictions. The resulting LLM-conditioned queries are projected into SAM3’s detector query space to drive accurate multi-instance segmentation in a single forward pass. This design equips SAM3 with high-level instruction understanding, compositional reasoning, and instance-level set prediction without modifying its core architecture. To support training and evaluation, we further construct Inst2Seg, a high-quality and large-scale instruction-based instance segmentation dataset and benchmark that couples free-form instructions with instance-level masks. Extensive experiments show that only 2B-scale InstructSAM achieves strong results across complex instruction-driven and phrase-level referring segmentation benchmarks, outperforming prior end-to-end methods and SAM3’s agentic pipeline while enabling efficient single-pass multi-instance prediction.

GitHub: https://github.com/DCDmllm/InstructSAM

Dataset: https://huggingface.co/datasets/CircleRadon/Inst2Seg

Model: https://huggingface.co/CircleRadon/InstructSAM-2B/

1 Introduction

Segmenting objects in images and videos is a foundational capability for embodied agents [1–3], autonomous perception [4], healthcare [5–7] and visual edition [8, 9]. The line work of Segment Anything Model (SAM) has significantly advanced this direction by enabling promptable segmentation with strong generalization [10–12]. In particular, the recent SAM3 [12] extends promptable segmentation to open-world concept-level, multiinstance settings, where a short noun phrase (e.g., “traffic cone”) can retrieve and segment multiple instances in a scene, as shown in Fig. 1(a). Despite this promising progress, a critical gap remains between concept-level prompting and real-world user intent. In practice, users rarely communicate their targets as isolated noun phrases; instead, they often issue complex, compositional instructions involving attributes (“the small mugs”), spatial constraints (“on the left”), relations (“next to the laptop”), exclusion (“except the one in front”), or counting (“the two largest”). Such instructions require nontrivial semantic parsing, visual reasoning, and instance-level grounding, as the target object set is often implicitly defined instead of explicitly specified by a single concept label.

Existing attempts to handle complex instructions mainly follow two paradigms. One common solution is an agentic decomposition-and-filtering pipeline, where a large vision-language model (VLM), such as QwenVL [13] or Gemini [14], rewrites the complex instruction into one or more concept-level prompts, repeatedly invokes SAM3 to generate candidate masks, and then post-filters the results with heuristics or verification prompts. However, this indirect process is slow, brittle, and prone to semantic loss, as rewriting may discard

(a) SAM3 Paradigm (b) Common MLLM-based

###### (c) Our Model

Segmentation Paradigm

Who are most likely to get the ball?

Person in grey

Who are most likely to get the ball?

Parallel Instance Queries

Mask queries

[Figure 1]

[Figure 2]

###### Text queries

[Figure 3]

###### Text queries

Text Encoder

###### LLM

###### LLM

Decoder

It is [SEG].

It is [SEG][SEG].

<ref_s> Two players wearing gray jersey facing the ball <ref_e> <mask_s> Decoder

###### Decoder

Decoder

mask1 mask2 mask3 mask4

0.92 0.77 0.83 0.01

mask1 mask2 mask3 mask4

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

mask

mask1 mask2

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

0.92 0.87 0.01 0.07

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Support simple phrase only Semantic level Repeated prediction

Complex reasoning Multi instance prediction

- Figure 1 Comparison of instruction-driven segmentation paradigms: (a) SAM3 handles concept-level prompts but struggles with complex instructions; (b) MLLM-based mask-token generation yields semantic masks or inconsistent multi-instance results; (c) our method uses LLM reasoning to condition an explicit set of instance-aware object queries that guide SAM3 for coherent multi-instance segmentation.

fine-grained constraints and iterative filtering can accumulate errors. Another line of work equips LLMs with a special segmentation token, i.e. [SEG], whose hidden state is decoded into a mask, as in LISA [15] and Sa2VA [16], as illustrated in Fig. 1(b). While effective for reasoning-driven semantic segmentation, this token-as-mask interface is not inherently instance-discriminative. LISA++ [17] extends this paradigm by emitting multiple [SEG] tokens for instance prediction. However, because [SEG] is a shared symbol without an explicit instance-binding mechanism, the resulting masks often collapse to duplicates or become unstable, producing repeated or inconsistent outputs. Moreover, autoregressive generation of multiple [SEG] tokens increases inference latency as the number of target instances grows.

In this paper, we propose InstructSAM, a unified framework for segmenting arbitrary instances under arbitrary instructions via an explicit reasoning-to-instance interface. Rather than forcing the LLM to directly “speak masks” token by token, InstructSAM leverages its general-purpose reasoning capability to interpret complex instructions and translate them into a set-structured, instance-aware query representations. These representations serve as an explicit interface to SAM3, enabling coherent and efficient multi-instance segmentation. Concretely, we introduce a bank of learnable queries into the LLM as parallel instance slots. Through bidirectional interactions among the queries, together with instruction and visual context, these slots are contextualized into instance-specific embeddings that capture potential target instances implied by the instruction. The resulting LLM-conditioned queries are then projected into SAM3’s detector query space, where they directly drive the detector and mask decoder to localize and segment multiple instances in a single forward pass. This design bridges instruction reasoning and mask prediction, enabling compositional understanding and coherent instance enumeration, as shown in Fig. 1(c).

To further advance instruction-based instance segmentation, we introduce Inst2Seg, a large-scale dataset and benchmark that couples free-form instructions with instance-level masks. Built through a carefully designed annotation pipeline, Inst2Seg contains 500K QA pairs for training and a dedicated benchmark with 3,328 manually verified instructions. The benchmark spans diverse real-world scenarios and instruction types, covering single-target, multi-target, and no-target cases to enables systematic evaluation of coherent instance-level mask prediction under complex instructions. Extensive experiments demonstrate that the 2B-scale InstructSAM achieves accurate instance-level segmentation under both complex instructions and referring phrases. It significantly outperforms prior state-of-the-art end-to-end approaches and SAM3’s agentic pipeline at the same model scale, while delivering robust performance across scenes with varying object densities and levels of semantic ambiguity.

We summarize our contributions as follows:

- • We present InstructSAM, a unified end-to-end framework for instruction-conditioned multi-instance

- segmentation via an explicit reasoning-to-instance query interface.
- • We introduce a bank of learnable queries within the LLM as parallel instance slots, coupled with a hybrid-attention mechanism for coherent, instruction-conditioned set prediction.
- • We construct Inst2Seg, a large-scale instruction-based instance segmentation dataset and benchmark covering single-target, multi-target, and no-target scenarios.
- • Extensive experiments demonstrate that 2B-scale InstructSAM substantially outperforms prior end-toend methods and SAM3’s agentic pipeline across established and newly introduced benchmarks.

### 2 Related Work

- 2.1 Segment Anything Models

The “Segment Anything” line of work has fundamentally reshaped generic visual segmentation by introducing promptable models that generalize across categories and domains. SAM [10] formulates segmentation as a prompt-to-mask task, where points, boxes, or coarse masks guide a mask decoder conditioned on image embeddings. Follow-up works [18–20] extend SAM along several practical axes, including efficiency and robustness. SAM2 [11] advances the paradigm to videos by introducing memory-based temporal propagation and interactive refinement. More recently, SAM3 [12] broadens promptable segmentation to open-world, concept-level multi-instance settings, enabling a short noun phrase to retrieve and segment multiple object instances. This capability significantly improves usability in multi-object scenes, yet it still primarily relies on concise concept prompts and is not designed to directly handle complex compositional instructions that require reasoning, exclusion, or counting. To address this issue, SAM3-I [21] equips SAM3 with instructionaware adapter and trains it to map natural-language instructions to masks. While promising, this direction typically requires modifying and retraining the segmentation model to internalize instruction understanding. In constrast, our goal is to preserve SAM3 as a strong open-world segmenter and interfacing it with a reasoning-capable VLM through an explicit query-based mechanism.

- 2.2 Multi-modal Grounded Segmentation

A growing body of work studies how to endow multi-modal large language models (MLLMs) [22–32] with pixel-level grounding, enabling them to respond to free-form instructions with segmentation masks. A dominant design paradigm is the embedding-as-mask interface: the MLLM is augmented with a special segmentation token (e.g., <SEG>), whose embedding is projected into the prompt space of a mask decoder (often SAM-style) and decoded into a mask in an end-to-end fashion [15, 16, 33–36]. These methods align phrase-level semantics with pixel outputs, but most still rely on emitting a segmentation token per semantic grounded region. To move from single-region semantic grounding to multi-instance prediction, LISA++ [17] yields multiple <SEG> tokens for instance segmentation and employs bipartite matching to assign each predicted mask to a ground-truth instance during training. In parallel, X-SAM [37] targets a broader “any segmentation” formulation by standardizing textual prompts with phrase delimiters. In contrast to directly generating mask tokens in an auto-regressive manner, our InstructSAM leverages the MLLM primarily for instruction-level reasoning and instance enumeration, and interfaces it with SAM3 through an explicit set of instance-aware object queries, enabling coherent and efficient multi-instance segmentation under complex instructions.

- 3 Method

In this section, we first formulate the task of instruction-driven instance segmentation task. We then present InstructSAM, an instance-aware segmentation framework that follows open-form instructions and predict a set of instance masks. Finally, we detail the training objectives used to optimize the proposed framework.

[Figure 17]

[Figure 18]

masks scores

Vision & text mask text

masks scores

[Figure 19]

[Figure 20]

[Figure 21]

Mask Decoder (SAM3)

masktextVision&text

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Seg Head Score Head

Image Encoder

[Figure 28]

[Figure 29]

Phrase Encoder MLP

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Pixel Decoder

Detector Decoder

[Figure 36]

SP EP SM

[Figure 37]

[Figure 38]

Fusion Decoder

[Figure 39]

[Figure 40]

[Figure 41]

Large Language Model

Phrase features Mask queries

Visual features

SM EM

Vision token Instance Query Bank (a) (b)

- Figure 2 Overview of the InstructSAM framework. (a) InstructSAM integrates a multimodal LLM, a set of parallel learnable mask queries, and a mask decoder to generate segmentation masks. (b) Illustration of the hybrid-attention design within the multimodal LLM.

- 3.1 Problem Formulation

Instruction-driven instance segmentation aims to predict instance-level masks from an input image and a free-form natural-language instruction. Formally, given an image ximg and an instruction text xtxt, the model outputs a variable-size set of instance masks M = {Mi}Ni=1, where each Mi ∈ {0,1}H×W denotes the binary mask of the i-th instance satisfying the instruction, and N is the number of selected instances, which can be zero. The instruction xtxt is open-form, ranging from a category name (e.g., “chair”) or a referring phrase (e.g., “the leftmost chair”) to a complex instruction involving attributes, relations, counting, exclusion, or implicit intent (e.g., “the objects on the table that should be thrown away”). Therefore, a model must jointly perform language understanding, visual grounding, and instance separation under open-vocabulary settings. We formulate this task as a set prediction problem:

Y = fθ(ximg,xtxt) = (Mi,si) Ni=1, s.t. Mi ∈ {0,1}H×W, si ∈ [0,1], N ≥ 0.

(1)

Here, si is the confidence score of the i-th predicted mask, N denotes the number of instances.

Compared with conventional semantic-level reasoning segmentation [15], this task is more challenging because it requires not only locating the relevant semantic regions but also separating and enumerating distinct object instances. It also differs from typical referring segmentation [21], where the query is usually a concise noun phrase that explicitly specifies the target. In contrast, instruction-driven instance segmentation must handle open-form and often implicit instructions. This capability is essential for embodied perception and robot manipulation, where agents must identify which specific instance to interact with (e.g., “pick up the mug closest to the sink”), enabling reliable grasp planning, collision avoidance, and sequential decision making.

- 3.2 Overview of InstructSAM

As illustrated in Fig. 2(a), InstructSAM consists of three components: (i) a multimodal LLM F for instruction understanding, multimodal fusion and instance-slot contextualization; (ii) a bank of parallel learnable mask queries Q that explicitly parameterize instance slots as the interface between instruction reasoning and mask prediction; and (iii) a set-prediction mask decoder D, instantiated with SAM3, for multi-instance localization and mask decoding.

Crucially, we inject a bank of learnable mask queries into the LLM as parallel instance slots, as shown in Fig. 2(a). These queries define an explicit slot space where different slots can specialize to different target instances within the same image. Given the instruction, visual features, and textual context produced by the LLM, each learnable query is contextualized into a semantically grounded instance embedding for downstream mask prediction. To encourage set-level coherence and suppress duplicate predictions, we further design a hybrid-attention pattern that allows each instance slot to globally integrate visual evidence, instruction cues,

#### and information from other slots, as illustrated in Fig. 2(b). The resulting LLM-conditioned query embeddings are then projected into SAM3’s detector query space and consumed by its detector and mask decoder to produce multiple instance masks in a single forward pass. This architecture enables InstructSAM to combine the reasoning capability of MLLMs with the strong open-world multi-instance segmentation ability.

- 3.2.1 Parallel Instance Query Bank

Given an image ximg and an instruction xtxt, the image encoder produces visual tokens V = Encimg(ximg) ∈ RL

v×d, while the instruction is tokenized into text embeddings T = Emb(xtxt) ∈ RL

t×d. We introduce a

learnable query bank Q = {qk}Kk=1 as parallel instance slots, where qk ∈ Rd and K controls the maximum number of instances that can be predicted in a single forward pass.

A key design of InstructSAM is to replace conventional autoregressively generated segmentation tokens with parallel learnable queries. Specifically, when the model encounters the trigger token <mask_start>, we insert the learnable query bank Q into the multimodal sequence and process it with the LLM in a single forward pass:

X = [V;T;Tphrase;<mask_start>;q1;...;qK], (2)

where Tphrase denotes a short target phrase (e.g., a concise referring description or resolved target phrase) generated by the LLM to provide auxiliary conditioning for segmentation. This phrase serves as a compact, grounded summary of the open-form instruction, which helps stabilize the interface with the mask decoder and reduce ambiguity, especially when the instruction involves implicit intent or multi-step reasoning.

The LLM then produces contextualized hidden states H = F(X) ∈ RL×d, from which we extract the query-specific embeddings:

zk = H[qk] ∈ Rd, k = 1,...,K. (3)

Each zk can be viewed as a grounded instance hypothesis: it integrates instruction semantics, global visual context, and query-level interactions, and is expected to encode both what to segment, namely the semantic intent, and where to segment, namely the localization cues. In this way, the query bank provides an explicit set-structured interface between instruction reasoning and downstream instance-level mask prediction.

- 3.2.2 Hybrid-Attention Design

To reconcile language generation with instance-level set prediction, we present a hybrid-attention pattern, as illustrated in Fig. 2(b). The key idea is to treat textual tokens and mask queries differently according to their roles, and instance queries should not be generated independently or sequentially. Text tokens follow the standard causal attention used for autoregressive language modeling, while mask queries are allowed to attend bidirectionally to other mask queries. This design preserves the language modeling ability of the LLM, while enabling instance slots to communicate with each other to capture the target set structure and suppress duplicate predictions.

Formally, let A ∈ {0,1}L×L be the attention mask. For text positions i ∈ Itext, we enforce causal attention by setting Aij = 1 only if j ≤ i. For query positions i ∈ Iquery, we allow full-context attention by setting Aij = 1 for all j ∈ (Ivision ∪ Itext ∪ Iquery). In this way, each query obtains a global view of the image, the instruction, and the other instance slots, enabling more stable and instance-discriminative mask prediction.

- 3.2.3 From Query to Mask

To realize the reasoning-to-instance interface, we translate LLM-conditioned instance queries into detectorcompatible prompts that directly control SAM3’s mask decoding process. Specially, a lightweight MLP projects each query embedding zk into the embedding space expected by the mask decoder D, yielding grounded mask-query embeddings {˜zk}Kk=1. In parallel, another MLP maps the phrase features to the required dimensionality, producing tp as auxiliary textual conditioning. Given the projected features, the fusion encoder conditions visual embeddings by cross-attending to the phrase tokens, producing instruction-aware image features. A subsequent detector then allows each mask query to cross-attend to these conditioned image features, refining instance-specific representations. Finally, a score head predicts the validity of each query, and a segmentation head generates its corresponding binary mask.

##### Table 1 Comparison of Inst2Seg benchmark with existing referring image segmentation benchmarks. Our Inst2Seg provides instruction-based instance-level evaluation covering single-target, multi-target, no-target, and reasoning scenarios. ST, MT, NT, and Reas. denote single-target, multi-targets, no-target, and reasoning, respectively.

Inst. Level

Prompt Type

Reas.

Benchmarks ST MT NT

Metric

RefCOCO ✓ Phrase IoU RefCOCO+ ✓ Phrase IoU RefCOCOg ✓ Phrase IoU gRefCOCO ✓ ✓ ✓ ✓ Phrase IoU GSEval ✓ ✓ Phrase IoU

ReasonSeg ✓ ✓ Instruction IoU Inst2Seg ✓ ✓ ✓ ✓ ✓ Instruction AP+IoU

- 3.3 Training Objectives

We train InstructSAM end-to-end with a multi-task objective that jointly optimizes: (i) a masked autoregressive loss Ltext, (ii) an instance segmentation loss Lseg, and (iii) a query-level presence loss Lpres. The overall loss is

L = λtextLtext + λsegLseg + λpresenceLpresence. (4)

Masked Auto-regressive Loss. Let y = (y1,...,yN) denote the target text sequence produced by the MLLM, and let x denote the multimodal conditioning context, including instruction tokens and image tokens. We optimize the standard auto-regressive negative log-likelihood, while masking out special segmentation-related tokens (e.g., instance query tokens q and <mask_end>), so that they do not contribute to the language modeling objective. Specifically, we introduce a binary mask mi ∈ {0,1} indicating whether the i-th token is supervised by the text loss (mi = 0 for masked tokens). The masked auto-regressive loss is:

Ltext = −

1

N i=1 mi

N

i=1

mi log pθ(yi | y<i,x). (5)

Segmentation Loss. Following DETR-style set prediction [38, 39], we perform bipartite matching to compute an optimal one-to-one assignment between predicted instance slots and ground-truth instances. For matched slots, we supervise the predicted masks with a weighted combination of per-pixel binary cross-entropy and Dice loss:

Lseg = λbceLbce + λdiceLdice, (6) where Lbce is computed over pixels and Ldice encourages overlap-aware mask quality.

Presence Loss. To identify which query slots correspond to valid target instances, we supervise each slot with a binary presence label. Specifically, after bipartite matching, we set tk = 1 if slot k is matched to a ground-truth instance and tk = 0 otherwise. We then apply a binary cross-entropy loss to the per-slot presence logits sˆk:

Lpresence =

1 K

K

k=1

BCE(ˆsk,tk). (7)

- 4 Inst2Seg Dataset

In this section, we present Inst2Seg, a large-scale instruction-based instance segmentation dataset and benchmark designed to couple free-form instructions with instance-level masks. It is designed to support fine-grained instruction reasoning and precise mask annotation for complex instruction-driven segmentation.

Training Data. We collect training images from two sources: (i) conventional exo-centric images sampled from SA-1B [10], COCO2017 [40], and (ii) ego-centric images curated from Ego4D [41], EPIC-KITCHENS [42],

Which part of the tool is sharpened for cutting?

Find any officers in this group who are not wearing helmets.

Which building has the sign indicating it sells extra-large sizes?

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

N/A

###### Multi Targets Single Target No Target

- Figure 3 Examples of the proposed Inst2Seg benchmark, covering diverse instruction types that require instance-level reasoning and segmentation.

and HD-EPIC [43]. For ego-centric subset, we crop clips with substantial scene variation and discard blurry or low-quality frames. Our annotation pipeline consists of four stages. (1) QA generation using Gemini 3 Flash [23] to produce localization-oriented referring questions with hard negatives and concise noun-phrase answers, along with an explicit ground field encoding counting/quantifiers for multi-instance targets; (2) object consolidation & box generation, where questions referring to the same target are merged into a shared object_id and Gemini predicts normalized 2D boxes; (3) mask annotation by prompting SAM2 [11] with the boxes to obtain pixel-accurate instance masks per object_id; and (4) filtering to remove low-quality or inconsistent samples. In total, we curate 100K images with 500K QA pairs.

Benchmark. The Inst2Seg benchmark comprises 986 images and 3,328 unique instructions. Compared with existing referring image segmentation benchmarks [44–47] (Table 1), Inst2Seg provides a more challenging evaluation setting by covering single-target, multi-target, and no-target cases under instruction prompts, spanning both object-level and part-level granularity. All benchmark instructions and masks are manually verified to ensure high quality. Representative examples are shown in Fig. 3.

Metrics. We adopt mean Average Precision (mAP) as the primary metric for evaluating instance-level predictions. To provide a more fine-grained analysis, we further stratify the results by the number of targets, including single-target, multi-target, and no-target cases. Since many prior methods do not explicitly distinguish individual instances, we additionally report generalized IoU (gIoU) as a complementary semanticlevel metric.

### 5 Experiments

- 5.1 Implementation Details

Our framework is built on the Qwen3-VL [22] backbone, with the mask decoder initialized from SAM 3 [21]. We set the number of learnable queries K = 10 by default. For parameter-efficient fine-tuning, we apply LoRA to the LLM with a rank of 256. Training is conducted in two stages: the first stage aligns the learnable query space with SAM3 for referring segmentation, while the second stage injects reasoning-oriented instruction knowledge. Training is conducted in two stages with different data compositions. The first stage focuses on alignment for referring segmentation, while the second stage aims to inject reasoning-oriented instruction knowledge. Detailed training data for each stage are provided in Appendix A. Each stage is trained for one epoch. For optimization, we set λtext = λmask = λscore = 1.0. In the segmentation loss, the weights for BCE and Dice losses are set to λbce = 2.0 and λdice = 0.5, respectively.

##### Table 2 Results on reasoning-based instruction segmentation benchmarks, including our proposed Inst2Seg (instance-level) and ReasonSeg [15] (semantic-level). Grey entries indicate that models are finetuned on the corresponding training data.

Inst2Seg ReasonSeg Overall Single-Target Multi-Target No-Target val test (all) test (short) test (long)

Method

mAP gIoU mAP gIoU mAP gIoU gIoU gIoU cIoU gIoU cIoU gIoU cIoU gIoU cIoU Multi-round Agent Pipeline

SAM3-AgentQwen2.5-VL-3B [12] 23.2 48.7 33.9 47.6 18.8 33.1 86.2 50.3 34.1 49.9 46.4 44.8 36.2 50.0 47.5 SAM3-AgentQwen2.5-VL-7B [12] 35.7 63.2 47.8 65.1 30.4 47.9 90.1 62.2 49.1 63.0 53.5 59.4 43.5 64.1 56.2 SAM3-AgentQwen3-VL-2B [12] 29.7 58.8 43.2 62.1 24.4 43.2 81.3 58.0 44.9 56.5 43.6 54.1 36.1 57.3 45.9

End-to-end Model

LISA-7B [15] 1.9 28.6 8.2 37.7 0.2 27.7 0.2 52.9 54.0 47.3 48.4 40.6 40.6 49.4 51.0 LISA++-7B [17] 2.2 25.8 7.7 34.1 0.5 24.9 0.2 64.2 68.1 57.0 59.5 49.6 51.1 59.3 61.7 PixelLM-7B [34] 4.6 27.2 13.9 36.7 0.8 22.9 5.1 44.7 37.4 44.0 39.7 38.4 36.7 45.8 40.6 SA2VA-4B [16] 8.2 52.6 30.1 59.2 1.1 40.2 56.9 59.2 60.0 56.9 55.8 53.1 53.0 58.2 56.4 SA2VA-8B [16] 9.4 53.9 35.5 65.2 1.4 46.3 32.4 65.2 61.1 59.5 57.8 55.1 49.8 60.9 60.0 X-SAM-3.8B [37] 11.0 36.6 33.4 51.9 2.0 30.0 0.0 56.6 32.9 57.8 41.0 47.7 48.1 56.0 40.8 InstructSAM-2B 31.5 60.4 52.6 66.8 22.2 44.0 74.3 62.5 65.0 61.1 61.0 56.0 51.1 62.7 63.3

- 5.2 Complex Reasoning Segmentation

Instance-level Inst2Seg. Table 2 reports results on the Inst2Seg benchmark across three subsets: single-target, multi-target, and no-target. For multi-round agentic pipeline, we evaluate SAM3-Agent [12] with Qwen2.5VL-3B and Qwen2.5-VL-7B. For end-to-end models, we compare with semantic-level methods, including LISA [15], SA2VA [16], and X-SAM [37], as well as instance-level methods LISA++ [17] and PixelLM [34].

Among end-to-end methods, InstructSAM-2B achieves the best mAP by a large margin, highlighting the superiority of instance-level instruction following. To complement mAP, we report gIoU as a semantic metric by taking the union of predicted masks per sample; the gap between gIoU and mAP indicates that instance discrimination is substantially harder than coarse semantic region localization, yet our InstructSAM remains strong on most subsets. Notably, for the no-target subset, we evaluate in a zero-shot setting without using any no-target instructions from the Inst2Seg training set. InstructSAM still achieves robust performance, demonstrating its generalization to invalid-target cases. Compared with multi-round agentic pipelines that require multiple interaction rounds and longer contexts, InstructSAM delivers leading performance at comparable model scale. For example, compared with SAM3-Agent based on Qwen2.5-VL-3B, InstructSAM improves mAP by +8.3 and gIoU by +11.7. These results indicate the effectiveness of InstructSAM for instruction-driven segmentation.

Semantic-level ReasonSeg. Table 2 also reports results on ReasonSeg benchmark for the reasoning semantic segmentation. Following the official protocol of [15], we report gIoU and cIoU on both validation and test splits. Compared to similarly sized models such as X-SAM [37] and SA2VA-4B [16], InstructSAM achieves substantially better performance, improving cIoU by +5.0 on the validation split and +5.2 on the overall test set. The improvement is especially notable on long instructions, where InstructSAM gains +6.9 cIoU on test (long), demonstrating stronger robustness to complex and lengthy descriptions.

- 5.3 Phrase-level Referring Expression Segmentation

gRefCOCO. As shown in Table 3, InstructSAM achieves strong performance on the gRefCOCO [46] benchmark. Since gRefCOCO provides instance-level annotations and contains multiple target instances, we report instance-level mAP in addition to the conventional IoU-based metric. InstructSAM-2B surpasses the strongest prior method, GSVA-7B [48], in semantic-level cIoU with gains of +6.6 on val, +3.1 on testA, and +4.9 on testB. It also achieves substantially higher mAP than multi-instance-capable methods such as PixelLM-7B [34] and X-SAM [37], demonstrating robust instance discrimination across diverse referring expressions.

##### Table 3 Results on phrase-level Referring Expression Segmentation benchmarks, including multi-object gRefCOCO and zero-shot GSEval. We report instance-level mAP and semantic-level cIoU/gIoU where applicable.

gRefCOCO GSEval val testA testB Stuff Part Multi Single All

Methods

mAP cIoU mAP cIoU mAP cIoU gIoU gIoU gIoU gIoU gIoU

LISA-7B [15] 28.1 53.9 41.5 63.6 31.9 55.6 85.2 21.2 71.5 42.8 57.6 GLAMM [33] — — — — — — 86.9 16.5 70.4 42.1 57.2 GSVA-7B [48] — 61.7 — 69.2 — 60.3 76.0 20.0 57.8 34.2 48.6 PSALM [49] — 42.0 — 52.4 — 50.6 39.0 10.0 53.7 36.9 37.7 EVF-SAM [50] — — — — — — 85.1 23.1 72.1 54.5 62.6 InstructSeg [51] — — — — — — 56.2 24.2 66.8 51.3 52.5 PixelLM-7B [34] 33.9 51.2 30.7 62.6 23.9 54.6 — — — — SA2VA-4B [16] 26.1 42.3 36.6 53.4 32.6 47.9 88.5 17.4 68.4 45.5 58.5 SA2VA-8B [16] 25.2 42.9 37.7 55.4 32.1 48.9 77.3 18.0 72.0 45.5 56.3 X-SAM-3.8B [37] 22.6 37.0 31.7 46.6 30.4 42.9 — — — — —

InstructSAM-2B 57.3 68.3 51.9 72.3 43.5 65.2 89.4 22.4 73.6 54.8 64.1

GSEval. Table 3 reports zero-shot results on GSEval [47], a comprehensive referring expression segmentation benchmark covering four challenging subsets: stuff, part, multi-object, and single-object. Since GSEval provides only semantic-level annotations, we follow the official protocol and report gIoU. InstructSAM achieves the best performance, outperforming the previous state-of-the-art method, EVF-SAM [50], by +1.5 gIoU.

RoboRefIt. We further evaluate InstructSAM on RoboRefIt [52], a challenging visual grounding benchmark designed for robotic perception and reasoning in indoor environments. RoboRefIt requires a robot to localize the target object specified by language commands. As shown in Table 4, InstructSAM-2B substantially outperforms other MLLM-based segmentation methods. Notably, it approaches the task-specific RefTR-r50 [52] on the in-distribution testA split and exceeds it by a large margin on the distribution-shifted testB split (+12.9), highlighting the strong generalization ability of our approach.

Table 4 Results on RoboRefIt benchmark. InstructSAM2B outperforms previous segmentation methods and shows strong generalization on the distribution-shifted testB split.

Method Task RoboRefIt Specific testA testB

RefTR-r50 [52] ✓ 85.5 61.5 RefTR-r101 [52] ✓ 83.9 60.7

LISA-7B [15] ✗ 36.1 28.7 SA2VA-4B [16] ✗ 56.8 34.0 InstructSAM-2B ✗ 82.5 74.4

- 5.4 Ablation Studies Key Designs in InstructSAM. We ablate two core components of InstructSAM in Table 5.

Removing the learnable queries forces the model to rely on autoregressively generated mask tokens without explicit query conditioning, resulting in consistent performance drops across all benchmarks. The degradation is most pronounced on the instance-level Inst2Seg benchmark, where mAP decreases from 31.5 to 20.1, demonstrating the importance of query bank for instance grounding. Replacing hybrid attention with plain causal attention also degrades performance, especially on ReasonSeg, where the score drops from 65.0 to 52.4 Replacing hybrid-attention with plain causal attention (w/o hybrid-attention) also degrades performance, especially on ReasonSeg, where the score drops from 65.0 to 52.4. This highlights the role of bidirectional query interaction and cross-modal fusion in enabling robust visual-text grounding and mask prediction.

Table 5 Ablation study of key design choices in InstructSAM. We report the average validation score on gRefCOCO, mAP on Inst2Seg, and cIoU on ReasonSeg-val.

gRefCOCO Inst2Seg ReasonSeg

#### w/o Query Bank 58.3 20.1 56.0 w/o Hybrid-Attention 61.2 29.5 52.4

InstructSAM 62.8 31.5 65.0

Scaling with Query Number. We analyze the sensitivity of InstructSAM to the query number K and scalability

##### Table6 Effect of query number on inference efficiency and segmentation performance. Larger query banks bring marginal performance changes but higher latency; “s” and “m” denote single-target and multi-target settings, respectively.

2Seg-s (mAP)

Inst2Seg-m (mAP)

Inst2Seg (mAP)

Query Num Infer Time (s) ↓ Inst

ReasonSeg (gIoU)

10 1.1 52.6 22.2 31.5 62.5 50 1.4 52.1 22.3 31.3 62.9 200 2.1 52.0 22.0 31.1 62.3

##### Table 7 Ablation study on phrase conditioning and LLM-conditioned queries. The results show that LLM-conditioned queries are the primary carrier of instruction semantics, while the generated phrase mainly serves as auxiliary conditioning.

Inst2Seg mAP

Inst2Seg gIoU

ReasonSeg test gIoU Ours 31.5 60.4 62.5 61.1

ReasonSeg val gIoU

Method

w/o phrase (dummy token) 29.1 (↓2.4) 58.5 (↓1.9) 60.7 (↓1.8) 58.9 (↓2.2) w/o LLM-conditioned query 16.7 (↓14.8) 42.1 (↓18.3) 45.0 (↓17.5) 42.9 (↓18.2)

to dense scenes. As shown in Table 6, increasing K from 10 to 200 leads to minimal performance change while inference time steadily increases. This suggests that a small number of well-conditioned queries is sufficient for most samples, consistent with the data distribution where the majority of instructions involve fewer than 10 target instances. Larger query banks may introduce redundant slots without clear performance gains.

Significance of LLM-conditioned query. To verify that the primary semantic signal comes from the LLMconditioned query representations, we conduct two inference ablations using the same trained checkpoint. First, we replace the generated noun phrase with a dummy token (token id 0) while keeping the rest of the architecture unchanged. This causes only a modest performance drop, suggesting that the noun phrase is not the main carrier of instruction semantics. Second, we remove the LLM-conditioned query representations. As shown in Table 7, this leads to much larger degradation across all metrics, demonstrating that the dominant semantic information is encoded in the instruction-conditioned queries after decoder fusion. These results indicate that the noun phrase mainly provides auxiliary textual conditioning for stability and compatibility with the SAM3 interface, while the LLM-conditioned queries serve as the core semantic representation for instruction-driven mask prediction.

Effectiveness of the Data Engine. Table 8 ablates our data engine on both the instance-level benchmark Inst2Seg and the semantic-level benchmark ReasonSeg. Removing the constructed training data (w/o training data) reduces performance on Inst2Seg, showing that high-quality multi-instance instruction-mask pairs are crucial for learning reliable instance-aware behavior. More importantly, removing the filtering stage (w/o filtering) causes substantial degradation on both benchmarks, with Inst2Seg mAP dropping from 31.5 to 11.9 and ReasonSeg cIoU/gIoU decreasing from 63.0/61.8 to 57.9/58.1. This suggests that naive MLLM–SAM3 data generation introduces considerable label noise. The full pipeline achieves the best performance, indicating that our data engine provides reliable supervision for instruction-driven instance segmentation.

EffectoftheAlignmentStage. Table 9 evaluates the alignment stage (stage 1) on referring and instruction-based segmentation benchmarks. Removing alignment causes consistent drops: on gRefCOCO, val mAP decreases from 57.3 to 41.3 (-16.0) and val cIoU drops from 68.3 to 57.7 (-10.6). The degradation is even more severe on instruction-following benchmarks, where Inst2Seg mAP drops from 31.5 to 8.1 (-23.4) and ReasonSeg val cIoU drops from 65.0 to 15.9 (-49.1). These results show that the alignment stage is critical for coupling language understanding with mask prediction, and improving generalization to compositional instructions.

Inference Efficiency. We compare inference efficiency under a controlled setting. All methods use the same Qwen3-VL-2B backbone and are evaluated under identical hardware conditions. Latency is measured on Inst2Seg and averaged over all instructions. As reported in Table 10, InstructSAM is substantially

Table 10 Inference-time comparison on Inst2Seg benchmark.

Model Infer Time (s) ↓

InstructSAM-2B 1.1 SAM3-Agent-Qwen3-VL-2B 29.6

##### Table 8 Effectiveness of the data engine, including the impact of the Inst2Seg data and the filtering strategy.

Inst2Seg ReasonSeg ST MT All cIoU gIoU

Setting

w/o Inst2Seg Data 47.6 17.6 25.9 62.1 60.3 w/o Filtering 10.2 13.1 11.9 57.9 58.1 Full Data Engine 52.6 22.2 31.5 63.0 61.8

Table 9 Ablation on the referring alignment pretraining (stage 1). This alignment stage substantially improves both referring and instruction-based segmentation performance.

Inst2Seg (mAP)

gRefCOCO val(mAP)

gRefCOCO val(cIoU)

ReasonSeg val(cIoU)

Stage 1

✓ 57.3 68.3 31.5 65.0

###### ✗ 41.3 (↓16.0) 57.7 (↓10.6) 8.1 (↓23.4) 15.9 (↓49.1)

[Figure 48]

[Figure 49]

[Figure 50]

Which set of icons tells me what I am forbidden from doing at this location?

If I need to cut the meat, which tools should I grab from the drying rack?

[Figure 51]

[Figure 52]

[Figure 53]

What architectural structures are primarily used to support the overhead canopy?

Find the text on the person's clothing that mentions a famous city.

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

SAM3+Agent SA2VA InstructSAM

SAM3+Agent SA2VA InstructSAM

- Figure 4 Qualitative comparison with SAM3-Agent-Qwen3-VL-2B [22] and SA2VA-4B [16]. InstructSAM better understands complex instructions and produces more accurate instance-level masks, especially in scenarios requiring fine-grained reasoning, object distinction, and multi-target localization. More qualitative results are provided in Appendix B.

faster than SAM3-Agent, highlighting the efficiency advantage of our unified framework over multi-step agentic execution.

- 5.5 Qualitive Results

Fig. 4 presents qualitative comparisons between InstructSAM and existing methods, including SAM3-AgentQwen3-VL-2B [22] and SA2VA-4B [16]. The examples cover diverse instruction-following scenarios, such as identifying forbidden signs, selecting tools, recognizing supporting structures, and locating text on clothing. Compared with the baselines, InstructSAM produces more accurate instance-level masks and better follows complex language instructions. In particular, it shows stronger capability in fine-grained visual reasoning and multi-target localization.

- 6 Conclusion

In this paper, we presented InstructSAM, a unified framework for instruction-driven multi-instance segmentation with arbitrary complex instructions. Built upon an explicit reasoning-to-instance query interface, InstructSAM translates arbitrary instructions into a set of parallel, learnable instance slots and projects them into SAM3’s detector query space. This design enables robust reasoning over free-form instructions involving attributes, relations, counting, exclusion and implicit intent. Across diverse instruction-driven and referring segmentation benchmarks, the compact 2B-scale InstructSAM consistently achieves strong performance, producing accurate and efficient multi-instance masks while substantially outperforming prior end-to-end methods and even surpassing multi-round agentic pipeline.

Appendix

- A More Training Details

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

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

[Figure 98]

[Figure 99]

Stage1 Stage2

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Figure A.1 The distribution of training datasets for InstructSAM.

- Stage 1: Referring Alignment Pretraining. In this stage, we pretrain the learnable mask queries and explicitly align the mask-query space produced by the LLM with that of SAM3. To achieve robust alignment, we leverage a large collection of category-level and phrase-level simple mask grounding data, as illustrated in Fig. A.1 (left). In total, 2.5M training samples are used for alignment pretraining. We optimize all parameters except the vision encoder in the mask decoder, which is kept frozen. The global batch size is set to 128. We use a learning rate of 5 × 10−6 for the MLLM, including the LLM, vision encoder, and projector, and 1 × 10−5 for the remaining trainable modules in the mask decoder. The MLLM is initialized from Qwen3-VL-2B, while the mask decoder is initialized from the pretrained SAM3.
- Stage 2: Reasoning Knowledge Fine-tuning. Starting from the aligned model obtained in Stage 1, we further fine-tune the system to incorporate reasoning-aware segmentation knowledge and enhance instruction-following for more complex referring and compositional scenarios. As shown in Fig. A.1 (right), Stage 2 combines instruction-level segmentation data with a subset of phrase-level referring data, maintaining grounding robustness while improving reasoning generalization. The total number of training samples in this stage is 0.5M. We continue to freeze the vision encoder of the mask decoder and fine-tune the remaining components with a smaller batch size of 64 for stable adaptation. The learning rates are reduced to 2 × 10−6 for the LLM, the MLLM vision encoder, and the projector, and set to 5 × 10−6 for other trainable modules. This stage is also trained for one epoch.

The detailed hyper-parameters for the multi-stage training are reported in Table A.1, and the dataset distribution is provided in Fig. A.1.

Table A.1 The Hyper-parameters in multi-stage training of InstructSAM. VE denotes vision encoder.

Item Stage1 Stage2

batch size 128 64 training epochs 1 1 lr of LLM 5e-6 2e-6 lr of VE in MLLM 5e-6 2e-6 lr of projector 5e-6 2e-6 lr of VE in mask decoder 0 0 lr of other modules 1e-5 5e-6 optimizer AdamW optimizer momentum β1 = 0.9, β2 = 0.999 weight decay 0.0 0.0 warmup ratio 0.03 0.03 LoRA rank 64 64

- B More Visualization Results

Instruction-basedInstanceSegmentation. Fig. C.1 presents qualitative results of InstructSAM on the Instructionbased Instance Segmentation task. Given a natural-language query that may include multiple attributes (e.g., object category, spatial relations, and contextual constraints), InstructSAM is able to accurately parse the instruction, localize the referred instance, and produce a precise instance-level segmentation mask. As illustrated, the model robustly handles complex and compositional descriptions, distinguishing the target object from visually similar distractors and cluttered backgrounds. These results demonstrate that InstructSAM effectively bridges language understanding and fine-grained mask generation, enabling reliable instance segmentation guided directly by user instructions.

Reasoning Segmentation. Fig. C.2 visualizes InstructSAM’s performance on the reasoning segmentation task. In this setting, the query cannot be resolved by category recognition alone; instead, the model must perform multi-step reasoning, such as identifying functional parts (e.g., “the part used to receive sound signals”), inferring intent or affordances (e.g., “something to sit on temporarily”). As shown, InstructSAM produces masks that align with the inferred evidence regions, demonstrating strong commonsense grounding and the ability to translate reasoning outcomes into precise, localized segmentations.

Referring Segmentation. Fig. C.3 illustrates the visualization results of InstructSAM on the Referring Segmentation task. In this setting, the model receives a referring expression that may describe the target region through appearance cues, relative position, or contextual relations (e.g., “far right person in the background” or “the bottles placed on the bench”). As shown in the examples, InstructSAM can accurately ground the expression in the scene and segment the corresponding region with clear boundaries, even under challenging conditions such as crowded scenes, small objects, and background clutter. These qualitative results further verify the strong language grounding ability of InstructSAM and its robustness in producing fine-grained masks aligned with diverse referring expressions.

- C Further Discussions

Although InstructSAM demonstrates strong performance on instruction-based segmentation, several limitations remain. First, the current version focuses on image inputs, primarily because constructing large-scale, high-quality instruction-mask supervision for videos is substantially more challenging. In video scenarios, multi-instance interactions, temporal correspondence, and frame-level mask consistency make data annotation and automatic data generation more complex, while also increasing the risk of hallucinated or inconsistent instructions. Extending the training and evaluation to video data, for example through joint training with video segmentation datasets, could further enhance temporal reasoning and mask consistency, potentially improving segmentation robustness in dynamic scenes. Second, effectively integrating segmentation datasets with existing large-scale conversational instruction involving complex reasoning remains an open challenge. Naive joint training can degrade performance on certain segmentation benchmarks, indicating non-trivial interference between datasets and objectives. Designing principled co-training strategies, such as data balancing schemes that incorporate conversational supervision while preserving or improving segmentation fidelity, is therefore crucial for further advancing instruction-following segmentation models.

References

- [1] Ronghao Dang, Yuqian Yuan, Yunxuan Mao, Kehan Li, Jiangpin Liu, Zhikai Wang, Xin Li, Fan Wang, and Deli Zhao. Rynnec: Bringing mllms into embodied world. arXiv preprint arXiv:2508.14160, 2025.
- [2] Yuqian Yuan, Ronghao Dang, Long Li, Wentong Li, Dian Jiao, Xin Li, Deli Zhao, Fan Wang, Wenqiao Zhang, Jun Xiao, et al. Eoc-bench: Can mllms identify, recall, and forecast objects in an egocentric world? arXiv preprint arXiv:2506.05287, 2025.
- [3] Zihao Xin, Wentong Li, Yixuan Jiang, Ziyuan Huang, Bin Wang, Piji Li, Jianke Zhu, Jie Qin, and Shengjun Huang. Agentvln: Towards agentic vision-and-language navigation. arXiv preprint arXiv:2603.17670, 2026.
- [4] Xiangtai Li, Henghui Ding, Haobo Yuan, Wenwei Zhang, Jiangmiao Pang, Guangliang Cheng, Kai Chen, Ziwei

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

What should I wear to protect my hands from hot water and chemicals while scrubbing?

Which items are being used specifically to display the national identity of the United Kingdom?

[Figure 139]

Which objects are used to add some natural greenery and decoration to the indoor window area?

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

potted plants.

yellow rubber gloves.

the Union Jack flags.

[Figure 147]

[Figure 148]

[Figure 149]

Find the formal decorations located at the very top of the front wall.

Find the individuals who are currently working together to move the number 48 car.

Where can I find the tall green plants that contrast with the glass and steel architecture?

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

the palm trees on the right side of the street.

[Figure 157]

three framed pictures at the top of the wall.

a group of people working on the number 48 car.

[Figure 158]

- Figure C.1 Visualization of InstructSAM on instruction-based instance segmentation.

Liu, and Chen Change Loy. Transformer-based visual segmentation: A survey. IEEE transactions on pattern analysis and machine intelligence, 2024.

- [5] Tianwei Lin, Wenqiao Zhang, Sijing Li, Yuqian Yuan, Binhe Yu, Haoyuan Li, Wanggui He, Hao Jiang, Mengze Li, Xiaohui Song, et al. Healthgpt: A medical large vision-language model for unifying comprehension and generation via heterogeneous knowledge adaptation. arXiv preprint arXiv:2502.09838, 2025.
- [6] Yihan Xie, Sijing Li, Tianwei Lin, Zhuonan Wang, Chenglin Yang, Yu Zhong, Wenjie Yan, Wenqiao Zhang, Xiaogang Guo, Jun Xiao, et al. Heartcare suite: A unified multimodal ecg suite for dual signal-image modeling

[Figure 159]

[Figure 160]

[Figure 161]

What part of the picture can help identify the ownership and registration of the car to ensure it is legally parked?

Please segment the ‘something that the cat uses to receive sound signals’ in the image.

What item in the picture can be classified as such a weapon?

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

knife.

[Figure 170]

license plate.

cat‘s ears.

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

While waiting for a consultation or interview, we cannot stand all the time and will need to rely on what object to sit down temporarily?

What object would people sit on when eating together at a dining table?

Please segment ‘something showing the price’.

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

couch.

something showing the price.

chair.

- Figure C.2 Visualization of InstructSAM on reasoning segmentation.

and understanding. arXiv preprint arXiv:2506.05831, 2025.

- [7] Tianwei Lin, Zhongwei Qiu, Wenqiao Zhang, Jiang Liu, Yihan Xie, Mingjian Gao, Zhenxuan Fan, Zhaocheng Li, Sijing Li, Zhongle Xie, et al. Omnict: Towards a unified slice-volume lvlm for comprehensive ct analysis. arXiv preprint arXiv:2602.16110, 2026.
- [8] Yuqian Yuan, Wenqiao Zhang, Juekai Lin, Yu Zhong, Mingjian Gao, Binhe Yu, Yunqi Cao, Wentong Li, Yueting Zhuang, and Beng Chin Ooi. Lmms meet object-centric vision: Understanding, segmentation, editing and generation. arXiv preprint arXiv:2604.11789, 2026.

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

Woman with arm up in the air.

Black bird head.

Far right person in the background.

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

far right person in the background.

woman with arm up in the air.

black bird head.

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

A plastic straw inserted into a brown drink.

The bottles placed on the bench.

[Figure 200]

Groom and bride.

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

Groom and bride.

a plastic straw.

the bottles placed on the bench.

- Figure C.3 Visualization of InstructSAM on referring segmentation.

- [9] Yu Zhong, Tianwei Lin, Ruike Zhu, Yuqian Yuan, Haoyu Zheng, Liang Liang, Wenqiao Zhang, Feifei Shao, Haoyuan Li, Wanggui He, et al. Unified personalized understanding, generating and editing. arXiv preprint arXiv:2601.06965, 2026.
- [10] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023.
- [11] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr,

- Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. In International Conference on Learning Representations, volume 2025, pages 28085–28128, 2025.
- [12] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.
- [13] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [14] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [15] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9579–9589, 2024.
- [16] Haobo Yuan, Xiangtai Li, Tao Zhang, Yueyi Sun, Zilong Huang, Shilin Xu, Shunping Ji, Yunhai Tong, Lu Qi, Jiashi Feng, et al. Sa2va: Marrying sam2 with llava for dense grounded understanding of images and videos. arXiv preprint arXiv:2501.04001, 2025.
- [17] Senqiao Yang, Tianyuan Qu, Xin Lai, Zhuotao Tian, Bohao Peng, Shu Liu, and Jiaya Jia. Lisa++: An improved baseline for reasoning segmentation with large language model. arXiv preprint arXiv:2312.17240, 2023.
- [18] Chaoning Zhang, Dongshen Han, Yu Qiao, Jung Uk Kim, Sung-Ho Bae, Seungkyu Lee, and Choong Seon Hong. Faster segment anything: Towards lightweight sam for mobile applications. arXiv preprint arXiv:2306.14289, 2023.
- [19] Lei Ke, Mingqiao Ye, Martin Danelljan, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu, et al. Segment anything in high quality. Advances in Neural Information Processing Systems, 36:29914–29934, 2023.
- [20] Xu Zhao, Wenchao Ding, Yongqi An, Yinglong Du, Tao Yu, Min Li, Ming Tang, and Jinqiao Wang. Fast segment anything. arXiv preprint arXiv:2306.12156, 2023.
- [21] Jingjing Li, Yue Feng, Yuchen Guo, Jincai Huang, Yongri Piao, Qi Bi, Miao Zhang, Xiaoqi Zhao, Qiang Chen, Shihao Zou, et al. Sam3-i: Segment anything with instructions. arXiv preprint arXiv:2512.04585, 2025.
- [22] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [23] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [24] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.
- [25] Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, and Jianke Zhu. Osprey: Pixel understanding with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28202–28211, 2024.
- [26] Yuqian Yuan, Hang Zhang, Wentong Li, Zesen Cheng, Boqiang Zhang, Long Li, Xin Li, Deli Zhao, Wenqiao Zhang, Yueting Zhuang, et al. Videorefer suite: Advancing spatial-temporal object understanding with video llm. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18970–18980, 2025.
- [27] Yuqian Yuan, Wenqiao Zhang, Xin Li, Shihao Wang, Kehan Li, Wentong Li, Jun Xiao, Lei Zhang, and Beng Chin Ooi. Pixelrefer: A unified framework for spatio-temporal object referring with arbitrary granularity. arXiv preprint arXiv:2510.23603, 2025.

- [28] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jie Qin, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal llm. International Journal of Computer Vision, 133(10):6794–6812, 2025.
- [29] Wenqiao Zhang, Tianwei Lin, Jiang Liu, Fangxun Shu, Haoyuan Li, Lei Zhang, He Wanggui, Hao Zhou, Zheqi Lv, Hao Jiang, et al. Hyperllava: Dynamic visual and language expert tuning for multimodal large language models. arXiv preprint arXiv:2403.13447, 2024.
- [30] Zhuonan Wang, Zhenxuan Fan, Siwen Tan, Yu Zhong, Yuqian Yuan, Haoyuan Li, Hao Jiang, Wenqiao Zhang, Feifei Shao, Hongwei Wang, et al. Mau-gpt: Enhancing multi-type industrial anomaly understanding via anomaly-aware and generalist experts adaptation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 26787–26795, 2026.
- [31] Wei Wang, Yuqian Yuan, Tianwei Lin, Wenqiao Zhang, Siliang Tang, Jun Xiao, and Yueting Zhuang. Crossview suite: Harnessing cross-view spatial intelligence of mllms with dataset, model and benchmark. arXiv preprint arXiv:2605.18621, 2026.
- [32] Haoyu Zheng, Tianwei Lin, Wei Wang, Zhuonan Wang, Wenqiao Zhang, Jiaqi Zhu, and Feifei Shao. Iad-unify: A region-grounded unified model for industrial anomaly segmentation, understanding, and generation. arXiv preprint arXiv:2604.12440, 2026.
- [33] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13009–13018, 2024.
- [34] Zhongwei Ren, Zhicheng Huang, Yunchao Wei, Yao Zhao, Dongmei Fu, Jiashi Feng, and Xiaojie Jin. Pixellm: Pixel reasoning with large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26374–26383, 2024.
- [35] Zechen Bai, Tong He, Haiyang Mei, Pichao Wang, Ziteng Gao, Joya Chen, Zheng Zhang, and Mike Zheng Shou. One token to seg them all: Language instructed reasoning segmentation in videos. Advances in Neural Information Processing Systems, 37:6833–6859, 2024.
- [36] Cilin Yan, Haochen Wang, Shilin Yan, Xiaolong Jiang, Yao Hu, Guoliang Kang, Weidi Xie, and Efstratios Gavves. Visa: Reasoning video object segmentation via large language models. In European Conference on Computer Vision, pages 98–115. Springer, 2024.
- [37] Hao Wang, Limeng Qiao, Zequn Jie, Zhijian Huang, Chengjian Feng, Qingfang Zheng, Lin Ma, Xiangyuan Lan, and Xiaodan Liang. X-sam: From segment anything to any segmentation. arXiv preprint arXiv:2508.04655, 2025.
- [38] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In CVPR, 2022.
- [39] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020.
- [40] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.
- [41] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022.
- [42] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Scaling egocentric vision: The epic-kitchens dataset. In European Conference on Computer Vision (ECCV), 2018.
- [43] Toby Perrett, Ahmad Darkhalil, Saptarshi Sinha, Omar Emara, Sam Pollard, Kranti Kumar Parida, Kaiting Liu, Prajwal Gatti, Siddhant Bansal, Kevin Flanagan, et al. Hd-epic: A highly-detailed egocentric video dataset. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23901–23913, 2025.
- [44] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014.

- [45] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 11–20, 2016.
- [46] Shuting He, Henghui Ding, Chang Liu, and Xudong Jiang. Grec: Generalized referring expression comprehension. arXiv preprint arXiv:2308.16182, 2023.
- [47] Rui Hu, Lianghui Zhu, Yuxuan Zhang, Tianheng Cheng, Lei Liu, Heng Liu, Longjin Ran, Xiaoxin Chen, Wenyu Liu, and Xinggang Wang. Groundingsuite: Measuring complex multi-granular pixel grounding. arXiv preprint arXiv:2503.10596, 2025.
- [48] Zhuofan Xia, Dongchen Han, Yizeng Han, Xuran Pan, Shiji Song, and Gao Huang. Gsva: Generalized segmentation via multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3858–3869, 2024.
- [49] Zheng Zhang, Yeyao Ma, Enming Zhang, and Xiang Bai. Psalm: Pixelwise segmentation with large multi-modal model. In European Conference on Computer Vision, pages 74–91. Springer, 2024.
- [50] Yuxuan Zhang, Tianheng Cheng, Lianghui Zhu, Rui Hu, Lei Liu, Heng Liu, Longjin Ran, Xiaoxin Chen, Wenyu Liu, and Xinggang Wang. Evf-sam: Early vision-language fusion for text-prompted segment anything model. arXiv preprint arXiv:2406.20076, 2024.
- [51] Cong Wei, Yujie Zhong, Haoxian Tan, Yingsen Zeng, Yong Liu, Hongfa Wang, and Yujiu Yang. Instructseg: Unifying instructed visual segmentation with multi-modal large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20193–20203, 2025.
- [52] Yuhao Lu, Yixuan Fan, Beixing Deng, Fangfu Liu, Yali Li, and Shengjin Wang. Vl-grasp: a 6-dof interactive grasp policy for language-oriented objects in cluttered indoor scenes. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 976–983. IEEE, 2023.

