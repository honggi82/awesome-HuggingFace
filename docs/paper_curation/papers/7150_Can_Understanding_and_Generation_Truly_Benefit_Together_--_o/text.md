# arXiv:2509.09666v5[cs.CV]31Mar2026

## Unified Multimodal Models as Auto-Encoders

Zhiyuan Yan1,2⋄,⋆, Kaiqing Lin1⋄, Zongjian Li1,3⋄, Junyan Ye4⋄, Hui Han1, Haochen Wang2,6⋆, Zhendong Wang5, Bin Lin1,3, Hao Li1, Xinyan Xiao2, Jingdong Wang2, Haifeng Wang2, Li Yuan1†

1Shenzhen Graduate School, Peking University 2Baidu, 3Rabbitpre AI, 4SYSU, 5USTC, 6CASIA

zhiyuanyan@stu.pku.edu.cn

[Figure 1]

Figure 1. Illustration of the key insight of our UAE, unifies image-to-text understanding and text-to-image generation under a reconstructive auto-encoding perspective. By optimizing reconstruction similarity through RL, the encoder (und. module) is trained to learn richer semantic representations while the decoder (gen. module) becomes better at recovering all semantics. Illustrations show strengthened fine-grained visual perception and improved complex instruction-following generation capability across RL training steps.

#### Abstract

Image-to-text (I2T) understanding and text-to-image (T2I) generation are two fundamental, important yet traditionally isolated multimodal tasks. Despite their intrinsic connection, existing approaches typically optimize them independently, missing the opportunity for mutual enhancement. In this paper, we argue that the both tasks can be connected under a shared Auto-Encoder perspective, where text serves

⋄ Equal Contribution, ⋆ Work done during an internship at the Baidu Star Program, † Corresponding Author

as the intermediate latent representation bridging the two directions — encoding images into textual semantics (I2T) and decoding text back into images (T2I). Our key insight is that if the encoder truly “understands” the image, it should capture all essential structure, and if the decoder truly “understands” the text, it should recover that structure faithfully. Building upon this principle, we propose Unified-GRPO, a post-training method based on reinforcement learning that jointly optimizes both modules through reconstructive rewards, maximizing the semantic consistency between the input and the generated images. Under this reconstruction objective, the encoder is encouraged to extract as much accu-

rate and comprehensive semantic information from the input image to maximize reconstruction quality, while the decoder is simultaneously optimized to generate conditioned on the encoder’s prior, enabling a self-evolving improvement.

Empirically, we find that using text as the intermediate representation and training under a reconstructive RL paradigm effectively benefits both I2T and T2I. The I2T module gains stronger fine-grained visual perception, such as small-object recognition, grounding, etc, while its dense embeddings and language priors, in turn, provide richer semantic signals that improve T2I fidelity and complex instruction following. These results demonstrate that the reconstructive RL establishes a mutually reinforcing cross-modal synergy within the auto-encoding framework.

#### 1. Introduction and Motivation

Unified multimodal models (UMMs) that support both generation and understanding have recently gained increasing popularity in both academia and industry [8, 16, 43, 60, 63, 69, 75, 82]. However, directly combining the understanding and generation models together leads to a sub-optimal result, as most existing arts on UMMs [7, 43, 63] suggest that optimizing diffusion-based generative objectives negatively degrade the understanding capability and learned representations (and conversely), making joint training brittle.

Consequently, some existing works decouple the UMM problem [48, 63], training understanding and generation modules separately, and missing out on potential cross-task mutual benefits. These design choices and empirical observations have dampened confidence in truly unified systems: absent demonstrable mutual gains, “unification” collapses into training two large components side by side.

In this work, we revisit the relationship between I2T and T2I from a conceptual standpoint and argue that a more principled linkage can be established by viewing them through a unified Auto-Encoder (AE) perspective. Under this view, text acts as an intermediate latent representation: the encoder extracts a semantic description from the input image (I2T), and the decoder reconstructs an image from this semantic representation (T2I). This perspective offers a natural and powerful unifying principle: if the encoder genuinely understands the image, it should capture all essential visual structure; if the decoder genuinely understands the text, it should faithfully recover that structure. Thus, highquality reconstruction becomes a proxy for enhancing both tasks simultaneously, revealing a pathway toward bidirectional synergy.

Building upon this insight, we introduce Unified-GRPO, a reinforcement-learning-based post-training method that jointly optimizes the encoder and decoder through reconstructive rewards. Unified-GRPO maximizes the semantic consistency between the original and reconstructed im-

[Figure 2]

Figure 2. The overall workflow of our method. Our post-training method, Unified-GRPO, utilizes the reconstruction objective for improved unified multimodal models (UMMs).

ages, encouraging the encoder to produce richer and more accurate textual semantics, while guiding the decoder to generate images that better adhere to the encoder’s descriptions. Through this cross-modal feedback loop, the two modules co-evolve: the encoder learns to encode more informative and comprehensive representations, and the decoder learns to generate more faithful and semantically grounded images, creating a self-reinforcing improvement cycle.

We conduct extensive experiments on visual understanding, generation, and unification tasks across a broad suite of benchmarks to verify that our post-training strategy with our core AE insight can improve the UMMs [8, 32]. Specifically, our method achieves significant improvement on image generation (e.g., from 0.73→0.86 on GenEval [14] and 0.296→0.475 on GenEval++ [76]), and largely improved fine-grained visual recognition and perception capability, e.g., from 0.05→0.45 on small object detection and from 0.15→0.75 on Person ReID of the MMT-Bench [77], consistent with the findings reported by Ross [58]) while maintaining the overall performance across visual understanding tasks. Furthermore, results on the proposed Unified-Bench show that our post-training method can largely improve the unification, resulting in a more coherent information flow between encoding and decoding.

In summary, our work makes the following contributions:

- • A unified Auto-Encoder perspective linking I2T and T2I: We propose a principled formulation where text serves as the intermediate representation connecting image encoding and decoding, offering a coherent bridge between multimodal understanding and generation.
- • Unified-GRPO, an RL-based post-training framework for cross-modal self-evolution: Through reconstructive rewards, our method jointly optimizes the encoder and decoder, enabling mutual reinforcement: richer semantic encoding improves generation, and more faithful generation strengthens fine-grained visual perception.
- • Broad applicability and consistent empirical gains: Unified-GRPO applies to various encoder–decoder multimodal systems, consistently improving text-to-image generation and enhancing fine-grained understanding (e.g., grounding, small-object recognition), while revealing interpretable trade-offs in OCR-heavy scenarios.

- Applying Unified-GRPO to UMM-1. For UMM-1, the autoregressive LLM πϕ is trained using reconstructive RL, while the diffusion transformer pθ remains frozen and acts as part of the reward environment (together with a CLIP encoder). Given an input image x, we sample a group of G

caption sequences {y(i)}Gi=1 from the old policy πϕ

old

. For each caption y(i), we extract its last hidden state h(Ti) and project it into a diffusion condition c(i) = g(h(Ti)), which is then used to synthesize a reconstructed image x˜(i) ∼ pθ(· | c(i)). The LLM is updated via the GRPO objective [52], where each trajectory oi corresponds to the token sequence of y(i), and the probability ratio is

rt(i)(ϕ) =

πϕ(yt(i) | x,y<t(i)) πϕ

old

(yt(i) | x,y<t(i))

.

This process encourages the LLM to emit hidden representations that maximize the diffusion’s reconstruction quality.

- Applying Unified-GRPO to UMM-2. For UMM-2, the same autoregressive model performs both I2T and T2I. Unified-GRPO is applied in an identical manner, except that the decoder Dϕ is now autoregressive rather than diffusionbased. The RL pipeline becomes: x −−→πϕ y,y −−→πϕ x,˜ with

UMM-1: AR for und., DiT for visual gen. UMM-2: AR for both und. and gen.

[Figure 3]

[Figure 4]

RM Image RM Image

| | |
|---|---|
|Image<br><br>Prompt| |

|Prompt<br><br>Text|
|---|

|Prompt<br><br>Text<br><br>AR|
|---|

Text

Image

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

###### AR

###### AR A AR

MM-DiT

Text

Prompt

Image

Image

Text

Encoding Decoding

Encoding Decoding

UniWorld, MetaQuery, etc Janus-Pro, X-Omni, etc

- Figure 3. Illustration of how Unified-GRPO integrates into two representative UMM architectures. UMM-1 uses an AR model for understanding and an MM-DiT for image generation, while UMM-2 employs a single AR backbone for both understanding and generation. In the diagram, “text” or “image” inside a rectangle denotes latent tokens, whereas those without a rectangle represent raw data inputs. Unified-GRPO can be applied to the LLM backbone in both architectures to provide reconstructive RL.

#### 2. UAE Methodology

Our goal is to unify image-to-text (I2T) understanding and text-to-image (T2I) generation within a single auto-encoding perspective, where text serves as the intermediate latent representation connecting the two directions. Given an input image x, a unified multimodal model (UMM) first produces a semantic description y (I2T), and another UMM reconstructs an image xˆ from y (T2I). We then adopt reconstructive reinforcement learning to maximize the semantic similarity between x and xˆ, enabling mutual improvement between understanding and generation.

reconstruction reward R(x,x˜) = cos(fCLIP(x),fCLIP(˜x)). This enables a fully AR model to co-evolve its understanding and generation abilities within a single shared token space. The specific implementation here is similar to previous work that incentivizes AR for improved image generation (such as T2I-R1 [23] and AR-GRPO [78]), and the key difference is that we use the reconstruction reward between the input and the generated image embeddings, enabling it to optimize both understanding and generation modules jointly.

##### 2.1. Unified-GRPO

We propose Unified-GRPO, a reconstructive reinforcement learning method designed to unify image-to-text (I2T) understanding and text-to-image (T2I) generation by training the model to maximize reconstruction fidelity.

##### 2.2. Unified-Bench: A Benchmark tailored for Evaluating the Unified Models

Motivation. As illustrated in Fig. 1, we view understanding (I→T) and generation (T→I) as a closed loop whose two halves should mutually enhance each other. Judging image realism alone or caption fidelity alone cannot reveal whether a system is truly unified. We introduce a reconstructionbased similarity, i.e., Unified-Score, to directly test whether the semantics extracted during understanding are sufficient for faithful regeneration, and whether regeneration in turn validates the completeness of the understanding.

To apply this framework to existing unified multimodal models (UMMs), we consider the two dominant architectural families shown in Fig. 3: (1) UMM-1, where an autoregressive language model (LLM) is responsible for multimodal understanding and provides language priors for a diffusion transformer (MM-DiT) used in image generation (e.g., UniWorld [32], MetaQuery [43], etc); and (2) UMM-2, where a single autoregressive model handles both visual understanding and visual generation in a shared token space (e.g., Janus-Pro [8], X-Omni [13], etc). In both families, the LLM plays a central role—either as the core understanding module (UMM-1) or as the backbone of both understanding and generation (UMM-2). Since GRPO and related RL algorithms have proven highly effective for training LLMs, we extend this idea to UMMs and employ GRPO to optimize the LLM components toward improved cross-modal reconstruction.

Protocol-1: Evaluation of the unified score from the reconstruction similarity. To quantify the unified score, we start from 100 diverse source images. The prompt, used to allow the model to generate cpation, is detailed in Supplementary. The same model then synthesizes an image from its own caption. We compute unified scores between the reconstruction and the source using four widely adopted vision backbones, CLIP [49], LongCLIP [80], DINO-v2 [42], and

###### Table 1. Ablation study on the proposed post-training on understanding, generation, and unification benchmarks. We apply our method to the two typical unified multimodal models and show the clear improvement.

Understanding Generation Unification MMB MMMU GenEval DPGBench Unified-Score

Model

UniWorld 83.5 58.6 84.0 81.2 79.0 w/ Ours 84.8 58.2 89.0 86.4 86.1 vs. Baseline +1.3% -0.4% +5% +5.2% +7.1%

Janus-pro 79.2 41.0 80.0 84.2 82.8 w/ Ours 80.3 41.6 84.3 88.9 89.1 vs. Baseline +1.1% +0.6% +4.3% +4.7% +6.3%

###### Table 2. Protocol-1 of Unified-Bench: comparing of unified score of different methods on Unified-Bench, the tailored benchmark for evaluating the unification between understanding and generation models in the UMMs.

Method CLIP LongCLIP DINO-v2 DINO-v3 Overall GPT-4o-Image [41] 90.42 94.37 81.74 77.27 85.95 BAGEL [11] 88.97 93.35 78.55 73.05 83.48 BLIP-3o [7] 84.84 90.24 68.31 62.86 76.56 Janus-Pro [8] 88.72 93.45 78.30 70.61 82.77 OmniGen2 [64] 88.36 93.11 77.70 74.07 83.31 Show-o [68] 80.18 86.75 58.20 51.51 69.16 UniWorld-V1 [32] 85.49 91.53 72.12 66.83 78.99 UAE 90.50 94.35 81.98 77.54 86.09

###### Table 3. Evaluating how “friendly” the output caption is for image generation. We use the data from Unified-Bench to assess the quality of the captions produced by the understanding model for better text-to-image generation. Bold indicates the best result.

Method CLIP LongCLIP DINO-v2 DINO-v3 Overall

Qwen-2.5-VL-3B [3] 88.34 92.62 73.91 70.02 80.85 Qwen-2.5-VL-7B [3] 88.26 92.89 76.12 70.96 81.92

###### UAE 90.50 94.35 81.98 77.54 86.09

DINO-v3 [54], and report per-backbone similarities and an overall summary.

Protocol-2: Quality Evaluation of the model’s output caption for reconstruction. We further evaluate caption quality through pairwise comparisons against various baselines, using four commercial LLM judges: Claude-4.1, GPT-4o, Grok-4, and o4-mini. The prompting strategy is detailed in Supplementary. For evaluation, we use pairwise winning rate (%), the percentage of times our model is preferred over baselines as the main metric.

#### 3. Experiments

##### 3.1. Ablation on Unified-GRPO

To comprehensively evaluate the effectiveness of the proposed Unified-GRPO, we implement our method on the two typical unified multimodal models: UniWorld [32] and Janus-Pro [8], among the understanding, generation, and unification benchmarks. Tab. 1 shows that applying UnifiedGRPO to both representative UMM architectures consistently improves their performance. The gains are most sig-

- Table 4. Benchmarking results of text-to-image generation capability on GenEval [15] benchmark. ‘†’ refers to the methods using the LLM rewriter. Bold indicates the best result, and underlined denotes the second best.

Method Single Two Counting Colors Position Color Overall

Janus Pro [8] 0.99 0.89 0.59 0.90 0.79 0.66 0.80 BLIP3-o 8B [7] - - - - - - 0.84 UniWorld-V1 [32] 0.99 0.93 0.79 0.89 0.49 0.70 0.80 UniWorld-V1† [32] 0.98 0.93 0.81 0.89 0.74 0.71 0.84 OmniGen2 [64] 1.00 0.95 0.64 0.88 0.55 0.76 0.80 X-Omni† [13] 0.98 0.95 0.75 0.91 0.71 0.68 0.83 BAGEL [11] 0.99 0.94 0.81 0.88 0.64 0.63 0.82 BAGEL† [11] 0.98 0.95 0.84 0.95 0.78 0.77 0.88

UAE 1.00 0.89 0.84 0.90 0.71 0.79 0.86 UAE† 1.00 0.97 0.82 0.95 0.73 0.84 0.89

- Table 5. Comparisons of challenging instruction following generation ability with other unified multimodal models on Geneval++ [15]. Bold indicates the best result, and underlined denotes the second best.

Method Color Count Color/Count Color/Pos Pos/Count Pos/Size Multi-Count Overall

Janus-Pro [8] 0.450 0.300 0.125 0.300 0.075 0.350 0.125 0.246 T2I-R1 [23] 0.675 0.325 0.200 0.350 0.075 0.250 0.300 0.311 BLIP3-o 4B [7] 0.125 0.225 0.100 0.450 0.125 0.550 0.225 0.257 BLIP3-o 8B [7] 0.250 0.250 0.125 0.600 0.125 0.575 0.225 0.307 OmniGen2 [64] 0.550 0.425 0.200 0.275 0.125 0.250 0.450 0.325 Bagel [11] 0.325 0.600 0.250 0.325 0.250 0.475 0.375 0.371

UAE 0.550 0.525 0.550 0.550 0.450 0.400 0.400 0.475

- Table 6. Protocol-2 of Unified-Bench: evaluating the quality of output caption of our trained understanding model (3B) against different opponents on Unified-Bench, evaluated by four judge models (using official commercial API). We use the metric of Pairwise winning rate (%) for evaluation. The Avg column reports the mean score across judges.

Our Wining Rate (%) Claude-4.1 GPT-4o Grok-4 o4-mini Avg

Opponent # Param

GPT-4o [41] - 47.4 89.4 30.6 21.2 47.2 Bagel [11] 7B 57.7 92.9 58.3 48.2 64.3 OmniGen2 [64] 3B 67.9 97.6 63.5 56.5 71.4 Show-o [68] 1.3B 97.8 100.0 89.8 91.0 94.7

Qwen-2.5-VL-3B [3] 3B 76.3 99.0 67.0 63.0 76.3 Qwen-2.5-VL-7B [3] 7B 68.8 99.0 62.0 56.0 71.5

nificant on generation and unification metrics, where reconstruction is directly optimized, yielding improvements of 4∼5% on generation and over 6% on unified reconstruction quality. Understanding performance exhibits only modest gains, which we attribute to the limited capacity of current generation models: imperfect reconstructions can introduce negative feedback to the encoder. Nevertheless, as we will show later (Sec. 3.5), Unified-GRPO can notably enhance fine-grained perceptual abilities, particularly in tasks involving subtle difference recognition and visual grounding via our reconstructive RL training. Since the UniWorld-based model demonstrates stronger performance in both generation and understanding compared to Janus, we adopt this architecture as the primary backbone for all subsequent experiments.

[Figure 14]

- Figure 4. Qualitative results on the complex and long-context generation. Our method can recover very detailed semantics from the highly descriptive input caption over the baseline, demonstrating that improved understanding can notably benefit generation.

[Figure 15]

- Figure 5. Reconstruction results vs. RL training steps. With the RL steps increasing, the understanding model (encoder) achieves better perception capability to produce an informative, detailed, yet accurate description to reconstruct the original image comprehensively; while the generation model (decoder) can take the rich description as input for recovering all semantics faithfully.

- 3.2. Unification Evaluation We assess the unified degree with the proposed UnifiedBench. Tab. 2 shows that our UAE achieves the best Overall unified score (86.09), surpassing GPT-4o-Image (85.95). Specifically, UAE obtains the top results on CLIP (90.50), DINO-v2 (81.98), and DINO-v3 (77.54), and statistical parity on LongCLIP (94.35 vs. 94.37). These consistent gains across contrastive (CLIP-family) and self-supervised (DINOfamily) features suggest that our UAE framework can preserve layout- and texture-level semantics that translate into more faithful reconstructions.
- 3.3. Text-to-Image Generation Evaluation

We evaluate UAE on two standard benchmarks: GenEval and its improved version GenEval++, which probe compositional understanding and instruction-following in increasingly chal-

lenging settings. More evaluations are in the Supplementary.

GenEval. As shown in Tab. 4, without considering LLM rewriting, our UAE attains the best Overall score among unified models (0.86). It leads on Counting (0.84) and Color attribution (0.79; +16 points vs. Bagel’s 0.63 and +3 vs. OmniGen2’s 0.76), co-leads on Colors (0.90), is second-best on Position (0.71), and reaches 0.89 on Two object (below the strongest 0.94–0.95). When considering LLM rewriting, e.g., using the same rewritten prompts with Bagel, our UAE achieves an overall score of 0.89 on average, demonstrating the SOTA performance in image generation.

GenEval++ (harder compositional control). GenEval++ [76] extends GenEval to prompts with three or more objects, each bearing distinct attributes and spatial relations, demanding comprehensive, multiconstraint satisfaction. In Tab. 5, UAE achieves the best

[Figure 16]

###### Figure 6. Qualitative examples on GenEval++. Under the complex and anti-realistic cases, our method demonstrates a clear advantage in multi-attribute instruction-following generation over other methods.

[Figure 17]

###### Figure 7. Qualitative examples showing how reconstruction-driven RL improves image-to-text understanding. Compared to the baseline, our model better identifies subtle differences and performs accurate visual grounding, demonstrating that reconstruction-driven RL encourages richer and more precise semantic extraction in image-to-text understanding.

- Table 7. High-level meta-tasks evaluation results on the comprehensive multimodal understanding benchmark: MMT-Bench [77]. Accuracy is the metric, and the Overall score is computed as the mean of all displayed subtasks.

Model Overall VR Loc Count HLN VC VG AR PLP I2IT RR Emo VI OCR DU IR 3D

Frequency Guess 32.3 30.0 28.2 28.2 43.4 28.2 29.1 30.0 29.4 30.8 33.5 30.1 52.1 30.4 37.6 29.9 26.5 Random Guess 27.9 27.1 28.1 25.0 41.6 25.0 24.8 26.6 21.2 33.4 10.5 25.4 50.8 27.2 30.3 24.3 25.5

InternVL-Chat-v1.2-34B 58.7 81.3 59.4 66.4 82.4 82.3 49.4 52.6 37.4 32.8 55.0 48.7 61.5 60.5 68.3 56.3 45.5 Qwen-VL-Plus 56.8 82.6 55.3 61.1 69.9 86.5 43.6 53.4 43.1 37.8 53.0 41.6 50.3 65.6 77.3 40.7 46.5 GPT-4V 54.1 85.3 55.6 51.6 69.6 80.3 25.0 47.7 48.2 31.8 52.5 45.1 47.9 68.0 69.8 44.9 42.0 GeminiProVision 56.2 84.7 43.6 56.4 65.9 80.1 33.0 57.4 40.3 31.5 58.5 55.2 47.5 59.5 71.6 68.4 45.2 DeepSeek-VL-7B 48.0 75.6 42.0 44.5 60.6 69.1 38.4 44.8 38.3 23.5 48.8 43.8 47.7 61.1 51.9 30.5 47.2 Claude3V-Haiku 47.4 74.3 44.8 51.1 63.6 67.6 26.9 46.2 35.5 22.8 50.0 35.2 42.9 54.4 69.8 34.6 38.2 ShareGPT4V-7B 47.8 74.2 36.0 50.9 62.4 71.6 35.4 46.2 39.2 21.8 59.8 44.3 54.5 47.8 47.9 27.8 45.2 LLaVA-v1.5-7B 46.1 72.8 34.3 47.5 61.6 68.1 34.0 46.6 36.0 22.2 58.0 42.5 57.6 45.0 40.8 26.1 44.8

Qwen-2.5-VL-3B 56.3 78.7 40.3 42.8 72.5 83.6 46.2 53.0 40.8 32.5 71.3 47.5 48.4 75.0 70.0 56.8 42.5 Ours (Qwen-3B) 56.5 80.1 47.3 44.7 72.8 84.1 47.1 53.5 46.6 32.7 71.3 48.3 57.6 68.8 58.4 50.6 40.0 vs. Baseline +0.2 +1.4 +7.0 +1.9 0.3 +0.5 +0.9 +0.5 +5.8 +0.2 +0.0 +0.8 +9.2 -6.2 -11.6 -6.2 -2.5

- Table 8. Evaluation results on fine-grained visual perception oriented sub-tasks on MMT-Bench [77]. Accuracy is the metric, and the Overall score is computed as the mean of all displayed subtasks. We show notable improvements across various fine-grained understanding tasks, highlighting the positive impact of generation on understanding.

Fine-grained Visual Recognition Color and Geometry Perception Model Overall

Salient Obj. Detection RGBD

Transparent Object Det.

Small Object Detection

Rotated Object Detection

Person Re-ID

Color Constancy

Color Assimilation

Geometrical Relativity

Geometrical Perspective

Polygon Localization

InternVL-Chat-V1.2-34B 63.4 28.5 66.5 64.5 46.7 60.0 34.5 44.5 82.5 75.0 46.1 Qwen-VL-Plus 62.3 44.5 47.5 59.5 60.0 50.5 47.5 29.0 58.3 43.0 63.8 GPT-4V 62.0 42.0 56.5 52.0 79.0 49.0 65.0 24.7 43.3 35.7 66.0 GeminiProVision 61.6 45.0 38.5 43.0 50.0 72.5 38.9 53.5 46.0 43.3 36.0 DeepSeek-VL-7B 53.2 40.0 53.5 43.5 36.7 32.5 27.5 52.0 54.2 56.0 23.4 Claude3V-Haiku 52.2 43.0 19.5 44.0 46.7 35.0 38.5 58.5 55.8 56.5 66.7 ShareGPT4V-7B 51.5 40.5 39.0 37.5 27.8 24.0 52.8 26.5 60.0 65.8 32.0 LLaVA-v1.5-7B 49.5 37.5 40.0 31.5 30.0 23.0 56.9 28.0 64.0 70.0 34.0 Frequency 31.7 26.0 26.0 27.5 28.9 30.0 52.8 51.0 50.5 53.3 31.5 Random 28.5 28.5 29.0 27.0 24.4 26.0 48.6 50.0 50.5 51.7 27.5

Qwen-2.5-VL-3B 32.5 25.0 15.0 5.0 33.3 15.0 28.6 50.0 60.0 58.3 35.0 Ours (Qwen-3B) 56.9 45.0 45.0 45.0 55.6 75.0 42.9 60.0 65.0 75.0 60.0 vs. Baseline +24.4 +20 +30 +40 +22.3 +60 +14.3 +10 +5 +16.7 +25

Overall score (0.475), leading on Color/Count (0.550) and Pos/Count (0.450), with runner-up performance on Color/Pos (0.550) and Multi-Count (0.400). Qualitative visualizations in Fig. 5 further show accurate attribute binding, disambiguation across multiple entities, and robust position–count consistency under long prompts. This highlights that our method can achieve notable improvement in complex instruction following.

##### 3.4. Image-to-Text Understanding Evaluation

Here, we conduct experiments to verify that after our posttraining, the encoder can achieve improved image-to-text, in terms of caption quality, and greater “generation-friendly”. Caption quality evaluation by commercial LLMs. As shown in Tab. 6, our understanding model (using Qwen-

- 2.5-VL-3B as the baseline) attains high average win rates: 94.7 vs. Show-o, 71.4 vs. OmniGen2, 64.3 vs. Bagel, and 76.3/71.5 vs. Qwen-2.5-VL (3B/7B), while remaining competitive with GPT-4o (47.2). The cross-judge agreement suggests our captions improve along multiple axes, completeness, attribute binding, relational, and spatial fidelity. Improving the understanding model as a better captioner suitable for generation. Under the Unified-Bench ”caption→generate→compare” protocol, captions produced

by our trained understanding model yield the highest reconstruction similarity across all four backbones (Tab. 3): 90.50 (CLIP), 94.35 (LongCLIP), 81.98 (DINO-v2), 77.54 (DINO-v3), with 86.09 Overall. These results indicate that the caption generated by our understanding model is more suitable for generation.

##### 3.5. Evaluation on the Understanding Benchmark.

We evaluate on MMT-Bench [77], which comprises highlevel meta-tasks1 The overall score remains essentially unchanged with a marginal improvement over the baseline (+0.2%; Tab. 7). However, if we zoom in to observe finegrained visual recognition suite (Tab. 8), the benefits of our generation-augmented training for perception become pronounced: we observe large absolute gains in Small Object Detection (+40.0%) and Person Re-ID (+60.0%), yielding a +24.4% increase in the fine-grained overall. These results indicate that generation does not harm understanding, but can instead enhance fine-grained visual perception capability.

1The tasks include VR (Visual Recognition), Loc (Spatial Localization), OCR (Text Reading), Count (Object Counting), HLN (Hallucination), IR (Image Retrieval), 3D, VC (Visual Caption), VG (Visual Grounding), DU (Document Understanding), AR (Action Recognition), PLP (Pixel-Level Perception), I2IT (Image-to-Image Translation), RR (Relation Reasoning), Emo (Emotion), and VI (Visual Illusion).

[Figure 18]

- Figure 8. Training dynamics of our reconstruction-oriented RL stage. Left: The reward score steadily increases as the policy learns to generate captions that more faithfully preserve the visual information in the input image. Right: The caption length gradually grows throughout training, indicating that the model is producing richer and more detailed textual descriptions. Together, these trends show that the RL optimization encourages the model to encode progressively more complete image information into text, ensuring that the downstream decoder receives a maximally informative representation.

##### 3.6. Case-Study in Fine-Grained Visual Perception

Spotting Subtle Differences. Fig. 7 (top) presents a challenging visual comparison task, where the baseline model fails to detect the fine difference between two images: one image shows two men standing near the car, while in the other image only one man is visible. The baseline incorrectly answers option A due to missing the subtle change. In contrast, our model, trained with Unified-GRPO, provides a detailed analysis of both images, accurately recognizing the presence and position of each person, the vehicle’s location, the outdoor parking setting, and the contextual cues.

Visual Object Grounding. Fig. 7 (bottom) demonstrates another demanding task requiring precise grounding. The input instruction asks the model to caption the scene and identify the corresponding region ID for “a skier in yellow, blue, orange, and pink clothing.” The baseline generates a generic caption and completely ignores the grounding instruction. After our training, the model not only follows the instruction faithfully but also grounds the described skier to the correct region ID by identifying the color composition of the outfit and matching it to the labeled bounding box.

#### 4. Related Work

Recent advancements in multimodal AI have led to the development of Unified Multimodal Models (UMMs) [81]. The architectural designs of current UMMs can be broadly categorized into two paradigms: (1) AR-based Approaches: In this setup, all modalities, including images and text, are tokenized and processed sequentially using an autoregressive transformer. Systems like Chameleon and EMU generate image tokens akin to language modeling by predicting the next token in a sequence [8, 28, 47, 55, 63, 65]. An evolution of this idea is seen in Show-o [68], which enhances token prediction with a discrete diffusion mechanism, introducing a structured denoising process during generation. (2) Hybrid AR-Diffusion Architectures: Some models combine autoregressive modeling with diffusion-based im-

age synthesis [75]. For instance, Transfusion and similar systems [11, 37, 53, 71, 82] extend a shared transformer backbone with a dedicated diffusion or flow-matching head for high-fidelity image generation. Alternatively, other approaches freeze a pre-trained MLLM and use learnable query modules or MLPs to extract and route intermediate representations to an external image generator [7, 32, 43]. A more recent direction integrates standard autoregressive language processing with masked-autoregressive reconstruction for visual data. MAR [29] enables image generation without relying on vector quantization, instead reconstructing patches in a flexible order. This approach has been adopted in models such as Harmon [12, 59, 66]. Meanwhile, some works [7, 13] use a discretized SigLIP [57] to convert images into tokens, training a single autoregressive model over these visual and language tokens, while employing a diffusion model for the final image decoding. Similar post-training works [61, 70] based on reconstruction demonstrate that using the dense image feature as the “rich text” condition for training diffusion models, which improves image generation. Additionally, RL-based frameworks have been proposed to enhance multimodal learning [34, 45, 72, 74].

#### 5. Conclusion

We show that an auto-encoder can serve as a foundational architecture for unifying image-to-text understanding and text-to-image generation. This paradigm leverages text as a shared intermediate latent representation. By introducing Unified-GRPO, we jointly optimize both, creating a synergistic feedback loop, enabling the auto-encoder principle to benefit both understanding and generation tasks simultaneously. This simple yet powerful design yields stronger fine-grained visual perception, richer semantic encoding, and improved complex instruction-following capability. Our findings highlight the value of treating multimodal tasks not as isolated objectives but as mutually reinforcing components of a unified system, paving the way for more coherent and synergistic multimodal learning.

Acknowledgment. This work was supported in part by the Natural Science Foundation of China (No. 62332002, 62425101), The Guangdong Grants (Grant No.2023ZT10X075), and Shenzhen Science and Technology Program (KQTD20240729102051063).

#### References

- [1] Stability AI. Sd3-medium. https://stability.ai/ news/stable-diffusion-3-medium, 2024. 5
- [2] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425– 2433, 2015. 1
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 4
- [4] Eslam Mohamed Bakr, Pengzhan Sun, Xiaoqian Shen, Faizan Farooq Khan, Li Erran Li, and Mohamed Elhoseiny. Hrs-bench: Holistic, reliable and scalable benchmark for textto-image models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20041–20053,

2023. 1

- [5] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. 1
- [6] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 5
- [7] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025. 2, 4, 8, 5
- [8] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025. 2, 3, 4, 8, 5
- [9] Jaemin Cho, Yushi Hu, Jason M Baldridge, Roopal Garg, Peter Anderson, Ranjay Krishna, Mohit Bansal, Jordi PontTuset, and Su Wang. Davidsonian scene graph: Improving reliability in fine-grained evaluation for text-to-image generation. In ICLR, 2024. 1
- [10] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017. 1
- [11] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 4, 8, 5
- [12] Lijie Fan, Luming Tang, Siyang Qin, Tianhong Li, Xuan Yang, Siyuan Qiao, Andreas Steiner, Chen Sun, Yuanzhen

- Li, Tao Zhu, et al. Unified autoregressive visual generation and understanding with continuous tokens. arXiv preprint arXiv:2503.13436, 2025. 8
- [13] Zigang Geng, Yibing Wang, Yeyao Ma, Chen Li, Yongming Rao, Shuyang Gu, Zhao Zhong, Qinglin Lu, Han Hu, Xiaosong Zhang, et al. X-omni: Reinforcement learning makes discrete autoregressive image generative models great again. arXiv preprint arXiv:2507.22058, 2025. 3, 4, 8
- [14] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-toimage alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023. 2, 1
- [15] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-toimage alignment. Advances in Neural Information Processing Systems, 36, 2024. 4
- [16] Agrim Gupta, Linxi Fan, Surya Ganguli, and Li Fei-Fei. Metamorph: Learning universal controllers with transformers. arXiv preprint arXiv:2203.11931, 2022. 2
- [17] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In EMNLP (1), 2021. 1
- [18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 1
- [19] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3,

2022. 1

- [20] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024. 1, 5
- [21] Kaiyi Huang, Chengqi Duan, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench++: An enhanced and comprehensive benchmark for compositional text-to-image generation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 1
- [22] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 1
- [23] Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, and Hongsheng Li. T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot. arXiv preprint arXiv:2505.00703, 2025. 3, 4
- [24] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 5
- [25] Tony Lee, Michihiro Yasunaga, Chenlin Meng, Yifan Mai, Joon Sung Park, Agrim Gupta, Yunzhi Zhang, Deepak Narayanan, Hannah Teufel, Marco Bellagente, et al. Holistic evaluation of text-to-image models. Advances in Neural Information Processing Systems, 36:69981–70011, 2023. 1

- [26] Baiqi Li, Zhiqiu Lin, Deepak Pathak, Jiayao Li, Yixin Fei, Kewen Wu, Xide Xia, Pengchuan Zhang, Graham Neubig, and Deva Ramanan. Evaluating and improving compositional text-to-visual generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5290–5301, 2024. 1
- [27] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024. 5
- [28] Hao Li, Yanhao Jia, Peng Jin, Zesen Cheng, Kehan Li, Jialu Sui, Chang Liu, and Li Yuan. Freestyleret: retrieving images from style-diversified queries. In European Conference on Computer Vision, pages 258–274. Springer, 2024. 8
- [29] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37:56424–56445, 2024. 8
- [30] Yi Li, Haonan Wang, Qixiang Zhang, Boyu Xiao, Chenchang Hu, Hualiang Wang, and Xiaomeng Li. Unieval: Unified holistic evaluation for unified multimodal understanding and generation. arXiv preprint arXiv:2505.10483, 2025. 1
- [31] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. Hunyuan-dit: A powerful multiresolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024. 5
- [32] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025. 2, 3, 4, 8, 5
- [33] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 1
- [34] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025. 8, 1
- [35] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 1
- [36] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022. 1
- [37] Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai yu, Liang Zhao, Yisong Wang, Jiaying Liu, and Chong Ruan. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation, 2024. 8

- [38] Zichen Miao, Jiang Wang, Ze Wang, Zhengyuan Yang, Lijuan Wang, Qiang Qiu, and Zicheng Liu. Training diffusion models towards diverse image generation with reinforcement learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10844–10853, 2024. 1
- [39] Yuwei Niu, Munan Ning, Mengren Zheng, Bin Lin, Peng Jin, Jiaqi Liao, Kunpeng Ning, Bin Zhu, and Li Yuan. Wise: A world knowledge-informed semantic evaluation for text-toimage generation. arXiv preprint arXiv:2503.07265, 2025. 1
- [40] OpenAI. Dall·e 3. https://openai.com/index/ dall-e-3, 2024. 5
- [41] OpenAI. Gpt-4o. https://openai.com/index/ introducing-4o-image-generation, 2025. 4
- [42] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 3
- [43] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025. 2, 3, 8
- [44] Yuang Peng, Yuxin Cui, Haomiao Tang, Zekun Qi, Runpei Dong, Jing Bai, Chunrui Han, Zheng Ge, Xiangyu Zhang, and Shu-Tao Xia. Dreambench++: A human-aligned benchmark for personalized image generation. arXiv preprint arXiv:2406.16855, 2024. 1
- [45] Caiyong Piao, Zhiyuan Yan, Haoming Xu, Yunzhen Zhao, Kaiqing Lin, Feiyang Xu, and Shuigeng Zhou. Towards policy-adaptive image guardrail: Benchmark and method. arXiv preprint arXiv:2603.01228, 2026. 8
- [46] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 5
- [47] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. arXiv preprint arXiv:2412.03069, 2024. 8
- [48] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2545–2555, 2025. 2, 5
- [49] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 3, 2
- [50] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2): 3, 2022. 9

- [51] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 1, 2
- [52] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 3, 1, 2
- [53] Weijia Shi, Xiaochuang Han, Chunting Zhou, Weixin Liang, Xi Victoria Lin, Luke Zettlemoyer, and Lili Yu. Llamafusion: Adapting pretrained language models for multimodal generation. arXiv preprint arXiv:2412.15188, 2024. 8
- [54] Oriane Sim´eoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 4
- [55] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 8
- [56] Yu Tian, Yue Liu, Shiqi Wang, and Sam Kwong. Quality assessment for text-to-image generation: A survey. IEEE MultiMedia, 2025. 1
- [57] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 8
- [58] Haochen Wang, Anlin Zheng, Yucheng Zhao, Tiancai Wang, Zheng Ge, Xiangyu Zhang, and Zhaoxiang Zhang. Reconstructive visual instruction tuning. arXiv preprint arXiv:2410.09575, 2024. 2
- [59] Peiyu Wang, Yi Peng, Yimeng Gan, Liang Hu, Tianyidan Xie, Xiaokun Wang, Yichen Wei, Chuanxin Tang, Bo Zhu, Changshi Li, et al. Skywork unipic: Unified autoregressive modeling for visual understanding and generation. arXiv preprint arXiv:2508.03320, 2025. 8
- [60] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 2, 5
- [61] XuDong Wang, Xingyi Zhou, Alireza Fathi, Trevor Darrell, and Cordelia Schmid. Visual lexicon: Rich image features in language space. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19736–19747, 2025. 8
- [62] Yibin Wang, Zhimin Li, Yuhang Zang, Yujie Zhou, Jiazi Bu, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Pref-grpo: Pairwise preference reward-based grpo for stable text-to-image reinforcement learning. arXiv preprint arXiv:2508.20751, 2025. 1
- [63] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai

- Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12966–12977, 2025. 2, 8
- [64] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025. 4, 5
- [65] Junfeng Wu, Yi Jiang, Chuofan Ma, Yuliang Liu, Hengshuang Zhao, Zehuan Yuan, Song Bai, and Xiang Bai. Liquid: Language models are scalable multi-modal generators. arXiv preprint arXiv:2412.04332, 2024. 8
- [66] Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Zhonghua Wu, Qingyi Tao, Wentao Liu, Wei Li, and Chen Change Loy. Harmonizing visual representations for unified multimodal understanding and generation. arXiv preprint arXiv:2503.21979,

2025. 8

- [67] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13294–13304, 2025. 5
- [68] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 4, 8, 5
- [69] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One Single Transformer to Unify Multimodal Understanding and Generation, 2024. 2
- [70] Ji Xie, Trevor Darrell, Luke Zettlemoyer, and XuDong Wang. Reconstruction alignment improves unified multimodal models. arXiv preprint arXiv:2509.07295, 2025. 8
- [71] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Showo2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025. 8
- [72] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Wang, Weiyun Ye, Shihao Geng, Yiren Zhao, Jiaming Li, Cunjian Li, Hang Sun, et al. Imagereward: Learning and evaluating human preferences for text-to-image generation. In Advances in Neural Information Processing Systems, 2023. 8, 1
- [73] Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan, Shen Yang, Qunlin Jin, Shurun Li, et al. Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation. arXiv preprint arXiv:2412.21059, 2024. 1
- [74] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025. 8, 1
- [75] Zhiyuan Yan, Junyan Ye, Weijia Li, Zilong Huang, Shenghai Yuan, Xiangyang He, Kaiqing Lin, Jun He, Conghui He, and Li Yuan. Gpt-imgeval: A comprehensive benchmark for diagnosing gpt4o in image generation. arXiv preprint arXiv:2504.02782, 2025. 2, 8

- [76] Junyan Ye, Dongzhi Jiang, Zihao Wang, Leqi Zhu, Zhenghao Hu, Zilong Huang, Jun He, Zhiyuan Yan, Jinghua Yu, Hongsheng Li, et al. Echo-4o: Harnessing the power of gpt4o synthetic images for improved image generation. arXiv preprint arXiv:2508.09987, 2025. 2, 5, 1
- [77] Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, Jiayi Lei, Quanfeng Lu, Runjian Chen, Peng Xu, Renrui Zhang, Haozhe Zhang, Peng Gao, Yali Wang, Yu Qiao, Ping Luo, Kaipeng Zhang, and Wenqi Shao. Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi, 2024. 2, 7
- [78] Shihao Yuan, Yahui Liu, Yang Yue, Jingyuan Zhang, Wangmeng Zuo, Qi Wang, Fuzheng Zhang, and Guorui Zhou. Argrpo: Training autoregressive image generation models via reinforcement learning. arXiv preprint arXiv:2508.06924,

2025. 3

- [79] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024. 1
- [80] Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang. Long-clip: Unlocking the long-text capability of clip. In European conference on computer vision, pages 310–325. Springer, 2024. 3
- [81] Xinjie Zhang, Jintao Guo, Shanshan Zhao, Minghao Fu, Lunhao Duan, Jiakui Hu, Yong Xien Chng, Guo-Hua Wang, QingGuo Chen, Zhao Xu, et al. Unified multimodal understanding and generation models: Advances, challenges, and opportunities. arXiv preprint arXiv:2505.02567, 2025. 8
- [82] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 2, 8

## Unified Multimodal Models as Auto-Encoders Supplementary Material

#### Supplementary Overview

- • Section 1: Additional Related works.
- • Section 2: Dataset details.
- • Section 3: Training settings.
- • Section 4: Qualitative examples.
- • Section 5: Additional experimental results.

#### 1. Additional Related works

Reinforcement Learning in Generative Models. The widespread success of Reinforcement Learning from Human Feedback (RLHF) in aligning large language models (LLMs) with human intent [10, 19] has inspired its application to text-to-image generation. In this context, a common strategy involves first training a reward model (RM) that learns from human judgments—either general aesthetic preferences [73] or alignment between prompts and generated images [72], followed by reinforcement learning to optimize the generative model accordingly [5]. Despite its promise, this two-stage approach faces significant limitations when applied to image editing tasks. Reward models are often brittle and challenging to design robustly [38], and they can be gamed through superficial changes that maximize reward without improving actual quality, a phenomenon known as ”reward hacking” [62]. More recently, alternative optimization frameworks like GRPO [52] have emerged as viable solutions, demonstrating effectiveness in tuning both diffusion and flow-matching based models. Extensions such as FlowGRPO [34] and DanceGRPO [74] illustrate the adaptability of these algorithms to complex generative processes, offering a more stable and fine-grained path toward aligning visual outputs with human expectations—particularly in dynamic, iterative editing scenarios where traditional methods fall short.

Benchmarking Multimodal Understanding, Generation, and Unification. Evaluating unified multimodal models (UMMs) typically involves aggregating performance across multiple specialized benchmarks, each targeting distinct capabilities. For assessing visual understanding, widely adopted benchmarks include ScienceQA [36], MMMU [79], VQA [2], GQA [22], and MM-Bench [35], all of which rely heavily on large-scale datasets with human-annotated images and labels. In contrast, our proposed UniBench introduces a novel paradigm as a VQA-style benchmark specifically designed for generated images, eliminating the dependency on real-image annotations by evaluating comprehension directly on synthesized content. For generative capability assessment, image quality is commonly mea-

sured using metrics such as FID [18], ImageReward [72], and LIQ [56], often evaluated on standard image corpora like MSCOCO [33] or LAION-5B [51]. Additional factors such as text-image alignment [17], fairness [25], and stylistic consistency [44] are also considered, drawing from benchmarks like HRS [4]. However, unified models place greater emphasis on instruction-following and coherent joint reasoning across perception and generation. As such, evaluation frameworks tailored to text-to-image synthesis, such as GenEval [14], DPG-Bench [20], and T2ICompBench++ [21], which are particularly relevant. These assess fine-grained attributes including object presence, spatial relations, counting accuracy, color fidelity, and positional reasoning [4, 9, 26]. Despite their utility, existing benchmarks are not specifically designed for the dual perceptiongeneration nature of UMMs, leaving a gap in comprehensive, integrated evaluation. To address world-knowledge grounding in image synthesis, WISE [39] was recently introduced to evaluate models’ implicit understanding of real-world constraints across domains such as food preparation, material physics, and object affordances. More recently, UniEval [30] proposes a new benchmark dedicated to unified multimodal modeling, covering a broader range of semantic, structural, and logical challenges with increased task difficulty and potential for model improvement.

#### 2. Dataset Details

RL stage data (1K). For the reinforcement learning (RL) phase, we curate a compact but highly refined dataset of 1,000 real-world photography images selected for exceptional compositional quality, visual clarity, and semantic richness. These images span diverse domains such as portrait photography, architectural shots, nature scenes, and dynamic street photography, all captured under realistic lighting and perspective conditions. In addition to these hand-picked photographs, we incorporate a specialized subset of synthetic yet photorealistic data from Echo-4o [76], which provides tightly aligned text-image pairs with expert-level captions and controlled visual variations. This combined RL dataset is used in a reconstruction-driven optimization framework: given a caption derived from one of these target images, the model is tasked with generating a new image, and its output is evaluated against the original using a learned reward model that assesses fidelity, detail preservation, and semantic alignment. Through this closed-loop paradigm, improved captioning leads to better reconstruction, which in turn refines generation capabilities.

###### Data for evaluation in Unified-Bench. To evaluate

the model’s performance on the proposed Unified-Bench, we randomly sample 100 images from the LAION-5B dataset [51] to serve as a dedicated test split. These images are selected without any filtering or curation based on content or aesthetic score, ensuring a representative and unbiased distribution across categories, styles, and complexity levels.

#### 3. Training Settings

Training details of Unified-GRPO. We employ the GRPO RL algorithm [52] to fine-tune only the LLM module while keeping other modules, like the corresponding visual encoder/decoder, frozen. We empirically observe that updating the visual encoder during RL training can lead to instability and degradation in image quality (see Fig. 9), such as anomaly artifacts, structural collapse, or semantic inconsistency, so we disable its gradient updates to preserve visual feature integrity. To enable effective sampling for RL-based image generation, we treat the combination of the diffusion decoder and a pre-trained CLIP model [49] as a unified, frozen reward module. This composite model operates purely in inference mode: given a generated image and its corresponding reconstructed image from the input caption, it computes a similarity score that serves as the final reward signal in the GRPO framework. During training, we use a learning rate of 1 × 10−6 and a batch size of 1 due to the high computational cost of diffusion-based RL. For each prompt, we generate 4 sampled images to estimate the policy gradient in GRPO, and we set the KL regularization coefficient 1×10−6, indicating that we only apply a minimal penalty for divergence from the reference policy, focusing solely on reward maximization. The temperature of LLM is set to be 1.0. The prompt used to do the LLM sampling is shown below (see Prompt. 5).

Note that we do not explicitly require the LLM to generate descriptive or comprehensive captions during training. After RL, the LLM autonomously produces longer and richer captions that are more conducive to high-fidelity image generation, even though no explicit supervision or loss is applied to the caption content itself. This emergent behavior suggests that the RL signal from image reconstruction quality implicitly guides the LLM toward generating more detailed and image-friendly textual descriptions.

#### 4. Qualitative Examples

Enhancing model’s comprehensive perception by the generation model. Fig. 11 contrasts captions used for reconstruction on a challenging example (small black dog wearing a yellow beanie and glasses). Baselines reveal three typical errors. (i) Category drift: some misidentify the subject as a monkey, causing the generator to synthesize an incorrect species. (ii) Attribute omissions or swaps: descrip-

[Figure 19]

Figure 9. Illustration of the reconstruction results when unfreezing ViT (the visual encoder of the MLLM) for joint training. We observe that the generated output collapses, semantically important details such as “two pumpkins” and “one candle” are missing. This degradation motivates us to keep the ViT frozen during finetuning across all experiments.

tions drop key items (beanie, glasses) or mismatch apparel colors, leading to reconstructions that caricature the outfit. (iii) Under-specified scenes: vague backgrounds and missing lighting cues prevent consistent photographic style at inference. UAE’s caption, in contrast, enumerates the full set of semantics—species, apparel type and color, eyewear, pose, occlusions (“ears are not visible”), background style (“blurred, park-like”), and lighting—producing a reconstruction that preserves identity, attire, and overall aesthetic. This example typifies the mechanism by which better understanding (denser, better-bound descriptions) yields better generation, echoing our Unified-Bench gains in Tab. 3.

#### 5. Additional Experimental Results

The text-to-image generation results on DPG-Bench. On DPG-Bench (Tab. 9), UAE achieves the top scores on Entity (91.43), Attribute (91.49), and Relation (92.07), and ranks second overall with 84.74, closely trailing Bagel (85.07). The sub-score pattern suggests UAE’s advantages come from faithful entity grounding and relation handling under long prompts, translating into competitive end-to-end generation quality within a unified architecture.

Prompt list used in Fig. 10. We provide the full caption for each sample in generation order, reading from left to right and top to bottom, row by row.

• Sample-1. A close-up portrait of a ginger tabby cat, its fur a rich tapestry of warm amber and deep russet stripes that catch the soft, directional light illuminating its face from the side, highlighting the velvety texture of its coat and the subtle contours of its cheekbones, while its large, luminous green eyes gaze intently off-camera with an expression of quiet contemplation and alert curiosity, framed by long, delicate white whiskers and perked ears that suggest attentiveness, all set against a dark, shadowy background that isolates the feline subject and enhances the dramatic, almost painterly quality of the image, emphasizing the cat’s regal poise and enigmatic presence.

[Figure 20]

###### Figure 10. Visualization results of UAE at 1024×1024 resolution.

[Figure 21]

Figure 11. Case study of the results from the proposed Unified-Bench, we see that our UAE enables to produce a more detailed, accurate, comprehensive description based on the input image, and reconstructs a similar result to the original image, showcasing the improved understanding and generation capabilities, and the better unification of the system.

- • Sample-2. The building on the left is a light beige color with a series of rectangular windows framed in red, some with small white panes. These windows have simple brick or mortar surrounds and are uniformly spaced, creating a rhythmic pattern across the facade. The ground floor features a small shop area with a white canopy providing shade for outdoor seating. The canopy is supported by metal poles and holds a few tables under its shelter. Behind this canopy, various items can be seen, including a few chairs and tables, indicating a caf´e or small eatery. A white umbrella stands next to the shop entrance, adding to the cozy atmosphere.Above the shop, the building has a series of small balconies with metal railings, each adorned with potted plants and hanging baskets, contributing to

the pedestrian-friendly urban design. The ground floor has a mix of business signs, some of which are partially visible but not legible, suggesting a bustling commercial area. There’s a dark green signboard affixed to one of the windows, possibly indicating a specialty shop or restaurant. The neighboring building on the right is a lighter shade of beige with a pastel green section near the top. Its windows are similarly framed in red, with larger panes and a more varied arrangement compared to the first building. This building features balconies with metal railings and small rectangular windows. The exterior walls show some wear and tear, with subtle moldings and patches of weathering, adding character to the structures. In front of these buildings lies a cobblestone street, which is partially shaded by

- Table 9. Comparisons of text-to-image generation ability on DPG-Bench [20] benchmark. Bold indicates the best result, and underlined denotes the second best.

Method Global Entity Attribute Relation Other Overall Dedicated T2I

SDXL [46] 83.27 82.43 80.91 86.76 80.41 74.65 PlayGroundv2.5 [27] 83.06 82.59 81.20 84.08 83.50 75.47 Hunyuan-DiT [31] 84.59 80.59 88.01 74.36 86.41 78.87 PixArt-Σ [6] 86.89 82.89 88.94 86.59 87.68 80.54 DALLE3 [40] 90.97 89.61 88.39 90.58 89.83 83.50 SD3-medium [1] 87.90 91.01 88.83 80.70 88.68 84.08 FLUX.1-dev [24] 82.1 89.5 88.7 91.1 89.4 84.0 OmniGen [67] 87.90 88.97 88.47 87.95 83.56 81.16

### Unified Model

Show-o [68] 79.33 75.44 78.02 84.45 60.80 67.27 EMU3 [60] 85.21 86.68 86.84 90.22 83.15 80.60 TokenFlow-XL [48] 78.72 79.22 81.29 85.22 71.20 73.38 Janus Pro [8] 86.90 88.90 89.40 89.32 89.48 84.19 BLIP3-o 4B [7] - - - - - 79.36 BLIP3-o 8B [7] - - - - - 81.60 UniWorld-V1 [32] 83.64 88.39 88.44 89.27 87.22 81.38 OmniGen2 [64] 88.81 88.83 90.18 89.37 90.27 83.57 BAGEL [11] 88.94 90.37 91.29 90.82 88.67 85.07 UAE 83.11 91.43 91.49 92.07 84.32 84.74

the shadows cast by the buildings. A large stone fountain occupies the foreground, its base circular and gray, with a worn, dark surface. The pavement around the fountain is paved with irregularly shaped stones, creating a rustic, old-world feel. The sunlight creates dramatic contrasts, with deep shadows and bright highlights accentuating the textures of the buildings and the cobblestones. The street is quiet, devoid of people, which enhances the serene and timeless atmosphere of the scene.

- • Sample-3. A photo of hearty Chinese meal.
- • Sample-4. This serene watercolor painting evokes the tranquil spirit of a traditional Chinese riverside village, where mist-laden mountains recede into a soft, pale sky, their layered silhouettes rendered in gentle washes of gray and muted green that dissolve into atmospheric haze; along the calm, reflective riverbank, white-walled houses with dark-tiled, upturned eaves nestle among lush trees, their architecture echoing classical Jiangnan aesthetics, while two slender wooden boats glide silently on the glassy water—one closer to the foreground with its simple mast

- and open cabin, the other a distant speck fading into the fog—imbuing the scene with quiet movement and timeless stillness, as the interplay of light and shadow across the rippling surface and the subtle gradations of ink suggest not only depth and distance but also a meditative harmony between nature and human habitation, capturing the essence of poetic rural life suspended in a dreamlike, almost ethereal moment.
- • Sample-5. A vibrant blue skateboard with bold, graffitistyle graphics—featuring swirling red and yellow patterns and stylized lettering—stands upright on cracked concrete, its bright red wheels and silver trucks catching the sunlight, casting a sharp shadow on the ground, while in the blurred background, a weathered wall adorned with colorful street art and a partially visible skate ramp hint at an urban skate park setting, blending raw energy with artistic expression under a clear, sunlit sky.
- • Sample-6. A solitary, gnarled tree with twisted, leafless branches stretches skyward like a skeletal sentinel in the heart of a vast desert landscape, its weathered trunk rooted

- firmly in the ochre sands that stretch to the horizon, dotted sparsely with low-lying shrubs; above, a dramatic expanse of billowing cumulus clouds drifts across a brilliant blue sky, casting shifting shadows over the arid terrain, while in the distance, the imposing silhouette of red rock mesas rises majestically against the horizon, lending a sense of ancient grandeur and timeless solitude to the scene, where nature’s raw resilience and stark beauty are captured in perfect harmony under the vast, open heavens.
- • Sample-7. A striking traditional East Asian ink painting captures the vibrant essence of a blossoming plum tree, its gnarled, darkly rendered branches—executed with bold, expressive brushstrokes of sumi ink—arching gracefully across the stark white paper to cradle clusters of vivid crimson flowers, each petal delicately shaped with fluid washes of red that convey both vitality and fragility, while subtle hints of green foliage at the lower left suggest the quiet emergence of new life; the composition balances dynamic movement with serene stillness, evoking themes of resilience and renewal as the blossoms defiantly bloom against the void, enhanced by the faint calligraphic inscription near the trunk and the small red seal in the corner, which together anchor the piece in cultural tradition and artistic intention.
- • Sample-8. In a breathtaking, sun-drenched meadow of lush rolling hills dotted with wildflowers and scattered boulders, a young boy with soft silver-gray hair and wide, awestruck blue eyes gazes upward in wonder as he gently cradles a radiant, living flame between his outstretched palms—a glowing, teardrop-shaped orb of golden-orange fire that pulses with warmth and light, its edges flickering with delicate embers against the backdrop of a brilliant blue sky streaked with fluffy white clouds and distant snow-capped mountains; dressed in a simple light-blue jacket over a crisp white shirt, the child embodies innocence and quiet awe, as if he has just summoned or discovered this mystical force, transforming the idyllic pastoral landscape into a realm where magic feels not only possible but tenderly held, evoking a sense of harmony between nature, wonder, and the boundless imagination of youth.
- • Sample-9. A vibrant, sun-drenched tropical beach unfolds under a brilliant azure sky dotted with fluffy white clouds, where the crystal-clear turquoise waters gently lap against golden sands lined with swaying palm trees casting dappled shadows on the shore, and at the heart of this serene paradise, the bold, three-dimensional white letters spelling “KEEP CALM” rise majestically from the sea’s edge, their clean, modern font contrasting with the organic beauty of nature while reinforcing the tranquil mood, as if the very landscape itself is whispering a soothing mantra of peace, relaxation, and escape from the chaos of everyday life.
- • Sample-10. A dazzling, multifaceted purple diamond rests regally upon a shimmering bed of iridescent violet sand,

its precisely cut facets catching and refracting beams of ethereal light that radiate from behind, casting a luminous glow across the scene and accentuating the gem’s deep amethyst hue with flashes of electric violet and cool silver highlights; the background dissolves into a dreamy, softly diffused gradient of lavender and indigo, enhancing the jewel’s otherworldly brilliance and making it appear almost suspended in a mystical twilight realm, where every angle of its polished surface seems to whisper secrets of rare beauty and enchanted allure, evoking both luxury and fantasy in a single, captivating moment.

- • Sample-11. In a rain-slicked, neon-drenched cyberpunk cityscape at night, a mysterious hooded figure stands silhouetted against a kaleidoscope of glowing skyscrapers and pulsating billboards, their face obscured by shadow as they hold aloft a luminous rectangular sign that boldly proclaims “UAE” in vibrant, electric-blue neon lettering, casting an otherworldly glow on their gloved hands and the wet pavement below, where reflections of magenta, cyan, and violet lights ripple across the glossy street like liquid electricity, evoking a futuristic vision of the United Arab Emirates as a nexus of technology, mystery, and urban energy under a dark, rain-streaked sky.
- • Sample-12. A cybernetic warrior stands resolute in the heart of a rain-lashed, neon-soaked metropolis, his face etched with intricate biomechanical tattoos that glow faintly under the pulsating pink and blue lights of towering holographic billboards, while his eyes are hidden behind sleek, futuristic visor goggles radiating a cool violet-blue luminescence that mirrors the city’s electric pulse; clad in a high-collared, armored black jacket accented with glowing orange circuitry along its seams, he exudes an aura of stoic intensity and technological prowess, as blurred silhouettes of passersby dissolve into the background, their forms swallowed by the misty haze and shimmering reflections on wet pavement, immersing him in a world where humanity and machine merge beneath the ceaseless drizzle and chromatic glow of a dystopian urban dreamscape.
- • Sample-13. As the sun dips below the horizon, casting a warm golden glow across the sky that fades into soft blues and purples, Shanghai’s iconic Oriental Pearl Tower stands tall and radiant, its spherical sections glowing with pink and purple hues that mirror the twilight, anchoring the city’s futuristic skyline against a backdrop of sleek glass skyscrapers and modern high-rises; below, the Huangpu River flows gently, reflecting the fading light and the silhouettes of bridges and riverside trees, while lush green foliage along the embankment frames the scene, adding a touch of nature to the urban grandeur, creating a serene yet dynamic panorama where technological marvels and natural beauty converge in perfect harmony at dusk.
- • Sample-14. Under a brooding, leaden sky that looms heavy with the promise of storm, a colossal wave rises

in furious majesty—its dark, churning body sculpted by unseen winds into a towering, curling crest that crashes forward in a froth of white foam and spray, its deep indigo and slate-gray depths hinting at the ocean’s raw, untamed power; above the tumult, a scattered flock of seabirds soars with outstretched wings, their silhouettes stark against the gloom as they ride the turbulent air currents, embodying both freedom and resilience amid nature’s overwhelming force, while the horizon vanishes beneath the swell, leaving only the primal drama of sea and sky locked in eternal, awe-inspiring conflict.

- • Sample-15. The image showcases a delectable pepperoni pizza presented on a rustic wooden board, set against a dark, textured background that adds a touch of sophistication. The pizza boasts a golden-brown crust with visible char marks from being cooked in a wood-fired oven, indicating a crispy texture. The cheese, melted and slightly browned in spots, blankets the pizza evenly, with some areas showcasing a rich, gooey appearance. The toppings are predominantly pepperoni slices, arranged in a somewhat circular pattern around the edges, while others lie scattered across the surface in various orientations. Each slice of pepperoni is glossy, indicating a fresh, juicy texture, and they are generously placed, making the pizza look hearty and appetizing. Interspersed among the pepperoni slices are small flecks of herbs, likely basil, adding a burst of green color and freshness to the dish. To the right side of the pizza, two fresh basil leaves are artistically placed, their vibrant green hues contrasting beautifully against the warm tones of the pizza and the wooden board. A few more basil leaves can be seen in the foreground at the bottom left corner, scattered more casually than the ones on the pizza itself. There are also a couple of slices of pepperoni lying outside the pizza, further enhancing the visual appeal of the presentation. The overall composition of the image is balanced, with the pizza centrally located, drawing the viewer’s attention immediately. The lighting is subtle yet adequate to highlight the textures and colors of the pizza, making it look inviting and mouth-watering. The slight shadows cast by the pizza and basil leaves add depth to the image, creating a three-dimensional feel.
- • Sample-16. The image depicts a serene night scene at a lively port town. The sky is filled with a bright starry Milky Way galaxy, casting a soft glow over the entire scene. The town features quaint, charming houses with warm yellow lights emanating from their windows, creating a cozy ambiance. At the forefront, there is a group of people gathered around wooden tables, enjoying their time together. They are engaged in conversation and laughter, with cups of coffee or tea in hand. A golden retriever dog sits by one of the tables, adding to the homely atmosphere. To the right, there is a tall streetlight and a small flower arrangement in a pot, further enhancing the quaint charm

of the setting. In the background, a harbor is visible with boats anchored, and the town extends with more houses and shops lining the streets, including a bakery sign.

- • Sample-17. From a high vantage point, the sun rises—or sets—in a blaze of golden-orange light that pierces through a dramatic sky streaked with soft pink, lavender, and deep blue clouds, casting long, ethereal shadows across a vast, snow-blanketed landscape of rolling hills and undulating valleys where a winding road snakes like a ribbon through the serene white expanse; frost-kissed shrubs dot the foreground, their dark branches dusted with snow and catching the warm glow, while the distant horizon fades into a hazy, dreamlike mist, blending earth and sky in a tranquil, almost otherworldly winter tableau that evokes both solitude and sublime beauty beneath the celestial spectacle of dawn or dusk.
- • Sample-18. The image captures the majestic Forbidden City in Beijing, China, bathed in the warm hues of a setting sun. The scene is dominated by several large, traditional Chinese buildings with elegant, ornate roofs painted in vibrant reds and golds. These buildings feature numerous golden dragons and intricate carvings, typical of imperial architecture. The main structure in the center is an imposing palace with multiple eaves and large golden pillars, its entrance flanked by smaller pavilions. The central building’s roof is adorned with intricate patterns and two large, pointed gables, adding to its grandeur. In front of the palace, a wide, open courtyard stretches out, paved with smooth, light-colored stones and bordered by white stone balustrades. These balustrades are decorated with sculpted figures and floral designs, providing a stark contrast to the dark stone of the buildings behind them. The courtyard is devoid of people, emphasizing the serene and historical atmosphere of the site. To the left, more buildings can be seen, each with their own distinct architectural features, though slightly obscured due to the architectural layout. The sky above is a soft gradient from pale blue at the horizon to a warm orange near the sun, which casts a gentle glow over the entire scene. A few wispy clouds are scattered across the sky, adding depth and dimension to the panoramic view. In the foreground, there is a series of white, stone railings and steps leading up to the palace, guiding the viewer’s eye towards the impressive structure. The entire area is bathed in the soft, golden light of the sunset, creating a peaceful and timeless quality that highlights the historical significance of this famous landmark.
- • Sample-19. In this serene, sunset-hued beach scene, a woman stands with her back to the viewer, gazing out at the ocean. She has long brown hair tied loosely behind her head and wears a flowing white sleeveless dress that reaches her ankles. She carries a pair of black flip-flops in her right hand. Her light brown and white dog sits attentively beside her on the sandy shore, their brown and

white fur contrasting with the warm, golden tones of the setting sun. The beach is bathed in the soft, orange glow of the setting sun, casting long shadows and highlighting the texture of the sand. In the distance, the gentle waves roll onto the shore, with the sun’s reflection shimmering on the water. To the left, a sailboat sails across the calm sea, its silhouette silhouetted against the warm sky. A wooden lifeguard chair with a red life buoy stands near the center-right of the scene, next to a blanket with a floral pattern draped over its legs. The beach is dotted with footprints, and tall grasses and shrubs frame the scene. A couple of seagulls fly low in the orange sky, adding to the tranquil atmosphere. In the background, a cliff rises, partially obscuring the view, and a few more sailboats are visible on the horizon.

- • Sample-20. An ancient Greek philosopher is talking on a wireless headset.
- • Sample-21. A serene elven woman with pointed ears and intricate silver face art gazes thoughtfully, clad in a dark green gown with gold trim. She stands in a mystical, moonlit forest where glowing blue mushrooms illuminate the shadowy trees around her.
- • Sample-22. The image depicts a small, well-lit home office setup in a cozy room with beige carpeting. The primary focus is a compact wooden desk positioned against a pale wall. The desk has a simple, light-colored finish and is supported by two metal legs, which appear to be adjustable for height. On the desk, there is a black keyboard and a laptop computer on the right side, along with a closed, black-framed flat-screen monitor to the left of the laptop. A white mouse and a pair of sunglasses rest on the keyboard. A single table lamp with a black shade stands next to the keyboard, casting a warm light over the workspace. To the left of the lamp, a small stack of books or papers rests on the desk surface. A black rolling chair with height-adjustable arms is stationed directly in front of the desk. The chair’s wheels are visible, indicating its portability. The computer monitor is accompanied by a webcam mounted above it on the wall. Below the desk, the floor is partially covered with a light-colored rug that contrasts with the carpeting. Adjacent to the desk, there is a potted plant with lush green leaves placed on a small round table or stand. The room’s background features a bookshelf filled with various books, some of which are visible through open shelves. A white cushioned armchair sits to the left of the desk, suggesting a cozy nook for relaxation or additional seating. On the wall behind the desk, near the corner, a rectangular mirror reflects part of the room, adding depth to the space. An overhead lighting fixture casts a soft yellow glow from above, highlighting the desk area while keeping the rest of the room dimly lit. The overall color palette includes neutral tones—beige, white, and shades of brown—creating a calming and functional

workspace atmosphere.

- • Sample-23. In a vibrant, arid desert landscape bathed in warm, golden hues of sunset, a group of three individuals ventures through a rugged, canyon-like terrain. The woman at the center, dressed in a practical olive-green safari outfit with rolled-up sleeves, khaki pants, and a belt bag slung over her shoulder, walks confidently towards the camera. Her dark hair is tied up in a bun, and she has a focused expression on her face as she gazes at the ground. A small, playful fox stands beside her, attentively looking ahead. The woman’s right hand holds a stainless steel water bottle, and her left arm is relaxed by her side. On the right, a man wearing a wide-brimmed straw hat, beige shirt, and cargo pants stands observing the surroundings, while his young son, dressed in an orange t-shirt and black shorts, looks back at them with a curious expression. The man and his son are positioned slightly behind the woman, who appears to be leading the way. In the foreground, a cactus plant with a yellow bloom adds to the desert ambiance. The background features towering red rock formations and sparse vegetation, including a few Joshua trees and desert scrub. A large eagle soars high above, its wings spread wide against the backdrop of a sky painted with swirling clouds in shades of orange, pink, and purple. The sand beneath their feet is dotted with footprints, suggesting they have been walking for some time. The entire scene is imbued with a sense of adventure and exploration, set against the timeless beauty of a desert canyon under a dramatic sunset sky.
- • Sample-24. Please generate a realistic image of the traditional Chinese Hotan Jade pendant. The pendant is a round jade brand, with a full color of turquoise. The jade is warm and delicate, and the surface is highly polished but not excessively reflective, showing the oily texture of real jade. A traditional Jiangnan garden landscape painting is carved in relief on the jade plaque: the upper half of the picture shows a group of Chinese style buildings arranged in a staggered manner, with roofs featuring upturned eaves and horsehead walls, and rich details. The buildings are interspersed with delicate elements such as small bridges, flowing water, weeping willows, and rockeries. The overall composition is complex but not messy, presenting a freehand feeling of traditional Chinese painting style. The lower part of the screen is relatively blank, with only winding rivers flowing from bottom to right, enhancing the spatial hierarchy. The pendant is hung on a gray green Chinese woven rope, with a simple and natural knot, tightly woven from multiple strands of fine thread, with a tough texture. It is decorated with a small red coral bead directly above it. The background of the picture is a light gray cloth in the style of physical photography as a reference, which is overall realistic and realistic. The style is modern high-quality still life photography, with clear

composition, soft lighting, and focus on the center of the jade plaque, blurring the background details.

- • Sample-25. A portrait of profound wisdom and quiet contemplation, this elderly man with a long, flowing white beard and deeply lined face is captured in dramatic chiaroscuro lighting against a dark void, his gaze fixed on something unseen beyond the frame.

More “Image-Text-Image” reconstruction results generated by our method. Here, we provide more visual examples of the “image-text-image” reconstruction pipeline using our method, i.e., our encoder processes the input image, generates the output descriptive caption, and then passes it through our decoder to recover it to pixels. See Fig. 12, Fig. 13, and Fig. 14 for details.

Scaling unified-bench for testing. We have expanded the evaluation by increasing the number of source images to 2000. The corresponding results are reported in Tab. 10, validating the effectiveness of our method.

Method Average CLIP LongCLIP DINO-v2 DINO-v3

Bagel 80.95 87.57 92.88 80.11 63.24 GPT-Image* 83.69 91.37 92.81 86.41 64.15 Ours 84.74 90.22 94.49 85.72 68.54

- Table 10. ‘*’ indicates the latest GPT-Image version used for comparison.

A deeper analysis of the performance drop on OCR/DU would be beneficial. We attribute the performance drop on OCR/DU to decoder-side text-rendering bottleneck, rather than to the limitations of our method.

In our setup, the decoder is instantiated with SD3, whose limited text generation capability yields a noisy supervision signal for the understanding model, thereby degrading OCR/DU performance. To verify, we replace SD3 with Qwen-Image (with stronger text rendering). As shown in Tab. 11, the performance drop completely disappears.

Decoder OCR DU Enc. Only 75.0 70.0 SD3 68.8 58.4 Qwen-Image 75.9 70.5

Table 11. Decoder choices for better OCR understanding.

MSE as the reward model. We train the unified system under an image-text-image (I2T2I) pipeline and use text as an intermediate representation; only semantic content is preserved. The CLIP model naturally captures this semantic similarity. Instead, MSE operates at the pixel level and penalizes low-level variations (lighting, texture, style) that do not affect semantics. Given this reason, our main paper utilizes the semantic encoder (CLIP) as the reward model, rather than pixel-level supervision.

The required T2I model/capability at the beginning of training. The T2I model is expected to possess basic long-instruction following capability, as our reconstructive RL process progressively produces more detailed intermediate captions. If the T2I model or text encoder is constrained by short token limits (e.g., CLIP-style 77-token encoders), it can bottleneck reconstruction quality and thus limit effective RL optimization. Importantly, this assumption is not restrictive in practice, as modern T2I foundations (e.g., Qwen/Longcat-Image) already show strong capacity for handling long and structured textual prompts, and thus naturally satisfy this and make our method broadly applicable.

Limitations and future works. In our experiment, we observe an unexpected decrease in understanding performance on visual-text recognition related tasks (Tab. 7 in the main paper), with accuracy dropping by approximately 10% on Document Understanding (DU) and OCR. This likely stems from the well-known difficulty of current image generation models in faithfully rendering text [50], which may introduce misleading reconstruction rewards during RL and hinder the encoder’s ability to learn reliable text semantics. This limitation means that the overall benefits of our framework are currently constrained by the weaknesses of the generation module. Improving text reconstruction fidelity is therefore an important direction for future work. Our framework naturally extends beyond the image, as it treats text as a universal representation; the same reconstructive principle could be applied to other modalities such as audio and video.

[Figure 22]

[Figure 23]

[Figure 24]

###### Prompt for the LLM Sampling

System Prompt: You are an expert vision-language model.

User Prompt: Your task is: Given an input image, generate a textual description of the image. If there is text in the image, transcribe it inside double quotes.

Now, carefully analyze the input image and output the full description.

Input Image: {{image path}}

Prompt Used to Perform LLM Judge for Caption Quality User Prompt:

You will conduct a multi-dimensional analysis of each caption based on the specific criteria listed below. For each criterion, you will assign a score from 1 (very poor) to 10 (excellent). After scoring, you must provide a detailed, structured comparative analysis and declare a final winner.

###### Evaluation Criteria & Scoring:

Please evaluate each caption against the following four criteria. Provide your scores in a markdown table.

###### 1. Comprehensiveness, Descriptive Richness, and Accuracy:

- • How deeply does the caption describe the image? Does it go beyond a superficial glance to include important, specific details (e.g., colors, textures, materials, lighting, background elements, expressions)?
- • Does it effectively and accurately describe the context (e.g., a black dog not a monkey, or brown eyes not black), environment (background description)?
- • Does the caption capture subtle nuances that a casual observer might miss?

###### 2. Linguistic Fluency and Naturalness:

- • Is the caption grammatically correct and wellwritten in natural-sounding English?
- • Does it flow like a human would describe the scene, or does it sound robotic, disjointed, or like a list of keywords?
- • Is the vocabulary choice sophisticated, appropriate, and engaging?

###### 3. Semantic and Compositional Insight:

- • Does it effectively capture and convey the overall mood, atmosphere, emotion, or narrative implied by the scene?
- • Does it demonstrate an understanding of the image’s composition (e.g., what is in the foreground vs. background)?

Based on the above rules, provide a comprehensive, head-to-head comparison of the two captions. Structure your analysis with subheadings for each of the four criteria. For each criterion, explicitly quote phrases from both Caption A and Caption B to illustrate your points and justify the difference in their scores. Explain not just what is different, but why one caption’s approach is superior for describing the provided image.

Finally, please declare the winner based on your detailed comparative analysis above. This section must contain only a single letter.

###### Final Answer: [A or B].

