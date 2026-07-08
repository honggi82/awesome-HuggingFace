# arXiv:2606.24849v1[cs.CV]23Jun2026

## IV-CoT: Implicit Visual Chain-of-Thought for Structure-Aware Text-to-Image Generation

### Zixuan Li1*, Haokun Lin1†, Yicheng Xiao3, Zhiwei Li1, Xinyang Song1, Zelong Zheng1, Yong He2, Heng Yao2, Ke Ding2, Chao Yu2, Chuan Yuan2†, Qi Li1†, Zhenan Sun1

1NLPR, Institute of Automation, Chinese Academy of Sciences 2Ant Group 3The University of Hong Kong

zixuan.li@nlpr.ia.ac.cn

### Abstract

Unified multi-modal large language models (MLLMs) have achieved strong text-to-image generation quality, but still struggle with structure-aware prompt following, where object counts, spatial relations, attribute bindings, and coarse layouts must be preserved. We attribute this limitation in part to the entanglement of structural planning and appearance rendering within a single conditioning stream. To address this issue, we propose Implicit Visual Chainof-Thought (IV-CoT), a latent visual reasoning framework for query-conditioned image generation. IV-CoT decomposes the visual conditioning queries into a structural-to-semantic cascade, where structural queries first form a latent visual plan and semantic queries then render appearance conditioned on this plan. To guide the structural queries, we introduce training-only sketch supervision, which encourages them to capture structure from sketches without requiring sketch extraction or intermediate decoding at inference time. IV-CoT performs implicit CoT reasoning in a single forward pass and achieves superior results on GenEval and T2I-CompBench. Visualizations and analyses demonstrate that the learned structural and semantic queries play complementary roles in structure-aware generation.

### 1 Introduction

Recent unified multi-modal large language models (MLLMs) have shown strong capabilities in generating realistic images from open-ended instructions (Zhou et al., 2025a; Wu et al., 2025a; Ma et al., 2025; Cui et al., 2025; Xiao et al., 2025; Xie et al., 2026). However, they still struggle with prompts that impose complex structural requirements (Huang et al., 2023; Ghosh et al., 2023; Zhang et al., 2025a; Jiang et al., 2025). When

*This work was done during the internship at Ant Group. †Corresponding author.

a prompt specifies multiple objects with distinct shapes, materials, attributes, and spatial arrangements, the model may produce visually plausible images while swapping attributes, omitting objects, or violating the requested layout. We refer to this setting as structure-aware prompt following; an example is illustrated in Figure 1.

Most unified MLLM-based generators convert the prompt into a visual conditioning stream through an understanding MLLM, where scene structure, object identity, attributes, and appearance details are compressed together (Pan et al., 2025; Wu et al., 2025b). Such entangled conditioning makes it difficult for the generator to distinguish what should determine the scene structure from what should control visual appearance. Recent works therefore explore Chain-of-Thought (CoT) reasoning for image generation, using intermediate reasoning steps to better handle complex scenes (Guo et al., 2025b; Wang et al., 2025).

Existing CoT-based generation methods mainly follow two explicit paradigms. Explicit Textual CoT generates intermediate verbal reasoning, scene descriptions, or numerical layouts before image synthesis (Deng et al., 2025; Jiang et al., 2026; Tian et al., 2026). However, language alone has limited spatial bandwidth for representing continuous 2D geometry, object boundaries, relative scale, and occlusion. Explicit Interleaved CoT incorporates visual intermediate states, such as masks, layouts, or draft images, to provide stronger structural cues (Guo et al., 2025a; Jiang et al., 2025; Qin et al., 2025). Yet this comes at the cost of explicit intermediate decoding, multi-stage pipelines, and potential error accumulation. These complementary limitations suggest that structure-aware generation needs an intermediate planning mechanism that preserves visual-spatial information while avoiding decoded reasoning states at inference time (Figure 1).

[Figure 1]

- Figure 1: Comparison of reasoning paradigms for text-to-image generation. (Left): direct generation, explicit textual CoT, explicit interleaved CoT, and IV-CoT differ in where intermediate reasoning states are represented. (Right): a structure-aware prompt-following example showing that IV-CoT better preserves the requested object layout without explicitly decoding intermediate images.

Latent reasoning provides a natural alternative, where intermediate deliberation is carried by hidden states or soft thought tokens rather than explicit outputs (Hao et al., 2024; Xu et al., 2025a,b; Ramji et al., 2026). Although related ideas have begun to appear in multimodal reasoning (Pham and Ngo, 2025; Chen et al., 2025a), latent reasoning for textto-image generation remains underexplored. The key question is: how should such latent states be organized before image rendering? We argue for a structure-first organization: the latent state should first anchor object boundaries, layout, and coarse shape, and then guide semantic appearance rendering. Without this structural scaffold, the generator may bind attributes to wrong objects, misplace spatial relations, or produce plausible details on incorrect structures.

We propose Implicit Visual Chain-of-Thought (IV-CoT), a structure-first latent reasoning framework in the query space of a unified MLLM-DiT generator. IV-CoT internalizes structural planning into a causal structural-to-semantic query cascade: structural queries are placed before semantic queries and first encode a latent visual plan, including object form, count, layout, and coarse spatial relations; semantic queries then attend to this plan to render appearance and fine-grained details. In this way, the “chain” is realized as an ordered latent dependency from structural planning to semantic rendering, rather than an external sequence of text, layouts, or decoded images.

We further use sketches as training-only structural guidance through a two-stage training scheme. Unlike sketch- or layout-conditioned generation methods that require external spatial controls at inference time, IV-CoT uses sketches only to shape

latent structural queries during training. In the first stage, sketch supervision encourages structural queries to encode contours, shapes, counts, and layouts while suppressing appearance factors such as color, texture, and lighting. In the second stage, IV-CoT is optimized for image generation with the structural objective retained as a regularizer, keeping structural queries aligned with the latent visual plan while semantic queries learn to render appearance details. At inference time, IVCoT takes only the text prompt and produces structural and semantic queries in a single forward pass, without extracting sketches, decoding intermediate images, or generating explicit reasoning traces. Our contributions are summarized as follows:

- • We formulate Implicit Visual Chain-of-Thought, where intermediate visual planning is internalized in latent query representations rather than externalized as explicit textual or visual intermediate states.
- • We instantiate this formulation with a causal structural-to-semantic query cascade and training-only sketch supervision, which shape structural queries into latent visual plans while preserving text-only, single-pass inference.
- • Using the same OpenUni-L-1024 backbone, IVCoT improves GenEval from 0.86 to 0.88 and T2I-CompBench from 0.5448 to 0.5743. Meanwhile, IV-CoT keeps single-pass inference and achieves 9-15× lower latency than explicit CoT methods. Visualizations and cross-prompt recombination show that structural queries encode recoverable and manipulable latent visual plans. The relevant code will be released upon acceptance of the paper.

### 2 Method

We propose Implicit Visual Chain-of-Thought (IV-CoT), a structure-first latent reasoning framework built upon a query-conditioned MLLM-DiT generator. Motivated by the difficulty of generation models in structure-aware prompt following (Figure 1), IV-CoT introduces an ordered structural-tosemantic query cascade that first encodes structural information and then performs semantic refinement. To enable such latent reasoning, we train the structural query inputs Q0s with a sketch-supervised objective and optimize image generation with structural regularization. This section first reviews the backbone architecture, followed by the proposed components and inference procedure.

- 2.1 Query-Conditioned MLLM-DiT Generation

We build IV-CoT on a unified MLLM-DiT generation architecture (Pan et al., 2025; Wu et al., 2025b). Given a text prompt y and a set of learnable visual query inputs Q0 ∈ RN×d, the MLLM produces continuous queries:

Q = ΦMLLM(y,Q0), (1)

where Q ∈ RN×d denotes the query states used for generation conditioning. In practice, a lightweight connector maps these query states into the conditioning space of the DiT; for simplicity, we use Q to denote the resulting conditioning sequence. The DiT conditions on Q to iteratively denoise a noisy image latent zt and recover a clean image latent zx, which is decoded into the output image.

In this formulation, visual information is compressed into a single flat query sequence, where structure-related factors, such as layout, shape and attributes, are entangled with appearance-related factors, such as color and texture. This provides the DiT generator with no explicit separation between structural planning and appearance rendering, leading to structure-aware prompt-following errors, as shown in Figure 3. IV-CoT addresses this limitation by assigning distinct roles to query groups and enforcing an ordered dependency from structure to semantics.

- 2.2 Structural-to-Semantic Query Cascade

IV-CoT partitions the learnable visual query inputs into two ordered groups: structural query inputs Q0s ∈ RNs×d and semantic query inputs

Q0m ∈ RNm×d. Given a text prompt y, we feed the MLLM with the ordered sequence

##### [y,Q0s,Q0m]. (2)

The MLLM outputs are then divided into structural and semantic conditioning queries:

Qs = Φs(y,Q0s), Qm = Φm(y,Qs,Q0m).

(3) Since the MLLM adopts causal self-attention, this ordering induces a one-way dependency between the two query groups. The structural queries are computed from the prompt and structural query inputs, without access to the semantic query inputs. In contrast, the semantic queries are placed after the structural queries and can therefore attend to both the prompt and the structural query states. This design establishes a structural-to-semantic cascade, encouraging the model to first form a latent visual plan and then render semantic appearance conditioned on it. The final conditioning sequence passed to the diffusion generator is

QIV-CoT = [Qs,Qm]. (4) 2.3 Sketch-Supervised Structural Constraint

As shown in Figure 2(a), IV-CoT first uses a sketchsupervised structural constraint to guide the structural queries toward visual planning. Given a target image x, we extract its sketch s with a fixed PiDiNet edge detector (Su et al., 2021) and encode it using the pretrained VAE encoder E:

zs = E(s). (5)

Since sketches suppress appearance factors such as color, texture, and lighting while preserving contours, object shapes, counts, and coarse layouts, the sketch latent zs defines a structure-focused clean latent for diffusion training.

Frozen-generator structural training. We feed the ordered sequence [y,Q0s] into the MLLM to obtain structural queries Qs, and use them to condition the DiT for sketch-latent denoising. Since zs is encoded by the diffusion VAE, it lies in the same latent space as image latents and can be used as the clean latent for diffusion training. During this stage, both the MLLM and DiT are frozen, and only the structural query inputs Q0s are optimized:

##### Lstruct = Ezs,t,ϵ ∥ϵ − ϵθ(zs,t,t,Qs)∥22 , (6)

Final image

[Figure 2]

Structural query

[Figure 3]

Prompt y

[Figure 4]

latent visual plan

DiT

MLLM

conditioned on

[Figure 5]

Semantic query

[Figure 6]

Query visuals are illustrative; only the final image is decoded at inference.

appearance details

###### [ , ]

(b) Inference:

concatenate

latent structural-to-semantic reasoning

[Figure 7]

Prompt y

Prompt y

[Figure 8]

MLLM

###### MLLM

[Figure 9]

[Figure 10]

DiT

###### [ , ]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

concatenate

[Figure 15]

[Figure 16]

DiT

[Figure 17]

[Figure 18]

[Figure 19]

DiT

(a) Training

structrural query semantic query gradient updates

- Figure 2: Illustration of IV-CoT for text-to-image generation. (a) During training, IV-CoT learns latent structural and semantic queries with structural and image-generation supervision. (b) During inference, the MLLM performs latent structural-to-semantic reasoning, and the resulting queries guide the DiT to generate the final image. No intermediate images are explicitly decoded; query visuals are shown only for illustration.

where zs,t denotes the noised sketch latent at diffusion step t. This frozen-generator design forces the structure required for sketch denoising to be encoded in Qs, rather than absorbed by adapting the MLLM or DiT.

2.4 Semantic Rendering with Structural Regularization

After structural training, we optimize IV-CoT for image generation. Given the ordered query sequence [y,Q0s,Q0m], the MLLM produces structural and semantic queries [Qs,Qm], which are concatenated as the conditioning sequence for the diffusion generator. The image-generation objective is

Limg = Ezx,t,ϵ ∥ϵ − ϵθ(zx,t,t,[Qs,Qm])∥22 ,

(7) where zx = E(x) denotes the clean image latent and zx,t is its noised version at diffusion step t.

To keep Qs aligned with the sketch-induced visual plan, we retain the structural loss Lstruct as a regularizer during image-generation training. Unlike the first stage, where the generator is frozen to shape Q0s, this stage optimizes the diffusion generator together with the query inputs. Gradients from Limg update both structural and semantic query inputs, whereas Lstruct mainly regularizes the struc-

tural branch. This prevents Qs from drifting toward appearance-only cues, while allowing Qm to complete identity, color, material, and texture details conditioned on the structural queries. The final objective is

Ltotal = Limg + λLstruct, (8) where λ controls the regularization strength. 2.5 Inference

As presented in Figure 2 (b), during inference, IVCoT uses the same full ordered query inputs as in semantic rendering, [y,Q0s,Q0m], and produces structural and semantic queries in one MLLM pass:

##### [Qs,Qm] = ΦMLLM([y,Q0s,Q0m]). (9)

No sketch is extracted or decoded. The DiT then performs standard diffusion decoding conditioned on the combined queries:

##### xˆ = ΦDiT(zT,[Qs,Qm]). (10)

Thus, IV-CoT introduces no explicit intermediate image generation, no reward-guided test-time search, and no additional visual decoding stage. The structural plan remains internal to the query sequence, while the visible output is produced by

the usual image generation process. Notably, this differentiates IV-CoT from explicit textual or interleaved CoT methods, such as T2I-R1, GoT-R1, and TWIG, which typically require multiple reasoning steps. In contrast, IV-CoT achieves higher efficiency, as evidenced by the results in Table 2.

### 3 Experiment

In this section, we conduct comprehensive experiments to answer the following Research Questions (RQs):

- RQ1: Does IV-CoT improve compositional and structure-aware prompt following while preserving inference efficiency?
- RQ2: Are structural supervision and query cascade both necessary?
- RQ3: Do structural queries encode recoverable visual plans and actively influence generated structure?
- RQ4: Does query separation enable zero-shot structure-appearance recombination?

#### 3.1 Setup

We instantiate IV-CoT on OpenUni-L to isolate the effect of the proposed query organization and structural supervision. We fine-tune IV-CoT on BLIP3-o (Chen et al., 2025c), ShareGPT-4o (Chen

- et al., 2025d), and Echo-4o (Ye et al., 2025). During structural training, we extract sketches from training images using a fixed PiDiNet edge detector as training-only supervision. No sketch is required at inference time. Implementation details are provided in Appendix A.

Baselines. We compare IV-CoT with two groups of methods. The first group includes unified multimodal generation models, such as JanusPro (Chen et al., 2025e), Emu3 (Wang et al., 2024), Show-o (Chen et al., 2026), MetaQuery-XL (Pan

- et al., 2025), OpenUni-L-1024 (Wu et al., 2025b), BAGEL (Deng et al., 2025), and TUNA-2 (Liu
- et al., 2026). The second group includes unified generation models with explicit CoT or reasoningenhanced generation, including T2I-R1 (Jiang

- et al., 2026), GoT-R1 (Duan et al., 2025), TWIGRL (Guo et al., 2025a), Uni-CoT (Qin et al., 2025), and Draco (Jiang et al., 2025). We instantiate IVCoT on OpenUni-L-1024 to isolate design effect.

Evaluation Benchmarks. We conduct the main evaluation on two widely used text-to-image gen-

eration benchmarks, GenEval (Ghosh et al., 2023) and T2I-CompBench (Huang et al., 2023).

#### 3.2 Main Results (RQ1)

Performance. Table 1 reports the main results on GenEval and T2I-CompBench. Compared with OpenUni-L-1024, IV-CoT improves the overall score from 0.86 to 0.88 on GenEval and from

- 0.5448 to 0.5743 on T2I-CompBench. The gains are particularly evident on structure-sensitive dimensions, including position and color attribution in GenEval, as well as spatial relation, shape, texture, and color in T2I-CompBench. These results support the effectiveness of our structurefirst, semantic-second pipeline for structure-aware prompt following. Compared with recent unified generation models and explicit CoT-based generation methods, IV-CoT achieves the best overall scores on both benchmarks. Notably, IV-CoT performs inference in a single forward pass, without decoded intermediate image or test-time reasoning.

Generation Samples. We provide a qualitative comparison with the OpenUni on structure-aware prompts in Figure 3. The baseline often generates visually plausible images but fails to satisfy key structural constraints, such as object count, spatial placement, and attribute binding. In contrast, IVCoT better preserves the specified visual organization while maintaining comparable image quality, further demonstrating the effectiveness of latent visual reasoning for structure-aware generation. Additional generation samples are provided in Appendix B.

Inference efficiency. We further compare inference efficiency with explicit CoT-based generation methods in Table 2. IV-CoT achieves the highest T2I-CompBench overall score while requiring only

- 1.693 seconds per sample, which is 9.01×, 9.89×, and 14.98× lower latency than T2I-R1, GoT-R1, and TWIG-RL, respectively. Details of latency measurement are provided in Appendix C.

#### 3.3 Ablation Study (RQ2)

RQ2 studies whether the gains of IV-CoT arise from its structural-to-semantic design rather than simpler alternatives. All ablation variants use the same training data, training budget, and inference settings. Base denotes OpenUni-L-1024, and Base + More Queries matches the number of visual queries used by IV-CoT to test the effect of query

GenEval T2I-CompBench Single Obj.

Method

Two Obj.

Counting

Colors Position

Color Attri.

Overall↑

Color Shape TextureSpatial NonSpatial

Complex

Overall↑

Unified models Janus-Pro-7B 0.98 0.85 0.56 0.89 0.77 0.64 0.78 0.6359 0.3528 0.7243 0.3378 0.3085 0.3559 0.4525 Emu3 0.98 0.71 0.34 0.81 0.17 0.21 0.54 0.7544 0.5706 0.7164 - - - Show-o 0.95 0.52 0.49 0.82 0.11 0.28 0.68 0.5600 0.4100 0.4600 0.2000 0.3000 0.2900 0.3700 MetaQuery-XL - - - - - - 0.80 - - - - - - OpenUni-L-1024 0.99 0.92 0.76 0.91 0.82 0.77 0.86 0.8216 0.6117 0.7308 0.4023 0.3102 0.3923 0.5448 BAGEL 0.99 0.94 0.81 0.88 0.64 0.63 0.82 0.8027 0.5685 0.7021 0.3488 0.3101 0.3824 0.5191 TUNA-2 0.99 0.96 0.80 0.91 0.84 0.76 0.87 - - - - - - -

Unified models with explicit CoT T2I-R1∗ 0.99 0.91 0.53 0.91 0.76 0.65 0.79 0.8130 0.5852 0.7243 0.3378 0.3090 0.3993 0.5281 GoT-R1∗ 0.99 0.94 0.50 0.90 0.46 0.68 0.75 0.8139 0.5549 0.7339 0.3306 0.3169 0.3944 0.5241 TWIG-RL∗ - - - - - - - 0.8249 0.6128 0.7319 0.3406 0.3199 0.5445 0.5624 Uni-CoT 0.99 0.96 0.84 0.92 0.57 0.71 0.83 - - - - - - Draco 1.00 0.99 0.81 0.91 0.70 0.76 0.86 - - - - - - -

IV-CoT (Ours) 1.00 0.96 0.78 0.92 0.86 0.79 0.88 0.8542 0.6296 0.7550 0.4734 0.3199 0.4136 0.5743

- Table 1: Model performance comparison on GenEval and T2I-CompBench. Best results are in bold and second-best results are underlined. Methods marked with ∗ use prompts from the T2I-CompBench training split during training.

Method Params Time↓ Rel.↓ T2I-Comp↑

T2I-R1 7B 15.261 9.01× 0.5281 GoT-R1 7B 16.744 9.89× 0.5241

TWIG-RL 7B 25.362 14.98× 0.5624 IV-CoT (Ours) 3.6B 1.693 1.00× 0.5743

- Table 2: Efficiency comparison with explicit CoT generation methods. Rel. denotes latency relative to IV-CoT.

pervision and the structural-to-semantic cascade are complementary: structural supervision shapes Qs into a latent visual plan, while the cascade allows Qm to render appearance conditioned on it.

3.4 Opening the Black Box: Interpreting Latent Visual Plans (RQ3)

capacity. Flat Sketch Aux applies the same sketchbased structural supervision to a flat query set, while Parallel Two-Query separates structural and semantic queries but generates them independently before concatenation. These variants isolate the effect of the ordered query cascade. IV-CoT w/o Structural Constraint keeps the cascade but removes sketch supervision on Qs, whereas Full IVCoT uses both sketch-supervised structural training and the structural-to-semantic query cascade.

Table 3 shows that Full IV-CoT performs best among all controlled variants. Increasing the number of queries yields only limited gains, indicating that the improvement is not simply due to larger query capacity. Flat Sketch Aux and Parallel TwoQuery remain below Full IV-CoT, suggesting that neither sketch supervision nor query partition alone fully explains the gain. Removing the structural constraint also weakens performance, confirming the need to explicitly guide Qs toward structural planning. These results indicate that structural su-

RQ3 examines whether structural queries encode latent visual plans and whether the generator uses structural and semantic queries in different ways.

Query decoding and perturbation. We first examine whether structural queries encode recoverable visual plans in Figure 4. We use Qs as the conditioning input to the DiT and visualize the corresponding sketch-domain generations. The results exhibit sketch-like structures that capture object shapes and scene layouts. In contrast, replacing Qs with randomly initialized queries before diffusion decoding substantially disrupts object layout, contours, and coarse shape. This suggests that Qs is actively used as a structural condition for image synthesis, rather than merely satisfying the sketch auxiliary loss. These visualizations are used only for analysis; IV-CoT does not decode intermediate sketches during actual inference.

Cross-attention proportion analysis. We further analyze how the diffusion generator allocates

[Figure 20]

- Figure 3: Qualitative comparison on structure-aware prompts. Each column corresponds to one prompt, with highlighted words indicating the target structural or attribute constraint. IV-CoT better maintains object count, spatial arrangement, attribute binding, and coarse object geometry while preserving visual quality.

Method #Queries Query Split Structural Supervision Cascade GenEval T2I-CompBench

Base 256 0.860 0.5448 Base + More Queries 512 0.869 0.5507 Flat Sketch Aux 512 ✓ 0.866 0.5496 Parallel Two-Query 512 ✓ ✓ 0.869 0.5562 IV-CoT w/o Structural Constraint 512 ✓ ✓ 0.861 0.5573 Full IV-CoT 512 ✓ ✓ ✓ 0.885 0.5743

Table 3: Ablation study of IV-CoT. All ablation variants are trained with the same data mixture and training budget.

[Figure 21]

- Figure 4: Visualization and perturbation of structural

group cardinality. Details and stage- and layer-wise visualizations are provided in Appendix D.

Figure 5 shows that Qs receives higher relative attention around object contours, boundaries, and coarse spatial structures, while Qm is more broadly activated over object interiors and appearancerelated regions. This does not indicate a hard separation, but suggests a soft functional specialization: structural queries provide spatial guidance, whereas semantic queries support appearance rendering conditioned on the structural plan.

queries. Qs visualization reveals sketch-like latent visual plans that preserve object contours, counts, and coarse layouts. Perturbing Qs with random queries before diffusion decoding disrupts the generated layout and object geometry, indicating that Qs acts as an active structural condition.

3.5 Zero-Shot Structure-Appearance Recombination (RQ4)

We further examine whether the learned query separation provides controllable handles in the latent space. Given two prompts yA and yB, IV-CoT produces query pairs (QAs ,QAm) and (QBs ,QBm). We then recombine them across prompts, e.g., [QAs ,QBm], and feed the mixed queries into the diffusion generator without additional training.

attention to structural and semantic queries during rendering. For each spatial latent position, we compute the relative cross-attention proportion assigned to Qs and Qm, averaged over selected middle denoising steps. Since the two query groups have the same size, the comparison is not biased by

As shown in Figure 6, the outputs often pre-

[Figure 22]

- Figure 5: Relative cross-attention proportion maps. Each row shows the generated image, the normalized

proportion assigned to structural queries Qs, and the complementary proportion assigned to semantic queries Qm. Structural queries receive higher relative attention around contours and spatial boundaries.

[Figure 23]

- Figure 6: Cross-prompt structure-appearance recombination. Structural queries from Prompt A are combined with semantic queries from Prompt B. The mixed outputs tend to preserve the coarse layout from Prompt A while adopting appearance attributes from Prompt B, suggesting partial controllability in the latent space.

serve the coarse layout or object configuration from prompt A while adopting salient appearance attributes from prompt B. This recombination behavior, although not explicitly trained, suggests that IV-CoT learns a partially controllable structureappearance separation in the latent query space.

While the separation is not perfectly orthogonal, the results indicate that structural queries encode manipulable latent visual plans.

### 4 Related Work

Explicit reasoning for image generation. With the great success of unified multi-modal large language models (Zhou et al., 2025b; Lin et al., 2025a; Xiao et al., 2026a,b), broader language-model applications and multimodal learning (Xiong et al., 2026, 2024; Li et al., 2025b, 2024; Hu et al., 2026; Chen et al., 2024, 2025b; Sun et al., 2025), and visual generation models (Yang et al., 2026; Song et al., 2025, 2026), recent work improves compositional image generation by externalizing intermediate reasoning as text, layouts, scene plans, or visual drafts. LLM-based planning methods first produce scene descriptions, layouts, or other intermediate representations before invoking a diffusion generator (Feng et al., 2023; Lian et al., 2023; Galun and Benaim, 2024; Koch et al., 2025). Recent work has explored explicit reasoning for visual generation through pre-generation textual planning, RLenhanced CoT, and post-generation reflection (Liao et al., 2025; Guo et al., 2025b; Tong et al., 2026; Zhang et al., 2025b; Gu et al., 2025; Zhang et al., 2025c; Li et al., 2025a; Zhuo et al., 2025; Huang et al., 2025). These methods externalize reasoning as textual plans, verification traces, visual drafts, or iterative refinement steps. IV-CoT instead keeps the intermediate visual plan inside latent queries and performs standard single-pass diffusion decoding at inference time.

Latent and continuous reasoning. Recent studies on latent or continuous reasoning suggest that intermediate deliberation can be carried by hidden states or soft thought tokens rather than naturallanguage rationales (Hao et al., 2024; Xu et al., 2025a,b; Ramji et al., 2026). Similar ideas have also been explored in multimodal understanding, where latent visual or multimodal tokens support reasoning without fully verbalizing intermediate steps (Pham and Ngo, 2025; Chen et al., 2025a; Wang et al., 2025; Xiong et al., 2026). Broader recent efforts explore latent representations for image generation via generation-time intervention (Mi et al., 2025; Chen et al., 2026; Sun et al., 2026). IVCoT instead structures the MLLM-DiT conditioning interface with structure-first queries, without intermediate decoding or test-time latent search.

### 5 Conclusion

We introduce IV-CoT, a structure-first latent reasoning framework for text-to-image generation that organizes conditioning queries into structural and semantic roles. With training-only sketch supervision and a structural-to-semantic query cascade, IVCoT keeps visual planning in latent representations without decoding intermediate sketches or textual rationales at inference time. Experiments and analyses on GenEval and T2I-CompBench show that IV-CoT improves structure-aware prompt following while maintaining high efficiency, with structural queries encoding recoverable and manipulable visual plans.

### Limitations

This work focuses on structure-aware prompt following for text-to-image generation. First, IV-CoT is not specifically optimized for rendering readable text within images. Although sketch supervision encourages structural queries to capture contours, layouts, and object configurations, accurate scenetext rendering requires fine-grained character-level alignment, spelling consistency, and typographyaware supervision, which are not explicitly modeled in our current training objectives. Second, we mainly evaluate IV-CoT in text-to-image generation and have not explored image editing scenarios, such as localized editing, instruction-guided revision, or multi-turn refinement. Extending latent structural-to-semantic reasoning to text-aware generation and editing remains a promising direction for future work.

### References

Chao Chen, Zhixin Ma, Yongqi Li, Yupeng Hu, Yinwei Wei, Wenjie Li, and Liqiang Nie. 2025a. Reasoning in the dark: Interleaved vision-text reasoning in latent space. arXiv preprint arXiv:2510.12603.

Harold Haodong Chen, Xinxiang Yin, Wen-Jie Shu, Hongfei Zhang, Zixin Zhang, Chenfei Liao, Litao Guo, Qifeng Chen, and Ying-Cong Chen. 2026. Show, don’t tell: Morphing latent reasoning into image generation. arXiv preprint arXiv:2602.02227.

Jian Chen, Yuxuan Hu, Haifeng Lu, Wei Wang, Min Yang, Chengming Li, and Xiping Hu. 2025b. Mghft: Multi-granularity hierarchical fusion transformer for cross-modal sticker emotion recognition. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 5794–5803.

Jian Chen, Wei Wang, Yuzhu Hu, Junxin Chen, Han Liu, and Xiping Hu. 2024. Tgca-pvt: Topic-guided context-aware pyramid vision transformer for sticker emotion recognition. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 9709–9718.

Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, and 1 others. 2025c. Blip3o: A family of fully open unified multimodal modelsarchitecture, training and dataset. arXiv preprint arXiv:2505.09568.

Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, and Benyou Wang. 2025d. Sharegpt-4o-image: Aligning multimodal models with gpt-4o-level image generation. arXiv preprint arXiv:2506.18095.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. 2025e. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811.

Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, and 1 others. 2025. Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, and 1 others. 2025. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683.

Chengqi Duan, Rongyao Fang, Yuqing Wang, Kun Wang, Linjiang Huang, Xingyu Zeng, Hongsheng Li, and Xihui Liu. 2025. Got-r1: Unleashing reasoning capability of mllm for visual generation with reinforcement learning. arXiv preprint arXiv:2505.17022.

Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. 2023. Layoutgpt: Compositional visual planning and generation with large language models. Advances in Neural Information Processing Systems, 36:18225–18250.

Ran Galun and Sagie Benaim. 2024. Generating intermediate representations for compositional text-toimage generation. arXiv preprint arXiv:2410.09792.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. 2023. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152.

Zeqi Gu, Markos Georgopoulos, Xiaoliang Dai, Marjan Ghazvininejad, Chu Wang, Felix Juefei-Xu, Kunpeng Li, Yujun Shi, Zecheng He, Zijian He, and 1 others. 2025. Improving chain-of-thought efficiency

for autoregressive image generation. arXiv preprint arXiv:2510.05593.

Ziyu Guo, Renrui Zhang, Hongyu Li, Manyuan Zhang, Xinyan Chen, Sifan Wang, Yan Feng, Peng Pei, and Pheng-Ann Heng. 2025a. Thinking-whilegenerating: Interleaving textual reasoning throughout visual generation. arXiv preprint arXiv:2511.16671.

Ziyu Guo, Renrui Zhang, Chengzhuo Tong, Zhizheng Zhao, Rui Huang, Haoquan Zhang, Manyuan Zhang, Jiaming Liu, Shanghang Zhang, Peng Gao, and 1 others. 2025b. Can we generate images with cot? let’s verify and reinforce image generation step by step. arXiv preprint arXiv:2501.13926.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. 2024. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769.

Yuxuan Hu, Jian Chen, Yuhao Wang, Zixuan Li, Jing Xiong, Pengyue Jia, Wei Wang, Chengming Li, and Xiangyu Zhao. 2026. Emotion and intention guided multi-modal learning for sticker response selection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 14883–14891.

Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. 2023. T2i-compbench: A comprehensive benchmark for open-world compositional text-toimage generation. Advances in Neural Information Processing Systems, 36:78723–78747.

Wenxuan Huang, Shuang Chen, Zheyong Xie, Shaosheng Cao, Shixiang Tang, Yufan Shen, Qingyu Yin, Wenbo Hu, Xiaoman Wang, Yuntian Tang, and 1 others. 2025. Interleaving reasoning for better text-toimage generation. arXiv preprint arXiv:2509.06945.

Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, and Hongsheng Li. 2026. T2i-r1: Reinforcing image generation with collaborative semantic-level and tokenlevel cot. Advances in Neural Information Processing Systems, 38:39856–39890.

Dongzhi Jiang, Renrui Zhang, Haodong Li, Zhuofan Zong, Ziyu Guo, Jun He, Claire Guo, Junyan Ye, Rongyao Fang, Weijia Li, and 1 others. 2025. Draco: Draft as cot for text-to-image preview and rare concept generation. arXiv preprint arXiv:2512.05112.

Jan-Hendrik Koch, Jonas Krumme, and Konrad Gadzicki. 2025. A two-stage system for layout-controlled image generation using large language models and diffusion models. arXiv preprint arXiv:2511.06888.

Shufan Li, Konstantinos Kallidromitis, Akash Gokul, Arsh Koneru, Yusuke Kato, Kazuki Kozuka, and Aditya Grover. 2025a. Reflect-dit: Inferencetime scaling for text-to-image diffusion transformers via in-context reflection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15657–15668.

Zixuan Li, Binzong Geng, Jing Xiong, Yong He, Yuxuan Hu, Jian Chen, Dingwei Chen, Xiyu Chang, Liang Zhang, Linjian Mo, and 1 others. 2025b. Ctr-sink: Attention sink for language models in click-through rate prediction. arXiv preprint arXiv:2508.03668.

Zixuan Li, Jing Xiong, Fanghua Ye, Chuanyang Zheng, Xun Wu, Jianqiao Lu, Zhongwei Wan, Xiaodan Liang, Chengming Li, Zhenan Sun, and 1 others. 2024. Uncertaintyrag: Span-level uncertainty enhanced long-context modeling for retrieval-augmented generation. arXiv preprint arXiv:2410.02719.

Long Lian, Boyi Li, Adam Yala, and Trevor Darrell. 2023. Llm-grounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models. arXiv preprint arXiv:2305.13655.

Jiaqi Liao, Zhengyuan Yang, Linjie Li, Dianqi Li, Kevin Lin, Yu Cheng, and Lijuan Wang. 2025. Imagegencot: Enhancing text-to-image in-context learning with chain-of-thought reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17214–17223.

Haokun Lin, Xinle Jia, Shaozhen Liu, Shujun Xia, Weitao Huang, Haobo Xu, Junyang Li, Yicheng Xiao, Xingrun Xing, Ziyu Guo, and 1 others. 2026a. Efficient diffusion language models: A comprehensive survey. Authorea Preprints, 3.

Haokun Lin, Xinle Jia, Haobo Xu, Bingchen Yao, Xianglong Guo, Yichen Wu, Zhichao Lu, Ying Wei, Qingfu Zhang, and Zhenan Sun. 2026b. Duquant++: Fine-grained rotation enhances microscaling fp4 quantization. arXiv preprint arXiv:2604.17789.

Haokun Lin, Teng Wang, Yixiao Ge, Yuying Ge, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun, and Ying Shan. 2025a. Toklip: Marry visual tokens to clip for multimodal comprehension and generation. arXiv preprint arXiv:2505.05422.

Haokun Lin, Haobo Xu, Yichen Wu, Ziyu Guo, Renrui Zhang, Zhichao Lu, Ying Wei, Qingfu Zhang, and Zhenan Sun. 2025b. Quantization meets dllms: A systematic study of post-training quantization for diffusion llms. arXiv preprint arXiv:2508.14896.

Zhiheng Liu, Weiming Ren, Xiaoke Huang, Shoufa Chen, Tianhong Li, Mengzhao Chen, Yatai Ji, Sen He, Jonas Schult, Belinda Zeng, and 1 others. 2026. Tuna-2: Pixel embeddings beat vision encoders for multimodal understanding and generation. arXiv preprint arXiv:2604.24763.

Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai Yu, and 1 others. 2025. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7739–7751.

Yapeng Mi, Yanpeng Zhao, Hengli Li, Chenxi Li, Huimin Wu, Xiaojian Ma, Song-Chun Zhu, Ying Nian Wu, and Qing Li. 2025. Milr: Improving multimodal image generation via test-time latent reasoning. arXiv preprint arXiv:2509.22761.

Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix JuefeiXu, and 1 others. 2025. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256.

Tan-Hanh Pham and Chris Ngo. 2025. Multimodal chain of continuous thought for latent-space reasoning in vision-language models. arXiv preprint arXiv:2508.12587.

Luozheng Qin, Jia Gong, Yuqing Sun, Tianjiao Li, Mengping Yang, Xiaomeng Yang, Chao Qu, Zhiyu Tan, and Hao Li. 2025. Uni-cot: Towards unified chain-of-thought reasoning across text and vision. arXiv preprint arXiv:2508.05606.

Keshav Ramji, Tahira Naseem, and Ramón Fernandez Astudillo. 2026. Thinking without words: Efficient latent reasoning with abstract chain-of-thought. arXiv preprint arXiv:2604.22709.

Xinyang Song, Libin Wang, Weining Wang, Zhiwei Li, Jianxin Sun, Dandan Zheng, Jingdong Chen, Qi Li, and Zhenan Sun. 2025. 3sgen: Unified subject, style, and structure-driven image generation with adaptive task-specific memory. arXiv preprint arXiv:2512.19271.

Xinyang Song, Libin Wang, Weining Wang, Shaozhen Liu, Dandan Zheng, Jingdong Chen, Qi Li, and Zhenan Sun. 2026. Unialignment: Semantic alignment for unified image generation, understanding, manipulation and perception. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 9116–9126.

Zhuo Su, Wenzhe Liu, Zitong Yu, Dewen Hu, Qing Liao, Qi Tian, Matti Pietikäinen, and Li Liu. 2021. Pixel difference networks for efficient edge detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5117–5127.

Xin Sun, Jianan Xie, Zhongqi Chen, Qiang Liu, Shu Wu, Yuehe Chen, Bowen Song, Zilei Wang, Weiqiang Wang, and Liang Wang. 2025. Divide-then-align: Honest alignment based on the knowledge boundary of rag. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11461–11480.

Yuwei Sun, Yuxuan Yao, Hui Li, and Siyu Zhu. 2026. The thinking pixel: Recursive sparse reasoning in multimodal diffusion latents. arXiv preprint arXiv:2604.25299.

Rui Tian, Mingfei Gao, Mingze Xu, Jiaming Hu, Jiasen Lu, Zuxuan Wu, Yinfei Yang, and Afshin Dehghan.

2026. Unigen: Enhanced training & test-time strategies for unified multimodal understanding and generation. Advances in Neural Information Processing Systems, 38:152386–152415.

Chengzhuo Tong, Ziyu Guo, Renrui Zhang, Wenyu Shan, Xinyu Wei, Zhenghao Xing, Hongsheng Li, and Pheng-Ann Heng. 2026. Delving into rl for image generation with cot: A study on dpo vs. grpo. Advances in Neural Information Processing Systems, 38:115632–115655.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, and 1 others. 2024. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869.

Yaoting Wang, Shengqiong Wu, Yuecheng Zhang, Shuicheng Yan, Ziwei Liu, Jiebo Luo, and Hao Fei. 2025. Multimodal chain-of-thought reasoning: A comprehensive survey. arXiv preprint arXiv:2503.12605.

Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, and 1 others. 2025a. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12966–12977.

Size Wu, Zhonghua Wu, Zerui Gong, Qingyi Tao, Sheng Jin, Qinyue Li, Wei Li, and Chen Change Loy. 2025b. Openuni: A simple baseline for unified multimodal understanding and generation. arXiv preprint arXiv:2505.23661.

Shujun Xia, Haokun Lin, Yichen Wu, Yinan Zhou, Zixuan Li, Zhongwei Wan, Xingrun Xing, Yefeng Zheng, Xiang Li, Caifeng Shan, and 1 others. 2025. Medrek: Retrieval-based editing for medical llms with keyaware prompts. arXiv preprint arXiv:2510.13500.

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. 2025. Omnigen: Unified image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13294–13304.

Yicheng Xiao, Lin Song, Yukang Chen, Yingmin Luo, Yuxin Chen, Yukang Gan, Wei Huang, Xiu Li, Xiaojuan Qi, and Ying Shan. 2026a. Mindomni: Unleashing reasoning generation in vision language models with rgpo. Advances in Neural Information Processing Systems, 38:88786–88810.

Yicheng Xiao, Wenhu Zhang, Lin Song, Yukang Chen, Wenbo Li, Nan Jiang, Tianhe Ren, Haokun Lin, Wei Huang, Haoyang Huang, and 1 others. 2026b. Spatialedit: Benchmarking fine-grained image spatial editing. arXiv preprint arXiv:2604.04911.

Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Chengyue Wu, Yujun Lin, Zhekai

Zhang, Muyang Li, Junyu Chen, and 1 others. 2025. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. arXiv preprint arXiv:2501.18427.

Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. 2026. Show-o2: Improved native unified multimodal models. Advances in Neural Information Processing Systems, 38:47490–47518.

Xingrun Xing, Zheng Liu, Shitao Xiao, Boyan Gao, Yiming Liang, Wanpeng Zhang, Haokun Lin, Guoqi Li, and Jiajun Zhang. 2025. Efficientllm: Scalable pruning-aware pretraining for architectureagnostic edge language models. arXiv preprint arXiv:2502.06663.

Jing Xiong, Qi Han, Yunta Hsieh, Hui Shen, Huajian Xin, Chaofan Tao, Chenyang Zhao, Hengyuan Zhang, Taiqiang Wu, Zhen Zhang, and 1 others. 2026. Mmformalizer: Multimodal autoformalization in the wild. arXiv preprint.

Jing Xiong, Zixuan Li, Chuanyang Zheng, Zhijiang Guo, Yichun Yin, Enze Xie, Zhicheng Yang, Qingxing Cao, Haiming Wang, Xiongwei Han, and 1 others. 2024. Dq-lore: Dual queries with low rank approximation re-ranking for in-context learning. In International Conference on Learning Representations, volume 2024, pages 41179–41203.

Haobo Xu, Sirui Chen, Ruizhong Qiu, Yuchen Yan, Chen Luo, Monica Cheng, Jingrui He, and Hanghang Tong. 2026. Prune as you generate: Online rollout pruning for faster and better rlvr. arXiv preprint arXiv:2603.24840.

Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. 2025a. Softcot: Soft chain-of-thought for efficient reasoning with llms. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23336– 23351.

Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. 2025b. Softcot++: Test-time scaling with soft chain-of-thought reasoning. arXiv preprint arXiv:2505.11484.

Yunqiao Yang, Haokun Lin, Guanzhong Wu, and Ying Wei. 2026. Concept-guided tokenization: Closing the gap between reconstruction and generation. In Fortythird International Conference on Machine Learning.

Junyan Ye, Dongzhi Jiang, Zihao Wang, Leqi Zhu, Zhenghao Hu, Zilong Huang, Jun He, Zhiyuan Yan, Jinghua Yu, Hongsheng Li, and 1 others. 2025. Echo4o: Harnessing the power of gpt-4o synthetic images for improved image generation. arXiv preprint arXiv:2508.09987.

Xinchen Zhang, Ling Yang, Guohao Li, Yaqi Cai, Yong Tang, Yujiu Yang, Mengdi Wang, Bin CUI, and

- 1 others. 2025a. Itercomp: Iterative compositionaware feedback learning from model gallery for textto-image generation. In International Conference

on Learning Representations, volume 2025, pages 31968–31988.

Yu Zhang, Yunqi Li, Yifan Yang, Rui Wang, Yuqing Yang, Dai Qi, Jianmin Bao, Dongdong Chen, Chong Luo, and Lili Qiu. 2025b. Reasongen-r1: Cot for autoregressive image generation models through sft and rl. arXiv preprint arXiv:2505.24875.

Yuyao Zhang, Jinghao Li, and Yu-Wing Tai. 2025c. Layercraft: Enhancing text-to-image generation with cot reasoning and layered object integration. arXiv preprint arXiv:2504.00010.

Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. 2025a. Transfusion: Predict the next token and diffuse images with one multi-modal model. In International Conference on Learning Representations, volume 2025, pages 6446–6469.

Yinan Zhou, Yaxiong Wang, Haokun Lin, Chen Ma, Li Zhu, and Zhedong Zheng. 2025b. Scale up composed image retrieval learning via modification text generation. IEEE Transactions on Multimedia.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, and 1 others. 2025. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479.

Le Zhuo, Liangbing Zhao, Sayak Paul, Yue Liao, Renrui Zhang, Yi Xin, Peng Gao, Mohamed Elhoseiny, and Hongsheng Li. 2025. From reflection to perfection: Scaling inference-time optimization for text-to-image diffusion models via reflection tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15329–15339.

### A Implementation Details

Backbone and query configuration. We instantiate IV-CoT on OpenUni-L, which combines a 2B InternVL3 (Zhu et al., 2025) MLLM with a 1.6B Sana diffusion generator (Xie et al., 2025). The model uses two groups of 256 visual queries, resulting in 512 conditioning queries after concatenation. The semantic queries are initialized from the pretrained OpenUni checkpoint, while the structural queries are initialized from the Stage-1 checkpoint. During Stage-2 training, the MLLM is frozen, and the Sana diffusion transformer, dual query inputs, and connector/projector modules are optimized. We set the structural regularization weight to λ = 0.3.

Training data. We train on a combined dataset of 128,393 image-text pairs from BLIP3o, ShareGPT4o-Image, and Echo-4o.

Optimization. We train IV-CoT on NVIDIA A800 80GB GPUs using bfloat16 mixed precision. We use AdamW with learning rate 2 × 10−5, β = (0.9,0.95), weight decay 0.05, and gradient clipping at 1.0. The learning rate is linearly warmed up for the first 10% of training steps and then decayed with a cosine schedule to 1 × 10−7. Unless otherwise specified, we set the random seed to 42.

### B Additional Generation Samples

We provide additional generation samples from IV-CoT in Figure 7. These examples cover diverse object categories, scenes, and visual styles, illustrating that IV-CoT maintains broad generation capability while preserving coherent visual structures.

### C Latency Measurement

As efficiency is important for the generation process (Lin et al., 2025b, 2026b,a; Xing et al., 2025; Xia et al., 2025; Xu et al., 2026), we measure latency on a single NVIDIA A800 80GB GPU with batch size 1. For each method, we report the average wall-clock inference time over 100 prompts, excluding model loading time. The time includes all steps required to produce the final image, including text processing, method-specific reasoning or intermediate generation, and final image synthesis.

### D Attention Analysis

Relative attention proportion. To further examine how the diffusion generator allocates attention

between structural and semantic queries during rendering, we compute the relative cross-attention proportion assigned to each query group. For each spatial latent position p, let A(p,q) denote the crossattention weight from position p to query q. We define

A(p,q), (11)

A(p,q) +

Z(p) =

q∈Qm

q∈Qs

A(p,q) Z(p)

rs(p) = q∈Qs

, (12) rm(p) = 1 − rs(p), (13)

where rs(p) and rm(p) denote the relative proportions assigned to structural queries Qs and semantic queries Qm, respectively. Since the two query groups contain the same number of queries, this group-wise normalization is not confounded by query-group size. The maps should therefore be interpreted as relative attention allocations between the two query groups, rather than absolute attention magnitudes.

Layer- and step-wise visualization. Figure 8 expands the main attention analysis across denoising steps and diffusion-transformer layer groups. Columns correspond to increasing denoising steps, and rows group consecutive layers. Within each triplet, from left to right, we show the intermediate denoised image, the relative attention proportion assigned to structural queries Qs, and the complementary proportion assigned to semantic queries Qm.

We observe a layer–step interaction. At early denoising steps, when the latent image state is still noisy, structural patterns are more visible in deeper layers, suggesting that deeper layers aggregate global information to recover coarse layouts. As denoising progresses, similar spatial patterns also appear in shallower layers, indicating that structural queries increasingly align with local object regions once coarse structures have emerged. The semantic-query maps show complementary and often more diffuse allocation patterns, suggesting soft functional specialization rather than a hard separation.

### E Use of AI Assistants

The authors used AI assistants for language polishing, wording suggestions, and submission-form preparation. All technical content, experiments, analyses, claims, and final text were reviewed and verified by the authors.

[Figure 24]

###### Figure 7: Additional qualitative samples generated by IV-CoT across diverse prompts. The examples cover objects, portraits, animals, natural scenes, and artistic styles, showing that the proposed structure-first latent reasoning framework maintains broad visual diversity while producing coherent image structures.

[Figure 25]

- Figure 8: Layer- and denoising-step-wise relative cross-attention proportion maps. Columns show denoising steps 5, 10, 15, and 20, and rows show grouped diffusion-transformer layers. Within each triplet, from left to right, we visualize the intermediate denoised image, the relative proportion assigned to structural queries Qs, and the complementary proportion assigned to semantic queries Qm. At early denoising steps, structural-query patterns become more organized in deeper layers; at later steps, similar spatial patterns also emerge in shallower layers, suggesting progressive structure formation across denoising and depth.

