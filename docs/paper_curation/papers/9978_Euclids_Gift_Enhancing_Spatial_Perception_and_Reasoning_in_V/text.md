# arXiv:2509.24473v3[cs.CV]19Nov2025

## Euclid’s Gift: Enhancing Spatial Perception and Reasoning in Vision-Language Models via Geometric Surrogate Tasks

Shijie Lian1,2*, Changti Wu3,2*, Laurence Tianruo Yang4,1†, Hang Yuan2, Bin Yu2, Lei Zhang3, Kai Chen5† 1Huazhong University of Science and Technology 2Zhongguancun Academy 3East China Normal University 4Zhengzhou University 5Zhongguancun Institute of Artificial Intelligence

###### Abstract

Spatial intelligence spans a rich suite of abilities, including visualising and transforming shapes, mentally rotating objects, judging relational positions and containment, and estimating numerosity. However, it still remains a critical unresolved challenge for Multimodal Large Language Models (MLLMs). To fill this gap, we propose to treat Euclidean geometry problem-solving as a surrogate task. Specifically, we meticulously constructed a curated multimodal dataset, called Euclid30K, comprising approximately 30K plane and solid geometry problems. Furthermore, to enable the model to learn and apply Euclidean principles from these geometry problems, we fine-tuned seven model variants (spanning 3–72B parameters) from the Qwen2.5VL, Qwen3VL, and RoboBrain2.0 families using Group Relative Policy Optimization (GRPO), inspiring the models to identify shapes, count, and relate entities, and perform multi-step deductive reasoning using Euclidean principles. Our experiments demonstrate that the resulting models achieve substantial zero-shot gains across four spatial reasoning benchmarks (Super-CLEVR, Omni3DBench, VSI-Bench, and MindCube) without any task-specific adaptations. Notably, after training on the Euclid30K, the mean VSI-Bench accuracy rose from 36.6% to 41.8% (+5.2%), and the mean MindCube accuracy rose from 31.4% to 38.1% (+6.7%). To our knowledge, this is the first systematic study showing that geometry-centric fine-tuning can confer vision-language models with broadly transferable spatial skills. Code and Euclid30K dataset can be found in this.

###### 1. Introduction

“The whole is greater than the part.”

— Euclid, Elements I, Common Notion 5 In recent years, multimodal large language models

*These authors contributed equally †Corresponding author

(MLLMs) [3, 46, 51] have achieved remarkable success across a broad range of vision-language tasks, from image captioning and visual question answering (VQA) to document understanding [4, 9, 33, 34, 42, 53]. State-ofthe-art models like GPT-5 [63], Gemini-2.5-Pro [22], and Qwen3VL-series [72] now rival or even surpass human performance on certain benchmarks, especially tasks requiring advanced language understanding or mathematical reasoning. For instance, models like GPT-4o [37], InternVL240B [18], and Qwen2.5VL-32B [9] have exceeded the average human score on the MathVista multimodal math benchmark [55], reflecting the rapid progress in integrating visual perception with high-level reasoning.

Despite recent progress, state-of-the-art MLLMs still fall short of genuine spatial intelligence [78, 85]. Spatial intelligence involves perceiving and mentally manipulating spatial relationships and spans several tasks, such as estimating quantity, interpreting spatial relations, and understanding geometric configurations [32, 78]. Nowadays, leading vision-language models (VLMs) still make occasional mistakes on tasks that young children solve with ease, such as determining object orientation or identifying which object is the nearest neighbor to a given object on its left [47, 59]. A recent evaluation on the Visual-Spatial Intelligence Benchmark (VSIBench) shows that more than 70% of the recorded errors arise from faulty spatial reasoning, not from deficiencies in visual recognition or language parsing [78]. This phenomenon is consistent with Moravec’s paradox [61], which suggests that high-level reasoning tasks are computationally simpler for VLM than low-level perceptual and sensorimotor skills. Closing this gap is essential for the next generation of VLMs [23].

Recent work on spatially aware VLMs, including Spatial-MLLM [76], SpaceVLM [14], VLM-3R [27], RoboBrain2.0 [8], and SIMS-V [12], attempts to provide specially constructed spatial datasets to improve model performance. However, tasks in these spatial datasets usually cover only a subset of real-world spatial tasks and may not enhance the model’s overall spatial intelligence. For example, Spatial-MLLM collects data from ScanQA [7],

Qwen3VL-4B

Qwen3VL-30B-A3B

RoboBrain2.0-7B

###### RoboBrain2.0-32B

###### Overall

Overall

Overall

Overall

Obj. Cnt.

Obj. Cnt.

Obj. Cnt.

Obj. Cnt.

Appr. Order

Appr. Order

Appr. Order

Appr. Order

Route Plan

Abs. Dist.

Route Plan

Abs. Dist.

Route Plan

Abs. Dist.

Route Plan

Abs. Dist.

Rel. Dir.

Obj. Size

Rel. Dir.

Obj. Size

Rel. Dir.

Obj. Size

Rel. Dir.

Obj. Size

Rel.

Room

Rel.

Room

Rel.

Room

Rel.

Room

Dist.

Size

Dist.

Size

Dist.

Size

Dist.

Size

###### Base Trained on the Euclidean geometry data (Euclid30K Dataset)

Figure 1. Performance gains on VSIBench after model training on Euclid30K, for more complete data please refer to Tab. 2.

SQA3D [57], and self-curated spatial QA data, and follows the eight tasks introduced in VSI-Bench [78] to build the Spatial-MLLM-120K dataset. The trained model therefore achieves state-of-the-art results on VSI-Bench, ScanQA, and SQA3D. However, its accuracy drops on the out-ofdomain MindCube benchmark [81]. This highlights a critical challenge in the field: while fine-tuning on task-specific datasets can achieve high in-domain performance, it may lead to over-specialization and fail to cultivate a more fundamental, generalizable spatial intelligence. To bridge this gap, VLMs must learn from a broader and more foundational range of spatial phenomena, thus extending their capabilities beyond the limitations of any single dataset.

In order to develop generalized spatial skills beyond any single benchmark, we attempt to explore a novel training paradigm that incorporates solving geometric problems as a surrogate task for enhancing spatial intelligence in the VLM. Geometry compresses centuries of mathematical study into formal descriptions of space [36], and to some extent it shapes our spatial representations in the world [28]. Therefore, learning to solve planar and solid geometry problems forces the model to internalize the axioms and constraints of Euclidean geometry, and provides the model with stronger out-of-domain generalization capabilities, because these principles are universal and independent of any single task. As shown in Fig. 1, these low-level geometry priors, like “Euclid’s Gift”, provide a principled foundation that supports zero-shot transfer to a wide range of downstream spatial tasks.

This suggests that the abilities required to solve geometric problems, including recognizing shapes and configurations, inferring spatial relationships such as parallel, angular, and relative positions, calculating or measuring geometric elements, and performing multistep logical reasoning, are also required for spatial perception tasks, like object counting, relational localization, and size estimation [20, 78]. Thus, by using geometry as a surrogate task, we aim to instill foundational Euclidean priors that support a significant and critical subset of spatial intelligence, i.e., static spatial perception and reasoning [21, 64].

To enable geometry-centric training, we curated and re-

leased the Euclid30K dataset, containing 29,695 geometry problems from open-source datasets including Geometry3K [54], MMK12 [60], SolidGeo [74], and WeMath2 [66], along with newly collected middle- and highschool exercises and competition problems. Due to the existing corpus’s significant bias toward plane geometry, we expanded Euclid30K with 3,996 solid geometry problems collected from textbooks and exercise materials. Solid geometry exposes the model to intuitive 3D concepts such as perspective and occlusion, polyhedral structures, and spatial rotation, which are equally crucial for constructing robust spatial prior knowledge. Additionally, all answers were reformatted for direct recognition by the rule-based reward function implemented via MathVerify [38].

Furthermore, in order to attribute the performance improvement strictly to the supervision of geometric knowledge rather than the interference of complex algorithms or data augmentation, we intentionally used the wellestablished GRPO framework and Euclid30K to train the models. Specifically, we used GRPO to fine-tune the Qwen2.5VL series (3B, 7B, 72B parameters), the Qwen3VL series (4B, 8B, 30B parameters), and the RoboBrain2.0 series (7B, 32B parameters). The resulting geometry-trained models deliver consistent gains on SuperCLEVR [47], Omni3D-Bench [59], VSI-Bench [78], and MindCube [81]. This suggests that the abstract geometric knowledge distilled from Euclid30K can be migrated to different spatial tasks and supports the model’s static spatial perception and reasoning capabilities, is an effective alternative for enhancing spatial intelligence in VLMs.

To our knowledge, this is the first work to use geometry problems as surrogate tasks to cultivate spatial intelligence in general-purpose VLMs. Unlike prior approaches that fine-tuned models on a single skill, our training paradigm uses geometric tasks to endow the model with a deeper, principle-driven understanding of space. This allows the model to go beyond the baseline in multiple spatial reasoning tasks without additional task-specific training. In summary, our contributions are as follows:

• We demonstrate that tackling geometry tasks can serve as an effective surrogate task for spatial intelligence. Learn-

ing to solve geometry problems assists models in acquiring the most basic perceptions of space, such as the axioms of Euclidean geometry. These low-level priors provide a principled foundation that supports the zero-shot transfer to a wide range of downstream spatial tasks.

- • We collected and constructed the Euclid30K, a VQA dataset of 29,695 geometry questions. Euclid30K offers a diverse and challenging set of problems covering a wide range of geometric concepts, designed to help models learn formal descriptions of space from geometric tasks.
- • Extensive experiments on the Qwen2.5VL, Qwen3VL, and RoboBrain2.0 show significant performance gains on four spatial benchmarks (Super-CLEVR, Omni3DBench, VSI-Bench, and MindCube), providing empirical evidence that the geometric curriculum reliably enhances spatial reasoning across diverse evaluation settings.

###### 2. Related Work

###### 2.1. Spatial Intelligence

Spatial intelligence is the capacity to reason about spatial relations, including mental rotation, viewpoint switching, and object counting [32, 78]. Benchmarks such as CLEVR [40], Super-CLEVR [47], Omni3D-Bench [59], VSI-Bench [78], and MindCube [81] show that accurate visual recognition alone is not enough; models must also perform explicit spatial reasoning. Early VLMs, such as ViLT [41], METER [26], Flamingo [3], and PaLI [17], improved perception yet still struggled with counting and orientation. Recent spatial-aware systems such as SpaceVLM [14], SpatialMLLM [76], SpatialLLM [56], RoboBrain [39], and RoboBrain 2.0 [8] fine-tune on curated spatial datasets, but these datasets cover only part of real-world scenarios and are costly to build. Our approach instead trains on readily available Euclidean geometry problems, treating them as surrogate tasks that transfer their principles to general visualspatial perception and reasoning without costly data collection or architectural changes.

###### 2.2. Multimodal Math & Geometry Datasets

Research on multimodal math reasoning has produced datasets with progressively broaden both scope and format. GeoQA [15], Geometry3K [54], and UniGeo [16] pioneered the use of image-text pairs for plane-geometry question answering, laying the groundwork for visual-symbolic reasoning. Subsequent datasets, such as MMMU [83], MathVista [55], We-Math [65], We-Math2 [66], MMMUPro [84], and GeoSense [77], expanded to a wider array of math domains and introduced more challenging multimodal tasks. The recently released SolidGeo3K [74] benchmark focuses on 3D geometry, providing skill tags and difficulty annotations to establish a dedicated testing platform for solid geometry reasoning.

###### 3. Method

This section first presents a domain-adaptation perspective that motivates using geometric problem solving to enhance spatial intelligence, then describes the construction of Euclid30K and our GRPO-based training framework. We further elaborate on the connection between geometry and spatial cognition from an educational-psychology viewpoint in Appendix B.

###### 3.1. Motivation: A Domain-Adaptation Perspective

This subsection presents a domain-adaptation view that motivates why training on geometric problem solving serves as an effective surrogate for spatial intelligence.

Let the source distribution DS denote geometry problem solving (e.g., Euclid30K) and the target distribution DT denote spatial intelligence. We train a VLM policy πθ on the source. The policy induces a hypothesis:

h(x) = I Ans(x;πθ) is correct ∈ {0,1}. (1)

Let H be the hypothesis class containing h. For h,h′ ∈ H, define the error (disagreement probabilities):

ϵS(h,h′) := Pr

[h(x) ̸= h′(x)],

x∼DS

Specifically, when f is an absolutely correct hypothesis, we abbreviate ϵS(h,f) as ϵS(h). The definition of ϵT(h,h′) and ϵT(h) follows similarly. Furthermore, the H∆H distance [11] between DS and DT is:

ϵS(h,h′) − ϵT(h,h′) . (2)

dH∆H(DS,DT) = 2 sup

h,h′∈H

Based on this definition, we can invoke the standard domain-adaptation bound [11], which connects the ϵT(h) to the ϵS(h) through the H∆H distance.

Standard Domain-Adaptation Bound For any h ∈ H, we have:

ϵT(h) ≲ ϵS(h) + 12 dH∆H(DS,DT). (3) (A detailed proof can be found in Appendix A.

In light of the bound in Eq. (3), the target error is governed by the source error ϵS(h) and the distribution discrepancy dH∆H(DS,DT). In practice, ϵS(h) can typically be reduced through optimization and data curation, which makes the discrepancy term pivotal [11, 58]. Consequently, if dH∆H(DS,DT) is small, we can regard the source distribution DS as a surrogate for the target distribution DT.

We hypothesize that formal Euclidean geometry compresses a broad set of spatial regularities—congruence,

Statistic Number Total Number 29,695

[Figure 1]

Q: As shown in Figure, in the cube $ABCD - A_1B_1C_1D_1$ with edge length $a$, cut off two corners along the planes

[Figure 2]

[Figure 3]

- $A_1C_1B$ and $AB_1D_1$ (that is, remove two tetrahedrons), then cut off two more corners along the planes
- $B_1D_1C$ and $A_1C_1D$. The volume of the remaining part is ______.? A: $\frac{a^3}{2}$ Type: Mathematical Expression

[Figure 4]

Mathematical Expression 16,804 Numeric 6,321 Multiple-Choice 2,618

###### Fig. 1 Fig. 2 Fig. 3

Q: The mechanical device in Fig.1 is called a "rotary engine". One of its core components, the rotor, is in the shape of a "triangular prism with curved lateral surfaces". Fig.3 shows a triangular prism with curved lateral surfaces. Its lateral edges are perpendicular to the base, and the base is a "Reuleaux triangle". A Reuleaux triangle is formed by drawing arcs with the 3 vertices of an equilateral triangle as the centers and the side - length of the equilateral

Q: The height of wine in the wine bottle is equal to the height of conical stemmed wine. Given that the inner diameter of the

###### Type

triangle as the radius, as shown in Fig.3 . If the height of the

bottom of the wine bottle is $8 cm$, and the inner diameter

[Figure 5]

of the upper opening of the stemmed wine glass is also $8 cm$. If all the wine in the wine bottle is poured into the stemmed wine glasses, how many glasses can be filled

triangular prism with curved lateral surfaces is $10$ and the distance between any two vertices of the base is $20$, then what is its lateral area? A. $100\pi$ B . $200\pi$ C. $300\pi$ D. $400\pi$

Plane (2D) / Solid (3D) 18,577 / 11,118 Newly collected

Type: Numeric A: B Type: Multiple-Choice

###### A: 9

Newly collected Solid Questions 3,996 Newly collected Images 3,792

Figure 2. The examples of the newly collected questions in Euclid30K. More examples can be found in the appendix.

###### Length

Max and Avg question length 1,598 / 229.8 Max and Avg answer length 501 / 8.9 Max and Avg image numbers 8 / 1.1

MMK12 [60], SolidGeo3K [74], and WeMath2 [66]. We employed Qwen2.5-VL-72B as an annotator to determine whether a problem belongs to plane or solid geometry.

- Table 1. Statistics of Euclid30K, Mathematical Expression, Numeric and Multiple-Choice are the three types of answers.

After filtering the existing corpus, we identified a significant imbalance: only about 7,000 solid geometry problems remained, while there were about 20,000 plane geometry problems. However, solid geometry encompasses more explicit three-dimensional spatial phenomena (e.g., perspective invariance, polyhedron truncation, volume-area relationships), which are equally crucial for VLMs in learning spatial knowledge. Furthermore, existing solid geometry problems predominantly focus on shape recognition, coordinate/angle/area calculations, and similar question types, with insufficient coverage of richer problem types. To address these gaps, we newly collected approximately 4,500 additional problems from K-12 textbooks and competition practice books, covering positional relationships, dynamic geometry, folding/unfolding, and contextualized word problems. All newly collected problems are used with publisher permission for academic purposes.

similarity, perspective, parallelism, intersection, and positional reasoning—into theorem-like constraints reused across downstream spatial-intelligence tasks. Compared to training on a narrow sub-skill (e.g., object counting, depth ordering, or size estimation), this breadth plausibly yields a smaller dH∆H between geometry and target tasks, consistent with cognitive-science evidence on the generality of geometric knowledge in human perception and reasoning [30, 44]. We further elaborate on this connection from an educational-psychology perspective in Appendix B.

Since the spatial-intelligence target-domain distribution is unknown and there is no canonical proxy dataset DˆT that faithfully spans the full space of spatial-intelligence tasks, dH∆H is not directly observable in practice. Nevertheless, our subsequent empirical results align with this qualitative hypothesis: after Euclid30K fine-tuning, models exhibit consistent gains across Omni3D-Bench, VSI-Bench, Super-CLEVR, and MindCube (in Tables 2,3, and 4), and qualitative cases in the Appendix (Figs. 5–12) further illustrate acquired, transferable skills such as perspective (near to far size), parallelism and similarity, and positional inference. These observations provide a principled rationale for geometry as a surrogate task for spatial intelligence.

As a result, we compiled about 32,500 candidate questions. Then, to further ensure the quality of collected geometric data, we designed a three-stage filtering process:

- • Duplicate Filtering: Since even similar texts can lead to vastly different meanings or solution processes when paired with different images, we uniquely identify each question through its image. Specifically, we use imagebased perceptual hashing to filter duplicate questions.
- • Question Splitting: Many materials contain multiple sub-questions under a single main question. We utilize the GPT-4o API [37] to detect and enumerate each subquestion, splitting them into independent instances.
- • Formula Formatting: We standardize formulae in questions and answers to LaTeX format via the DeepSeekV3.1 API [25]. This ensures answers can be correctly parsed by MathVerify [38] for verification. For example, we replace ’π/2’ with ’\frac{\ pi}{2}’.

###### 3.2. Euclid30K

Plane and solid geometry give axiomatized abstractions of spatial phenomena. Training on such problems compels models to internalize Euclidean constraints such as angle and ratio preservation, similarity, and congruence, thereby providing an effective surrogate curriculum for cultivating spatial perception and reasoning skills.

All conversions are subsequently verified and, when necessary, corrected by human annotators to ensure the final dataset is correct and consistent. After the aforementioned filtering, splitting, formatting and human verification, we

Unfortunately, there are currently no large-scale, highquality training datasets tailored for diverse geometric problems. To address this, we filtered the required corpus from open-source resources such as Geometry3K [54],

Euclid30K

###### Origin Question 1

###### Origin Question N Question N

Question 1-1

Question 1-2

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

… Q: Asshown in the figure, on the edge of a dihedral angle of 60°, there are two points A and B. AC and BD are line segments perpendicular to AB in the two half - planes of this dihedral angle. A = 4cm, AC = 6cm, BD = 8cm. What is the length of CD？

…

Q: Build a geometric solid with small cubes. Its front view and top view are shown in the figure. Then the minimum number of cubes needed is ____ and the maximum number of cubes needed is ____ ？

Q: Build a geometric solid with small cubes. Its front view and top view are shown in the figure. Then the minimum number of cubes needed is ____ ？

Q: Build a geometric solid with small cubes. Its front view and top view are shown in the figure. Then the maximum number of cubes needed is ____ ？

Q: As shown in the figure, on the edge of a dihedral angle of $60^{\circ}$, there are two points $A$ and $B$. $AC$ and $BD$ are line - segments perpendicular to $AB$ in the two half - planes of this dihedral angle. $AB = 4cm$, $AC = 6cm$, $BD = 8cm$. What is the length of $CD$？

…

A: 8 and 12

A: 2√17

A: 8

A: 12

A: $2 \sqrt {17}$

Euclid30K + VLM Improve

Spatial Perception and Reasoning

Spatial Relation

###### Object Counting

###### Size Estimation

Distance Perception

###### Mental Rotation

[Figure 11]

[Figure 12]

[Figure 13]

|[Figure 14]|
|---|

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

…

How many table in the room？

What is the size of this bathroom?

How far is the stove from the sofa?

What's on the right after turning left 90°?

What color is the bicycle beside bus?

Figure 3. Enhancing spatial perception and reasoning capabilities in models using the geometric problem-solving dataset (Euclid30K).

ultimately obtained 29,695 geometry problems, which were compiled into the Euclid30K dataset. Summary statistics about Euclid30K are reported in Table 1. Additionally, we also list some examples of our newly collected data in Fig. 2. Notably, the final Euclid30K dataset contains 3,996 newly collected/generated solid geometry problems, exceeding SolidGeo [74], the largest prior solid geometry dataset, which contained 3,113 problems.

Finally, each instance in Euclid30K is represented as a triple < image(s) i, text problem p, answer a >, where i may contain one or multiple figures, p states the given conditions and questions, and a is one of: a LaTeX expression, a numeric value, or the index of the correct option.

###### 3.3. RL Training in Euclid30K

We follow the standard GRPO training practice [69] to enhance the geometry-solving ability of the Qwen2.5VL (3B, 72B), Qwen3VL (4B, 8B, 30B), and the RoboBrain2.0 (7B, 32B), but introduce some modifications based on recent work [82] to accelerate training. Specifically, during training, we first sample a set of outputs {oi}Gi=1 for each question q = < image i, textual problem p > from the policy model πθ

, and computing J (θ):

old

|oi|

G

1 γ

min ri,t(θ)Aˆi, clip(ri,t(θ),1 ± ϵ)Aˆi ,

J (θ) =

t=1

i=1

(4) where γ = Gi=1 |oi| and ri,t(θ) = ππθ(oi,t|q,oi,<t)

θold(oi,t|q,oi,<t). Advantage function Aˆi = Ri−mean({Ri}

G i=1)

std({Ri}Gi=1) computed using

the group-level rewards {Ri}Gi=1. In Eq. 4, we follow DAPO [82] using the token-level policy gradient loss and ensure that each batch has valid gradients by over-sampling and filtering out prompts whose answers are either entirely correct or entirely wrong. Then, we optimize the policy model πθ by maximizing the following objective:

[J (θ) − β · KL[πθ∥πref]]. (5)

LGRPO(θ) = Eq,o

i

In reinforcement learning with verifiable rewards for LLMs, the design of the reward function is critical [69, 76]. In addition to a formatting reward applied to all task types, we introduce task-dependent reward modelling to ensure that it accurately reflects the proximity between the predicted and ground-truth answers. Specifically, if the answer is a mathematical expression containing variables in LaTeX format, we invoke MathVerify [38] to perform exact symbolic equivalence checking, so algebraically identical forms (e.g., 2πr and (2r)π) receive the same reward. For purely numeric answers, to mitigate the risk of reward hacking [75], we forgo the conventional mean relative accuracy (MRA) metric [78] and instead grant the reward if the prediction lies within ±1% band around the ground truth:

 

pred − gt

gt ≤ 0.01, 0, otherwise,

1,

(6)

R =



where pred is the model prediction and gt is the ground truth value. This strategy filters out rough answers that have no practical relevance to the geometric conclusions,

Numerical Question Multiple-Choice Question

Methods

Overall Obj. Cnt. Abs. Dist. Obj. Size Room Size Rel. Dist. Rel. Dir. Route Plan Appr. Order

Proprietary Models GPT-4o [37] 46.2 5.3 43.8 38.2 37.0 41.3 31.5 28.5 34.0

- Gemini-1.5 Pro [34] 49.6 28.8 58.6 49.4 46.0 48.1 42.0 68.0 48.8

- Gemini-2.0 Flash [35] 52.4 30.6 66.7 31.8 56.0 46.3 24.5 55.1 45.4 Gemini-2.5-Pro[22] - - - - - - - - 47.8

Open-source Models

InternVL2-40B [19] 34.9 26.9 46.5 31.8 42.1 32.2 34.0 39.6 36.0 VILA-1.5-40B [48] 22.4 24.8 48.7 22.7 40.5 25.7 31.5 32.9 31.2 LLaVA-OneVision-72B [45] 43.5 23.9 57.6 37.5 42.5 39.9 32.5 44.6 40.2 LLaVA-Video-72B [87] 48.9 22.8 57.4 35.3 42.4 36.7 35.0 48.6 40.9

Spatial Models

M2-Reasoning-7B [2] 41.0 34.0 60.9 55.4 40.7 47.3 29.9 28.8 42.3 Spatial-MLLM-4B [76] 65.3 34.8 63.1 45.1 41.3 46.2 33.5 46.3 48.4

- Qwen2.5VL-series Qwen2.5VL-3B [9] 35.6 23.4 34.9 16.6 34.4 40.7 26.3 21.8 29.2 Qwen2.5VL-Euclid-3B 38.3 ↑ 26.8 ↑ 35.4 ↑ 22.2 ↑ 37.0 ↑ 43.2 ↑ 36.6 ↑ 16.3 32.0 ↑ Qwen2.5VL-72B [9] 13.6 19.6 40.9 41.1 37.7 35.3 34.0 36.2 32.3 Qwen2.5VL-Euclid-72B 22.5 ↑ 27.2 ↑ 55.7 ↑ 43.3 ↑ 44.9 ↑ 37.1 ↑ 32.5 36.6 ↑ 37.5 ↑

- Qwen3VL-series Qwen3VL-4B [72] 28.5 33.0 32.6 43.5 40.3 40.0 33.0 33.2 35.5 Qwen3VL-Euclid-4B 33.3 ↑ 37.4 ↑ 49.5 ↑ 48.3 ↑ 46.5 ↑ 46.3 ↑ 34.0 ↑ 42.9 ↑ 42.3 ↑ Qwen3VL-8B [72] 10.8 28.1 37.9 44.1 31.3 36.4 37.1 40.3 33.3 Qwen3VL-Euclid-8B 16.5 ↑ 29.8 ↑ 36.0 47.7 ↑ 34.5 ↑ 36.6 ↑ 38.1 ↑ 45.1 ↑ 35.5 ↑ Qwen3VL-30B-A3B [72] 27.4 32.3 53.6 44.0 42.1 36.2 35.5 48.9 40.0 Qwen3VL-Euclid-30B-A3B 33.5 ↑ 37.5 ↑ 64.1 ↑ 43.1 48.7 ↑ 49.6 ↑ 35.6 ↑ 54.4 ↑ 45.8 ↑

RoboBrai2.0-series RoboBrain2.0-7B [8] 46.0 32.7 58.9 35.9 45.9 41.5 30.9 55.2 43.0 RoboBrain2.0-Euclid-7B 66.4 ↑ 36.9 ↑ 66.3 ↑ 40.5 ↑ 48.3 ↑ 45.3 ↑ 35.6 ↑ 57.8 ↑ 49.6 ↑ RoboBrain2.0-32B [8] 50.5 37.0 59.2 28.4 43.2 46.1 34.5 39.5 43.1 RoboBrain2.0-Euclid-32B 59.2 ↑ 39.4 ↑ 63.4 ↑ 47.8 ↑ 48.7 ↑ 47.5 ↑ 33.5 57.0 ↑ 49.6 ↑

- Table 2. Evaluation on VSI-Bench [78]. Qwen2.5VL-Euclid, Qwen3VL-Euclid and RoboBrain2.0-Euclid indicate the Qwen2.5VL [9], Qwen3VL [72] and RoboBrain2.0 [8] trained with GRPO [69] on the Euclid30K dataset.

while allowing for rounding or floating-point truncation errors that are common in the generation process. Further, to prevent unit mismatches from interfering with reward computation, we manually append unit prompts (e.g., Please answer in meters) to questions in Euclid30K whenever the ground-truth answer carries a specific unit. In addition, for multiple choice questions, we use exact match reward.

###### 4. Experiment

In this section, we evaluate the zero-shot generalisation of models trained on geometry data to spatial intelligence tasks, using four benchmarks: Super CLEVR, Omni3D Bench, VSI Bench, and MindCube. We also conduct a causal ablation study that contrasts models trained on equal amounts of geometry data and spatial data. Additional experiments, along with detailed settings, prompt templates, and dataset settings, are provided in the Appendix C. More-

over, more visualization and analysis of results are presented in Appendix D and Figs. 4—12 in the Appendix.

###### 4.1. Comparisons on VSI-Bench

As noted in VSIBench, spatial reasoning ability is the primary bottleneck limiting MLLM performance on the VSIBench test [78]. Therefore, to better demonstrate how models perceive scenes and perform spatial reasoning, and to verify whether they genuinely acquire spatial intelligence from geometric knowledge, we deviate from the original VSI-Bench setup, which uses prompts such as ”Please answer the question using a single word or phrase” and constrains the maximum response length to 16 tokens. Instead, we follow the prompt configuration described in RoboBrain2.0 [8] Sec. B, which encourages the model to first reason about the problem before providing an answer, and we set the maximum response length to 1024 tokens. For fairness, all compared models (baselines and

Methods SuperClevr Omni3DBench Qwen2.5VL-series Qwen2.5VL-3B 70.0 24.7 Qwen2.5VL-Euclid-3B 75.2 ↑ 26.5 ↑ Qwen2.5VL-72B 72.6 30.4 Qwen2.5VL-Euclid-72B 83.1 ↑ 32.9 ↑ Qwen3VL-series Qwen3VL-4B 55.4 27.7 Qwen3VL-Euclid-4B 61.2 ↑ 31.7 ↑ Qwen3VL-8B 48.3 34.0 Qwen3VL-Euclid-8B 49.0 ↑ 35.0 ↑ Qwen3VL-30B-A3B 64.1 36.7 Qwen3VL-Euclid-30B-A3B 70.2 ↑ 38.9 ↑ RoboBrain2.0-series RoboBrain2.0-7B 47.4 14.2 RoboBrain2.0-Euclid-7B 85.2 ↑ 21.2 ↑ RoboBrain2.0-32B 59.5 34.8 RoboBrain2.0-Euclid-32B 75.6 ↑ 36.8 ↑

- Table 3. Evaluation on SuperClevr [47] & Omni3D Bench [59].

Euclid30K-fine-tuned variants) use the same prompt template and a 1024-token response budget. This setup allows us to observe the model’s intermediate reasoning process and assess whether it has internalized transferable spatial priors from Euclid30K training.

- Tab. 2 shows that Euclid30K fine-tuning yields an av-

erage accuracy improvement of 5.2% across all evaluated model variants in the Qwen2.5VL, Qwen3VL, and RoboBrain2.0 families. The effect is most pronounced on RoboBrain2.0, where the Euclid-trained 7B and 32B versions reach 49.6% overall, outstripping the best open-source reference (Spatial-MLLM-4B at 48.4%) and also surpassing the strongest proprietary baselines reported by VSIBench (Gemini-1.5 Pro and Gemini-2.5-Flash-preview-0417, 47.8% and 48.8%). Furthermore, the visualization examples in the Appendix D.1 demonstrate that these models have internalized fundamental spatial knowledge, such as perspective scaling, size estimation, and relational reasoning, by engaging with geometric knowledge.

4.2. Comparisons on SuperClevr & Omni3DBench

- Tab. 3 demonstrates that the fine-tuned Euclid30K model achieves improved accuracy on both SuperClevr and Omni3DBench, two classic spatial intelligence datasets. For Qwen2.5VL, the 3B model rises from 70.0 to 75.2 on Super-CLEVR and from 24.7 to 26.5 on Omni3D-Bench; the 72B model adds 10.5 percentage points on SuperCLEVR and about 2.5 on Omni3D-Bench. Qwen3VL shows a similar pattern. The dense 4B variant improves by

- 5.8 and 4.0 points on the two datasets, and the MoE 30BA3B improves by 6.1 and 2.2 points. Overall, Euclid30K delivers substantial accuracy gains, demonstrating that the

Methods Rotation Among Around Overall Proprietary Models

GPT-4o [37] 32.7 40.2 29.2 38.8 Claude-4-Sonnet [4] 48.4 44.2 47.6 44.8

Spatial Models

- RoboBrain1.0-7B [39] 35.8 38.3 29.5 37.4 SpaceMantis [14] 37.7 21.3 29.3 22.8 Space-Qwen [14] 38.0 33.7 26.3 33.3 Spatial-MLLM [76] 38.4 20.9 32.8 32.1 Qwen2.5VL-series Qwen2.5VL-3B [9] 14.3 22.8 24.1 20.4 Qwen2.5VL-Euclid-3B 33.5 ↑ 43.0 ↑ 40.0 ↑ 38.9 ↑ Qwen2.5VL-72B [9] 31.5 31.4 30.6 31.2 Qwen2.5VL-Euclid-72B 43.0 ↑ 35.6 ↑ 31.6 ↑ 36.7 ↑ Qwen3VL-series Qwen3VL-4B [72] 19.3 25.3 33.7 26.1 Qwen3VL-Euclid-4B 31.7 ↑ 29.2 ↑ 38.0 ↑ 33.0 ↑ Qwen3VL-8B [72] 25.3 32.4 44.7 34.2 Qwen3VL-Euclid-8B 40.0 ↑ 35.1 ↑ 48.0 ↑ 41.0 ↑ Qwen3VL-30B-A3B [72] 37.7 32.3 49.3 39.8 Qwen3VL-Euclid-30B-A3B 39.6 ↑ 33.8 ↑ 48.6 40.7 ↑

- RoboBrain2.0-series RoboBrain2.0-7B [8] 39.4 38.8 38.6 38.9 RoboBrain2.0-Euclid-7B 36.0 46.5 ↑ 36.2 39.4 ↑ RoboBrain2.0-32B [8] 21.2 35.4 31.0 29.2 RoboBrain2.0-Euclid-32B 38.9 ↑ 37.8 ↑ 33.8 ↑ 36.8 ↑

Table 4. Evaluation on MindCube [81].

geometry-centric curriculum provides broadly transferable spatial priors rather than mere data volume benefits.

The substantial performance gains observed for RoboBrain2.0 after Euclid30K fine-tuning are plausibly linked to its prior exposure: its SFT corpus includes substantial spatial-intelligence data, which likely instills implicit, intuitive notions about spatial phenomena. Training on Euclid30K then supplies explicit first-principles geometric reasoning, formalizing and consolidating those earlier intuitions, thereby enabling the model to generalize more rapidly to downstream tasks. This synergy between implicit spatial exposure and explicit geometric grounding may account for the pronounced performance improvements.

###### 4.3. Comparisons on MindCube

As shown in Tab. 4, fine-tuning on Euclid30K improves the overall MindCube accuracy across all variants in the three model families, with an average gain of 6.7 percentage points. It is also worth noting that models trained using Euclidean geometry datasets outperform existing spatial models (most of which are also based on Qwen backbones, but trained on a larger spatial corpus) in terms of generalization ability. For example, Spatial-MLLM combines the Qwen 2.5-VL-3B and VGGT [73] backbones, trains on 120K spa-

Numerical Question Multiple-Choice Question

Methods

Overall Obj. Cnt. Abs. Dist. Obj. Size Room Size Rel. Dist. Rel. Dir. Route Plan Appr. Order

- Qwen2.5VL-72B 13.6 19.6 40.9 41.1 37.7 35.3 34.0 36.2 32.3 Qwen2.5VL-Space-72B 15.6 24.8 40.7 41.4 43.4 37.8 29.4 33.5 33.2 Qwen2.5VL-Euclid-72B 22.5 27.2 55.7 43.3 44.9 37.1 32.5 36.6 37.5

- Qwen3VL-30B-A3B 27.4 32.3 53.6 44.0 42.1 36.2 35.5 48.9 40.0 Qwen3VL-Space-30B-A3B 29.9 36.8 58.0 43.4 47.9 49.1 35.9 52.7 44.2 Qwen3VL-Euclid-30B-A3B 33.5 37.5 64.1 43.1 48.7 49.6 35.6 54.4 45.8

RoboBrain2.0-32B 50.5 37.0 59.2 28.4 43.2 46.1 34.5 39.5 43.1 RoboBrain2.0-Space-32B 58.0 36.9 62.2 47.8 46.9 44.5 34.0 42.1 46.7 RoboBrain2.0-Euclid-32B 59.2 39.4 63.4 47.8 48.7 47.5 33.5 57.0 49.6

Table 5. Ablation experiment on VSI-Bench [78]. We compare training a model on a 30K subset of the spatial intelligence dataset Clevr-CoGenT v.s. the geometric dataset Euclid30K to verify that the geometric dataset serves as a surrogate task to improve the spatial intelligence capabilities of the model. Bolding indicates the best score within each model type.

tial samples, and scores 32.1% on MindCube. In contrast, Qwen2.5VL-Euclid-3B, trained with only 30K geometric datas, scored 38.9% on MindCube, representing a relative improvement of 6.8 percentage points. These results suggest that learning accurate Euclidean priors from a compact geometry course provides more transferable spatial knowledge than extending generalized spatial data alone.

###### 4.4. Ablation Study

To isolate the contribution of our Euclid30K dataset from the potential reasoning enhancements provided by the GRPO algorithm, we conducted a causal ablation study. Specifically, we randomly sampled a subset equal in size to Euclid30K on the non-geometric spatial intelligence dataset Clevr-CoGenT [40] and used the exact same GRPO setup to train Qwen2.5VL, Qwen3VL and RoboBrain2.0. This design ensures that performance gains after training on geometric data can be directly attributed to the fact that the geometric task as surrogate task for spatial intelligence, rather than due to the effects of GRPO or data incrementation.

Tab. 5 shows that models trained on Euclid30K achieve markedly higher overall accuracy than those fine-tuned on the equal-sized Clevr-CoGenT split. Specifically, Euclid30K training improves average accuracy from 38.5% to 44.3% (+5.8 points), whereas Clevr-CoGenT training yields a smaller improvement from 38.5% to 41.4% (+2.9 points). This demonstrates that the performance gains from Euclid30K exceed those attributable to additional data or GRPO-induced generalization alone, confirming that structured geometric knowledge provides a more transferable foundation for spatial reasoning. Ablation results for additional model sizes are provided in Appendix D.2.

###### 5. Discussion

Supervised Fine-tuning (SFT). For mathematical problems, collecting step-by-step solution annotations is sub-

stantially more expensive than answer-only labels. Accordingly, Euclid30K does not currently include process annotations, which makes it less suited to SFT routines that rely on explicit reasoning traces. Nevertheless, geometry QA can still serve as a surrogate task for spatial intelligence under SFT. In Appendix Appendix E.1, we report SFT results using other geometry datasets that provide process supervision. As future work, we plan to extend Euclid30K with solution-process annotations to better support SFT.

Model-Specific Performance Variations. Although geometry tasks provide beneficial Euclidean priors that enhance spatial reasoning across Tabs. 2 to 4, improvements exhibit model-specific variation. We offer several insights into this phenomenon. Performance gains appear linked to characteristics inherited from the model’s earlier training stages. Specifically, when a model already possesses rudimentary spatial concepts, training on Euclid30K refines and consolidates that knowledge through first-principles geometric reasoning, yielding larger improvements (e.g., RoboBrain2.0). Conversely, if a model must discover these first principles from scratch using only geometric data, gains tend to be more gradual. We provide additional experiments exploring this phenomenon in Appendix E.2.

###### 6. Conclusion

This study shows that using geometry as a surrogate task provides an alternative way to achieve transferable Spatial Perception and Reasoning. MLLMs trained on planar and solid geometry learn basic Euclidean priors that can be transferred to multiple spatial tasks without extra finetuning. Training solely on our Euclid30K yields consistent, significant gains across four unseen benchmarks for seven model. These generalization gains validate our core hypothesis that learning basic Euclidean geometric principles is an effective strategy for developing transferable spatial skills.

###### References

- [1] David Acuna, Guojun Zhang, Marc T. Law, and Sanja Fidler. f-domain adversarial learning: Theory and algorithms. In Proceedings of the 38th International Conference on Machine Learning, pages 66–75. PMLR, 2021. 13
- [2] Inclusion AI, Fudong Wang, Jiajia Liu, Jingdong Chen, Jun Zhou, Kaixiang Ji, Lixiang Ru, Qingpei Guo, Ruobing Zheng, Tianqi Li, et al. M2-reasoning: Empowering mllms with unified general and spatial reasoning. arXiv preprint arXiv:2507.08306, 2025. 6
- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736,

2022. 1, 3

- [4] Anthropic. Introducing claude 4. Anthropic Blog, 2025. 1, 7
- [5] Laszlo Aszalos and Maria Bako. How can we improve the spatial intelligence. In 6th International Conference on Applied Informatics, Eger, Hungary, 2004. 14
- [6] Jafar A Aziz, Dwi Juniati, and Pradnyo Wijayanti. Students’ reasoning with logical mathematical and visual spatial intelligence in geometry problem solving. In International Joint Conference on Science and Engineering (IJCSE 2020), pages 203–207. Atlantis Press, 2020. 14
- [7] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. 1
- [8] BAAI-RoboBrain-Team. Robobrain 2.0 technical report. arXiv preprint arXiv:2507.02029, 2025. 1, 3, 6, 7, 24, 25, 26, 27
- [9] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 6, 7, 20, 21
- [10] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, and Elad Shulman. ARKitscenes - a diverse real-world dataset for 3d indoor scene understanding using mobile RGB-d data. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1), 2021. 15
- [11] Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman Vaughan. A theory of learning from different domains. Machine Learning, 79(1-2):151–175, 2010. 3, 13
- [12] Ellis Brown, Arijit Ray, Ranjay Krishna, Ross Girshick, Rob Fergus, and Saining Xie. SIMS-V: Simulated instructiontuning for spatial video understanding. arXiv preprint arXiv:2511.04668, 2025. 1
- [13] Jeffrey Buckley, Niall Seery, and Donal Canty. Investigating the use of spatial reasoning strategies in geometric problem

- solving. International Journal of Technology and Design Education, 29(2):341–362, 2019. 13
- [14] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Danny Driess, Pete Florence, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. arXiv preprint arXiv:2401.12168, 2024. 1, 3, 7
- [15] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 513–523,

2021. 3

- [16] Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, and Xiaodan Liang. Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3313– 3323, 2022. 3
- [17] Xi Chen, Xiao Wang, Soravit Changpinyo, Anthony J Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794, 2022. 3
- [18] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 1
- [19] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Kaipeng Zhang, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2025. 6
- [20] Sijie Cheng, Zhicheng Guo, Jingwen Wu, Kechen Fang, Peng Li, Huaping Liu, and Yang Liu. Egothink: Evaluating first-person perspective thinking capability of visionlanguage models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14291–14302, 2024. 2
- [21] Douglas H Clements and Michael T Battista. Geometry and spatial reasoning. Handbook of research on mathematics teaching and learning: A project of the National Council of Teachers of Mathematics, pages 420–464, 1992. 2
- [22] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 1, 6

- [23] Yasser Dahou, Ngoc Dung Huynh, Phuc H Le-Khac, Wamiq Reyaz Para, Ankit Singh, and Sanath Narayan. Vision-language models can’t see the obvious. arXiv preprint arXiv:2507.04741, 2025. 1
- [24] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas A. Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. 2017 IEEE Conference on Computer Vision and Pattern Recognition, pages 2432–2443, 2017. 15
- [25] DeepSeek-AI. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 4
- [26] Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang Zhu, Pengchuan Zhang, Lu Yuan, Nanyun Peng, et al. An empirical study of training end-to-end vision-and-language transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18166–18176, 2022. 3
- [27] Zhiwen Fan, Jian Zhang, Renjie Li, Junge Zhang, Runjin Chen, Hezhen Hu, Kevin Wang, Huaizhi Qu, Dilin Wang, Zhicheng Yan, Hongyu Xu, Justin Theiss, Tianlong Chen, Jiachen Li, Zhengzhong Tu, Zhangyang Wang, and Rakesh Ranjan. Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction. arXiv preprint arXiv:2505.20279, 2025. 1
- [28] Usman Farooq and George Dragoi. Experience of euclidean geometry sculpts the development and dynamics of rodent hippocampal sequential cell assemblies. Nature Communications, 15(1):8417, 2024. 2
- [29] Usman Farooq and George Dragoi. Experience of euclidean geometry sculpts the development and dynamics of rodent hippocampal sequential cell assemblies. Nature Communications, 15(1):8417, 2024. 14
- [30] Jacob Feldman. What is a visual object? Trends in Cognitive Sciences, 7(6):252–256, 2003. 4, 13
- [31] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing HONG, Jianhua Han, Hang Xu, Zhenguo Li, et al. G-llava: Solving geometric problem with multi-modal large language model. In The Thirteenth International Conference on Learning Representations, 2025. 18
- [32] Howard Gardner. Frames of mind: The theory of multiple intelligences. Basic books, 2011. 1, 3
- [33] GemmaTeam, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ram´e, Morgane Rivi`ere, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025. 1
- [34] Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1, 6
- [35] Google. Gemini 2.0 flash: Model card, 2025. Model card published April 15, 2025. 6
- [36] Jeremy J. Gray. Epistemology of geometry. Stanford Encyclopaedia of Philosophy, 2013. Revised 2013-September. 2

- [37] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 1, 4, 6, 7
- [38] Kydl´ıˇcek Hynek and Gandenberger Greg. Math-verify. github repository, 2025. 2, 4, 5
- [39] Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, Yao Mu, Pengju An, et al. Robobrain: A unified brain model for robotic manipulation from abstract to concrete. arXiv preprint arXiv:2502.21257, 2025. 3, 7
- [40] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2901–2910, 2017. 3, 8, 16
- [41] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Visionand-language transformer without convolution or region supervision. In International conference on machine learning, pages 5583–5594. PMLR, 2021. 3
- [42] KimiTeam. Kimi-VL technical report. arXiv preprint arXiv:2504.07491, 2025. 1
- [43] K. M. Kyaw and T. Vid´akovich. The relationship between spatial reasoning and geometric reasoning in teachers. Eurasia Journal of Mathematics, Science and Technology Education, 21(8):em2684, 2025. 13
- [44] Brenden M Lake, Tomer D Ullman, Joshua B Tenenbaum, and Samuel J Gershman. Building machines that learn and think like people. Behavioral and Brain Sciences, 40:e253,

2017. 4, 13

- [45] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 6
- [46] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 1

- [47] Zhuowan Li, Xingrui Wang, Elias Stengel-Eskin, Adam Kortylewski, Wufei Ma, Benjamin Van Durme, and Alan L Yuille. Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14963–14973, 2023. 1, 2, 3, 7, 15, 16, 26, 27
- [48] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26679–26689, 2023. 6
- [49] Dingkun Liu, Zhu Chen, Jingwei Luo, Shijie Lian, and Dongrui Wu. Mirepnet: A pipeline and foundation model for eeg-based motor imagery classification. arXiv preprint arXiv:2507.20254, 2025. 13
- [50] Dingkun Liu, Siyang Li, Ziwei Wang, Wei Li, and Dongrui Wu. Spatial distillation based distribution alignment

- (sdda) for cross-headset eeg classification. arXiv preprint arXiv:2503.05349, 2025. 13
- [51] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 1
- [52] Yuhong Liu, Beichen Zhang, Yuhang Zang, Yuhang Cao, Long Xing, Xiaoyi Dong, Haodong Duan, Dahua Lin, and Jiaqi Wang. Spatial-ssrl: Enhancing spatial understanding via self-supervised reinforcement learning. arXiv preprint arXiv:2510.27606, 2025. 17
- [53] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan. Deepseek-vl: Towards real-world visionlanguage understanding. arXiv preprint arXiv:2403.05525,

2024. 1

- [54] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In The Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, 2021. 2, 3, 4
- [55] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations, 2024. 1, 3
- [56] Wufei Ma, Luoxin Ye, Celso M de Melo, Alan Yuille, and Jieneng Chen. Spatialllm: A compound 3d-informed design towards spatially-intelligent large multimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17249–17260, 2025. 3
- [57] Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, Song-Chun Zhu, and Siyuan Huang. Sqa3d: Situated question answering in 3d scenes. In International Conference on Learning Representations, 2023. 2
- [58] Yishay Mansour, Mehryar Mohri, and Afshin Rostamizadeh. Domain adaptation: Learning bounds and algorithms. arXiv preprint arXiv:0902.3430, 2009. 3, 13
- [59] Damiano Marsili, Rohun Agrawal, Yisong Yue, and Georgia Gkioxari. Visual agentic ai for spatial reasoning with a dynamic api. arXiv preprint arXiv:2502.06787, 2025. 1, 2, 3, 7, 15, 16, 20, 21
- [60] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365, 2025. 2, 4
- [61] Hans Moravec. Mind children: The future of robot and human intelligence. Harvard University Press, 1988. 1
- [62] Nora S Newcombe and Andrea Frick. Early education for spatial intelligence: Why, what, and how. Mind, Brain, and Education, 4(3):102–111, 2010. 14
- [63] OpenAI. Introducing gpt-5. OpenAI Blog, 2024. 1
- [64] Marios Pittalis and Constantinos Christou. Types of reasoning in 3d geometry thinking and their relation with spatial

ability. Educational Studies in mathematics, 75(2):191–212,

2010. 2

- [65] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024. 3
- [66] Runqi Qiao, Qiuna Tan, Peiqing Yang, Yanzi Wang, Xiaowan Wang, Enhui Wan, Sitong Zhou, Guanting Dong, Yuchen Zeng, Yida Xu, Jie Wang, Chong Sun, Chen Li, and Honggang Zhang. We-math 2.0: A versatile mathbook system for incentivizing visual mathematical reasoning. arXiv preprint arXiv:2508.10433, 2025. 2, 3, 4
- [67] Nova Riastuti, Mardiyana, and Ikrar Pramudya. Analysis of students geometry skills viewed from spatial intelligence. In AIP Conference Proceedings, page 020024. AIP Publishing LLC, 2017. 14
- [68] N Riastuti, M Mardiyana, and I Pramudya. Students’ errors in geometry viewed from spatial intelligence. In Journal of Physics: Conference Series, page 012029. IOP Publishing,

2017. 14

- [69] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 5, 6, 18
- [70] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient RLHF framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297, 2025. 14
- [71] Ne¸se Dokumacı S¨utc¸¨u and Beh¸cet Oral. The effects of geometrical-mechanical intelligence games on the spatial abilities. International Online Journal of Primary Education, 9(2):171–196, 2020. 14
- [72] Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 1, 6, 7, 22, 23
- [73] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025. 7
- [74] Peijie Wang, Chao Yang, Zhong-Zhi Li, Fei Yin, Dekang Ran, Mi Tian, Zhilong Ji, Jinfeng Bai, and Cheng-Lin Liu. Solidgeo: Measuring multimodal spatial math reasoning in solid geometry. arXiv preprint arXiv:2505.21177, 2025. 2, 3, 4, 5
- [75] Lilian Weng. Reward hacking in reinforcement learning. lilianweng.github.io, 2024. 5
- [76] Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. arXiv preprint arXiv:2505.23747, 2025. 1, 3, 5, 6, 7
- [77] Liangyu Xu, Yingxiu Zhao, Jingyun Wang, Yingyao Wang, Bu Pi, Chen Wang, Mingliang Zhang, Jihao Gu, Xiang Li, Xiaoyong Zhu, et al. Geosense: Evaluating identification and application of geometric principles in multimodal reasoning. arXiv preprint arXiv:2504.12597, 2025. 3

- [78] Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025. 1, 2, 3, 5, 6, 8, 14, 15, 16, 17, 22, 23, 24, 25
- [79] Rui Yang, Ziyu Zhu, Yanwei Li, Jingjia Huang, Shen Yan, Siyuan Zhou, Zhe Liu, Xiangtai Li, Shuangye Li, Wenqian Wang, et al. Visual spatial tuning. arXiv preprint arXiv:2511.05491, 2025. 18, 19
- [80] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 12–22, 2023. 15
- [81] Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, Saining Xie, Manling Li, Jiajun Wu, and Li Fei-Fei. Spatial mental modeling from limited views. arXiv preprint arXiv:2506.21458, 2025. 2, 3, 7, 14, 15, 16
- [82] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. 5
- [83] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 3
- [84] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of The Association for Computational Linguistics, 2025. 3
- [85] Huanyu Zhang, Chengzu Li, Wenshan Wu, Shaoguang Mao, Ivan Vuli´c, Zhang Zhang, Liang Wang, Tieniu Tan, Furu Wei, et al. A call for new recipes to enhance spatial reasoning in mllms. arXiv preprint arXiv:2504.15037, 2025. 1
- [86] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmmseval: Reality check on the evaluation of large multimodal models. arXiv preprint arXiv:2407.12772, 2024. 14
- [87] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 6
- [88] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System

Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. 18

[89] Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, and Yuwen Xiong. Easyr1: An efficient, scalable, multi-modality rl training framework. github repository, 2025. 14

## Euclid’s Gift: Enhancing Spatial Perception and Reasoning in Vision-Language Models via Geometric Surrogate Tasks

### Supplementary Material

###### Supplementary Contents

- A. Proof of Standard Domain-Adaptation Bound 13
- B. Evidence from Educational Psychology 13
- C. Detailed Experimental Setup 14

- C.1. Training setup . . . . . . . . . . . . . . . . 14
- C.2. Test setup . . . . . . . . . . . . . . . . . . . 14
- C.3. Prompt templates . . . . . . . . . . . . . . . 14
- C.4. Dataset Setup . . . . . . . . . . . . . . . . . 15

- D. More Experiment and Visualization 15

- D.1. More experiments about main results . . . . 15
- D.2. More experiments about ablation study . . . 15
- D.3. Comparison with Other Surrogate Tasks . . . 17

- E. More Discussion 18

- E.1. Supervised Fine-tuning (SFT). . . . . . . . . 18
- E.2. Model-Specific Performance Variations . . . 19

- F. More Visualization about Euclid30K 19

###### A. Proof of Standard Domain-Adaptation Bound

This section provides the complete proof for the standard domain-adaptation bound introduced in Sec. 3.1. We follow the same notation as in the main text.

Let h∗ ∈ H be the (ideal) hypothesis defined by

h∗ = arg min h∈H

ϵS(h) + ϵT(h) , (7)

and denote ϵideal = ϵS(h∗) + ϵT(h∗). By the definition of dH∆H, we have

ϵS(h,h∗) − ϵT(h,h∗) ≤ 21dH∆H(DS,DT). (8) Then, by the triangle inequality and Eq. (8), we have:

ϵT(h) ≤ ϵT(h∗) + ϵT(h,h∗)

= ϵT(h∗) + ϵS(h,h∗) + ϵT(h,h∗) − ϵS(h,h∗) ≤ ϵT(h∗) + ϵS(h,h∗) + ϵT(h,h∗) − ϵS(h,h∗) ≤ ϵT(h∗) + [ϵS(h) + ϵS(h∗)] + 21dH∆H(DS,DT)

= ϵS(h) + ϵideal + 12 dH∆H(DS,DT),

(9) The above proof refers to [11] and [58].

In the setting considered in this paper we examine generalisation from (i) formal geometry to broad spatial intelligence and (ii) spatial sub-skills to broader spatial intelligence. Given continued scaling of model capacity and data, along with advances in training methodologies, it is reasonable to anticipate the emergence of a sufficiently strong hypothesis h∗ for which the source-target joint errors in both regimes satisfy:

ϵS(h∗) ≈ ϵT(h∗) ≈ 0, (10)

i.e., the error magnitudes on geometry tasks and spatialintelligence tasks (and likewise on spatial sub-tasks and full spatial intelligence) become simultaneously negligible. When this asymptotic condition holds for both sides of our comparisons, the corresponding ideal terms ϵideal in each bound are effectively 0, making their omission justified. Consequently, Eq. (9) simplifies to:

ϵT(h) ≲ ϵS(h) + 21dH∆H(DS,DT), (11)

Therefore, if dH∆H(DS,DT) is sufficiently small, the population gap ϵT(h)−ϵS(h) also becomes small, which allows the source distribution to serve as a reliable surrogate for the target distribution [1, 49, 50].

###### B. Evidence from Educational Psychology

Complementing the domain-adaptation view in the previous subsection, we now present evidence from educational psychology that echoes the cognitive-science perspective on the generality of geometric knowledge in perception and reasoning [30, 44].

There is extensive evidence in educational psychology that geometry problem solving is closely related to spatial intelligence, can serve as an informative indicator of spatial ability, and can be used to improve it through targeted practice.

First, numerous correlational studies document a substantive link between geometric and spatial reasoning. Kyaw and Vid´akovich report a moderate positive correlation between teachers’ geometric and spatial reasoning (r = 0.47), with 3D matching and measurement tasks predicting spatial scores [43]. In STEM and graphical education, higher spatial ability is associated with better problemsolving performance and more effective strategies [13]. Newcombe and Frick emphasize that spatial representations and transformations are central cognitive resources that support reasoning in domains that are not obviously

spatial—for example, through the use of graphs and diagrams [62].

Second, several studies show that performance on geometry tasks is a sensitive proxy for spatial ability. Analyses of middle-school students reveal that geometry skills and error patterns systematically vary with spatial-intelligence levels [67, 68]. Differences in dominance between logicalmathematical and visual-spatial intelligence yield distinct pathways for geometric reasoning, further tying geometry problem solving to spatial constructs [6]. These results support the use of geometry assessments as indicators of students’ spatial proficiency.

Third, intervention studies demonstrate that providing structured geometric activities can improve spatial intelligence. Programmatic practice with polyhedra and computer-generated spatial problems yields measurable gains [5]. Geometrical-mechanical intelligence games, implemented in quasi-experiments with pre/post testing, significantly enhance spatial visualisation and spatial relations skills [71]. Overall, the balance of evidence indicates that well-designed geometric practice is an effective means to cultivate spatial abilities.

Moreover, neuroscience research reveals that early exposure to Euclidean geometric structures fundamentally shapes spatial representations. Studies examining hippocampal activity in rodents reared in spherical versus cuboid environments demonstrate that experience with canonical Euclidean features (edges, corners, planes) enriches the repertoire of preconfigured neuronal patterns and enhances the brain’s ability to discriminate between distinct spatial layouts [29]. While these findings originate from animal models, they provide convergent biological evidence that geometric experience during development can refine spatial coding mechanisms—a principle that may extend to learning systems more broadly.

Taken together, these findings motivate our surrogatetask choice. Our results suggest that the same relationship generalises beyond human learners to large multimodal models: training on formal geometry induces domain-invariant structure that transfers to diverse spatialintelligence benchmarks. This observation is consistent with the domain-adaptation analysis in the previous subsection and provides an educational-psychology rationale for our geometry-first curriculum.

###### C. Detailed Experimental Setup

This section summarises the key hyperparameters, evaluation settings, prompt templates, and datasets settings used throughout the paper.

###### C.1. Training setup

In this paper, we follow the default settings of VeRL [70] and EasyR1 [89] to train the Qwen2.5-VL series, Qwen3-

VL series, and the RoboBrain2.0 series. Specifically, we train for 10 epochs in 64 NVIDIA H100 GPUs using Adam optimizer with a learning rate of 1 × 10−6 and a weight decay of 1 × 10−2. In GRPO, we perform 8 rollouts per question and set the default sampling temperature to 1. The KL divergence coefficient β in Eq. 5 is set to 1 × 10−2.

Unless stated otherwise, we fix the random seed at 1 to guarantee determinism. We adopt a context window of 1024 tokens for both the prompt and the response, and use a rollout batch of 512 samples. The actor network updates with a global batch size of 128 and a maximum gradient norm of 1.0. Images are resized so that the total pixel count lies between 512 × 512 and 2048 × 2048. All remaining hyper-parameters, including PPO clip ratio, learningrate schedule, and parallelism settings, follow the default EasyR1 recipe and can be found in our GitHub repository.

###### C.2. Test setup

Inference is conducted with the lmms-eval toolkit [86] to ensure consistent decoding across models. In the test, to ensure the reproducibility of the results, we follow VSIBench [78] and MindCube [81] to set the temperature to 0. Finally, to ensure that the model performs sufficient spatial inference, we set the maximum generation length of model responses at 1024 tokens. For reproducibility, detailed testing scripts are provided in our GitHub repository.

###### C.3. Prompt templates

Euclid-tuned models. During both training and evaluation, we use the following template:

###### Euclid-tuned Models Prompt Template

You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think></think> tags. The final answer MUST BE put in \boxed{}.

Baseline variants. RoboBrain2.0 expects the answer inside < answer>< /answer> tags; we therefore replace the last line with, like:

###### Vanilla RoboBrain2.0 Prompt Template

You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think></think> tags. The final answer MUST BE put in <answer></answer>.

Because Qwen2.5VL-Instruct and Qwen3VL-Instruct was tuned with supervised instruction data that often begins

with phrases like “think step by step,” keeping the same cue in your evaluation prompt aligns the test-time input with the style encountered during training. This consistency helps the model interpret the prompt as intended and reduces the risk of unexpected formatting effects.

###### Vanilla QwenVL-Instruct Prompt Template

You FIRST think step by step and then provide the final answer. The final answer MUST BE put in \boxed{}.

During evaluation, to mitigate prompt-template bias, we run each model with both its native instruction-style prompt and the unified Euclid reasoning template, and report the better-performing variant. When processing VSI-Bench tasks, we make minor adjustments to the above templates to follow the benchmark’s original settings as closely as possible and ensure consistent results—for example, we prepend “These are frames of a video.” to every prompt.

###### C.4. Dataset Setup

In this subsection, we provide an introduction and configuration details for the dataset used in the main page.

Setup in VSI-Bench. VSI-Bench [78] contains more than 5,130 egocentric videos question-answer pairs sourced from ARKitScenes[10], ScanNet [24], and ScanNet++ [80]. The task types are divided into numerical question tasks (e.g., object counting, absolute distance estimation, object size estimation, and room size estimation) and multiple choice tasks (e.g., relative distance estimation, relative direction reasoning, route planning, and spatiotemporal appearance-order). For the evaluation metrics, we align with the VSIBench setting. In addition, for the Qwen2.5VL-series and RoboBrain2.0-series, we use 32 frames uniformly sampled from the scene video as input frames in the inference process.

Setup in Super-CLEVR and Omni3D-Bench. Super-CLEVR [47] contains a 5,000-image test split that probes how well a model handles changes in visual complexity, concept distribution, and composition, making it a strong measure of two-dimensional spatial reasoning. Omni3D-Bench [59] adds 500 questions to the Omni3D dataset, each requiring a model to locate objects in three-dimensional space and estimate their relative distances and sizes. Together, these benchmarks test both planar and volumetric aspects of spatial understanding, providing complementary evidence of a model’s geometric competence. For the evaluation metrics, we follow the settings of VSIBench [78]. Specifically, we calculate mean relative accuracy (MRA) across confidence thresholds C = {0.5,0.55...,0.95} for the numerical question tasks and report exact-match accuracy for multiple-choice tasks.

Setup in MindCube. MindCube [81] is a recent benchmark crafted to scrutinize the spatial-reasoning capabilities of VLMs under partial observability and dynamic viewpoints, challenging the VLM to maintain object consistency across viewpoints and to reason about occluded or invisible elements. MindCube defines three canonical camera trajectories: Rotation (camera stays in place but rotates to look around; 1,081 samples), Around (camera moves around objects in a circular path; 1,869 samples), and Among (camera moves among objects in a circular path; 18,204 samples). Since all questions follow a multiple-choice format, we evaluate models by exact-match accuracy between the predicted option and the ground-truth answer.

###### D. More Experiment and Visualization

###### D.1. More experiments about main results

To present the quantitative gains more intuitively, Fig. 4 plots the base models and their Euclid30K-tuned counterparts side by side. The light bars show consistent accuracy improvements on Super-CLEVR [47], Omni3DBench [59], VSI-Bench [78], and MindCube [81], confirming that a compact geometry curriculum injects transferable spatial priors across both Qwen2.5VL, Qwen3VL and RoboBrain2.0 families. Additionally, we include results for Qwen2.5VL-7B and Qwen2.5VL-32B in the Fig. 4, which exhibit consistent improvements across all four benchmarks following the same geometric surrogate-task training, reaffirming that geometry serves as an effective surrogate task for spatial intelligence and further validating the robustness and generality of this approach.

Beyond the aggregate accuracy gains, Figures5–12 qualitatively illustrate how geometry tuning alters intermediate reasoning. After Euclid30K training the model produces more coherent multi view descriptions (Fig.7), applies geometric similarity relations correctly (Fig.10), uses quadrant and cardinal directional cues with fewer ambiguities (Fig.8, Fig.11), and leverages perspective driven size cues (near–far size scaling) more systematically (Fig.12). It also shows clearer distance estimation chains (Fig.5, Fig.9), improved object size estimation (Fig.6, Fig.12), more reliable counting with cross view consistency (Fig. 7), and fewer heuristic shortcuts (e.g., premature guesses without spatial justification). These qualitative traces are consistent with the quantitative improvements: they suggest the model has internalized foundational Euclidean principles (similarity, proportionality, relative position, viewpoint coherence) and can deploy them across distinct downstream spatial tasks.

###### D.2. More experiments about ablation study

Tab. 6 presents a detailed analyses of the ablation study results on VSI-Bench [78], complementing the summary in Sec. 4.4. To isolate the contribution of Euclid30K from po-

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

| | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

Figure 4. Performance improvement on SuperClevr [47], Omni3DBench [59], VSIBench [78], and MindCube [81] after the model has been trained on Eculid30K.

tential confounds such as additional training data or GRPOinduced generalization, we trained each model variant on a 30K-sample subset of the spatial-intelligence dataset ClevrCoGenT [40] using identical hyperparameters, rollout budgets, and training epochs.

Overall Performance. Across all model families, Euclid30K training consistently yields higher overall accuracy than Clevr-CoGenT training. For instance, Qwen2.5VL3B improves from 29.2% (base) to 32.0% with Euclid30K versus 31.3% with Clevr-CoGenT; Qwen3VL-30B-A3B rises from 40.0% to 45.8% with Euclid30K versus 44.2% with Clevr-CoGenT; and RoboBrain2.0-32B jumps from 43.1% to 49.6% with Euclid30K versus 46.7% with ClevrCoGenT. These results indicate that Euclid30K provides a more transferable spatial foundation than an equal volume of non-geometric spatial data.

Task-Specific Patterns. Clevr-CoGenT training yields targeted improvements on tasks closely aligned with its original design, such as object counting and relative direction. For example, Qwen2.5VL-3B trained on Clevr-CoGenT achieves 40.5% on object counting, slightly outperforming Euclid30K training (38.3%), and similar patterns appear for relative direction in several variants. Conversely, Euclid30K training produces broader gains across multiple categories, particularly on size-estimation tasks (like object size and absolute distance estimation), where geometric rea-

soning principles directly apply. Qwen2.5VL-32B trained on Euclid30K reaches 55.8% on object size versus 44.3% with Clevr-CoGenT, and 46.0% on room size versus 40.8%.

This ablation study reveals a fundamental trade-off: task-specific spatial datasets enhance performance on related categories, whereas structured geometric datasets provide a more general-purpose spatial reasoning foundation that transfers more uniformly across diverse task types. Euclid30K’s strength lies in instilling first-principles geometric knowledge—distance, proportion, similarity, and spatial relations—that applies broadly, even to tasks not explicitly represented in the training corpus. The pattern holds across Qwen2.5VL, Qwen3VL, and RoboBrain2.0 families, and across parameter scales ranging from 3B to 72B, from dense to MoE. This consistency suggests that the advantage conferred by geometric surrogate tasks is not an artifact of a particular architecture or capacity regime, but reflects a more general principle: formal Euclidean reasoning offers a robust substrate for spatial intelligence.

In summary, Tab. 6 substantiates the claim that geometry serves as an effective surrogate task for spatial intelligence, with Euclid30K training delivering superior overall performance and broader generalization compared to equal-sized non-geometric spatial datasets.

Numerical Question Multiple-Choice Question

Methods

Overall Obj. Cnt. Abs. Dist. Obj. Size Room Size Rel. Dist. Rel. Dir. Route Plan Appr. Order

Qwen2.5VL-series

Qwen2.5VL-3B 35.6 23.4 34.9 16.6 34.4 40.7 26.3 21.8 29.2 Qwen2.5VL-Space-3B 40.5 24.5 30.1 29.8 33.9 43.0 29.9 18.8 31.3 Qwen2.5VL-Euclid-3B 38.3 26.8 35.4 22.2 37.0 43.2 36.6 16.3 32.0

Qwen2.5VL-7B 39.5 17.8 16.9 5.8 33.8 36.7 24.7 22.8 24.8 Qwen2.5VL-Space-7B 42.4 17.8 24.5 9.2 36.7 38.5 29.4 23.8 27.8 Qwen2.5VL-Euclid-7B 38.8 22.8 37.3 23.2 38.3 38.5 25.8 26.5 31.4

Qwen2.5VL-32B 22.4 27.0 37.9 38.0 39.4 40.1 33.5 41.3 34.9 Qwen2.5VL-Space-32B 45.6 25.9 44.3 40.8 42.0 39.3 28.4 35.3 37.7 Qwen2.5VL-Euclid-32B 38.7 30.9 55.8 46.0 43.7 37.1 34.0 33.2 39.9

- Qwen2.5VL-72B 13.6 19.6 40.9 41.1 37.7 35.3 34.0 36.2 32.3 Qwen2.5VL-Space-72B 15.6 24.8 40.7 41.4 43.4 37.8 29.4 33.5 33.2 Qwen2.5VL-Euclid-72B 22.5 27.2 55.7 43.3 44.9 37.1 32.5 36.6 37.5

- Qwen3VL-series

Qwen3VL-4B 28.5 33.0 32.6 43.5 40.3 40.0 33.0 33.2 35.5 Qwen3VL-Space-4B 32.1 37.1 44.7 51.0 49.3 43.8 37.6 38.5 41.8 Qwen3VL-Euclid-4B 33.3 37.4 49.5 48.3 46.5 46.3 34.0 42.9 42.3

Qwen3VL-30B-A3B 27.4 32.3 53.6 44.0 42.1 36.2 35.5 48.9 40.0 Qwen3VL-Space-30B-A3B 29.9 36.8 58.0 43.4 47.9 49.1 35.9 52.7 44.2 Qwen3VL-Euclid-30B-A3B 33.5 37.5 64.1 43.1 48.7 49.6 35.6 54.4 45.8

RoboBrain2.0-series

RoboBrain2.0-7B 46.0 32.7 58.9 35.9 45.9 41.5 30.9 55.2 43.0 RoboBrain2.0-Space-7B 66.4 34.4 65.8 41.0 46.6 46.5 36.6 52.3 48.7 RoboBrain2.0-Euclid-7B 66.4 36.9 66.3 40.5 48.3 45.3 35.6 57.8 49.6

RoboBrain2.0-32B 50.5 37.0 59.2 28.4 43.2 46.1 34.5 39.5 43.1 RoboBrain2.0-Space-32B 58.0 36.9 62.2 47.8 46.9 44.5 34.0 42.1 46.7 RoboBrain2.0-Euclid-32B 59.2 39.4 63.4 47.8 48.7 47.5 33.5 57.0 49.6

Table 6. Ablation experiment on VSI-Bench [78]. We compare training a model on a 30K subset of the spatial intelligence dataset Clevr-CoGenT v.s. the geometric dataset Euclid30K to verify that the geometric dataset serves as a surrogate task to improve the spatial intelligence capabilities of the model. Bolding indicates the best score within each model type.

Numerical Question Multiple-Choice Question

Methods

Overall Obj. Cnt. Abs. Dist. Obj. Size Room Size Rel. Dist. Rel. Dir. Route Plan Appr. Order

Qwen2.5VL-7B 39.5 17.8 16.9 5.8 33.8 36.7 24.7 22.8 24.8 Spat1-SSRL-7B [52] 37.1 24.6 22.9 17.0 36.3 35.1 33.5 27.3 29.2 Qwen2.5VL-Euclid-7B 38.8 22.8 37.3 23.2 38.3 38.5 25.8 26.5 31.4

Table 7. Comparisons with Spat1-SSRL-7B on VSI-Bench. Bolding indicates the best score within each model type.

Methods SuperClevr Omni3DBench MindCube

Qwen2.5VL-7B 76.1 28.3 30.0 Spat1-SSRL-7B [52] 76.3 33.1 30.6 Qwen2.5VL-Euclid-7B 86.2 31.1 31.1

- Table 8. Comparisons with Spat1-SSRL-7B on SuperClevr, Omni3D Bench, and MindCube. Bolding indicates the best score within each model type.

###### D.3. Comparison with Other Surrogate Tasks

Beyond our geometry-first curriculum, several studies explore alternative surrogate tasks to strengthen spatial intelligence. In this subsection, we compare against a representative approach, Spatial-SSRL [52]. Spatial-SSRL proposes a self-supervised pipeline that derives verifiable signals from ordinary RGB/RGB-D images via five pretext tasks capturing 2D/3D spatial structure: shuffled patch re-

Numerical Question Multiple-Choice Question

Methods

Overall Obj. Cnt. Abs. Dist. Obj. Size Room Size Rel. Dist. Rel. Dir. Route Plan Appr. Order

Qwen3VL-30B-A3B 27.4 32.3 53.6 44.0 42.1 36.2 35.5 48.9 40.0 + GeometricSFT 29.6 33.8 56.9 39.1 46.8 47.8 35.1 60.2 43.7

- Table 9. Evaluation on VSI-Bench. + GeometricSFT indicate the Qwen3VL-30B-A3B trained with SFT on the Geo170K dataset [31]. Bolding indicates the best score within each model type.

Methods

Numerical Question Multiple-Choice Question

Overall Obj. Cnt. Abs. Dist. Obj. Size Room Size Rel. Dist. Rel. Dir. Route Plan Appr. Order

Qwen2.5VL-7B 39.5 17.8 16.9 5.8 33.8 36.7 24.7 22.8 24.8 VST-7B [79] 56.5 35.1 69.0 53.3 50.1 11.7 28.4 50.3 44.3 VST-Euclid-7B 66.4 38.4 74.2 60.1 56.5 48.6 41.8 57.8 55.5

- Table 10. Evaluation on VSI-Bench . VST-Euclid indicate the VST [79] trained with GRPO [69] on the Euclid30K dataset. Bolding indicates the best score within each model type.

Methods SuperClevr Omni3DBench MindCube

Qwen3VL-30B-A3B 64.1 36.7 39.8 + GeometricSFT 66.5 40.5 38.3

- Table 11. Evaluation on SuperClevr, Omni3D Bench, and MindCube. + GeometricSFT indicate the Qwen3VL-30B-A3B trained with SFT on the Geo170K dataset [31]. Bolding indicates the best score within each model type.

zq

Methods SuperClevr Omni3DBench MindCube

Qwen2.5VL-7B 76.1 28.3 30.0 VST-7B [79] 83.1 36.5 35.5 VST-Euclid-7B 86.3 37.1 34.8

- Table 12. Comparisons with Spat1-SSRL-7B on SuperClevr, Omni3D Bench, and MindCube. VST-Euclid indicate the VST [79] trained with GRPO [69] on the Euclid30K dataset. Bolding indicates the best score within each model type.

###### Category Parameter Value

Dataset image max pixels 262144 Dataset video max pixels 16384 Dataset cutoff len 8192

Method finetuning type lora Method lora rank 8 Method lora target all

Train per device train batch size 4 Train gradient accumulation steps 4 Train learning rate 1e-4 Train num train epochs 1 Train lr scheduler type cosine Train weight decay 0.01 Train warmup ratio 0.1 Train bf16 true

Table 13. Supervised Fine-tuning (SFT) configuration in LLaMAFactory framework [88].

ordering, flipped patch recognition, cropped-patch inpainting, regional depth ordering, and relative 3D position prediction. The released configuration uses 81K image-QA pairs to train a 7B model with RLVR.

Tab. 7 compares a 7B backbone trained on our 30K Euclid30K geometry corpus (RLVR, identical decoding as Sec. C) with Spatial-SSRL-7B on VSI-Bench. Euclid30K achieves a higher overall score (31.4 vs. 29.2) with substantially fewer training samples (30K vs. 81K; 63% fewer). Gains concentrate on size- and distance-estimation categories where geometric constraints are directly applicable, while Spatial-SSRL shows stronger performance on route planning and appearance order. Cross-benchmark results in

Tab. 8 show that Euclid30K improves Super-CLEVR and remains competitive on Omni3D-Bench and MindCube under the same inference protocol.

###### E. More Discussion

###### E.1. Supervised Fine-tuning (SFT).

For mathematical problems, collecting step by step solution annotations is substantially more expensive than answer only labels. Accordingly, Euclid30K does not currently include process annotations, which makes it less suited to SFT routines that rely on explicit reasoning traces. Nevertheless, geometry QA can still serve as a surrogate task for

spatial intelligence under SFT. In this subsection, in order to show that geometry can as a surrogate task for spatial intelligence not only under reinforcement learning but also under supervised fine tuning, we conduct SFT on Geo170K, a dataset that supplies explicit reasoning trajectories. Providing these intermediate steps helps the model internalize Euclidean principles.

As shown in Tab. 9, Qwen3VL-30B-A3B improves on VSI Bench from 40.0 to 43.7 overall after Geo170K SFT (the gain is smaller than the 40.0 to 45.8 increase achieved with Euclid30K RL). In Tab. 11, cross benchmark results also rise on SuperCLEVR (64.1 to 66.5) and Omni3D Bench (36.7 to 40.5), while MindCube shows a modest decrease (39.8 to 38.3). The decline on MindCube is plausibly due to the composition of Geo170K: most problems are plane geometry items and provide little direct supervision for viewpoint change or three dimensional mental imagery. Even with this limitation, the consistent gains on the majority of spatial benchmarks reinforce our claim that solving geometry problems is an effective surrogate task for spatial intelligence.

###### E.2. Model-Specific Performance Variations

As noted in Sec. 5.2, we hypothesize that the performance gains after geometry tuning are model specific and depend on capabilities inherited from earlier training data. In this subsection, we use VST-7B-RL [79] as a representative case to test this hypothesis. VST-7B first applies supervised fine-tuning on the VST-P dataset with 4.1M samples that cover 19 spatial skills across single-image, multi-image, and video. It then uses reinforcement learning on the VSTR dataset with 135K reasoning samples. We start from the released VST-7B-RL checkpoint, add Euclid30K training with GRPO and keep decoding and evaluation identical those to used in the rest of this appendix. Results in Tab. 10 show clear gains on VSI Bench. The overall score moves from 44.3 to 55.5. Tab. 12 shows small improvements on SuperCLEVR and Omni3DBench, and a small decrease on MindCube from 35.5 to 34.8.

A reasonable explanation is that the VST dataset has limited coverage of viewpoint changes and spatiotemporal consistency. The samples of ”camera motion” and ”camera-camera” combined account for only about 3% of the total dataset. However, MindCube heavily relies on these patterns. Meanwhile, the plane geometry problems in Euclid30K do not include explicit supervision signals for viewpoint transformations. This suggests that the role of Euclid30K is to formalize and reinforce the spatial concepts already present in the model, leading to significant improvements in tasks such as metric estimation and relational reasoning. However, when certain key concepts (such as robust viewpoint transformation ability) are weakly represented in the pretraining, geometric fine-tuning may offer little bene-

fit and might even slightly interfere with the model’s existing heuristic strategies. These support our hypothesis that models with richer prior spatial concepts obtain larger gains from Euclid30K, while gaps in prior coverage limit or occasionally reduce transfer.

###### F. More Visualization about Euclid30K

Fig. 13–15 provide additional Euclid30K examples, illustrating diversity in problem types, diagram styles, and reasoning complexity. These samples complement the main paper by showing the range of geometric configurations and textual formulations present in the Euclid30K dataset.

Question: Which object is closer to the camera: the washing machine or the kettle?

[Figure 24]

Options: Washing Machine, Kettle GT: Washing Machine

Qwen2.5VL Response: To determine which object is closer to the camera, let's analyze the image step by step:

- 1. Identify the washing machine: It is located in the lower part of the image, partially under the counter.
- 2. Identify the kettle: It is located on the counter, to the right of the washing machine.
- 3. Compare the positions: The kettle is positioned on the counter, which is closer to the camera than the washing machine, which is partially hidden under the counter. Therefore, the object closer to the camera is the kettle. Qwen2.5VL Answer: Kettle

Qwen2.5VL -Eculid Response: To determine which object is closer to the camera, let's analyze the image:

- 1. The washing machine is positioned in the foreground, directly in front of the camera.
- 2. The kettle is placed on the countertop, behind the washing machine. Since the washing machine is in front of the kettle and closer to the camera, it is the object that is closer to the camera. Final answer: washing machine Qwen2.5VL Answer: Washing Machine

- Figure 5. The response and final answer for Qwen2.5VL-7B [9] and Qwen2.5VL-Eculid-7B in Omni3DBech [59].

[Figure 25]

Question: How many objects of the volume of the bedside table can you fit in an object with the volume of the bed?

GT: 15.407

|Qwen2.5VL Response: To determine how many objects of the volume of the bedside table can fit in an object with the volume of the bed, we need to compare the volumes of the two objects.<br><br>1. Volume of the Bed: The bed appears to be a standard double bed, which typically has dimensions of approximately 1.5 meters by 2 meters. Assuming the height of the bed is about 0.2 meters, the volume of the bed can be calculated as：Volume of the bed = 1.5 m × 2 m × 0.2 m = 0.6 m3<br>2. Volume of the Bedside Table: The bedside table appears to be a standard nightstand, which typically has dimensions of approximately 0.5 meters by 0.5 meters by 0.5 meters. The volume of the bedside table can be calculated as: Volume of the bedside table = 0.5 m × 0.5 m × 0.5 m = 0.125 m3<br>3. Comparison: To find out how many bedside tables can fit into the bed, we divide the volume of the bed by the volume of the bedside table:<br><br><br>Volume of the bed<br><br>Volume of the bedside table<br><br>=<br><br>0.6 m3<br><br>0.125 m3<br><br>= 4.8 Since we are looking for a whole number of bedside tables that can fit, we round down to the nearest whole number. Qwen2.5VL Answer: 4 (MRA:0) Qwen2.5VL -Eculid Response: To determine how many objects of the volume of the bedside table can fit in an object with the volume of the bed, we need to compare the volumes of the two objects.<br><br>1. Volume of the Bed: The bed appears to be a standard double bed, which typically has a volume of around 1.5 cubic meters (assuming standard dimensions and a thickness of the mattress).<br>2. Volume of the Bedside Table: The bedside table appears to be a small, rectangular table. Estimating its dimensions, it might be about 0.5 meters in length, 0.4 meters in width, and 0.5 meters in height, giving it a volume of approximately 0.1 cubic meters.<br>3. Comparison: To find out how many bedside tables can fit into the bed, we divide the volume of the bed by the volume of the bedside table:<br><br><br>1.5 cubic meters 0.1 cubic meters<br><br>= 15. Therefore, the number of bedside tables that can fit into the bed is 15.<br><br>Qwen2.5VL Answer: 15 (MRA:1)|
|---|

- Figure 6. The response and final answer for Qwen2.5VL-7B [9] and Qwen2.5VL-Eculid-7B in Omni3DBech [59].

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Question: How many table(s) are in this room? GT: 3

Qwen3VL Response: Looking at the frames, I can identify several tables:

- 1. In the initial frames, there is a small wooden table next to a doorway, with a lamp and a phone on it.
- 2. Later, in the living room, there is a larger wooden coffee table in the center of the room.
- 3. Also in the living room, there is a small wooden side table next to the brown leather ottoman, which has a lamp and a phone on it.
- 4. There is also a small wooden side table next to the Christmas tree, which has a phone on it. Therefore, there are 4 tables visible in the room. Qwen3VL Answer: 4

Qwen3VL-Eculid Response: Looking at the frames, I can identify several tables:

- 1. In the initial frames, there is a small wooden table next to the doorway, with a lamp and a phone on it.
- 2. Later, in the living room, there is a larger wooden coffee table in the center of the room.
- 3. Also in the living room, there is a small wooden side table next to the brown leather ottoman, which has a lamp and a phone on it (this is the same table seen earlier near the doorway).
- 4. There is also a small wooden side table next to the Christmas tree, which has a small item on it.\n\nTherefore, there are at least 3 distinct tables visible in the room: the coffee table, the side table next to the ottoman, and the side table next to the Christmas tree. Qwen3VL-Eculid Answer: 3

Figure 7. The response and final answer for Qwen3VL-4B [72] and Qwen3VL-Eculid-4B in VSIBench [78].

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Question: If I am standing by the door and facing the tv, is the trash bin to my front-left, front-right, back-left, or back-right? The directions refer to the quadrants of a Cartesian plane (if I am standing at the origin and facing along the positive y-axis).",

Options: A. back-right, B. front-right, C. back-left, D. front-left GT: D

Qwen3VL Response: Let's analyze the scene step by step:

- 1. Identify the starting position: You are standing by the door. From the frames, the door is on the right side of the room, near the red wall and the window. So, you are on the right side of the room.
- 2. Determine the direction you are facing: You are facing the TV. The TV is mounted on the white wall, which is opposite the red wall where the door is. Therefore, you are facing the white wall.
- 3. Locate the trash bin: The trash bin is visible in the frames near the window, which is on the far side of the room, next to the door. It is positioned in the corner where the red wall and the window/wall meet.
- 4. Determine the relative position: Since you are standing by the door (on the right side of the room) and facing the TV (which is on the opposite wall), you are facing the white wall. The trash bin is in the corner on the far side of the room, which is to your left and behind you. From your perspective, it is in the back-left quadrant. Therefore, the trash bin is to your back-left. The correct option is C. Qwen3VL Answer: C

Qwen3VL-Eculid Response: Let's analyze the scene step by step:

1. Identify the door and the TV: The door is visible in the frame, and the TV is mounted on the wall. The person is standing by the door.\n2. Determine the facing direction: The question asks what is to the front-left, front-right, etc., when facing the TV. So, we need to determine the position of the trash bin relative to the TV from the perspective of someone standing at the door and facing the TV.

- 3. Locate the trash bin: The trash bin is visible in the frame, positioned near the wall, to the left of the TV from the viewer‘s perspective.
- 4. Determine the relative position: If a person is standing by the door and facing the TV, they are facing the wall where the TV is mounted. The trash bin is located to the left of the TV. Therefore, from the person's perspective, the trash bin is in the front-left direction. The final answer is D. Qwen3VL-Eculid Answer: D

Figure 8. The response and final answer for Qwen3VL-30B-A3B [72] and Qwen3VL-Eculid-30B-A3B in VSIBench [78].

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Question: Measuring from the closest point of each object, which of these objects (heater, trash can, door, cup) is the closest to the microwave?

Options: A. heater, B. trash can, C. door, D. Cup GT: D. Cup

RoboBrain2.0 Response: In the visual input, the microwave is positioned on the countertop, slightly to the right of the center.

The heater is located on the left side of the countertop, near the sink. The trash can is not visible in the

image, suggesting it is not in the immediate vicinity of the microwave. The door is visible on the far left side of the frame, indicating it is not near the microwave. The cup is on the countertop, closer to the microwave than the heater, but not as close as the heater itself.

To determine which object is closest to the microwave, I focus on the spatial arrangement of the heater and the cup. The heater is positioned directly adjacent to the microwave, with no visible gap between them. The cup, while also on the countertop, is further away from the microwave compared to the heater.

Therefore, based on the visual analysis, the heater is the closest object to the microwave. The heater is directly adjacent to the microwave, making it the closest object in terms of proximity. Thus, the answer is A. heater

###### RoboBrain2.0 Answer: A. heater RoboBrain2.0-Eculid Response:

In the visual scene, the microwave is positioned on the countertop, slightly to the right of the center. The heater is located on the left side of the countertop, near the sink. The trash can is not visible in the image, suggesting it is not in the immediate vicinity of the microwave. The door is visible in the background, slightly to the right of the microwave. The cup is on the countertop, near the microwave, and appears to be the closest object to the microwave.

To determine which object is closest to the microwave, I focus on the spatial arrangement of these objects. The cup is directly on the countertop, adjacent to the microwave, indicating it is the closest object to the microwave. The heater is on the left side of the countertop, further away from the microwave. The door is in the background, suggesting it is not the closest. The trash can, if present, would be further away from the microwave than the cup. Therefore, based on the visual analysis of the scene and the spatial proximity of the objects to the microwave, the object closest to the microwave is the cup. This aligns with option D. Thus, the answer is: D.

###### RoboBrain2.0 Answer: D. Cup

Figure 9. The response and final answer for RoboBrain2.0-7B [8] and RoboBrain2.0-Eculid-7B in VSIBench [78].

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Question: What is the length of the longest dimension (length, width, or height) of the pillow, measured in centimeters?

###### GT: 73

RoboBrain2.0 Response: In the visual input, the pillow is prominently visible on the bed, positioned towards the center of the frame. The pillow is covered with a striped duvet, which provides a clear visual cue for its approximate size. The pillow appears to be elongated, with a noticeable length extending horizontally across the bed. To determine the length of the pillow, I focus on its visible dimensions relative to the bed and other objects in the scene. The pillow's length is visually estimated by comparing it to the bed's width and the visible portion of the bed's length. The pillow's length seems to be approximately 100 cm, as it

###### spans a significant portion of the bed's width. This estimation is based on the pillow's apparent

size relative to the bed's dimensions and the visible length of the bed itself. Therefore, the length of the pillow, measured in centimeters, is approximately 100 cm.

###### RoboBrain2.0 Answer: 100 (MRA: 0.3) RoboBrain2.0-Eculid Response:

From the visual input, the pillow is prominently placed on the bed, with its distinct rectangular shape and soft texture. The pillow's dimensions are visually apparent, with its length extending horizontally across the bed. The pillow's height is relatively low, typical of a standard pillow, and its width spans a

significant portion of the bed's surface.

To determine the pillow's length, I focus on its horizontal span across the bed. The pillow's length is visually estimated by comparing it to the bed's width and the visible portion of the bed's surface. The pillow appears to extend approximately 70 centimeters along the bed's length, which is consistent with the typical dimensions of a standard pillow.

The pillow's length is the primary dimension of interest, as the task specifies measuring the longest dimension. The pillow's width and height are not the focus here, as the task specifies the length. Therefore, based on the visual analysis of the pillow's dimensions in the scene, the pillow's length is approximately 70 centimeters. This conclusion is supported by the pillow's visible horizontal span across the bed, which aligns with the typical dimensions of a standard pillow.

###### RoboBrain2.0 Answer: 70 (MRA: 1.0)

Figure 10. The response and final answer for RoboBrain2.0-7B [8] and RoboBrain2.0-Eculid-7B in VSIBench [78].

[Figure 46]

Question: Is the number of big yellow objects to the left of the rubber car the same as the number of motorbikes that are right of the regular bus? True or False.

GT: False

###### RoboBrain2.0 Response:

In the visual input, I focus on identifying the big yellow objects to the left of the rubber car. The rubber car is a prominent yellow vehicle located centrally in the scene. To its left, there is a large yellow bicycle, which is a big yellow object. This bicycle is positioned prominently to the left of the rubber car.

Next, I identify the motorbikes in the scene. There are two motorbikes visible: one is a red motorbike

###### located to the left of the yellow bicycle, and the other is a gold motorbike positioned to the right of the yellow bicycle. The gold motorbike is the one that is right of the regular bus, which is a blue bus located on the far right side of the scene.

Now, I compare the number of big yellow objects to the left of the rubber car with the number of motorbikes to the right of the regular bus. There is one big yellow object (the bicycle) to the left of the rubber car, and there is one motorbike (the gold one) to the right of the regular bus. Therefore, the number of big yellow objects to the left of the rubber car is equal to the number of motorbikes to the right of the regular bus. This leads to the conclusion that the statement is True.

RoboBrain2.0 Answer: True RoboBrain2.0-Eculid Response: To determine the number of big yellow objects to the left of the rubber car, I first identify the rubber car, which is a yellow minivan located centrally in the image. To its left, I observe a large yellow bicycle. This bicycle is a prominent, big yellow object. Next, I focus on the motorbikes in the scene. There are two motorbikes visible: one is a large yellow motorcycle positioned to the right of the minivan, and the other is a smaller red motorbike located to the left of the minivan. Now, I compare the number of big yellow objects to the left of the rubber car with the number of motorbikes to the right of the regular bus. The big yellow object to the left of the minivan is the bicycle. The motorbikes are positioned to the left of the minivan, not to the right of the bus.

Therefore, the number of big yellow objects to the left of the rubber car (the bicycle) is not the same as the number of motorbikes to the right of the regular bus. The bicycle is not on the right side of the bus, and the motorbikes are not on the right side of the bus either. Thus, the answer is: False.

###### RoboBrain2.0 Answer: False

Figure 11. The response and final answer for RoboBrain2.0-7B [8] and RoboBrain2.0-Eculid-7B in SuperCLEVR [47].

[Figure 47]

Question: What number of objects are tiny cyan rubber things or

metallic objects that are on the left side of the bus?

###### GT: 3

RoboBrain2.0 Response: In the visual input, I observe several objects, including motorcycles, a bus, and airplanes. The task requires identifying objects that are either tiny cyan rubber things or metallic objects located on the left side of the bus. First, I focus on the metallic objects on the left side of the bus. The bus is positioned towards the upper

center of the scene. To its left, there are two prominent metallic objects: a gold-colored motorcycle and

a gold-colored scooter. These objects are clearly metallic and positioned to the left of the bus. Next, I consider the tiny cyan rubber things. Upon examining the scene, there are no objects that appear to be tiny and distinctly cyan rubber. The bus itself is not a tiny object, and the other objects, such as the motorcycles and airplanes, are not tiny either. Based on this analysis, the metallic objects on the left side of the bus are the gold-colored motorcycle and scooter. These are the only objects that meet the criteria of being metallic and positioned to the left of the bus. Therefore, the answer is that there are two objects that are metallic and on the left side of the bus

###### RoboBrain2.0 Answer: 2

RoboBrain2.0-Eculid Response: In the visual scene before me, I observe several metallic objects, including a blue airplane, a yellow scooter, and a yellow motorcycle. The blue airplane is positioned prominently in the foreground, while the yellow scooter and motorcycle are situated towards the left side of the scene. The bus, which is a metallic object, is located further back in the scene, behind the motorcycle. To determine the number of metallic objects on the left side of the bus, I focus on the objects that are positioned to the left of the bus. The yellow scooter and the yellow motorcycle are both on the left side of the bus. The blue airplane, although not directly to the left of the bus, is also considered as part of the left side due to its position relative to the bus.

The blue airplane, being a metallic object, is included in the count. The yellow scooter and motorcycle are also metallic and are positioned to the left of the bus. Therefore, the metallic objects on the left side of the bus are the blue airplane, the yellow scooter, and the yellow motorcycle.

Thus, based on the visual analysis of the scene and the spatial arrangement of the objects, the number of metallic objects on the left side of the bus is three. Therefore, the answer is: 3.

###### RoboBrain2.0 Answer: 3

Figure 12. The response and final answer for RoboBrain2.0-7B [8] and RoboBrain2.0-Eculid-7B in SuperCLEVR [47].

Problem: As shown in the figure, it is the three - view drawings of a part. Then the surface area of this part is?

[Figure 48]

- A. $15\pi \mathrm{cm}^{2}$
- B. $24\pi \mathrm{cm}^{2}$
- C. $51\pi \mathrm{cm}^{2}$
- D. $66\pi \mathrm{cm}^{2}$

#### Answer: B

Problem: As shown in the figure, six small cubes are placed as shown. If a small cube with a label is removed, and both its front view and top view change, the label of this small cube is?

[Figure 49]

A. (1) B. (2) C. (3) D. (4)

#### Answer: C

Problem: The geometric solids in the figure are all

stacked by cubes with a side length of 1. The surface area

of the 1-th geometric solid is 6. Then the surface area of the 20-th geometric solid is?

[Figure 50]

A. 1320 B. 1260 C. 1200 D. 1140

#### Answer: B

- Figure 13. More examples from the Euclid30K dataset.

Problem: As shown in the figure, a rectangular iron sheet can be folded into an uncovered cuboid box after cutting off four corners. According to the data marked in the figure, find the base area of this box (unit: centimeter).

[Figure 51]

##### Answer: $a^2+ab - 8a - 4b + 16$

Problem: As shown in the figure, in the triangular prism $ABC

- A_1B_1C_1$, both the side - lengths of the base and the length of the lateral edge are equal to $1$, and angle $BAA_1$

[Figure 52]

= angle $ CAA_1 $ = $60^{\circ}$. Find the cosine value of the angle formed by the skew lines $AB_1$ and $BC_1$.",

##### Answer: $\sqrt{2}$

Problem: As shown in the figure, in the cube $ABCD A_{1}B_{1}C_{1}D_{1}$, $P$ is an equal - dividing point of the diagonal $BD_{1}$ into three parts. The number of different values of the distances from $P$ to each surface of the cube is?

[Figure 53]

A. 2 pieces B. 3 pieces

C. 4 pieces D. 6 pieces Answer: A

- Figure 14. More examples from the Euclid30K dataset.

Problem: As shown in the figure, it is the lateral expansion

diagram of a food packaging box. Please calculate the surface

area of this packaging box according to the dimensions marked in the figure.",

[Figure 54]

###### Answer: $2b^{2}+4ab$

Problem: As shown in the figure, in triangle $ABC$, $AB \perp AC$. If $AD \perp BC$, then $AB^{2}=BD \cdot BC$. Similarly, there is a proposition: In the triangular pyramid $A-BCD$, AD $\perp$ the plane $ABC$. If the projection of point $A$ inside \triangle $BCD$ is $M$, then $S_{triangle ABC}^{2}=S_{triangle BCM} \cdot S_{triangle BCD}$. The above - mentioned proposition is?

[Figure 55]

- A. True proposition
- B. Adding the condition “$AB \perp AC$” makes it a true proposition.
- C. Adding the condition “$M$ is the orthocenter of triangle $BCD$” makes it a true proposition.
- D. Adding the condition "The triangular pyramid $A-BCD$ is a regular triangular pyramid" makes the proposition true."

[Figure 56]

[Figure 57]

[Figure 58]

Answer: A

Figure 15. More examples from the Euclid30K dataset.

