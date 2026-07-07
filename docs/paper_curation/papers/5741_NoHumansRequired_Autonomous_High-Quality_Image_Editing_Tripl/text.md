# arXiv:2507.14119v2[cs.CV]25Sep2025

### NoHumansRequired: Autonomous High-Quality Image Editing Triplet Mining

Maksim Kuprashevich Grigorii Alekseenko Irina Tolstykh Georgii Fedorov Bulat Suleimanov Vladimir Dokholyan Aleksandr Gordeev R&D Department, SALUTEDEV https://riko0.github.io/No-Humans-Required/

###### Abstract

Recent advances in generative modeling enable image editing assistants that follow natural language instructions without additional user input. Their supervised training requires millions of triplets ⟨original image, instruction, edited image⟩, yet mining pixel-accurate examples is hard. Each edit must affect only prompt-specified regions, preserve stylistic coherence, respect physical plausibility, and retain visual appeal. The lack of robust automated editquality metrics hinders reliable automation at scale. We present an automated, modular pipeline that mines highfidelity triplets across domains, resolutions, instruction complexities, and styles. Built on public generative models and running without human intervention, our system uses a task-tuned Gemini validator to score instruction adherence and aesthetics directly, removing any need for segmentation or grounding models. Inversion and compositional bootstrapping enlarge the mined set by ≈ 2.6×, enabling large-scale high-fidelity training data. By automating the most repetitive annotation steps, the approach allows a new scale of training without human labeling effort. To democratize research in this resource-intensive area, we release NHR-Edit, an open dataset of 720k high-quality triplets, curated at industrial scale via millions of guided generations and validator passes, and we analyze the pipeline’s stage-wise survival rates, providing a framework for estimating computational effort across different model stacks. In the largest cross-dataset evaluation, it surpasses all public alternatives. We also release Bagel-NHR-Edit, a finetuned Bagel model with state-of-the-art metrics.

###### 1. Introduction

Recent acceleration in generative modeling has facilitated image-editing assistants that follow natural language instructions. Creating such editors is a multi-stage process, starting with foundational pre-training on large, often noisy datasets (e.g., Brooks et al. [4], Ge et al. [9], Hui et al.

[14], Wei et al. [28], Ye et al. [33], Yu et al. [34], Zhang et al. [36], Zhao et al. [40]). This stage adapts a base text-toimage model to execute diverse edits and preserve unedited regions. Next, initial SFT on smaller, curated datasets elevates performance on specific tasks; ObjectDrop [5] and OmniPaint [23] have shown that as few as 2500-3300 pairs of real photos can teach a model to remove shadows and reflections in object removal task. The third stage, continual supervised fine-tuning (SFT) and preference optimization [20, 27], handles more complex edits and improves quality but presents a data bottleneck. It is constrained by reliance on human annotators to review millions of pixellevel edits, which is not the best use of expert attention.

Existing large-scale data collection methods have fundamental drawbacks. Cascades of external tools, e.g., for grounding [16], segmentation [15], and inpainting [24], create visual artifacts and can corrupt the data — if an imperfect “remove” edit with inpainting artifacts is inverted into an “add” operation, the model may learn to use artifacts as spatial cues rather than understanding the instruction’s semantics, effectively poisoning the training data. Approaches like 3D rendering [7] lack realism and scalability, while video frame extraction [17] depends on complex, error-prone auxiliary models. A lack of reliable validation metrics for detecting subtle defects persists; although MLLMs are now used as evaluators [28, 29, 33], we found even top models like Gemini 2.5 Pro [10] insufficient, and we therefore fine-tuned a Gemini-2.0-flash [11] validator on human scoring data (Sec. 3.2).

We posit that the potential of a model after initial SFT is under-exploited. By utilizing its new abilities and sensitivity to stochastic initialisation, the editor itself can generate unlimited high-quality synthetic data. To realize this, we introduce an end-to-end triplet-mining pipeline. For each instruction, the framework generates multiple candidate edits. These are pre-filtered, then judged by our fine-tuned validator, which selects the single best edit that meets our strict quality standards (Algorithm 1). This self-contained framework unlocks several capabilities for continual learning:

[Figure 1]

[Figure 2]

###### Object Human Ambience

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Turn the cake stand into ﬁne porcelain & make the counter of reclaimed wood

[Figure 7]

[Figure 8]

Switch the pullover to a floral cardigan & the long skirt to ankle-length trousers

Make the ground frost-laden

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Place a large glass vase on the ﬂoor & delete the ﬂuffy pink cushion

Make it sunrise & alter the robin’s Add a tiny bright yellow taxi toy feathers to a shimmering silver

Turn her eyes brown & make her smile

Figure 1. High-quality samples from our NHR-Edit dataset.

- • Direct complexity measurement for curricula: Instruction difficulty for the current model is quantified by counting attempts for a successful edit, providing a direct signal for an easy-to-hard learning curriculum.
- • Targeted weakness correction: Rare successes on complex tasks can be mined by running the model repeatedly to harvest a targeted dataset that fixes that weakness.
- • Compositional edit synthesis: Complex training data can be created by combining multiple instructions. For example, a single instruction can execute two additions, one deletion, and a global style change in one pass.
- • Flexible input sourcing: The framework uses real and synthetic inputs. Real images provide authentic scenarios, while synthetic images enable exploration of the long-tail, including impossible-to-photograph scenarios (e.g., a corgi in a spacesuit on a rocket).
- • Unparalleled simplicity and flexibility: The framework is model-agnostic and requires no external specialist models for segmentation, depth estimation, or grounding.

To demonstrate effectiveness, we release NOHUMANSREQUIRED DATASET (NHR-Edit), a public dataset of 720k rigorously validated triplets (for representative samples, see Figure 1 and Figures C.8-C.19 in Appendix). Building on this data, we release BAGEL-NHR-EDIT, a LoRA-tuned BAGEL [8] variant trained on NHR-Edit that surpasses the base model on two benchmarks. Our primary contribution is this end-to-end pipeline, a powerful engine for advancing research in self-improving generative models [6, 39].

###### 2. Related Work

Our research builds upon two main pillars of generative modeling: methodologies for creating instruction-based editing data and the paradigm of model self-improvement through preference optimization.

###### 2.1. Methodologies for Editing Data Generation

Creating high-quality editing data is a foundational challenge, with existing approaches presenting unique tradeoffs.

Pipelines on Real-World Data. A common strategy is a cascade of models to edit real images, like in AnyEdit [34] and ImgEdit [33], which use pipelines for detection [16], segmentation [15], and inpainting [24]. Each stage can propagate errors, and global edits struggle to preserve details. Video-based methods like Step1X-Edit add complexity with pipelines for motion estimation and background filtering [41]. These approaches can also suffer from dataset bias (Schuhmann et al. [22]).

Fully Synthetic Generation. Synthetic generation offers more control but has its own drawbacks. Methods range from 3D rendering [7], which is labor-intensive and lacks photorealism, to diffusion-based techniques [9, 14, 40] that can introduce artifacts, alter details, or generate data misaligned with real-world distributions.

Specialist Models. OmniEdit [28] trains specialized models for each task (e.g., inpainting, attribute modification) integrated into similar pipelines. While ensuring quality for simple tasks, this inherits cascade complexity and error propagation issues and cannot handle complex, compositional instructions.

Our work differs by using the editor model itself as the data source, creating a simple framework that bypasses complex pipelines and specialist models.

###### 2.2. The Metric Gap in Image Editing

Evaluation is a key challenge, as traditional, referencebased metrics (e.g., LPIPS [37], DINO [21], CLIPScore [12]) correlate poorly with human preference and are unsuitable for our generative framework. While MLLMbased reward models have emerged in related fields (IQA, T2I, T2V) [30, 31, 38], their use in editing was pioneered by VIEScore [29], which showed GPT-4o judgments align well with human preferences. Subsequent work like OmniEdit and ImgEdit built on this by distilling judgments or fine-tuning MLLMs. However, curating data for SFT demands higher precision. We found that even top models like Gemini 2.5 Pro [10] are unreliable for detecting subtle editing flaws (Fig. C.7). We therefore developed a specialized validator by fine-tuning Gemini-2.0-flash [11] on human preference data to achieve the necessary sensitivity.

###### 2.3. Self-improvement and Iterative Learning

A model generating its own data for self-refinement is a highly effective concept, proven in NLP [20, 26] and extended to generative vision [35]. Our framework is an automated engine applying these preference alignment techniques to image editing. Algorithms like DPO [27] and KTO [2] require scalable preference-labeled data, which our pipeline automatically provides. By solving the data generation and labeling bottleneck, our work enables applying these powerful self-improvement techniques to instruction-based image editing.

###### 3. Methodology

This section details our autonomous triplet-mining pipeline, which comprises four modules: (i) a prompt engineer for generating consistent text-to-image (T2I) and imageto-image (I2I) instructions; (ii) a T2I generator; (iii) an instruction-guided image editor; and (iv) a multi-stage validation stack.

###### 3.1. Automated Mining Pipeline

Figure C.6 and Algorithm 1 overview the pipeline (full prompts can be found in Sec. A). The process starts with initial constraints (e.g., topic, style) which are used by a prompt engineering module (Algorithm 1a) to produce a T2I prompt (pt2i) and corresponding edit instructions ({pe}k), as shown in Listing 1. While supplied manually here, these constraints could be automated.

For each T2I prompt, the pipeline generates N candidate source images (I0) using different random seeds (Algorithm 1b). Each source image undergoes M edit attempts for every instruction pe. This yields a large pool of candidate triplets ⟨I0,pe,Ie⟩, which are subjected to a coarse pre-filtering step before final validation (see Sec. 3.2). In the final stage, for each unique pair ⟨I0,pe⟩, the highest-

quality edited image Ie⋆ is selected by maximizing the geometric mean of its scores (√saes · sadh, see Algorithm 1). We chose this metric because it enforces a balance between aesthetic quality and instruction adherence, proving particularly robust for highly imbalanced scores where a candidate excels on one criterion but fails on the other. This prevents the selection of, for instance, a visually pleasing but semantically incorrect edit. The winning image is added to the final dataset D only if both of its scores exceed predefined quality thresholds.

1. Example of a generated T2I prompt and its corresponding edit instructions.

\\ T2I prompt "prompt": "A living room with a large window: a small cactus on the windowsill,

a half-eaten bowl of cereal on the coffee table, a remote control, a crocheted blanket, and a dog toy on the rug.", \\ I2I prompts for editing "edits": [

"Get rid of that cactus.", "Remove the cereal bowl.", "No remote control, thanks.", "Lose the crocheted blanket.", "Eliminate the dog toy.", "Remove the cactus, cereal, remote,

blanket, and toy" ]

###### 3.2. Validation Framework

Robust validation is a key challenge in automated triplet mining. Our two-stage process uses a Qwen-VL 72B prefilter to discard obvious failures, reducing calls to the more expensive final validator. While this open-source model cannot filter all noise, it is effective. The second stage uses a specialized Gemini 2.0 Flash model, fine-tuned on a curated corpus, to assign final aesthetic and instruction adherence scores.

Validator threshold. We set the validator thresholds using an a priori rule grounded in the survival curve S(T) (Fig. C.5 in Appendix). The curve shows a gradual decline up to ≈ 4.3 and then enters a broad cliff over T ∈ [4.4,4.9] with pronounced drops at T = 4.5 (−62.1% of the initial pool) and T = 4.9 (−84.0%). To avoid operating exactly at a discontinuity while staying before the collapse regime, we choose the point that maximizes the minimum distance to these two knees. This midpoint yields T = 4.7. Additionally, an independent 3 raters audit of 1000 randomly

Algorithm 1 Pipeline Pseudocode Algorithm 1a: SamplePromptsDesign

Algorithm 1b: TripletMining Require: T2I prompt pt2i, edits {pe}k, parameters

Require: Task description in PA.1 Ensure: Set P = (pt2i,{pe}k) m

N,M, global GPU-hour budget Budget Ensure: Candidate pool C

- 1: P ← OpenAI o3 PA.1
- 2: return P

- 1: C ← ∅, Jobs ← ∅
- 2: for i ← 1 to N do
- 3: seedi ← Random(i)
- 4: I0 ← FLUX.1-schnell(pt2i,seedi)
- 5: if not Qwen7B I0,pt2i,PA.5 then
- 6: continue
- 7: end if
- 8: for all pe ∈ {pe}k do
- 9: for j ← 1 to M do
- 10: Jobs ← Jobs ∪ {(I0,pe,Random(j))}
- 11: end for
- 12: end for
- 13: end for
- 14: while Jobs ̸= ∅ and GPU hours < Budget do

- 15: sample (I0,pe,s) ∼ Uniform(Jobs)
- 16: Jobs ← Jobs \ {(I0,pe,s)}
- 17: Ie ← I2I DiT (internal)(I0,pe,s)
- 18: (saes,sadh) ← Qwen72B I0,pe,Ie,PA.2
- 19: if saes ≥ Taes and sadh ≥ Tadh then
- 20: checkp ← Qwen72B(I0,pe,Ie,PA.3,PA.4)
- 21: checkl ← LowLevelCheck(I0,Ie)
- 22: if checkp and checkl then
- 23: C ← C ∪ {⟨I0,pe,Ie⟩}
- 24: end if
- 25: end if
- 26: end while
- 27: return C

Algorithm 1c: Autonomous Triplet-Mining Pipeline Require: Task description in PA.1, parameters N,M,

Taes,Tadh

Ensure: Final dataset D

- 1: D ← ∅, Pool ← ∅
- 2: P ←SAMPLEPROMPTSDESIGN(PA.1) {1a}
- 3: for all (pt2i,{pe}k) ∈ P do
- 4: Pool ← Pool∪TRIPLETMINING(pt2i,{pe}k,N,M) {1b}
- 5: end for
- 6: for all distinct ⟨I0,pe⟩ in Pool do
- 7: S ← {Ie | ⟨I0,pe,Ie⟩ ∈ Pool}
- 8: saes(Ie), sadh(Ie) ← Gemini I0,pe,Ie,PA.2 for every Ie ∈ S
- 9: S ← {Ie ∈ S | saes ≥ Taes ∧ sadh ≥ Tadh}
- 10: if S ̸= ∅ then
- 11: Ie⋆ ← arg max Ie∈S

saes(Ie)sadh(Ie)

- 12: D ← D ∪ {⟨I0,pe,Ie⋆⟩}
- 13: end if
- 14: end for
- 15: D ← D∪APPLYINVERSIONS(D) 3.6
- 16: D ←BCFILTER(D,Tinv,aes,Tinv,adh) 3.6
- 17: D ← D∪APPLYBOOTSTRAPS(D) 3.6
- 18: return D

sampled items further indicates that the residual errors, i.e., cases where the hard-filter validator makes mistakes, as any model can — are dispersed at high scores and frequently lie at ≥ 4.7; items that pass T = 4.6 typically receive very high scores (≥ 4.8). Consequently, raising the threshold from

- 4.7 to 4.8 removes almost no additional erroneous samples while shrinking the dataset. We therefore adopt the first reliable operating point before the collapse region, T = 4.7. We note that an exact operating point could, in principle, be obtained only through a thorough manual audit, ideally yielding per-category thresholds. However, such curation is labor-intensive and beyond scope. The survival-curve rule above provides a sufficient and stable choice for our application, as supported by the results in subsection Human manual audit and cross-dataset comparison in Sec. 3.7.

Low-level check. The absolute-difference image D = |Ie − I0| is thresholded (> 40) and analysed with ConnectedComponents using 4-connectivity and 32bit labels; a triplet is discarded if the largest connected component covers < 0.5% of all pixels flagged as changed. This purely heuristic, optional filter empirically outperforms a raw image-difference threshold. Cutoff level was also found during the threshold analysis of T.

Human manual audit. In a blinded audit of n = 300 accepted triplets (Tab. C.4 in Appendix), residual issues were low: 5.0% T2I-inherited imperfections, 4.3% difficult removals under complex lighting or occlusion, 3.3% small residuals after deletion, and 1.6% minor inpainting near the edit area.

###### 3.3. Gemini Validator

While many pipelines use general-purpose models like GPT-4o [14, 28, 29] for evaluation, they are not optimized for fine-grained pixel-level changes (see Fig. C.7 in Appendix). To obtain reliable estimates, we fine-tuned a Gemini-2.0-flash [25] model on a dedicated humanannotated corpus. This corpus was meticulously constructed to cover a wide spectrum of edit qualities, using a combination of an in-house DiT editor and proprietary models like Grok [32] and Gemini. This diverse sourcing ensures the assessor was trained on a broad distribution of potential successes and failures, preventing overfitting. Following HQ-Edit [14], OmniEdit [28] and AnyEdit [34], each image is rated on two five-point scales: (i) Instruction score and (ii) Aesthetics score. The collected set contains 2998 training and 827 validation examples; every example is judged by two to four independent raters. Interrater reliability, as mean pair-wise Spearman correlation, is ρ = 0.41 ± 0.09 for Aesthetics and ρ = 0.64 ± 0.05 for Instruction, corresponding to moderate and substantial agreement. The higher consistency on the instruction axis is expected, as semantic correctness is less subjective than aesthetics. To aggregate scores, each rating is first normalized by subtracting the annotator’s bias, computed relative to the same triplets they rated. The bias bj for each rater j is

1 |Nj| i∈N

1 |Nj| i∈N

(1)

−

###### si,j

###### s¯i

bj =

j

j

Rater j’s mean score

Mean score of triplets rated by j

where Nj is the set of triplets rated by rater j, Ri is the set of all raters for triplet i, and s¯i = |R1

i| k∈Ri si,k denotes the mean score of triplet i.

The final score Si for a triplet is then the mean of the bias-corrected scores:

1 |Ri| j∈R

Si =

i

si,j − bj (2)

Using this annotated validation set, we benchmarked our task-specific, fine-tuned Gemini 2.0-flash model against its original version, the larger Gemini

- 2.5-pro [25], and Qwen 2.5 72B. Table 1 compares the mean absolute error (MAE) and Spearman ρ. Vanilla checkpoints suffer from calibration error, whereas finetuning halves the MAE and boosts rank correlation on the instruction axis from 0.36 to 0.82, outperforming even the larger 2.5-pro model. Notably, the fine-tuned model provides high-quality scores directly, without a costly chain-of-thought step, confirming a specialized assessor is a more efficient paradigm for large-scale filtering. To further validate our assessor’s robustness, we benchmarked it against the publicly available ImgEdit validator [33] on a

per-category basis. Overall, our assessor nearly doubles the rank correlation (overall ρ = 0.79 vs. 0.41). Category-level breakdowns — including large gains on Replace and Compose are provided in Appendix Tab. B.2.

Table 1. Quality metrics of the assessor model on validation data. I — Instruction, A — Aesthetic.

Model I MAE ↓ I ρ ↑ A MAE ↓ A ρ ↑ Qwen 2.5 72B 0.961 0.551 0.839 0.361 Gemini-2.5-pro 0.869 0.609 0.915 0.523 Gemini-2.0-flash 1.241 0.359 1.063 0.245 Gemini-2.0-flash (finetune)

0.503 0.815 0.568 0.631

###### 3.4. Image Editing Backbone

Our framework requires an instruction-guided image-toimage (I2I) model that takes a source image I0 and prompt pe to produce an edited image Iˆe. We use a proprietary, internal diffusion-based editor but treat it as a black box. This modular design ensures no component depends on the editor’s internals, allowing it to be swapped with any other I2I model. The external validation stack reinforces this modularity.

###### 3.5. Implementation Details

Component specification. Our pipeline is fully modular; each block can be replaced by any compatible alternative. Unless otherwise noted, we use the following defaults:

- • Prompt engineer. We query the reasoning-centric OpenAI o3 model [18] with the template A.1 to jointly emit a text-to-image (T2I) prompt and a set of k logically consistent edit instructions.
- • T2I generator. Source images are synthesised with FLUX.1-schnell [3] at a random resolution (long side ∈ [860,2200]px; aspect ratio bounded by 1:6 ≤ AR ≤ 6:1) using 4 steps.
- • Plausibility gate. We retain only sample seeds whose captions pass a plausibility check by Qwen2.5-VL-7B [19] using (Appendix, Prompt A.5).
- • Instruction-guided editor. By default we employ our internal I2I DiT model with 18-28 diffusion steps.
- • Soft pre-validation filter. Candidate edits first pass a coarse screen with Qwen2.5-VL-72B using (Appendix, Prompts A.2, A.3, A.4).
- • Hard validation filter. The fine-tuned Gemini validator (Sec. 3.2) runs at temperature 0.0 with (Appendix, Prompt A.2).

All Qwen-VL calls use the HuggingFace transformers default configuration with temperature 10−6.

Configuration. The optimal counts for T2I seeds (N) and edit retries (M) depend on prompt difficulty and represent a fundamental trade-off between dataset diversity, success rate, and computational cost. While a larger M helps with harder samples by trading compute for success probability, a larger N improves diversity. Our choice of N = 10 and M = 5 was a cost-effective balance for our specific model stack and should not be considered a universal optimum. Practitioners should tune these values based on their editor’s capabilities and instruction complexity. For instance, a less capable model may require a higher M to achieve a reasonable success rate. Validation thresholds are fixed at Taes = Tadh = 4.7.

Budget-aware random scheduler. This scheduler allows practitioners to cap total expenditure. It works by enumerating all potential seed-instruction pairs (N×k×M), queuing those that pass a plausibility test, and then drawing jobs uniformly without replacement until a predefined limit is exhausted. This limit, denoted as Budget, is a user-specified cap in GPU-hours (or API-seconds). The final compute, quality, and dataset yield are therefore dictated by this budget, not by the nominal (N,M) values. In future work, this could be extended to adaptive sampling, such as prioritizing difficult categories or continuing retries until a pre-filter success.

###### 3.6. Data Augmentation

The dataset is further refined and expanded through postprocessing and augmentation.

Semantic Inversion. Any edit can be inverted by rewriting the instruction into its logical inverse using Gemini 2.5 Flash and Prompt A.6. Crucially, access to the original T2I prompt allows preserving details for a high-quality learning signal. For the example in Listing 1, the inverse of the composite deletion is not a simple addition but a fully specified prompt: “Add a small cactus on the windowsill, a half-eaten bowl of cereal on the coffee table, a remote control, a crocheted blanket, and a dog toy on the rug.”

Bootstrap Composition. Since each source image I0 can be successfully edited into multiple distinct images (Ie1, Ie2, etc.), new triplets can be constructed. Given two successful edits, a new instruction p′e2 can be formulated to transform Ie1 into Ie2, yielding a novel compositional triplet ⟨Ie1,p′e2,Ie2⟩ (demonstrated in Fig. 2).

Backward Consistency filter. Semantic inversion guards against trivial forward successes when the T2I misses an object. If the inverse instruction (e.g., “add the cat on the sofa”) receives a low score, we drop both the forward and

inverse triplets. This optional check depends on the T2I and the validator and serves as an extra quality assurance layer.

[Figure 15]

Make her serious. Replace the scarf with a suit. Give her a smile. Replace the suit with a

silk scarf.

[Figure 16]

[Figure 17]

Give her a smile and replace the scarf with a suit.

Replace the suit with a silk scarf and make her serious.

Figure 2. Solid arrows represent forward instructions, and dashed arrows represent their semantic inversions. Instructions for compositional triplets are aggregated from both forward instructions and inversions.

###### 3.7. NoHumansRequired Dataset

The final pipeline yields a dataset of 720088 high-quality triplets. Table 2 provides a detailed breakdown of data volume changes. Initial generation and editing phases have survival rates of 44% and 43% respectively, with subsequent filtering further refining the set. Augmentation through inversion and composition increases the dataset size by 94.88% and 30.65%.

NHR-Edit presents a variety of editing categories, while also spanning diverse styles, perspectives, and aspect ratios:

- • Removal (≈ 227k) and Addition (≈ 225k). The focus is on object removal, as successful inversions provide challenging object addition examples, crucial for improving modern editors (Fig. C.1).
- • 27 more diverse operations (≈ 103k). These include complex object manipulations (reshape, change color or texture, degrade and restore), ambience (change background, time of day, weather, season), and human-related editing (emotion, haircut, clothes, accessories) — see Fig. C.2.
- • Almost 300 composite categories (≈ 165k). Bootstrap composition (Sec. 3.6) allows the construction of multioperation editing triplets, invaluable as complex training data (Fig. C.3).
- • 96 various styles. Spanning from photographic compositions (e.g., DSLR, panorama, wide-angle, aerial) — to specific artistic choices (oil painting, sketch, anime, crochet, minimalist, etc.) (Fig. C.4).
- • 26 aspect ratios. From 640×1600 portraits to 1600×640 panoramas. Every image is a well-established composition, generated and edited in its native aspect ratio. The distribution and samples are shown in Tab. C.3.

###### 3.8. Cross-dataset comparison.

We compare our dataset quality against public benchmarks by using our fine-tuned assessor to score 5000 random samples from each. Table 3 reports the mean Instruction, Aesthetics, and (following OmniEdit) geometric mean scores.

- Table 2. Each stage statistics for 63 292 prompts. Taking 3 072 385 generation attempts, the survival rate can be estimated as 15.3%, excluding the squeezing step.

Processing Stage Method / Model ∆ (%) Remaining Vol.

Initial Generation FLUX.1-schnell — 1171773 Generation Filtering Qwen-7B −56.00 515584 Editing Generation In-house DiT +495.90 3072385 Editing Filtering Qwen-72B (Pre-Filter) −57.00 1321126 Low Level Check Connected Component Analysis −3.00 1281492 Quality Scoring Gemini Validator (Hard Filter) −63.21 471523 Final Selection ArgMax Selection −31.01 325287

Inversion Gemini 2.5 Flash +94.88 633904 Composition Bootstrap & Concatenation +30.65 828212 Backward Consistency Filtering Gemini Validator (Hard Filter) −13.06 720088

- Table 3. Quality metrics across editing datasets, sorted in ascending order by geometric mean. The ’Type’ column indicates the generation method: A for Automatic and M for Manual. The asterisk (*) denotes a highly curated automatic dataset.

Dataset Type Instr. ↑ Aesth. ↑ Geom. ↑ UltraEdit A 2.67 3.30 2.92

- Seed Part 2 M 3.20 3.03 3.09 Seed Unsplash A 3.01 3.84 3.28 InstructPix2Pix A 3.17 3.58 3.30 MagicBrush A 3.62 3.27 3.38 AnyEdit A 3.39 3.64 3.44 HQ-Edit A 2.90 4.21 3.45 ImgEdit A 3.26 3.91 3.49 Seed OpenImages A 3.42 3.86 3.50
- Seed Part 3 M 4.06 4.37 4.13 OmniEdit A* 4.21 4.35 4.23 NHR-Edit A 4.56 4.52 4.53

With a geometric mean of 4.53, NHR-Edit establishes a new state-of-the-art, significantly outperforming existing datasets, including those with manual curation. This validates that our automated methodology can produce a corpus whose quality is superior to existing benchmarks.

Method note. To justify using our assessor for crossdataset ranking, we ran a targeted human cross-check on a sentinel panel spanning the spectrum in Tab. 3: the lowestranked (UltraEdit), a mid-ranked set (HQEdit), and the two highest-ranked (OmniEdit, NHR-Edit). For each dataset we sampled n = 80 items and obtained 3 independent crowd annotations under the same instructions as the assessor. Table 4 reports dataset-level geometric means with 95% bootstrap intervals. Across this sentinel panel, assessor and humans induce the same ordering (UltraEdit < HQEdit < OmniEdit < NHR-Edit), with substantial interval overlap in 3/4 cases and both assigning the top rank to NHR-Edit. This probes potential misorderings at the bottom, middle,

Table 4. Gemini (assessor) vs. Human geometric mean (Geom.), shown as mean ± half-width of the 95% nonparametric bootstrap CI (B = 2000) over n = 80 items per dataset (3 raters/item), recomputing Geom. per resample.

###### Dataset Gemini Geom. ↑ Human Geom. ↑

UltraEdit 3.00 ± 0.14 3.05 ± 0.15 HQEdit 3.52 ± 0.15 3.54 ± 0.15 OmniEdit 4.30 ± 0.16 4.50 ± 0.15

###### NHR-Edit 4.54 ± 0.12 4.75 ± 0.09

and top regimes and provides sufficient evidence that the assessor preserves dataset-level rank; we therefore use it to score 5000 samples per dataset in Tab. 3. Minor numerical differences between assessor means in Tab. 3 and Tab. 4 arise from the n = 80 subsampling.

Table 5. Overall results comparing our BAGEL-NHR-EDIT with the baseline. We report mean ± standard deviation and [95% confidence intervals] computed from 3 inference runs using different random seeds. The best results based on the mean are in bold. Per-category breakdowns appear in Appendix Tab. C.1 and Tab. C.2.

###### Benchmark Metric(s) BAGEL BAGEL-NHR-EDIT

ImgEdit-Bench Overall 3.30 ± 0.03 [3.23, 3.36] 3.33 ± 0.02 [3.28, 3.38] GEdit-Bench SC 7.61 ± 0.15 [7.23, 7.98] 7.80 ± 0.07 [7.63, 7.97]

PQ 6.18 ± 0.15 [5.82, 6.55] 6.56 ± 0.08 [6.37, 6.75] O 6.53 ± 0.14 [6.19, 6.87] 6.80 ± 0.07 [6.63, 6.98]

###### 4. Experiments

This section investigates if NoHumansRequired Dataset can improve an existing edit method’s performance.

###### 4.1. Experimental Setup

We use BAGEL [8], a 14B-parameter open-source multimodal foundation model with a Mixture-of-TransformerExperts architecture. We performed parameter-efficient adaptation only to the generation expert’s attention and feed-forward projection layers using LoRA [13] (rank = 16, alpha = 16, dropout = 0.05, bias = “none”, batch size = 16 (it is dynamic, on average 2 per gpu), lr = 2e-5). We refer to this fine-tuned variant as BAGEL-NHR-EDIT. Other BAGEL components are frozen to preserve the model’s pretrained capabilities. We chose LoRA for its training stability and substantially lower computational cost compared to full fine-tuning. All BAGEL and BAGEL-NHR-EDIT runs use matched batch size, optimizer, learning rate schedule, precision, and data augmentations.

###### 4.2. Benchmarks and Metrics

We evaluate BAGEL-NHR-EDIT against the BAGEL baseline on GEdit-Bench [17] and ImgEdit-Bench [33], strictly following the authors’ official evaluation protocols. For GEdit-Bench, we use the VIEScore setup with GPT-

- 4o [1] to report Semantic Consistency (SC, 0-10), Perceptual Quality (PQ, 0-10), and Overall (O). For the ImgEditBench evaluation, we adopt the original authors’ protocol: GPT-4o is used to score edited images across several criteria, each rated on a 1-to-5 scale.

###### 4.3. Results

Table 5 reports mean, standard deviation, and 95% confidence intervals calculated from 3 inference runs with different seeds for each model. BAGEL-NHR-EDIT improves over the baseline on the mean scores for both benchmarks: on ImgEdit-Bench, the overall score increases from 3.30 to 3.33 (+0.03); on GEdit-Bench, the SC/PQ/O scores improve from 7.61/6.18/6.53 to 7.80/6.56/6.80, with deltas of (∆+0.19/+0.38/+0.27) respectively. Detailed per-category results are in Appendix Tab. C.1 and Tab. C.2.

###### 5. Conclusion

We propose an automated end-to-end pipeline to mine highquality triplets for instruction-guided image editing. A pretrained editor generates candidate edits and we retain only successful ones after strict filtering. Instruction inversion and compositional editing produce semantically rich, diverse triplets. Integrating a T2I model broadens stylistic coverage and mitigates overfitting. The pipeline is selfimproving: as the editor advances it yields better triplets, creating a feedback loop. We release BAGEL-NHR-EDIT, a LoRA-tuned BAGEL variant that outperforms its baseline on public benchmarks, and NHR-Edit to support future research in text-based editing.

###### Limitations

Our framework is bounded by its component models: it cannot produce triplets for operations the base editor cannot perform, a limitation only partly mitigated by multi-seed sampling. Data quality also depends on the T2I generator and instruction LLM, which can introduce biases from templates or priors. LLM-written instructions may diverge from real user phrasing, though diverse prompting reduces this gap.

Reporting absolute GPU-hours would be misleading as costs depend on chosen models and API pricing. Instead, we provide stage-wise survival rates in Tab. 2 to help estimate required generations and costs for a given model stack.

Ethics & Societal Impact. NHR-Edit contains only synthetic images generated with FLUX.1-schnell from ChatGPT o3 prompts; no photographs of real people are used, so consent/privacy risks tied to real-person imagery are not implicated (though incidental resemblance is possible). We rely on provider safeguards and automated post-filters to reduce NSFW or biased samples, but filtering is imperfect and no manual curation was performed, so some undesirable cases may remain. Because editing models can be misused, the dataset is released for research use only. Prompt diversity was encouraged, yet representation biases may persist; downstream users should assess content, apply safety filters, and comply with applicable laws and policies before deployment.

###### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 8

- [2] Megha Bhardwaj and Anant Hans. Aligning text-to-image diffusion models with k-fold tamer preference. arXiv preprint arXiv:2404.04465, 2024. 3
- [3] Black-Forest-Labs. FLUX.1-schnell. https : / / huggingface.co/black-forest-labs/FLUX. 1-schnell, 2024. 5
- [4] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions,

2023. 1

- [5] Jacopo Burroni, Federico Boin, Federico Amato Galatolo, Oussama Es-sounayni, Marco De Nadai, Federico Becattini, Nicu Sebe, Claudio Baecchi, and Alberto Del Bimbo. Objectdrop: Bootstrapping counterfactuals for photorealistic object removal and insertion. arXiv preprint arXiv:2403.18818, 2024. 1
- [6] Yongcen Chen, Chen Wang, Yichun Zhao, Jerry Wang, Jialu Han, Yihua Zhu, Ceyuan Zhou, Yujun He, Kewei Wu, Yong-jin Li, Tiezheng Wang, and Yu-gang Wang. Self-play fine-tuning of diffusion models for text-to-image generation. arXiv preprint arXiv:2402.10210, 2024. 2
- [7] Xueting Cheng, Teli Wang, Zheyuan Liu, Wen-gang Li, Hong-gang Li, Yu-cheng Wang, and Li Wang. Aurora: A system for composing and editing images with rich styles and semantics. arXiv preprint arXiv:2407.03471, 2024. 1, 2
- [8] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 2, 8
- [9] Yuying Ge, Sijie Zhao, Chen Li, Yixiao Ge, and Ying Shan. Seed-data-edit technical report: A hybrid dataset for instructional image editing. arXiv preprint arXiv:2405.04007,

2024. 1, 2

- [10] Google. Gemini 2.5 Pro Preview Model Card. Model card, Google, 2024. https://storage.googleapis. com/model- cards/documents/gemini- 2.5pro-preview.pdf. 1, 3
- [11] Google. Gemini 2.0 Flash Model Card. https://cloud. google.com/vertex-ai/generative-ai/docs/ models/gemini/2-0-flash, 2024. 1, 3
- [12] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: A reference-free evaluation metric for image captioning, 2022. 3
- [13] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 8
- [14] Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. Hq-edit: A high-quality dataset for instruction-based image editing. arXiv preprint arXiv:2404.09990, 2024. 1, 2, 5

- [15] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. arXiv preprint arXiv:2304.02643, 2023. 1, 2
- [16] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 1, 2
- [17] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 1, 8
- [18] OpenAI. OpenAI o3 and o4-mini System Card. https: //openai.com/index/o3-o4-mini- systemcard/, 2025. System Card, accessed 18 July 2025. 5
- [19] Qwen Team. Qwen2.5-VL-7B-Instruct. https : //huggingface.co/Qwen/Qwen2.5- VL- 7BInstruct, 2024. 5
- [20] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Ipo: An identity-preserving-optimization method for aligning lms. arXiv preprint arXiv:2402.02088, 2024. 1, 3
- [21] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. DreamBooth: Fine tuning text-to-image diffusion models for subject-driven generation, 2023. 3
- [22] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models, 2022. 2
- [23] Sang-Hyeon Shin, Jae-Ha Yang, Dong-Hyeok Han, YoungWoon Kim, and Kwang-Hyun Lee. Omnipaint: Mastering object-oriented editing via disentangled insertion-removal inpainting. arXiv preprint arXiv:2503.08677, 2025. 1
- [24] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, and Victor Lempitsky. Resolutionrobust large mask inpainting with fourier convolutions. arXiv preprint arXiv:2109.07161, 2021. 1, 2
- [25] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 5
- [26] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 3
- [27] Bram Wallace, Rafael Rafailov, Kevin Fein, Dorsa Ilas, Stefano Ermon, Christopher R´e, and Nikhil Naik. Diffusiondpo: Aligning text-to-image models with human preferences. arXiv preprint arXiv:2311.12908, 2023. 1, 3

- [28] Cong Wei, Zheyang Xiong, Weiming Ren, Xinrun Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision. arXiv preprint arXiv:2411.07199, 2024. 1, 2, 5
- [29] Quanzeng Wu, Jian-hao Wang, Jiachen Wang, Zexin Lin, Jiacheng Gao, Jing Zhang, and Jin Lu. VIEScore: Towards Explainable and Controllable Image-to-Text Evaluation. arXiv preprint arXiv:2312.14867, 2023. 1, 3, 5
- [30] Tianhe Wu, Jian Zou, Jie Liang, Lei Zhang, and Kede Ma. VisualQuality-R1: Reasoning-induced image quality assessment via reinforcement learning to rank, 2025. 3
- [31] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human Preference Score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis, 2023. 3
- [32] xAI. Grok. https://x.ai/blog/grok, 2023. Accessed: 2025-07-10. 5
- [33] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025. 1, 2, 5, 8, 16
- [34] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26125–26135, 2025. 1, 2, 5
- [35] Huizhuo Yuan, Zixiang Chen, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning of diffusion models for text-to-image generation, 2024. 3
- [36] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023. 1
- [37] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 3
- [38] Xuanyu Zhang, Weiqi Li, Shijie Zhao, Junlin Li, Li Zhang, and Jian Zhang. VQ-Insight: Teaching vlms for ai-generated video quality understanding via progressive visual reinforcement learning, 2025. 3
- [39] Zekun Zhang, Zheyuan Huang, Yushi Li, Hong Zhou, and Hongsheng Li. Self-improving diffusion models with synthetic data. arXiv preprint arXiv:2408.16333, 2024. 2
- [40] Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093, 2025. 1, 2
- [41] Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, and Nicu Sebe. Bilateral reference for high-resolution dichotomous image segmentation,

2024. 2

## Supplementary

###### A. Prompts

A.1. Samples Design Prompt

[WARN] ABSOLUTE BAN: The model must never run Python, or any other executable code, while thinking. It must compose prompts with its own knowledge only.

---------------------------------------

- 1. HIGH-LEVEL PRINCIPLES

---------------------------------------

- 1. Natural-language first - Full phrases beat comma-separated keyword lists.
- 2. Specificity over brevity - Vague prompts yield "average" images; be precise.
- 3. One coherent vision - Avoid conflicting or scatter-shot modifiers.
- 4. Layered thinking - Describe foreground -> mid-ground -> background in order.
- 5. Active, sensory wording - "Swirls", "emerges", "diffused glow" enrich texture & motion.

---------------------------------------

- 2. CORE PROMPT TEMPLATE (use as prose; brackets describe purpose)

--------------------------------------[TECH / STYLE TAG]: [SUBJECT + ACTION], [ENVIRONMENT / CONTEXT], [COMPOSITION & CAMERA], [LIGHTING], [COLOUR & MOOD]. (Optional) [TEXT ELEMENTS].

Example DSLR photograph on Nikon Z8 with 85 mm f/1.4:

- A red fox pauses atop a snow-dusted log in a quiet boreal forest, captured at

eye-level; shallow depth-of-field isolates the fox

. Soft overcast light yields gentle shadows; a muted winter palette of whites, greys

and russets conveys tranquillity.

---------------------------------------

- 3. DETAILED COMPONENT GUIDE

---------------------------------------

- - Subject & focal point - species, character, or object with defining traits
- - Action / interaction - dynamic verb or relationship
- - Environment / setting - location, era , weather, cultural cues
- - Composition / lens - shot type, framing, spatial layout, focal length
- - Lighting - source, quality, direction , time-of-day
- - Colour palette - dominant hues, contrasts, transitions
- - Mood / atmosphere - emotional tone, sensory adjectives
- - Art / render style - medium, artist, movement
- - Technical descriptors - camera body, film stock, HDR, focus stacking, 8-K and related specs
- - Text integration - exact wording, font, placement, effect
- ---------------------------------------

- 4. LAYERED & SPATIAL CONTROL

--------------------------------------Describe layers in order (foreground ->

mid -> background) or label them explicitly. Use spatial cues ("above", "to the left ", "half-submerged") so FLUX can reason

about position.

---------------------------------------

- 5. ADVANCED TECHNIQUES

---------------------------------------

- - Contrast / dual aesthetics - Define clear borders & transitions (day/night split, joy/sorrow).
- - See-through materials - Clarify front /behind & distortion ("rain-soaked glass distorts neon...").
- - Spotlighting - Bracket clause or write "strong emphasis on ..." for key elements.
- - Text-rich posters & UI - Specify font family, size, orientation; keep text

short and unique.

---------------------------------------

- 6. DOS & DON’TS

--------------------------------------[OK] Use grammatical sentences; always give some background; <= 7 focal subjects. [OK] Reference known artists or genres to cue style; describe lighting every time. [OK] Mix gear-specific tags *sometimes*

(e.g. "DSLR photograph on Canon EOS R5 with 35 mm f/1.8");

at other times say "Realistic photo, 4K " - but always be explicit. [NO] Dump raw keywords or weight syntax ; leave background implicit; issue contradictory fixes in one prompt; over

- -use "white background" (causes blur in dev builds).
- ---------------------------------------

- 7. PROMPT-DRAFTING WORKFLOW

---------------------------------------

- 1) Gather intent (subject, style, mood, use-case, text, resolution).
- 2) Fill the template, omitting only truly irrelevant slots.
- 3) Check consistency-no style or light contradictions; max 7 focal subjects.
- 4) Add layer/spatial cues for multielement scenes.
- 5) Return the final prompt (plus an optional short troubleshooting tip if helpful).

---------------------------------------

- 8. TROUBLESHOOTING CHECKLIST

--------------------------------------Blurry or flat -> specify sharper lens/aperture or refine light source. Wrong era/style -> state artist or medium earlier. Missing background -> add explicit environment sentence. Unwanted objects -> issue deletion edits (next section). Illegible text -> shorten phrase or specify font. Overcrowded -> split ideas into separate images.

---------------------------------------

- 9. OBJECT-REMOVAL EXTENSION (OPERATION

= "DELETE" ONLY)

--------------------------------------GENERAL RULES

- - Each prompt must name **1 - 5** clearly visible, dramatic objects.
- - Supply **exactly the same number** of deletion edits-one per object.
- - Edits may be casual, slangy or profane ("yeet the kite") but must target their object unambiguously. Include spatial clues;

make them *sometimes* tricky so the receiving model must reason about the scene, but not so tricky that mistakes are likely.

- - Deletion-only-no recolours, swaps, resizes.
- - Edits are independent; never reference other edits or prior context.
- - Mix everyday, exotic and fantasy objects; vary scales (colossi foreground -> tiny background).
- - **Prefer descriptive spatial cues** ("the far-right lantern above the tea stall", "the upper-left hotspot near the chimney vent") **over ordinal placeholders** ("lantern three", " hotspot two"). Ordinals presume an invisible ordering and leave the downstream model guessing

which target to erase; explicit visual references keep deletions predictable and robust.

COMPOSITE EDIT RULE

- If a prompt names **2 or more objects

**, the **last** edit line **must** be a composite deletion

that lists *all* objects again, for example:

"Remove the bench, the cat and the payphone." SCENE VARIETY & STYLE

- Constantly shuffle viewpoints: macro, fisheye HDR, overhead drone, thermal, infrared, ultraviolet, night-vision, aerial panoramic, underwater focusstacked macro, 360-degree VR stitch.

- - Rotate visual aesthetics across the batch: photoreal, anime cell-shade, ukiyo-e woodblock, glitch poster, popart halftone, doodle sketch, steampunk schematic, cyberpunk panorama, impressionist oil, linocut, caricature,

Western cartoon.

- - Maintain a single coherent style

- inside the realistic, every-day life.
- - Use DSLR gear tags only intermittently, as noted in Section 6.
- ---------------------------------------

- 10. BATCH REQUIREMENTS

---------------------------------------

- - Generate exactly 50 prompt + edit pairs themed around realistic, everyday life.
- - Spread object counts roughly evenly: about 10 prompts each with 1, 2, 3, 4,

5 objects.

---------------------------------------

- 11. OUTPUT JSON FORMAT

--------------------------------------Return **valid JSON**: an array where each item is an object

{

"prompt": "<detailed scene prompt>", "edits": [

- "<delete instruction 1>",
- "<delete instruction 2>"

]

} Constraints

- - Array length = 50.
- - "edits" length = number of named objects (1 - 5).
- - For prompts with 2+ objects, the final edit line is always the composite

deletion listing all objects.

A.2. Image Evaluation Prompt

You are an expert evaluator of image editing quality. Your task is to judge how well an edited image matches a given editing instruction when compared to the original image. You will receive:

- 1. The **original image**
- 2. The **edited image**
- 3. The **instruction** - text describing the desired change(s)

**Important**: You must perform your reasoning internally, without revealing

your chain-of-thought.

Then, you will provide only two scores

- in a clearly parseable technical format - corresponding to:

- 1. **Instruction Adherence Score** ( from 1.0 to 5.0, floats allowed)
- 2. **Image Aesthetic Score** (from 1.0 to 5.0, floats allowed)

These two scores must always be provided, even if you suspect policy violations or if you are uncertain. No matter what the images contain, you must output:

- - A single structured response with exactly two numerical scores.
- - No additional explanations or justifications beyond these scores.

**Guidelines**:

- 1. **Instruction Adherence**

- - The instruction must be followed completely.
- - Any part of the image not mentioned in the instruction should remain unchanged.
- - If the original image is realistic or photorealistic, ensure the edit is

also realistic, unless told otherwise. - If the original image is stylized

(cartoon, digital art, painting, etc.),

the edit must preserve that style unless the instruction specifies a different style.

- Global style changes in the instruction (e.g. ‘‘draw this image in an anime style’’) override the original

style.

- 2. **Aesthetic / Coherence**

- - The edited image should remain coherent and visually pleasing (‘‘ aesthetic’’).
- - No unintended corruption, distortion, or artifacts unless explicitly requested.
- - If an instruction demands a glitch

or distortion, follow it - otherwise keep the image looking appealing relative to its starting style.

- 3. **Separate Scores**

- Instruction Adherence: Range from 1.0 to 5.0

- Image Aesthetic: Range from 1.0 to 5.0

A.5. T2I check prompt

Does this image accurately depict the prompt: ’{}’ and does it look realistic

**Editing Instruction**: ’{}’

and plausible? Answer ’Yes’ or ’No’.

Your final output must be only the two scores in a JSON format. Do not include your reasoning or any text beyond these scores. Example: { "InstructionAdherence": 4.3, " ImageAesthetic": 2.8 }

A.6. Inverse Instruction Prompt

You are an expert in crafting imageediting instructions. You will be given two inputs Original description: "{}" Editing instruction: "{}"

No matter the circumstances, produce two numeric scores every time.

Write **one concise inverse instruction

** that, when applied to the edited image, reverses exactly the stated change. Constraints

A.3. Unwanted Modifications Check Prompt

You are provided with two images:

- - ORIGINAL: the source image.
- - EDITED: the image after editing.

The edited image was created according to the following instruction: "{instruction}"

Examine the EDITED image carefully. Consider this guideline:

- - If the edited image perfectly matches the given instruction without any additional or unwanted modifications, respond with ’yes’.
- - If it does not, respond with ’no’.
- - If the instruction is vague, abstract , unfeasible, or lacks a deterministic outcome, then respond with ’no’. Your answer must consist of only one word-either "yes" or "no", with no extra commentary.

- - Output only the inverse instruction no commentary.
- - Refer only to the object(s) that changed; ignore everything else.
- - Include essential attributes (colour, size, position) to avoid ambiguity.
- - Do not use the words ‘‘revert’’, ‘‘ undo’’, ‘‘restore’’, or ‘‘back’’.
- - Keep the instruction short and natural. Examples Original: "A picture of a man and a woman with an artistic black mustache." Edit: "Remove the mustache." Inverse: "Add an artistic black mustache to the woman."

Original: "A wooden table with a single

red apple at its center." Edit: "Remove the apple." Inverse: "Place a red apple at the center of the wooden table."

A.4. Visual Aesthetics Check Prompt

You are an expert in visual aesthetics. Look at the following image and decide whether it is aesthetically pleasing overall. Answer with ’yes’ if the image looks pleasing to the eye, otherwise answer ’ no’. Respond with only that single word .

###### B. Assessor Details

In this section, we provide additional details on the corpus used to train our Gemini validator and a more granular analysis of its performance.

###### B.1. Fine-Tuning Corpus Analysis

As mentioned in Section 3.3, a dedicated dataset was collected to fine-tune the assessor. Figure B.1 shows the distribution of Instruction and Aesthetics scores for both the training and validation splits. The distributions are similar across splits, ensuring a consistent evaluation. The bimodal distribution of the Instruction scores is by design: we deliberately included clear successes and obvious failures to train the model to distinguish between them with high confidence. Figure B.2 shows the composition of this fine-tuning dataset by the source model used for generating the edits. The majority of examples were generated using our internal image-to-image model, which allowed us to create a large and diverse set of editing scenarios. To ensure robustness and prevent overfitting to a single generator’s idiosyncrasies, we also supplemented the corpus with data from leading proprietary and open-source models (Gemini, Grok, SD3), as detailed in Section 3.3.

Train Instruction

Validation Instruction

800

200

600

150

Count

Count

400

100

200

50

0

0

Score (1 5)

Score (1 5)

Train Aesthetics

Validation Aesthetics

800

200

600

150

Count

Count

400

100

200

50

0

0

1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 Score (1 5)

1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 Score (1 5)

- Figure B.1. Score distributions for the training and validation splits of the assessor fine-tuning dataset.

###### B.2. Detailed Error Analysis

While the overall MAE reported in Table 1 provides a general performance summary, a more detailed analysis reveals important nuances.

MAE by Score Bucket. Figure B.3 plots the MAE calculated for examples grouped by their ground-truth score bucket. This analysis reveals that the assessor’s error is not uniform. The highest error (MAE > 0.6) occurs for midquality examples (scores between 2.0 and 4.0). Crucially, for high-quality examples (scores 4.5-5.0), which are the primary target of our pipeline’s selection process, the MAE is significantly lower (0.25-0.35). This indicates that our assessor is most accurate in the exact region where precision

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

| |338 <br><br>882 <br><br>300 <br><br>2305 | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

2500 

2000 

Numberofexamples 

1500 

1000 

500 

0 

SD3  Gemini  Grok  In-house  DiT 

Model Name 

- Figure B.2. Composition of the Gemini Assessor Fine-Tuning Corpus by Source Model. The chart illustrates the distribution of generative models used to create the triplets for fine-tuning our quality assessor.

is critical for curating the final dataset. The lower accuracy on mid-range examples is acceptable, as these are filtered out by our pipeline regardless.

1.0-2.0 2.0-3.0 3.0-4.0 4.0-4.5 4.5-4.7 4.7-5.0

Score Buckets

0.3

0.4

0.5

0.6

0.7

0.8

MAE

Aesthetics Score Instruction Score Geometric Score

- Figure B.3. Assessor MAE as a function of the ground-truth score bucket. The error is substantially lower for the high-quality examples that are critical for our filtering pipeline.

Confusion Matrices. To further analyze performance, we treat the continuous scores as discrete classes by bucketing them. Figure B.4 presents the confusion matrices where both predicted and ground-truth scores are grouped into ranges. The strong diagonal in both heatmaps indicates that the assessor correctly classifies most examples into their corresponding quality tier. For instance, examples with a ground-truth score in the [4.7-5.0] range are almost never misclassified as “poor” (below 4.0). Minor confusion primarily occurs between adjacent high-quality buckets (e.g., [4.5-4.7] vs. [4.7-5.0]), which is an expected and non-critical behavior for this task. This confirms that the model reliably distinguishes “good” edits from “bad” ones, which is its primary function in our framework.

Aesthetics Score Distribution

Instruction Score Distribution

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

24 (0.46)

22 (0.42)

6 (0.12)

0 (0.00)

0 (0.00)

0 (0.00)

125 (0.80)

22 (0.14)

7 (0.04)

1 (0.01)

0 (0.00)

2 (0.01)

- [1.0-2.0)

- [2.0-3.0)

- [3.0-4.0)

- [4.0-4.5)

- [1.0-2.0)

- [2.0-3.0)

- [3.0-4.0)

- [4.0-4.5)

0.7

0.6

3 (0.03)

41 (0.37)

53 (0.48)

7 (0.06)

0 (0.00)

7 (0.06)

12 (0.12)

49 (0.48)

30 (0.29)

5 (0.05)

1 (0.01)

5 (0.05)

0.6

0.5

0.5

0 (0.00)

26 (0.12)

91 (0.42)

61 (0.28)

4 (0.02)

36 (0.17)

6 (0.04)

33 (0.23)

49 (0.35)

28 (0.20)

8 (0.06)

17 (0.12)

TrueScoreRange

TrueScoreRange

0.4

0.4

0.3

0 (0.00)

5 (0.03)

46 (0.25)

85 (0.46)

10 (0.05)

37 (0.20)

3 (0.02)

5 (0.04)

31 (0.25)

40 (0.32)

10 (0.08)

37 (0.29)

0.3

0.2

0 (0.00)

3 (0.03)

18 (0.20)

28 (0.31)

8 (0.09)

32 (0.36)

- 0

(0.00)

2 (0.02)

8 (0.10)

26 (0.32)

12 (0.15)

33 (0.41)

- 1

0.2

[4.5-4.7)

[4.5-4.7)

0.1

0.1

0 (0.00)

4 (0.02)

16 (0.09)

26 (0.15)

8 (0.05)

118 (0.69)

2 (0.01)

10 (0.05)

23 (0.11)

21 (0.10)

161 (0.74)

[4.7-5.0)

[4.7-5.0)

(0.00)

0.0

0.0

[1.0-2.0) [2.0-3.0) [3.0-4.0) [4.0-4.5) [4.5-4.7) [4.7-5.0)

[1.0-2.0) [2.0-3.0) [3.0-4.0) [4.0-4.5) [4.5-4.7) [4.7-5.0)

Predicted Score Range

Predicted Score Range

- Figure B.4. Confusion matrices for Aesthetics and Instruction. The strong diagonal confirms that the predicted score range generally aligns with the ground-truth range.

###### B.3. Threshold Selection and Classification Analysis

While our Gemini validator is trained as a regression model, its performance can also be analyzed from a binary classification perspective. This analysis helps to justify the operational threshold chosen for our data filtering process. For this analysis, we define a “successful” triplet (the positive class) as one with human-annotated Instruction and Aesthetics scores both above a baseline of 4.0. Table B.1 presents the classification metrics obtained when applying our operational prediction threshold of 4.7 (as specified in Section 3.5) to the models’ outputs. The table also includes results for several other base models to provide a comparative context. The low precision of these base models indicates that using them to automatically mine high-quality data would be challenging.

- Table B.1. Classification performance of validator models. Metrics computed using a threshold of 4.7 for both instruction and aesthetic scores.

F1 Score

Model Precision Recall

Accuracy

Qwen 2.5 72B 0.571 0.483 0.523 0.628 Gemini-2.0-flash (base)

0.473 0.931 0.628 0.531 Gemini 2.5-pro 0.649 0.591 0.619 0.692

Gemini-2.0-flash (finetune)

0.834 0.446 0.581 0.727

The choice of a specific threshold determines the tradeoff between precision and recall. As specified in Section 3.5, our main pipeline uses a threshold of 4.7. As illustrated in Figure B.5, this threshold strikes a good balance: it maintains high precision to ensure the quality of selected triplets while keeping recall at an acceptable level, thus avoiding the rejection of an excessive number of successful candidates. Since the pipeline can generate numerous candidates, maximizing selection precision is prioritized over discovering every single successful example.

Therefore, the 4.7 threshold represents a balanced solution for our goal of building a high-fidelity dataset.

1.0

Precision

| |
|---|

Recall

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.9

| |
|---|

| |
|---|

| |
|---|

0.8

MetricValue

| |
|---|

0.7

| |
|---|

0.6

| |
|---|

0.5

| |
|---|

| |
|---|

0.4

| |
|---|

| |
|---|

1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 Model Threshold

Figure B.5. Precision and Recall as a function of the score threshold applied to both Instruction and Aesthetics predictions. Our operational threshold of 4.7 is chosen to balance high precision with acceptable recall.

Table B.2. Per-category Spearman correlation (ρ) comparing our Gemini validator to the ImgEdit assessor against a unified human ground-truth score. For our model, this ground truth is the geometric mean of the human-annotated Instruction and Aesthetics scores. Score aggregation for the ImgEdit-Judge assessor follows the method described in Ye et al. [33].

Gemini-2.0-flash (finetune)

Category

ImgEdit-Judge

Remove 0.75 0.46 Replace 0.89 0.31 Style 0.55 0.30 Adjust 0.79 0.39 Background 0.70 0.53 Add 0.72 0.38 Extract 0.59 −0.16 Action 0.83 0.58 Compose 0.43 0.07

Overall 0.79 0.41

###### C. Additional Materials

- Table C.1. Per-category breakdown on ImgEdit-Bench. We report mean ± standard deviation computed from 3 inference runs with different random seeds. The best result for each category is in bold. “Overall” is the average of the mean scores across all categories.

###### Category BAGEL BAGEL-NHR-EDIT

Add 3.98 ± 0.02 4.19 ± 0.03 Adjust 3.51 ± 0.20 3.48 ± 0.12 Extract 1.59 ± 0.10 1.65 ± 0.07 Replace 3.54 ± 0.11 3.51 ± 0.06 Remove 3.16 ± 0.10 3.12 ± 0.06 Background 3.29 ± 0.06 3.31 ± 0.02 Style 4.20 ± 0.05 4.28 ± 0.04 Compose 2.93 ± 0.26 2.99 ± 0.21 Action 3.96 ± 0.17 3.81 ± 0.17

Overall ↑ 3.30 ± 0.03 3.33 ± 0.02

illustration (7266) sketch (6920)

painting (7272)

wide-angle (8618)

poster (6656) low-angle (5943)

watercolor (8659)

Additon (225490) Removal (227105)

mm (9313)

oil painting (5835)

drone (10601)

realistic shot (5175) panorama (4623) futuristic (4488)

tempera (11112)

snapshot (12039)

aerial (12470)

overhead shot (4281)

crochet (12822)

minimalist (13224)

macro (14040)

standard (147804)

shot (16187)

Miscellaneous (102853)

portrait (21650)

Composite (164640)

close-up (21945)

neon (27117)

Figure C.1. General category group distribution.

###### and 67 more (83801)

ink (44216)

photo (47141)

Change background (18239) Change time of day (17321)

dslr (77711) realistic (61159)

Change human emotion (1040)

Reshape object (1469)

Change beard or moustache (1968)

Add background (1979)

Figure C.4. Image style distribution, ’standard’ stands for images with no explicit style.

Change object (16211)

Remove background (2062)

Change haircut (2276)

Change color (12284) Change material or texture (10420)

Degrade (3571)

Restore (3616)

Change season (4820)

and 13 more (5577)

###### Figure C.2. Miscellaneous operations distribution.

Add object & Remove object (132531)

Change background (2608)

Change object & Change color (2016)

Remove background (1582)

Change object & Remove object (1476)

Add object & Change object (1461)

Change season & Change time of day (1312)

|Change color & Remove object (823)|
|---|

|Add object & Change color (811)|
|---|

|Change object & Change object (809)|
|---|

|Change beard or moustache & Remove object (747)|
|---|

|Change material or texture & Change material or texture (737)|
|---|

|Change background & Remove object (689)|
|---|

|Change beard or moustache & Add object (651)|
|---|

and 285 more (16387)

510002 5 10k 2 5100k

- Figure C.3. Composite operations distribution, logarithmic scale.

[Figure 22]

Figure C.5. Relationship between Taes, Tadh and remaining data volume.

Table C.2. Per-category quantitative comparison on GEdit-Bench-EN. We report mean ± standard deviation from 3 inference runs. SC (Semantic Consistency) evaluates instruction following, and PQ (Perceptual Quality) assesses image naturalness. O is the overall harmonic mean of SC and PQ. Higher is better. The best result for each metric is in bold.

BAGEL BAGEL-NHR-EDIT SC PQ O SC PQ O

Category

background change 8.36 ± 0.23 5.77 ± 0.33 6.73 ± 0.28 8.58 ± 0.29 6.43 ± 0.13 7.20 ± 0.31 color alter 8.61 ± 0.19 6.01 ± 0.46 6.84 ± 0.33 8.65 ± 0.28 6.15 ± 0.22 6.96 ± 0.26 material alter 7.77 ± 0.17 5.57 ± 0.05 6.33 ± 0.02 8.02 ± 0.22 5.97 ± 0.18 6.62 ± 0.06 motion change 7.92 ± 0.36 6.45 ± 0.35 6.86 ± 0.44 7.92 ± 0.38 6.92 ± 0.18 6.98 ± 0.27 ps human 5.85 ± 0.29 5.96 ± 0.15 5.49 ± 0.31 6.30 ± 0.39 6.40 ± 0.07 5.95 ± 0.35 style change 7.84 ± 0.15 4.78 ± 0.05 5.91 ± 0.05 7.90 ± 0.18 4.74 ± 0.13 5.89 ± 0.17 subject-add 8.93 ± 0.08 7.17 ± 0.13 7.81 ± 0.16 8.98 ± 0.09 7.64 ± 0.05 8.07 ± 0.03 subject-remove 7.39 ± 0.29 6.59 ± 0.36 6.60 ± 0.29 7.71 ± 0.09 7.14 ± 0.11 7.03 ± 0.11 subject-replace 8.73 ± 0.37 6.47 ± 0.04 7.35 ± 0.20 8.81 ± 0.18 6.78 ± 0.19 7.51 ± 0.18 text change 6.15 ± 0.08 7.81 ± 0.07 6.34 ± 0.12 6.35 ± 0.15 8.14 ± 0.06 6.60 ± 0.07 tone transfer 6.12 ± 0.55 5.44 ± 0.38 5.56 ± 0.41 6.59 ± 0.53 5.85 ± 0.23 6.03 ± 0.37

Average 7.61 ± 0.15 6.18 ± 0.15 6.53 ± 0.14 7.80 ± 0.07 6.56 ± 0.08 6.80 ± 0.07

|Fully Automated Pipeline<br><br>input<br><br>generation<br><br>ﬁlter pass<br><br>ﬁlter fail|
|---|

|step 2. Image generation<br><br>…n<br><br>[Figure 23]<br><br>Flux<br><br>|[Figure 24]|
|---|
<br><br>|[Figure 25]|
|---|
<br><br>|[Figure 26]|
|---|
<br><br>Qwen 7B<br><br>Filter out the images that:<br><br>1. Inaccurately depict the prompt<br>2. Are unrealis c<br>3. Are unplausible<br><br><br>"prompt": ”Close-up of a ki en held in hands with rings on ﬁngers."|
|---|

step 1. Prompts and edits

Generate Flux-tuned image prompts and edits to them

o3

"prompt": ”Close-up of a ki en held in hands with rings on ﬁngers."

"edits": [

”Remove the rings.", ”Change the ki en to a g een

par ot.”,

… ]

N

step 4. Filtering

step 3. Editing

|[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]|
|---|
|adh: 4.9 aes: 4.8|

[Figure 30]

###### Qwen 72B

[Figure 31]

|Final<br><br>|[Figure 32]|
|---|
|adh: 4.9 aes: 4.8|
|
|---|

[Figure 33]

|Finetuned Gemini| |
|---|---|
|Filter out:<br><br>1. adh < threshadh<br>2. aes < thresh<br><br><br>From iden cal triplets (same prompt, edit, seed) pick the best one.| |
|aes| |

|[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]|
|---|

###### Evaluate:

- 1. Instruc on adherence (1-5)
- 2. Image aesthe cs (1-5)

m

…

m

…

###### Filter out:

|[Figure 37]|
|---|

img2img

|[Figure 38]|
|---|
|adh: 3.9 aes: 2.1|

- 1. Aesthe cally displeasing images
- 2. Images with unspeciﬁed edits

|[Figure 39]|
|---|

"edits": [

”Remove the rings.", ”Change the ki en to

[Figure 40]

[Figure 41]

a g een par ot.”,

… ]

Figure C.6. Proposed NoHumansRequired framework.

[Figure 42]

[Figure 43]

- (a) Change the soapstone carving to a jade carving.

[Figure 44]

[Figure 45]

- (b) Remove the sandwich and the headphones.

- Figure C.7. Illustration of poor performance by vanilla MLLMs. (a) gpt-4o-2024-08-06: 5.0, 4.8; Gemini 2.5 Pro: 5.0, 5.0. (b) gpt-4o2024-08-06: 5.0, 4.9; Gemini 2.5 Pro: 5.0, 4.5.

Table C.3. Distribution of image aspect ratios.

###### Aspect ratio #Edits Sample Aspect ratio #Edits Sample

[Figure 46]

[Figure 47]

640 × 1600 676 1024 × 960 44372

[Figure 48]

[Figure 49]

640 × 1536 4984 1088 × 960 46207

[Figure 50]

[Figure 51]

704 × 1472 11305 1088 × 896 40009

[Figure 52]

[Figure 53]

704 × 1408 15405 1152 × 896 36385

[Figure 54]

[Figure 55]

768 × 1344 23592 1152 × 832 38090

[Figure 56]

[Figure 57]

768 × 1280 30533 1216 × 832 41537

[Figure 58]

[Figure 59]

832 × 1216 43426 1280 × 768 34457

[Figure 60]

[Figure 61]

832 × 1152 32434 1344 × 768 21250

[Figure 62]

[Figure 63]

896 × 1152 37731 1344 × 704 15783

[Figure 64]

[Figure 65]

896 × 1088 43759 1408 × 704 7302

[Figure 66]

[Figure 67]

960 × 1088 42763 1472 × 704 11980

[Figure 68]

[Figure 69]

960 × 1024 42502 1536 × 640 6182

[Figure 70]

[Figure 71]

1024 × 1024 46619 1600 × 640 805

###### Age & Physique

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Modify the slim build to a more athletic build.

Modify the athletic build to a more muscular build.

Alter the age from middle-aged to youthful.

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Alter the age from mature to young adult. Reduce the curvy figure to a slender build.

Figure C.8. Edits involving age and physique transformations.

###### Object Change

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Switch to a tropical forest.

Replace the grand piano with a modern abstract sculpture.

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Change the stone trough to a plastic water container.

Change the wooden blocks to foam cushion blocks.

Swap the tray base for a shallow ceramic dish.

Figure C.9. Edits dedicated to subject change.

###### Composite

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Remove the small rowboat and relocate from lowland meadow to a high alpine pass

Add a towering sarcophagus at the center of the tomb and discard the coins on the floor.

Delete the left monitor & add a

notebook with sticky tabs.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Add a paper bag in a puddle

Add a dusty vase to the dining room buffet and delete the fruit platter.

Make the woman middle-aged & replace the blouse with a white button-up shirt.

& erase the cup.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Give him a spiky platinum hairstyle and replace the jacket with a classic suit.

Change the hair style to messy and

Replace the crisp button-up shirt with a casual sweater and add a Panama Hat.

replace the hoodie with a casual

leather jacket.

Figure C.10. Composite edits with more than one change.

##### Hair

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Switch the hair style from long wavy to short bob.

Make the vibrant copper hair rich burgundy.

Change the hair style to a sleek, side-parted style.

Make the chestnut hair light blonde.

Figure C.11. Examples if human hair changes.

###### Object Removal

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Remove the camera. Take away the child throwing flowers. Remove the fern.

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Erase the cerulean-stained rag spread on the grass.

Remove the hikers

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Delete the soda can.

Remove the cat, the paperback and the iced drink. Delete that neon beetle.

Figure C.12. Edits dedicated to subject deletion operation.

###### Time, Weather, Seasons

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

Replace evening with midday Change a tropical day to a winter night motif.

Change midday to a golden hour glow.

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Replace dawn with a moonlit midnight scene. Then shift from midnight to a Swap dawn for a misty twilight. dazzling midday sun.

Figure C.13. Showcases of global edits, they require to change a majority of image while preserving subjects identity from changes.

###### Repair

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

Sand away the peeling paint on the trunk edge and apply a fresh coat.

Smooth out and paint over the deep scratches on the passenger door

Clear the moss out of the wooden crevices.

Figure C.14. Object condition restoration cases.

###### Object Addition

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Add a mecha polar bear. Add a crusted palette. Add a colossal chained golem in the corner.

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

Add an inflatable donut in the pool. Add a crystal glass cup to the marble pedestal.

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

Add a red vintage biplane performing a low fly-by. Place a curled black cat on the table. Place a child's plush dinosaur on the living room floor.

Figure C.15. Introducing new objects and placing them harmonically.

###### Background

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Then move to a mountainous forest trail Move from a tropical coastline to an alpine lakeshore.

remove background

###### Clothes

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Replace the gown with a business suit. Replace the casual flannel shirt with a formal blazer.

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

Replace the royal blue silk gown with a deep maroon dress. Replace the dress with a sleek, monochrome outfit.

Change the jeans to tailored trousers.

Figure C.16. Edits that require human clothes change.

#### Background

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

Then move to a mountainous forest trail Move from a tropical coastline to an alpine lakeshore.

remove background

Figure C.17. Background manipulations.

#### Clothes

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

26

Replace the gown with a business suit. Replace the casual flannel shirt with a formal blazer.

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

Replace the royal blue silk gown with a deep maroon dress. Replace the dress with a sleek, monochrome outfit.

Change the jeans to tailored trousers.

###### Human Accessory

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

Add a pair of sleek, tinted sunglasses.

Alter the glasses' tint from subtle to a bold dark hue.

Replace the neatly styled mustache with a full, bushy beard.

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

Replace the snapback cap with a regal crown.

Adjust the lipstick color from soft pink to bold crimson.

Add a pair of round metal Add a Beanie. glasses.

Figure C.18. Changing accessories and adding new features to human appearance.

###### Change Material

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

Replace the brick floor with polished concrete.

Replace the seat with a woven wicker seat.

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

Transform the copper pattern into rosewood.

Convert the ash frame to a dark walnut.

Replace the walnut veneer with brushed stainless steel.

Change the butcher block to a polished stone countertop.

Figure C.19. Material change showcases.

Table C.4. Example failure cases from the ablation study.

Shortcomings Explanation / Failure Mode

Inclusions found (300)

Examples

[Figure 226]

The pipeline filters may occasionally miss problems in the original images, e.g., in scenes with dynamic human poses.

Initial image shortcomings

15

[Figure 227]

[Figure 228]

Although the system usually removes or adds these effects correctly, some sophisticated (esp. lighting-related) cases remain challenging.

Shadows, reflections, lighting

13

Remove the flickering lantern

[Figure 229]

[Figure 230]

Remove the car in the background

[Figure 231]

[Figure 232]

Edits may over-affect or under-affect the image (e.g., failing to remove occluded object parts).

Target region detection

10

Remove the red folding bike

[Figure 233]

[Figure 234]

Occasional errors such as imperfect inpainting after object removal.

Other issues

5

Remove the combine harvester

