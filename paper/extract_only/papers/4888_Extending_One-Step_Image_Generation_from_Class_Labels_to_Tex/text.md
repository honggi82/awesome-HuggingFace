# arXiv:2604.18168v1[cs.CV]20Apr2026

## Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation

Chenxi Zhao1,∗Chen Zhu2, Xiaokun Feng2, Aiming Hao2, Jiashu Zhu2, Jiachen Lei2, Jiahong Wu2,†Xiangxiang Chu2, Jufeng Yang1‡ 1 College of Computer Science, Nankai University 2 AMAP, Alibaba Group

zhaochenxi@mail.nankai.edu.cn hongxi.wjh@alibaba-inc.com

Ours SANA-Sprint Ours SANA-Sprint Ours SANA-Sprint

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Miniature city inside a vintage light bulb, tiny traffic and street lamps, dust on the glass, shallow DOF tilt-shift effect

Portrait of a conductor made of swirling ink and smoke, motion captured mid-crescendo, stark high-contrast monochrome

High-altitude balloon view of a volcanic eruption beneath, ash cloud forming a dragon, orange embers in stratosphere blue

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Close-up of a violin carved from ice, bow shedding frost, breath visible, studio rim lights, high-detail macro

Samurai armor woven from autumn leaves, forest floor carpeted in reds and golds, soft backlight, leaf veins sharp

A greenhouse on the moon growing lavender, Earth hanging low, regolith footprints, sun harsh, shadows crisp

Figure 1. Visual comparison of Our Model and SANA-Sprint in 4-step inference on challenging text. Our model achieves superior image quality and instruction following. The blue text denotes examples where SANA-Sprint fails.

#### Abstract

Few-step generation has been a long-standing goal, with recent one-step generation methods exemplified by MeanFlow achieving remarkable results. Existing research on MeanFlow primarily focuses on class-to-image generation. However, an intuitive yet unexplored direction is to extend the condition from fixed class labels to flexible text inputs, enabling richer content creation. Compared to the limited class labels, text conditions pose greater challenges to the model’s understanding capability, necessitating the effective integration of powerful text encoders into the MeanFlow framework. Surprisingly, although incorporating text conditions appears straightforward, we find that integrating powerful LLM-based text encoders using conventional training strategies results in unsatisfactory performance. To uncover the underlying cause, we conduct detailed analyses and reveal that, due to the extremely limited number of refinement steps in the MeanFlow generation, such as only one

∗Work done during the internship at AMAP. †Project Leader. ‡Corresponding author.

step, the text feature representations are required to possess sufficiently high discriminability. This also explains why discrete and easily distinguishable class features perform well within the MeanFlow framework. Guided by these insights, we leverage a powerful LLM-based text encoder validated to possess the required semantic properties and adapt the MeanFlow generation process to this framework, resulting in efficient text-conditioned synthesis for the first time. Furthermore, we validate our approach on the widely used diffusion model, demonstrating significant generation performance improvements. We hope this work provides a general and practical reference for future research on textconditioned MeanFlow generation. The code is available at https://github.com/AMAP-ML/EMF.

#### 1. Introduction

Generative models, exemplified by diffusion models [1, 2] and flow matching [3, 4], have achieved remarkable success in image content creation. Since generating high-quality images typically requires many denoising iterations, few-step generation [5–8] aims to reduce denoising steps to improve

efficiency, becoming an active research direction. As a representative and promising approach, flow map methods [7, 8] model the average velocity between two time steps, enabling efficient one-step generation. In particular, MeanFlow [9], a principled extension of flow matching, shows that flow maps can achieve performance comparable to standard models.

The acceleration potential demonstrated by MeanFlow has garnered widespread interest in subsequent research [10– 12]. Although these studies improve MeanFlow from various perspectives, their experiments primarily focus on class label conditioned image generation in the ImageNet setting [13]. To enable richer and more diverse content creation, an intuitiveyetunexploreddirectionistoextendtheconditioning from fixed class labels to flexible text inputs. Compared to the limited class labels, text conditions impose greater challenges on the generative model’s semantic understanding capabilities. Adapting to text conditioning thus necessitates the effective integration of powerful text encoders into the MeanFlow framework.

To enhance text understanding and instruction-following capabilities, modern text-to-image(T2I) generation models, such as SANA-1.5 [14], commonly replace the CLIP [15] or T5 [16] encoder with large language models (LLMs) [17, 18]. Following their practice, we attempt to integrate LLM-based text encoders into the MeanFlow framework to achieve one-step T2I generation. Surprisingly, we find that directly applying the widely used diffusion training loss to adapt LLM-based text encoders with diffusion models fails to yield satisfactory results. This motivates us to conduct detailed analyses to uncover the underlying cause.

The well-known stability issue of the JVP term has been repeatedly identified as the primary bottleneck in scaling consistency-based methods to large-scale applications such as T2I [19–21], so directly applying Mean Flow to T2I tasks is not an easy task. By contrasting our failed experiments with those in which MeanFlow succeeds, we offer two observations. First, fine-tuning MeanFlow on a pretrained model is substantially easier than training from scratch on DiT: a pretrained model already encodes a velocity field, so MeanFlow mainly needs to map instantaneous to average velocity [20, 22, 23]. Yet this presumed advantage is doubtful, since even large, state-of-the-art text-to-image models—which are more expressive than DiT—also struggle to learn the average velocity, calling into question the practical benefit of the pretrained velocity field for MeanFlow. Second, a recent line of work [24–30] investigates representation learning for image generation, aiming to enhance class separability and improve generation quality through stronger visual representations. Unlike class-conditional settings—where supervision is relatively clean and unambiguous and most studies center on diffusion models—T2I relies on complex textual conditions whose semantics must be carefully parsed and precisely grounded in the visual space. As a result, mak-

ing MeanFlow effective hinges on prioritizing the quality of the text encoder. Yet the true role of text encoders in visual generation remains insufficiently understood.

To further validate our hypothesis, we conducted the following analyses. To probe the instantaneous velocity in the generative velocity field, we examined how reducing the number of denoising iterations affects different text representations. Specifically, we evaluated standard generative models equipped with various text encoders under limitediteration settings. The results reveal substantial differences across text representations in their ability to preserve semantic fidelity when the number of steps is constrained. This indicates that, although some models achieve strong final performance, their underlying velocity fields may be of low quality and are only corrected through multiple denoising steps [31]. Through targeted analyses of the text encoders, we distilled two core insights:1) High-quality text representations are required to exhibit strong semantic discriminability, effectively capturing subtle differences among semantically similar texts; 2) They also need to possess good semantic disentanglement, clearly reflecting the distinct semantic components within the text. These two properties help reduce the difficulty of semantic discrimination under limited denoising steps, thereby improving the semantic fidelity of the generative model. We hypothesize that these characteristics alleviate the semantic discrimination burden faced by generative models under a limited number of denoising steps, thereby making them better suited to the MeanFlow framework.

Building on these insights, we, for the first time, enable MeanFlow to be effectively applied to T2I generation. Specifically, we validate the proposed method on the recently popular diffusion model BLIP3o-NEXT, achieving significant improvements across multiple evaluation benchmarks while demonstrating the scalability of our approach. We hope this work provides a general and practical reference for future research on text-conditioned MeanFlow generation.

In summary, our contributions are as follows:

- • To the best of our knowledge, we are the first to explore and realize the extension of conditioning in MeanFlow-based one-step generation from fixed class labels to flexible text inputs, enabling rich and efficient image generation.
- • Integrating powerful LLM-based text encoders into the Mean Flow framework, we find that under a limited number of iterations, different textual representations yield velocity fields of varying quality, which in turn induce pronounced differences in semantic fidelity. Furthermore, we systematically analyze the key properties of high-quality textual representations—discriminability and disentanglement.
- • Experiments on BLIP3o-NEXT validate our design, and our one-step T2I model, EMF (Extending MeanFlow to T2I), achieves competitive results on standard benchmarks.

#### 2. Related Work

- 2.1. Text to Image Generation

The field of video generation has witnessed significant advancements over the past year. These improvements stem frommulti-facetedinnovations: architecturaltransitionsfrom U-Net [32] to DiT [4, 33], denoising paradigm shifts from diffusion [1, 2] to flow matching [3, 4, 34, 35] optimization, and the evolution of text encoders from early text foundation models [15, 16] to LLMs [17, 18, 36–39]. Representative works such as the Stable Diffusion [4, 40, 41] and PixArt [42–44] series have continuously improved image generation capabilities. Recent large-scale models like FLUX [45, 46], Nano Banana [47], Qwen-Image [48], and HunyuanImage

- 3.0 [49] have demonstrated the ability to synthesize complex content and accurately edit images. To enhance semantic understanding and instruction following abilities of generative models, models such as Playground v3 [50], SANA-1.5 [14], and BLIP3o-NEXT [51] focus on integrating LLMs [17, 18] effectively into the generation framework. Meanwhile, given that high-quality image synthesis typically requires multiple denoising iterations, reducing the number of denoising steps to improve generation efficiency has become another important research direction.

- 2.2. Few-step Generation

Although diffusion models achieve excellent generation quality, their iterative sampling process incurs high computational cost. Considerable efforts have been devoted to accelerating sampling to fewer or even one step. A representative approach is Consistency Model [5, 6, 20], which enforces self-consistency by requiring predictions remain invariant under repeated model application or temporal interpolation across varying noise levels. Such constraints promote coherent and predictable generation trajectories, enabling accurate approximation with substantially fewer steps. Despite extensive studies on consistency models [22, 23, 52, 53], these methods are generally heuristic and introduced as external regularizers without rigorous theoretical foundations [11]. Recent works propose learning a flow map [7, 8] between two time steps to accelerate inference by reducing discretization error. In particular, MeanFlow [9] presents a principled framework for one-step generation, introducing average velocity as the ratio of displacement over a time interval. Unlike Flow Matching [3, 4] that models instantaneous velocity per time step, MeanFlow rigorously derives the relation between average and instantaneous velocities and designs a theoretically grounded training objective accordingly. Furthermore, MeanFlow achieves one-step generation performance comparable to standard multi-step models, attracting numerous follow-up improvements [10–12]. However, existing MeanFlow based studies primarily focus on class label-conditioned image generation. This work systematically explores and

implements extending conditioning from fixed class labels to flexible text inputs.

#### 3. Method

##### 3.1. Preliminary

MeanFlow. To avoid the costly ODE integration in standard flow matching inference, MeanFlow learns a flow map 𝑢𝜃(𝑧𝑡, 𝑡,𝑟) that directly predicts the transition from 𝑧𝑡 at time 𝑡 to 𝑧𝑟 at time 𝑟. The transition is defined as

𝑧𝑟 = 𝑧𝑡 + (𝑟 − 𝑡)𝑢𝜃(𝑧𝑡, 𝑡,𝑟), 𝑟 > 𝑡. (1)

For the true ODE trajectory, the ideal flow map corresponds to the average velocity over [𝑡,𝑟]. However, computing this quantity requires access to the full trajectory and is therefore expensive in practice. MeanFlow instead derives a self-consistent target by differentiating the transition equation along the trajectory, which gives

𝑢𝜃(𝑧𝑡, 𝑡,𝑟) = 𝑣(𝑧𝑡, 𝑡) + (𝑟 − 𝑡)

𝑑 𝑑𝑡

𝑢𝜃(𝑧𝑡, 𝑡,𝑟). (2)

Here, the total derivative is computed as 𝑑𝑡𝑑 𝑢𝜃 = 𝜕𝑡𝑢𝜃 + (∇𝑧𝑡𝑢𝜃)𝑣(𝑧𝑡, 𝑡), which can be efficiently implemented via JVP. Based on this relation, the target is defined as

𝑢˜(𝑧𝑡, 𝑡,𝑟) = 𝑣(𝑧𝑡, 𝑡) + (𝑟 − 𝑡) 𝑑𝑡𝑑 𝑢𝜃(𝑧𝑡, 𝑡,𝑟), and the model is trained with

LMF(𝜃) = E𝑡,𝑧𝑡,𝑟 ∥𝑢𝜃(𝑧𝑡, 𝑡,𝑟) − sg(𝑢˜(𝑧𝑡, 𝑡,𝑟))∥2 , (3)

where sg(·) denotes the stop-gradient operator for stable optimization.

##### 3.2. Different Text Representations Show Distinct Few-Step Generation Potentials

Existing studies on MeanFlow have achieved significant progress in class label-conditioned image generation tasks. This work attempts to extend MeanFlow to T2I generation, aiming to support richer and more diverse content creation. To enable semantic understanding and instruction following for flexible text inputs, recent mainstream T2I models have gradually replaced earlier foundational text encoders (such as CLIP [15] and T5 [16]) with powerful LLM-based text encoders. Following this trend, we attempt to effectively adapt LLM-based text encoders to the MeanFlow framework.

Reducing the number of generation steps limits a model’s refinement capacity [31, 54]. From the perspective of the velocity field, fewer steps are equivalent to taking a larger step along the instantaneous velocity at each time step, thereby reducing opportunities for gradual corrections to trajectory details and semantic boundaries. To examine how fewer steps affect different text representations, we evaluate two standard generative models (SANA-1.5 [14] and BLIP3o-NEXT [51]) under constrained-iteration settings. These models share the

1Step 2Step 4Step 8Step 12Step 16Step 20Step

1Step 2Step 4Step 8Step 12Step 16Step 20Step

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

SANA-1.5BLIP3o-NEXT

Sana-1.5Blip3o-NEXT

    )  

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

    (       

Inference Steps

- Figure 2. Left: Performance gap in few-step generation — For "Ducks float leisurely on vibrant, clear blue water.", SANA-1.5 misses the subject "ducks" in early steps, while BLIP3o-NEXT preserves it, yielding greater robustness and a more accurate velocity-field direction. Right: Under few-step sampling, BLIP3o-NEXT consistently outperforms SANA-1.5 on GenEval. BLIP3o-NEXT shows stronger subject preservation and downstream metric gains in few-step regimes.

same diffusion backbone but employ distinct text encoders. As shown in Fig. 2, even when the total number of steps is drastically reduced to 1, BLIP3o-NEXT maintains basic semantic integrity, whereas SANA-1.5 exhibits a substantial loss of semantic fidelity under few-step settings. The results in Fig. 2 further indicate that different text representations display varying robustness to velocity-field integration errors induced by step reduction. Evidently, the text representation associated with BLIP3o-NEXT demonstrates higher potential and quality for few-step generation; its ability to preserve basic semantic integrity even in the one-step regime suggests that the direction of BLIP3o-NEXT’s velocity field is more correct and better aligned with the target semantics. Subsequent experiments also confirm that this representation is better suited for MeanFlow-based one-step generation.

##### 3.3.High-QualityTextRepresentationsExhibitDiscriminability and Disentanglement

During image synthesis, the textual condition directly governs the quality of the generated output. When the text encoder is inadequate, it struggles to build a proper velocity field, causing slow model convergence and often requiring several corrective steps before an image aligns with its textual description. As stated in the previous section, under a multistep sampling setting, the model can repair the sampling denoising trajectory; thus the final outcome (quantified by GenEval) may remain unchanged, though the computational effort varies. To more precisely evaluate distinct text encoders, we examine two key properties—discriminability and disentanglement—across four encoders: the Blip3oNEXT text encoder [51], the SANA-1.5 text encoder [31], Clip-vit-large-patch14 [15], and T5-v1_1-xxl [16].

training split of COCO 2017 [55], we first encode each textual prompt with the text encoder under evaluation. We then compute cosine similarities between this text embedding and the image embeddings of all 118k image–caption pairs, ranking the pairs by similarity to obtain the top-k matches. First, weperformmeanpoolingalongthesequencedimension of the embeddings.

𝐿∑︁𝑠𝑒𝑞

1 𝐿𝑠𝑒𝑞

h(𝑥) =

e𝑡(𝑥), (4)

𝑡=1

Then we compute the cosine similarity.

h(𝑥)⊤h(𝑦) ∥h(𝑥)∥2 ∥h(𝑦)∥2

cos(𝑥, 𝑦) = 1 −

(5)

Fig. 3 visualizes the two most similar pairs retrieved for a representative query, where the queries were drawn from 1,000 samples randomly selected from the 118k dataset. The qualitative results reveal a clear pattern: because both the SANA-1.5 text encoder and T5 are trained exclusively on linguistic corpora and lack explicit vision–language alignment, the retrieved images exhibit low semantic relevance. In contrast, encoders such as BLIP3o-NEXT and CLIP, which are explicitly aligned on image–text pairs during pre-training, return qualitatively superior matches. To further quantify retrieval performance, we re-encode the retrieved images with a strong vision backbone (DINOv3 [57]) and calculate the cosine similarity between these image embeddings and the embedding of the query image. The aggregated scores, reported in Tab.1, provide a rigorous metric for comparing the alignment capabilities of different text encoders.

Discriminability. For a vision–language dataset composed of paired images and captions, an effective text encoder should generate representations that are well aligned with their corresponding image embeddings. Inspired by Wu et al. [56], we assess the cross-modal encoding quality through an image–text retrieval experiment. Specifically, on the 118k

Table 1. DINO evaluation of image-feature similarity for textretrieved images.

###### Model BLIP3o-NEXT CLIP Gemma T5 Score 0.734 0.730 0.713 0.634

###### Query

|BLIP3o-NEXT Text Encoder ✓|CLIP|
|---|---|
|[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>A wooden bench sitting in a grass field.<br><br>swing hanging by chains in woods.|[Figure 45]<br><br>[Figure 46]<br><br>A park with two wooden benches filled with lots of green grass.<br><br>a teddy bear wearing a green bow sitting on a swing|

[Figure 47]

[Figure 48]

A wooden swing hangs above plush, green, grass.

A the

|SANA-1.5 Text Encoder|T5|
|---|---|
|[Figure 49]<br><br>[Figure 50]<br><br>A wood bench under a tree in front of some bushes.<br><br>wooden bench under the shade a big tree in a wooded area.|[Figure 51]<br><br>[Figure 52]<br><br>Two sailboats travel in calm waters on the ocean.<br><br>A garden with broccoli, tomatoes and other vegetables|

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

coco train2017

[Figure 58]

[Figure 59]

[Figure 60]

Encoder

1000 samples

###### Query

A of

- Figure 3. On the COCO2017 train set [55], we encode query prompts using different text encoders, retrieve the Top-2 similar texts, and visualize their corresponding images. Among them, the images retrieved using the BLIP3o-NEXT text encoder are the most similar. This indicates that the distributions of its text and image representations are closely aligned, exhibiting strong discriminability.

Disentanglement. Another crucial property is that the text encoder’s output should be highly disentangled. Intuitively, after encoding a complete prompt, the resulting text embedding should retain the linguistic structure of the original text—i.e., exhibit semantic disentanglement. Moreover, when we shorten the prompt via sentence reduction to form subsequences, the distances between their embeddings and that of the full prompt should remain as small as possible.

ment. Leveraging the strong capabilities of the BLIP3oNEXT representation space, we propose an adaptation of the MeanFlow framework for T2I generation.

Specifically, given a pre-trained flow matching backbone conditioned on textual embeddings, we modify its architecture to explicitly support MeanFlow’s bidirectional time conditioning. In standard flow matching, a single temporal embedding layer 𝜙time(𝑡) is used to represent the current generation time 𝑡. In our adaptation, we duplicate the temporal embedding parameters to yield two separate embedding layers: 𝜙interval(·), 𝜙end(·), encodes the interval length 𝑡 − 𝑟, and the segment end time 𝑡, respectively. Given a start time 𝑟 and an end time 𝑡, we construct the conditional temporal embedding as: 𝜙cond(𝑡,𝑟) = 𝜙interval(𝑡 − 𝑟) + 𝜙end(𝑡).

Motivated by this idea, we conducted experiments on the entire set of prompts in DPG-Bench [58]. For each original prompt, we randomly removed portions of the text to create an ablated version. We then encoded both the original and ablated prompts with several different text encoders and recomputed the cosine distance between their embeddings. The experimental results are summarized in Tab. 2.

The conditioning embedding 𝜙cond and text features 𝜓text(𝑥text) jointly condition the velocity network:

The experimental results show that, compared with CLIP and T5, autoregressive text encoders trained with the nexttoken prediction paradigm perform better. In particular, the BLIP3o-NEXT text encoder and Gemma [38] achieve strong results and exhibit good disentanglement.

𝑢𝜃 𝑧𝑡, 𝑡,𝑟, 𝜓text = 𝑓𝜃 𝑧𝑡, 𝜙cond(𝑡,𝑟), 𝜓text . (6) During training, we adaptively sample timesteps (𝑡,𝑟)

from either a uniform or logit-normal distribution as follows: 𝑡,𝑟 ∼ 𝑝(·; 𝜇(𝑝), 𝜎(𝑝)), 𝑡 ≠ 𝑟, (7)

- Table 2. Evaluation of Text Encoder Disentanglement via Subsequence Similarity.

where 𝑝 is either uniform U(0, 1) or logit-normal, and the parameters 𝜇(𝑝), 𝜎(𝑝) are interpolated between initial and final values according to training progress 𝑝 ∈ [0, 1]. The ratio of non-equal timesteps (𝑡 ≠ 𝑟) is also increased adaptively throughout training. This strategy ensures balanced exposure to both short- and long-range segments, promoting stable learning of the mean velocity field.

Model BLIP3o-NEXT CLIP Gemma T5 Score 0.999 0.967 0.987 0.893

##### 3.4. Extending MeanFlow to T2I Generation

The full training procedure for our T2I MeanFlow adaptation minimizes the standard MeanFlow objective:

Building upon our evaluation in Sec. 3.3, the BLIP3o-NEXT text encoder consistently outperforms other LLM-based text encoders in terms of both discriminability and disentangle-

L𝑀𝐹(𝜃) = E𝑧𝑡,𝑡,𝑟 ∥𝑢𝜃(𝑧𝑡, 𝑡,𝑟, 𝜓text) − sg(𝑢tgt)∥2 , (8)

with 𝑢tgt defined as:

𝑑 𝑑𝑡

𝑢𝜃(𝑧𝑡, 𝑡,𝑟, 𝜓text), (9)

𝑢tgt = 𝑣𝜃(𝑧𝑡, 𝑡, 𝜓text) + (𝑟 − 𝑡)

where 𝜓text is produced by the BLIP3o-NEXT encoder and injected as the text condition. The derivative term is computed via Jacobian–vector products as in Eq. (14), and stop-gradient is applied to 𝑢tgt to stabilize training.

This adaptation extends MeanFlow to handle complex text-based conditioning in modern T2I models, enabling accurate and semantically faithful generation even in the one-step regime.

#### 4. Experiment

In this section, we provide a detailed description of our experimental setup, present the results of our method on mainstream image generation benchmarks and state-of-theart models, and offer deeper analyses and insights.

##### 4.1. Implementation Details

Training Recipe. We use approximately 170,000 samples (BLIP3o-60k [59], shareGPT-4o [60], and Echo-4o [61]) for our training. The learning rate is set to 1e-5, the batch size is 128, and the experiment runs for 150 epochs. We conduct experiments based on the BLIP3o-NEXT model, while keeping all other experimental settings consistent with BLIP3o-NEXT.

Evaluation details. We evaluate T2I generation on GenEval [62] and DPG-Bench [58]. GenEval provides a precise, attribute-focused evaluation of text–image faithfulness, while DPG-Bench emphasizes challenging long-form prompts that test instruction following and compositional robustness. In addition, we evaluated human perceptual preferences on the HPS-v2 dataset [63].

##### 4.2. Comparison with State-of-the-arts

In our GenEval tests, the model reached a score of 0.90 with just 4 sampling steps, nearly matching BLIP3o-NEXT’s 0.91, and outperforming nearly all other pretrained models, which usually require more than 20 steps. In addition, we surpassed every distilled model, which typically needs one or more teacher models during training, whereas our approach continues training from a single set of pretrained weights and still achieves high-quality generation in very few steps. Fig. 5 further illustrates qualitative examples from our approach, demonstrating high semantic fidelity and visual detail under 4-step generation.

As Tab. 4 shows, on the more challenging DPG-Bench our model maintains performance close to BLIP3o-NEXT, and across HPSv2 it yields substantial gains in human-preference alignment over BLIP3o-NEXT’s few-step sampling, matching the performance attained with 30 sampling steps.More

0.90

0.8

0.85 0.74

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.6

GenEval

0.4

| |
|---|

4 Steps 2 Steps 1 Steps

0.2

| |
|---|

0.0

1 2 3 4 5 6 7 8 9 10 Training Steps (w)

Figure 4. Ablation study of sampling steps in T2I MeanFlow. Strong 4-step performance is reached at ~1w training steps, while fewer steps need more training.

experimental comparisons on DPG-Bench are provided in the appendix.

##### 4.3. Ablation of Sampling Steps in T2I MeanFlow

Toinvestigatetheimpactofsamplingstepsongenerationquality, we monitor the performance of BLIP3o-NEXT throughout the training process under our MeanFlow framework. At each training checkpoint, the model is evaluated using the GenEval metric with three sampling configurations: 1-step, 2-step, and 4-step, as illustrated in Fig. 4.

We observe that under our MeanFlow training framework, the model not only delivers rapid performance improvements but also converges stably across different sampling step settings. With 4-step sampling, high generation quality is achieved within roughly 10k training steps, reaching a GenEvalscoreof0.90by60ksteps. Eveninmorechallenging few-step scenarios, the framework remains robust: with 2step sampling, the model attains a GenEval score of 0.85 at 70k steps, while 1-step sampling reaches 0.74 at 90k steps.

#### 5. Discussion

Our experimental method achieves significant improvements over BLIP3o-NEXT. However, several questions remain to be discussed.

• Can our method scale beyond two steps? Prior work on consistency-distilled models demonstrates that they can directly produce strong image generations with very few steps. However, increasing the sampling steps in such models typically yields marginal gains in image quality, and in some cases even negative gains at larger steps [7, 12, 54], making it difficult to achieve a favorable trade-off between inference time and generation quality. In contrast, our model continues to benefit from additional sampling steps: performance rises from 0.74 at 1-step to 0.90 at 4step, as shown in Fig. 4. Notably, this 4-steps result already approaches the BLIP3o-NEXT baseline obtained with 30 sampling steps (0.91). Furthermore, when extending our

- Table 3. GenEval results for pretrained, unified, and distilled models, plus few-step comparisons of BLIP3o-NEXT vs our MeanFlow adaptation. Our method attains the best distilled-model performance and rivals larger models even at 4-step sampling.

Model #Params Steps

Single Object

Two Objects

Counting Colors Position

Color Attribution

Overall Pretrained Models

SD3.5-L [4] 8B 28 0.98 0.89 0.73 0.83 0.34 0.47 0.71 FLUX.1-dev [45] 12B 50 0.98 0.81 0.74 0.79 0.22 0.45 0.66 SANA-1.5 [31] 4.8B 20 0.99 0.93 0.86 0.84 0.59 0.65 0.81 Cosmos-Predict2 [64] 0.6B 35 1.00 0.97 0.74 0.86 0.59 0.70 0.81 PixArt-𝛼 [42] 0.6B 20 0.98 0.50 0.44 0.80 0.08 0.07 0.48 Lumina-Image 2.0 [65] 2.6B 50 - 0.87 0.67 - - 0.62 0.73 HiDream-I1-Full [66] 3B 50 1.00 0.98 0.79 0.91 0.60 0.72 0.83 Seedream 3.0 [67] / / 0.99 0.96 0.91 0.93 0.47 0.80 0.84 GPT Image 1 [High] [68] / / 0.99 0.92 0.85 0.92 0.75 0.61 0.84 BLIP3o-NEXT [51] 3B 30 0.99 0.95 0.88 0.90 0.92 0.79 0.91

Unified Models

MetaQuery-L [69] 3B 30 - - - - - - 0.78 BLIP3-o-8B [59] 8B 30 - - - - - - 0.83 OpenUni-B-512 [70] 1.6B 30 0.99 0.91 0.74 0.90 0.77 0.73 0.84 Tar-7B [71] 9.6B 50 - 0.92 0.83 0.65 - - 0.83 TBAC-UniImage-3B [72] 4.6B 30 0.99 0.94 0.77 0.92 0.83 0.75 0.87 Qwen-Image [48] 20B 50 0.99 0.92 0.89 0.88 0.76 0.77 0.87

Distilled Models

SDXL-LCM [52] 2.6B 4 0.99 0.55 0.38 0.85 0.07 0.14 0.50 SDXL-Turbo [41] 2.6B 4 1.00 0.72 0.49 0.82 0.11 0.21 0.56 SDXL-Lightning [73] 2.6B 4 0.98 0.61 0.44 0.84 0.11 0.21 0.53 Hyper-SDXL [74] 2.6B 4 1.00 0.77 0.48 0.89 0.11 0.23 0.58 SDXL-DMD2 [75] 2.6B 4 1.00 0.76 0.52 0.88 0.11 0.24 0.58 SD3.5-L-Turbo [4] 8B 4 0.99 0.89 0.68 0.78 0.23 0.54 0.68 FLUX.1-schnell [45] 12B 4 0.99 0.88 0.64 0.78 0.30 0.52 0.69

SANA-Sprint [19] rCM [21] 14B 4 1.00 0.98 0.80 0.86 0.59 0.73 0.83

- 0.6B 4 1.00 0.90 0.71 0.89 0.61 0.54 0.77

- 1.6B 4 1.00 0.92 0.59 0.91 0.54 0.55 0.75

BLIP3o-NEXT and Ours under Few-Step Generation

BLIP3o-NEXT [51]

- 3B 1 0.81 0.40 0.40 0.56 0.38 0.23 0.46

- 3B 2 0.92 0.68 0.55 0.66 0.60 0.40 0.63

- 3B 4 0.99 0.93 0.84 0.84 0.86 0.70 0.86

EMF

- 3B 1 0.98 0.86 0.66 0.69 0.80 0.47 0.74

- 3B 2 0.99 0.91 0.81 0.86 0.86 0.66 0.85

- 3B 4 1.00 0.94 0.88 0.92 0.91 0.76 0.90

- Table 4. DPG-Bench and HPS-v2.1 results. Our MeanFlow adaptation matches BLIP3o-NEXT’s performance using far fewer sampling steps, with 4-step generation rivaling the 30-step baseline on both benchmarks.

DPG-Bench HPS-v2.1 Global Entity Attribute Relation Other Overall anime concept-art paintings photo Average

Model Steps

- 1 69.10 73.48 79.92 69.09 73.60 57.05 19.77 17.54 18.23 18.64 18.54

- 2 79.35 79.16 77.71 79.66 82.32 67.38 23.51 21.70 22.78 21.82 22.45

BLIP3o-NEXT

4 85.99 85.04 87.78 86.44 88.53 78.15 28.13 26.22 27.18 26.30 26.96 30 88.55 86.82 90.14 88.01 86.21 82.05 30.27 29.15 28.99 29.26 29.42

- 1 85.24 85.85 85.19 82.37 82.65 77.36 (+20.31) 26.64 25.60 25.53 25.32 25.77 (+7.23)

- 2 85.63 88.15 85.96 85.69 86.20 79.44 (+12.06) 28.02 26.83 27.03 26.96 27.21 (+4.76)

EMF

4 88.01 87.27 88.24 88.78 87.68 81.20 (+3.05) 30.03 29.02 29.09 28.86 29.25 (+2.29)

model’s sampling steps to 8, the DPG-Bench score increases from 81.20 to 81.94 compared to the 4-step setting. We attribute this to MeanFlow’s nature as a stable discretization ofanunderlyingcontinuousgenerativeflow—eachaddedstep more faithfully follows the average velocity field and reduces the approximation error. As a result, our method scales gracefully beyond 2 steps, delivering sustained improvements in both quantitative metrics and perceptual fidelity without sufferingfromthesaturationordegradationpatternsobserved

in conventional consistency-distilled approaches.

• Is the convergence speed of the MeanFlow dependent on the domain of the training data? Our initial attempt to apply MeanFlow fine-tuning on SANA-1.5 failed, likely due to a domain mismatch between SANA-1.5’s training data and MeanFlow’s data. To remove this confound, we re-trained SANA-1.5 text encoder with flow matching on the exact SFT data and hyperparameters used by BLIP3o-NEXT. As

Ours Sana sprint Flux.1-Schnell SDXL-DMD2 SD3.5-L-Turbo

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

A moonlit mangrove where roots form musical staffs, bioluminescent notes drifting on the tide, long exposure glow.

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Cyborg peregrine falcon mid-dive, titanium feathers splaying, Mach cone vapor, ultra telephoto compression.

Figure 5. 4-step sampling comparison of our method with existing distilled models. Our method achieves superior semantic fidelity and visual detail while closely matching complex text prompts. The blue text denotes examples where other models fail.

- Table 5. GenEval scores of SANA-1.5’s experiment. The encoder was additionally fine-tuned on SFT data to match the same domain, yet results show it still fails to achieve effective MeanFlow generation.

Sample Method

EncoderSFT

MeanFlow Train

Sampling Steps

GenEval

Flow Matching 20 0.81 Flow Matching ✓ 20 0.85

MeanFlow ✓ 4 0.50 MeanFlow ✓ 20 0.83 MeanFlow ✓ ✓ 4 0.47 MeanFlow ✓ ✓ 20 0.82

shown in Tab. 5, encoder fine-tuning improved SANA-1.5’s GenEval score from 0.81 to 0.85, but an additional MeanFlow stage remained ineffective. Notably, while MeanFlow did not learn the average velocity field, the model reached similar performance with 20 sampling steps, indicating MeanFlow does not disrupt the original trajectories.

Lastly, we also trained the SFT variant of BLIP3o-NEXT with Mean Flow, and present the 4-step GenEval test results during training in Fig. 6. The results again show that the SFT version of BLIP3o-NEXT converges stably, whereas SANA-1.5 exhibits training instability regardless of whether the text encoder is fine-tuned.

- 6. Conclusion

In this work, we present the first exploration and implementation of extending MeanFlow’s original class-labelconditioned one-step generation to flexible text conditioning, enabling richer and more efficient T2I synthesis. Through systematic analyses, we identify that high-quality text representations in few-step generation settings require both strong semantic discriminability and semantic disentangle-

0.90

0.9

0.80

0.8

GenEval

0.7

BLIP3o-NEXT RL

BLIP3o-NEXT SFT

SANA

0.6

SANA-encoder-train

0.50 0.47

0.5

0.4

1 2 3 4 5 6 7 8 9 10 Training Steps (w)

Figure 6. 4-step GenEval performance of different text encoders under MeanFlow training.

ment, which substantially improve semantic fidelity when only a limited number of denoising iterations are available. Guided by these insights, we adopt BLIP3o-NEXT’s powerful LLM-based text encoder—validated to possess the required semantic properties—and adapt MeanFlow on top of the BLIP3o-NEXT framework, achieving efficient text-conditioned synthesis. Empirical results validate our approach, achieving competitive one-step T2I generation with markedly improved synthesis quality. We believe this work offers valuable practical guidance and a strong reference for future research on text-conditioned MeanFlow generation.

#### 7. Acknowledge

This work was supported by National Natural Science Foundation of China (No. 62532004), Shenzhen Science and Technology Program (No. JCYJ20240813114229039), Natural Science Foundation of Tianjin (No. 24JCZXJC00040), Supercomputing Center of Nankai University.

#### References

- [1] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1, 3
- [2] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502,

2020. 1, 3

- [3] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In ICLR, 2023. 1, 3
- [4] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flowtransformersforhigh-resolutionimagesynthesis. InFortyfirst international conference on machine learning, 2024. 1, 3, 7, 2
- [5] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. 2023. 1, 3
- [6] Jonathan Heek, Emiel Hoogeboom, and Tim Salimans. Multistep consistency models. arXiv preprint arXiv:2403.06807,

2024. 3

- [7] Amirmojtaba Sabour, Sanja Fidler, and Karsten Kreis. Align your flow: Scaling continuous-time flow map distillation. arXiv preprint arXiv:2506.14603, 2025. 2, 3, 6
- [8] Nicholas M Boffi, Michael S Albergo, and Eric Vanden-Eijnden. Flow map matching. arXiv preprint arXiv:2406.07507, 2(3):9, 2024. 1, 2, 3
- [9] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025. 2, 3
- [10] Huijie Zhang, Aliaksandr Siarohin, Willi Menapace, Michael Vasilkovsky, SergeyTulyakov, QingQu, andIvanSkorokhodov. Alphaflow: Understanding and improving meanflow models. arXiv preprint arXiv:2510.20771, 2025. 2, 3
- [11] Yi Guo, Wei Wang, Zhihang Yuan, Rong Cao, Kuan Chen, Zhengyang Chen, Yuanyuan Huo, Yang Zhang, Yuping Wang, Shouda Liu, et al. Splitmeanflow: Interval splitting consistency in few-step generative modeling. arXiv preprint arXiv:2507.16884, 2025. 3
- [12] Kyungmin Lee, Sihyun Yu, and Jinwoo Shin. Decoupled meanflow: Turning flow models into flow maps for accelerated sampling. arXiv preprint arXiv:2510.24474, 2025. 2, 3, 6
- [13] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 2
- [14] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629,

2024. 2, 3

- [15] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–

8763. PmLR, 2021. 2, 3, 4

- [16] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 2, 3, 4
- [17] Qwen Team et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2(3), 2024. 2, 3
- [18] GemmaTeam, MorganeRiviere, ShreyaPathak, PierGiuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024. 2, 3
- [19] Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Song Han, and Enze Xie. Sana-sprint: One-step diffusion with continuous-time consistency distillation. arXiv preprint arXiv:2503.09641,

2025. 2, 7

- [20] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024. 2, 3
- [21] Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale diffusion distillation via score-regularized continuous-time consistency. arXiv preprint arXiv:2510.08431, 2025. 2, 7
- [22] Zhengyang Geng, Ashwini Pokle, William Luo, Justin Lin, and J Zico Kolter. Consistency models made easy. arXiv preprint arXiv:2406.14548, 2024. 2, 3
- [23] Yang Song and Prafulla Dhariwal. Improved techniques for trainingconsistencymodels. arXivpreprintarXiv:2310.14189,

2023. 2, 3

- [24] Jiachen Lei, Keli Liu, Julius Berner, Haiming Yu, Hongkai Zheng, Jiahong Wu, and Xiangxiang Chu. Advancing endto-end pixel space generative modeling via self-supervised pre-training. arXiv preprint arXiv:2510.12586, 2025. 2
- [25] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.
- [26] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.
- [27] Minglei Shi, Haolin Wang, Wenzhao Zheng, Ziyang Yuan, Xiaoshi Wu, Xintao Wang, Pengfei Wan, Jie Zhou, and Jiwen Lu. Latent diffusion model without variational autoencoder. arXiv preprint arXiv:2510.15301, 2025.
- [28] Ge Wu, Shen Zhang, Ruijing Shi, Shanghua Gao, Zhenyuan Chen, Lei Wang, Zhaowei Chen, Hongcheng Gao, Yao Tang, Jian Yang, et al. Representation entanglement for generation: Training diffusion transformers is much easier than you think. In NeurIPS, 2025.
- [29] Xiangxiang Chu, Renda Li, and Yong Wang. Usp: Unified selfsupervised pretraining for image generation and understanding. In ICCV, 2025.
- [30] Ji Xie, Trevor Darrell, Luke Zettlemoyer, and XuDong Wang. Reconstruction alignment improves unified multimodal models. arXiv preprint arXiv:2509.07295, 2025. 2

- [31] Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, Han Cai, et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer, 2025. 2, 3, 4, 7
- [32] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer,

2015. 3

- [33] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 3

- [34] Chubin Chen, Jiashu Zhu, Xiaokun Feng, et al. S2-guidance: Stochastic self guidance for training-free enhancement of diffusion models. arXiv preprint arXiv:2508.12880, 2025. 3
- [35] Chubin Chen, Sujie Hu, Jiashu Zhu, Meiqi Wu, Jintao Chen, Yanxun Li, Nisha Huang, Chengyu Fang, Jiahong Wu, Xiangxiang Chu, et al. Taming preference mode collapse via directional decoupling alignment in diffusion reinforcement learning. arXiv preprint arXiv:2512.24146, 2025. 3
- [36] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609,

2023. 3

- [37] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [38] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024. 5
- [39] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786,

2025. 3

- [40] RobinRombach, AndreasBlattmann, DominikLorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3
- [41] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 3, 7
- [42] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Zhongdao Wang, James T Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-𝛼: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In ICLR, 2024. 3, 7, 2
- [43] Junsong Chen, Yue Wu, Simian Luo, Enze Xie, Sayak Paul, Ping Luo, Hang Zhao, and Zhenguo Li. Pixart-{\delta}: Fast and controllable image generation with latent consistency models. arXiv preprint arXiv:2401.05252, 2024.

- [44] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-𝜎: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, pages 74–91. Springer, 2024. 3, 2
- [45] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 3, 7, 2
- [46] Rui Lan, Yancheng Bai, Xu Duan, Mingxing Li, Dongyang Jin, Ryan Xu, Dong Nie, Lei Sun, and Xiangxiang Chu. Fluxtext: A simple and advanced diffusion transformer baseline for scene text editing. arXiv preprint arXiv:2505.03329, 2025. 3
- [47] Google. Nano banana. https://developers. googleblog.com/en/introducing-gemini-25-flash-image, 2025. 3
- [48] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint

- arXiv:2508.02324, 2025. 3, 7, 2

[49] Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint

- arXiv:2509.23951, 2025. 3

- [50] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Chase Lambert, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024. 3
- [51] Jiuhai Chen, Le Xue, Zhiyang Xu, Xichen Pan, Shusheng Yang, Can Qin, An Yan, Honglu Zhou, Zeyuan Chen, Lifu Huang, et al. Blip3o-next: Next frontier of native image generation. arXiv preprint arXiv:2510.15857, 2025. 3, 4, 7
- [52] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 3, 7
- [53] Ling Yang, Zixiang Zhang, Zhilong Zhang, Xingchao Liu, Minkai Xu, Wentao Zhang, Chenlin Meng, Stefano Ermon, and Bin Cui. Consistency flow matching: Defining straight flows with velocity consistency. arXiv preprint arXiv:2407.02398, 2024. 3
- [54] Zidong Wang, Yiyuan Zhang, Xiaoyu Yue, Xiangyu Yue, Yangguang Li, Wanli Ouyang, and Lei Bai. Transition models: Rethinking the generative learning objective. arXiv preprint arXiv:2509.04394, 2025. 3, 6
- [55] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 4, 5
- [56] Sitong Wu, Haoru Tan, Zhuotao Tian, Yukang Chen, Xiaojuan Qi, and Jiaya Jia. Saco loss: Sample-wise affinity consistency for vision-language pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27358–27369, 2024. 4
- [57] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc

- Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 4
- [58] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024. 5, 6
- [59] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025. 6, 7, 2
- [60] Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, and Benyou Wang. Sharegpt-4o-image: Aligning multimodal models with gpt4o-level image generation. arXiv preprint arXiv:2506.18095,

2025. 6

- [61] Junyan Ye, Dongzhi Jiang, Zihao Wang, Leqi Zhu, Zhenghao Hu, Zilong Huang, Jun He, Zhiyuan Yan, Jinghua Yu, Hongsheng Li, et al. Echo-4o: Harnessing the power of gpt-4o synthetic images for improved image generation. arXiv preprint arXiv:2508.09987, 2025. 6
- [62] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-toimage alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023. 6
- [63] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341,

2023. 6

- [64] NVIDIA. Cosmos world foundation model platform for physical ai, 2025. 7
- [65] Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Jiakang Yuan, Xinyue Li, Dongyang Liu, et al. Luminaimage 2.0: A unified and efficient image generative framework. arXiv preprint arXiv:2503.21758, 2025. 7, 2
- [66] Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng Zhang, Fengbin Gao, Peihan Xu, et al. Hidream-i1: A high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705, 2025. 7, 2
- [67] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025. 7, 2
- [68] OpenAI. Gpt-image-1, 2025. 7, 2
- [69] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025. 7, 2
- [70] Size Wu, Zhonghua Wu, Zerui Gong, Qingyi Tao, Sheng Jin, Qinyue Li, Wei Li, and Chen Change Loy. Openuni: A simple baseline for unified multimodal understanding and generation. arXiv preprint arXiv:2505.23661, 2025. 7, 2
- [71] Jiaming Han, Hao Chen, Yang Zhao, Hanyu Wang, Qi Zhao, Ziyan Yang, Hao He, Xiangyu Yue, and Lu Jiang. Vision as a

- dialect: Unifying visual understanding and generation via textaligned representations. arXiv preprint arXiv:2506.18898, 2025. 7, 2
- [72] JunzheXu, YuyangYin, andXiChen. Tbac-uniimage: Unified understanding and generation by ladder-side diffusion tuning. arXiv preprint arXiv:2508.08098, 2025. 7, 2
- [73] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024. 7
- [74] Yuxi Ren, Xin Xia, Yanzuo Lu, Jiacheng Zhang, Jie Wu, Pan Xie, Xing Wang, and Xuefeng Xiao. Hyper-sd: Trajectory segmented consistency model for efficient image synthesis. In NeurIPS, 2024. 7
- [75] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455– 47487, 2024. 7, 2
- [76] Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Xiangyang Zhu, Fu-Yun Wang, Zhanyu Ma, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. Advances in Neural Information Processing Systems, 37:131278–131315, 2024. 2
- [77] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024. 2
- [78] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, Dayou Chen, Jiajun He, Jiahao Li, Wenyue Li, Chen Zhang, Rongwei Quan, Jianxiang Lu, Jiabin Huang, Xiaoyan Yuan, Xiaoxiao Zheng, Yixuan Li, Jihong Zhang, Chao Zhang, Meng Chen, Jie Liu, Zheng Fang, Weiyan Wang, Jinbao Xue, YangyuTao, Jianchen Zhu, Kai Liu, Sihuan Lin, Yifu Sun, Yun Li, Dongdong Wang, Mingtao Chen, Zhichao Hu, Xiao Xiao, Yan Chen, Yuhong Liu, Wei Liu, Di Wang, Yong Yang, Jie Jiang, and Qinglin Lu. Hunyuandit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding, 2024. 2
- [79] OpenAI. DALL·E 3. https://openai.com/research/dall-e-3, September 2023. 2
- [80] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast highresolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024. 2

## Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation

### Supplementary Material

#### 8. Velocity Field Learning Challenges: ClassLabel vs. Text Conditions

denoising trajectory is relatively smooth. This smoothness indicates that the instantaneous velocity at each step closely aligns with the overall average velocity, making it straightforward for the model to predict this average. This stability is rooted in the embedding space, where class-label features form sparse clusters with large inter-class margins, ensuring category integrity and attribute accuracy even in single-step generation.

𝑡 − 𝑟 𝑢(𝑧,𝑟,𝑡)

𝑣

𝑣

𝑢(𝑧, 𝑟, 𝑡)

𝑢(𝑧, 𝑟,𝑡)

In stark contrast, the higher complexity and coupled nature of textual conditions lead to a more tortuous denoising trajectory. This winding path causes a significant divergence between the instantaneous and average velocities, often manifesting as early-stage semantic drift. Consequently, the model struggles to converge on the correct average velocity, necessitating additional corrective steps. This difficulty is exacerbated by the nature of textual embeddings, which reside in densely packed neighborhoods and inherently complicate the estimation of a stable average velocity.

𝑡 − 𝑟 𝑢(𝑧,𝑟, 𝑡)

- Figure 7. Denoising Trajectory Comparison. Simple class-label conditioning (left) yields a smooth path, whereas complex text conditioning (right) results in a tortuous path.

Extending MeanFlow from class-label conditioning to textual conditioning introduces fundamentally different challenges for velocity field learning.

These observations directly link the challenges of textconditioned MeanFlow to the key properties of high-quality textual representations introduced in the main text: strong discriminability and disentanglement are essential for preserving semantic fidelity when the velocity field is learned under limited denoising steps.

Representation Separability. Class labels are discrete and well-separated in the embedding space, enabling the velocity field to maintain a stable direction. Consequently, the denoising trajectory is smooth, with the instantaneous velocity at each step closely aligning with the overall average velocity. This stability makes predicting the average velocity straightforward, ensuring high fidelity even in few-step generation. In contrast, textual embeddings form dense and continuous distributions where semantically related prompts (e.g. blue teapot vs. red teapot) occupy neighboring regions, reducing the discriminability of the representation. This density forces the velocity field to navigate fine-grained semantic distinctions, resulting in a more tortuous trajectory. The instantaneous velocity frequently diverges from the average, leading to semantic drift and necessitating additional corrective iterations to converge on the target concept.

#### 9. Additional Experiment on text encoder

We conducted analyses of the post-trained SANA-1.5 and OpenUni text encoders, and ran mean-flow experiments on OpenUni. We chose OpenUni because it shares the SANA1.5 diffusion backbone, but uses a InternVL3–based text encoder. Tab. 7 compares the two encoders. After training, Gemma becomes less discriminative but more disentangled, which helps 20-step generation by refining the language space. In contrast, mean-flow few-step generation needs strong image–text discriminability, so it still fails even after encoder training. We also train mean flow on OpenUni under thesamesetup(Tab.8). OpenUniperformsbetterthanSANA1.5, benefiting from stronger text-encoder representations, but it still falls short of the original model due to insufficient discriminability.

Instruction Complexity. Class labels typically encapsulate a single semantic concept, whereas natural language prompts often bind multiple attributes, objects, and spatial relations (e.g., a blue ceramic teapot on a wooden table next to a vase of red tulips). In few-step regimes, the model has limited opportunities for correction. Therefore, inadequate disentanglement of these semantic components can easily lead to binding errors, missing objects, or incorrect attribute assignments.

#### 10. Inference Time Comparison.

When generating images from the same prompt and timing diffusion sampling only, BLIP3o-NEXT on H200 takes 1.24 s with 30 steps, while ours takes 0.22/0.12/0.08 s (4/2/1 steps). For end-to-end generation with different prompts,

The generation dynamics differ significantly between class-label and textual conditioning, a contrast visualized in Fig. 7. Under the simpler class-label conditioning, the

- Table 6. Quantitative evaluation results on DPG-Bench. Our method consistently outperforms distilled few-step models of comparable scale under the same denoising step settings.

Model #Params Steps Global Entity Attribute Relation Other Overall Pretrained Models

PixArt-𝛼 [42] 0.6B 20 74.97 79.32 78.60 82.57 76.96 71.11 Lumina-Next [76] 4B 20 82.82 88.65 86.44 80.53 81.82 74.63 Playground v2.5 [77] / / 83.06 82.59 81.20 84.08 83.50 75.47 Hunyuan-DiT [78] 1.5B 50 84.59 80.59 88.01 74.36 86.41 78.87 PixArt-Σ [44] 0.6B 20 86.89 82.89 88.94 86.59 87.68 80.54 DALL-E 3 [79] / / 90.97 89.61 88.39 90.58 89.83 83.50 FLUX.1 [Dev] [45] 12.7B 50 74.35 90.00 88.96 90.87 88.33 83.84 SD3 Medium [4] 2B 50 87.90 91.01 88.83 80.70 88.68 84.08 HiDream-I1-Full [66] 3B 50 76.44 90.22 89.48 93.74 91.83 85.89 Lumina-Image 2.0 [65] 2.6B 50 - 91.97 90.20 94.85 - 87.20 Seedream 3.0 [67] / / 94.31 92.65 91.36 92.78 88.24 88.27 GPT Image 1 [High] [68] / / 88.89 88.94 89.84 92.63 90.96 85.15

Unified Models

MetaQuery-L [69] 3B 30 - - - - - 81.10 BLIP3-o-8B [59] 8B 30 - - - - - 80.73 OpenUni-B-512 [70] 1.6B 20 85.87 87.33 86.54 86.91 89.43 80.29 Tar-7B [71] 9.6B 50 - 88.62 88.05 93.98 - 84.19 TBAC-UniImage-3B [72] 4.6B 30 83.52 87.94 87.80 87.17 87.02 80.97 Qwen-Image [48] 20B 50 91.32 91.56 92.02 94.31 92.73 88.32

Distilled Models

SDXL-DMD2 [75] 2.6B 4 81.16 80.68 82.47 83.52 80.05 74.24 SD3.5-L-Turbo [4] 8B 4 90.99 87.43 87.42 87.81 86.10 81.97 SD3.5-Turbo [80] 8B 4 80.12 86.13 84.73 91.86 78.29 79.03 FLUX.1-schnell [45] 12B 4 86.62 90.82 88.35 93.45 82.00 84.94 SANA-Sprint [19] 1.6B 4 83.84 88.54 88.50 87.40 86.41 81.08

BLIP3o-NEXT and Ours under Few-Step Generation

BLIP3o-NEXT

- 3B 1 73.60 69.10 73.48 79.92 69.09 57.05

- 3B 2 82.32 79.35 79.16 77.71 79.66 67.38 3B 4 88.53 85.99 85.04 87.78 86.44 78.15 3B 30 86.21 88.55 86.82 90.14 88.01 82.05

EMF

- 3B 1 85.24 85.85 85.19 82.37 82.65 77.36

- 3B 2 85.63 88.15 85.96 85.69 86.20 79.44 3B 4 88.01 87.27 88.24 88.78 87.68 81.20 3B 8 89.07 88.13 88.96 87.49 86.34 81.94

- Table 7. Experiments on discriminability and disentanglement metrics for the trained SANA-1.5 and OpenUni text encoders.

Table 8. Results of OpenUni trained on Mean Flow.

###### Steps FM-GenEval MF-Geneval

Metric Value Disc. (Gemma-train) 0.694 Disc. (OpenUni) 0.724 Dise. (Gemma-train) 0.997 Dise. (OpenUni) 0.996

20 0.86 0.76 4 0.73 0.70 2 0.31 0.61 1 0.11 0.59

BLIP3o-NEXT (30 steps) takes 11.3 s, whereas our 4-step version takes 9.87 s. The remaining time is mostly spent on autoregressive text-embedding generation.

#### 11. User Study and ImageReward Result

Considering instruction-following ability, we conducted PickScore and a user study on 50 prompts (similar to Fig.1 in our manuscript). We recruited 20 users, who compared images generated by five models for each prompt and answered: “Which result best matches the prompt?”

Table 9. Performance comparison across different models.

###### Model PickScore User Study

SDXL-DMD2 0.14 0.09 SD3.5-L-Turbo 0.16 0.13 FLUX.1-schnell 0.17 0.12 SANA-Sprint 0.25 0.16 Ours 0.28 0.49

All models use 4-step generation, and both experiments show that our method performs better.

#### 12. Additional Quantitative and Qualitative Results

We provide supplementary quantitative and qualitative evaluations to further validate the effectiveness of our approach under limited denoising steps.

DPG-Bench evaluation. Generating high-fidelity images from complex and detail-rich textual prompts in a limited number of denoising iterations is a highly challenging task. To assess our model’s capability in this regime, we conduct extensive tests on DPG-Bench, which focuses on long-form prompts with intricate attribute bindings and spatial relationships. As reported in Tab. 6, our method consistently outperforms equally sized distilled few-step models under the same step setting, despite the inherent difficulty of the benchmark. Notably, with only 8 sampling steps, our model delivers performance on par with the BLIP3o-NEXT baseline using 30 steps, and even under the challenging 1-step regime, it surpasses widely-used distilled models such as SDXL-DMD2 and Playground v2.5 in overall score.

Vertical comparison across sampling steps. We additionally present the few-step generation results of our MeanFlow adaptation under 1-step, 2-step, 4-step, and 8-step settings, comparing them with the BLIP3o-NEXT baseline trained with standard Flow Matching under the same sampling step configurations.

As shown in Fig. 8, our method achieves an effective tradeoff between inference speed and output quality: whereas the Flow Matching baseline exhibits noticeable blurring and loss of fine details when the number of sampling steps is reduced, our MeanFlow sampling retains salient object structures and fine-grained textures even at extremely low step counts, producing visually coherent and semantically faithful images at a fraction of the baseline’s inference time.

Horizontal few-step comparison. We also present sideby-side comparisons between our model and other few-step approaches at same 4-step settings (Fig. 9). These results highlight our model’s ability to preserve fine-grained details and adhere to textual instructions more faithfully than existing distilled models, across a diverse set of challenging prompts.

1-step 2-Step 4-step 30(FM)/8(MF)-step

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

On a rustic wooden table,

FlowMatchingMeanFlow

three ripe eggplants with a glossy royal purple skin

are carefully arranged in a neat row. Their plump, oblong shapes complement the table's textured surface, and they cast soft shadows in the warm, ambient light. Nearby, the woven pattern of a tan-colored napkin peeks out from beneath the vibrant, richly colored vegetables.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

- Figure 8. Representative visual results on DPG-Bench. Compared to the blurred outputs of few-step Flow Matching (FM) inference, our MeanFlow (MF) approach produces relatively sharp images even with a single sampling step, and with 8 sampling steps achieves visual quality comparable to Flow Matching using 30 steps, demonstrating a favorable trade-off between generation speed and visual fidelity.

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Ours Sana sprint Flux.1-Schnell SDXL-DMD2 SD3.5-L-Turbo

Museum of raincoats organized by storms survived, tags tell

weather stories, cool gray palette.

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Portrait of a botanist with veins of leaves in the skin, greenhouse highlights, humid breath.

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Astronaut barbershop cutting hair into nebula shapes, clippings sparkle, chrome tools, humor.

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Opera singer composed of swirling colored smoke, spotlight slicing haze, spectral form, velvet curtains.

- Figure 9. Additional comparisons under 4-step sampling between our method and existing distilled models. Our approach achieves higher semantic fidelity and richer visual details, closely adhering to complex text prompts. Blue text indicates cases where competing models fail to accurately render the described content.

