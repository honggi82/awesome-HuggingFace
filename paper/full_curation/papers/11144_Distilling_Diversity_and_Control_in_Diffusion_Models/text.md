## Distilling Diversity and Control in Diffusion Models

##### Rohit Gandikota∗ David Bau Northeastern University

Base Model: 9.22s Distilled Model: 0.64s Diversity Distillation (Ours): 0.64s

Slider Distilled from SDXL-Base

- Big Eyes +

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

# arXiv:2503.10637v4[cs.GR]10Nov2025

[Figure 7]

DMDLightningTurboLCM

[Figure 8]

“Cartooncharacter”

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

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

[Figure 27]

[Figure 28]

(a) (b) (c) (d)

Figure 1. Diversity Distillation: (a) SDXL-Base is very slow (9.22s) but has good sample diversity for the prompt ”Cartoon character”, sampling a wide range of styles, creatures, backgrounds, and poses. (b) SDXL-DMD2 is fast (0.64s) but sacrifices diversity. With the same prompt, samples all have the same style, pose, species, and context. (c) We show how the diversity of the base model can be distilled into the fast model by substituting the first timestep, achieving both speed and diversity (0.64s). Control Distillation: (d) Despite the lack of sample diversity in distilled models, control mechanisms like Concept Sliders trained on base models transfer perfectly to distilled variants, demonstrating that the representational structure for diversity exists but is not spontaneously activated during generation.

#### Abstract

model for only the first critical timestep before switching to the distilled model. This single intervention restores sample diversity while maintaining computational efficiency. We provide both causal validation and theoretical support showing why the very first timestep concentrates the diversity bottleneck in distilled models. Our code and data are available at distillation.baulab.info

Distilled diffusion models generate images in far fewer timesteps but suffer from reduced sample diversity when generating multiple outputs from the same prompt. To understand this phenomenon, we first investigate whether distillation damages concept representations by examining if the required diversity is properly learned. Surprisingly, distilled models retain the base model’s representational structure: control mechanisms like Concept Sliders and LoRAs transfer seamlessly without retraining, and SliderSpace analysis reveals distilled models possess variational directions needed for diversity yet fail to activate them. This redirects our investigation to understanding how the generation dynamics differ between base and distilled models. Using xˆ0 trajectory visualization, we discover distilled models commit to their final image structure almost immediately at the first timestep, while base models distribute structural decisions across many steps. To test whether this first-step commitment causes the diversity loss, we introduce diversity distillation, a hybrid approach using the base

#### 1. Introduction

Distilled diffusion models generate images in far fewer timesteps but lack the sample diversity of their original base model counterparts. In this paper we ask: Why do distilled diffusion models collapse in sample diversity?

This limitation fundamentally constrains practical applications. When users generate multiple images from the same prompt, they expect varied outputs [24] that explore different creative interpretations. Figure 1 illustrates this

∗Correspondence to gandikota.ro@northeastern.edu

problem: while base models produce diverse structural compositions across random seeds, distilled variants converge to visually similar results despite their computational advantages. This diversity collapse limits the utility of efficient models in creative workflows, design exploration, and applications requiring multiple candidate generations [24].

The challenge is particularly puzzling given recent advances in distillation quality. Diffusion models demonstrate unprecedented generation quality [4, 14, 19, 27, 28], yet their computational demands create deployment barriers. Modern distillation techniques [19, 20, 23, 32, 38, 39] have successfully maintained image quality while reducing inference steps from 50-100 to just 1-4. Some distilled models even achieve better distributional diversity than their base counterparts. However, distributional diversity is different from sample diversity. Distributional diversity is the ability to cover the full spectrum of training data across varied prompts, while sample diversity is the ability to generate diverse images for a single prompt under different seeds. SDXL-DMD2 [38] shows superior distributional diversity with lower FID [13] scores on COCO [21] datasets but exhibits poor sample diversity, making the collapse even more mysterious.

To solve this puzzle, we first investigate whether distillation damages the model’s concept representations by analyzing the top variation directions [10] in base models and examining if these directions exist in distilled models. Surprisingly, we find that distilled models retain the variational directions needed for diversity—they simply fail to activate them during generation. This leads us to hypothesize that timestep dynamics could be the reason, supported by theoretical analysis showing how distillation collapses the output variance at early timesteps. We test this by visualizing what models predict at intermediate timesteps [14, 36], revealing that distilled models commit to their final structure almost immediately at the first step while base models distribute decisions across many steps. To causally test our hypothesis, we introduce a hybrid inference technique and demonstrate that sample diversity can be drastically improved by simply modifying the first timestep of distilled models.

Our investigation reveals that the problem lies not in what distilled models learn, but in how they generate. Through careful analysis of model representations and generation dynamics, we identify the root cause and demonstrate a simple solution that restores sample diversity without sacrificing efficiency. Our findings challenge conventional assumptions about the diversity-efficiency tradeoff and provide actionable insights for both researchers developing distillation methods and practitioners deploying efficient diffusion models.

#### 2. Related Works

Diffusion Distillation: While diffusion models [14, 33, 34] excel at high-quality image synthesis, their requirement for 20-100 sampling steps creates significant computational bottlenecks. Diffusion distillation techniques address this limitation by finetuning base models that maintain quality with fewer steps. Progressive distillation [30] established the foundation by iteratively training student models to match teacher outputs with half the sampling steps. Recent approaches have further improved efficiency through distinct methodologies: Adversarial Diffusion Distillation [32], implemented in SDXL-Turbo, integrates score distillation with adversarial training to enable high-fidelity generation in just 1-4 steps, effectively combining diffusion guidance with GAN-like discriminators. Distribution Matching Distillation [38], featured in SDXLDMD2, takes a different approach by focusing on matching output distributions rather than specific trajectories, eliminating regression loss and implementing a two time-scale update rule that significantly improves training stability. For balancing quality and mode coverage, Progressive Adversarial Diffusion Distillation [20] in SDXL-Lightning employs staged training with specialized latent-space discriminators, offering flexibility through checkpoints optimized for 1-8 step inference. Latent Consistency Models [23], applied in SDXL-LCM, ensure consistency in latent representations across noise levels for distillation, reducing steps to 4-8 while preserving generation quality. Despite these advances in efficiency, the relationship between model distillation and sample diversity has remained largely unexplored.

Concept Representation: Research in concept representation for diffusion models has evolved from basic personalization to sophisticated control mechanisms [2, 3, 8, 25, 37, 40]. Textual Inversion [6] captures the semantics of a concept with learnable embeddings in text space without modifying model weights, allowing personalization with just a few images. DreamBooth [29] advanced this approach by fine-tuning models with unique identifiers and a specialized prior preservation loss. Custom Diffusion [17] streamlined this process by optimizing only crossattention layers, reducing storage requirements to just 3% of model weights while enabling multi-concept customization simultaneously. For precise attribute manipulation, Concept Sliders [8] introduced low-rank adaptors that create interpretable controls over specific visual attributes like age or weather conditions. This technique was expanded in SliderSpace [10], which decomposes a model’s visual variation capabilities, i.e. sample diversity, into multiple controls from a single prompt, enhancing creative exploration. Recent works have addressed the issue of suboptimal mode following in finetuned models by implementing an inference-time guidance annealing [15]. Complementary to

Base→Base Base→DMD Base→LCM Base→Turbo Base→Lightning

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

ConceptSliders

“BigEyes”

Distilled

“CartoonCharacter”

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

CustomDiffusion

Distilled

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

“Car”Variation32

Sliderspace

Distilled

Figure 2. Control directions (Sliders [17]), customization adapters (Custom Diffusion [8]), and variational directions (Sliderspace [10]) trained on SDXL-Base transfer to all distilled models without additional finetuning. SliderSpace results suggest that top variation directions that capture sample diversity in the base model exist in distilled models but are not spontaneously activated during generation.

these control mechanisms, hierarchical concept trees [1, 35] were developed to enable intuitive exploration of related visual concepts. Recent work has also addressed ethical concerns through targeted concept removal techniques by editing selective weights [7, 9, 22], redirecting concept representations [16, 26]. Since distillation modifies the UNet model of diffusion, in this work, we mainly focus on custom concept and control representations that are captured in UNet modules. Our work uniquely explores whether such control mechanisms can be distilled from base to efficient models without additional training.

#### 3. Control Distillation

Having established that distilled models suffer from reduced sample diversity (Figure 1), we investigate the underlying cause. Our first hypothesis is that distillation erodes the model’s concept space: that the distilled model fails to learn or retain the representations necessary for diverse generation. If this were true, the model would naturally produce less varied outputs because the required conceptual building blocks would be missing or damaged.

To test this hypothesis, we examine whether the representational structure needed for diversity is preserved in distilled models. We pose this as representational compat-

ibility: do the variation directions and control mechanisms present in base models transfer to distilled models without retraining?

###### 3.1. Experimental Setup

We test three complementary families of controls that probe different aspects of model representations:

Concept Sliders [8, 10] provide fine-grained control over visual attributes (e.g., age, weather, eye size) through low-rank adaptations. These test whether distilled models preserve the semantic understanding needed to manipulate specific visual properties.

Customization mechanisms including Custom Diffusion [17] and DreamBooth [29] capture nuanced, userdefined concepts through specialized training procedures. These mechanisms test whether distilled models retain the capacity to encode and recall complex, personalized concepts.

SliderSpace [10] decomposes the seed-induced variations of a base diffusion model into interpretable, continuous directions—each corresponding to a controllable visual factor (e.g., texture, lighting, layout). This directly analyzes whether the variational components responsible for sample diversity are present in distilled models.

###### Method Concept Base→Base Base→DMD Base→LCM Base→Turbo Base→Lightning

Age 20.4 17.8 27.1 19.0 24.8 Smile 19.7 21.4 19.5 33.5 14.0 Muscular 34.6 26.7 33.8 39.0 33.2

Concept Sliders [8]

Lego 32.2 26.8 26.0 30.3 29.7 Watercolor style 34.3 31.4 29.6 27.5 39.2 Crayon style 32.7 27.8 24.7 29.5 32.5

Customization [17, 29]

Direction 1 29.3 28.1 24.1 27.7 22.4 Direction 16 32.9 32.0 33.4 28.1 29.9 Direction 32 30.7 28.9 29.3 30.5 31.4

Sliderspace [10] (Top nth Variation)

Table 1. We show the percentage change in CLIP score from the original image and the edited image. Higher values indicate stronger attribute change or style transfer. Control effectiveness is largely preserved when transferring from base to different distilled models, with only minor variations across distillation techniques. Importantly, SliderSpace directions for the concept “car” which capture the base’s natural variational structure transfer. This demonstrates that the representational components needed for diversity exist in distilled models.

###### x̂0

visualization for Debugging SDXL Artifacts

For each mechanism, we perform bidirectional transfer experiments: training on base models and applying to distilled models, and vice versa. We study four SDXL distilled families—SDXL-Turbo [32], SDXL-Lightning [20], SDXL-LCM [23], and SDXL-DMD2 [38]—which span diverse distillation procedures. Additional training details and experiments on additional base models (SD 2.1 [28] and DiT-backboned PixArt [4]) are provided in the Appendix D.

SDXL Generated Image

T=5

T=10

T=15

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

“Imageofdogandcatsittingonsofa”

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

|[Figure 52]|
|---|

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

|[Figure 58]|
|---|

|[Figure 59]|
|---|

###### 3.2. Results

Figure 3. xˆ0 visualization reveals generation inconsistencies. When prompted with “Image of dog and cat sitting on sofa,” the SDXL model produces an image with only a dog. However, xˆ0 visualization at T = 10 shows the model initially conceptualizing a cat face (red box) before abandoning this element in the final generation. This demonstrates how diffusion models can discard semantic elements during the denoising process.

Our experiments reveal surprising evidence that contradicts our initial hypothesis. Figure 2 demonstrates seamless transfer of control mechanisms across model variants. For example, a ”comical big eyes” slider trained on SDXL effectively controls SDXL-Turbo’s generations, despite Turbo requiring only 1-4 steps compared to SDXL’s 20-100 steps.

what is missing to why existing diversity mechanisms fail to activate. If the required variational directions exist but are not used, the problem must lie in the generation dynamics rather than the representational capacity.

Table 1 quantifies this compatibility through CLIP scores [12]. Transfer effectiveness remains consistently high across all tested combinations, with positive score changes indicating successful attribute manipulation. While absolute scores vary due to different guidance scales across models, the consistent positive changes confirm that concept representations are preserved during distillation. We show more experiments for the Distillation→Base evidence in Appendix B.

#### 4. xˆ0: Visualizing Intermediate Latents

Having established that distilled models possess the representational components needed for diversity, we must understand why these components fail to activate during generation. Our investigation shifts from what distilled models learn to how they generate. To analyze the generation dynamics, we employ xˆ0 trajectory visualization as a diagnostic tool.

Most tellingly, SliderSpace [10] analysis reveals that variation directions learned on base models transfer perfectly to distilled variants without retraining (Tab 1 and Fig. 2). This indicates that the factors needed for diverse generation are present and accessible in distilled models. However, during free generation with different random seeds, distilled models rarely activate these factors.

In the standard diffusion formulation [14, 33], x0 represents the clean image that is progressively corrupted with noise over T timesteps. During reverse generation, at each timestep t, the model predicts noise ϵθ(xt,t) to compute the next denoising step:

These results directly contradict our initial hypothesis that distillation destroys concept representations. The concepts remain encoded and controllable in distilled models; what fails is their spontaneous activation during fewstep sampling. This finding redirects our investigation from

Let x0 be an initial image and xT be pure Gaussian noise. The forward diffusion process gradually adds noise

according to a variance schedule βt, with corresponding noise level parameters αt = 1−βt and cumulative parameters α¯t = ts=1 αs. The generative process aims to reverse this diffusion, starting from xT and progressively denoising to reconstruct x0. At timestep t, the model predicts noise ϵθ(xt,t) to compute the next step:

√1 − αtϵθ(xt,t) √αt

xt −

(1)

xt−1 =

Using this predicted noise ϵθ, we can estimate what the model believes the final clean image will be at any intermediate timestep t, by extrapolating the same noise prediction all the way to x0:

√1 − α¯tϵθ(xt,t) √α¯t

xt −

(2)

xˆ0|t =

Visualizing these xˆ0|t trajectories across timesteps reveals when structural decisions are made and how they evolve during generation [36]. This technique allows us to peer into the model’s ”thought process” and understand the temporal dynamics of image formation.

###### 4.1. xˆ0 for Investigating Generation Artifacts

Before applying this technique to our diversity investigation, we demonstrate its utility for understanding generation artifacts. In Figure 3, when prompted with ”Image of dog and cat sitting on sofa,” the SDXL model produces a final image containing only a dog’s face. However, xˆ0|t visualization at early timesteps (T = 10) reveals that the model initially conceptualized a cat face (highlighted in red box) before abandoning this element in later steps. This insight exposes how diffusion models can ”change their mind” during generation, sometimes discarding semantic elements present in the prompt.

###### 4.2. xˆ0 for Investigating Sample Diversity

We now apply xˆ0 visualization to compare base and distilled models under identical conditions: same seed and prompt (”image of a dog”). Standard visualizations of intermediate latents xt (Fig 4.a) show only subtle differences across base and distilled models. In stark contrast, the xˆ0 trajectories (Fig 4.b) expose a dramatic pattern that could potentially explain the diversity collapse.

Distilled models commit to their final image structure almost immediately after the first timestep. Their xˆ0|t predictions quickly converge to the final output. Base models, conversely, distribute structural decision-making across many timesteps, with their xˆ0|t predictions gradually refining and evolving toward the final image.

Figure 5 quantifies this phenomenon by measuring DreamSim distances [5] between intermediate xˆ0|t predictions and final outputs across COCO-10k [21] prompts. The data reveals that distilled models achieve a large fraction of

their final structure after a single timestep, while base models require approximately 30% of their total inference steps to reach comparable structural definition. We provide more qualitative evidence in Appendix F.

Our discovery suggests a testable hypothesis: the first timestep is the primary culprit behind diversity loss in distilled models. We provide theoretical analysis (detailed in Appendix A) showing why the first timestep concentrates the diversity bottleneck. The analysis demonstrates how timestep compression in distillation amplifies decisionmaking pressure early in the process, with the amplification factor being largest at initial timesteps.

#### 5. Diversity Distillation

Our xˆ0 analysis identified a clear hypothesis: the first timestep is the primary bottleneck causing diversity collapse in distilled models. To test this causally, we design experiments that directly manipulate the first timestep and measure the impact on sample diversity. If our hypothesis is valid, targeted interventions at this critical step should restore diversity without affecting efficiency.

We develop two approaches to test this hypothesis. First, we replace the first timestep of distilled model inference with the corresponding step from the base model. Second, we examine whether simply skipping the problematic first timestep can improve diversity. Both approaches target the identified bottleneck while preserving the computational advantages of distilled models.

Algorithm 1 implements our hybrid inference strategy. The approach uses the base model for the critical first timestep(s) to establish diverse structural foundations, then transitions to the distilled model for efficient completion. This design directly tests whether the first timestep controls diversity while maintaining computational efficiency.

Algorithm 1 Hybrid Inference for Diversity Distillation Require: Base model fbase, distilled model fdistil, total

timesteps T, transition point k Ensure: Generated image x0

- 1: Initialize xT ∼ N(0,I)
- 2: for t = T,T − 1,...,1 do
- 3: if t > T − k then ▷ Critical timesteps for diversity
- 4: xt−1 ← fbase(xt,t,prompt)
- 5: else ▷ Efficient refinement timesteps
- 6: xt−1 ← fdistil(xt,t,prompt)
- 7: end if
- 8: end for
- 9: return x0

###### 5.1. Experimental Results

We evaluate our approach across two complementary diversity metrics that capture different aspects of model per-

(b) x̂0

(a) Standard Denoising Visualization Visualization

0% 30% 60% 100%

0% 30% 60% 100%

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

SDXL-Base

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

DMDLCM

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

LightningTurbo

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Figure 4. Comparison of standard diffusion visualization vs. xˆ0 visualization. (a) Standard visualization of intermediate latents shows subtle differences between base and distilled models. (b) xˆ0 visualization reveals dramatic differences in how models predict the final output. Distilled models commit to final image structure in the first timestep, while base models gradually refine structure across multiple steps, explaining the observed mode collapse in distilled models.

[Figure 100]

formance. Distributional diversity measures how well generated samples match the real training data distribution across varied prompts, assessed through FID [13] (lower is better). Sample diversity measures variation among outputs generated from the same prompt with different random seeds, quantified by average pairwise DreamSim distance [5] (higher is better).

xSimilaritywithFinalImage0

Distributional Diversity. Table 3 demonstrates that our hybrid approach achieves superior distributional diversity compared to both base and distilled models. Measuring FID against the COCO-30k dataset, our method achieves better scores than the base model while maintaining the computational efficiency of distilled inference. This confirms that our intervention does not compromise the model’s ability to handle diverse prompts. We hypothesize the improvement in FID compared to the base model is due to the usage of the distilled model in final timesteps. Prior work suggests that varied guidance levels in final timesteps can affect FID

Diffusion Timesteps

- Figure 5. Measuring the dreamsim distance between intermediate

xˆ0 visualization and final generated image reveals that distilled models establish structural image composition within the initial diffusion step, whereas base models require approximately 30% of steps to achieve comparable structural definition.

Base Model Distilled Model Diversity Distillation (Ours)

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

“imageofacar” “Imageofastronaut,

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

mutedcolors,8k”

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

- Figure 6. Visual comparison of generation diversity. Each row shows three different generations (different random seeds) for the same prompt using: (left) base model, (middle) distilled model, and (right) our diversity distillation approach. Note how the distilled model produces similar car types, poses, and astronaut orientations across seeds, while our approach restores diversity in colors, contexts, and compositions comparable to the base model while maintaining similar inference speed.

Method Steps FID(↓) IS(↑) CLIP(↑) Time (s)(↓) Base 50 12.74 24.74 31.83 9.22

Prompt Base Distilled Hybrid (Ours) Sunset beach 0.396 0.271 0.373 Cute puppy 0.233 0.199 0.265 Futuristic city 0.237 0.198 0.283 Person 0.484 0.347 0.461 Van Gogh art 0.337 0.305 0.366 Average 0.337 0.264 0.350

Distilled 4 15.52 27.20 31.69 0.64 Hybrid (Ours) 4 10.79 26.13 32.12 0.64

Skip First Timestep 3 10.12 24.69 31.71 0.53

Table 3. Measuring distributional diversity using FID shows our approach achieves comparable or better diversity than SDXL-Base while maintaining SDXL-DMD [38] efficiency. While our focus is sample diversity (Table 2), this confirms our intervention preserves dataset coverage. Skipping first timestep achieves superior FID but lower generative quality (CLIP [12], IS [31]).

Table 2. Sample diversity measured by average pairwise DreamSim distance (higher is more diverse). Our hybrid approach restores the lost diversity in distillation with a simple intervention.

metrics [18].

Sample Diversity. Table 2 reveals that our approach dramatically restores sample-level diversity. For each prompt, we generate 100 images with different random seeds and calculate average pairwise DreamSim distances. The hybrid approach restores the diversity lost during distillation.

Figure 6 provides visual confirmation of these quantitative results. The distilled model clearly exhibits reduced structural variety across random seeds, producing similar compositions and layouts. Our hybrid approach successfully restores this diversity, generating varied structural arrangements comparable to the base model while maintaining fast inference speeds.

Causal Evidence. To causally close the loop, we also provide an experiment in the Appendix E where keeping the first timestep with the distilled model and replacing the later timesteps with the base model does not help with improvement in sample diversity. This demonstrates that the first timestep is indeed the critical bottleneck, validating our mechanistic understanding derived from xˆ0 visualization.

###### 5.2. Hyperparameter Analysis and Efficiency

Figure 7 analyzes the key parameters of our approach. Most importantly, using the base model for just the first timestep (k=1) provides substantial diversity gains with minimal computational overhead (Fig 7.b), confirming our hypothesis about first-step criticality. The guidance scale analysis shows optimal performance around zero guidance from the base model, suggesting that natural diversity is preserved

[Figure 137]

(a) (b) (c)

- Figure 7. (a) Impact of guidance scale from the base model on diversity shows optimal performance around 0 guidance. (b) Effect of the number of distilled model steps (k) being replaced by base model inference. Running distilled model from first timestep (k = 1) provides diversity gains with minimal computational overhead. (c) Comparing the total timesteps of base model when replacing the first timestep of distilled model shows that replacing 1-1 timesteps of distilled with base is most ideal.

without additional steering(Fig 7.a). Finally, setting the base model’s noise schedule similar to distilled (n=4) has optimal results (Fig 7.c). We find that using exact inference conditions for distilled models where the model weights are swapped with base for the first timestep is ideal.

For scenarios where loading both models simultaneously is not feasible, we explore skipping the first timestep entirely. Table 3 shows this approach provides significant diversity improvements, though our hybrid method achieves superior quality as measured by CLIP and Inception scores. We provide qualitative examples in the Appendix G.

Our hybrid method demonstrates that the conventional efficiency-diversity tradeoff in distilled models can be resolved through targeted intervention. By identifying and addressing the specific timestep responsible for diversity collapse, we restore sample variety without sacrificing computational advantages. Our theoretical support in Appendix A provides a principled explanation for our empirical observations and suggests that distributing first-timestep decisions across multiple steps during training could be a promising direction for future distillation methods.

#### 6. Limitations

While our approach significantly improves diversity without substantial computational overhead, some limitations remain. First, our method requires loading both base and distilled models in memory, increasing resource requirements compared to traditional inference. Future distillation work could explore our insights to design diversitypreserving mechanisms directly into a distilled model.

Second, our analysis focuses primarily on visual diversity metrics. Further investigation is needed to understand the impact on semantic diversity: the range of concepts and compositions a model can generate. Developing more diversity metrics that capture both visual and semantic varia-

tions can provide deeper insights into distillation.

Finally, our approach treats all prompts uniformly, but different concepts may benefit from different base/distilled step allocations. Adaptive inference strategies that dynamically adjust the transition point based on prompts could further optimize the quality-efficiency trade-off.

#### 7. Conclusion

This work addresses a fundamental limitation of distilled diffusion models: the trade-off between computational efficiency and sample diversity. Our contributions are threefold: (1) We demonstrate that distilled models retain all the variational directions needed for diversity, contradicting the hypothesis that distillation damages concept representations—the required diversity mechanisms exist but fail to activate during generation; (2) We identify the root cause using xˆ0 trajectory visualization, revealing that distilled models concentrate structural decision-making in the first timestep while base models distribute decisions across many steps, and provide theoretical support showing why the first timestep creates a diversity bottleneck; and (3) We validate this theory by developing diversity distillation, demonstrating that targeted intervention at a single timestep restores full base model sample diversity as measured by DreamSim scores with strong empirical validation.

Our experimental results challenge the conventional diversity-efficiency trade-off. Diversity distillation restores the diversity of the original base model while maintaining the computational efficiency of distilled inference (0.64s vs. 9.22s per image). By providing both mechanistic understanding and theoretical grounding for why diversity collapse occurs, our approach eliminates this traditional trade-off without additional training or model modifications, opening new possibilities for deploying efficient yet diverse generative models in creative applications.

#### Acknowledgment

RG and DB are supported by Open Philanthropy and NSF grant #2403304.

#### Code

Our methods are available as open-source code. Source code, and data sets for reproducing our results can be found at distillation.baulab.info and at our GitHub repo github.com/rohitgandikota/distillation

#### References

- [1] Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel CohenOr, and Dani Lischinski. Break-a-scene: Extracting multiple concepts from a single image. In SIGGRAPH Asia 2023 Conference Papers, pages 1–12, 2023. 3
- [2] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22560–22570, 2023. 2
- [3] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM transactions on Graphics (TOG), 42(4):1–10, 2023. 2
- [4] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 2, 4
- [5] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344,

2023. 5, 6

- [6] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 2
- [7] Rohit Gandikota, Joanna Materzynska, Jaden FiottoKaufman, and David Bau. Erasing concepts from diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2426–2436, 2023. 3
- [8] Rohit Gandikota, Joanna Materzy´nska, Tingrui Zhou, Antonio Torralba, and David Bau. Concept sliders: Lora adaptors for precise control in diffusion models. In European Conference on Computer Vision, pages 172–188. Springer, 2024. 2, 3, 4
- [9] Rohit Gandikota, Hadas Orgad, Yonatan Belinkov, Joanna Materzy´nska, and David Bau. Unified concept editing in diffusion models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5111–5120, 2024. 3

- [10] Rohit Gandikota, Zongze Wu, Richard Zhang, David Bau, Eli Shechtman, and Nick Kolkin. Sliderspace: Decomposing the visual capabilities of diffusion models. arXiv preprint arXiv:2502.01639, 2025. 2, 3, 4
- [11] Jiayi Guo, Xingqian Xu, Yifan Pu, Zanlin Ni, Chaofei Wang, Manushree Vasu, Shiji Song, Gao Huang, and Humphrey Shi. Smooth diffusion: Crafting smooth latent spaces in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7548– 7558, 2024. 2
- [12] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718,

2021. 4, 7

- [13] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 2, 6
- [14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2, 4
- [15] Rohit Jena, Ali Taghibakhshi, Sahil Jain, Gerald Shen, Nima Tajbakhsh, and Arash Vahdat. Elucidating optimal rewarddiversity tradeoffs in text-to-image diffusion models. arXiv preprint arXiv:2409.06493, 2024. 2
- [16] Nupur Kumari, Bingliang Zhang, Sheng-Yu Wang, Eli Shechtman, Richard Zhang, and Jun-Yan Zhu. Ablating concepts in text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22691–22702, 2023. 3
- [17] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1931–1941, 2023. 2, 3, 4
- [18] Tuomas Kynk¨a¨anniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. Advances in Neural Information Processing Systems, 37:122458–122483, 2024. 7
- [19] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 2
- [20] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxllightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024. 2, 4
- [21] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pages 740–755. Springer, 2014. 2, 5
- [22] Shilin Lu, Zilan Wang, Leyang Li, Yanzhu Liu, and Adams Wai-Kin Kong. Mace: Mass concept erasure in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6430– 6440, 2024. 3

- [23] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 2, 4
- [24] ManBearScientist. Shop for the right seed. https: / / www . reddit . com / r / StableDiffusion / comments / x6kjdh / steps _ for _ getting _ better _ images/, 2022. [Accessed 16-09-2025]. 1, 2
- [25] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI conference on artificial intelligence, pages 4296–4304, 2024. 2
- [26] Minh Pham, Kelly O Marshall, Chinmay Hegde, and Niv Cohen. Robust concept erasure using task vectors. arXiv preprint arXiv:2404.03631, 2024. 3
- [27] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2
- [28] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 4
- [29] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 2, 3, 4
- [30] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022. 2
- [31] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016. 7
- [32] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–103. Springer,

2024. 2, 4

- [33] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015. 2, 4
- [34] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2
- [35] Yael Vinker, Andrey Voynov, Daniel Cohen-Or, and Ariel Shamir. Concept decomposition for visual exploration and inspiration. ACM Transactions on Graphics (TOG), 42(6): 1–13, 2023. 3

- [36] Binxu Wang and John J Vastola. Diffusion models generate images like painters: an analytical theory of outline first, details later. arXiv preprint arXiv:2303.02490, 2023. 2, 5
- [37] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2

- [38] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. arXiv preprint arXiv:2405.14867, 2024. 2, 4, 7
- [39] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024. 2
- [40] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 2

## Distilling Diversity and Control in Diffusion Models Supplementary Material

#### A. Theoretical Support

We provide a theoretical justification for the empirical observation that distilled diffusion models lose most of their sample diversity at the first denoising timestep. Our argument relies on the standard DDPM forward process, the amplification structure of xˆ0|t, and the statistical effect of mean-squared-error (MSE) based distillation.

###### A.1. Preliminaries

Let x0 ∈ Rd be a clean data sample, and consider the forward diffusion process

q(xt | x0) = N xt;√α¯t x0, (1 − α¯t)I , (A.1)

where α¯t = ts=1 αs and αs ∈ (0,1).

The reverse model is often parameterized by predicting the noise εθ(xt,t). From the forward relation, one can form an estimator of x0:

√1 − α¯t εθ(xt,t) √α¯t

xt −

. (A.2)

xˆ0|t =

Equation (A.2) will be central to our analysis, as it determines how prediction errors or randomness in εθ propagate into variability in xˆ0|t.

###### A.2. Sensitivity

Lemma 1 (Sensitivity). Let ∆ε denote a perturbation in the noise prediction at timestep t. Then the induced change in xˆ0|t is

∆ˆx0|t = − 1−α¯

α¯t ∆ε. (A.3) Proof. Differentiate (A.2) with respect to ε:

t

∂xˆ0|t ∂ε

√1 − α¯t √α¯t

= −

I.

Multiplying by ∆ε yields the result.

| |
|---|

The amplification factor

At =

1 − α¯t α¯t

(A.4)

quantifies how strongly prediction variability at timestep t is magnified in the clean-sample estimate. Since α¯t ≪ 1 at early timesteps, At is very large.

###### A.3. Distillation and Conditional Variance

Distilled diffusion models are typically trained to minimize an MSE-style loss between student and teacher outputs. A standard fact from estimation theory is that the MSE minimizer of a random target is the conditional mean:

s∗(x) = E[Y | X = x]. (A.5)

Thus, when the teacher output Y has conditional variance Var(Y | X), the student collapses this variance and learns to reconstruct the mean.

By the law of total variance,

Var(Y ) = Var(E[Y | X]) + E[Var(Y | X)]. (A.6)

MSE distillation removes the second term, reducing sample diversity by precisely E[Var(Y | X)].

∆Var = Var(Y )−Var(s∗(x)) = E[Var(Y | X)]. (A.7)

###### A.4. Amplification of Diversity Loss at Early Timesteps

Let the teacher produce a stochastic noise prediction εT(xt,t;ξ), where ξ captures randomness in sampling. Then, from Lemma 1,

1 − α¯t α¯t

Var(εT(xt,t) | xt). (A.8) Taking expectation over xt,

Var(ˆx0|t | xt) =

1 − α¯t α¯t

Ex

Ex

Var(ˆx0|t | xt) =

Var(εT(xt,t) | xt) .

t

t

(A.9) Proposition 1 (Amplified Diversity Loss). For an MSEtrained student, the reduction in total variance of xˆ0|t due to distillation satisfies

1 − α¯t α¯t

Var(εT(xt,t) | xt) . (A.10)

Ex

∆Var ≥

t

Proof. Direct application of the law of total variance to the random variable xˆ0|t, substituting from (A.9).

| |
|---|

###### A.5. Main Result

Theorem 1 (First-Timestep Dominance). Let α¯t denote the cumulative product of noise schedule parameters. Then for small α¯t (early timesteps), the amplification factor (1 − α¯t)/α¯t is maximal. Consequently, the diversity loss ∆Var induced by MSE-based distillation is largest at the earliest timesteps. In particular, the first denoising step dominates the reduction of sample diversity in distilled diffusion models.

Proof. Since α¯t is monotonically increasing in t with α¯0 ≈ 0, the fraction (1−α¯t)/α¯t is strictly decreasing in t. Hence the bound on ∆Var is largest at the earliest timestep, completing the proof.

| |
|---|

This proof applies in cases in which the teacher uses a stochastic ϵT(xt,t;ξ) to create variance in its output. This is the case for the ADD distillation method [32] that is used to train SDXL-Turbo, which uses stochastically sampled teacher for reconstruction loss. The variance trade-off in such cases is also analyzed in previous works [11].

We have also tested our methods on models that do not use a stochastic teacher, such as DMD [39], which uses the Distribution Matching distillation method which uses a deterministic teacher, in which the teacher’s Var(Yt | xt,t) = 0. Despite this difference, we still measure a large drop in sample diversity concentrated at the first timestep. We hypothesize that in these cases, the loss in diversity is due to sparse sampling of the teacher relative to the large diversity of text prompts, which allows the student to collapse to sparsely-sampled modes in a similar way as seen in the stochastic teachers, despite the deterministic teachers’ theoretical diversity: for example in DMD only 100k textconditioned teacher samples are used. The loss in diversity due to sparse sampling of a teacher would be amplified in the early timesteps due to Eq (A.4).

#### B. Control Distillation: Reverse Transfer

In the main paper, we demonstrated that control mechanisms trained on base models can be seamlessly transferred to distilled models. Here, we present additional results for the reverse direction: transferring control mechanisms trained on distilled models to base models. This bidirectional transfer capability further validates our hypothesis that concept representations are preserved during the distillation process.

We note that while most control mechanisms transferred effectively, we encountered difficulties training LoRA adaptations on LCM due to its specialized architecture and training procedure. These challenges highlight potential avenues for future research in developing more universally transferable control mechanisms.

#### C. Skip Step Approach

In the main paper, we introduced a resource-efficient alternative to our hybrid approach: skipping the first timestep altogether in distilled model inference. We provide additional qualitative comparisons between this approach and our hybrid method in Figure C.1.

The skip-first-step approach provides a reasonable compromise when resource constraints prevent loading both models simultaneously. However, our quantitative analysis

in the main paper and these qualitative examples demonstrate that the hybrid approach consistently achieves superior results in terms of both diversity and quality.

#### D. Generalization Across Model Backbones

To assess the generality of our findings, we extend our analysis to different diffusion model architectures beyond SDXL. We evaluate two additional model pairs: PixArtAlpha (base) with PixArt-Delta (distilled), and SD 2.1 (base) with SD-Turbo (distilled).

Table D.1 shows that the diversity collapse phenomenon and the effectiveness of our solution generalize across different model architectures. PixArt-Delta exhibits similar sample diversity reduction compared to PixArt-Alpha, with our hybrid approach restoring diversity while maintaining efficiency. Similarly, SD-Turbo shows reduced sample diversity compared to SD 2.1, which our method successfully addresses. We show qualitative results in Figure D.1

Model Architecture Sample Diversity Time (s) Our Method PixArt-Alpha DiT 0.342 4.1 PixArt-Delta DiT 0.198 0.9 0.339 SD 2.1 UNet 0.298 3.2 SD-Turbo UNet 0.171 0.7 0.294

Table D.1. Sample diversity (DreamSim distance) across different model architectures for the prompt “image of a car” across 100 samples. Our hybrid approach consistently restores diversity regardless of the underlying architecture (DiT vs UNet) or distillation method. Please refer to Fig D.1 for qualitative samples

The consistent pattern across DiT-based (PixArt) and UNet-based (SD 2.1/Turbo) architectures demonstrates that the first-timestep diversity bottleneck is a fundamental characteristic of diffusion distillation, not specific to particular model designs.

#### E. Causal Validation: Testing Later Timesteps

To strengthen our causal claim that the first timestep is the critical bottleneck, we conduct a complementary experiment: replacing the final timesteps of distilled models with base model steps while keeping the first timestep from the distilled model.

Table E.1 shows that replacing the final timesteps does not yield diversity improvements compared to our firsttimestep intervention. This result provides strong causal evidence that the diversity bottleneck is concentrated at the beginning of the generation process, not distributed throughout the timesteps. The minimal improvement from modifying later steps confirms our xˆ0 analysis: once the structural decisions are made in the first timestep, later steps primarily refine details rather than introduce fundamental variations. We show qualitative examples in Figure E.1

Turbo→Base DMD→Base Lightning→Base

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

SDXL-BaseDistilledModel

[Figure 142]

[Figure 143]

[Figure 144]

Golden Dragon Mythical Warrior Chibi Style

- Figure B.1. Reverse Control Transfer: Control mechanisms (Custom Diffusion [17] and Concept Sliders [8]) trained on distilled models can be effectively transferred to base models without retraining. This bidirectional transferability confirms that concept representations are preserved during diffusion distillation. Note: LCM LoRA transfers were excluded due to training difficulties with the LCM architecture.

tion during the denoising process. We present additional visualizations in Figures F.1, F.2 that further illuminate the differences between base and distilled models.

Method Sample Diversity Time (s)

SDXL-Base 0.357 9.22 SDXL-DMD2 0.264 0.64

These visualizations reinforce our key finding: distilled models compress the diversity-generating behavior distributed across early timesteps in base models into a single initial step, explaining the observed mode collapse. This insight directly informed our hybrid inference approach, which strategically leverages the diversity-generating capabilities of base models in critical early steps.

Replace First Timestep (Ours) 0.350 0.64 Replace Final Timesteps 0.251 6.61

Table E.1. Causal validation experiment. Replacing final timesteps with base model steps provides minimal diversity improvement (DreamSim distance) compared to our first-timestep intervention, confirming that the first timestep is the critical bottleneck. The analysis is done for the prompt “image of a car” across 100 samples.

#### G. Mode Collapse and Diversity

The main paper introduced our finding that distilled diffusion models suffer from reduced sample diversity (mode collapse) compared to their base counterparts. We provide additional qualitative examples in Figure G.1-G.4 that visually demonstrate this phenomenon across various prompts

#### F. Extended xˆ0 visualization Analysis

The main paper introduced xˆ0 visualization technique for analyzing how diffusion models develop structural informa-

Diversity Distillation (Ours) Skipping First Timestep (Ablation)

[Figure 145]

[Figure 146]

“Imageofatruck”“Imageofadog”

[Figure 147]

[Figure 148]

[Figure 149]

- Figure C.1. Qualitative comparison between (left) our hybrid approach, (right) skip-first-step approach. The skip-first-step approach improves diversity over the standard distilled model but exhibits reduced quality compared to our hybrid method, particularly in fine details and coherence.

and model variants.

These examples highlight the significant diversity loss in distilled models. While the distilled models produce high-quality images, they often converge to similar structural compositions regardless of random seed initialization. Our diversity distillation approach effectively addresses this limitation, restoring the variety of outputs comparable to the base model while maintaining computational efficiency.

Base Model Distilled Model Hybrid Inference

[Figure 150]

[Figure 151]

[Figure 152]

Pixart-α(DiT)SD2.1(UNet)

[Figure 153]

[Figure 154]

[Figure 155]

- Figure D.1. Generalization across model architectures. Sample diversity comparison for the prompt ”image of a car” across different diffusion model architectures. Top row: PixArt-α (DiT-based base model) shows diverse car types, colors, and contexts, while PixArt-δ (distilled) produces similar red sports cars with repetitive compositions. Bottom row: SD 2.1 (UNet-based base model) generates varied car styles and settings, while SD-Turbo (distilled) exhibits reduced diversity with similar silver/white cars in repetitive urban contexts. Our hybrid inference approach restores diversity in both architectures, demonstrating that the first-timestep bottleneck is architecture-agnostic.

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

SDXL-Base SDXL-DMD Base (T=0) → DMD (T>0) DMD (T=0) → Base (T>0)

- Figure E.1. Causal validation of first-timestep importance. Visual comparison for the prompt ”image of a car” showing: (left) SDXLBase with diverse car types, colors, and contexts; (middle-left) SDXL-DMD with reduced diversity showing similar red sports cars; (middle-right) our hybrid approach using base model for first timestep (T=0) then DMD for remaining steps, successfully restoring diversity; (right) control experiment using DMD for first timestep (T=0) then base model for remaining steps, showing minimal diversity improvement. This demonstrates that the first timestep, not later steps, controls sample diversity.

x0 Visualization

Base Model Distilled Model

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

- Figure F.1. Extended xˆ0 visualization comparison between SDXL-Base and SDXL-DMD for the prompt. The visualization reveals that DMD commits to final structural composition within the first timestep, while Base gradually develops structure across multiple steps. This pattern is consistent across different content types and prompts.

x0 Visualization

Base Model Distilled Model

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

- Figure F.2. Extended xˆ0 visualization comparison between SDXL-Base and SDXL-DMD for the prompt. The visualization reveals that DMD commits to final structural composition within the first timestep, while Base gradually develops structure across multiple steps. This pattern is consistent across different content types and prompts

[Figure 176]

[Figure 177]

[Figure 178]

###### Figure G.1. Comparison of generation diversity across different models for the prompt ”image of a toy.” Each image shows different seeds for the same model. Note the structural similarity in distilled model outputs compared to the greater variation in base model and our hybrid approach.

[Figure 179]

[Figure 180]

[Figure 181]

###### Figure G.2. Comparison of generation diversity for ”image of a flower” Distilled models (middle column) produce structurally similar outputs across different seeds, while our approach (right column) restores diversity comparable to the base model (left column) while maintaining the speed advantage of distilled models.

[Figure 182]

[Figure 183]

[Figure 184]

###### Figure G.3. Additional diversity comparison for ”city street”Distilled models (middle column) produce structurally similar outputs across different seeds, while our approach (right column) restores diversity comparable to the base model (left column) while maintaining the speed advantage of distilled models.

[Figure 185]

[Figure 186]

[Figure 187]

###### Figure G.4. Diversity comparison for abstract prompt: ”picture of a monster” Distilled models (middle column) produce structurally similar outputs across different seeds, while our approach (right column) restores diversity comparable to the base model (left column) while maintaining the speed advantage of distilled models.

