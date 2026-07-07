# arXiv:2511.22625v2[cs.CV]1Dec2025

## ReasonEdit: Towards Reasoning-Enhanced Image Editing Models

[Figure 1]

[Figure 2]

[Figure 3]

Step1X-Image Team StepFun https://github.com/stepfun-ai/Step1X-Edit

#### Abstract

[Figure 4]

[Figure 5]

[Figure 6]

Recent advances in image editing models have shown remarkable progress. A common architectural design couples a multimodal large language model (MLLM) encoder with a diffusion decoder, as seen in systems such as Step1X-Edit and Qwen-Image-Edit, where the MLLM encodes both the reference image and the instruction but remains frozen during training. In this work, we demonstrate that unlocking the reasoning capabilities of MLLM can further push the boundaries of editing models. Specifically, we explore two reasoning mechanisms, thinking and reflection, which enhance instruction understanding and editing accuracy. Based on that, our proposed framework enables image editing in a thinking–editing–reflection loop: the thinking mechanism leverages the world knowledge of MLLM to interpret abstract instructions, while the reflection reviews editing results, automatically corrects unintended manipulations, and identifies the stopping round. Extensive experiments demonstrate that our reasoning approach achieves significant performance gains, with improvements of ImgEdit (+4.3%), GEdit (+4.7%), and Kris (+8.2%) when initializing our DiT from the Step1XEdit (ReasonEdit-S), and also outperforms previous opensource methods on both GEdit and Kris when integrated with Qwen-Image-Edit (ReasonEdit-Q).

Figure 1. ReasonEdit achieves progressive performance gains, employing Thinking to interpret abstract instructions and Reflection to audit and correct the initial results.

models exhibit limited visual reasoning capabilities, which restricts their ability to handle complex or abstract instructions. More importantly, such limitations prevent them from benefiting fully from test-time scaling, a paradigm that has driven significant improvements in language models.

Turning to the visual reasoning domain, recent advances have explored reasoning-enhanced visual generation through unified understanding and generation [11, 28, 46], reflection-based refinement [26, 51], and chain-of-thought modeling [20, 48, 57]. These studies highlight the potential of reasoning for controllable and efficient generation. For instance, BAGEL [11] introduces a thinking mode that leverages the world knowledge of MLLMs to interpret abstract instructions in image generation and editing, while OmniGen2 [51] integrates reflection capabilities of MLLMs into generation. Despite these advances, most existing efforts remain centered on image generation [6, 23], leaving the application of reasoning to image editing largely underexplored. A key underlying challenge lies in the substantial hallucinations of MLLMs during paired image understanding [12, 19], particularly in capturing the differences between reference and edited results [15, 47] and in generating appropriate refined instructions for subsequent editing.

#### 1. Introduction

Image editing with diffusion models has witnessed rapid progress, moving from early mask-based approaches such as BrushNet [22] and PowerPaint [60], to instruction-driven systems like InstructPix2Pix [7] and OmniGen [53], and more recently to multimodal frameworks that integrate an MLLM encoder with a diffusion decoder, like Step1XEdit [33] and Qwen-Image-Edit [50]. These advances have substantially improved controllability and usability, enabling more diverse and flexible image editing. However, state-of-the-art instruction-based methods still face challenges in generalizing instructions, as most models keep MLLM encoders frozen during training. As a result, current

To this end, we propose ReasonEdit, a fundamental editing model with two reasoning capbilities: thinking and reflection. The former primarily transforms ambiguous, colloquial, and informal editing instructions into clear, standardized, and actionable directives by constructing Thinking Pairs, which are structured as abstract-to-concrete instruction pairs. The latter is designed to perform iterative self-correction, refinement, and termination during the editing process by restructuring the paired image understanding as multiple cascaded single-image understanding tasks. We achieve this by constructing Reflection Triples that form an

Instruction Ref Image w/ thinking w/o thinking

Instruction Ref Image w/ thinking w/o thinking

Replace the arid sandy ground with lush green grass while maintaining the hilly terrain and blue sky.

Make the leaf appear wilted and discolored, with drooping edges and pale green veins.

###### [Thinking]

Modify it to appear as if the desertification control measures are effective.

[Figure 7]

[Figure 8]

[Figure 9]

Symptoms of potassium deficiency in this leaf.

[Figure 10]

[Figure 11]

[Figure 12]

Instruction Ref Image Thinking

Reflection round2

Reflection round1

Set the smartphone vertically at the center against a completely black background. Add a halo effect around the phone's screen using the blue light. Place the title "BEYOND THE EDGE" in an extremely thin white sans-serif font at the top center of the image.

<#Think>The second image has a white background instead of a black one, and there is no halo effect created by the screen's blue light. Additionally, the text position does not align with the instruction, which requires it to be placed at the top center of the image. <#Reflection>Use a black background, adjust the smartphone's position so that its screen emits blue light, and create a halo effect around it. Finally, place the text "BEYOND THE EDGE" at the top center of the image.

Use the image object as the main visual. Place the smartphone vertically at the center of a completely black background. The blue light emitted from the screen should be the only light source, creating a halo diffusion effect around it to enhance the visual language of "Technology as Light." Do not introduce any other objects. Only place the title "BEYOND THE EDGE" in extremely fine white sans-serif font at the top center of the image. The overall composition should be centered and symmetrical, with a minimalist, cool, and futuristic tension, suitable for high-end tech brand visual displays or new product launch posters.

[Reflection]

[Figure 13]

[Figure 14]

[Figure 15]

<#Success>

- Figure 2. Illustration of the ReasonEdit’s reasoning capabilities. The thinking module illustrates how a model decomposes abstract instructions into clear, actionable commands. The reflection pipeline, conversely, showcases the model’s ability to perform an iterative self-correction loop, refining an intermediate generated image to achieve a more accurate final result.

iterative cycle defined by three core image states: <original image, editing instructions, edited image, reflection instructions, reflection-corrected image, VIEScore [25]>. To train these image editing reasoning capabilities, our architecture integrates a MLLM as the Reasoner and a DiT as the Generator. We employ a multi-stage training strategy: initially, the model is trained independently on image editing and thinking tasks, followed by a joint training phase. This progressive approach simplifies the learning objectives at each stage, leading to smoother convergence and a more effective, gradual acquisition of both editing and reasoning capabilities. Our contributions can be summarized as follows:

such as BrushNet [22], BrushEdit [27], PowerPaint [60], and FLUX.1-Fill-dev [3], typically employ an edit-area mask together with textual instructions to achieve localized and high-quality edits. Beyond mask-based control, recent works have further explored enhancing editing controllability by incorporating multiple visual conditions. For instance, OminiControl [45], ACE [16], and ACE++ [36] unify diverse conditional signals such as depth maps and keypoints within a single model, thereby enabling more flexible and versatile editing capabilities.

While visual conditions offer precise control, they raise the usage threshold. In contrast, instruction-based models enable editing through natural language, but often struggle to align semantic understanding with fine-grained manipulation. Pioneering efforts such as Instruct-Pix2Pix [7], MagicBrush [56], UltraEdit [59], AnyEdit [55], and OmniGen [53] construct large-scale instruction–image pairs to support purely instruction-driven editing, yet still face challenges in fidelity and quality. Recent approaches address this challenge by leveraging priors from advanced text-toimage models [2, 4, 8], as in ICEdit [58], Hidream-E1 [17], and FLUX.1-Kontext-dev [5]. Another line of work integrates multimodal large language model encoders with diffusion decoders, such as Qwen2VL-Flux [34], MetaQueries [39], BLIP3-o [9], UniWorld-v1 [31], Step1XEdit [33], and Qwen-Image-Edit [50].

- • A reasoning-enhanced editing model that natively supports a thinking–editing–reflection workflow. Thinking mode allows parsing original instructions leveraging the world knowledge of MLLMs, enabling the model to tackle more complex editing tasks. Reflection mode enables iterative refinement by reviewing and correcting the results of previous edits.
- • A comprehensive data construction pipeline consisting of <original image, editing instructions, edited image, reflection instructions, reflection-corrected image, VIEScore>, which supports end-to-end training of the thinking–editing–reflection loop.
- • A flexible training framework that demonstrates consistent performance gains by initializing our DiT from advanced models, such as ReasonEdit-S (based on Step1XEdit) improving ImgEdit (+4.3%), GEdit (+4.7%), and Kris (+8.2%); and ReasonEdit-Q (based on Qwen-ImageEdit) yielding ImgEdit (+2.8%), GEdit (+3.4%), and Kris (+6.1%), while outperforming previous open-source methods on GEdit and Kris.

Although existing models have achieved notable progress in instruction-based editing, their reliance on frozen MLLM encoders limits performance on complex or abstract instructions. Motivated by this, ReasonEdit unlocks the reasoning capability of MLLMs through joint optimization with the diffusion decoder, thereby improving semantic understanding and extending the boundaries of controllable image editing.

#### 2. Related Work

##### 2.1. Image Editing Models

##### 2.2. Reasoning-Enhanced Visual Generation

The test-time scaling paradigm has rapidly extended from language to multimodal domains, giving rise to several reasoning-enhanced visual generation models. ThinkD-

Diffusion models have demonstrated remarkable progress in generative modeling, particularly for producing highfidelity and diverse image editing results. Early approaches,

iff [37] introduces multimodal in-context reasoning into diffusion models via a “think-then-diffuse” inference scheme, while BAGEL [11] enables a thinking mode by jointly training visual understanding and generation tasks. In addition to pre-thinking mechanisms before generation, some works explore reflection strategies to refine outputs, such as OmniGen2 [51] and Reflect-DiT [26]. Others, including ImageCoT [57], MINT [48], and IRG [20], employ multimodal chain-of-thought reasoning to guide generation. Beyond text-to-image, GoT [14] integrates reasoning with diffusion models using large-scale reasoning-chain data for controllable generation and editing. Uni-CoT [41] further decomposes multimodal chain-of-thought learning into macroand micro-level components with auxiliary tasks, enabling efficient training for complex reasoning.

Compared with the above approaches, ReasonEdit focuses more on exploring thinking and reflection mechanisms for editing tasks, enhancing instruction understanding and editing accuracy. While Uni-CoT [41] is a concurrent work, our method adopts different base models, training data composition, and training paradigm.

#### 3. Method

This section introduces the training data construction and the training of our model. We first elaborate on the construction of our edit reasoning data. Following this, we describe the training of the proposed REASONEDIT, in which we present the model design, the multi-stage training strategy. Finally, and a thorough account of the specific training details, ensuring clarity of the our method.

##### 3.1. Data Construction

To facilitate supervised fine-tuning of our reasoning model, we have developed two distinct datasets: thinking and reflection. The former consists of abstract instruction-clear multi-step decomposition instruction pairs, while the latter includes triples that encompass multiple cascaded singleimage understanding tasks.

Thinking Pairs consist of abstract-to-concrete instruction pairs. Each pair links an abstract instruction, which captures a user’s original request in ambiguous, colloquial, or informal language, with its corresponding set of concrete, actionable commands. The concrete counterpart translates the initial user intent into one or more precise, standardized, and executable directives. For instance, the abstract entry “symptoms of potassium deficiency in leaves” is paired with the concrete command “Render the leaves yellow and desiccate the leaf tips.” For more complex requests, this structure facilitates a logical decomposition into a single, cascaded sequence of directives. As an illustration, a multifaceted request like “Make the image more dramatic with a vintage feel” is deconstructed into a single, composite instruction:

“Increase the image contrast. Apply a sepia tone filter. Add a subtle vignette effect”.

To construct the Thinking Pairs dataset, we devised a three-step process combining categorization, annotation, and review, leveraging different advanced Vision-Language Models (VLMs) as annotators. First, we classified a large pool of raw instructions as either already clear or as abstract and complex. Then, in a two-way annotation process, we generated the corresponding abstract instructions for the clear commands and decomposed the complex instructions into clear, actionable sub-directives. Finally, a rigorous review ensured that each pair met our specific abstract-toconcrete requirements. We also supplemented the dataset with a small number of simple instructions that did not require rewriting. This ensures our model can learn to handle both complex requests by decomposing them and simple requests by outputting them directly.

The Thinking Pairs dataset is built from an initial 500k image-instruction pair pool, which we categorize into 112k complex and 388k simple instructions. After annotating the entire set, a rigorous review process selects 150k highquality abstract-to-concrete pairs. Specifically, 62k of these pairs come from simplifying complex instructions, and 88k are created by adding an abstract layer to simple ones. To ensure versatility, we also include 50k pairs of simple, unedited instructions. The final dataset totals 200k carefully curated pairs.

Reflection Triples is constructed from a large collection of existing image-editing pairs, designed to facilitate a model’s multi-step, cascaded reasoning capabilities. Each core triple consists of an <Input Image>, a <Generated Image>, and a <Target Image>. This structure models a chained editing process: the <Generated Image> represents an intermediate output from an initial edit on the <Input Image>, providing the crucial context for a multiround, single-image reflection process through which the model evaluates the generated output and performs subsequent adjustments to produce the refined <Target Image>. In some instances, the generated image and the target image are identical if no further edits are required.

To mitigate hallucination issues inherent in single-pass, dual-image evaluation, we designed a multi-round, singleimage reflection pipeline to enable robust reflection. This process begins by generating a target image description based on the input image and instruction, which acts as a faithful and concise blueprint for the intended outcome. A quantitative evaluation then provides a consistency score and rationale, with metrics designed to assess the presence of conflicts, omissions, and hallucinations. This comprehensive evaluation serves as a strong prior for the final reflection and decision-making phase, where the model assesses the edit’s success using the original image, the generated image, and the original instruction as a basis. The

process yields one of three conclusions: Successful Edit: Indicated by reasoning content and a <#Success> tag, signifying consistency between the generated and target images. Refinable Edit: For edits that are not fully successful but allow for refinement, reasoning content and a <#Reflection> tag are returned, along with a secondary editing instruction based on the generated image. Failed Edit: If an edit fails due to irrecoverable flaws, reasoning content and a <#Failed> tag are provided. Furthermore, the model conducts a final scoring of the generated image, assessing both semantic accuracy and image quality, to determine the optimal round at which the iterative process should terminate.

The Reflection Triples is constructed from an initial pool of 500k image-editing pairs. To diversify the modalities of intermediate images, we generate an additional 500k images using four mainstream editing methods [5, 33, 38, 43]. We then apply our previously described reflection pipeline, utilizing an advanced VLM to automate the process. After a final, rigorous manual screening, the curation yields 180k valid data pairs, with an approximate ratio of 3:1:1 for success, reflection, and failed examples. We then utilize GPT-

- 4.1 [38] to evaluate the VIEScore for each of these 180k valid data pairs.

##### 3.2. Training

With the reasoning-enhanced dataset, we utilize a multistage training strategy to effectively integrate reasoning and image editing capabilities into a single unified model.

###### 3.2.1. Model Design

As shown in Fig. 3, our model integrates an MLLM as the Reasoner and a DiT [40] as a Generator. Specifically, we directly adopt Step1X-Edit [33] and Qwen-Image-Edit [50] as our base architectures, which employ Qwen2.5VL 7B Instruct [1] for text embedding and a DiT as their diffusion heads, initialized from these respective models. In contrast to the original base models, we enhance the MLLM and diffusion transformer with Thinking and Reflection capabilities on image editing. This is achieved through a multistage training strategy and subsequent fine-tuning on our reasoning-enhanced dataset, thereby progressively refining the model’s performance. It is important to note that, while Step1X-Edit and Qwen-Image-Edit serve as our chosen implementations, our proposed method is broadly applicable across various image editing approaches.

###### 3.2.2. Multi-stage Training

Prior work highlights that such reconciliation often necessitates dedicated architectural advancements, for instance, in vision encoders [35, 42, 49], to mitigate conflicts during early joint training on both understanding and generation. To address these complex dynamics and effectively integrate enhanced reasoning with generative processes, we

adopt a multi-stage training strategy. This progressive approach decomposes the intricate joint optimization into simpler, focused tasks: initially cultivating the MLLM’s explicit Thinking and Reflection, subsequently adapting the Generator (DiT) to these refined MLLM on image editing, and culminating in a comprehensive joint fine-tuning of both components to achieve superior overall performance.

REASONING LEARNING STAGE . This initial stage is dedicated to cultivating the MLLM’s explicit Thinking and Reflection capabilities tailored for image editing tasks. To efficiently adapt the model while mitigating catastrophic forgetting of its foundational knowledge, and to isolate reasoning training, we employ Low-Rank Adaptation (LoRA) [18] on the linear layers in attention modules. During this phase, the DiT remains frozen. Training is conducted on the constructed Thinking Pairs and Reflection Triples datasets (cf. Sec. 3.1), optimizing with a standard Next Token Prediction (NTP) loss,

LNTP = Et

i −

L

log pθ(tk|t1,t2,··· ,tk−1) (1)

k=1

where tk represents the k-th token in a sequence of length L, and pθ is the probability predicted by the MLLM parameterized by θ.

EDIT LEARNING STAGE . Following the dedicated tuning of the MLLM’s reasoning abilities, this stage focuses on adapting the Generator, specifically the DiT model. To leverage the MLLM’s refined contextual understanding without interference, its parameters are kept frozen throughout this phase. The DiT is trained using a flow matching loss [32], with a dual objective that encompasses both textto-image (T2I) generation and direct image editing tasks. Including T2I data is crucial; their significantly larger scale and broader domain coverage are instrumental in enriching the model’s general generative knowledge, which in turn substantially improves its proficiency in diverse editing scenarios. The flow matching loss is formulated as,

0∼D,x1∼N(0,I),c∥ut(x|c)−vt(x|x0,c)∥22

LFM = Et∼U(0,1),x

(2) where t is uniformly sampled from [0,1], x0 is a data point from the dataset D, x1 is standard Gaussian noise, and c represents the conditioning information (e.g., a text or a reference image). The DiT model ut is trained to predict the target vector field x1 − x0 at the interpolated point xt = (1 − t)x0 + tx1.

UNIFIED TUNING STAGE . After the preceding stages, this final stage unifies and jointly fine-tunes both the MLLM

Ref Image

[Reflection-round1]

[Figure 16]

Instruction

[Reflection-round2]

Ref Image

Instruction Craft the outerwear from lambskin leather.

Result round1

Result round1 Instruction

Craft the outerwear from lambskin leather.

Result round2

[Figure 17]

[Figure 18]

Unified Fine-tuned MLLM

[Figure 19]

[Figure 20]

<#Success>

###### [Thinking]

###### [Reflection step-by-step]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Uniﬁed Fine-tuned MLLM

Unified Fine-tuned MLLM

###### First Query

<Target Description>

Result round1, Instruction

<#Reflection> Replace the fur coat with a lambskin leather coat while maintaining the overall scene and background consistency.

Concrete Instruction

A long-haired girl in an outerwear crafted from lambskin leather.

[Figure 25]

Change the outerwear material to lambskin leather.

Craft the outerwear from lambskin leather.

###### Second Query

[Generation-round1]

[Generation-round2]

<Consistency Score>

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Result round2, Target Description A long-haired girl in an outerwear crafted from lambskin leather.

{score:10, reason: The image and the description are in full alignment, with no artifacts or distortions.}

###### Unified Fine-tuned MLLM

###### Unified Fine-tuned MLLM

[Figure 30]

Hidden States VAE . . .

Hidden States VAE . . .

[Figure 31]

[Figure 32]

Noisy Latent

Noisy Latent

[Figure 33]

[Figure 34]

VAE

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Connector

Connector

###### Third Query

[Figure 39]

Result round1,

Result round2 ,

Instru ction,

Consistency Score

Result-round1

<Conclusion>

Result-round2

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

{score:10, reason}

DiT

Craft the outerwear from lambskin leather.

DiT

VAE

VAE

VAE

[Figure 47]

[Figure 48]

<#Success>

- Figure 3. The model architecture and inference pipeline of REASONEDIT are structured around two core components: the Thinking and Reflection processes which serve as the Reasoner, and a DiT acting as the Generator. These Reasoner and Generator modules undergo a multi-stage training process. During inference, they operate in an interleaved and sequential manner, progressively yielding more precise image editing results through their integrated Thinking and Reflection capabilities.

and the DiT. This comprehensive joint optimization is crucial for ensuring that the understanding and generative processes seamlessly complement each other. The joint training loss for this stage is formulated as,

Speed Ulysses [21]. For the DiT, both tensor parallelism and sequence parallelism were applied, enabling effective scaling across multiple nodes and GPUs and accelerating the training process.

Ljoint = LFM + ωNTP · LNTP (3)

#### 4. Experiments

##### 4.1. Experimental Settings

###### 3.2.3. Training Details

Benckmark. We conduct our experiments on three widelyused benchmarks: GEdit-Bench [33] and ImgEdit-Bench [54] for evaluating broad and comprehensive foundational image editing capabilities, and KRIS-Bench [52] for assessing a model’s advanced reasoning skills and ability to interpret abstract instructions. These benchmarks collectively enable a thorough evaluation of our model’s performance, ranging from foundational editing tasks to complex, abstract reasoning challenges.

We utilized Step1X-Edit V1.1 [44] and Qwen-ImageEdit [50] as our pretrained models. The first reasoning learning stage involved training the MLLM on 32 H800 GPUs (4 nodes, 8 GPUs/node) for 16 hours, completing 50,000 steps with an initial learning rate of 1 × 10−4. The second edit learning stage scaled to 128 GPUs (16 nodes, 8 GPUs/node), training for 38.9 hours and 28,000 steps at a learning rate of 1×10−5. During this stage, we use in-house 14.4M T2I samples and 2.4M image editing samples for training. The final stage consisted of 20 hours of training, completing 12,000 steps with a learning rate of 6×10−6 and the NTP loss weight ωNTP of 0.1. Similar to BAGEL [11] and Mogao [30], during this stage, FlexAttention [13] and a packed data format [10] were utilized to support efficient hybrid training for both understanding and generation tasks, especially on the Reflection Triples. To optimize training performance and scalability, distributed training employed several parallelization strategies. Specifically, the MLLM and the Connector utilized sequence parallelism and Deep-

Metrics. For the GEdit-Bench, we evaluate performance using three metrics-Semantic Consistency (SQ), Perceptual Quality (PQ), and an Overall Score (O)-which are automatically assessed by VIEScore [24] using GPT-4.1. On the ImgEdit-Bench, we use GPT-4.1 to assign 1-5 ratings across three dimensions-instruction adherence, imageediting quality, and detail preservation-where the final score for the latter two is capped by instruction adherence. On the KRIS-Bench, we use GPT-4o to assign 1-5 ratings across four dimensions-Visual Consistency, Visual Quality, In-

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Instruction] Change the animal in the image to China's most famous national treasure animal.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Instruction] Correct the unreasonable parts of the vehicle in the image.

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Instruction] Change to three spoons.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Instruction] Modify this photo to be set in a desert.

- Figure 4. Comprehensive qualitative evaluation of leading image editing models. The results demonstrate that our proposed approach, which incorporates thinking and reflection mechanisms, significantly outperforms the editing model.

struction Following, and the novel Knowledge Plausibility, which assesses consistency with real-world knowledge.

GEdit-Bench and ImgEdit-Bench primarily evaluate a model’s foundational editing capabilities. While our thinking and reflection mechanisms provide performance gains, their full impact may be less pronounced on these relatively simple tasks compared to more complex ones. This is consistent with the design of our dataset, where the thinking and reflection modules are specifically tailored for complex instructions and multi-step editing.

##### 4.2. Experimental Results

Quantitative results are first reported on GEdit-Bench [33]and ImgEdit-Bench [54] to assess foundational editing capabilities (cf. Sec. 4.2.1), with the evaluation then shifting to the more complex KRIS-Bench [52] for an assessment of abstract reasoning skills (cf. Sec. 4.2.2).

As shown in Fig. 4, a qualitative comparison demonstrates that our approach excels at precisely altering target areas while faithfully maintaining the integrity of unedited regions, such as backgrounds, facial features, and hairstyles. This capability addresses a key challenge in image editing by effectively mitigating common failures related to consistency and fidelity, resulting in stable performance and accurate responsiveness to a wide range of commands.

###### 4.2.1. Evaluation on GEdit-Bench and ImgEdit-Bench

As shown in Table 1, our method achieves superior performance on the foundational instruction benchmarks ImgEdit-Bench [54] and GEdit-Bench [33]. Both ReasonEdit-S and ReasonEdit-Q exhibit consistent and substantial improvements over their respective base models, achieving gains of +4.3% and +4.7% on ImgEdit and GEdit for ReasonEdit-S, and +2.8% and +3.4% for ReasonEditQ, respectively. Importantly, although the two methods are built upon different underlying models - Step1X-Edit v1.1 for ReasonEdit-S and Qwen-Image-Edit for ReasonEditQ - both variants significantly surpass the performance of their corresponding underlying editors. ReasonEdit-S ranks third on GEdit-Bench, outperforming Qwen-Image-Edit, while ReasonEdit-Q achieves the highest overall score. On ImgEdit-Bench, ReasonEdit-S and ReasonEdit-Q place second and third among all open-source models, trailing the top entry by only 0.08 and 0.12 points, respectively.

###### 4.2.2. Evaluation on Kris-Bench

On the KRIS-Bench [52], the proposed approach demonstrates the best performance among all open-source models, including those that also employ a thinking-based mechanism (e.g., BAGEL-Thinking), and surpasses several closed-source methods. This superiority is further substantiated by substantial improvements over their respective base models (w/o thinking, w/o reflection): ReasonEdit-S (built upon Step1X-Edit v1.1 [33]) attains a performance gain of +8.2%, while ReasonEdit-Q (derived from Qwen-

Table 1. Comprehensive quantitative evaluation of leading image editing models. We validate our approach using two variants: ReasonEditS (initialized from Step1X-Edit v1.1) and ReasonEdit-Q (initialized from Qwen-Image-Edit). Our method achieves significant performance gains over the base models (w/o thinking, w/o reflection) and achieves state-of-the-art performance among open-source models on both GEdit and Kris (with ReasonEdit-Q), while also proving to be highly competitive with several closed-source models.

GEdit-Bench KRIS-Bench ImgEdit-Bench Semantic Consistency

Models

Factual Knowledge

Conceptual Knowledge

Procedural Knowledge

Quality Overall

###### Overall Overall

Gemini 2 flash (Apr. 2025) 6.87 7.44 6.51 65.26 59.65 62.90 62.41 Gemini 2.5 flash (Sep. 2025) 8.25 8.29 7.89 77.03 78.29 75.93 77.29 4.30 Doubao (seed edit 1.6, Apr. 2025) 7.22 7.89 6.98 63.30 62.23 54.17 60.70 Doubao (Seedream 4.0, Aug. 2025) 9.17 7.95 8.40 78.10 76.86 76.93 77.31 4.46 GPT4o (Apr. 2025) 7.74 8.13 7.49 79.80 81.37 78.32 80.09 GPT4o (Sep. 2025) 8.74 7.67 8.01 81.16 78.24 77.09 79.00 4.30

close-source models

ICEdit [58] 4.94 7.39 4.87 46.99 42.73 27.76 40.70 3.05 Omnigen [53] 5.88 5.87 5.01 33.11 28.02 23.89 28.85 2.96 Omnigen 2 [51] 7.16 6.77 6.41 57.36 44.20 47.79 49.71 3.44 BAGEL-thinking [11] 7.70 6.51 6.66 66.18 61.92 49.02 60.18 3.56 BAGEL [11] 7.48 6.80 6.60 60.26 55.86 51.69 56.21 3.20 Uniworld-V1 [31] 4.93 7.43 4.85 47.71 44.80 47.92 50.27 3.26 Hidream-I1 (E1) [17] 5.66 6.06 5.01 43.31 50.05 37.64 44.72 3.17 Hidream-E1.1 [17] 7.15 6.65 6.42 43.52 44.71 36.08 42.25 3.97 Flux-Kontext-dev [5] 7.16 7.37 6.51 53.28 50.36 42.53 49.54 3.97 UniWorld-FLUX.1-Kontext-Dev [29] 7.28 7.49 6.74 55.50 51.39 43.76 51.04 4.02 UniWorld-Qwen-Image-Edit [29] 8.36 7.87 7.76 61.72 56.38 46.69 55.98 4.48 Step1X-Edit v1.1 [33] 7.66 7.35 6.97 53.05 54.34 44.66 51.59 3.90 ReasonEdit-S (base) 7.77 7.65 7.24 58.23 60.55 46.21 56.33 4.22 ReasonEdit-S (thinking) 8.02 7.64 7.36 (+1.7%) 59.79 62.76 49.78 58.64 (+4.1%) 4.18 (-0.9%) ReasonEdit-S (thinking+reflection) 8.18 7.85 7.58 (+4.7%) 62.44 65.72 50.42 60.93 (+8.2%) 4.40 (+4.3%) Qwen-Image-Edit-2509 [50] 8.00 7.86 7.56 61.47 56.79 47.07 56.15 4.27 ReasonEdit-Q (base) 8.12 7.94 7.51 62.29 62.22 44.53 58.05 4.24 ReasonEdit-Q (thinking) 8.20 7.96 7.61 (+1.3%) 62.44 64.49 52.02 60.81(+4.8%) 4.27 (+0.7%) ReasonEdit-Q (thinking+reflection) 8.34 7.97 7.77 (+3.4%) 63.92 64.85 52.41 61.57 (+6.1%) 4.36 (+2.8%)

open-source models

Image-Edit [50]) achieves an improvement of +6.1%.

The performance gains are attributed to the method’s ability to simplify abstract and difficult editing tasks into clear, actionable steps for the editing model. Furthermore, the reflection pipeline provides a crucial mechanism to analyze the correctness of an edit and formulate strategies for improvement. This iterative process of self-correction allows the model to identify and rectify subtle errors, effectively mitigating hallucination and improving overall fidelity. The method’s demonstrated effectiveness on both complex and simple tasks (cf. Sec. 4.2.1) proves its versatility and robust generalization.

As shown in Fig. 4, many methods often misinterpret or fail to respond correctly to abstract or complex instructions. Our proposed thinking module effectively aids the editing model in understanding such instructions and executing them accurately. Furthermore, the reflection pipeline enhances this process by enabling the model to identify and rectify subtle errors, formulate precise refinement strategies, and prevent the compounding of mistakes that are common in multi-step editing tasks. Simultaneously, many models struggle with maintaining consistency in complex scenarios, often leading to unintended alterations in unedited regions because they lack a robust understanding of the entire scene’s structure. In contrast, our approach ensures high consistency by faithfully preserving elements that should remain unchanged.

##### 4.3. Ablation Studies

To systematically evaluate the contribution of each component of the proposed method, a series of ablation studies on ReasonEdit-S (built upon Step1X-Edit v1.1 [33] and the MLLM Qwen2.5VL 7B Instruct, hereafter Qwen) using KRIS-Bench, as its abstract and challenging nature makes it an ideal testbed for verifying the reasoning and reflection capabilities of the model.

Table 2. Ablation of Multi-Stage Training. This table evaluates the performance contributions of each stage in the training pipeline, from the pre-trained baseline to the final unified model, highlighting the cumulative benefits of fine-tuning the generator and reasoning modules at each step.

KRIS-Bench Factual Knowledge

Methods

Conceptual Knowledge

Procedural Knowledge

###### Overall

Pre-trained Generator (Step1X-Edit V1.1 [33]) 53.05 54.34 44.66 51.59 Pre-trained Generator + Qwen Reasoning 54.05 57.44 41.26 52.41 Pre-trained Generator + Qwen-tuned Reasoning 56.32 62.00 46.17 56.24 Base Generator W/O Reasoning 55.80 55.28 43.78 52.74 Base Generator + Qwen-tuned Reasoning 60.54 62.16 48.26 58.29 Unified Tuned (ReasonEdit-S) 62.44 65.72 50.42 60.93

Impact of Multi-Stage Training. To evaluate the contribution of the reasoning learning stage, we compare the performance of the Pre-trained Generator (Step1X-Edit v1.1 [33]) when integrated with either a base (untuned) Qwen model or a fine-tuned Qwen model. When the Pretrained Generator is augmented with the base Qwen model leveraging our thinking and reflection mechanism, only a marginal performance gain of 0.82 points is observed. In

contrast, fine-tuning Qwen on our reasoning data consistently and significantly outperforms this base configuration. This highlights that foundational multimodal large language models, without domain-specific adaptation, struggle to effectively grasp the nuances of image editing, thereby underscoring the critical necessity of tailoring the MLLM to these specific demands. After the edit learning stage, in isolation, the Base Generator achieves a degree of performance improvement over the Pre-trained Generator, demonstrating its role in adapting the generative capabilities. Finally, this multi-stage strategy culminates in the optimal performance of the unified training, providing a substantial performance increase from the Base Generator + Qwen-tuned Reasoning model to the Unified Tuned model (58.29 vs. 60.93), validating the synergistic benefits of training the entire pipeline as a whole.

- Table 3. Ablation Study on the Contributions of the Thinking and Reflection Modules. The table shows the performance of four model variants on the KRIS-Bench, demonstrating the benefits of each component and the synergy of their combination.

Methods

KRIS-Bench Factual Knowledge

Conceptual Knowledge

Procedural Knowledge

Overall

ReasonEdit-S (w/o thinking, w/o reflection) 58.23 60.55 46.21 56.33 ReasonEdit-S (w/ thinking, w/o reflection) 59.79 62.76 49.78 58.64 ReasonEdit-S (w/o thinking, w/ reflection) 61.40 64.16 48.16 59.39 ReasonEdit-S (w/ thinking, w/ reflection) 62.44 65.72 50.42 60.93

Ablation of Thinking and Reflection. To understand the individual and combined contributions of the thinking and reflection modules, four variants are compared: (1) a baseline model without either module; (2) a model with only the thinking module; (3) a model with only the reflection module; and (4) the full model incorporating both. The results on KRIS-Bench (see Tab. 3) show a gradual improvement in performance with the addition of each component. The thinking module alone provides a significant performance boost, confirming its effectiveness in handling complex instructions. The thinking + reflection module proves beneficial across all benchmarks (see Tab. 1), as it effectively rectifies errors. The full model, with both modules integrated, achieves the highest scores, highlighting the synergistic relationship between understanding an instruction and correcting subsequent errors.

- Table 4. Ablation Study on Reflection Pipelines. The table compares three different reflection mechanisms on KRIS-Bench, highlighting the effectiveness of the proposed multi-round pipeline.

KRIS-Bench Factual Knowledge

Methods

Conceptual Knowledge

Procedural Knowledge

###### Overall

Base Generator 55.80 55.28 43.78 52.74 Base Generator + dual-image pipeline 52.97 61.84 41.12 53.79 Base Generator + single-image pipeline 54.81 56.92 43.70 53.04 Base Generator + our multi-round pipeline 60.54 62.16 48.26 58.29

Comparison of Reflection Pipelines. To ensure consistency in the DiT parameters, this ablation study is con-

ducted by combining the Base Generator (the DiT after the edit learning stage) with each reflection pipeline. Tab. 4 compares three distinct approaches to the reflection process—a dual-image pipeline, a pure single-image pipeline, and the proposed multi-round prior pipeline. The dualimage pipeline, which relies on a direct comparison between the initial input and the generated output, is often prone to hallucinations. Conversely, a pure single-image approach struggles with tasks that require a clear beforeand-after comparison, such as Portrait Beautification or motion/expression-related edits. As shown in the table, the proposed multi-round single-image prior pipeline is superior. This is attributed to the method’s ability to combine the benefits of both approaches, allowing it to perform a selfcorrection loop on the generated image itself while leveraging key prior information from the multi-round process.

Reflection Performance Curve. The effect of varying reflection rounds is evaluated on KRIS-Bench using ReasonEdit-S (see Tab. 5). The results show that incorporating reflection consistently improves performance over the Thinking-only baseline (58.64). Specifically, two reflection rounds yield a score of 60.93, while extending to three or four rounds brings only marginal improvements (+0.06 and +0.14, respectively) with higher computational cost. We further evaluate a naive re-roll baseline (see Tab. 6). This simpler strategy yields only minor gains, peaking at 59.24 after three attempts, and even drops to 59.09 at four attempts with increased cost. The comparison indicates that the targeted reflection mechanism markedly outperforms an unguided re-roll strategy, highlighting the role of structured reasoning in iterative error correction (see Fig. 5).

[Figure 85]

- Table 5. Performance-efficiency curve of reflection rounds.

Reflection Rounds 0 (Thinking) 1 2 3 4 Performance 58.64 60.08 60.93 60.99 61.07 Time(s) 40 80 120 160 200

- Table 6. Performance-efficiency curve of re-roll times.

Re-roll Times 0 (Thinking) 1 2 3 4 Performance 58.64 58.84 59.00 59.24 59.09 Time(s) 39 78 117 156 195

Figure 5. PerformanceEfficiency Curve.

#### 5. Conclusion

In this work, we present ReasonEdit, a fundamental image editing framework that demonstrates the crucial role of explicit reasoning in achieving robust and versatile performance. The proposed method introduces a novel pipeline with two core capabilities: thinking and reflection. By training these capabilities on a curated collection of Thinking Pairs and Reflection Triples, the framework learns to convert abstract user requests into actionable commands and to perform self-correction in an iterative loop. Extensive experiments on a range of benchmarks validate the efficacy

of this approach, with the model achieving state-of-the-art performance among open-source methods while remaining highly competitive with several closed-source models. This work provides a new perspective on reasoningenhanced image editing, showing that a structured pipeline for instruction understanding and self-correction is vital for building models that can handle both simple and complex editing tasks with high fidelity and consistency.

#### Contributors and Acknowledgments

Core Contributors: Fukun Yin, Shiyu Liu, Yucheng Han, Zhibo Wang, Peng Xing, Rui Wang, Wei Cheng, Yingming Wang, Aojie Li, Zixin Yin, Pengtao Chen, Xiangyu Zhang, Daxin Jiang, Xianfang Zeng, Gang Yu.

Corresponding Authors: Xianfang Zeng (zengxianfang@stepfun.com), Gang Yu (yugang@stepfun.com), Daxin Jiang (djiang@stepfun.com).

#### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 4
- [2] Black Forest Labs. Flux.1 [dev]. https://huggingface. co/black-forest-labs/FLUX.1-dev, 2024. 2
- [3] Black Forest Labs. Flux.1 fill [dev]. https : / / huggingface.co/black-forest-labs/FLUX.1-Filldev, 2024. Accessed: 2025-04-19. 2
- [4] Black Forest Labs. Flux.1 [schnell]. https : / / huggingface.co/black-forest-labs/FLUX.1-schnell,

- 2024. 2

[5] BlackForestLabs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space,

- 2025. 2, 4, 7

- [6] Manuel Brack, Felix Friedrich, Katharia Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting, and Apolin´ario Passos. Ledits++: Limitless image editing using text-to-image models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8861–8870, 2024. 1
- [7] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18392–18402, 2022. 1, 2

- [8] Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng Zhang, Fengbin Gao, Peihan Xu, et al. Hidream-i1: A high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705, 2025. 2
- [9] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025. 2
- [10] Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36:2252–2274, 2023. 5
- [11] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 1, 3, 5, 7
- [12] Peng Ding, Jingyu Wu, Jun Kuang, Dan Ma, Xuezhi Cao, Xunliang Cai, Shi Chen, Jiajun Chen, and Shujian Huang. Hallu-pi: Evaluating hallucination in multi-modal large language models within perturbed inputs. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 10707–10715, 2024. 1
- [13] Juechu Dong, Boyuan Feng, Driss Guessous, Yanbo Liang, and Horace He. Flex attention: A programming model for generating optimized attention kernels. arXiv preprint arXiv:2412.05496, 2024. 5
- [14] Rongyao Fang, Chengqi Duan, Kun Wang, Linjiang Huang, Hao Li, Shilin Yan, Hao Tian, Xingyu Zeng, Rui Zhao, Jifeng Dai, Xihui Liu, and Hongsheng Li. Got: Unleashing reasoning capability of multimodal large language model for visual generation and editing. arXiv preprint arXiv:2503.10639, 2025. 3
- [15] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pages 148–166. Springer, 2024. 1
- [16] Zhen Han, Zeyinzi Jiang, Yulin Pan, Jingfeng Zhang, Chaojie Mao, Chen-Wei Xie, Yu Liu, and Jingren Zhou. ACE: All-round creator and editor following instructions via diffusion transformer. In The Thirteenth International Conference on Learning Representations, 2025. 2
- [17] HiDream-ai. Hidream-e1. https://github.com/HiDreamai/HiDream-E1, 2025. 2, 7
- [18] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. 4
- [19] Wen Huang, Hongbin Liu, Minxin Guo, and Neil Gong. Visual hallucinations of multi-modal large language models. In Findings of the Association for Computational Linguistics: ACL 2024, pages 9614–9631, 2024. 1

- [20] Wenxuan Huang, Shuang Chen, Zheyong Xie, Shaosheng Cao, Shixiang Tang, Yufan Shen, Qingyu Yin, Wenbo Hu, Xiaoman Wang, Yuntian Tang, et al. Interleaving reasoning for better text-to-image generation. arXiv preprint arXiv:2509.06945, 2025. 1, 3
- [21] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Reza Yazdani Aminadabi, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. System optimizations for enabling training of extreme long sequence transformer models. In Proceedings of the 43rd ACM Symposium on Principles of Distributed Computing, page 121–130, New York, NY, USA, 2024. Association for Computing Machinery. 5
- [22] Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion. In European Conference on Computer Vision, pages 150–168. Springer,

2024. 1, 2

- [23] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6007–6017, 2023. 1
- [24] Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, and Wenhu Chen. Viescore: Towards explainable metrics for conditional image synthesis evaluation. arXiv preprint arXiv:2312.14867, 2023. 5
- [25] Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, and Wenhu Chen. Viescore: Towards explainable metrics for conditional image synthesis evaluation. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12268–12290, 2024. 2
- [26] Shufan Li, Konstantinos Kallidromitis, Akash Gokul, Arsh Koneru, Yusuke Kato, Kazuki Kozuka, and Aditya Grover. Reflect-dit: Inference-time scaling for text-to-image diffusion transformers via in-context reflection. arXiv preprint arXiv:2503.12271, 2025. 1, 3
- [27] Yaowei Li, Yuxuan Bian, Xu Ju, Zhaoyang Zhang, Ying Shan, and Qiang Xu. Brushedit: All-in-one image inpainting and editing. ArXiv, abs/2412.10316, 2024. 2
- [28] Zijie Li, Henry Li, Yichun Shi, Amir Barati Farimani, Yuval Kluger, Linjie Yang, and Peng Wang. Dual diffusion for unified image generation and understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2779–2790, 2025. 1
- [29] Zongjian Li, Zheyuan Liu, Qihui Zhang, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Yang Ye, Wangbo Yu, Yuwei Niu, and Li Yuan. Uniworld-v2: Reinforce image editing with diffusion negative-aware finetuning and mllm implicit feedback. arXiv preprint arXiv:2510.16888, 2025. 7
- [30] Chao Liao, Liyang Liu, Xun Wang, Zhengxiong Luo, Xinyu Zhang, Wenliang Zhao, Jie Wu, Liang Li, Zhi Tian, and Weilin Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025. 5
- [31] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang,

- Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025. 2, 7
- [32] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. 4
- [33] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 1, 2, 4, 5, 6, 7
- [34] Pengqi Lu. Qwen2vl-flux: Unifying image and text guidance for controllable image generation, 2024. 2
- [35] Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai Yu, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7739–7751,

2025. 4

- [36] Chaojie Mao, Jingfeng Zhang, Yulin Pan, Zeyinzi Jiang, Zhen Han, Yu Liu, and Jingren Zhou. Ace++: Instructionbased image creation and editing via context-aware content filling. arXiv preprint arXiv:2501.02487, 2025. 2
- [37] Zhenxing Mi, Kuan-Chieh Wang, Guocheng Qian, Hanrong Ye, Runtao Liu, Sergey Tulyakov, Kfir Aberman, and Dan Xu. I think, therefore i diffuse: Enabling multimodal incontext reasoning in diffusion models. In Forty-second International Conference on Machine Learning, 2025. 3
- [38] OpenAI. Introducing 4o image generation, 2025. 4
- [39] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025. 2
- [40] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 4

- [41] Luozheng Qin, Jia Gong, Yuqing Sun, Tianjiao Li, Mengping Yang, Xiaomeng Yang, Chao Qu, Zhiyu Tan, and Hao Li. Uni-cot: Towards unified chain-of-thought reasoning across text and vision. arXiv preprint arXiv:2508.05606,

2025. 3

- [42] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2545–2555, 2025. 4
- [43] Yichun Shi, Peng Wang, and Weilin Huang. Seededit: Align image re-generation to image editing. arXiv preprint arXiv:2411.06686, 2024. 4

- [44] StepFun AI. Step1x-edit v1.1 diffusers. https:// huggingface . co / stepfun - ai / Step1X - Edit - v1p1 diffusers, 2025. Accessed: 2025-09-23. 5
- [45] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision,

2025. 2

- [46] NextStep Team, Chunrui Han, Guopeng Li, Jingwei Wu, Quan Sun, Yan Cai, Yuang Peng, Zheng Ge, Deyu Zhou, Haomiao Tang, et al. Nextstep-1: Toward autoregressive image generation with continuous tokens at scale. arXiv preprint arXiv:2508.10711, 2025. 1
- [47] Fei Wang, Xingyu Fu, James Y. Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, Tianyi Lorena Yan, Wenjie Jacky Mo, HsiangHui Liu, Pan Lu, Chunyuan Li, Chaowei Xiao, Kai-Wei Chang, Dan Roth, Sheng Zhang, Hoifung Poon, and Muhao Chen. Muirbench: A comprehensive benchmark for robust multi-image understanding. In The Thirteenth International Conference on Learning Representations, 2025. 1
- [48] Yi Wang, Mushui Liu, Wanggui He, Longxiang Zhang, Ziwei Huang, Guanghao Zhang, Fangxun Shu, Zhong Tao, Dong She, Zhelun Yu, et al. Mint: Multi-modal chain of thought in unified generative models for enhanced image generation. arXiv preprint arXiv:2503.01298, 2025. 1, 3
- [49] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12966–12977, 2025. 4
- [50] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report,

2025. 1, 2, 4, 5, 7

- [51] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025. 1, 3, 7
- [52] Yongliang Wu, Zonghui Li, Xinting Hu, Xinyu Ye, Xianfang Zeng, Gang Yu, Wenbo Zhu, Bernt Schiele, MingHsuan Yang, and Xu Yang. Kris-bench: Benchmarking next-level intelligent image editing models. arXiv preprint arXiv:2505.16707, 2025. 5, 6
- [53] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13294–13304, 2025. 1, 2, 7
- [54] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A uni-

- fied image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025. 5, 6
- [55] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26125–26135, 2025. 2
- [56] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023. 2
- [57] Renrui Zhang, Chengzhuo Tong, Zhizheng Zhao, Ziyu Guo, Haoquan Zhang, Manyuan Zhang, Jiaming Liu, Peng Gao, and Hongsheng Li. Let’s verify and reinforce image generation step by step. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 28662– 28672, 2025. 1, 3
- [58] Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with in-context generation in large scale diffusion transformer. arXiv, 2025. 2, 7
- [59] Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093, 2024. 2
- [60] Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen. A task is worth one word: Learning with task prompts for high-quality versatile image inpainting. In European Conference on Computer Vision, pages 195–211. Springer, 2024. 1, 2

#### A. Appendix

##### A.1. Illustration of Reflection Pipelines

We conduct an ablation study on the proposed multi-round reflection pipeline against two alternative designs, as presented in Table 4 and further illustrated in Fig. 6. The Dual Image Reflection pipeline directly inputs the reference image, edit instruction, and result image, tasking the MLLM with generating thinking and reflection concurrently. Nevertheless, we found that current MLLMs frequently exhibit hallucinations in image editing tasks under this unified input scheme. Our investigation then led to the Single Image Reflection pipeline, which first guides the MLLM to describe the target image based on the edit instruction and reference image. Subsequently, using this target description, the MLLM evaluates the result image, offering detailed reasoning, identifying failures, suggesting refinements, or confirming success. A key drawback here is that the MLLM loses the essential context of the reference image during its final conclusion, leading to less effective assessments. The proposed pipeline addresses these limitations by decomposing the reflection into three distinct sub-procedures: target description, result assessment, and refinement conclusion. In the initial two sub-procedures, the MLLM receives only a single image as input (e.g., reference image for target description, result image for assessment), which significantly enhances accuracy. In the final stage, the MLLM is provided with all relevant information to formulate a comprehensive conclusion for subsequent actions, thereby maintaining full contextual awareness.

#### B. More Results

To further complement the quantitative analyses in the main paper, this section presents additional qualitative results demonstrating the breadth and robustness of our framework. As shown in Figures 7–10, these examples cover diverse editing instructions, multi-round reasoning-and-reflection processes, and comparisons against state-of-the-art methods.

#### C. Failure Cases

To provide a more comprehensive evaluation of ReasonEdit, we also explicitly analyze scenarios where our model produces unsatisfactory results. As illustrated in Fig. 11, we present representative failure cases encountered during our experiments. We believe that analyzing these limitations will inspire future work to further improve the robustness of reasoning-based editing models.

All in one Assess & Conclude

Target Describe

Instruction

Ref. Instruction Image

Result Image

Ref. Instruction Image

Result Image

|The result image should contain ….|
|---|

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

MLLM

MLLM

### MLLM

|<#Think> … <#Reflection>|
|---|

|The result image should contain ….|
|---|

|Consistency Score: <#Think> … <#Reflection>|
|---|

(a) Dual-image Reflection

(b) Single-image Reflection

Target Describe

Result Assess

###### Refinement Conclude

Instruction

|The result image should contain ….|
|---|

Ref. Instruction Image

Result Image

Ref. Image

Result Image

|Consistency Score: Reason:|
|---|

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

MLLM

MLLM

MLLM

|The result image should contain ….|
|---|

|… Consistency Score: Reason:|
|---|

|<#Think> … <#Reflection>|
|---|

(c) Multi-round Reflection

- Figure 6. Three distinct MLLM-based reflection pipelines for image editing. (a) Dual-image Reflection processes the instruction, reference image, and result image simultaneously in a single MLLM call to produce a combined thinking and reflection output. (b) Single-image Reflection decomposes this into two sequential MLLM calls: first, generating a target description from the instruction and reference image, and then assessing the result image against this description to provide a consistency score and reflection. (c) Our proposed Multi-round Reflection pipeline further refines the process into three dedicated stages: (1) Target Describe, which formulates a target image description from the input instruction and reference image; (2) Result Assess, which evaluates the generated image against this target description to output consistency scores and reasons; and (3) Refinement Conclude, where the MLLM analyzes the assessment to provide success/failure judgments and, if necessary, detailed instructions for further image modification, leveraging both the reference and result images. This multi-round approach enables a more granular and iterative refinement of image editing outcomes.

###### Original Image W/O Rensoning W/ Rensoning

[Figure 98]

[Figure 99]

[Figure 100]

###### [Thinking]

###### [Reflection]

Remove the background.

<#Think> The user's instruction was to remove the background and keep the Superman figure. The second image successfully removed the background and isolated the subject, fulfilling the request perfectly without introducing any artifacts. <#Success>

[Instruction] Can you remove the background from this image? Only keep the Superman figure.

[Figure 101]

[Figure 102]

[Thinking] Add a missing leg to the right side of the chair, matching the existing wooden structure.

[Figure 103]

###### [Reflection]

<#Think> The user wants me to complete the legs of the chair. The second image shows that all four legs have been added to the chair, matching the style and color of the existing ones, and there are no other significant changes or artifacts. <#Success>

[Instruction] Complete the legs of the chair.

[Thinking] Replace the bicycle wheels with regular bicycle wheels.

###### [Reflection]

[Figure 104]

[Figure 105]

[Figure 106]

<#Think> The user wants me to correct the unreasonable parts of the bicycle in the image. The first image shows a bicycle with square wheels, which is unrealistic. The second image has successfully changed these square wheels into standard circular ones, fulfilling the instruction perfectly without introducing any artifacts. <#Success>

[Instruction] Correct the unreasonable parts of the bicycle in the image.

[Thinking] Change the color of the bread to a dark brown.

###### [Reflection]

[Figure 107]

[Figure 108]

[Figure 109]

<#Think>The user wants me to make the bread appear overcooked. The output image has successfully changed the light, uncooked bread into a dark, burnt-looking slice, which effectively conveys the concept of being overcooked without introducing any artifacts or unwanted changes. <#Success>

[Instruction] Modify to appear overcooked.

[Thinking] Replace the woman's floral dress with a traditional Japanese kimono.

###### [Reflection]

[Figure 110]

[Figure 111]

[Figure 112]

<#Think> The user wants me to dress the woman in traditional Japanese clothing. The output image successfully replaces the original floral dress with a traditional Japanese kimono, which includes a haori, hakama, and obi, while maintaining the subject's pose and the background. The edit is high quality without any noticeable artifacts. <#Success>

[Instruction] Dress the woman in the image in traditional Japanese clothing.

[Thinking] Add holes to the leaves.

[Reflection] <#Think> The user wants me to add pest infestation to the leaves. The second image clearly shows several holes in the leaves, which convincingly represents pest infestation, and the rest of the image remains consistent without any noticeable artifacts. <#Success>

[Figure 113]

[Figure 114]

[Figure 115]

[Instruction] After the leaves in the image encounter pest infestation.

[Figure 116]

[Thinking] Add a printed document in front of the printer.

###### [Reflection]

[Figure 117]

[Figure 118]

<#Think> The user wants me to place a printed document in front of the printer. The output image successfully adds a printed document onto the existing sheet of paper in front of the printer, fulfilling all aspects of the command without introducing any artifacts. <#Success>

[Instruction] Place the printed document in front of the printer.

- Figure 7. Qualitative examples illustrating the thinking capabilities of our MLLM Reasoner across various image editing instructions.

###### Original Image W/O Rensoning W/ Rensoning

[Thinking] Invert the colors of the image.

###### [Reflection-round1]

[Figure 119]

[Figure 120]

[Figure 121]

<#Think> The user instructed to invert the colors of the image, changing the black text to white and the white background to black. The resulting image did not perform this color inversion at all; it remains identical to the original image. <#Failed>

###### [Reflection-round2]

[Figure 122]

[Instruction] Invert the colors of the image.

<#Think> The user wants me to invert the colors of the image. The original image has a black logo on a white background. The output image has the same logo on a light gray background. The core instruction was to invert the colors, which would result in a white logo on a black background. Since this was not done, the edit is unsuccessful and requires correction. <#Reflection> Invert the colors of the image, so the text becomes white and the background turns black.

[Figure 123]

###### [Reflection-round3]

<#Think> The user instructed to invert the colors of the image. The original image has black text on a white background, while the edited image has white text on a black background, which perfectly fulfills the instruction without any other alterations. <#Success>

- [Reflection-round1] <#Think> The user wants me to evaluate an image editing task where the bird's action is changed to "flapping its wings and flying high". The second image shows the bird flapping its wings, which is part of the requested action. However, the bird is still sitting on the branch, not "flying high", so the edit is only partially successful. <#Reflection> Make the bird fly off the branch, creating a blurred background to indicate motion and distance.
- [Reflection-round2] <#Think> The user wants me to change the bird's action to flapping its wings and flying high. The second image successfully depicts the bird flapping its wings, but it is still perched on a branch, failing to convey the concept of "flying high". This is a partial success that requires a correction. <#Reflection> Remove the branch the bird is perched on, so it appears to be flying freely in the air.
- [Reflection-round3] <#Think> The user instructed to change the bird's action to flapping its wings and flying high. The output image successfully depicts the bird mid-flight with wings flapping, accurately fulfilling both parts of the command without introducing any noticeable artifacts. <#Success>

[Thinking] Make the bird flapping its wings and in mid-flight.

[Figure 124]

[Figure 125]

[Figure 126]

[Instruction] Change the bird's action to flapping its wings and flying high.

[Figure 127]

[Figure 128]

- [Reflection-round1] <#Think> The user instructed to make the shape a circle with an eccentricity of 0. The model successfully changed the shape from an oval to a circle, but the resulting shape is still an ellipse, not a perfect circle with zero eccentricity. This requires a correction. <#Reflection> Make the circle perfectly round, with no visible oval shape.
- [Reflection-round2] <#Think> The user instructed to make the shape a circle with eccentricity 0, but the resulting image is still an ellipse. The instruction was not followed correctly, so a reflection is needed to change the shape from an ellipse to a circle.

[Thinking] Change the shape to a perfect circle.

[Figure 129]

[Figure 130]

[Figure 131]

[Instruction] Make it a circle with eccentricity 0.

[Figure 132]

<#Reflection> Change the shape from an ellipse into a perfect circle.

[Reflection-round3] <#Think> The user's instruction was to make the shape a circle with an eccentricity of 0. The resulting image is a perfect circle with no apparent eccentricity, which exactly matches the instruction. <#Success>

[Figure 133]

###### [Reflection-round1]

[Figure 134]

[Figure 135]

[Figure 136]

[Thinking] Replace the text "听喜欢的歌 ⾛ 喜欢的路" with " 我爱你".

<#Think> The user instructed to remove all text and add "我爱你". The model successfully added the text "我爱你", but it failed to remove the original lines of handwritten text below it. <#Reflection> Remove the handwritten text "⾛喜欢的路".

###### [Reflection-round2]

[Instruction] Remove the text and add “我爱你”.

[Figure 137]

<#Think> The user's instruction was to remove the existing text and add the text "我爱你". The output image has successfully removed the original text and added the specified text "我爱你" in a consistent style, while keeping the rest of the image, including the background and a small heart symbol, intact. <#Success>

- Figure 8. Qualitative multi-round examples illustrating the thinking and the reflection progressively correct and improve the quality of the generate results.

###### Original Image W/O Rensoning W/ Rensoning

[Thinking] Replace the seagull with an owl.

- [Reflection-round1] <#Think> The user wants me to replace the seagull with a bird of wisdom from Greek mythology. The instruction is vague about which specific bird, but the most well-known bird associated with wisdom in Greek mythology is the owl. The output image replaced the seagull with an eagle, which is incorrect. Therefore, the edit failed because the wrong animal was chosen. <#Failed>
- [Reflection-round2] <#Think> The user wants to replace the seagull with a bird from Greek mythology associated with wisdom. The output image shows an eagle, which is a real animal, not a mythical bird. Therefore, the edit is only partially successful because it failed to specify the mythical nature of the bird. <#Reflection> Change the eagle into a mythical bird of wisdom from Greek mythology, such as Athena's owl or a phoenix.
- [Reflection-round3] <#Think> The user wants me to replace the seagull with a bird of wisdom from Greek mythology. The output image successfully replaced the seagull with an owl, which is considered a bird of wisdom in Greek mythology. The new creature fits the context of the prompt, the background was preserved well, and there are no noticeable artifacts. <#Success>

[Figure 138]

[Figure 139]

[Figure 140]

[Instruction] Replace the creature in the image with the bird famed for wisdom in Greek mythology.

[Figure 141]

[Figure 142]

- [Reflection-round1] <#Think> The user instructed to change the Egyptian pyramid into a Maya stepped pyramid. The output image changed the pyramid's structure, adding steps, but failed to change its cultural origin, as the new pyramid remains distinctly Egyptian in appearance. <#Reflection> Make the pyramid look like it belongs to the Maya civilization by changing its architectural style to feature rounded corners instead of sharp ones, adding decorative carvings, and incorporating a small temple or temple-like structure on top.
- [Reflection-round2] <#Think> The user wants me to change the Egyptian pyramid into a Maya stepped pyramid. The output image successfully transforms the original pyramid into a structure that is distinctly Mayan in style, featuring multiple platforms and detailed carvings, while preserving the desert setting from the original image without any noticeable artifacts. <#Success>

[Thinking] Replace the pyramid with a stepped pyramid from the Maya civilization.

[Figure 143]

[Figure 144]

[Figure 145]

[Instruction] Change the bird's action to flapping its wings and flying high.

[Figure 146]

- [Reflection-round1] <#Think> The edit failed because the chameleon's vibrant green color was not changed to blend into the environment; instead, the background was changed to a blurry brown, which is not a natural camouflage for the chameleon. <#Reflection> Change the chameleon's color to a brownish-green that matches the background.
- [Reflection-round2] <#Think> The user wants to make the chameleon blend into its environment. The original image has a bright, colorful chameleon against a blurry background. The edited image changes the chameleon's color scheme to match the earthy tones of the background, effectively achieving the blending effect without introducing any artifacts. <#Success>

[Thinking] Change the chameleon's colors to match the surrounding environment.

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Instruction] The organism in the image makes changes to blend into the environment.

- [Reflection-round1] <#Think> The user wants me to place the King of Fruits on the table. The King of Fruits refers to the Durian, which was correctly placed on the table in the second image. However, the image shows two durians instead of one, which is a deviation from the singular noun in the instruction. <#Reflection> Remove the left durian so only one remains.
- [Reflection-round2] <#Think> The user instructed to place the 'King of Fruits', which is a golden-orange durian, on the table. However, the output image is completely missing the fruit, and the table remains empty, indicating a complete failure to execute the core command. <#Failed>
- [Reflection-round3] <#Think> The user's instruction was to place the 'King of Fruits' on the table. The resulting image shows a durian, which is a well-known Southeast Asian fruit and the subject of the phrase "King of Fruits". The placement on the table is realistic, and there are no visible artifacts or unwanted changes to the rest of the image. <#Success>

[Thinking] Add a durian fruit on the wooden table.

[Figure 151]

[Figure 152]

[Figure 153]

[Instruction] Place the Southeast Asian speciality known as the 'King of Fruits' on the table.

[Figure 154]

[Figure 155]

- Figure 9. Qualitative multi-round examples illustrating the thinking and the reflection progressively correct and improve the quality of the generate results.

###### UniWorld-Qwen

###### Input image Flux-Kontext-dev Step1X-Edit v1.1 Qwen-Image-Edit BAGEL BAGEL-thinking ReasonEdit-S ReasonEdit-Q

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Instruction] What the fruit looks like when ripe in the picture.

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Instruction] Add a watermelon

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Instruction] Modify to appear more desertified.

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Instruction] Change the animal in the image to the one with the longest neck in the world.

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Instruction] Use the object in the image as the main visual focus. Place the light blue handbag slightly below the center of the frame. The background should feature a dawn pink gradient sky with a soft-focus city silhouette. The ground is made of a smooth mirror material with a slightly blurred reflection, creating an airy and ethereal composition. Include minimalistic white feather elements slowly falling in the sky to enhance the dreamlike feeling. The overall color scheme should be unified in three colors: blue, white, and pink, with a soft, modern texture and a fantasy atmosphere, suitable for urban women's affordable luxury brand poster style. Position the title at the top center of the image, using a thin serif font, with the copy reading: "Lighter than the wind, more resolute than you."

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Instruction] Complete the pants.

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Instruction] Modify the speed limit to show a non-speeding scenario.

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Instruction] Correct the unreasonable parts of the arm in the image.

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Instruction] Without altering or beautifying anything else, just shape my eyebrows to suit me.

###### Figure 10. More qualitative comparison of our method and state-of-the-art approaches.

###### Original Image Instruction Rensoning

Correct the violations in the image.

[Thinking] Remove the cigarette from the man's mouth.

[Reflection] <#Think> The user wants me to correct the image. The instruction was "Correct the violations in the image". By comparing the two images, I can see that the cigarette has been removed, which fulfills the correction request. The rest of the image remains consistent without any noticeable artifacts. <#Success>

[Figure 237]

[Figure 238]

[Cause of failure] The failure stems from incomplete planning within the Thinking process. To correct the violation in the image, there are two viable solutions: either (1) modify the environment (e.g., remove or alter the "No Smoking" sign) or (2) modify the subject's behavior (e.g., change the person to a non-smoking state). The model selects the second approach but fails to formulate a comprehensive plan. It removes the cigarette but neglects to adjust the corresponding hand gesture, resulting in a semantic inconsistency where the subject retains a smoking pose despite the object's absence.

Pour a little water onto this pile of dry ice.

[Thinking] Add a small puddle of water beside the dry ice pile.

[Reflection] <#Think> The user‘s instruction was to pour a little water onto the pile of dry ice. The output image shows the dry ice now sitting on a wet surface with a visible puddle of water and associated vapor, perfectly matching the request without any noticeable artifacts or unintended changes. <#Success>

[Figure 239]

[Figure 240]

[Cause of failure] The failure stems from insufficient physical knowledge regarding material interactions. The model merely adds water beside the dry ice, treating it as a generic solid, and overlooks the specific physical reaction known as sublimation. Consequently, it fails to generate the characteristic thick fog that naturally occurs when water contacts dry ice, resulting in a scientifically inaccurate editing outcome.

Change to one apple.

[Thinking] Remove all but one apple.

[Reflection-round1] <#Think> The user instructed to change the image to show one apple. The output image removed all the apples and only kept the plate, which fails to follow the core instruction. <#Failed>

[Figure 241]

[Figure 242]

[Figure 243]

- [Reflection-round2] <#Think> The user wants me to change the image to show one apple. However, the resulting image is completely empty, showing only the transparent plate without any apples, which directly contradicts the core instruction. <#Failed>
- [Reflection-round3] <#Think> The user instructed to change the image to have one apple. The resulting image removed all apples and replaced them with an empty plate, which completely fails to fulfill the core requirement of the instruction. <#Failed>

[Figure 244]

[Cause of failure] The failure stems from the insufficient capability of the generation model. Across three attempts, the model failed to execute the selective retention of a single apple, resulting in the removal of all objects despite the correct planning.

Figure 11. Failure cases.

