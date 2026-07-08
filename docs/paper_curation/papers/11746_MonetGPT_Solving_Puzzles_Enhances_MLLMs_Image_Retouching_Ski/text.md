arXiv:2505.06176v1[cs.GR]9May2025

MonetGPT: Solving Puzzles Enhances MLLMs’ Image Retouching Skills

NILADRI SHEKHAR DUTT, University College London, UK DUYGU CEYLAN, Adobe Research, UK NILOY J. MITRA, University College London, Adobe Research, UK

[Figure 1]

[Figure 2]

[Figure 3]

- Fig. 1. We present MonetGPT, an image operation-aware multimodal large language model (MLLM), that provides automatic suggestions for image retouching. Given a photograph (left), MonetGPT analyzes it to identify a set of issues and possible adjustments to fix them. The solution steps are then translated to a set of procedural operations, along with respective parameter settings, drawing from a given library of operations, which occurs in three stages. (Visual puzzles, on which we train the MLLM, are not shown here.)

Retouching is an essential task in post-manipulation of raw photographs. Generative editing, guided by text or strokes, provides a new tool accessible to users but can easily change the identity of the original objects in unacceptable and unpredictable ways. In contrast, although traditional procedural edits, as commonly supported by photoediting tools (e.g., Gimp, Lightroom), are conservative, they are still preferred by professionals. Unfortunately, professional quality retouching involves many individual procedural editing operations that is challenging to plan for most novices. In this paper, we ask if a multimodal large language model (MLLM) can be taught to critique raw photographs, suggest suitable remedies, and finally realize them with a given set of pre-authored procedural image operations. We demonstrate that MLLMs can be first made aware of the underlying image processing operations, by training them to solve specially-designed visual puzzles. Subsequently, such an operation-aware MLLM can both plan and propose edit sequences. To facilitate training, given a set of expert-edited photos, we synthesize a reasoning dataset by procedurally manipulating the expert edits and then grounding a pretrained LLM on the visual adjustments, to synthesize reasoning for finetuning. The proposed retouching operations are, by construction, understandable by the users, preserve object details and resolution, and can be optionally overridden. We evaluate our setup on a variety of test examples and show advantages, in terms of explainability and identity preservation, over existing generative and other procedural alternatives. Code, data, models, and supplementary results can be found via our project website at https://monetgpt.github.io.

CCS Concepts: • Computing methodologies → Image processing; Natural language processing; Machine learning; Computer vision; • Humancentered computing → Interaction design.

Authors’ Contact Information: Niladri Shekhar Dutt, niladri.dutt.22@ucl.ac.uk, University College London, UK; Duygu Ceylan, ceylan@adobe.com, Adobe Research, UK; Niloy J. Mitra, n.mitra@ucl.ac.uk, University College London, Adobe Research, UK.

This work is licensed under a Creative Commons Attribution 4.0 International License. © 2025 Copyright held by the owner/author(s). ACM 1557-7368/2025/8-ARTxx https://doi.org/10.1145/3730926

Additional Key Words and Phrases: LLMs, image retouching, skill learning, interpretability, procedural edits, edit sequence, regression, copilot, agent

# ACM Reference Format:

Niladri Shekhar Dutt, Duygu Ceylan, and Niloy J. Mitra. 2025. MonetGPT: Solving Puzzles Enhances MLLMs’ Image Retouching Skills. ACM Trans. Graph. 44, 4, Article xx (August 2025), 12 pages. https://doi.org/10.1145/ 3730926

1 Introduction

We regularly retouch captured images to improve their presentation. For example, users adjust contrast and brightness, manipulate exposure, or correct color profiles. Such adjustments, often consisting of a series of procedural operations, are preferred by professional users as the operations are non-destructive, and can be applied at different resolutions. Further, such edits are interpretable, supported by many established image manipulation tools, and, unlike generative editing counterparts, better preserve the identity of the source content.

Unfortunately, effectively using procedural edits is difficult and beyond the reach of most novices. There are two main challenges. First, users should learn how to apply the individual operations using the tool – referred to as the command knowledge. Second, they have to plan, based on the source image, which set of operations to use and propose appropriate parameter values for the chosen operations – referred to as the strategic knowledge. While the first one can be lowered with practice on a given toolset (e.g., Gimp), the second one is often hard to overcome as planning, using a library of operations, is open-ended and inherently more difficult.

In a breakthrough effort, the Exposure framework [Hu et al. 2018] demonstrated that it is possible to learn, using a reinforcement learning setup, sequences of procedural edits directly from artist retouching examples. However, the effectiveness of such an approach is limited by the paucity of expert edits available for training. In this paper, we ask if it is instead possible to start from the knowledge

[Figure 4]

- Fig. 2. Generative tools, for example instructPix2Pix [Brooks et al. 2023] or MGIE [Fu et al. 2024], produce impressive image enhancements but can result in identity loss (e.g., faces, hands, objects) and are harder to override by users. Procedural approaches are more controllable as they restrict operations to a given set of user-prescribed operation library, and can be overridden or applied in parts. Current MLLMs (bottom-left: e.g., GPT

- o1 applied using a library of operations, presented as doc-strings), do not have a good internal model of image operations, and perform worse than
- our operation-aware variant (bottom-right). See Section 5 for evaluation.

embedded in frontier models, trained on significantly larger and diverse datasets, and adapt them to our specialized retouching task with limited data from expert artists.

We present MonetGPT for procedural image retouching. MonetGPT introduces an effective fine-tuning strategy to adapt multimodal large language models (MLLMs), even on a limited retouching dataset [Bychkovsky et al. 2011; Liang et al. 2021]. Once fine-tuned, the MLLM can identify problems in a source image, plan a sequence of fixes to improve the image, and finally translate the fixes using a given library of procedural edits. For example, as in Figure 1, MonetGPT suggests an edit sequence to retouch the input raw photograph. In addition to proposing a sequence of edit operations with associated parameters, our method also provides explanation in the form of what issue each of the proposed adjustments attempts to solve. Further, an user can ignore or override any of the proposed changes and run the rest of the procedure (e.g., experts sometimes violate photographic guidance to highlight an aspect/subject of the image).

We found current MLLMs to be versatile but not quite powerful in proposing meaningful retouchings on raw photographs (see Figure 2). Directly fine-tuning such an MLLM on artists’ edit sequences only partially improves the results (see Section 5). We attribute this to MLLM’s lack of understanding of what each of the image editing operations entails. As humans, we build mental models of these operations based on our experience (e.g., what does increasing the

brightness slider do to images?). We mimic the same skill learning [Lövdén et al. 2020] for MLLMs – to this end, we design specific visual puzzles, involving the given imaging operations, and train an MLLM on solving these puzzles. As a result the MLLM becomes operation-aware and is then capable of planning high-quality image retouching sequences. We demonstrate how to do so in an unpaired training setup with limited amount of data from artists.

We evaluated MonetGPT on a diverse set of input images, comparing its retouching results against generative editing methods (LLMbased editing, InstructPix2Pix [Brooks et al. 2023], and MGIE [Fu et al. 2024]), procedural editing techniques from prior work (Exposure [Hu et al. 2018]), custom-designed variations of MLLMs integrated with procedural pipelines (Gemini 2.0 with chain-of-thought, fine-tuned variants), and commercial software (GooglePhoto AutoEnhance). We evaluated the resultant edits extensively through qualitative analysis by expert reviewers and novice users. Please refer to supplemental for more evaluation. In summary, we

- (i) propose the first framework for MLLM-guided interpretable procedural image retouching, enabling non-destructive editing on high-resolution 16-bit images.,
- (ii) train MLLMs on strategically-designed visual puzzles to make them operation-and aesthetics-aware, and use to plan our edit sequences with associated parameter estimation;
- (iii) demonstrate, via extensive evaluation and comparisons, the feasibility and benefits of operation-aware MLLM-guided procedural editing workflows over chain-of-thought reasoning MLLMs and generative alternatives.

Refer to our webpage at monetgpt.github.io for code and supplementary results.

2 Related Work 2.1 Image Retouching

Image retouching is an essential and commonly used workflow for post-processing raw images. Many commercial image editing software and web-based tools provide a vast number of filters that can be used for image enhancement. Given the difficulty in choosing which filters to apply in which settings, there has been a lot of research to automate parts of this workflow. A popular thread of research focuses on using pairs of input and edited images to predict the parameters of individual filters, such as global tone adjustment or color adjustment. Earlier work [Bychkovsky et al. 2011; Yan et al. 2014] used machine learning methods such as Gaussian process regression or support vector machines to tackle this task. Later, such methods were replaced with deep neural networks as surrogate functions for various image processing operations [Chen et al. 2017; Liu et al. 2022; Yan et al. 2016]. More recently, approaches that aim to perform image enhancement directly by predicting residual image layers [Kim et al. 2020b] or per-pixel color and channel intensity transforms [Kim et al. 2020a] have been developed. Li et al. [2023] strike a balance between global and per-pixel editing by using a set of piecewise linear curves to retouch different spatial regions of an input image. However, in such approaches, it is not possible to further edit or control the results since the edits are not linked to specific image processing operations.

Closer to our problem setup, several works have looked into how to best select the type and parameters of a predefined set of operations to enhance image quality and aesthetics. Notably, Exposure [Hu et al. 2018] presents an RL-based framework where image retouching is cast as a planning problem: a discriminator that classifies an image as retouched or not is used to provide the reward function for the RL agent. In a similar setup, Shi et al. [2021] propose to edit an image given a text prompt by generating a sequence of image editing operations and the corresponding parameters. The operations are chosen from a predefined stack of differentiable filters and a sequence modeler (i.e., LSTM decoder) is used to guide the planning. Fischer et al. [2020] also propose a framework to optimize the parameters of a set of differentiable neural image filters using a neural image assessor to evaluate the image quality. Instead of training a planning algorithm from scratch, we explore the power of pre-trained multimodal large language models to aid learning from a limited set of expert edits.

- 2.2 Generative Edits

Over the past few years, there has been a transformational breakthrough in conditional and unconditional image generation, first with the use of GANs [Goodfellow et al. 2014] and most recently diffusion-based image generators [Rombach et al. 2021]. Specifically, with the success of text-to-image generators, many works have explored editing via text prompts [Brooks et al. 2022; Cao et al. 2023; Hertz et al. 2022], spatial guidance [Zhang et al. 2023], and other user interactions [Mou et al. 2023]. As the large language models advance, editing paradigms [Pan et al. 2023; Peng et al. 2023; Santos et al. 2024; Xiao et al. 2024] that leverage the model’s linguistic reasoning capabilities emerge. While being very powerful, such methods re-generate every pixel in the edited image and hence often struggle with identity preservation (see Figure 2).

- 2.3 Reusing LLMs for Graphics Tasks

We are witnessing a revolution in the domain of (multi-modal) large language models with many successful examples [Achiam et al. 2023; Jiang et al. 2023; Touvron et al. 2023]. These models specialize on various tasks such layout planning [Aguina-Kang et al. 2024; Feng et al. 2024; Littlefair et al. 2025; Yang et al. 2023], 3D editing [Huang et al. 2024], and embodied interaction [Qi et al. 2024]. In the context of image editing, ClickDiffusion [Helbling et al. 2024] first generates a new layout given a text prompt from which conditional image generation is performed. Chain-of-thought (CoT) [Wei et al. 2024] is used at inference time to better leverage the priors of the LLM to create the new layouts at inference time. Fu et al. [2024] leverage an MLLM not only to obtain expressive instructions but additional visual guidance to condition a diffusion-based generator, which is fine-tuned (see Figure 2). Ours differs from these methods since we do not cast image editing as a single, closed-box generation but represent it as applying a set of predefined image filters and utilize an MLLM to predict the sequence and parameters of such filters.

Most recently, several work [Hang et al. 2024; Zhenyu et al. 2024] explore a framework where an MLLM is used as an agent to plan a sequence of editing operations to be applied given a source image and a target description. The planning is guided by a feedback

mechanism in an iterative workflow. In contrast, we fine-tune the MLLM which is then queried at test time directly to generate the editing operations. In a concurrent effort, ComfyGen [Gal et al. 2024] finetunes an LLM to choose a ComfyUI workflow from a given set of flows to accomplish a desired generative image editing task. To the best of our knowledge, our method is the first where an MLLM is fine-tuned to reason about a set of procedural image editing operations and their parameters.

3 Design Considerations

Our goal is to aesthetically retouch any given image 𝐼𝑆 using a combination of operations drawn from a library L of pre-defined procedural filters. We author a library (see supplemental for details) with three types/stages of operations: (i) lighting adjustments (e.g., blacks, contrast, exposure, highlights, whites, shadows); (ii) colorandtemperaturesadjustments (e.g.,saturation,temperature,

tint); and (iii) color-specific adjustments (e.g., hue, luminance, saturation) for eight different colors. We execute the adjustments in the three stages, as listed above. The ability to fine-tune colors across eight distinct and precise ranges allows us to make meaningful, localized adjustments, providing a practical alternative to using masks for certain editing challenges. We assume that each of the listed functions in the library can be executed by specifying a source image and function parameters (e.g., can be coded a C++, Python, or even neural blocks).

Designing Visual Puzzles. Although MLLMs have rich global priors, they perform poorly when queried directly to generate procedural image retouching operations (see Figure 2 and Section 5). Similarly, fine-tuning them to directly generate the parameters of a set of operations given a source image leads to severe over-fitting due to scarcity of training data (see Section 5). Instead, MonetGPT proposes to make the MLLMs more (image) operation-aware, by designing specific visual puzzles with different goals and training them using suitable datasets. Specifically, solving these puzzles helps an MLLM develop the following knowledge:

- (a) what each image operation does, i.e., relations among a source image, a single (image) operation, and the resultant image;
- (b) how much to apply any image operation, i.e., what is an aesthetic application of an operation on any source/intermediate image;
- (c) how far we are from an ‘optimal’ image, i.e., building an internal model of a desirable retouched image; and finally,
- (d) how to plan a series of operations to get to an ‘optimal’ image, i.e., learn how to create an editing plan. With this motivation, we design three puzzles (described in Section 4): Puzzle A helps develop the skill (a); Puzzle B helps develop skills (b,c), and Puzzle C helps develop skill (d).

Our visual puzzles serve as proxy loss functions for the various image adjustment operations when fine-tuning the MLLM.

Generating a Reasoning Dataset. Once we design the visual puzzles, we use a pretrained MLLM (we use Gemini 2.0 flash) to additionally generate a reasoning solution corresponding to each puzzle. This step allows us to leverage pretrained MLLMs to reason about each edit operation by explaining why a particular operation was

used and what problem it fixes by grounding it on the actual adjustments and basing its reasoning on the visual changes to prevent hallucination. (See Section 5 for a comparison with a baseline with Gemini2.0 with our library L.) We shall use this to build a dataset to fine-tune a MLLM to acquire this reasoning when the actual adjustments are not supplied. Next, we describe our method.

- 4 Method

MonetGPT is a novel framework that leverages MLLMs for advanced reasoning to facilitate procedural image retouching. Pretrained MLLMs lack the domain knowledge required to comprehend underlying image retouching operations and their associated adjustment values. To address this limitation, we design a set of puzzles specifically targeted at bridging these knowledge gaps. We discover that by solving these puzzles, the MLLM can become an agent with expert-level domain knowledge, capable of retouching images effectively. In the following, we first introduce the visual puzzles we design to fine-tune the MLLM, and then discuss how we utilize the fine-tuned model at inference time. Finally, we provide details of the procedural image filters we use and how we execute

- them in practice.

- 4.1 Puzzle A: Gaining Understanding of Individual Operations

As the first step, the MLLM must visually understand the impact of each individual operation on an image and how this impact varies across different adjustment levels. To do this, we randomly sample an operation O ∈ L and an associated adjustment value 𝑉 from a predefined library of image retouching operations. This operation is

- then applied to a source image 𝐼𝑆 to produce an edited image 𝐼𝐸. The 𝐼𝑆 and 𝐼𝐸 images are stitched together and presented to an MLLM
- to identify the operation and adjustment value, i.e., given the image pairs the MLLM should predict the operation and its corresponding change amount.

We discover that directly querying a pretrained MLLM to identify the operation and adjustment value produces poor results (presented as ‘Gemini 2.0+CoT+library’ option in Section 5). However, when provided with the actual operation and adjustment value and grounded on the observed visual changes, the MLLM generates detailed and convincing reasoning 𝑅 that accurately explains in rich textual descriptions, how the effect of (O,𝑉) on 𝐼𝑆 results in 𝐼𝐸. Thus, given pairs (𝐼𝑆,𝐼𝐸), we get associated 𝑅 as well as (single) edit (O,𝑉), see Figure 3. Grounding the MLLM with the true adjustment value ensures its reasoning aligns with the specific operation O, avoiding unrelated or hypothetical explanations.

We use the extracted reasoning during the training phase to teach the MLLM in a supervised fashion to identify O and regress𝑉. (Note that we synthetically generate this training data as we have access to the image processing filters in L.) Instead of directly inferring the operation and adjustment value, the MLLM needs to explain the reasoning by eliciting visual differences in the image caused by O in tandem with the degree of adjustment. By elaborating on the visual changes in the edited image, the MLLM encodes these visual nuances within its text representation, enabling it to effectively learn the effects of various operations in library L. We quantized

[Figure 5]

- Fig. 3. Puzzle A. This puzzle intends to teach what any single operation O ∈ L, along with its value 𝑉, does to a source image 𝐼𝑆 to produce an edited image 𝐼𝐸. The visual puzzle being, given an ordered pair (𝐼𝑆,𝐼𝐸), one has to predict (O,𝑉 ). Using (𝐼𝑆,𝐼𝐸), we also generate corresponding reasoning 𝑅.

(normalized) parameter values (see also [Wang et al. 2024b]) as they are easier to tokenize and use with MLLMs.

[Figure 6]

- Fig. 4. Puzzle B. This puzzle intends to teach about image aesthetics under any single operation O ∈ L. The visual puzzle being, given a set of randomly

ordered images (𝐼𝐸,𝐼𝑉1,𝐼𝑉2,𝐼𝑉3,𝐼𝑉4 ), generated from an expert-edited final image 𝐼𝑋 by applying operation O with perturbed values {𝑉𝑖 }, one has to order the set of images, based on low to high values 𝑉𝑖, as well as identify the optimal image 𝐼𝑋 along with the perturbations values to go back to 𝐼𝑋 from each image. Note that we implicitly assume that any perturbation of an expert-edited image results in a worse image. Using the image set and the operation, we also generate corresponding reasoning 𝑅.

- 4.2 Puzzle B: Understanding Image Aesthetics

Understanding image aesthetics is essential for defining how an enhanced version of an image should appear. An MLLM must acquire the capability to visually identify the optimal appearance of an image when an operation is adjusted to its ideal parameter value. To achieve this, we design a second puzzle involving four random variations (𝐼𝑉 ) of adjustments for a sampled operation, O, applied to an image edited by an expert, 𝐼𝑋. Note that we assume that any sufficiently large adjustment made to 𝐼𝑋 degrades the image and results in a suboptimal edit.

We construct the puzzles by stitching 𝐼𝑋 alongside the four adjusted images in a random order. The MLLM is first tasked with ordering all five images from the lowest to highest adjustment values. The adjustment range for the operations is defined on a perceptually linear scale, ranging from [−100, +100]. Once ordered, the MLLM must then identify the image with the optimal level of O (i.e., identify 𝐼𝑋) and justify its reasoning. Additionally, it must determine the adjustment level required to transform a randomly selected image from 𝐼𝑉 into the optimal image, 𝐼𝑋. See Figure 4. Note that this process implicitly assumes that the operation is invertible.

Similar to Section 4.1, instead of providing a direct one-sentence answer to the puzzle, we query a pretrained model with the correct answers and task it with generating reasoning grounded in the observed visual changes. By training the MLLM to solve this puzzle and detail its reasoning, it acquires an intrinsic ability to recognize the visual characteristics of an optimally adjusted image and estimate the adjustment value required to transform any source image to the optimal one. This aesthetic understanding is critical when planning edits that involve multiple operations and adjustments.

- 4.3 Puzzle C: Generating a Plan for Image Retouching

MLLMs have demonstrated significant success in solving complex tasks, such as mathematical proofs, by decomposing them into manageable steps [Wei et al. 2024]. However, the abstract and subjective nature of image editing presents a large state space, making it challenging for an MLLM to directly predict multiple operations along with their corresponding adjustment values. While existing MLLMs can suggest basic adjustments, such as modifying exposure or saturation, they struggle to produce comprehensive editing plans with precise adjustments.

To bridge this gap, we design a third puzzle, where we aim to enable MLLMs to generate expert-level retouching plans, including suitable operations and adjustment values, to enhance a source image 𝐼𝑆 into its expertly retouched counterpart 𝐼𝑋. To achieve this, we modify expert-edited images to create poorer-quality variants, which serve as the 𝐼𝑆 images requiring enhancement. Note that, once again, we only make use of expert-edited images 𝐼𝑋, while we use synthetic edit plans by procedurally perturbing them. However, unlike previous puzzles which focused on learning operations individually, we alter multiple parameters within specific categories: (i) Lighting adjustments, (ii) Color and temperature adjustments, and (iii) Color-specific adjustments.

For the planning stage, our design choice is motivated by the following considerations: (i) Inversion feasibility: By modifying a limited set of parameters within a category, the operations remain

invertible, enabling the reconstructed image to closely match 𝐼𝑋.

- (ii) Reduced complexity: Generating a comprehensive plan involving numerous operations simultaneously is inherently challenging. Dividing the process into sequential stages—starting with lighting, followed by color-temperature adjustments, and finally color-specific fine-tuning—simplifies the task and aligns with expert workflows.
- (iii) Reasoning clarity: Finally, similar to the first two puzzles, we want to generate reasoning behind the edits by querying a pretrained MLLM to analyze visual changes corresponding to specific adjustments. When many operations are applied simultaneously, it becomes challenging to disentangle which operation is responsible for a given visual change. Hence, we split our task into stages.

We synthetically generate a dataset of 𝐼𝑆 images derived from expert edits 𝐼𝑋. For each 𝐼𝑆-𝐼𝑋 pair, we task the MLLM with generating reasoning for every adjustment. This involves identifying the adjustment to be applied, noting the degree of change, and linking it to the corresponding visual issue and solution. Specifically, we structure the reasoning into a triplet <Adjustment, Issue, Solution> for each operation, as described next (see Figure 5).

Adjustment: The operation and its degree of adjustment. Issue: The visual problem addressed by the adjustment by referencing specific elements in the image. Solution: The visual improvement achieved through the adjustment.

[Figure 7]

Fig. 5. Puzzle C. This puzzle intends to teach how to generate a retouching plan. The visual puzzle being, given an ordered pair (𝐼𝑆,𝐼𝐸), one has to come up with a retouching plan {(O𝑖,𝑉𝑖 )} listing the operations, from L, along with the associated parameter values. Using the image retouching sequence and the operations, we also generate corresponding reasoning 𝑅, in the form of <Adjustment, Issue, Solution> for each operation.

We reformatted the detailed triplets in the style of instructions to generate a plan P. Note, during training the MLLM only has access to 𝐼𝑆 and it must generate a plan, P, that will lead to𝐼𝑋. When fine-tuned on this dataset, the MLLM is capable of generating comprehensive plans during inference, without hallucination. Additionally, the MLLM uses chain-of-thought (CoT) reasoning to regress from highlevel reasoning to precise parameter values, referencing a predefined legend that maps adjustments to numerical ranges. This structured approach ensures that the MLLM produces meaningful insights and solutions aligned with expert edits, instead of independently guessing adjustment plans.

An important aspect of this puzzle involves teaching the MLLM to recognize when no further adjustments are needed for a particular stage. This prevents unnecessary edits that could degrade a well-adjusted image. To train this skill, we introduce an additional challenge: the MLLM when queried to generate an editing plan for a given stage, it must instead justify why no further edits are required. As in earlier puzzles, synthetic reasoning is generated to train the MLLM by assuming that any further modification to an expert-edited image 𝐼𝑋 for a specific category of operations would result in a degradation of quality.

- 4.4 Inference: Reasoning as a Pathway for Regression

Once the MLLM is trained on the three puzzles, we use it to generate editing plans by leveraging reasoning. Reasoning bridges the gap between adjustment values and the underlying intent behind each editing operation. This effectively induces a pathway for the fine-tuned MLLM M to regress precise adjustment values regress from high-level reasoning. Given 𝐼𝑆, our goal is to predict a set of adjustment operations and corresponding values, 𝐴. We condition an MLLM M on 𝐼𝑆 to first generate an editing plan P as,

M(·|𝐼𝑆) := P. (1)

We then drive M to generate the final adjustment values 𝐴 based on P,

M(·|P,𝐼𝑆) := 𝐴. (2)

As noted in Section 4.3, we divide our plan generation into three separate stages. Hence, we apply our procedural pipeline on the source image to get an edited image, which we feed back to infer the next stage of operations, as dictated by the inferred plan. See Figure 1 for an example.

- 4.5 Authoring a Library with a Reduced Parameter Space

Existing image enhancement software with Python bindings, such as GIMP, are overly complicated to code for. For instance, GIMP’s procedural database (PDB) often requires multiple function calls and the specification of numerous parameter values to execute a single adjustment operation, unnecessarily expanding the parameter space. Other Python-based image enhancement libraries, such as OpenCV and Pillow, provide a more straightforward interface but offer a very limited range of operations.

To address these limitations and making use of open source options, we developed a Python library of image adjustment operations that simplifies the process of defining and executing adjustments.

The library uses modular functions, where each adjustment operation is controlled by a single master parameter. Sub-parameters are either fixed or dynamically derived from the master parameter, significantly reducing the complexity of the parameter space. The library provides an approximately comparable subset of tools to those available in platforms like Google Photos and Lightroom.

More importantly, the fine-tuned MLLM possesses a deep visual understanding of the operations within the library, enabling it to generate detailed plans with adjustment values that accurately reflect how each modification impacts the image.

Leveraging this understanding, the fine-tuned MLLM acts as an agent that generates editing plans by generating operations and corresponding adjustment values as output in a structured JSON format, eliminating the need to write code and instead solely focus on capturing the visual impact of each adjustment. Our library directly processes this JSON file to apply the adjustments seamlessly. Additionally, we design the parameter values to follow a mostly perceptually linear scale, ranging from [-100,+100] to ensure a consistent control over the adjustments. Our operations are non-destructive by construction and can operate on high-resolution 16-bit images compared to other generative solutions.

5 Evaluation

ImplementationDetails.We fine-tuneQwen-VL-7B-Instruct [Wang

et al. 2024a], a 7-billion-parameter MLLM (M), using DoRA adapters [Liu et al. 2024]. We fine-tune using the llamafactory [Zheng et al. 2024] framework. We configure DoRAZ[Liu et al. 2024] with a dropout rate of 0.2, an adapter rank of 256, and an alpha rank of 512. A learning rate of 1e−4 is employed with cosine scheduling, and the model is fine-tuned for a single epoch, which takes approximately 8 hours on an H100 GPU; direct regression baseline takes 2.5 hours. The training dataset comprises synthetically generated puzzles paired with corresponding reasoning, generated using Gemini 2.0 Flash Experimental [Team 2024]. We created the puzzles by sampling expert edited images from PPR10K [Liang et al. 2021]. As detailed in Section 4, we apply random adjustments to each image, yielding a synthetic puzzle dataset with approximately 7k samples for puzzle A, 5k for puzzle B, and 13k for puzzle C. Our library includes a total of 33 operations, implemented in Python either from scratch or by extending existing libraries such as OpenCV and Pillow (see supplemental). To ensure the accuracy of our library, we verify that applying the expert adjustment values from the PPR10K dataset to the source images produces results that closely match the desired targets.

Inference time. For inference, we used an RTX 4090. For each retouching, ours takes 25 sec for full staged pipeline execution, while direct regression takes 10 sec, and Exposure takes about 2 sec.

Dataset. We evaluate our method on a variety of images curated from the PPR10K [Liang et al. 2021] and Adobe5k [Bychkovsky et al. 2011] datasets, which provide access to source images as well as expert edits. Recall that our data generation does not make use of the pairing information. For testing, we select images for which the expert versions have not been seen during our training.

Baselines. We perform extensive comparisons with the following methods. (i) Exposure [Hu et al. 2018], which is an RL-based

[Figure 8]

## Fig. 6. Each row depicts an input image and a retouched version produced by each of the baselines. While generative baselines (MGIE) struggle to preserve the identity (last row), Exposure or Gemini sometimes generate too bright or dark results (third row). Direct regression with an MLLM fails to make sufficient enhancements. Our method is capable of providing balanced and aesthetic enhancements.

[Figure 9]

[Figure 10]

[Figure 11]

"Blacks": -34,"Clarity": 5, "Contrast": 5,

"Exposure": 13, "Highlights": -63, "LuminanceAdjustmentYellow": 9,

"Blacks": 15,"Contrast": 15,"Highlights": -15,

"Shadows": 15, "Whites": 15, "Exposure": 3, "Temperature": -8, "Tint": 12, "Saturation": 10, "LuminanceAdjustmentGreen": -15, "SaturationAdjustmentGreen": -15

"LuminanceAdjustmentRed": 3, "LuminanceAdjustmentOrange": 25,

"Saturation": -5, "Whites": -37,

"SaturationAdjustmentYellow": -5, "SaturationAdjustmentOrange": -15,

"Shadows": 10, "Sharpness": -15, "Temperature": -1, "Tint": 1

[Figure 12]

[Figure 13]

[Figure 14]

"Blacks": -34,"Clarity": 5, "Contrast": 5,

"Exposure": 13, "Highlights": -63, "LuminanceAdjustmentYellow": 9,

"Blacks": -15,"Contrast": 10,"Highlights": 10,

"Shadows": -15, "Whites": -15, "Exposure": -3, "Temperature": -3, "Tint": -18, "Saturation": 15, "LuminanceAdjustmentYellow": -15, "SaturationAdjustmentYellow": -10

"LuminanceAdjustmentRed": 3, "LuminanceAdjustmentOrange": 25,

"Saturation": -5, "Whites": -37,

"SaturationAdjustmentYellow": -5, "SaturationAdjustmentOrange": -15,

"Shadows": 10, "Sharpness": -15, "Temperature": -1, "Tint": 1

[Figure 15]

[Figure 16]

[Figure 17]

"Blacks": 30, "Contrast": 10,"Highlights": 10, "Shadows": -10,"Whites": -15, "Exposure": 15,"Temperature": 6, "Tint": 3,"Saturation": 15, "LuminanceAdjustmentOrange": -15, "HueAdjustmentOrange": -5

"Blacks": -34,"Clarity": 5, "Contrast": 5,

"Exposure": 11, "Highlights": -63, "LuminanceAdjustmentYellow": 9,

"LuminanceAdjustmentRed": 3, "LuminanceAdjustmentOrange": 25,

"Saturation": -5, "Whites": -37,

"SaturationAdjustmentYellow": -5, "SaturationAdjustmentOrange": -15,

"Shadows": 10, "Sharpness": -15, "Temperature": -1, "Tint": 1

input image MLLM regression baseline

MonetGPT (ours)

[Figure 18]

input image

[Figure 19]

balanced

"Blacks": 20, "Contrast": 10,

"Highlights": 10, "Shadows": 10, "Whites": 15, "Exposure": 18, "Temperature": -6,

"Tint": -15, "Saturation": 15,

"SaturationAdjustmentOrange": -15, "LuminanceAdjustmentOrange": 10

[Figure 20]

vibrantretro

"Contrast": -25, "Highlights": -25,

"Shadows": 20, "Whites": 5, "Exposure": 0, "Temperature": 12, "Saturation": 10, "SaturationAdjustmentRed": 50, "HueAdjustmentOrange": 0, "LuminanceAdjustmentOrange": 10

[Figure 21]

"Blacks": 15, "Contrast": 15, "Highlights": 15, "Shadows": 15,

"Whites": 15, "Exposure": 15, "Temperature": -6,

"Tint": 18, "Saturation": 40,

"SaturationAdjustmentRed": 10, "LuminanceAdjustmentRed": 15

- Fig. 7. (Left) MonetGPT responds to subtle changes in the input image, producing different edit plans. In contrast, our regression baseline largely misses the subtle variations in the input and proposes almost identical retouching plans. (Right) MonetGPT can produce different edit plans based on style tags, here we show retro, balanced, and vibrant producing different edit plans, resulting different retouched results.

framework to suggest a sequence of operations and their parameters to enhance an image; (ii) Unpaired Image Enhancement [Kosugi and Yamasaki 2020], similar to Exposure, with an editing interface; (iii) RSFNet [Ouyang et al. 2023], which generates pixel-level attention maps using region-specific filters but it requires paired retouched images for training that may be hard to obtain; (iv) MGIE [Fu et al. 2024] utilizes an MLLM to derive expressive instructions and additional guidance to enable instruction guided image editing. In our experiments, we use the fixed instruction ‘Enhance the image like a professional image editing expert using Lightroom’; (v) Gemini+library, where we use Gemini 2.0 at inference time with chainof-thought (CoT) reasoning along with our library L, but with no additional training or fine-tuning. In particular, we interact with Gemini in three stages. Given a source image, we first prompt it to write a detailed plan of the adjustments to be made without providing any operation names or parameters. This effectively enforces the MLLM first to reason about what aspects of the image require enhancement. We then provide three categories of operations, i.e., lighting adjustments, color and temperature adjustments, and colorspecific operations, similar to ours. We prompt the MLLM to choose which operations need to be applied along with a reasoning. Finally, we ask the MLLM to provide the parameters of the selected operations, which we then translate and execute using our library. We refer to the supplementary material for more details about each step of this prompting; (vi) InstructP2P [Brooks et al. 2023]; and finally (vii) Google Photo, where we also use the autoenhance feature available as part of Google Photos1 as a black box commercial alternative.

Finally, to ablate our method, we also present regression that refer to an MLLM guided variant where we directly fine-tune the MLLM

1https://photos.google.com/

(same setup as ours) to directly regress the parameters of a set of image editing operations. We perform this fine-tuning using the PPR10K dataset where we have access to source and edited image pairs along with corresponding adjustment operations and values. Following prior works such as Exposure [Hu et al. 2018], we train it to emulate a single expert’s adjustments to minimize ambiguity.

Table 1. Evaluation for generalization on the Adobe5k dataset. All methods are trained on the PPR10k dataset except MGIE (requires very large data) and GooglePhotos (closed source and proprietary).

Method SSIM ↑ LPIPS ↓ PSNR ↑ Histogram ↑ Exposure [Hu et al. 2018] 0.63 0.14 15.12 47.21

Unpaired [Kosugi et al. 2020] 0.83 0.12 21.73 83.98 RSFNet [Ouyang et al. 2023] 0.88 0.08 21.85 80.26 InstructP2P [Brooks et al. 2023] 0.61 0.22 16.99 73.90

MGIE [Fu et al. 2024] 0.74 0.08 22.94 79.95 GeminiCoT [Team 2024] + our library 0.80 0.14 17.83 63.71

GooglePhotos 0.90 0.06 25.86 86.47 MLLM Regression 0.84 0.10 20.89 82.05

Ours 0.90 0.07 23.75 79.50

Quantitative Comparison. We train MonetGPT and baselines on a single expert (expert A) fromPPR10k [Liang et al.2021] and evaluate on 400 images randomly sampled from Adobe5k [Bychkovsky et al. 2011] to test for generalization. We do not re-train MGIE due to its significantly larger data requirements (1M+), nor Google Photos, which is closed source. We also note that training solely on PPR10k makes generalization more challenging due to its limited image variety (mostly portraits). For evaluation, we compute several standard metrics: PSNR measures pixel-wise fidelity, while SSIM and LPIPS [Zhang et al. 2018] serve as perceptual quality indicators. We additionally compute histogram intersections based on Hu et al. [2018] to assess how well the predicted image distributions

- Fig. 8. User preference study comparing our method against baselines Exposure [Hu et al. 2018] (white-box system) and MGIE [Fu et al. 2024] (instructionguided MLLM enhancer). We presented participants with a source image alongside a pair of edited images, where our result was randomly paired with one of the baselines. Responses were collected from users experienced in photo retouching (expert users) and those with varying familiarity (novice users). Participants could also select neither if both edits failed to improve upon the original image. Both expert and novice groups demonstrated a preference for

- our results, as shown.

match expert edits for contrast, luminance, and color saturation (we show the mean of the three histograms in Table 1). The Adobe5k dataset provides edits by five different experts. For each sample and metric, we take the highest score achieved against any of these five experts (except for histogram intersections, which consider all experts). Given the subjective nature of image retouching, matching any expert edit can be considered a desirable outcome for an edited sample. Results in Table 1 show that our method outperforms all open-source baselines on three of the four metrics and achieves performance comparable to the closed-source Google Photos.

Qualitative Comparison. Traditional image comparison metrics often fail to capture the enhancement quality given the subjective nature of image retouching. Therefore, we conduct user studies and expert judgment to evaluate our results. In particular, we first conduct a user study where we select 15 source images and generate enhanced versions using Exposure, Gemini-CoT, and our method. We present each participant the source image as well as a pair of enhanced results and ask them to choose the option that has better aesthetic quality and visual enhancement. They are also provided with the option of choosing neither of the options if they think the source image looks aesthetically more plausible. Each pair is constructed from our result as well as one of the baselines and presented in random order resulting in a total of 200 questions. We collect answers from 15 novice users with varying skills in photo retouching as well as 10 photography experts.

Discussion of results. We provide a sample set of results in Figure 6, and refer to the supplementary material for more examples including detailed explanations for our edits. IP2P and MGIE, being generative methods, are limited in terms of the resolution of images it can generate and often struggle to preserve the content of the source image. Exposure and Gemini CoT+library baselines frequently result in over-exposed, high contrast, too bright or too dark images (e.g., rows 2 and 8). The auto enhance option in Google Photos is a strong baseline, which also suggests parameters of vari-

- ous image editing operations, most likely based on machine learning

methods. Our MLLM regression baseline fails to learn any meaningful signal from the paired dataset of adjustment settings (note: ours

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Fig. 9. Autoregressive editing. The autoregressive nature of MLLMs, combined with our staged editing pipeline, allows the user to edit the plan (P) at any stage. The refined plan is used to determine subsequent parameter values. Moreover, the edited plan P′ enables the MLLM to generate plans for subsequent stages that are consistent with the edits. The bottom image (P′′∗) shows a result following further modifications to the second and third stages, after the first stage was modified to generate P′′.

Fig. 10. For each operation in our library, we show a violin plot of the adjustment values proposed by MonetGPT and the baseline which directly regresses the values on 100 images from the PPR10k dataset. While the baseline overfits and predicts the same values, MonetGPT utilizes the full range of values.

is unpaired), which further confirms the need for using reasoning as a pathway.

Perceptual User Study. Our visual observations are confirmed by our user study on 50 images from– Adobe5k and Reddit. As shown in Figure 8, our method is preferred against all baselines, both by 15 novice users as well as 10 experts. Beyond completing the user study, we also collected verbal feedback from the experts regarding what worked well and areas for potential improvement in the edits. Overall, the experts expressed a strong preference for our approach, while providing constructive suggestions for minor refinements, such as ‘making the skin tones slightly more saturated’ and ‘curve adjustments to better emphasize focal points’.

Effect of Image Operation-Awareness. To highlight the impact of puzzle-solving on image operation-awareness, we test MonetGPT on the same scene with variations in lighting (primarily), as shown in Figure 7-left. Unlike our MLLM regression baseline, MonetGPT generates significantly different retouching plans, tailored to the lighting conditions of each image. In this example, MonetGPT adjusts its outputs, producing distinct adjustments for inputs with balanced lighting, overly bright conditions, or underexposed.

Personalized retouching. Image retouching is subjective and there is no single ‘optimal’ solution, as stylistic preferences vary widely. Our framework is trained primarily to emulate a particular expert’s style, which can introduce subjectivity of a single expert when considering an edit as optimal. However, our framework’s inherent flexibility—enabled by combining MLLMs with a procedural design—allows it to generalize effectively to diverse stylistic requests specified by users. Users can guide the retouching process via natural-language (e.g., requesting increased vibrancy or softer tones), making the model adaptable to individual preferences. As shown in Figure 7-right: we show three different styles, enabled by providing the following additional tags to MonetGPT as prompts: ‘nostalgic retro vibe’, ‘balanced’, and ‘vibrant and punchy colors’, and applied to the same input. We query the LLM to characterize the features of a particular style tag and then add them to our template prompts. The autoregressive nature of MLLMs further allows us the user to edit the plan at any stage and generate subsequent parameter values and plans for the next stages in tandem with the changes made by the user as presented in Figure 9.

6 Conclusion

We demonstrated that MLLMs can learn procedural image retouching operators through training on specially designed visual puzzles. Once trained, the MLLM can critique photographs, propose fixes, and suggest sequences of retouching operations with corresponding parameters. These suggestions can then be translated into executable calls using a function library. We evaluated our approach, MonetGPT, on benchmark datasets, demonstrating advantages over various alternatives. Notably, our method requires no inference-time optimization (e.g., iterative feedback), is compatible with existing MLLMs, and is explainable by design (with detailed reasoning).

Limitations and Future work.

- (i) Currently, MonetGPT supports a limited set of global opera-

tions, excluding crops or regional edits. Supporting object-specific operations could involve using semantic masking networks to presegment images. However, obtaining sufficient artist-edited images linked with regional masks remains a challenge.

- (ii) We trained MonetGPT on a dataset of 8k expert-edited images.

Consequently, artist-specific aesthetic priors or biases are likely reflected in our model. Training on larger, more diverse datasets could mitigate bias, enable learning priors over editing parameters, and potentially facilitate developing aesthetic scoring models.

- (iii) Image retouching is subjective, lacking a single optimal solu-

tion. Our model can make errors, sometimes resulting in artifacts like saturated regions. We expect improved training data and better modeling of perturbation distributions in synthetic augmentation to partially address this. A human-in-the-loop system could further enhance user satisfaction.

- (iv) Our work focused on procedural operations and did not in-

clude generative filters. Future work could explore casting specific generative edits as neurosymbolic modules, allowing MonetGPT to incorporate them. However, this might compromise the procedural interpretability that is a key advantage of our current system.

Acknowledgments

We thank Rishabh Kabra, Ruchira Ray, Tobias Ritschel, Chen Liu, Sylvain Paris, and Morten Hannemose for their comments and suggestions. Niloy was supported by gifts from Adobe and UCL AI Centre.

References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023).

Rio Aguina-Kang, Maxim Gumin, Do Heon Han, Stewart Morris, Seung Jean Yoo, Aditya Ganeshan, R Kenny Jones, Qiuhong Anna Wei, Kailiang Fu, and Daniel Ritchie. 2024. Open-Universe Indoor Scene Generation using LLM Program Synthesis and Uncurated Object Databases. arXiv preprint arXiv:2403.09675 (2024).

- Tim Brooks, Aleksander Holynski, and Alexei A. Efros. 2022. InstructPix2Pix: Learning to Follow Image Editing Instructions. In CVPR. 18392–18402. https://api. semanticscholar.org/CorpusID:253581213
- Tim Brooks, Aleksander Holynski, and Alexei A. Efros. 2023. InstructPix2Pix: Learning to Follow Image Editing Instructions. In CVPR.

Vladimir Bychkovsky, Sylvain Paris, Eric Chan, and Fredo Durand. 2011. Learning photographic global tonal adjustment with a database of input / output image pairs. In CVPR. 97–104.

Ming Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. 2023. MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing. In ICCV. 22503–22513. https://api.semanticscholar.org/ CorpusID:258179432

Qifeng Chen, Jia Xu, and Vladlen Koltun. 2017. Fast Image Processing With FullyConvolutional Networks. In Proceedings of the IEEE International Conference on Computer Vision (ICCV).

Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. 2024. LayoutGPT: Compositional Visual Planning and Generation with Large Language Models. Advances in Neural Information Processing Systems 36 (2024).

Michael Fischer, Konstantin Kobs, and Andreas Hotho. 2020. NICER: Aesthetic image enhancement with humans in the loop. arXiv preprint arXiv:2012.01778 (2020). Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Wang, Yinfei Yang, and Zhe Gan. 2024. Guiding Instruction-based Image Editing via Multimodal Large Language Models. In ICLR. https://arxiv.org/abs/2309.17102

Rinon Gal, Adi Haviv, Yuval Alaluf, Amit H. Bermano, Daniel Cohen-Or, and Gal Chechik. 2024. ComfyGen: Prompt-Adaptive Workflows for Text-to-Image Generation. arXiv preprint arXiv:2410.01731 (2024).

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative Adversarial Nets. In Advances in Neural Information Processing Systems, Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K.Q. Weinberger (Eds.), Vol. 27. Curran Associates, Inc. https://proceedings.neurips.cc/paper_files/paper/2014/file/ 5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf

Tiankai Hang, Shuyang Gu, Dong Chen, Xin Geng, and Baining Guo. 2024. CCA: Collaborative Competitive Agents for Image Editing. (2024). arXiv:2401.13011 [cs.CV] Alec Helbling, Seongmin Lee Lee, and Polo Chau. 2024. ClickDiffusion: Harnessing

LLMs for Interactive Precise Image Editing. arXiv preprint arXiv:2012.01778 (2024).

Amir Hertz, Ron Mokady, Jay M. Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2022. Prompt-to-Prompt Image Editing with Cross Attention Control. arXiv preprint arXiv:2208.01626 (2022). https://api.semanticscholar.org/CorpusID: 251252882

Yuanming Hu, Hao He, Chenxi Xu, Baoyuan Wang, and Stephen Lin. 2018. Exposure: A White-Box Photo Post-Processing Framework. ACM Transactions on Graphics (TOG) 37, 2 (2018), 26.

Ian Huang, Guandao Yang, and Leonidas Guibas. 2024. BlenderAlchemy: Editing 3D Graphics with Vision-Language Models. arXiv:2404.17672 [cs.CV] https://arxiv. org/abs/2404.17672

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7B. arXiv preprint arXiv:2310.06825 (2023).

- Han-Ul Kim, Young Jun Koh, and Chang-Su Kim. 2020a. Global and Local Enhancement Networks for Paired and Unpaired Image Enhancement. In ECCV (Glasgow, United Kingdom). Springer-Verlag, Berlin, Heidelberg, 339–354. https://doi.org/10.1007/ 978-3-030-58595-2_21
- Han-Ul Kim, Young Jun Koh, and Chang-Su Kim. 2020b. PieNet: Personalized Image Enhancement. In European Conference on Computer Vision.

Satoshi Kosugi and Toshihiko Yamasaki. 2020. Unpaired image enhancement featuring reinforcement-learning-controlled image editing software. In Proceedings of the AAAI conference on artificial intelligence, Vol. 34. 11296–11303.

Chongyi Li, Chunle Guo, Shangchen Zhou, Qiming Ai, Ruicheng Feng, and Chen Change Loy. 2023. FlexiCurve: Flexible Piecewise Curves Estimation for Photo Retouching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. 1092–1101.

Jie Liang, Hui Zeng, Miaomiao Cui, Xuansong Xie, and Lei Zhang. 2021. PPR10K: A Large-Scale Portrait Photo Retouching Dataset with Human-Region Mask and Group-Level Consistency. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

Gabrielle Littlefair, Niladri Shekhar Dutt, and Niloy J. Mitra. 2025. FlairGPT: Repurposing LLMs for Interior Designs. arXiv:2501.04648 [cs.GR] https://arxiv.org/abs/2501. 04648

Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. 2024. DoRA: Weight-Decomposed LowRank Adaptation. arXiv:2402.09353 [cs.CL] https://arxiv.org/abs/2402.09353

Yihao Liu, Jingwen He, Xiangyu Chen, Zhengwen Zhang, Hengyuan Zhao, Chao Dong, and Yu Qiao. 2022. Very Lightweight Photo Retouching Network With Conditional Sequential Modulation. IEEE Trans. Multi. 25 (June 2022), 4638–4652. https://doi.org/10.1109/TMM.2022.3179904

Martin Lövdén, Benjamín Garzón, and Ulman Lindenberger. 2020. Human skill learning: expansion, exploration, selection, and refinement. Current Opinion in Behavioral Sciences 36 (2020), 163–168.

Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. 2023. Dragondiffusion: Enabling drag-style manipulation on diffusion models. arXiv preprint arXiv:2307.02421 (2023).

Wenqi Ouyang, Yi Dong, Xiaoyang Kang, Peiran Ren, Xin Xu, and Xuansong Xie. 2023. Rsfnet: A white-box image retouching approach using region-specific color filters. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 12160–12169.

Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, and Furu Wei. 2023. Kosmos-G: Generating Images in Context with Multimodal Large Language Models. arXiv preprint arXiv:2310.02992 (2023). https://api.semanticscholar.org/CorpusID: 263620748

Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. 2023. Kosmos-2: Grounding Multimodal Large Language Models to the World. arXiv preprint arXiv:2306.14824 (2023). https://api.semanticscholar.org/ CorpusID:259262263

Zekun Qi, Runpei Dong, Shaochen Zhang, Haoran Geng, Chunrui Han, Zheng Ge, Li Yi, and Kaisheng Ma. 2024. ShapeLLM: Universal 3D Object Understanding for Embodied Interaction. arXiv preprint arXiv:2402.17766 (2024).

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2021. High-Resolution Image Synthesis with Latent Diffusion Models. arXiv:2112.10752 [cs.CV]

Rodrigo Santos, João Silva, and António Branco. 2024. Leveraging LLMs for On-the-Fly Instruction Guided Image Editing. arXiv:2403.08004 [cs.CL] https://arxiv.org/abs/ 2403.08004

Jing Shi, Ning Xu, Yihang Xu, Trung Bui, Franck Dernoncourt, and Chenliang Xu. 2021. Learning by Planning: Language-Guided Global Image Editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 13590–13599.

Gemini Team. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv:2403.05530 [cs.CL] https://arxiv.org/abs/2403.05530 Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023).

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. 2024a. Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution. arXiv preprint arXiv:2409.12191 (2024).

Zhengyi Wang, Jonathan Lorraine, Yikai Wang, Hang Su, Jun Zhu, Sanja Fidler, and Xiaohui Zeng. 2024b. LLaMA-Mesh: Unifying 3D Mesh Generation with Language Models. arXiv:2411.09595 [cs.LG] https://arxiv.org/abs/2411.09595

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2024. Chain-of-thought prompting elicits reasoning in large language models. In Proc. NeurIPS. Article 1800, 14 pages.

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. 2024. OmniGen: Unified Image Generation. arXiv preprint arXiv:2409.11340 (2024). https://api.semanticscholar.org/ CorpusID:272694523

Jianzhou Yan, Stephen Lin, Sing Bing Kang, and Xiaoou Tang. 2014. A Learning-to-Rank Approach for Image Color Enhancement. In CVPR. 2987–2994.

Zhicheng Yan, Hao Zhang, Baoyuan Wang, Sylvain Paris, and Yizhou Yu. 2016. Automatic Photo Adjustment Using Deep Neural Networks. ACM Trans. Graph. 35, 2, Article 11 (Feb. 2016), 15 pages.

Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, Chris Callison-Burch, Mark Yatskar, Aniruddha Kembhavi, and Christopher Clark. 2023. Holodeck: Language Guided Generation of 3D Embodied AI Environments. https://doi.org/10.48550/ ARXIV.2312.09067

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding Conditional Control to Text-to-Image Diffusion Models. In ICCV. Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In CVPR.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. 2024. LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations). Association for

Computational Linguistics, Bangkok, Thailand. http://arxiv.org/abs/2403.13372 Wang Zhenyu, Li Aoxue, Li Zhenguo, and Liu Xihui. 2024. GenArtist: Multimodal LLM as an Agent for Unified Image Generation and Editing. arXiv preprint arXiv:2407.05600 (2024).

