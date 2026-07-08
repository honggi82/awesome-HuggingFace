## IMG: Calibrating Diffusion Models via Implicit Multimodal Guidance

Jiayi Guo1,2* Chuanhao Yan1,2* Xingqian Xu1 Yulin Wang2 Kai Wang1 Gao Huang2† Humphrey Shi1†

1SHI Labs @ Georgia Tech 2Tsinghua University https://github.com/SHI-Labs/IMG-Multimodal-Diffusion-Alignment

(a) Concept Comprehension (b) Aesthetic Quality One morning I chased an elephant in my pajamas.

# arXiv:2509.26231v1[cs.CV]30Sep2025

The clock on the side of the metal building is gold and black.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

(c) Object Addition (d) Object Correction

A woman on the top of a butterfly. A rubber ball and a metallic car.

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

FLUX FLUX + IMG (Ours) FLUX

FLUX + IMG (Ours)

Figure 1. The multimodal misalignment issue. Even the latest state-of-the-art diffusion model, FLUX.1 [dev] (FLUX) [37], may overlook or misinterpret concepts in prompts. Assisted with our proposed Implicit Multimodal Guidance (IMG) framework, the promptimage misalignment issues are significantly mitigated in various aspects such as concept comprehension, aesthetic quality, object addition, and correction. In each case, both images are generated with the same random seed for fair comparison.

### Abstract

re-generation-based multimodal alignment framework that requires no extra data or editing operations. Specifically, given a generated image and its prompt, IMG a) utilizes a multimodal large language model (MLLM) to identify misalignments; b) introduces an Implicit Aligner that manipulates diffusion conditioning features to reduce misalignments and enable re-generation; and c) formulates the re-alignment goal into a trainable objective, namely Iteratively Updated Preference Objective. Extensive qualitative and quantitative evaluations on SDXL, SDXL-DPO, and FLUX show that IMG outperforms ex-

Ensuring precise multimodal alignment between diffusiongenerated images and input prompts has been a longstanding challenge. Earlier works finetune diffusion weight using high-quality preference data, which tends to be limited and difficult to scale up. Recent editing-based methods further refine local regions of generated images but may compromise overall image quality. In this work, we propose Implicit Multimodal Guidance (IMG), a novel

*Equal contribution. †Corresponding authors.

isting alignment methods. Furthermore, IMG acts as a flexible plug-and-play adapter, seamlessly enhancing prior finetuning-based alignment methods. Our code will be available at https://github.com/SHI-Labs/IMG– Multimodal-Diffusion-Alignment.

Prompt: A woman on top of a butterfly.

[Figure 9]

[Figure 10]

Pretrained Models (Detector, LLM)

IMG (MLLM, Aligner)

[Figure 11]

[Figure 12]

Encoder

Encoder

[Figure 13]

[Figure 14]

[Figure 15]

Features

Features

Edit Instruction: Add a woman

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Diffusion

Extra Editing

Diffusion

### 1. Introduction

Misaligned

Edited

Misaligned

Aligned

Recently, diffusion models have become powerful text-toimage (T2I) generation tools capable of producing diverse and realistic images. However, most methods still face input-output misalignment challenges based on human inspection. More specifically, these models may overlook or misinterpret a few aspects of prompts, thus creating undesired visual results with misinterpretations. Fig. 1 shows evidence that even the latest state-of-the-art diffusion model, FLUX [37], generates images that require further refinement in terms of prompt awareness and adherence.

|[Figure 20]|
|---|

|[Figure 21]|
|---|

[Figure 22]

[Figure 23]

a) Alignment via Extra Editing (Existing methods)

b) Alignment via Re-generation (Ours)

a) Existing methods b) Ours

Pipeline Complexity High (Extra Editing) Low (Re-generation)

Reuse Diffusion Process ✗ ✓

Performance (HPS score↑)

28.14 29.48

Due to the aforementioned challenges, exploring methods that improve prompt-image alignment has become an emerging and critical area of research. Early works in this field primarily employ the preference-based weight finetuning. Methods such as [11, 57] finetune diffusion models on high-quality prompt-image pairs. Later studies [5, 10, 13, 67, 80] apply reinforcement learning from human feedback (RLHF), exploring rewarding algorithms and datasets that are proven to be effective for large language models (LLMs) [1, 12]. However, these methods are constrained by the limited availability of high-quality finetuning data, which is difficult to scale up further. In contrast, the recent LLM-driven image editing method [70] eliminates the need for fine-tuning model weights. It combines an open-set detector [49] with an LLM [1] to identify misalignments in generated images and produces language instructions for image editing (see Fig. 2a). Leveraging powerful LLMs for verification and reflection on generation results is a promising research direction [19, 53, 54]. However, the current editing pipeline primarily focuses on improving alignment in locally edited regions, and we empirically find that the overall image quality often fails to maintain the pre-editing level. Additionally, its detector sometimes omits critical misalignments, resulting in inaccurate editing instructions from the LLM.

Figure 2. Comparison between our Implicit Multimodal Guidance (IMG) and existing editing-based alignment methods. a) Existing methods require additional editing operations that improve alignment in local regions but may compromise overall image quality. b) In contrast, IMG employs a re-generation-based alignment framework by manipulating diffusion conditioning features, ensuring pipeline simplicity and high-quality outputs.

fusion models. Subsequently, the Implicit Aligner takes as input both features of MLLM and misaligned images, producing better-aligned diffusion conditioning features to reduce misalignment and enable re-generation. To train the Implicit Aligner, we introduce an Iteratively Updated Preference Objective, which combines direct preference optimization [60] and self-play finetuning [8].

To conclude, IMG distinguishes itself from prior approaches by: a) serving as a flexible plug-and-play adapter that seamlessly integrates with both base diffusion models and their finetuned versions once trained, b) leveraging the native diffusion generation process to maintain pipeline simplicity and high-quality outputs, and c) eliminating the need for additional editing operations, making it a more straightforward and efficient solution. Built on these principles, we conduct extensive qualitative and quantitative evaluations across popular T2I models, including SDXL [57], SDXL-DPO [67], and FLUX [37], as well as both finetuning-based and editing-based alignment methods. Results consistently demonstrate that IMG effectively re-aligns image outputs to improved versions and outperforms existing alignment methods.

In this work, we explore enhancing alignment performance without additional data required by finetuning-based methods or extra editing operations in editing-based methods. To this end, we propose the Implicit Multimodal Guidance (IMG), a novel diffusion alignment framework that improves prompt adherence through a simple yet effective re-generation process. Specifically, IMG involves a multimodal large language model (MLLM) and a newly introduced Implicit Aligner. The MLLM first detects misalignment between the prompts and the images generated by dif-

### 2. Related Work

Diffusion model alignment is a research area focusing on improving the prompt adherence of diffusion models in

terms of human perception. Early efforts [11, 57, 64, 71, 74] directly finetune diffusion models on high-quality datasets to produce visually appealing results. Later studies conduct rewarding algorithms based on reinforcement learning [5, 10, 13, 31, 73] and preference learning [67, 76, 80] using datasets labeled with human preferences [20, 36]. For example, DDPO [5] utilizes black-box reward functions to optimize diffusion models within specific prompt sets. Diffusion-DPO [67] re-formulates the direct preference optimization [60] into a differentiable diffusion objective via the evidence lower bound. Inspired by the rapid advances of large language models (LLMs) [1, 12, 42, 61], recent works explore integrating LLMs inside the diffusion system to enhance prompt comprehension and representation [14, 23, 41, 70, 77]. For instance, LMD [41] performs layout-grounded image generation using captioned bounding boxes generated from prompts by LLMs [1]. SLD [70] introduces an LLM-driven image editing framework that utilizes open-set detectors [49] and LLMs [1] to identify misalignments and edit initial generation results. Additionally, recent studies explore the use of MLLMs to aid in image editing [15, 18, 30, 47, 55] and customization [27, 40, 56, 66, 81, 84]. However, these methods primarily focus on executing human-provided instructions, rather than automatically correcting misalignment.

A woman on the top of a butterfly.

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

A rubber ball and a metallic car.

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Initial Generation Instruct Pix2Pix SLD

IMG (Ours)

Figure 3. Comparison with editing-based methods. We evaluate the performance of Instruct Pix2Pix [6] and SLD [70] with IMG. For Instruct Pix2Pix, the instructions are “add a woman” and “make the ball a rubber ball”, generated by our finetuned MLLM.

example, a canonical formulation for xt is defined as such:

##### xt = αtx0 + σtϵ, (1)

where t ∼ U(0,T), ϵ ∼ N(0,I) is random Gaussian noise, αt and σt are predefined functions of t. The reverse process iteratively transforms xT into x0, assessing intermediate xt through a well-trained deep neural net ϵθ [25, 43, 46, 65].

Considering the text-to-image generation task, given an image x0 and text condition cT, the training objective of ϵθ is formulated as:

Diffusion adapters [21, 22, 29, 50, 62, 69, 75, 78, 82, 83] aim to extend the capability of diffusion models by incorporating additional input conditions beyond text prompts. Pioneering works like ControlNet [82] and T2I-Adapter [50] introduce structure-conditioned adapters to enable controllable image synthesis. On top of these baselines, works such as Composer and Uni-ControlNet [29, 58, 83] propose a unified adapter capable of handling various structural conditions. Recently, diffusion adapters enabling visualencoding [9, 32, 34, 44, 75, 78] have continuously gained attention in variation, editing and video tasks. For instance, Prompt-free Diffusion [75] introduces the SeeCoder as an image context encoder, replacing the original text encoder to support image prompts. IP-Adapter [78] integrates CLIP image encoder [59] with decoupled cross-attention layers to create an effective image-prompt adapter.

0,ϵ,cT,t ∥ϵ − ϵθ(xt,cT,t)∥22 . (2)

Ldiff = Ex

Diffusion with image prompt [75, 78, 82, 83] has recently gained popularity in the community, in which diffusion models reconstruct images similar to the input image, usually through a frozen T2I model attaching an additional image prompt (IP) encoder that takes visual inputs. Thus, the training objective is extended to include an additional image condition cI, which is the encoded features of x0:

0,ϵ,cT,cI,t ∥ϵ − ϵθ(xt,cT,cI,t)∥22 . (3)

Ldiff = Ex

During inference, users input both cI and, optionally, cT to create or enhance a content-consistent variant of x0. IP encoders [72, 78] are widely available for diffusion models such as SD series [64] and FLUX [37].

### 3. Methodology

#### 3.2. MLLM-driven Misalignment Analysis

In this section, we first go through preliminaries of diffusion models with text and image prompts [64, 78] in Sec. 3.1. Then, we provide an in-depth explanation of our method: Implicit Multimodal Guidance (IMG) in Secs. 3.2 to 3.4.

Diffusion models sometimes misinterpret or overlook parts of a prompt when generating images. To address this issue, our first step is to identify these misalignments via Multimodel Large Language Models (MLLMs) that are capable of visual question answering [3].

#### 3.1. Preliminaries

While MLLMs [42] can describe image contents, there has been limited effort to customize them as misalignment detectors. In IMG, we address this by introducing a customized MLLM, which we finetune using instruction-based image data [6], enabling the MLLM to analyze and respond

Diffusion models [37, 64] are a class of generative methods involving forward and reverse processes. The forward process is usually known as a gradual procedure, transforming the data point x0 into Gaussian noise xT with T steps. For

Prompt: One morning I chased an elephant in my pajamas.

Construct Instruction Diffusion

[Figure 32]

How to make the <Image> match the intended prompt: <Prompt>?

[Figure 33]

[Figure 34]

Initial Image

[Figure 35]

MLLM

[Figure 36]

Guidance Features

IP Encoder Implicit Aligner

Aligned Image

[Figure 37]

Initial Image Features

[Figure 38]

[Figure 39]

[Figure 40]

Initial Generation MLLM Analysis Re-generation

Aligned Image Features

Diffusion

- Figure 4. Overview of the Implicit Multimodal Guidance (IMG) framework. Given an initial image that exhibits misalignments with its prompt, IMG begins by conducting an MLLM-driven misalignment analysis. Following this, IMG utilizes an Implicit Aligner to translate the initial image features into better-aligned features according to the MLLM’s guidance. Finally, these aligned image features are incorporated as new conditions to re-generate images with improved prompt-image alignment.

set up an image re-generation process conditioned on implicit features, highlighted as Implicit Multimodal Guidance (IMG). A complete diagram of our IMG framework is shown in Fig. 4, in which a) we generate an initial image that may exhibit misalignments via a diffusion model; b) we detect potential misalignments via MLLM using questions same as those in the finetuning process, (e.g., “How to make the <Image> match the intended prompt: <Prompt>?”); and c) we re-generate the image conditioned on aligned features produced by the newly introduced Implicit Aligner.

to potential misalignments.

Specifically, the instruction-based dataset contains original images I0, edited images I1, prompts for both images T0 & T1, and edit instructions TE. We utilize the data by taking out I0, T1 and TE as our training triplets. While I0 and T1 are fed into the MLLM as inputs, we request the model to describe alignment through question prompts such as “How to make the <Original Image> match the intended prompt: <Edited Prompt>?”, then supervise model outputs against TE. Through experiments in Sec. 4.5, we demonstrate that our specialized finetuning makes MLLM a much more reliable model for misalignment detection. For more details regarding MLLM finetuning, please see our supplementary.

Generally, our Implicit Aligner functions as an adapter network for the original diffusion model, facilitating the realignment of the initial image to better adhere to its prompt. Instead of explicitly editing the initial image [6, 70], our design implicitly refines the diffusion conditioning features. Structured as a stack of cross-attention layers, the Implicit Aligner takes MLLM hidden states, namely guidance features, as one input and initial image features, embedded via an image prompt (IP) encoder, as the other. The guidance features promote modifications on initial features, producing better-aligned features that are trained to fit the embedded features of a well-aligned image. According to the properties of image prompt diffusion described in Sec. 3.1, the diffusion model can then re-generate a more faithfully aligned image from these re-aligned features.

#### 3.3. Implicit Multimodal Guidance

Our next step in IMG is to improve the generated images by removing the previously detected misalignment. A straightforward baseline is to edit these images through existing editing methods like [6, 70]. In Fig. 3, we provide a quick demo using Instruct Pix2Pix [6] and SLD [70], both showing unsatisfactory results. For Instruct Pix2Pix, we use the responses generated by our finetuned MLLM as instructions. However, the result shows that the editing takes place incorrectly. For SLD, although it successfully generates a woman in the first case, the overall aesthetic quality is degraded, and SLD fails to deal with the second case.

In the next section, we will introduce Iteratively Updated Preference Objective, the training objective of our Implicit Aligner that leverages human preference datasets [36].

The unsuccessful try-out on existing editing methods motivates us to tackle the challenge from a new angle:

#### 3.4. Iteratively Updated Preference Objective

As aforementioned, we leverage the same open-source human preference datasets [36] adopted in finetuning-based alignment methods [67] to train our Implicit Aligner. These datasets consist of triplets {cT,xw0 ,xl0}, where xw0 and xl0 represent the human-preferred (winning) and non-preferred (losing) images based on the prompt cT (i.e., xw0 ≻ xl0). Let clI and cwI denote the image prompt features of xl0 and xw0 . The direction from clI to cwI indicates a meaningful alignment-improving trajectory, conditioned on cT [68, 78]. Such trajectory suggests a basic objective for our Implicit Aligner, in which the network fθ fits cwI based on input conditions c = (h,clI), where h represents the MLLM guidance features and clI represents the image features, formulated as the following:

||cwI − fθ(c)||22. (4)

Lbase = Ec,cw

I

Besides the basics, we also draw inspiration from direct preference optimization (DPO) [60] and self-play finetuning (SPIN) [8] for enhanced feature alignment. In our case, we use DPO to finetune fθ from a reference aligner fref (a copy of fθ from an earlier training iteration), encouraging the network to output fθ(c) close to the preferred features cwI while keeping distance from the nonpreferred features clI. SPIN also forces fθ(c) to be close to cwI while additionally forces fθ(c) to be more preferred than crefI = fref(c). Adapting both DPO [60, 67] and SPIN [8, 80], we derive our objective as the following:





pθ(cwI |c) pref(cwI |c) − log

pθ(clI|c) pref(clI|c)

Lpref = Ec,cw

ℓ

log

 

 

I ,clI

DPO





pθ(crefI |c) pref(crefI |c)

pθ(cwI |c) pref(cwI |c) − log

+log

,

 

 

SPIN

(5) where ℓ is a monotonically decreasing convex function, usually implemented as ℓ(x): = log(1 + e−x) [67, 80]. Following [35, 39, 63], we define pθ and pref as Gaussian distributions with means predicted by fθ and fref and a constant variance σ, i.e., p(cI|c) = N(cI;f(c),σ). With some transformations, the objective in Eq. 5 can be simplified to:

I ,clI[ℓ(−[2(||cwI − fθ(c)||22 − ||cwI − fref(c)||22) − (||clI − fθ(c)||22 − ||clI − fref(c)||22) − ||fref(c) − fθ(c)||22])].

Lpref = Ec,cw

(6) The detailed derivation of Lpref can be found in the sup-

plementary. The final training objective is then a combination of Lbase and Lpref with a ratio parameter λ:

L = Lbase + λLpref. (7)

Unlike prior works such as DPO [67] and SPIN [80], which require a fixed reference net fref with frozen weights, our Implicit Aligner can be trained from scratch and the reference net can be runtime updated. Specifically, we first randomly initialize fref and later iteratively copy fθ to fref whenever fθ outperforms fref. In practice, we execute the substitution when fθ(c) is closer to cwI than fref(c) for k consecutive iterations. Thereafter, we name our objective function L the Iteratively Updated Preference Objective.

### 4. Experiments

#### 4.1. Experimental Setup

Baselines and models. Our experiments are conducted on two diffusion models: SDXL [57] and FLUX.1 [dev] (FLUX) [37]. We also compare IMG against DiffusionDPO (SDXL-DPO) [67], the top-performing finetuningbased alignment method, and SLD [70], the leading editingbased alignment method. We further compare IMG with leading compositional generation methods, ELLA [26] and CoMat [33]. For MLLM, we finetune LLaVA 1.5-13b [42] on the Instruct-Pix2Pix dataset [6] for 1 epoch following [42] and extract the last hidden layer features for guidance. We use the IP-Adapter [72, 78], trained on SDXL and FLUX to enable image prompts and extract image features.

Implementation details. IMG’s Implicit Aligner is trained on the Pick-a-Pic training set [36] (the same as SDXLDPO [67]) for 100K iterations with 8 A100 GPUs and a batch size of 8. The training data contains 851K preferred and non-preferred image pairs under specific prompts. We use the AdamW [45] optimizer with a constant learning rate of 1×10−4 and a weight decay of 1×10−4. The ratio parameter λ in Eq. 7 is set to 1. The reference model updating step k in Sec. 3.4 is set to 10. We determine the optimal hyperparameters via the average Pick Score [36] across generated images on 500 Pick-a-Pic test set prompts. Pick Score is a caption-aware alignment scoring model trained on Picka-Pic. For evaluation, we report the Human Preference Scores (HPS) on the Human Preference Datasets (HPD) test set [71], which includes 3,400 prompts across 5 categories and on the Parpi-Prompts [79], a diverse prompt dataset of 1,632 prompts ranging from brief concepts to complex sentences. We also report results on the T2I-CompBench [28], which contains 1800 test prompts to validate compositional image generation capabilities. We apply 50 sampling steps for SDXL and SDXL-DPO, and 30 for FLUX. More implementation details are provided in the supplementary.

Human Preference Datasets (HPD)

Model

Parti-Prompts User Study Anime Concept-Art Drawbench Painting Photo Average

SDXL [57] 28.59 27.69 28.04 27.74 28.10 28.03 27.71

77.6%

SDXL + IMG (Ours) 29.14 (85.5%) 28.31 (92.8%) 28.48 (81.5%) 28.33 (91.5%) 28.49 (80.3%) 28.56 (87.2%) 28.13 (79.2%) SDXL-DPO [67] 28.98 28.12 28.40 28.18 28.29 28.39 27.99

75.5%

SDXL-DPO + IMG (Ours) 29.42 (84.8%) 28.55 (86.1%) 28.68 (77.0%) 28.58 (83.8%) 28.66 (77.1%) 28.79 (82.6%) 28.34 (76.0%) FLUX [37] 29.77 28.98 29.44 28.90 29.14 29.21 29.23

66.9% FLUX + IMG (Ours) 30.06 (67.0%) 29.17 (62.0%) 29.80 (64.0%) 29.20 (69.5%) 29.41 (64.8%) 29.48 (65.7%) 29.61 (71.7%)

Table 1. Quantitative comparison with base models and finetuning-based alignment methods on HPD and Parti-Prompts. We report the average HPS scores and the win rates of IMG over competing methods. Higher HPS scores indicate better preference alignment. Additionally, we conduct user studies to assess the real human preference rates of IMG.

Model Color↑ Shape↑ Texture↑ Spatial↑ Non-Spatial↑ Complex↑

A robot penguin wearing a top hat and playing a vintage trumpet under a rainbow.

The city of London on Mars.

SDXL [57] 0.5552 0.4878 0.5213 0.1915 0.3125 0.3403 SDXL-ELLA [26] 0.7260 0.5634 0.6686 0.2214 0.3069 SDXL-CoMat [33] 0.7827 0.5329 0.6468 0.2428 0.3187 0.3680 SDXL-DPO [67] 0.6957 0.5430 0.6428 0.2415 0.3160 0.3561 SDXL-DPO + IMG (Ours) 0.7294 0.5811 0.6548 0.2484 0.3199 0.3874

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

FLUX [37] 0.7852 0.5633 0.6942 0.2792 0.3129 0.3891 FLUX + IMG (Ours) 0.8060 0.5773 0.7008 0.2826 0.3139 0.3983

Photograph of a wall along a city street with a watercolor mural of foxes in a jazz band.

Table 2. Quantitative comparison with base models and finetuning-based alignment methods on T2I-CompBench. The best results are in bold and the second-best results are underlined.

The rectangular mirror was hung above the white sink.

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Quantitative comparison. In Tab. 1, we conduct quantitative evaluations on the HPD [71] and Parti-Prompts [79] datasets, which contain thousands of prompts across diverse categories and complexities. We evaluate images generated by the base models and those re-generated by IMG, using HPS scores [71], and report the win rates of IMG over the base models. The results demonstrate that IMG serves as a general framework that consistently enhances alignment across different base models. Notably, IMG shares the same training data as SDXL-DPO, yet when integrated with SDXL, it outperforms SDXL-DPO and achieves an average win rate of 84.6% over SDXL. Furthermore, when integrated with SDXL-DPO, IMG achieves even higher performance, with an average win rate of 80.5% over SDXLDPO. This suggests that IMG not only enhances alignment more effectively but also complements finetuningbased methods boosting their performance without requiring additional data. When combined with the state-of-theart FLUX model, IMG also achieves a strong alignment performance with an average win rate of 67.6%. In addition, we conduct user studies where 33 evaluators were asked to do an A-B test on 30 random image pairs generated by each base model and IMG with the same prompt. Each unique pair was assessed by 3 evaluators, and only fully consistent votes were used to compute the final win rates. Results show that approximately 70% of users prefer our re-aligned images over the originals. In Tab. 2, beyond general prompt sets, we further assess IMG on the compositional image generation benchmark, T2I-CompBench [50]. Without specialized training on compositional prompts like CoMat [33], IMG operates in a zero-shot manner while achieving leading performance with both SDXL and FLUX.

A man is holding a telescope and observing a comet in the night sky.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

SDXL SDXL + IMG (Ours) SDXL-DPO SDXL-DPO + IMG (Ours)

- Figure 5. Qualitative comparison with base models and finetuning-based alignment methods. The first two rows show that IMG addresses various misalignment types across different prompts, while the last row shows that IMG resolves misalignment issues that challenge both models.

#### 4.2. Comparison with Base Models and Finetuningbased Methods

Qualitative comparison. In addition to comparing with FLUX [37] in Fig. 1, we perform further quantitative evaluations by integrating IMG with the widely used SDXL [57] model and its finetuned variant, SDXL-DPO [67]. The Implicit Aligner of IMG is trained using SDXL’s image prompt encoder [78] and seamlessly shared with SDXLDPO. As shown in Fig. 5, the results highlight a) IMG’s effectiveness in addressing diverse alignment issues from various aspects, such as concept comprehension (e.g., a penguin with a robotic body and foxes playing musical instruments), aesthetic quality (e.g., a well-constructed city on Mars), object addition (e.g., a comet in the sky) and object correction (e.g., a rectangular mirror), and b) IMG’s flexibility to operate with SDXL-DPO without additional training. The last row also showcases cases where both SDXL and SDXL-DPO struggle, whereas IMG demonstrates clear improvements in alignment across both models.

A baby daikon radish in a tutu walking a dog.

A spherical tennis ball and a conical tennis shoe.

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

A girl diving into a pool.

The comfortable armchair embraced the tired body and the hard wooden floor.

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Initial Generation One Round IMG Two Round IMG

Initial Generation SLD IMG (Ours)

- Figure 6. Qualitative comparison with editing-based methods. IMG surpasses SLD in visual quality and image comprehension.

Figure 7. Multi-round generation results. IMG continuously improves prompt-image alignment by executing multiple rounds.

SDXL SDXL-DPO FLUX

SDXL SDXL-DPO FLUX

Base Model 28.03 28.39 29.21 Base Model + IMG (LLaVA) 28.56 28.79 29.48 Base Model + IMG (Qwen-VL) 28.48 28.73 29.52

Base Model 28.03 28.39 29.21 Base Model + SLD [70] 27.35 27.52 28.14 Base Model + IMG (Ours) 28.56 28.79 29.48

Table 4. Generalization capability to different MLLMs. We further integrate IMG with Qwen-VL [4]. The results indicate consistent improvement across different MLLMs.

Table 3. Quantitative comparison with editing-based methods. SLD [70] is designed to enhance alignment through local editing but often at the cost of overall image quality.

#### 4.4. Discussion

#### 4.3. Comparison with Editing-based Methods.

MLLM misalignment detection accuracy. We evaluate our finetuned MLLM as follows: a) We sample 300 instruction-based editing cases from SEED-Data-Edit [16] as the test set; b) Both the original and the finetuned MLLM detect misalignment by predicting editing instructions; and c) GPT-4 [1] is employed to verify whether the predictions are semantically consistent with the ground truth. As a result, the finetuned MLLM achieves a 76.3% “yes” rate, compared to 42.3% for the original MLLM.

Qualitative comparison. In Fig. 6, we compare IMG with the leading editing-based alignment method, SLD [70]. In the first case, although SLD recognizes the missing tennis shoe, it incorrectly deletes the intended tennis ball. In the second case, SLD fails to recognize the unintended human gesture, resulting in no edits. These limitations arise primarily because the LLM in SLD interprets image content through text-based bounding box descriptions provided by a detector, which introduces several risks: a) inaccurate detection results that lead to incorrect editing instructions from the LLM (e.g., removing the tennis ball) and b) an inability to address quality-related misalignments (e.g., overlooking the unintended gesture). In contrast, IMG, leveraging an MLLM that processes both images and prompts as input, more accurately identifies potential misalignments.

Generalization to different MLLMs. In Tab. 4, we additionally integrate IMG with Qwen2.5-VL-7B. Evaluations on the HPD benchmark show consistent alignment improvements across different MLLMs, highlighting IMG’s strong generalization capability.

Multi-round generation. In Fig. 7, we demonstrate that IMG functions as an iterable framework that continuously enhances alignment through multiple rounds. For details that are not fully aligned in the first round IMG, e.g., the dog-like shape of the daikon radish and the action of diving, a second round of IMG further refines these details.

Quantitative Comparison. In Tab. 3, we further compare IMG with SLD on the HPD benchmark. The results reveal that SLD leads to performance degradation, aligning with our qualitative observations and analysis. While SLD may enhance alignment in locally edited regions, the accumulated errors from the detector, LLM, and editing pipeline ultimately reduce overall image quality and alignment performance. In contrast, IMG improves alignment through a native and more reliable re-generation process.

Dense prompt generation. Generating images from long, complex prompts remains a challenging task for diffusion models [26, 38, 79]. In Fig. 8, we integrate IMG with FLUX and compare it against leading community models, including Stable Diffusion 3.5 Large (SD3.5) [2], PixArt-σ [7] and

A towering Gundam robot, painted in a striking combination of white, blue, and red, stands with its gleaming sword raised high against the backdrop of a sprawling metropolis. The cityscape features an array of tall, glass skyscrapers that reflect the fading light of the sunset. Beyond the urban expanse, a majestic mountain range looms, leading to the tranquil expanse of the ocean, while above, a dark, ominous moon dominates the twilight sky. The entire scene is rendered in a vivid, high-contrast style reminiscent of a detailed anime illustration.

A whimsical scene featuring a small elf with pointed ears and a green hat, sipping orange juice through a long straw from a disproportionately large orange. Next to the elf, a curious squirrel perches on its hind legs, while an owl with wide, observant eyes watches intently from a branch overhead. The orange's vibrant color contrasts with the muted browns and greens of the surrounding forest foliage.

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

SD3.5 FLUX SD3.5 FLUX

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

PixArt-𝜎 Playground v2.5 FLUX + IMG (Ours)

PixArt-𝜎 Playground v2.5 FLUX + IMG (Ours)

- Figure 8. Dense prompt generation. We integrate IMG with FLUX [37] and compare it against leading community models, including Stable Diffusion 3.5 Large (SD3.5) [2], PixArt-σ [7] and Playground v2.5 [38], using complex dense prompts.

Guidance features SDXL SDXL-DPO Base Model 22.34 22.57

Ratio SDXL SDXL-DPO Base Model 22.34 22.57

Components SDXL SDXL-DPO Base Model 22.34 22.57

No Guidance 22.22 22.39 Text Embedding 22.38 22.68 Original MLLM 22.59 22.81

0 22.56 22.76 0.5 22.56 22.80

+ Lbase 22.56 22.76 + Lpref−DPO 22.65 22.88 + Lpref−SPIN (IMG) 22.74 22.98

1 (IMG) 22.74 22.98 2 22.50 22.79

Finetuned MLLM (IMG) 22.74 22.98

(c) Impact of each loss component.

(b) Impact of different λ values in Eq. 7.

(a) Impact of different guidance features.

Table 5. Ablation studies. We examine the impact of different (a) guidance features, (b) loss ratios, and (c) loss components on the Pick-a-Pic test set via the average Pick Score. Higher Pick Scores indicate better preference alignment.

Playground v2.5 [38]. The results demonstrate that IMG substantially enhances details in generated images (as highlighted by the colored texts and boxes), enabling FLUX to stand out among competitors.

ment capability. IMG achieves the best alignment scores with our “Finetuned MLLM”. This aligns with the accuracy improvement in Sec. 4.4 and validates the effectiveness of our customized finetuning task in Sec. 3.2.

Impact of different λ values. In Tab. 5b, we examine the impact of varying the loss ratio λ in Eq. 7, which controls the strength of Lpref. Setting λ = 0 reduces the objective to Lbase alone. We find that a small λ may not sufficiently activate the effect of Lpref, while an excessively large λ can lead to training instability and thus suboptimal performance, as the Implicit Aligner is trained from scratch rather than finetuned from a reference model. Empirically, λ = 1 provides a balanced and effective trade-off.

#### 4.5. Ablation Studies

In Tab. 5, we examine the impact of different implicit guidance features, loss components, and loss ratios on the Picka-Pic test set via Pick Score to determine the optimal training scheme and hyperparameters.

Impact of different guidance features. In Tab. 5a, we examine various guidance features used by the Implicit Aligner to refine initial image features into better-aligned features. “No Guidance”, which directly uses initial image features as aligned ones, performs worse than the “Base Model”, underscoring the need for feature alignment. Using “Text Embedding” (text prompt embeddings from the diffusion text encoder) to guide feature alignment provides only minor improvements, as it lacks information on specific misalignments. “Original MLLM” features offer significant gains across two base models, making SDXL + IMG competitive with SDXL-DPO and highlighting MLLM’s align-

Impact of each loss component. In Tab. 5c, we examine each loss component in Eq. 7. Using the basic objective Lbase (Eq. 4), SDXL + IMG already achieves alignment performance competitive with SDXL-DPO. Furthermore, IMG is compatible with SDXL-DPO, enabling additional performance gains. For Lpref, we separately evaluate the impact of its DPO and SPIN components in Eq. 5. Results demonstrate that these two components provide progressive performance improvements in alignment.

### 5. Conclusion

In this paper, we propose Implicit Multimodal Guidance (IMG), a novel re-generation-based alignment framework. Unlike existing finetuning-based and editing-based approaches, IMG enhances alignment performance without requiring additional finetuning data or explicit editing operations. Specifically, given a generated image and its prompt, IMG involves an MLLM that identifies potential misalignments and an Implicit Aligner that reduces misalignments and facilitates re-generation by refining diffusion conditioning features. The Implicit Aligner is optimized through a trainable Iteratively Updated Preference Objective. Extensive qualitative and quantitative evaluations on SDXL, SDXL-DPO, and FLUX show that IMG outperforms existing alignment methods. Furthermore, IMG acts as a flexible plug-and-play adapter, seamlessly enhancing prior finetuning-based alignment methods.

### Acknowledgments

This research was supported in part by National Science Foundation under Award #2427478 - CAREER Program, and by National Science Foundation and the Institute of Education Sciences, U.S. Department of Education under Award #2229873 - National AI Institute for Exceptional Education. This project was also partially supported by cyberinfrastructure resources and services provided by College of Computing at the Georgia Institute of Technology, Atlanta, Georgia, USA.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv:2303.08774, 2023. 2, 3, 7
- [2] Stability AI. Stable diffusion 3.5 large. https: / / huggingface . co / stabilityai / stable diffusion-3.5-large, 2024. 7, 8
- [3] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. VQA: Visual question answering. In ICCV, 2015. 3
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 7
- [5] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In ICLR, 2023. 2, 3
- [6] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 3, 4, 5, 1, 2
- [7] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. PixArt-Sigma: Weak-to-strong train-

- ing of diffusion transformer for 4k text-to-image generation. arXiv:2403.04692, 2024. 7, 8
- [8] Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models. In ICML, 2024. 2, 5, 3
- [9] Mang Tik Chiu, Yuqian Zhou, Lingzhi Zhang, Zhe Lin, Connelly Barnes, Sohrab Amirghodsi, Eli Shechtman, and Humphrey Shi. Brush2prompt: Contextual prompt generator for object inpainting. In CVPR, 2024. 3
- [10] Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv:2309.17400, 2023. 2, 3
- [11] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv:2309.15807, 2023. 2, 3
- [12] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv:2407.21783, 2024. 2, 3
- [13] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Reinforcement learning for fine-tuning text-to-image diffusion models. In NeurIPS, 2024. 2, 3
- [14] Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and generation with large language models. In NeurIPS,

2024. 3

- [15] Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Yang Wang, Yinfei Yang, and Zhe Gan. Guiding instruction-based image editing via multimodal large language models. In ICLR,

2024. 3

- [16] Yuying Ge, Sijie Zhao, Chen Li, Yixiao Ge, and Ying Shan. Seed-data-edit technical report: A hybrid dataset for instructional image editing. arXiv preprint arXiv:2405.04007,

2024. 7

- [17] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. In NeurIPS, 2023. 3
- [18] Vidit Goel, Elia Peruzzo, Yifan Jiang, Dejia Xu, Xingqian Xu, Nicu Sebe, Trevor Darrell, Zhangyang Wang, and Humphrey Shi. Pair diffusion: A comprehensive multimodal object-level image editor. In CVPR, 2024. 3
- [19] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 2
- [20] Jiayi Guo, Chaoqun Du, Jiangshan Wang, Huijuan Huang, Pengfei Wan, and Gao Huang. Assessing a single image in reference-guided image synthesis. In AAAI, 2022. 3
- [21] Jiayi Guo, Chaofei Wang, You Wu, Eric Zhang, Kai Wang, Xingqian Xu, Humphrey Shi, Gao Huang, and Shiji Song. Zero-shot generative model adaptation via image-specific prompt learning. In CVPR, 2023. 3

- [22] Jiayi Guo, Xingqian Xu, Yifan Pu, Zanlin Ni, Chaofei Wang, Manushree Vasu, Shiji Song, Gao Huang, and Humphrey Shi. Smooth diffusion: Crafting smooth latent spaces in diffusion models. In CVPR, 2024. 3
- [23] Jiayi Guo, Junhao Zhao, Chaoqun Du, Yulin Wang, Chunjiang Ge, Zanlin Ni, Shiji Song, Humphrey Shi, and Gao Huang. Everything to the synthetic: Diffusion-driven testtime adaptation via synthetic-domain alignment. In CVPR,

2025. 3

- [24] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS Workshops, 2021. 2
- [25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 3
- [26] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv:2403.05135, 2024. 5, 6, 7, 1, 3
- [27] Jiannan Huang, Jun Hao Liew, Hanshu Yan, Yuyang Yin, Yao Zhao, Humphrey Shi, and Yunchao Wei. Classdiffusion: More aligned personalization tuning with explicit class guidance. In ICLR, 2025. 3
- [28] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. In NeurIPS, 2023. 5, 1
- [29] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv:2302.09778, 2023. 3
- [30] Yuzhou Huang, Liangbin Xie, Xintao Wang, Ziyang Yuan, Xiaodong Cun, Yixiao Ge, Jiantao Zhou, Chao Dong, Rui Huang, Ruimao Zhang, et al. Smartedit: Exploring complex instruction-based image editing with multimodal large language models. In CVPR, 2024. 3
- [31] Arman Isajanyan, Artur Shatveryan, David Kocharian, Zhangyang Wang, and Humphrey Shi. Social reward: Evaluating and enhancing generative AI through million-user feedback from an online creative community. In ICLR, 2024. 3
- [32] Jitesh Jain, Jianwei Yang, and Humphrey Shi. Vcoder: Versatile vision encoders for multimodal large language models. In CVPR, 2024. 3
- [33] Dongzhi Jiang, Guanglu Song, Xiaoshi Wu, Renrui Zhang, Dazhong Shen, Zhuofan Zong, Yu Liu, and Hongsheng Li. Comat: Aligning text-to-image diffusion model with imageto-text concept matching. In NeurIPS, 2024. 5, 6, 1
- [34] Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si, Dahua Lin, Yu Qiao, Chen Change Loy, and Ziwei Liu. Videobooth: Diffusion-based video generation with image prompts. In CVPR, 2024. 3
- [35] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2015. 5
- [36] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. In NeurIPS, 2023. 3, 4, 5, 1, 2
- [37] Black Forest Labs. Flux. https : / / blackforestlabs.ai/, 2024. 1, 2, 3, 5, 6, 8

- [38] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2.5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv:2402.17245, 2024. 7, 8
- [39] Wanhua Li, Xiaoke Huang, Jiwen Lu, Jianjiang Feng, and Jie Zhou. Learning probabilistic ordinal embeddings for uncertainty-aware regression. In CVPR, 2021. 5, 2
- [40] Wei Li, Xue Xu, Jiachen Liu, and Xinyan Xiao. Unimog: Unified image generation through multimodal conditional diffusion. In ACL, 2021. 3
- [41] Long Lian, Boyi Li, Adam Yala, and Trevor Darrell. Llmgrounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models. TMLR, 2024. 3
- [42] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2024. 3, 5, 1, 2
- [43] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2022. 3
- [44] Zeyu Liu, Zanlin Ni, Yeguo Hua, Xin Deng, Xiao Ma, Cheng Zhong, and Gao Huang. Coda: Repurposing continuous vaes for discrete tokenization. arXiv preprint arXiv:2503.17760,

2025. 3

- [45] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 5, 2
- [46] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. In NeurIPS,

2022. 3

- [47] Haoming Lu, Hazarapet Tunanyan, Kai Wang, Shant Navasardyan, Zhangyang Wang, and Humphrey Shi. Specialist diffusion: Plug-and-play sample-efficient fine-tuning of text-to-image diffusion models to learn any unseen style. In CVPR, 2023. 3
- [48] Peter McCullagh. Generalized linear models. Routledge,

2019. 2

- [49] Matthias Minderer, Alexey Gritsenko, and Neil Houlsby. Scaling open-vocabulary object detection. In NeurIPS, 2023. 2, 3
- [50] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2I-Adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv:2302.08453, 2023. 3, 6
- [51] Alfred M¨uller. Integral probability metrics and their generating classes of functions. Advances in applied probability,

1997. 3

- [52] David A Nix and Andreas S Weigend. Estimating the mean and variance of the target probability distribution. In ICNN. IEEE, 1994. 2
- [53] OpenAI. Openai o1. https://openai.com/o1/,

2024. 2

- [54] OpenAI. Openai o3-mini. https://openai.com/ index/openai-o3-mini/, 2024. 2
- [55] Wenqi Ouyang, Yi Dong, Lei Yang, Jianlou Si, and Xingang Pan. I2vedit: First-frame-guided video editing via image-tovideo diffusion models. arXiv:2405.16537, 2024. 3

- [56] Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, and Furu Wei. Kosmos-g: Generating images in context with multimodal large language models. In ICLR,

2024. 3

- [57] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In ICLR, 2023. 2, 3, 5, 6, 1
- [58] Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, et al. Unicontrol: A unified diffusion model for controllable visual generation in the wild. arXiv:2305.11147, 2023. 3
- [59] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 3
- [60] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In NeurIPS, 2024. 2, 3, 5
- [61] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020. 3
- [62] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv:2204.06125, 2022. 3
- [63] Jiawei Ren, Mingyuan Zhang, Cunjun Yu, and Ziwei Liu. Balanced mse for imbalanced visual regression. In CVPR,

2022. 5, 2

- [64] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 3
- [65] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2020. 3
- [66] Kunpeng Song, Yizhe Zhu, Bingchen Liu, Qing Yan, Ahmed Elgammal, and Xiao Yang. Moma: Multimodal llm adapter for fast personalized image generation. In ECCV, 2024. 3
- [67] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In CVPR, 2024. 2, 3, 5, 6, 1
- [68] Haofan Wang, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards stylepreserving in text-to-image generation. arXiv:2404.02733,

2024. 5

- [69] Jiangshan Wang, Yue Ma, Jiayi Guo, Yicheng Xiao, Gao Huang, and Xiu Li. Cove: Unleashing the diffusion feature correspondence for consistent video editing. In NeurIPS,

2024. 3

- [70] Tsung-Han Wu, Long Lian, Joseph E Gonzalez, Boyi Li, and Trevor Darrell. Self-correcting llm-controlled diffusion models. In CVPR, 2024. 2, 3, 4, 5, 7

- [71] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv:2306.09341, 2023. 3, 5, 6, 1
- [72] XLabs-AI. X-flux. https://github.com/XLabsAI/x-flux, 2024. 3, 5, 1
- [73] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. arXiv:2304.05977, 2023. 3
- [74] Xingqian Xu, Zhangyang Wang, Eric Zhang, Kai Wang, and Humphrey Shi. Versatile Diffusion: Text, images and variations all in one diffusion model. arXiv:2211.08332, 2022. 3
- [75] Xingqian Xu, Jiayi Guo, Zhangyang Wang, Gao Huang, Irfan Essa, and Humphrey Shi. Prompt-free diffusion: Taking” text” out of text-to-image diffusion models. In CVPR, 2024. 3
- [76] Kai Yang, Jian Tao, Jiafei Lyu, Chunjiang Ge, Jiaxin Chen, Weihan Shen, Xiaolong Zhu, and Xiu Li. Using human feedback to fine-tune diffusion models without any reward model. In CVPR, 2024. 3
- [77] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and CUI Bin. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. In ICML, 2024. 3
- [78] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. IPAdapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv:2308.06721, 2023. 3, 5, 6, 1
- [79] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. TMLR,

2022. 5, 6, 7, 1

- [80] Huizhuo Yuan, Zixiang Chen, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning of diffusion models for text-to-image generation. arXiv:2402.10210, 2024. 2, 3, 5
- [81] Gong Zhang, Kihyuk Sohn, Meera Hahn, Humphrey Shi, and Irfan Essa. Finestyle: Fine-grained controllable style personalization for text-to-image models. In NeurIPS, 2024. 3
- [82] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 3
- [83] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K Wong. UniControlNet: All-in-one control to text-to-image diffusion models. In NeurIPS, 2023. 3
- [84] Zhuofan Zong, Dongzhi Jiang, Bingqi Ma, Guanglu Song, Hao Shao, Dazhong Shen, Yu Liu, and Hongsheng Li. Easyref: Omni-generalized group image reference for diffusion models via multimodal llm. In ICML, 2024. 3

## IMG: Calibrating Diffusion Models via Implicit Multimodal Guidance Supplementary Material

### A. Implementation Details

[Figure 75]

#### A.1. Baselines and Models.

Our experiments are based on two diffusion models: SDXL [57], a widely adopted base diffusion model for alignment tasks, and FLUX.1 [dev] (FLUX) [37], a recent state-of-the-art flow-matching-based diffusion transformer. To compare with finetuning-based methods, we use the top-performing finetuned variant of SDXL, SDXL-DPO, which applies the Diffusion-DPO [67] technique, demonstrating the superiority of IMG and its compatibility with finetuning-based methods. For comparison with editingbased methods, we adopt the leading SLD as our baseline to highlight the advantages of IMG in visual comprehension and aesthetic quality. We further compare IMG with leading compositional generation methods, ELLA [26] and CoMat [33], to evaluate the compositional generation capabilities. For MLLM, we finetune LLaVA 1.5-13b [42] on the Instruct-Pix2Pix dataset [6] for 1 epoch, using the finetuning task format shown in Fig. 11, and extract features from the last hidden layer for guidance. We use the IP-Adapter [72, 78], trained on SDXL and FLUX, to enable image prompts and extract image features, with the image prompt scale set to 0.2. The Implicit Aligner takes both MLLM and image features as input and is implemented as a stack of 4 cross-attention layers and 2 linear layers. A detailed illustrative diagram of Implicit Aligner is shown in Fig. 9, accompanied by its execution pseudo code in Fig. 10. During inference, we execute the forward pass 3 times to enhance performance.

Figure 10. Pseudo code of Implicit Aligner. Our Implict Aligner (1) projects MLLM features to the same dimension as image features; (2) conducts cross-attention between initial image features and projected MLLM features; and (3) processes attention outputs with a linear layer as aligned image features.

851K pairs of preferred and unpreferred images generated under specific prompts. The preference labels are annotated by human observers. To determine the optimal training scheme and hyperparameters, we conduct ablation studies by evaluating the average Pick Score [36] across generated images using 500 unique prompts from the Pick-a-Pic test set. The Pick Score is a caption-aware preference scoring model trained on Pick-a-Pic. For evaluation, we report Human Preference Scores v2 (HPS v2) across generated images on the Human Preference Datasets v2 (HPD v2) test set [71], which includes 3,400 prompts across five categories, as well as the Parpi-Prompts [79], a diverse dataset of 1,632 prompts ranging from brief concepts to complex sentences. HPS v2 is a caption-aware preference scoring model trained on HPD v2. We also report results on the T2I-CompBench [28], which contains 1800 test prompts to validate compositional image generation capabilities. For each test in user studies, 33 evaluators were asked to do an A-B test on 30 random image pairs generated by the base model and IMG with the same prompt. Each unique pair was assessed by 3 evaluators, and only fully consistent votes were used to compute the final win rates. For MLLM finetuning, we extract triplets of {Original Image, Edited Prompt, Edit Instruction} from the CLIP-filtered InstructPix2Pix dataset [6], which contains 313K samples.

MLLM Guidance Features

Initial Image Features

| | |
|---|---|
| | |

Linear

As queries

Projected MLLM Features

4 × Cross-Attention

As keys and values

Linear

Implicit Aligner

Aligned Image Features

#### A.3. MLLM Finetuning.

To customize a pretrained MLLM as a misalignment detector, we finetune LLaVA 1.5-13b [42] on the InstructPix2Pix dataset [6] for 1 epoch. We use training triplets consisting of original images I0, edited prompts T1, and edit instructions TE. While I0 and T1 are fed into the MLLM as inputs, we prompt the model to describe the alignment by asking questions such as, ’How can the <Original Image> match the intended prompt: <Edited Prompt>?’, and supervise the model’s outputs against TE (see Fig. 11). To

Figure 9. Detailed architecture of Implicit Aligner. Our Implicit Aligner contains 4 cross-attention layers and 2 linear layers. The number of color cubes here represents the token dimensions rather than the number of tokens.

#### A.2. Datasets and Benchmarks.

For Implicit Aligner training, we use the same Pick-a-Pic training set [36] as Diffusion-DPO [67], which consists of

updating step k in Sec. 3.4 is set to 10. The training process takes about 10-15 hours. For evaluation, we set classifierfree guidance [24] to 7.5 for SDXL and SDXL-DPO, and 3.5 for FLUX. Sampling steps are set to 50 for SDXL and SDXL-DPO, and 30 for FLUX. The MLLM in IMG consumes about 4% additional inference time and 15G (QwenVL-7B) - 25G (LLaVA-13B) GPU memory.

Edited Prompt: A photography of a polar bear running on the snow.

Original Image

[Figure 76]

How to make the <Original Image> match the intended prompt: <Edited Prompt>?

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Edit Instruction: Turn the wolf into a polar bear.

[Figure 81]

[Figure 82]

[Figure 83]

MLLM

### B. Objective Derivation

- Figure 11. MLLM finetuning on instruction-based image data. We conduct finetuning on {Original Image, Edited Prompt, Edit Instruction} triplets from image editing datasets [6] to enhance MLLM’s comprehension on prompt-image misalignments.

[Figure 84]

[Figure 85]

[Figure 86]

What method will align the image with the prompt: A rubber ball and a metallic car?

[Figure 87]

To align the image with the prompt, one can use the following method:

- 1. Start by placing the rubber ball on the table.
- 2. Position the metallic car so that it is directly above the rubber ball.
- 3. Adjust the angle and distance of the car until it appears as if the car is driving over the rubber ball.
- 4. Take the photo from a perspective that captures both the rubber ball and the metallic car in the frame. By following these steps, the image will show a rubber ball and a metallic car in a way that aligns with the prompt.

[Figure 88]

Make the ball a rubber ball.

Original MLLM

Finetuned MLLM

- Figure 12. Text response comparison of the original MLLM and our finetuned MLLM. The original MLLM primarily outlines an image generation process based on the prompt, while our finetuned MLLM emphasizes aligning the input image with the provided prompt, showcasing its misalignment detection capability.

This section presents the detailed derivation of our proposed Iteratively Updated Preference Objective L in Sec. 3.4, which is a combination of a basic objective Lbase and a preference objective Lpref. To enhance generality and clarity, we substitute the cwI and clI in Sec. 3.4 with more general forms, xw and xl. These denote the preferred and non-preferred outputs of a regression model fθ (the Implicit Aligner in IMG), under a given condition c. In essence, the training procedure operates on triplets {c,xw,xl}.

#### B.1. Basic Objective

The primary goal of fθ is to predict the preferred sample xw, given the condition c, as formalized in Eq. 4:

w||xw − fθ(c)||22. (8)

Lbase = Ec,x

Minimizing the above Mean Square Error(MSE) is a well-established approach, equivalent to performing maximum likelihood estimation (MLE) in regression settings [39, 52, 63]. Within this framework, fθ(c) predicts the mean of a noisy distribution, which is assumed to follow a Gaussian distribution with constant variance σI, consistent with the probabilistic interpretation [48]:

pθ(xw|c) = N(xw|fθ(c),σI). (9)

The MSE in Eq. 8 equals the negative log-likelihood (NLL) of pθ(xw|c) [52]. Consequently, training the regression model fθ using MSE implicitly enables it to approximate the conditional data distribution pdata(xw|c).

prevent overfitting, we randomly select one of 100 different misalignment detection questions for each sample. The fine-tuning hyperparameters follow the standard configurations in [42]. In Fig. 12, we compare the text responses of the original MLLM and our fine-tuned MLLM. The original MLLM primarily outlines an image generation process based on the prompt, while our finetuned MLLM emphasizes aligning the input image with the provided prompt, showcasing its misalignment detection capability.

#### B.2. Preference Objective

Besides the basics, we also draw inspiration from direct preference optimization (DPO) [67] and self-play finetuning (SPIN) [80] to enhance alignment. These preference learning techniques adhere to a common RLHF principle [60]: optimize the conditional distribution pθ(x|c) to maximize a latent reward model r(c,x), while regularizing the KL-divergence from a reference distribution pref:

#### A.4. IMG Training and Evaluation.

Ec,x[r(c,x)] − ηKL (pθ(x|c)||pref(x|c)). (10)

max

pθ

Our Implicit Aligner is trained on the Pick-a-Pic training set [36] for 100K iterations with 8 A100 GPUs and a batch size of 8. We use the AdamW [45] optimizer with a constant learning rate of 1×10−4 and a weight decay of 1×10−4. The ratio parameter in Eq. 7 is set to 1. The reference model

Here pθ and pref are prediction distributions of fθ and fref, respectively, where fref is a copy of fθ from an earlier training iteration, as defined in Eq. 9. The hyperparameter η controls the strength of the regularization.

As demonstrated in [60], the unique global optimal solution of pθ(x|c) in Eq. 10 is expressed as:

which aligns with Eq. 5. Using the equivalence between MSE and NLL under the Gaussian prior, as discussed in Sec. B.1, we obtain a simplified version of Lpref for implementation as follows:

pθ(x|c) = pref(x|c)exp(r(c,x)/η)/Z(c), (11) where Z(c) = x

pref(x0|c)exp(r(c,x0)/η) is the partition function. The reward model is reformulated as:

w,xl[ℓ(−[2(||xw − fθ(c)||22 − ||xw − fref(c)||22) − (||xl − fθ(c)||22 − ||xl − fref(c)||22) − ||fref(c) − fθ(c)||22])],

Lpref = Ec,x

0

pθ(x|c) pref(x|c)

+ η log Z(c). (12)

r(c,x) = η log

(17) which is consistent with Eq. 6. As discussed in Sec. 3.4, the reference model fref is iteratively updated. Specifically, we first randomly initialize fref and later iteratively copy fθ to fref whenever fθ outperforms fref. In practice, we execute the substitution when fθ(c) is closer to xw than fref(c) for k consecutive iterations, i.e.,

From the perspective of integral probability metric (IPM) [51], DPO [67] maximizes the reward gap between preferred and non-preferred data distributions, while SPIN [80] maximizes the reward gap between preferred data distribution and reference data distribution, i.e., xref = fref(c) ∼ pref(x|c). As introduced in Sec. 3.4, we establish a combined objective of DPO and SPIN:

||xw − fθ(c)||22 < ||xw − fref(c)||22. (18)

w,xl,xref[r(c,xw) − r(c,xl)

max

Ec,x

To summarize, The final Iteratively Updated Preference Objective is a combination of Lbase and Lpref, weighted by a ratio parameter λ:

r

DPO

(13)

##### + µ(r(c,xw) − r(c,xref)

)],

SPIN

L = Lbase + λLpref. (19)

where µ is a hyperparameter that controls the trade-off. As demonstrated by [8], a more general form of the optimization problem in Eq. 13 is:

### C. Additional Quantitative Results

In Tab. 6, we present additional quantitative results on GenEval [17] and DPGBench [26]. IMG shows consistent improvements across two benchmarks.

w,xl,xref[ℓ(r(c,xw) − r(c,xl)

min

Ec,x

(14)

r

+ µ(r(c,xw) − r(c,xref)))],

Model GenEval↑ DPGBench↑

where ℓ represents any monotonically decreasing convex loss function. Eq. 13 can be viewed as the maximization version of Eq. 14, where l(a) = −a. However, using such a linear loss function leads to an unbounded objective value, which may cause undesirable negative infinite values of r(c,xl) and r(c,xref) during continuous training. To address this issue, we adopt a logistic loss function as suggested by [67, 80]:

SDXL-DPO 0.59 76.81 SDXL-DPO + IMG (Ours) 0.61 78.72

FLUX 0.68 80.60 FLUX + IMG (Ours) 0.70 82.77

Table 6. Results on GenEval [17] and DPGBench [26].

### D. Additional Qualitative Results

l(a) := −log sigmoid(a) = log(1 + exp(−a)), (15)

- In Fig. 13, we compare IMG with leading MLLM-based image editing methods [15, 30]. IMG showcases better alignment performance and visual quality.
- In Fig. 14 and Fig. 15, we present additional qualitative

which is non-negative, smooth, and exhibits an exponentially decaying tail as a → ∞. The logistic loss function helps prevent the excessive growth of the reward value r, ensuring a stable training process.

results to show the superior prompt adherence and aesthetic quality achieved by integrating IMG with various models.

By substituting the reward model r in Eq. 14 with Eq. 12 and empirically setting η and µ to 1, we obtain the final preference objective as follows:

A rubber ball and a metallic car. (Instruction: Make the ball a rubber ball)

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

pθ(xw|c) pref(xw|c) − log

pθ(xl|c) pref(xl|c)

Lpref = Ec,x

w,xl,xref ℓ log

Initial Generation MIGE SmartEdit

IMG (Ours)

pθ(xref|c) pref(xref|c)

pθ(xw|c) pref(xw|c) − log

,

+log

Figure 13. Comparison between MLLM-based editing and IMG.

(16)

A green banana and a brown horse. A blue cup and a green cell phone.

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

A woman is holding a yoga mat and heading to a class. A fish eating a pelican.

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

A fish near a car. A milk container in a refrigerator.

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

FLUX FLUX + IMG (Ours) FLUX FLUX + IMG (Ours)

Figure 14. Additional qualitative results by integrating IMG with FLUX.

A leather jacket and a glass vase. A green apple and a red kiwi.

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

A dog is standing on its hind legs and trying to catch a frisbee.

Two kittens curled up in a white sheet that looks soft.

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

A giant cat sleeps in the middle of a StarCraft 2 map. A sheep to the right of a wine glass.

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

SDXL SDXL + IMG (Ours) SDXL-DPO SDXL-DPO + IMG (Ours)

Figure 15. Additional qualitative results by integrating IMG with SDXL and SDXL-DPO.

