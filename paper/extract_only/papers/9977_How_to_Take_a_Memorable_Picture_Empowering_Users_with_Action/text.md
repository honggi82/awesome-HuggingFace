## arXiv:2602.21877v2[cs.CV]19Mar2026

### How to Take a Memorable Picture? Empowering Users with Actionable Feedback

Francesco Laiti1,2,3 Davide Talon4 Jacopo Staiano1 Elisa Ricci1,4

1University of Trento 2University of Pisa 3Travelbrain srl 4Fondazione Bruno Kessler laitifranz.github.io/MemCoach

[Figure 1]

[Figure 2]

###### Abstract

Memorability 0.92 👍

Memorability 0.64 👎

[Figure 3]

[Figure 4]

Image memorability, i.e., how likely an image is to be remembered, has traditionally been studied in computer vision either as a passive prediction task, with models regressing a scalar score, or with generative methods altering the visual input to boost the image likelihood of being remembered. Yet, none of these paradigms supports users at capture time, when the crucial question is how to improve a photo memorability. We introduce the task of Memorability Feedback (MemFeed), where an automated model should provide actionable, human-interpretable guidance to users with the goal to enhance an image future recall. We also present MemCoach, the first approach designed to provide concrete suggestions in natural language for memorability improvement (e.g., “emphasize facial expression,” “bring the subject forward”). Our method, based on Multimodal Large Language Models (MLLMs), is training-free and employs a teacher-student steering strategy, aligning the model internal activations toward more memorable patterns learned from a teacher model progressing along least-to-most memorable samples. To enable systematic evaluation on this novel task, we further introduce MemBench, a new benchmark featuring sequence-aligned photoshoots with annotated memorability scores. Our experiments, considering multiple MLLMs, demonstrate the effectiveness of MemCoach, showing consistently improved performance over several zero-shot models. The results indicate that memorability can not only be predicted but also taught and instructed, shifting the focus from mere prediction to actionable feedback for human creators.

[Figure 5]

# 🎉

v

[Figure 6]

###### Bring them closer, both should smile and face each other

MemCoach

Figure 1. Given an input photo, memorability feedback aims to generate natural-language suggestions to guide users toward a more memorable shot. MemCoach provides memorability-aware feedback, effectively assisting users to capture memorable images.

els trained to regress a scalar memorability score from images [17, 32, 62] and explaining what makes an image memorable [26–28]. These works identified key intrinsic factors such as the presence of people [13], indoor scenes [9], or emotional expressions [9], rather than objects and panoramic views [27], as well as extrinsic ones, including context and the observer [8]. More recent generative approaches have attempted to manipulate memorability, leveraging editing models to automatically enhance an image’s likelihood of being remembered [19, 59]. However, these paradigms are inherently passive and opaque: prediction models merely report how memorable an image is, while generative models directly alter the image, losing control on the changes. In contrast, when taking a picture, humans seek actionable feedback: “What should I change in this shot to make it more memorable?”, rather than a numerical score or an automated edit, especially considering that, as humans, we generally fail to judge what is memorable [28]. Similarly, in the context of computational photography, scoring models have been developed to assess the quality of images [41, 72] or to produce free-form critiques that are often verbose and difficult to operationalize as constructive feedback [53].

###### 1. Introduction

Memorability, i.e., the likelihood that an image will be remembered by human observers, is an intrinsic property of a picture that can be predicted from visual content alone [8, 26, 27, 30]. Previous research has largely focused on measuring this property by introducing prediction mod-

To address this gap, we introduce Memorability Feedback (MemFeed), the task of providing users with actionable and interpretable feedback to improve image memorability. Instead of predicting or editing, an automated model is used to provide guiding feedback: given a user’s image, it generates natural-language suggestions describing concrete compositional or semantic changes that could increase memorability (e.g., “bring the subjects closer”, “the subjects should smile and face each other”), effectively verbalising how to improve the shot in terms of memorability (see Fig. 1). Leveraging the reasoning and vision-language capabilities of Multimodal Large Language Models (MLLMs), we propose MemCoach, a novel approach that bridges perceptual memorability research and photographic assistance. MemCoach employs a training-free steering strategy that redirects MLLM activations toward memorability-aware feedback, i.e., suggestions enhancing memorability, as distilled from a teacher model indicating how to transition from less to more memorable images across multiple views of the same scene. This contrasts with the model’s default neutral feedback, which lacks memorability awareness.

To evaluate methods on the novel task of MemFeed, we introduce MemBench, a new benchmark based on the PPR10K [43] dataset. It includes multiple images from the same photoshoot, each annotated with its memorability score. The proposed evaluation metrics are based on the quality of the model feedback, i.e., the memorability difference between the image the model is currently observing and the one after feedback implementation (as estimated by an editing model), as well as their perplexity on ground-truth effective feedback. Across four open-source MLLMs, our experiments show that MemCoach consistently enhances performance over standard zero-shot models.

###### Our contribution is three-fold:

- • We investigate and formalize the task of Memorability Feedback, where a model should provide humanunderstandable and actionable feedback to a user on how to make a shoot more memorable. To the best of our knowledge, this problem has not been previously studied.
- • We introduce MemBench, a benchmark for memorability feedback training and evaluation.
- • We present MemCoach, a novel training-free method leveraging a teacher-student strategy and activation steering to inject memorability information for useful guidance. Our results show that MemCoach can be effectively applied to multiple MLLMs.

###### 2. Related Work

Memorability. Memorability refers to the probability that an observer will recall an image or a video after a quick view of it [8, 26–28, 32]. Early research [26, 27] revealed that this is not a subjective phenomenon: memorability is a quantifiable property of visual content that is stable across observers.

This property holds for both images [18, 27, 32, 38, 48, 52, 74] and videos [12, 14, 34, 36, 47, 48, 58]. Thus, the community has focused on understanding the intrinsic property underlying memorable visual content, finding that semantics plays a crucial role with faces and animals [13], things [13], indoor [9] or less cluttered scenes [19] and images conveying negative emotions [9] having an increased memorability score. This is in stark contrast with the original belief that natural vistas and aesthetic beauty make an image memorable [26]. Furthermore, researchers have investigated the influence of extrinsic factors like visual context, eye movements and the role of the observer [8]. Most similar to our work, [19, 31, 59, 60] leverage editing models for increasing the memorability of images at hand. In contrast, our goal is to develop models that provide users with natural language feedback on how to enhance an image’s memorability.

Photographic feedback. Recent efforts [24, 54, 71] focus on curating photograph datasets annotated with professional critiques and aesthetic feedback. Models trained on these data can explain compositional strengths and weaknesses in natural language, providing users with critique-like feedback. However, they fall short on translating critique into concrete, actionable instructions that a user can execute at the moment of shooting. Research on photographic guidance has largely focused on aesthetic scoring or rule-based feedback [29, 42, 73] (e.g., rule of thirds) with overlays to assist the user. Similarly, works like [15, 44, 46] propose diversified views and adaptive composition grids to improve image quality. While effective for novice photographers, these approaches mainly offer static rule enforcement or post-hoc critique rather than adaptive, scene-specific coaching. Most recently, [20] highlights the increasing demand for interactive photographic guidance. However, such systems are proprietary, and a formalized framework, including publicly available benchmarking data and evaluation metrics, is still lacking in the literature.

MLLMs and steering. Starting from early approaches limited to learning a shared embedding space where visual and textual representations are aligned [55, 77], recent research efforts have focused on generative methods capable of coherent question-answering [2, 4, 23, 40, 45, 78]. Under the linear activations hypothesis [51], steering approaches [56, 65, 79] show that model behaviour can be modified via linear displacements of model intermediate representations [56, 65, 79]. Steering typically involves building contrasting sample sets differing in a target concept, computing a mean-difference vector, and adding or subtracting it from activations to control the concept at inference time [6, 11, 16, 56, 57, 64]. In contrast with these works, we design a teacher-student steering approach for actionable memorability feedback. To the best of our knowledge, MemCoach is the first activation steering strategy for MLLMs applied to perceptual tasks.

Data Pipeline

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

| | |
|---|---|
| | |
| | |

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Predictor dss

FEEDBACK

[Figure 18]

(a) Collect images from a scene

(b) Regress memorability

(c) Rank images within scene

(d) Actions generation

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

FEEDBACK Predictor dss

[Figure 24]

[Figure 25]

[Figure 26]

Memorability analysis IR - RM

Input: image + feedback

Editing model generation

Edited image

[Figure 27]

FEEDBACK

Editing Evaluation Perplexity Evaluation

- Figure 2. Overview of MemBench generation and evaluation. Top: Data pipeline for constructing MemBench, including scene grouping, memorability regression, image ranking, and generation of actionable memorability-aware feedback. Bottom: Evaluation pipeline assessing feedback quality through editing-based memorability improvement and perplexity scoring.

###### 3. Memorability Feedback

depicted in Fig. 2. We start by collecting images from the same scene and group them together (Fig. 2-a). In a second step, for each image we evaluate its memorability by means of a predictor M, a pre-trained regressor (Fig. 2-b) built upon CLIP [55] features and trained on publicly available memorability datasets [18, 27, 32], reaching state-of-the-art performance. See Supp. Mat. for predictor M details. Once images are associated to the corresponding memorability score, photographs within the same scene are ranked and pairs (xS,xD) are constructed from less to more memorable images (Fig. 2-c).

In this section, we formally define the task of Memorability Feedback and present MemBench, a novel benchmark to test the effectiveness of models in providing actionable and human-interpretable guidance to take memorable images.

###### 3.1. Task Definition

We frame the memorability feedback task as a transformation problem over visual content. Given a source image xS with an associated memorability score mS ∈ [0,1], the objective is to design an automated model capable of generating a natural language actionable feedback a that, when implemented on xS, would get to the destination image xD, such that the resulting memorability score mD satisfies mD > mS (Fig. 1). Here, we assume mD = M(xD) and mS = M(xS) are estimated by a memorability prediction model M. This task departs from conventional memorability prediction, as it requires models not only to assess the current memorability level, but to proactively identify and verbalize actions capable of increasing it. The generated feedback must be both semantically grounded in the visual content and operationally feasible. In this formulation, success depends on the model’s ability to reason about image properties that influence human memory and to translate such reasoning into targeted and constructive guidance.

Extracting actionable memorability feedback. For each image pair (xS,xD), we prompt a captioning model ψ that allows for interleaved images, to describe the feedback a necessary to transform the source into the destination image (Fig. 2-d): a = ψ(xS,xD,pa) where pa is the feedback elicitation prompt: “Determine the actions required to transform ⟨xS⟩ into ⟨xD⟩”. Contrary to computational photography adjustments focusing on post-hoc corrections (e.g., “make the image brighter”), we focus on semantic actions that a user can take on-the-fly for a better shot, e.g., “Face each other” (see Supp. Mat. for qualitative samples). We rely on INTERNVL3.5 8B [67] as captioning model.

Benchmark statistics. MemBench comprises approximately 10K images grouped into 1,570 scenes, with an average of 6.5 images per scene. The word cloud in Fig. 3a illustrates the most frequent terms appearing in the collected feedback. As shown, suggestions span a wide range of semantic categories, including references to body parts (e.g., “hand”, “face”), verbs (e.g., “holding”, “remove”), and photographic concepts (e.g., “background”, “lighting”). Source images exhibit an average memorability score of 0.63, while the most memorable images within the same scene range between [0.51, 1.0], indicating some overlap between the two distributions (Fig. 3-b). Feedback varies in length, ranging from 7 to 102 words (Fig. 3-c). Finally, in Fig. 3-d, we categorize atomic sub-actions in the feedback

###### 3.2. MemBench

We introduce MemBench, a benchmark for memorabilityaware feedback. MemBench builds upon PPR10K [43] by augmenting image pairs, of the same scene, with naturallanguage semantic action descriptions that specify how the visual content differs between a lower-memorability image and a higher-memorability counterpart.

Data pipeline. We built upon PPR10K [43], a portrait photo retouching dataset with several different scenes. PPR10K offers multiple shoots per scene, where each taken photograph may differ both in subjects and composition as well as framing and lighting. A visualization of the data pipeline is

- (a)WordCloud

- (b) Source-target Memorability (c) Feedback Length

[Figure 28]

(d) Categorization & Co-occurrence

- Figure 3. MemBench statistics. Data analysis in terms of (a) most frequent words; (b) distribution of memorability scores for the least and most memorable images within each scene; (c) feedback length as measured by content words; and (d) categorization of atomic sub-actions, where the width of each chord indicates the frequency of co-occurrence between categories.

using GPT-5-MINI as an automatic annotator and report their co-occurrence patterns (see Supp. Mat.). Most sub-actions relate to subject posing, followed by semantic adjustments, while co-occurrence statistics highlight strong correlations between framing and posing and the interplay between lighting and semantic changes.

Evaluation protocol. As we propose a novel task, we also introduce evaluation metrics for Memorability Feedback (see Fig.2-bottom), covering two main axes: real world effectiveness and likelihood of memorable actions. On the one hand, editing metrics probe the effectiveness of provided feedback by emulating real-world user behavior; we use FLUX.1 KONTEXT [37] as in-context image editing model e(·,·) which applies the guidelines provided by the memorability feedback: starting from the source image xS and the feedback a, the destination image is obtained as edited output xˆD = e(xS,a). Hence, IMPROVEMENT RATIO (IR) evaluates the fraction of time the edited image has larger memorability than the source one, i.e., IR = x

D

1[mD ≥ mS], with 1[c] the indicator function evaluating the satisfaction of the c condition. Instead, RELATIVE MEMORABILITY (RM) is defined as: RM = (mD − mS)/mS, accounting for relative memorability improvements. In the Supp. Mat., we report experiments with different editing models and a different memorability predictor. On the other hand, we evaluate the likelihood that a model provides improving memorability feedback by computing the PERPLEXITY on ground truth memorability-aware feedback from the same captioning model. We use an 80-20 train/test scenes split, evaluating the feedback model only on scenes not seen during training.

- 4. Method

However, when naively prompted, MLLMs lack a concrete understanding of what makes an image memorable (Sec.4.1). In Sec. 4.2 we hence describe how to enable a multimodal large language model to effectively perform MemFeed.

###### 4.1. MLLMs Lack Memorability Understanding

Since even humans provide inconsistent judgments of memorability [28], we first investigate whether contemporary MLLMs are able to capture the underlying factors that make an image memorable. We conduct a preliminary study on the LaMem dataset [32], where each image is annotated with its memorability score. Specifically, we prompt recent MLLMs with a simple question, by asking whether a given image is memorable “Is this image memorable? Output only yes or no.” and interpret the likelihood of the yes token with respect to the no token as the predicted memorability score. Following prior works [18, 27, 32], we evaluate the results in terms of Spearman’s rank correlation [61] against the ground-truth scores. As shown in Tab. 1, despite extensive pretraining, MLLMs exhibit no correlation with human annotations, remaining far below the cross-annotator consistency upper bound. Consequently, they also fail to provide reliable or effective feedback for enhancing memorability (see Fig. 4). Indeed, we observe a marginal IR improvement for all zero-shot models when prompted with pm = “Determine the actions required to improve the memorability of ⟨xS⟩” compared to the EDITING BASELINE, implemented by providing an empty-string instruction to the editing model e(·,“ ”), leaving the image unaltered except for the model’s default bias.

###### 4.2. MemCoach

Our goal is to design a model with the ability to provide actionable feedback or, in other words, suggestions that when applied to a user’s photo, can enhance its memorability. Given their capability to jointly interpret visual inputs and generate coherent textual descriptions, multimodal large language models are naturally well-suited for this task.

We introduce MemCoach, a training-free approach to elicit memorability feedback in state-of-the-art MLLMs thanks to a novel knowledge-distillation activation steering strategy.

Method overview. Fig. 5 depicts our approach. In the initial contrasting data generation step (Fig. 5-left), MemCoach

Spearman Rank (↑)

Model

Inter-annotator∗ 0.68 QWEN2.5VL [5] -0.06 INTERNVL3.5 [67] -0.01 IDEFICS3 [39] -0.07 LLAVA-OV [3] 0.08

Editing Baseline

0.68

|Qwe<br><br>Inte<br><br>Idef<br><br>LLaV|n2.<br><br>rnVL<br><br>ics3<br><br>A-O|5VL<br><br>3.5<br><br>V| | | | |0|.68<br><br>0.7<br><br>0.7<br><br>0.70|3<br><br>3| |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

IR

- Table 1 & Figure 4. MLLMs lack memorability understanding. Left: Memorability prediction performance in terms of Spearman’s Rank Correlation (↑). (∗) is reported from [32]. Right: Improvement ratio of zero-shot models with respect to the EDITING BASELINE, marginal improvement is observed.

leverages multiple images corresponding to the same scene to construct a paired dataset where the default behaviour of a student MLLM asked to provide memorability feedback (i.e., neutral feedback) is compared to the one of a teacher model generating actions that will transform the source image to a destination image that is known to be more memorable (i.e., memorability-aware feedback). Then, the second steering vector extraction step (Fig. 5-center) extracts a memorability steering vector on student activations to capture the latent-space deviations introduced by memorability-aware feedback. Finally, at inference time (Fig. 5-right), the MLLM steering step uses such vector to shift the student model activations toward more effective suggestions.

Contrasting data generation. We build paired memorability feedback samples based on the difference in memorability improvement that they induce.

Formally, consider a dataset D = {(Xi)}Ni=1 where for

each scene i, the set of images Xi = {xi1,...,xiM} are captured within the same shooting session. Our goal is to

generate feedback pairs (f+i ,f−i ). f+i corresponds to the memorability-aware feedback provided by a teacher model which effectively describes how to get to more memorable images. f−i corresponds to the default student behavior when asked to suggest for improved memorability. To this end, we follow the data generation pipeline in Sec. 3. Each image is evaluated for its memorability score with M and ranked accordingly. Consider xiS as the least memorable image in Xi, i.e., the source image we want to provide feedback on, and xiD the most memorable image within the set, or, in other words, the desired output we would like to get with the provided feedback. Let the teacher model ϕteach be a MLLM that, when observing a pair of images (x,x′), elicits the corresponding actions to move from image x to image x′ and ϕstud the student model we are interested to enable for effective feedback for memorability on an observed image x. On the one hand, we leverage the teacher model to extract memorability-aware feedback f+i = ϕteach(xiS,xiD,pa), with pa the feedback elicitation prompt in Sec. 3, yielding

the actionable instructions on how to move from xiS to xiD and consequently, improve the current image memorability. On the other side, we collect ϕstud default neutral feedback, f−i = ϕstud(xiS,pm), where pm is the memorability feedback prompt in Sec. 4.1. We construct paired contrasting data as:

F+ = {f+i }i, F− = {f−i }i, (1) with i = 1,...,N. In summary, paired data in Eq. (1) captures the discrepancy between student-default and teacherprivileged memorability-aware feedback: for the same source image xS the privileged knowledge of the memorability target is opposed to the student’s default uninformed suggestions.

Steering vector extraction. Starting from the available contrasting data, this step aims to characterize the student activation-space directions capturing the systematic shift between memorability-aware feedback and neutral one.

Despite both sets providing valid suggestions on the source image xiS, feedback in F+ improves memorability, whereas the ones in F− have limited effect. Inspired by steering strategies [65], we therefore leverage their discrepancies to disentangle the factors that improve memorability. To this end, we construct the input to the student model by using:

f+i = {(xiS,pm,f+i )}f+i ∈F+, (2) f−i = {(xiS,pm,f−i )}f−i ∈F−,

where f+i and f−i are placed in the assistant turn of the chat template, paired with the same source image xiS and prompt pm, thus inducing different responses for identical inputs. Then, we independently feed f+i and f−i to the student model to collect its activations on the two different types of feedback. Let define h(l) as the activation of ϕstud at layer l = 1,...,L, where L is the number of its layers, and let hi,+(l) and hi,−(l) denote the aware and neutral feedback activations for the i-th sample at layer l. We extract the memorability steering vector r(l) at layer l as:

1 N

r(l) =

N

hi,+(l) − hi,−(l), (3)

i=1

This vector characterizes the shift between memorabilityaware and neutral feedback in the model activation space, acting as a distilled representation of the teacher’s privileged knowledge, later used to steer the uninformed student toward more effective memorability guidance.

Inference with MLLM steering. At inference time, we aim to endow the student model with the capability to improve memorability, without relying on the teacher privileged information. Given a user-provided image x and the memorability instruction prompt pm, we first compute the

(c) Inference with MLLM Steering

(a) Constrasting Data Generation

(b) Steering Vector Extraction

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Mem0.85

Mem0.94

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

##### .

[Figure 37]

[Figure 38]

Teacher MLLM

Layer 1 Layer l

Layer 1 Layer l

Layer 1 Layer l

…

…

Paired contrasting data

+ …

[Figure 39]

…

…

Layer L

Layer L

[Figure 40]

Layer L

[Figure 41]

###### -

Mem0.85

Student MLLM

OUTPUT FEEDBACK

Seq tokens Steering vector

Avg

Stand straight, face forward with both hands holding the red object

[Figure 42]

Figure 5. Overview of the proposed method. (a) Contrasting data generation : paired samples are built by coupling the memorabilityaware guidance of a teacher MLLM with the neutral responses of a student MLLM on the same scene; (b) Steering vector extraction : activation differences between memorability-aware and neutral feedback are averaged to obtain a memorability steering vector capturing the latent shift toward effective suggestions for memorability; (c) Inference with MLLM steering : the student activations are shifted using the memorability steering vector to produce improved, memorability-oriented feedback without additional training.

student default activations h(l) and then steer the model by injecting the memorability steering vector r(l) extracted in the previous step. Formally, the activations are shifted as:

for image perceptual evaluation, namely Q-INSTRUCT [71] and AESEXPERT [24]. Finally, we also report EDITING BASELINE corresponding to the empty string as feedback proposed to the editing model (see Sec. 4.1).

h˜(l) = h(l) + α · r(l), (4)

Implementation details. Unless stated otherwise, we use INTERNVL3.5 8B [67] for both teacher and student models and employ the MemBench training split to generate contrasting examples. We fix the steering layer to l = 12 and the coefficient to α = 55, selected via tuning on a held-out subset of the training data. To ensure structured outputs, we adopt the outlines library [68] for constrained decoding (see Supp. Mat. for further details).

where α is a hyperparameter controlling the steering strength. Intuitively, Eq. (4) shifts the model’s intermediate representation toward the activation patterns observed when generating effective feedback, thereby distilling the teacher’s guidance into the student’s latent space. After steering, the forward propagation proceeds through the remaining layers, with subsequent feedback modulated by the steered activations, thereby altering the student behavior. Notably, this steering procedure is training-free, modelagnostic, and operates entirely at the activation level, making it compatible with any MLLM model that provides access to its intermediate representations.

###### 5.1. Quantitative Results

Table 2 reports the quantitative comparison of different MLLMs when asked for memorability feedback, as evaluated in terms of both editing metrics and perplexity. As can be noted, results highlight a consistent advantage of MemCoach across both axes of evaluation. Here we only report results for MemCoach when using INTERNVL3.5 model. We observe a marked increase in IR, indicating that feedback produced by the steered model more frequently leads to edits that raise the memorability of the resulting images. This gain is further confirmed by a higher RM, showing that the relative increase in memorability is not only more frequent but also larger. MemCoach yields a +5% IR with respect to the strongest zero-shot GPT-5 MINI [50] and +31.81% gain on the RM metric with respect to its base INTERNVL3.5 model. Importantly, despite its training-free nature, MemCoach outperforms state-of-the-art large-scale aesthetics-specialized approaches, showcasing the benefit of

###### 5. Experiments

Baselines. We consider a wide range of MLLMs models, including QWEN2.5 VL 7B [5], INTERNVL3.5 8B [67], IDEFICS3 8B [39], and LLAVA-OV 7B [3], under several configurations: as Teacher oracle, models take advantage of privileged information where more memorable destination images are fed as input together with the source image, and the MLLM should only focus on generating a feedback describing their difference; as zero-shot, instead, models are prompted with pm to generate suggestions (see Sec. 4.1); we include GPT-5 MINI [50] as a representative of proprietary models. For completeness, we compare with state-of-the-art aesthetics-specialized MLLMs trained

- Table 2. Comparison with state-of-the-art models. MemFeed performance of MemCoach when comparing to several

teacher oracle , zero-shot and aesthetics specialized MLLMs. MemCoach achieves the best results in the considered metrics. Best results in bold.

Editing Perplexity

Model

IR (↑) RM% (↑) (↓) Edit model 0.68 3.72 n.d. Teacher oracles LLAVA-OV [3] 0.74 5.93 5.73 IDEFICS3 [39] 0.80 9.84 29.21 QWEN2.5VL [5] 0.83 10.16 2.34 INTERNVL3.5 [67] 0.85 11.92 2.40 Aesthetics specialized

AESEXPERT [24] 0.73 6.67 5.97 Q-INSTRUCT [71] 0.73 5.31 5.36

Zero-shot baselines GPT-5 MINI [50] 0.75 7.03 n.d. LLAVA-OV [3] 0.70 5.87 7.58 IDEFICS3 [39] 0.73 6.64 20.19 QWEN2.5VL [5] 0.68 4.26 10.23 INTERNVL3.5 [67] 0.73 5.47 5.49 MemCoach (Ours) 0.80 7.21 4.99

the presented approach with respect to models trained on other perceptual metrics. Notably, MemCoach closes the gap of training-free strategies with teacher oracle baselines that take advantage of their privileged knowledge of the scene. Turning to the likelihood of ground-truth feedback, the lower perplexity achieved by MemCoach confirms its improved alignment with human-like memorability-aware feedback: the reduced uncertainty over ground-truth feedback suggests that the steered MLLM better captures the linguistic regularities associated with memorability-increasing suggestions. Preliminary user studies in the Supp. Mat. confirm MemCoach effectiveness and the quality of the provided feedback.

We then demonstrate that the integration of MemCoach into different multimodal backbones consistently enhances their ability to generate memorability-aware feedback. Results are shown in Tab. 3. In terms of IR, MemCoach yields consistent gains for all models, with the strongest improvement observed for QWEN2.5VL and LLAVA-OV.

###### 5.2. Qualitative Evaluation

We qualitatively analyze the feedback provided by MemCoach in Fig. 7, where source images observed by the model (left) are shown with the provided natural-language suggestions (bottom) and the imagined destination image (right), as generated by the in-context editing model. The examples highlight the variety of suggestions the model proposes, ranging from fine-grained compositional adjustments, such as altering gaze direction, pose, or hand position, to semantic interventions involving object removal or face expression change. Feedback is naturally interpretable and actionable,

Table 3. Generalization to different MLLMs. MemFeed performance of MemCoach when applied to different architectures. MemCoach generalizes to different models, enhancing their ability to produce memorability feedback.

Editing Perplexity IR (↑) RM% (↑) (↓)

Model

LLAVA-OV [3] 0.70 5.87 7.58 MemCoach-LLAVA 0.73 (+4.29%) 5.04 (-14.14%) 14.05 (+85.36%)

IDEFICS3 [39] 0.73 6.64 20.19 MemCoach-IDEFICS 0.75 (+2.74%) 6.69 (+0.75%) 19.81 (-1.88%)

QWEN2.5VL [5] 0.68 4.26 10.23 MemCoach-QWEN 0.74 (+8.82%) 5.49 (+28.87%) 13.90 (+35.87%)

INTERNVL3.5 [67] 0.73 5.47 5.49 MemCoach-INTERNVL 0.80 (+9.59%) 7.21 (+31.81%) 4.99 (-9.11%)

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Stand up

Hands on the hips

Hold with both hands

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Look at the camera Look at each other

Smile

Figure 6. Common feedback patterns on source images. MemCoach favors symmetric and socially connected compositions, reflecting principles of human photography.

expressed in concise textual instructions (mostly involving verbs “Bring”, “Stand”, “Remove”) that can be directly implemented, effectively verbalizing how to take a memorable picture. Interestingly, cases in the figure also expose trade-offs between normalization and distinctiveness. In line with previous memorability studies [19], positive cases often relate to conventional photographic strategies (e.g., centered framing, and minimal occlusion). Conversely, failure cases show the negative effect of removing semantically out-of-context elements (e.g., skulls, feathered headdresses), underscoring the dual nature of memorable images, where both clarity and the extrinsic notion distinctiveness [8] shape the MemFeed task.

Common feedback patterns. In Fig. 6, we analyze the most recurrent feedback patterns that MemCoach associates with improved memorability. Interestingly, these suggestions reveal an emergent understanding of photographic composition and social engagement. Many instructions promote symmetry and balance, such as “hold with both hands” or ‘‘hands on the hips”, which encourage centered and symmetric poses that naturally guide the viewer’s attention toward the subject [8, 35]. Others focus on directing the subjects’ gaze, such as “look at the camera” or “look at each other”, reinforcing its role as emotionally resonant cue.

+16.9%

+21.7%

+15.0%

+ 21.2%

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Look straight ahead, eyes open and facing forward with a neutral expression

Stand up, face forward, place hands on hips

Remove the cans from the hair, lower the hand from the chin

Stand straight, face forward, with both hands clasped in front

+11.4%

+10.7%

-3.7%

-12.2%

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Lie down, close the eyes, rest the hands on the chest

Remove the skull from the head, lower the arms to the sides

Lower the hand from the head, remove the feathered headdress

Bring the hands together, and rest the chin on the hands, remove the gloves

Figure 7. Qualitative feedback from MemCoach. For each source image (left), the model provides natural-language feedback (bottom) that is applied to produce the destination image (right). Each score represents the Relative Memorability (RM), indicating how suggested feedback affects memorability. MemCoach provides human-interpretable and actionable feedback that translates into semantic changes for overall improved memorability. Observed failure cases propose to remove out-of-context elements.

###### 5.3. Ablation Study

Data efficiency of steering. Figure 8-top compares the improvement ratio as a function of the available training data. In the low-data regime, MemCoach consistently outperforms Low-Rank [22] fine-tuning, showing that steering requires far fewer samples to capture memorability-relevant directions. With only 1% of the training data, MemCoach already reaches performance on par with full-data fine-tuning, while maintaining stable gains as more data become available.

Impact of main components. Tab. 4 analyzes the main design choices underlying MemCoach. QWENCONTRASTING reports model performance when the memorability-aware feedback in the contrasting data generation is extracted from a different teacher (QWEN2.5VL). As can be noted, steering continues to provide a positive effect, though with reduced marginal benefit compared to the INTERNVL3.5. Confirming the importance of per-sample contrast, the DIFF(MEAN) variant, which averages activations before differencing, yields lower editing performance (6.64 RM) than our subtraction-before-averaging formulation (7.21 RM), presented in Eq. 3. Finally, Fig. 8-bottom ablates the steering parameter α in terms of IR: performance improvement is initially observed with increasing coefficient values, with performance saturating with larger alphas.

Table 4. Ablation analysis. MemFeed performance of

MemCoach when ablating on the contrasting data teacher and steering vector computation.

Editing Perplexity IR (↑) RM% (↑) (↓)

Model

QWEN-CONTRASTING 0.73 5.68 5.13 DIFF(MEAN) 0.78 6.64 4.39

MemCoach (Ours) 0.80 7.21 4.99

0.80

0.75

IR

Zero-shot Finetuned MemCoach

0.70

1 10 30 50 75 100 Training Set Percentage (%)

0.80

0.80

0.79

0.78 0.78

IR

0.75

0.76

0.74 MemCoach

0.72

0.70

40 45 50 55 60 65 70 α (Steering Coefficient)

Figure 8. Data efficiency. Top: performance vs number of training/contrasting samples. Bottom: performance vs coefficient α.

###### 6. Conclusion

We introduced the challenging problem of Memorability Feedback, a new task that shifts the study of memorability from passive prediction to actionable guidance. To foster future research on the setting, we present MemBench along with MemFeed evaluation metrics to assess the quality of provided feedback. We proposed MemCoach, a novel model-agnostic activation steering framework that distills how to improve the memorability of an image from an oracle teacher model to a student MLLM, aiming to provide naturallanguage feedback at capture time. Experimental validation of the approach demonstrates that steering multimodal large language models towards memorability-aware activations yields more effective and human-aligned feedback than zeroshot strategies, while requiring only minimal data. Beyond memorability, our findings suggest that activation steering offers a general and efficient route to endow MLLMs with perceptual skills, paving the way for future research on interactive and explainable visual guidance systems.

###### Acknowledgments

We acknowledge ISCRA for awarding this project access to the LEONARDO supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CINECA (Italy). This work was supported by the Ministero delle Imprese e del Made in Italy (IPCEI Cloud DM 27 giugno 2022 – IPCEI-CL-0000007) and European Union (Next Generation EU), the EU Horizon ELIAS (No. 101120237), and ELLIOT (No. 101214398). This work was carried out in the Vision and Learning joint laboratory of FBK and UniTN. Francesco Laiti is supported by PNRR funding (Innovative Doctorates program).

###### References

- [1] Abien Fred Agarap. Deep learning using rectified linear units (relu). arXiv preprint arXiv:1803.08375, 2018. 13
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millicah, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. In Proceedings of the 36th International Conference on Neural Information Processing Systems, 2022. 2
- [3] Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Chunsheng Wu, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025. 5, 6, 7
- [4] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609,

2023. 2

- [5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 5, 6, 7
- [6] Max Belitsky, Dawid J Kopiczko, Michael Dorkenwald, M Jehanzeb Mirza, James R Glass, Cees GM Snoek, and Yuki M Asano. Kv cache steering for controlling frozen llms. arXiv preprint arXiv:2507.08799, 2025. 2
- [7] Black Forest Labs. FLUX.2 [klein], 2026. 18
- [8] Zoya Bylinskii, Phillip Isola, Constance Bainbridge, Antonio Torralba, and Aude Oliva. Intrinsic and extrinsic effects on image memorability. Vision research, 2015. 1, 2, 7, 13
- [9] Zoya Bylinskii, Lore Goetschalckx, Anelise Newman, and Aude Oliva. Memorability: An image-computable measure of information utility. In Human perception of visual information: Psychological and computational perspectives, 2021. 1, 2
- [10] Mathilde Caron, Alireza Fathi, Cordelia Schmid, and Ahmet Iscen. Web-scale visual entity recognition: An llm-driven

- data approach. In Advances in Neural Information Processing Systems, 2024. 13
- [11] Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. Persona vectors: Monitoring and controlling character traits in language models. arXiv preprint arXiv:2507.21509, 2025. 2
- [12] Romain Cohendet, Claire-H´el`ene Demarty, Ngoc QK Duong, and Martin Engilberge. Videomem: Constructing, analyzing, predicting short-term and long-term video memorability. In Proceedings of the IEEE/CVF international conference on computer vision, 2019. 2
- [13] Rachit Dubey, Joshua Peterson, Aditya Khosla, Ming-Hsuan Yang, and Bernard Ghanem. What makes an object memorable? In Proceedings of the ieee international conference on computer vision, 2015. 1, 2
- [14] Th´eo Dumont, Juan Segundo Hevia, and Camilo L Fosco. Modular memorability: Tiered representations for video memorability prediction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023. 2
- [15] Jane L. E, Ohad Fried, Jingwan Lu, Jianming Zhang, Radom´ır Mech, Jose Echevarria, Pat Hanrahan, and James A. Landay. Adaptive photographic composition guidance. In Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems, 2020. 2
- [16] Simone Facchiano, Stefano Saravalle, Matteo Migliarini, Edoardo De Matteis, Alessio Sampieri, Andrea Pilzer, Emanuele Rodol`a, Indro Spinelli, Luca Franco, and Fabio Galasso. Video unlearning via low-rank refusal vector. arXiv preprint arXiv:2506.07891, 2025. 2
- [17] Jiri Fajtl, Vasileios Argyriou, Dorothy Monekosso, and Paolo Remagnino. Amnet: Memorability estimation with attention. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2018. 1, 13, 14
- [18] Lore Goetschalckx and Johan Wagemans. Memcat: a new category-based image set quantified on memorability. PeerJ,

2019. 2, 3, 4, 13

- [19] Lore Goetschalckx, Alex Andonian, Aude Oliva, and Phillip Isola. Ganalyze: Toward visual definitions of cognitive image properties. In Proceedings of the ieee/cvf international conference on computer vision, 2019. 1, 2, 7, 16
- [20] Google. Level up your photography skills with camera coach,

2025. 2

- [21] Thomas Hagen and Thomas Espeseth. Image memorability prediction with vision transformers. arXiv preprint arXiv:2301.08647, 2023. 13, 14, 18
- [22] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 2022. 8
- [23] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, et al. Language is not all you need: Aligning perception with language models. Advances in Neural Information Processing Systems, 2023. 2
- [24] Yipo Huang, Xiangfei Sheng, Zhichao Yang, Quan Yuan, Zhichao Duan, Pengfei Chen, Leida Li, Weisi Lin, and Guangming Shi. Aesexpert: Towards multi-modality foundation

- model for image aesthetics perception. In Proceedings of the 32nd ACM International Conference on Multimedia, 2024. 2, 6, 7
- [25] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. 13
- [26] Phillip Isola, Devi Parikh, Antonio Torralba, and Aude Oliva. Understanding the intrinsic memorability of images. Advances in neural information processing systems, 2011. 1, 2, 13
- [27] Phillip Isola, Jianxiong Xiao, Antonio Torralba, and Aude Oliva. What makes an image memorable? In CVPR 2011,

2011. 1, 2, 3, 4

- [28] Phillip Isola, Jianxiong Xiao, Devi Parikh, Antonio Torralba, and Aude Oliva. What makes a photograph memorable? IEEE transactions on pattern analysis and machine intelligence,

2013. 1, 2, 4, 13

- [29] Jaspal Singh Kahlon and Gongbo Liang. Portraid: An aidriven portrait assistant for professional-quality image composition. In Proceedings of the 2025 ACM Southeast Conference,

2025. 2

- [30] Aditya Khosla, Jianxiong Xiao, Phillip Isola, Antonio Torralba, and Aude Oliva. Image memorability and visual inception. In SIGGRAPH Asia 2012 Technical Briefs, 2012. 1
- [31] Aditya Khosla, Wilma A Bainbridge, Antonio Torralba, and Aude Oliva. Modifying the memorability of face photographs. In Proceedings of the IEEE international conference on computer vision, 2013. 2
- [32] Aditya Khosla, Akhil S Raju, Antonio Torralba, and Aude Oliva. Understanding and predicting image memorability at a large scale. In Proceedings of the IEEE international conference on computer vision, 2015. 1, 2, 3, 4, 5, 13
- [33] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2017. 13
- [34] Rukiye Savran Kiziltepe, Lorin Sweeney, Mihai Gabriel Constantin, Faiyaz Doctor, Alba Garc´ıa Seco de Herrera, ClaireH´el´ene Demarty, Graham Healy, Bogdan Ionescu, and Alan F. Smeaton. An annotated video dataset for computing video memorability. Data in Brief, 2021. 2
- [35] Prajneya Kumar, Eshika Khandelwal, Makarand Tapaswi, and Vishnu Sreekumar. Eye vs. ai: Human gaze and model attention in video memorability. arXiv preprint arXiv:2311.16484,

2023. 7

- [36] Prajneya Kumar, Eshika Khandelwal, Makarand Tapaswi, and Vishnu Sreekumar. Seeing eye to ai: Comparing human gaze and model attention in video memorability. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2025. 2
- [37] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742,

2025. 4, 14

- [38] Souad Lahrache and Rajae El Ouazzani. A survey on image memorability prediction: From traditional to deep learning models. In 2022 2nd International Conference on Innovative Research in Applied Science, Engineering and Technology (IRASET), 2022. 2, 13
- [39] Hugo Lauren¸con, Andr´es Marafioti, Victor Sanh, and Leo Tronchon. Building and better understanding vision-language models: insights and future directions. In Workshop on Responsibly Building the Next Generation of Multimodal Foundational Models, 2024. 5, 6, 7
- [40] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning. PMLR, 2023. 2
- [41] Leida Li, Hancheng Zhu, Sicheng Zhao, Guiguang Ding, and Weisi Lin. Personality-assisted multi-task learning for generic and personalized image aesthetics assessment. IEEE Transactions on Image Processing, 2020. 1
- [42] Yi-Feng Li, Chuan-Kai Yang, and Yi-Zhen Chang. Photo composition with real-time rating. Sensors, 20(3), 2020. 2
- [43] Jie Liang, Hui Zeng, Miaomiao Cui, Xuansong Xie, and Lei Zhang. Ppr10k: A large-scale portrait photo retouching dataset with human-region mask and group-level consistency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021. 2, 3, 13
- [44] Oliver Limoyo, Jimmy Li, Dmitriy Rivkin, Jonathan Kelly, and Gregory Dudek. Photobot: Reference-guided interactive photography via natural language. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),

2024. 2

- [45] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in neural information processing systems, 2023. 2
- [46] Shuai Ma, Zijun Wei, Feng Tian, Xiangmin Fan, Jianming Zhang, Xiaohui Shen, Zhe Lin, Jin Huang, Radom´ır Mˇech, Dimitris Samaras, et al. Smarteye: assisting instant photo taking via integrating user preference with deep view proposal network. In Proceedings of the 2019 CHI conference on human factors in computing systems, 2019. 2
- [47] Iv´an Mart´ın-Fern´andez, Sergio Esteban-Romero, Fernando Fern´andez-Mart´ınez, and Manuel Gil-Mart´ın. Parameterefficient adaptation of large vision—language models for video memorability prediction. Sensors (Basel, Switzerland),

2025. 2

- [48] Anelise Newman, Camilo Fosco, Vincent Casser, Allen Lee, Barry McNamara, and Aude Oliva. Multimodal memorability: Modeling effects of semantics and decay on video memorability. In European Conference on Computer Vision, 2020. 2, 13
- [49] Thao Nguyen, Haotian Liu, Yuheng Li, Mu Cai, Utkarsh Ojha, and Yong Jae Lee. Yo’llava: Your personalized language and vision assistant. In NeurIPS, 2024. 18
- [50] OpenAI. Gpt-5 system card. Technical report, OpenAI, 2025. 6, 7, 15, 17
- [51] Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. ICML, 2024. 2

- [52] Shay Perera, Ayellet Tal, and Lihi Zelnik-Manor. Is image memorability prediction solved? In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, 2019. 2, 13
- [53] Daiqing Qi, Handong Zhao, Jing Shi, Simon Jenni, Yifei Fan, Franck Dernoncourt, Scott Cohen, and Sheng Li. The photographer’s eye: Teaching multimodal large language models to see, and critique like photographers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 1
- [54] Daiqing Qi, Handong Zhao, Jing Shi, Simon Jenni, Yifei Fan, Franck Dernoncourt, Scott Cohen, and Sheng Li. The photographer’s eye: Teaching multimodal large language models to see, and critique like photographers. In Proceedings of the Computer Vision and Pattern Recognition Conference,

2025. 2

- [55] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR,

2021. 2, 3

- [56] Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. Steering llama 2 via contrastive activation addition. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024. 2
- [57] William F Shen, Xinchi Qiu, Meghdad Kurmanji, Alex Iacob, Lorenzo Sani, Yihong Chen, Nicola Cancedda, and Nicholas D Lane. Lunar: Llm unlearning via neural activation redirection. arXiv preprint arXiv:2502.07218, 2025. 2
- [58] Harini SI, Somesh Singh, Yaman K Singla, Aanisha Bhattacharyya, Veeky Baths, Changyou Chen, Rajiv Ratn Shah, and Balaji Krishnamurthy. Long-term ad memorability: Understanding and generating memorable ads. In Proceedings of the IEEE international conference on computer vision, 2024. 2, 13, 14
- [59] Aliaksandr Siarohin, Gloria Zen, Cveta Majtanovic, Xavier Alameda-Pineda, Elisa Ricci, and Nicu Sebe. How to make an image more memorable? a deep style transfer approach. In Proceedings of the 2017 ACM on international conference on multimedia retrieval, 2017. 1, 2
- [60] Oleksii Sidorov. Changing the image memorability: From basic photo editing to gans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2019. 2
- [61] C Spearman. The proof and measurement of association between two things. International Journal of Epidemiology,

2010. 4

- [62] Hammad Squalli-Houssaini, Ngoc QK Duong, Marquant Gwena¨elle, and Claire-H´el`ene Demarty. Deep learning for predicting image memorability. In 2018 IEEE international conference on acoustics, speech and signal processing (ICASSP), 2018. 1
- [63] Ying Tai, Jian Yang, Xiaoming Liu, and Chunyan Xu. Memnet: A persistent memory network for image restoration. In

- Proceedings of the IEEE international conference on computer vision, 2017. 13, 14
- [64] Davide Talon, Federico Girella, Ziyue Liu, Marco Cristani, and Yiming Wang. Seeing the abstract: Translating the abstract language for vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference,

2025. 2

- [65] Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248, 2023. 2, 5
- [66] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models,

2022. 14

- [67] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 3, 5, 6, 7
- [68] Brandon T Willard and R´emi Louf. Efficient guided generation for large language models. arXiv preprint arXiv:2307.09702, 2023. 6, 14
- [69] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2020. 14
- [70] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. preprint arXiv:2508.02324, 2025. 18
- [71] Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Kaixin Xu, Chunyi Li, Jingwen Hou, Guangtao Zhai, et al. Q-instruct: Improving low-level visual abilities for multi-modality foundation models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024. 2, 6, 7
- [72] Xiaoran Wu. Interpretable aesthetic analysis model for intelligent photography guidance systems. In Proceedings of the 27th International Conference on Intelligent User Interfaces,

2022. 1

- [73] Xiaoran Wu and Jia Jia. Tumera: Tutor of photography beginners. arXiv preprint arXiv:2109.11365, 2021. 2
- [74] Amit Zalcher, Navve Wasserman, Roman Beliy, Oliver Heinimann, and Michal Irani. Don’t judge before you clip: A unified approach for perceptual tasks. arXiv preprint arXiv:2503.13260, 2025. 2, 13, 14
- [75] Jerrold H. Zar. Spearman Rank Correlation. John Wiley & Sons, Ltd, 2005. 13
- [76] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training.

- In Proceedings of the IEEE/CVF international conference on computer vision, 2023. 13
- [77] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, 2023. 2
- [78] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. In The Twelfth International Conference on Learning Representations, 2024. 2
- [79] Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405, 2023. 2

### How to Take a Memorable Picture? Empowering Users with Actionable Feedback

#### Supplementary Material

In this supplementary material, we provide additional details on MemBench and MemCoach. Section A presents qualitative examples from MemBench dataset and describes its construction pipeline. Section B gives additional implementation details of our proposed MemCoach method with a discussion regarding potential implications of our work, while Section C provides preliminary user study experiments. Finally, in Section D, we demonstrate the consistency of our framework across different editing models and memorability predictors, including an analysis of feedback quality generated by MemCoach.

###### A. MemBench Additional Details

###### A.1. Data Examples

Fig. 10 and 11 present data examples from the MemBench dataset. For each image pair, we show (from left to right) the source image (red frame), the destination image (blue frame), and the corresponding feedback generated by the multimodal model. The memorability scores assigned by the predictor M are shown beneath each image.

###### A.2. Construction Details

Images within each scene are ranked using the memorability predictor M. Pairs of least and most memorable images are then selected to construct contrastive training data. Evaluation is performed on a random held-out set of unseen scenes, where feedback is generated starting from the scene’s least memorable image.

###### A.3. Image Pre-processing

PPR10K [43] provides images in RAW format, with an average image file size of 42MB. To reduce storage requirements, enable efficient processing, and ensure compatibility with MLLM vision processors, we convert all RAW files to JPEG format while preserving their original aspect ratio. Conversions are performed using rawpy and PIL Python libraries, both executed with default parameters.

###### A.4. Memorability Predictor

To build the memorability predictor M, we follow the approach proposed in [74] where a frozen visual feature extractor is followed by an MLP head trained for regression. The final model outputs a single continuous value in [0,1] corresponding to the predicted memorability score.

Training. We train the regressor M on three widely used memorability datasets: LaMem [32], MemCat [18], and

SUN [17]. Each image is associated with a ground-truth memorability score in the range [0,1]. For feature extraction, we employ the vision tower of OpenCLIP [25], specifically the ViT-SO400M-14-SigLIP-384 model [76] pretrained on WebLI [10]. The resulting visual embeddings (dimension 1152) serve as input to the regression head. The MLP consists of two fully connected layers: a 256-dimensional hidden layer with a ReLU activation [1], followed by a 1-dimensional linear output layer. The model is trained end-to-end only on the MLP parameters, using the mean squared error (MSE) loss. We train for 100 epochs using the Adam optimizer [33] with a learning rate of 1×10−4 and weight decay set to 0.0. Empirically, we find that using the raw (unnormalized) OpenCLIP features leads to improved Spearman’s rank correlation [75]. A summary of the model performance is reported in Tab. 5.

Table 5. Model M performance. Comparison of memorability predictor models trained on CLIP-like embeddings. We report feature dimensionality, whether feature normalization is applied, and Spearman’s rank correlation. The used predictor model achieves the highest correlation among all evaluated variants. (*) is reported from [74].

Spearman Rank (↑)

Model Fts dim Normalization

Human* n.d. n.d. 0.68 ViT-L-14-quickgelu 768 ✓ 0.73 ViT-L-14-quickgelu 768 × 0.81 ViT-SO400M-14-SigLIP-384 1152 ✓ 0.76 MemBench M 1152 × 0.82

Validation. To assess the effectiveness of our memorability predictor M, we compare it against state-of-the-art memorability models [17, 21, 58, 63, 74] on the LaMem test set (Tab. 6). While prior approaches are typically trained solely on LaMem, except for [58], our model M leverages three datasets (LaMem, MemCat, and SUN), achieving the highest Spearman’s rank correlation among all evaluated methods.

Limitations. Our automatic pipeline relies heavily on the initial ranking of images, which is determined by a memorability predictor M. Although this dependency introduces a potential source of bias, we treat the memorability model as a well-established black-box component, consistent with prior literature [8, 18, 26–28, 32, 38, 48, 52, 74]. In this sense, our framework is agnostic to the specific predictor used: given any target scoring criterion, it may substitute an alternative ranking signal and construct systems tailored to their own objectives.

Table 6. Comparison with state-of-the-art memorability predictors. Spearman Rank correlation on the LaMem test set. Our model, trained on LaMem, MemCat, and SUN, achieves the highest correlation among all methods. (*) is reported from [74].

Spearman Rank (↑)

Model Pretrained

MemNet* [63] LaMem 0.64 AMNet* [17] LaMem 0.67 Human* n.d. 0.68 ViTMem* [21] LaMem 0.71 Henry [58] LaMem + MemCat + SUN 0.72 Henry [58] LaMem 0.74 PerceptCLIP [74] LaMem 0.74

MemBench M LaMem + MemCat + SUN 0.82

###### A.5. Editing Baseline

For in-context image editing, we employ FLUX.1 KONTEXT [37], as described in Sec. 3.2, via diffusers library [66] from HuggingFace (open-source model version with tag FLUX.1-Kontext-dev). We use the default configuration recommended by the original authors, setting the number of inference steps to 28, the guidance scale to 2.5, and fixing the seed generator to 0 to ensure reproducibility. The default aspect ratio of the model is 1:1, specifically 1024 × 1024 pixel resolution.

###### A.6. Prompting and Structured Feedback

Feedback elicitation prompt. To generate the full MemBench dataset, we rely on prompt pa, presented in Sec. 3.2. The complete prompt pa is reported below.

System: You are an observer.

###### User:

|SOURCE IMAGE A|
|---|

|DESTINATION IMAGE B|
|---|

Your task is to determine the actions required to transform Image A into Image B. Strictly avoid both explicit and implicit references to the images when suggesting action items, and ensure that each action item is fully self-contained.

Produce a structured JSON object that must include:

– actions: a list of precise and well-informative semantic actions.

Respond with a valid JSON object and no explanation.

Output: {

"actions": [

- <#1 sub-action>,
- <#2 sub-action>,

..., <#k sub-action>

] }

The resulting output is then parsed as a JSON object.

During early experimentation, models frequently produced feedback that referred directly to the target image rather than describing the transformation itself (e.g. “Adjust the brightness to match the one in Image B”). To address this issue, we refined the prompt to explicitly forbid image-referential phrasing and require self-contained action descriptions.

In cases where multiple source images receive the same memorability score, ties are resolved by sorting according to the filename identifier in descending order.

Structured, formally valid JSON outputs. To ensure consistently structured outputs while maintaining flexibility in the definition of output fields, we adopt the outlines library [68] for constrained decoding; compatible with the transformers library [69], it allows to enforce a predefined output schema. For extracting the feedback divided into subactions, we define the following class specification:

class ActionListOutput(BaseModel):

actions: List[str] = Field( description="A list of actions.", min_items=1, max_items=10

)

Listing 1. Class specification to ensure valid JSON schema.

This setup enforces syntactic validity, guarantees reliable parsing, and enables systematic storage of feedback samples in JSON format. The schema-based design also allows for straightforward modifications, such as adding or removing fields when extending the output format.

###### A.7. Feedback Sub-actions Categorization

To categorize the atomic sub-actions contained in each feedback instance, we employ GPT-5 MINI [50] as an automatic annotator. The model is prompted with a taxonomy covering six high-level categories: Framing, Lighting, Posing, Semantics, Intent, and Aesthetics. The prompt used for annotation is reported below:

User: Consider the following action that a photographer can do. Categorize the action into: FRAMING – Zoom/Crop/Reframe, Angle and Viewpoint, Balance and Symmetry LIGHTING – Lighting direction/strength/temperature, Exposure adjustment, Shadows control POSING – Pose adjustments, Facial expressions, Subjects interaction, Clothes SEMANTICS – Add/remove objects or people, Change background, Include contextual cues INTENT – Change narrative emphasis, Mood and atmosphere AESTHETICS – Color grading/filters, Contrast/sharpness, Blur and focus Here the action: <input action>

This setup guarantees consistent labeling across all subactions, enabling downstream analysis of category frequencies and co-occurrence patterns.

###### B. MemCoach Additional Details

MemCoach is a training-free method that applies activation steering to modulate the internal representations of a multimodal model. All experiments were conducted using the PyTorch framework on a single NVIDIA A100 GPU (64 GB).

Below, we provide the implementation details for generating and injecting the steering vector, complementing the

description in Sec. 4.2.

Extraction stage. We target the residual module within a specific language Transformer block of the multimodal backbone (e.g., layer l = 55 in our best-performing configuration). For a chosen layer l, we register a forward hook to capture the activation tensor h(l) for each input sequence i. Each input sequence, as defined in Eq. 2, is first tokenized into input IDs. For each input i, we compute the mean over the sequence dimension to obtain a single activation representation. No normalization or additional post-processing is applied. Since we operate in batches, we record the starting index of padding tokens and exclude padded positions from the mean computation. No generation is performed during this stage: the model is only run in forward mode. The memorability steering vector r(l) is computed as described in Sec. 4.2.

Inference stage. Inference requires selecting a steering coefficient α and the target layer l. During the model’s forward pass, the vector r(l) is injected, scaled by α, at the specified layer, following Eq. 4. Injection is applied uniformly along the sequence dimension. The forward computation then proceeds as usual, but with altered activation patterns at layer l, steering the model toward memorability-aware behaviour.

Different models configuration. Tab. 7 reports the optimal layer index l and steering coefficient α identified for the four open-source models we evaluate. Hyperparameters are optimized on a held-out split of the training set.

Table 7. Optimal steering configuration across models. Bestperforming layer index l and steering coefficient α obtained via hyperparameter tuning for each evaluated open-source multimodal model. We also report the language-model depth (LM Depth) and the corresponding IR score.

Model Layer Coefficient LM Depth IR

MemCoach-IDEFICS 13 30 32 0.75 MemCoach-LLAVA 20 143 28 0.73 MemCoach-QWEN 12 26 28 0.74 MemCoach-INTERNVL 12 55 36 0.80

Implication of optimizing memorability. Enhancing memorability raises ethical concerns, including potential manipulation, undue influence on viewer perception, and the risk of homogenizing visual expression by favoring conventional cues over diversity. At the same time, in assistive and controlled contexts, e.g. education, creative exercises, or personal photography coaching, memorability optimization can enhance communication, reinforce learning recall, and provide actionable guidance without overriding individual intent. Balancing these risks and benefits is essential, emphasizing the need for transparency, user agency, and context-aware application to ensure that memorability interventions remain supportive rather than prescriptive.

###### C. User Studies

The following experiments are preliminary user studies designed to evaluate the effectiveness of MemCoach. They aim to provide early validation and insights on the effect of MemCoach on human memorability (Sec. C.1), evaluate the effectiveness of the approach for real-life guidance (Sec. C.2), and probe the quality of feedback according to users (Sec. C.3).

###### C.1. Human Memorability Alignment

To assess how images from different settings drive human memorability, we conduct a memorability experiment with 47 valid users (avg. 15.6 annotations/image), following previous work [19].

Experiment setup details. Participants completed a continuous recognition (repeat-detection) visual memory task in which they viewed a stream of images and pressed the space bar whenever they detected a repeat of an image previously shown within the same session. Each session consisted of 150 images presented sequentially, with each image displayed for 600 ms followed by an 800 ms blank interstimulus interval. The sequence contained 40 target pairs, where each target image appeared once and was repeated after 22-93 intervening images. An additional 10 vigilance pairs were included as attention checks, in which the repeat occurred after 1-4 intervening images. The remaining 50 images were fillers presented only once to maintain spacing and reduce predictability of repeats. Participants could respond at any time during the 1400 ms trial window (image plus blank interval). The same task protocol was used across three stimulus variants, i.e. MemCoach, INTERNVL3.5 , and source images xs, with identical timing and sequence structure; only the image sources differed. Sequence order was deterministic and reproducible, using a fixed seed. Participants were instructed not to repeat the game after completing one. If participants missed more than 50% of the vigilance repeats in a run, user results were excluded from the analyses.

Results. Consistent with the editing experiments, Figure 9 shows both INTERNVL3.5 and MemCoach increase the average memorability wrt source images xs, with MemCoach achieving a large margin of improvement. The gap between methods indicates that MemCoach more effectively shifts images toward higher memorability regimes, while INTERNVL3.5 provides only moderate improvements. Overall, results highlight the importance of explicitly injecting memorability-aware signals to achieve stronger memorability outcomes.

###### C.2. Human-in-the-loop Evaluation

To evaluate real-world effectiveness, we conduct a preliminary human-in-the-loop evaluation measuring whether users can successfully follow the generated feedback to produce

[Figure 71]

Figure 9. Human memory performance across three different image settings.

more memorable images.

Experiment setup details. We implemented a mobile app allowing a user to use MemCoach in real-life scenarios. The app presents a live camera viewfinder and allows users to point their phone at any scene of their choice. Upon capturing a frame, users could request either a memorability score alone or a memorability score accompanied by actionable textual feedback generated by MemCoach.

The feedback is displayed directly on screen as a naturallanguage suggestion, guiding the user toward a more memorable composition. Users are free to implement the feedback by adjusting their framing or scene and re-capturing the image, observing how the memorability score changed between attempts. When the feedback pertained to the objects or subjects of the scene, the phone was kept as stable as possible between captures to isolate the effect of compositional changes. Each submitted image,

[Figure 72]

together with its predicted score and generated feedback, was logged for offline analysis. The app requires no installation and runs entirely in the browser, served over a secure HTTPS connection via a public tunnel to ensure accessibility across devices and operating systems.

Results. We collected 27 scenes and evaluated them using the memorability predictor M. Despite the domain shift, MemCoach consistently improves memorability, achieving an IR of 0.52 and a relative gain (RM) of +4.9%. These improvements reflect both the frequency and magnitude of successful memorability enhancements, indicating that the generated feedback effectively guides actionable changes even in previously unseen scenarios. Qualitative inspection confirms that suggestions focus on semantically meaningful adjustments, such as subject positioning, gaze direction, and interaction cues, rather than superficial edits. Overall, these results demonstrate the potential of human-in-the-loop memorability optimization and motivate future exploration of larger-scale, user-centered studies and strategies for robust real-world deployment.

###### C.3. Feedback Quality Evaluation

To understand the effectiveness of memorability feedback beyond score improvements, we conducted a human study with 28 participants (381 annotations), rating the generated feedback from MemCoach on a 1–5 Likert scale along three dimensions: Clearness (clarity of steps), Relevance (scenespecificity), and Feasibility (realism of applying it).

Experiment setup details. Participants completed a feedback-rating task in which they viewed a source image together with feedback and provided three scalar judgments about the feedback. For each item, participants were asked to rate:

- (i) Clearness. “How clear are the steps needed to change the photo?” (1: Not at all, 5: Very clear);
- (ii) Relevance. “How specific is the advice for this scene?” (1: Very general, 5: Very specific);
- (iii) Feasibility. “How realistic and sensible is this feedback to apply in real-world conditions (e.g., given physical constraints, tools, and context)?” (1: Not realistic or sensible, 5: Completely realistic and sensible).

Each item yielded three numeric ratings, and items were presented in a randomized order with identical instructions and scale anchors across conditions, ensuring consistent evaluation of clarity, scene-specificity, and real-world actionability. Results. A summary of the experiment performance is reported in Tab. 8. MemBench Oracle achieves consistently high scores across all dimensions, confirming the advantage of access to memorability-aware privileged information. MemCoach maintains strong clearness while improving feasibility, indicating that its suggestions are generally easier to interpret and implement in practice. This gain, however, comes with a slight reduction in scene-specificity, suggesting a tendency toward more generic but broadly applicable guidance. Overall, the results highlight a trade-off between precision and practicality, where MemCoach favors actionable and reliable feedback over highly tailored but less consistently executable suggestions.

Table 8. Feedback quality results.

Model Clearness Relevance Feasibility

Oracle 4.19 4.30 4.11 MemCoach 3.96 3.76 4.32

###### D. Additional Analyses D.1. Positive and Failure Cases

To analyze the factors underlying successful and unsuccessful memorability feedback, we designed a structured annotation pipeline based on an LLM-as-judge approach that classifies each sample according to a fixed taxonomy.

Experiment setup details. For each sample, the judge receives three inputs: the original source image, the generated feedback text, and the measured memorability outcome (Improved or Worsened) as determined by our memorability predictor M. Based on the outcome, one of two dedicated prompt templates is selected. Both templates frame the model as an expert annotator performing classification rather than creative analysis, but present mutually exclusive category sets tailored to the direction of change. For improved samples, the judge selects among five improvement categories: (i) Posing / Body Configuration, (ii) Framing / Composition, (iii) Lighting / Visibility, (iv) Semantic Clarity, and (v) Emotional or Social Salience. For worsened samples, it selects among five failure categories: (i) Template Over-Normalization, (ii) Distinctiveness Suppression, (iii) Feasibility / Actionability Failure, (iv) Attention Dilution, and (v) Perceptual Degradation. In both cases, the model is required to first produce a one-sentence justification grounded strictly in the visible image content and the provided feedback, serving as a reasoning step before the final category assignment, followed by a single primary category, with explicit prohibitions against introducing new categories, referencing memorability scores, or speculating beyond the observable evidence. Annotations have been generated using GPT-5 MINI [50] model.

Results. Positive effects are driven primarily by posing (74.9%), which plays the dominant role, followed by emotional saliency (23.11%), and to a much smaller extent by framing (1.59%). This distribution indicates that improvements are largely attributable to how subjects are positioned and the emotional cues conveyed, with only marginal influence from compositional framing. Conversely, failures are predominantly caused by over-normalization (76.19%), which emerges as the principal limiting factor, along with distinctiveness suppression (22.22%), which further reduces the effectiveness of the outcome. Additional failure modes contribute only marginally, including perceptual degradation (1.59%), while instances of attention dilution or action infeasibility occur only rarely and have a negligible overall impact.

###### D.2. Editing Instruction-Following

To decouple feedback quality from the editor’s instructionfollowing, we compare memorability changes from (i) human-in-the-loop ground truth (Sec. C.2) and (ii) edited images using the same feedback. Similar performance is noted (0.52 IR, +4.9% RM ground truth vs 0.55 IR, +2.19% RM edited) and the moderate correlation (ρ = 0.51) between destination image memorabilities confirms the editor as a valid proxy for automated evaluation.

###### D.3. Leveraging Predictor Biases

To mitigate shortcut learning, we ran a cross-predictor experiment using different memorability predictors for steering, i.e. ViTMem (VM) [21], and evaluation (our memorability predictor M, MB). As shown in Tab. 9, using VM alone or the cross-predictor setup (VM → MB) consistently confirms the effectiveness of MemCoach, indicating that the observed improvements are not an artifact of a single predictor. These results demonstrate that MemCoach ’s feedback remains robust across varying evaluation criteria, effectively enhancing memorability regardless of the specific predictor used, and mitigating concerns of shortcut learning or predictor-specific bias.

Table 9. Performance of our framework on different settings. We report the results using different editing models for the evaluation and different memorability predictors.

Edit Model Mem Predictor

Model

Qwen-IE FLUX.2-k VM VM → MB Edit model 0.69 0.68 0.64 0.69 Zero-shot baselines

GPT-5-MINI 0.78 0.80 0.73 0.76 LLAVA-OV 0.59 0.68 0.76 0.71 IDEFICS3 0.69 0.68 0.73 0.73 QWEN2.5VL 0.54 0.61 0.73 0.69 INTERNVL3.5 0.78 0.74 0.68 0.73

MemCoach

MemCoach-LLAVA 0.80 0.73 0.69 0.74 MemCoach-IDEFICS 0.85 0.74 0.77 0.76 MemCoach-QWEN 0.82 0.76 0.77 0.81 MemCoach-INTERNVL 0.88 0.83 0.83 0.82

###### D.5. Generalization

As a first study, we focus on human-centric images, given the strong influence of human presence and attributes on image memorability. To assess how well this setting generalizes beyond such content, we conduct preliminary experiments on non-human images using the same editing proxy pipeline introduced in the main paper. Results on objects and landmarks from the Yo’LLaVA dataset [49] indicate that MemCoach performs on par with QWEN2.5VL (0.79 IR), i.e., with no degradation in performance relative to the human domain. Exploring a broader extension beyond human-centric images is left for future work.

###### D.4. Multiple Editing Models

In Tab. 9, we evaluate MemCoach across multiple editing backbones, including Qwen-Image Edit1 [70] and FLUX.2klein2 [7]. Across all editors, MemCoach consistently increases both the frequency and magnitude of memorability improvements compared to baseline zero-shot and default feedback. Gains are robust to variations in model architecture and editing style, indicating that the proposed approach generalizes across different latent spaces and editing mechanisms. Qualitative inspection confirms that MemCoach directs edits toward semantically meaningful transformations rather than generic low-level changes, producing feedback that is both actionable and visually effective. These results reinforce that the benefit of memorability-aware guidance is model-agnostic and not confined to a specific editing pipeline.

- 1Model card: https : / / huggingface . co /

lightx2v / Qwen - Image - Lightning. Weight name: Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32

- 2Model card: https://huggingface.co/black-forest-

labs/FLUX.2-klein-9B

|[Figure 73]|
|---|

|[Figure 74]|
|---|

###### Feedback a:

- 1. Rotate the perspective to face forward.
- 2. Bring the hands up to cover the mouth.
- 3. Hold a small yellow flower between the fingers.
- 4. Adjust the hair to frame the face evenly.

Mem score: 0.992

Mem score: 0.525

|[Figure 75]|
|---|

|[Figure 76]|
|---|

###### Feedback a:

- 1. Adjust the position of the person on the left to face forward with a slight smile.
- 2. Raise the head of the person on the right and have them look slightly to the side with a gentle smile.
- 3. Ensure both individuals are standing upright and close together, with the person on the right holding the handlebars of the scooter.
- 4. Maintain the floral arrangement and attire of both individuals as they are.

Mem score: 0.753

Mem score: 0.803

|[Figure 77]|
|---|

|[Figure 78]|
|---|

###### Feedback a:

- 1. Adjust the position of the left arm to rest on the hip, ensuring the hand is placed firmly on the hip bone.
- 2. Shift the weight slightly to the left leg to create a more dynamic stance.
- 3. Tilt the head slightly to the right to add a subtle angle to the gaze.
- 4. Ensure the right hand is relaxed by the side, with fingers gently curved.
- 5. Adjust the lighting to maintain consistency in shadows and highlights across the outfit.

Mem score: 0.965

Mem score: 0.885

###### Feedback a:

- 1. Crop the image to focus on the upper body of the person, removing the background elements such as the airplane and the chair.
- 2. Adjust the angle to a closer, more intimate shot, centering the person’s face and upper torso.
- 3. Ensure the person’s hand is positioned near the face, with the index finger touching the lips, maintaining the same pose but in a tighter frame.
- 4. Remove any elements that are not part of the person’s attire or immediate surroundings, such as the grass and distant trees.
- 5. Enhance the lighting to create a softer, more even illumination on the person’s face and clothing.

|[Figure 79]|
|---|

|[Figure 80]|
|---|

Mem score: 0.578

Mem score: 0.981

Figure 10. A set of qualitative examples from MemBench.

|[Figure 81]|
|---|

|[Figure 82]|
|---|

###### Feedback a:

- 1. Stand up from the seated position on the railway track.
- 2. Hold a wicker basket with both hands in front of the body.
- 3. Adjust the hair to fall naturally over the shoulders.
- 4. Shift the gaze to the side while maintaining a smile.
- 5. Ensure the background changes to a green train car with visible rust and metal details.

Mem score: 0.980

Mem score: 0.716

|[Figure 83]|
|---|

|[Figure 84]|
|---|

###### Feedback a:

- 1. Adjust the person’s posture to a standing position.
- 2. Raise both arms to hold flowers above the head.
- 3. Ensure the person is surrounded by lush greenery.
- 4. Remove the basket and gardening tools from the scene.
- 5. Adjust the lighting to create a brighter and more vibrant atmosphere.

Mem score: 0.710

Mem score: 0.752

###### Feedback a:

- 1. Reposition the couple from a seated to a lying down pose on a grassy surface.
- 2. Adjust the angle of the shot to an overhead view.
- 3. Remove the stone wall and window background, replacing it with a grassy area.
- 4. Ensure the bride’s dress and veil spread out naturally on the grass.
- 5. Place the groom’s jacket and pants neatly on the grass beside them.
- 6. Position the bride’s bouquet on the grass near her hand.

|[Figure 85]|
|---|

|[Figure 86]|
|---|

Mem score: 0.807

Mem score: 0.892

###### Feedback a:

- 1. Adjust the position of the person so they are standing more centrally within the frame.
- 2. Shift the perspective slightly to the right to include more of the water and the sculpture on the left.
- 3. Reduce the brightness and contrast to create a softer, more subdued lighting effect.
- 4. Reposition the person’s arm so it is relaxed by their side, not holding onto the structure.
- 5. Ensure the reflection on the water is more prominent by adjusting the angle of the light source.

|[Figure 87]|
|---|

|[Figure 88]|
|---|

Mem score: 0.749

Mem score: 0.829

###### Feedback a:

- 1. Reposition the individuals so that they are facing each other, with one person lifting the other into their arms.
- 2. Adjust the arms so that the lifted person’s arms are wrapped around the other person’s neck.
- 3. Ensure the lifted person’s legs are bent at the knees and held by the other person.
- 4. Shift the gaze of both individuals to look at each other affectionately.
- 5. Maintain the scenic background with mountains and clouds, but adjust the angle slightly to accommodate the new pose.

|[Figure 89]|
|---|

|[Figure 90]|
|---|

Mem score: 0.798

Mem score: 0.990

20

Figure 11. A set of qualitative examples from MemBench.

