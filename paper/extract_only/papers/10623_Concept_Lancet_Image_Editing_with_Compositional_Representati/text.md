## Concept Lancet: Image Editing with Compositional Representation Transplant

Jinqi Luo, Tianjiao Ding, Kwan Ho Ryan Chan, Hancheng Min, Chris Callison-Burch, René Vidal University of Pennsylvania

jinqiluo@upenn.edu

# arXiv:2504.02828v1[cs.CV]3Apr2025

##### Abstract

Diffusion models are widely used for image editing tasks. Existing editing methods often design a representation manipulation procedure by curating an edit direction in the text embedding or score space. However, such a procedure faces a key challenge: overestimating the edit strength harms visual consistency while underestimating it fails the editing task. Notably, each source image may require a different editing strength, and it is costly to search for an appropriate strength via trial-and-error. To address this challenge, we propose Concept Lancet (CoLan), a zeroshot plug-and-play framework for principled representation manipulation in diffusion-based image editing. At inference time, we decompose the source input in the latent (text embedding or diffusion score) space as a sparse linear combination of the representations of the collected visual concepts. This allows us to accurately estimate the presence of concepts in each image, which informs the edit. Based on the editing task (replace/add/remove), we perform a customized concept transplant process to impose the corresponding editing direction. To sufficiently model the concept space, we curate a conceptual representation dataset, CoLan-150K, which contains diverse descriptions and scenarios of visual terms and phrases for the latent dictionary. Experiments on multiple diffusion-based image editing baselines show that methods equipped with CoLan achieve state-of-the-art performance in editing effectiveness and consistency preservation. More project information is available at https://peterljq.github.io/project/colan.

##### 1. Introduction

How can we edit a given image following a specified conceptual guidance, say Cat→Dog or Sketch→Painting? This problem was briefly discussed in the early work of Image Analogies [12] in the 2000s, widely studied in the era of generative adversarial networks [19, 70], and recently revolutionized by diffusion models [14, 48]. Not only do diffusion models shine in producing realistic images [42, 46], but they also allow for conditioning on multimodal guidance such as text prompts [21, 45] and regional masks [1, 16, 68], making

|[Figure 1]|
|---|

Adorable [cat] [dog] sits on green grass …

###### dog Concept Lancet for Editing

cat

sitting …

grass

time

|[Figure 2]|
|---|

0.57

0.25

0.48

-0.15

Diffusion Model Backbone

Figure 1. Given a source image and the editing task, our proposed CoLan generates a concept dictionary and performs sparse decomposition in the latent space to precisely transplant the target concept.

it handy for image editing.

In this paper, we consider the task of utilizing diffusion models for image editing that impose desired concepts based on user prompts. Specifically, given a source image and its source caption, our task is to modify the content, appearance, or pattern of the image based on a given target prompt. To accomplish it with diffusion models, a basic idea is to first use the score predictor to perform the noising process (e.g., DDIM Inversion [49]) and then follow the denoising process conditioned on the target concept. Such conditioning in the diffusion-based backbones typically happens in a structured latent space (score or text embedding space; to be detailed in §2), where one moves a source latent vector towards an edit direction (i.e., the shift from the source concept to the target concept). A fruitful series of works have contributed to enhancing the inversion process [17, 18, 27, 36, 51], innovating attention controls [4, 11, 40, 65], and tuning the backbone with conditional instructions [3, 9, 39, 71].

Despite the remarkable progress, an often overlooked issue is the magnitude of the concept representation to be imposed, that is, determining how far to move along the edit direction. Prior works [40, 65] typically assume a certain amount of editing without adjusting the magnitude based on the contexts or existing concepts of the given image. Such a heuristic choice can be problematic as we show in Figure 2: adding too much of the edit may overwhelm the image with the target concept (Image (4)), while too little may fail to fully remove the source concept (Image (2)). Hence, since the presence of the source concept varies in every image, proper estimation of the edit magnitude is necessary for

accurate editing.

To address the aforementioned issue, we draw inspiration from low-dimensional [22, 28] and compositional [25, 59] structures in latent representations of diffusion models. In this paper, we propose Concept Lancet (CoLan), a zero-shot plug-and-play framework to interpret and manipulate sparse representations of concepts for diffusion-based image editing. Our intuition is to sufficiently model the latent spaces to analyze how much each concept is present in the source representation, which allows for accurate transplant from the source concept to the target one in the proper magnitude. We state our contributions and workflows as follows:

- • (§3.1) To allow for such an analysis, one needs a dictionary of directions representing diverse concepts. Existing dictionaries are limited since they either contain concepts without clear visual meanings (e.g., “hardship”), lack phrases (e.g., “made of wood”), or have only a limited number of concepts. Thus, we collect a conceptual representation dataset, CoLan-150K, of diverse descriptions for visual concepts and compute a dictionary of concept vectors to the latent (text embedding or score) space.
- • (§3.2) At inference time, we propose to decompose the source latent vector as a linear combination of the collected visual concepts to inform the edit. To mitigate the optimization inefficiency with an overcomplete dictionary, we instruct a vision-language model (VLM) to parse image-prompt tuples into a representative list of visual concepts as dictionary atoms. For common editing tasks of replacing, we switch the source concept vector in the decomposition with our target concept vector and synthesize the edited image with the backbone. The task of adding or removing concepts can be recasted as special cases of concept replacing, detailed in the method section.
- • (§4) We conduct quantitative comparisons on multiple diffusion-based image editing baselines and qualitative evaluations on the visual synthesis. Methods equipped with CoLan achieve state-of-the-art performance in editing effectiveness and consistency preservation. Notably, the plug-andplay design of our method provides flexibility in the choice of backbones and latent spaces.

##### 2. Preliminaries in Diffusion-Based Editing

This section briefly discusses diffusion-based image editing and how it involves a representation manipulation process in either the text embedding or the score space.

DDIM Inversion. Diffusion model samples a new image z0 by gradually denoising a standard Gaussian zT through the following reverse-time conditional denoising process:

√1 − αtϵθ(zt,t,c) √αt

zt −

zt−1 = √αt−1 ·

+ 1 − αt−1 − σt2 · ϵθ(zt,t,c)

+ σtϵt, with ϵt ∼ N(0,I), (1)

Edit Direction Concept Direction CoLan

(1) (2) (3) (4) Visualized Image

|[Figure 3]<br><br>(1)|
|---|

|[Figure 4]<br><br>(2)|
|---|

𝑑others

|[Figure 5]<br><br>(3)|
|---|

|[Figure 6]<br><br>(4)|
|---|

| |
|---|

| |
|---|

Figure 2. Representation manipulation in diffusion models involves adding an accurate magnitude of edit direction (e.g., Image (3) by CoLan) to the latent source representation. Figure 5 and Figure 7 show more examples.

where zt is the denoised image at time t (t = 0,...,T), c is the text embedding of the caption of the image to be sampled, and ϵθ(zt,t,c) models the score function [50] for noise prediction. With the choice of σt = 0, the denoising process in (1) enables DDIM inversion [49]. Specifically, one replaces the forward process with the iterative calling of ϵθ(·,t,c) to predict the forward noise for the source z0. The forward process often stores anchor information (e.g., attention map) [4, 11, 40, 65]. Then the reverse process samples the target image z′0 with visual features corresponding to altered concepts. The key ideas are first to utilize the anchor information such that the denoising process can faithfully reconstruct z0 when conditioned on c (often called sampling in the reconstruction/source branch), and then to further implant (1) with a representation manipulation procedure.

For simplicity, we describe the basic DDIM paradigm here. We further elaborate variants of inversion methods for exact recovery of z0 [18, 65] and different diffusion backbones (e.g., Stable Diffusion [43, 45], Consistency Model [34, 51]) in Appendix §6 and §7.

Steering Reverse Process. Prior works have different choices of representation manipulation in the space of text embeddings [44] or diffusion scores [31] to vary visual synthesis. A general paradigm is to impose an editing direction (∆ϵ or ∆c) to the latent representation (the score function prediction ϵθ(·;t,c) or the text embedding c, respectively) in the editing backbone. Sampling with manipulated concepts is often called sampling in edit/target branch.

• (Text Embedding Space) For instance, the work of [40] edits the concept by manipulating the c in Equation 1 as

cedit = csource + w (cA − cB),

where ∆c = cA − cB is a generalized direction denoting the shift from concept B towards concept A. In fact, this formulation of Vector Addition (VecAdd) can be traced back to word embeddings in language models: the work of [35] observes that cqueen ≈ cking+w·(cwoman − cman), where cking can be viewed as a source latent vector and cwoman − cman as

|[Figure 7]|
|---|

|[Figure 8]|
|---|

Concept 1: cat

Sparse Decomposition

Image Editing Backbone

VisionLanguageModel

Stimulus 1

...... Stimulus K

RepresentationConstruction

Source Representation

Edited Representation

#### =

### =

###### 2: grass

###### Source Image

|𝑤1|
|---|
| |
|𝑤2|
| |
|𝑤3|

###### Edited Image

......

𝑤1

𝑤2

...

Adorable [cat] sits on green grass, with an

###### …

###### …

𝑤3

###### N: sitting

attentive look.

......

…

…

Source Prompt

|𝑤𝑁|
|---|

𝑤𝑁

𝑑𝑑𝑜𝑔 𝑑2 𝑑3

𝑑𝑁

𝑑𝑐𝑎𝑡 𝑑2 𝑑3

𝑑𝑁

###### Target: dog

Adorable [dog] sits on green grass, with an

LLM

Stimulus 1 ......

attentive look.

Concept Interpretation

Concept Transplant

Target Prompt

Stimulus K

Figure 3. The CoLan framework. Starting with a source image and prompt, a vision-language model extracts visual concepts (e.g., cat, grass, sitting) to construct a concept dictionary. The source representation is then decomposed along this dictionary, and the target concept (dog) is transplanted to replace the corresponding atom to achieve precise edits. Finally, the image editing backbone generates an edited image where the desired target concept is incorporated without disrupting other visual elements.

an editing direction. Recent years have also witnessed such an idea being applied to steering activations of large language models (LLMs) [33, 57, 72] and VLMs [6, 30, 53, 54].

• (Score Space) Such an idea has also been applied in the score space by adding a score ϵ with a ∆ϵ. For instance, the work of [2, 8, 13] considered the recipe

ϵedit = ϵθ(·;t,csource)+w(ϵθ(·;t,ctarget)−ϵθ(·;t,csource)), where ϵθ(·;t,csource) can be treated as the source latent vector ϵ, ϵθ(·;t,ctarget)−ϵθ(·;t,csource) as the edit direction ∆ϵ, and w controls the amount of edit. One can also implement csource = ∅ to have the unconditional score. More generally, there is a line of works [5, 8, 13, 59, 62, 65] involving the above formulation to steer or edit the synthesis.

##### 3. Our Method: Concept Lancet

With the above background we are ready to propose our method for accurate representation manipulation in diffusionbased image editing. The high-level idea is that, instead of arbitrarily setting the amount of edit, we will estimate what and how much concepts are present in the source image to inform the edit. This is done via collecting a dictionary of concept vectors in the latent space and decomposing the source latent vector into a linear combination of the dictionary atoms to allow the concept transplant procedures, which we shall discuss in §3.1 and §3.2 respectively.

###### 3.1. Concept Dictionary Synthesis

Here the main goal is to collect a diverse set of concepts (and the corresponding concept vectors in the latent space) that are both visually meaningful and relevant for image editing, such that the decomposition of a source latent vector captures important visual elements and allows potential modifications for effective editing. This naturally boils down to two steps: curating visual concepts for stimulus synthesis and extracting

a concept vector from the stimuli. We describe our approach below and compare it with the alternatives in the literature.

Curating Visual Concepts. Constructing domain-specific concepts is widely adopted for evaluating and controlling generative foundation models [23, 24, 26, 29, 63, 64]. To model the rich semantics of a given concept, an emerging line of work collects textual concept stimuli (i.e., a set of examples, descriptions, and scenarios) for downstream LLM or diffusion editing tasks [33, 40, 57, 72]. There are three issues when applying these concepts in editing images:

- • Many concepts for editing LLMs [33, 72], such as “honesty” or “hardship,” are not catered to image editing in diffusion models. Existing concept stimuli are typically in a specialized format for LLM activation reading (e.g., begin with second-person pronouns).
- • Such concepts primarily focus on single-word descriptor (e.g. “love”, “friendship”), rather than multi-word phrases (e.g., “wearing sunglasses” or “made of wood”) that are helpful to model visual space.
- • Existing collection of concepts for image editing has a limited number of concept vectors open-sourced (e.g., less than 20 in [40] and less than 50 in [32, 41]). To address these issues, we curate a comprehensive set of visual concepts relevant to image editing tasks. Specifically, for each editing task that consists of a source image, a source prompt, and an editing prompt, we employ a VLM [37] to parse the image prompts tuple and generate a list of relevant visual concepts. This step ensures that our concepts are both visually grounded and editing-relevant.

We then instruct an LLM [38] with in-context demonstrations of stimulus synthesis to generate diverse stimuli for each concept to capture various contexts in which the concept shows up. The instructions are shown in Appendix §10. After collecting concepts across all editing tasks, we obtain 5,078 concepts and a total of 152,971 concept stim-

###### bird

###### ancient

###### office desk

###### wearing hat

An office desk often features

Ancient civilizations, such as the

Birds are characterized by their

A cowboy tips his hat in greeting, a traditional gesture of respect

compartments or drawers for

Egyptians and Mesopotamians, made

feathers, beaks, and egg-laying abilities.

organizing stationery, documents,

significant advances in writing and architecture.

and politeness.

and personal items.

Owls are nocturnal birds with

A sleek, minimalist desk with a

A graduate wears a mortarboard cap,

The ruins of ancient Rome, with its iconic Colosseum, offer insight

silent flight, an adaptation for

glass top offers a modern touch to

tossing it into the air upon

hunting at night. Woodpeckers use …

a corporate office space. Many modern desks are equipped …

receiving their diploma. A firefighter wears a helmet to …

into the grandeur of past empires. The ancient Silk Road …

Figure 4. Samples of the concept stimuli from CoLan-150K. Additional samples are attached in the Appendix §8.

uli, which we call CoLan-150K. Figure 4 shows samples of the concept stimuli in our dataset. Compared to existing collections of conceptual representation for diffusion-based editing, CoLan-150K represents a significant scaling up and provides richer and more diverse representations for each concept. By sampling various observations of a concept, the large set of stimuli helps accurately estimate a representation that is robust to changes in contexts.

Concept Vector Extraction. Given the stimuli for each concept, we now need to extract a representative direction in the latent space. Let x be a concept (e.g., “wearing sunglasses”) and sx1,...,sxK be its corresponding stimuli. We first map each stimulus into the latent space using the text encoder E of the diffusion model1. To read a robust concept vector from the collection of embeddings of stimuli, we draw inspiration from prior arts on Representation Reading (RepRead) and propose two options: Arithmetic Average (Avg) [33, 40, 52, 72] or Principal Component Analysis (PCA) [30, 33, 72] on the set of embedding vectors. Avg directly returns the mean of all stimulus embeddings and PCA returns the first principal component of embeddings as the concept vector:

###### dx = RepRead(E(sx1),...,E(sxK)). (2)

For each given source image, a specific collection of concept vectors {dx

i}Ni=1 will form the concept dictionary, which will be further used for decomposition analysis during inference (§3.2). Figure 4 shows samples of concepts and their associated stimuli. We use Avg for the representation reading stage in the experiments since it is more computationally efficient.

###### 3.2. Concept Transplant via Sparse Decomposition

Now that we have obtained a concept dictionary, we are ready to describe how we decompose the latent code of the image along the dictionary and transplant the concept.

Selecting Task-Specific Concepts. While our concept dictionary provides a comprehensive collection of visual concepts, not all concepts are relevant to a specific editing task. To avoid spurious decompositions and make the method efficient, the VLM parses the source image-prompt pair and identifies pertinent task-relevant concepts, as we have done

1For simplicity, we describe the concept extraction in the text embedding space; see the Appendix §7 for the case of the score space.

in §3.1. The corresponding concept vectors are then assembled into a dictionary matrix D ∈ Rd×N, where d is the dimension of the latent space and N is the number of concepts in the dictionary. More details of constructing the dictionary in a specific latent space (e.g., CLIP text embedding space) are shown in Appendix §7.

Concept Analysis. Given a source latent vector v (either from the text encoder or score function), we decompose it along the directions in D through sparse coding. That is, we solve the following optimization problem:

∥v − Dw∥22 + λ∥w∥1 (3)

w∗ = argmin

w

where solutions of concept coefficients w ∈ Rn and λ > 0 is a regularization parameter that controls the sparsity of the solution. In practice, we realize the sparse solver with Elastic Net [66]. Such a decomposition yields

###### v = Dw∗ + r (4)

where w∗ contains the solved coefficients of each concept vector for composition and r is the residual not explained by the concepts in D.

Concept Transplant. To perform the representation manipulation, we construct a modified dictionary D′ by replacing the column of the source concept vector with that of the target concept. The edited latent representation is then obtained as v′ = D′w∗ + r. This transplant scheme preserves the compositional coefficients estimated from the source representation while substituting the relevant concept vector. It imposes the desired concept while maintaining the overall structure of the remaining concepts in the source image.

We note that this concept replacing scheme generalizes to concept insertion and removal. Indeed, concept removal can be viewed as setting the target concept as the null concept; we extract a direction for the null concept using the same procedure as described in §3.1 with stimuli as empty sentences. On the other hand, the case of concept insertion is more subtle since there is no explicit source concept to replace. Hence we instruct the VLM to comprehend the source image and the target prompt to suggest an appropriate source concept as the counterpart of the target concept. For example, if the task is to add concept [rusty] to an image of a normal bike, the VLM will identify the concept [normal] for the concept dictionary and the following replacement.

Table 1. Evaluation of different baselines using Concept Lancet or Vector Addition. The best performance of each category is in bold and the second best is underlined. For each metric under Consistency Preservation, the number on the left is evaluated on the whole image, and the number on the right is evaluated on the background (outside the edit mask).

Consistency Preservation Edit Effectiveness (%, ↑) StruDist (×10−3, ↓)

Representation Manipulation

Inversion Backbone

LPIPS (×10−3, ↓)

SSIM (%, ↑)

Target Image

Target Concept

PSNR (↑)

N.A. DDIM P2P [11] 69.01 39.09 15.04 17.19 340.3 221.3 56.56 70.36 24.35 21.10 N.A. DI P2P [11] 11.02 5.963 22.71 27.24 114.1 54.68 75.08 84.57 24.82 22.07 N.A. DI MasaCtrl [4] 23.34 10.40 19.12 22.78 160.8 87.38 71.12 81.36 24.42 21.37

VecAdd DI P2P-Zero [40] 53.04 25.54 17.65 21.59 273.8 142.4 61.78 76.60 23.16 20.81 CoLan DI P2P-Zero [40] 15.91 6.606 23.08 26.08 120.3 68.43 75.82 83.55 23.84 21.13

VecAdd VI InfEdit [65] 27.18 17.24 21.83 27.99 136.6 56.65 71.70 84.64 24.80 22.04 CoLan VI InfEdit (E) [65] 16.21 8.025 22.13 28.04 125.9 55.05 74.96 84.72 24.90 22.12 CoLan VI InfEdit (S) [65] 13.97 6.199 23.42 28.46 110.3 53.04 75.51 85.12 24.94 22.45

##### 4. Experimental Results

We provide quantitative evaluations for baselines in §4.1 and qualitative observations in §4.2. Finally, we provide visual analysis of the concept vectors from CoLan-150K in §4.3.

###### 4.1. Quantitative Evaluation

We perform a standardized quantitative evaluation of CoLan against current methods with PIE-Bench [18]. Its editing tasks are based on a broad collection of image sources (e.g., TEdBench [20], TI2I benchmark [56]) with diverse scene types and editing categories.

Baselines. We compare editing backbones that fall into two categories based on their concept transfer approach: (1) mechanistic swap of attention maps including P2P [11] and MasaCtrl [4], and (2) representation manipulation that enables us to plug CoLan in the diffusion score space (S) of InfEdit [65] and the text embedding space (E) of both InfEdit and P2P-Zero [40]. We cover multiple inversion approaches such as DDIM [49], Direct Inversion (DI) [18], and Virtual Inversion (VI) [65]. Further implementation details can be found in Appendix §7.

Metrics. The two main criteria are Consistency Preservation and Edit Effectiveness. Consistency Preservation is a set of metrics aimed at evaluating the amount of semantic information preserved during image editing. We report the Structure Distance (StruDist) [55], PSNR [15], LPIPS [69], and SSIM [58]. On the other hand, Edit Effectiveness measures the correctness of the edited part, and it is evaluated by two metrics: Target Image metric computes the CLIP similarity [44, 61] between the edited text and the edited image, whereas Target Concept metric computes the CLIP similarity between the edited text and the edit-masked region of the target image.

Results. Table 1 reports our results. All backbones equipped with CoLan have improved Edit Effectiveness, which indicates that CoLan accurately edits images towards the desired target concept. Moreover, we observe that backbones with

CoLan achieve better consistency preservation across the board. For instance, on the P2P-Zero backbone, CoLan is able to achieve a lower StruDist and LPIPS by nearly 50% and a higher PSNR and SSIM by about 10%. While DI with P2P achieves the best StruDist, CoLan ranks a very close second for StruDist and overall achieves better performance on all rest of the consistency metrics. We argue that StruDist computes an average difference between the DINO-V2 feature maps of the two images. Hence this single metric is largely dependent on a specific transformer, and checking holistically four metrics is a fairer way for consistency evaluation. Notably, InfEdit with CoLan in the score space has the most outstanding performance across multiple metrics.

Additionally, Table 2 shows the average time of sparse decomposition of CoLan using the CLIP space of InfEdit and P2P-Zero backbones. We observe that, since VLM helps make the dictionary concise, the decomposition only occupies a small proportion of the total editing time. This demonstrates that CoLan is efficient and inexpensive relative to the overall computation cost of inference in diffusion models. Furthermore, Table 3 compares the editing performance of CoLan given different dictionary sizes. As expected, we observe that a larger CoLan dictionary is better at capturing the presence of existing concepts in the source image, leading to stronger editing performance. Overall, our quantitative experiments demonstrate that the concept transplant process of CoLan benefits from proper accurate and sparse concept representations that exist in the CLIP space and the diffusion score space for better image editing performance.

###### 4.2. Qualitative Observation

This section provides qualitative results of edited images. We compare the visual quality between images edited with a given backbone and that complemented with CoLan.

###### 4.2.1. Visual Comparison

Each target image can be segmented into two parts: i) the region of interest, which corresponds to the source concept and should be edited to express the target concept; and ii)

|a poster of [a bus driving down] a road with mountains in the background. [a bus driving down] → []|
|---|

|photo of a [goat] and a cat standing on rocks<br><br>near the ocean. [goat] → [horse]|
|---|

|a plate with [steak] on it. [steak] → [salmon]|
|---|

Source Image P2P-Zero + CoLan Source Image P2P-Zero + CoLan Source Image P2P-Zero + CoLan

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

|a lone tree is reflected in the water at night<br><br>[with a bright moon]. [with a bright moon] → []|
|---|

|a [meerkat] puppy wrapped in a blue towel.<br><br>[meerkat] → [lion]|
|---|

|blueberries []. [] → [and strawberries]|
|---|

Source Image P2P-Zero + CoLan Source Image P2P-Zero + CoLan Source Image P2P-Zero + CoLan

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

- Figure 5. Visual comparisons of CoLan in the text embedding space of P2P-Zero. Texts in gray are the original captions of the source images from PIE-Bench, and texts in blue are the corresponding edit task (replace, add, remove). [x] represents the concepts of interest, and [] represents the null concept.

0

0.4

0.8

winter

trees

relaxed

perched

orange

fence

eyes

ears

daylight

cat

an orange

cat sitting

on top of a fence

[Figure 27]

[Figure 28]

0

0.25

0.5

round

plate

organic

orange

earthy

crumbly

creamy

clean

classic

cake

a round cake with orange frosting on a wooden plate

0

0.4

0.8

tranquility

tiger

serene

stripes

pond

nature

habitat

lush

greenalgae

fur

a tiger swimming in a pond of green algae

[Figure 29]

0

0.45

0.9

white

trees

restful

nature

lawn

grass

fur

domestic

dog

calm

a white dog lying on grass

[Figure 30]

- Figure 6. The histograms of solved magnitudes of the concept atoms in CoLan decomposition (text embedding space). As there are tens of concepts in a single dictionary, the histogram includes the concepts whose CoLan coefficients have the top 10 largest magnitudes.

the background, whose contents should be intact through the editing process. Here, we qualitatively analyze these two aspects when using CoLan for image editing.

Ideally, the provided editing should be accurately reflected in the region of interest. We observe that editing with the backbone alone often results in either exaggerated or understated editing. For example, in the task of modifying from [spaceship] to [eagle] in Figure 7 (caption: “a woman in a dress standing in front of a [spaceship]”), the InfEdit backbone alone yields an edited image where the region of interest only resembles an ambiguous bird, whereas an eagle is clearly visible when plugging with CoLan. Moreover, in Figure 5, the example with the caption “a [meerkat] puppy wrapped in a blue towel.” has a blue towel wrapped around the meerkat in the source image. With the P2P-Zero backbone alone, the towel is missing from the output image, whereas the output after plugging CoLan has the blue towel in nearly the same position as that in the source image.

As seen, for both the regions of interest and backgrounds, edited images are of higher quality when a backbone method

Table 2. Average time of sparse decomposition in CoLan for different backbones.

Proportion (%) P2P-Zero

Time (s)

Backbone Metric

Editing Process 38.74 100

Sparse Decomposition 0.153 0.394 Infedit (S)

Editing Process 2.198 100 Sparse Decomposition 0.084 3.82

runs with CoLan. We postulate that this is possible because CoLan respects the geometry of the concept vectors via sparse decomposition. By identifying the correct coefficients, our concept transplant is precise and does not significantly affect non-targeted semantics.

###### 4.2.2. Representation Decomposition

One of the key steps in our approach (discussed in §3.2) is to linearly decompose the latent representation (from the editing backbone) into a sparse combination of dictionary atoms. The success of our downstream editing task hinges on finding a proper set of concepts coefficients that accurately

|[] the arctic sky.<br><br>[] → [a castle in]|
|---|

|[] a white horse running in the field. [] → [watercolor of]|
|---|

a woman in a dress standing in front of a [spaceship]. [spaceship] → [eagle]

Source Image InfEdit + CoLan Source Image InfEdit + CoLan

Source Image InfEdit + CoLan

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

|orange and daisy []. [] → [and glass of water]|
|---|

|white [tiger] on brown ground. [tiger] → [cat]|
|---|

|[smoke]. [smoke] → [fire]|
|---|

Source Image InfEdit + CoLan Source Image Infedit + CoLan Source Image InfEdit + CoLan

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

- Figure 7. Visual comparisons of CoLan in the score space (first row) and text embedding space (second row) of InfEdit. Texts in gray are the original captions of the source images from PIE-Bench, and texts in blue are the corresponding edit task (replace, add, remove).

0

0.3

0.6

white

steamed

smoothtexture

round

light

background

garnish

dumplings

brown bowl

asiancuisine

arrangement

three white dumplings on brown bowl

|[Figure 49]|
|---|

|[Figure 50]|
|---|

0

- 0.5
- 1

wildlife

thinlegs

small

shiny

sharpfocus

perched

detailed

plumage

colorful

bird

beak

a colorful bird standing on a branch

| |
|---|

0

0.4

0.8

smoothsurface

sealing

mechanism

reflective

lid

kitchen item

glassjar

functional

empty

container

closed

a glass jar with a lid closed

|[Figure 51]|
|---|

0

0.25

0.5

teacup

simple

reading

personalspace

natural

materials

indoors

cozy

comfortable

brown

book

brown tea cup and a book

|[Figure 52]|
|---|

- Figure 8. The histograms of solved magnitudes of the concept atoms in CoLan decomposition (score space). The histogram includes the concepts whose CoLan coefficients have the top 10 largest magnitudes.

reflects the semantics in the source image. Here we verify that CoLan indeed finds and analyzes representative concepts that are visibly contributive to the given image.

Figures 6 and 8 present the magnitude histograms of the concept coefficients solved by CoLan in the CLIP space and score space respectively. For decompositions in the score space (Figure 8), take as an example on the leftmost captioned “a colorful bird standing on a branch”. CoLan finds the top three concepts in the image including “bird”, “beak”, and “detailed plumage”, all of which are concepts relevant to the bird in the provided image. Similarly, take the second image captioned “an orange cat sitting on top of a fence” in Figure 6. The top concepts in the histogram are key semantics including “cat”, “fence” and “orange”. Overall, in both spaces, CoLan is able to find descriptive concepts and solve coefficients to accurately reflect the composition of semantics.

###### 4.3. Representation Analysis in CoLan-150K

This section studies the concept vectors obtained from diverse concept stimuli of our CoLan-150K dataset. We evaluate the grounding of the concept vectors in §4.3.1 and the variability of the concept in the edited images in §4.3.2.

Table 3. Average performance of backbones with CoLan for different dictionary sizes (N).

Backbone Metric N = 5 N = 10 N = 20 N = 30 P2P-Zero

LPIPS (×10−3, ↓) 135.6 107.1 80.12 72.85

Target Concept 20.83 20.99 21.10 21.14 Infedit (S)

LPIPS (×10−3, ↓) 56.28 55.87 53.96 53.11 Target Concept 22.05 22.09 22.38 22.40

###### 4.3.1. Concept Grounding

An extracted concept vector is grounded when the vector serves effectively in the editing backbone to impose the corresponding visual semantics in the image. For instance, if we use representation reading [33, 40, 57, 72] to convert the stimuli under [green] to the concept vector, then we would expect to see the color ‘green’ as we add this vector in the image editing backbone.

We verify that our concept vectors are grounded in the following way. For a given concept [x], we extract its concept vector from CoLan-150K. Then we generate the edited images by adding the concept vector in the backbone for every source image. Lastly, we evaluate the difference be-

-[fresh]

-2.0 -1.6 -1.2 -0.8 -0.4 0

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

AppleLotus

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

- Figure 9. Visualizations of edited images with decreasing strength of the concept [fresh] extracted from our CoLan-150K dataset. The values on top correspond to the coefficient wfresh for removing

the concept dfresh. CoLan solves w∗fresh of −0.977 for the apple and −1.16 for the lotus.

tween the CLIP(source image, “x”) and CLIP(edited image, “x”). If the given concept vector is indeed grounded, we would expect to see an increase in the metric. In Table 4, we sample three of the concept directions [watercolor], [dog], [wearing hat], and apply P2P-Zero with CoLan to every source image in PIE-Bench. We further divided the results based on the four image types: Artificial, Natural, Indoor, and Outdoor. Across all image types and our given concepts, we observe a significant increase in the CLIP similarity, which means that the edited images are indeed towards the desired concept direction, and the concept vectors are grounded. The results with more concepts and visualization can be found in Appendix §8.

###### 4.3.2. Comparing Editing Strengths

As we argued in §2, proper image editing requires wellestimated edit strength that depends on the presence of concepts in the given source image. Visualizing the progressive changes of the source image along the desired edit direction [7, 10, 41, 47] offers insights for estimating edit strength. Here we compare the editing effects of the concept vectors from our CoLan-150K dataset with grids of coefficients. Figure 9 and Figure 10 experiment with two scenarios: concept removal and concept addition, respectively.

Take the top row of Figure 10 as an example. Our task here is to add the target concept [green] to our source image of an apple. Our method CoLan solves the concept coefficient w∗green = 0.586. Comparing to edited images by a range of wgreen, we observe that the edited images with wgreen > 0.586 gradually appear over-edited and corrupted, whereas images with 0 < wgreen < 0.586 are still underedited for the target concept. More concretely, we observe that the green color is not visible at wgreen = 0 and is visible

- at wgreen = 0.6. Eventually, a brown patch appears on top at wgreen = 0.9, and the apple morphed into a corrupted object
- at wgreen = 1.5. Similarly, for concept removal in the second row Figure 9, our method CoLan solves the concept coeffi-

cient w∗fresh = −1.16. We observe that result images of the lotus with wfresh < −1.16 appear over-edited, whereas those with 0 > wfresh > −1.16 are under-edited. In summary, our

+[green]

0 0.3 0.6 0.9 1.2 1.5

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

FlowerApple

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Figure 10. Visualizations of edited images with increasing strength of the concept [green] extracted from our CoLan-150K dataset. The values on top correspond to the coefficient wgreen for adding the concept vector dgreen. CoLan solves w∗green of 0.586 for the apple and 0.695 for the rose.

Table 4. Grounding of sampled concept directions in CoLan-150K.

Concept Direction

Image Type

Increase (%, ↑)

Source Edited

Artificial 15.20 18.08 18.95 Natural 12.37 18.31 48.02 Indoor 12.94 16.69 28.98

[watercolor]

Outdoor 14.19 19.08 34.46

Artificial 14.18 19.28 35.97 Natural 13.28 18.65 40.49 Indoor 12.46 18.29 46.81

[dog]

Outdoor 13.08 18.35 40.29

Artificial 12.58 14.77 17.41 Natural 11.73 14.02 19.49 Indoor 10.25 12.18 18.83

[wearing hat]

Outdoor 11.28 13.34 18.28

results demonstrate that a suitable choice of the strength is important for high-quality image editing, and CoLan outputs a solution that has edit effectiveness while preserving the visual consistency.

##### 5. Conclusion

This paper presents Concept Lancet (CoLan), a zero-shot, plug-and-play framework for principled representation manipulation in diffusion-based image editing. By leveraging a large-scale curated dataset of concept representation (CoLan150K), we extract a contextual dictionary for the editing task and perform sparse decomposition in the latent space to accurately estimate the magnitude of concept transplant. Image editing backbones plugging with CoLan achieve state-of-theart performance in editing tasks while better maintaining visual consistency. Through extensive quantitative and qualitative evaluations across multiple perspectives, we demonstrate CoLan’s strong capability to interpret and improve the image editing process. We provide further discussions on limitations, future developments, and societal impacts in Appendix §9.

##### Acknowledgment

This research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via a grant of the BENGAL Program and a grant of the HIATUS Program (Contract #2022-22072200005), by the Defense Advanced Research Projects Agency’s (DARPA) SciFy program (Agreement No. HR00112520300), and by the Penn Engineering Dean’s Fellowship. The views expressed are those of the author and do not reflect the official policy or position of the Department of Defense or the U.S. Government.

##### References

- [1] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In CVPR, 2022.
- [2] Manuel Brack, Felix Friedrich, Katharia Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting, and Apolinario Passos. Ledits++: Limitless image editing using text-toimage models. In CVPR, 2024.
- [3] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023.
- [4] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In ICCV, 2023.
- [5] Siyi Chen, Huijie Zhang, Minzhe Guo, Yifu Lu, Peng Wang, and Qing Qu. Exploring low-dimensional subspaces in diffusion models for controllable image editing. arXiv preprint arXiv:2409.02374, 2024.
- [6] Guillaume Couairon, Matthijs Douze, Matthieu Cord, and Holger Schwenk. Embedding arithmetic of multimodal queries for image retrieval. In CVPR, 2022.
- [7] Rohit Gandikota, Joanna Materzynska, Tingrui Zhou, Antonio Torralba, and David Bau. Concept sliders: Lora adaptors for precise control in diffusion models. arXiv preprint arXiv:2311.12092, 2023.
- [8] Rohit Gandikota, Joanna Materzy´nska, Jaden FiottoKaufman, and David Bau. Erasing concepts from diffusion models. In ICCV, 2023.
- [9] Zigang Geng, Binxin Yang, Tiankai Hang, Chen Li, Shuyang Gu, Ting Zhang, Jianmin Bao, Zheng Zhang, Han Hu, Dong Chen, and Baining Guo. Instructdiffusion: A generalist modeling interface for vision tasks. arXiv preprint arXiv:2309.03895, 2023.
- [10] Zhenliang He, Wangmeng Zuo, Meina Kan, Shiguang Shan, and Xilin Chen. Attgan: Facial attribute editing by only changing what you want. In IEEE TIP, 2019.
- [11] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. Prompt-to-prompt image editing with cross-attention control. In ICLR, 2023.
- [12] Aaron Hertzmann, Charles E. Jacobs, Nuria Oliver, Brian Curless, and David H. Salesin. Image analogies. In SIGGRAPH, 2001.

- [13] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.
- [15] Alain Horé and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In ICPR, 2010.
- [16] Wenjing Huang, Shikui Tu, and Lei Xu. Pfb-diff: Progressive feature blending diffusion for text-driven image editing. arXiv preprint arXiv:2306.16894, 2023.
- [17] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly DDPM noise space: Inversion and manipulations. In CVPR, 2024.
- [18] Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Pnp inversion: Boosting diffusion-based editing with 3 lines of code. In ICLR, 2024.
- [19] Tero Karras, Samuli Laine, and Timo Aila. A Style-Based Generator Architecture for Generative Adversarial Networks. In CVPR, 2019.
- [20] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, 2023.
- [21] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Textto-image diffusion models are zero-shot video generators. In ICCV, 2023.
- [22] Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. Diffusionclip: Text-guided diffusion models for robust image manipulation. In CVPR, 2022.
- [23] Tony Lee, Michihiro Yasunaga, Chenlin Meng, Yifan Mai, Joon Sung Park, Agrim Gupta, Yunzhi Zhang, Deepak Narayanan, Hannah Benita Teufel, Marco Bellagente, Minguk Kang, Taesung Park, Jure Leskovec, Jun-Yan Zhu, Li Fei-Fei, Jiajun Wu, Stefano Ermon, and Percy Liang. Holistic evaluation of text-to-image models. In NeurIPS Datasets and Benchmarks Track, 2023.
- [24] Bowen Li, Zhaoyu Li, Qiwei Du, Jinqi Luo, Wenshan Wang, Yaqi Xie, Simon Stepputtis, Chen Wang, Katia Sycara, Pradeep Ravikumar, Alexander Gray, Xujie Si, and Sebastian Scherer. Logicity: Advancing neuro-symbolic ai with abstract urban simulation. In NeurIPS Datasets and Benchmarks Track, 2024.
- [25] Hang Li, Chengzhi Shen, Philip Torr, Volker Tresp, and Jindong Gu. Self-discovering interpretable diffusion latent directions for responsible text-to-image generation. In CVPR, 2024.
- [26] Manling Li, Shiyu Zhao, Qineng Wang, Kangrui Wang, Yu Zhou, Sanjana Srivastava, Cem Gokmen, Tony Lee, Li Erran Li, Ruohan Zhang, et al. Embodied agent interface: Benchmarking llms for embodied decision making. In NeurIPS Datasets and Benchmarks Track, 2024.
- [27] Senmao Li, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang, and Jian Yang. Stylediffusion: Prompt-embedding inversion for text-based editing. arXiv preprint arXiv:2303.15649, 2023.
- [28] Qiyao Liang, Ziming Liu, Mitchell Ostrow, and Ila R Fiete. How diffusion models learn to factorize and compose. In NeurIPS, 2024.

- [29] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B. Tenenbaum. Compositional visual generation with composable diffusion models. In ECCV, 2022.
- [30] Sheng Liu, Haotian Ye, Lei Xing, and James Zou. Reducing hallucinations in vision-language models via latent space steering. arXiv preprint arXiv:2410.15778, 2024.
- [31] Calvin Luo. Understanding diffusion models: A unified perspective. arXiv preprint arXiv:2208.11970, 2022.
- [32] Jinqi Luo, Zhaoning Wang, Chen Henry Wu, Dong Huang, and Fernando De La Torre. Zero-shot model diagnosis. In CVPR, 2023.
- [33] Jinqi Luo, Tianjiao Ding, Kwan Ho Ryan Chan, Darshan Thaker, Aditya Chattopadhyay, Chris Callison-Burch, and René Vidal. Pace: Parsimonious concept engineering for large language models. In NeurIPS, 2024.
- [34] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.
- [35] Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. Linguistic Regularities in Continuous Space Word Representations. In NAACL HLT, 2013.
- [36] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. arXiv preprint arXiv:2211.09794, 2022.
- [37] OpenAI. GPT-4V(ision) System Card, 2023.
- [38] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [39] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155, 2022.
- [40] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In SIGGRAPH, 2023.
- [41] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery. In ICCV, 2021.
- [42] William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022.
- [43] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.
- [45] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2021.
- [46] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour,

- Raphael Gontijo-Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022.
- [47] Yujun Shen and Bolei Zhou. Closed-form factorization of latent semantics in gans. In CVPR, 2021.
- [48] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. arXiv preprint arXiv:1503.03585, 2015.
- [49] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021.
- [50] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021.
- [51] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.
- [52] Nishant Subramani, Nivedita Suresh, and Matthew Peters. Extracting Latent Steering Vectors from Pretrained Language Models. In ACL Findings, 2022.
- [53] Yoad Tewel, Yoav Shalev, Idan Schwartz, and Lior Wolf. Zerocap: Zero-shot image-to-text generation for visual-semantic arithmetic. In CVPR, 2022.
- [54] Matthew Trager, Pramuditha Perera, Luca Zancato, Alessandro Achille, Parminder Bhatia, and Stefano Soatto. Linear spaces of meanings: compositional structures in visionlanguage models. In ICCV, 2023.
- [55] Narek Tumanyan, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Splicing vit features for semantic appearance transfer. In CVPR, 2022.
- [56] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-toimage translation. In CVPR, 2023.
- [57] Alexander Matt Turner, Lisa Thiergart, David Udell, Gavin Leech, Ulisse Mini, and Monte MacDiarmid. Activation addition: Steering language models without optimization. arXiv preprint arXiv:2308.10248v3, 2023.
- [58] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. In IEEE TIP, 2004.
- [59] Zihao Wang, Lin Gui, Jeffrey Negrea, and Victor Veitch. Concept algebra for (score-based) text-controlled generative models. In NeurIPS, 2023.
- [60] Zhaoning Wang, Ming Li, and Chen Chen. Luciddreaming: Controllable object-centric 3d generation. arXiv preprint arXiv:2312.00588, 2023.
- [61] Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, and Nan Duan. Godiva: Generating open-domain videos from natural descriptions. arXiv preprint arXiv:2104.14806, 2021.
- [62] Chen Henry Wu and Fernando De la Torre. Unifying diffusion models’ latent space, with applications to cyclediffusion and guidance. arXiv preprint arXiv:2210.05559, 2022.
- [63] Xindi Wu, Dingli Yu, Yangsibo Huang, Olga Russakovsky, and Sanjeev Arora. Conceptmix: A compositional image generation benchmark with controllable difficulty. In NeurIPS Datasets and Benchmarks Track, 2024.

- [64] Yongliang Wu, Shiji Zhou, Mingzhuo Yang, Lianzhe Wang, Heng Chang, Wenbo Zhu, Xinting Hu, Xiao Zhou, and Xu Yang. Unlearning concepts in diffusion model via concept domain correction and concept preserving gradient. In AAAI, 2025.
- [65] Sihan Xu, Yidong Huang, Jiayi Pan, Ziqiao Ma, and Joyce Chai. Inversion-free image editing with natural language. arXiv preprint arXiv:2312.04965, 2023.
- [66] Chong You, Chun Guang Li, Daniel P Robinson, and Rene Vidal. Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering. In CVPR, 2016.
- [67] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why vision-language models behave like bags-of-words, and what to do about it? arXiv preprint arXiv:2210.01936, 2022.
- [68] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023.
- [69] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.
- [70] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networkss. In ICCV, 2017.
- [71] Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen. A task is worth one word: Learning with task prompts for high-quality versatile image inpainting. In ECCV, 2024.
- [72] Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, Zico Kolter, and Dan Hendrycks. Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405, 2023.

## Concept Lancet: Image Editing with Compositional Representation Transplant Supplementary Material

We organize the supplementary material as follows.

- • §6 covers additional details of prior arts on diffusion-based editing to complement those mentioned in §2.
- • §7 provides descriptions for collecting the dataset CoLan150K and implementing concept transplant method CoLan.
- • §8 gives extra visualizations on the dataset and the method.
- • §9 discusses limitations, future works and societal impacts.
- • §10 details the prompting templates used in dataset collection and inference.

##### 6. Prior Arts for Diffusion-Based Editing

To generate a new image z0 based on text prompts, diffusion models sample from a standard Gaussian zT and recursively denoise it through the reverse process [49]:

zt−1 =√αt−1fθ(zt,t,c) + 1 − αt−1 − σt2ϵθ(zt,t,c)

+ σtϵt, with ϵt ∼ N(0,I). (5)

Here zt is the denoised image at time t, c is the text embedding of the caption of the image to be sampled, ϵθ(zt,t,c) and fθ(zt,t,c) are two networks that predict the score function [50] and the denoised image z0 respectively given c and zt. As we elaborate below, different choices for αt,σt,fθ give rise to a class of diffusion models for editing.

DDIM-Based Editing. By choosing σt = 0 and fθ(zt,t,c) = z

√1−αtϵθ(zt,t,c)

t−

√αt for every t, the denoising process in (5) yields DDIM sampling [49]. To make sure such a process generates the source image z0 faithfully, one replaces the stand Gaussain zT with noise computed from a special forward process that iteratively adds deterministic noises, computed via ϵθ(·,t,c), to the source image z0. Some regularization can improve the statistical properties of these noises, resulting in better image editability during the denoising process [40]. Recently, the work of [18] have proposed Direct Inversion (DI) to add further guidance, allowing exact recovery of z0 following the source branch and then improving the visual quality of the edited image when concept transfer is imposed.

Consistency-Model-Based Editing. Instead of parameterizing fθ using the learned score ϵθ, one can learn a separate network for fθ(zt,t,c) to approximate the flow map of the probablity flow ODE [50], the deterministic counterpart of DDPM [14] sampling. With the above and the choice of σt = √1 − αt−1 for every t, the process in (5) gives Multi-step Consistency Model Sampling [51], and fθ(zt,t,c) in this case is called the Consistency Model [34, 51]. Through a trained consistency model, one can ideally denoise zt into z0 in one pass of fθ. However, the

denoised z(0t) := fθ(zt,t,c) has low quality if zt is close to a Gaussian, thus a multi-step sampling is adopted to improve the sampled image quality [50]. For the image editing purpose, [65] propose Virtual Inversion (VI) that guides the process to sample the source image at every time t in the

source branch, i.e., z(0t) = z0,∀t.

##### 7. Framework Details

Dataset Collection. Each concept in the CoLan-150K approximately consists of 30 stimuli. We use GPT-4o (with vision module) [38] for parsing source input and proposing the concepts. After curating all concepts, we use GPT-4o (without vision module) to generate diverse concept stimuli. The instructions for them are shown in §10.

Concept Transplant. When constructing the dictionary in the CLIP text embedding space, each concept vector is a sequence of tokens flattened as a single vector of dimension d = 77 × 768 = 59136, where 77 is the maximum number of tokens after padding and 768 is the dimension of token embeddings. For plugging CoLan on the text embedding space of P2P-Zero, we refer to analyzing the process of c + ∆c in Algorithm 1 of [40]. For plugging CoLan on the text embedding space of InfEdit, we refer to decomposing the embedding of its source branch to solve the coefficients. For plugging CoLan on the score space of InfEdit, we refer to analyzing the εconsτ

in Algorithm 2 of [65]. Specifically, given a concept x, its direction dx for concept dictionary in the score space at the time step t is generated as follows:

− εsrcτ

+ εtgtτ

n

n

n

ϵx = ϵθ(·;t,RepRead(E(sx1),...,E(sxK)))

where the RepRead(·) corresponds to the representation reading algorithms described in §3.1.

Evaluation Detail. In Table 1, we evaluate all diffusionbased editing baselines with the backbone of Stable Diffusion V1.5 [45], and consistency-based baselines with the Latent Consistency Model [34] (Dreamshaper V7) which is distilled from Stable Diffusion V1.5. The hyperparameter for the sparsity regularizer λ = 0.01. The null embedding or ∅ in the paper refers to the CLIP embedding of the empty string. When adding/inserting a target concept, as there is no counterpart described in the source caption, we instruct the VLM to propose a counterpart present in the source image and revise the source caption. The revised dataset will be open-sourced together with all concept stimuli. We use P2P-Zero as the backbone for the representation analysis in CoLan-150K and comparing editing strengths. The experi-

|a [black] bird with a yellow beak and yellow feet.<br><br>[black] → [green]|
|---|

|[two boats are docked on the shore of] a lake. [two<br><br>boats are docked on the shore of] → []|
|---|

|a woman in sunglasses sitting at a table [with a bottles of water]. [with a bottles of water] → []|
|---|

###### Source Image Backbone + CoLan Source Image Backbone + CoLan Source Image Backbone + CoLan

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

|a bird standing on [clods]. [clods] → [eggs]|
|---|

|a painting of a woman with [colorful paint] on her<br><br>face. [colorful paint] → [drab paint]|
|---|

|a [white] dog lying on grass.<br><br>[white] → [leopard]|
|---|

Source Image Backbone + CoLan Source Image Backbone + CoLan Source Image Backbone + CoLan

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

|a [colorful] bird standing on a branch.<br><br>[colorful] → [red]|
|---|

|three white [dumplings] on brown bowl.<br><br>[dumplings] → [sushi]|
|---|

|an orange van with [surfboards] on top.<br><br>[surfboards] → [flowers]|
|---|

Source Image Backbone + CoLan Source Image Backbone + CoLan Source Image Backbone + CoLan

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

|[photograph] - window of the world by jimmy kirk. [photograph] → [painting]|
|---|

|a cat sitting in the [grass].<br><br>[grass] → [rocks]|
|---|

|a [rabbit] is sitting in a pile of colorful eggs. [rabbit] → [cat]|
|---|

###### Source Image Backbone + CoLan Source Image Backbone + CoLan Source Image Backbone + CoLan

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

|a painting of a cup with a [smoke] coming out of it. [smoke] → [flower]|
|---|

|a cute dog holding a [red] heart. [red] → [pink]|
|---|

|a [tiger] swimming in a pond of green algae.<br><br>[tiger] → [dog]|
|---|

###### Source Image Backbone + CoLan Source Image Backbone + CoLan Source Image Backbone + CoLan

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

|a painting of [a dog in] the forest.<br><br>[a dog in] → []|
|---|

|a big ship [in a bottle] on the dark ocean. [in a bottle] → []|
|---|

|a man sitting on a rock with [trees] in the<br><br>background. [trees] → [a city]|
|---|

###### Source Image Backbone + CoLan Source Image Backbone + CoLan Source Image Backbone + CoLan

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

- Figure 11. Additional visual comparison of CoLan in the text embedding space of P2P-Zero. We observe that the backbone plugging with CoLan has editing results that visually better align with the task.

ments in §4 are performed on a workstation of 8 NVIDIA A40 GPUs.

Pipeline. Algorithm 1 shows the full procedure of our proposed framework CoLan. The first part of the algorithm is to extract a set of concept vectors from the input editing image-text tuples based on § 3.1), followed by the second part where we transplant the target concept via sparse decom-

position in § 3.2. In the first part, we instruct a VLM to parse the source input into a set of relevant concepts, and then we instruct an LLM to generate concept stimuli for every concept. Using the concept stimuli, we extract a collection of concept vectors using representation reading from the latent space of our diffusion model. Then, in the second part of CoLan, we decompose the text embedding or diffusion score

Algorithm 1: Concept Lancet (CoLan) for Diffusion-based Image Editing Input: Frozen diffusion-based image editing backbone Fθ, image editing tuples (source prompt, source image, target

prompt) P = {(pi,qi,p′i)}Ni=1q Parse P with the vision-language model to collect the concepts X = VLM(P) For each concept xi ∈ X : ▷ §3.1: Concept Dictionary Synthesis

Instruct the LLM to synthesize concept stimuli {sx Extract the concept vector dx

- i
- j }Kj=1 = LLM(xi)

= RepRead((sx Stack concept vectors {dx

- i
- j }Kj=1)

i

i}N

i=1 as columns of the concept dictionary D.

x

For each source prompt-image pair (pi,qi) ∈ P: ▷ §3.2 Concept Transplant via Sparse Decomposition Encode pi to the text embedding space or (pi,qi) to the diffusion score space as the source representation v Solve for the compositional coefficients that reconstruct the source w∗ = argminw ∥v − Dw∥22 + λ∥w∥1 Curate a modified dictionary D′ by replacing the column of the source concept with that of the target concept Obtain the edited latent representation as v′ = D′w∗ + r. Generate the edited image through the image editing backbone qi′ = Fθ(v′).

Output: The edited images Q′ = {q′i}Ni=1q .

Source Images

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

SourceEdited

Fixed Edit Strength of [dog] = 0.7 Without CoLan Estimation

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Edit Strength of [dog] Estimated by CoLan

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Edited

- Figure 12. Visualizations of editing results. The first row shows the source images, the second row shows the results with the fixed edit strength of 0.7 for the concept [dog] without CoLan analysis, and the third row shows the edit results with CoLan analysis.

of the source representation using sparse coding techniques. After obtaining the coefficients of each concept vector, we perform a transplant process with the customized operation of removing, adding, or replacing. Finally, we synthesize the edited images with the modified latent representation with the image editing backbone.

##### 8. Additional Results

This section provides additional results for CoLan. It includes more editing improvements with baseline models and visualization of concept instances from our CoLan-150K dataset.

Visual Comparison. Figure 11 shows additional visualization of the image editing results. The experiment settings follow §4.2. We observe that the editing backbone has a better editing performance after plugging CoLan.

Concept Grounding. Figure 13 visualizes the edited images with the extracted concept vectors [watercolor], [dog], and [wearing hat] from the stimuli of our CoLan-150K dataset. We observe that the edited images correctly reflect the semantic meaning of the concepts, which indicates that our concept stimuli successfully ground the concept. Figure 14 further shows additional samples of concepts and their stimuli. Note that there are approximately 30 stimuli per concept, and our figure shows the first three for each concept.

Edit Strength. Figure 12 shows the editing results from source images of the cat to the target concept dog without or with CoLan. The synthesis setting follows the Comparing Editing Strengths section in §4 and we fix the edit strength to 0.7 if CoLan is not used. From the second row of the figure, we observe that different source images of the cat require different magnitudes of editing, and simply choosing

[dog] [wearing hat]

[watercolor]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Source

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

Edited

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

- Figure 13. Visualizations of concept grounding for sampled concepts from our CoLan-150K dataset. We observe that the extracted concept vectors from our dataset corresponds to the desired semantics by visualization.

a unified strength for all source images will frequently result in unsatisfactory results for different images (under-edit or over-edit). Then in the third row of the figure, we show that editing with CoLan results in more consistent and reasonable visual results. This is because our framework adaptively estimates the concept composition of each image and solves customized edit strengths for each source image.

##### 9. Limitations, Future Works, Societal Impacts

While CoLan demonstrates strong performance for diffusionbased image editing, we elaborate on potential limitations and directions for future work in this section.

Limitation. The current framework primarily operates upon diffusion-based backbones with attention-control mechanisms where source concepts correspond to certain regions of interest. It will be challenging to perform spatial manipulations that require editing across different sectors of attention maps. For instance, consider tasks such as moving the cat from right to left or relocating the table to the corner, which shall require non-trivial operations in the attention modules. Another challenge lies in handling numerical modifications, such as changing the number of objects (e.g., changing an image of two cats to have three cats) or composing numerical relations with multiple different objects (e.g., adding two apples to an image of three bananas).

Future Work. Future work could explore methods to enhance CoLan’s capabilities to handle spatial relationships and global layout modifications while preserving its precise concept manipulation advantages. For numerical editing, it is worthy exploring the bag-of-words effect of CLIP or how the diffusion model shall encode numerical concepts in a way

that straightforward manipulation is permitted [60, 63, 67]. The precise mapping between numerical concepts and their representations in the latent space warrants further investigation to enable more sophisticated counting-based edits.

Societal Impact. Image editing frameworks with high useraccessibility (through the prompt-based interface) raise considerations about potential misuse. The ability to perform precise conceptual edits could be exploited to create misleading, controversial, or deceptive content. While our framework focuses on enhancing editing quality, future development should incorporate safeguarding against malicious requests and protecting copyrights in content creation.

##### 10. Prompting Template

As mentioned in Section 3.1, we instruct the VLM to perform two tasks: rewriting prompts for concept addition or insertion and constructing detailed concept dictionaries. We then instruct the LLM to synthesize concept stimuli. Figure 15 shows the instructions (prompting template) for rewriting captions by identifying source concepts and generating rewritten prompts tailored for image editing tasks. Figure 16 shows the instructions for constructing a comprehensive concept list by parsing multimodal information from the source input. This ensures that the list captures diverse and unique aspects of the source image and prompt. Finally, Figure 17 shows the instructions for generating diverse and contextually rich concept stimuli, which enables the mapping to conceptual representations.

###### brown hair

###### candle

###### cloudy sky

###### desert

Brown hair can range in shade from light caramel to deep chocolate, providing a rich variety of options. Individuals with brown hair may have natural highlights that give

Deserts are characterized by their arid environment and receive less than 25 centimeters of rain annually. The Sahara Desert is the largest hot desert in the world, stretching

Candles provide soft, ambient

The sky is overcast with thick

lighting that can create a cozy

cumulus clouds, creating a soft,

atmosphere in any room.

diffuse light over the landscape. A cloudy sky casts a muted gray hue, affecting the mood of the

A flickering candle casts dancing

shadows on the walls, adding a

depth and dimension to their hair

scene below with an introspective

across multiple countries in North

sense of warmth and comfort. Scented candles are infused with fragrances like lavender or

color. Chestnut brown is a warm, reddish-

calm. Patches of blue peek through

Africa. Erosion from wind and water can

brown hair color that glows

scattered clouds, hinting at the

create dramatic rock formations

vanilla, helping to create a

beautifully in sunlight.

possibility of a clearing sky.

found in desert environments.

relaxing environment.

###### digital art

###### elegant

###### fireplace

###### fluffy

Digital art is created using

An elegant ballroom filled with

A fireplace crackles softly,

Fluffy clouds drift lazily across

software tools such as Adobe

chandeliers and lavish decorations

providing both warmth and a cozy

the sky, resembling puffs of cotton

Photoshop, Corel Painter, or

sets the stage for a grand event.

ambiance on a cold winter night.

wool.

Procreate.

The evolution of digital art has

The elegance of a swan gliding gracefully across a serene lake at

The mantel above the fireplace is often used to display photographs

A fluffy white cat purrs contentedly as it curls up on a

enabled artists to explore new

styles and techniques not possible

dusk.

or holiday decorations. Fireplaces can be fueled by wood,

soft cushion.

with traditional media.

Artists often use digital tablets

The minimalist design of a Scandinavian interior reflects a

and styluses to draw and paint,

gas, or electricity, each offering

Freshly fallen snow creates a

simulating the feel of natural

distinct characteristics and

fluffy blanket over the landscape.

refined sense of elegance.

media.

maintenance needs.

###### garden

###### glass container

###### historic architecture

###### in the city

Gardens can be designed to include

Historic architecture often

Glass containers are typically

A bustling city street is full of

a variety of plants, including ornamental flowers, shrubs, and

features intricate carvings and stonework that tell a story of the

transparent, allowing for easy

traffic, with cars honking and

viewing of their contents.

people rushing by.

trees.

era in which it was built.

A vegetable garden is cultivated to

The Roman Colosseum, with its

A mason jar is a popular type of

The city skyline is dominated by

grow produce such as tomatoes,

massive stone arches and elliptical

glass container used for preserving

towering skyscrapers, casting long

lettuce, and carrots for home

structure, is a testament to

food through canning.

shadows as the sun sets.

consumption.

ancient engineering prowess.

A butterfly garden is planted with

Glass containers are often used to

Victorian-era buildings are

Sidewalk cafes in the city are

colorful, nectar-rich flowers like

store spices in the kitchen, with

characterized by their ornate

popular spots for people to sip

milkweed and lantana to attract

airtight seals to maintain

detail, steep gables, and patterned

coffee and watch passersby.

butterflies.

freshness.

brickwork.

###### keyboard

###### living room

###### looking at the camera

###### marble wall

A mechanical keyboard features

The living room often serves as the

Marble walls are often associated

A child looks directly into the

individual switches for each key,

central gathering space for

with luxury and elegance due to the

camera, capturing an expression of innocent curiosity.

providing tactile feedback and

families to relax and entertain

stone's rich veining and natural

durability.

guests.

beauty.

Wireless keyboards offer

A cozy living room features a large, plush sofa adorned with soft, colorful throw pillows.

The bride and groom pose for a photo, both looking directly at the camera with wide smiles.

A marble wall can have a highly polished finish that reflects light and brightens a room.

portability and convenience by

using Bluetooth or RF technology for connectivity. Ergonomic keyboards are designed to

The unique patterns of veining in

The living room is often equipped with a wall-mounted television and a sound system for entertainment.

A group of friends gathers for a selfie, all leaning in and making eye contact with the camera lens.

reduce strain on the wrists and

marble walls can create a one-of-a-

hands, often featuring split designs or curved layouts.

kind visual impact, with swirls and lines varying from slab to slab.

###### on the grass

###### purple

###### spaceship

###### snowflakes

Purple has long been associated

Spaceships are engineered for

A toddler crawls on the grass, giggling as they touch the soft blades for the first time.

Snowflakes are intricate ice crystals that form in the atmosphere under cold conditions.

with royalty and luxury, stemming

travel beyond Earth's atmosphere,

from the rarity of the dye historically used to create it.

equipped with advanced propulsion systems.

A field of lavender in bloom paints the landscape a soft purple, with a calming fragrance wafting through the air.

Each snowflake has a unique pattern, with no two snowflakes ever being identical due to their complex formation process.

A couple enjoys a picnic on the

A spaceship hovers silently in deep

grass under the shade of a

space, its surface gleaming under

sprawling oak tree.

distant starlight.

Morning dew glistens like tiny

The twilight sky often transitions

The interior of a spaceship often

Snowflakes typically have a

pearls on the grass, catching the

through shades of purple as the sun

features zero-gravity living

hexagonal shape due to the

light of the rising sun.

sets, creating a peaceful ambiance.

quarters and control panels.

molecular structure of ice.

###### table lamp

###### tennis ball

###### vibrant

###### wooden

A tennis ball is typically made of

Table lamps often come with a

Wooden furniture often showcases

a rubber core covered in a fuzzy felt material, giving it a

The vibrant colors of a sunset cast a warm glow over the horizon.

fabric shade that diffuses light to

the natural grain and beauty of the

create a cozy ambiance.

wood used.

distinctive texture. Tennis balls are often bright

A touch-activated table lamp allows

A vibrant city street bustling with

The creaking of a wooden floor adds character to an old house.

users to adjust the brightness with a simple tap.

yellow to enhance visibility during play on the court.

life, full of people, street vendors, and musicians.

The standard diameter of a tennis

Table lamps can feature adjustable

A vibrant painting captures the eye

A wooden sculpture carved from oak

ball is about 6.7 centimeters,

necks, making them perfect for reading or focused tasks.

with its bold use of color and dynamic brushstrokes.

can highlight the artist's skill and attention to detail.

conforming to international

regulations.

###### Figure 14. Additional samples of the concept stimuli from CoLan-150K. Each concept consists of approximately 30 stimuli and this figure samples the first three for a concept.

###### Rewriting Captions for Concept Addition/Insertion

You are one of the best experts in Generative Models and Concept Learning in the world. You are very good at designing concept dictionary to research the representation in latent space from CLIP or Score-based Generative Models, which have wide applications in image editing. You are a great expert in understanding and parsing multimodal information from a given image. Now, given a source prompt, a target prompt, and a source image, your task is to rewrite the source prompt for the image editing task. Usually, there is a focused pair of concepts in the source prompt and the target prompt to be edited (e.g., "cat" to "dog"). The source concept is usually annotated in the brackets ("[]") in the source prompt. However, in some editing tasks, there is no clear source concept mentioned in the source prompt. Hence, for these tasks, you are required to comprehend the source image and identify the corresponding source concept. After comprehending the source image, you need to generate a re-written source prompt with a clearly annotated source concept.

Here are two demonstrations: Source Prompt: a slanted mountain bicycle on the road in front of a building Target Prompt: a slanted [rusty] mountain bicycle on the road in front of a building Source Concept: "" Target Concept: "rusty" Source Image: (IMG) Re-written Source Prompt: a slanted [new] mountain bicycle on the road in front of a building

Source Prompt: two birds sitting on a branch Target Prompt: two [origami] birds sitting on a branch Source Concept: "" Target Concept: "origami" Source Image: (IMG) Re-written Source Prompt: two [real] birds sitting on a branch

The identified source concept should not be the same as the target concept. The response MUST be with brackets ("[]") around the source concept. You should not use "without" frequently. Try your best to comprehend the image. You should only output the re-written source prompt. DO NOT print anything else such as "Here are ...", "Sure, ...", "Certainly, ...". DO NOT print quotation marks unless necessary. Just return the string.

Source Prompt: <input> Target Prompt: <input> Source Concept: <input> Target Concept: <input> Source Image: <input> Re-written Source Prompt: <fill the response here>

- Figure 15. The instructions for rewriting the task of concept addition/insertion with the VLM-found source concept as the counter-part.

###### Concept Dictionary Construction

You are one of the best experts in Generative Models and Concept Learning in the world. You are very good at designing concept dictionary to research the representation in latent space from CLIP or Score-based Generative Models, which have wide applications in image editing. You are a great expert in understanding and parsing multimodal information from a given image. Now, given a source prompt, a target prompt, and a source image, your task is to parse the given information into a concept list. The concept list consists of concepts, attributes, objects, and items that comprehensively describe the source image and cover the source prompt. Your concept list must have at least 15 concepts. As the concept list is for the task of image editing, there is a focused pair of concepts in the source prompt and the target prompt to be edited. The source concept is usually annotated in the bracket ("[]") in the source prompt. You must put the focused concept in the source prompt as the FIRST atom in the concept list. You must NOT put the focused concept in the target prompt in the concept list.

Here are three demonstrations: Source Prompt: a [round] cake with orange frosting on a wooden plate Target Prompt: a [square] cake with orange frosting on a wooden plate Source Concept: "round" Target Concept: "square" Source Image: (IMG) Concept List: ["round", "cake", "orange", "frosting", "wooden", "plate", "swirl", "creamy", "crumbly", "smooth", "rustic", "natural", "muted", "handmade", "warm", "minimalist", "unfrosted", "botanical", "bark", "inviting", "cozy", "textured", "simple", "organic", "earthy", "soft", "classic", "contrasting", "neutral", "clean"]

Source Prompt: a painting of [a dog in] the forest Target Prompt: a painting of the forest Source Concept: "a dog in" Target Concept: "" Source Image: (IMG) Concept List: ["a dog in", "painting", "forest", "trees", "leaves", "sunlight", "vibrant colors", "orange hues", "pink trees", "purple plants", "playful", "cartoonish", "nature", "animals", "butterflies", "fantasy", "surreal", "whimsical", "tall trees", "shadows", "depth", "light beams", "foliage", "dynamic", "warm tones", "imaginative", "dreamlike", "motion", "soft textures", "layered composition", "bright atmosphere"]

Source Prompt: blue light, a black and white [cat] is playing with a flower Target Prompt: blue light, a black and white [dog] is playing with a flower Source Concept: "cat" Target Concept: "dog" Source Image: (IMG) Concept List: ["cat", "black", "white", "blue light", "flower", "playing", "paws", "stone path", "curious", "whiskers", "small", "fluffy", "outdoor", "pink petals", "focused", "nature", "detailed fur", "green stem", "bright", "youthful", "movement", "natural light", "close-up", "gentle", "exploration", "soft shadows", "grass between stones", "alert", "innocent", "delicate"]

The concepts in the list should not be redundant or repetitive. Each concept in the list represents a unique perspective of objects, styles, and contexts. The response MUST be in Python list format. You should have at least 15 concepts in the list. You should only output the Python list. DO NOT print anything else such as "Here are ...", "Sure, ...", "Certainly, ...". Just return the list ["", "", ...,

..., ""].

Source Prompt: <input> Target Prompt: <input> Source Concept: <input> Target Concept: <input> Source Image: <input> Concept List: <fill the response here>

Figure 16. The instructions for the VLM to parse the source image-prompt tuple into the concept list for the concept dictionary.

###### Concept Stimulus Synthesis

You are one of the best experts in Generative Models and Concept Learning in the world. You are very good at generating concept stimuli to research the representation in latent space from CLIP or Score-based Generative Models, which have wide applications in image editing. You are a great expert in providing relevant information and scenarios based on a given concept. Now, given a concept, your task is to generate 30 (THIRTY) instances of concept stimuli for a given concept. As the concept stimuli will be used for the task of image editing, we need comprehensive, diverse, and accurate descriptions and examples for the concept.

Here are three demonstrations of the concept and its corresponding concept stimuli: Concept: dog Concept Stimuli: [ "Dogs are known for their loyalty and strong bonds with humans.", "A dog wags its tail excitedly when it sees its owner after a long day.", "Puppies often chew on objects as a way to explore their environment.", "The sound of a dog’s bark can vary depending on its breed and mood.", "Dogs rely heavily on their sense of smell, which is far more sensitive than that of humans.", "A dog runs alongside its owner during a morning jog, full of energy.",

... ]

Concept: cat Concept Stimuli: [ "Cats are known for their graceful, stealthy movements.", "A cat stretches lazily under the warm afternoon sun.", "Kittens explore their surroundings with curiosity and playfulness.", "A cat’s purring has been shown to have a calming effect on humans.", "Stray cats often rely on their instincts and sharp senses for survival.", "The eyes of a cat reflect light in the dark, giving them superior night vision.",

... ]

Concept: cake Concept Stimuli: [ "Cakes are often baked in layers and filled with frosting or cream in between each layer.", "A slice of cake reveals its moist interior, topped with a rich layer of chocolate ganache.", "Cakes are a common centerpiece for celebrations such as birthdays, weddings, and anniversaries.", "A cake adorned with fresh berries and whipped cream makes for a light, summery dessert.", "Cupcakes are miniature cakes baked in individual paper liners and often topped with buttercream frosting.", "The aroma of a freshly baked vanilla cake fills the kitchen with a warm, sweet scent.",

... ]

The concept stimuli in the list should not be redundant or repetitive. Each stimulus in the list represents a unique perspective (e.g., styles, contexts, examples, attributes, descriptions, usages) of the concept. The response MUST be in Python list format. You should have at least 30 stimuli in the list. You should only output the Python list. DO NOT print anything else such as "Here are ...", "Sure, ...", "Certainly, ...". Just return the list ["", "", ...,

..., ""]. Concept: <input> Concept Stimuli: <fill the response here>

Figure 17. The instructions for the LLM to generat diverse stimuli given a concept.

