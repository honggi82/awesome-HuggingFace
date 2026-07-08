# arXiv:2602.15727v2[cs.CV]28Jun2026

## Spanning the Visual Analogy Space with a Weight Basis of LoRAs

Hila Manor2†,1 , Rinon Gal2† , Haggai Maron2,1 , Tomer Michaeli1 , and Gal Chechik2,3

1Technion, Israel 2NVIDIA, Israel 3Bar-Ilan University, Israel

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

“Turn this photo into an architectural rendering”

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

“Give this animal a fantastical set of armor”

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

“Turn this image into the Clay_Toy style”

“Turn this into image into the Ghibli style”

| |
|---|

| |
|---|

Fig. 1: LoRWeB. We present a novel method for analogy-based editing via learnable mixing of LoRAs. Given a prompt and an image triplet {a, a′, b} depicting a desired transformation, LoRWeB dynamically constructs a single LoRA from a learnable basis of LoRAs, and produces an editing result b′ that applies the same analogy to b.

Abstract. Visual analogy learning enables image editing via demonstration rather than textual description, allowing users to specify complex transformations difficult to articulate in words. Given a triplet {a, a′, b}, the goal is to generate b′ such that a : a′ :: b : b′. Recent methods adapt text-to-image models with a single Low-Rank Adaptation (LoRA) module, but they face a fundamental limitation: attempting to capture the diverse space of visual transformations within a fixed module constrains generalization. Inspired by recent work showing that LoRAs in constrained domains span meaningful, interpolatable semantic spaces, we propose LoRWeB, which specializes the model for each analogy task in a single inference pass. LoRWeB dynamically composes learned transformation primitives, informally, choosing a point in a “space of LoRAs”. We introduce two key components: (1) a learnable basis of LoRAs to span the space of different visual transformations, and (2) a lightweight encoder that dynamically weighs these basis LoRAs given the input analogy pair. Comprehensive evaluations demonstrate state-of-the-art performance and significantly improved generalization to unseen transformations. Our findings suggest LoRA basis decompositions are a promising direction for flexible visual manipulation tasks. See our website for code.

Keywords: Image analogies · Image editing · LoRA · Flow models † This work was done while HM and RG were working at NVIDIA.

### 1 Introduction

Text-based image editing models [5,6,59,71,79] have recently emerged as powerful tools for controllable image generation and manipulation, enabling users to modify images through textual descriptions. However, many visual transformations are inherently difficult to articulate precisely through text alone. For example, consider describing the transformation that converts a photograph into the style of a specific painting, or conveying an exact target pose through text. Such inherent limitations motivate the need for alternative paradigms that can capture and apply complex visual transformations.

Visual analogy learning [27] offers a compelling solution to this challenge by enabling models to understand transformations through examples rather than explicit descriptions. In this paradigm, given a triplet of images {a,a′,b}, the goal is to generate an image b′ such that the visual relationship a : a′ :: b : b′ holds. That is, the transformation applied between a and a′ should be analogously applied to b to produce b′. This approach allows users to specify complex visual changes through demonstration, making it possible to capture nuanced transformations that would be difficult or impossible to describe textually.

Early learning-based approaches trained stand-alone analogy models directly from analogy data [4,39,54,68,69,72], but this lead to limited task diversity and image quality, or required extensive compute. More recent work aims to leverage the rich prior of powerful text-to-image backbones by adapting them to the visual analogy task, using a single Low-Rank Adaptation (LoRA) module [19,41,60]. While effective, these methods face a fundamental limitation: they attempt to capture the diverse space of possible transformations within a single adaptation module. This constraint may limit the model’s ability to generalize across the rich variety of relationships that exist in images.

We hypothesize that specializing the model to each specific analogy task at inference time may improve performance and generalization. While this objective could theoretically be achieved via hypernetworks that generate task-specific LoRAs [60], these are notoriously difficult to train and often suffer from instability [48]. Instead, we draw inspiration from recent work [12] demonstrating that fine-tuned LoRAs (e.g., for personalization tasks) can form an interpretable weight space. In this space, individually-trained LoRAs act as a basis of fundamental and semantic visual traits, and interpolating between the weights of these LoRAs can effectively cover new points in this semantic space, which allows for creating new, blended concepts. Building on this insight, we explore a similar principle for visual analogy learning and propose LoRWeB, a two-component system: (1) a learnable basis of LoRA modules and (2) a lightweight encoder that dynamically combines LoRAs from this basis at inference time based on the input analogy pair. These components are jointly trained, enabling the model to compose appropriate transformations for novel analogies unseen during training.

Existing methods typically encode analogy images using vision-language models such as CLIP [51] or SigLIP [75] and provide these encodings as context to the generative model. This can provide the higher-level semantic understanding needed for understanding the analogy task. However, this might lead to loss of

detail in fine-grained visual detail preservation. Recent advances have shown that diffusion models can extract remarkably accurate visual details through extended attention mechanisms [5,8]. Thus, we leverage this capability by providing the full analogy triplet directly to the diffusion model through an extended-attention mechanism, while reserving CLIP-based encodings specifically for LoRA selection. This approach allows LoRWeB to balance fine-detail consistency with the higher-level semantics required to understand the analogy task.

We evaluate LoRWeB against established baselines and show it achieves stateof-the-art results. Our contributions include: (1) a novel architecture that decomposes visual analogy learning into a basis of LoRAs with dynamic composition, and (2) a comprehensive evaluation showing improved generalization to unseen transformations compared to existing single-LoRA approaches.

### 2 Related Work

Visual Analogies. Visual analogies, also known as “Image Analogies” [27], “Visual Prompting” [4] or “Visual Relations” [19], is the task of learning a transformation from a pair of before-and-after exemplars and applying it analogously to new images. Early non-neural methods learned explicit per-pair filters for simpler tasks such as style transfer [27], or per-pair optimization for relighting in

##### 3D [15]. With the advent of network-based methods, initial works proposed models conditioned on image embeddings or NeRF [45] representations to present analogies through simple vector arithmetic [14,23,36,54]. While these methods showed promise on datasets of simple, isolated objects, they struggled with the complexity of real-world images, and still mostly tackled style-transfer analogies. Newer methods instead treat analogy learning as in-context learning, where the model is directly conditioned on the exemplar pair and a reference image, and is trained to successfully synthesize the matching target [4,68,69,72]. More recently, some works adapt pre-trained text-to-image foundation models to the new task, going beyond simple style-transfer analogies. For example, [61] use per-sample optimization, backpropegating through the entire diffusion process. However, such approaches can require dozens of minutes to edit every image, and their memory requirements can be challenging with newer, larger models. Another approach adapts the foundation model directly using a LoRA module [9,19,28]. These methods, while showing impressive results, still struggle to generalize to unseen tasks. Our approach aims to tackle this limitation by avoiding the bottleneck of a single LoRA, opting instead to train a basis of adapters which can be mixed to achieve greater flexibility and better generalization.

Diffusion-Based Image Editing. The unprecedented semantic control offered by large scale text-to-image diffusion models [5,53,55] has inspired extensive work leveraging them as priors for image editing. Early works add noise to an image and remove it conditioned on a novel prompt [44], though such methods significantly change image structure. Subsequent work improved content preservation by manipulating internal feature representations [25, 49, 67]

or the model’s denoising trajectory [11,24,31,34]. Recent works go beyond text and incorporate different control modalities for enhanced precision, such as ControlNet [8,77], or attention-sharing [1,18,26,65]. Others explore text-free editing to enable modifications that cannot be textually described [22,42], though without direct control. Transformer-based diffusion models further popularized attention-sharing for maintaining subject consistency in personalization [17,56] and editing [7,62,64]. Among these, Flux.1-Kontext [5] was specifically trained for text-based editing, incorporating input images via extended attention mechanisms. Our work extends this model’s capabilities to visual analogies.

LoRA and Weight Bases. LoRA [28] is a parameter-efficient fine-tuning method that modifies a model using low-rank matrices learned on top of the existing weights. Its success lead to a range of downstream approaches trying to improve on the original formula. Of these, a line of work explores the combination of multiple LoRa modules, either to combine them post-tuning [58,76], or as a means of turning an existing model into a mixture of experts [13,43,70]. In visual content generation, a recent work [12] showed that independently trained LoRA weights can span a semantic basis, and interpolations between them can be meaningful. However, to use their findings on a practical task such as face personalization, their approach required 65,000 independently trained LoRAs, and further employed PCA to span this basis, using test-time tuning per subject. Similar observations on weight bases of LoRAs were made in language processing: LoraHub [29] independently trained LoRAs and combined them post-tuning with test-time optimization to enable new tasks at inference given multiple reference outputs of the new tasks, and Sci-LoRA [10] combined LoRAs for tasks like text simplification across different scientific domains. We propose to further expand on this idea by learning a joint basis of LoRAs, along with the router to mix and match between them efficiently at inference time. Thus, we can learn a basis that is more amenable to interpolations, and enable better downstream generalization. Our approach does not necessitate the extensive inference-time compute required by prior approaches, rather creating an input dependent LoRA with a single forward pass at inference time, without test-time optimization.

### 3 Method

#### 3.1 Preliminaries

Low-Rank Adaption. LoRA [28] offers a parameter-efficient alternative to conventional fine-tuning of large models by learning low-rank matrices that adapt the pre-trained weights. Specifically, starting from a frozen pre-trained weight matrix W0 ∈ Rm×n, the update of the weights is represented as the product of two learned low-rank matrices ∆W = BA, where B ∈ Rm×r and A ∈ Rr×n, and the rank r is typically r ≪ min(m,n). This formulation drastically reduces the number of trainable parameters, while typically maintaining model performance. The final weights of the model are then updated to W = W0 + αr BA, where α is a scaling constant.

[Figure 17]

[Figure 18]

| |
|---|

[Figure 19]

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

| |
|---|

| |
|---|

| |
|---|

[Figure 20]

| |
|---|

| |
|---|

| |
|---|

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

| |
|---|

[Figure 25]

[Figure 26]

| |
|---|

| |
|---|

[Figure 27]

[Figure 28]

| |
|---|

| |
|---|

| |
|---|

- Fig. 2: LoRWeB Overview. We first encode a and a′, that describe a visual transformation (e.g. adding a hat to the man), and b, which should be edited analogously (e.g. adding a hat to the woman) with CLIP [51], and a small learned projection module. The similarity between the encoded vector and a set of learned keys determines the linear coefficients for combining the learned LoRAs into a single, mixed LoRA. This mixed LoRA is injected into a conditional flow model (e.g. Flux.1-Kontext [5]). Next, we build a 2 × 2 composite image from {a, a′, b}. The conditional flow model gets this composite image as its input, along with a guiding edit prompt, and produces a composite image with the edited results b′ in the bottom-right quadrant.

Flow Models. Flow-based generative models [2,37,38] learn a series of transformations to map samples from one probability distribution x1 ∼ p, to samples from another x0 ∼ q. In the generative context, p is typically taken as the standard normal distribution, while q is the data distribution in a latent space [55]. Then, These models learn a time-dependent velocity field vθ(zt,t) that models the direction from a noisy sample towards the data manifold. The noisy sample zt is a linearly interpolated latent between the two data distributions, and is given as zt = (1 − t)x0 + tx1. The rectified flow-matching training loss for a conditional model conditioned on a text prompt c is given by

0,x1,y,c ∥vθ(zt,t,y,c) − (x1 − x0)∥22 . (1) Here, the velocity field is optionally conditioned on a context image y.

L = Et∼p(t),x

#### 3.2 LoRWeB

Our objective is to perform visual analogy completion [27], where the model infers a proposed edit from a given image pair and applies it to a new image. Formally, two reference images, a,a′ ∈ RD, are related by an unknown transformation T : RD → RD such that a′ = T (a). Given a new image b ∈ RD, the goal is to generate b′ ∈ RD such that b′ ≈ T (b).

Naive Solutions and Limitations. Using a pre-trained conditional generative model, such as FLUX.1-Kontext [5], existing solutions for this task fine-tune the

model using a single LoRA [57]. For example, given the input triplet {a,a′,b}, one can construct a composite 2 × 2 image y = [a,a′;b,b], as shown in the bottom-left part of Fig. 2, which serves as the conditioning input. The goal of the model is to output x0 = [a,a′;b,b′], such that the bottom-right quadrant was transformed from b to b′, by training over Eq. (1). While these approaches perform well when the transformation T is constrained to the analogy types seen in the training set, they struggle to generalize to new, diverse transformations. We propose this arises in part because the single adapter struggles to capture the wide range of analogical relationships, from different style transfers to objects insertion or layout modifications.

A more advanced solution could be to span the diverse set of possible analogies using multiple adapters. Recently, [12] demonstrated that LoRAs trained for model personalization can span a semantic basis. Inspired by this, we propose to learn such a basis for task LoRAs. A naïve adaptation of [12] to analogy tasks would require us to first optimize a single adapter for each of N analogy types seen during training, such that each LoRA module i excels at a different subset of visual edits. Once the specialized adapters are trained, they can be linearly combined to obtain an equivalent single “novel” adapter

A = eiAi, B = eiBi, (2)

where the coefficients ei are optimized for each analogy task separately through the use of Eq. (1) and the reference pair of images {a,a′}. The model using the combined LoRA is then used to transform b to b′.

However, this approach requires training a large number of models, and a testtime tuning phase for every new analogy. Indeed, [12] required 65,000 LoRAs to capture the constrained space of faces, and collecting a significant number of different analogy pairs is more difficult.

Our Approach. Instead, we propose LoRWeB (Low-Rank Weight Basis). Rather than training individual LoRAs and combining them only at inference time, we propose to simultaneously train a basis of LoRA adapters, jointly with an encoder that predicts linear-combination coefficients for each input analogy pair. Specifically, we maintain a set of N rank-r LoRAs, and associate each Ai,Bi pair where i ∈ {1,...,N} with a learnable key vector ki ∈ Rd, as depicted in the right part of Fig. 2. Next, we define an encoder network based on a frozen, pre-trained ViT [74], E, e.g. CLIP [51]. The encoder takes as input the conditioning image triplet, {a,a′,b}, passes them through the ViT, concatenates the results and projects them through a small learnable projection module P that outputs the results as a query vector q ∈ Rd:

q(a,a′,b) = P E(a),E(a′),E(b) . (3) Then, based on the conditioning query, we compute N coefficients with

ei(a,a′,b) = softmax

q(a,a′,b)KT √

, (4)

d i

where K ∈ Rd×N contains the key vectors {ki}Ni=1 in its columns. The final LoRA combination follows

∆W = BA = ei(BiAi), (5) and is marked as “Mixed LoRA” in Fig. 2.

Importantly, a single rank-r′ LoRA is a special case of LoRWeB. Specifically, if the learned mixing router collapses to a constant output for any input {a,a′b}, then ei will be constant for all inputs. This will result in the same static mixed N rank r LoRA combination. This mixture can yield a matrix of rank r′ = Nr, which is equivalent to finetuning a single rank r′ LoRA. However, as LoRWeB differently combines LoRAs for different inputs, this dramatically increases the expressive power of our representation. Indeed, a single rank r′ LoRA is just a single point in the space of all rank r′ LoRAs, which LoRWeB aims to span.

We use the same pre-trained encoder across different network layers, but train individual LoRWeB modules, including LoRAs, keys and projections for each targeted weight matrix W0 in the network. This enables capturing different semantic elements for each weight and layer in the model.

### 4 Experiments

Settings. We evaluate our approach using Flux.1-Kontext [5] as the pre-trained conditional flow model and CLIP [51] as the image encoder backbone. For our LoRAs Basis, we match the capacity of prior work [19], using N = 32 adapters, each of rank r = 4, with d = 128 as the learned key dimension. We project the CLIP-encoder’s output to Rd using a single fully-connected layer. To save on compute, during training we set the resolution to a maximum of 512 × 512 images, resizing on the long-edge of images. Additional implementation details are

- in App. A.1. Importantly, the use of a very lightweight CLIP encoder, combined with efficient Einsum matrix multiplications, ensure the inference overhead of LoRWeB is minimal. Indeed, LoRWeB inference takes 33.4 seconds, compared to 32.4 seconds for a single r = 128 LoRA. See further details on our inference efficiency in App. A.2. We compare LoRWeB to four recent baselines: A standard Flux.1-Kontext LoRA of similar parameter capacity (equivalent to LoRWeB with N = 1,r = 128), as well as three prior visual analogy methods based on Flux.1-Dev (RelationAdapter [19], VisualCloze [35] and EditTransfer [9]). We additionally compare to Diffusion Image Analogies (DIA) [61] and PairEdit [41],
- in App. B.1. DIA relies on inversion into CLIP space and expensive per-sample backpropagation through the entire diffusion process of Stable Diffusion 1.4 [55]. Nevertheless, for a fuller assessment, we compare their method with ours, including an additional variant we design to adapt DIA to Flux.1-Kontext. PairEdit also uses per-sample optimization, training three Flux LoRAs for each input triplet: A concept LoRA and a content LoRAs trained in parallel to describe the transformation from a to a′, and an inversion LoRA to reconstruct b. Once all LoRAs are trained they can be used to generate b′.

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

“Add a halo of flowers around this animal’s head”

“Convert this image into pop art style”

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

“Convert this image into mosaic art style” “Convert this image into pencil drawing art style”

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

“Convert this image into minimalist art style” “Add a halo of fire”

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

“Make this person look like a clown”

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

“Change the standing pose to a squatting pose.”

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

“Convert the subject to a robot with white…”

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

“Turn this photo into a surrealist floating sculpture”

“Raise the peron’s arms.”

- Fig. 3: LoRWeB visual analogy results. Using a LoRA Basis allows LoRWeB to generalize to a wide variety of new analogy tasks, from adding objects to transferring specific styles or makeup or copying pose changes. Please zoom in for more details.

Dataset. We train our model using the public Relation252k [19] set, which contains 16K analogy image pairs across 208 tasks. Since the train-set split of Relation252k is not fully publicly available, and only 10 unseen analogy tasks were released, we extend it with a custom validation set to evaluate visual analogies. Specifically, we focus on analogies that were not found in the training set, which we create in the following manner: First, we collect over 100 Unsplash1 photos covering diverse concepts from three categories: animals, persons, and general objects. Next, we create analogy pairs with a focus on two categories: transformations which are in-domain for the base text-to-image model, and transformations that are not. For in-domain transformations, we first use an LLM to summarize the training prompts for each task in the training-set of Relation252k, yielding 208 representative prompts. Next, we ask the LLM to generate novel prompts

1 https://unsplash.com/

LoRA

Relation

Edit

Ours

VisualCloze

| |
|---|

Aadpter

Transfer

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

“Make this look like it’s growing moss”

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

“Turn this person into a steampunk portrait”

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

“Add bioluminescent glowing elements”

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

“Give this creature a crown of crystals”

- Fig. 4: Comparisons with baseline methods on unseen tasks. We compare LoRWeB with four recent baselines: RelationAdapter [19], VisualCloze [35] and EditTransfer [9], as well as a standard Flux.1-Kontext LoRA of similar parameter capacity. Our approach generalizes across more diverse tasks, and better maintains the visual details of both the subject and the analogy.

that differ from the training set’s prompts and manually verify that they match the given concept categories. We filter prompts where Flux.1-Kontext fails to produce a meaningful edit, and further randomly select 15 prompts per concept category from the remainder. We generate three images per prompt, obtaining a total of 135 analogy pairs. For out-of-domain analogies, we collect 18 community LoRAs for Flux.1-Kontext from HuggingFace, which were trained to enable edits the base model failed with. We use these pre-trained LoRAs, and repeat the previous random sampling strategy to get 135 analogy pairs. Finally, we randomly select as the input images b two images from the matching concept category, with a similar aspect ratio to a and a′, and crop them to the exact size. Our resulting set contains 540 analogy triplets across 90 tasks and 3 concept categories. Including the unseen set of Relation252K, this gives 100 tasks across 840 analogy triplets. On all experiments, we first aggregate the results per analogy task, and then aggregate over all tasks. More details appear in App. A.3.

#### 4.1 Qualitative Evaluations

Figures 1 and 3 include results of analogy-based editing using LoRWeB. Notably, the model generalizes to new tasks covering style transfer, background

Ours r = 128 LoRA RelationAdapter EditTransfer VisualCloze

- 5

- 6

- 7

- 8

−→Preservation(VLM)

0.5

←−LPIPS

0.4

0.3

4.8 5.0 5.2 5.4 5.6 5.8 Edit Accuracy (VLM) −→

0.05 0.10 0.15 0.20 CLIP Directional −→

- Fig. 5: Quantitative comparisons. (left) Accuracy of the applied edit and preservation of b in b′ using Gemma-3 [63]. Top right is better. (right) CLIP directional similarity and LPIPS between b′ and b. Bottom-right is better. Our method pushes the Pareto front of edit accuracy-preservation, achieving higher edit accuracy while strongly preserving the input image.

0 50 100

Win Rate

User Study

Pairwise VLM

83.6 16.4 70.4 29.6

Ours Vs. Edit Transfer

0 50 100

Win Rate

70.0 30.0 58.5 41.5

Ours Vs. RelationAdapter

0 50 100

Win Rate

69.1 30.9 68.1 31.9

Ours Vs. VisualCloze

0 50 100

Win Rate

57.6 42.4 57.9 42.1

Ours Vs. LoRA r=128

- Fig. 6: Pairwise image comparisons. We compare LoRWeB to four baselines on overall edit quality preference via both a user study and using a VLM. LoRWeB produces edits that are favored by both. Error bars are the 68% Wilson score interval.

replacements, object insertion, object displacement and more. In Fig. 4 we show qualitative comparisons of LoRWeB against the baselines. Notably, existing approaches either struggle with maintaining the content of the original image, or fail on some of the tasks. Crucially, some baselines struggle to maintain subject identities, e.g. the cat in RelationAdapter or the woman in VisualCloze, or to accurately capture the details of the transformation, e.g. the precise crown in the cat example. LoRWeB shows greater adaptability and succeeds in a wider range of tasks. Additional results appear in App. B.3 and App. B.6.

#### 4.2 Quantitative Evaluations

Automated Evaluation Metrics. For quantitative evaluations, we follow prior work [9, 21, 60] and evaluate performance across standard metrics such as LPIPS [78] between the source and generated image, and CLIP directional similarity between both analogy pairs. In addition, we build on recent image editing work [30], which demonstrates that VLMs often better correlate with human preference than CLIP-based methods, and implement a VLM-based assessment protocol. Specifically, we conduct two VLM-based experiments: In the first, we provide Gemma-3 [63] with {a,a′,b,b′}, and ask the VLM to evaluate the quality of results on two criteria: consistency with the source image, and accuracy of the applied transformation relative to the reference transformation.

Table 1: Results for the ablation study of LoRWeB described in Sec. 4.3, for different hyperparameter and architecture choices.

CLIP Pairwise VLM (%) ↑ (VLM) (VLM) Dir. ↑ LoRA r=128 ET VC RA

Pres. ↑ Acc. ↑ LPIPS ↓

Model

LoRWeB (full, r = 4, N = 32) 7.87 5.94 0.31 0.21 57.9 70.4 68.1 58.5 + r = 16 8.13 4.92 0.20 0.11 51.8 63.9 62.4 49.6 + r = 16, N = 8 7.82 5.49 0.29 0.19 59.9 73.1 67.0 56.7 + N = 16 7.74 5.95 0.31 0.23 60.4 70.5 68.5 56.6 + Tanh activation 7.94 4.49 0.18 0.09 48.2 58.3 51.8 42.1 + 2 × 2 Enc. Input 7.90 5.75 0.28 0.20 61.9 73.3 68.2 53.9 + SigLip2 7.83 5.82 0.31 0.21 59.0 71.7 71.5 55.5 + SigLip2 & 2 × 2 Enc. Input 7.85 5.71 0.29 0.20 59.5 73.8 66.8 58.3

We name these metrics Preservation (VLM) and Edit Accuracy (VLM), respectively. As a second quality metric, we take a 2-alternative-forced-choice design (2AFC). We show Gemma-3 {a,a′,b}, the b′ result of our model, and the b′ result generated by a baseline, and ask it to select the image that best applies the analogy. We report this metric as Pairwise VLM. The prompts given to the VLM and further details appear in App. A.4. The results are shown in Fig. 5 and Fig. 6. When considering preservation and editing accuracy tradeoffs (Fig. 5), our model pushes the Pareto front, achieving high edit accuracy while better maintaining the input’s structure and appearance.

User Study. Beyond automated metrics, we also conduct a two-alternative forced choice user study. We show each user a reference pair (a,a′), an input image b, and two results (one from our model and one of a random baseline), in a randomized order, filtering out cases where no method succeeded in editing. Users are asked to select their preferred editing result. In total, we collected responses from 33 users covering 45 image pairs. The results (Fig. 6) align with the automated metrics, showing that users favor our approach over all baselines. In in App. A.4 we additionally evaluate the alignment between the scores of the VLM and the preferences of humans.

All in all, our experiments demonstrate that our approach can meaningfully improve on the existing state of the art, and better generalize to unseen tasks.

- 4.3 Ablations We next study the importance of different components of LoRWeB.

Capacity Effect. We compare LoRWeB across modified capacities in both basis sizes N and ranks r. Specifically, we compare our original variation ({N =32,r= 4}), with {N =8,r=16}, {N =16,r=4} and {N =32,r=16}. We use the same evaluation setup as in Sec. 4.2. Results are reported in Tab. 1. Reducing the basis size while maintaining the capacity (r=16,N =8) leads to a slight drop in

performance, as does simply reducing capacity (r = 4,N = 16). This highlights the importance of a large basis for generalization. Similarly, a naïve increase in rank can hamper editability, which we hypothesize to be a consequence of the data, leading to increased overfitting. We provide additional capacity results for LoRWeB and a single LoRA with a higher capacity in App. B.2. Here, again, naïve parameter addition does not strictly correlate with better performance.

Similarity Normalizing Function. The normalization function choice in Eq. (4) can also affect the learned basis. For example, the used softmax is bound to [0,1], hence it cannot result in negative coefficients for any LoRA. An alternative approach is to use Tanh, which is instead bound to [−1,1]. In practice, we find it to drastically underperform. We propose that this may be due to Tanh allowing the model to compose mixed LoRAs with much greater norms, possibly taking the model too far out of domain. Another alternative can be a differential activation function, as proposed by several recent work [46, 73]. However, we leave further investigation of activations to future work.

Layout of Encoder Input. In our approach, we elected to separately encode each of the conditioning analogy images using CLIP, and concatenate their representations. Our intuition is that CLIP requires resizing the image to 224×224, which can severely constrain the level of detail in each quadrant of the 2×2 grid that we provide Flux as a context. Moreover, concatenated features could allow the model to better understand which encoding represents each conditioning image (i.e. a, a′ and b), allowing it to better reason over the analogy. We verify this experimentally by comparing to a version that provides CLIP with just the context image (the 2 × 2 grid). As seen in Tab. 1, this diminishes performance, mainly decreasing the editing-accuracy metrics.

Alternative Image Encoders. Although our approach uses CLIP [51] as the encoder backbone, we validate our robustness to an alternative, common choice: SigLIP2 [66]. The results in Tab. 1 indicate that this change does not significantly alter our performances. We leave further tuning of encoders to future work.

Importance of Prompts and Reference Images. We follow existing baselines and use prompts to augment the model’s understanding. Since our goal is analogy based editing, and not simply text-based modification, we verify that indeed the output of our model depends not only on the text prompt, but on the analogy pair itself. Specifically, we conduct two complementing experiments. First, we examine how the same input image, b, reacts to different reference pairs {a,a′} under the same editing prompt. As can be seen in Fig. 7, the reference pair dictates the details of the analogy task, and particularly the visual details that are not captured by the prompts. For example, in the first row it copies the poster banner from the analogy image and adapts its specific style, and in the second row it matches the design and colors of the given crown. In comparison,

Ours VisualCloze RelationAadpter Edit Transfer

| |
|---|

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

“Turn this person into a vintage poster design”

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

“Give this creature a crown of crystals”

- Fig. 7: Effect of different reference analogy pairs. LoRWeB directly leverages the analogy pair to understand the details of the proposed task, applying an edit that is beyond just text-based editing based on the given prompt. For example, when the prompt is “Give this create a crown of crystals”, the analogy context passes information on the amount and color of the crystals.

we observe that some of the baselines are insensitive to the analogy pair, instead relying almost entirely on the prompt. Here, again, it can be seen some of the baselines opt to change the identity in the image (e.g. the dog of RelationAdapter and VisualCloze). For the second experiment, we examine how the same input triplet {a,a′,b} react to reducing the detail level in the prompt. Here, we use an LLM [20] to omit details from prompts in our evaluation set (instruction appears in App. A.4). When comparing the pairwise VLM preference rate for such detail-reduced prompts, our results are preferred over RelationAdapter, EditTransfer and VisualCloze 59.4%, 70.4% and 66.8%. Qualitative examples can be seen in Fig. 8. These results show that LoRWeB can better rely on the analogy images, when compared to prior work which relies heavily on the text prompt itself. As both experiments demonstrate, our approach has learned to perform analogy-based editing, and to a greater degree than the existing baselines. We experiment with unaligned prompts and input images in App. B.5.

Relation Adapter

Edit Transfer

LoRA Ours VisualCloze

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

“Add a halo of flowers around this animal's head"

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

“Add a headpiece."

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

“Add a swirling galaxy background"

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

“Change the background."

- Fig. 8: Effect of less detailed prompts for the same input. While we follow prior approaches in using textual prompts to understand the context of the task, the details of the applied transformation are achieved by understanding the reference pair of images a and a′. For example, when asked to “Add a halo of flowers around this animal’s head”, most approaches add a similar-looking flower crown to the dogs head. However, when the prompt changes to only “Add a headpiece”, only LoRWeB understands from a : a′ the headpiece is a flower crown that should be added on top of the dog’s head.

### 5 Discussion

We introduced LoRWeB, a modular framework for visual analogy completion that learns a basis of LoRA adapters and dynamically composes them using a shared encoder conditioned on the input analogy. Our approach addresses the limitations of single-adapter fine-tuning or multi-adapter optimization at inference time by enabling flexible, layer-specific adaptations to diverse and unseen transformations. Through extensive comparisons, we showed how LoRWeB outperforms and generalizes better than competing naive LoRA-based methods across various visual analogy tasks. However, this generalization is not with-

out limitations. For example, LoRWeB may still struggle with tasks that are significantly different from the training corpus (See App. B.7 for examples).

Additionally, a strong assumption in image analogies is the access to a reference image pair {a,a′} where the noticeable change depicts the transformation alone, while keeping the rest of the image details exactly the same (e.g. depicting the same person in different poses in the same environment). In practical circumstances, finding such reference pairs in the wild, which convey the exact needed transformation, can be difficult. Nevertheless, we show in App. B.4 that similarity isn’t strictly required, and LoRWeB can work on non-identical input pairs with textual guidance, within limit. Indeed, an interesting future work could explore better decomposing the components of a depicted transformation (e.g. the pose and the background), and allow for interactively choosing which components are used when applying the transformation to b.

While our focus here is on visual analogy completion, a similar LoRA-basis approach could be broadly applicable, possibly replacing LoRAs in other tasks where generalization is needed. We hope to explore this direction in future work.

### Acknowledgments

This research was partially supported by the Israel Science Foundation (grant no. 2318/22) and the Planning and Budgeting Committee of the Israeli Council for Higher Education. The authors are grateful to Matan Kleiner and Yoad Tewel for their insightful discussions and input throughout this work.

### References

- 1. Alaluf, Y., Garibi, D., Patashnik, O., Averbuch-Elor, H., Cohen-Or, D.: Crossimage attention for zero-shot appearance transfer. In: ACM SIGGRAPH 2024 conference papers. pp. 1–12 (2024) 4
- 2. Albergo, M.S., Vanden-Eijnden, E.: Building normalizing flows with stochastic interpolants. In: The Eleventh International Conference on Learning Representations

(2023) 5

- 3. Anthropic: Introducing claude 4. https://www.anthropic.com/news/claude-4

(2025) 21

- 4. Bar, A., Gandelsman, Y., Darrell, T., Globerson, A., Efros, A.: Visual prompting via image inpainting. Advances in Neural Information Processing Systems 35, 25005–25017 (2022) 2, 3
- 5. Black Forest Labs, Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., et al.: FLUX. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742 (2025) 2, 3, 4, 5, 7
- 6. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 18392–18402 (2023) 2
- 7. Cai, S., Chan, E.R., Zhang, Y., Guibas, L., Wu, J., Wetzstein, G.: Diffusion selfdistillation for zero-shot customized image generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 18434–18443 (2025) 4

- 8. Cao, M., Wang, X., Qi, Z., Shan, Y., Qie, X., Zheng, Y.: Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 22560– 22570 (2023) 3, 4
- 9. Chen, L., Mao, Q., Gu, Y., Shou, M.Z.: Edit transfer: Learning image editing via vision in-context relations. arXiv preprint arXiv:2503.13327 (2025) 3, 7, 9, 10
- 10. Cheng, M., Gong, J., Eldardiry, H.: Sci-LoRA: Mixture of scientific LoRAs for cross-domain lay paraphrasing. In: Findings of the Association for Computational Linguistics: ACL 2025. pp. 18524–18541 (2025) 4
- 11. Deutch, G., Gal, R., Garibi, D., Patashnik, O., Cohen-Or, D.: Turboedit: Textbased image editing using few-step diffusion models. In: SIGGRAPH Asia 2024 Conference Papers. pp. 1–12 (2024) 4
- 12. Dravid, A., Gandelsman, Y., Wang, K.C., Abdal, R., Wetzstein, G., Efros, A., Aberman, K.: Interpreting the weight space of customized diffusion models. Advances in Neural Information Processing Systems 37, 137334–137371 (2024) 2, 4, 6
- 13. Feng, W., Hao, C., Zhang, Y., Han, Y., Wang, H.: Mixture-of-LoRAs: An efficient multitask tuning method for large language models. In: Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). pp. 11371–11380 (2024) 4
- 14. Fischer, M., Li, Z., Nguyen-Phuoc, T., Bozic, A., Dong, Z., Marshall, C., Ritschel, T.: Nerf analogies: Example-based visual attribute transfer for nerfs. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4640–4650 (2024) 3
- 15. Fišer, J., Jamriška, O., Lukáč, M., Shechtman, E., Asente, P., Lu, J., S`ykora, D.: Stylit: illumination-guided example-based stylization of 3d renderings. ACM Transactions on Graphics (TOG) 35(4), 1–11 (2016) 3
- 16. Fu, S., Tamir, N., Sundaram, S., Chai, L., Zhang, R., Dekel, T., Isola, P.: Dreamsim: Learning new dimensions of human visual similarity using synthetic data. In: Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., Levine, S. (eds.) Advances in Neural Information Processing Systems. vol. 36, pp. 50742–50768. Curran Associates, Inc. (2023) 24
- 17. Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A.H., Chechik, G., Cohen-or, D.: An image is worth one word: Personalizing text-to-image generation using textual inversion. In: The Eleventh International Conference on Learning Representations (2023) 4
- 18. Gal, R., Lichter, O., Richardson, E., Patashnik, O., Bermano, A.H., Chechik, G., Cohen-Or, D.: LCM-lookahead for encoder-based text-to-image personalization. In: European Conference on Computer Vision. pp. 322–340. Springer (2024) 4
- 19. Gong, Y., Song, Y., Li, Y., Li, C., Zhang, Y.: Relationadapter: Learning and transferring visual relation with diffusion transformers. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025) 2, 3, 7, 8, 9, 21, 24, 34
- 20. Google: Introducing gemini 3. https://blog.google/products-and-platforms/ products/gemini/gemini-3/ (2025) 13, 24
- 21. Gu, Z., Yang, S., Liao, J., Huo, J., Gao, Y.: Analogist: Out-of-the-box visual in-context learning with image diffusion model. ACM Transactions on Graphics (TOG) 43(4), 1–15 (2024) 10
- 22. Haas, R., Huberman-Spiegelglas, I., Mulayoff, R., Graßhof, S., Brandt, S.S., Michaeli, T.: Discovering interpretable directions in the semantic latent space of

- diffusion models. In: 2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG). pp. 1–9. IEEE (2024) 4
- 23. He, M., Liao, J., Chen, D., Yuan, L., Sander, P.V.: Progressive color transfer with dense semantic correspondences. ACM Transactions on Graphics (TOG) 38(2), 1–18 (2019) 3
- 24. Hertz, A., Aberman, K., Cohen-Or, D.: Delta denoising score. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 2328–2337

(2023) 4

- 25. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-Or, D.: Prompt-to-prompt image editing with cross-attention control. In: The Eleventh International Conference on Learning Representations (2023) 3
- 26. Hertz, A., Voynov, A., Fruchter, S., Cohen-Or, D.: Style aligned image generation via shared attention. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4775–4785 (2024) 4
- 27. Hertzmann, A., Jacobs, C.E., Oliver, N., Curless, B., Salesin, D.H.: Image analogies. In: Proceedings of the 28th Annual Conference on Computer Graphics and Interactive Techniques. p. 327–340. SIGGRAPH ’01, Association for Computing Machinery, New York, NY, USA (2001). https://doi.org/10.1145/383259.383295, https://doi.org/10.1145/383259.383295 2, 3, 5
- 28. Hu, E.J., yelong shen, Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-rank adaptation of large language models. In: International Conference on Learning Representations (2022) 3, 4
- 29. Huang, C., Liu, Q., Lin, B.Y., Pang, T., Du, C., Lin, M.: LoraHub: Efficient crosstask generalization via dynamic loRA composition. In: First Conference on Language Modeling (2024) 4
- 30. Huang, Y., Huang, J., Liu, Y., Yan, M., Lv, J., Liu, J., Xiong, W., Zhang, H., Cao, L., Chen, S.: Diffusion model-based image editing: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025) 10, 24
- 31. Huberman-Spiegelglas, I., Kulikov, V., Michaeli, T.: An edit friendly DDPM noise space: Inversion and manipulations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12469–12478 (2024) 4
- 32. Ishikawa, R., Fujii, R., Saito, H., Hachiuma, R.: Human preference-aligned concept customization benchmark via decomposed evaluation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 7002–7011 (2025) 24
- 33. Kawar, B., Zada, S., Lang, O., Tov, O., Chang, H., Dekel, T., Mosseri, I., Irani, M.: Imagic: Text-based real image editing with diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6007–6017

(2023) 35

- 34. Kulikov, V., Kleiner, M., Huberman-Spiegelglas, I., Michaeli, T.: Flowedit: Inversion-free text-based editing using pre-trained flow models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19721–19730

(2025) 4

- 35. Li, Z.Y., Du, R., Yan, J., Zhuo, L., Li, Z., Gao, P., Ma, Z., Cheng, M.M.: Visualcloze: A universal image generation framework via visual in-context learning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 18969–18979 (October 2025) 7, 9
- 36. Liao, J., Yao, Y., Yuan, L., Hua, G., Kang, S.B.: Visual attribute transfer through deep image analogy. ACM Transactions on Graphics (TOG) 36(4), 1–15 (2017) 3

- 37. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. In: The Eleventh International Conference on Learning Representations (2023) 5
- 38. Liu, X., Gong, C., qiang liu: Flow straight and fast: Learning to generate and transfer data with rectified flow. In: The Eleventh International Conference on Learning Representations (2023) 5
- 39. Liu, Y., Chen, X., Ma, X., Wang, X., Zhou, J., Qiao, Y., Dong, C.: Unifying image processing as visual prompting question answering. In: International Conference on Machine Learning. pp. 30873–30891. PMLR (2024) 2
- 40. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (2019) 21
- 41. Lu, H., Chen, J., Yang, Z., Gnanha, A.T., Wang, F.L., Qing, L., Mao, X.: Pairedit: Learning semantic variations for exemplar-based image editing. In: The Thirtyninth Annual Conference on Neural Information Processing Systems (2025) 2, 7, 27
- 42. Manor, H., Michaeli, T.: Zero-shot unsupervised and text-based audio editing using DDPM inversion. In: Salakhutdinov, R., Kolter, Z., Heller, K., Weller, A., Oliver, N., Scarlett, J., Berkenkamp, F. (eds.) Proceedings of the 41st International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 235, pp. 34603–34629. PMLR (21–27 Jul 2024) 4
- 43. Mao, F., Hao, A., Chen, J., Liu, D., Feng, X., Zhu, J., Wu, M., Chen, C., Wu, J., Chu, X.: Omni-effects: Unified and spatially-controllable visual effects generation. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 7927–7935 (2026) 4
- 44. Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.Y., Ermon, S.: SDEdit: Guided image synthesis and editing with stochastic differential equations. In: International Conference on Learning Representations (2022) 3
- 45. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99–106 (2021) 3
- 46. Misrahi, A., Chirkova, N., Louis, M., Nikoulina, V.: DiffLoRA: Differential lowrank adapters for large language models. arXiv preprint arXiv:2507.23588 (2025) 12
- 47. OpenAI: Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024) 21
- 48. Ortiz, J.J.G., Guttag, J., Dalca, A.V.: Magnitude invariant parametrizations improve hypernetwork learning. In: The Twelfth International Conference on Learning Representations (2024) 2
- 49. Parmar, G., Kumar Singh, K., Zhang, R., Li, Y., Lu, J., Zhu, J.Y.: Zero-shot imageto-image translation. In: ACM SIGGRAPH 2023 conference proceedings. pp. 1–11

(2023) 3

- 50. Peng, Y., Cui, Y., Tang, H., Qi, Z., Dong, R., Bai, J., Han, C., Ge, Z., Zhang, X., Xia, S.T.: Dreambench++: A human-aligned benchmark for personalized image generation. In: The Thirteenth International Conference on Learning Representations (2025) 24
- 51. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Meila, M., Zhang, T. (eds.) Proceedings of the 38th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 139, pp. 8748–8763. PMLR (18–24 Jul

2021) 2, 5, 6, 7, 12, 27

- 52. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research 21(140), 1–67 (2020) 27
- 53. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125

(2022) 3

- 54. Reed, S.E., Zhang, Y., Zhang, Y., Lee, H.: Deep visual analogy-making. Advances in neural information processing systems 28 (2015) 2, 3
- 55. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 1824, 2022. pp. 10674–10685. IEEE (2022). https://doi.org/10.1109/CVPR52688. 2022.01042, https://doi.org/10.1109/CVPR52688.2022.01042 3, 5, 7, 27
- 56. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 22500–22510 (2023) 4
- 57. Ryu, S.: Cloneofsimo/lora: Low-rank adaptation for fast text-to-image diffusion fine-tuning (2023) 6
- 58. Shah, V., Ruiz, N., Cole, F., Lu, E., Lazebnik, S., Li, Y., Jampani, V.: ZipLoRA: Any subject in any style by effectively merging LoRAs. In: European Conference on Computer Vision. pp. 422–438. Springer (2024) 4
- 59. Sheynin, S., Polyak, A., Singer, U., Kirstain, Y., Zohar, A., Ashual, O., Parikh, D., Taigman, Y.: Emu edit: Precise image editing via recognition and generation tasks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8871–8879 (2024) 2
- 60. Song, X., Cui, J., Zhang, H., Shi, J., Chen, J., Zhang, C., Jiang, Y.G.: LoRA of change: Learning to generate LoRA for the editing instruction from a single before-after image pair. arXiv preprint arXiv:2411.19156 (2024) 2, 10
- 61. Šubrtová, A., Lukáč, M., Čech, J., Futschik, D., Shechtman, E., S`ykora, D.: Diffusion image analogies. In: ACM SIGGRAPH 2023 Conference Proceedings. pp. 1–10 (2023) 3, 7, 27
- 62. Tan, Z., Liu, S., Yang, X., Xue, Q., Wang, X.: Ominicontrol: Minimal and universal control for diffusion transformer. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (2025) 4
- 63. Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Ramé, A., Rivière, M., et al.: Gemma 3 technical report. arXiv preprint arXiv:2503.19786 (2025) 10, 22
- 64. Tewel, Y., Gal, R., Samuel, D., Atzmon, Y., Wolf, L., Chechik, G.: Add-it: Trainingfree object insertion in images with pretrained diffusion models. In: The Thirteenth International Conference on Learning Representations (2025) 4
- 65. Tewel, Y., Kaduri, O., Gal, R., Kasten, Y., Wolf, L., Chechik, G., Atzmon, Y.: Training-free consistent text-to-image generation. ACM Transactions on Graphics (TOG) 43(4), 1–18 (2024) 4
- 66. Tschannen, M., Gritsenko, A., Wang, X., Naeem, M.F., Alabdulmohsin, I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., et al.: SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786 (2025) 12
- 67. Tumanyan, N., Geyer, M., Bagon, S., Dekel, T.: Plug-and-play diffusion features for text-driven image-to-image translation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1921–1930 (2023) 3

- 68. Wang, X., Wang, W., Cao, Y., Shen, C., Huang, T.: Images speak in images: A generalist painter for in-context visual learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6830–6839 (2023) 2, 3
- 69. Wang, Z., Jiang, Y., Lu, Y., He, P., Chen, W., Wang, Z., Zhou, M., et al.: Incontext learning unlocked for diffusion models. Advances in Neural Information Processing Systems 36, 8542–8562 (2023) 2, 3
- 70. Wu, X., Huang, S., Wei, F.: Mixture of loRA experts. In: The Twelfth International Conference on Learning Representations (2024) 4
- 71. Xiao, S., Wang, Y., Zhou, J., Yuan, H., Xing, X., Yan, R., Li, C., Wang, S., Huang, T., Liu, Z.: Omnigen: Unified image generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 13294–13304 (2025) 2
- 72. Yang, Y., Peng, H., Shen, Y., Yang, Y., Hu, H., Qiu, L., Koike, H., et al.: Imagebrush: Learning visual in-context instructions for exemplar-based image manipulation. Advances in Neural Information Processing Systems 36, 48723–48743 (2023) 2, 3
- 73. Ye, T., Dong, L., Xia, Y., Sun, Y., Zhu, Y., Huang, G., Wei, F.: Differential transformer. In: The Thirteenth International Conference on Learning Representations

(2025) 12

- 74. Zhai, X., Kolesnikov, A., Houlsby, N., Beyer, L.: Scaling vision transformers. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12104–12113 (2022) 6
- 75. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pre-training. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 11975–11986 (2023) 2
- 76. Zhang, J.C., Xiong, Y.J.: Subject or style: Adaptive and training-free mixture of LoRAs. arXiv preprint arXiv:2508.02165 (2025) 4
- 77. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 3836–3847 (2023) 4
- 78. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 586–595 (2018) 10
- 79. Zhang, Z., Xie, J., Lu, Y., Yang, Z., Yang, Y.: Enabling instructional image editing with in-context generation in large scale diffusion transformer. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025) 2

### A Experimental Details

#### A.1 Implementation Details

In all our experiments, we train for 10K steps on 1 H100 GPU, setting 8-bit AdamW [40] as the optimizer with a learning rate of 10−3, β1 = 0.9,β2 = 0.99,

- a weight decay value of 0.05, and bfloat16 mixed-precision training. We enable gradient checkpointing, and use a batch size of 6 for all experiments, except for when r = 16,N = 32 where the batch size is set to 4. As for the encoders, the CLIP checkpoint we use is openai/clip-vit-large-patch14. For the SigLIP2 version in the ablations, we test google/siglip2-base-patch16-224. Both output a vector in R768.

#### A.2 Efficiency Analysis

Compared to using a single, standard LoRA module, LoRWeB increases the computational load during inference due to using multiple LoRA modules. However, the inference overhead of our approach is minimal. At inference time, LoRWeB requires passing the input triplet {a,a′b} through the CLIP encoder once and later combines the LoRA basis for each layer. As CLIP is very lightweight compared to Flux.1-Kontext, and LoRA mixing is computed by an efficient Einsum matrix multiplication, this incurs negligible runtime overhead. Averaged over 100 images on an A100 GPU, LoRWeB inference takes 33.4±0.4 seconds, compared to 32.4 ± 0.3 seconds for a single r = 128 LoRA. This means only a +3.1% increase in runtime. Furthermore, our current implementation re-computes the LoRA mixing every timestep. However, as the mixed LoRA of LoRWeB is fixed for all timesteps, a more efficient implementation of caching the mixing result from the first timestep can further reduce cost.

#### A.3 Custom Inference Dataset

All images gathered from Unsplash for the inference dataset extension are free to use under the Unsplash license2. To simulate in-domain prompts, we use GPT-4o [47] and Claude Sonnet 4 [3] to summarize the training prompts of Relation252k [19] as described in Sec. 4, and generate novel prompts. The 15 randomly selected prompts per concept category (animals, objects, and persons) appear in Tab. S1. The 18 pre-trained LoRA adapters are sourced from HuggingFace3, and cover a range of transformation types such as style transfer, object modification, and artistic reinterpretation. Specifically, we use the following community LoRAs, with their provided trigger prompt:

- – day-dream/MechAnything-Kontext-Dev-Lora
- – drbaph/Fluffy-kontext-LoRA

- 2 https://unsplash.com/license
- 3 https://https://huggingface.co/

- – fal/3D-Game-Assets-Kontext-Dev-LoRA
- – fal/Cubist-Art-Kontext-Dev-LoRA
- – fal/Gouache-Art-Kontext-Dev-LoRA
- – fal/Minimalist-Art-Kontext-Dev-LoRA
- – fal/Mosaic-Art-Kontext-Dev-LoRA
- – fal/Pencil-Drawing-Kontext-Dev-LoRA
- – fal/Plushie-Kontext-Dev-LoRA
- – fal/Pop-Art-Kontext-Dev-LoRA
- – fal/Watercolor-Art-Kontext-Dev-LoRA
- – gokaygokay/Bronze-Sculpture-Kontext-Dev-LoRA
- – gokaygokay/Low-Poly-Kontext-Dev-LoRA
- – gokaygokay/Marble-Sculpture-Kontext-Dev-LoRA
- – gokaygokay/Oil-Paint-Kontext-Dev-LoRA
- – Kontext-Style/Clay_Toy_lora
- – Kontext-Style/Ghibli_lora
- – Kontext-Style/Paper_Cutting_lora .

To match between a,a′ and b images of different sizes, we only choose b images with an original aspect ratio distanced 0.15 from the aspect ratio of a and a′, and crop b to a’s aspect ratio. The images are resized to the same size with a maximum long edge of 512 before entering Flux.1-Kontext.

- A.4 VLM Based Evaluation Part of our automated evaluation metrics include the use of Gemma-3 [63] as

- a VLM to evaluate our results. We use two VLM-based experiments. In the first, we ask the VLM to evaluate our results on two criteria: consistency with the source image b and accuracy of the applied transformation relative to the reference transformation described by {a,a′}. For this, we provide Gemma-3 with {a,a′,b,b′}, and the following prompt:

- You are given 4 images: A (original image), A’ (edited version of A), B (another original image), and B’ ( an output of an editing method). A, A’ and B are reference images that are given to some editing method in order to generate B’. The method tries to

infer the transformation that A underwent to produce A’, and then tries (maybe unsuccessfully) to apply the exact same transformation to B - in order to generate B’. Your task is to evaluate the resulting B’: Was the same transformation applied well?

Specifically , assess B’ under two metrics , editing accuracy , and consistency with the original image B , 1-10 integers only:

- 1) editing accuracy: Evaluate how closely B’ applies the transformation seen from A to A’. Are there missing elements , are there redundant elements? Quantify the precision of the editing.
- 2) consistency: Asses how well the edited image B’ maintains the context of the original image B. Does

it preserve the identity , objects , and layout in B that did not require a change , based on the

infered transformation from A to A’?

Consider in your evaluations other visual factors such as the localization of the edits , existence of redundant elements , style/strength/magnitude/colors

of changes.

First , describe in detail what the transformation from A to A’. Then describe what elements of it are present or missing in B’, detailing precisely what ’ s wrong regarding each metric.

Then , return a strict JSON with this scheme: \{" metrics ":\{" accuracy":<1-10>,"consistency":<1-10>\}, "explanation ":"the reasoning you described above "\}.

The JSON is parsed automatically, and we report the numeric values as Preservation (VLM) and Edit Accuracy (VLM).

In the second quality metric, we take a 2-alternative-forced-choice design (2AFC). We show Gemma-3 five images: {a,a′,b}, the b′ result of our model, and the b′ result generated by one baseline, and ask it to select the image that better applies the analogy via the following prompt:

- You are given 5 images: A (original image), A’ (edited version of A), B (another original image), and 2 B’

images (outputs of 2 editing methods). A, A’ and B are reference images that are given to some

editing method in order to generate B’. The methods

try to infer the transformation that A underwent to produce A’, and then tries (maybe unsuccessfully ) to apply the exact same transformation to B - in order to generate B’.

Your task is to evaluate the resulting B’s: In which of

the two methods was the same transformation applied well?

Specifically , assess B’ under two metrics , editing accuracy , and consistency with the original image B , 1-10 integers only:

- 1) editing accuracy: Evaluate how closely B’ applies the transformation seen from A to A’. Are there missing elements , are there redundant elements? Quantify the precision of the editing.

- 2) consistency: Asses how well the edited image B’ maintains the context of the original image B. Does

it preserve the identity , objects , and layout in B that did not require a change , based on the

inferred transformation from A to A’?

Consider in your evaluations other visual factors such as the localization of the edits , existence of redundant elements , style/strength/magnitude/colors

of changes.

First , describe in detail what the transformation from A to A’. Then describe what elements of it are present or missing in B’1 and B’2, detailing precisely what ’s wrong regarding each metric.

Then , return a strict JSON with this scheme: \{"better ":<1 or 2>,"explanation ":"the reasoning you described above"\}

We report the winrates parsed from the JSON outputs as pairwise VLM. Additionally, we used an LLM [20] to modify the prompts for the varying

prompts ablation reported in Sec. 4.3. To generate these prompts, we used the following prompt:

I am going to give you a file which contains 192

editing prompts. Generate an output file , like that: For each prompt in the file , please edit it slightly , so the prompt will have less information of the specific editing task , but will still convey largely a similar editing task.

Please generate only 1 option per prompt in a file. The format of each output file should be a JSON like: [{" original": original_prompt , "edit": edited_prompt

},...].

Also , do not change the order of appearance , so that later when I check I can see which prompt turned into what by just comparing line numbers.

Alignment With Humans. While VLMs have been used in the past as a metric aligned with human preference [30,32,50], even in the context of visual analogies [19], we further validate their use in our task. Specifically, we test the alignment between the scores of the VLM and the preferences of humans from our user study described in Sec. 4. Following [16], We calculate the percentage of times the votes of each user agreed with the votes of the VLM and average over all users. We find this average user-VLM agreement to be 66.7%. As a baseline, we also compute the average agreement between different users. Namely, we compute the percentage of times the votes of each pair of users agreed and

##### average over all user pairs. We find that this average user-user agreement is 74.2%. This means that our VLM based approach achieves a 89.9% evaluation consistency with the evaluation of humans. We also note that the mean standard deviation of user votes is 0.3423, which is similar to the empirical standard deviation of the VLM’s predictions from the users mean, which is given by 0.4649.

Table S1: List of prompts generated for the inference sets Category Prompt

Animals Add a collar with a bell Animals Add a mountainous background Animals Give this animal clockwork mechanical parts Animals Add a flowing mane Animals Add camouflage patterns Animals Give this animal ethereal ghost-like transparency Animals Add a flowing river background Animals Add metallic golden fur highlights Animals Give this animal translucent fairy wings Animals Add a halo of fire Animals Give this animal a fantastical set of armor Animals Give this creature a crown of crystals Animals Add a halo of flowers around this animal’s head Animals Give this animal bioluminescent markings Animals Make this creature look sleepy Objects Add a swirling galaxy background Objects Render the object entirely as if it’s made from hand-knitted or hand-

crocheted yarn Objects Add bioluminescent glowing elements Objects Turn this into a candy or confectionery version Objects Add flowing fabric or silk textures Objects Turn this into a steampunk mechanical design Objects Add intricate filigree patterns Objects Turn this into a vintage advertisement poster Objects Give this object a coat of rust Objects Turn this photo into a cross-section diagram Objects Make this look ancient and archaeological Objects Turn this photo into a surrealist floating sculpture Objects Make this look like it’s growing moss Objects Turn this photo into an architectural rendering Objects Make this look like it’s made of clouds Persons Add a cape or cloak Persons Add elaborate hairstyling with ornaments Persons Make this person look heroic Persons Add a serene, forested background Persons Add golden hour lighting to this portrait Persons Make this person look like a clown Persons Add a swirling vortex background Persons Add natural outdoor lighting to this portrait Persons Make this person look like royalty Persons Add body paint or decorative patterns Persons Add temporary tattoos Persons Turn this person into a holographic projection Persons Add elaborate eye makeup Persons Make this person look ethereal Persons Turn this person into a steampunk portrait

### B Additional Results

#### B.1 Comparison to Diffusion Image Analogies (DIA) and PairEdit

DIA [61] is a per-sample optimization approach for image analogies, originally introduced for Stable Diffusion 1.4 [55]. Specifically, this method uses backpropegation through the entire diffusion process to invert the reference and input images inot the models latent space, and its CLIP [51] space. This takes 30GB for SD1.4 (and > 10[mins] per image on an H100 GPU). This is challenging with newer, larger models. Additionally, in contrast to our aproach, the reliance of DIA on inversion into CLIP space means it is not as adaptable to newer models (e.g., CLIP is not used in Flux.2). Nevertheless, we compare to DIA to allow a more complete evaluation.

DIA is introduced as an interactive approach and presents 20 default hyperparameter options for the user to choose from. We therefore manually check all 20 options over 540 test images and perform the final evaluation with the best performing hyperparameter configuration (σ = 12, λ = 0.63). Finally, we also adapt DIA to Flux.1-Kontext. First, we replace their b inversion by setting

- b as the context image. Next, as backpropagation through the entire flow process is computationally infeasible, we replace the CLIP embedding inversion by setting the CLIP (E) embeddings as E(b)+E(a′)−E(a). Finally, Flux.1-Kontext includes an additional text encoder, T5 [52]. Therefore, for a fairer comparison, we provide the editing prompt to the T5 encoder. We term this approach DIA-Kontext.

PairEdit [41] is also a per-sample optimization approach for examplar-based image-editing, designed for Flux.1-Dev. Specifically, this method trains three distinct LoRAs for each input triplet {a,a′b}. Given a and a′, the method first jointly trains a content LoRA, which reconstructs the source image using the standard flow loss, and a semantic LoRA, which aims to capture the semantic transformation between the two images by optimizing a semantic loss. The join training encourages the semantic LoRA to disentangle the semantic differences from the image content. Next, to apply this transformation to a new image

- b, PairEdit requires training an inversion LoRA of b. Finally, at inference the inversion LoRA and the semantic LoRA are aplpied together to produce b′. This optimization approach takes dozens of minutes and intensive compute to edit a single new image (more than 25 minutes on an A100 GPU). In contrast, LoRWeB requires no test time training.

Qualitative results can be seen in Figs. S1 and S2, and quantitative evluations appear in Tab. S2. As can be seen, LoRWeB outperforms all approaches. Specifically, while this performance boost may seem trivial over DIA due to their use of a weaker pre-trained model (SD1.4), LoRWeB also outperforms DIA-Kontext. DIA-Kontext preserves the identity of the object, yet struggles to accurately perform the transformation. While PairEdit can sometimes score lower on LPIPS, its overall edit-balance of edit-adherence and content preservation is lacking, yielding sub-par results across all metrics.

Ours DIA

DIA-Kontext

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

“Turn this image into the Clay_Toy style”

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

“Convert this image into pop art style”

- Fig. S1: Qualitative comparison with DIA & DIA-Kontext. LoRWeB outperforms DIA and our Flux-adapter variant DIA-Kontext by a large margin. Notably, DIA-Kontext does preserves the identity of both the cat and the woman, yet struggles to accurately perform the transformation.

Ours PairEdit(0.6) PairEdit (1)

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

“Convert this image into minimalist art style”

PairEdit(0.8)

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

“Convert this image into watercolor art style”

[Figure 192]

[Figure 193]

[Figure 194]

- Fig. S2: Qualitative comparison with PairEdit. LoRWeB outperforms PairEdit, across multiple PairEdit strengths variations.

Table S2: Quantitave comparison with DIA, DIA-Kontext, and PairEdit

CLIP Our Pairwise ↑ (VLM) (VLM) Dir. ↑ VLM vs. (%)

Pres. ↑ Acc. ↑ LPIPS ↓

Method

Ours 7.87 5.94 0.31 0.21 –

DIA 3.56 3.44 0.59 0.12 86.5% DIA-Kontext 4.78 3.56 0.63 0.17 68.6% PairEdit s = 0.6 6.68 4.55 0.24 0.11 75.6% PairEdit s = 0.8 4.44 3.60 0.44 0.16 86.3%

#### B.2 Additional Quantitative Results

We conduct two additional experiments with LoRWeB of a larger capacity (r = 4,N = 64), as well as a single LoRA with higher capacity, of r = 256. The results, along with a detailed table of the numerical values in Fig. 5, appear in Tab. S3. As evident, naïve parameter addition does not strictly correlate with better performance, and can cause the methods to overfit.

Table S3: additional results for the ablation study of LoRWeB described in Sec. 4.3, for different hyperparameter and architecture choices.

CLIP Pairwise VLM (%) ↑ (VLM) (VLM) Dir. ↑ LoRA r = 128 ET VC RA

Pres. ↑ Acc. ↑ LPIPS ↓

Model

LoRWeB (full, r = 4, N = 32) 7.87 5.94 0.31 0.21 57.9 70.4 68.1 58.5 LoRWeB on (r = 4, N = 64) 7.80 5.48 0.27 0.19 56.5 67.7 66.3 52.6 LoRA r = 128 7.99 5.70 0.27 0.20 N/A N/A N/A N/A LoRA r = 256 7.88 5.48 0.26 0.18 N/A N/A N/A N/A VisualCloze 5.24 4.93 0.53 0.21 N/A N/A N/A N/A RelationAdapter 7.01 5.93 0.43 0.22 N/A N/A N/A N/A Edit-Transfer 7.38 4.79 0.31 0.04 N/A N/A N/A N/A

#### B.3 Additional Qualitative Results

We provide additional qualitative results of our method in Fig. S3, as well as more comparisons of our method to the 4 baselines from Sec. 4 in Fig. S4.

| |
|---|

| |
|---|

| |
|---|

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

“Convert this image into watercolor art style”

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

“Turn this into a steampunk mechanical design”

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

“Convert this image into bronze version”

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

“Convert this image into watercolor art style”

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

“Add a serene, forested background”

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

“Turn this image into the Clay_Toy style”

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

“Turn this person into a steampunk portrait”

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

“Convert this image into watercolor art style”

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

“Convert this image into watercolor art style”

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

“Convert this image into pencil drawing art style”

“Turn this into image into the Ghibli style”

###### Fig. S3: LoRWeB visual analoy results. The use of a LoRA Basis allows LoRBA to generalize to a wide varity of new analogy tasks, from changing given images to certain styles such as clay toys or bronze sculptures, changing the backgrounds, or changing the cloths of the person. Please zoom in for more details.

LoRA

Relation

Edit

Ours

VisualCloze

| |
|---|

Aadpter

Transfer

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

“Give this animal a fantastical set of armor”

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

“Turn this photo into a surrealist floating sculpture”

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

“Turn this image into the Clay_Toy style”

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

“Convert this image into minimalist art style”

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

“Convert this image into watercolor art style”

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

“Make this look like it’s made of clouds”

###### Fig. S4: Comparisons with baseline methods on unseen tasks. Our approach generalizes more across diverse tasks, and better maintains the visual details of both the subject and the analogy.

Ours

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

“Give this animal a fantastical set of armor”

- Fig. S5: The effect of using non-identical a : a′. Here, we use the top-right example from Fig. 1. While exact alignment of a and a′ helps, and LoRWeB was trained on such samples, empirically we find LoRWeB can work on images depicting more than the intended transformation, with text guidance, to some limit.

#### B.4 Sensitivity to Non-Identical Input Pairs

As mentioned in Sec. 5, in image analogies a commonplace assumption is the access to a reference image pair {a,a′} where the noticeable change depicts the transformation alone, while keeping the rest of the image details exactly the same. However, in practical circumstances, finding such reference pair of input images, which depict only a single modification, is hard to come by in the wild. Indeed, such pairs are more common when shooting a series of images by design, e.g. in a professional photo-shoot. Still, we empirically find that LoRWeB can work on non-identical input pairs with textual guidance, within limit. For example, in Fig. S5 we repeat the example from Fig. 1 with different a images. As can be seen, changing the position of the dog in a did not negatively effect the resulting b, as the edit is accurate and the identity of the dog in b is preserved. However, changing the a to a different dog distorts the output b, leading to loss of identity preservation, as well as the image background.

| |
|---|

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

“Add camouflage patterns”

- Fig. S6: The effect of using unaligned textual prompt c and input images pair a : a′. Using text prompts that describe a different edit than that seen in the analogy images pair can produce a combined-editing effect, displaying elements from both types of input.

#### B.5 Effect of Misalignment in Text and Image Inputs

As discussed in Sec. 4.3, while we follow existing baselines in using prompts, this necessitates validating the balance of prompt and input images effect on the output. In Sec. 4.3 we conduct two experiments to provide insight on the effects of different prompts on the output, and the effect of different image inputs on the output. Here, we further investigate the effect of the two inputs by examining how misalignment affects the output. Specifically, we provide to LoRWeB a conditioning prompt c which does not describe the analogy seen in the set a reference image pairs {a,a′}. Indeed, LoRWeB was not trained on such inputs, and was trained on aligned data, where the prompts roughly describe the analogy seen in the input pair. Empirically, we find this can either cause LoRWeB to ignore one of the inputs, or create an interesting combined-editing effect. As seen in Fig. S6, this combined-editing effect results an output image which displays features seen in the analogy pair as well as features understood from the textual prompt. This further strengthens that LoRWeB has learned to perform textually-guided analogy-based editing, using both the input images and the textual prompt.

#### B.6 Sensitivity to Non-Flux Generated Images

Our evaluation set is composed of the test set of Relation252K [19] as well as our custom set of edited Unsplash images. The reference image pairs in Relation252K were edited by using MidJourney, or curated from existing benchmark datasets. Our custom set was generated by instructing Flux with editing prompts. In Fig. S7, we further verify the sensitivity of LoRWeB to reference image pairs not generated by Flux. Specifically, we test over images from TEdBench, which contains images edited by Imagic [33]. As can be seen, the use of non-Flux generated images as the reference pair does not hampers the performance of LoRWeB.

| |
|---|

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

“Make the animal jump.”

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

“Make the animal jump.”

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

“Add a hat to the animal.”

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

“Add a hat to the animal.”

- Fig. S7: Sensitivity to reference image pairs not generated by Flux. We explicitly test LoRWeB on cases where the input reference pair of images, {a, a′}, was not generated by Flux. Specifically, here the images were drawn from TEdBench, where a′ was generated by Imagic [33]. The use of image not generated by Flux does not hamper the performance of LoRWeB, verifying it is not biased towards Flux-generated images.

#### B.7 Failure Cases

LoRWeB better generalizes than competing approaches across various visual analogy tasks. However, this generalization is not without limitations. Here we present some failure cases of LoRWeB. Specifically, LoRWeB may still struggle with tasks that are significantly different from the training corpus. As can be seen in the first row of Fig. S8, turning b into a cubist art style proved difficult for all approaches. Here, a′ was created by using a community LoRA, such that the base model itself also had difficulties in creating this specific style. Additionally, as seen in the second row of Fig. S8, LoRWeB might only partly understand the analogy if it includes several components. Here, the analogical difference includes a black collar and a large bell, yet LoRWeB only added the bell. Finally, if the difference between the reference pair a and a′ is small, such as closing the eyes of the animal (third row of Fig. S8), LoRWeB might struggle reflecting the changes. This might be due to the use of a CLIP models which downsamples the images to a size of 224 × 224, which might make it less sensitive to small details.

LoRA

Relation

Edit

Ours

VisualCloze

| |
|---|

Aadpter

Transfer

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

“Convert this image into cubist art style”

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

“Add a collar with a bell”

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

“Make this creature look sleepy”

- Fig. S8: Examples of failure cases of LoRWeB. When the tasks are significantly different from those seen in the training corpus (first row) or require modifying only small details (third row) LoRWeB might struggle with applying the needed transformation. Additionally, LoRWeB might only partly apply a transformation if it requires modifying multiple elements (second row).

