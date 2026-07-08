arXiv:2505.23758v1[cs.CV]29May2025

# LoRAShop: Training-Free Multi-Concept Image Generation and Editing with Rectified Flow Transformers

Yusuf Dalva Hidir Yesiltepe Pinar Yanardag Virginia Tech

https://lorashop.github.io/

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>|
|---|

|[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>|
|---|

|[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>|
|---|

|[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>|
|---|

|[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>|
|---|

Generation with Single/Multiple Concepts

|[Figure 30]|
|---|

|[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>|
|---|

|[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>|
|---|

|[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>|
|---|

Original

Edited

|[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>|
|---|

|[Figure 49]|
|---|

|[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>|
|---|

|[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>|
|---|

|[Figure 56]|
|---|

Original

Original

Edited

Edited

Editing with Single/Multiple Concepts

- Figure 1. LoRAShop. We present LoRAShop, a training-free framework enabling the simultaneous use of multiple LoRA adapters for generation and editing. By identifying the coarse boundaries of personalized concepts as subject priors, we allow the use of multiple LoRA adapters by eliminating the “cross-talk” between different adapters.

###### Abstract

We introduce LoRAShop, the first framework for multiconcept image editing with LoRA models. LoRAShop builds on a key observation about the feature interaction patterns inside Flux-style diffusion transformers: conceptspecific transformer features activate spatially coherent regions early in the denoising process. We harness this observation to derive a disentangled latent mask for each concept in a prior forward pass and blend the corresponding LoRA weights only within regions bounding the concepts to be personalized. The resulting edits seamlessly integrate multiple subjects or styles into the original scene while preserving global context, lighting, and fine details. Our experiments demonstrate that LoRAShop delivers better identity preservation compared to baselines. By eliminating retraining and external constraints, LoRAShop turns personalized diffusion models into a practical ‘photoshop-with-LoRAs’ tool and opens new avenues for compositional visual storytelling and rapid creative iteration.

###### 1. Introduction

The rapid progress in Text-to-image (T2I) generative models [27, 28, 32] has opened new creative avenues such as content generation [4, 21, 37] and editing [3, 5, 6, 22, 38, 48, 49], but users often desire customized outputs with specific topics or styles not present in the original training data [51]. Personalization techniques that fine-tune a pretrained generative model on a small set of user-provided images have emerged to meet this need. Notably, methods like DreamBooth [29] and Low-Rank Adaptation (LoRA) [12] allow T2I models to be customized, capturing userspecific concepts (e.g. a particular pet, a unique face, or a distinct art style) and regenerating them in new contexts with high fidelity. While single-concept personalization is a relatively simple task, multi-concept generation is a challenging problem: Given multiple fine-tuned concept models (e.g. several LoRAs trained on different subjects), how can we compose them to synthesize a coherent image containing all the custom concepts? Achieving such compositions is challenging because independently trained LoRAs can interfere with each other when combined, leading to identity distortions or one concept dominating the other – a phenomenon sometimes called “LoRA crosstalk” [10, 20, 24, 35]. Simply merging or applying multiple LoRAs naively often causes one concept to vanish or entangle attributes with the other [10]. Recent research indeed highlights that multi-concept generation remains nontrivial: combining personalized models typically degrades individual concept quality unless special measures are taken [10, 24, 35]. However, these methods still require training a new combined model or a fine-tuning process (e.g., im-

posing constraints during each LoRA’s training or running a post hoc alignment optimization).

While existing techniques can achieve multi-concept generation – i.e. producing a new image containing several personalized concepts – none of these methods addresses the task of multi-concept editing: modifying a given image to insert multiple new concepts. Multi-concept image editing presents a different set of challenges. Here, the goal is not to generate a scene from scratch, but to start from an input image and seamlessly blend in additional personalized elements (each defined by a LoRA model) into that image. A naive approach to this problem might be to apply iterative inpainting: for example, masking a region in the image and prompting the diffusion model (with the LoRA loaded) to generate the new concept in that area. Unfortunately, off-the-shelf inpainting with personalized diffusion models often yields artifacts and inconsistencies. The injected object or character may not blend naturally with the lighting and context of the original image, or the model may unintentionally alter the surrounding content. Another approach could be face-swapping or identity transfer, where a person’s face in the image is replaced with a personalized face (using a LoRA of that person). Although this can handle a single face, it often does not preserve the full appearance of the person, such as body features, and can produce unrealistic results.

In this paper, we propose LoRAShop, a novel framework that enables multi-concept image editing with LoRA models, without requiring any additional training, special auxiliary inputs, or external segmentation. Given an input image and a set of LoRA modules (each encoding a different concept), LoRAShop allows the user to insert each concept into the image at a desired location in a disentangled way. One of our key observations is a disentangled mask extraction technique that leverages the internal representations of the rectified-flow model to localize the influence of each subject to be personalized. In essence, as each LoRA is applied during the denoising process, our method extracts a coarse mask that delineates the regions where that concept significantly contributes to the image. By combining these masks with the user’s concept specifications, LoRAShop is able to blend multiple concepts directly into the diffusion latent in a controlled manner (see Fig. 1). Our experiments show that LoRA subjects blend naturally into the original scene, and their identities/styles match the LoRA concepts with high fidelity. Our approach does not require training of any new model or ensemble; it directly utilizes existing LoRAs and the base rectified-flow model at inference time, making it efficient and user-friendly. We believe that LoRAShop fills an important gap between personalized generation and image editing, opening the door to new creative workflows (such as “LoRAshopping” with generative models) that were previously impractical.

###### Stage 1: Self-Supervised Subject Prior Extraction Stage 2: Prior-Guided Blending of Residual Features

|Image Tokens| |
|---|---|
| | |

|Text Tokens| |
|---|---|
| | |

|Image Tokens|
|---|

|Double-Stream Block| | |
|---|---|---|
| | | |

|[Figure 57]| |
|---|---|
| | |

|[Figure 58]|
|---|

|[Figure 59]| |
|---|---|
| | |

###### Double-Stream Block

|[Figure 60]| |
|---|---|
| | |

Subject-prior Extraction (Sec. 3.2)

Blending Residual Features(Sec. 3.3)

|Single-Stream Block| | |
|---|---|---|
| | | |

[Figure 61]

###### +

|[Figure 62]|
|---|

|[Figure 63]| |
|---|---|
| ||[Figure 64]|
|---|
|

|[Figure 65]|
|---|

|Single-Stream Block| | |
|---|---|---|
| | | |

|Text Tokens|
|---|

|Image Tokens|
|---|

- Figure 2. LoRAShop Framework. LoRAShop enables multi-subject generation and editing over a two-stage training-free pipeline. First,

we extract the subject prior Mˆc′, which gives a coarse-level prior on where the concept of interest, c′, is located. Following, we introduce a blending mechanism over the transformer block residuals, which both enables seamless blending of customized features and bounds the region-of-interest for the LoRA adapter utilized.

###### 2. Related Work

Personalized Image Generation. Personalized image generation aims to inject a user-defined concept, typically a face, style, or object, into a text-to-image model so it can be used in future generations. Early work relied on Textual Inversion (TI) [9], which learns a single embedding that reproduces a user’s concept. TI is lightweight, but struggles to learn concepts involve high level of detail, where it learns to reconstruct the target concept with diffusion loss. DreamBooth (DB) [29] improves fidelity by fine-tuning selected model weights and reserving a rare token for the new concept, though at a higher compute cost. Later methods seek better quality–efficiency trade-offs: P+ [40] extends TI with a richer token representation; Custom Diffusion [19] trains only cross-attention layers; and DB-LoRA [31] applies low-rank adaptation [12] to store each concept in a small rank-limited update. Recent encoder-based systems such as StyleDrop [36], HyperDreamBooth [30], Taming Encoder [15], IP-Adapter [46], MS-Diffusion [42], MIPAdapter[13], InfiniteYou [16], OmniGen [45] and UNO [43] predict adapter features directly from reference images, enabling near-instant personalization but often with some loss of identity fidelity compared with full DreamBooth tuning.

Merging Multiple Concepts. Combining LoRAs for style and subject control remains as a challenging tasks, as combined adapters usually optimize overlapping representations. In achieving such a combination of personalized concepts, current work still faces certain challenges. Simple weight averaging [31] is fast but quickly causes in-

terference. Mix-of-Show [10] trains special embeddingdecomposed LoRAs that avoid this clash, yet it needs the original data and cannot use community models, such as those available on platforms like civit.ai [2]. ZipLoRA [33] merges one style and one content adapter but breaks down with more than one content LoRA. On the other hand, OMG [18] is based on an external segmenter to apply separate concepts, whose errors propagate to the result. Orthogonal Adaptation [24] keeps LoRAs in separate subspaces with additional constraints introduced, reducing cross-talk, but adds training overhead and likewise assumes data access. Our proposed approach differs from existing multi-concept generation methods since our main goal is ’editing’ as opposed to generation. Moreover, our method does not require any input conditions such as keypoints or segmentation masks.

###### 3. Method

We propose LoRAShop, a new training-free pipeline that enables the use of multiple LoRA adapters through a targeted feature blending scheme for multi-subject generation and editing. Our method, Multi-Subject Residual Blending (MSRB), consists of two fundamental stages: 1) the extraction of a subject prior that effectively highlights the spatial regions where each subject is intended to appear, and 2) the application of a residual feature blending scheme within the diffusion transformer that selectively merges the outputs of different LoRA adapters. This allows us to spatially combine features corresponding to distinct concepts, enabling coherent and disentangled multi-subject genera-

###### Concepts

Representative Images for each concept are shown below

|[Figure 66]|
|---|

|[Figure 67]|
|---|

|[Figure 68]|
|---|

|[Figure 69]|
|---|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|

|[Figure 73]|
|---|

<Watson> <Armas> <Gal>

<Kiernan> <Billie> <Lee> <DiCaprio> <Pitt>

|[Figure 74]|
|---|

|[Figure 75]|
|---|

|[Figure 76]|
|---|

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

<Kiernan> + <Watson>

<Billie> + <Watson>

<Billie> + <Gal>

Original

Original + <Kiernan> + <Watson> + <Gal>

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

Original + <Lee> + <DiCaprio> + <Pitt>

<Gal> + <Kiernan>

<Watson> + <Armas>

<Gal> + <Armas>

Original

- Figure 3. Editing Generated & Real Images with LoRAShop. We provide qualitative editing results with different human concepts. LoRAShop can achieve both edits on real and generated images. Due to non-intersecting subject prior extraction scheme of our framework, LoRAShop can perform edits with multiple concepts in one denoising pass.

tion and editing without any additional training.

###### 3.1. Preliminaries

Multi-Modal Diffusion Transformers. Multi-modal diffusion transformers (MM-DiT) [8] extend the DiT architecture by processing text and image tokens in two tightly coupled streams, enabling end-to-end text-to-image generation. Rectified-flow models such as FLUX adopt this design and alternate between two transformer block types. We denote blocks that keep separate parameter sets for the text and image streams as double-stream blocks, and those that apply a shared transformation to both streams as single-stream blocks. During the denoising trajectory, the network first aligns textual and visual features within the double-stream blocks and subsequently refines the fused representation in the single-stream blocks. All feature updates propagate through residual connections, an architectural property that our generation and editing protocol leverages directly.

Personalization via Low-Rank Adaptation. Low-Rank Adaptation (LoRA) [12] was originally introduced as a lightweight fine-tuning method for large language models. Instead of updating the full weight matrix W0 ∈ Rd×k, LoRA learns a low-rank increment, formulating the finetuned weights as W = W0 + ∆W = W0 + BA with B ∈ Rd×r, A ∈ Rr×k, and an intrinsic rank r ≪ min(d,k). Because only A and B are trained, the additional parameter count and memory footprint scale linearly with r, making LoRA especially attractive for large backbones. Following its success in NLP, LoRA has been

adopted for text-to-image diffusion models and, more recently, for rectified-flow transformers such as FLUX. We leverage single-subject LoRA adapters trained for rectifiedflow transformers and introduce a training-free mechanism that allows multiple adapters, each corresponding to a different subject, to be used simultaneously without any additional optimization.

###### 3.2. Self-Supervised Subject Prior Extraction

Training several LoRA adapters so they can be applied simultaneously is costly and often infeasible for large-scale denoisers: Every additional adapter consumes optimization memory, and jointly fine-tuning many of them tends to introduce interference and distribution drift. To bypass this bottleneck, we first predict, in inference time, where each personalized subject will emerge in latent space and then confine every adapter’s effect to the pixels assigned to that subject. The binary masks that delimit these regions are our subject priors. We extract each prior once in a short pseudo-denoising run that proceeds only until timestep γ, when latents are still close to noise, yet crossattention already carries strong spatial cues [5, 11]. Rectified flow transformers such as FLUX provide well-localized cross-attention maps. In particular, the map from the last block that still keeps the text and image streams separate (the double-stream block) gives the sharpest separation. For a prompt c and the token subset c′ naming one subject we compute

√

Mc′ = softmax QiKcT′/

d , (1)

###### portrait of a man and a woman, they are chefs

###### Block 19

Block 12

|[Figure 90]|
|---|

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

|[Figure 99]|
|---|

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

Block 20

Block 58

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

|[Figure 110]|
|---|

|[Figure 111]|
|---|

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Figure 4. Ablation Study. Ablation on transformer blocks, where Block 19 shows superior ability for separation between subjects.

where Qi are the image queries, Kc′ the keys of c′, and d the key dimension.

Because raw attention may fragment, we iteratively blur Mc′ with a 3 × 3 Gaussian kernel and renormalize until the super-threshold area forms a single connected component. Thresholding at the τ posterior quantile then produces the final binary mask, which we denote by Mˆc′, for the subject c′.

When multiple subjects are present, these masks can intersect, leading to undesirable “LoRA cross-talk”. To obtain non-overlapping maps, we stack the smoothed attention

maps { Mu}Nu=1, determine for every spatial position (i,j) the subject u with the strongest response,

k⋆(i,j) = arg max

, Mu(i,j)

u

Mmax(i,j) = Mk⋆(i,j)(i,j) (2) and finally define one-hot priors

###### Mˆu(i,j) = 1! u = k⋆(i,j) . (3)

The set Mˆu partitions the latent canvas without overlap and serves as the spatial guide for adapter mixing during generation and editing.

###### 3.3. Prior-Guided Blending of Residual Features

The diffusion transformer proceeds as usual, but at every block we overwrite the residual feature tensors wherever a subject prior is active. At block ℓ the frozen backbone produces a collection of R residual tensors, Fbaseℓ,r ∈ RS×C with r = 1,...,R, corresponding to the outputs of multi-modal attention, MLP, and any other sublayer that feeds a skip connection. In parallel, the k-th LoRA adapter contributes its counterparts F(ℓ,rk). The binary priors Mˆc′ ∈ {0,1}S indicate which latent tokens belong to subject c′.

For each token position p we turn the priors into weights, so the weights sum to one on the subject tokens and to zero

on background tokens.

Mˆk(p)

, ε ≪ 1, (4)

αc′(p) =

N u=1

Mˆu(p) + ε

Whether the block is double-stream (text and image kept separate) or single-stream, we treat it the same way: only image tokens are blended; prompt tokens keep their backbone residuals. For every image token p and every residual index r we substitute

N

αc′(p)F(ℓ,rk)(p), (5)

F˜ℓ,r(p) =

k=1

and feed F˜ℓ,r back through the block’s skip connection. If no subject claims token p ( u Mˆu(p) = 0), we leave Fbaseℓ,r (p) unchanged. Blending is disabled during the first until timestep t, letting the backbone establish the overall layout of the scene before subject-specific features are inserted. Because we mix the residual outputs of every sublayer rather than changing any weights, all adapters remain independent, and each subject influences exactly the tokens selected by its prior across the entire depth of the transformer.

###### 3.4. Editing with LoRAShop

LoRAShop intervenes only in the feature space of a rectified flow transformer: it neither modifies the noise schedule nor alters any model weights. During the reverse diffusion process, we overwrite residual features solely at token positions indicated by the subject priors, leaving all other tokens unchanged. Because this operation is local and linear, the global denoising trajectory, and thus the overall scene layout, remains intact. The same mechanism integrates seamlessly with inversion. We adopt the RF-Solver pipeline of [41], which uses a second-order solver to recover the latent noise corresponding to a target image. After reconstructing the latent, we utilize LoRAShop to edit the inverted latent. As illustrated in Fig. 1 and Fig. 3, this enables regioncontrolled insertion of multiple personalized concepts into

Ours DreamBooth IP-Adapter InfiniteYou UNO OmniGen

Ref.

|[Figure 120]|
|---|

|[Figure 121]|
|---|

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

a)

a happy elf, standing beneath glowing trees in a forest where moonlight streams through twisted branches.

Ref. DreamBooth MS-Diffusion MIP-Adapter

Ours

OmniGen

UNO

[Figure 128]

|[Figure 129]|
|---|

|[Figure 130]|
|---|

|[Figure 131]|
|---|

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

[Figure 135]

##### b)

[Figure 136]

[Figure 137]

a woman in a chef’s uniform, and a woman in a barista apron in a warmly lit café interior

Original Ours ReFace Original Ours ReFace

|[Figure 138]|
|---|

|[Figure 139]|
|---|

|[Figure 140]|
|---|

|[Figure 141]|
|---|

|[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>|
|---|

|[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>|
|---|

#### c)

- Figure 5. Qualitative Comparisons. We provide qualitative comparisons on three mainstream tasks: single-subject generation, multisubject generation and face swapping. Over all of the benchmarked tasks, LoRAShop provides superior performance against competing approaches.

real images while faithfully preserving the properties of the input.

###### 4. Experiments

We evaluate LoRAShop on both image generation and image editing tasks. For generation, we measure how well the method renders a single personalized subject and how reliably it composes multiple personalized subjects in one scene. For editing, we evaluate identity transfer on real images, replacing a person’s appearance with that encoded by a LoRA adapter. We provide the details of our experimental protocol along with the results in this section.

Experimental Setup We use FLUX.1-dev, as the rectified-flow transformer on which we build our approach. Our approach is based on utilizing pre-trained LoRA adapters for tasks such as single/multi-concept generation and editing. In all of our experiments, we use the LoRAs available at diffusers [39] library. We provide a complete list of LoRAs used in our experiments in the supplementary material, along with visual representations of these concepts for ease of understanding. Unless otherwise mentioned, we set the editing timestep t = 0.90,

γ = 0.94 and τ = 0.7, where we apply the proposed blending scheme (Sec. 3.3) onward timestep t during the reverse process. Our approach requires no training over the pre-trained adapters and can perform the aforementioned personalization task in inference time. We conduct our experiments using one NVIDIA L40S GPU. LoRAShop can generate images using two concepts approximately in 50 seconds, as opposed to the manual inference time of FLUX.1-dev which requires 30 seconds per image. Furthermore, since LoRAShop can apply each concept sequentially, we introduce no memory constraints on how many concepts that can be applied to a given image. See Fig. 2 for 4 subject generation results.

###### 4.1. Qualitative Results

We qualitatively assess the effectiveness of our method in single & multiple subjects and in generated & real images. To assess the visual performance of our framework, we demonstrate its capabilities in experiments on human subjects. Although LoRAShop can also perform edits on a variety of types of subject, we perform our experiments on human subjects due to the high level of details such concepts involve, and their wide-usage in customization tasks. Since

- Table 1. Quantitative Comparisons on Single-Subject Generation. We provide quantitative comparisons on single-subject generation. Our method outperforms the competing FLUX-based approaches in the overall performance, measured over identity similarity, prompt alignment and visual quality.

###### Method ID ↑ CLIP-T ↑ HPS ↑ Aesthetics ↑

DreamBooth 0.755 ± 0.089 0.429 ± 0.055 0.305 ± 0.030 6.311 ± 0.505 IP-Adapter-FLUX 0.309 ± 0.077 0.330 ± 0.053 0.272 ± 0.026 6.340 ± 0.408 InfiniteYou 0.683 ± 0.068 0.439 ± 0.039 0.307 ± 0.026 6.490 ± 0.459 Omni-Gen 0.657 ± 0.066 0.434 ± 0.043 0.311 ± 0.030 6.514 ± 0.448 UNO 0.486 ± 0.137 0.415 ± 0.051 0.289 ± 0.030 6.303 ± 0.527 Ours 0.740 ± 0.066 0.439 ± 0.047 0.321 ± 0.028 6.499 ± 0.529

our method requires no fine-tuning for LoRA-adapters, we can use any adapter trained for our base model. Furthermore, since our approach does not focus on a specific type of residuals (e.g. attention layer outputs), but operates on the overall representation space, we can also use LoRAs with different ranks and different sets of fine-tuned parameters together.

Editing on Generated and Real Images. We provide editing results with LoRAShop on both male and female subjects, where these LoRAs are trained with different sets of combinations, which involve different sets of weights, ranks, and presence of a trigger word. Presented in Fig. 3, LoRAShop can both perform edits on real & generated images, without altering any subject-independent details. Note that, since LoRA adapters offer us a way to utilize the rich semantics in the weight space of the denoiser, our approach can also perform changes to the body of the edited subject (Fig. 3, row 1), which exceeds the limits of the face swapping task and provide us an advanced way of editing images with customized concepts. Furthermore, as our subject prior extraction algorithm provides non-intersecting masks, our approach facilitates performing multiple edits with distinct LoRA adapters in a single denoising pass.

Qualitative Comparisons. We provide qualitative comparisons of our approach with competing methods on singlesubject, multi-subject and face swapping tasks. Since our proposed approach performs generation by editing, we enable blending of the residual features. While vanilla approaches such as DreamBooth [29] achieve subject-based generation results, since they fine-tune the weights of the original denoiser, they result in reduced prompt alignment and visual coherence. On the other end, encoder-based approaches such as IP-Adapter [46], InfiniteYou [16], UNO [43] and OmniGen [45] struggle to encode the identity features that are effectively captured by DreamBooth. In this regard, our approach offers the best of both worlds (Fig.

- 5 (a)), where we personalize only the regions related to the identity, which both achieves superior prompt alignment and personalization performance.

For multi-subject generation, we provide comparisons

with FLUX-based approaches such as UNO [43], OmniGen [45], DreamBooth [29] and SDXL[25]-based approaches MS-Diffusion [42] and MIP-Adapter [13]. We use federated averaging for DreamBooth, as a baseline towards multi-subject personalization. As we demonstrate Fig. 5 (b), our subject priors mitigate the confusion between similar concepts effectively, where the remaining approaches either attempt to merge the two identities into one, or fail to capture the identity accurately. In this regard, our approach outperforms the competing methods for multi-subject generation as a training-free solution, which can effectively reflect multiple concepts effectively, mitigating the “crosstalk” effect between the concepts. Additionally, we provide comparisons with methods combining multiple LoRA adapters in Fig. 7, where our method offers compositions with high quality, without any pose input. To benchmark our method in terms of editing, we select the face swapping task, where we use identity LoRAs to represent the identity to be inserted into the original image. As we qualitatively benchmark in Fig. 5 (c), our approach extends the limitations of identity swapping, which was a task that is limited with swapping the faces until today. As LoRA adapters are capable of capturing physical features in addition to facial features, LoRAShop enables the transfer of physical features in addition to the face of the source identity, in addition to superior fidelity against methods based on inpainting such as ReFace [1].

###### 4.2. Quantitative Results

Extending our benchmark in qualitative experiments, we benchmark the editing and generation performance of LoRAShop on three mainstream tasks. Specifically, we benchmark the performance of LoRAShop for single & multi concept generation along with face swapping task. We provide the details of each constructed benchmark below.

Single-Subject Generation. Following previous work, we populate a set of varying identities and generation prompts to benchmark our generation results. Among publicly available LoRA adapters, we select 15 identity LoRAs and generate a total of 520 images where 15 generation prompts

- Table 2. Quantitative Comparisons on Multi-Subject Generation. We benchmark our approach against FLUX and SDXL based methods. LoRAShop achieves superior identity preservation over multiple subjects, while maintaining the prompt alignment and visual quality of the base model.

Method ID ↑ CLIP-T ↑ HPS ↑ Aesthetics ↑

SDXL

based

OMG 0.305 ± 0.14 0.217 ± 0.09 0.212 ± 0.05 6.017 ± 0.35 MS-Diffusion 0.206 ± 0.05 0.251 ± 0.08 0.253 ± 0.03 6.119 ± 0.24 MIP-Adapter 0.209 ± 0.06 0.243 ± 0.07 0.236 ± 0.03 6.111 ± 0.30

FLUX

based

DreamBooth 0.444 ± 0.08 0.248 ± 0.08 0.259 ± 0.04 6.113 ± 0.30 OmniGen 0.453 ± 0.09 0.256 ± 0.08 0.258 ± 0.04 6.264 ± 0.26 UNO 0.270 ± 0.07 0.252 ± 0.08 0.255 ± 0.04 6.113 ± 0.36 Ours 0.532 ± 0.12 0.252 ± 0.08 0.260 ± 0.04 6.124 ± 0.29

- Table 3. User Study. We present user study results on identity preservation (Q1), and prompt alignment (Q2) for multi-subject generation task.

Method User Study - Q1 ↑ User Study - Q2 ↑

OMG 2.591 ± 0.25 3.332 ± 0.55 MS-Diffusion 2.596 ± 0.27 2.753 ± 0.19 MIP-Adapter 2.889 ± 0.26 3.123 ± 0.50

SDXL

based

DreamBooth 3.196 ± 0.10 4.060 ± 0.14 OmniGen 3.340 ± 0.32 4.012 ± 0.26 UNO 2.711 ± 0.23 3.587 ± 0.44 Ours 3.762 ± 0.25 4.230 ± 0.13

FLUX

based

were applied to each identity separately. To adequately assess both the personalization, prompt alignment and visual coherence of the generated outputs, we construct our benchmark prompts with themes such as artistic creations, contexts defined by activities and superficial concepts (see Fig. 5 (a)). We provide the complete list of prompts we use for our benchmark in the supplementary material. To assess both identity preservation, text alignment and visual coherence of the generated images, we utilize ArcFace embeddings [7], CLIP-T similarity [26], HPS score [44] and Aesthetics score1. We present the quantitative results in Table 1. As quantitative metrics also show, our approach leads to a sweet spot between identity preservation, prompt alignment, and visual coherence, as we utilize the generative priors in our residual blending scheme.

Multi-Subject Generation. In addition to our benchmark for single-subject generation, we also benchmark our approach against multi-subject generation methods. Using the 15 subjects that we used in our benchmark for single-subject generation, we initially generate random pairs of identities with corresponding prompts to create a benchmark for the two-subject generation task. In our evaluations, we compare our method with both FLUX-based methods UNO

1https : / / github . com / christophschuhmann / improved-aesthetic-predictor

[43], OmniGen [45] and DreamBooth (FedAvg) [29] and SDXL-based methods OMG [18], MS-Diffusion [42] and MIP-Adapter [13]. As we present in the results in Table 2, our approach achieves both superior prompt alignment, visual coherence, and identity preservation.

User Study. Supplementary to our benchmark on multisubject generation, we also conducted a user study to perceptually evaluate the generation quality of our approach. We conducted our study on 50 participants over Prolific.com crowdsourcing platform, where each participant is asked to assess 70 images involving multiple subjects. In our study, we evaluated the generation performance in which users are asked to rate the images in two aspects on a Likert scale (1: poor, 5: excellent): (Q1) alignment with the target identities and (Q2) alignment with the generation prompt. We provide the result of our study in Tab. 3. As our results also demonstrate, LoRAShop outperforms the competing approaches in both prompt alignment and identity preservation. Please see Appendix for additional details about the user study.

Face Swapping. We also benchmark our approach in the face swapping task. We compare our method with an inpainting-based swapping approach ReFace [1]. Although our approach does not involve any hard constraints for content preservation such as inpainting masks that restrict the regions to be edited, our method still achieves competitive

- Table 4. Quantitative Comparisons on Face Swapping. We benchmark LoRAShop against REFace [1]. While performing on-par in input preservation, LoRAShop introduces significant improvements in identity preservation.

###### Method ID ↑ DINO ↑ CLIP-I ↑ LPIPS ↓

ReFace 0.330 ± 0.091 0.982 ± 0.012 0.940 ± 0.038 0.031 ± 0.033 Ours 0.709 ± 0.101 0.970 ± 0.019 0.926 ± 0.037 0.050 ± 0.019

|𝑡 = 0.97|
|---|

|𝑡 = 0.94|
|---|

|𝑡 = 0.90|
|---|

|𝑡 = 0.97|
|---|

|𝑡 = 0.94|
|---|

|𝑡 = 0.90|
|---|

Original

Original

|[Figure 148]|
|---|

|[Figure 149]|
|---|

|[Figure 150]|
|---|

|[Figure 151]|
|---|

|[Figure 152]|
|---|

|[Figure 153]|
|---|

|[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>|
|---|

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

a knight, and a sorcerer, a magical aura surrounding them

a woman in a yellow raincoat, and a woman under a black umbrella

Original 𝛾 = 1.0 𝛾 = 0.94 𝛾 = 0.86 Original 𝛾 = 1.0 𝛾 = 0.94 𝛾 = 0.86

[Figure 164]

[Figure 165]

[Figure 166]

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

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

a bride, alongside a bridesmaid, both facing forward

a graduate and a woman in a floral dress, posing for a graduation photo

|𝜏 = 0.6|
|---|

|𝜏 = 0.7|
|---|

|𝜏 = 0.5|
|---|

|𝜏 = 0.5|
|---|

|𝜏 = 0.6|
|---|

|𝜏 = 0.7|
|---|

Original

Original

|[Figure 192]<br><br>[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]<br><br>|
|---|

[Figure 197]

[Figure 198]

|[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]<br><br>|
|---|

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

a woman in a blue dress and a woman in a black dress, in a casino

a gala portrait of a woman in a red gown, beside a woman in a silver dress

- Figure 6. Ablation Study. (a) Ablations on hyperparameters time step t, subject’s prior extraction step γ, and the posterior threshold for binarization of the subject’s prior masks τ. (b) Ablation on transformer blocks, where Block 19 shows superior ability for separation between subjects.

performance in terms of input preservation, which we measure using DINO [23], CLIP-I [26], and LPIPS [50] metrics. Furthermore, LoRAShop leads to significant improvements in identity preservation properties. Note that our approach extends the bounds of the face swapping task and can perform full identity transfer by editing the physical appearance, in comparison to inpainting-based swapping approaches.

###### 4.3. Ablation Studies

Ablations on Transformer Blocks. To further justify the use of the last double-stream block for subject prior extraction, and to provide an investigation over the roles of different transformer blocks, we provide ablations over the masks extracted from different transformer blocks in Fig. 4. As shown by the attention masks Mc′ extracted for the subject c′ (e.g. woman), we observe that through the double-stream blocks (blocks 0-19), FLUX constructs the semantic context and is able to perform the separation between different

concepts at the end of these blocks. In the single-stream blocks, we observe that the model attempts to focus more on the visual details, which results in maps spread out over different entities. Building up on this observation, we build our subject prior extraction scheme on the attention maps produced by the last double-stream block (e.g. Block 19).

Ablations on Editing Parameters. Complementary to the block selection, LoRAShop includes three additional hyperparameters for editing, which are the editing time step t, the subject’s prior extraction step γ, and the posterior threshold for binarization of the subject’s prior masks τ. We provide ablations on these hyperparameters in Fig. 6. Similarly to the trend observed in diffusion-based editing methods, LoRAShop is able to preserve the adapter-irrelevant features of the input image better when the edit is performed in later timesteps. Considering that the effect should be effective enough and preserve certain features of the input image, we achieve a good balance for the timestep t. Regarding the subject priors extracted prior to the denoising

Ours

Ours

|[Figure 220]|
|---|

|[Figure 221]|
|---|

|[Figure 222]|
|---|

|[Figure 223]|
|---|

LoRACLR

Orthogonal Adaptation

LoRACLR Orthogonal Adaptation

|[Figure 224]|
|---|

|[Figure 225]|
|---|

|[Figure 226]|
|---|

|[Figure 227]|
|---|

Mix-of-Show Prompt+

Mix-of-Show Prompt+

|[Figure 228]|
|---|

|[Figure 229]|
|---|

|[Figure 230]|
|---|

|[Figure 231]|
|---|

<Lebron> & <Margot> & <Taylor>, performing a surgery together in an operating room…

<Lebron> & <Pitt> & <Gosling> & <Messi>, on the deck of a wooden ship, in adventurous fantasy style…

- Figure 7. Qualitative Comparisons with Multi Composition Methods. We compare our method with multi-composition methods operating on multiple LoRA adapters, LoRAShop outperforms the competing approaches while not relying on a pose input, and thus generate compositions with diverse settings.

steps, we recognize that the introduced parameters have a significant impact on the quality of the mask. In general, we find γ = 0.94 and τ = 0.7 as suitable hyperparameters, which we utilize in all of our experiments for complete and accurate enough masks.

###### 5. Discussion

Limitations and Broader Impact. Because the extracted masks inherit the latent biases of the underlying diffusion model (e.g., greater attention to faces, stereotypical gender features, or saturated colors) [17, 47], they can sometimes mislocate or underrepresent certain regions, leading to less coherent or unbalanced edits, particularly for concepts underrepresented in the model’s pretraining data. Our mask extraction leverages attention patterns unique to the Flux architecture; other diffusion backbones (e.g., SDXL-Turbo) may require re-tuning of threshold parameters or yield less coherent masks. This limits immediate portability across all T2I models. Like other powerful editing tools, LoRAShop can be used to create non-consensual content. We encourage deployment within responsible-AI guardrails, but broader ethical safeguards remain necessary. Nevertheless, LoRAShop demonstrates—for the first time—training-free, region-controlled multi-concept editing with LoRAs, unlocking new creative workflows and research directions in compositional image manipulation.

Conclusion. We presented LoRAShop, the first trainingfree framework that enables region-controlled multiconcept image editing with off-the-shelf LoRA modules. By uncovering, and exploiting, spatially coherent activa-

tion patterns inside Flux diffusion transformers, we devised a disentangled latent-mask extraction procedure that lets each LoRA act only where it is intended, eliminating crossconcept interference. Without any extra optimization, segmentation, or auxiliary guidance, LoRAShop seamlessly blends multiple personalized subjects or styles into an input image, preserving both global context and fine local detail. Beyond advancing the state of the art in personalized image editing, LoRAShop turns diffusion models into an intuitive “photoshop-with-LoRAs,” opening new possibilities for collaborative storytelling, product visualization, and rapid creative iteration.

###### References

- [1] Sanoojan Baliah, Qinliang Lin, Shengcai Liao, Xiaodan Liang, and Muhammad Haris Khan. Realistic and efficient face swapping: A unified approach with diffusion models. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 1062–1071. IEEE, 2025. 7, 8, 9
- [2] Civitai. https://civitai.com, 2020. 3
- [3] Yusuf Dalva and Pinar Yanardag. Noiseclr: A contrastive learning approach for unsupervised discovery of interpretable directions in diffusion models. arXiv preprint arXiv:2312.05390, 2023. 2
- [4] Yusuf Dalva, Yijun Li, Qing Liu, Nanxuan Zhao, Jianming Zhang, Zhe Lin, and Pinar Yanardag. Layerfusion: Harmonized multi-layer text-to-image generation with generative priors. arXiv preprint arXiv:2412.04460, 2024. 2
- [5] Yusuf Dalva, Kavana Venkatesh, and Pinar Yanardag. Fluxspace: Disentangled semantic editing in rectified flow transformers, 2024. 2, 4

- [6] Yusuf Dalva, Hidir Yesiltepe, and Pinar Yanardag. Gantastic: Gan-based transfer of interpretable directions for disentangled image editing in text-to-image diffusion models. arXiv preprint arXiv:2403.19645, 2024. 2
- [7] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In CVPR, pages 4690–4699, 2019. 8, 15
- [8] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 4

- [9] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 3
- [10] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models. arXiv preprint arXiv:2305.18292, 2023. 2, 3, 13
- [11] Alec Helbling, Tuna Han Salih Meral, Ben Hoover, Pinar Yanardag, and Duen Horng Chau. Conceptattention: Diffusion transformers learn highly interpretable features. arXiv preprint arXiv:2502.04320, 2025. 4
- [12] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 2, 3, 4
- [13] Qihan Huang, Siming Fu, Jinlong Liu, Hao Jiang, Yipeng Yu, and Jie Song. Resolving multi-condition confusion for finetuning-free personalized image generation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3707–3714, 2025. 3, 7, 8
- [14] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 15
- [15] Xuhui Jia, Yang Zhao, Kelvin CK Chan, Yandong Li, Han Zhang, Boqing Gong, Tingbo Hou, Huisheng Wang, and Yu-Chuan Su. Taming encoder for zero fine-tuning image customization with text-to-image diffusion models. arXiv preprint arXiv:2304.02642, 2023. 3
- [16] Liming Jiang, Qing Yan, Yumin Jia, Zichuan Liu, Hao Kang, and Xin Lu. InfiniteYou: Flexible photo recrafting while preserving your identity. arXiv preprint, arXiv:2503.16418,

2025. 3, 7

- [17] Tahira Kazimi, Ritika Allada, and Pinar Yanardag. Explaining in diffusion: Explaining a classifier through hierarchical semantics with text-to-image diffusion models. arXiv preprint arXiv:2412.18604, 2024. 10
- [18] Zhe Kong, Yong Zhang, Tianyu Yang, Tao Wang, Kaihao Zhang, Bizhu Wu, Guanying Chen, Wei Liu, and Wenhan Luo. Omg: Occlusion-friendly personalized multiconcept generation in diffusion models. arXiv preprint arXiv:2403.10983, 2024. 3, 8

- [19] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023. 3
- [20] Tuna Han Salih Meral, Enis Simsar, Federico Tombari, and Pinar Yanardag. Clora: A contrastive approach to compose multiple lora models. arXiv preprint arXiv:2403.19776,

2024. 2

- [21] Tuna Han Salih Meral, Enis Simsar, Federico Tombari, and Pinar Yanardag. Conform: Contrast is all you need for highfidelity text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9005–9014, 2024. 2
- [22] Tuna Han Salih Meral, Hidir Yesiltepe, Connor Dunlop, and Pinar Yanardag. Motionflow: Attention-driven motion transfer in video diffusion models. arXiv preprint arXiv:2412.05275, 2024. 2
- [23] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 9
- [24] Ryan Po, Guandao Yang, Kfir Aberman, and Gordon Wetzstein. Orthogonal adaptation for modular customization of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7964–7973, 2024. 2, 3, 13
- [25] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 7
- [26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 8, 9, 15
- [27] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125,

2022. 2

- [28] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [29] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023. 2, 3, 7, 8, 16
- [30] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for

- fast personalization of text-to-image models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6527–6536, 2024. 3
- [31] Simo Ryu. Low-rank adaptation for fast text-to-image diffusion fine-tuning, 2023. 3
- [32] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487, 2022. 2
- [33] Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora: Any subject in any style by effectively merging loras. arXiv preprint arXiv:2311.13600, 2023. 3
- [34] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 15
- [35] Enis Simsar, Thomas Hofmann, Federico Tombari, and Pinar Yanardag. Loraclr: Contrastive adaptation for customization of diffusion models, 2024. 2, 13
- [36] Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, et al. Styledrop: Text-to-image generation in any style. arXiv preprint arXiv:2306.00983,

2023. 3

- [37] Kavana Venkatesh, Yusuf Dalva, Ismini Lourentzou, and Pinar Yanardag. Context canvas: Enhancing text-to-image diffusion models with knowledge graph-based rag. arXiv preprint arXiv:2412.09614, 2024. 2
- [38] Kavana Venkatesh, Connor Dunlop, and Pinar Yanardag. Crea: A collaborative multi-agent framework for creative content generation with diffusion models. arXiv preprint arXiv:2504.05306, 2025. 2
- [39] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/ diffusers, 2022. 6
- [40] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. p+: Extended textual conditioning in text-toimage generation. arXiv preprint arXiv:2303.09522, 2023. 3, 13
- [41] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. arXiv preprint arXiv:2411.04746, 2024. 5
- [42] Xierui Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. MS-diffusion: Multi-subject zero-shot image personalization with layout guidance. In The Thirteenth International Conference on Learning Representations, 2025. 3, 7, 8
- [43] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation. arXiv preprint arXiv:2504.02160, 2025. 3, 7, 8

- [44] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341,

2023. 8

- [45] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024. 3, 7, 8
- [46] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 3, 7

- [47] Hidir Yesiltepe, Kiymet Akdemir, and Pinar Yanardag. Mist: Mitigating intersectional bias with disentangled crossattention editing in text-to-image diffusion models. arXiv preprint arXiv:2403.19738, 2024. 10
- [48] Hidir Yesiltepe, Yusuf Dalva, and Pinar Yanardag. The curious case of end token: A zero-shot disentangled image editing using clip. arXiv preprint arXiv:2406.00457, 2024. 2
- [49] Hidir Yesiltepe, Tuna Han Salih Meral, Connor Dunlop, and Pinar Yanardag. Motionshop: Zero-shot motion transfer in video diffusion models with mixture of score guidance. arXiv preprint arXiv:2412.05355, 2024. 2
- [50] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 9, 15
- [51] Matthew Zheng, Enis Simsar, Hidir Yesiltepe, Federico Tombari, Joel Simon, and Pinar Yanardag Delul. Stylebreeder: Exploring and democratizing artistic styles through text-to-image models. Advances in Neural Information Processing Systems, 37:34098–34122, 2024. 2

# LoRAShop: Training-Free Multi-Concept Image Generation and Editing with Rectified Flow Transformers

## Supplementary Material

###### Table of Contents

- A. Details of User Study 13
- B. Supplementary Generation and Editing Examples 13
- C. Additional Comparisons 13
- D. Detailed Masking and Blending Algorithm 13
- E. Experiment Details 14
- F. List of LoRA Adapters 15

###### A. Details of User Study

We provide a sample question for the user study conducted in Fig. 8. To assess both the identity preservation and prompt alignment capabilities of our approach, we direct two questions to the participants of our study. The users are also provided representative examples of the personalized subjects, where these images are outsourced from assets available for public use. Then, the users are asked to rate the provided image on a Likert scale, where 1 corresponds to an unsuccessful generation and 5 corresponds to a successful generation.

[Figure 232]

Figure 8. User Interface of our User Study.

###### B. Supplementary Generation and Editing Examples

Supplementary to the editing and generation examples provided in the main paper, we provide supplementary results from LoRAShop in this section. Specifically, we provide examples of four subject generation in Fig. 9, three subject generation in Fig. 10, two subject generation in 11, and a combination of human and non-human adapters in Fig. 12. As we demonstrate qualitatively, our approach can both handle multiple instances of the same type of entities (e.g. woman) and different type of entities (e.g. man, sunglasses, clothing).

###### C. Additional Comparisons

We compare our method against multi-concept LoRA composition approaches, including Mix-of-Show [10], LoRACLR [35], Orthogonal Adaptation [24], and Prompt+ [40] in Fig. 15. Notably, the first three methods require a pose condition for generating compositions, and the first two depend on specialized LoRA models such as EdLoRA, which limits their applicability when using community LoRAs from platforms like Civit.ai. In contrast, our method operates without pose conditions, retraining, or model merging, enabling successful composition using arbitrary LoRA models out of the box. Additionally, our method can compose LoRAs with different characteristics (e.g. different ranks and different sets of parameters), by operating on output space only. We also highlight that other methods do not support Flux, thus we visually compare with their Stable Diffusion-based generations.

We also note that LoRACLR [35], Orthogonal Adaptation [24], and Prompt+ [40] do not have publicly available implementations, which prevents us from conducting a quantitative comparison.

###### D. Detailed Masking and Blending Algorithm

To further clarify the details of our method, we provide detailed descriptions of subject prior extraction and residual blending scheme introduced in the main paper. For the subject prior extraction, we provide the details of blob construction algorithm in 1. In addition, to further clarify the blending process, we describe the blending process for a given residual (from a transformer block) in Alg. 2. Note that this blending operation is applicable for all residual features outputted inside the transformer blocks.

|[Figure 233]|
|---|

|[Figure 234]|
|---|

|[Figure 235]|
|---|

a knight, an elf, a sorcerer and a superheroine standing in a sunny forest

|[Figure 236]|
|---|

|[Figure 237]|
|---|

|[Figure 238]|
|---|

Women in red, blue, black and green suits, playing poker in a casino

|[Figure 239]|
|---|

|[Figure 240]|
|---|

[Figure 241]

Women in a floral dress, a trench coat, a business suit and a red dress, walking in downtown

- Figure 9. Multi-subject composition results on four human subjects. As our approach does not rely on any other external conditioning like pose conditioning, LoRAShop can utilize the generative capabilities of FLUX, and thus generate outputs with high fidelity and superior prompt alignment. In the provided examples, we utilize the concepts <Margot>, <Gal>, <Kiernan> and <Beer>.

- Algorithm 1 HOMOGENEOUSBLOB

Require: Soft mask M ∈ [0,1]B×H∗W×1, image size (H,W), Gaussian size k, variance σ, threshold t, maximum passes

P, mode flatten, distance parameter λ

- 1: M ← reshape(M,⟨B,1,H,W⟩)
- 2: G ← GaussianKernel(k,σ)
- 3: M ← renorm(M) ▷ 0–1 scaling
- 4: for p = 1 to P do
- 5: M ← renorm conv2d(M,G)
- 6: if every batch sample has ≤ 1 connected component above t then
- 7: break
- 8: end if
- 9: end for
- 10: ▷ Homogenise the blob
- 11: P ← {M=max(M)} ▷ Use the global peak as a single-pixel marker
- 12: M ← morph reconstruct P,M ▷ Flood-fill outward until original mask intensity is reached

- 13: M ← renorm(M) ▷ Rescale result to [0,1]; yields a flat, uniform blob
- 14: Mˆ ← reshape(M,⟨B,H∗W,1⟩)
- 15: return Mˆ

###### E. Experiment Details

In this section, we provide supplementary details on our quantitative evaluations and provide the specifics of the

|[Figure 242]|
|---|

|[Figure 243]|
|---|

|[Figure 244]|
|---|

portrait photo of three men attending a friend’s wedding, in a sunny garden

|[Figure 245]|
|---|

|[Figure 246]|
|---|

|[Figure 247]|
|---|

a gala setting, three men wearing suits in italian style, in a luxurious dinner

|[Figure 248]|
|---|

|[Figure 249]|
|---|

|[Figure 250]|
|---|

Three men in a gym, wearing sport clothes

- Figure 10. Multi-subject composition results for three subjects. We provide generation results for the subjects <Pitt>, <Elon> and <DiCaprio>. We provide generation results on three different generation prompts, with different compositions of the subjects.

metrics utilized and the prompt sets used. We provide the prompts that we use for the evaluation of the single-subject and multi-subject generation tasks in Table 8 and Table 9, where we generate the prompt set with GPT-4o[14]. In the following, we provide the details for each of the metrics that we use in our evaluations.

- • ID: We use the InsightFace2 codebase for the ID similarity metric. Specifically, we use ArcFace [7] embeddings provided in their implementation, using the buffalo l variant.

- • CLIP: To assess text-to-image similarity for single/multi subject generation tasks, and image-to-image similarity for face swapping benchmark, we utilize CLIP [26] as our feature extractor. In all of our experiments, we use the big-G variant of the model3.
- • HPS: As a secondary metric to quantify text-to-image alignment, we utilize the Human Preference Score (HPS), which is fine-tuned with user preferences. In our experiments, we use the HPSv24 variant.
- • Aesthetics: To assess the quality of the generated images,

- 2https://github.com/deepinsight/insightface
- 3https://huggingface.co/laion/CLIP- ViT- bigG-

14-laion2B-39B-b160k

- 4https://github.com/tgxs002/HPSv2

we utilize the aesthetics score for single and multi subject generation tasks. We use the second version of the predictor in all of our experiments5.

- • DINO: As a secondary metric to assess the input preservation for the face swapping task, we use DINO for our benchmark. We use the checkpoints from https: //huggingface.co/facebook/dinov2-base.
- • LPIPS: Following the common practice from image editing tasks, we utilize LPIPS [50] score with VGG [34] backbone.

For all of the competing approaches, we use the default hyperparameter setups and their corresponding official implementations.

###### F. List of LoRA Adapters

We provide a complete list of LoRA adapters used in this section. Specifically, we provide the list of the LoRA adapters for woman subjects in Tab. 5, man subjects in Tab. 6 and non-human subjects in Tab. 7. For each of the adapters, we provide representative images for each, to help readers identify the subjects. Note that, we provide this

5https : / / huggingface . co / shunk031 / aesthetics predictor-v2-sac-logos-ava1-l14-linearMSE

|[Figure 251]|
|---|

|[Figure 252]|
|---|

|[Figure 253]|
|---|

|[Figure 254]|
|---|

a woman in a black business suit, next to a woman in a beige blazer, in a modern office setting

|[Figure 255]|
|---|

|[Figure 256]|
|---|

|[Figure 257]|
|---|

|[Figure 258]|
|---|

two men sitting in a park, wearing casual clothes

|[Figure 259]|
|---|

|[Figure 260]|
|---|

|[Figure 261]|
|---|

|[Figure 262]|
|---|

a man and a woman, having a business dinner in a fancy restaurant, posing for a photo

- Figure 11. Multi-subject composition results for two subjects. We provide generation results for the concepts <Armas>, <Sabrina>, <Pitt>, <DiCaprio>. As we demonstrate in the examples, LoRAShop can perform compositions between the same type (e.g. womanwoman) and different type (e.g. man-woman) of identity concepts.

list as a legend, where the adpater icons are in match with the ones used in the main paper. As exceptions, we train the LoRA adapters for <Gosling> and <Lebron> using Dreambooth [29].

|[Figure 263]|
|---|

|[Figure 264]|
|---|

a woman posing in front of a car

|[Figure 265]|
|---|

|[Figure 266]|
|---|

|[Figure 267]|
|---|

|[Figure 268]|
|---|

a man/woman and a cat, playing together in a sunny backyard

|[Figure 269]|
|---|

|[Figure 270]|
|---|

|[Figure 271]|
|---|

|[Figure 272]|
|---|

a man/woman with sunglasses, wearing a jacket. Standing in the downtown at night

- Figure 12. Multi-subject composition results generated by our method on different types of objects. As can be seen in the examples LoRAShop can perform combinations between different types of concepts.

### Original +<Watson> +<Gal> +<Kiernan> +<Beer>

|[Figure 273]|
|---|

|[Figure 274]|
|---|

|[Figure 275]|
|---|

|[Figure 276]|
|---|

|[Figure 277]|
|---|

|[Figure 278]|
|---|

|[Figure 279]|
|---|

|[Figure 280]|
|---|

|[Figure 281]|
|---|

|[Figure 282]|
|---|

|[Figure 283]|
|---|

|[Figure 284]|
|---|

|[Figure 285]|
|---|

|[Figure 286]|
|---|

|[Figure 287]|
|---|

|[Figure 288]|
|---|

|[Figure 289]|
|---|

|[Figure 290]|
|---|

|[Figure 291]|
|---|

|[Figure 292]|
|---|

- Figure 13. Face Swapping results with LoRAShop. As we demonstrate in the provided examples, our editing approach offers a seamless blending between the input subject and the target identity, while preserving the input characteristics.

###### Original +<Watson> +<Gal> +<Kiernan> +<Beer>

|[Figure 293]|
|---|

|[Figure 294]|
|---|

|[Figure 295]|
|---|

|[Figure 296]|
|---|

|[Figure 297]|
|---|

|[Figure 298]|
|---|

|[Figure 299]|
|---|

|[Figure 300]|
|---|

|[Figure 301]|
|---|

|[Figure 302]|
|---|

|[Figure 303]|
|---|

|[Figure 304]|
|---|

|[Figure 305]|
|---|

|[Figure 306]|
|---|

|[Figure 307]|
|---|

|[Figure 308]|
|---|

|[Figure 309]|
|---|

|[Figure 310]|
|---|

|[Figure 311]|
|---|

|[Figure 312]|
|---|

Figure 14. Face Swapping results with LoRAShop.

Ours

|[Figure 313]|
|---|

LoraCLR Orthogonal Adaptation

|[Figure 314]|
|---|

|[Figure 315]|
|---|

Mix-of-Show Prompt+

|[Figure 316]|
|---|

|[Figure 317]|
|---|

<Lebron> and <Taylor>, inside a futuristic spaceship, sci-fi realism

Figure 15. Comparison with state-of-the-art multi composition methods, on two subject generation task.

- Algorithm 2 RESIDUALBLENDING

###### Require:

Rbase ∈ RS×C ▷ residual from frozen backbone R(k) ∈ RS×C for k = 1,...,N ▷ residuals from N LoRA adapters Mˆk ∈ {0,1}S for k = 1,...,N ▷ token-wise subject priors I ⊆ {1,...,S} ▷ indices of image tokens ε small constant ▷ avoids divide-by-zero

Ensure: R ∈ RS×C ▷ blended residual tensor

- 1: for each token index p = 1,...,S do
- 2: if p ∈/ I then ▷ prompt token: no blending
- 3: R(p) ← Rbase(p)
- 4: continue
- 5: end if
- 6: sumMask ← Nu=1

Mˆu(p)

- 7: if sumMask = 0 then ▷ background token
- 8: R(p) ← Rbase(p)
- 9: else ▷ token claimed by a subject
- 10: for k = 1 to N do
- 11: αk ← Mˆk(p)/(sumMask + ε) ▷ normalise prior to a weight
- 12: end for
- 13: R(p) ← Nk=1 αk R(k)(p) ▷ blend adapter residuals according to weights
- 14: end if
- 15: end for
- 16: return R ▷ ready for the block’s skip connection

###### Adapter Icon Adapter Tag URL of the Adapter

[Figure 318]

<Armas> https://huggingface.co/Trenddwdw/Ana_de_Armas

[Figure 319]

<Billie> https://huggingface.co/punzel/flux_billie_eilish

[Figure 320]

<Watson> https://huggingface.co/punzel/flux_emma_watson

[Figure 321]

<Gal> https://huggingface.co/punzel/flux_gal_gadot

[Figure 322]

<Kiernan> https://huggingface.co/punzel/flux_kiernan_shipka

[Figure 323]

<Margot> https://huggingface.co/punzel/flux_margot_robbie

[Figure 324]

<Margot> https://huggingface.co/punzel/flux_emma_stone

[Figure 325]

<Beer> https://huggingface.co/punzel/flux_madison_beer

[Figure 326]

https : / / huggingface . co / mmaluchnick / sabrina carpenter-flux-model

<Sabrina>

[Figure 327]

https : / / huggingface . co / DeZoomer / TaylorSwift FluxLora

<Taylor>

Table 5. Image-and-text comparison table.

###### Adapter Icon Adapter Tag URL of the Adapter

[Figure 328]

<DiCaprio> https://huggingface.co/openfree/leonardo-dicaprio

[Figure 329]

<Pitt> https://huggingface.co/Trenddwdw/Brad_Pitt

[Figure 330]

<Lee> https://huggingface.co/openfree/bruce-lee

[Figure 331]

https://huggingface.co/roelfrenkema/flux1.lora. elonmusk

<Elon>

[Figure 332]

<Messi> https://huggingface.co/namita2991/messi

Table 6. Image-and-text comparison table.

###### Adapter Icon Adapter Tag URL of the Adapter

[Figure 333]

<Lumiva> https://huggingface.co/Litqecko/lumiva-glasses

[Figure 334]

https :// huggingface. co / Oscar2384 / Loewe _ Hybrid _ bomber_jacket_in_nappa

<Jacket>

[Figure 335]

https://huggingface.co/martintomov/moncler-dress1000-v1

<Dress>

[Figure 336]

<Tower> https://huggingface.co/seawolf2357/ntower

[Figure 337]

https://huggingface.co/ginipick/flux-lora-ericcat

<Cat>

[Figure 338]

https://huggingface.co/seawolf2357/flux-lora-carrolls-royce

<Royce>

Table 7. Image-and-text comparison table.

- P1 a woman/man rendered in a stylized manner is centered in the image, standing in front of a backdrop of expressive brushstrokes and vibrant color blocks.
- P2 a woman/man illustrated in pencil is centered in the frame, with fine shading and linework defining her/his face, placed against a softly sketched background.
- P3 a woman/man illustrated with smooth digital brushwork is centered in the image, with soft ambient lighting and a clean gradient background behind her/him.
- P4 a woman/man rendered in art deco style is centered in the scene, framed by angular gold patterns and symmetrical borders in an ornate composition.
- P5 a woman/man is centered in the image, rendered in a cyberpunk painting style with neon reflections casting pink and blue highlights across her/his face, glowing circuitry traced along her/his cheekbones, and a blurred futuristic cityscape of holograms and rain-soaked signs behind her/him.
- P6 a woman/man with a happy expression, sitting near a tall window with natural light falling across her/his face, while shadows from nearby plants frame the soft background.
- P7 a beautiful woman/man is centered in a cozy room filled with bookshelves and warm lighting, her/his face lit by a glowing screen as she/he laughs during a video call.
- P8 a woman/man with a nervous expression on a misty morning trail, the background gently blurs into distant trees and dew-covered grass.
- P9 a woman/man with a happy expression in a warmly lit kitchen, preparing a meal with a relaxed expression, surrounded by ingredients and subtle reflections from the counter.
- P10 a woman/man is centered at a caf´e table, sketching in a notebook with soft light falling on her/his face, as the background softly fades into rustic textures and furniture.
- P11 a woman/man knight with a fierce expression, wearing intricately detailed medieval armor, standing on a battlefield at sunset as orange light reflects off her/his head and the silhouettes of fallen weapons surround her/him.
- P12 a woman/man sorcerer is centered in the image, casting a glowing spell with both hands, her/his face illuminated by swirling magical energy, while runes float in the air and a faint aura pulses around her/him in the twilight mist.
- P13 a futuristic cyborg woman/man is centered in the image, with a metallic faceplate, cybernetic implants across her/his jaw and temple, and glowing blue circuitry along her/his neck, standing in front of a neon-lit skyline under a starless night sky.
- P14 a woman/man dragon rider is centered in the image, her/his face framed by windswept hair and a dark leather hood, with the neck of a black-scaled dragon behind her/him and storm clouds swirling in the sky, her/his expression fierce and focused as wind lifts her/his cloak around her/his shoulders.
- P15 a happy woman/man elf in a portrait photo setting, with long silver hair and pointed ears, cloaked in forestgreen robes, standing beneath ancient glowing trees in an enchanted forest where magical particles float in the air and moonlight streams through twisted branches.

Table 8. Prompt list for single-subject generation.

- P1 a close-up profile photo of a woman in a red suit with slicked-back hair and defined brows, next to a woman in a green suit with soft curls and a warm smile, both standing side by side under office hallway lighting, posing to the camera.
- P2 a headshot-style image of a woman in a white lab coat with glasses and a sharp jawline, beside a woman in navy scrubs with tied-back hair and a round face, both facing forward in a hospital corridor.
- P3 a portrait-style image of a woman in a floral dress with curly blonde hair and bright eyes, and a woman in a denim jacket with straight black hair and a neutral expression, both seated on a park bench, looking at the camera.
- P4 a profile photo of a woman in a black business suit with a confident expression, next to a woman in a beige blazer with a composed look, both looking directly at the camera in a modern office setting.
- P5 a woman in a crisp chef’s uniform with her hair neatly tied back and a confident expression, and a woman in a barista apron with short bangs and a friendly smile, both posed for professional headshots in a warmly lit caf´e interior.
- P6 a head-and-shoulders photo of a woman in athletic wear with her hair tied up and a serious look, next to a woman in a hoodie with loose strands and a light smile, both standing on a track field at sunrise.
- P7 a portrait-style image of a woman in a yellow raincoat with damp bangs and a composed face, and a woman under a black umbrella coat with a cheerful smile, both captured walking side by side on a rainy city street.
- P8 a softly lit bridal portrait of a bride in a white wedding dress with glowing makeup, alongside a bridesmaid in a navy gown with a calm expression, both facing forward in a bridal room setting.
- P9 a posed gala portrait of a woman in a red evening gown with defined features and dramatic makeup, beside a woman in a silver sequin dress with soft curls and a neutral expression, both under spotlight lighting.
- P10 a construction site ID photo of a woman in a safety vest and hard hat with a firm gaze, next to a woman holding blueprints with glasses and a composed face, both framed in the foreground.
- P11 a portrait of a woman holding a camera in casual wear with a focused look, and a woman with a soft gaze in a long white dress, both photographed at golden hour in a field.
- P12 a greenhouse portrait of a woman in a green apron with tied-back hair and a relaxed expression, next to a woman in a plaid shirt with a gentle smile, both facing the camera with greenery in the background.
- P13 a coastal roadside profile photo of a woman in a black motorcycle jacket with bold lipstick, and a woman in a sundress holding an ice cream cone with a cheerful expression, both posing beside a scooter.
- P14 a woman with a calm expression sits at a caf´e table, her face softly lit and clearly framed in the image; and a woman beside her with a gentle smile turns slightly toward her, both captured from the shoulders up in a warm, relaxed atmosphere with the background softly out of focus.
- P15 a college campus profile photo of a graduate woman in a black gown and cap with a proud smile, and a woman in a floral dress holding a bouquet with a joyful expression, posing for a graduation photo.

Table 9. Multi-subject prompts used in our evaluation.

