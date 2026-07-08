#### Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation

Omer Dahary1 Or Patashnik1,2 Kfir Aberman2 Daniel Cohen-Or1,2 1Tel Aviv University 2Snap Research

# arXiv:2403.16990v1[cs.CV]25Mar2024

### “3 ginger kittens and 2 gray kittens…”

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

“… on the stairs” “… on the street” “… in a basket” “… on a tree”

Figure 1. Our method bounds the attention to enable layout control over a pre-trained text-to-image diffusion model. Bounded Attention effectively reduces the impact of the innate semantic leakage during denoising, encouraging each subject to be itself. Our method can faithfully generate challenging layouts featuring multiple similar subjects with different modifiers (e.g., ginger and gray kittens).

##### Abstract

##### 1. Introduction

Text-to-image diffusion models have an unprecedented ability to generate diverse and high-quality images. However, they often struggle to faithfully capture the intended semantics of complex input prompts that include multiple subjects. Recently, numerous layout-to-image extensions have been introduced to improve user control, aiming to localize subjects represented by specific tokens. Yet, these methods often produce semantically inaccurate images, especially when dealing with multiple semantically or visually similar subjects. In this work, we study and analyze the causes of these limitations. Our exploration reveals that the primary issue stems from inadvertent semantic leakage between subjects in the denoising process. This leakage is attributed to the diffusion model’s attention layers, which tend to blend the visual features of different subjects. To address these issues, we introduce Bounded Attention, a training-free method for bounding the information flow in the sampling process. Bounded Attention prevents detrimental leakage among subjects and enables guiding the generation to promote each subject’s individuality, even with complex multi-subject conditioning. Through extensive experimentation, we demonstrate that our method empowers the generation of multiple subjects that better align with given prompts and layouts.

In recent years, text-to-image generation has undergone a significant shift with the integration of conditional diffusion models [23, 26, 28, 29], allowing for the facile generation of high-quality and diverse images. The use of attention layers in architectures of such generative models has been identified as a major factor contributing to improved quality of generated images [15, 28]. However, these models struggle to accurately generate scenes containing multiple subjects, especially when they are semantically or visually similar.

In this work, we study the problem of multi-subject image generation in attention-based diffusion models. Our contribution is twofold. First, we recognize the underlying reasons for the difficulty in generating images containing multiple subjects, especially those sharing semantic similarities. Second, building on our insights, we present a method aimed at mitigating semantic leakage in the generated images, allowing control over the generation of multiple subjects (see Figure 1).

We demonstrate an innate bias within the common attention-based architectures utilized in diffusion models, which predisposes them to leak visual features between subjects. In particular, the functionality of attention layers is designed to blend features across the image. Therefore, they inherently lead to information leakage between subjects.

Catastrophic Incorrect Subject neglect attribute binding fusion

[Figure 5]

[Figure 6]

[Figure 7]

“A ginger kitten “A spotted lizard “A watercolor painting and a gray puppy” and a blue fruit” and a marble statue”

- Figure 2. Misalignment in layout-to-image generation include (i) catastrophic neglect [6] where the model fails to include one or more subjects mentioned in the prompt within the generated image, (ii) incorrect attribute binding [6, 27] where attributes are not correctly matched to their corresponding subjects, and (iii) subject fusion [39] where the model merges multiple subjects into a single, larger subject.

This phenomenon is particularly noticeable when subjects are semantically similar and, therefore, attend to each other (Figure 2).

A plethora of works tries to mitigate the cross-subject leakage issue, either by modifying the sampling process to better follow different subjects in the prompt [6, 11], or by coupling the global prompt with layout information via segmentation maps or bounding boxes labeled with subject classes or local prompts [7, 16, 17, 22]. However, the majority of these methods still encounter difficulties in accurately aligning to input layouts, particularly in scenarios involving two or more semantically similar subjects.

In our approach, we guide the image generation with a spatial layout [17, 22]. To address cross-subject leakage, we introduce the Bounded Attention mechanism, utilized during the denoising process to generate an image. This method bounds the influence of irrelevant visual and textual tokens on each pixel, which otherwise promotes leakage. By applying this mechanism, we encourage each subject to be itself, in the sense that it hinders the borrowing of features from other subjects in the scene. We show that bounding the attention is needed both in the cross- and selfattention layers. Moreover, we find additional architectural components that amplify leakage, modify their operation, and present remedies to them.

We show that our method succeeds in facilitating control over multiple subjects within complex, coarse-grained layouts comprising numerous bounding boxes with similar semantics. Particularly challenging examples are demonstrated in Figure 1, where we successfully generate five kittens with a mix of adjectives. We conduct experiments on both Stable Diffusion [28] and SDXL [23] architectures and demonstrate the advantage of our method compared to previous ones, both supervised and unsupervised.

##### 2. Related work

Text-to-image diffusion models. Diffusion models, trained on huge datasets [30], have demonstrated their power in learning the complex distributions of diverse natural images [3, 23, 26, 28, 29]. Augmenting attention layers into diffusion models and conditioning them on textual prompts by coupling them with visually-aligned text encoders [25] leads to powerful text-to-image models [14]. In this work, we specifically examine two such open-source text-to-image diffusion models: Stable Diffusion [28], and the more recent SDXL [23].

Semantic alignment in text-to-image synthesis. A critical drawback of current text-to-image models pertains to their limited ability to faithfully represent the precise semantics of input prompts. Various studies have identified common semantic failures and proposed mitigating strategies, such as adjusting text embeddings [11, 32, 34], or optimizing noisy signals to strengthen or align cross-attention maps [6, 27]. Nevertheless, these methods often fall short in generating multiple subjects, and do not adhere to positional semantics, such as subjects’ number or location.

Layout-guided image synthesis. Addressing the semantic alignment concern, alternative approaches advocate for conditioning the diffusion process on layout information, either by training new models from scratch [24, 40] or finetuning an existing one [2, 17, 37, 38]. Despite their promise, these methods demand extensive computational resources and prolonged training times. Moreover, they are constrained by the layout distribution of the training data and the models’ architectural bias to blend subject features, a limitation that our Bounded Attention aims to overcome.

To circumvent these challenges, numerous researchers explore training-free techniques, where the generation process itself is modified to enforce layout constraints. Several optimization-based works employ techniques similar to classifier-free guidance to localize the cross-attention [7, 9, 36, 39] and/or self-attention maps [22]. While effective in aligning random noise with the intended layout, guiding attention maps to distinct regions may inadvertently lead to undesired behavior, particularly when different subjects share similar semantics and visual features. Furthermore, these methods often exhibit a deteriorating effect on visual quality, thereby limiting their applicability to only the initial denoising steps and neglecting finer control over shape and visual details, which are determined only in later stages [21]. Bounded Attention addresses these shortcomings by regulating attention computation throughout the entire denoising process.

Another approach involves generating each subject separately in its own denoising process [4, 18]. While these

methods inherently address catastrophic neglect, they tend to generate disharmonious images, and remain susceptible to leakage when merging subjects in subsequent stages. In contrast, masking attention maps to input bounding boxes [12] or attenuating attention in specific segments [16] represents a milder variant of this strategy, aimed at avoiding visible stitching. However, these methods often fall short of fully mitigating subject leakage, compromising semantic alignment. In comparison, Bounded Attention is able to carefully govern information propagation among subjects in a single denoising process.

While both the trained models and training-free techniques aim to generate numerous objects, they do not mitigate the inherent leakage caused by attention mechanisms. Unlike our Bounded Attention technique, these methods encounter challenges in effectively generating and controlling a multitude of subjects, especially when they share semantic similarity. Notably, existing techniques struggle to accurately generate even two semantically similar subjects, whereas our method, as demonstrated succeeds in generating five and even more subjects.

##### 3. Preliminaries

Latent diffusion models. In this work, we examine Stable Diffusion [28] and SDXL [23], which are both publicly available latent diffusion models. These models operate in the latent space of a pretrained image autoencoder, and are thus tasked with denoising a latent representation of the image, where each latent pixel corresponds to a patch in the generated image. Starting from pure random noise zT, at each timestep t, the current noisy latent zt is passed to a denoising UNet ϵθ, trained to predict the current noise estimate ϵθ (zt,y,t) using the guiding prompt encoding y.

Attention layers. At each block, the UNet utilizes residual convolution layers, producing intermediate features ϕ(l)(zt), where l denotes the layer’s index. These, in turn, are passed to attention layers, which essentially average different values Vt(l) according to pixel-specific weights:

ϕ(l+1)(zt) = A(tl)Vt(l), where (1) A(tl) = softmax Q(tl)K(tl)

⊺

. (2)

Here, the keys K(tl) = fK(l) C(tl) and values Vt(l) =

fV(l) C(tl) are linear projections of context vectors C(tl). In the cross-attention layers we inject semantic context from

the prompt encoding C(tl) ≡ y, while self-attention layers utilize global information from the latent itself C(tl) ≡ ϕ(l)(zt).

The weighting scheme is determined by the attention map A(tl) which represents the probability of each pixel

to semantically associate with a key-value pair. This is done by linearly projecting the latent noisy pixels to queries

Q(tl) = fQ(l) ϕ(l)(zt) and computing their inner product with the keys. It has been widely demonstrated that the

cross-attention maps are highly indicative of the semantic association between the image layout and the prompt tokens [13]. Meanwhile, the self-attention maps govern the correspondence between pixels, and thus form the image’s structure [31].

##### 4. Semantic Leakage

We begin by studying the causes of semantic leakage in Stable Diffusion [28], and examine the limitations of existing layout-to-image approaches.

###### 4.1. On Subject Similarity

Figure 2 illustrates various misalignment failures observed in state-of-the-art layout-to-image training-free methods. As we shall show, these failures are more prevalent for subjects that share semantic or visual similarity.

2 ∈ R2 be 2D latent coordinates corresponding to two semantically similar subjects s1,s2 in the generated image. Intuitively, we expect that along the denoising process, the queries corresponding to these pixels,

Let xs

,xs

1

Q(tl) [xs

],Q(tl) [xs

], will be similar and hence also their attention responses. This, in turn, also implies that they will share semantic information from the token embeddings through the cross-attention layers or visual information via the self-attention layers.

1

2

To explore this hypothesis, we investigate the model’s behavior when tasked with generating two subjects and analyze their attention features in both cross- and self-attention layers. Subsequently, we meticulously examine these features and demonstrate how their behavior sheds light on the leakage observed in generated images.

###### 4.2. Cross-Attention Leakage

To analyze the leakage caused by cross-attention layers, we examine the cross-attention queries. We depict these queries in the plots in Figure 3, where each point corresponds to a single query projected to 2D with PCA. To label each point with its corresponding subject, we compute the subject masks by averaging cross-attention maps [13] and color each projected query according to the subject’s text color. The leftmost plot, in which the two subjects were generated separately, serves as a reference point to the relation between the queries when there is no leakage. For comparative analysis, we also present results for Layout-guidance [7], as a simple representative of current training-free layout-guided approaches, and Bounded Attention, which we shall cover in the sequel.

We consider the cross-attention queries in two examples:

A kitten A puppy A kitten and a puppy

[Figure 8]

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

A hamster A squirrel A hamster and a squirrel

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

SD SD LG BA

- Figure 3. Cross-Attention Leakage. We demonstrate the emergence of semantic leakage at the cross-attention layers. We show two examples: a puppy a kitten, and a hamster and a squirrel. In the two leftmost columns, the subjects were generated separately using Stable Diffusion (SD). In the right three columns, we generate a single image with the two subjects using three different methods: Stable Diffusion (SD), Layout Guidance (LG), and Bounded Attention (BA, ours). Under each row, we plot the two first principal components of the cross-attention queries. As can be seen, the separation of the queries (blue and red) reflects the leakage between the subjects in the generated images.

“a kitten” and “a puppy”, and “a hamster” and a “squirrel”. As can be seen in the reference plots, the kitten and puppy queries share some small overlap, and the hamster and squirrel queries are mixed together. The level of separation between the red and blue dots in the plots reveals the semantic similarity of the two forming subjects.

Clearly, vanilla Stable Diffusion struggles to adequately generate the two subjects within the same image. This is apparent by the visual leakage between the two subjects, as the model cannot avoid averaging their distinct visual properties. For example, the puppy has the visual features of a kitten, like raised ears and a triangular nose, while the squirrel loses its distinct ear shape and its feet are now pink like the hamster. Respectively, it can be seen that the queries of the two subjects are mixed, even for the kitten and the puppy which are more separated in the reference plot.

Meanwhile, Layout Guidance (LG), which optimizes zt to have attention peaks for each noun token at their corresponding region, exhibits interesting results. Its optimization objective implicitly encourages the separation between subjects’ cross-attention queries. This can have positive effects, like the hamster and squirrel having unique colors, but

at the same time yields unwanted artifacts, like the puppy losing its face altogether. Moreover, it can inadvertently push the latent signal out of distribution, causing quality degradation, as evident by the hamster’s and squirrel’s cartoonish texture. When it overly pushes the latent out of distribution, it leads to the catastrophic neglect phenomenon (Figure 2).

In comparison, when examining the plots of our method alongside the reference plots, our approach preserves the feature distribution of the subjects’ queries, and successfully generates the two subjects, even when the queries are as mixed as in the hamster and the squirrel.

The above analysis yields two immediate conclusions: (i) Semantic similarity between subjects is reflected by their queries proximity, and leads to mixed queries when the subjects are generated together. This in turn leads to leakage between the subjects in the generated images, and (ii) enforcing semantic separation by modifying the semantic meaning of the cross-attention queries is harmful. The former observation represents a crucial architectural limitation in current diffusion models, and the latter pinpoints to a previously unexplored weakness in the widely used latent optimization methodology. Bounded Attention is designed to overcome these limitations.

###### 4.3. Self-Attention Leakage

We now turn to analyze the leakage caused by self-attention layers. It has been shown that self-attention features exhibit dense correspondences within the same subject [5] and across semantic similar ones [1]. Hence, they are suspected as another source of leakage, that we shall study next.

Here, we choose to examine the self-attention maps as a means to understand the leakage. In Figure 4 we focus on representative pixels (marked in yellow) associated with the subjects’ eyes and legs, where visual leakage is most pronounced. As expected, features from one subject’s eye or leg attend to the semantic similar body parts of the other. As a result, the features of each of the yellow points are directly affected by the features of the counterpart subject, causing leakage. In both images, the crab and the frog have similar appearances. In vanilla SD, both have a crab-like color and limbs with frog’s toe pads. In LG, both have froglike eyes and crab legs.

Notably, this tendency to rely on similar patches aids the model in denoising the latent signal and is essential for achieving coherent images with properly blended subjects and backgrounds. Nevertheless, it has the drawback of leaking features between disjointed subjects. Therefore, completely disjointing the visual features during denoising [4], or naively pushing self-attention queries apart through optimization [22], can lead to subpar results. Consequently, we introduce the Bounded Attention mechanism to mitigate leakage and guide the latent signal towards subject separa-

SD LG

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

- Figure 4. Self-Attention Leakage. We demonstrate the emergence of semantic leakage at the self-attention maps of two subjects: a crab and a frog. The images are generated by Stable Diffusion (SD) and Layout-guidance (LG). The top row highlights specific pixels, such as those of a subject’s eye or leg, while the bottom row present their respective self-attention maps.

A kitten A puppy 64 × 64 32 × 32 16 × 16

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

A lizard A fruit 64 × 64 32 × 32 16 × 16

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

- Figure 5. We generate different subjects, and plot the first two principal components of the cross-attention queries at different layers of the UNet, where each layer is of different resolution. The high semantic similarity between the kitten and the puppy is expressed by the proximity of their queries through all layers. Meanwhile, the lizard and fruit share similar texture, and hence only their high-resolution queries are entangled.

bility, while avoiding detrimental artifacts.

It’s important to highlight that the leakage resulting from both cross- and self-attention layers is intertwined and mutually reinforced. Therefore, addressing the leakage caused by only one of these layers is insufficient to prevent the leakage in the generated images.

###### 4.4. Levels of Similarity

In the two previous sections, our focus was primarily on subjects that share semantic similarity. Building upon the observation that the UNet’s inner layers dictate the subject’s semantics and shape, while the outer layers control its style and appearance [33], we analyzed the inner UNet layers. However, leakage can also occur prominently when generating subjects that share visual similarity rather than semantic. We now turn to explore this scenario and demonstrate that, in such cases, the leakage originates from the UNet’s outer layers.

In Figure 5, we visualize cross-attention queries at dif-

ferent decoder’s layers, when generating the kitten, puppy, lizard, and fruit in isolation. As can be seen, the queries of the kitten and the puppy are mixed across all UNet’s layers, aligning with the visual and semantic similarity between these animals. On the other hand, the queries of the lizard and the fruit are overlapped only in the highest-resolution layer, aligning with the lack of semantic similarity between them. A closer look reveals that the lizard and the fruit share surprisingly similar textures, which explains the overlap of the queries in the highest-resolution layer. As explained in the previous sections, this overlap causes leakage between the two subjects, in this case, a visual rather than a semantic leakage (see Figure 2).

##### 5. Bounded Attention

Our method takes as input n distinct textual subjects S = {si}ni=1 contained within a global prompt y, along with their corresponding bounding boxes B = {bi}ni=1. Our objective is to condition the generation process on y, S, and B, while preserving the intended semantics of each subject, all without requiring any training or fine-tuning.

Figure 6 illustrates an overview of our method. There, the input prompt y is “A kitten and a puppy”, S = {“kitten”,“puppy”}, and the two corresponding bounding boxes {b1,b2} are illustrated in the top left corner.

Bounded Attention operates in two modes: Bounded Guidance and Bounded Denoising. Specifically, at the beginning of the denoising process, for t ∈ [T,Tguidance], we perform a Bounded Guidance step followed by a Bounded Denoising step. In the guidance step, we utilize the Bounded Guidance loss. This interval of timesteps constitutes the optimization phase. Then, for t ∈ [Tguidance,0], we apply only Bounded Denoising steps.

In both modes, we manipulate the model’s forward pass by adopting an augmented weighting scheme in the attention layers, that safeguard the flow of information between the queries and the keys:

⊺

A(tl) = softmax Q(tl)K(tl)

+ Mt , (3) where l represents the layer index, t represents the diffusion timestep, and Q(tl),K(tl) are the queries and keys of the l-th attention layer. Mt represents time-specific masks composed of zeros and −∞ elements. We refer to A(tl) as the Bounded Attention map.

When indexing A(tl), we use pixel coordinates x for rows, and attention-type-specific context vectors c for columns. In locations [x,c], where Mt [x,c] = −∞, it holds that A(tl) [x,c] = 0. Therefore, these masks prevent harmful information flow between pixels in self-attention layers, and between pixels and token embeddings in crossattention layers.

|𝑧𝑡𝑜𝑝𝑡 = 𝑧𝑡 − 𝛽∇𝑧𝑡Σ𝑖ℒ𝑖2|
|---|

Inputs

|| |
|---|
<br><br>| |
|---|
|
|---|

[Figure 52]

[Figure 53]

[Figure 54]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|𝑧𝑡|
|---|

|𝑧𝑡−1|
|---|

|𝑧𝑡𝑜𝑝𝑡|
|---|

“A kitten and a puppy”

Guidance Step Denoising Step

|𝑄𝑐𝑎<br><br>| | | | | | | |
|---|---|---|---|---|---|---|
| | |𝐾𝑐𝑎| | | | |
<br><br>|=|
|---|
<br><br>𝑄𝑠𝑎 𝐾𝑠𝑎<br><br>ℒ𝑖 = 1 −<br><br>|=|
|---|
<br><br>𝐴𝑐𝑎( )<br><br>𝐴𝑠𝑎( )<br><br>|
|---|

|𝐴𝑠𝑎 =<br><br>𝑄𝑐<br><br>|𝑎<br><br>𝑎 𝐾𝑠𝑎<br><br>|
|---|---|
|𝑄𝑠| |
| | |

𝐴𝑐𝑎 =

𝐾𝑐𝑎

Bounded Guidance Bounded Denoising

- Figure 6. Bounded Attention operates in two modes: guidance and denoising. In each mode, strict constraints are imposed to bound the attention of each subject solely to itself and, possibly, to the background, thereby preventing any influence from other subjects’ features. In guidance mode, we minimize a loss that encourages each subject’s attention to concentrate within its corresponding bounding box. When calculating this loss, we mask the other bounding boxes, to prevent artifacts as shown in Figure 3. We simplify the visualization of the loss by setting α = 1 in Equation 4. During the denoising step, we confine the attention of each subject solely to its bounding box, along with the background in the self-attention. This strategy effectively prevents feature leakage while maintaining the natural immersion of the subject within the image. To demonstrate each of these modes, we show the attention map of a specific key, marked with ⋆. For the cross-attention map, we show the key corresponding to the “kitten” token, and for the self-attention map, we show a key that lies in the kitten’s target bounding box.

###### 5.1. Bounded Guidance

In Bounded Guidance, we backpropagate through the diffusion model to steer the latent signal toward the desired layout, using Gradient Descent. Our Bounded Guidance loss encourages the Bounded Attention map of each key corresponding to subject si, to be within the bi bounding box. To this end, for each subject key we consider the ratio between the attention within the corresponding bounding box, to the entire Bounded Attention map (see Figure 6).

Formally, we aggregate the following loss on the different subjects:

###### Aˆ [x,c]

x∈bi, c∈Ci

, (4)

Li = 1 −

Aˆ [x,c] + α

Aˆ [x,c]

x∈bi, c∈Ci

x∈/bi, c∈Ci

where i denotes the index of subject si, Aˆ is the mean Bounded Attention map, averaged across heads and layers, and α is a hyperparameter that magnifies the significance of disregarding attention towards the background, as we explain later. Similarly to the above, When indexing Aˆ , pixel coordinates x represent rows, and attention-typespecific context vectors c represent columns. We designate

Ci as the set of all si-related context vectors, i.e., pixel coordinates in bi for self-attention layers, and the token identifiers of si in cross-attention layers. Additionally, for crossattention layers, we include the first padding token [EoT] in Ci to enhance layout alignment [39].

For each subject si, the mask Mt should block the influence of opposing keys (tokens in sj and pixels in bj for j ̸= i), to avoid the artifacts illustrated in Figure 3. This fosters that the queries of different subjects, including similar ones, are not erroneously forced to be far apart. Utilizing this loss, our Bounded Guidance step is defined as

ztopt = zt − β∇zt i L2i. (5)

Integrating this loss within the cross-attention layers encourages the localization of each subject’s semantics within its bounding boxes [7]. However, as cross-attention responses tend to peak around more typical patches associated with the subject’s semantics (e.g., the face of a human, the legs of a crab, etc.), it may lack control over the subject’s boundaries. By applying the loss within the selfattention layers, we encourage each subject to establish its own boundaries close to its bounding box, thereby discouraging subject fusion (see Figure 2).

In the computation of the loss, we also introduce a hyper-

Cluster SA

|| |
|---|
<br><br>| |
|---|
| |
|---|---|
| | |

| |
|---|

[Figure 55]

| |
|---|

| |
|---|

“A kitten and a puppy”

“A kitten and a puppy”

[Figure 56]

[Figure 57]

|𝑧𝑡|
|---|

|𝑧𝑡−1|
|---|

Bounded Attention Denoising Step

- Figure 7. In the first phase, the coarse shapes of the subjects are formed. Then, we move to the next phase, illustrated here. During this phase, we apply Bounded Denoising steps using the finegrained subject masks. At regular intervals, we refine these masks by clustering the self-attention (SA) maps.

parameter α to reinforce attention to the background. This adjustment aids in preventing subject amalgamation, where a redundant subject is realized from different subject semantics in the background.

To preserve image quality, we limit the application of this mode to an initial time interval [T,Tguidance], following similar works [6, 7].

- 5.2. Bounded Denoising

In Bounded Denoising, we compute the model’s output and use it as the next latent in the series. Here, the masks aim to reduce semantic leakage between subjects, as detailed in Section 4, and to prevent unintended semantics from leaking to the background. Unlike Bounded Guidance and typical attention-based guidance approaches, Bounded Denoising can be applied throughout all time steps to mitigate leaks in fine details, which emerge only in later stages [19].

However, coarse masking in later stages may degrade image quality and result in noticeable stitching. To address this, after the optimization phase, for t ∈ [Tguidance,0], we replace each bounding box with a fine segmentation mask obtained by clustering the self-attention maps [21] (see Figure 7). Since the subject outlines are roughly determined in the initial time steps and evolve gradually thereafter [21], we refine the masks periodically. The resulting Bounded Attention maps of cross-attention layers are visualized in

- Figure 8. Notably, this mechanism also addresses imperfect align-

ments between subjects and bounding boxes after the guidance phase, which are more common when generating numerous subjects. Thus, employing this technique enhances the robustness of our method to seed selection, ensuring proper semantics even when subjects extend beyond their initial confines (see Figures 1,10). Compared to methods that necessitate strict input masks yet remain susceptible to leakage [4, 8, 16], our method offers greater user control with simpler inputs and more satisfactory outcomes.

Generated Image Bounded Self-Attention Map

[Figure 58]

[Figure 59]

Bounded Cross-Attention Maps

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

black asian black asian

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

woman man man woman

Figure 8. By segmenting the average self-attention maps from the UNet’s bottleneck (top-right corner), we can refine the input bounding boxes into dense segmentation masks, even amidst highmagnitude noise. Bounded Attention utilizes these masks to align the evolving image structure (represented by the self-attention maps) with its semantics (reflected by the cross-attention maps), ensuring each subject retains its distinct characteristics during denoising.

Method Details. Further details on the adaptation of Bounded Attention to the cross- and self-attention layers, along with a description of the subject mask refinement process, are provided in Appendix A.

##### 6. Experiments

In this section, we conduct both qualitative and quantitative experiments to assess the effectiveness of our Bounded Attention method. We compare our approach with three training-free baseline methods: Layout-guidance (LG) [7], BoxDiff (BD) [36], and MultiDiffusion (MD) [4]. Additionally, we include comparisons with GLIGEN [17] and ReCo [37], which necessitate training. Since Attentionrefocusing (AR) [22] is based on GLIGEN, we categorize it as a trained method for the purpose of our evaluation. For fairness, when comparing our method with other methods, we use Stable Diffusion.

###### 6.1. Qualitative Results

SDXL results. We begin our experiments by showcasing the effectiveness of our method in challenging scenarios, particularly when tasked with generating multiple semantically similar subjects using SDXL.

In Figure 9, we demonstrate that Bounded Attention is capable of generating subjects with complex positional rela-

###### “a [base] with [cake] and [icing] and [toppings] on a table”

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

“cake stand”, “chocolate cake”, “cutting board”, “cheese cake”, “ceramic plate”, “sponge cake”, “whipped cream”, “strawberries” “vanilla frosting”, “blueberries” “caramel frosting”, “raspberries”

“a golden retriever and a german shepherd and a boston terrier and an english bulldog and a border collie in a pool”

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

- Figure 9. Bounded Attention in SDXL enables precise control over multiple subjects, producing natural images where each subject seamlessly integrates into the scene while retaining its distinct features.

tions and occlusions, such as stacked cake layers, as well as visually similar subjects that naturally blend into the background, like dogs of various breeds partially submerged in a pool. Our approach produces all subjects with their distinctive characteristics, even when altering the prompt, seed, or bounding box assignment.

Moving to Figure 10, we generate multiple semantically similar subjects with different modifiers across vari-

- ous seeds. It is evident that Vanilla SDXL fails to follow the prompts due to semantic leakage. In the first row, it inaccurately generates the number of dogs and kittens and mixes between their colors. In the middle row, the clothing compositions combine fabrics, silhouettes, and colors mentioned in the prompt. Finally, in the last row, it merges the appearance of the subjects while leaking the pink attribute into the background.

We provide more results in Appendix B.

Non-curated results. Next, we conduct a non-curated comparison with the training-free baseline methods and present the results in Figure 11. We showcase the initial six images sampled from seed 0 for each method. We anticipate that in the absence of Bounded Attention, semantic leakage may freely blend subject features, hindering the intended layout’s formation.

It is evident from the results that none of the competing methods is able to consistently construct the input lay-

- out. Layout Guidance [7] frequently neglects one of the subjects, and even when it generates three subjects, it struggles to avoid leakage, resulting in puppies with kitten-like features or incorrect color assignments. BoxDiff [36] often

generates the correct number of subjects but suffers from artifacts in the form of blobs. Similar to Layout Guidance, it encounters difficulties in properly constructing the puppy. Surprisingly, even MultiDiffusion [4], which generates the subjects separately, faces challenges in generating them all, with some disappearing or merging together in its bootstrapping phase.

In contrast, our method consistently outperforms these approaches, producing three subjects that align with the both prompt and layout in all six images.

Comparisons with baselines. We present a qualitative comparison in Figure 12. All competing methods, including those trained specifically for the layout-to-image task, exhibit significant visual and semantic leakage. The trainingfree methods perform the worst: MultiDiffusion produces disharmonious, low-quality images, while optimizationbased methods often result in object fusion, combining different semantics without adhering to the layout.

The training-based approaches closely follow the layout but fail to convery the correct semantics. In the first row, they neglect the corduroy jacket, leaking the denim texture into the other jacket, or even fusing them together. In the other rows, the elephant’s or penguin’s features leak into the rhino or bear, respectively. Moreover, due to the rigidity of these approaches, stemming from being trained on perfect bounding boxes, they are unable to depict the penguin riding the bear.

In comparison, our method generates images that align with the input layout and prompt, ensuring each subject retains its unique attributes, semantics, and appearance.

”A gray kitten and a ginger kitten and a black kitten and a white dog and a brown dog on a bed”

[Figure 80]

[Figure 81]

[Figure 82]

VanillaSDXLBoundedAttention

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

“A realistic photo of a window dressing with three mannequins wearing a blue velvet dress and a pink tulle gown and a brown fur coat.”

[Figure 87]

[Figure 88]

[Figure 89]

VanillaSDXLBoundedAttention

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

“3D Pixar animation of a cute unicorn and a pink hedgehog and a nerdy owl traveling in a magical forest.”

[Figure 94]

[Figure 95]

[Figure 96]

VanillaSDXLBoundedAttention

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

- Figure 10. Bounded Attention in SDXL promotes faithful generation of all subjects and their respective modifiers.

###### 6.2. Quantitative Results

Dataset evaluation. We evaluate our method’s effectiveness using the DrawBench dataset [29], known for its challenging prompts designed to test a model’s ability to compose multiple subjects with specific quantities and relations. We use the evaluation procedure from previous work [22, 35].

Our results, alongside those of other training-free methods, are summarized in Table 1. Unlike other approaches that do not account for semantic leakage, our method demonstrates notable improvements in the counting category. While other methods struggle to surpass the recall rates of vanilla SD, Bounded Attention enhances recall by 0.1, representing a noteworthy advancement. Moreover, it improves counting precision and spatial accuracy, highlighting the effectiveness of Bounded Attention in addressing semantic misalignments.

Method Counting Spatial Precision Recall F1 Accuracy

Stable Diffusion [28] 0.74 0.78 0.73 0.19 Layout-guidance [7] 0.72 0.78 0.72 0.35 BoxDiff [36] 0.81 0.78 0.76 0.28 MultiDiffusion [4] 0.70 0.55 0.57 0.15 Ours 0.83 0.88 0.82 0.36

Table 1. Quantitative evaluation on the DrawBench dataset.

LG [7] BD [36] MD [4] Our score vs. 0.85 0.72 0.95

Table 2. User study results.

User study. The automatic metrics utilized in Table 1 have limitations in capturing semantic leakage, as they rely on an object detector [35] trained on real images, where such erroneous leakage between subjects is absent. To address this issue, we conducted a user study.

For a comprehensive evaluation, we enlisted ChatGPT to provide us with five pairs of visually similar yet distinctly recognizable animals, along with a suitable background prompt and four different layouts. Subsequently, for each subject pair and its corresponding background prompt, we generated six non-curated images for each layout using our training-free baseline. For each prompt-layout condition, users were presented with an image set comprising the six images generated by our method and six images produced by a competing method. Users were then tasked with selecting realistic images that accurately depicted the prompt and layout, devoid of any leakage between subjects.

In total, responses were collected from 32 users, resulting in 330 responses. The results of the user study are summarized in Table 2, where we report the estimated conditional probability of a selected image being generated by our method compared to corresponding competing methods.

###### 6.3. Ablation Studies

To assess the significance of each component, we conduct an ablation study where we systematically vary our method’s configurations by omitting one component in each setting. Two examples, generated with SDXL (top row) and SD (bottom row), are illustrated in Figure 13.

Guidance is crucial for aligning the latent signal with the intended layout. However, attempting to guide the latent signal without our Bounded Guidance mechanism leads to subpar results, as seen in the partial alignment of the lizard with its bounding box, and the puppy’s distorted form. The issue arises from the inherent query entanglement between the two semantically similar subjects in each exam-

“A gray kitten and a ginger kitten and black puppy in a yard.”

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

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

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Bounded Attention Layout Guidance BoxDiff MultiDiffusion Figure 11. Comparison of the first six images generated from the seed 0.

“A denim jacket and a corduroy jacket and a leather handbag in a closet.”

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

“A penguin riding a white bear in the north pole.”

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

“A big red elephant and a far away rhino in a jungle.”

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

BA (ours) LG [7] BD [36] MD [4] GLIGEN [17] AR [22] ReCo [37]

- Figure 12. Qualitative comparison of our method with baseline methods: Above each row, we display the input prompt, where each subject’s color matches the color of its corresponding bounding box. We compare with both training-free methods (2nd-4th columns) and trained models (5th-7th columns). See Appendix B for more results.

ple. Without Bounded Guidance, the optimization in the top row reaches a plateau, where aligning the lizard with its bounding box reduces its loss but also increases the turtle’s. In the bottom row, the optimization pushes the two subject queries away from each other, creating artifacts.

Meanwhile, forgoing Bounded Denoising results in noticeable semantic leakage. In the top example, the lizard is replaced by a turtle, with the “red” attribute erroneously

leaking to the wrong subject. Similarly, in the bottom example, the puppy is replaced with a kitten.

Lastly, incorporating mask refinement in the later stages preserves fine details and prevents them from leaking. Without mask refinement, the kitten’s legs lose the details of their ginger fur texture, the turtle’s facial features resemble those of a lizard, and the lizard exhibits shell-like contours on its back.

“A red lizard and a turtle on the grass.”

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

SDXL

StableDiffusion

“A ginger kitten and a gray puppy on the front stairs.”

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

w/o G w/o BG w/o BD w/o MR Full Method

- Figure 13. Qualitative ablation. We ablate our method by skipping the guidance step (G), performing guidance without Bounded Guidance (BG), not applying Bounded Denoising (BD), and not performing Mask Refinement (MR). We show zoomed-in images of the two rightmost configurations.

##### 7. Conclusions

We introduce Bounded Attention, a technique designed to regulate the accurate generation of multiple subjects within an image. This approach encourages each subject to “be yourself”, emphasizing the importance of preserving individuality and uniqueness without being excessively influenced by other subjects present in the image. Our development of the Bounded Attention technique stemmed from an in-depth analysis of the root causes behind the misalignment observed between the provided prompt and the resulting generated image. Our investigation revealed that this misalignment primarily arises due to semantic leakage among the generated subjects, a phenomenon observed in both the cross and self-attention layers.

While Bounded Attention effectively mitigates a significant portion of semantic leakage, it does not entirely eliminate it. Our findings demonstrate a marked improvement in performance compared to other methods that seek to achieve semantic alignment. However, residual leakage persists, which we attribute to imperfect optimization during the guidance mode and inaccurate segmentation of the subject prior to the second phase.

While Bounded Attention excels in generating multiple subjects with plausible semantic alignment, its performance may vary across different layouts. Achieving success with Bounded Attention hinges on a strong match between the seed and the layout. Moving forward, we aim to explore methods for generating well-suited seeds tailored to specific layouts. One potential avenue is to introduce noise to the layout image, thereby creating a seed that aligns more closely with the desired outcomes.

##### Acknowledgement

We thank Guy Tevet for fruitful discussions and useful suggestions. This work is partly supported by a research gift from Snap Inc.

##### References

- [1] Yuval Alaluf, Daniel Garibi, Or Patashnik, Hadar AverbuchElor, and Daniel Cohen-Or. Cross-image attention for zeroshot appearance transfer. arXiv preprint arXiv:2311.03335,

2023. 4

- [2] Omri Avrahami, Thomas Hayes, Oran Gafni, Sonal Gupta, Yaniv Taigman, Devi Parikh, Dani Lischinski, Ohad Fried, and Xi Yin. Spatext: Spatio-textual representation for controllable image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18370–18380, 2023. 2
- [3] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 2
- [4] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. 2023. 2, 4, 7, 8, 9, 10, 16
- [5] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. arXiv preprint arXiv:2304.08465, 2023. 4
- [6] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–10, 2023. 2, 7
- [7] Minghao Chen, Iro Laina, and Andrea Vedaldi. Training-free

- layout control with cross-attention guidance. arXiv preprint arXiv:2304.03373, 2023. 2, 3, 6, 7, 8, 9, 10, 14, 16
- [8] Guillaume Couairon, Marlene Careil, Matthieu Cord, St´ephane Lathuiliere, and Jakob Verbeek. Zero-shot spatial layout conditioning for text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2174–2183, 2023. 7
- [9] Yuki Endo. Masked-attention diffusion guidance for spatially controlling text-to-image generation. arXiv preprint arXiv:2308.06027, 2023. 2
- [10] Dave Epstein, Allan Jabri, Ben Poole, Alexei Efros, and Aleksander Holynski. Diffusion self-guidance for controllable image generation. Advances in Neural Information Processing Systems, 36, 2024. 14
- [11] Weixi Feng, Xuehai He, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Pradyumna Narayana, Sugato Basu, Xin Eric Wang, and William Yang Wang. Training-free structured diffusion guidance for compositional text-to-image synthesis. arXiv preprint arXiv:2212.05032, 2022. 2
- [12] Yutong He, Ruslan Salakhutdinov, and J Zico Kolter. Localized text-to-image generation for free via cross attention control. arXiv preprint arXiv:2306.14636, 2023. 3
- [13] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 3
- [14] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 2
- [15] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 1
- [16] Yunji Kim, Jiyoung Lee, Jin-Hwa Kim, Jung-Woo Ha, and Jun-Yan Zhu. Dense text-to-image generation with attention modulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7701–7711, 2023. 2, 3, 7
- [17] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22511–22521, 2023. 2, 7, 10, 16
- [18] Long Lian, Boyi Li, Adam Yala, and Trevor Darrell. Llmgrounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models. arXiv preprint arXiv:2305.13655, 2023. 2
- [19] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 7
- [20] Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. Teaching clip to count to ten. arXiv preprint arXiv:2302.12066, 2023. 14
- [21] Or Patashnik, Daniel Garibi, Idan Azuri, Hadar AverbuchElor, and Daniel Cohen-Or. Localizing object-level shape

- variations with text-to-image diffusion models. arXiv preprint arXiv:2303.11306, 2023. 2, 7, 14
- [22] Quynh Phung, Songwei Ge, and Jia-Bin Huang. Grounded text-to-image synthesis with attention refocusing. arXiv preprint arXiv:2306.05427, 2023. 2, 4, 7, 9, 10, 16
- [23] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint

- arXiv:2307.01952, 2023. 1, 2, 3

[24] Leigang Qu, Shengqiong Wu, Hao Fei, Liqiang Nie, and Tat-Seng Chua. Layoutllm-t2i: Eliciting layout guidance from llm for text-to-image generation. arXiv preprint

- arXiv:2308.05095, 2023. 2

- [25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2
- [26] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 1, 2

- [27] Royi Rassin, Eran Hirsch, Daniel Glickman, Shauli Ravfogel, Yoav Goldberg, and Gal Chechik. Linguistic binding in diffusion models: Enhancing attribute correspondence through attention map alignment. arXiv preprint arXiv:2306.08877, 2023. 2
- [28] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2, 3, 9
- [29] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 1, 2, 9
- [30] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models, 2022. 2
- [31] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1921–1930, 2023. 3
- [32] Hazarapet Tunanyan, Dejia Xu, Shant Navasardyan, Zhangyang Wang, and Humphrey Shi. Multi-concept t2izero: Tweaking only the text embeddings and nothing else. arXiv preprint arXiv:2310.07419, 2023. 2, 14
- [33] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. p+: Extended textual conditioning in text-to-

- image generation. arXiv preprint arXiv:2303.09522, 2023. 5
- [34] Jennifer C White and Ryan Cotterell. Schr\”{o} dinger’s bat: Diffusion models sometimes generate polysemous words in superposition. arXiv preprint arXiv:2211.13095,

2022. 2

- [35] Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2. https://github. com/facebookresearch/detectron2, 2019. 9
- [36] Jinheng Xie, Yuexiang Li, Yawen Huang, Haozhe Liu, Wentian Zhang, Yefeng Zheng, and Mike Zheng Shou. Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7452–7461, 2023. 2, 7, 8, 9, 10, 16
- [37] Zhengyuan Yang, Jianfeng Wang, Zhe Gan, Linjie Li, Kevin Lin, Chenfei Wu, Nan Duan, Zicheng Liu, Ce Liu, Michael Zeng, et al. Reco: Region-controlled text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14246–14255,

2023. 2, 7, 10, 16

- [38] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 2
- [39] Peiang Zhao, Han Li, Ruiyang Jin, and S Kevin Zhou. Loco: Locally constrained training-free layout-to-image synthesis. arXiv preprint arXiv:2311.12342, 2023. 2, 6, 14
- [40] Guangcong Zheng, Xianpan Zhou, Xuewei Li, Zhongang Qi, Ying Shan, and Xi Li. Layoutdiffusion: Controllable diffusion model for layout-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22490–22499, 2023. 2

## Appendix

##### A. Method Details

###### A.1. Bounded Cross-Attention

In the following, we elaborate on our design choices regarding the implementation of Bounded Attention into crossattention layers.

First, we have observed that tokens in the prompt y that do not correspond to attributes of the subjects or the subjects themselves can unexpectedly result in leakage between subjects within the image.

Specifically, we found that conjunctions, positional relations, and numbers, carry semantics related to other subjects around them. Consequently, they can freely introduce conflicting features into the background and subjects, disrupting the intended layout. Given that vision-language models like CLIP struggle with encapsulating compositional concepts [20], such tokens are generally irrelevant for multi-subject generation, prompting us to exclude them after computing the token embeddings. We achieve this automatically using an off-the-shelf computational POS tagger. Moreover, padding tokens can significantly impact the resulting layout. Specifically, the first [EoT] token typically attends to the image’s foreground subjects in SD and can even generate subjects on its own [7, 39]. However, it remains crucial for subject generation, particularly those specified last in the prompt [32]. We also observe similar behavior in SDXL. To address this, we confine this token’s attention within the union of bounding boxes.

Second, our masking strategy in Bounded Attention for cross-attention layers exhibits slight variations between Bounded Guidance and Bounded Denoising. In Bounded Denoising, we confine each subject token in si to interact solely with its corresponding box pixels bi. Conversely, in Bounded Guidance we employ a similar strategy but allow all tokens to attend to the background. Thus, the two modes complement each other: Bounded Guidance refines the random latent signal to achieve a reasonable initial alignment with the input layout, while Bounded Denoising enforces the layout constraints.

###### A.2. Bounded Self-Attention

In self-attention layers, we use the same masking scheme for both Bounded Guidance and Bounded Denoising modes. Specifically, for each subject si, we exclude all keys corresponding to bj where j ̸= i. We found that enabling subjects to interact with the background is crucial for generating natural-looking images, where the subjects seamlessly blend into the scene. Subjects should attend to the background to maintain high visual quality, as overly restrictive masking can lead to degradation, especially for small bounding boxes. Moreover, the background should attend

to the subjects to integrate them into the environment and facilitate lighting effects such as shadows and reflections.

In contrast to Bounded Cross-Attention, we apply Bounded Self-Attention both to the prompt-conditioned noise estimation ϵθ (zt,y,t), and its unconditional counterpart ϵθ (zt,ϕ,t). We observed that excluding the unconditional noise estimation introduces significant disparities between the conditional and unconditional directions in classifier-free guidance, consequently leading to noticeable artifacts.

###### A.3. Subject Mask Refinement

To derive the segmentation masks in [Tguidance,0], we employ a technique akin to self-segmentation, previously introduced for localized image editing [21].

In self-segmentation, first, all self-attention maps are averaged across heads, layers and timesteps. Then, each pixel is associated with its corresponding averaged self-attention map, which serves as input for a clustering method, such as KMeans. Each cluster is labeled as belonging to the background or to a specific subject’s noun, by aggregating and comparing the corresponding cross-attention maps of these nouns within each cluster.

When adapting this technique for generation, we encounter the challenge of lacking the more reliable latestage attention maps. To address this, we observe that selfattention maps formed at the UNet’s middle block and first up-block are more robust to noise. Therefore, starting at Tguidance, we cluster the time-specific self-attention maps, averaging only at the mentioned layers.

For labeling, we initially compute cross-attention masks

Mcrossi for each subject si, by applying a soft threshold to the subject’s cross-attention maps [10],

Mcrossi = norm sigmoid s · norm A ˆ crossi − σnoun .

(6)

Here, Aˆ crossi is the mean Boudned Cross-Attention map of the subject’s last noun in the prompt, averaged across heads and layers, and norm denotes the L1 normalization operation. We use hyperparameters s,σnoun, where σnoun defines the soft threshold, and s controls the binarization sharpness.

Then, we calculate the Intersection over Minimum (IoM) between each self-attention cluster Cselfj and cross-attention mask Mcrossi ,

Mcrossi [x] · Cselfj [x]

IoM(i,j) = x

. (7)

min x Mcrossi [x], x Cselfj [x]

For each cluster Cselfj , we determine the subject index with the highest IoM, imax (j) = arg maxi IoM(i,j). We

“A classic guitar and an electric guitar and a cello and a violin on the wall of a music shop.”

[Figure 159]

[Figure 160]

Figure 14. Failure cases.

assign cluster Cselfj to subject si

max(j), if IoM(imax (j),j) ≥ σcluster, where σcluster is a hyperparameter.

We repeat this process at uniform time intervals to enable continuous evolution of the shapes. To maintain real-time performance, we execute KMeans directly on the GPU. For temporal consistency and faster convergence, we initialize the cluster centers with the centers computed at the previous interval.

- A.4. Implementation Details

In our experiments, we utilized 50 denoising steps with Tguidance = 0.7, conducting 5 Gradient Descent iterations for each Bounded Guidance step. In Eq. 4, we set α to the number of input subjects. The loss computation was performed using attention layers 12 to 19 in SD and 70 to 81 in SDXL.

For the step size β in Eq. 5, we initiated the optimization process with β ∈ [8,15] for SD and β ∈ [10,25] for SDXL, with higher values utilized for more challenging layouts. We employed a linear scheduler, concluding the optimization with β ∈ [2,5] for SD and β ∈ [5,10] for SDXL. Early stopping was applied if the average loss between subjects dropped to 0.2 or lower.

Subject mask refinement was conducted every 5 steps, computing the cross and self-attention masks from attention layers 14 to 19 in SD and 70 to 81 in SDXL. We set s = 10 and σnoun = 0.2 when computing the cross-attention masks in Eq. 6. Additionally, when running KMeans, the number of self-attention clusters was set to 2 to 3 times the number of input subjects, with a threshold of σcluster = 0.2.

- B. Additional Results

- B.1. Failure Cases

In Figure 14, we present two failure cases of our method. In these instances, we attempted to generate five similar stringed instruments within a challenging layout. Although each bounding box contains the correct instrument, the generated images do not accurately adhere to the prompt.

In the left image, the two guitars appear as paintings on the wall, deviating from the intended representation. Similarly, in the right image, redundant parts of instruments are generated at the margins of the bounding boxes, leading to inconsistencies in the output.

###### B.2. Qualitative Results

Comparisons with baselines. In Figure 15, we present additional qualitative comparison results. While all competing methods fail to accurately generate all subjects from the prompt, our method successfully preserves each subject’s intended semantics. For example, in the third row, none of the methods accurately generate the correct type of pasta in each dinnerware, and in the fourth row, none achieve the correct number of watercolor paintings.

SDXL results. We present additional results in SDXL in Figures 16 and 17. In these figures, Vanilla SDXL exhibits evident semantic leakage. For example, in Figure 16, it doesn’t accurately generate all subjects with their intended modifiers. In the top row, a turtle is depicted with octopus tentacles in the second-to-right column. In the middle row, the green modifier intended for the alien spills over into the background. Lastly, none of the images in the last row include orchids as intended.

“A cactus in a clay pot and a fern in a porcelain pot.”

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

“A cow and a horse eating hay in a farm.”

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

“A white plate with ravioli next to a blue bowl of gnocchi on a table.”

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

“A marble statue and two watercolor paintings in a room.”

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

“A shark chasing a dolphin in the ocean.”

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

BA (ours) LG [7] BD [36] MD [4] GLIGEN [17] AR [22] ReCo [37] Figure 15. More qualitative results of our method in comparison to baseline methods.

“A realistic photo of a turtle and a jellyfish and an octopus and a starfish in the ocean depths.”

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

VanillaSDXLBoundedAttention

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

“A science fiction movie poster with an astronaut and a robot and a green alien and a spaceship.”

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

VanillaSDXLBoundedAttention

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

“A porcelain pot with tulips and a metal can with orchids and a glass jar with sunflowers”

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

VanillaSDXLBoundedAttention

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

“A realistic photo of a highway with a tourist bus and a school bus and a fire engine.”

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

VanillaSDXLBoundedAttention

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

“A realistic photo of a highway with a semi trailer and a concrete mixer and a helicopter.”

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

VanillaSDXLBoundedAttention

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

“A realistic photo of a tool shed with a lawn mower and a bucket and a ladder and tools attached to the wall.”

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

VanillaSDXLBoundedAttention

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

