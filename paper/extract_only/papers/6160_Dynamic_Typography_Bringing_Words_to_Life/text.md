# arXiv:2404.11614v3[cs.CV]5Nov2024

## Dynamic Typography: Bringing Text to Life via Video Diffusion Prior

Zichen Liu∗,1 Yihao Meng∗,1 Hao Ouyang2 Yue Yu1 Bolin Zhao1 Daniel Cohen-Or†,3 Huamin Qu†,1

1 HKUST 2 Ant Group 3Tel-Aviv University

[Figure 1]

Figure 1. Given a letter and a text prompt that briefly describes the animation, our method automatically semantically reshapes a letter and animates it in vector format while maintaining legibility. Our approach allows for a variety of creative interpretations that can dynamically bring words to life. Our code is available at: https://animate-your-word.github.io/demo/.

### Abstract

### 1. Introduction

Text animation serves as an expressive medium, transforming static communication into dynamic experiences by infusing words with motion to evoke emotions, emphasize meanings, and construct compelling narratives. Crafting animations that are semantically aware poses significant challenges, demanding expertise in graphic design and animation. We present an automated text animation scheme, termed “Dynamic Typography”, which combines two challenging tasks. It deforms letters to convey semantic meaning and infuses them with vibrant movements based on user prompts. Our technique harnesses vector graphics and an end-to-end optimization-based framework. This framework employs neural displacement fields to convert letters into base shapes and applies per-frame motion, encouraging coherence with the intended textual concept. Perceptual loss regularization and shape preservation techniques are employed to maintain legibility and structural integrity throughout the animation process. We demonstrate the generalizability of our approach across various text-tovideo models and highlight the superiority of our end-toend methodology over baseline methods, which might comprise separate tasks. Through quantitative and qualitative evaluations, we demonstrate the effectiveness of our framework in generating coherent text animations that faithfully interpret user prompts while maintaining readability.

*Indicates Equal Contribution. †Indicates Corresponding Author.

Text animation is the art of bringing text to life through motion. By animating text to convey emotion, emphasize meaning, and create a dynamic narrative, text animation transforms static messages into vivid, interactive experiences [22, 24]. The fusion of motion and text, not only captivates viewers, but also deepens the message’s impact, making text animation prevail in movies, advertisements, website widgets, and online memes [55].

This paper introduces a specialized text animation scheme that focuses on animating individual letters within words. This animation is a compound task: The letters are deformed to embody their semantic meaning and then brought to life with vivid movements based on the user’s prompt. We refer to it as “Dynamic Typography”. For example, the letter “M” in “CAMEL” can be animated with the prompt “A camel walks steadily across the desert” in Fig. 1. This animation scheme opens up a new dimension of textual animation that enriches the reading experience.

However, crafting such detailed and prompt-aware animations is challenging. Traditional text animation methods demand considerable expertise in graphic design and animation [24], making them less accessible to non-experts. Our methodology aims to automate the text animation process to make it more accessible and efficient. Following prior research in font generation and stylization [18, 27, 53], we represent each input letter and every output frame as a vectorized, closed shape by a collection of B´ezier curves. This vector representation is resolution-

independent, keeping text clear at any scale and offering substantial editability, as users can easily adjust text appearance through control points. However, this shift to vector graphics introduces unique challenges in text animation. Most current video generation methods [6, 14, 35, 52, 56] fall short in this new scenario as they are designed within the rasterized pixel-based scenario instead of vectorized shapes, and are hard to render readable text. Although the most recent work, LiveSketch [13], introduces an approach to animate arbitrary vectorized sketches, it struggles to preserve legibility and consistency throughout animation when the input becomes vectorized letters, causing visually unpleasant artifacts including flickering and distortion.

To address these challenges, we design an end-to-end optimization-based framework that utilizes two neural displacement fields, represented in coordinates-based MLP. The first field deforms the original letter into the base shape, setting the stage for animation. Subsequently, the second neural displacement field learns the per-frame motion applied to the base shape. The two fields are jointly optimized using the score-distillation sampling (SDS) loss [41] to integrate motion priors from a pre-trained text-tovideo model [50] to encourage the animation to align with the intended textual concept. To preserve the legibility of the letter throughout the animation, we apply perceptual loss [59] as a form of regularization on the base shape to maintain a perceptual resemblance to the original letter. Additionally, to preserve the overall structure and appearance during animation, we introduce a novel shape preservation regularization based on the triangulation [16] of the base shape, which forces the deformation between the consecutive frames to adhere to the principle of being conformal with respect to the base shape.

Our approach is designed to be data-efficient, eliminating the need for additional data collection or the fine-tuning of large-scale models. Furthermore, our method generalizes well to various text-to-video models, enabling the incorporation of upcoming developments in this area. We quantitatively and qualitatively tested our text animation generation method against various baseline approaches, using a broad spectrum of prompts. The results demonstrate that the generated animation accurately and aesthetically interprets the input text prompt descriptions. Our method outperforms various baseline models in preserving legibility and promptvideo alignment. Overall, our framework demonstrates its efficacy in producing coherent text animations from user prompts while maintaining the readability of the text, which is achieved by the key design of the learnable base shape and associated shape preservation regularization.

### 2. Related Work

#### 2.1. Static Text Stylization

Text stylization enhances text aesthetics while maintaining readability, including artistic text style transfer and semantic typography. Artistic text style transfer migrates stylistic elements from source images onto text, typically using texture synthesis [9, 57] and GANs [2, 19, 30, 51]. Semantic typography combines semantic understanding with visual representation, creating visual forms that convey meaning. Notable works include Word-as-Image [18], which uses Score Distillation Sampling [41] with diffusion prior [44] to create meaningful letter deformations, and DS-Fusion [48], which employs latent diffusion to blend semantic-related styles into glyphs.

While these works produce only static images with limited semantic expression, our Dynamic Typography introduces motion to text, enhancing viewer engagement and aesthetic appeal [33].

#### 2.2. Dynamic Text Animation

Given animations’ effectiveness in capturing attention [5], research has explored dynamic text animations, particularly in dynamic style transfer and kinetic typography. Dynamic style transfer adapts visual style and motion from reference videos to text, with works like [31] transferring animations from dynamic text videos, and Yang et al. [58] introducing a scale-aware Shape-Matching GAN for diverse styles.

Kinetic typography [11] integrates motion with text to enhance messages. While traditionally labor-intensive, recent works [11, 12, 23, 34, 54] aim to automate this process. For instance, Wakey-Wakey [55] uses a motion transfer model [47] to animate text using meme GIFs.

However, these approaches require specific driven videos and are limited to simple motion patterns. In contrast, our method generates arbitrary motion patterns using only text prompts as input.

#### 2.3. Text and Image-to-Video Generation

Text-to-Video generation has advanced significantly with diffusion models. Several approaches extend Stable Diffusion (SD) [44] by incorporating temporal information, including AnimateDiff [14], LVDM [15], MagicVideo [60], VideoCrafter [6], and ModelScope [50]. Image-to-Video generation methods like DynamiCrafter [56], Motion-I2V [46], Gen-2 [45], Pika Labs [21], and SVD [4] generate videos from images and prompts.

However, existing open-source models fail to maintain text readability during motion. Training models for text animation requires large text animation datasets, which are scarce. One recent work LiveSketch [13] animates arbitrary vectorized sketches without extensive training. This work leverages the motion prior from pre-trained text-

[Figure 2]

- Figure 2. An overview of the framework. Given a letter represented as a set of control points, the Base Field deforms it to the shared base shape, setting the stage to add per-frame displacement. Then the base shape is duplicated across k frames, and the Motion Field predicts the displacement for each control point at each frame, infusing movement into the base shape. Each frame is rendered by the differentiable rasterizer R and concatenated as the output video. The base and motion field are jointly optimized by the video prior from frozen pre-trained video foundation model using Score Distillation Sampling LSDS, under regularization on legibility Llegibility and structure preservation Lstructure.

to-video diffusion model using score distillation sampling [41] to guide the motion of input sketches. However, when the input becomes vectorized letters, LiveSketch struggles to preserve legibility and consistency throughout the animation, leading to flickering and distortion artifacts that severely degrade video quality. Our method, in contrast, generates consistent, prompt-aware text animations while preserving readability.

3. Preliminary

- 3.1. Vector Representation and Fonts

Modern font formats like TrueType [40] and PostScript [1] utilize vector graphics to define glyph outlines. These outlines are typically collections of B´ezier or B-Spline curves, enabling flexible text rendering at any scale [18]. Our method outputs each animation frame in the same vector representation as our input.

[Figure 3]

- Figure 3. B´ezier curves representation of letter “B”. The endpoints are marked in orange, and the inner control points are in blue.

#### 3.2. Score Distillation Sampling

The objective of Score Distillation Sampling (SDS), originally introduced in the DreamFusion [41], is to leverage pre-trained diffusion models’ prior knowledge for the textconditioned generation of different modalities [20]. SDS optimizes the parameters θ of the parametric generator G (e.g., NeRF [32]), ensuring the output of G aligns well with the prompt. For illustration, assuming G is a parametric image generator. First, an image x = G(θ) is generated. Next, a noise image zτ(x) is obtained by adding a Gaussian noise ϵ at the diffusion process’s τ-th timestep:

##### zτ(x) = ατx + στϵ, (1)

where ατ, and στ are diffusion model’s noising schedule, and ϵ is a noise sample from the distribution N(0,1).

For a pre-trained diffusion model ϵϕ, the gradient of the SDS loss LSDS is formulated as:

∂x ∂θ

, (2)

∇ϕLSDS = w(τ)(ϵϕ(zτ(x);y,τ) − ϵ)

where y is the conditioning input to the diffusion model and w(τ) is a weighting function. The diffusion model predicts the noise added to the image x with ϵϕ(zτ(x);y,τ). The discrepancy between this prediction and the actual noise ϵ measures the difference between the input image and one that aligns with the text prompt. In this work, we adopt this strategy to extract the motion prior from the pre-trained text-to-video diffusion model [50].

In alignment with the setting outlined in Iluz et al. [18], we use the FreeType [7] font library to extract the outlines of the specified letter. Subsequently, these outlines are converted into a closed curve composed of several cubic B´ezier curves, as illustrated in Fig. 3, to achieve a coherent representation across different fonts and letters. We iteratively subdivide the letter’s B´ezier segments until reaching a pre-defined threshold to ensure sufficient control points to represent semantic deformation.

Since SDS is used within the pixel-based scenario, we utilize DiffVG [25] as a differentiable rasterizer. This allows us to convert our vector-defined content into pixel space in a differentiable way for applying the SDS loss.

### 4. Method

Dynamic Typography focuses on animating individual letters within words based on the user’s prompt. The letter is deformed to embody the word’s semantic meaning and then brought to life by infusing motion based on the user’s prompt. To achieve visually appealing animations, we identify three crucial requirements for Dynamic Typography:

- • Legibility Preservation. The deformed letter should remain legible in each frame during animation.
- • Semantic Alignment. The letter should be deformed and animated in a way that aligns with the semantic information in the text prompt.
- • Temporal Consistency. The deformed letter should move coherently while preserving a relatively consistent appearance in each animation frame.

Problem Formulation: The original input letter is initialized as a cubic B´ezier curves control points set (Fig. 3), denoted as Pletter = {pi}Ni=1 = {(xi,yi)}Ni=1 ∈ RN×2, where x,y refers to control points’ coordinates in SVG canvas, and N is the total number of control points of the indicated letter. The output video consists of k frames, each represented by a set of control points, denoted as V = {Pt}kt=1 ∈ Rk×N×2, where Pt is the control points for t-th frame.

Our goal is to learn the per-frame displacement to be added on the set of control point coordinates of the original letter’s outline. This displacement represents the motion of the control points over time, creating the animation that depicts the user’s prompt. We denote the displacement for t-th frame as ∆Pt = {∆pti}Ni=1 = {(∆xti,∆yit)}Ni=1 ∈ RN×2, where ∆pti refers to the displacement of the i-th control point in the t-th frame. The final video can be derived as V = {Pletter + ∆Pt}kt=1.

One straightforward strategy can be first deforming the static letter with existing method [18], then utilizing an animation model [13] designed for vector graphics composed of B´ezier curves to animate the deformed letter. However, this non-end-to-end formulation suffers from conflicting prior knowledge. The deformed letter generated by the first model may not align with the prior knowledge of the animation model. This mismatch can lead the animation model to alter the appearance of the deformed letter, leading to considerable visual artifacts including distortion and inconsistency, see Fig. 4.

Therefore, to ensure the coherence of the entire process, we propose an end-to-end framework that directly maps the original letter to the final animation, as illustrated in Fig. 2. To address the complexity of learning per-frame displacement that converts the letter into animation, we represent the video as a learnable base shape and per-frame motion added on the base shape (§4.1). Additionally, we incorporate legibility regularization loss based on perceptual similarity

[Figure 4]

Figure 4. Illustration of the prior knowledge conflict issue. The left is the deformed “R” for “BULLFIGHTER” with prompt “A bullfighter holds the corners of a red cape in both hands and waves it” generated by Iluz et al. [18], the right is generated by Gal et al. [13] to animate the deformed letter with the same prompt. The mismatch in prior knowledge between separate models leads to significant appearance changes and severe artifacts, as highlighted by the red circles.

to maintain letter legibility (§4.2). Finally, we introduce a mesh-based structure preservation regularization loss to ensure appearance and structure integrity between frames, mitigating inconsistent artifacts (§4.3).

#### 4.1. Base Field and Motion Field

Learning the per-frame displacement that directly converts the input letter into animation frames is challenging. Directly optimizing the displacement may lead to severe artifacts that degrade the quality of the animation, including distortion, flickering, and abrupt appearance changes in the adjacent frame. Inspired by the dynamic NeRFs [39, 42] and CoDeF [37], we propose modeling the generated video in two neural displacement fields: the base field and the motion field, to address the complexity of this deformation. Both fields are represented by coordinate-based Multilayer Perceptron (MLP). To better capture high-frequency variation and represent geometry information, we project the coordinates into a higher-dimensional space using the NeRF [32] positional encoding:

γ(p) = sin(20πp), cos(20πp), . . . , sin(2L−1πp), cos(2L−1πp) ,

(3)

where p refers to control point coordinates. Further details can be found in the appendix.

The objective of the base field, denoted as B, is to learn a shared shape for every animation frame, serving as a base to infuse motion. It is defined by a function B : γ(Pletter) → PB, which maps the original letter’s control points coordinates Pletter into base shapes’ coordinates PB, both in RN×2.

The motion field, denoted as M, encodes the correspondence between the control points in the base shape and those in each video frame. Inspired by CoDeF [37], we represent the video as a 3D volume space, where a control point at t-th frame with coordinate (x,y) is represented by (x,y,t). We duplicate the shared base shape k times across k frames and convert the coordinates into 3D volume space, written as PB′ : {(PB,t)}kt=1. The motion field is defined as a function

M : γ(PB′ ) → PV that maps control points from the base shape to their corresponding locations PV ∈ Rk×N×2 in each video frame.

To better model motion, we represent PV as PB + ∆P, focusing on learning the per-frame displacements ∆P = {∆Pt}kt=1 to be applied on the base shape. Following Gal et al. [13], we decompose the motion into global motion (modeled by an affine transformation matrix shared by all control points of an entire frame) and local motion (predicted for each control point separately). Consider the i-th control point on the base shape with coordinate (xB,i,yB,i), its displacement on t-th frame ∆pti is summed by its local and global displacement:

∆pti = ∆pt,locali + ∆pt,globali , (4)

 

 

 

 

 

 −

 

  ,

sx shxsydx shysx sy dy

cos θ sin θ0 − sin θcos θ0

xB,i yB,i 1

xB,i yB,i 1

∆pt,globali 1

=

0 0 1

0 0 1

(5)

where ∆pt,locali and all elements in the per-frame global transformation matrix are predicted by the MLP in the

motion field.

To train the base field and motion field, we distill prior knowledge from the large-scale pretrained text-tovideo model using SDS computed in Eq. 2. At each training iteration, we use a differentiable rasterizer [25], denoted as R, to render our predicted control points set PV into a rasterized video (pixel format video). We select a diffusion timestep τ, draw a sample from a normal distribution for noise ϵ ∼ N(0,1), and then add the noise to the rasterized video. The video foundation model denoise this video, based on the user prompt describing a motion pattern closely related to the word’s semantic meaning (e.g. “A camel walks steadily across the desert” for “M” in “CAMEL”). We jointly optimize the base field and motion field using the SDS loss to generate videos aligned with the desired text prompt. The visualized base shape demonstrates alignment with the prompt’s semantics, as shown in Fig. 5.

[Figure 5]

Figure 5. Base shape of “Y” for “GYM” with prompt “A man doing exercise by lifting two dumbbells in both hands.”

#### 4.2. Legibility Regularization

A critical requirement for Dynamic Typography is ensuring the animations maintain legibility. For example, for “M” in “CAMEL”, we hope the “M” takes on the appearance of

a camel while being recognizable as the letter “M”. When employing SDS loss for training, the text-to-video foundation model’s prior knowledge naturally deforms the letter’s shape to match the semantic content of the text prompt. However, this significant appearance change compromises the letter’s legibility throughout the animation.

Thus, we propose a regularization term that enforces the letter to be legible, working alongside the SDS loss to guide the optimization process. Specifically, we leverage Learned Perceptual Image Patch Similarity (LPIPS) [59] as a loss to regularize the perceptual distance between the rasterized images of the base shape R(PB) and the original letter R(Pletter):

Llegibility = LPIPS(R (PB),R (Pletter)). (6)

Benefiting from our design, we only need to apply this LPIPS-based legibility regularization term to the base shape, and the motion field will automatically propagate this legibility constraint to each frame.

#### 4.3. Structure Preservation Regularization

The optimization process alters the positions of control points, sometimes leading to complex intersections between B´ezier curves, as illustrated in Fig. 6d. The rendering of Scalable Vector Graphics (SVG) adheres to the non-zero rule or even-odd rule [10], which determines the fill status by drawing an imaginary line from the point to infinity and counting the number of times the line intersects the shape’s boundary. The frequent intersections between B´ezier curves complicate the boundary, leading to alternating black and white “holes” within the image. Furthermore, these intersections between B´ezier curves vary abruptly between adjacent frames, leading to severe flickering effects that degrade animation quality, see Fig. 6. In addition, the unconstrained degrees of freedom in motion could alter the appearance of the base shape, leading to noticeable discrepancies in appearance between adjacent frames and temporal inconsistency.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

(a) frame 1 (b) frame 2 (c) frame 3 (d) frame 1 vis.

Figure 6. Adjacent frames of animation for letter “E” in “JET”. A large area of alternating black and white “holes” occur within each frame, as highlighted within the red circles, causing severe flickering between the adjacent frames. (d) is the visualization of frame 1, highlighting the control points and the associated B´ezier curves. The illustration reveals frequent intersections among the B´ezier curves leading to the flickering artifacts.

To address these issues, we adopt Delaunay Triangulation [3, 8] on the base shape based on control points, as shown in Fig. 7. By maintaining the structure of the triangular meshes, we prevent frequent intersections between B´ezier curves, while preserving the relative consistency of local geometry information across adjacent frames.

[Figure 10]

- Figure 7. Illustration of the Mesh-based structure preservation. We first apply this regularization between the base shape and the input letter. We propagate the structural constraint to every frame by regularizing the last frame with the base shape and regularizing every frame with its next frame.

The whole regularization process is illustrated in Fig. 7. We employ the angle variation [18] of the corresponding triangular meshes in adjacent frames as regularization:

k

m

1 k × m

∥Ti,t+1 − Ti,t∥22, (7)

t=1

i=1

where m refers to the total number of triangular meshes in each frame, Ti,t ∈ R3 refers to the corresponding angles in i-th triangular mesh in the t-th frame. Particularly, the (k + 1)-th frame refers to the base shape. Consequently, the structural constraint with the base shape is propagated to every frame, allowing the preservation of the geometric structure throughout the animation. Furthermore, to ensure the base shape is geometrically similar to the input letter, we apply the same triangulation-based constraints between the base shape and the letter. Lstructure can be formulated as:

m

1 m

∥Ti,letter − Ti,B∥22

Lstructure =λ1 ·

i=1

(8)

k

m

1 k × m

∥Ti,t+1 − Ti,t∥22,

+ λ2 ·

t=1

i=1

where λ1 and λ2 are weight hyperparameters .

We find that this angle-based na¨ıve approach effectively maintains the triangular structure, thereby alleviating the

frequent intersections of the B´ezier curves and preserving a relatively stable appearance across different frames without significantly affecting the motion liveliness.

### 5. Experiments

We create a dataset via a workshop to evaluate our method’s ability. The dataset covers Dynamic Typography samples for all letters in the alphabet, featuring a variety of elements such as animals, humans, and objects, in a total of 33 samples. Each sample includes a word, a specific letter within the word to be animated, and a concise text prompt describing the desired animation. For each sample, a video with 24 frames will be generated. Each sample takes 1000 epochs for optimization, about 40 minutes on a H800 GPU.

To illustrate our method’s capabilities, we present some generated results in Fig. 1. These animations vividly bring the specified letter to life while adhering to the prompt and maintaining the word’s readability. The implementation details and more results can be found in our appendix and supplementary materials.

#### 5.1. Comparisons

We compare our method with approaches from two distinct categories: the pixel-based strategies leveraging either textto-video or image-to-video methods, and the vector-based animation method.

Within the pixel-based scenario, we compare our model against the leading text-to-video generation models Gen2 [45] (ranked first in the EvalCrafter [26] and VBench [17] benchmark) – a commercial web-based tool, and DynamiCrafter [56], the state-of-the-art model for imageto-video generation conditioned on text. For text-to-video generation, we append the prompt with “which looks like a letter β,” where β represents the specific letter to be animated. In the image-to-video case, we use the stylized letter generated by the Word-as-Image [18] as the conditioning image. Within the vector-based scenario, we utilize LiveSketch [13] as a framework to animate vector images. To ensure a fair comparison, we condition the animation on the stylized letter generated by the Word-as-Image [18] as well. For fairness, all animations to be compared are composed of 24 frames, each lasting 3 seconds, and are rendered at a resolution of 256 × 256.

Qualitative Comparison We present two samples for visual comparison in Fig. 8. For more comparisons and full videos, please check our supplementary materials. While achieving high resolution and realism, Gen-2 struggles to generate frames that keep the letter’s shape, which greatly harms the legibility. With DynamiCrafter, the “SWAN” animation exhibits minimal movement, while the “GYM” animation features unrealistic motion that deviates from the user’s prompt. Although LiveSketch can depict the user’s prompt through animation, it sacrifices legibility. Also,

[Figure 11]

- Figure 8. Visual comparisons between the baselines and our model. Text-to-video model (Gen-2) generates colorful images but fails to maintain the shape of the original letter. The pixel-based image-to-video model (DynamiCrafter) produces results with minimal, sometimes unreasonable motion. The general vector animation model (LiveSketch) struggles to preserve legibility or maintain a consistent appearance.

the animated letter fails to maintain a stable appearance throughout the animation, as demonstrated in the “SWAN” example. Our model strikes a balance between promptvideo alignment and letter legibility. It consistently generates animations that adhere to the user’s prompt while preserving the original letter’s form. This allows the animation to seamlessly integrate within the original word, as showcased by the in-context results in Fig. 8.

|Method|Perceptual Input Conformity (↑)<br><br>|Text-to-Video Alignment(↑)|
|---|---|---|
|Gen-2 DynamiCrafter LiveSketch Ours|0.1495 0.5151 0.4841 0.5301<br><br>|23.3687 17.8124 20.2402 21.4391|

- Table 1. Quantitative results between the baselines and our model. The best score for each metric is highlighted in red. Our model gets the best score in PIC, and second-best score in text-to-video alignment, indicating a balance between faithfully representing the user’s prompt and maintaining the legibility.

Quantitative Comparison Tab. 1 presents the quantitative evaluation results. We employ two metrics, Perceptual Input Conformity (PIC) and Text-to-Video Alignment. Following DynamiCrafter [56], we compute Perceptual Input Conformity (PIC) using DreamSim’s [41] perceptual similarity metric between each output frame and the input letter, averaged across all frames. This metric assesses how well the animation preserves the original letter’s appearance. To evaluate the alignment between the generated videos and their corresponding prompts (Text-to-Video Alignment), we leverage the X-CLIP score [29], which extends CLIP [43] to video recognition, to obtain frame-wise image embeddings and text embeddings. The average cosine

similarity between these embeddings reflects how well the generated videos align with the corresponding prompts.

While Gen-2 achieves the highest text-to-video alignment score, it severely suffers in legibility preservation (lowest PIC score). Conversely, our model excels in PIC, indicating the effectiveness in maintaining the original letter’s form. Our model also achieves the second-best textto-video alignment score, indicating it faithfully depicts the animation prompt while preserving legibility.

#### 5.2. Ablation Study

We conduct ablations to analyze the contribution of each component in our proposed method: learnable base shape, legibility regularization, and mesh-based structure preservation regularization. Visual results in Fig. 9 showcase the qualitative impact of removing each component. Quantitative results in Tab. 2 further confirm their effectiveness.

In addition to Perceptual Input Conformity (PIC) and XCLIP score (Text-to-Video Alignment), we employ warping error to assess temporal consistency, following EvalCrafter [26]. This metric estimates the optical flow between consecutive frames using the RAFT model [49] and calculates the pixel-wise difference between the warped image and the target image. The lower warping error indicates smoother and more temporally consistent animations.

The calculation of legibility and structure preservation regularization loss involves the base shape. Hence, when removing the learnable base shape, the legibility loss Llegibility is computed between every output frame and the input letter, and the structure preservation loss Lstructure is only applied between every pair of consecutive frames.

Base Shape: As observed in Fig. 9 (row 2), removing the shared learnable base shape results in inconsistent animations. Specifically, as highlighted by the red circle, the

[Figure 12]

- Figure 9. Visual comparisons of ablation study. Some artifacts are highlighted in red cycles. Removing base shape or structure preservation regularization results in shape deviation and inconsistent artifacts. Without legibility regularization, each animation frame loses the letter “R” shape.

|Method|Warping Error(↓)|PIC (↑)|T2V Align. (↑)|
|---|---|---|---|
|Full Model No Base Shape No Legibility No Struc. Pre.|0.01645 0.03616 0.01561 0.01777<br><br>|0.5310 0.5178 0.4924 0.4906|21.4447 20.0568 20.2857 20.6285|

- Table 2. Quantitative results of the ablation study. The best score for each metric is highlighted in red. The full model gets the best in PIC, text-to-video alignment, and second-best in optical flow warping error, indicating the effectiveness of each module.

appearance of the bullfighter deviates significantly between frames, harming legibility. The finding is also supported by Tab. 2 (row 2), where removing the base shape results in significant degradation under all three metrics.

Legibility Regularization: Without the perceptual regularization on the base shape, the base shape struggles to preserve legibility. As a result, each animation frame loses the letter “R” shape in Fig. 9 (row 3), leading to lower PIC in Tab. 2 (row 3).

Structure Preservation Regularization: Removing meshbased structure preservation allows the base shape’s structure to deviate from the original letter, causing the discontinuity between the bullfighter and cape in the base shape and all frames, as highlighted in Fig. 9 (row 4). Without this regularization term, the animation shows inconsistent appearances across different frames, degrading legibility and leading to the lowest PIC in Tab. 2 (row 4).

#### 5.3. Generalizability

Our framework, leveraging Score Distillation Sampling (SDS), achieves generalization across various diffusionbased text-to-video models. To demonstrate this, we apply different base models for computing LSDS, including

the 1.7-billion parameter text-to-video model from ModelScope [50], AnimateDiff [14], and ZeroScope [28]. Fig. 10 presents visual results for the same animation sample (“K” in “Knight”) with each base model.

[Figure 13]

Figure 10. Visual results of the same animation sample (“K”) using different text-to-video base models. All generated animations accurately depict the user’s prompt and maintain the “K” shape, showing the generalizability of our methods across different textto-video foundation models.

While the animation exhibits deformations and animation styles unique to each model, all animations accurately depict the user’s prompt and maintain the “K” shape. This showcases the generalizability of our method. Hence, future advancements in text-to-video models with stronger prior knowledge will benefit our approach.

### 6. Conclusion

We propose a text animation scheme, termed “Dynamic Typography”, that deforms letters to convey semantic meaning and animates them vividly based on user prompts. To automate text animation generation, we propose an end-to-end optimization-based framework that leverages the video diffusion prior. Our method faithfully depicts the user’s prompt in animation while preserving the input letter’s legibility, and is generalizable to different fonts, prompts, and languages. Nevertheless, there remain several limitations. First, the motion quality can be bounded by the video foundation model, which may be unaware of specific motions in some cases. Luckily, our framework is modelagnostic, facilitating integration with future diffusion-based video foundation model advancements. Besides, challenges arise when user-provided text prompts deviate significantly from original letter shapes, complicating the model’s ability to strike a balance between generating semantic-aware vivid motion and preserving the legibility of the original letter. We hope that our work can open the possibility for further research of semantic-aware text animation that incorporates the rapid development of video generation models.

### References

- [1] Adobe Systems Inc. 1990. Adobe Type 1 Font Format. Addison Wesley Publishing Company. 3
- [2] Samaneh Azadi, Matthew Fisher, Vladimir Kim, Zhaowen Wang, Eli Shechtman, and Trevor Darrell. 2018. MultiContent GAN for Few-Shot Font Style Transfer. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. https://doi.org/10.1109/cvpr. 2018.00789 2
- [3] C Barber and Hannu Huhdanpaa. 1995. Qhull. The Geometry Center, University of Minnesota. 6
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023). 2
- [5] Bay-Wei Chang and David Ungar. 1993. Animation: from cartoons to the user interface. In Proceedings of the 6th Annual ACM Symposium on User Interface Software and Technology (Atlanta, Georgia, USA) (UIST ’93). Association for Computing Machinery, New York, NY, USA, 45–55. https://doi.org/10.1145/168642.168647 2
- [6] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. 2023. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512 (2023). 2
- [7] Werner Lemberg David Turner. 2009. FreeType library. Retrieved Mar 19, 2024 from https://freetype. org/ 3, 12
- [8] Boris Delaunay et al. 1934. Sur la sphere vide. Izv. Akad. Nauk SSSR, Otdelenie Matematicheskii i Estestvennyka Nauk 7, 793-800 (1934), 1–2. 6
- [9] Noa Fish, Lilach Perry, Amit Bermano, and Daniel CohenOr. 2020. SketchPatch. ACM Transactions on Graphics (Dec 2020), 1–14. https://doi.org/10.1145/ 3414685.3417816 2
- [10] James D Foley. 1996. Computer graphics: principles and practice. Vol. 12110. Addison-Wesley Professional. 5
- [11] Shannon Ford, Jodi Forlizzi, and Suguru Ishizaki. 1997. Kinetic typography. In CHI ’97 extended abstracts on Human factors in computing systems looking to the future - CHI ’97. https://doi.org/10.1145/1120212.1120387 2
- [12] Jodi Forlizzi, Johnny Lee, and Scott Hudson. 2003. The kinedit system. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems. https://doi. org/10.1145/642611.642677 2
- [13] Rinon Gal, Yael Vinker, Yuval Alaluf, Amit H. Bermano, Daniel Cohen-Or, Ariel Shamir, and Gal Chechik. 2023. Breathing Life Into Sketches Using Text-to-Video Priors.

(2023). arXiv:2311.13608 [cs.CV] 2, 4, 5, 6, 12, 14

- [14] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. 2024. AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. In

- The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id= Fx2SbBgcte 2, 8
- [15] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. 2022. Latent Video Diffusion Models for High-Fidelity Video Generation with Arbitrary Lengths. (Nov 2022). 2
- [16] Kai Hormann and G¨unther Greiner. 2000. MIPS: An efficient global parametrization method. Curve and Surface Design: Saint-Malo 1999 (2000), 153–162. 2
- [17] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. 2024. VBench: Comprehensive Benchmark Suite for Video Generative Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6
- [18] Shir Iluz, Yael Vinker, Amir Hertz, Daniel Berio, Daniel Cohen-Or, and Ariel Shamir. 2023. Word-As-Image for Semantic Typography. ACM Trans. Graph. 42, 4, Article 151 (jul 2023), 11 pages. https://doi.org/10.1145/ 3592123 1, 2, 3, 4, 6, 12
- [19] Yue Jiang, Zhouhui Lian, Yingmin Tang, and Jianguo Xiao. 2019. SCFont: Structure-Guided Chinese Font Generation via Deep Stacked Networks. Proceedings of the AAAI Conference on Artificial Intelligence (Sep 2019), 4015–4022. https://doi.org/10.1609/aaai. v33i01.33014015 2
- [20] Oren Katzir, Or Patashnik, Daniel Cohen-Or, and Dani Lischinski. 2024. Noise-free Score Distillation. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id= dlIMcmlAdk 3
- [21] Pika labs. 2023. Pikalabs. https://www.pika.art/ 2
- [22] Joonhwan Lee, Soojin Jun, Jodi Forlizzi, and Scott E. Hudson. 2006. Using kinetic typography to convey emotion in text-based interpersonal communication. In Proceedings of the 6th Conference on Designing Interactive Systems (University Park, PA, USA) (DIS ’06). Association for Computing Machinery, New York, NY, USA, 41–49. https: //doi.org/10.1145/1142405.1142414 1
- [23] Johnny C. Lee, Jodi Forlizzi, and Scott E. Hudson. 2002. The kinetic typography engine. In Proceedings of the 15th annual ACM symposium on User interface software and technology. https://doi.org/10.1145/571985.571997 2
- [24] Johnny C. Lee, Jodi Forlizzi, and Scott E. Hudson. 2002. The kinetic typography engine: an extensible system for animating expressive text. In Proceedings of the 15th Annual ACM Symposium on User Interface Software and Technology (Paris, France) (UIST ’02). Association for Computing Machinery, New York, NY, USA, 81–90. https://doi. org/10.1145/571985.571997 1
- [25] Tzu-Mao Li, Michal Luk´aˇc, Micha¨el Gharbi, and Jonathan Ragan-Kelley. 2020. Differentiable vector graphics rasterization for editing and learning. ACM Transactions on Graphics (Dec 2020), 1–15. https://doi.org/10.1145/ 3414685.3417871 3, 5

- [26] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. 2023. Evalcrafter: Benchmarking and evaluating large video generation models. arXiv preprint arXiv:2310.11440 (2023). 6, 7
- [27] Raphael Gontijo Lopes, David Ha, Douglas Eck, and Jonathon Shlens. 2019. A learned representation for scalable vector graphics. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7930–7939. 1
- [28] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. 2023. VideoFusion: Decomposed Diffusion Models for High-Quality Video Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 8, 12
- [29] Yiwei Ma, Guohai Xu, Xiaoshuai Sun, Ming Yan, Ji Zhang, and Rongrong Ji. 2022. X-CLIP: End-to-End Multigrained Contrastive Learning for Video-Text Retrieval. In Proceedings of the 30th ACM International Conference on Multimedia (¡conf-loc¿, ¡city¿Lisboa¡/city¿, ¡country¿Portugal¡/country¿, ¡/conf-loc¿) (MM ’22). Association for Computing Machinery, New York, NY, USA, 638–647. https://doi.org/10.1145/3503161.3547910 7
- [30] Wendong Mao, Shuai Yang, Huihong Shi, Jiaying Liu, and Zhongfeng Wang. 2022. Intelligent typography: Artistic text style transfer for complex texture and structure. IEEE Transactions on Multimedia (2022). 2
- [31] Yifang Men, Zhouhui Lian, Yingmin Tang, and Jianguo Xiao. 2019. DynTypo: Example-Based Dynamic Text Effects Transfer. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). https://doi. org/10.1109/cvpr.2019.00602 2
- [32] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106. 3, 4, 12
- [33] Mitsuru Minakuchi and Yutaka Kidawara. 2008. Kinetic typography for ambient displays. In Proceedings of the 2nd international conference on Ubiquitous information management and communication. https://doi.org/10. 1145/1352793.1352805 2
- [34] Mitsuru Minakuchi and Katsumi Tanaka. 2005. Automatic kinetic typography composer. In Proceedings of the 2005 ACM SIGCHI International Conference on Advances in computer entertainment technology. https://doi. org/10.1145/1178477.1178512 2
- [35] Haomiao Ni, Changhao Shi, Kai Li, Sharon X Huang, and Martin Renqiang Min. 2023. Conditional Image-toVideo Generation with Latent Flow Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18444–18455. 2
- [36] OpenAI. 2024. GPT-4 Technical Report. arXiv:2303.08774 [cs.CL] 14
- [37] Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. 2023. Codef: Content deformation fields

- for temporally consistent video processing. arXiv preprint arXiv:2308.07926 (2023). 4
- [38] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. 2021. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 5865–5874. 12
- [39] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T. Barron, Sofien Bouaziz, Dan B Goldman, Ricardo MartinBrualla, and Steven M. Seitz. 2021. HyperNeRF: A HigherDimensional Representation for Topologically Varying Neural Radiance Fields. ACM Trans. Graph. 40, 6, Article 238 (dec 2021). 4
- [40] Laurence Penny. 1996. A History of TrueType. Retrieved Mar 19, 2024 from https://www.truetypetypography.com 3
- [41] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. 2023. DreamFusion: Text-to-3D using 2D Diffusion. In The Eleventh International Conference on Learning Representations. https://openreview.net/forum?id= FjNys5c7VyY 2, 3, 7
- [42] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. 2020. D-NeRF: Neural Radiance Fields for Dynamic Scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4
- [43] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763. 7
- [44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. 2022. High-Resolution Image Synthesis with Latent Diffusion Models. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.1109/ cvpr52688.2022.01042 2
- [45] RunwayML. 2023. Runway. https://research. runwayml.com/gen2 2, 6, 14
- [46] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, et al. 2024. Motion-I2V: Consistent and Controllable Image-to-Video Generation with Explicit Motion Modeling. arXiv preprint arXiv:2401.15977

(2024). 2

- [47] Aliaksandr Siarohin, St´ephane Lathuili`ere, Sergey Tulyakov, Elisa Ricci, and Nicu Sebe. 2019. First Order Motion Model for Image Animation. Neural Information Processing Systems,Neural Information Processing Systems (Jan 2019). 2
- [48] Maham Tanveer, Yizhi Wang, Ali Mahdavi-Amiri, and Hao Zhang. 2023. DS-Fusion: Artistic Typography via Discriminated and Stylized Diffusion. (Mar 2023). 2
- [49] Zachary Teed and Jia Deng. 2020. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part II 16. Springer, 402–419. 7

- [50] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. 2023. Modelscope textto-video technical report. arXiv preprint arXiv:2308.06571

(2023). 2, 3, 8, 12

- [51] Wenjing Wang, Jiaying Liu, Shuai Yang, and Zongming Guo. 2019. Typography With Decor: Intelligent Text Style Transfer. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/ 10.1109/cvpr.2019.00604 2
- [52] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. 2024. Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems 36 (2024). 2
- [53] Yizhi Wang and Zhouhui Lian. 2021. DeepVecFont: Synthesizing High-quality Vector Fonts via Dual-modality Learning. ACM Transactions on Graphics 40, 6 (2021), 15 pages. https://doi.org/10.1145/3478513.3480488 1
- [54] Liwenhan Xie, Xinhuan Shu, Jeon Cheol Su, Yun Wang, Siming Chen, and Huamin Qu. 2023. Creating emordle: Animating word cloud for emotion expression. IEEE Transactions on Visualization and Computer Graphics (2023). 2
- [55] Liwenhan Xie, Zhaoyu Zhou, Kerun Yu, Yun Wang, Huamin Qu, and Siming Chen. 2023. Wakey-Wakey: Animate Text by Mimicking Characters in a GIF. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology. https://doi.org/10.1145/ 3586183.3606813 1, 2
- [56] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. 2023. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190 (2023). 2, 6, 7, 14
- [57] Shuai Yang, Zhouhui Lian, and Zhongwen Guo. 2016. Awesome Typography: Statistics-Based Text Effects Transfer. Cornell University - arXiv,Cornell University - arXiv (Nov 2016). 2
- [58] Shuai Yang, Zhangyang Wang, and Jiaying Liu. 2021. Shape-Matching GAN++: Scale Controllable Dynamic Artistic Text Style Transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence (Jan 2021), 1–1. https://doi.org/10.1109/tpami.2021. 3055211 2
- [59] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition. 586–595. 2, 5
- [60] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. 2022. MagicVideo: Efficient Video Generation With Latent Diffusion Models. (Nov 2022). 2

Appendix

- A. Implementation Details

The base field and motion field are jointly optimized. Following Gal et al. [13], we interleavely optimize local motion and global motion. Adam optimizer is adopted, and the learning rates are set to be 5e − 3, 5e − 3, and 1e − 3 to learn the base field, local motion, and global motion. For the global motion, we set the scaling factor to be 2.0, 1e−2, 5e−2, and 1e−1 for translation, rotation, scale, and shear.

We set the regularization weight to be 5e3 for the Llegibility and 1e3, 1e4 for the λ1, λ2 in Lstructure. We observe that Llegibility often plays a dominant role. When perceptual regularization is employed from the start, the base shape typically retains the original letter’s form, preventing any semantic deformations. Hence, following Iluz et al. [18], we gradually increase the weight of Llegibility to make its effects after semantic deformation has taken place.

We use the text-to-video-ms-1.7b model in ModelScope [28, 50] for the diffusion backbone. We apply augmentations including random crop and random perspective to all video frames. We intend to make our code available to release all the details and support further research.

- B. Frequency-based Encoding and Annealing

NeRF [32] has highlighted that a heuristic application of sinusoidal functions to input coordinates, known as “positional encoding”, enables the coordinate-based MLPs to capture higher frequency content, as denoted by:

γ(p) = sin(20πp), cos(20πp), . . . , sin(2L−1πp), cos(2L−1πp) ,

(9)

where p refers to the point coordinates.

We found that this property also applies to our MLPs that use coordinates of control points as input. This allows the MLPs in the base and motion field to more effectively represent high-frequency information, corresponding to the detailed geometric features. Additionally, when using coordinate-based MLPs to model motion, a significant challenge is capturing both minute and large motions. Following Nerfies [38], we employ a coarseto-fine strategy that initially targets low-frequency (largescale) motion and progressively refines the high-frequency (localized) motions. Specifically, we use the following formula to apply weights to each frequency band j in the positional encoding of the MLPs within the motion field.

1 − cos(π · clamp(α − j,0,1)) 2

, (10)

wj(α) =

where α(t) = LtN , t is the current training iteration, and N is a hyper-parameter for when α should reach the maximum

number of frequencies L.

[Figure 14]

- Figure 11. Visual comparison with and without frequency encoding and annealing. The geometry and motion quality degrades when removing annealed frequency-based encoding.

Fig. 11 shows the visual ablation comparisons with and without the frequency encoding and annealing. When removing annealed frequency-based encoding, the geometry and motion quality of the generated animation suffer. To be specific, the butterfly animation in Fig. 11 (row 2) exhibits a lack of geometry details, and the bullfighter animation in Fig. 11 (row 4) shows unreasonable motion, leading to the degradation of the animation quality.

C. Effect Analysis of Control Points

By adjusting the number of control points on the B´ezier curves, we can alter the appearance of the generated text animations, thus producing more diverse outcomes. Generally, increasing the number of control points enriches the geometric details, as shown in Fig. 12 (row 2). However, including too many control points may lead to self-intersection of the B´ezier curves, causing abrupt and frequent changes of black and white regions, as highlighted by the red circles in Fig. 12 (row 3). Users can choose the sample they desire based on their preferences.

[Figure 15]

- Figure 12. The effect of the number of control points. The first row displays the result generated with the default number of control points extracted by FreeType [7], which is 75. In the second and third rows, the number of control points is increased to 204 and 420 respectively.

[Figure 16]

###### Figure 13. Dynamic Typography over different fonts for the same animation sample. The corresponding fonts in the first, second, and third rows are KaushanScript-Regular, Segoe Print, and Roboto-Bold respectively. The animated letter “H” preserves the unique style of each font while faithfully depicting the prompts, allowing it to be seamlessly integrated into the word “HARMONY” under different fonts.

[Figure 17]

###### Figure 14. Dynamic Typography over different prompts for the same letter “M” to be animated. Our approach displays different visual effects based on three different prompts, offering a completely different reading experience for the same word.

[Figure 18]

###### Figure 15. Dynamic Typography over different languages. In the first row, we animate the Chinese character “从” in the word “跟 从”, meaning “following” in English. In the second row, we animate the hiragana character 「つ」from the Japanese word 「つり」, meaning “fishing” in English. This demonstrates the potential of our proposed methodology to generate Dynamic Typography over different languages.

### D. Generalizability over Different Fonts, Prompts and Languages

Our method generalizes well in generating Dynamic Typography over different fonts, text prompts, and languages. When the letter to be animated takes different fonts, the generated animation preserves the unique style of each font, as illustrated in Fig. 13. In Fig. 14, by adjusting the input prompts, our method can generate animation with distinct visual effects for the same letter to be animated. Fig. 15 demonstrates two Dynamic Typography samples in Chinese and Japanese, indicating the potential to generalize into different languages.

### E. User Study

We conduct a user study to further compare our animation result with three baseline models: DynamiCrafter [56], Gen-2 [45], and LiveSketch [13], using the same dataset as the quantitative analysis. In the study, users are asked to select their preferred sample from four options, each produced by one of the four models. Their selections are guided by the three crucial requirements for Dynamic Typography as described in the method section in the main body, i.e., legibility preservation, semantic alignment, and temporal consistency. Users are required to answer 8 questions that are randomly sampled from the dataset. We collect 62 responses in total.

[Figure 19]

- Figure 16. User Study results. Users are required to select the best animation based on legibility preservation, semantic alignment, and temporal consistency. The pie chart shows the percentage of each method being selected as the best animation.

According to the final result shown in Fig. 16, our animation demonstrates transcendent distinction among the baseline models. Out of 44.76% responses, the animation generated by our method is selected to be the best, indicating that our generated animations best meet the crucial requirements.

### F. Failure Case

We observe that, in some samples, the semantic meaning and corresponding motion specified by the user-provided text prompt significantly diverged from the original shape of the letter. In such cases, the model struggles to simultaneously maintain the shape of the letter and convey the vivid semantic information of the text prompt. As a result, the letter either undergoes minimal change in shape, retaining its original form, or it completely loses its original shape, compromising legibility.

[Figure 20]

Figure 17. Failure case illustration. The first row is generated with the default weight for legibility and structure preservation loss, which suffers from minimal semantic deformation. In the second row, we reduce the weight of these regularization losses, which compromises the legibility.

For example, in Fig. 17, when we incorporate the legibility regularization, the shape of the letter “R” remains unchanged, maintaining its original form while performing the “launch” action. Conversely, when we reduce the weight of the legibility regularization, it transforms completely into the shape of a rocket, losing the characteristic contours of the letter “R”, thus sacrificing the legibility.

### G. GPT-4V as Dynamic Typography Designer

As illustrated in Fig. 17, if the user-specified text prompt deviates too much from the chosen letter’s shape, it can hinder the creation of vivid animations. We can utilize the powerful visual and semantic understanding capabilities of Vision Language Models (VLMs) to assist users in selecting appropriate letters and prompts.

In the experiment, we provide current state-of-the-art VLM, GPT-4V [36], with a snapshot of an animation generated, along with the corresponding chosen word, letter, and text prompt as an example to facilitate in-context learning by GPT-4V. Subsequently, we request GPT-4V to design text animations by following the paradigm exemplified in the previous experiment. We require GPT-4V to generate outputs including the word, the selected letter, and the text prompt, explicitly demanding that it considers the similarity between the letter’s original shape and its shape after deformation.

We list some samples designed by GPT-4V as follows:

[Figure 21]

Figure 18. Some results generated by our method based on pairs of prompts and letters that are designed with the assistance of GPT-4V.

Word: CAT Chosen Letter: C Text Prompt: “A cat curls up to sleep.” Animation Idea: The “C” naturally curls tighter into a

circular shape, resembling a cat curling up.

Word: KICK Chosen Letter: K Text Prompt: “A soccer player kicks a ball.” Animation Idea: The angled legs of the “K” mimic the

motion of kicking, with one leg drawing back and then striking forward.

Word: SNAKE Chosen Letter: S Text Prompt: “A snake slithers through grass.” Animation Idea: The natural curve of “S” undu-

lates slightly, resembling the slithering movement of a snake.

The generated animations designed by GPT-4V are shown in Fig. 18. We found that GPT-4V has the potential to design proper pairs of prompts and letters that carefully consider the natural shapes of the letters and how they can effectively transform into the desired actions or characteristics with minimal deviation, ensuring the animations are feasible and visually coherent.

