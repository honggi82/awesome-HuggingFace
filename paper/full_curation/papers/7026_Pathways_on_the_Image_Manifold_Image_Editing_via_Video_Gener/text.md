### Pathways on the Image Manifold: Image Editing via Video Generation

Noam Rotstein Gal Yona Daniel Silver Roy Velich David Bensa¨ıd Ron Kimmel

Technion - Israel Institute of Technology

# arXiv:2411.16819v4[cs.CV]20Mar2025

#### Abstract

|[Figure 1]|
|---|

|[Figure 2]|
|---|

“A photo of a tomato in a blue tennis court.”

Recent advances in image editing, driven by image diffusion models, have shown remarkable progress. However, significant challenges remain, as these models often struggle to follow complex edit instructions accurately and frequently compromise fidelity by altering key elements of the original image. Simultaneously, video generation has made remarkable strides, with models that effectively function as consistent and continuous world simulators. In this paper, we propose merging these two fields by utilizing image-tovideo models for image editing. We reformulate image editing as a temporal process, using pretrained video models to create smooth transitions from the original image to the desired edit. This approach traverses the image manifold continuously, ensuring consistent edits while preserving the original image’s key aspects. Our approach achieves stateof-the-art results on text-based image editing, demonstrating significant improvements in both edit accuracy and image preservation. Visit our project page.

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

|[Figure 10]|
|---|

|[Figure 11]|
|---|

"A vase of flowers filled with green water."

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

|[Figure 19]|
|---|

|[Figure 20]|
|---|

"A photo of a teddy bear doing pushups."

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

#### 1. Introduction

Image editing has witnessed remarkable advancements through deep learning and text-guided diffusion networks. These developments have set a new benchmark for image manipulations, enhancing both control and quality. However, current approaches continue to face significant limitations in real-world scenarios. These methods often struggle with two key challenges: achieving precise edits that accurately reflect the intended modifications, and preserving the essential characteristics of the original image content.

Figure 1. Visualization of Frame2Frame’s editing process. Temporal progression of our video-based approach. Starting from the source image (leftmost), frames illustrate the natural evolution toward the target edit (rightmost). Our method produces temporally coherent intermediate states while preserving fidelity to both the source content and the editing intent.

ing edit accuracy and content preservation.

State-of-the-art techniques predominantly rely on textguided diffusion models, which iteratively denoise random latent representations to generate edited images. Such methods condition the generation process on both the source image—using techniques such as latent inversion [32] or model fine-tuning [22]—and the target edit description. However, these approaches require the model to generate a single output image that preserves source image fidelity while implementing complex edits, often compromis-

In this paper, we propose a paradigm shift in image editing by reformulating it as a video generation task. Rather than a single-state transition, our approach harnesses temporal coherence: the source image serves as the initial frame of a video that progressively and naturally transforms toward the target edit. This temporal evolution allows the editing process to unfold through physically plausible intermediate states, providing a continuous path between source

and target images, as illustrated in Figure 1 and Figure S9 in the appendix. This temporal approach leverages the sophisticated world understanding embedded in recent video generation models, which have achieved breakthrough results in temporal coherence and visual quality through training on large-scale internet data [2, 48]. From a geometrical perspective, conventional editing approaches project initial noise onto the natural image manifold, targeting a single point where the image aligns with both the source and the edit request. In contrast, our approach generates a continuous path along the manifold between the original and edited image, producing a smooth realistic transition across different image states, as thoroughly discussed in Section 4.

We implement the proposed approach through a structured pipeline called Frame2Frame (F2F). First, we transform the edit instruction into a Temporal Editing Caption a scenario describing how the edit should naturally evolve over time—using a pretrained Vision-Language Model (VLM). Next, a state-of-the-art image-to-video model generates a temporally coherent sequence guided by the temporal caption. Finally, we identify the frame that best realizes the desired edit with the assistance of a VLM. Extensive experiments demonstrate improvements over existing imageto-image approaches. We evaluate on TEdBench [22] and PosEdit, a newly curated dataset derived from UTD-MHAD [6], which focuses on human pose transformations. PosEdit pairs source images with ground-truth targets of the same subject in different poses, enabling rigorous evaluation of both edit accuracy and source fidelity. Beyond commonly defined editing tasks, our framework shows promising results in more classical computer vision problems such as de-blurring, de-noising, and relighting by recasting them as temporal progressions, suggesting broader applications for video-based image transformations.

###### Our main contributions include:

- 1. Reformulating image editing as a generative video task– leveraging temporal coherence to create edit paths on the natural image manifold, enabling high-fidelity manipulations while preserving source characteristics.
- 2. Frame2Frame: an end-to-end framework that realizes the reformulation through three key components: (1) temporal editing captions, (2) generated video-based editing, and (3) automated frame selection.
- 3. Comprehensive evaluation showing state-of-the-art performance on TEdBench and PosEdit, a new dataset for evaluating human pose edits.

#### 2. Related Efforts

##### 2.1. Image Editing

Text-based image editing has advanced significantly with the success of generative diffusion models [19]. These models, in their text-to-image version, generate images through

a denoising diffusion process conditioned on input text, relying on ground truth text-image pairs for training [37, 39]. In contrast, image editing often lacks predefined ground truth data for source and target images, posing unique challenges. This limitation has led researchers to explore diverse editing methodologies [20]. For example, SDEdit [30] injects noise into an image and then denoises it based on an editing target prompt. Imagic [22] fine-tunes a text-toimage model on a single image, subsequently interpolating between input and target text embeddings to produce edits. Other methods first invert the input image into a diffusion model’s latent space [21, 32] and then generate the edited image from that latent representation using various techniques for structure preservation and manipulation [18, 34]. InstructPix2Pix [5] synthesizes an editing dataset using the approach in [18], filters it, and employs this dataset to train a diffusion model in a supervised fashion. Paint-by-Inpaint

- [47] further explores this supervised approach, generating a real-image dataset for object insertion.

2.2. Generative Video Models

Recent years have seen remarkable advancements in video generation, evolving from domain-specific systems [15] and brief clips to models capable of generating diverse, highfidelity content. This progress has stemmed from paradigm shifts and the large-scale expansion of datasets [7] and architectures. Approaches have evolved from recurrent networks [16, 42] and generative adversarial networks (GANs) [9, 43, 44] to latent diffusion models (LDMs) [2, 3, 11], which leverage large U-Net or transformer-based architectures alongside vast internet-sourced datasets. Notable recent efforts include Stable Video Diffusion [2], which trains an LDM on large curated datasets, and OpenAI’s Sora [28], which achieves impressive results by integrating large architectures with extensive public and private datasets. These models, often termed ”world simulators” due to their emergent understanding of physical dynamics and temporal coherence. Our work builds upon CogVideoX

- [48], a transformer-based latent diffusion model that employs a 3D Variational Autoencoder to compress videos across both spatial and temporal dimensions, enhancing coherence. To improve text-video alignment, CogVideoX also integrates an expert transformer with adaptive LayerNorm, enabling deep fusion of visual and textual modalities.

Close to our work, several recent efforts have utilized the world simulation capabilities of video diffusion models for various computer vision tasks. Make-A-Video3D [41] temporally extends static NeRFs using Score Distillation Sampling from video models. ViVid-1-to-3 [25] generates images along a camera trajectory around objects to enable novel view synthesis. PhysDreamer [51] models rigid object properties through 3D Gaussians and material fields, trained via distillation from pretrained video generators.

Input Image F2F LEDITS++

[Figure 28]

[Figure 29]

[Figure 30]

Figure 2. Editing Manifold Pathway. Given an input image and target caption ”A happy person making a heart shape with their hands”, our method generates a continuous path on the natural image manifold. Each generated frame (indicated by black arrows) represents a plausible intermediate state between the source and target, maintaining temporal consistency throughout the transformation. As a result, in contrast to the competing approach, F2F achieves the desired edit while preserving the ”AI” text on the person’s shirt.

##### 2.3. Image Editing and Video

The intersection of image editing and video has received limited attention. Existing methods focus on sampling pairs of random frames from videos to build image pair datasets that capture the same subject under varying conditions. For example, AnyDoor [8] uses the paired frames as an augmentation method, segmenting foreground objects in each frame and assigning one masked object as the target edited appearance of the subject. MagicFixup [1] employs these frames to build a dataset focused on refining user-made subject coarse 2D edits. Recently, drag-based editing approaches [29, 40], used frames extracted from video and computed optical flow to collect a dataset aimed at editing by spatially dragging points within the image. In contrast to these approaches, which rely on frame sampling for image data collection, we are the first to directly perform image editing using generative video models.

#### 3. Frame2Frame

We present Frame2Frame, a framework that reformulates image editing as a temporal transformation process. Our approach leverages video generation models to create natural transitions between source and target images, achieving consistent and realistic edits. The proposed method has three main steps, as illustrated in Figure 3.

- 3.1. Temporal Editing Captions Text-based image editing methods typically operate on two inputs: a source image Is and a target caption c, where c

specifies the desired modifications to Is. Our approach differs fundamentally by modeling editing as a temporal process. This requires a novel type of prompt—the Temporal Editing Caption, denoted by c˜—that describes the sequential transformation from source to target image. We construct c˜by combining information from Is and c to create a description of how the desired edit unfolds over time.

To automate this process, we leverage recent advances in Vision-Language Models (VLMs) [10, 14, 38, 45, 46]. The VLM, specifically ChatGPT-4o [33], is instructed to produce a concise video scenario that highlights how elements within the image change or move over time. The generated caption captures the essential transformations while maintaining a static camera perspective unless movement is necessary. To improve generation quality, we utilize in-context learning (ICL) [12], providing the VLM with nine exemplar prompt-caption pairs. The complete prompt template, ICL examples, and an ablation study comparing this approach with directly using the target captions are included in Section B of the Appendix.

##### 3.2. Video Generation

We employ CogVideoX (I2V-5B) [48], a pretrained generative video latent diffusion model utilizing a transformerbased architecture. Specifically, we use its image-to-video variant, which has been fine-tuned to generate videos starting from an input image Is. During generation, Is is encoded and concatenated with noise in latent space, where

the model applies a denoising process guided by the temporal caption c˜. As elaborated in Section 4, this conditioning allows generated videos to start from Is and evolve naturally along the image manifold, maintaining temporal coherence and consistency. Additionally, the model’s transformer architecture enables effective fusion of visual and textual information, allowing precise control over the editing process through our temporal captions. Formally, given the video generator G, we define the generation process as:

[Figure 31]

[Figure 32]

Temporal Caption

Video Generator

”The cat swiftly springs, using his hind legs to leap into the air.”

"A photo of a jumping cat.”

|Frame Se|lection|
|---|---|

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Figure 3. Frame2Frame Overview. Given a source image and editing prompt, our pipeline proceeds in three steps. First, a Vision-Language Model generates a temporal caption describing the transformation. Next, this caption guides a video generator to create a natural progression of the edit. Finally, our frame selection strategy identifies the optimal frame that best realizes the desired edit, producing the final image of the cat mid-leap.

G(Is,c˜) = V = {f1,...,fT}

where V denotes the generated video with T frames, and ft represents the frame at timestep t.

##### 3.3. Frame Selection

We observed that the optimal number of frames required for an edit can vary—small changes may be completed in fewer frames, while more extensive transformations often necessitate additional ones. Additionally, later frames tend to deviate further from the source image. Thus, even though V serves as an editing path originating from Is, there is no guarantee that fT is the optimal edited image in V . For instance, in Figure 3, the edited image of the cat might be expected to capture the midpoint of its jump, but the last frame shows the jump already completed. Therefore, we aim to identify the optimal edited frame, denoted ft∗, which corresponds to the earliest timestep t that achieves the desired edit. The transition from the initial frame f1 (or Is) to the final edited frame ft∗ motivates our method’s name, Frame2Frame.

#### 4. Editing Manifold Pathway

To illustrate the advantages of our method over conventional image-to-image approaches, we aim to visualize the editing process within the natural image manifold, where realistic images reside. Given the high dimensionality of this space, we project it into a lower-dimensional representation suitable for interpretation. To achieve this, we first generate three sets of images, each containing 200 samples, using the text-to-image generator FLUX.1-dev [26]. The sets are based on the following prompts:

- 1. AI: “Full-body portrait of a happy person wearing a shirt with the word ’AI’ on it”.
- 2. AI + Heart Hands: “Full-body portrait of a happy person wearing a shirt with the word ’AI’ on it and making a heart shape with their hands”.
- 3. Heart Hands: “Full-body portrait of a happy person making a heart shape with their hands”.

To automate the selection of t∗ and avoid manual frameby-frame review, we employ an automated approach. After generating the sequence V , we sample every fourth frame, imprinting each with a unique identifier and assembling them into an image collage alongside Is. Inspired by [23], which introduces a novel approach to video comprehension by transforming videos into image grids, we use a VLM, specifically GPT-4o, to assist in selecting t∗ by providing it with the collage and the editing prompt c. The VLM is tasked with identifying the frame that best fulfills the editing intent, evaluating each frame’s alignment with c and fidelity to Is. The model is instructed to select the optimal frame with the lowest index that completes the edit. An ablation study, detailed in Section C of the appendix, evaluates the effectiveness of our automated approach against the baseline of selecting the final frame of the video.

We manually filter the outcomes to ensure alignment with descriptions, with examples provided in the supplementary material in Section D. Then, we use a CLIP ViT-B/32 model [36] to extract a 512-dimensional feature vector for each image. Using these features, along with 25 feature vectors extracted from random noise images, we perform Principal Component Analysis (PCA) to reduce the dimensionality into a two-dimensional subspace. The resulting subspace, now suitable for visualization, is depicted in Figure 2.

As illustrated in the figure, the natural images form a smooth manifold, distinctly separated from the distribution of noise images. Within this manifold, there is a clear semantic progression: images of people with ’AI’ shirts (green cluster) are close to images of people with ’AI’ shirts making a heart shape (purple cluster), which are adjacent to images of people only making a heart shape (red cluster). Thus, transitioning smoothly along the manifold allows a person with an ’AI’ shirt to perform a heart shape with their hands while preserving the shirt’s text.

Note that while we refer to an optimal t∗, the definition of the required edit can be subjective and dependent on user preferences. For example, in Figure 3, different users may select frames of the cat jumping based on variations in its altitude. Thus, frame selection could serve as a flexible and customizable advantage in certain scenarios.

|Model|Source LPIPS↓ CLIP-I↑<br><br>|Target CLIP↑<br><br>|
|---|---|---|
|SDEdit Pix2Pix-ZERO Imagic LEDITS++ FlowEdit|0.30 0.85 0.29 0.84 0.52 0.86 0.23 0.87 0.22 0.89<br><br>|0.60 0.62 0.63 0.63 0.61|

###### F2F 0.22 0.89 0.63

Table 1. TEdBench Results. Quantitative comparison on TEdBench benchmark. Source metrics (LPIPS and CLIP-I) measure content preservation, while Target metric (CLIP) measures edit accuracy. Our Frame2Frame (F2F) method achieves better or comparable performance across all metrics.

Consider an original image from the AI group that we wish to edit using the target prompt: “A happy person making a heart shape with their hands”. Current editing methods generate a single image, which may cause an abrupt transition to the red cluster, effectively removing the ’AI’ on the shirt. This behavior is illustrated in the figure, where the edit by LEDITS++ is positioned near the red cluster, and the ’AI’ text on the shirt disappears. In contrast, our method leverages video generation to perform the edit smoothly, moving along the manifold with incremental changes until reaching the required edit in the purple cluster, as indicated by black arrows in Figure 2. The temporal editing caption guiding this process is: “A happy person very slowly raising their hands to form a heart shape”. This gradual progression lets us reach the purple cluster, resulting in an edited image where the person maintains the ‘AI’ on their shirt while making a heart shape—a faithful preservation of the original image’s key attributes. This experiment illustrates that the proposed paradigm enables smooth traversal across the image manifold, enabling consistent edits while preserving the essential characteristics of the original image.

#### 5. Experiments

We evaluate Frame2Frame against state-of-the-art image editing methods, including LEdits++ [4], SDEdit [31], Pix2Pix-Zero [35] Imagic [22], and FLUX-based FlowEdit [24]. Our evaluation spans two benchmarks: the established TEdBench [22] for general image editing, and our newly introduced PosEdit, specifically designed for human pose editing. Finally, we conduct a human evaluation to assess our method based on real user preferences.

##### 5.1. Evaluation Protocol

We conduct our experiments following a consistent evaluation protocol across all methods and benchmarks. Following common practice [4, 22], for each method and source image, we manually select the best result from fifteen ran-

|Model<br><br>|Source LPIPS↓ CLIP-I↑<br><br>|Target<br><br>LPIPS↓ CLIP-I↑CLIP↑|
|---|---|---|
| | | |

GT (Reference) 0.08 0.91 0 1 0.61 SDEdit 0.39 0.61 0.39 0.64 0.57 Pix2Pix-ZERO 0.39 0.57 0.40 0.60 0.56 LEDITS++ 0.26 0.65 0.28 0.69 0.64 F2F 0.14 0.82 0.15 0.84 0.64

Table 2. PosEdit Results. Quantitative evaluation on PosEdit. Source metrics assess similarity to the original image, while Target metrics include LPIPS and CLIP-I comparisons to the groundtruth target image, along with the CLIP score for edit accuracy. “GT” provides ground-truth target image metrics for context.

dom seeds based on visual quality and edit accuracy, ensuring the same seed set is used across all methods. For all methods, we use the default hyperparameters and settings as provided in their official implementations or official Hugging Face repositories 1.

Image Preprocessing and Generation Details. We use CogVideoX [48] as our video generation backbone, which operates at a fixed resolution of 720×480 pixels. Both TEdBench and PosEdit benchmarks consist of images with 1:1 aspect ratio. To accommodate CogVideoX’s resolution requirements while preserving image content, we first resize all source images to 480×480 pixels, then horizontally pad them with black pixels to reach the required 720 × 480 resolution (adding 120 pixels on each side). After generating the video sequence, we crop the central 480 × 480 region of the selected frame to remove the padding and resize it to the final evaluation resolution of 512×512 pixels, matching the standard resolution used in prior work. For video generation, we adopt the default hyperparameters proposed by CogVideoX: a guidance scale of 6, 49 generated frames per sequence, and 50 denoising inference steps. This configuration generates videos approximately 6 seconds in duration at a frame rate of 8 frames per second.

Method-Specific Requirements. Pix2Pix-Zero [35] requires additional source image descriptions along with the standard inputs. We generate these automatically using BLIP-2 [27], a state-of-the-art image captioning model, ensuring consistent source descriptions across all experiments. For our method, we transform the original editing prompts from both benchmarks into temporal editing captions, as described in Section 3.1.

##### 5.2. TEdBench Evaluation Results

We quantitatively evaluate our method and baselines on the TEdBench benchmark using three complementary met-

1LEdits++, Pix2Pix-Zero, SDEdit

Pix2Pix Zero

Source SDEdit Imagic LEDITS++ FlowEdit F2F

Target Edit

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

“A photo of a cat in a grass field.”

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

“A photo of an open door.”

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

“A torn up teddy bear.”

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

“A photo of a bird leaning down.”

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

“A photo of a chair sawed in half.”

- Figure 4. Qualitative Results on TEdBench. Comparison with other methods across various editing tasks. Our approach consistently produces edits that better align with the target prompt while preserving the source image’s content and structure. For instance, in the teddy bear example, our method uniquely achieves complex structural modifications while maintaining high visual quality.

rics. For each method, we compute the metrics between the source image and its corresponding best edited version. LPIPS [50] measures perceptual similarity to assess how much the edit preserves the source image’s content, with lower values indicating better preservation. CLIP-I evaluates the similarity between source and edited images in CLIP’s [36] feature space, where higher values indicate better preservation of semantic content. Finally, CLIP score measures alignment between the edited image and the target prompt, with higher values indicating better adherence to the editing instruction. As shown in Table 1, our method achieves strong performance across all metrics, demonstrating effective balance between preserving source content and achieving the desired edit.

The qualitative advantages of our method are visually evident in the comparisons presented in Figure 4. The figure demonstrates our method’s superior performance across diverse editing scenarios, producing results that both faithfully execute the intended edits while maintaining strong alignment with the source image. Additional examples showcasing these capabilities are provided in the appendix in Section A.

##### 5.3. PosEdit Benchmark

We introduce PosEdit, a benchmark for human pose editing derived from the UTD-MHAD dataset [6]. UTD-MHAD includes RGB videos of 8 subjects (4 females and 4 males)

performing predefined actions in a controlled indoor environment. We carefully curated 58 editing tasks encompassing 8 distinct action categories, ranging from simple poses like a raised hand to complex athletic poses such as basketball shooting and lunging.

The proposed benchmark specifically focuses on human pose manipulation. Unlike TEdBench, each editing task in PosEdit includes a ground truth frame extracted from the same subject in the target pose. The availability of reference images allows for a more comprehensive evaluation of both editing accuracy and identity preservation. In the dataset, each editing task consists of two images: a source image showing the subject in a neutral standing pose with arms relaxed at their sides, and a ground-truth edited target image capturing the subject performing in a specific pose. The benchmark also provides a prompt for each task that describes the target pose the subject should achieve (e.g., ”A person in a basketball shooting posture.”).

Benchmark Evaluation. The evaluation on PosEdit follows the same protocol established for TEdBench: generating multiple variants using nine different seeds and manually selecting the best result for each method. However, PosEdit’s ground-truth target images enable additional evaluation metrics. Beyond measuring preservation of source content (via LPIPS and CLIP-I between source and edited images) and edit accuracy (via CLIP score with the prompt),

Source LEDITS++ F2F Ground

Truth

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

"A person performing a squat."

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

"A person jogging.”

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

"A person making a basketball shooting posture.”

- Figure 5. Qualitative Results on PosEdit. Comparison between our Frame2Frame method and LEDITS++ on human motion editing tasks. For each example, we show the source image, edited results from both methods, and the ground-truth target image. Our method better preserves subject identity while achieving more natural pose transitions. The evaluation metrics for each image are provided in Section F of the appendix.

we compute LPIPS and CLIP-I metrics between the edited image and its corresponding ground-truth target. This enhanced evaluation directly assesses both the accuracy of the transformation and preservation of identity features. Additionally, we include the evaluation metrics of the groundtruth images as a reference to the rest of the results.

Table 2 demonstrates that our method consistently outperforms competitive methods across all metrics, with particularly strong performance in similarity measures with the target ground-truth image. This advantage highlights our method’s superior ability to preserve key features while achieving the desired edit, as illustrated by the qualitative comparisons in Figure 5. The figure shows that our method generates more natural pose transitions while maintaining crucial identity attributes such as facial features, body proportions, and clothing details.

Two important implementation notes: First, for Pix2PixZero, which requires source image descriptions, we use the same static prompt across all tasks: ”A person standing naturally with his arms relaxed at his sides.” Second, we exclude Imagic [22] from this comparison as their official implementation is not publicly available and their results are only reported on TEdBench.

##### 5.4. Human Evaluation Survey

To further evaluate our method, we conducted a human survey. As indicated in Table 1, LEDITS++ emerges as the most competitive method compared to ours, making it a natural choice for comparison. Each participant was presented with 20 randomly sampled TEdBench samples, including the source image, the target edit prompt, and the corresponding edited outputs from both methods. Inspired

|Method<br><br>|Edit Accuracy Overall Per-Image<br><br>|Edit Quality<br><br>Overall Per-Image|
|---|---|---|
| | | |

F2F 54.1% 53.0% 65.6% 67.0% LEDITS++ 45.9% 47.0% 34.4% 33.0%

Table 3. Human Survey Results. Human evaluation on TEdBench shows that F2F surpasses LEDITS++ in edit accuracy while offering a significant advantage in preserving the original image.

by the methodology proposed by [49], participants evaluated each comparison based on two criteria: (1) edit accuracy relative to the prompt, and (2) edit quality—defined as the preservation of visual fidelity to the source image, seamless integration of edited elements, and the overall natural appearance of modifications. We collected responses from 59 randomly selected online evaluators. To ensure an unbiased assessment, evaluators were not informed of the study’s objectives. The results were quantified using two metrics: (i) overall global preference, expressed as a percentage, and (ii) aggregated per-image preference, where a tie resulted in each method receiving 0.5 points.

Our results, summarized in Table 3, demonstrate that F2F outperforms LEdits++ on both metrics. Specifically: For edit quality, F2F achieved a global preference score of 53%, slightly surpassing the 47% obtained by LEdits++. In terms of edit accuracy, F2F achieved a 65.6% global preference compared to LEdits++’s 34.4%. The per-image preference results mirrored this trend, indicating robustness across individual examples and no significant influence of outliers. These findings reinforce our claim that smooth temporal editing—using video as a medium—preserves essential scene characteristics while successfully performing the edit. Our results suggest that the gap between F2F and LEdits++ in source preservation is larger than reflected by the LPIPS scores in Table 1. The full survey details are provided in the supplementary materials in Section G.

##### 5.5. Additional Vision Tasks

We demonstrate our framework’s applicability beyond traditional editing by applying it to fundamental image manipulation tasks: denoising, deblurring, outpainting, and relighting. For these experiments, we utilize Runway Gen-3 [13] as our video generation backbone, as it showed superior performance on these specific tasks compared to other generative video models. Since these tasks serve as extensions and potential applications of our method, we focus on qualitative results, leaving quantitative evaluation for future work. Our results, shown in Figure 6, demonstrate strong performance across all tasks. We associate this success to the natural alignment between these operations and common video scenarios: deblurring maps to camera focusing, outpainting to camera motion, and relighting to time-lapse lighting changes. This alignment enables our

Original Denoising Deblurring Outpainting Relighting

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

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

Figure 6. Additional Vision Tasks. Qualitative results of our image-to-video-to-image editing approach on selected traditional tasks.

method to leverage the video model’s learned temporal dynamics, achieving high-quality results without task-specific training. Full prompts are provided in Section I of the supplementary material.

Extended Results. Additional examples illustrating our method’s editing performance across diverse backgrounds are provided in Section H of the Appendix.

#### 6. Limitations

While our approach addresses key shortcomings in current image editing methods, it also introduces unique challenges. For example, natural camera motion sometimes appears in video sequences, which, when replicated in generated content, can lead to unintended perspective shifts. As with all generative models, video models are trained on specific data domains, making it challenging to produce results that deviate significantly from the model’s training data, which predominantly includes real-world transformations. Despite these challenges, F2F demonstrates success in several cases. For example, in the middle row of Figure 1, the vase is ”magically” filled with green water, showcasing the model’s ability to perform imaginative edits, with additional examples discussed in Section H of the supplementary. Additionally, our method is computationally intensive, as transforming an image into a video sequence is resource-heavy and often slower than other image editing methods. However, video generation efficiency is advancing rapidly, as demonstrated by models like Runway Turbo2 and LTX-Video [17], which can execute the process in

2Runway Turbo API

seconds, making this approach increasingly less resourceintensive. Additionally, optimizing the number of frames per edit—currently fixed at 49 for CogVideoX—could enable faster editing possibilities.

#### 7. Conclusions

We introduced Frame2Frame, a novel approach that reformulates image editing through video generation. By leveraging video models’ inherent understanding of temporal transformations, our method achieves state-of-the-art editing results while maintaining high fidelity to source images. We demonstrated our framework’s effectiveness on standard benchmarks, introduced PosEdit for human pose editing, and showed promising results on more classical vision tasks. As video generation technology advances, we expect our approach to enable increasingly sophisticated image manipulations while maintaining natural and physically plausible results.

Future Research. Our work opens several promising directions. First, fine-tuning video generators specifically for image editing presents an exciting opportunity, including straightforward solutions like enforcing static camera scenarios or using datasets curated for editing tasks, alongside more complex approaches yet to emerge. Secondly, a key direction could involve reducing the overhead of full video generation while preserving the benefits of gradual, temporally coherent transformations, potentially enhancing both the efficiency and speed of editing. The speed can be affected both in terms of per-frame average generation speed or the number of frames generated per edit, which is currently fixed at 49 for CogVideoX.

#### References

- [1] Hadi Alzayer, Zhihao Xia, Xuaner Zhang, Eli Shechtman, Jia-Bin Huang, and Michael Gharbi. Magic fixup: Streamlining photo editing by watching dynamic videos. arXiv preprint arXiv:2403.13044, 2024. 3

- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2

- [3] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 2

- [4] Manuel Brack, Felix Friedrich, Katharina Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting, and Apolin´ario Passos. Ledits++: Limitless image editing using text-to-image models. 2023. 5
- [5] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 2

- [6] Chen Chen, Roozbeh Jafari, and Nasser Kehtarnavaz. Utdmhad: A multimodal dataset for human action recognition utilizing a depth camera and a wearable inertial sensor. In 2015 IEEE International conference on image processing (ICIP), pages 168–172. IEEE, 2015. 2, 6

- [7] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331, 2024. 2

- [8] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6593–6602, 2024. 3

- [9] Aidan Clark, Jeff Donahue, and Karen Simonyan. Adversarial video generation on complex datasets. In ICCV, 2019. 2

- [10] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning,

2023. 3

- [11] Duolikun Danier, Fan Zhang, and David Bull. Ldmvfi: Video frame interpolation with latent diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, 2024. 2

- [12] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Tianyu

- Liu, et al. A survey on in-context learning. arXiv preprint arXiv:2301.00234, 2022. 3
- [13] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models,

2023. 7

- [14] Roy Ganz, Yair Kittenplon, Aviad Aberdam, Elad Ben Avraham, Oren Nuriel, Shai Mazor, and Ron Litman. Question aware vision transformer for multimodal reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13861–13871, 2024. 3

- [15] Yue Gao, Yuan Zhou, Jinglu Wang, Xiao Li, Xiang Ming, and Yan Lu. High-fidelity and freely controllable talking head video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5609–5619, 2023. 2

- [16] David Ha and J¨urgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018. 2

- [17] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103,

2024. 8

- [18] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 2

- [19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2

- [20] Yi Huang, Jiancheng Huang, Yifan Liu, Mingfu Yan, Jiaxi Lv, Jianzhuang Liu, Wei Xiong, He Zhang, Shifeng Chen, and Liangliang Cao. Diffusion model-based image editing: A survey. arXiv preprint arXiv:2402.17525, 2024. 2

- [21] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly ddpm noise space: Inversion and manipulations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12469–12478, 2024. 2

- [22] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023. 1, 2, 5, 7

- [23] Wonkyun Kim, Changin Choi, Wonseok Lee, and Wonjong Rhee. An image grid can be worth a video: Zeroshot video question answering using a vlm. arXiv preprint arXiv:2403.18406, 2024. 4, 12

- [24] Vladimir Kulikov, Matan Kleiner, Inbar HubermanSpiegelglas, and Tomer Michaeli. Flowedit: Inversion-free text-based editing using pre-trained flow models. arXiv preprint arXiv:2412.08629, 2024. 5

- [25] Jeong-gi Kwak, Erqun Dong, Yuhe Jin, Hanseok Ko, Shweta Mahajan, and Kwang Moo Yi. Vivid-1-to-3: Novel view synthesis with video diffusion models. In Proceedings of

- the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6775–6785, 2024. 2
- [26] Black Forest Labs. Flux.1-dev. https://github.com/ black-forest-labs/FLUX.1-dev, 2024. 4
- [27] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 5

- [28] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177, 2024. 2

- [29] Grace Luo, Trevor Darrell, Oliver Wang, Dan B Goldman, and Aleksander Holynski. Readout guidance: Learning control from diffusion features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8217–8227, 2024. 3

- [30] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 2

- [31] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022. 5

- [32] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023. 1, 2

- [33] OpenAI. Chatgpt-4o (october 2024 version). https:// chat.openai.com/chat, 2024. 3
- [34] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-toimage translation. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023. 2

- [35] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-toimage translation. In ACM SIGGRAPH 2023 Conference Proceedings, New York, NY, USA, 2023. Association for Computing Machinery. 5

- [36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 4, 6

- [37] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2

- [38] Noam Rotstein, David Bensa¨ıd, Shaked Brody, Roy Ganz, and Ron Kimmel. Fusecap: Leveraging large language models for enriched fused image captions. In Proceedings of the

- IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5689–5700, 2024. 3
- [39] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 2

- [40] Yujun Shi, Jun Hao Liew, Hanshu Yan, Vincent YF Tan, and Jiashi Feng. Instadrag: Lightning fast and accurate dragbased image editing emerging from videos. arXiv preprint arXiv:2405.13722, 2024. 3

- [41] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, et al. Text-to-4d dynamic scene generation. arXiv preprint arXiv:2301.11280,

2023. 2

- [42] Nitish Srivastava, Elman Mansimov, and Ruslan Salakhudinov. Unsupervised learning of video representations using lstms. In International conference on machine learning. PMLR, 2015. 2

- [43] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. Mocogan: Decomposing motion and content for video generation. In CVPR, 2018. 2

- [44] Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. Generating videos with scene dynamics. In NeurIPS, 2016. 2

- [45] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023. 3

- [46] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models, 2024. 3
- [47] Navve Wasserman, Noam Rotstein, Roy Ganz, and Ron Kimmel. Paint by inpaint: Learning to add image objects by removing them first. arXiv preprint arXiv:2404.18212,

2024. 2

- [48] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2, 3, 5

- [49] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. Advances in Neural Information Processing Systems, 36, 2024. 7, 14

- [50] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6

- [51] Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y Feng, Changxi Zheng, Noah Snavely, Jiajun Wu, and William T Freeman. Physdreamer: Physics-based interaction with 3d objects via video generation. In European Conference on Computer Vision. Springer, 2025. 2

## Appendix

#### A. Additional Editing Examples

We provide additional qualitative examples of the editing results on the TEdBench benchmark, generated based on the experimental setup detailed in Section 5.2. These examples further supplement those presented in Figures 1 and 4. Each example is accompanied by the temporal editing caption used to perform the edit, which was generated using the method described in Section 3.1. Furthermore, in Figure S9, we supplement Figure 1 with additional video examples generated by our method, illustrating the transition from the source image (left) to the target edit (right).

|Source Image|Target Editing Caption|Temporal Editing Caption|Edited Image|
|---|---|---|---|
|[Figure 121]|A photo of a vase of red roses.|The carnations slowly transform into vibrant red roses.|[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]|
|[Figure 129]|Pizza with pepperoni.|Pepperoni slices are carefully placed on the pizza.| |
|[Figure 130]|A photo of a couple holding their hands on a beach.|The couple on the beach gradually moves closer to each other and holds hands.| |
|[Figure 131]|A photo of a tree. Taken in autumn.|The tree's green leaves gradually turn vibrant shades of red and orange.| |
|[Figure 132]|A photo of a traffic jam.|Cars begin to fill the onceclear highway, gradually slowing to a standstill.| |
|[Figure 133]|A person in a greetting pose to Namaste hands.|The person's hands slowly move together into a Namaste position.| |
|[Figure 134]|A photo of a sitting dog.|The dog gradually lowers its back legs, settling into a seated position.| |

|Source Image|Target Editing Caption|Temporal Editing Caption|Edited Image|
|---|---|---|---|
| | | | |
|[Figure 135]|A goat jumping over a cat.|The goat leaps gracefully into the air over the unmoving cat.|[Figure 136]|
|[Figure 137]|A photo of a car in Manhattan.|The car drives through a bustling Manhattan street.|[Figure 138]|
|[Figure 139]|A girl riding a horse.|The horse lowers its head, revealing a girl seated on its back.|[Figure 140]|
|[Figure 141]|A teddy bear holding a cup.|The teddy bear gently lifts a cup into its arm.|[Figure 142]|
|[Figure 143]|A photo of an open box.|The flaps of the box gently unfold, revealing the open interior.|[Figure 144]|
|[Figure 145]|A photo of a giraffe eating the grass below.|The giraffe gently lowers its neck to nibble on the grass.|[Figure 146]|
|[Figure 147]|Two bananas.|A second banana gently rolls into view, coming to rest beside the first banana.|[Figure 148]|
|[Figure 149]|A cyclist riding in a street.|A cyclist gradually emerges, pedaling along the cobblestone street.|[Figure 150]|
|[Figure 151]|A white horse in a grass field.|The snow beneath the white horse melts, revealing a lush grass field.|[Figure 152]|

Figure S7. TEdBench Editing Examples.

Figure S8. TEdBench Editing Examples.

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

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

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

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

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

Figure S9. Generated Video Examples. Additional video sequences illustrating the temporal evolution from the source image to the target edit.

#### B. Temporal Editing Captions

##### B.1. VLM Instruction

As outlined in Section 3.1, we propose a framework for automatically generating the temporal editing caption by leveraging the original target editing prompt in conjunction with the source image. The instruction given to the VLM, along with the source image is:

’Write a one-sentence description of a short video that begins with the provided image and smoothly transitions into a scene of a ”CAPTION”, highlighting how elements in the image undergo changes or movement over time. Keep the description simple, concise and short, focusing only on essential changes and actions without altering unnecessary details. Avoid mentioning elements that do not contribute to the main change needed, and focus the description on the main transitions. Do not add objects that are not in the original image or described in the final scene. The camera should remain static unless movement is absolutely necessary. Ensure all transitions happen within a few second duration without mentioning the length or using the word ”video”.’

Here, ”CAPTION” is replaced with the target caption specific to the image. Additionally, as explained, in-context learning is employed to provide the VLM with examples alongside the instruction. Before processing the desired

source image and edit prompt, the instruction is presented to the VLM nine times, each paired with a distinct example consisting of a source image, target caption, and corresponding temporal editing caption. Examples of these are illustrated in Figure S10.

| |Target Editing Caption|Temporal Editing Caption|
|---|---|---|
|[Figure 234]|A photo of domes in the Grand Canyon under the golden sunrise.|The sun rises slowly in the early morning.|
|[Figure 235]|Two red Campari shots|Red-colored Campari is poured into shot glasses until they are filled.|
|[Figure 236]|A photo of a magician holding a hat with a rabbit in it.|A magician slowly pulls a rabbit from his hat and reveals it gradually.|

Figure S10. In Context Learning Examples.

##### B.2. Ablation

To assess the impact of the Temporal Editing Caption, we conduct an ablation experiment comparing its use against directly using the target editing captions from the TEdBench benchmark. Apart from this modification, we adhere to the same protocols as described in the original experiment in Section 5.2. As shown in Table S4, this setup preserves a similar resemblance to the source image but underperforms in terms of image editing performance.

|Model<br><br>|Source LPIPS↓ CLIP-I↑|Target CLIP↑<br><br>|
|---|---|---|
|Original Captions Temporal Captions|0.21 0.89<br><br>0.22 0.89<br><br><br>|0.60 0.63|

Table S4. Temporal Editing Captions Ablation.

#### C. Frame Selection C.1. VLM Instruction

As detailed in Section 3.3, our method selects the frame that best aligns with the intended edit from each generated video. To automate this process, inspired by [23], we create a collage of uniformly sampled frames from the video, along with the source image and target editing caption, and prompt a VLM to identify the optimal frame. The model is

instructed to select the earliest frame (i.e., with the lowest index) that satisfies the editing intent, minimizing deviation from the original image. In both this process and the best seed selection process (applied across all methods), if none of the edited frames successfully fulfill the desired edit, the original image is retained as the final output.

The instruction provided to the VLM is:

‘The image displays the source photo at the top, with a collage of 12 edited versions beneath it. The target edit image caption was: “CAPTION”. Your task is to choose the image from 1 to 12 that best follows this edit fully and naturally. If none of the images follows the edit, select image 0. If multiple images follow the edit equally, prioritize the one with the lowest number possible. Avoid selecting images that appear to follow the edit but are not edits of the original image. Additionally, avoid images where camera motion, zoom, or image quality differs significantly, or where the content does not appear stable relative to the original source. Respond with: “The selected edit is:x” where x is the number of your chosen edit.’

Here, ”CAPTION” refers to the target editing caption. Examples of the collages can be found in Figure S11.

[Figure 237]

Figure S11. Frame Selection Collage. The target editing caption for this example is: “A photo of a cat yawning.”.

##### C.2. Ablation

To validate the effectiveness of our approach, we compare it to the naive solution of using the last frame of the generated video as the edited output. The evaluation follows the same protocol described in Section 5.2. As can be seen in Table S5, this naive approach results in a lower target CLIP

score for the edited outputs, highlighting the advantages of our method.

|Model<br><br>|Source LPIPS↓ CLIP-I↑<br><br>|Target CLIP↑|
|---|---|---|
|Last Frame Selected Frame|0.24 0.9<br><br>0.22 0.89<br><br>|0.61 0.63|

Table S5. Frame Selection Ablation.

#### D. Editing Manifold Pathway

As elaborated in Section 4, to simulate the image manifold, we generated 200 images across three distinct categories using FLUX.1-dev. Examples of these generated images are shown in Figure S12.

AI Shirt

Heart Hands

AI + Heart

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

Figure S12. Flux.1-dev Generations

#### E. Inference Hyperparameters

We use the 5B-parameter image-to-video version of CogVideoX, ensuring consistency by applying the same inference hyperparameters across all experiments. The model generates a fixed 49 frames per sample. During inference, we use the default denoising scheduler in CogVideoX, which is based on DDIM with V-prediction. We perform 50 denoising steps and set the classifier-free guidance scale for text conditioning to 6.0.

#### F. PosEdit

In Section 5.3, we outline the construction of a human pose editing dataset. This dataset encompasses 58 editing tasks, distributed across 8 distinct action categories, featuring 8 different subjects. The source images consistently depict a neutral standing pose with arms relaxed at the sides, while the target poses vary according to the edit category. Each editing category is paired with a target caption and a temporal caption. Figure S13 and Figure S14 illustrates examples for each action category. Additionally, in Figure S15, we complement Figure 5 with quantitative results, demonstrating how the numerical evaluation aligns with its intended measurement objectives.

|Source Image|Target Editing Caption| |Temporal Editing Caption|Ground Truth Target Image|
|---|---|---|---|---|
|[Figure 250]|A person holding their bent arms with closed hands near their face.|A person raises their hands to their face, bending their arms and closing their hands| |[Figure 251]|
|[Figure 252]|A person performing a squat.|A person bending their knees and lowering into a squat.| |[Figure 253]|
|[Figure 254]|A person is lunging with their right foot forward.|A person stepping smoothly into a lunge with their right foot forward.| |[Figure 255]|
|[Figure 256]|A person making a basketball shooting posture.|A person raises their arms and mimics a basketball shot.| |[Figure 257]|

- Figure S13. PosEdit Examples.

|Source Image|Target Editing Caption| |Temporal Editing Caption|Ground Truth Target Image|
|---|---|---|---|---|
|[Figure 258]|A person stretches their left hand to the left.|A person raises their left hand and stretches it to the left| |[Figure 259]|
|[Figure 260]|A person joging|A person mimicking a jogging motion by slowly raising their legs.| |[Figure 261]|
|[Figure 262]|A person clapping their hands|A person smoothly brings their hands together to start clapping.| |[Figure 263]|
|[Figure 264]|A person making a basketball shooting posture.|A person raises their arms and mimics a basketball shot.| |[Figure 265]|

- Figure S14. PosEdit Examples.

Source LEDITS++ F2F

Ground Truth

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

Source

Source

Source

LPIPS CLIP-I

0.28 0.66

LPIPS CLIP-I

0.17 0.78

LPIPS CLIP-I

0.10 0.86

"A person performing a squat."

Target

Target

Target

LPIPS CLIP-I CLIP

0.29 0.71 0.68

LPIPS CLIP-I CLIP

0.14 0.83 0.70

LPIPS CLIP-I CLIP

-

[Figure 270]

0.53

[Figure 271]

[Figure 272]

[Figure 273]

Source

Source

Source

LPIPS CLIP-I

0.38 0.65

LPIPS CLIP-I

0.13 0.80

LPIPS CLIP-I

0.08 0.88

"A person jogging.”

Target

Target

Target

LPIPS CLIP-I CLIP

0.39 0.65 0.62

LPIPS CLIP-I CLIP

0.11 0.90 0.81

LPIPS CLIP-I CLIP

0.51

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

"A person making a basketball shooting posture.”

Source

Source

Source

LPIPS CLIP-I

0.18 0.59

LPIPS CLIP-I

0.13 0.80

LPIPS CLIP-I

0.11 0.88

Target

Target

Target

LPIPS CLIP-I CLIP

0.19 0.72 0.64

LPIPS CLIP-I CLIP

0.11 0.90 0.81

LPIPS CLIP-I CLIP

0.79

Figure S15. PosEdit Quantitative Evaluation Examples.

#### G. Human Survey

As detailed in Section 5.4, we conducted a human evaluation survey to assess our method’s performance based on real user preferences. Following the framework in [49], the survey questions evaluated (1) the accuracy of the edit relative to the prompt and (2) the quality of the edit, defined as the preservation of visual fidelity to the source image. Each participant reviewed 20 edits, comparing our method with LEDITS++. Examples of the pages shown to the evaluators are provided in Figures S16 and S17.

[Figure 278]

Figure S16. Survey Example.

[Figure 279]

Figure S17. Survey Example.

#### H. Further Ablations

We present additional ablation results demonstrating the ability of our method to handle two distinct editing challenges: (1) diverse backgrounds and (2) out-of-videodistribution edits.

Backgrounds. Similar to our image generation approach in Section 4, we generate eight images of two objects with varying backgrounds. Each image generation prompt follows one of the two templates below, where LOCATION is replaced with one of the following settings: colorful amusement park, sandy beach, magical fairy tale forest, futuristic cityscape, old library, snowy mountain peak, bustling train station, and quaint village square.

- 1. “A small brown teddy bear sitting in LOCATION.”
- 2. “A tennis ball lies in LOCATION.” Out-of-video-distribution editing. As discussed in Section 6, one might assume that our method would fail to perform edits requiring temporal transformations that deviate significantly from typical real-world videos (the model’s training set). To show that this is not necessarily the case, we selected two unreal editing processes, using the following temporal captions:

- 1. “The teddy bear slowly rips, revealing stuffing coming out.”
- 2. “The tennis ball gradually transforms into a ripe red tomato.”

These edits depict unrealistic events that do not ordinarily occur in real-life footage. However, as shown in Fig. S18, our method successfully handles both the background variations and the out-of-distribution nature of these transformations. Moreover, the full temporal sequences in Fig. S9 demonstrate that these edits occur seamlessly without any external interaction (the teddy bear rips apart spontaneously, and the tennis ball morphs magically into a tomato).

|[Figure 280]<br><br>[Figure 281]<br><br>[Figure 282]<br><br>[Figure 283]<br><br>[Figure 284]<br><br>[Figure 285]<br><br>[Figure 286]<br><br>[Figure 287]<br><br>[Figure 288]<br><br>[Figure 289]<br><br>[Figure 290]<br><br>[Figure 291]<br><br>[Figure 292]<br><br>[Figure 293]<br><br>[Figure 294]<br><br>[Figure 295]<br><br>[Figure 296]<br><br>[Figure 297]<br><br>[Figure 298]<br><br>[Figure 299]<br><br>[Figure 300]<br><br>[Figure 301]<br><br>[Figure 302]<br><br>[Figure 303]<br><br>[Figure 304]<br><br>[Figure 305]<br><br>[Figure 306]<br><br>[Figure 307]<br><br>[Figure 308]<br><br>[Figure 309]<br><br>[Figure 310]<br><br>[Figure 311]|
|---|

Figure S18. Ablation Examples.

#### I. Additional Vision Tasks Captions

As outlined in Section 5.5 and demonstrated in Figure 6, we showcase our framework’s applicability for additional, more classic vision tasks that are not typically classified as image editing. For these tasks, we employ Runway Gen3 as our video generator. Empirically, these tasks required longer and more descriptive captions. The temporal editing captions used for each task are as follows:

- 1. Relighting: ’The scene’s lighting shifts gradually, changing to night. The sun is setting, and artificial lights replace it. The camera is static. Time-lapse. Cinematic.’
- 2. Outpainting: ’The image expands, adding new surroundings seamlessly beyond the original frame.’
- 3. Denoising: ’The image clears up as noise fades away, revealing smoother, cleaner details.’
- 4. Debluring: ’The camera comes into focus, revealing sharp details and enhanced clarity, as though a camera lens has adjusted perfectly. Nothing moves. Static image.’

