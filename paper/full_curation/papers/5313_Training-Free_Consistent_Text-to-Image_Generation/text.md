# arXiv:2402.03286v3[cs.CV]30May2024

### Training-Free Consistent Text-to-Image Generation

YOAD TEWEL, NVIDIA, Israel and Tel Aviv University, Israel OMRI KADURI, Independent Scientist, Israel RINON GAL, NVIDIA, Israel and Tel Aviv University, Israel YONI KASTEN, NVIDIA, Israel LIOR WOLF, Tel Aviv University, Israel GAL CHECHIK, NVIDIA, Israel YUVAL ATZMON, NVIDIA, Israel

“walking in the park”

“writing numbers on a blackboard”

“drinking a beer in the bar” “feeding a stray cat”

“eating dinner in a restaurant”

Subject Description

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

“A photo of an old man wearing a hat.”

SingleSubjectMultiSubject

“wearing a school uniform”

“walking with his mom” “reading a book” “climbing a tree” “eating his food”

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

“An old story illustration of a kid.”

“in the snow, wearing

“having a countryside picnic with her cat”

“hiking with her cat, in the mountains”

“wearing headphones, with her cat”

headphones” “cooking some food”

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

“A hyper-realistic digital painting of a happy girl, brown eyes.”

Fig. 1. ConsiStory transforms a set of input prompts with recurring subjects into a series of images that maintain the same subject identity and adhere to the provided text. It can also maintain consistent identities for multiple subjects. Importantly, ConsiStory does not involve any optimization or pre-training.

Text-to-image models offer a new level of creative flexibility by allowing users to guide the image generation process through natural language. However, using these models to consistently portray the same subject across diverse prompts remains challenging. Existing approaches fine-tune the model to teach it new words that describe specific user-provided subjects or add image conditioning to the model. These methods require lengthy persubject optimization or large-scale pre-training. Moreover, they struggle to align generated images with text prompts and face difficulties in portraying multiple subjects. Here, we present ConsiStory, a training-free approach that enables consistent subject generation by sharing the internal activations of the pretrained model. We introduce a subject-driven shared attention block and correspondence-based feature injection to promote subject consistency

between images. Additionally, we develop strategies to encourage layout diversity while maintaining subject consistency. We compare ConsiStory to a range of baselines, and demonstrate state-of-the-art performance on subject consistency and text alignment, without requiring a single optimization step. Finally, ConsiStory can naturally extend to multi-subject scenarios, and even enable training-free personalization for common objects.

Code will be available at our project page.

Additional Key Words and Phrases: Text-to-Image, Consistent, Story, Diffusion, Training-Free

###### ACM Reference Format:

Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, and Yuval Atzmon. 2024. Training-Free Consistent Text-to-Image Generation. ACM Trans. Graph. 43, 4, Article 52 (July 2024), 18 pages. https://doi.org/10. 1145/3658157

Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for third-party components of this work must be honored. For all other uses, contact the owner/author(s). © 2024 Copyright held by the owner/author(s). 0730-0301/2024/7-ART52 https://doi.org/10.1145/3658157

1 INTRODUCTION

Large-scale text-to-image (T2I) diffusion models empower users to create imaginative scenes from text, but their stochastic nature

poses challenges when trying to portray visually consistent subjects across an array of prompts. Such consistency is crucial for many applications: from illustrating books and stories, through designing virtual assets, to creating graphic novels and synthetic data.

In the field of consistent image generation, current approaches [Avrahami et al. 2023b; Feng et al. 2023; Jeong et al. 2023; Liu et al. 2023] predominantly rely on personalization, a process where the text-to-image model learns a new word to represent a specific subject in a given image set. However, these personalization-based methods suffer from several drawbacks: They require per-subject training; they struggle to portray multiple consistent subjects simultaneously in one image; and they can suffer from trade-offs between subject consistency and prompt-alignment. Alternatives, like training image-conditioned diffusion models (e.g. using an encoder [Gal et al. 2023; Wei et al. 2023; Ye et al. 2023]), require significant computational resources, and their extension to multi-object scenes remains unclear. A common thread in all these approaches is that they attempt to enforce consistency a posteriori. That is, they operate to make generated images consistent with a specific, given target. Such approaches have two drawbacks. They are bound to constrain the model’s “creativity" to the given target image, and they tend to drive the model away from its training distribution.

We show here that the limitations of a posteriori methods can be avoided, and propose a way to achieve consistency in a zero-shot manner - without conditioning on existing images. The key idea is to promote cross-frame consistency a priori during generation. To achieve this, we leverage the internal feature representations of the diffusion model to align the generated images with each other, without any need to further align them with an external source. In doing so, we can enable on-the-fly consistent generation (Fig. 1), without requiring lengthy training or backpropagation, making generation roughly ×20 faster than the current state-of-the-art.

Our approach operates in three steps. First, we localize the subject across a set of noisy generated images. We then encourage subject consistency by allowing each generated image to attend to subject patches in other frames via an extension of the self-attention mechanism. This leads to more consistent subjects across the batch but causes the layout diversity to greatly diminish, as observed in other contexts that use similar extension [Hertz et al. 2023]. Our second step is, therefore, to maintain diversity in two ways: by incorporating features from a vanilla, non-consistent sampling step, and by introducing a new inference-time dropout on the shared keys and values. Finally, we aim to enhance consistency in finer details. To achieve this, we align the self-attention output features between corresponding subject pixels across the entire set.

Our full method, which we term ConsiStory, combines these components to enable training-free consistent generation. We compare ConsiStory to prior approaches and demonstrate that by aligning features during the generative process, we not only substantially speed up the process, but also maintain better prompt-alignment. Importantly, our method is trivial to extend to multi-subject scenes, avoiding pitfalls introduced by personalization-based approaches.

Finally, we show that ConsiStory is compatible with existing editing tools like ControlNet [Zhang et al. 2023b], we introduce methods for re-using the consistent identities, and even apply our ideas to

training-free personalization for common object classes, being the first to show training-free personalization, with no encoder use.

In summary, this paper makes the following contributions: First, we present a training-free method for achieving subject consistency across varying prompts. Second, we develop new techniques to combat layout collapse in extend-attention applications. Additionally, we share a new benchmark dataset for consistency evaluation.

2 RELATED WORK

Consistent T2I generation. is the task of synthesizing a set of images that portray visually consistent subjects. Early works utilized extensive fine-tuning and personalization [Gal et al. 2022; Ruiz et al.

- 2022] to promote consistency. [Jeong et al. 2023] replaces a character face using a personalized model and image editing. [Gong et al.
- 2023] iteratively generates multi-character images using personalized LoRA models [Ryu 2023], and requires pre-training a text-tolayout model. [Feng et al. 2023; Liu et al. 2023] involve fine-tuning a T2I model on storyboard datasets and conditioning it on image frames. This resembles encoder-based personalization methods like IP-Adapter [Ye et al. 2023] and ELITE [Wei et al. 2023]. Finally, [Avrahami et al. 2023b] is a concurrent work that trains a personalized LoRA model iteratively by extracting repeated identities from generated image sets.

ConsiStory does not tune or personalize the pre-trained T2I model. It seamlessly generates consistent images from text prompts alone.

Attention-based Consistency. In the realm of videos, a common practice is to increase temporal consistency by sharing self-attention keys and values [Wu et al. 2023] across frames. This can be done for generation [Ceylan et al. 2023; Khachatryan et al. 2023; Wu et al. 2023] or for video editing [Geyer et al. 2023; QI et al. 2023]. Others use attention keys and values from a source image in order to inject a consistent identity across video frames [Chang et al. 2023; Hu et al. 2023; Tu et al. 2023; Xu et al. 2023].

Whenconsideringimages,earlyworks in text-based editing [Hertz

et al. 2022; Parmar et al. 2023; Tumanyan et al. 2023] proposed to maintain the structure of an image by extracting its attention masks or features, and injecting them into follow-up generations.

More recent works explored extended-attention mechanisms to maintain consistent appearances when modifying image layouts [Caoetal.2023;Mouetal.2023],orfortraining-freeappearance[Alaluf et al. 2023] and style-transfer [Hertz et al. 2023] tasks.

Our method draws on these attention-sharing ideas but applies them to the task of consistent T2I generation. We do not draw features from existing images or align entire frames, but develop tools to enable subject-level consistency across novel images.

Appearance transfer using dense correspondence maps. has been widely studied. [Liao et al. 2017] transfer appearance between images with similar structures using VGG-based maps. [Benaim et al. 2020; Tumanyan et al. 2022] trained generative models to leverage these mappings for image-to-image translation. Recently, diffusion models have been found to establish strong zero-shot correspondence between images [Hedlin et al. 2023; Luo et al. 2023b; Zhang et al. 2023a], enabling applications like instance swapping, image editing, and robust registration.

Here, we leverage the diffusion-based DIFT maps [Tang et al. 2023] to share features across multiple images and encourage the generation of subjects with consistent appearance. This aligns features throughout the denoising process rather than doing appearance transfer as a post-hoc step.

###### 3 PRELIMINARIES: SELF-ATTENTION IN T2I MODELS

Our method manipulates self-attention in T2I diffusion models. We start by outlining its mechanism and introducing key notations.

A self-attention layer receives a series of tokens, each of which contains features describing a single image patch. Each such token undergoes linear projections through three self-attention matrices: 𝑾𝐾, 𝑾𝑉 and 𝑾𝑄. The results of these projections are known as “Keys”, “Values” and “Queries”, respectively.

More concretely, consider the 𝑖𝑡ℎ image entry in the generated batch. Let 𝑥𝑖 ∈ R𝑃×𝑑 be a sequence of 𝑃 input token vectors with feature dimension 𝑑. We define 𝐾𝑖 = 𝑥𝑖 · 𝑾𝐾, 𝑉𝑖 = 𝑥𝑖 · 𝑾𝑉 , 𝑄𝑖 = 𝑥𝑖 · 𝑾𝑄. The self-attention map is then given by:

𝐴𝑖 = softmax 𝑄𝑖𝐾𝑖⊤/√︁𝑑𝑘 ∈ R𝑃𝑥𝑃, (1)

where𝑑𝑘 is the feature dimension of𝑾𝐾,𝑾𝑄 projections. Intuitively, this map provides a relevancy score between every pair of patches in the image. It is then used to weight how much the “Value” features of a given target patch should influence a source patch ℎ𝑖 = 𝐴𝑖 ·𝑉𝑖, where ℎ denotes an intermediary, hidden feature set.

These are projected using a fourth, “output-projection" matrix,

𝑾𝑂, yielding 𝑥𝑖𝑜𝑢𝑡 = 𝑾𝑂 · ℎ𝑖, which is then summed with the input features 𝑥𝑖 to create the input for the next layer.

Our method intervenes in this self-attention mechanism by allowing images in a generated batch to attend to each other, and be influenced by each other’s 𝑥𝑜𝑢𝑡 activations.

###### 4 METHOD

Our goal is to generate a set of images portraying consistent subjects across an array of prompts. We propose to do so by better aligning the internal activation of the T2I model during image denoising. Importantly, we aim to enforce consistency exclusively through an inference-based mechanism, without additional training.

Our approach is comprised of three main components. First, we introduce a subject-driven self-attention mechanism (SDSA), aimed at sharing subject-specific information across relevant model activations in the generated image batch. Second, we observe that the above component comes at the cost of reducing the variation in the generated layouts. Therefore, we propose strategies for mitigating this form of mode collapse through an attention-dropout mechanism, and by blending query features obtained from a vanilla, non-consistent, sampling step. Third, we incorporate a feature injection mechanism to further refine the results. There, we map features from one generated image to another based on a cross-image dense-correspondence map derived from the diffusion features. Below, we outline each of these components in detail.

4.1 Subject-driven self-attention

Consider a simple idea for promoting consistency: expanding the self-attention, so that queries from one image can also attend to keys

and values from other images in the batch. This enables repeated objects to naturally attend to each other, thus sharing visual features across images. This idea is often used in video generation and editing works [Wu et al. 2023], leading to increased consistency across frames. However, generated videos differ from our scenario. First, they are created with a single prompt that is shared across frames. Second, they typically require little variation in backgrounds or layout from one frame to the next. In contrast, we want each frame to follow a unique prompt, and we want to maintain diversity in backgrounds and layouts. Naïvely employing these video-based mechanisms leads to uniform backgrounds and drastically reduced alignment with each image’s prompt - in line with expectations for a single video scene, but in direct conflict with our objectives.

One way to tackle these limitations is by reducing the amount of information being shared at background patches. As we are only concerned about sharing subject appearance, we mask the expanded self-attention, so that queries from one image can only match keys and values from the same image, or from regions containing the subject in other images. This way, features for repeated subject elements can be shared, while background features remain separate.

To this end, we employ a similar approach to prior art [Cao et al. 2023; Chefer et al. 2023] and identify noisy latent patches that are likely to contain the subject using cross-attention features. Specifically, we average and threshold the cross-attention maps related to the subject token across diffusion steps and layers to create subject-specific masks (details in Appendix B). With these masks, we propose Subject-Driven Self-Attention (SDSA) where attention is masked so each image can only attend to its own patches or the subject patches within the batch (See Fig. 2).

𝐾+ = [𝐾1 ⊕ 𝐾2 ⊕ . . . ⊕ 𝐾𝑁 ] ∈ R𝑁·𝑃×𝑑𝑘 𝑉+ = [𝑉1 ⊕ 𝑉2 ⊕ . . . ⊕ 𝑉𝑁 ] ∈ R𝑁·𝑃×𝑑𝑣 𝑀𝑖+ = [𝑀1 . . . 𝑀𝑖−1 ⊕ 1 ⊕ 𝑀𝑖+1 . . . 𝑀𝑁 ] (2)

𝐴𝑖+ = softmax 𝑄𝑖𝐾+⊤/√︁𝑑𝑘 + log𝑀𝑖+ ∈ R𝑃×𝑁·𝑃 ℎ𝑖 = 𝐴𝑖+ ·𝑉+ ∈ R𝑃×𝑑𝑣. (3)

Here, 𝑀𝑖 is the subject mask for the 𝑖𝑡ℎ entry in the batch, and ⊕ indicates matrix concatenation. We use standard attention masking, which null-out softmax’s logits by assigning their scores to −∞ according to the mask. Note that the Query tensors remain unaltered, and that the concatenated mask 𝑀𝑖+ is set to be an array of 1’s for patch indices that belong to the 𝑖𝑡ℎ image itself.

4.2 Enriching layout diversity

The use of SDSA restores prompt alignment and avoids background collapse. However, we observe that it can still lead to excessive similarity between image layouts. For example, subjects will typically be generated in similar locations and poses.

To improve the diversity of our results, we propose two strategies: first, incorporating features from a vanilla, non-consistent sampling step; and second, further weakening the subject-driven shared attention through a dropout mechanism.

Using Vanilla Query Features. Recent work [Alaluf et al. 2023], demonstrated that one can use diffusion models to combine the

[Figure 16]

[Figure 17]

Diffusion U-net

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Subject Driven Self-Attention

[Figure 18]

[Figure 19]

|Masked|Attention.<br><br>[Figure 20]<br><br>[Figure 21]<br><br>| |
|---|---|---|
|.| |.|

Prompt Set

SD SA

SD SA

SD SA

FI FI FI

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Subject Localization

[Figure 28]

[Figure 29]

[Figure 30]

Self-Attention Dropout

- Fig. 2. Architecture outline (left): Given a set of prompts, at every generation step we localize the subject in each generated image 𝐼𝑖. We utilize the cross-attention maps up to the current generation step, to create subject masks 𝑀𝑖. Then, we replace the standard self-attention layers in the U-net decoder with Subject Driven Self-Attention layers that share information between subject instances. We also add Feature Injection for additional refinement. Subject Driven Self-Attention (right): We extend the self-attention layer so the Query from generated image 𝐼𝑖 will also have access to the Keys from all other images in the batch (𝐼𝑗, where 𝑗 ≠ 𝑖), restricted by their subject masks 𝑀𝑗. To enrich diversity we: (1) Weaken the SDSA via dropout and (2) Blend Query features with vanilla Query features from a non-consistent sampling step, yielding 𝑄1∗.

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Correspondence Feature Injection

[Figure 40]

[Figure 41]

[Figure 42]

- Fig. 3. Feature Injection: To further refine the subject’s identity across images, we introduce a mechanism for blending features within the batch. We extract a patch correspondence map between each pair of images (Middle), and then inject features between images based on that map (Right).

generated queries towards the vanilla queries, resulting in:

𝑄𝑡∗ = (1 − 𝜈𝑡)𝑄𝑡𝑆𝐷𝑆𝐴 + 𝜈𝑡𝑄𝑡𝑣𝑎𝑛𝑖𝑙𝑙𝑎, (4) where 𝑣𝑡 is a linearly decaying blending parameter (see Appendix).

Self-Attention Dropout. Our second strategy to enhance layout variation involves weakening SDSA using a dropout mechanism. Specifically, at each denoising step, we randomly nullify a subset of patches from 𝑀𝑖 by setting them to 0. This weakens the attention sharing between different images and subsequently promotes richer layout variations. Notably, by adjusting the dropout probability, we can regulate the strength of consistency, and strike a balance between visual consistency and layout variations.

Through these two mechanisms, we aim to tackle two aspects of the layout-collapse problem: Query-feature blending allows to retain aspects of diversity from the non-consistent sampling, while attention-dropout encourages the model to rely less on the shared keys and values, avoiding over-consistency. By mixing them, we achieve increased diversity without significant harm to consistency.

appearance of one image with the structure of another. They do so by injecting self-attention Keys and Values from the appearance image, and Queries from the structure image. Inspired by this, we aim to enhance pose variation by aligning more closely with a structure predicted by a more diverse vanilla forward pass (i.e. without our modifications). We focus on the early steps of the diffusion process, which have been shown to primarily control layout [Balaji et al. 2022; Patashnik et al. 2023], and apply the following query-blending mechanism: Let 𝑧𝑡 be the noisy latents at step 𝑡. We first apply a vanilla denoising step to 𝑧𝑡, without SDSA, and cache the selfattention queries generated by the diffusion network:𝑄𝑡𝑣𝑎𝑛𝑖𝑙𝑙𝑎. Then, we denoise the same latents 𝑧𝑡 again, this time using SDSA. During this second pass, for all SDSA layers, we linearly interpolate the

4.3 Feature injection

The shared attention mechanism notably improves subject consistency but may struggle with fine visual features, which may hurt the subject’s identity. Hence, we propose to further improve consistency through a novel cross-image Feature Injection mechanism.

Here, we aim to improve the similarity of features from corresponding regions (e.g. the left eye) across different images in the batch. Specifically, we find that substantial texture information is contained in the self-attention output features, 𝒙𝑜𝑢𝑡, and aligning these features between matching areas can enhance consistency.

To align these features, we first build a patch correspondence map between every pair of images 𝐼𝑡 and 𝐼𝑠 in the batch, using DIFT [Tang et al. 2023] features 𝐷𝑡 and 𝐷𝑠 (See Appendix). We denote the correspondence map by 𝐶𝑡→𝑠. Intuitively, when applied

TI

“exploring

“wearing a training harness”

“sticking head out of the car window” “swimming”

“wearing a small sweater”

among ﬂowers” “in the open” “in a meadow”

“in a ship”

|[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]|
|---|

|[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]|
|---|

IP-AdapterOurs

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

DB-LoRA

“An origami style of a happy hedgehog” “A watercolor illustration of a puppy”

- Fig. 4. Qualitative Results We evaluated our method against IP-Adapter, TI, and DB-LORA. Some methods failed to maintain consistency (TI), or follow the prompt (IP-Adapter). Other methods alternated between keeping consistency or following text, but not both (DB-LoRA). Our method successfully followed the prompt while maintaining consistency. Additional results are shown at Figure A.1

on patch index 𝑝 from 𝐼𝑡, 𝐶𝑡→𝑠 [𝑝] yields the most similar patch in 𝐼𝑠, as illustrated in Fig. 3.

- 4.4 Anchor images and reusable subjects

As an additional optimization, we can reduce the computational complexity of our approach by designating a subset of generated images as “anchor images”. Rather than sharing keys and values across all generated images during SDSA steps, we allow the images to only observe keys and values derived from the anchors. Similarly, for feature injection, we only consider the anchors as valid feature sources. Note that the anchors can still observe each other during generation, but they too do not observe features from non-anchor images. We find that for most cases, two anchors are sufficient.

This offers several benefits: First, it allows for faster inference and reduced VRAM requirements, because it restricts the size of extended attention. Second, it can improve generation quality in large batches, where we notice it can reduce visual artifacts. Most importantly, we can now reuse the same subjects in novel scenes by creating a new batch where the same prompts and seeds are used to re-create the anchor images, but the non-anchor prompts have changed. Through this mechanism, our approach can yield reusable subjects, and unlimited consistent image generation.

- 4.5 Multi-subject consistent generation

Then, to promote feature similarity, we can blend corresponding features based on this mapping. We extend this idea to a many-toone scenario, where each image 𝐼𝑡 is blended with the other images in the batch. For each patch index 𝑝 in image 𝐼𝑡, we compare its corresponding patches in all other images and select the one with the highest cosine similarity in the DIFT feature space. Formally:

src(𝑝) = argmax

similarity(𝐷𝑡 [𝑝],𝐷𝑠 [𝐶𝑡→𝑠 [𝑝]]), (5)

𝑠≠𝑡

where src(𝑝) is the “best” source patch for the target patch 𝑝, and similarity is the cosine similarity score.

Finally, we blend the self-attention output layer features of the target image 𝒙𝑡𝑜𝑢𝑡, and its corresponding source patches, 𝒙𝑠𝑜𝑢𝑡.

𝒙ˆ𝑡𝑜𝑢𝑡 = (1 − 𝛼) · 𝒙𝑡𝑜𝑢𝑡 + 𝛼 · src(𝒙𝑡𝑜𝑢𝑡), (6)

where 𝛼 is a blending parameter, and src(𝒙𝑡𝑜𝑢𝑡) ∈ R𝑃×𝑑 is the tensor obtained by pooling the corresponding features for each patch 𝑝 in

𝒙𝑡𝑜𝑢𝑡 from the associated patch 𝑠𝑟𝑐(𝑝).

In practice, to enforce consistency between appearances of the same subject, without affecting the background, we exclusively apply the feature injection according to the subject masks 𝑀𝑖. Additionally, we apply a threshold to inject features only between patches with high enough similarity in the DIFT space (see Appendix). This approach ensures that features contributing to the appearance of the subject are collectively drawn from all source images, promoting a more comprehensive and representative synthesis.

Personalization-based approaches struggle in maintaining consistency over multiple subjects within a single image [Gu et al. 2023; Kumari et al. 2022; Po et al. 2023; Tewel et al. 2023]. However, with ConsiStory, multi-subject consistent generation is possible in a simple, straightforward manner, by simply taking a union of the subject masks. When the subjects are semantically different, information leakage between them is not a concern. This is due to the exponential form of the attention softmax, which acts as a gate that suppresses

tend to either overfit the single training image, producing no variations, or underfit and fail to maintain consistency. IP-Adapter similarly struggles to match complex prompts, particularly when styles are involved. Our method can successfully achieve both subject consistency and text-alignment. Additionally, in Fig. 6 we show that with different initial noise inputs, our method can generate varied sets of consistent images. In Fig. 7 we focus on photo-realistic face imagery and demonstrate that we can maintain a consistent identity while varying attributes such as expression, hair color, or tattoos.

“walking in the park”

“playing with a ball”

“walking in the snow”

“sitting in the house”

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

##### ✓

boy

###### OursLoRADB

✓

dog

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

✓

boy

###### ✗

dog

“Neonpunk style of a boy with black spiky hair, with his dog”

“in the middle of a rainstorm” “perched on a rock” “sitting by the ﬁre” “surﬁng in the sea”

“reading a book”

- Fig. 5. Multiple Subjects: ConsiStory generates multiple consistent subjects, while other methods often neglect at least one subject.

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

information leakage between unrelated subjects. Similarly, thresholding the correspondence map during feature injection yields a gating effect that safeguards against information leakage. Additional details such as hyperparameter choices are in the supplementary.

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

- 5 EXPERIMENTS

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

We begin our evaluation by comparing ConsiStory with a range of prior and concurrent baselines. We open with a qualitative comparison, showing that our method can achieve improved subjectconsistency and higher prompt-alignment when compared to the state-of-the-art. Next, we evaluate our method quantitatively, including a large-scale user study which demonstrates that users typically favor our results. Moving on, we conduct an ablation study to highlight the contribution and effect of each component in our method. Finally, we conclude with a set of extended applications, showing that our method is compatible with existing tools such as ControlNet [Zhang et al. 2023b], and it can even be used to enable training-free personalization for common object classes.

“A plastic owl”

- Fig. 6. Seed Variation. Given different starting noise, ConsiStory generates different consistent set of images.

“looking at the camera” “screaming angrily” “smiling wide” “with a face tattoo” “with purple hair”

“A close-up photo of a man with beard and sunglasses”

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

- Fig. 7. Facial expression variation: ConsiStory is able to generate images of the same identity while facial attributes.

- 5.1 Evaluation baselines

We compare our method to three classes of baselines: (1) The baseline SDXL model, without adaptations. (2) Optimization-based personalization approaches that teach the model about a new subject by fine-tuning parts of the model: Textual Inversion (TI) [Gal et al. 2022] fine-tunes the text encoder’s word embeddings. DreamBoothLoRA (DB-LORA) [Ryu 2023] tunes the diffusion U-Net using Low Rank Adaptation [Hu et al. 2021]. (3) Encoder-based approaches that take a single image as input and output a conditioning code to the diffusion model: IP-Adapter [Ye et al. 2023] and ELITE [Wei et al. 2023]. For the personalization and encoder baselines, we first generate a single image of a target subject using a prompt describing the subject, then use it to personalize the diffusion model. All methods except ELITE are based on a pre-trained SDXL model.

For ConsiStory, we use two anchor images and 0.5 dropout. For the automated metric, we also use lower dropout values.

- 5.2 Qualitative Results

Multi-Subject Generation. In Fig. 1 (bottom) we demonstrate that ConsiStory can create scenes with multiple consistent subjects. In Fig. 5 we further compare our method to LORA-DB. Notably, LORA-DB tends to neglect the consistency of one or even both subjects. This pitfall is common when combining personalization approaches, as they learn each subject in isolation. In contrast, our method simply builds on the diffusion model’s inherent compositional ability. In Figures A.2, A.3 we provide additional comparisons.

Fig. 17 provides more results with single and multiple subjects.

5.3 Quantitative evaluation

In Fig. 4 and Fig. A.1, we show qualitative comparisons. Our method can achieve a high degree of subject consistency, while better adhering to the text prompts. Tuning-based personalization approaches

Next, we perform quantitative evaluations with automated metrics. First, we use each baseline to generate 100 image sets, where each set contains 5 images depicting a shared subject under different prompts.

Our evaluation prompts were created using ChatGPT [OpenAI 2022], with the following protocol: Each prompt consisted of three parts: (1) a subject description, e.g., “A red dragon” (2) a setting description, e.g., “blowing bubbles” or “in a castle”, and (3) a style descriptor, e.g., “Origami style”. For subject descriptions, we utilized both detailed examples (“A red dragon”), and non-detailed ones (“A dragon”). For setting descriptions, we asked ChatGPT to provide descriptions that naturally fit the subject. Each of the 100 sets contains prompts sharing the same subject description and style but with varying setting descriptions. Further details can be found in Appendix D.

0.85

| | |
|---|---|
| | |

d=0

SubjectConsistency

| | |
|---|---|
| | |

d=0.15

| | |
|---|---|
| | |

0.75

d=0.5

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Ours

IP-Adapter

ELITE

LoRA DreamBooth

| | |
|---|---|
| | |

0.65

Textual Inversion

| | |
|---|---|
| | |

SDXL

0.55 0.60 0.65 Text Similarity

- Fig. 8. Subject Consistency VS Textual Similarity: ConsiStory (green) achieves the optimal balance between Subject Consistency and Textual Similarity. Encoder-based methods such as ELITE and IP-Adapter often overfit to visual appearance, while optimization-based methods such as LoRA-DB and TI do not exhibit high subject consistency as in our method. 𝑑 denotes different self-attention dropout values. Error bars are S.E.M.

50% Win Rate

Visual Textual

Visual Textual

Visual Textual

61% 39%

88% 12%

91% 9%

60% 40%

56% 44%

58% 42%

LoRADBTIIP-Adapter

- Fig. 9. User Study results indicate a notable preference among participants for our generated images both in regards to Subject Consistency (Visual) and Textual Similarity (Textual).

We follow the concurrent work of Avrahami et al. [2023b] and evaluate the methods on two axes - prompt-alignment, and subject consistency. For prompt-alignment, we use CLIP to measure the similarity between each generated image and its conditioning prompt, and report the average CLIP-score [Hessel et al. 2021] over all 500 generated images. For consistency evaluation, we use DreamSim [Fu et al. 2023], which has been shown to better correlate with human

judgment of inter-image similarity. We calculate the pair-wise similarity between each pair of images in each of the 100 sets. To focus on subject consistency, we follow Dreamsim’s background removal protocol. We report the average score over these sets. In our results, the error bars represent the Standard Error of the Mean (S.E.M).

The results are provided in Fig. 8. As can be seen, our method is situated on the Pareto-front. Our standard setup matches SDXL in text-alignment scores, demonstrating that promoting consistency a-priori can better maintain the model’s knowledge.

However, qualitatively, we observed that the consistency metric tends to bias its scores towards configurations with slight layout changes, rather than assessing consistency in the subject’s identity. Therefore, given the limitations of the automated metric, we conducted a large-scale user study. We concentrated on the most effective techniques, omitting both vanilla SDXL and ELITE. We used the standard two-alternative forced-choice format. Users faced two types of questions: (1) Subject-consistency, where they were shown two sets of 5 images each. They were asked to choose the set that better shows the same subject, ignoring background, pose, and image quality. (2) Text-alignment, where they selected the image that best matched a textual description from two images. We gathered 500 responses per baseline for each question type, totaling 3, 000 responses. The results are shown in Fig. 9. Despite being a training-free approach, ConsiStory outperforms the baselines, both regarding textual alignment and subject consistency.

Runtime comparison. We conducted a runtime analysis of the main methods, focusing on their time-to-consistent-subject (TTCS) using an H100 GPU. Our method, ConsiStory, achieved the fastest TTCS result, at 32 seconds for generating two anchors and an image based on a new prompt. This is ×25 faster than the SoTA approach by Avrahami et al. [2023b] which is estimated at 13 minutes on an H100 GPU. Moreover, our method is ×8-14 faster than LORA-DB (4.5 min.) and TI (7.5 min.). Comparing our approach to encoderbased methods like IP-Adapter is more convoluted. These techniques require weeks of pre-training, but once trained, they can generate an anchor and another image based on a new prompt in just 8 seconds.

5.4 Ablation study

We now move to evaluate the impact of different components of our own method through an ablation study of the following components: (1) SDSA steps (2) Feature Injection (FI) (3) Attention dropout and query-feature blending. (4) Without using a subject mask. Qualitative comparisons are provided in Figures 11, 12. Quantitative results are provided in Appendix A.4.

Removing SDSA leads to poor consistency, both regarding shape and texture. Removing FI yields subjects with similar shapes but less accurate identities. Finally, removing the vanilla query blending and attention dropout leads to significant reductions in layout diversity. We quantify this loss of diversity in Appendix C.

Additional examples on the effects of the feature injection are provided in Fig. 10. In cases where the subject can take forms with significantly different semantics (e.g., a humanoid robot versus a factory’s robotic arm), the DIFT features may lead to noisy correspondences. While such noisy correspondence maps can lead to sub-optimal subject consistency, we observe that in the majority of

“in the beach” “on the snow” “on a branch”

“in his house” “eating dinner” “in a factory”

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

w/oFICorrespondenceOurs

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

“A 3D animation of a cute dragon”

“A photo of a smart robot”

- Fig. 10. Additional feature injection results. Our method can handle scenarios where the estimated correspondence maps are noisy, or even when parts of the objects are obscured.

Oursw/oVariation

|[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>[Figure 125]|
|---|

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

w/oFIw/oSDSA

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

- Fig. 11. Component Ablation We ablated several components: SubjectDriven Self Attention (SDSA), Feature Injection (FI), and our variation enriching strategies: self-attention dropout and Query-feature blending (Variation). All ablated cases fail to maintain consistency as our method.

we want to ensure that our alternative approach maintains this compatibility. In Fig. 13 we show that pose-based controls with ControlNet successfully guide our consistent image generation method.

Training-free Personalization. We are the first to show training-free personalization (Fig. 14), where ConsiStory enables personalization without any tuning or encoder use. Specifically, we show how to personalize common subject classes. Given two images of a subject, we invert them using Edit Friendly DDPM-Inversion [HubermanSpiegelglas et al. 2023]. These latents and noise maps are used as anchors for ConsiStory, allowing the rest of the batch to draw on their visual appearance. This application requires minor modifications of ConsiStory, which we detail in Appendix E.

In Fig. A.6 we show that this approach struggles with complex objects, and is incompatible with style-changing prompts. Yet, we believe it can serve as a quick and cheaper alternative to current personalization methods, and leave it for future work.

“in the garden” “in the park” “on a dirt road” “sitting on a hill” “looking at the sky”

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Oursw/oMask

cases, feature injection still contributes to more consistent results. We also highlight the method’s ability to handle occluded objects (e.g. the robots legs, which do not appear in one of the frames). This is grounded in the thresholding mechanism, which drops patches with low similarity scores.

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

5.5 Extended Applications

“A photo of a teddy bear”

Spatial Controls. We first demonstrate that ConsiStory is compatible with existing guided generation tools like ControlNet [Zhang et al. 2023b]. As these are compatible with standard personalization methods [Avrahami et al. 2023a; Gal et al. 2022; Zhang et al. 2023b],

Fig. 12. Subject Masking: Without the subject mask, there is noticeable background leakage across images.

“A photo”

Pose Control

[Figure 150]

[Figure 151]

[Figure 152]

FailedStyleTransfer

“B&W Sketch” “3D animation” “Watercolor”

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

“a photo of an athletic 50 year-old man with gray hair”

[Figure 157]

[Figure 158]

[Figure 159]

“… in the beach” “… in the countryside” “… in the park”

IncorrectLocalization

- Fig. 13. ControlNet Integration. Our method can be integrated with ControlNet to generate a consistent character with pose control.

[Figure 160]

[Figure 161]

“in the Grand Canyon” “in Boston”

“on a wooden ﬂoor” “on top of snow”

Inverted Anchors

[Figure 162]

[Figure 163]

[Figure 164]

Inverted Anchors

[Figure 165]

- Fig. 14. Training-Free Personalization. We utilize edit-friendly inversion to invert 2 real images per subject. These inverted images used as anchors in our method for training-free personalization.

[Figure 166]

- Fig. 15. Limitations: Our method often struggles with different styles in the same set of image (Top), and is dependent on the quality of the model cross-attention to localize the subject correctly (Bottom).

“in a studio” “in a meadow”

“eating piece of cake”

“A photo of a young man, posing”

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

“in a studio” “in a meadow”

“eating piece of cake”

- Fig. 16. Model Bias. The underlying SDXL model may exhibit biases towards certain ethnic groups, and our approach inherits them. Our method can generate consistent subjects belonging to diverse groups when these are highlighted in the prompt.

###### 6 LIMITATIONS

Our approach has several limitations, shown in Fig. 15. First, it relies on the localization of objects through cross-attention maps. This process may occasionally fail, particularly when dealing with unusual styles. However, from our observations so far, such failures appear relatively infrequent (fewer than 5%), and they can be resolved by simply changing the seed. Another limitation is found in the entanglement between appearance and style. Our method struggles to separate the two, and hence we are limited to consistent generations where images share the same style.

the given prompts. Moreover, our method can be easily extended to tackle more challenging cases such as multi-subject scenarios, and even enable training-free personalization for common objects.

Additionally, we observe that the underlying SDXL model may exhibit biases towards certain groups. We demonstrate that these biases can be significantly reduced by specifying modifiers like gender or ethnicity, as illustrated in Figure 16.

We hope that our results will assist in a consistent generation of creative endeavors and that they will inspire others to continue exploring training-free alternatives to personalization-based tasks.

###### 7 CONCLUSIONS

ACKNOWLEDGMENTS

We introduced ConsiStory, a training-free approach for creating visually consistent subjects using a pre-trained text-to-image diffusion model. When compared to the state of the art, our method is not only ×20 faster, but can better preserve the output’s alignment with

We thank Assaf Shocher, Eli Meirom, Chen Tessler, Dvir Samuel and Yael Vinker for useful discussions and for providing feedback on an earlier version of this manuscript. This work was completed as part of the first author’s PhD thesis at Tel-Aviv University.

“hiking in the mountains with walking stick”

“standing in the street”

“drinking hot tea on a rainy night”

“meeting a friend in the park”

“graduating from college”

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

“A photorealistic illustration of a bunny, full body”

“dressed in a “curled up asleep” miniature jacket”

“wearing a ﬂower “in a meadow” crown”

“wearing a small hat”

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

“A pink hedgehog with a cute smile”

## SingleSubjectMultiSubject

“preparing a gourmet meal”

“walking in the “baking a cake” street at night” “taking a bath”

“sitting on a stool”

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

“A 3D animation of a young chef with curly hair”

“in a blooming meadow”

“wearing a silver bridle”

“dressed in a glow of moonlight”

“dressed in a blanket of mist”

“meeting a group of children”

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

“A unicorn with a midnight blue coat and a crystal horn”

“on a table” “in a room” “on the ground” “with ﬂowers” “being held by a person”

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

###### “A photo of a vase”

“sitting on the mat, wearing a hat”

“walking in the street, with his

“sitting on the couch with her owl”

“standing outside with her owl”

“eating his food while wearing a hat”

“the owl sitting on a branch”

“jumping over a puddle”

hat on” “with her pet owl”

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

“A photo of a dog” “A 3D animation of a 12 years old girl”

“eating in a restaurant”

“exploring the rainforest”

“walking in the sand”

“looking through a junkyard”

“throwing the ball up”

“going to sleep, holding his ball”

“kicking a ball” “holding the ball”

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

“A 3D animation of a baby in a suit, with his hamster” “A hyper-realistic digital painting of a young ginger boy”

Fig. 17. Additional Qualitative Results We demonstrate that ConsiStory successfully generates a consistent subject while following the prompt for various subjects (Top). Furthermore, we show that ConsiStory still succeeds even in the case of multiple interacting subjects (Bottom).

REFERENCES

Yuval Alaluf, Daniel Garibi, Or Patashnik, Hadar Averbuch-Elor, and Daniel Cohen-Or. 2023. Cross-Image Attention for Zero-Shot Appearance Transfer. arXiv:2311.03335 [cs.CV]

Moab Arar, Rinon Gal, Yuval Atzmon, Gal Chechik, Daniel Cohen-Or, Ariel Shamir, and Amit H. Bermano. 2023. Domain-agnostic tuning-encoder for fast personalization of text-to-image models. In SIGGRAPH Asia 2023 Conference Papers. 1–10.

Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel Cohen-Or, and Dani Lischinski. 2023a. Break-A-Scene: Extracting Multiple Concepts from a Single Image. In SIGGRAPH Asia 2023 Conference Papers (, Sydney, NSW, Australia,) (SA ’23). Association for Computing Machinery, New York, NY, USA, Article 96, 12 pages. https://doi.org/10.1145/3610548.3618154

Omri Avrahami, Amir Hertz, Yael Vinker, Moab Arar, Shlomi Fruchter, Ohad Fried, Daniel Cohen-Or, and Dani Lischinski. 2023b. The Chosen One: Consistent Characters in Text-to-Image Diffusion Models. arXiv preprint arXiv:2311.10093 (2023). Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, and MingYu Liu. 2022. eDiff-I: Text-to-Image Diffusion Models with Ensemble of Expert Denoisers. arXiv preprint arXiv:2211.01324 (2022).

Sagie Benaim, Ron Mokady, Amit Bermano, Daniel Cohen-Or, and Lior Wolf. 2020. Structural-analogy from a Single Image Pair. Computer Graphics Forum n/a (2020). https://doi.org/10.1111/cgf.14186 arXiv:https://onlinelibrary.wiley.com/doi/pdf/10.1111/cgf.14186

Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. 2023. MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 22560–22570.

Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. 2023. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 23206–23217.

Di Chang, Yichun Shi, Quankai Gao, Jessica Fu, Hongyi Xu, Guoxian Song, Qing Yan, Xiao Yang, and Mohammad Soleymani. 2023. MagicDance: Realistic Human Dance Video Generation with Motions & Facial Expressions Transfer. arXiv preprint arXiv:2311.12052 (2023).

Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. 2023. Attendand-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG) 42, 4 (2023), 1–10.

Zhangyin Feng, Yuchen Ren, Xinmiao Yu, Xiaocheng Feng, Duyu Tang, Shuming Shi, and Bing Qin. 2023. Improved Visual Story Generation with Adaptive Context Modeling. arXiv preprint arXiv:2305.16811 (2023).

Stephanie Fu, Netanel Yakir Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. 2023. DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data. In Thirty-seventh Conference on Neural Information Processing Systems. https://openreview.net/forum?id=DEiNSfh1k7 Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. 2022. An Image is Worth One Word: Personalizing Text-toImage Generation using Textual Inversion. https://doi.org/10.48550/ARXIV.2208. 01618

Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel CohenOr. 2023. Encoder-based domain tuning for fast personalization of text-to-image models. ACM Transactions on Graphics (TOG) 42, 4 (2023), 1–13.

Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. 2023. TokenFlow: Consistent Diffusion Features for Consistent Video Editing. arXiv preprint arxiv:2307.10373

(2023).

Yuan Gong, Youxin Pang, Xiaodong Cun, Menghan Xia, Haoxin Chen, Longyue Wang, Yong Zhang, Xintao Wang, Ying Shan, and Yujiu Yang. 2023. TaleCrafter: Interactive Story Visualization with Multiple Characters. arXiv preprint arXiv:2305.18247 (2023).

Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. 2023. Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models. arXiv preprint arXiv:2305.18292 (2023).

Eric Hedlin, Gopal Sharma, Shweta Mahajan, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, and Kwang Moo Yi. 2023. Unsupervised Semantic Correspondence Using Stable Diffusion. (2023). arXiv:2305.15581 [cs.CV]

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2022. Prompt-to-prompt image editing with cross attention control.

(2022). Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. 2023. Style Aligned Image Generation via Shared Attention. (2023).

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. CLIPScore: A Reference-free Evaluation Metric for Image Captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, MarieFrancine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (Eds.). Association for Computational Linguistics, Online and Punta Cana, Dominican Republic, 7514–7528. https://doi.org/10.18653/v1/2021.emnlp-main.595

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, and Weizhu Chen. 2021. LoRA: Low-Rank Adaptation of Large Language Models. ArXiv abs/2106.09685 (2021).

Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. 2023. Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation. arXiv preprint arXiv:2311.17117 (2023).

Xun Huang and Serge Belongie. 2017. Arbitrary style transfer in real-time with adaptive instance normalization. In Proceedings of the IEEE international conference on computer vision. 1501–1510.

Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. 2023. An Edit Friendly DDPM Noise Space: Inversion and Manipulations. arXiv preprint arXiv:2304.06140 (2023).

Hyeonho Jeong, Gihyun Kwon, and Jong Chul Ye. 2023. Zero-shot generation of coherent storybook from plain text story using diffusion models. arXiv preprint arXiv:2302.03900 (2023).

Xuhui Jia, Yang Zhao, Kelvin CK Chan, Yandong Li, Han Zhang, Boqing Gong, Tingbo Hou, Huisheng Wang, and Yu-Chuan Su. 2023. Taming encoder for zero finetuning image customization with text-to-image diffusion models. arXiv preprint arXiv:2304.02642 (2023).

Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. 2023. Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 15954–15964. Aliasghar Khani, Saeid Asgari Taghanaki, Aditya Sanghi, Ali Mahdavi Amiri, and

Ghassan Hamarneh. 2023. Slime: Segment like me. arXiv preprint arXiv:2309.03179

(2023). Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu.

2022. Multi-Concept Customization of Text-to-Image Diffusion. arXiv (2022).

Dongxu Li, Junnan Li, and Steven CH Hoi. 2023. Blip-diffusion: Pre-trained subject representation for controllable text-to-image generation and editing. arXiv preprint arXiv:2305.14720 (2023).

Jing Liao, Yuan Yao, Lu Yuan, Gang Hua, and Sing Bing Kang. 2017. Visual Attribute Transfer Through Deep Image Analogy. ACM Trans. Graph. 36, 4, Article 120 (July 2017), 15 pages. https://doi.org/10.1145/3072959.3073683

Chang Liu, Haoning Wu, Yujie Zhong, Xiaoyun Zhang, and Weidi Xie. 2023. Intelligent Grimm–Open-ended Visual Storytelling via Latent Diffusion Models. arXiv preprint arXiv:2306.00973 (2023).

Grace Luo, Lisa Dunlap, Dong Huk Park, Aleksander Holynski, and Trevor Darrell. 2023b. Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence. In Advances in Neural Information Processing Systems.

Jinqi Luo, Kwan Ho Ryan Chan, Dimitris Dimos, and René Vidal. 2023a. Knowledge Pursuit Prompting for Zero-Shot Multimodal Synthesis. arXiv preprint arXiv:2311.17898

(2023).

Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. 2023. Dragondiffusion: Enabling drag-style manipulation on diffusion models. arXiv preprint arXiv:2307.02421 (2023).

OpenAI. 2022. ChatGPT. https://chat.openai.com/. Accessed: 2023-10-15. Nobuyuki Otsu. 1979. A Threshold Selection Method from Gray-Level Histograms.

IEEE Transactions on Systems, Man, and Cybernetics 9, 1 (1979), 62–66. https: //doi.org/10.1109/TSMC.1979.4310076

Xichen Pan, Pengda Qin, Yuhong Li, Hui Xue, and Wenhu Chen. 2024. Synthesizing Coherent Story With Auto-Regressive Latent Diffusion Models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). 2920–2930.

Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and JunYan Zhu. 2023. Zero-shot image-to-image translation. In ACM SIGGRAPH 2023 Conference Proceedings. 1–11.

Or Patashnik, Daniel Garibi, Idan Azuri, Hadar Averbuch-Elor, and Daniel Cohen-Or.

2023. Localizing Object-level Shape Variations with Text-to-Image Diffusion Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

Ryan Po, Guandao Yang, Kfir Aberman, and Gordon Wetzstein. 2023. Orthogonal Adaptation for Modular Customization of Diffusion Models. arXiv preprint arXiv:2312.02432 (2023).

Chenyang QI, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. 2023. FateZero: Fusing Attentions for Zero-shot Text-based Video Editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 15932–15942.

Elad Richardson, Kfir Goldberg, Yuval Alaluf, and Daniel Cohen-Or. 2023. ConceptLab: Creative Generation using Diffusion Prior Constraints. arXiv:2308.02669 [cs.CV]

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2022. DreamBooth: Fine Tuning Text-to-image Diffusion Models for Subject-Driven Generation. (2022).

Simo Ryu. 2023. Low-rank Adaptation for Fast Text-to-Image Diffusion Fine-tuning. https://github.com/cloneofsimo/lora.

Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. 2023. Instantbooth: Personalized textto-image generation without test-time finetuning. arXiv preprint arXiv:2304.03411

(2023).

Chenyang Si, Ziqi Huang, Yuming Jiang, and Ziwei Liu. 2023. FreeU: Free Lunch in Diffusion U-Net. arXiv preprint arXiv:2309.11497 (2023).

Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. 2023. Emergent Correspondence from Image Diffusion. In Thirty-seventh Conference on Neural Information Processing Systems. https://openreview.net/forum?id= ypOiXjdfnU

Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. 2023. Key-locked rank one editing for text-to-image personalization. In ACM SIGGRAPH 2023 Conference Proceedings. 1–11.

Shuyuan Tu, Qi Dai, Zhi-Qi Cheng, Han Hu, Xintong Han, Zuxuan Wu, and Yu-Gang Jiang. 2023. MotionEditor: Editing Video Motion via Content-Aware Diffusion. arXiv preprint arXiv:2311.18830 (2023).

Narek Tumanyan, Omer Bar-Tal, Shai Bagon, and Tali Dekel. 2022. Splicing ViT Features for Semantic Appearance Transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10748–10757.

Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. 2023. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1921–1930.

Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. 2023. ELITE: Encoding Visual Concepts into Textual Embeddings for Customized Text-to-Image Generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 15943–15953.

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. 2023. Tune-a-video: Oneshot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7623–7633.

Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. 2023. MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. 2023. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721 (2023).

Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Polania Cabrera, Varun Jampani, Deqing Sun, and Ming-Hsuan Yang. 2023a. A Tale of Two Features: Stable Diffusion Complements DINO for Zero-Shot Semantic Correspondence. (2023).

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023b. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.

#### Appendix: Training-Free Consistent Text-to-Image Generation

###### A ADDITIONAL RESULTS

We provide additional Qualitative and Quantitative results of our method. In Fig. A.1, Fig. A.2 and Fig. A.3 we show additional single and multi subject qualitative comparisons to existing baselines. In Fig. A.4 and Fig. A.5 we provide quantitative ablation results with respect to Textual Similarity, Subject Consistency, and Layout Diversity.

###### B ADDITIONAL IMPLEMENTATION DETAILS

Subject-Driven Self-Attention. is applied at all timesteps, in the decoder layers of the U-net.

Feature injection and DIFT features: Feature injection is applied at timesteps 𝑡 ∈ [680, 900], with 𝛼 = 0.8. We only inject patches with similarity scores above a threshold that is set automatically by the Otsu method.

For feature injection, we denoise the image from 𝑡 = 1000 to 𝑡 = 261, for computing the DIFT features at 𝑡 = 261 as in [Tang et al. 2023]. We then denoise again from 𝑡 = 1000 to 𝑡 = 0, guiding the feature injection with the precomputed DIFT features.

Pose Variation. We apply the injection of Vanilla Query Features over the first 5 denoising steps, with 𝜈𝑡 values that linearly decay with𝑡 from 0.9 to 0.8. Self-Attention Dropout is applied with 𝑝 = 0.5.

Diffusion Process. Images were sampled with 50 DDIM steps and a guidance scale of 5. Similarly to [Alaluf et al. 2023; Luo et al. 2023a], we used Free-U [Si et al. 2023] to enhance the generation quality.

Extracting per subject mask. To extract subject masks from noisy latent patches, we collect all cross-attention maps that relate to each subject’s token, across all previous diffusion steps, and all crossattention layers of resolution 32 × 32. We then average them, and threshold them using the “Otsu’s method” [Otsu 1979]. The subject mask at the generation step 𝜏, is given by:

𝑚𝑖 = E𝑙𝐿=0E𝜏𝑡=𝑇 [𝐴𝑖] 𝑀𝑖 = 𝑜𝑡𝑠𝑢 𝑚𝑖 ∈ {0, 1}𝑃, (7)

where 𝐿 is the number of network layers, 𝑃 is the number of patches and E denotes averaging.

Constructing DIFT pairwise correspondence maps. For a source image 𝐼𝑠 and a target image 𝐼𝑡, we denote their DIFT features by 𝐷𝑠, 𝐷𝑡 ∈ R𝑃×𝑑𝐷𝐼𝐹𝑇 respectively, where 𝑑𝐷𝐼𝐹𝑇 is the feature dimension. The cross-image patch similarity scores are given by the cosine similarity between these features:

𝐷𝑠 [𝑝𝑠] · 𝐷𝑡 [𝑝𝑡] ∥𝐷𝑠 [𝑝𝑠]∥ ∥𝐷𝑡 [𝑝𝑡]∥

Sim(𝐼𝑠,𝐼𝑡)𝑝𝑠,𝑝𝑡 =

, (8)

where 𝑝𝑠, 𝑝𝑡 are the indices of specific patches in the source and target image respectively and 𝐷𝑠 [𝑝𝑠], 𝐷𝑡 [𝑝𝑡] are the DIFT matrix rows (i.e. feature-vectors) matching these patches. Given these patchsimilarityscores,wecancalculateapatch-wise dense-correspondence

map 𝐶𝑡→𝑠:

∀𝑝𝑡 ∈ 𝐼𝑡 : 𝐶𝑡→𝑠 [𝑝𝑡] = argmax

Sim(𝐼𝑠,𝐼𝑡)𝑝𝑠,𝑝𝑡 . (9)

𝑝𝑠 ∈𝐼𝑠

Here, for each patch 𝑝𝑡 in the target image 𝐼𝑡, 𝐶𝑡→𝑠 [𝑝𝑡] gives the index of the most similar patch in the source image, 𝐼𝑠.

Control Net. To enhance the impact of the pose from ControlNet, we raise the Self-Attention Dropout value to 0.7.

C DIVERSITY EVALUATIONS

In our quest to enhance layout diversity, we have proposed two strategies: Vanilla Query Injection and Self-Attention Dropout. To quantitatively assess the contributions of these strategies, we construct an automatic evaluation metric for layout diversity.

Specifically, we utilize the DIFT features and dense correspondence maps between image pairs. We measure layout diversity by calculating the average displacement between corresponding points across each image pair. A low displacement score indicates that the subject’s layout is largely aligned in both images. If this persists across the entire image set, we can conclude that the images have limited diversity. Notably, we opt for a geometric-diversity metric rather than an image-based one since the latter may conflate layout diversity with undesired changes in the subject’s appearance. In Fig. A.5, we show the results of our diversity metric when ablating components from our method. Scores are normalized to the displacement value of images sampled from a vanilla non-consistent SDXL model. Our full method attains high diversity scores, surpassing solutions that omit one or more of the layout enrichment components.

D PROMPTS DATASET DETAILS

To evaluate our method on a large scale, we constructed a prompts dataset. We asked ChatGPT to construct sets of 5 prompts each, where every prompt in a set contains the same recurring subject. In addition, we instructed it to generate the following metadata for each prompt set:

- • Superclass: We asked the model to group each set into one of the following superclasses: humans, animals, fantasy, inanimate.
- • Subject Token: A single token representing the subject (e.g., ’cat’, ’boy’).
- • Subject Description: The description of the recurring subject (e.g., "A 16-year-old girl with blonde hair").
- • Description Level: This refers to whether the subject description is generic or detailed. For example, a generic description might simply be "A dog", while a detailed description could elaborate as "A dog with brown and white fur, fluffy hair, and pointy ears".
- • Style: The style of the requested image; for example, "A 3D animation of."

We generated 100 prompt sets, comprising a total of 500 images, spanning various superclasses, description levels, and styles. We aimed to cover a wide spectrum of visual themes and subjects, ensuring a broad and inclusive representation across the different categories.

“inside a hollowed-out tree”

“atop a dew-covered ﬂower”

“in a grassy yard”

“dressed in a bandana”

“wearing a training harness”

“under a full moon”

“collecting morning dew”

“in a pet store”

|[Figure 220]<br><br>[Figure 221]<br><br>[Figure 222]<br><br>[Figure 223]|
|---|

|[Figure 224]<br><br>[Figure 225]<br><br>[Figure 226]<br><br>[Figure 227]|
|---|

###### IP-AdapterTIDB-LoRAOurs

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

###### ELITEELITE

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

“A hyper-realistic digital painting of a fairy”

“A photo of a puppy”

“drinking coﬀee with his friend in the heart of NYC”

“looking at a painting in a museum”

“reviewing a paper in his cozy apartment”

“dressed as an astronaut”

“digging with a trowel”

“in a dense forest”

“jogging on the beach”

“wearing green knitted hat”

|[Figure 260]<br><br>[Figure 261]<br><br>[Figure 262]<br><br>[Figure 263]|
|---|

|[Figure 264]<br><br>[Figure 265]<br><br>[Figure 266]<br><br>[Figure 267]|
|---|

OursIP-AdapterTIDB-LoRA

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

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

[Figure 299]

“A hyper-realistic digital painting of a teenage boy with black hair, a slight build”

“A 3D animation of a cute nerdy mouse”

- Fig. A.1. Additional Qualitative Comparisons We evaluated our method against IP-Adapter, TI, ELITE and DB-LORA. Some methods failed to maintain consistency (TI) or follow the prompt (IP-Adapter). Other methods alternated between keeping consistency or following text, but not both (DB-LoRA). Our method successfully followed the prompt while maintaining consistency.

“sitting on the mat, wearing a hat”

“walking in the street, with his hat on”

“sitting in the couch with her owl”

“standing outside with her owl”

“eating his food while wearing a hat”

“jumping over a puddle”

“the owl sitting on a branch”

“with her pet owl”

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

###### OursLoRADB

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

“A photo of a dog”

“A 3D animation of a 12 years old girl”

“resting in the wild, wearing his goggles”

“in the city, wearing his goggles”

“going to sleep, holding his ball”

“throwing the ball up”

“reading the newspaper”

“holding the ball”

“wearing goggles”

“kicking a ball”

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

###### OursLoRADB

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

“An illustration of a wolf”

“A hyper-realistic digital painting of a young ginger boy”

- Fig. A.2. Additional Qualitative Multi-Subject Comparisons We evaluated our method against DB-LORA for generation of multiple consistent subjects. Lora DB tends to neglect the consistency of at least one of the subjects, while our method succeeds in both.

OursTheChosenOne

“A hyper-realistic digital painting of a happy girl, brown eyes.”

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

- Fig. A.3. Comparison to The Chosen One, Multi-Subject We evaluated our method against a concurrent work by [Avrahami et al. 2023b]. Notably, while ConsiStory is training-free, in contrast to the concurrent work that requires an iterative optimization process, we demonstrate that our method outperforms it in preserving consistency for multiple subjects.

0.80

| |Ours<br><br>Ours (w/o Dropout & Query Injection)<br><br>Ours (w/o Dropout)<br><br>Ours (w/o FI)<br><br>Ours (w/o SDSA)| |
|---|---|---|
| | | |
| | | |
| | | |

SubjectConsistency

0.75

0.70

0.600 0.625 0.650 Text Similarity

Fig. A.4. Quantitative Component Ablation We conducted a quantitative evaluation of ConsiStory by ablating various components: Self-Attention Dropout (Dropout), Query Injection, Feature Injection (FI), and SubjectDriven Self-Attention (SDSA). Notably, omitting either SDSA or FI resulted in reduced subject consistency. Eliminating the variation enhancement mechanisms (Dropout and Query Injection) decreased textual similarity.

Our prompt dataset is provided as a YAML file in the supplemental materials.

0.8

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

0.7

Diversity

0.6

0.5

w/o Dropout & Query Injection

w/o Dropout Dropout = 0.3 Ours

Methods

- Fig. A.5. Layout Diversity Ablation Our method attains the highest Diversity Score, when compared to variations omitting the Dropout and/or the Query Injection components.

- E TRAINING-FREE PERSONALIZATION DETAILS

We implemented edit-friendly inversion [Huberman-Spiegelglas et al. 2023] for SDXL and set the guidance scale to 2.0. We inverted two real-images, as shown at Fig. 14, and used them as anchors in our method. However, in our method, the anchors are generated with attention sharing between them, which is not suitable for the case of inverted anchors. Therefore, we modify our anchoring process such that anchors will only share attention maps with generated images, rather than with other anchors. We observed that naively using the inverted image attention features with our method gave poor personalization results, and we hypothesize it is related to a distribution shift between the features of the inverted and generated images. Hence, we added Adain [Huang and Belongie 2017] to align the self-attention keys of inverted features with the self-attention keys of the generated images.

Since edit-friendly inversion is based on DDPM, we modified our generation process to also use DDPM scheduling with 100 generation steps. We used the default guidance scale of 5.0 for the generated images. We note that this training-free personalization only works for simple objects, and provide failure cases at Fig. A.6.

“with Eiﬀel Tower in

the background” “in a box”

“on a wooden ﬂoor” “in the ocean”

“A photo of a cat statue…”

“A photo of a vase…”

Inverted Anchors

[Figure 340]

Inverted Anchors

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

Fig. A.6. Training-Free Personalization Failure Our training-free personalization method may fail on non common subjects.

- F USER STUDY DETAILS

We evaluate the models through two Amazon Mechanical Turk user studies using a two-alternative forced choice protocol. In the first study, named “Visual consistency", raters saw two image sets, each produced by a different approach. They chose the set where the subject’s identity remained most consistent. In the second study

named “Textual alignment", users received a text description and two images from different approaches. They selected the image that better matched the description.

- F.1 Visual Consistency

For the first study, in each trial, raters were presented with two sets of images, each set depicting a subject in various situations. They were instructed to select the set where the subject’s identity remained consistent across all images. This decision was to be based solely on the subject’s features and identity, disregarding elements such as background, clothing, or pose. The focus was on the consistency of the subject’s identity, including aspects like eye color, texture, facial features, and other subtle details. Example images and solutions were provided for guidance. Figure A.7 illustrates the experimental framework used in the trials. Figure A.8 displays the example images provided to help guide raters through the instructions.

The study included 100 trials of images from 100 unique subjects. Each trial was repeated 5 times with 5 different raters. Each image set consisted of 5 images generated from 5 distinct prompt settings. For the ConsiStory images, we selected the variant with less stringent identity preservation parameters. Raters were paid 0.15 per trial. To maintain the quality of the study, we only selected raters with an Amazon Mechanical Turk “Masters” qualification, demonstrating a high degree of approval rate over a wide range of tasks. Furthermore, we also conducted a qualification test on the prescreened pool of raters, consisting of 5 curated trials that were simple.

- F.2 Textual Alignment

For the second study, in each trial, raters were presented with a textual description and two images. They were instructed to determine which image better matched the textual description. They were advised to focus on the details of the description. For example, if the text states "A bear is drinking from a cup", raters should choose the image where the bear is actually drinking from the cup. For guidance, example images with solutions were provided. Figure A.9 illustrates the experimental framework that was used in trials. The text descriptions only included the general subject class (e.g. dog”) rather than fully detailed descriptions (e.g. A cute brown and white fluffy puppy dog with blue eyes”). This helped raters focus on the subject’s setting. Figure A.10 displays the examples provided to help guide raters through the instructions.

The study included 500 trials of images from 100 unique subjects, each subject had 5 distinct prompt settings. For the ConsiStory images, we selected the variant with less stringent identity preservation parameters. We paid $0.05 per trial. To maintain the quality of the study, we only selected raters with an Amazon Mechanical Turk “Masters” qualification. Furthermore, we also conducted a qualification test on the prescreened pool of raters, consisting of 5 curated trials that were simple.

[Figure 346]

Fig. A.7. One trial of the visual consistency user study.

[Figure 347]

[Figure 348]

Fig. A.8. Examples provided in the visual consistency user study.

[Figure 349]

Fig. A.9. One trial of the textual alignment user study.

[Figure 350]

[Figure 351]

Fig. A.10. Examples provided in the textual alignment user study.

