# arXiv:2307.00522v1[cs.CV]2Jul2023

## LEDITS: Real Image Editing with DDPM Inversion and Semantic Guidance

Linoy Tsaban, Apolinário Passos HuggingFace {linoy, apolinario}@huggingface.co

[Figure 1]

Figure 1: LEDITS- DDPM inversion with semantic guidance for real image editing. Real images edited purely with DDPM inversion and with both DDPM inversion and semantic guidance (LEDITS). In this combined approach we first apply DDPM Inversion on the input image, and then edit by performing the reverse diffusion process using the inverted latents and the desired target prompt, together with semantic guidance.

### ABSTRACT

Recent large-scale text-guided diffusion models provide powerful image generation capabilities. Currently, a significant effort is given to enable the modification of these images using text only as means to offer intuitive and versatile editing. However, editing proves to be difficult for these generative models due to the inherent nature of editing techniques, which involves preserving certain content from the original image. Conversely, in text-based models, even minor modifications to the text prompt frequently result in an entirely distinct result, making attaining one-shot generation that accurately corresponds to the user’s intent exceedingly challenging. In addition, to edit a real image using these state-of-the-art tools, one must first invert the image into the pre-trained model’s domain - adding another factor affecting the edit quality, as well as latency. In this exploratory report, we propose LEDITS - a combined lightweight approach for real-image editing, incorporating the Edit

Friendly DDPM inversion technique with Semantic Guidance, thus extending Semantic Guidance to real image editing, while harnessing the editing capabilities of DDPM inversion as well. This approach achieves versatile edits, both subtle and extensive as well as alterations in composition and style, while requiring no optimization nor extensions to the architecture. Code and examples are available on the project’s webpage.

### 1 Introduction

The exceptional realism and diversity of image synthesis using text-guided diffusion models have garnered significant attention, leading to a surge in interest. The advent of large-scale models [1, 2, 3, 4, 5, 6] has sparked the imaginations of countless users, granting unprecedented creative freedom in generating images. Consequently, ongoing research endeavors have emerged, focusing on exploring ways to utilize these powerful models for image editing. Recent developments in intuitive text-based editing showcased the ability of diffusion based methods to manipulate images using text alone [7, 8, 9, 10, 11, 12, 13].

In a recent work by Brack et al.[7] the concept of semantic guidance (SEGA) for diffusion models was introduced. SEGA requires no external guidance, is calculated during the existing generation process and was demonstrated to have sophisticated image composition and editing capabilities. The concept vectors identified with SEGA were demonstrated to be robust, isolated, can be combined arbitrarily, and scale monotonically. Additional studies explored alternative methods of engaging with image generation that are rooted in semantic understanding, such as Prompt-to-Prompt [8], which leverages the semantic information of the model’s cross-attention layers that associates pixels with tokens from the text prompt. While operations on the cross-attention maps enable various changes to the generated image, SEGA does not require token-based conditioning and allows for combinations of multiple semantic changes.

Text-guided editing of a real image with state-of-the-art tools requires inverting the given image, which poses a significant challenge in leveraging them for real images. This requires finding a sequence of noise vectors that once used as input for a diffusion process, would produce the input image. The vast majority of diffusion-based editing works use the denoising diffusion implicit model (DDIM) scheme [8, 9, 11, 12, 14, 15], which is a deterministic mapping from a single noise map to a generated image.

In the work of Huberman et al. [16], an inversion method for the denoising diffusion probabilistic model (DDPM) scheme was proposed. They suggest a new way to compute noise maps involved in the diffusion generation process of the DDPM scheme, so that they behave differently than the ones used in regular DDPM sampling: they are correlated across timesteps and have a higher variance. Edit Friendly DDPM inversion was shown to achieve state-of-the-art results on text-based editing tasks (either by itself or in combination with other editing methods) and can generate diverse results for each input image and text, contrary to DDIM inversion-based methods.

In this overview we aim to casually explore the combination and integration of the DDPM inversion and SEGA techniques, which we refer to as LEDITS. LEDITS consists of a simple modification to the semantically guided diffusion generation process. This modification extends the SEGA technique to real images as well as introduces a combined editing approach that makes use of the editing capabilities of both methods simultaneously, showing competitive qualitative results with state-of-the-art methods.

### 2 Related Work

#### 2.1 Edit friendly DDPM inversion

A significant challenge of diffusion-based methods for image editing and manipulation is the extension to real images that requires inverting the generation process. In particular, inversion of DDPM sampling scheme [1] posed a major challenge that was recently addressed by Huberman et al. [16]. In their work, they suggest an alternative inversion, that consists of a novel way to compute the T + 1 noise maps involved in the diffusion generation process of the DDPM scheme, so that they are better suited for editing.

In the DDPM sampling scheme, the reverse diffusion process starts from a random noise vector xT ∼ N(0,I) and iteratively denoises it using

xt−1 = µˆt(xt) + σtzt t = T,...,1 (1) where zt are iid standard normal vectors, and

)/√α¯ + 1 − α¯t−1 − σt2ϵθ

√1 − α¯tϵθ

µˆt(xt) = √α¯t−1(xt −

(2)

t

t

[Figure 2]

- Figure 2: LEDITS overview. Top: inversion of the input image. We first apply DDPM inversion on the original image to obtain the inverted latents and corresponding noise maps. Bottom: We use the inverted latents to drive the reverse diffusion process with semantic guidance. In each denoising step we compute the noise estimate according to the SEGA logic and compute the updated latents according to the DDPM scheme, using pre-computed noise maps.

where ϵθ

is the neural network noise estimate of xt, and σt = ηβt(1 − α¯t−1)/(1 − α¯t) where βt stands for a variance schedule and η ∈ [0,1] with η = 1 corresponding to the original DDPM work. The edit friendly DDPM inversion method constructs the sequence x1,..,xT such that structures within the image x0 are more strongly “imprinted” into the noise maps z1,...,zT that are extracted by isolating zt from eq.1.

t

#### 2.2 Semantic Guidance

The concept of Semantic Guidance [7] was introduced to enhance fine grained control over the generation process of text guided diffusion models. SEGA extends principles introduced in classifier-free guidance by exclusively interacting with the concepts already present in the model’s latent space. The calculation takes place within the ongoing diffusion iteration and is designed to impact the diffusion process across multiple directions. More specifically, SEGA uses multiple textual descriptions ei, representing the given target concepts of the generated image, in addition to the text prompt p.

### 3 LEDITS - DDPM Inversion X SEGA

We propose a straightforward integration that consists of a simple modification to the SEGA scheme of the diffusion denoising process. This modification allows the flexibility of editing with both methods while still maintaining complete control over the editing effect of each component. First, we apply DDPM inversion on the input image to estimate the latent code associated with it. To apply the editing operations, we perform the denoising loop such that for each timestep t, we repeat the logic used in SEGA but with the DDPM inversion scheme, using the pre-computed noise vectors. More specifically, we start the denoising process with xT computed with DDPM inversion. Let ϵθ

be the the diffusion model’s (DM), noise estimate with semantic guidance (following the SEGA logic) in timestep t. Then we update the latents according to eq.1 such that

t

xt−1 = µˆt(xt;ϵθ

) + σtzt

t

[Figure 3]

- Figure 3: Image editing with LEDITS. LEDITS extends fine-grained control over edit operations and introduces flexibility and versatility. We show images edited purely with DDPM Inversion (forth column from the right) and images edited with LEDITS, using both methods simultaneously (three leftmost and rightmost columns) - these images were edited by using the described target prompt (in black) in addition to SEGA concepts (stated in blue). SEGA semantic vectors maintain their monotonically scaling property when used in LEDITS - the gradual effect of increasing/decreasing the strength of SEGA concepts can be observed from the third column on the right to the rightmost column, and from the third column to the left to the leftmost column.

where zt is the corresponding noise map, obtained from the inversion process. A pseudo-code of our method is summarized in Alg. 1. A general overview is provided in Fig. 2.

Algorithm 1 LEDIT Input: Input image I, target prompt ptar and edit concepts e1,...,ek Output: Output image I˜

- 1: Compute the inverted latent and noise maps xT,z1,...,zT using DDPM inversion over I;
- 2: cp

tar

,ce

1

,...,ce

k ← DM.encode(ptar,e1,...,ek)

- 3: for t = T,...,1 do
- 4: ϵθ

t

= DM.predict − noise(xt,cp

tar

,ce

1

,...,ce

k

)

- 5: xt−1 ← µˆt(xt;ϵθ

t

) + σtzt ▷ update latents

- 6: end for
- 7: I˜ ← DM.decode(x0) return I˜

- 4 Experiments

We explored two editing workflows: The first, using DDPM purely for inversion (i.e. target prompt=””), such that a perfect reconstruction of the original image is achieved and editing is done by performing semantic guidance with SEGA edit concepts. The second is performing two editing operations simultaneously by choosing a target prompt that reflects a desired output, in addition to semantic guidance with SEGA edit concepts.

[Figure 4]

- Figure 4: Comparisons. We show results for editing real images using pure DDPM inversions, DDPM inversion with prompt-to-prompt and LEDITS respectively. Results shown here were obtained with the first editing workflow, using DDPM purely for inversion and SEGA for editing. All images were generated using the same seed.

We observe that both approaches add diversity and versatility to the pure DDPM inversion outputs (figures 4 5), and extend the amount of control over edit operations. In addition, our experiments indicate that SEGA guidance vectors generally maintain their properties of robustness and monotonicity as can be seen in figures 1,3. Our qualitative experiments show competitive results with state-of-the-art methods and demonstrate the following properties: fidelity vs. creativity - The combined approach adds another layer of flexibility in tuning the effect of the desired edit, balancing between preserving the original image semantics and applying creative edits. flexibility and versatility - adding SEGA editing concepts on top of the ddpm edit (reflected in the target prompt) maintains the quality of the DDPM edit (Fig. 1, 3). Complementing capabilities - The combined control can compensate for the limitations of one approach or the other in various cases. In Fig. 5. we explore the effect of the skip-steps and target guidance scale (the strength parameter of the classifier-free scale) parameters on the edited output, when using solely DDPM inversion for the editing operation. In comparison, we also examine the effect of SEGA concepts with increasing edit guidance scales when editing solely with SEGA (and using DDPM for inversion). We observe that the pure DDPM inversion edited outputs and pure SEGA edited outputs range differently on the scale of fidelity to the source image and compliance with the target prompt.

In addition, given the straightforward integration of the two methods, we maintain the performance advantages of the two techniques, thus making this overall approach lightweight.

[Figure 5]

- Figure 5: Parameter effect in DDPM inversion vs. LEDITS. We show the effect of the parameters skip steps and target guidance scale on the output image when using pure DDPM inversion (top panel) compared to the effect of the edit concepts guidance scales when using LEDITS.

### 5 Conclusion

In this report, we explored the combination of the DDPM inversion technique with semantic guidance and introduced LEDITS. We show that this efficient and lightweight approach spans a wide range of editing capabilities and extends the level of fine-grained control users have over the effect of editing operations. Our results indicate LEDITS generally maintains the individual strengths of each method, including SEGA properties such as robustness, and monotonicity. Our qualitative experiments indicate the two techniques can be used simultaneously for independent editing operations leading to more diverse outputs without harming the fidelity to the semantics of the original image and compliance with the editing prompts.

### 6 Limitations

Given the casual and exploratory nature of this report, we leave quantitative evaluations for future works and outside the scope of this work. The purpose of this report was to merely explore and suggest an intuitive editing workflow for real images, demonstrate it’s qualitative abilities and potentially drive further works along this path.

### 7 Methods

#### 7.1 Implementation

The implementation of our approach builds on the Stable Diffusion and Semantic Stable Diffusion pipelines from the HuggingFace diffusers library. For all experiments and evaluations we used StableDiffusion-v-1-5 checkpoint.

For the DDPM Inversion implementation, we used the official implementation at - https://github.com/DDPM-inversion. Our implementation is available on the project’s webpage.

- 7.2 Experiments All images used for our analysis were downloaded from: https://www.pexels.com/.

| |Method target cfg skip warm-up threshold edit concepts cfg τX,τa| |
|---|---|---|

| |LEDITS 15 36 1 0.95 7 -<br><br>DDPM 15 36 - - - DDPM + P2P 9 12 - - - 0.6,0.2| |
|---|---|---|

Table 1: Hyper-parameters used in experiments shown in Fig. 4. Target cfg and skip correspond to strength and Tskip from the original DDPM inversion paper [16] Edit concepts cfg corresponds to the guidance scale used for each SEGA concept individually, threshold and warm-up stand for λ,δ respectively from the original SEGA paper [7]. τX,τa are the cross- and self-attentions parameters used for P2P.

In all experiments, we configured all methods to use 100 forward and backward steps. Table 1 summarizes the huper-parameters we used for all methods to produce the results shown in Fig. 4. DDPM and P2P hyper-parameters used for Fig. 4 were set with identical values to those used in [16] for quantitative assessments.

### References

- [1] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.
- [2] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.
- [3] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.
- [4] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.
- [5] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.
- [6] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.
- [7] Manuel Brack, Felix Friedrich, Dominik Hintersdorf, Lukas Struppek, Patrick Schramowski, and Kristian Kersting. Sega: Instructing diffusion using semantic dimensions. arXiv preprint arXiv:2301.12247, 2023.
- [8] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.
- [9] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. arXiv preprint arXiv:2210.11427, 2022.
- [10] Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. Diffusionclip: Text-guided diffusion models for robust image manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2426–2435, 2022.
- [11] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1921–1930, 2023.
- [12] Bram Wallace, Akash Gokul, and Nikhil Naik. Edict: Exact diffusion inversion via coupled transformations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22532–22541, 2023.
- [13] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023.

- [14] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023.
- [15] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. arXiv preprint arXiv:2302.03027, 2023.
- [16] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly ddpm noise space: Inversion and manipulations. arXiv preprint arXiv:2304.06140, 2023.

