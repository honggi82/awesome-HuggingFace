# arXiv:2408.00735v1[cs.CV]1Aug2024

## TurboEdit: Text-Based Image Editing Using Few-Step Diffusion Models

Gilad Deutch

Rinon Gal

Daniel Garibi

Tel-Aviv University

NVIDIA, Tel-Aviv University

Tel-Aviv University

Or Patashnik

Tel-Aviv University

Daniel Cohen-Or

Tel-Aviv University

[Figure 1]

SDXLTurbo

“Yellow butterfly” “Red flower” “Bee” “Dragonfly”

SDXLTurbo

“Koala” “Racoon” “Dog” “On the beach,

Input

with volleyballs”

Figure 1: Our method enables text-based editing of real images with as few as 3 diffusion steps (0.321 seconds on an A5000 GPU).

### ABSTRACT

these methods commonly rely on the multi-step nature of the diffusion process. In such many-step scenarios, any deviation from train-time statistics can be lessened by injecting smaller changes across many steps, or by allowing the final steps of the diffusion process to draw the result back to the prior distribution, thereby correcting for any artifacts that may arise along the way.

Diffusion models have opened the path to a wide range of textbased image editing frameworks. However, these typically build on the multi-step nature of the diffusion backwards process, and adapting them to distilled, fast-sampling methods has proven surprisingly challenging. Here, we focus on a popular line of text-based editing frameworks - the “edit-friendly” DDPM-noise inversion approach. We analyze its application to fast sampling methods and categorize its failures into two classes: the appearance of visual artifacts, and insufficient editing strength. We trace the artifacts to mismatched noise statistics between inverted noises and the expected noise schedule, and suggest a shifted noise schedule which corrects for this offset. To increase editing strength, we propose a pseudo-guidance approach that efficiently increases the magnitude of edits without introducing new artifacts. All in all, our method enables text-based image editing with as few as three diffusion steps, while providing novel insights into the mechanisms behind popular text-based editing approaches. Project page: https://turboedit-paper.github.io/

Recently, model distillation methods [Salimans and Ho 2022; Sauer et al. 2023; Song et al. 2023] have enabled the creation of “fast” diffusion models, from which novel images can be sampled in few (18) steps. Ideally, we would like to use such models as a backbone for image editing, thereby accelerating the editing process. However, absent sufficient steps, existing text-based editing approaches tend to create noticeable artifacts, or show poor performance.

Here, we propose to address these limitations by analyzing the behaviour of one such family of editing methods — the DDPM noiseinversion framework [Huberman-Spiegelglas et al.2023; Tsaban and Passos 2023; Wu and la Torre 2023]. There, inversion takes the form of a series of per-timestep noise maps. These are pre-calculated, such that using them in place of the random noise samples in the DDPM backwards process, will lead to a re-construction of the original image under a given prompt (see section 3 for more details). Importantly, editing through this approach is simple, requiring only a change of the text describing the image, while still using the same pre-calculated noise samples during the DDPM backwards process. However, attempting to apply this approach to image editing with fast-sampling methods (e.g., SDXL Turbo [Sauer et al. 2023]) leads to

### 1 INTRODUCTION

The unprecedented expressive power of large-scale text-to-image diffusion models [Nichol et al. 2021; Ramesh et al. 2022; Rombach et al. 2022] has contributed to a rise of text-based editing frameworks. Through these frameworks, users are empowered to modify existing, real images using natural language instructions. However,

the creation of severe visual artifacts, and significantly diminished adherence to the novel prompts (i.e. insufficient editing strength).

To tackle the visual artifacts, we analyze the statistics of the inverted noise maps, and observe that they behave more closely to the noise expected from earlier (more noisy) diffusion steps. We hypothesize that in the case of few-step models, there is no time to correct for the noise-distribution shift induced by these corrections, and hence artifacts arise. To overcome them, we propose a shifted denoising schedule, where the denoising sampler is instructed to remove noise as if it was observing an earlier, more noisy step. Additionally, in typical noise-inversion schemes, there is little impact to the noise used at the last denoising step, and indeed it is often simply discarded. In the case of few-step models, the last step’s impact is non-trivial, and we find it helpful to inject noise also in this step, while explicitly normalizing it towards the correct statistics.

To overcome the insufficient editing strength, we first rephrase the noise-inversion approach, demonstrate that it bears similarities to Delta-Denoising methods [Hertz et al. 2023a], and that under certain conditions the DDPM-inversion and Delta-Denoising approaches are exactly equivalent. This re-phrasing not only provides additional insight as to why DDPM noise-inversion approaches are successful, but it reveals that the inversion process itself can be skipped and replaced with an evaluation of a single correction term at each denoising step. Hence, it can be combined into the same batch as the backwards denoising steps themselves, further decreasing editing times. Finally, we identify specific terms in the edit-friendly denoising process that are responsible for the impact of the prompt, and strengthen them in a similar approach to classifierfree guidance (CFG, [Ho and Salimans 2021]). We show that under commonly occurring assumptions, this pseudo-guidance approach is equivalent to re-introducing CFG into the fast-sampling method, but using fewer network evaluation steps.

By combining these components, we enable real-image editing with as few as 3 diffusion steps, achieving ×5-×500 speedup compared with existing editing methods, while preserving and even improving their output quality. Our code will be made public.

### 2 RELATED WORK

Fast Diffusion Sampling. Early diffusion models [Ho et al. 2020; Sohl-Dickstein et al. 2015] were notoriously slow to sample from, requiring hundreds of expensive neural evaluations (and several minutes) to produce a single image. To overcome this hurdle, a range of advanced sampling methods were proposed [Karras et al. 2022; Liu et al. 2022; Lu et al. 2022, 2023; Song et al. 2020]. These are typically grounded in ordinary differential equation solvers, and can successfully reduce the sampling process to several dozen steps without modifying the underlying denoising model.

To further reduce sampling times, a recent line of work proposes to employ a distillation process [Salimans and Ho 2022], where a pre-trained diffusion model is fine-tuned using objectives that promote sampling in few (1-8) steps. These can range from the use of adversarial [Goodfellow et al. 2014] losses [Lin et al. 2024; Sauer et al. 2023] to distribution matching [Yin et al. 2024] and consistency objectives [Kim et al. 2024; Luo et al. 2023a,b; Song et al. 2023].

The emergence of these fast-sampling methods offers an opportunity to speed up existing workflows. However, existing control and editing methods often struggle in this few-step regime. Hence, several approaches were proposed to enable additional controls [Parmar et al. 2024; Xiao et al. 2023] or improve personalization [Gal et al. 2024; Guo et al. 2024] using few-step models. Our work seeks to bring another crucial component into this fast-sampling realm the text-based image editing workflow.

Text-based image editing. The unprecedented semantic control offered by large scale text-to-image diffusion models [Podell et al. 2024; Ramesh et al. 2022; Rombach et al. 2022] has inspired a large volume of work that leverages them as a prior for image editing. Early works proposed simple approaches, such as adding noise to an image and removing it conditioned on a novel prompt [Meng et al.

- 2022]. However, such methods often lead to significant changes in the image shape and layout. Hence, more advanced methods proposed to use in-painting and other localization approaches [Avrahami et al. 2023; Brack et al. 2023; Nichol et al. 2021; Patashnik et al.
- 2023] to modify a single image region. Others works align attention maps [Hertz et al. 2022; Mokady

et al. 2023] or other internal feature representations [Parmar et al. 2023; Tumanyan et al. 2022] to better preserve the original image structure. The outputs of such methods can also be distilled into another diffusion model, trained to modify a conditioning image based on an instruction prompt [Brooks et al. 2023]. In another approach, the diffusion model itself is fine-tuned on the source image [Kawar et al. 2023; Valevski et al. 2023] to better align it to the original content. However, such approaches are often costly in both time and compute. Finally, a recent self-attention sharing approach [Cao et al. 2023] has proven effective in maintaining image content across prompts [Tewel et al.2024], and in transferring styles [Hertz et al. 2023b] or appearances [Alaluf et al. 2023a] across images.

Common to all these approaches is that they operate in a manystep regime, i.e. they typically require dozens of diffusion steps, and struggle when applied to fast sampling models [Parmar et al. 2024]. We aim to enable text-based editing in the fast sampling regime.

Diffusion inversion. Applying editing techniques to a real image commonly requires one to first find some latent representation of the image, which can be fed into the model in order to reconstruct the image. This latent can then be perturbed, directly or through modifications of the generative pass, to affect a change in the image.

Initial inversion efforts focused on GANs [Goodfellow et al. 2014], opting for either direct optimization [Abdal et al. 2019, 2020; Zhu et al. 2016, 2020] or encoder-based approaches [Alaluf et al. 2021a,b; Dinh et al. 2022; Parmar et al. 2022; Richardson et al. 2020; Tov et al. 2021]. With the rise of diffusion-based editing, several works sought to invert real images into the diffusion space, often by determining some initial noise that will be cleaned into a specific image. Initial inversion approaches overwrote the low-frequency content of a generated image with the low-pass filtered content from a source image [Choi et al. 2021], enabling scribble-based modifications or texture changes. Others relied on inverting the deterministic DDIM [Song et al. 2020] process [Dhariwal and Nichol 2021], but these commonly require many steps to be accurate, and struggle to modify the image through inference-time prompt changes [Mokady

et al. 2023]. To overcome this issue, several works intervene in the CFG process and replace the null text condition with a learned embedding that represents the original image [Han et al. 2023; Miyake et al. 2023; Mokady et al. 2023]. However, these commonly require lengthy optimization. As an alternative to optimization, DDIM-based inversion can also be improved by leveraging a fixedpoint iterative scheme [Garibi et al. 2024; Meiri et al. 2023; Pan et al. 2023], but such solutions require dozens of inversion steps even when attached to a fast-sampling backwards process [Luo et al. 2023b].

Rather than focusing on predicting an initial noise that will re-create the image with deterministic sampling, an alternative approach is to consider the DDPM generative process, and invert the image into the intermediate noise maps that are added to a generated image [Huberman-Spiegelglas et al. 2023; Wu and la Torre 2023]. This approach serves as our backbone for few-step editing, and we expand on it in greater detail in section 3.

Finally, some methods invert sets of images into the text conditioning space of the model [Alaluf et al. 2023b; Dong et al. 2022; Gal et al. 2022, 2023; Voynov et al. 2023; Zhang et al. 2023]. However, these are typically used for personalization, where the goal is not to preserve the structure of an image, but to learn a global representation that allows to re-create a concept in novel scenes.

### 3 PRELIMINARIES

We begin with a high level overview of the DDPM noise-inversion approach of Huberman-Spiegelglas et al. [2023]. There, the goal is to find a meaningful latent representation that can be used to reconstruct an image, such that this latent can be manipulated in more intuitive ways than with existing approaches like DDIM inversion. To this end, the authors propose to use the DDPM-noise space (i.e. the noise maps added to the image at each step of the DDPM denoising process), and term it an “edit-friendly” noise space.

More concretely, recall the DDPM denoising equation: 𝑥𝑡−1 = 𝜇𝑡 (𝑥𝑡,𝑐) + 𝜎𝑡𝑧𝑡, 𝑡 = 𝑇, . . ., 1, (1)

where 𝑧𝑡 ∼ N(0, 𝑰), 𝑐 is a conditioning prompt, 𝜇𝑡 (𝑥𝑡,𝑐) is derived from the denoising network’s output through:

1 √𝛼𝑡

1 − 𝛼𝑡 √1 − 𝛼¯𝑡

𝜖𝜃 (𝑥𝑡,𝑡,𝑐) , (2)

𝜇𝑡 (𝑥𝑡,𝑐) =

𝑥𝑡 −

where 𝜖𝜃 (𝑥𝑡,𝑡,𝑐) is the noise prediction and 𝛼𝑡,𝜎𝑡 are derived from the noising schedule.

In the noise-inversion process, given an image 𝑥0, one first calculates its noisy representations through the standard forward diffusion process equation, using a different independently-sampled noise for each time-step:

√1 − 𝛼¯𝑡 𝜖˜𝑡, 1, . . .,𝑇, (3)

𝑥𝑡 = √𝛼¯𝑡𝑥0 +

where 𝜖˜𝑡 ∼ N(0, 𝑰) are statistically independent. This independence is a core difference between the “edit-friendly” approach and prior noise inversion approaches (CycleDiffusion, [Wu and la Torre 2023]), and the authors show that it is crucial for achieving pleasing results.

Given two such noisy images, 𝑥𝑡 and 𝑥𝑡−1, one then calculates the noise that would be needed for eq. (1) to clean 𝑥𝑡 into 𝑥𝑡−1:

𝜎𝑡𝑧𝑡 = 𝑥𝑡−1 − 𝜇𝑡 (𝑥𝑡,𝑐) (4)

Finally, to perform text-based editing, one can simply denoise

an image from the pre-calculated 𝑥𝑇 under a novel prompt 𝑐ˆ, while applying the inverted noises at each step in place of the typical DDPM noise samples, i.e.:

𝑥ˆ𝑡−1 = 𝜇𝑡 (𝑥ˆ𝑡,𝑐ˆ) + 𝜎𝑡𝑧𝑡 = 𝜇𝑡 (𝑥ˆ𝑡,𝑐ˆ) + 𝑥𝑡−1 − 𝜇𝑡 (𝑥𝑡,𝑐). (5)

Throughout the rest of the paper, we make use of this “editfriendly” DDPM noise-inversion technique. We demonstrate how it can be adapted to work with a few-step SDXL-Turbo model, and provide additional insights on the reason behind its success.

### 4 METHOD

In the following section, we outline a series of modifications that can enable the DDPM noise-inversion editing flow to operate in the few-step diffusion domain, and specifically with SDXL-Turbo. We analyze the two primary issues with the baseline results - the appearance of visual artifacts, and insufficient editing strength, and suggest a fix for each. Throughout the process, we shed some additional light on the DDPM-inversion process itself.

### 4.1 Treating the visual artifacts

As noted above, directly applying the DDPM noise-inversion approach to SDXL-Turbo leads to considerable artifacts in the generated images (see figs. 6 and 7). In their original work, HubermanSpiegelglas et al. [2023] observe that their inverted noises follow different dynamics than the standard Gaussian noise used in DDPM generation. Drawing on their observations, we hypothesize that the emergence of visual artifacts can be traced back to such deviations. To verify this hypothesis, we analyze the statistics of the noises derived from a noise-inversion process, and compare them with the scale of noises injected during a standard DDPM generation flow. Specifically, we apply standard edit-friendly inversion using an SDXL model with a few-step schedule, and compare the standard deviation of 𝑥𝑡−1−𝜇𝑡 (𝑥𝑡) i.e. the DDPM-inversion corrections, with 𝑁 (0,𝜎𝑡2), i.e. the standard DDPM noise. Results are in fig. 2.

An immediate observation is that the per-pixel standard deviations of the noise-inversion corrections are higher than those of the DDPM-noise schedule, throughout the entire process. Importantly, this contrasts with the dynamics observed in many-step scenarios [Huberman-Spiegelglas et al. 2023], where the statistics eventually converge towards the end of the generation process. A second observation, is that the noise statistics seem to deviate by an approximately constant margin. In other words, we notice that throughout much of the process, the inverted noises behave like a standard noise originating from a time-step shifted by roughly 200 steps to the past (see also fig. 11 in the supplementary). All-in-all, this disagreement between the noise statistics and the timestep used to condition the model and the denoising step, leads to a train-test mismatch and results in the creation of artifacts.

Following these observations, we propose to alleviate the artifacts by re-aligning the denoising timestep schedule to the scale of the noises. Specifically, during the inversion process, for each timestep 𝑡 we first add noise to 𝑥0 according to the standard DDPM noising schedule, and create𝑥𝑡. However, we inject a larger timestep 𝑡 + Δ into the denoising model and the scheduler’s denoising step:

𝜎𝑡 · 𝑧𝑡 = 𝑥𝑡−1 − 𝜇𝑡+Δ(𝑥𝑡,𝑐), (6)

Edit Friendly

1.0

+ Timestep Shift + Norm Clipping Real Schedule

0.8

0.6

NoiseSTD

0.4

0.2

0.0

200 400 600 800 Time steps

- Figure 2: Comparison of the pixel-wise standard deviations of inverted noise maps, and the expected distribution. The scale of corrections predicted by standard edit-friendly DDPM inversion (red, eq. (4)) is consistently higher than the expected noise schedule (green). The higher values approximately align with a shift along the x-axis: i.e., edit-friendly noise scales align with earlier steps in the diffusion process. We thus propose a time-shifted inversion schedule, where the image is cleaned “as-if” it belonged to a time-point aligning with its noise scale, rather than the real step. In practice, shifting the schedule by a constant 200 steps serves to provide good alignment (blue) and resolve most artifacts. To correct the statistics of the last step, we further apply norm-clipping to the predicted noise at that stage (purple). Shaded regions indicate the 68% confidence interval.

where the subscript 𝑡 + Δ denotes the timestep input of both the denoising network and the scheduler. By applying this correction, we can once again compare the statistics of the inverted noises with standard noise schedule (fig. 2, blue) and see that they are in much better agreement across all timesteps.

Note that for the generative pass, we must also employ a similar shift to keep the two network prediction components synchronized:

𝑥ˆ𝑡−1 = 𝜇𝑡+Δ(𝑥ˆ𝑡,𝑐ˆ) + (𝑥𝑡−1 − 𝜇𝑡+Δ(𝑥𝑡,𝑐)) . (7) This synchronization is crucial for our analysis in section 4.2.

As an additional correction, we find it beneficial to clip the norm of the last noise-inversion correction (fig. 2, purple). The motivation here arises from the contrast with multi-step approaches, where the last correction is small and can be skipped. Here, the last step is large, and its associated correction still captures many of the details of the original image. Hence, we do not want to discard it. Instead, we simply clip it to avoid the introduction of novel artifacts. This is reminiscent of other editing methods, e.g. [Hertz et al. 2022] where the control is decreased towards the end of the generation process.

### 4.2 Improving prompt alignment

Having treated the visual artifacts, we must now deal with the second limitation: insufficient editing strength. The issue here is that changing the prompt between the noise-inversion step and the generation pass, often leads to little or no change in the final image.

To investigate this limitation, we once again consider the editfriendly inference formula (eq. (5)), which we can re-write as:

𝑥ˆ𝑡−1 = 𝑥𝑡−1 + (𝜇𝑡 (𝑥ˆ𝑡,𝑐ˆ) − 𝜇𝑡 (𝑥𝑡,𝑐)) . (8) Under this framing, we can already see a hint to an underlying reason behind the noise-inversion approach’s success. Specifically, the second term captures the difference between the model’s prediction on the edited image, using the novel prompt, and the same prediction on the original image using the original prompt that describes it. This is analogous to the correction term found in Delta Denoising Score (DDS, [Hertz et al. 2023a]) where the authors show that the performance of score distillation sampling methods (SDS, [Poole et al. 2023]) on image editing can be significantly improved if one computes a loss using a similar difference between the network’s prediction on an edited image with the new prompt, and the prediction on the original image with the original prompt that describes it. The intuition provided for such a term in DDS (and later expanded on in NFSD [Katzir et al. 2023]), is that this difference helps cancel out components of the denoising network’s prediction that are unrelated to the prompt (e.g., the direction towards a cleaner image, or any general network bias). The existence of a similar term in the edit-friendly process implies that the same mechanism contributes to its improved performance in image editing tasks. Indeed, we will later show that the connection between EF and DDS is much deeper, with the two methods being functionally equivalent.

We can further expand this analysis of the edit-friendly process by adding and subtracting a cross-term:

𝑐𝑟𝑜𝑠𝑠−𝑝𝑟𝑜𝑚𝑝𝑡

𝑐𝑟𝑜𝑠𝑠−𝑡𝑟𝑎𝑗𝑒𝑐𝑡𝑜𝑟𝑦

𝜇𝑡 (𝑥ˆ𝑡,𝑐ˆ) − 𝜇𝑡 (𝑥ˆ𝑡,𝑐) +

𝜇𝑡 (𝑥ˆ𝑡,𝑐) − 𝜇𝑡 (𝑥𝑡,𝑐) . (9)

𝑥ˆ𝑡−1 = 𝑥𝑡−1 +

Now, we can identify two different directions of change to the original image. The first term is the difference between predictions for the generation trajectory under the novel and original prompts. Hence, one can regard this term as one that represents the direction that takes an image on the novel trajectory from the old prompt to the new one. The second term is the difference between predictions for the new trajectory and the old one, under the same prompt. Hence, it can be regarded as the direction that takes an image from the old trajectory, to the new one. Intuitively, applying both of them to 𝑥𝑡−1, itself a point on the original trajectory, will first shift it to the new trajectory and then pull it further in the direction of the difference between prompts.

Recall that our goal was to strengthen the effect of the prompt. Hence, we can simply draw inspiration from CFG and extrapolate along this cross-prompt direction. Importantly, we do not want to increase the weight of the cross-trajectory term, because we do not want to overshoot the new trajectory. Indeed, in fig. 3 we investigate the behaviour of these two terms and demonstrate that scaling the cross-trajectory term leads to the creation of novel visual artifacts and increased saturation, while scaling the cross-prompt term leads to stronger editing effects. As such, we propose to convert the edit-friendly inference equation as follows:

𝑥ˆ𝑡−1 = 𝑥𝑡−1 +𝜇𝑡 (𝑥ˆ𝑡,𝑐) −𝜇𝑡 (𝑥𝑡,𝑐) +𝑤 · (𝜇𝑡 (𝑥ˆ𝑡,𝑐ˆ) − 𝜇𝑡 (𝑥ˆ𝑡,𝑐)) , (10) where 𝑤 is the pseudo-guidance scale.

𝑤𝑝 = 1 𝑤𝑝 = 1.5 𝑤𝑝 = 2

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Input Image “a polygonal illustration of a cat and a bunny"

𝑡𝑡𝑡2151===𝑤𝑤.𝑤

- Figure 3: We show the effect of increasing the strength of

the cross-prompt term (𝑤𝑝) and cross-trajectory term (𝑤𝑡) in the DDPM inversion. While both terms can help increase the condition in the edited image, as we increase the crosstrajectory term we see artifacts and saturation.

In the supplementary, we further analyze the connection between this guidance formula and a re-introduction of CFG into SDXL-Turbo. Specifically, we show that our approach is equivalent to applying CFG during both inversion and inference, so long as:

𝜇𝑡 (𝑥ˆ𝑡,𝑐) − 𝜇𝑡 (𝑥𝑡,𝑐) ≈ 𝜇𝑡 (𝑥ˆ𝑡,𝜙) − 𝜇𝑡 (𝑥𝑡,𝜙), (11) where 𝜙 indicates the null prompt. Experimentally, we find that this condition commonly holds for SDXL-Turbo predictions. Specifically, we find an average cosine similarity of 0.93 between the two sides of eq. (11), compared to an averaged cosine similarity of −0.04 between these two sides and the cross-prompt term of eq. (9).

Hence our approach roughly aligns with a re-introduction of CFG into SDXL-Turbo, but requires fewer neural function evaluations (3 instead of 4), or equivalently a smaller batch size. In both cases, this allows us to gain further speed improvements over existing approaches. Moreover, this result sheds further light on the observations of Huberman-Spiegelglas et al. [2023], which report better performance when applying CFG during both the inversion and the generation process with multi-step models. If CFG is only applied to one pass and not the other, then CFG also increases the weight of the cross-trajectory term which may harm the results.

Finally, while the derivation of eq. (10) was rooted in the “editfriendly” DDPM-inversion process, we note that its evaluation does not require any pre-computation of a multi-step noise map. Indeed, since neither 𝑥𝑡−1 nor 𝜇𝑡 (𝑥𝑡,𝑐) depend on a multi-step denoising pass (they depend only on a closed-form projection of 𝑥0 via eq. (3)), they can simply be calculated at inference-time, with 𝜇𝑡 (𝑥𝑡,𝑐) being predicted as part of the same batch. This approach is more closely related to CycleDiffusion’s [Wu and la Torre 2023] denoising pass, and can effectively cut the number of editing steps by half.

### 4.3 Connecting EF and DDS

In section 4.2 we observed edit-friendly inversion and DDS share a similar structure. In both cases, the image is being edited by shifting it along the vector between the diffusion predictions on the new trajectory with the new prompt, and the diffusion predictions on the original image with a prompt that describes it. In appendix C we demonstrate that this connection runs deeper, and that under an appropriate choice of timesteps and learning rates during the DDS optimization process, both methods become functionally equivalent, with the DDS corrections terms collapsing to eq. (5). Please see the supplementary for the proof, an empirical demonstration, and a deeper discussion of this equivalency.

### 4.4 Implementation details

We implemented our method on top of SDXL-Turbo [Sauer et al. 2023]. Unless otherwise noted, all results use a pseudo-guidance scale 𝑤 = 1.5 and 4 denoising steps (starting at 𝑡 = 599 with Δ = 200). We clip the norm of the final step corrections to a maximum of 15.5. All experiments are performed on a single NVIDIA A5000 GPU.

### 5 RESULTS

We structure our experimental verification in two parts. In the first part, we demonstrate our methods ability to edit real images in few steps. There, we conduct a series of evaluations, where we contrast our method against a range of baselines. These include both existing multi-step baselines, as well as several few-step alternatives. We demonstrate that our approach can match, or even exceed the quality of current multi-step editing methods, while being significantly faster. Then, we conduct an ablation study where we demonstrate the effect of individual components in our proposed solution.

### 5.1 Evaluation

We being by evaluating our method, starting with a qualitative evaluation. In fig. 4 we show a range of editing results achieved with our method using 4 diffusion steps. These include object-level modifications, style changes, or object replacement. Additional results are shown in fig. 10.

Next, we compare our method against a series of baselines, including both many-step approaches as well as few-step sampling alternatives. In fig. 5 we compare our method to multi-step methods. We consider a large range of approaches, including both optimization based methods (Null-text, [Mokady et al. 2023]), featurepreserving ones (PnP, [Tumanyan et al. 2022]), the baseline editfriendly editing approach and the recently introduced ReNoise [Garibi et al. 2024] which performs multi-step inversion (∼ 40 steps) but re-generates the image via a fast sampling method (∼ 4 steps). Our method achieves comparable or better results than most multi-step baselines, while being significantly faster.

In fig. 6 we compare our method to SDEdit [Meng et al. 2022] and vanilla applications of edit-friendly inversion with SDXL-Turbo using 4 steps. Our method can better maintain the details of the original image, while also achieving better prompt alignment. Notably, the edit-friendly approach creates significant artifacts. Please zoom in to better see the results.

Original Image Editing results

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

“White wedding cake” “Pagoda” “Snowy tower” “Colorful” “Bonsai” “Macaroons”

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

“Poodle” “Cat” “Tiger” “Full moon” “Sitting” “Cardboard”

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

“Watercolor” “Robots” “Wooden sculpture” “In the forest” “Huge babies” “Art gallery”

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

“Bronze” “Robot” “Pink toy” “At sunset” “Made of flowers” “Carousel horse”

#### Figure 4: Qualitative editing results of our method. All results use 4 diffusion steps.

For a quantitative evaluation, we employ 4 commonly used metrics. The first two are the CLIP-space [Radford et al. 2021] similarity between the input image and the editing result (CLIP-I), and the CLIP-space similarity between the target text and the editing result (CLIP-T). We further evaluate the CLIP-directional similarity [Brooks et al. 2023; Gal et al. 2021], i.e. the cosine-similarity between the CLIP-space direction that connects the pre- and postediting image, and the direction that points from a description of the original image to a description of the target edit (CLIP-Dir). Intuitively, this metric measures how well the image-space change aligns with the prompt difference, without changing content that is unrelated to the prompt. Finally, we also compare the LPIPS [Zhang et al. 2018] score between the pre- and post-edited images.

The results are shown in table 1. Notably, our method achieves favorable results on both prompt-alignment metrics, and indeed outperforms all other methods which do not also suffer from significant deviation from the original image. On the image preservation front, our approach does not outperform the state-of-the-art, but it achieves comparable results to many multi-step methods, despite being quicker by a factor of ×5 − ×500.

To further validate our results, we also conduct a user preference study. We used a two-alternative forced choice setup, where each user was shown the source image, the target prompt, and two editing results including our method and one baseline. Users were asked to select the image that better aligns with the prompt, while

preserving the content of the original image. We focused on the 5 most performant baselines, including both multi- and few-step approaches. In total, we collected 202 responses from 25 users. Results are shown in fig. 8. Overall, our approach is strongly preferred over competing few-step methods, while being competitive with multi-step results. Plug & Play achieve higher user preference, but they are slower by a factor of ×500. All in all, these results demonstrate that our approach can successfully edit real images in as few as 3 diffusion steps, without sacrificing quality.

### 5.2 Ablation study

Next, we turn to an ablation study where we analyze the effect of the different components inherent in our approach. Specifically, we consider: (1) the effect of disabling the time-shift applied to the network predictions, (2) the effect of removing the noise-clipping at the final denoising step, (3) the effects of other noise normalization schemes, such as shifting only the last step or directly re-normalizing the noise statistics (4) the effect of disabling the pseudo-guidance, and (5) the performance of the naive edit-friendly DDPM inversion in this few-step setup. Visual results are provided in fig. 7 while quantitative results are found in table 2.

Removing the time-shift maintains the text-alignment, but harms the image-to-image similarity because of the introduction of visual artifacts. A similar effect can be observed to a lesser extent when

Edit-Friendly + P2P

Null-text + P2P

Input Ours (4 steps) Ours (3 steps) PnP Edit-Friendly

ReNoise (1) ReNoise (0.75)

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

“RedFlower"

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

“Dogmade oflego"

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

“Bearcubs inthesnow"

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

“Cartoon"“Withglasses"

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

#### Figure 5: Comparisons against multi-step editing methods. Our results are on-par with existing baselines, while being 𝑥5-𝑥300 faster.

Edit-Friendly (4 steps)

Input Ours (4 steps) Ours (3 steps) SDEdit (0.5) SDEdit (0.75)

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

“Withearrings"“Watercolor"“Wearingascarf"

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

#### Figure 6: Comparisons with few-step methods. Our method can better preserve the content of the original image, while applying meaningful edits.

Table 1: Quantitative comparisons against text-based editing baselines. Bold indicates the best scoring method, underline indicates the second best, and red indicates the worst. EF denotes edit-friendly DDPM-inversion. Editing times include both inversion and generation, and are computed on a machine with a single A5000.

CLIP-T ↑ CLIP-I ↑ CLIP-Dir ↑ LPIPS ↓ Time ↓ EF (100 steps) 0.276 0.760 0.173 0.102 29.4s

EF + P2P (100 steps)

0.266 0.801 0.159 0.090 43.2s Plug & Play 0.281 0.746 0.197 0.116 202s

Null-text + P2P 0.263 0.779 0.194 0.091 150s

- ReNoise (0.75) 0.271 0.759 0.182 0.110 1.75s

- ReNoise (1.0) 0.275 0.727 0.187 0.123 2.30s

Ours (4 steps) 0.291 0.745 0.216 0.118 0.412s Ours (3 steps) 0.291 0.748 0.211 0.118 0.321s

EF (4 steps) 0.269 0.756 0.120 0.105 0.576s EF (3 steps) 0.266 0.737 0.114 0.115 0.429s

SDEdit (0.75) 0.301 0.671 0.193 0.191 0.065s SDEdit (0.5) 0.281 0.751 0.163 0.149 0.098s

removing the noise clipping or when shifting only the last step. Explicit re-normalization of noise maps leads to overly-smoothed results and loss of detail. Without pseudo-guidance, we observe significantly diminished prompt-alignment. Finally, with all methods disabled, the results show significantly diminished scores on all fronts, and have considerable visual artifacts.

Ours w/o guidance

Ours w/ last step shift

Ours w/ STD re-norm

Input

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Ours w/o timestep shift

Ours w/o norm clipping

Ours Edit-friendly

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

“Old car”

- Figure 7: Qualitative ablation results. Removing the pseudoguidance leads to diminished editing results. Removing the timestep shift or the norm clipping leads to increased visual artifacts, and alternative noise normalization schemes may lead to overly smoothed results. Removing both of our components and returning to vanilla “edit-friendly” leads to severe artifacts. Please zoom in to better view the artifacts.

Table 2: Quantitative ablation study. Bold indicates the best scoring method, underline indicates the second best. EF denotes edit-friendly DDPM-inversion.

CLIP-T ↑ CLIP-I ↑ CLIP-Dir ↑ LPIPS ↓ Ours 0.291 0.745 0.216 0.118

w/o timestep shift 0.291 0.736 0.183 0.124 w/o clipping 0.289 0.741 0.197 0.116 Shift last step 0.296 0.728 0.189 0.128

w/ STD re-norm 0.282 0.701 0.192 0.155 w/o guidance 0.274 0.795 0.181 0.096 EF (4 steps) 0.269 0.756 0.120 0.105

SDEdit (0.5)

Renoise (1)

Ours

PnP

Null-text + P2P

EF (100 steps)

0.00 0.25 0.50 0.75 1.00 User Preference [%]

- Figure 8: User study results. We show the % of users who preferred each method when compared to ours. Error bars are the 68% confidence interval. 6 LIMITATIONS InputEdited

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

“Red car” “Wearing a hat” “Crossing arms”

- Figure 9: Method limitations. Our approach may display attribute leakage when editing an object, and may struggle to add novel objects or significantly modify poses.

Our method shares many of the limitations of prior text-based editing work. In particular, it can struggle with changing geometries through the prompt (e.g. asking for a person to cross his arms). Similarly, it may struggle to introduce new objects into the scene

(adding a hat to a horse). Another limitation can be found in prompt leakage to undesired areas of the scene (e.g. turning a car red may lead to a shift in background color). In addition, it can struggle with stylization when converting real imagery to painting or vice versa. We show examples of limitations and failure cases in fig. 9.

### 7 CONCLUSIONS

We presented TurboEdit, a fast text-based editing method that leverages newly introduced fast-sampling methods to significantly reduce the time it takes to manipulate an image. Notably, our method achieves sub-second editing times, enabling an interactive editing experience.

Our approach is derived from a deeper analysis of the dynamics of existing noise-inversion approaches, leading to a series of technically simple fixes. Hence, we hope that our work can provide not only a useful application for creative workflows, but also shed a light on components and ideas that should be kept in mind when adapting other tools to the few-step regime.

In the future, we hope to investigate further improvements for editing flows in the few-step regime, such as a more principled alignment of noise schedules to the expected statistics, or exploring ways to improve geometric modifications and object insertion.

### Acknowledgements

This work was partially supported by ISF (grants 2492/20 and 3441/21).

### REFERENCES

- Rameen Abdal, Yipeng Qin, and Peter Wonka. 2019. Image2stylegan: How to embed images into the stylegan latent space?. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 4432–4441.
- Rameen Abdal, Yipeng Qin, and Peter Wonka. 2020. Image2stylegan++: How to edit the embedded images?. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 8296–8305.

Yuval Alaluf, Daniel Garibi, Or Patashnik, Hadar Averbuch-Elor, and Daniel Cohen-Or. 2023a. Cross-image attention for zero-shot appearance transfer. arXiv preprint arXiv:2311.03335 (2023).

Yuval Alaluf, Or Patashnik, and Daniel Cohen-Or. 2021a. ReStyle: A Residual-Based StyleGAN Encoder via Iterative Refinement. arXiv preprint arXiv:2104.02699 (2021).

Yuval Alaluf, Elad Richardson, Gal Metzer, and Daniel Cohen-Or. 2023b. A Neural SpaceTime Representation for Text-to-Image Personalization. arXiv:2305.15391 [cs.CV]

Yuval Alaluf, Omer Tov, Ron Mokady, Rinon Gal, and Amit H. Bermano. 2021b. HyperStyle: StyleGAN Inversion with HyperNetworks for Real Image Editing. arXiv:2111.15666 [cs.CV]

Omri Avrahami, Ohad Fried, and Dani Lischinski. 2023. Blended Latent Diffusion. ACM Trans. Graph. 42, 4, Article 149 (jul 2023), 11 pages. https://doi.org/10.1145/3592450

Manuel Brack, Felix Friedrich, Katharina Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting, and Apolinário Passos. 2023. Ledits++: Limitless image editing using text-to-image models. arXiv preprint arXiv:2311.16711 (2023).

Tim Brooks, Aleksander Holynski, and Alexei A. Efros. 2023. InstructPix2Pix: Learning to Follow Image Editing Instructions. In CVPR.

Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. 2023. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 22560–22570.

Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. 2021. ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE Computer Society, 14347–14356.

Prafulla Dhariwal and Alexander Nichol. 2021. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems 34 (2021), 8780–8794.

Tan M Dinh, Anh Tuan Tran, Rang Nguyen, and Binh-Son Hua. 2022. Hyperinverter: Improving stylegan inversion via hypernetwork. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 11389–11398.

Ziyi Dong, Pengxu Wei, and Liang Lin. 2022. Dreamartist: Towards controllable oneshot text-to-image generation via positive-negative prompt-tuning. arXiv preprint arXiv:2211.11337 (2022).

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. 2022. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022).

Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. 2023. Designing an encoder for fast personalization of text-to-image models. arXiv preprint arXiv:2302.12228 (2023).

Rinon Gal, Or Lichter, Elad Richardson, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. 2024. LCM-Lookahead for Encoder-based Text-to-Image Personalization. arXiv:2404.03620 [cs.CV]

Rinon Gal, Or Patashnik, Haggai Maron, Gal Chechik, and Daniel Cohen-Or. 2021. Stylegan-nada: Clip-guided domain adaptation of image generators. arXiv preprint arXiv:2108.00946 (2021).

Daniel Garibi, Or Patashnik, Andrey Voynov, Hadar Averbuch-Elor, and Daniel CohenOr. 2024. ReNoise: Real Image Inversion Through Iterative Noising. arXiv preprint arXiv:2403.14602 (2024).

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative adversarial nets. Advances in neural information processing systems 27 (2014).

Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, and Qian He. 2024. PuLID: Pure and Lightning ID Customization via Contrastive Alignment. arXiv preprint arXiv:2404.16022 (2024).

Ligong Han, Song Wen, Qi Chen, Zhixing Zhang, Kunpeng Song, Mengwei Ren, Ruijiang Gao, Yuxiao Chen, Di Liu, Qilong Zhangli, et al. 2023. Improving negativeprompt inversion via proximal guidance. arXiv preprint arXiv:2306.05414 (2023).

Amir Hertz, Kfir Aberman, and Daniel Cohen-Or. 2023a. Delta Denoising Score. arXiv:2304.07090 [cs.CV]

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2022. Prompt-to-prompt image editing with cross attention control.

- (2022).

Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. 2023b. Style aligned image generation via shared attention. arXiv preprint arXiv:2312.02133

- (2023).

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems 33 (2020), 6840–6851. Jonathan Ho and Tim Salimans. 2021. Classifier-Free Diffusion Guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.

Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. 2023. An Edit Friendly DDPM Noise Space: Inversion and Manipulations. arXiv preprint arXiv:2304.06140 (2023).

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. 2022. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems 35 (2022), 26565–26577.

Oren Katzir, Or Patashnik, Daniel Cohen-Or, and Dani Lischinski. 2023. Noise-Free Score Distillation. arXiv:2310.17590 [cs.CV]

Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. 2023. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6007–6017.

Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. 2024. Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion. In The Twelfth International Conference on Learning Representations. https://openreview. net/forum?id=ymjI8feDTD

Juil Koo, Chanho Park, and Minhyuk Sung. 2024. Posterior distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13352–13361.

Shanchuan Lin, Anran Wang, and Xiao Yang. 2024. SDXL-Lightning: Progressive Adversarial Diffusion Distillation. arXiv:2402.13929 [cs.CV]

Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. 2022. Pseudo Numerical Methods for Diffusion Models on Manifolds. In International Conference on Learning Representations. https://openreview.net/forum?id=PlKWVd2yBkY

- Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. 2022. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. In Advances in Neural Information Processing Systems, Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (Eds.). https://openreview. net/forum?id=2uAaGwlP_V
- Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. 2023. DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models. https://openreview.net/forum?id=4vGwQqviud5

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. 2023a. Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference. arXiv:2310.04378 [cs.CV]

Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolinário Passos, Longbo Huang, Jian Li, and Hang Zhao. 2023b. LCM-LoRA: A Universal StableDiffusion Acceleration Module. arXiv:2311.05556 [cs.CV]

Original Image Editing results

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

“Snowing” “Watercolor” “Zebra” “Tiger” “Dragon” “On a skateboard”

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

“Golden sculpture” “Wooden statue” “At the beach” “Monk” “Batman” “Elmo”

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

“Bear” “Hippo” “Maneki Neko” “Van Gogh” “Made of metal” “Colorful lotus”

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

“Halloween pumpkin” “Near Fuji” “Shopping at Walmart” “Cherry blossom” “Minecraft graphics” “Bouncy castle”

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

“The Earth” “Fireball” “Marble sculpture” “Futuristic robot” “The galaxy” “Crystal skull”

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

“Simpsons characters” “In a basket” “Cracked” “In a salad” “Marbles” “Macaroons”

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

“Fox” “Wearing a crown” “Robot” “Wearing sunglasses” “On a mushroom” “Monkey”

#### Figure 10: Additional qualitative editing results of our method. All results use 4 diffusion steps.

Barak Meiri, Dvir Samuel, Nir Darshan, Gal Chechik, Shai Avidan, and Rami Ben-Ari.

2023. Fixed-point Inversion for Text-to-image diffusion models. arXiv preprint arXiv:2312.12540 (2023).

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. 2022. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations. In International Conference on Learning Representations. Daiki Miyake, Akihiro Iohara, Yu Saito, and Toshiyuki Tanaka. 2023. Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models. arXiv preprint arXiv:2305.16807 (2023).

Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2023. Nulltext inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6038–6047.

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. 2021. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741 (2021).

Zhihong Pan, Riccardo Gherardi, Xiufeng Xie, and Stephen Huang. 2023. Effective Real Image Editing with Accelerated Iterative Diffusion Inversion. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 15912–15921. Gaurav Parmar, Yijun Li, Jingwan Lu, Richard Zhang, Jun-Yan Zhu, and Krishna Kumar Singh. 2022. Spatially-Adaptive Multilayer Selection for GAN Inversion and Editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 11399–11409.

Gaurav Parmar, Taesung Park, Srinivasa Narasimhan, and Jun-Yan Zhu. 2024. OneStep Image Translation with Text-to-Image Models. arXiv preprint arXiv:2403.12036

(2024). Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and JunYan Zhu. 2023. Zero-shot Image-to-Image Translation. arXiv:2302.03027 [cs.CV]

Or Patashnik, Daniel Garibi, Idan Azuri, Hadar Averbuch-Elor, and Daniel CohenOr. 2023. Localizing object-level shape variations with text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 23051–23061.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2024. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=di52zR8xgf Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. 2023. DreamFusion: Text-to-3D using 2D Diffusion. In The Eleventh International Conference on Learning Representations. https://openreview.net/forum?id=FjNys5c7VyY

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020 (2021).

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 (2022).

Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro, and Daniel Cohen-Or. 2020. Encoding in Style: a StyleGAN Encoder for Image-toImage Translation. arXiv preprint arXiv:2008.00951 (2020).

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-Resolution Image Synthesis with Latent Diffusion Models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022. IEEE, 10674–10685. https://doi.org/10.1109/CVPR52688. 2022.01042

Tim Salimans and Jonathan Ho. 2022. Progressive Distillation for Fast Sampling of Diffusion Models. In International Conference on Learning Representations. https: //openreview.net/forum?id=TIdIXIpzhoI

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. 2023. Adversarial Diffusion Distillation. arXiv:2311.17042 [cs.CV]

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. 2015. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning. PMLR, 2256–2265.

Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020. Denoising Diffusion Implicit Models. In International Conference on Learning Representations.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. 2023. Consistency models. In Proceedings of the 40th International Conference on Machine Learning. 32211–32252.

Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, and Yuval Atzmon. 2024. Training-Free Consistent Text-to-Image Generation. arXiv:2402.03286 [cs.CV]

Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, and Daniel Cohen-Or. 2021. Designing an Encoder for StyleGAN Image Manipulation. arXiv preprint arXiv:2102.02766 (2021).

Linoy Tsaban and Apolinário Passos. 2023. Ledits: Real image editing with ddpm inversion and semantic guidance. arXiv preprint arXiv:2307.00522 (2023).

Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. 2022. Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation. arXiv preprint arXiv:2211.12572 (2022).

Dani Valevski, Matan Kalman, Eyal Molad, Eyal Segalis, Yossi Matias, and Yaniv Leviathan. 2023. Unitune: Text-driven image editing by fine tuning a diffusion model on a single image. ACM Transactions on Graphics (TOG) 42, 4 (2023), 1–10.

Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. 2023. p+: Extended textual conditioning in text-to-image generation. arXiv preprint arXiv:2303.09522

(2023). Chen Henry Wu and Fernando De la Torre. 2023. A Latent Space of Stochastic Diffusion Models for Zero-Shot Image Editing and Guidance. In ICCV.

Jie Xiao, Kai Zhu, Han Zhang, Zhiheng Liu, Yujun Shen, Yu Liu, Xueyang Fu, and ZhengJun Zha. 2023. CCM: Adding Conditional Controls to Text-to-Image Consistency Models. arXiv preprint arXiv:2312.06971 (2023).

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T Freeman, and Taesung Park. 2024. One-step Diffusion with Distribution Matching Distillation. CVPR (2024).

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In CVPR.

Yuxin Zhang, Weiming Dong, Fan Tang, Nisha Huang, Haibin Huang, Chongyang Ma, Tong-Yee Lee, Oliver Deussen, and Changsheng Xu. 2023. ProSpect: Prompt Spectrum for Attribute-Aware Personalization of Diffusion Models. ACM Transactions on Graphics (TOG) 42, 6 (2023), 244:1–244:14.

Jun-Yan Zhu, Philipp Krähenbühl, Eli Shechtman, and Alexei A Efros. 2016. Generative visual manipulation on the natural image manifold. In European conference on computer vision. Springer, 597–613.

Peihao Zhu, Rameen Abdal, Yipeng Qin, and Peter Wonka. 2020. Improved StyleGAN Embedding: Where are the Good Latents? arXiv:2012.09036 [cs.CV]

Supplementary Materials

- A CONNECTION BETWEEN PSEUDO-GUIDANCE AND CLASSIFIER-FREE GUIDANCE

Let us now provide proof, that under the assumption that

𝜇𝑡 (𝑥ˆ𝑡,𝑐) − 𝜇𝑡 (𝑥𝑡,𝑐) ≈ 𝜇𝑡 (𝑥ˆ𝑡,𝜙) − 𝜇𝑡 (𝑥𝑡,𝜙) (12) Our proposed guidance is equivalent to using CFG but with fewer network evaluations. We consider the case where CFG is applied bothduringinversionandduring generation. Let 𝜇 (𝑥,𝜙 + 𝜆𝑠 · (𝑐 − 𝜙)) be the notation for a U-net prediction followed by a scheduler step with CFG strength 𝜆𝑠. Then, the following holds:

𝜇 (𝑥,𝜙 + 𝜆𝑠 · (𝑐 − 𝜙)) = 𝜇 (𝑥,𝜙) + 𝜆𝑠 · [𝜇 (𝑥,𝑐) − 𝜇 (𝑥,𝜙)], (13) i.e., one can first perform a scheduler step after summing the U-net predictions, or first perform two scheduler steps and then sum the results. We can now write the inference equation as:

𝑥ˆ𝑡−1 = 𝑥𝑡−1 + [𝜇 (𝑥ˆ𝑡,𝜙 + 𝜆𝑠(𝑐ˆ − 𝜙)) − 𝜇 (𝑥𝑡,𝜙 + 𝜆𝑠(𝑐 − 𝜙))]. (14)

Adding and subtracting the cross term 𝜇𝑡 (𝑥ˆ𝑡,𝜙 + 𝜆𝑠(𝑐 − 𝜙)), we get:

𝑥ˆ𝑡−1 = 𝑥𝑡−1 + [𝜇 (𝑥ˆ𝑡,𝜙 + 𝜆𝑠(𝑐ˆ − 𝜙)) − 𝜇𝑡 (𝑥ˆ𝑡,𝜙 + 𝜆𝑠(𝑐 − 𝜙)) + 𝜇𝑡 (𝑥ˆ𝑡,𝜙 + 𝜆𝑠(𝑐 − 𝜙)) − 𝜇 (𝑥𝑡,𝜙 + 𝜆𝑠(𝑐 − 𝜙))]

= 𝑥𝑡−1 + [𝜇 (𝑥ˆ𝑡,𝜙) + 𝜆𝑠 [𝜇 (𝑥ˆ𝑡,𝑐ˆ) − 𝜇 (𝑥ˆ𝑡,𝜙)]

− 𝜇 (𝑥ˆ𝑡,𝜙) − 𝜆𝑠 [𝜇 (𝑥ˆ𝑡,𝑐) − 𝜇 (𝑥ˆ𝑡,𝜙)] + 𝜇 (𝑥ˆ𝑡,𝜙) + 𝜆𝑠 [𝜇 (𝑥ˆ𝑡,𝑐) − 𝜇 (𝑥ˆ𝑡,𝜙)] − 𝜇 (𝑥𝑡,𝜙) − 𝜆𝑠 [𝜇 (𝑥𝑡,𝑐) − 𝜇 (𝑥𝑡,𝜙)]]

= 𝑥𝑡−1 + 𝜇 (𝑥ˆ𝑡,𝜙) − 𝜇 (𝑥𝑡,𝜙) + 𝜆𝑠 [

𝜇 (𝑥ˆ𝑡,𝑐ˆ) − 𝜇 (𝑥ˆ𝑡,𝜙) − 𝜇 (𝑥ˆ𝑡,𝑐) + 𝜇 (𝑥ˆ𝑡,𝜙)

+ 𝜇 (𝑥ˆ𝑡,𝑐) − 𝜇 (𝑥ˆ𝑡,𝜙) − 𝜇 (𝑥𝑡,𝑐) + 𝜇 (𝑥𝑡,𝜙)]

= 𝑥𝑡−1 + 𝜇 (𝑥ˆ𝑡,𝜙) − 𝜇 (𝑥𝑡,𝜙) + 𝜆𝑠 [ (15)

𝜇 (𝑥ˆ𝑡,𝑐ˆ) − 𝜇 (𝑥ˆ𝑡,𝑐)

+ (𝜇 (𝑥ˆ𝑡,𝑐) − 𝜇 (𝑥𝑡,𝑐)) − (𝜇 (𝑥ˆ𝑡,𝜙) − 𝜇 (𝑥𝑡,𝜙))] (16)

If 𝜇𝑡 (𝑥ˆ𝑡,𝑐) − 𝜇𝑡 (𝑥𝑡,𝑐) ≈ 𝜇𝑡 (𝑥ˆ𝑡,𝜙) − 𝜇𝑡 (𝑥𝑡,𝜙) then the term in line (15) is approximately zero and line (14) can be replaced: 𝜇 (𝑥ˆ𝑡,𝜙) − 𝜇 (𝑥𝑡,𝜙) → 𝜇 (𝑥ˆ𝑡,𝑐) − 𝜇 (𝑥𝑡,𝑐) and we get that DDPM inversion with CFG is equivalent to our proposed guidance. Empirically, we find that relation of eq. (12) often holds for 3 and 4 steps with SDXL-Turbo, but we offer no guarantees for the general case.

- B FURTHER ANALYSIS ON TIME SHIFTS

Here we provide further analysis on the behaviour of the inversion noise scales, and demonstrate the importance of applying the shifts throughout the entire generation process. In fig. 11(blue) we plot the distribution of offsets between the DDPM-inversion corrections and the denoising step for which the true noise schedule is closest to this step. These shifts are approximately distributed around the 200-step shift, which inspires our choice to correct for them through a 200 step offset. We further plot the same distribution of offsets

after applying our fix. Now, most predicted noises match the step in which they are used, and the overall shift is diminished.

In fig. 12 we ask whether it is important to shift the schedule throughout the entire generative process, or if it is sufficient to shift only the last step. The intuition behind this experiment is that applying only a final-step shift can lead to increased smoothing of the image (due to the removal of extra noise), and possibly account for the artifact removal on its own. However, we find experimentally that this is not the case. As can be seen, the unshifted steps accumulate strong artifacts, and the final step fails to overcome them. Moreover, we observe that this approach leads to worse preservation of background objects. All in all, we find that shifting the entire schedule is beneficial.

Without Shift With Shift

| |
|---|

0.5

0.4

Density

0.3

0.2

0.1

0.0

200 100 0 100 200 300 400 500 Noise scale offset

Figure 11: For each image in our evaluation set, and for each denoising time step, we calculate the time-offset between the inverted noises and the closest point on the true noise schedule that has the same level of noise. With vanilla edit-friendly DDPM inversion (blue), nearly all offsets are concentrated in the [100, 300] step region. When inverting using a shifted schedule (green), the offsets are more concentrated around 0.

### C EDIT-FRIENDLY AND DELTA-DENOISING EQUIVALENCE

Here, we provide proof that Delta Denoising Score (DDS) and EditFriendly DDPM Inversion (EF) are equivalent, under an appropriate choice of learning rate and timestep sampling for the DDS method.

First, recall the DDPM sampler’s definition of 𝜇𝑡 (𝑥𝑡,𝑐):

1 − 𝛼𝑡 √1 − 𝛼¯𝑡

1 √𝛼𝑡

𝝐𝜙𝑡 (x𝑡,𝑐) , (17) and the noising equation:

x𝑡 −

𝜇𝑡 (𝑥𝑡,𝑐) =

√1 − 𝛼¯𝑡 𝜖˜𝑡, (18)

𝑥𝑡 = √𝛼¯𝑡𝑥0 +

where 𝛼𝑡 is derived from the diffusion noising schedule and 𝛼¯𝑡 = 𝑡 𝑠=1 𝛼𝑠 and 𝝐𝜙𝑡 denotes the diffusion model’s output.

For simplicity, we denote 𝑎𝑡 = √1𝛼𝑡 , 𝑏𝑡 = √11−−𝛼𝛼¯𝑡𝑡 , 𝑐𝑡 = √𝛼¯𝑡 and 𝑑𝑡 = √1 − 𝛼¯𝑡, and re-write these equations as:

𝜇𝑡 (𝑥𝑡,𝑐) = 𝑎𝑡 𝑥𝑡 − 𝑏𝑡𝜖𝑡𝜙 (𝑥𝑡,𝑐) , (19)

𝑥𝑡 = 𝑐𝑡𝑥0 + 𝑑𝑡𝜖˜𝑡 (20) Next, recall that DDS operates through an iterative optimization

scheme. Let 𝑥 be the original image, 𝑥ˆ𝑖 the optimized image at

Final step timestep shift

Full timestep shift

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Input Image "wooden statue"

Iteration2Iteration3Iteration4Iteration1

Figure 12: Comparison of our full timestep shift approach with an extra shifted timestep at the final step. We show the edited image at each iteration of the edit process before adding the noise. The extra shifted step has a "cleaning" effect that removes some of the artifacts, yet the image is already too damaged and dissimilar to the original image. Our full timestep shift approach keeps the image visually appealing at each step.

iteration 𝑖, and let 𝑥𝑡 and 𝑥ˆ𝑡𝑖 be their respective representations after noising to time 𝑡 via eq. (20). The DDS gradient is then:

∇𝑥ˆLDDS = 𝜖𝑡𝜙 𝑥 ˆ𝑡𝑖,𝑐ˆ − 𝜖𝑡𝜙 (𝑥𝑡,𝑐) , (21) where 𝑐 and 𝑐ˆ are the prompts describing the original image and the edit target respectively and 𝑡 is sampled randomly for each optimization iteration. Since this gradient is applied directly to 𝑥ˆ𝑖, we thus have:

𝑥ˆ𝑖+1 = 𝑥ˆ𝑖 −𝛾(𝜖𝑡𝜙 (𝑥ˆ𝑡𝑖,𝑐ˆ) − 𝜖𝑡𝜙 (𝑥𝑡,𝑐)) (22) Next, we consider DDS under the following two conditions:

- (1) Rather than sampling diffusion timesteps randomly, we do so sequentially following a standard denoising schedule (i.e. we start

with 𝑡 = 𝑇 and move towards 𝑡 = 0). We will thus use 𝑥ˆ0𝑡 to denote the un-noised, DDS-optimized image at the iteration corresponding

to timestep 𝑡, and 𝑥ˆ𝑡𝑡 to denote its noised version at this timestep.

- (2) We select a time-dependent learning rate given by:

1 − 𝛼𝑡 √𝛼¯𝑡√1 − 𝛼¯𝑡

𝑏𝑡 𝑐𝑡

(23) Plugging these conditions into eq. (22) we have:

=

𝛾𝑡 =

𝑏𝑡 𝑐𝑡 (𝜖𝑡𝜙 (𝑥ˆ𝑡𝑡,𝑐ˆ) − 𝜖𝑡𝜙 (𝑥𝑡,𝑐))

𝑥ˆ0𝑡−1 = 𝑥ˆ0𝑡 −

1 𝑐𝑡

1 𝑐𝑡

𝑏𝑡 𝑐𝑡 (𝜖𝑡𝜙 (𝑥ˆ𝑡𝑡,𝑐ˆ) − 𝜖𝑡𝜙 (𝑥𝑡,𝑐)) +

== 𝑥ˆ0𝑡 −

𝑥𝑡 −

𝑥𝑡, and by repeatedly applying eqs. (19) and (20) we have:

1 𝑐𝑡

1 𝑐𝑡

𝑏𝑡 𝑐𝑡 (𝜖𝑡𝜙 (𝑥ˆ𝑡𝑡,𝑐ˆ) − 𝜖𝑡𝜙 (𝑥𝑡,𝑐)) +

𝑥ˆ0𝑡−1 = 𝑥ˆ0𝑡 −

𝑥𝑡 −

𝑥𝑡

1 𝑐𝑡

𝑏𝑡 𝑐𝑡

𝑏𝑡 𝑐𝑡

𝑑𝑡 𝑐𝑡

𝜖𝑡𝜙 (𝑥𝑡,𝑐) −

𝜖𝑡𝜙 (𝑥ˆ𝑡𝑡,𝑐ˆ) + 𝑥0 +

= 𝑥ˆ0𝑡 +

𝜖˜𝑡

𝑥𝑡 −

1 𝑐𝑡 (𝑐𝑡𝑥ˆ0𝑡 + 𝑑𝑡𝜖˜𝑡) −

1 𝑐𝑡 (𝑥𝑡 − 𝑏𝑡𝜖𝑡𝜙 (𝑥𝑡,𝑐)) −

𝑏𝑡 𝑐𝑡

𝜖𝑡𝜙 (𝑥ˆ𝑡,𝑐ˆ) + 𝑥0

=

1 𝑐𝑡𝑎𝑡

1 𝑐𝑡

𝑏𝑡 𝑐𝑡

𝜖𝑡𝜙 (𝑥ˆ𝑡,𝑐ˆ) + 𝑥0

𝑥ˆ𝑡 −

=

𝜇(𝑥𝑡,𝑐) −

1 𝑐𝑡 (𝑥ˆ𝑡 − 𝑏𝑡𝜖𝑡𝜙 (𝑥ˆ𝑡,𝑐ˆ)) −

1 𝑐𝑡𝑎𝑡

= 𝑥0 +

𝜇(𝑥𝑡,𝑐)

1

𝑐𝑡𝑎𝑡 (𝜇(𝑥ˆ𝑡,𝑐ˆ) − 𝜇(𝑥𝑡,𝑐)). (24) From the definitions of 𝑐𝑡, 𝑎𝑡 and 𝛼¯𝑡 we further have:

= 𝑥0 +

−1

−1

𝑡−1

𝑡

1 √𝛼𝑡

1 𝑐𝑡−1

1 𝑐𝑡𝑎𝑡

(25)

=

=

=

𝛼𝑠

𝛼𝑠

𝑠=1

𝑠=1

Adding noise (eq. (20)) to eq. (24) and plugging in this constant relation gives:

1 𝑐𝑡−1 (𝜇(𝑥ˆ𝑡,𝑐ˆ)) − 𝜇(𝑥𝑡,𝑐)) + 𝑑𝑡−1𝜖˜𝑡−1

𝑥ˆ𝑡−1 = 𝑐𝑡−1(𝑥0 +

= (𝑐𝑡−1𝑥0 + 𝑑𝑡−1𝜖˜𝑡−1) + (𝜇(𝑥ˆ𝑡,𝑐ˆ) − 𝜇(𝑥𝑡,𝑐))

= 𝑥𝑡−1 + (𝜇(𝑥ˆ𝑡,𝑐ˆ) − 𝜇(𝑥𝑡,𝑐))

which is exactly equal to the edit friendly step (eq. (5)).

This surprising result confirms that DDS and Edit-friendly share not only a similar correction term, but indeed under the specific choice of consecutive timesteps samples and a specific learning rate, the two methods converge to the exact same update rule. We further verify this empirically by editing a set of images using both methods, where DDS uses the modified schedule and learning rate. The results are shown in fig. 13. As can be observed, the results are visually identical, further confirming our theoretical observation.

Finally, we note that a similar analysis can be applied to methods that combine DDS and EF (e.g., Posterior Distillation Sampling [Koo et al. 2024]). Indeed, through an appropriate choice of learning rates and sampling steps, such methods can also be made to follow eq. (5) for the specific case of image editing.

Input DDS Edit Friendly

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

WatercolorPixelArtApplesCityOranges

#### Figure 13: We show empirically that DDS with sequential timestep sampling and the learning rate of eq. (23) is equivalent to edit-friendly DDPM inversion. We run both methods using the same number of steps and guidance.

