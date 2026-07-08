## VideoGuide: Improving Video Diffusion Models without Training Through a Teacher’s Guide

# arXiv:2410.04364v3[cs.CV]8Dec2024

Dohun Lee∗, Bryan S Kim∗, Geon Yeong Park, Jong Chul Ye Kim Jaechul Graduate School of AI, KAIST ∗: Equal Contribution

{leedh7, bryanswkim, pky3436, jong.ye}@kaist.ac.kr

###### Base Ours Base Ours

"A drone view of celebration with Christmas tree and fireworks" "A boat sailing in the middle of the ocean"

###### Base Ours Base Ours

"Slow motion footage of a racing car" "A dog drinking water"

Figure 1. VideoGuide is a novel framework for improving temporal consistency while preserving imaging quality, enabling high-quality video generation for diverse text prompts. By applying VideoGuide to underperforming base models, we can significantly improve temporal consistency with no additional training or fine-tuning. Best viewed with Acrobat Reader. Click each image to play the video clip.

#### Abstract

Text-to-image (T2I) diffusion models have revolutionized visual content creation, but extending these capabilities to textto-video (T2V) generation remains a challenge, particularly in preserving temporal consistency. Existing methods that aim to improve consistency often cause trade-offs such as reduced imaging quality and impractical computational time. To address these issues we introduce VideoGuide, a novel

framework that enhances the temporal consistency of pretrained T2V models without the need for additional training or fine-tuning. Instead, VideoGuide leverages any pretrained video diffusion model (VDM) or itself as a guide during the early stages of inference, improving temporal quality by interpolating the guiding model’s denoised samples into the sampling model’s denoising process. The proposed method brings about significant improvement in temporal consistency and image fidelity, providing a cost-effective and prac-

tical solution that synergizes the strengths of various video diffusion models. Furthermore, we demonstrate prior distillation, revealing that base models can achieve enhanced text coherence by utilizing the superior data prior of the guiding model through the proposed method. Project Page: https:

//dohunlee1.github.io/videoguide.github.io/

#### 1. Introduction

Text-to-image (T2I) diffusion models have made significant advances in visual generation, enabling user interactive image generation with enriched text descriptions. Now the AI community is looking deeper into the potential of T2I diffusion models, exploring their application to the higher dimensional field of video generation. Text-to-video (T2V) diffusion models aim to extend the capabilities of their imagebased counterparts by generating coherent video sequences from text descriptions, handling both spatial and temporal dimensions simultaneously.

Text-to-video (T2V) diffusion models often face a challenging trade-off between temporal consistency and image quality, where improvements in one frequently degrade the other. This compromise results in diminished perceived quality and negatively impacts downstream tasks, such as T2V personalization. Although recent works [4, 35] have attempted to address aspects of temporal quality, they often do so at the expense of either visual fidelity or inference speed. In this work, we address the clear need for a robust approach that refines the temporal stability of pretrained T2V models without sacrificing image quality. We propose a novel framework that greatly improves the quality of generated samples without requiring any training or fine-tuning.

Specifically, we introduce VideoGuide, a general framework that uses any pretrained video diffusion model as a guide during early steps of reverse diffusion sampling. The choice of pretrained teacher VDM is flexible; it can be freely selected from any existing VDMs, or even be the model itself. In any case, the VDM that acts as the guide provides a consistent video trajectory by proceeding in its own denoising for a small number of steps. The teacher model’s denoised sample is then interpolated with the original denoising process to guide the sample towards a direction with better temporal quality. Through interpolation, the student VDM is able to follow the temporal consistency of the teacher VDM to produce samples of enhanced quality. Such interpolation only needs to be involved in the first few steps of inference, but is strong enough to guide the entire denoising process towards more desirable results.

VideoGuide is a versatile framework that allows any pretrained video diffusion model to be used for distillation in a plug-and-play fashion. By integrating a high-performing VDM as a video guide, our framework elevates underperforming VDMs to achieve state-of-the-art quality, which is

particularly useful when the student model possesses unique traits unavailable for the teacher model. Additionally, we find that interpolating the teacher model’s denoised sample provides the student model with an enhanced noise prior, guiding it to generate samples previously unattainable.

In particular, we show two representative cases of how VideoGuide can be applied to combine the best of both worlds: unique functions provided by the student model and high temporal stability provided by the teacher model. In AnimateDiff [13], a motion module is trained that can be interleaved into any pretrained T2I model. The scheme works for any personalized image diffusion model and grants easy application of controllable and extensible modules [12, 38], but not without consequences. Specifically, fixing the T2I weights limits interaction between the temporal module and generated spatial features, hence harming temporal consistency. Applying VideoGuide with an open-source state-ofthe-art model without personalization capability [3] as the teacher model, we can greatly enhance the temporal quality of AnimateDiff. Thus, personalization and controllability is provided by the student model, while temporal consistency is refined by the teacher model. Likewise, LaVie [34] is a multifaceted T2V model that offers various functions including interpolation and super-resolution in a cascaded generation framework, but shows substandard temporal consistency. Using VideoGuide, we can upgrade its temporal consistency with an external model while maintaining its multiple functions.

The synergistic effects that our framework can bring are not limited to these two cases but are, in fact, boundless. As powerful video diffusion models emerge, existing models will not become obsolete but actually improve through the guidance our method provides. Moreover, as VideoGuide can be applied solely during inference time, these benefits can be enjoyed with no cost at all. Our contributions can be summarized as follows:

- 1. We propose VideoGuide, a novel framework for enhancing temporal consistency and motion smoothness while maintaining the imaging quality of the original VDM.
- 2. We show how any existing VDM can be incorporated into our framework, enabling boosted performance of inadequate models along with newfound synergistic effects among models.
- 3. We provide evidence of prior distillation, in which the informative prior of teacher models can be utilized to create samples of improved text coherency.

#### 2. Related Works

The Diffusion Model. Diffusion probabilistic models [17] have achieved great success as generative models. To address the significant computational cost that arises from operating in pixel space, Latent Diffusion Models (LDMs) [29] learn the diffusion process in latent space. LDMs utilize

[Figure 1]

- Figure 2. Overall Pipeline. VideoGuide is a framework for enhancing temporal quality without additional training, leveraging the capabilities of any pretrained VDM. Throughout the denoising process of the sampling VDM, the guiding VDM receives an intermediate latent zt and provides a temporally consistent sample zt−τ by proceeding in its own denoising for a small number of steps τ. The sample

zt−τ is then denoised and interpolated with the denoised zt to produce a fused latent zt′. Such interpolation only needs to take part in the first few steps of inference, and effectively guides samples towards a direction of improved temporal consistency. To further ensure

model flexibility in refining high-frequency areas for better image quality, the latent zt′ is passed through a Low-Pass Filter (LPF). Overall, VideoGuide is a straightforward addition to the original pipeline, yet it is powerful enough to significantly enhance temporal consistency without compromising imaging quality or motion smoothness.

an encoder-decoder framework where the encoder E and the decoder D are trained together to reconstruct the input data. This training aims to satisfy the relation x = D(z0) = D(E(x)), where z0 is the latent representation of the corresponding clean pixel image x. Thus the forward diffusion process in latent space is defined as follows:

###### zt = √α¯tz0 + √1 − α¯tϵ, (1)

where α¯t is a pre-determined noise scheduling coefficient, and ϵ ∼ N(0,I) represents Gaussian noise sampled from a standard normal distribution. The reverse diffusion process is directed by a score-based neural network, denoted as the diffusion model ϵθ, which is trained using the denoising score matching framework [17, 31]. The training objective for this model is formulated as follows:

Et,ϵ∼N(0,I)||ϵ − ϵθ(zt,t)||22. (2)

min

θ

Following the formulation of DDIM [30], the reverse deterministic sampling from the posterior distribution p(zt−1|zt,z0) is given by:

zt−1 = √α¯t−1z0|t + 1 − α¯t−1ϵθ(zt,t) (3)

√1 − α¯tϵθ(zt,t) √α¯t

zt −

(4)

z0|t =

where the denoised sample at timestep t, denoted as z0|t, can be obtained using Tweedie’s formula [9].

Guidance as Optimization Problems. Applying guidance for diffusion models during sampling time assists diffusion models in exploring the latent space with fidelity to the desired manifold, yielding samples that are tailored to specific criteria. Specifically, guidance can be viewed as addressing the optimization problem minz∈M ℓ(z) for a given loss function ℓ(z), where M represents the clean data manifold.

A direct approach can be seen in the context of diffusion model-based inverse problem solvers [5, 6, 8]. In particular, diffusion posterior sampling (DPS) [6] defines the loss function as the manifold-constrained gradient (MCG) [5] of a noisy sample zt ∈ Mt. The loss function ∇mcgzt ℓ(zt) = ∇zt

ℓ(z0|t) is then incorporated into Eq. (3) as follows:

zt−1 = √α¯t−1(z0|t − γt∇zt

ℓ(z0|t))

(5)

+ 1 − α¯t−1ϵθ(zt,t)

where γt denotes step size. To reduce computational overhead, Eq. (5) can be equivalently viewed as follows:

√α¯t−1(z0|t − γt∇z0|t

zt−1 ≈

ℓ(z0|t))

(6)

+ 1 − α¯t−1ϵθ(zt,t)

This approach, known as decomposed diffusion sampling

(DDS) [8], avoids computing the score Jacobian, aligning with methods from [27].

Depending on how the loss function ℓ(z) is defined, it is possible to address various tasks, such as solving inverse problems [6] and generating text-conditioned images [7].

In this work, we are the first to address the video consistency problem from the perspective of an optimization problem, introducing a novel function designed to enhance sample quality in the video domain. This function is integrated into the reverse diffusion process, similar to Eq. (6), to provide a simple yet effective guidance term.

Video Diffusion Model & Consistent Video Generation. The Video Diffusion Model (VDM), originally proposed in [18], operates the diffusion process in the video domain. Similar to LDMs, many recent VDMs [2, 14, 36] are trained in the latent space to reduce computational cost. In Latent VDMs (LVDMs), a temporal layer is incorporated to facilitate frame interaction along the temporal axis during training. By modifying zt to zt1:N in Eqs. (1)-(6), the diffusion model can be extended to the video domain. For simplicity, we will use the notation zt to represent the latent for video generation instead of zt1:N.

One of the main challenges in utilizing diffusion models for video generation lies in maintaining temporal consistency. In the video domain, PYoCo [10] introduces a carefully designed progressive video noise prior to better leverage image diffusion models for video generation. However, PYoCo primarily focuses on the noise distribution during the training stage and requires extensive fine-tuning on video datasets. Recent studies [4, 11] aim to enhance video consistency by leveraging techniques such as attention injection and pretrained text-to-image models, but either encounter image degradation issues [4] or involve a lengthy pipeline to establish initial states [11]. Other approaches introduce improvements in temporal consistency but are primarily focused on long video generation, making them less suitable for the basic 16-frame scenario [20, 28].

FreeInit [35] addresses the issue of video consistency by iterative refinement of the initial noise. This method aims to resolve the training-inference discrepancy in video diffusion models by reinitializing noise with a spatio-temporal filter, ensuring the refined noise better aligns with the training distribution. While this approach enhances frame-to-frame consistency, it has a significant drawback: repeated iteration leads to the loss of fine details and imaging quality degradation. Additionally, the iterative nature of the method induces high computational costs, prolonging the generation process.

#### 3. VideoGuide

In this section, we present VideoGuide, a novel guiding framework which improves the video consistency without compromising significant computational costs in contrast to prior works. Our framework is based on a teacher-guided

latent optimization objective, which when minimized during the early stage of reverse sampling process, progressively improves the temporal consistency of generated video. We begin by outlining the overall optimization framework.

###### 3.1. Video Consistency Guidance

An important contribution of this work is revealing that video consistency can be enhanced by recasting guidance as an optimization problem. We introduce a new objective that regularizes the sampling path of the reverse diffusion process to improve the quality of generated video samples:

ℓ(z0;ψ,ϵ,t) = ||ϵψ(√α¯tz0 + √1 − α¯tϵ,t) − ϵ||22, (7)

where ψ conceptually represents a general teacher model which can denoise the noisy video latents reasonably well. That said, this regularizer represents an ideal condition that high-quality video samples should satisfy: the ideal indistribution video samples should be well reconstructed from random perturbations followed by denoising using teacher video diffusion models.

Then, we can now integrate the optimization step of (7) in reverse sampling process as a guidance in terms of denoised video estimates z0|t. This reads a single DDIM iterate as follows:

zt−1 = √α¯t−1(z0|t − γt∇z0|t

ℓ(z0|t;ψ,ϵ,t))

(8)

+ 1 − α¯t−1ϵθ(zt,t)

Observe that Eq. (7) can be reformulated by swapping ϵψ(zt,t) into an expression of zt and z0ψ|t, and swapping ϵ into an expression of zt and z0 as follows:

√α¯tz0ψ|t √1 − α¯t −

√α¯tz0 √1 − α¯t

2

zt −

zt −

ℓ(z0;ψ,ϵ,t) =

(9)

2

2 2

α¯t 1 − α¯t

z0 − z0ψ|t

=

Plugging Eq. (9) into Eq. (8), the gradient term dissolves into an interpolation scheme and the update rule can be simplified as follows:

zt−1 = √α¯t−1z′ + 1 − α¯t−1ϵθ(zt,t) where z′ = β · z0|t + (1 − β) · z0ψ|t

(10)

1−α¯t . z0ψ|t can be obtained with the renoising process of z0|t as below:

with β = 1 − 2γ

tα¯t

√1 − α¯tϵψ(zt,t) √α¯t

zt −

z0ψ|t =

(11)

where zt = √α¯tz0|t + √1 − α¯tϵ, ϵ ∼ N(0,I)

Specifically, a key challenge in our approach is the lack of a direct model for ψ, which, ideally, would take the form of a consistency model [32]. To address this, we approximate ψ using an iterative reverse sampling method to predict the endpoint of the probability flow ODE (PF-ODE). By performing reverse sampling over multiple steps τ in the original model θ, we generate samples that serve as a proxy for the PF-ODE endpoint. Consequently, we can redefine the terms as follows:

z0ψ|t ≈ z0|t−τ, z′ = β · z0|t + (1 − β) · z0|t−τ (12)

This approach offers a cost-effective solution for approximating ψ in video diffusion models, eliminating the need to train a separate consistency model.

To address the increased sampling time caused by interpolating multiple τ samples at each reverse step, we propose applying a low-frequency filter during the early timesteps of the diffusion process. Recent work [37] indicates that these initial stages primarily establish low-frequency structures, with high-frequency components contributing minimally to image quality. By introducing a low-frequency filter early in the diffusion trajectory, we can accelerate convergence toward consistent samples without sacrificing quality.

Unlike previous methods [35], which apply the filter only to the initial noise, our approach iteratively applies the filter along the entire diffusion path, ensuring stability throughout the trajectory. Additionally, by incorporating early stopping and leveraging our filter’s ability to rapidly achieve sample consistency, we prevent the image degradation typically observed with prolonged optimization [15, 22] of score distillation sampling (SDS) loss [27], which is similar to our objective. Specifically, we define our updates using low-pass (LPFγ) and high-pass (HPFγ) filters with a cutoff frequency γ, streamlining the sampling process while maintaining high quality.

zt−1 = LPFγ(zt−1) + HPF1−γ(ϵ) where ϵ ∼ N(0,I)

(13)

###### 3.2. Guidance with External VDMs

The assumption of ψ in Sec. 3.1 holds for any teacher model that provides a reliable estimate of the denoised sample. This brings us to realize that ψ does not necessarily have to be approximated by the same base model. It is possible to plug in any video diffusion model to approximate ψ and the denoising process would be guided to follow the temporal consistency of the supplemented latent. Here, we demonstrate the steps required for utilizing denoised sam-

ples z0(G|t−) τ of an external guidance model G to enhance the performance of the base sampling model S.

Renoising into the Guidance Domain. Different video diffusion models are trained on varying noise schedules and distributions, and matching such discrepancies is a mandatory

process. When utilizing a guiding model with conflicting factors, the intermediate latent zt of the sampling model must be transformed to align with the noise schedule and distribution of the guiding model. The transformation process can be defined as follows:

zt(G) = α¯t(G)z0(S|t) + 1 − α¯t(G)ϵ where ϵ ∼ N(0,I)

(14)

where (S) denotes the components related to the base sampling model and (G) denotes the components related to the

external guiding model. Specifically, z0(S|t) is the denoised sample from zt(S) at timestep t, and α¯t(G) is derived from the noise schedule of the guiding diffusion model. The resulting outcome zt(G) can then be denoised with the guiding model for a sufficient number of timesteps τ up to z0(G|t−) τ. Thus, the PF-ODE endpoint originally approximated by ψ is now represented as z0(G|t−) τ, the denoised output of the guiding model G.

Interpolation of Denoised Samples. Interpolating the denoised samples of the two models S and G can be expressed as below:

zt(−S)1 = √α¯t−1(β · z0(S|t) + (1 − β) · z0(G|t−) τ)

(15)

+ 1 − α¯t−1ϵ(θS)(zt,t)

Note that the only difference from Eq. (10) is the introduction of the z0(G|t−) τ term, where originally z0(S|t)−τ would be used. LPFγ can then be used on zt(−S)1 as in Eq. (13) for replacing high-frequency components:

zt(−S)1 = LPFγ(zt(−S)1) + HPF1−γ(ϵ) where ϵ ∼ N(0,I)

(16)

In cases where external video diffusion models (VDMs) are used as guidance, the low-frequency filter serves an additional role in preserving domain fidelity. By controlling high-frequency components and introducing Gaussian noise, it prevents unwanted domain drift toward the guidance model while selectively distilling only its temporal stability. This approach captures the temporal consistency of the guiding diffusion model while preserving unique characteristics, such as those in AnimateDiff, without compromising image quality. Notably, these synergistic effects are achieved without additional training or fine-tuning, enabling users to flexibly employ preferred video diffusion models (VDMs) in a plug-and-play manner.

###### 3.3. Prior Distillation

Each video diffusion model spans its own specific data distribution, causing sample generation to be restricted to the

data prior the model has been trained on. Thus, if the data prior of a model is substandard, the generation results of the model are also inherently substandard. This is especially noticeable when using personalized text-to-image (T2I) models such as Dreambooth or LoRA in AnimateDiff, in which substandard results that do not align with the given text prompt are frequently observed. Prior work [10] elaborates on the importance of data prior for VDMs, but the proposed solution involves extensive fine-tuning, making it impractical for simple use cases. On the other hand, VideoGuide comes as a potential solution in such cases, where the interpolation between two models exhibit a form of prior distillation. Through the guidance of a generalized video diffusion model (e.g. [3]) the base sampling model is able to refer to the denoised sample provided by the guidance model, and steer its sampling process towards a relevant outcome. This allows for the effective generation of diverse objects, even while retaining the style of the original data domain. For the case of AnimateDiff, the approach allows for broader customization without the need for directly training the personalized T2I model on a wider range of data. Extensive analysis concerning this issue is provided in Sec. 5.2.

#### 4. Experiments

Experimental Settings. In our experiments, we leverage multiple open-source Text-to-Video (T2V) diffusion models to explore the combined strengths of each. For the guiding diffusion model, we choose Videocrafter2 [3] due to its strong performance in temporal consistency, as measured by the VBench [19] benchmark. For sampling, we employ AnimateDiff [13] for flexible personalization of video content, and Lavie [34] to enhance video quality and increase frame count through super-resolution and interpolation techniques. This integration combines the temporal consistency of the guiding model with the advantages of the sampling model. All experiments were conducted using DDIM with 50 steps for sampling. For our experiments with AnimateDiff, we set the number of interpolation steps I = 5, β = 0.5, and τ = 10, and used the Butterworth filter with a normalized frequency of 0.25 and a filter order of n = 4. Additional experimental details are provided in Supplementary Material. Evaluation Metrics. To validate the improvement in video consistency with our proposed method, we evaluate five key metrics: subject consistency, background consistency, imaging quality, motion smoothness, and dynamic degree. For subject consistency evaluation, DINO [1] feature similarity between frames is measured to assess consistency of the subject’s appearance throughout the video. Background consistency is evaluated using CLIP feature similarity between frames to evaluate overall scene consistency. Imaging quality is also a key metric in that maintaining original image quality is essential for generation and enabling customization. Thus we evaluate image quality using the multi-scale image qual-

ity transformer (MUSIQ) [21], which measures frame-wise low-level distortion such as noise, blur, and over-exposure. To ensure smooth motion, we employ a video interpolation model [24] to assess consistency of motion across video frames. To compare the magnitude of the motion in the videos, we utilize RAFT [33] to estimate the optical flow, and calculate the mean of top 5% largest magnitude of the flows.

###### 4.1. Comparison Results

Qualitative results for various prompts and base models are shown in Fig. 3. Samples from the base model show impairment in temporal consistency, such as fluctuation in color or abrupt change in subject appearance. FreeInit [35] moderately solves the problem of temporal consistency but at the cost of considerable degradation in imaging quality, such as smoothing out of textural details. In contrast, the proposed VideoGuide significantly enhances temporal consistency without loss of imaging quality. Furthermore, VideoGuide solves sudden frame shifts frequently observed in LaVie by providing smooth frame transitions. Additional qualitative results are included in Supplementary Material.

Quantitative comparison shows that our approach consistently outperforms the baseline model, achieving substantial improvements in both subject and background consistency. When using AnimateDiff [13] as the baseline, our method delivers the best results across all key metrics, except for dynamic degree, due to an inherent trade-off. Our method greatly enhances temporal consistency with a comparatively small reduction in dynamic degree, ensuring stable temporal coherence without compromising image quality. For LaVie [34], our method achieves higher temporal consistency and an increased dynamic degree compared to previous methods, highlighting its ability to enhance temporal stability with minimal impact on dynamic motion.

Incorporating an external model [3] for guidance enhances performance by achieving higher temporal consistency without compromising dynamic motion, compared to self-guided cases in both AnimateDiff and LaVie. This result indicates that using a higher-performing external model for guidance can lead to superior video quality. Additionally, our method successfully addresses challenges such as sudden visual shifts, as discussed in Supplementary Material.

Regarding computational efficiency, iterative initial noise refinement in prior work [35] requires performing DDIM sampling over multiple iterations, resulting in high computational cost. In contrast, our method only introduces a small number of additional sampling steps. This difference leads to a significant reduction in inference time, yielding a ×1.8 ∼ ×2.4 improvement in generation speed for AnimateDiff and a ×2.1 ∼ ×3.0 improvement for Lavie as shown in Tab. 2.

Our approach achieves these advancements without sac-

Subject consistency (↑)

Background Consistency (↑)

Imaging Quality (↑)

Motion Smoothness (↑)

Dynamic Degree (↑)

Method

AnimateDiff [13] 0.9183 0.9437 0.6647 0.9547 26.67 AnimateDiff + FreeInit [35] 0.9487 0.9604 0.6173 0.9705 19.28 AnimateDiff + Ours (with AnimateDiff) 0.9520 0.9600 0.6566 0.9731 15.25 AnimateDiff + Ours (with VideoCrafter2) 0.9614 0.9664 0.6671 0.9772 16.78

LaVie [34] 0.9534 0.9599 0.6750 0.9658 14.69 LaVie + FreeInit [35] 0.9625 0.9643 0.6533 0.9757 10.69 LaVie + Ours (with Lavie) 0.9629 0.9652 0.6780 0.9725 12.39 LaVie + Ours (with VideoCrafter2) 0.9635 0.9643 0.6796 0.9723 12.63

Table 1. Quantitative comparison of video generation. Bold: best, underline: second best.

[Figure 2]

- Figure 3. Qualitative Comparison. VideoGuide is applied on various base models for different text prompts. For each prompt, frames of generated samples from four different models are displayed: (i) Base model (first row); (ii) Base model with FreeInit (second row); (iii) Base model with VideoGuide (self-guided case) (third row); (iv) Base model with VideoGuide (external model-guided case) (fourth row). AD, VC, LV indicate guidance models of AnimateDiff, VideoCrafter-2.0, LaVie, respectively. Samples for the base model show substandard temporal consistency, especially regarding color fluctuation and subject appearance change. Applying FreeInit improves consistency but introduces degradation in imaging quality, such as smoothing out of textural details. In contrast, applying VideoGuide significantly enhances temporal consistency while preserving imaging quality, both for the self-guided and the external model-guided case.

Method AnimateDiff LaVie Baseline 11.38 6.86 FreeInit 51.98 30.18

Ours (self-guided) 21.68 10.01 Ours (VC-guided) 29.73 14.07

- Table 2. Inference time for video generation(s). Both the selfguided case and the VideoCrafter2-guided case show significant decrease in inference time compared with previous method [35]. Bold: best, underline: second best.

Interpolation Scale β SC BC

β = 0.9 0.9518 0.9599 0.8 0.9546 0.9609 0.7 0.9576 0.9628 0.6 0.9605 0.9649 0.5 0.9614 0.9664

Interpolation Step Number I SC BC

I = 1 0.9524 0.9618

- 2 0.9489 0.9588
- 3 0.9546 0.9612
- 4 0.9602 0.9645

- 5 0.9614 0.9664

Guidance Step Number τ SC BC

τ = 1 0.9444 0.9558 3 0.9531 0.9611 5 0.9582 0.9641 7 0.9611 0.9658 10 0.9614 0.9664

- Table 3. Ablation study regarding interpolation scale β, number of interpolation steps I, and number of guidance sampling steps τ. Subject consistency (SC) and background consistency (BC) is compared for various parameters. Bold: best, underline: second best.

rificing image quality, highlighting its importance for both personalization-focused applications and high-quality video generation. Overall, VideoGuide proves to be essential for optimizing temporal coherence and enhancing overall quality across various model frameworks.

#### 5. Analysis 5.1. Ablation Study

Parameter Selection. An analysis is performed to assess how varying parameters of the guiding diffusion model impacts temporal consistency. Specifically, we examine the effects of three factors: interpolation scale β, number of interpolation steps I, number of guidance sampling steps τ.

Our ablation studies prove that all three parameters are closely related to temporal consistency. Decrease in interpolation scale β leads to improved subject and background consistency. Note that the minimum value of β is constrained to 0.5 to mitigate the risk of distribution shift. Increasing the number of interpolation steps I also leads to improvement in temporal consistency, which proves that our interpolation scheme is indeed effective. Furthermore, increasing the number of guidance sampling steps τ enhances consistency, indicating that blending intermediate latents with better-denoised versions enhances overall consistency as expected. Such ablation study highlights the trade-off between consistency improvement and computational efficiency, offering insight into optimal parameter settings for the guiding diffusion model.

[Figure 3]

Figure 4. Prior Distillation Results. VideoGuide solves degraded performance regarding text coherency by enabling the utilization of a superior data prior. Example results for certain ambiguous prompts are displayed. For each prompt, the same random seed is shared for both methods. AnimateDiff directs generation of ‘beetle’ and ‘jaguar’ towards car samples due to a substandard data prior. Using VideoGuide, users can distill a superior prior for correct generation.

###### 5.2. Prior Distillation

Degraded performance due to a substandard data prior is an issue only solvable through extra training. However VideoGuide provides a workaround to this matter by enabling the utilization of a superior data prior. Fig. 4 demonstrates example cases. For all instances, generated samples are guided towards a result of better text coherence while maintaining the style of the original data domain. Additional examples of prior distillation are provided in Supplemental Material.

#### 6. Conclusion

In this work, we introduced VideoGuide, a novel and versatile framework that enhances the temporal quality of pretrained text-to-video (T2V) diffusion models without the need for additional training or fine-tuning. Our approach provides temporally consistent samples to intermediate latents during the early stages of the denoising process, guiding the low frequency components of latents towards a direction of high temporal consistency. The samples provided are not confined to the base model; any superior pretrained VDM can be selected for distillation. By doing so, we empower underperforming models with improved motion smoothness and temporal consistency while maintaining their unique traits and strengths, including personalization and controllability. We demonstrate the effectiveness of VideoGuide on various base models, and prove its ability to enhance temporal consistency without sacrifice of imaging quality or motion smoothness compared to prior methods. The potential of VideoGuide extends far beyond the cases discussed, as VideoGuide ensures that even existing models can remain relevant and competitive by leveraging the strengths of superior models. As video diffusion models continue to evolve, new and emerging VDMs will only enhance the pertinence of VideoGuide over time, broadening the scope of VDMs utilizable as a video guide.

#### References

- [1] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the International Conference on Computer Vision (ICCV), 2021. 6
- [2] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter1: Open diffusion models for high-quality video generation, 2023. 4
- [3] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models, 2024. 2, 6
- [4] Xuweiyi Chen, Tian Xia, and Sihan Xu. Unictrl: Improving the spatiotemporal consistency of text-to-video diffusion models via training-free unified attention control, 2024. 2, 4, 5
- [5] Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye. Improving diffusion models for inverse problems using manifold constraints. In Advances in Neural Information Processing Systems, 2022. 3
- [6] Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In International Conference on Learning Representations, 2023. 3, 4
- [7] Hyungjin Chung, Jeongsol Kim, Geon Yeong Park, Hyelin Nam, and Jong Chul Ye. Cfg++: Manifold-constrained classifier free guidance for diffusion models. arXiv preprint arXiv:2406.08070, 2024. 4, 1
- [8] Hyungjin Chung, Suhyeon Lee, and Jong Chul Ye. Decomposed diffusion sampler for accelerating large-scale inverse problems. In The Twelfth International Conference on Learning Representations, 2024. 3, 4
- [9] Bradley Efron. Tweedie’s formula and selection bias. Journal of the American Statistical Association, 106(496):1602–1614,

2011. 3

- [10] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, MingYu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models, 2024. 4, 6
- [11] Xiefan Guo, Jinlin Liu, Miaomiao Cui, Liefeng Bo, and Di Huang. I4vgen: Image as free stepping stone for text-to-video generation. arXiv preprint arXiv:2406.02230, 2024. 4
- [12] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. arXiv preprint arXiv:2311.16933, 2023. 2
- [13] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. International Conference on Learning Representations, 2024. 2, 6, 7, 5
- [14] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv:2211.13221, 2022. 4

- [15] Amir Hertz, Kfir Aberman, and Daniel Cohen-Or. Delta denoising score, 2023. 5
- [16] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 1
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 2, 3
- [18] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models, 2022. 4
- [19] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 6, 2
- [20] Gu Jiaxi, Wang Shicong, Zhao Haoyu, Lu Tianyi, Zhang Xing, Wu Zuxuan, Xu Songcen, Zhang Wei, Jiang Yu-Gang, and Xu Hang. Reuse and diffuse: Iterative denoising for text-to-video generation. arXiv preprint arXiv:2309.03549,

2023. 4

- [21] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148–5157, 2021. 6
- [22] Subin Kim, Kyungmin Lee, June Suk Choi, Jongheon Jeong, Kihyuk Sohn, and Jinwoo Shin. Collaborative score distillation for consistent visual editing. In Advances in Neural Information Processing Systems, 2023. 5
- [23] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 4
- [24] Zhen Li, Zuo-Liang Zhu, Ling-Hao Han, Qibin Hou, ChunLe Guo, and Ming-Ming Cheng. Amt: All-pairs multi-field transforms for efficient frame interpolation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR),

2023. 6

- [25] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 4, 5
- [26] William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022. 4
- [27] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv, 2022. 4, 5
- [28] Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. Freenoise: Tuning-free longer video diffusion via noise rescheduling, 2023. 4
- [29] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models, 2021. 2
- [30] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In 9th International Conference on Learning Representations, ICLR, 2021. 3, 5
- [31] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based

- generative modeling through stochastic differential equations. In 9th International Conference on Learning Representations, ICLR, 2021. 3
- [32] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023. 5
- [33] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow, 2020. 6
- [34] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023. 2, 6, 7
- [35] Tianxing Wu, Chenyang Si, Yuming Jiang, Ziqi Huang, and Ziwei Liu. Freeinit: Bridging initialization gap in video diffusion models. arXiv preprint arXiv:2312.07537, 2023. 2, 4, 5, 6, 7, 8
- [36] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv:2310.12190, 2023. 4
- [37] Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, and Chao Dong. Scaling up to excellence: Practicing model scaling for photorealistic image restoration in the wild, 2024. 5
- [38] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. 2
- [39] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024. 4, 5

## VideoGuide: Improving Video Diffusion Models without Training Through a Teacher’s Guide

### Supplementary Material

The supplementary material is organized as follows:

- • Section A: Importance of the low-pass filter.
- • Section B: About classifier-free guidance.
- • Section C: Pseudocodes for our algorithm.
- • Section D: More experimental details.
- • Section E: More quantitative results: user study.
- • Section F: Application on DiT-based models.
- • Section G: Comparison with orthogonal methods.
- • Section H: Limitations.
- • Section I: More qualitative examples.

#### A. Importance of Low-Pass Filter

To evaluate the role of the low-pass filter in our methodology, we conduct experiments by varying the interpolation step I, both with and without the low-pass filter. These experiments are averaged across 800 prompts from the VBench categories for consistent evaluation. We apply the low-pass filter for the initial 5 timesteps, based on the observation that the mid-to-late timesteps in the diffusion process focus on generating mid- and high-frequency details. Replacing these frequencies with random components via the low-pass filter in the mid-to-late timesteps would result in degraded video quality, necessitating the early-timestep limitation. All corresponding results are presented in Fig. 5.

We measure the effects of the filter on Subject Consistency, Background Consistency, and Imaging Quality. Both Subject Consistency and Background Consistency steadily improves as the number of interpolation steps increases, demonstrating the effectiveness of our interpolation scheme in enhancing temporal coherence. Meanwhile, Imaging Quality is maintained up to approximately 10 interpolation steps without the low-pass filter. Beyond this point, a significant drop in quality is observed, indicating that excessive interpolation exacerbates the blurring effects caused by prolonged SDS optimization, as noted earlier in this work.

The improvement in consistency is significantly accelerated when using the low-pass filter. This acceleration is achieved while mitigating the decline in imaging quality typically associated with increased interpolation steps. Furthermore, application of the filter also reduces computational overhead during interpolation. Specifically, the consistency achieved at I = 4 with the filter is comparable to the consistency achieved at I = 50 without the filter, offering approximately a 7-fold reduction in inference time. Such results prove the effectiveness of the low-pass filter in balancing consistency improvement, imaging quality preservation, and computational efficiency.

#### B. Classifier-Free Guidance

Off-Manifold Behavior of CFG Recent study [7] demonstrates that employing a high Classifier- Free Guidance (CFG) [16] scale (w > 1.0) in the early timesteps of diffusion sampling leads to off-manifold behavior. This phenomenon results in denoised samples exhibiting problems such as color saturation and abrupt transitions, which negatively affect the interpolation between samples during these timesteps. We solve this by applying a lower guidance scale w during the early stages of sampling, ensuring smoother interpolation between the denoised samples. As illustrated in Fig. 6 (a), when using a high CFG scale (w = 7.5), the influence of the guiding diffusion model becomes minimal due to significant color saturation, making it difficult for the output of the guiding model to be reflected effectively. In contrast, as illustrated in Fig. 6 (b), a lower CFG scale (w = 0.8) facilitates smoother interpolation between the sampling diffusion model and the guiding diffusion model.

Configuration SC (↑) BC (↑) Base

+ CFG 0.9183 0.9437 + CFG++ 0.9176 0.9435

FreeInit

+ CFG 0.9487 0.9604 + CFG++ 0.9473 0.9604

Ours

+ CFG Interp. 0.9598 0.9635 + CFG++ Interp. 0.9614 0.9664

Table 4. Comparison of consistency metrics (SC: Subject Consistency, BC: Background Consistency) across different configurations using CFG and CFG++ in AnimateDiff. Our approach with interpolated CFG++ achieves the best performance, significantly enhancing both subject and background consistency.

We provide quantitative analysis for using CFG and CFG++ across the Base Model, Base Model + FreeInit, and Base Model + VideoGuide (Ours) during the interpolation. As shown in Tab. 4, metrics for Base and FreeInit decrease when CFG++ is used, and metrics improve only when CFG++ is applied to our interpolation scheme. This implies the significant positive impact on consistency of CFG++ within the proposed interpolation scheme, especially compared to CFG. Also, this supports the idea that smooth interpolation of denoised samples positively impacts model performance, as discussed above.

[Figure 4]

- Figure 5. Comparison of Subject Consistency, Background Consistency, and Imaging Quality across interpolation steps (I) with and without the application of the low-frequency filter. Results indicate that the low-frequency filter accelerates convergence towards consistency while maintaining imaging quality.

[Figure 5]

- Figure 6. (a) The interpolation process between denoised samples from the sampling model (S) and the guiding model (G) for high guidance scale w = 7.5 is shown. (b) The interpolation process for low guidance scale w = 0.8 is shown. Both interpolations are performed at T = 980 and β = 0.7. Results indicate that with high guidance scale w, influence of the guiding diffusion model is significantly reduced due to color saturation.

guidance scale of w = 0.8 in DDIM 50-step sampling. After completing the interpolation step, we revert to CFG reverse sampling with a CFG scale of 7.5. In FreeInit, we use a Butterworth filter with a normalized frequency of 0.25, filter order n = 4, and perform 5 iterations, as recommended in prior work. The same filter is applied in our experiments with FreeInit. For AnimateDiff, we configure the guiding model with parameters I = 5, β = 0.5, and τ = 10. In the case of LaVie, we set I = 3, β = 0.5, and τ = 10 to optimize inference speed. Additionally, the τ intervals are not uniformly spaced as in the standard 50-step DDIM sampling. To better leverage temporally consistent samples, we divide the remaining interval into 25 steps for reverse sampling during guidance steps.

- D.3. Figure Explanation Base models used for Figure 3:

- (a) AnimateDiff with pretrained T2I model RealisticVision.
- (b) AnimateDiff with pretrained T2I model RealisticVision.
- (c) LaVie.
- (d) LaVie. Base model used for Figure 4: AnimateDiff with pretrained T2I model ToonYou.

- E. User Study

#### C. Pseudo Code

Pseudo codes regarding our algorithm are provided. For clarity, the pseudo code describing our algorithm adopts the CFG++ reverse sampling framework for the entire process.

#### D. Experimental Details

###### D.1. Prompt Selection

We conduct a user study to evaluate generated video samples using three criteria: Text Alignment, Overall Quality, and Smooth and Dynamic Motion, with all metrics scored on a 1 to 5 scale. A total of 30 participants provided ratings for each metric, offering comprehensive feedback on the generated videos. Tab. 5 shows that our method surpasses the baseline and previous work in all evaluated aspects.

In all experiments, we utilize 800 prompts from various categories in VBench [19] to evaluate the model’s ability to generate across diverse categories.

###### D.2. Hyperparameter Selection

We employ a classifier-free guidance (CFG) scale of 7.5 during inference for both base models (AnimateDiff, LaVie) and FreeInit-applied cases. During interpolation of the denoised samples, we apply CFG++ reverse sampling with a

###### Text Alignment

• Measures how well the video corresponds to the prompt,

- Algorithm 1 VideoGuide with Sampling Diffusion Model Require: guidance scale λ ∈ [0,1], guiding steps I, interpolation scale β, extra step τ

- 1: Initialize zT ∼ N(0,I)
- 2: for t = T,...,1 do
- 3: ϵˆθ(zt,t) = ϵθ(zt,t,ϕ) + λ[ϵθ(zt,t,c) − ϵθ(zt,t,ϕ)]
- 4: z0|t = (zt −

√1 − α¯tϵˆθ(zt,t))/√α¯t

- 5: zt = √α¯tz0|t + √1 − α¯tϵ, where ϵ ∼ N(0,I)

- 6: if T − t < I then
- 7: for j = 0,...,τ do
- 8: zt−j−1 = √α¯t−j−1z0|t−j + 1 − α¯t−j−1ϵθ(zt−j,t − j,ϕ)

- 9: end for
- 10: z0′|t = β · z0|t + (1 − β) · z0|t−τ
- 11: zt−1 = √α¯t−1z0′|t + √1 − α¯t−1ϵθ(zt,t,ϕ)

- 12: zt−1 = LPFγ(zt−1) + HPFγ(ϵ), where ϵ ∼ N(0,I)
- 13: else
- 14: zt−1 = √α¯t−1z0|t + √1 − α¯t−1ϵθ(zt,t,ϕ)

- 15: end if
- 16: end for
- 17: Output: Final video z0

- Algorithm 2 VideoGuide with Guiding Diffusion Model

Require: guidance scale λ ∈ [0,1], guiding steps I, interpolation scale β, extra step τ, Guiding Model G parameterized by ψ,

noise schedule α¯(G) of G

- 1: Initialize zT ∼ N(0,I)
- 2: for t = T,...,1 do
- 3: ϵˆθ(zt,t) = ϵθ(zt,t,ϕ) + λ[ϵθ(zt,t,c) − ϵθ(zt,t,ϕ)]
- 4: z0|t = (zt −

√1 − α¯tϵˆθ(zt,t))/√α¯t

- 5: zt(G) = α¯t(G)z0|t + 1 − α¯t(G)ϵ, where ϵ ∼ N(0,I)

- 6: if T − t < I then
- 7: for j = 0,...,τ do
- 8: z0(G|t−) j = (zt(−Gj) − 1 − α¯t(−Gj)ϵˆψ(zt(−Gj),t − j)/ α¯t(−Gj)

- 9: zt(−Gj)−1 = α¯t(−Gj)−1z0(G|t−) j + 1 − α¯t(−Gj)−1ϵψ(zt(−Gj),t − j,ϕ)

- 10: end for
- 11: z0′|t = β · z0|t + (1 − β) · z0(G|t−) τ
- 12: zt−1 = √α¯t−1z0′|t + √1 − α¯t−1ϵθ(zt,t,ϕ)

- 13: zt−1 = LPFγ(zt−1) + HPFγ(ϵ), where ϵ ∼ N(0,I)
- 14: else
- 15: zt−1 = √α¯t−1z0|t + √1 − α¯t−1ϵθ(zt,t,ϕ)

- 16: end if
- 17: end for
- 18: Output: Final video z0

focusing on semantic coherence.

• Question: Do you think the videos reflect the given text condition well? (5: Strongly Agree / 4: Agree / 3: Neutral / 2: Disagree / 1: Strongly Disagree)

###### Overall Quality

• Assesses the video’s visual consistency, image degradation, and aesthetic appeal.

• Question: Do you think the video’s overall quality is good? (rich detail, unchanging objects) (5: Strongly Agree / 4: Agree / 3: Neutral / 2: Disagree / 1: Strongly Disagree)

###### Smooth and Dynamic Motion

- • Evaluates the naturalness and fluidity of the motion in the video.
- • Question: Do you think the video’s overall motion is

smooth and dynamic? (5: Strongly Agree / 4: Agree / 3: Neutral / 2: Disagree / 1: Strongly Disagree)

Method TA OQ SDM Base 3.72 2.84 2.9 Base + FreeInit 3.97 3.35 3.38 Base + VideoGuide (Ours) 4.36 4.37 4.36

Table 5. User Study. Text Alignment (TA), Overall Quality (OQ), Smooth and Dynamic Motion (SDM) are evaluated among methods. Bold: best, underline: second best.

#### F. Application on DiT-based Models

We further evaluate the robustness of our methodology by applying it to different architectures and schedulers. Specifically, we present further evaluation on models that use Diffusion Transformer (DiT) [26] architecture: Open-Sora v1.0 [39] and Open-Sora v1.2 [39]. Each model employs a standard DDIM scheduler (50 steps) and a rectified flow [25] scheduler, respectively. In the rectified flow-based configuration, the objective for training can be formulated as follows:

zt = (1 − t)z0 + tϵ where t ∈ [0,1]

θˆ = argmin

E ||(z0 − ϵ) − vθ(zt,t)||22

θ

(17)

Using the objective above we can redefine our method as below:

+ ti · vθ(zt

= zt

z0|t

,ti) ϵθ(zt

i

i

i

i − (1 − ti) · vθ(zt

,ti) = zt

,ti) zt

i

i

= (1 − ti−1)f(z0|t

,β,τ) + ti−1ϵθ(zt

,ti)

i−1

i

i

(18) where f(z0|t

,β,τ) is the interpolation function between z0|t

i

with scale β. The results in Tab. 6 demonstrate that our method improves temporal consistency for both baselines while preserving imaging quality and introducing only a minimal decrease in dynamic degree. These findings indicate that our methodology enhances performance regardless of the underlying architecture and scheduler.

and z0|t

i−τ

i

#### G. Comparison with Orthogonal Methods

A recent study, UniCtrl [4], attempts to improve semantic consistency and motion quality in an approach orthogonal to ours. In this section, we compare the performance of each technique and assess the feasibility of combining them. Following the recommendation in the paper, we use a motion injection degree of c = 0.2, while maintaining the same experimental configuration as described in Section D. As illustrated in Table 7, UniCtrl [4] improves temporal consistency but at the cost of a significant reduction in dynamic degree and imaging quality.

#### H. Limitations

While our approach significantly improves the performance of baseline models, it relies on sharing the same Variational Auto-Encoder(VAE) [23] space. In practice, many latent diffusion models utilize the same VAE, making this requirement generally feasible. However, if the VAE spaces differ, one potential solution is to decode, interpolate, and re-encode the features. This process, however, incurs additional computational overhead and risks losing fine details due to iterative encoding-decoding. Developing an effective method to address compatibility across different VAE spaces remains an avenue for future research.

##### I. More Qualitative Examples Additional samples are provided in following pages:

- • Supplemental examples of prior distillation.
- • Qualitative comparison for various methods.
- • Qualitative comparison for various base models.
- • Usage of VideoGuide to solve sudden frame shifts in LaVie samples.

- I.1. Prior Distillation
- I.2. More Qualitative Comparison Results
- I.3. More Qualitative Results
- I.4. LaVie Sudden Shift

Subject consistency (↑)

Background Consistency (↑)

Imaging Quality (↑)

Motion Smoothness (↑)

Dynamic Degree (↑)

Method

OpenSora v1.0 [39] (DDIM [30]) 0.9735 0.9689 0.6615 0.9678 4.97 OpenSora v1.0 + VideoGuide (self-guided) 0.9763 0.9689 0.6738 0.9754 3.88 OpenSora v1.2 [39] (Rectified Flow [25]) 0.9725 0.9696 0.6582 0.9881 12.68 OpenSora v1.2 + VideoGuide (self-guided) 0.9808 0.9748 0.6689 0.9903 11.07

Table 6. Quantitative comparison of video generation in DiT-based architecture.

Subject consistency (↑)

Background Consistency (↑)

Imaging Quality (↑)

Motion Smoothness (↑)

Dynamic Degree (↑)

Method

AnimateDiff [13] 0.9183 0.9437 0.6647 0.9547 26.67 AnimateDiff + UniCtrl [4] 0.9259 0.9413 0.6032 0.9584 14.96 AnimateDiff + Ours 0.9614 0.9664 0.6671 0.9772 16.78 AnimateDiff + UniCtrl + Ours 0.9639 0.9628 0.5883 0.9776 5.02

Table 7. Quantitative comparison with orthogonal methods.

[Figure 6]

- Figure 7. Qualitative Results of VideoGuide on Open-Sora v1.0.

[Figure 7]

- Figure 8. Qualitative Results of VideoGuide on Open-Sora v1.2.

[Figure 8]

###### Figure 9. Qualitative Comparison of UniCtrl and VideoGuide.

[Figure 9]

###### Figure 10. Prior Distillation. For each prompt, we share the same random seed for both methods.

[Figure 10]

###### Figure 11. More Qualitative Comparison Results of VideoGuide. Top: AnimateDiff with ToonYou, Bottom: AnimateDiff with RCNZCartoon

[Figure 11]

[Figure 12]

[Figure 13]

###### Figure 14. More Qualitative Results of VideoGuide on AnimateDiff (with ToonYou).

[Figure 14]

###### Figure 15. More Qualitative Results of VideoGuide on AnimateDiff (with RCNZCartoon).

[Figure 15]

###### Figure 16. More Qualitative Results of VideoGuide on AnimateDiff (with FilmVelvia).

[Figure 16]

###### Figure 17. More Qualitative Results of VideoGuide on LaVie.

[Figure 17]

###### Figure 18. More Qualitative Results of VideoGuide on LaVie.

[Figure 18]

Figure 19. VideoGuide helps solve the issue of sudden frame shifts in LaVie samples. By integrating an external guiding model, VideoGuide provides smoother frame transitions to the base model. LV indicates that guidance model of LaVie is used (the self-guided case), and VC indicates that guidance model of VideoCrafter2 is used. Guidance given with the external model VideoCrafter2 solves sudden frame shift unsolvable by other methods.

