## Enhance-A-Video: Better Generated Video for Free

# arXiv:2502.07508v3[cs.CV]27Feb2025

Yang Luo1 Xuanlei Zhao1 Mengzhao Chen2 Kaipeng Zhang2 Wenqi Shao2† Kai Wang1† Zhangyang Wang3 Yang You1

1National University of Singapore 2Shanghai Artificial Intelligence Laboratory 3University of Texas at Austin

[Figure 1]

Figure 1. Enhance-A-Video boosts diffusion transformers-based video generation quality at minimal cost - no training needed, no extra learnable parameters, no memory overhead. Detailed captions are available in Appendix G.

### Abstract

DiT-based video generation has achieved remarkable results, but research into enhancing existing models remains relatively unexplored. In this work, we introduce a training-free approach to enhance the coherence and quality of DiT-based generated videos, named Enhance-A-Video. The

†corresponding author Code: NUS-HPC-AI-Lab/Enhance-A-Video

core idea is enhancing the cross-frame correlations based on non-diagonal temporal attention distributions. Thanks to its simple design, our approach can be easily applied to most DiT-based video generation frameworks without any retraining or fine-tuning. Across various DiT-based video generation models, our approach demonstrates promising improvements in both temporal consistency and visual quality. We hope this re-

search can inspire future explorations in video generation enhancement.

### 1. Introduction

Diffusion transformer (DiT) models (Peebles & Xie, 2022) have revolutionized video generation, enabling the creation of realistic and compelling videos (Yang et al., 2024; Brooks et al., 2024; Lin et al., 2024; Xu et al., 2024a; Kong et al., 2025). However, achieving temporal consistency across frames while maintaining fine-grained details remains a significant challenge. Many existing methods generate videos that suffer from unnatural transitions and degraded quality as illustrated in Figure 2, which fundamentally limits their practical applicability in real-world scenarios and professional applications (Yan et al., 2023; Henschel et al., 2024).

[Figure 2]

- Figure 2. Video sample of HunyuanVideo model with unnatural head movements, repeated right hands and conflicting glove color.

Video generation enhancement (He et al., 2024) is designed for addressing the above limitations, where two objectives are primarily considered: (i) maintaining temporal consistency across frames, which ensures smooth and coherent transitions, and (ii) improving spatial details, which enhances the visual quality of each frame. In UNet-based video generation (Zhang et al., 2023a; Guo et al., 2024; Xu et al., 2024b; Li et al., 2025), Upscale-A-Video (Zhou et al., 2024) integrated a local-global temporal strategy for better temporal coherence, and VEnhancer (He et al., 2024) designed a video ControlNet (Zhang et al., 2023b) to enhance spatial and temporal resolution simultaneously. Nevertheless, the exploration of enhancing DiT-based video generation remains limited, particularly in addressing challenges of temporal consistency and spatial detail preservation.

In DiT-based video generation, temporal attention (Tan et al.,

- 2023) plays a crucial role in ensuring coherence among frames, further preserving fine-grained details. Through careful analysis of temporal attention in DiT blocks, we made an important observation as shown in Figure 3: crossframe temporal attentions (non-diagonal elements) are sig-

nificantly lower than intra-frame attentions (diagonal elements) in some blocks. This unbalanced distribution of cross-frame and intra-frame attention may lead to inconsistencies among frames, such as abrupt transitions and blurred details in generated videos.

Is there an efficient method to utilize the cross-frame information to improve consistency across frames? The intensity of cross-frame information is directly related to the mean of non-diagonal temporal attention weights. By leveraging the calculated cross-frame intensity, it becomes possible to promote video quality by adjusting imbalanced cross-frame dependencies while maintaining frame-level detail.

Building on these insights, we propose a novel, training-free, and plug-and-play approach, Enhance-A-Video, to improve the temporal and spatial quality of DiT-based generated videos. The method introduces two key innovations: a crossframe intensity to capture cross-frame information within the temporal attention mechanism and an enhance temperature parameter to scale calculated cross-frame intensity. By strengthening cross-frame correlations from the temperature perspective, our approach enhances temporal consistency and preserves fine visual details effectively. A notable advantage is that this method can be readily integrated into prevalent DiT-based video generation frameworks with negligible computational overhead.

We conduct a comprehensive experimental evaluation of our approach across several benchmark DiT-based video generation models including HunyuanVideo (Kong et al., 2025), CogVideoX (Yang et al., 2024), LTX-Video (HaCohen et al., 2024) and Open-Sora (Zheng et al., 2024). By incorporating Enhance-A-Video during the inference phase, these models demonstrate a significant improvement in generated video quality by reducing temporal inconsistencies and refining visual fidelity with minimal extra cost.

### 2. Related Work

Video Generation. Recent advancements in video generation have been driven by powerful diffusion transformerbased models (Chen et al., 2024; Ma et al., 2024; Gao et al., 2024; Lu et al., 2024). Sora (Brooks et al., 2024) has demonstrated exceptional capabilities in generating realistic and long-duration videos, establishing itself as a significant milestone in text-to-video generation. CogVideoX (Yang et al., 2024) introduced a 3D full attention mechanism and expert transformers to improve motion consistency and semantic alignment. HunyuanVideo (Kong et al., 2025) introduces a hybrid stream block with enhanced semantic understanding. However, several important challenges such as temporal inconsistency and the loss of fine-grained spatial details in video generation still persist.

Temperature Parameter. The temperature parameter is

[Figure 3]

- Figure 3. Visualization of temporal attention distributions in Open-Sora for blocks 2, 14, and 26 at denoising step 30, where non-diagonal elements are considerably weaker than diagonal elements.

a well-known concept in deep learning, primarily used to control the distribution of attention or output probabilities in generative models (Peeperkorn et al., 2024; Renze & Guven,

- 2024). In natural language generation tasks, the temperature is often adjusted during inference to modulate the diversity of the generated text (Holtzman et al., 2020). A higher temperature increases randomness, promoting creativity, while a lower temperature encourages deterministic and coherent outputs. Recently, the concept has been explored in vision-related tasks, such as visual question answering and multimodal learning (Chen et al., 2021), where temperature adjustments are applied to balance multimodal attention distributions. However, its application in DiT-based video generation, particularly in enhancing temporal attention, remains underexplored.

### 3. Methodology

#### 3.1. Diffusion Transformer Models

Diffusion Transformer models are inspired by the success of diffusion models in generating high-quality images and videos by iteratively refining noisy data (Ho et al., 2022; Blattmann et al., 2023; Esser et al., 2024). These models combine the strengths of diffusion processes and transformer architectures to model temporal and spatial dependencies in video generation. The forward diffusion process adds noise to the data over T timesteps, gradually converting it into a noise distribution. Starting from clean data x0, the noisy data at timestep t is obtained as:

xt = √αtxt−1 + √1 − αtzt, for t = 1,...,T, (1)

where αt controls the noise schedule and zt ∼ N(0,I) is Gaussian noise. As t increases, xt approaches a standard normal distribution N(0,I). To recover the original data distribution, the reverse diffusion process progressively removes noise from xt until reaching x0:

pθ(xt−1|xt) = N(xt−1;µθ(xt,t),Σθ(xt,t)), (2)

where µθ and Σθ are learned parameters representing the mean and covariance of the denoised distribution.

#### 3.2. Temporal Attention in DiT Blocks

DiT-based video generation models employ temporal transformer blocks focusing on cross-frame interactions. Each temporal block consists of feed-forward networks, selfattention modules, and optional cross-attention layers.

The temporal self-attention module computes attention weights between frames, allowing the model to aggregate information from past and future frames. For video latent z ∈ RB×F×C×H×W with batch size B, F frames, C channels, and spatial dimensions H × W, it reshapes features by merging spatial dimensions into the batch size, yielding z˜ ∈ R(B×H×W)×F×C. Self-attention (Vaswani et al., 2023) is then applied along the frame axis:

A = Attention(Q(z˜),K(z˜)) ∈ R(B×H×W)×F×F (3) where Q and K denote the Query and Key heads, and A satisfies Fj=1 A(b,i,j) = 1.

Temporal attention is crucial for balancing coherence and flexibility in video generation. However, findings in Figure 3 reveal that standard temporal attention mechanisms often underemphasize cross-frame interactions, as attention weights for non-diagonal elements are typically much lower than diagonal elements. This shortcoming can lead to temporal inconsistencies like flickering or unexpected transitions, further affecting the spatial content negatively.

#### 3.3. Temperature in DiT-based Video Generation

The temperature is a critical concept in large language model (LLM) inference, controlling the randomness and coherence of the generated tokens. The probability P(x) of generating a token x is adjusted using the temperature τ as:

exp z(τx) x′ exp z(x

P(x) =

′) τ

(4)

enhance temperature

Cross-Frame Intensity

drop and mean

Temporal Attention Map Enhance Block

Temporal Attention

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | |
|---|---|
| | |

Hidden States

multiplication sum

non-diagonal attention

diagonal attention (dropped)

- Figure 4. Overview of the Enhance Block. The block computes the average of non-diagonal elements from the temporal attention map as Cross-Frame Intensity (CFI). The CFI is scaled by the temperature parameter and fused back to enhance the temporal attention output.

where z(x) represents the unnormalized logit for token x, and τ > 0 controls the degree of randomness: a lower τ makes the output more deterministic, while a higher τ

increases diversity by flattening the probability distribution. In video generation, a similar temperature principle can be considered when using DiT models, where the temporal attention mechanism controls the relationship between generated frames. Equation 5 presents a direct usage of temperature in temporal attention of DiT models.

Attention(Q,K) = softmax

QK⊤ τ ·

√dk

(5)

In particular, properly increased temperatures amplify nondiagonal temporal attention, allowing the DiT model to draw global information from multiple frames during generation, leading to better spatial diversity and temporal consistency. On the other hand, setting the temperature extremely high results in uniform attention across all frames, possibly generating unexpected or prompt-irrelevant content.

However, video generation requires a proper balance between cross-frame and intra-frame attention, if we directly apply the LLM-style temperature adjustment similarly to change the original attention weights , we always fail to enhance target cross-frame dependencies suitably. Directly applying τ to temporal attention causes increasing changes as the model deepens and denoising steps accumulate, which can lead to overly smooth motion, loss of visual details, and unstable video generation, as illustrated in Appendix A.

#### 3.4. Enhance Block

To better adaptively adjust the temperature in the temporal attention mechanism, we propose a novel method, EnhanceA-Video, to enhance temporal consistency in video generation by utilizing the non-diagonal temporal attention with enhance temperature parameter. The cross-frame intensity is measured by the non-diagonal temporal attention, where higher values enable the model to focus on a broader temporal context, corresponding to higher temperature. By further introducing the enhance temperature parameter to scale the cross-frame intensity, we appropriately adjust the temporal attention outputs as a training-free enhancement.

As presented in Figure 4, we design an Enhance Block as a parallel branch to the temporal attention mechanism. The Enhance Block operates as follows:

First, the temporal attention map A ∈ RF×F is computed, where F is the number of frames. The diagonal elements Aii correspond to intra-frame attention, and the non-diagonal elements Aij (i ̸= j) represent cross-frame attention.

Next, the Cross-Frame Intensity (CFI) is calculated by averaging the non-diagonal elements of the attention map:

F

1 F(F − 1)

CFI =

i=1

F

Aij. (6)

j=1 j̸=i

The CFI is then multiplied by the enhance temperature parameter τ to enhance cross-frame correlations better:

CFIenhanced = clip((τ + F) · CFI,1). (7)

[Figure 4]

- Figure 5. Temporal attention difference map between original CogVideoX model and w/ Enhance-A-Video of layer 29 at denoising step

50. Non-diagonal elements in the attention matrix of w/ Enhance-A-Video show higher values (shown in blue), while diagonal elements have reduced values (shown in red).

Noticeably, the enhanced Cross-Frame Intensity (CFIenhanced) is clipped at a minimum value of 1, which prevents excessive deterioration of cross-frame correlations during enhancement.

Finally, the output of the Enhance Block (CFIenhanced) is utilized to enhance the original temporal attention block output Oattn in the residual connection (He et al., 2015; Si et al., 2024):

Ofinal = CFIenhanced · Oattn + H. (8)

where H represents the hidden states that are inputs of the attention block.

When CFIenhanced exceeds 1, indicating significant crossframe information, the ratio of temporal attention block outputs is correspondingly amplified in Ofinal. Otherwise, the connection defaults to a standard residual connection. Since Oattn is relatively small compared to H, modest enhancements (small CFIenhanced) to Oattn slightly affect the Ofinal distribution, enabling Enhance-A-Video to enhance cross-frame attention without substantially altering original attention patterns. The complete analytical details are available in Appendix B.

The temporal attention difference map in Figure 5 shows the difference between the temporal attention of the original CogVideoX model and w/ Enhance-A-Video, illustrating how Enhance-A-Video properly strengthens cross-frame attention. Specifically, certain non-diagonal elements (blue areas) are moderately increased (e.g., 0.9×10−2), indicating

enhanced cross-frame correlations. Meanwhile, the diagonal elements experience a minimal reduction (3.3 × 10−2 at most), which ensures stable intra-frame attention and preserves existing fine-grained visual details. More analysis can be found in Appendix A.

### 4. Experiments

#### 4.1. Setup

To evaluate the effectiveness of our proposed Enhance-AVideo method, we conduct experiments on video generation models incorporating two types of attention mechanisms: 3D full attention and spatial-temporal attention. Specifically, we choose several representative models for each category:

3D Full Attention Model: HunyuanVideo (Kong et al., 2025), CogVideoX (Yang et al., 2024) and LTX-Video (HaCohen et al., 2024), which employ 3D full attention to model spatial and temporal dependencies simultaneously.

Spatial-Temporal Attention Model: Open-Sora (Zheng et al., 2024) and Open-Sora-Plan v1.0.0 (Lin et al., 2024), which decompose the attention mechanism into separate spatial and temporal components for computational efficiency and scalability.

We follow the original setup of these methods exactly and incorporate the Enhance Block exclusively into the temporal attention modules of these models during the inference phase without additional retraining or fine-tuning. For 3D full attention models, we reshape the 3D attention to fo-

[Figure 5]

- Figure 6. Qualitative results of Enhance-A-Video on HunyuanVideo. Captions: (a) An antique car drives along a dirt road through golden wheat fields. Dust rises softly as wheat brushes against the car with distant trees meeting a blue sky. (b) A baseball player grips a bat in black gloves, wearing a blue-and-white uniform and cap, with a blurred crowd and green field highlighting his focused stance.

cus on calculating temporal attention and the corresponding CFIenhanced, which is then applied to enhance the 3D attention outputs in the same way.

- 4.2. 3D Full Attention Model

[Figure 6]

- Figure 7. Qualitative results of Enhance-A-Video on CogVideoX. Captions: (a) A cute and happy Corgi playing in the park, in a surrealistic style. (b) Balloon full of water exploding in extreme slow motion.

augmentation in HunyuanVideo improved the model’s video generation capabilities effectively. The results shown in Figure 6 demonstrate that Enhance-A-Video consistently produces more realistic images with better details.

In the first case, HunyuanVideo’s output shows a driverless car moving unnaturally in reverse, while Enhance-A-Video generates a car moving realistically in the correct direction. In the second case, HunyuanVideo produces conflicting artifacts - duplicate right hands and unnatural head movement. In contrast, Enhance-A-Video captures the baseball player’s motion with natural fluidity and richer detail.

By applying Enhance-A-Video to CogVideoX (Yang et al., 2024), we observe significant improvements in promptvideo consistency, temporal coherence, and visual detail. In caption (b) of Figure 7, CogVideoX fails to accurately capture the prompt describing a “balloon full of water”, generating only vague water splashes without the balloon. In contrast, the enhanced model produces videos that better align with the given prompts while delivering smoother transitions and clearer visuals.

LTX-Video (HaCohen et al., 2024) is a real-time latent text-to-video diffusion model that generates high-quality, temporally consistent videos efficiently. The integration of Enhance-A-Video into LTX-Video further improves temporal consistency and enhances spatial details. As exhibited in Figure 8, the enhanced model produces videos with sharper textures, more vivid colors, and smoother transitions com-

HunyuanVideo (Kong et al., 2025) is a state-of-the-art textto-video diffusion model recognized for its ability to produce high-resolution and temporally coherent videos from textual prompts. Our implementation of Enhance-A-Video

pared to the baseline LTX-Video.

The snow-covered mountains (top row) and river scene (bottom row) generated by Enhance-A-Video display clearer structures and more natural color gradients, while the baseline results appear less detailed and slightly blurred. This demonstrates that Enhance-A-Video effectively strengthens cross-frame attention, leading to more realistic and visually appealing videos.

[Figure 7]

- Figure 8. Qualitative results of Enhance-A-Video on LTX-Video. Captions: (a) The camera pans over snow-covered mountains, revealing jagged peaks and deep, narrow valleys. (b) An emeraldgreen river winds through a rocky canyon, forming reflective pools amid pine trees and brown-gray rocks.

#### 4.3. Spatial-Temporal Attention Model

Open-Sora (Zheng et al., 2024) is an efficient text-tovideo generation model that utilizes a decomposed spatialtemporal attention mechanism to balance computational efficiency and video quality. Incorporating the Enhance-AVideo augmentation into Open-Sora significantly improved temporal consistency and spatial detail preservation. As demonstrated in Figure 9, the enhanced model produces videos with more natural motion transitions and more realistic visual details. Besides, the results on Open-Sora-Plan v1.0.0 are provided in Appendix D.

#### 4.4. Quantitative Analysis

We evaluated video quality through a blind user study of 110 participants. Each person compared two videos generated from the same text prompt and random seed - one from baseline models and one from w/ Enhance-A-Video. The videos were shown in random order to prevent bias. Participants chose which video they preferred based on three criteria: temporal consistency, prompt-video consistency, and overall visual quality.

- Figure 10 presents the main user study results for chosen models and w/ Enhance-A-Video of each evaluation criterion. The results show that models using Enhance-AVideo received the majority of preference, demonstrating that Enhance-A-Video notably enhances the text-to-video

[Figure 8]

- Figure 9. Qualitative results of Enhance-A-Video on Open-Sora. Caption: A cake.

[Figure 9]

- Figure 10. User study results comparing baseline models and w/ Enhance-A-Video across evaluation criteria.

models’ performance in all evaluated aspects 1:

Temporal Consistency. The usage of Cross-Frame Intensity (CFI) and the enhance temperature parameter strengthens cross-frame connections. This results in smoother motion transitions and improved frame-to-frame alignment, which creates a more stable and coherent visual experience in the generated video.

Prompt-video Consistency. In diffusion-based video generation, video frames are progressively denoised based on the prompt. However, the lack of temporal attention in cross-frame information transmission causes the semantic alignment between the video and the prompt to deviate gradually during generation. Enhancing cross-frame information by Enhance-A-Video ensures that objects and actions in the scene remain consistent with the prompt. This smooth

1Appendix C contains a comprehensive analysis with specific user study examples.

[Figure 10]

- Figure 11. Ablation study on the enhance temperature parameter in the Enhance Block. Moderate values balance temporal consistency and visual diversity, while extreme values degrade performance.

semantic evolution avoids abrupt or inconsistent content, improving the alignment between the generated video and the given prompt.

Visual Quality. By using CFI and the enhanced temperature parameter, the model makes better use of information from adjacent frames to improve details, especially in object textures and edges. The improved cross-frame attention smooths the denoising process and reduces random changes, allowing the model to generate more consistent motion and avoid unrealistic movements.

Moreover, we conducted independent evaluations using VBench (Huang et al., 2024) for each video generation model with 5 random seeds. Table 1 shows that integrating Enhance-A-Video consistently improves VBench scores across all models 2. These results confirm that Enhance-AVideo effectively boosts temporal consistency and visual quality with minimal overhead.

Table 1. Comparison of VBench Score for CogVideoX, Open-Sora, and LTX-Video models without and with Enhance-A-Video.

Method CogVideoX Open-Sora LTX-Video

Baseline 77.27 79.04 71.93 w/ Enhance-A-Video 77.34 79.16 72.04

#### 4.5. Ablation Study

Impact of Temperature. To better understand the impact of the temperature parameter, we conduct an ablation study by varying the enhance temperature parameter in the Enhance Block. Results in Figure 11 indicate that moderate temperature values achieve the best balance between temporal consistency and diversity, while extreme values (too low or too high) will degrade performance.

Effects of Clipping. Figure 12 illustrates that applying

2VBench may not fully reflect the quality advancements of Enhance-A-Video as discussed in Appendix F.

[Figure 11]

Figure 12. Visual comparison of video generation results with and without the clipping mechanism in the Enhance Block.

the clipping effectively stabilizes cross-frame attention, resulting in clearer visuals and smoother motion. Without clipping, the model produces noticeable artifacts such as motion blur and distorted details, highlighting the necessity of clipping for maintaining temporal consistency and preserving spatial fidelity.

Minimal Overhead. To evaluate the inference efficiency of the proposed Enhance-A-Video (EAV) method, we conducted an ablation study on two prevail video generation models in Table 2 using 1 A100 GPU. These negligible increases in the two models indicate that the Enhance-A-Video method is highly efficient and scales well when integrated into large video generation models.

Table 2. Comparison of inference efficiency for HunyuanVideo and CogVideoX models with and without Enhance-A-Video.

Time (min)

Model

Overhead w/o EAV w/ EAV

HunyuanVideo 50.32 50.72 0.8% CogVideoX 1.53 1.57 2.1%

### 5. Conclusion

This paper presents Enhance-A-Video, a simple yet effective method that improves temporal consistency and visual quality in DiT-based video generation. By pioneering the exploration of cross-frame information and the temperature concept in DiT blocks, the method offers a straightforward yet powerful solution for video generation enhancement. Its robust generalization and ease of implementation suggest promising future developments in better video generation.

### Impact Statement

This paper presents research aimed at advancing video generation by improving the temporal consistency and visual quality of diffusion transformer-based models through a training-free, plug-and-play approach. Our experiments use publicly available models and benchmarks, posing no risk of harmful societal consequences while contributing to applications in entertainment, education, and media production with minimal computational overhead.

### References

Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., and Lorenz, D. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., Ng, C., Wang, R., and Ramesh, A. Video generation models as world simulators. 2024.

- Chen, S., Xu, M., Ren, J., Cong, Y., He, S., Xie, Y., Sinha, A., Luo, P., Xiang, T., and Perez-Rua, J.-M. Gentron: Diffusion transformers for image and video generation, 2024. URL https://arxiv.org/abs/2312.04557.
- Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. An empirical study of training self-supervised vision transformers. In International Conference on Computer Vision (ICCV), 2021.

Christiano, P. F., Leike, J., Brown, T. B., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., Muller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., Podell, D., Dockhorn, T., English, Z., Lacey, K., Goodwin, A., Marek, Y., and Rombach, R. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024. doi: 10.48550/arXiv.2403.03206.

Gao, P., Zhuo, L., Liu, D., Du, R., Luo, X., Qiu, L., Zhang, Y., Lin, C., Huang, R., Geng, S., Zhang, R., Xi, J., Shao, W., Jiang, Z., Yang, T., Ye, W., Tong, H., He, J., Qiao, Y., and Li, H. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers, 2024. URL https://arxiv.

org/abs/2405.05945.

Guo, Y., Yang, C., Rao, A., Liang, Z., Wang, Y., Qiao, Y., Agrawala, M., Lin, D., and Dai, B. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. International Conference on Learning Representations, 2024.

HaCohen, Y., Chiprut, N., Brazowski, B., Shalem, D., Moshe, D., Richardson, E., Levin, E., Shiran, G., Zabari, N., Gordon, O., Panet, P., Weissbuch, S., Kulikov, V., Bitterman, Y., Melumian, Z., and Bibi, O. Ltxvideo: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

- He, J., Xue, T., Liu, D., Lin, X., Gao, P., Lin, D., Qiao, Y., Ouyang, W., and Liu, Z. Venhancer: Generative space-time enhancement for video generation, 2024. URL https://arxiv.org/abs/2407.07667.
- He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778, 2015. URL https: //api.semanticscholar.org/CorpusID: 206594692.

Henschel, R., Khachatryan, L., Hayrapetyan, D., Poghosyan, H., Tadevosyan, V., Wang, Z., Navasardyan, S., and Shi, H. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. arXiv preprint arXiv:2403.14773, 2024.

Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., and Fleet, D. J. Video diffusion models. arXiv preprint arXiv:2204.03458, 2022.

Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. The curious case of neural text degeneration. In International Conference on Learning Representations (ICLR), 2020.

Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., and Liu, Z. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., Wu, K., Lin, Q., Yuan, J., Long, Y., Wang, A., Wang, A., Li, C., Huang,

D., Yang, F., Tan, H., Wang, H., Song, J., Bai, J., Wu, J., Xue, J., Wang, J., Wang, K., Liu, M., Li, P., Li, S., Wang, W., Yu, W., Deng, X., Li, Y., Chen, Y., Cui, Y., Peng, Y., Yu, Z., He, Z., Xu, Z., Zhou, Z., Xu, Z., Tao, Y., Lu, Q., Liu, S., Zhou, D., Wang, H., Yang, Y., Wang, D., Liu, Y., Jiang, J., and Zhong, C. Hunyuanvideo: A systematic framework for large video generative models, 2025. URL https://arxiv.org/abs/2412.03603.

Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), Proceedings of the 17th International Conference on Machine Learning (ICML 2000), pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann.

Li, X., Liu, Y., Cao, S., Chen, Z., Zhuang, S., Chen, X., He, Y., Wang, Y., and Qiao, Y. Diffvsr: Enhancing realworld video super-resolution with diffusion models for advanced visual quality and temporal consistency, 2025. URL https://arxiv.org/abs/2501.10110.

Lin, B., Ge, Y., Cheng, X., Li, Z., Zhu, B., Wang, S., He, X., Ye, Y., Yuan, S., Chen, L., Jia, T., Zhang, J., Tang, Z., Pang, Y., She, B., Yan, C., Hu, Z., Dong, X., Chen, L., Pan, Z., Zhou, X., Dong, S., Tian, Y., and Yuan, L. Open-sora plan: Open-source large video generation model, 2024. URL https://arxiv.org/abs/ 2412.00131.

Lu, Y., Liang, Y., Zhu, L., and Yang, Y. Freelong: Trainingfree long video generation with spectralblend temporal attention, 2024. URL https://arxiv.org/abs/ 2407.19918.

Ma, X., Wang, Y., Jia, G., Chen, X., Liu, Z., Li, Y.-F., Chen, C., and Qiao, Y. Latte: Latent diffusion transformer for video generation, 2024. URL https://arxiv.org/ abs/2401.03048.

Peebles, W. S. and Xie, S. Scalable diffusion models with transformers. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4172–4182, 2022. URL https://api.semanticscholar.

org/CorpusID:254854389.

Peeperkorn, M., Kouwenhoven, T., Brown, D. G., and Jordanous, A. K. Is temperature the creativity parameter of large language models? ArXiv, 2024. doi: 10.48550/arXiv.2405.00492.

Renze, M. and Guven, E. The effect of sampling temperature on problem solving in large language models. ArXiv, abs/2402.05201, 2024. URL https: //api.semanticscholar.org/CorpusID: 267547769.

Si, C., Huang, Z., Jiang, Y., and Liu, Z. Freeu: Free lunch in diffusion u-net. In CVPR, 2024.

Tan, C., Gao, Z., Wu, L., Xu, Y., Xia, J., Li, S., and Li, S. Z. Temporal attention unit: Towards efficient spatiotemporal predictive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18770–18782, 2023.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need, 2023. URL https://arxiv.org/ abs/1706.03762.

Xu, J., Zou, X., Huang, K., Chen, Y., Liu, B., Cheng, M., Shi, X., and Huang, J. Easyanimate: A high-performance long video generation method based on transformer architecture, 2024a. URL https://arxiv.org/abs/ 2405.18991.

Xu, Y., Park, T., Zhang, R., Zhou, Y., Shechtman, E., Liu, F., Huang, J.-B., and Liu, D. Videogigagan: Towards detail-rich video super-resolution, 2024b. URL https: //arxiv.org/abs/2404.12388.

Yan, W., Hafner, D., James, S., and Abbeel, P. Temporally consistent transformers for video generation. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 39062–39098. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/ v202/yan23b.html.

Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., Yin, D., Gu, X., Zhang, Y., Wang, W., Cheng, Y., Liu, T., Xu, B., Dong, Y., and Tang, J. Cogvideox: Text-to-video diffusion models with an expert transformer, 2024. URL https://arxiv.org/abs/2408.06072.

Zhang, D. J., Wu, J. Z., Liu, J.-W., Zhao, R., Ran, L., Gu, Y., Gao, D., and Shou, M. Z. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arXiv:2309.15818, 2023a.

Zhang, L., Rao, A., and Agrawala, M. Adding conditional control to text-to-image diffusion models, 2023b.

Zheng, Z., Peng, X., Yang, T., Shen, C., Li, S., Liu, H., Zhou, Y., Li, T., and You, Y. Open-sora: Democratizing efficient video production for all, 2024. URL https: //arxiv.org/abs/2412.20404.

Zhou, S., Yang, P., Wang, J., Luo, Y., and Loy, C. C. Upscale-A-Video: Temporal-consistent diffusion model for real-world video super-resolution. In CVPR, 2024.

### A. Temperature Method Comparison

In Figure 13(a) and (b), where the temperature parameter τ and Cross-Frame Intensity are directly applied in temporal attention calculation separately as presented in Equation 9 and 10, the diagonal elements (e.g., 27.4, 6.3) show a significant weakening of intra-frame attention, leading to the severe loss of spatial details and resulting in blurry and unrealistic textures. Additionally, the large negative values in the off-diagonal regions indicate an overabundant distributed enhancement of cross-frame attention, resulting in limited improvement in video quality.

QK⊤ τ ·

Attention(Q,K,V ) = softmax

V (9)

√dk

QK⊤

Attention(Q,K,V ) = softmax

V (10)

√dk

CFIenhanced ·

In contrast, Figure 13(c) using the Enhance-A-Video method shows modest changes along the diagonal, with values close to zero, preserving intra-frame attention and maintaining fine-grained details. Moreover, the negative values in the off-diagonal regions (e.g., -1.3, -0.9) reflect a targeted and moderate enhancement of cross-frame attention, significantly improving motion coherence and overall video quality.

[Figure 12]

- Figure 13. Temporal attention difference maps and corresponding generated videos comparing three temperature enhancement methods. (a) Temperature Attention Scaling τ = 1.1. (b) CFI Attention Scaling. (c) Enhance-A-Video Method.

### B. CFI Distribution and L2 Norm Proportion in Residual Connection

The CFIenhanced values in Figure 14(a) range between 1.12-1.18, indicating a modest enhancement of keyframes containing important temporal information. Figure 14(b) shows two low proportions calculated as follows:

propCogvideoX = ||Oattn||2

||H||2

(11)

[Figure 13]

- Figure 14. (a) The distribution of CFIenhanced during the inference of CogVideoX w/ Enhance-A-Video in layer 4. (b) The proportion of l2 norms between Oattn and H in residual connection in layer 4.

propw/ Enhance-A-Video = ||CFIenhanced · Oattn||2

||H||2

(12)

suggesting that attention outputs are relatively small compared to hidden states in the residual connection. Consequently, applying CFIenhanced to attention outputs rather than attention allows for enhancing important information with minimal disruption to the original attention distribution. Thus, Enhance-A-Video improves temporal consistency while preserving existing spatial details.

C. User Study Example

[Figure 14]

- Figure 15. Selected user study example with the caption: A woman with curly hair sits comfortably in the driver’s seat of a sleek, modern car, her eyes focused on the road ahead.

In the first example shown in Figure 15, 37% of participants preferred the video from the basic HunyuanVideo model over the version enhanced by Enhance-A-Video. This unexpected preference occurred because these participants overlooked the enhanced details (marked by two squares) in the improved version. The enhanced model actually produced more precise and detailed elements, particularly in the interior door handle and steering wheels, demonstrating how Enhance-A-Video can improve the baseline model’s ability to generate fine visual details.

Nevertheless, the enhanced result received an overwhelming 91% of votes in the second example presented in Figure 16. The superior quality of the silver plane in the enhanced version is immediately apparent, making it a much clearer improvement over the original HunyuanVideo compared to the previous example, where the differences are more subtle and require

[Figure 15]

- Figure 16. Selected user study example with the caption: A sleek, silver airplane soars gracefully through a vast, azure sky, its wings cutting through wispy, cotton-like clouds.

careful observation to notice.

In general, Enhance-A-Video introduces more visual details in generated videos, but the limited observation time in the user study prevented volunteers from noticing this advantage. However, with the release of advanced models like Sora, the demand for detailed and realistic video generation continues to grow. This trend underscores the growing importance of Enhance-A-Video in refining details and its role as a valuable tool for achieving higher-quality video generation.

### D. More Experimental Results

[Figure 16]

Figure 17. Qualitative results of Enhance-A-Video on Open-Sora-Plan.

Open-Sora-Plan v1.0.0 (Lin et al., 2024) is a text-to-video generation model leveraging a multi-resolution latent diffusion framework for high-quality and temporally coherent videos. As shown in Figure 17, for the left example, Enhance-A-Video creates clearer leaves and sharper flower details, removing the blur seen in the baseline model. In the right example, the enhanced version delivers clearer water flow and better-defined rocks, showcasing natural lighting and textures. These improvements highlight Enhance-A-Video’s ability to enhance cross-frame attention and produce visually high-quality videos.

### E. Limitations

Our approach shows modest quantitative improvements, primarily limited by the temperature parameter requiring different optimal values for each prompt. In future work, we plan to develop an adaptive temperature mechanism using RLHF (Christiano et al., 2017) to adjust this parameter based on the specific prompt context automatically. Besides, we focused solely on enhancing temporal attention without addressing spatial attention or cross-attention mechanisms, which are crucial for preserving spatial coherence and prompt alignment. Future work could explore incorporating these mechanisms to improve spatial video quality and semantic consistency.

### F. Discussion on VBench

The VBench benchmark does not fully reflect the substantial quality improvements achieved by Enhance-A-Video. Take the Aesthetic Quality metric as an example: the Aesthetic Quality metric in VBench is designed to evaluate the human-perceived visual quality of video frames. In the comparison of airplane footage in Figure 18 that achieves a majority of votes from user-study participants, the Enhance-A-Video version shows noticeably better detail and clarity in rendering the aircraft compared to the HunyuanVideo baseline, yet it receives a lower Aesthetic Quality score (55.59 vs 57.06). This scoring discrepancy suggests that VBench may not effectively catch actual improvements in video enhancement quality in some cases.

[Figure 17]

- Figure 18. Comparison of video quality between HunyuanVideo and w/ Enhance-A-Video on a caption: A sleek, silver airplane soars gracefully through a vast, azure sky, its wings cutting through wispy, cotton-like clouds. The sun glints off its polished surface, creating a dazzling spectacle against the endless blue expanse. As it glides effortlessly, the contrail forms a delicate, white ribbon trailing behind, adding to the scene’s ethereal beauty. The aircraft’s engines emit a soft, distant hum, blending harmoniously with the serene atmosphere. Below, the earth’s curvature is faintly visible, enhancing the sense of altitude and freedom. The scene captures the essence of flight, evoking a feeling of wonder and exploration.

In another example from Figure 19, while the video produced by Enhance-A-Video more accurately captures the prompt’s details—such as “sandy hair”, “sandcastles and beach toys”—it nonetheless receives a lower Aesthetic Quality rating when compared to the baseline HunyuanVideo model.

[Figure 18]

- Figure 19. Comparison of video quality between HunyuanVideo and w/ Enhance-A-Video on a caption: A young girl, wearing a widebrimmed straw hat and a colorful swimsuit, carefully applies sunblock to her younger brother’s face on a sunlit beach. The boy, with sandy hair and a playful grin, sits patiently on a striped beach towel, surrounded by sandcastles and beach toys. The gentle waves of the ocean provide a soothing soundtrack as seagulls call in the distance. The girl’s hands move with care, ensuring every inch of his face is protected, while the sun casts a warm glow over the scene, highlighting the siblings’ bond and the carefree joy of a summer day by the sea.

### G. Captions for Figure 1

- Caption 1 (top row): A young girl with curly hair, wearing a bright yellow dress, sits cross-legged on a wooden floor, surrounded by an array of colorful markers and crayons. She carefully colors a large piece of cardboard, her face a picture of concentration and creativity. The cardboard, propped up against a cozy living room couch, is filled with whimsical drawings of flowers, stars, and animals. Sunlight streams through a nearby window, casting a warm glow over her workspace. Her small hands move deftly, adding vibrant hues to her imaginative artwork, while her expression reflects pure joy and artistic focus.
- Caption 2 (bottom row): A young girl, wearing a wide-brimmed straw hat and a colorful swimsuit, carefully applies sunblock to her younger brother’s face on a sunlit beach. The boy, with sandy hair and a playful grin, sits patiently on a striped beach towel, surrounded by sandcastles and beach toys. The gentle waves of the ocean provide a soothing soundtrack as seagulls call in the distance. The girl’s hands move with care, ensuring every inch of his face is protected, while the sun casts a warm glow over the scene, highlighting the siblings’ bond and the carefree joy of a summer day by the sea.

