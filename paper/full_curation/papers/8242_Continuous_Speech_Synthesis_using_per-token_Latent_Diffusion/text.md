# Speech Synthesis From Continuous Features Using Per-Token Latent Diffusion

Arnon Turetzky1,2,§, Avihu Dekel2,§, Nimrod Shabtay2,3, Slava Shechtman2, David Haws2, Hagai Aronowitz2, Ron Hoory2, Yossi Adi1

1The Hebrew University of Jerusalem 2IBM Research 3Tel Aviv University §Core Contributors

## arXiv:2410.16048v2[eess.AS]23Nov2025

Abstract—We present SALAD, a zero-shot text-to-speech (TTS) autoregressive model operating over continuous speech representations. SALAD utilizes a per-token diffusion process to refine and predict continuous representations for the next time step. We compare our approach against a discrete variant of SALAD as well as publicly available zero-shot TTS systems, and conduct a comprehensive analysis of discrete versus continuous modeling techniques. Our results show that SALAD achieves superior intelligibility while matching the speech quality and speaker similarity of ground-truth audio.

I. INTRODUCTION

Discrete autoregressive (AR) models have become the leading approach in natural language processing. Their remarkable success has motivated researchers to extend these models to image and speech domains. Modeling continuous domains with discrete approaches typically involves quantization. In image generation, quantization is often realized using discrete autoencoders [1], which are further optimized with adversarial losses [2], [3], [4]. In audio generation, techniques like Residual Vector Quantization (RVQ) [5], [6] iteratively quantize the residual, producing multiple discrete codes per frame that can be modeled autoregressively [2], [7], [8].

While presenting impressive results, discrete approaches have several limitations. First, modeling multiple dependent streams simultaneously is inherently complex and often requires additional strategies like two-stage modeling [7], [9], [10] or token interleaving [8]. Second, similarity among consecutive codec codes has been shown to cause robustness issues, leading to continuous stretches of silence or persistent noise [9]. Third, quantization methods inherently impose a fidelity upper-bound, constrained by continuous representations such as mel-spectrogram features [11]. This raises the question: Can we optimize an AR model to directly generate high-quality speech over continuous features?

Recently, several studies have explored language modeling with continuous-valued tokens. In image generation, GIVT [12] proposes representing the continuous distribution of images obtained by a VAE using a Gaussian Mixture Model (GMM), while AR-Diffusion [13] suggests a per-token image diffusion head to model similar representations. For speech generation, [14] presented an AR modeling approach over mel spectrograms, incorporating a Gaussian sampling module

followed by a post-net. Concurrently with our work, [15] propose a similar modeling approach to GIVT but for spoken data.

Inspired by AR-Diffusion [13], we propose SALAD (Speech synthesis with Autoregressive LAtent Diffusion), an autoregressive per-token latent diffusion model for zero-shot speech synthesis over continuous features. By operating directly in continuous space, SALAD eliminates the need for signal quantization, thus enabling higher-fidelity speech generation. As an autoregressive model, SALAD also naturally supports variable-length outputs, addressing a fundamental challenge not encountered in fixed-length image generation methods. We use semantic tokens [16], [17], [18] which are not used for signal synthesis but rather to enhance robustness and define the generation-stopping condition.

We evaluate SALAD against the discrete representation alternative employing RVQ quantization, specifically following the delay prediction pattern introduced by [8] (see Figure 1). Additionally, we compare SALAD with two leading publicly available zero-shot TTS systems, XTTS [19] and VoiceCraft [20]. Our experimental results demonstrate that SALAD achieves superior intelligibility and maintains speech quality and speaker similarity comparable to ground-truth audio, as confirmed by both objective metrics and subjective listening tests 1.

Our contributions: (i) We introduce SALAD, the first autoregressive per-token latent diffusion model for zero-shot speech synthesis directly over continuous acoustic representations, eliminating the need for discrete quantization. (ii) We empirically demonstrate that continuous latent diffusion modeling can surpass discrete modeling approaches in intelligibility and match them in quality and speaker similarity. (iii) Through comprehensive experiments and ablation studies, we analyze the advantages and trade-offs of continuous versus discrete speech modeling, providing practical insights into the design of zero-shot speech generation models.

1Samples available at: https://s3.us-south.objectstorage.softlayer.net/ zk-wav-data/Webpages/SynthesisPerTokenLatentDiffusion/index.html

|-1.5|-0.2|-1.6|0.4|
|---|---|---|---|
|-0.3|-0.7|-0.2|0.|
|0.|-2.4|0.7|-1.3|

|0.3|
|---|
|-0.1|
|-1.3|

[Figure 1]

VAE Latents

Diffusion Head

### Linear Transformer

|31|53|41|39|
|---|---|---|---|
|12|1|95|47|
|93|65|23|24|

|39|
|---|
|47|
|12|

RVQ Matrix

RVQ Heads

Transformer

Embed

Fig. 1: Continuous vs. discrete modeling

II. RELATED WORK

Zero-Shot TTS. Inspired by the success of in-context learning, there has been significant interest in zero-shot TTS systems that generalize to unseen speakers during inference, offering flexibility and improved quality [7]. Zero-shot TTS methods typically formulate the task as language modeling over text and audio tokens, leveraging short speaker prompts to synthesize speech aligned with the target speaker’s voice characteristics [20], [21], [22], [23]. Our method follows this paradigm but operates directly over continuous speech representations.

Semantic Tokens. Quantized embeddings from selfsupervised audio models, termed semantic tokens [16], [17], [18], capture phonetic and prosodic information beneficial for speech synthesis [24], [25], [26], unconditional audio generation [27], and multimodal text-audio tasks [28]. SALAD utilizes semantic tokens as an auxiliary representation defining a precise stopping condition for continuous acoustic modeling.

RVQ Codes Prediction. Discrete audio generation methods commonly utilize Residual Vector Quantization (RVQ), involving multiple quantization layers [5], [6]. Methods such as AudioLM [27] flatten RVQ codes into sequences, whereas others, like Vall-E [7], employ two-stage prediction approaches or token interleaving strategies [8]. Differently, SALAD directly predicts a continuous latent space, thus avoiding the need to predict multiple residuals codes.

Continuous Models. When learning a continuous distribution, recent works typically use diffusion models, which were developed to sample from complex continuous probability distributions, inspired by non-equilibrium thermodynamics [29]. Several works attempt to synthesize speech using a diffusion process, which has the challenge of generating variable length outputs [30], [31], [32]. For that end, most diffusion-based works rely on a duration predictor that predicts the audio length in advance, which might be inferior to determining the length on-the-fly during synthesis [21], [22]. MELLE [14] predicts Mel spectrograms autoregressively using a Gaussian sampling module, and parameterizes the next frame using a Gaussian distribution, which restricts it to learn only unimodal

distributions. MELLE relies on an additional binary classifier that indicates when to stop, which is a highly imbalanced classification problem. In contrast, SALAD operates on VAE latent tokens, which allows sampling diverse inputs while training, and uses a diffusion head, capable of modeling multimodal distributions. SALAD relies on semantic-tokens to determine the stopping condition, a more balanced representation which also provides contextual information.

III. BACKGROUND

Definitions. We denote the raw audio sequence as a = (a1,...,am) where ai ∈ [−1,1] with sampling rate fS. The text is y = (y1,..,yk) where yi ∈ A, and A is the text vocabulary. We obtain compressed audio representations using a variational autoencoder (VAE), trained with adversarial losses to obtain high-fidelity reconstructions. The VAE’s encoder E predicts a sequence of means and variances of normal distribution: (µ1,...,µn),(σ12,...,σn2) = E(a) where σi,µi ∈ Rd and d is the VAE bottleneck dimension. The VAE downsamples the sequence with a stride r. We sample xi ∼ N(µi,σi2) and denote x = (x1,..,xn) as the continuous acoustic tokens. Hereafter, we refer to these continuous VAE latents interchangeably as acoustic tokens or continuous acoustic representations, following prior usage in speech-diffusion literature[14]. The VAE’s decoder D is used for reconstruction aˆ1,...,aˆm = D(x1,..,xn). We also extract semantic tokens and denote them by w = (w1,..,wm), which have the same downsampling stride as the VAE. Our goal is to predict the audio based on the desired text and the speaker prompt. Denoting the speaker prompt latent features as s = s1,...,sp, our training objective can be formulated by: p(x|y,s).

Diffusion Process. A diffusion process starts from a continuous signal, and gradually destroys it using a forward noise process. Our method performs latent diffusion, and attempts to predict the VAE latent vectors x1,...,xn. Given noising coefficients β0,...,βT and some continuous vector x, we

define√1 − βxt0xt=−1x+and√βϵtϵ∼. ThisN(0,Iiterative); the Markovdenoisingstructureprocessiscanxtbe= simplified. By defining αt = 1−βt and α¯t = ti=1 αi, we get that xt = √α¯tx + √1 − α¯tϵ. The diffusion process is often

[Figure 2]

[Figure 3]

[Figure 4]

(a) Training (b) Inference

Fig. 2: The per-token diffusion head

defined such that α¯T → 0 and xT distributes closely to the standard normal distribution. Diffusion models ϵθ are trained to perform the reverse diffusion process, which denoises the corrupted signal by predicting the added noise. Their denoising loss is defined as L(x) = Eϵ,t ∥ϵ − ϵθ(t,xt)∥2 . Most diffusion models operate on a sequence x1,..,xn and attempt to denoise all tokens in parallel using ϵθ(t,xt1,...,xtn). Per-Token Diffusion Head. [13] proposed an MLP diffusion head for image generation. Unlike standard diffusion models, the diffusion head denoises each token independently, which gives additional flexibility when defining the conditioning information (e.g., predicting on previously predicted tokens). The authors rely on a transformer model Θ that extracts contextual per-token conditioning vectors z1,..,zn based on the input features and optional context vectors that we denote by C

#### z = z1,...,zn = Θ(C,x1,...,xn).

The diffusion head (noise estimator) ϵθ takes a contextual conditioning vector z and attempts to model the continuous distribution p(x|z). Given a target token x, a diffusion process is being applied conditioned on z. The loss is:

#### L(x,z) = Eϵ,t ∥ϵ − ϵθ(xt,t,z)∥2 . (1)

During training, t ∼ [T],ϵ ∼ N(0,I) is sampled for each token x, resulting in noisy targets xt, and L(x,z) is minimized. See Figure 2a. The denoising network is trained jointly with the transformer Θ, and the gradient with respect to z is propagated to the transformer. K different values of t,ϵ may be sampled for a given context vector and target z,x, with the additional complexity of just the MLP head rather than the entire model. During inference, a continuous vector is sampled from xT ∼ N(0,I) and the final result is obtained from the reverse diffusion process (see Figure 2b):

1 √αt

xt−1 =

βt √1 − α¯t

xt −

ϵθ(xt,t,z) + βtϵ. (2)

IV. SALAD

SALAD is an end-to-end zero-shot TTS model, that given text and a speaker prompt autoregressively predicting continuous VAE latents x using per-token latent diffusion. Our

method utilizes semantic tokens w as an auxiliary representation that provides contextual information and determines the stopping condition. We project both the semantic and acoustic representations into a shared space, and sum them to obtain a fused representation. We adopt the delay pattern suggested by [8], such that each acoustic token xi is predicted based on the semantic token wi, for which we define ri = f(wi,xi−1).

Formally, model predicts both discrete semantic tokens and continuous acoustic tokens autoregressively, based on the text and speaker, using a causal transformer:

p (w1,x0),...,(wn,xn−1) | t,s =

n

p (wi,xi−1) | t,s,r1,...,ri−1 . (3)

i=1

For each autoregressive step i, we extract contextual features from our transformer backbone, based on the text and speaker prompt zi = Θ(t,s,r1,...ri), which is used to predict wi+1 using the cross-entropy loss Ls by the MLP head, and xi using the diffusion loss La by the diffusion head (see the system overview in Figure 3). We halt the generation once the semantic prediction head samples an EOS token. We note that audio duration is predicted on the-fly based on the model’s predictions, unlike most diffusion-based TTS models, where the audio duration is predetermined.

V. EXPERIMENTS A. Experimental Setup

Datasets. We train our models on the English subset of multi-lingual LibriSpeech (MLS) [33], which contains 10M examples of 10-20 seconds, resulting in 45K hours. To avoid over-exposure of a few speakers, we limit the maximal number of utterances per speaker to 10K, resulting in 5.2M examples. We evaluate all models on LibriSpeech test-clean [34], which consists of 2620 utterances by 40 speakers. All speakers in the test set are excluded from the training set. We filter the dataset to utterances with lengths of 8-25 seconds, and then limit to at most 15 samples per speaker, resulting in 564 utterances for evaluation.

Tokenization. To derive acoustic tokens, we train continuous β-VAE-GAN, with a varying bottleneck dimension d ∈ {8,16,24,32}, and set the KL-divergence regularization

[Figure 5]

Fig. 3: System overview. Conditioned on text t and a speaker prompt s, SALAD emits semantic tokens w and acoustic latents x, which a VAE decoder converts to waveform.

to β = 5 · 10−5, as done in [12]. We also train discrete RVQ-GAN models with q ∈ {4,8,12} codebooks, each with 1024 entries. In addition, we apply quantizer dropout [5] with p = 0.5. All compression models are trained on MLS-English, DAPS, LibriTTS, LibriTTS-R and LJ-Speech, which balance between high and mid quality recordings [35]. The all-training hyperparameters follow the original recipe proposed by [36]. We extract semantic tokens by quantizing the embeddings of the 11th layer of W2V-BERT [37] using minibatch K-means with 1024 centroids.

Architecture. We use a transformer backbone with d = 1024, dff = 4096, 24 layers, 16 heads, sinusoidal positional embedding, GeLU activation, and a dropout rate of 0.1, resulting in models with roughly 350M parameters. VAE embeddings are projected using a linear layer, while RVQ tokens are embedded using Q lookup tables, which are summed into a single embedding. We use Classifier-Free Guidance (CFG) [38] and randomly omit the speaker prompt with p = 0.1 during training.

RVQ codes are predicted using a Q MLP heads with four hidden layers. For discrete distributions we apply top k = 10 sampling, with a temperature of τ = 1, a repetition penalty of 1.05, and a CFG scale of α = 3.

We use a diffusion process with T = 1000 steps, where betas are logarithmicly spaced between β0 = 2e − 4 and βT = 0.03. Our per-token diffusion head is an MLP network with 12 residual layers, that predicts the noise ϵ given the transformer embedding vector z, the noisy input xt, and the diffusion step t. Each residual block consists of layer normalization, linear layer, SiLU activation, and dropout with p = 0.1. During inference, we apply 20 diffusion steps for sampling, with a default noise scale of 1. We use the AdamW optimizer, with lrmax = 3e − 4 and lrmin = 3e − 5, weight decay 0.1, and a clip gradient norm of 1, and train with FP16 mixed precision. We linearly warm up the learning rate from lrmin across 32K iterations to lrmax and decay the learning rate back to lrmin over 300K steps using a cosine schedule. Each global batch size has ∼150K acoustic tokens. Each model was trained with 8 A100 80GB GPUs.

Metrics. We measure Audio Quality using UTMOS [39] which produces a quality score in the range of 1-5 (higher is better). Intelligibility is measured by the character error rate (CER) in percentages (%) between the ground-truth text and the Whisper transcripts [40] of the synthesized audio. Speaker

Model Type UTMOS CER Sim. GT – 4.121 0.528 0.736 XTTS Disc. 3.91 0.787 0.544 VoiceCraft Disc. 3.986 4.067 0.598 SALAD Disc. 4.27 2.298 0.600 SALAD Cont. 4.28 0.739 0.539

TABLE I: Objective evaluation of LibriSpeech test-clean

Similarity is measured by the cosine similarity to the prompt, comparing the embedding of WavLM-TDNN [41], a popular speaker verification model. This metric was also reported in Vall-E and subsequent studies [7], [9]. The similarity score predicted is in the range of [−1,1], where a larger value indicates a higher similarity.

For the subjective listening tests, we selected one random utterance for every speaker in LibriSpeech test-clean (20 female and 20 male speakers), resulting in 40 utterances for evaluation. For each sample, we selected a three-second-long speaker prompt from another random utterance of the same speaker. Each system synthesizes the desired utterance based on the same text and speaker prompt. All experiments were conducted on the Amazon Mechanical Turk crowd-sourcing platform with votes collected from 39-58 subjects qualified as masters [42].

In the first listening test we assess speech quality and naturalness by the standard 5-point scale Mean Opinion Score (MOS) [43]. 25 distinct subjects assessed each utterance. We report the average scores and the 95% confidence interval.

In the second listening test we asses the Speaker Similarity by a 4-level pairwise similarity test, as in [44], [45], where subjects were presented with (utterance, prompt) pairs and asked to rank speaker similarity of each pair on a 4-level categorical scale (definitely different speakers, probably different speakers, probably the same speaker, definitely the same speaker). Each utterance was assessed by 20 distinct subjects on average. We report the mean similarity score and the 95% confidence interval while attaching 1-4 numerical values to the above categories, as in [45].

We further compare SALAD with two leading publicly available zero-shot TTS systems: XTTS [19], run with its official package using default inference settings; and VoiceCraft [20], using the 330MTTSEnhanced model configured for top-k sampling (k = 40), nucleus sampling (p = 1),

[Figure 6]

[Figure 7]

(a) MOS Listening Test (1-5 scale) (b) Speaker Similarity Listening Test (1-4 scale)

- Fig. 4: Subjective listening results

[Figure 8]

(a) VQ/VAE reconstruction (b) Codewise Noise sensitivity (c) Codewise cross entropy loss

[Figure 9]

[Figure 10]

- Fig. 5: High-fidelity RVQ codes

temperature = 1, and repetition penalty = 3. The VoiceCraft authors report that this configuration outperforms their reimplementation of VALL-E2[9], which we do not include directly in our evaluation, as it is not publicly available and its demo set is too limited for reliable batch metrics.

B. Results

Subjective Evaluation. We conduct the two subjective listening tests, described above, to compare the following systems: (1) Ground Truth audio (2) XTTSv2 [19] (3) SALAD (4) SALAD - Discrete. Figure 4a reports the mean opinion score (MOS) results, suggesting that the difference between the ground-truth audio (GT) to SALAD is statistically insignificant (p > 0.01). Figure 4b presents the speaker similarity average score with 95% confidence intervals, suggesting similar or better speaker similarity scores for all the systems but XTTSv2. More precise analysis with two-sided Wilkinson rank-sum test [46] reveals that SALAD do not differ (p >> 0.01) from the GT in terms of speaker similarity, while SALAD-discrete is marginally better than the GT (p = 0.0105).

Objective Evaluation. We evaluate all models on zero-shot TTS. Given a text and a three-second speaker prompt, which is taken randomly from another utterance of the same speaker, the model attempts to synthesize the audio with the identity and prosody similar to the prompt. All models use the same random prompt for each sample.

Table I shows that SALAD achieves superior intelligibility (lowest CER) making it the most reliable model when having

to synthesize an exact text, with comparable UTMOS quality to the discrete baseline. However, in terms of speaker similarity SALAD is comparable to XTTS but the discrete baseline is superior.

C. Ablation Study

High-Fidelity Modeling. When increasing the number of RVQ codebooks or the VAE embedding dimension, the reconstruction quality increases, but language modeling can be difficult [22]. Figure 5a shows the reconstruction quality measured by PESQ [47], for different numbers of RVQ codebooks and VAE embedding dimensions. One concern regarding RVQ modeling is that the fine codes quantize noise, leading to a high gradient contribution of random classification problems. We measure the noise sensitivity per codebook by adding Gaussian noise into raw samples, compressing them with the RVQ model, and checking the ratio of change per codebook. Figure 5b suggests that fine codebooks are more noisesensitive. Figure 5c shows per-codebook validation crossentropy loss in the discrete 12-codebook model, indicating difficulty in reducing uncertainty in finer codebooks. This occurs despite the delay pattern in Section IV, where finer RVQ levels are conditioned on coarser layers of the same frame. Finally, we compare the generation quality with lesscompressed representations. Table II shows that higher fidelity causes a greater intelligibility drop in the high-fidelity discrete model than in the continuous model.

[Figure 11]

[Figure 12]

[Figure 13]

(a) CFG scale (b) Noise scale (c) Diffusion steps

Fig. 6: Inference hyperparameters influence

UTMOS ↑ Intelligibility ↓ Similarity ↑

Continuous (d=32) 4.271 1.157 0.545 Discrete (Q=12) 4.203 5.461 0.597

TABLE II: Discrete vs. continuous SALAD with high-fidelity representations. Intelligibility (CER) and Similarity (speaker similarity) are defined in Section. V (Metrics).

VAE Sampling Unlike discrete codebooks or Mel spectrograms, VAE models enable diverse sampling, which may enhance robustness and mitigate the mismatch between training and inference(where prediction rely on previous noisy outputs). To validate this, we compare two SALAD models - one sampling from the VAE distribution x = µ + ϵ · σ and the other using only the mean x = µ. The results presented in Table III, show a large gap in UTMOS and intelligibility indicating that sampling improves synthesized quality. Listening to VAE-NoSample outputs we observed increasing speaker-inconsistency artifacts, likely due to the absence of sampling noise during training. We suspect this noise improves robustness, and we hypothesize that the high similarity score of VAE-NoSample results from these artifacts.

Inference Hyperparameters We investigate CFG, noise scale and the number diffusion steps. In every experiment, we fix all values to the default inference values following those described in Section V, and change only a single hyperparameter. The CFG linear extrapolation coefficient increases the speaker similarity, but degrades the automatic quality metric, as seen in Figure 6a. Next, we scale the noise level added in each diffusion step by scaling the βtϵ term in Equation 2, and see improvements in similarity but degradation in the UTMOS quality score (Figure 6b). We also examine the number of diffusion steps, which improve similarity until reaching 20 diffusion steps, and also degrade UTMOS (Figure 6c).

VI. DISCUSSION

Compressing complex signals like audio and images involves a tradeoff between perception and generation. While compression can degrade perception by losing information, it benefits generation by simplifying the learned distribution. Developing generative methods for less-compressed representations could help mitigate this tradeoff. Though RVQ enables high-fidelity compression, it may quantize noise. Using

UTMOS ↑ Intelligibility ↓ Similarity ↑

Sample 4.280 0.739 0.539 NoSample 3.468 1.891 0.613

TABLE III: Influence of VAE sampling during training

-0.2cm

continuous representations can be more robust to noise, as continuous models scale the noise according to its magnitude.

Future work. Our approach could be extended by developing multimodal models capable of both perception and generation. Additionally, exploring generation stopping conditions that do not rely on discrete representations would be valuable. Adaptive inference strategies, such as adjusting the number of diffusion steps per token (e.g., using more steps for early tokens), could improve efficiency. Finally, developing quality metrics for the diffusion process could enable the use of decoding algorithms like beam search during inference.

Limitations The diffusion head inference process is slower than the RVQ prediction heads, as it requires an iterative process for the token sampling. Moreover, it does not allow to measure likelihood or confidence, which can be useful for decoding algorithms such as beam search or confidence based unmasking. Optimal balancing of the discrete and continuous losses in SALADis not easy to obtain. During training, the gradients of the discrete semantic loss increase, while the gradients of the continuous diffusion loss decrease.

Ethical Statement SALAD is capable of zero-shot voice cloning, which presents potential risks, including misuse for voice spoofing. To mitigate these risks, it is essential to implement strict protocols that verify and ensure the speaker’s consent and authorization before their voice is used in any realworld application, especially when dealing with previously unseen speakers. Safeguards must be in place to prevent unauthorized use and protect individuals’ privacy and identity.

AI Tools Usage AI-based tools may have assisted in code writing and paraphrasing, but all content was thoroughly reviewed by the authors and used only as a support tool.

ACKNOWLEDGMENT

This research work was partially supported by Mobileye Academic Grant Program and by the Israel Innovation Authority, grant number 78563.

REFERENCES

- [1] A. Van Den Oord, O. Vinyals, et al., “Neural discrete representation learning,” Advances in neural information processing systems, vol. 30, 2017.
- [2] P. Esser, R. Rombach, and B. Ommer, “Taming transformers for high-resolution image synthesis,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 12873–12883.
- [3] A. Ramesh et al., “Zero-shot text-to-image generation,” in International conference on machine learning, Pmlr, 2021, pp. 8821–8831.
- [4] O. Gafni, A. Polyak, O. Ashual, S. Sheynin, D. Parikh, and Y. Taigman, “Make-a-scene: Scene-based text-toimage generation with human priors,” in European Conference on Computer Vision, Springer, 2022, pp. 89– 106.
- [5] N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi, “Soundstream: An end-to-end neural audio codec,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 495–507, 2021.
- [6] A. D´efossez, J. Copet, G. Synnaeve, and Y. Adi, “High fidelity neural audio compression,” arXiv preprint arXiv:2210.13438, 2022.
- [7] C. Wang et al., “Neural codec language models are zero-shot text to speech synthesizers,” arXiv preprint arXiv:2301.02111, 2023.
- [8] J. Copet et al., “Simple and controllable music generation,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [9] S. Chen et al., “Vall-e 2: Neural codec language models are human parity zero-shot text to speech synthesizers,” arXiv preprint arXiv:2406.05370, 2024.
- [10] Z. Zhang et al., “Speak foreign languages with your own voice: Cross-lingual neural codec language modeling,” arXiv preprint arXiv:2303.03926, 2023.
- [11] K. C. Puvvada, N. R. Koluguri, K. Dhawan, J. Balam, and B. Ginsburg, “Discrete audio representation as an alternative to mel-spectrograms for speaker and speech recognition,” in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), IEEE, 2024, pp. 12111–12115.
- [12] M. Tschannen, C. Eastwood, and F. Mentzer, “Givt: Generative infinite-vocabulary transformers,” arXiv preprint arXiv:2312.02116, 2023.
- [13] T. Li, Y. Tian, H. Li, M. Deng, and K. He, “Autoregressive image generation without vector quantization,” arXiv preprint arXiv:2406.11838, 2024.
- [14] L. Meng et al., “Autoregressive speech synthesis without vector quantization,” arXiv preprint arXiv:2407.08551, 2024.
- [15] W. Lin and C. HE, “Continuous autoregressive modeling with stochastic monotonic alignment for speech synthesis,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=cuFzE8Jlvb

- [16] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “Hubert: Selfsupervised speech representation learning by masked prediction of hidden units,” IEEE/ACM transactions on audio, speech, and language processing, vol. 29, pp. 3451–3460, 2021.
- [17] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “Wav2vec 2.0: A framework for self-supervised learning of speech representations,” Advances in neural information processing systems, vol. 33, pp. 12449– 12460, 2020.
- [18] Y.-A. Chung et al., “W2v-bert: Combining contrastive learning and masked language modeling for selfsupervised speech pre-training,” in 2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), IEEE, 2021, pp. 244–250.
- [19] E. Casanova et al., “Xtts: A massively multilingual zero-shot text-to-speech model,” arXiv preprint arXiv:2406.04904, 2024.
- [20] P. Peng, P.-Y. Huang, D. Li, A. Mohamed, and D. Harwath, “Voicecraft: Zero-shot speech editing and text-tospeech in the wild,” arXiv preprint arXiv:2403.16973, 2024.
- [21] M. Le et al., “Voicebox: Text-guided multilingual universal speech generation at scale,” Advances in neural information processing systems, vol. 36, 2024.
- [22] K. Shen et al., “Naturalspeech 2: Latent diffusion models are natural and zero-shot speech and singing synthesizers,” arXiv preprint arXiv:2304.09116, 2023.
- [23] M. Łajszczak et al., “Base tts: Lessons from building a billion-parameter text-to-speech model on 100k hours of data,” arXiv preprint arXiv:2402.08093, 2024.
- [24] E. Kharitonov et al., Speak, read and prompt: Highfidelity text-to-speech with minimal supervision, 2023. arXiv: 2302 . 03540 [cs.SD]. [Online]. Available: https://doi.org/10.48550/arXiv.2302.03540
- [25] R. Huang et al., “Make-a-voice: Unified voice synthesis with discrete representation,” arXiv preprint arXiv:2305.19269, 2023.
- [26] Z. Borsos, M. Sharifi, D. Vincent, E. Kharitonov, N. Zeghidour, and M. Tagliasacchi, “Soundstorm: Efficient parallel audio generation,” arXiv preprint arXiv:2305.09636, 2023.
- [27] Z. Borsos et al., AudioLM: A language modeling approach to audio generation, 2023. arXiv: 2209.03143 [cs.SD]. [Online]. Available: https : / / doi . org / 10 . 48550/arXiv.2209.03143
- [28] P. K. Rubenstein et al., “Audiopalm: A large language model that can speak and listen,” arXiv preprint arXiv:2306.12925, 2023.
- [29] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.
- [30] Z. Kong, W. Ping, J. Huang, K. Zhao, and B. Catanzaro, “Diffwave: A versatile diffusion model for audio synthesis,” arXiv preprint arXiv:2009.09761, 2020.

- [31] N. Chen, Y. Zhang, H. Zen, R. J. Weiss, M. Norouzi, and W. Chan, “Wavegrad: Estimating gradients for waveform generation,” arXiv preprint arXiv:2009.00713, 2020.
- [32] V. Popov, I. Vovk, V. Gogoryan, T. Sadekova, and M. Kudinov, “Grad-tts: A diffusion probabilistic model for text-to-speech,” in International Conference on Machine Learning, PMLR, 2021, pp. 8599–8608.
- [33] V. Pratap, Q. Xu, A. Sriram, G. Synnaeve, and R. Collobert, “Mls: A large-scale multilingual dataset for speech research,” arXiv preprint arXiv:2012.03411, 2020.
- [34] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Librispeech: An asr corpus based on public domain audio books,” in 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP), IEEE, 2015, pp. 5206–5210.
- [35] S. Shechtman and A. Dekel, “Low bitrate high-quality rvqgan-based discrete speech tokenizer,” in Interspeech 2024, 2024, pp. 4174–4178. DOI: 10.21437/Interspeech. 2024-2366
- [36] R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, and K. Kumar, “High-fidelity audio compression with improved rvqgan,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [37] L. Barrault et al., “Seamless: Multilingual expressive and streaming speech translation,” arXiv preprint arXiv:2312.05187, 2023.
- [38] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv preprint arXiv:2207.12598, 2022.
- [39] T. Saeki, D. Xin, W. Nakata, T. Koriyama, S. Takamichi, and H. Saruwatari, “Utmos: Utokyo-sarulab system for voicemos challenge 2022,” arXiv preprint arXiv:2204.02152, 2022.
- [40] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in International conference on machine learning, PMLR, 2023, pp. 28492–28518.
- [41] S. Chen et al., “Wavlm: Large-scale self-supervised pretraining for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–1518, 2022. DOI: 10 . 1109 / JSTSP. 2022 . 3188113
- [42] I. Sodr´e and F. Brasileiro, “An analysis of the use of qualifications on the amazon mechanical turk online labor market,” Computer Supported Cooperative Work (CSCW), vol. 26, pp. 837–872, 2017.
- [43] F. Ribeiro, D. Florˆencio, C. Zhang, and M. Seltzer, “Crowdmos: An approach for crowdsourcing mean opinion score studies,” in 2011 IEEE international conference on acoustics, speech and signal processing (ICASSP), IEEE, 2011, pp. 2416–2419.
- [44] M. Wester, Z. Wu, and J. Yamagishi, “Analysis of the voice conversion challenge 2016 evaluation results,” in

- Interspeech 2016, 2016, pp. 1637–1641. DOI: 10.21437/ Interspeech.2016-1331
- [45] Z. Kons, S. Shechtman, A. Sorin, R. Hoory, C. Rabinovitz, and E. D. S. Morais, “Neural tts voice conversion,” in 2018 IEEE Spoken Language Technology Workshop (SLT), IEEE, 2018, pp. 290–296.
- [46] F. Wilcoxon, “Individual comparisons by ranking methods. biom. bull., 1, 80–83,” 1945.
- [47] A. W. Rix, J. G. Beerends, M. P. Hollier, and A. P. Hekstra, “Perceptual evaluation of speech quality (pesq)-a new method for speech quality assessment of telephone networks and codecs,” in 2001 IEEE international conference on acoustics, speech, and signal processing. Proceedings (Cat. No. 01CH37221), IEEE, vol. 2, 2001, pp. 749–752.

