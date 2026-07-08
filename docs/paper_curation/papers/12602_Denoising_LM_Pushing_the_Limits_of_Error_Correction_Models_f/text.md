arXiv:2405.15216v2[cs.LG]16Mar2026

# Revisiting ASR Error Correction with Specialized Models

###### Zijin Gu, Tatiana Likhomanenko, Richard He Bai, Erik McDermott, Ronan Collobert, Navdeep Jaitly† Apple, †Google; work done at Apple

Language models play a central role in automatic speech recognition (ASR), yet most methods rely on text-only models unaware of ASR error patterns. Recently, large language models (LLMs) have been applied to ASR correction, but introduce latency and hallucination concerns. We revisit ASR error correction with compact seq2seq models, trained on ASR errors from real and synthetic audio. To scale training, we construct synthetic corpora via cascaded TTS and ASR, finding that matching the diversity of realistic error distributions is key. We propose correction-first decoding, where the correction model generates candidates rescored using ASR acoustic scores. With 15x fewer parameters than LLMs, our modelachieves1.5/3.3%WERonLibriSpeechtest-clean/other, outperformsLLMs, generalizesacrossASRarchitectures (CTC, Seq2seq, Transducer) and diverse domains, and provides precise corrections in the low-error regime where LLMs struggle.

Correspondence: Zijin Gu: zijin@apple.com Date: March 18, 2026

## 1 Introduction

Language models are a key component of modern automatic speech recognition (ASR) systems, providing linguistic constraints that complement acoustic modeling Synnaeve et al. (2020). In conventional pipelines, this interaction is typically realized through shallow fusion, rescoring, or related heuristics that combine acoustic scores with probabilities from text-only language models Chorowski and Jaitly (2016); Kannan et al. (2018). While effective, these approaches rely on language models that are trained independently of ASR systems and are therefore unaware of the error patterns produced during decoding, often requiring careful tuning and increased computational cost.

Recent advances in large language models (LLMs) have renewed interest in post-hoc ASR correction Chen et al. (2023); Yang et al. (2023); Hu et al. (2024); Fathullah et al. (2024); Radhakrishnan et al. (2023); Fang et al. (2025); Liu et al. (2025), demonstrating that strong generative models can fix recognition errors using powerful linguistic priors. However, LLM-based correction introduces practical challenges, including high inference latency, limited controllability, and a tendency to hallucinate corrections that diverge from the acoustic evidence Hu et al. (2024); Fang et al. (2025); Liu et al. (2025). Moreover, LLMs are fundamentally ill-suited for correcting already-accurate ASR systems: reducing WER from 30% to 20% requires coarse linguistic fixes that LLMs can provide, but reducing WER from 5% to 3% demands precise, character-level corrections that preserve acoustic evidence—a regime where LLMs struggle and often degrade performance through over-correction Wu et al. (2023); Yang et al. (2023). This raises a fundamental question: do specialized error correction models provide a better accuracy–efficiency trade-off than generic large language models for ASR error correction?

In this work, we revisit ASR error correction with specialized language models. Rather than relying on language models trained only on clean text, which are agnostic to ASR error patterns, or on generic LLMs for post-hoc correction, we propose to explicitly learn the conditional distribution that maps corrupted ASR hypotheses to clean transcripts. We study error correction language models (ECLMs), compact conditional sequence-to-sequence models trained specifically for this purpose. This separation offers a key advantage: the

acoustic model can focus purely on mapping speech to plausible token sequences, while the ECLM learns to correct the systematic errors that acoustic models make—without requiring joint optimization or heuristic score combination.

A central challenge in training such models is the scarcity of paired ASR hypotheses and reference transcripts. To overcome this limitation, we construct large-scale training data by cascading text-to-speech (TTS) systems with ASR models Guo et al. (2019); Hu et al. (2022), supplemented with real ASR hypotheses from labeled audio, enabling the generation of arbitrarily large noisy-clean text pairs that capture realistic error distributions. Through systematic ablations, we find that the effectiveness of ECLMs depends less on the perceptual quality of synthetic speech and more on matching the diversity and structure of realistic ASR error distributions. We achieve this through multi-speaker synthesis, noise augmentation, and mixing synthetic and real data sources. Crucially, this decoupled approach scales efficiently: text corpora can be expanded independently of acoustic data, and a single ECLM transfers across different ASR architectures without retraining.

Beyond model training, we also revisit decoding. In conventional approaches, ASR generates an n-best list using beam search, e.g., with an n-gram LM, and a neural LM then rescores these hypotheses. Rather than rescoring ASR hypotheses with a language model, we propose correction-first decoding, in which the ECLM generates candidate corrections that are subsequently evaluated using acoustic scores from the ASR system. This approach enables interaction between acoustic and linguistic evidence without relying on expensive ASR beam sizes and language-model rescoring.

Our main contributions are as follows:

- • We demonstrate that our ECLM achieves state-of-the-art WER on LibriSpeech (1.5% test-clean, 3.3% test-other) without using external audio data, matching self-supervised approaches that use 60k hours of unlabeled audio, while outperforming both off-the-shelf and fine-tuned LLMs with 15× fewer parameters, lower latency, and no hallucination.
- • We propose correction-first decoding, which outperforms conventional neural LM rescoring while offering comparable computational cost.
- • We demonstrate that ECLMs generalize across ASR architectures (QuartzNet, Conformer, Whisper, Parakeet-TDT) spanning CTC, seq2seq, and transducer paradigms, and across diverse domains, reducing WER by 13% relative on average across eight datasets spanning conversational and prepared speech.
- • Through systematic ablations, we identify key ingredients for effective training data: multi-speaker TTS, noise augmentation, mixing synthetic and real data, and—counterintuitively—noisier TTS systems outperform high-quality ones.

Our results show that in the low-error regime where LLMs overcorrect and hallucinate, compact specialized models that learn the ASR error distribution offer a superior alternative to both neural LM rescoring and generic LLMs.

## 2 Related work

LM integration for ASR Finding better techniques to leverage LMs most effectively with acoustic models has been a long standing research problem Veselý et al. (2013); Toshniwal et al. (2018). Earlier techniques back-propagated errors from LMs into ASRs using sequence discriminative criterion (e.g. Maximum Mutual Information) Kingsbury (2009); Jaitly et al. (2012); Veselý et al. (2013). Later approaches attempted to merge features of text-only LMs with features of ASRs in either the shallower or deeper layers of the model Gulcehre

- et al. (2015); Sriram et al. (2018); Kannan et al. (2018). Another line of work Meng et al. (2021); McDermott et al. (2019); Variani et al. (2020) attempts to subtract the language model learned in the end-to-end ASR models, while integrating external LMs.

Error correction models Error correction models post-process outputs from an ASR system to fix errors Guo et al. (2019); Shivakumar et al. (2019); Tanaka et al. (2018); Zhang et al. (2019). These approaches

Text y TTS Synth.Audio ASR y

(a)

(y, y) ECLM

Real Audio

ASR y

(b)

Text y

- Figure 1 Data generation pipeline for training the error correction model. Noisy hypotheses y˜ are obtained by running ASR on either (a) synthetic audio generated via multi-speaker TTS from a text corpus, or (b) real audio. These are paired with clean references y to create training examples (˜y, y) for the ECLM.

typically convert first-pass ASR hypotheses into cleaned up text. Even with just one best hypothesis from the ASR model, Transformer-based error correction models have shown improvements over the baseline acoustic models and acoustic models with n-gram language models Hrinchuk et al. (2020); Zhao et al. (2021). Further improvements can be made by using n-best ASR hypotheses as inputs to the clean up model as this provides more information Leng et al. (2021); Ma et al. (2023); by using more advanced WER based metrics, instead of cross-entropy for model optimization Hori et al. (2016); or by using different variants that incorporate more compact inputs, such as phonemes Wang et al. (2020); Dutta et al. (2022) and word lattices Ma et al. (2020); Dai et al. (2022). However, the efficacy of these approaches has been limited by the scarcity of paired ASR output and ground truth transcriptions, and thus have not been able to outperform conventional neural

LM integration. As a result most approaches start with a pretrained language model, e.g., BART Lewis et al. (2020), that is finetuned with the limited noisy ASR data Zhao et al. (2021); Dutta et al. (2022); Ma et al.

- (2023). Others incorporate different data augmentation strategies, such as SpecAugment Ma et al. (2023) and dropout Hrinchuk et al. (2020). While these are helpful, the improvement may be limited by the data size. Guo et al. (2019) have attempted to resolve this problem by using synthetic data for training error correction models on top of listen, attend and spell (LAS) Chan et al. (2015) models. They show that error correction can improve results when combined with neural LMs, but error correction by itself is inferior to neural LM integration.

LLM-based error correction More recently, large language models (LLMs) have been explored for ASR error correction with promising results. HyPoradise Chen et al. (2023) proposed using LLMs with hypothesesto-transcription prompting, while Yang et al. (2023) showed that LLMs can correct ASR errors through incontext learning and task-activating prompts. Hu et al. (2024) demonstrated that LLMs are efficient learners for noise-robust speech recognition with minimal finetuning. Other works have explored prompting LLMs with speech features directly Fathullah et al. (2024); Wu et al. (2023) or combining Whisper with LLaMA for cross-modal correction Radhakrishnan et al. (2023). While LLM-based methods show promise, they have notable limitations for practical deployment: (i) Latency: API-based LLMs introduce network latency unsuitable for real-time ASR; (ii) Cost: per-token pricing makes large-scale transcription expensive; (iii) Reliability: proprietary models may change or be discontinued, creating dependency risks and making results hard to reproduce; (iv) Offline deployment: many applications require on-device processing without internet access. In contrast, our ECLM is a compact (< 1B parameters), self-contained model that can be deployed offline while being specifically trained for error correction. More fundamentally, LLM-based methods excel at correcting linguistic errors (grammar, word choice, semantic coherence) but struggle with acoustic errors (phonetic confusions, homophones, character-level mistakes) because they lack knowledge of the ASR’s error distribution. Our ECLM is trained on TTS-generated errors that mimic the ASR’s actual mistake patterns, enabling fine-grained character-level corrections that generic LLMs cannot achieve.

## 3 Error correction for speech recognition

### 3.1 Probabilistic formulation

To motivate the error correction approach and our variant of it, we consider a probabilistic model pcorr(y|x), which we call correction-based speech recognition. This is a cascade of two stochastic, discrete models – an ASR model which produces sequences y˜ of discrete tokens from audio input x with probability pASR(˜y|x), and an ECLM which transforms an input sequence, y˜, into an output sequence y with probability pECLM(y|y˜). Here y˜ denotes any ASR hypothesis and yˆ = arg maxy˜ pASR(˜y|x) denotes the greedy hypothesis. Under this model

 

 . (3.1)

log pcorr(y|x) = log

pECLM(y|y˜)pASR(˜y|x)

y ˜

Most error correction approaches optimize log pECLM(y|yˆ), where yˆ = arg max

pASR(˜y|x), (3.2)

y˜

which can be viewed as an approximation to the lower bound of Equation (3.1) log pcorr(y|x) ≥

pASR(˜y|x)log pECLM(y|y˜) (3.3)

y ˜

under the assumption that the ASR model is sharply peaked on a single transcript (the inequality comes from applying Jensen’s inequality). Note, however, that using samples from the posterior of the model, pcorr(˜y|x,y), would be the correct distribution to optimize the above model on, but we (and prior works on error correction) do not follow that approach here.

### 3.2 Data generation

We would like to optimize Equation (3.2) in order to improve log pcorr. This requires paired data {(ˆy,y)} where yˆ is a corrupted ASR hypothesis and y is the clean reference. Unlike conventional LMs which model clean text alone, ECLMs are conditional models whose input distribution must reflect realistic ASR errors. The central challenge is therefore constructing a training distribution that is close to the true error distribution encountered at test time. We address this through three complementary strategies: synthetic data generation via TTS, mixing in real ASR hypotheses, and noise augmentation.

Synthetic data via TTS Given a large text corpus with distribution p(y), we generate paired data by cascading a TTS system with an ASR system (see Figure 1(a)). For each sentence y, the TTS system synthesizes audio x˜, and the ASR system produces a hypothesis yˆ from x˜. The resulting dataset {(ˆy,y)} is used to train the ECLM. This approach captures the biases of the ASR system—phonetic confusions, insertion/deletion patterns, and character-level errors—more faithfully than random text corruptions Norouzi

- et al. (2016).1

Mixing real and synthetic data While TTS-based generation can scale to arbitrarily large text corpora, the resulting error distribution may not perfectly match that of real speech. We therefore mix a small proportion of real ASR hypotheses (obtained by running the ASR on labeled audio) into the synthetic training data. This grounds the ECLM’s error distribution more closely to real conditions and provides complementary error patterns that TTS alone may not produce.

Noise augmentation To further diversify the error distribution, we apply two forms of augmentation: (1) random character substitution in the input transcript, and (2) frequency masking of the spectrogram before ASR inference. Both strategies increase the variety of errors without shifting the distribution away from realistic ASR mistakes.

We ablate all of these factors—TTS system choice, real data mixing, noise augmentation, and scaling—in Sections 5.4.1–5.4.3.

1As initial experiments we also tried carefully designed by hand corruptions, e.g. n-gram modifications of the text which did not work in practice.

- 3.3 Decoding techniques At inference time, given audio x, we seek the transcript y∗ that maximizes

y∗ = arg max

y

y ˜

pECLM(y|y˜)pASR(˜y|x). (3.4)

Since the exact optimization is intractable, we consider two approximations: greedy decoding, which requires no access to acoustic scores, and correction-first decoding, which reintroduces acoustic evidence through rescoring.

Greedy decoding The ECLM takes the greedy ASR hypothesis yˆ and decodes y∗ independently of acoustic scores2

y∗ = arg max

pASR(˜y|x), (3.5)

pECLM(y|yˆ); yˆ = arg max

y

y˜

where both maximizations are approximated by greedy search. This is the simplest form of error correction: it operates on text alone, without access to audio or ASR scores—a property not achievable with conventional LM rescoring.

Algorithm 1 Correction-first decoding Require: Audio x, ASR model pASR, ECLM pECLM, beam size B, weight λ Ensure: Corrected transcript y∗

- 1: yˆ ← arg maxy˜ pASR(˜y|x) ▷ ASR greedy decoding
- 2: Y ← BeamSearch(pECLM(·|yˆ),B) ▷ ECLM generates n-best
- 3: for each y ∈ Y do
- 4: Compute log pASR(y|x) ▷ ASR acoustic rescore
- 5: end for
- 6: y∗ ← arg maxy∈Y λlog pECLM(y|yˆ) + log pASR(y|x)

Correction-first decoding To reintroduce acoustic evidence, we propose correction-first decoding (Algorithm 1). The ECLM generates an n-best list of corrected candidates via beam search from the greedy ASR hypothesis yˆ, and each candidate is rescored using a weighted combination of the ECLM score and the ASR acoustic score, with a single hyperparameter λ tuned on the validation set. This approach assumes the correct transcript appears within the ECLM’s beam; in practice, a beam size of 10 suffices.

[Figure 1]

- Figure 2 Comparison of decoding approaches. (a) LM rescoring: ASR generates an n-best beam, e.g., with n-gram LM, then neural LM rescores. (b) Correction-first decoding: ASR produces a greedy hypothesis, ECLM generates candidates via beam search, and ASR rescores using acoustic likelihood.

2For CTC models, yˆ is obtained by taking the argmax over label probabilities at each frame, deduplicating, and removing blanks.

This reverses the conventional neural LM rescoring pipeline (Figure 2): in standard rescoring Synnaeve et al.

(2020), the ASR generates a beam (typically with a weak n-gram LM), which is rescored by the neural LM.3 In correction-first decoding, the ECLM generates the beam from a single greedy ASR output, and ASR scores are used for rescoring. We adopt nucleus sampling with threshold 0.9 to ensure full-length beams. The computational cost of both approaches is comparable, while ECLM greedy decoding is cheaper as it avoids beam search entirely.

## 4 Experimental details

### 4.1 Training data

Unless stated otherwise, all ECLMs are trained on data derived from the LibriSpeech Panayotov et al. (2015) LM corpus (40M sentences, 800M words). We synthesize audio from this corpus using three TTS systems: Tacotron-2 Shen et al. (2018) (single-speaker), YourTTS Casanova et al. (2022) (zero-shot multi-speaker), and RichTTS Bai et al. (2024) (zero-shot multi-speaker with D-vector speaker embeddings Variani et al. (2014)). For multi-speaker systems, we randomly select speakers from the LibriSpeech ASR training set. The synthesized audio is processed by a CTC-based ASR system Graves et al. (2006) to produce hypotheses, yielding paired data {(ˆy,y)} for ECLM training. We compare TTS systems in Section 5.4.2.

As discussed in Section 3, we also incorporate a small proportion (10%) of real ASR hypotheses obtained by running the baseline ASR on the LibriSpeech 960h training set. This grounds the error distribution in real acoustic conditions and complements the synthetic TTS data.

For the cross-domain generalization experiments (Section 5.3.2), we train a separate ECLM on ParakeetTDT-v2 NVIDIA NeMo (2024) transcriptions of the English subset of the Granary dataset Koluguri et al. (2025), a large-scale open-source collection of pseudo-labeled speech across 25 European languages sourced from the wild, of which around 120k hours are English.

### 4.2 ASR models

- Table 1 summarizes the ASR models used in our experiments. The primary model is a Transformer-based CTC encoder with 255M parameters, referred to as the baseline ASR throughout the paper. All CTC models (Transformer, Conformer Gulati et al. (2020), QuartzNet Kriman et al. (2020)) are trained on LibriSpeech 960h with a character vocabulary. Whisper models Radford et al. (2023) and Parakeet-TDT-v2 NVIDIA

- NeMo (2024) are pretrained on much larger proprietary audio corpora and use word-piece tokenization. Table 1 ASR models used in this work.

#### Model Decoder Params Training data

Transformer CTC 255M LS 960h Conformer CTC 102M LS 960h QuartzNet CTC 7M LS 960h Whisper-base Enc-Dec 74M proprietary Whisper-small Enc-Dec 244M proprietary Parakeet-TDT-v2 TDT 0.6B Granary

The baseline ASR (Transformer CTC, 255M) consists of a 1D convolution (kernel 7, stride 3), sinusoidal positional embedding, and 36 pre-LayerNorm transformer blocks (embedding dim 768, 4 heads, MLP dim 3072, dropout 0.1, layer drop 0.1). The Conformer (102M) uses 16 conformer blocks (embedding dim 512, 4 heads, MLP dim 2048, dropout 0.1). All acoustic models use 80-channel log-mel filterbanks (25ms window, 10ms stride) with SpecAugment Park et al. (2019) (2 frequency masks, max width 30; 10 time masks, max width 50,

3Joint decoding with ASR and neural LM is also possible but more expensive, and in practice the oracle WER of the first-pass n-gram beam is already very low.

ratio 0.1). Models are trained with AdamW (weight decay 1e-6) and gradient clipping of 1.0/0.5/1.0 for Transformer/Conformer/QuartzNet, with peak learning rates of 0.001/0.002/0.0022 and warmup of 64k/10k/64k steps respectively. Training continues until greedy WER plateaus on dev-clean and dev-other.

### 4.3 Language models

ECLM The ECLM is a Transformer encoder-decoder trained with cross-entropy loss and a character vocabulary, with 16 encoder layers and 4 decoder layers (dropout and layer drop 0.1, sinusoidal positional embedding). We vary the embedding and MLP dimensions for different model sizes: 512/2048 (69M), 768/3072 (155M), and 1280/6144 (484M), following Synnaeve et al. (2020). The training data is generated from LibriSpeech LM text corpus by different TTS systems but from the baseline ASR model. ECLMs are trained with a dynamic batch size of 160k tokens, AdamW optimizer (weight decay 0.01, gradient clipping 0.1), learning rate warmup of 64k steps, followed by a constant learning rate of 0.001 for 300k steps, then step decay (rate 0.5, step size 200k) until greedy WER plateaus on dev-clean and dev-other. The same training hyperparameters are used across all experiments, varying only model size and training data.

Neural LM The neural LM is a Transformer decoder-only model with 20 layers, trained with cross-entropy loss and a 10K word-piece vocabulary on the LibriSpeech LM text corpus. It shares the same embedding and MLP dimensions with the ECLM. Training uses AdamW (weight decay 1e−8) with a cosine learning rate schedule (peak 0.001, 500k steps, 16k warmup). Perplexity for 69M, 155M, and 484M LMs is 34.45, 31.49, and 30.9 respectively. For neural LM rescoring (Figure 2a), we generate an n-best list from CTC beam-search decoding with a 4-gram word-level LM (top-200k words), and rescore using neural LM and CTC ASR scores, similar to Synnaeve et al. (2020).

LLM baselines To benchmark against generic LLMs, we evaluate three instruction-tuned models for ASR error correction: Llama-3.1-8B-Instruct Grattafiori et al. (2024), Llama-3.1-70B-Instruct Grattafiori et al.

- (2024), and Mistral-7B-Instruct-v0.3 Jiang et al. (2023). All models are used off-the-shelf without any finetuning on ASR data. We evaluate in both zero-shot and 5-shot settings, where the few-shot examples are drawn from the LibriSpeech dev set. The prompt template is shown in Figure 3. Inference is performed with greedy decoding (temperature 0) on a single H100 GPU.

|System prompt: You are an ASR error correction system. Correct any transcription errors in the following text. Only fix errors–-do not add, remove, or rephrase content. Output only the corrected transcript. Few-shot user prompt: Correct ASR transcription errors. Examples: Input: “he went too the see yesterday” Output: “he went to the sea yesterday” Input: “their our many problems with this” Output: “there are many problems with this” Input: “the whether is nice today” Output: “the weather is nice today” Input: “she could of done better” Output: “she could have done better” Input: “i need to by some food” Output: “i need to buy some food” Input: “{hypothesis}” Output:|
|---|

- Figure 3 Prompt template used for LLM-based ASR error correction (5-shot). The system prompt instructs conservative correction; the few-shot examples demonstrate common ASR error types (homophones, phonetic confusions). Zero-shot uses the system prompt only. Compute All models are trained on A100 with 80GB memory with 1 node of 8 GPUs for 3-5 days. Some

models we train with larger batch size which results in longer training with gradient accumulation. For data generation by YourTTS model we use CPU inference and it takes 100k CPU hours (as it is not batched version of public code). With RichTTS (116M), we spent 12,800 GPU hours to synthesize 40M audios (roughly 137k hours of data). Correction-first decoding with beam of 64 is sufficient, while the beam of 10 gives similar results within 0.1% variation in WER, which takes about 10 mins to finish on 8 A100 GPU for dev sets.

## 5 Results

We organize our experiments to answer five questions: (1) How does our ECLM compare to neural LM rescoring and LLM-based correction? (2) What types of errors can ECLMs fix that LLMs cannot? (3) Does a single ECLM generalize across ASR architectures and domains? (4) What are the key ingredients for effective synthetic training data? (5) How does performance scale with model size, data size, and speaker diversity?

We use LibriSpeech (LS) as the primary benchmark throughout, as it is the most widely used evaluation set for ASR, enables direct comparison with prior work, and provides a large text-only LM corpus (800M words) that can be used for synthetic data generation via TTS. Importantly, modern ASR systems already achieve very low WER on LibriSpeech, making it a challenging testbed for error correction: reducing WER from 5% to 3% requires precise, character-level fixes rather than coarse linguistic corrections, and any overcorrection risks degrading already-accurate transcriptions. We then evaluate generalization to other architectures and domains in Section 5.3. All WER scores are computed after applying the Whisper text normalizer Radford et al. (2023) for consistent evaluation across models.

### 5.1 Main results

##### Table 2 compares neural LM rescoring, LLM-based correction, and our ECLM on LibriSpeech test set.

- Table 2 LibriSpeech test sets WER (%) comparing neural LM rescoring, LLM-based correction, and our ECLM. Latency measured on a single H100 GPU.

Model Params LS-clean LS-other Latency (ms) Halluc. (%) baseline ASR – 2.2 5.3 – – Neural LM

+ NeLM rescoring 0.5B 2.0 4.1 – – Generic LLMs

+ Mistral

7B 0-shot 32.0 37.0 579 11.5 5-shot 20.0 25.8 774 6.6

+ Llama

8B

0-shot 16.7 23.3 295 4.9 5-shot 7.1 13.4 243 3.1

+ Llama

70B

0-shot 8.8 13.0 1202 4.6 5-shot 19.3 19.5 1767 4.9

Specialized LMs

+ ECLM (greedy) 0.5B 2.0 4.1 44 0 + ECLM (corr.-first) 0.5B 1.6 3.6 457 0

ECLM vs. neural LM rescoring Our ECLM with greedy decoding achieves comparable WER to neural LM rescoring, despite not requiring beam search from the ASR. With correction-first decoding, our model significantly outperforms neural LM rescoring (1.6%/3.6% vs. 2.0%/4.1%), achieving 20% and 12% relative WER reduction on LS-clean and LS-other respectively.

ECLM vs. generic LLMs We evaluate LLMs using the prompt template shown in Figure 3. All LLMs degrade the baseline ASR, even the best result—Llama-70B zero-shot (8.8%/13.0%)—which has 140× more parameters than our ECLM. Scaling from 8B to 70B yields only modest WER improvement at 4–6× higher latency (1202 ms vs. 243–295 ms), suggesting that LLMs do not scale efficiently for error correction in the lowWER regime. Few-shot prompting helps smaller models (Llama-8B 5-shot: 7.1%/13.4%) but hurts the 70B model (19.3%/19.5%), likely due to over-reliance on example patterns. We also fine-tuned Mistral-7B-v0.1 with both LoRA and full fine-tuning on our training data, where full fine-tuning achieving 2.0%/4.8% WERstill worse than our ECLM with greedy (2.0%/4.1%) and correction-first decoding (1.6%/3.6%), confirming that architecture matters more than simply adapting a generic LLM. Our ECLM runs at 44 ms—5–40× faster than LLMs. More critically, all LLMs exhibit high hallucination rates (3–12%), whereas our ECLM produces zero hallucinations. We define hallucination rate as the percentage of words in the model’s output that appear in neither the ASR hypothesis nor the reference transcript, measuring words fabricated without acoustic or textual support.

Acoustic error: ECLM corrects, LLM reverses meaning

LLM hallucination: injects world knowledge

ASR: in my opinion the are oul criminals ECLM: in my opinion they are all criminals LLM: in my opinion there are no criminals Ref: in my opinion they are all criminals

ASR: who were john peter judas and mary ECLM: who were john peter judas and mary LLM: who were john the baptist and mary Ref: who were john peter judas and mary

Acoustic error: ECLM corrects, LLM picks wrong word

LLM overcorrection: changes correct output

ASR: she baised her face with a pathetic look ECLM: she raised her face with a pathetic look LLM: she biased her face with a pathetic look Ref: she raised her face with a pathetic look

ASR: her mother went to hide ECLM: her mother went to hide LLM: her mother went to shop Ref: her mother went to hide

Word boundary error: ECLM corrects, LLM rewrites

Simple spelling: both ECLM and LLM correct

ASR: margaret was all o stunned ECLM: margaret was almost stunned LLM: margaret was all aghast Ref: margaret was almost stunned

ASR: what is needed for succiss in warfare ECLM: what is needed for success in warfare LLM: what is needed for success in warfare Ref: what is needed for success in warfare

ASR Error Correct Fix LLM Error

Figure 4 Qualitative comparison on dev-other. Red: ASR errors, blue: correct ECLM fixes, orange: LLM errors.

### 5.2 Qualitative analysis: why do LLMs struggle 

To understand why LLMs underperform despite their strong language modeling capabilities, we examine error types qualitatively. We categorize ASR errors into: (1) acoustic errors—phonetic confusions and homophones where the incorrect word sounds similar to the correct one; and (2) linguistic errors—grammar and word choice issues unrelated to sound.

- Figure 4 shows representative examples. LLMs struggle with acoustic errors: “baised”→“raised” requires knowledge of ASR error patterns, but the LLM picks the linguistically plausible “biased” instead. Similarly, the LLM rewrites “all o stunned” as “all aghast” rather than recovering the correct word boundary “almost stunned”. More dangerously, LLMs can reverse meaning entirely—correcting “the are oul criminals” to “there are no criminals” instead of “they are all criminals”. Our ECLM, trained on synthetic and real ASR errors that capture actual error distributions, learns these acoustic confusion patterns and makes precise character-level corrections.

Most critically, LLMs hallucinate on already-correct ASR output. They inject world knowledge (“john peter judas”→“john the baptist”), or substitute plausible but incorrect words (“hide”→“shop”). In the low-WER regime where remaining errors are subtle phonetic confusions, LLMs over-correct while our ECLM makes conservative, targeted fixes.

### 5.3 Generalization across architectures and domains

- 5.3.1 Generalization across ASR architectures

Table 3 LibriSpeech test WER (%) across ASR architectures. The same ECLM and neural LM from Table 2 are used without retraining.

Model Params LS-clean LS-other QuartzNet-CTC

7M

6.5 16.9

+ neLM (rescoring) 2.9 8.6 + ECLM (greedy) 2.8 8.3 + ECLM (corr.-first) 2.3 7.1

Conformer-CTC

102M

2.6 5.6 + neLM (rescoring) 2.2 4.2 + ECLM (greedy) 2.2 4.1 + ECLM (corr.-first) 1.7 3.6

Whisper-base

74M

4.5 10.9

+ ECLM (greedy) 3.1 7.9 Whisper-small

244M

3.3 7.6 + ECLM (greedy) 2.7 6.1

To test whether a single ECLM trained on noise from the baseline ASR transfers to other ASR architectures and sizes, we apply the same ECLM (0.5B) to QuartzNet (7M), Conformer (102M), and Whisper (base/small) without any retraining. Table 3 shows that the ECLM consistently improves all ASR models, with correction-first decoding outperforming neural LM rescoring across architectures. Notably, the ECLM improves Whisper—an encoder-decoder model trained on proprietary data with word-piece tokenizationdespite being trained only on CTC-based ASR errors with character tokenization, demonstrating that error patterns generalize across architectures.

- 5.3.2 Generalization across domains and datasets

To evaluate whether the proposed error correction approach generalizes beyond LibriSpeech, we train an ECLM (0.5B) on Parakeet-TDT-v2 NVIDIA NeMo (2024) transcriptions of the English subset of the Granary dataset Koluguri et al. (2025). Both Parakeet hypotheses and Granary labels are lowercased and stripped of punctuation (except hyphens) before training and evaluation, to prevent the ECLM from spending capacity on trivial casing and punctuation differences rather than learning meaningful error corrections. We then apply this ECLM to correct Parakeet outputs across eight diverse datasets spanning two categories: conversational and noisy speech (CHiME-6 Watanabe et al. (2020), CallHome Cieri et al. (2004), Switchboard Cieri et al. (2004), CommonVoice Ardila et al. (2019)), and read or prepared speech with already-low baseline WER (LibriSpeech Panayotov et al. (2015), VoxPopuli Wang et al. (2021), TED-LIUM 3 Hernandez et al. (2018)). This setting reflects real-world deployment, where a single correction model must generalize across domains, acoustic conditions, and speaker populations without per-dataset adaptation.

- Table 4 reflects several key results. On conversational and noisy datasets (CHiME-6, CallHome, Switchboard), where Parakeet produces frequent errors due to overlapping speech, disfluencies, and background noise, the ECLM provides substantial improvements: 24% relative reduction on CHiME-6, 13% on CallHome, and 7% on Switchboard. On prepared speech datasets (LibriSpeech, TED-LIUM, VoxPopuli), where Parakeet already performs well, the ECLM preserves performance without degradation.

The variation in gains across datasets is driven by multiple factors beyond ASR accuracy alone. First, the Granary training labels are pseudo-labeled by Whisper, so the ECLM’s correction ceiling is bounded by the quality of these labels—on domains where Whisper itself is less accurate, the ECLM has less room to improve. Second, the domain distribution matters: conversational datasets are acoustically closer to the diverse conditions in Granary, while prepared speech domains like LibriSpeech and TED-LIUM are underrepresented. Third, when the input hypothesis is already correct, the ECLM learns to pass it through

Table 4 WER (%) for Parakeet-TDT-v2 corrected by our ECLM (0.5B) across diverse datasets.

Dataset Parakeet + ECLM (greedy) Rel. ↓ Conversational / noisy CHiME-6 34.0 25.8 24% CallHome 15.5 13.5 13% Switchboard 11.1 10.3 7% CommonVoice (en) 10.1 10.0 2% Prepared / low-WER

VoxPopuli (en) 6.5 6.4 1% LS-clean 2.1 2.1 0% LS-other 4.0 4.0 0% TED-LIUM 3 3.9 3.9 0%

Average 10.9 9.5 13%

unchanged, avoiding the overcorrection problem that plagues LLMs. On average, the ECLM reduces WER from 10.9% to 9.5% (13% relative), demonstrating that a single ECLM can serve as a universal post-processor across ASR architectures, domains, and acoustic conditions.

### 5.4 Ablations

As argued in Section 3, the effectiveness of ECLMs depends critically on matching the training error distribution to real ASR errors. We systematically ablate three axes: the composition of training noise (Section 5.4.1), the role of TTS system choice and synthetic vs. real data (Section 5.4.2), and scaling behavior along model size, data size, and speaker diversity (Section 5.4.3). All ablations are evaluated on the LibriSpeech dev sets.

- 5.4.1 Noise distribution in training data

Table 5 traces the data construction path to our best results, showing the incremental effect of each strategy. Starting from 40M {(ˆy,y)} pairs generated by YourTTS, the baseline ECLM improves dev-other but slightly degrades dev-clean (row 2), suggesting the synthetic errors are too clean for the model to learn meaningful corrections. We progressively diversify the error distribution through three strategies, each carefully chosen to avoid shifting the training distribution too far from real ASR errors: (1) random character substitution at rate s=10% (row 4); (2) frequency masking of the spectrogram before ASR inference, similar to SpecAugment Park et al. (2019) (row 5); (3) mixing in 10% real ASR hypotheses from the LibriSpeech 960h training set to ground the distribution in real acoustic conditions (row 7). Additionally, combining data from two TTS systems (YourTTS + RichTTS) outperforms either system alone (row 6), as different TTS models produce complementary error patterns. Combining all strategies yields the best results (row 8).

- 5.4.2 Synthetic data analysis

TTS audio vs. real audio To assess whether synthetic data can substitute for real labeled audio, we compare ECLMs trained on TTS-generated hypotheses vs. hypotheses from ∼60k hours of real speech from Libriheavy Kang et al. (2023) (400M words), both decoded by the baseline ASR. For a fair comparison, we generate TTS data with RichTTS for the same sentences. As shown in Table 6, the gap is small: ECLMsynthetic is slightly worse on dev-clean (1.7% vs. 1.5%) but outperforms ECLM-real on dev-other (3.7% vs.

- 4.1%), suggesting that TTS-generated diversity compensates for the lack of real acoustic conditions. This validates the TTS-based pipeline as a scalable alternative to collecting labeled audio.

TTS quality vs. error diversity We evaluate the audio quality of each TTS system by measuring ASR WER on the synthesized speech (Table 7). Tacotron produces the cleanest audio (6% WER with Whispersmall), followed by YourTTS (10%) and RichTTS (12%). However, when training ECLMs on data from each system separately, Tacotron performs worst for error correction despite its higher audio quality (Table 5, row 2 vs. rows 3–4). This is because cleaner audio leads to fewer ASR errors in the training pairs, leaving the

Table 5 LibriSpeech dev WER (%) under different training data compositions. s: character substitution rate; FM: frequency masking.

Greedy Corr.-first Model clean other clean other baseline ASR 2.1 5.5 – –

+ ECLM, Tacotron + s=10% 4.1 6.6 2.0 4.8 + ECLM (YourTTS) 2.3 4.9 1.6 4.0

+ s=10% 2.2 4.6 1.5 3.8 + FM 2.3 4.6 1.6 3.7 + RichTTS 2.1 4.2 1.5 3.7 + FM + real 2.0 4.3 1.5 3.8 + RichTTS + FM + real 1.9 3.9 1.5 3.4

+ ECLM, RichTTS + s=10% 2.2 4.3 1.6 3.6

Table 6 LibriSpeech dev WER (%) for ECLMs trained on synthetic vs. real audio.

Greedy Corr.-first Model clean other clean other baseline ASR 2.1 5.5 – –

+ ECLM-synthetic 2.4 4.5 1.7 3.7 + ECLM-real 2.0 4.6 1.5 4.1

ECLM with mostly identity mappings and insufficient signal to learn meaningful corrections. Additionally, Tacotron is single-speaker, further limiting the diversity of error patterns compared to the multi-speaker YourTTS and RichTTS. This finding is counterintuitive but consistent with our central thesis: what matters for ECLM training is not TTS fidelity, but the richness and diversity of the resulting error distribution.

Table 7 LibriSpeech dev WER (%) on TTS-generated audio and resulting ECLM performance (greedy) when trained on each TTS system.

Whisper-small Baseline ASR ECLM (greedy) TTS System clean other clean other clean other

Tacotron 5.9 5.2 6.8 6.2 4.1 6.6 YourTTS 8.2 9.8 8.6 10.5 2.3 4.9 RichTTS 7.7 11.9 9.5 16.7 2.2 4.3

- 5.4.3 Scalability analysis

A key practical question is whether ECLMs follow predictable scaling behavior. We find that ECLMs benefit consistently from scaling along three axes (Figure 5).

Model size WER decreases consistently as model size increases from 69M to 1B for both greedy and correction-first decoding (Figure 5a), with the largest gains between 69M and 484M.

Text corpus size We construct three corpora by subsampling or augmenting the LibriSpeech LM corpus:

- 0.5x (400M words), 1x (800M words), and 2x (1.6B words, augmented with Gutenberg text Xu et al. (2020)). Larger corpora yield lower WER (Figure 5b), confirming that ECLMs benefit from greater textual diversity in training. Speaker diversity We train five ECLMs (155M) on data synthesized from 1, 10, 100, 1k, and 2k speakers.

###### (a) Model size

###### (b) Data size

###### (c) Speaker diversity

- 1

- 2

- 3

- 4

- 5

- 6

- 1

- 2

- 3

- 4

- 5

- 6

- 1

- 2

- 3

- 4

- 5

- 6

WER(%)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

69M 155M 484M 1B

400M (0.5x)

800M (1x)

1.6B (2x)

1 10 100 1k 2k

Model size

Number of speakers

Training data (words)

Greedy (dev-other)

Greedy (dev-clean)

Baseline (dev-other) Baseline (dev-clean)

Corr.-first (dev-other)

Corr.-first (dev-clean)

- Figure 5 LibriSpeech dev WER (%) as a function of model size, training data size, and speaker diversity. Solid lines: dev-other; dashed lines: dev-clean.

More speakers reduce WER, particularly for greedy decoding, by increasing the variability of ASR error patterns in the training data (Figure 5c). Gains saturate around 100 speakers on LibriSpeech dev sets; we use all available speakers for final experiments.

### 5.5 Comparison with prior work

A natural question is how our approach compares to systems that achieve low WER through fundamentally different strategies: larger architectures, self-supervised pretraining on unlabeled audio, or massive supervised training corpora. Table 8 places our best system in context. Following prior work showing that TTS data can improve ASR Hu et al. (2022), we also train a Transformer ASR variant with a mixture of LibriSpeech and TTS audio, denoted baseline ASR (LS+TTS). This stronger baseline with correction-first decoding achieves

- 1.5% and 3.3% WER on test-clean and test-other, respectively. Without using any external audio data, our results match wav2vec 2.0 and HuBERT models that leverage 60k hours of unlabeled audio for selfsupervised pretraining, and outperform Zipformer Yao et al. (2024), the strongest specialized model in the no-external-audio category. Our results are also competitive with state-of-the-art models from the Open ASR Leaderboard, including Granite Speech 3.3-8B Saon et al. (2025) (1.4%/2.9%), Canary-Qwen-2.5B NVIDIA

NeMo (2025) (1.6%/3.1%), and Qwen3-ASR-1.7B Shi et al. (2026) (1.6%/3.4%)—all of which use significantly more parameters and large-scale training data. This suggests that error correction is a complementary and cost-effective path to low WER, orthogonal to scaling acoustic models or training data.

## 6 Conclusion

We revisit ASR error correction with specialized language models. Our experiments demonstrate that ECLMs—compact seq2seq models trained on a mixture of ASR errors from real and synthetic audiooutperform both neural LM rescoring and generic LLMs, including fine-tuned variants, while being 15× smaller, 5–40× faster, and free of hallucination.

The central insight is that matching the training error distribution to real ASR mistakes matters more than model scale. Through systematic ablations, we show that multi-speaker TTS, noise augmentation, mixing synthetic and real data, and—counterintuitively—noisier TTS systems all contribute to a more representative error distribution.

We propose correction-first decoding, which reverses the conventional LM rescoring pipeline and outperforms it at comparable computational cost. A single ECLM generalizes across ASR architectures (QuartzNet, Conformer, Whisper, Parakeet) spanning CTC, seq2seq, and transducer paradigms, and across domains, reducing WER by 13% relative on average across eight diverse datasets spanning conversational and prepared speech.

Our results suggest that in the era of large language models, compact specialized models that capture task-

Table 8 LibriSpeech test WER (%) comparison with prior work.

Model Audio Data LS-clean LS-other No external audio data

Transformer Synnaeve et al. (2020) LS-960h 2.3 5.2 Context-Net (L) Han et al. (2020) LS-960h 1.9 4.1 Conformer (Transducer) Gulati et al. (2020) LS-960h 1.9 3.9 ASAPP-ASR Pan et al. (2020) LS-960h 1.8 4.5 E-branchformer + ILME Kim et al. (2023) LS-960h 1.8 3.7 Zipformer Yao et al. (2024) LS-960h 1.6 3.6 SYNT++ Hu et al. (2022) LS-960h 2.4 6.3 LAS + SC + LM Guo et al. (2019) LS-960h 4.3 –

Self-supervised pretraining (LL-60k)

wav2vec 2.0-Large Baevski et al. (2020) LL-60k 1.8 3.3 data2vec 2.0 Baevski et al. (2023) LL-60k 1.7 3.0 HuBERT-Large Hsu et al. (2021) LL-60k 1.9 3.3 HuBERT-XL Hsu et al. (2021) LL-60k 1.8 2.9 Conformer XXL Zhang et al. (2020) LL-60k 1.5 3.1

Large-scale supervised / Speech LLMs

Whisper large-v3 Radford et al. (2023) 5M hrs 2.0 3.7 Parakeet-TDT-v2 NVIDIA NeMo (2024) 120k hrs 1.7 3.2 Canary-Qwen-2.5B NVIDIA NeMo (2025) 234k hrs 1.6 3.1 Granite Speech 3.3-8B Saon et al. (2025) 76k hrs 1.4 2.9 Qwen3-ASR-1.7B Shi et al. (2026) 40M hrs + 1.6 3.4

baseline ASR (LS+TTS) + corr.-first LS-960h 1.5 3.3

specific distributions remain a compelling alternative—particularly in the low-error regime where LLMs overcorrect and hallucinate.

## References

Rosana Ardila et al. Common voice: A massively-multilingual speech corpus. arXiv preprint arXiv:1912.06670, 2019. Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised

learning of speech representations. Advances in neural information processing systems, 33:12449–12460, 2020.

Alexei Baevski, Wei-Ning Hsu, Alexis Conneau, and Michael Auli. Efficient self-supervised learning with contextualized target representations for vision, speech and language. In International Conference on Machine Learning. PMLR, 2023.

Richard He Bai, Tatiana Likhomanenko, Ruixiang Zhang, Zijin Gu, Zakaria Aldeneh, and Navdeep Jaitly. dmel: Speech tokenization made simple. arXiv preprint arXiv:2407.15835, 2024.

Edresson Casanova, Julian Weber, Christopher D Shulby, Arnaldo Candido Junior, Eren Gölge, and Moacir A Ponti. Yourtts: Towards zero-shot multi-speaker tts and zero-shot voice conversion for everyone. In International Conference on Machine Learning, pages 2709–2720. PMLR, 2022.

William Chan, Navdeep Jaitly, Quoc V Le, and Oriol Vinyals. Listen, attend and spell. arXiv preprint arXiv:1508.01211, 2015.

Chen Chen, Yuchen Hu, Chao-Han Huck Yang, Sabato Marco Siniscalchi, Pin-Yu Chen, and Eng Siong Chng. HyPoradise: An open baseline for generative speech recognition with large language models. In Advances in Neural Information Processing Systems, 2023.

Jan Chorowski and Navdeep Jaitly. Towards better decoding and language model integration in sequence to sequence models. arXiv preprint arXiv:1612.02695, 2016.

Christopher Cieri, David Miller, and Kevin Walker. The fisher corpus: A resource for the next generations of speechto-text. In LREC, volume 4, pages 69–71, 2004.

Lingfeng Dai, Lu Chen, Zhikai Zhou, and Kai Yu. Latticebart: Lattice-to-lattice pre-training for speech recognition. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6112–6116. IEEE, 2022.

Samrat Dutta, Shreyansh Jain, Ayush Maheshwari, Souvik Pal, Ganesh Ramakrishnan, and Preethi Jyothi. Error correction in asr using sequence-to-sequence models. arXiv preprint arXiv:2202.01157, 2022.

Yangui Fang, Baixu Chen, Jing Peng, Xu Li, Yu Xi, Chengwei Zhang, and Guohui Zhong. Fewer hallucinations, more verification: A three-stage llm-based framework for asr error correction. arXiv preprint arXiv:2505.24347, 2025.

Yassir Fathullah, Chunyang Wu, Egor Shanber, Ke Du, Evgeny Lakomkin, Xutai Jia, and Juan Pino. Prompting large language models with speech recognition abilities. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 13351–13355. IEEE, 2024.

Aaron Grattafiori et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen Schmidhuber. Connectionist temporal classification:

labelling unsegmented sequence data with recurrent neural networks. In Proceedings of the 23rd International Conference on Machine Learning, pages 369–376, 2006.

Anmol Gulati et al. Conformer: Convolution-augmented transformer for speech recognition. In Proc. Interspeech, 2020.

Caglar Gulcehre, Orhan Firat, Kelvin Xu, Kyunghyun Cho, Loic Barrault, Huei-Chi Lin, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. On using monolingual corpora in neural machine translation. arXiv preprint arXiv:1503.03535, 2015.

Jinxi Guo, Tara N Sainath, and Ron J Weiss. A spelling correction model for end-to-end speech recognition. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5651–5655. IEEE, 2019.

Wei Han, Zhengdong Zhang, Yu Zhang, Jiahui Yu, Chung-Cheng Chiu, James Qin, Anmol Gulati, Ruoming Pang, and Yonghui Wu. Contextnet: Improving convolutional neural networks for automatic speech recognition with global context. In Proc. Interspeech, 2020.

François Hernandez, Vincent Nguyen, Sahar Ghannay, Natalia Tomashenko, and Yannick Esteve. Ted-lium 3: Twice as much data and corpus repartition for experiments on speaker adaptation. In Speech and Computer: 20th International Conference, SPECOM 2018, Leipzig, Germany, September 18–22, 2018, Proceedings 20, pages 198– 208. Springer, 2018.

Takaaki Hori, Chiori Hori, Shinji Watanabe, and John R. Hershey. Minimum word error training of long short-term memory recurrent neural network language models for speech recognition. In 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5990–5994, 2016. doi: 10.1109/ICASSP.2016.7472827.

Oleksii Hrinchuk, Mariya Popova, and Boris Ginsburg. Correction of automatic speech recognition with transformer sequence-to-sequence model. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 7074–7078. IEEE, 2020.

Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3451–3460, 2021.

Ting-Yao Hu, Mohammadreza Armandpour, Ashish Shrivastava, Jen-Hao Rick Chang, Hema Koppula, and Oncel Tuzel. Synt++: Utilizing imperfect synthetic data to improve speech recognition. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 7682–7686. IEEE, 2022.

Yuchen Hu, Chen Chen, Chao-Han Huck Yang, Ruizhe Li, Dong Zhang, Zhehuai Chen, and Eng Siong Chng. Large language models are efficient learners of noise-robust speech recognition. In The Twelfth International Conference on Learning Representations, 2024.

Navdeep Jaitly, Patrick Nguyen, Andrew Senior, and Vincent Vanhoucke. Application of pretrained deep neural networks to large vocabulary speech recognition. In Proc. Interspeech, pages 2578–2581, 2012. doi: 10.21437/ Interspeech.2012-10.

Albert Q Jiang et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

Wei Kang, Xiaoyu Yang, Zengwei Yao, Fangjun Kuang, Yifan Yang, Liyong Guo, Long Lin, and Daniel Povey. Libriheavy: a 50,000 hours asr corpus with punctuation casing and context. arXiv preprint arXiv:2309.08105, 2023.

Anjuli Kannan, Yonghui Wu, Patrick Nguyen, Tara N Sainath, Zhijeng Chen, and Rohit Prabhavalkar. An analysis of incorporating an external language model into a sequence-to-sequence model. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5828. IEEE, 2018.

Kwangyoun Kim, Felix Wu, Yifan Peng, Jing Pan, Prashant Sridhar, Kyu J Han, and Shinji Watanabe. Ebranchformer: Branchformer with enhanced merging for speech recognition. In 2022 IEEE Spoken Language Technology Workshop (SLT), pages 84–91. IEEE, 2023.

Brian Kingsbury. Lattice-based optimization of sequence classification criteria for neural-network acoustic modeling. In 2009 IEEE International Conference on Acoustics, Speech and Signal Processing, pages 3761–3764. IEEE, 2009.

Nithin Rao Koluguri, Monica Sekoyan, George Zelenfroynd, Sasha Meister, Shuoyang Ding, Sofia Kostandian, He Huang, Nikolay Karpov, Jagadeesh Balam, Vitaly Lavrukhin, Yifan Peng, Sara Papi, Marco Gaido, Alessio Brutti, and Boris Ginsburg. Granary: Speech recognition and translation dataset in 25 european languages. arXiv preprint arXiv:2505.13404, 2025.

Samuel Kriman, Stanislav Beliaev, Boris Ginsburg, Jocelyn Huang, Oleksii Kuchaiev, Vitaly Lavrukhin, Ryan Leary, Jason Li, and Yang Zhang. Quartznet: Deep automatic speech recognition with 1d time-channel separable convolutions. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6124–6128. IEEE, 2020.

Yichong Leng et al. Fastcorrect 2: Fast error correction on multiple candidates for automatic speech recognition. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 4328–4337, 2021.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7871–7880, 2020.

Yutong Liu, Ziyue Zhang, Cheng Huang, Yongbin Yu, Xiangxiang Wang, Yuqing Cai, and Nyima Tashi. Listening, imagining & refining: A heuristic optimized asr correction framework with llms. arXiv preprint arXiv:2509.15095, 2025.

Rao Ma, Hao Li, Qi Liu, Lu Chen, and Kai Yu. Neural lattice search for speech recognition. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 7794–7798. IEEE, 2020.

Rao Ma, Mark JF Gales, Kate M Knill, and Mengjie Qian. N-best t5: Robust asr error correction using multiple input hypotheses and constrained decoding space. arXiv preprint arXiv:2303.00456, 2023.

Erik McDermott, Hasim Sak, and Ehsan Variani. A density ratio approach to language model fusion in end-to-end automatic speech recognition. 2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 434–441, 2019. URL https://api.semanticscholar.org/CorpusID:211243101.

Zhong Meng, Sarangarajan Parthasarathy, Eric Sun, Yashesh Gaur, Naoyuki Kanda, Liang Lu, Xie Chen, Rui Zhao, Jinyu Li, and Yifan Gong. Internal language model estimation for domain-adaptive end-to-end speech recognition. In 2021 IEEE Spoken Language Technology Workshop (SLT), pages 243–250. IEEE, 2021.

Mohammad Norouzi et al. Reward augmented maximum likelihood for neural structured prediction. Advances In

Neural Information Processing Systems, 29, 2016. NVIDIA NeMo. Parakeet-tdt-0.6b-v2. https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2, 2024. NVIDIA NeMo. Canary-qwen-2.5b. https://huggingface.co/nvidia/canary-qwen-2.5b, 2025. Jing Pan, Joshua Shapiro, Jeremy Wohlwend, Kyu J Han, Tao Lei, and Tao Ma. Asapp-asr: Multistream cnn and

self-attentive sru for sota speech recognition. In Proc. Interspeech, 2020.

Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: an asr corpus based on public domain audio books. In 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 5206–5210. IEEE, 2015.

Daniel S. Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, and Quoc V. Le. SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. In Proc. Interspeech, pages 2613–2617, 2019. doi: 10.21437/Interspeech.2019-2680.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International Conference on Machine Learning, pages 28492–28518. PMLR, 2023.

Srijith Radhakrishnan, Chao-Han Huck Yang, Sumeer Ahmad Khan, Rohit Kumar, Narsis A. Kiani, David GomezCabrero, and Jesper Tegnér. Whispering LLaMA: A cross-modal generative error correction framework for speech recognition. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. URL https: //openreview.net/forum id=OETPPc15XG.

George Saon et al. Granite-speech: Open-source speech-aware llms with strong english asr capabilities. arXiv preprint arXiv:2505.08699, 2025.

Jonathan Shen et al. Natural tts synthesis by conditioning wavenet on mel spectrogram predictions. In 2018 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 4779–4783. IEEE, 2018.

Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, et al. Qwen3-asr technical report. arXiv preprint arXiv:2601.21337, 2026.

Prashanth Gurunath Shivakumar, Haoqi Li, Kevin Knight, and Panayiotis Georgiou. Learning from past mistakes: improving automatic speech recognition output via noisy-clean phrase context modeling. APSIPA Transactions on Signal and Information Processing, 8:e8, 2019.

Anuroop Sriram, Heewoo Jun, Sanjeev Satheesh, and Adam Coates. Cold fusion: Training seq2seq models together with language models, 2018. URL https://openreview.net/forum id=rybAWfx0b.

Gabriel Synnaeve et al. End-to-end ASR: from supervised to semi-supervised learning with modern architectures. In

ICML 2020 Workshop on Self-supervision in Audio and Speech, 2020. URL https://openreview.net/forum id=OSVxDDc360z. Tomohiro Tanaka, Ryo Masumura, Hirokazu Masataki, and Yushi Aono. Neural error corrective language models for

automatic speech recognition. In Proc. Interspeech, pages 401–405, 2018.

Shubham Toshniwal, Anjuli Kannan, Chung-Cheng Chiu, Yonghui Wu, Tara N Sainath, and Karen Livescu. A comparison of techniques for language model integration in encoder-decoder speech recognition. In 2018 IEEE spoken language technology workshop (SLT), pages 369–375. IEEE, 2018.

Ehsan Variani, Xin Lei, Erik McDermott, Ignacio Lopez Moreno, and Javier Gonzalez-Dominguez. Deep neural networks for small footprint text-dependent speaker verification. In 2014 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 4052–4056. IEEE, 2014.

Ehsan Variani, David Rybach, Cyril Allauzen, and Michael Riley. Hybrid autoregressive transducer (hat). In ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6139–6143,

2020. doi: 10.1109/ICASSP40776.2020.9053600. Karel Veselý, Arnab Ghoshal, Lukáš Burget, and Daniel Povey. Sequence-discriminative training of deep neural networks. In Proc. Interspeech, pages 2345–2349, 2013. doi: 10.21437/Interspeech.2013-548. Changhan Wang et al. Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation. arXiv preprint arXiv:2101.00390, 2021. Haoyu Wang, Shuyan Dong, Yue Liu, James Logan, Ashish Agrawal, and Yang Liu. Asr error correction with augmented transformer for entity retrieval. In Proc. Interspeech, 2020. Shinji Watanabe et al. Chime-6 challenge: Tackling multispeaker speech recognition for unsegmented recordings. arXiv preprint arXiv:2004.09249, 2020. Jian Wu et al. On decoder-only architecture for speech-to-text and large language model integration. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 1–8. IEEE, 2023.

Qiantong Xu, Tatiana Likhomanenko, Jacob Kahn, Awni Hannun, Gabriel Synnaeve, and Ronan Collobert. Iterative Pseudo-Labeling for Speech Recognition. In Proc. Interspeech, pages 1006–1010, 2020. doi: 10.21437/Interspeech. 2020-1800.

Chao-Han Huck Yang et al. Generative speech recognition error correction with large language models and taskactivating prompting. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 1–8. IEEE, 2023.

Zengwei Yao, Liyong Guo, Xiaoyu Yang, Wei Kang, Fangjun Kuang, Yifan Chen, Zengrui Liu, and Daniel Povey. Zipformer: A faster and better encoder for automatic speech recognition. In International Conference on Learning Representations, 2024.

Shiliang Zhang, Ming Lei, and Zhijie Yan. Investigation of Transformer Based Spelling Correction Model for CTCBased End-to-End Mandarin Speech Recognition. In Proc. Interspeech, pages 2180–2184, 2019. doi: 10.21437/ Interspeech.2019-1290.

Yu Zhang, James Qin, Daniel S Park, Wei Han, Chung-Cheng Chiu, Ruoming Pang, Quoc V Le, and Yonghui Wu. Pushing the limits of semi-supervised learning for automatic speech recognition. arXiv preprint arXiv:2010.10504, 2020.

Yun Zhao, Xuerui Yang, Jinchao Wang, Yongyu Gao, Chao Yan, and Yuanfu Zhou. Bart based semantic correction for mandarin automatic speech recognition system. arXiv preprint arXiv:2104.05507, 2021.

