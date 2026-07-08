# arXiv:2510.26865v2[cs.CV]24Mar2026

## Do Vision-Language Models Measure Up? Benchmarking Visual Measurement Reading with MeasureBench

#### BAAI FlagEval Team∗

Project page: https://flageval-baai.github.io/MeasureBenchPage/

TL;DR: Fine-grained visual understanding tasks such as visual measurement reading have been surprisingly challenging for frontier general-purpose vision-language models. We introduce MeasureBench, a benchmark with diverse images of measuring instruments collected from both real-world images and a new data synthesis pipeline.

### Abstract

Reading measurement instruments is eﬀortless for humans and requires relatively little domain expertise, yet it remains surprisingly challenging for current vision-language models (VLMs) as we ﬁnd in preliminary evaluation. In this work, we introduce MeasureBench, a benchmark on visual measurement reading covering both real-world and synthesized images of various types of measurements, along with an extensible pipeline for data synthesis. Our pipeline procedurally generates a speciﬁed type of gauge with controllable visual appearance, enabling scalable variation in key details such as pointers, scales, fonts, lighting, and clutter. Evaluation on popular proprietary and open-weight VLMs shows that even the strongest frontier VLMs struggle with measurement reading in general. We have also conducted preliminary experiments with reinforcement ﬁnetuning (RFT) over synthetic data, and ﬁnd a signiﬁcant improvement on both in-domain synthetic subset and real-world images. Our analysis highlights a fundamental limitation of current VLMs in ﬁne-grained spatial grounding. We hope this resource and our code releases can help future advances on visually grounded numeracy and precise spatial perception of VLMs, bridging the gap between recognizing numbers and measuring the world.

### 1 Introduction

Recent advances in vision-language models (VLMs) have demonstrated impressive capabilities in tackling complex reasoning tasks that combine textual and visual information. Models or systems such as GPT-5 (OpenAI, 2025a) and Gemini 2.5 Pro (Gemini Team, 2025) achieve human-expert level performance on collegelevel problems in MMMU (Yue et al., 2024) and MMMU-Pro (Yue et al., 2025). Even on Humanity’s Last Exam (HLE) (Phan et al., 2025), a benchmark measuring the frontier of human knowledge, state-of-the-art models achieve accuracies exceeding 25%, substantially surpassing the human average.

That said, state-of-the-art VLMs still struggle with ﬁne-grained perception, e.g., low-level visual cues, precise geometry, and subtle changes, even when their high-level reasoning appears strong. Existing ﬁne-grained evaluations are well represented by text reading and chart reasoning (Singh et al., 2019; Masry et al., 2022; Tang et al., 2025), or by similarly artiﬁcial low-level vision tests such as BlindTest (Rahmanzadehgervi et al., 2024) and SalBench (Dahou et al., 2025). However, they rarely require mapping physical scales to numeric values.

Visual instrument reading tasks usually require ﬁne-grained visual perception, light quantitative reasoning, and basic arithmetic operations. Examples include reading pressure gauges in industrial settings, and thermometers or even as simple as clocks in daily life. Accurate interpretation of these instruments is crucial for safety, eﬃciency, and decision-making across domains for vision language models or future embodied AI systems. While a few existing studies have covered very speciﬁc types of reading such as clocks (Saxena et al., 2025; Yang et al., 2022), rulers (Matuzevičius, 2023; Pan et al., 2025), industrial gauges (Izquierdo-Domenech et al., 2025; Valente et al., 2025), and household meters (Van et al., 2025), they do not span the broad diversity of instruments or reading designs.

To ﬁll this gap, we introduce MeasureBench, a benchmark for evaluating VLMs on measuring instrument reading across 26 instrument types and four types of readout designs. Each image is paired with a reading

∗Full list of authors attached at the end.

[Figure 3]

Figure 1: MeasureBench real-world samples with four commonly used reading designs.

question. MeasureBench comprises 2,442 image–question pairs: 1,272 diverse real-world images collected and human-annotated, and 1,170 synthetic images generated with randomized readings for 39 instruments.

Our data synthetic pipeline has two complementary backends: (i) a 2D programmatic renderer for diverse layouts with full control over fonts and geometry; and (ii) a 3D Blender renderer for photorealistic scenes with realistic lighting, materials, reﬂections, and occlusions. The pipeline is fully automated and readily scalable in both breadth (instrument types) and depth (variations). This pipeline can be used to generate additional data for training or evaluation.

We evaluate a number of modern VLMs on MeasureBench and report these key ﬁndings:

- • Persisting diﬃculty. Current VLMs still struggle with instrument reading, with the best model achieving only 30.2% accuracy on the real-world set and 26.3% on the synthetic set.
- • Object and text recognition seem easy, but inferring numbers is hard. Models exhibit strong image understanding and reach over 90% accuracy on unit recognition. Yet they falter on mapping scales to numeric values.
- • Systematic ﬁne-grained errors. Models often “know how to read” but miss details: They misinterpret pointer positions, confuse adjacent ticks, and mismatch values to scale markings, leading to oﬀ-target answers.

With our data synthetic pipeline that produces accurately annotated readings, we have also conducted preliminary experiments of reinforcement learning using synthetic data. Results are encouraging in that the synthetic subset of MeasureBench can get signiﬁcantly improved, but not as promising on real-world images.

In summary, our contributions include:

- • We present MeasureBench, a comprehensive benchmark targeting ﬁne-grained instrument reading across 26 instrument types and 2,442 image–question pairs.
- • We provide a controllable 2D/3D synthesis pipeline that produces precise labels for sketch or photorealistic images with randomized readings for 39 instruments.
- • We deliver a standardized evaluation of 18 contemporary VLMs and an analysis of their failure modes, highlighting concrete gaps in low-level perception and precise geometric reasoning.
- • Preliminary RFT experiments using our synthetic pipeline show promise for data curation, but highlight the need for better visual representations to improve generalization.

[Figure 5]

Statistics Number Total Questions 2442 Real-World Images 1272 (52%)

- * Dial/Linear/Dig./Comp. 711/361/96/104
- * Instrument Types 26 Synthetic Images 1170 (48%)
- * Dial/Linear/Dig./Comp. 750/300/60/60
- * Instrument Types 16
- * Instrument Appearances 39

Table 1: Key statistics of MeasureBench. Figure 2: Distribution of reading designs and instrument types.

### 2 MeasureBench

#### 2.1 Overview of MeasureBench

We introduce MeasureBench, a comprehensive benchmark for evaluating the ability to read values from measuring instruments. MeasureBench comprises two main components: (i) a diverse set of instrument images with standardized annotations, and (ii) a data synthesis framework for generating additional training and evaluation data. By visual appearance, we categorize instruments into four readout designs (see also Figure 1 for examples from the real-world images in MeasureBench):

- • Dial: Analog instruments with one or more pointers (e.g., ammeters and pressure gauges which typically have a single pointer, whereas clocks often have two or three).
- • Digital: Devices with electronic or mechanical digital readouts (e.g., pulse oximeters and electromechanical electricity meters).
- • Linear: Instruments with linear scales and no pointers (e.g., rulers with a single scale, and vernier calipers with a main and a vernier scale).
- • Composite: Instruments combining multiple readout designs, such as dial calipers and complex water meters.

As shown in Table 1, MeasureBench contains 2,442 questions: 1,272 real-world images and 1,170 synthetic images. The real-world subset spans 26 instrument types, while the synthetic subset covers 16 types with 39 distinct appearances. To better explore the capability of VLMs in ﬁne-grained instrument reading, we place greater emphasis on dial and linear instruments because digital devices primarily test OCR capabilities, and composite instruments are comparatively rare in practice.

#### 2.2 Evaluation metrics

Measurement error is natural when reading from any instrument that does not explicitly display a deterministic digital value on the screen. Therefore, we determine the correctness of the ﬁnal reading via interval matching instead of a strict value, along with the correctness of unit prediction.

Answer extraction. To get the reading from natural language output, we extract the ﬁnal answer after common markers (e.g., “Answer:”) or inside \boxed{}. Our evaluation script will speciﬁcally parse: (i) numeric: integers, decimals, scientiﬁc notation, and fractions (a/b→ﬂoat). If multiple scalars appear, use the rightmost. (ii) time: the ﬁrst hh:mm[:ss] pattern, converted to seconds. Preserve adjacent tokens for unit matching. 1

Answer matching. Each sample in our benchmark includes one or more ground-truth candidates2, each contains a closed numeric interval for value grading, optionally along with a set of acceptable unit substrings

1Unicode characters are normalized for equivalence matching. 2Some instruments express the same quantity in diﬀerent units thereby diﬀerent values, such as degrees in Celsius

and Fahrenheit. During evaluation, we only adopt the single ground-truth that maximizes the score.

to indicate a correct unit in a model response. A prediction is value-correct if the number parsed from the model response falls within the interval of any candidate, and unit-correct if any unit string in that candidate can match the extracted answer. Fully-correct requires value-correct and, when speciﬁed, unit-correct for the same candidate. If multiple candidates exist, score against the one that maximizes correctness (prefer fully-correct; otherwise prefer value-correct; break ties by smaller relative error, then by narrower interval).

#### 2.3 Real-world subset curation

We assembled a real-world subset of images from three sources: (i) Google Image Search using instrumentspeciﬁc keywords, restricted to images under permissive licenses for usage, (ii) photos contributed by team members under private authorization, and (iii) images purchased from a third-party vendor. We removed low-quality images (e.g., blurry, low-resolution, or occluded) and annotated the remaining images using a standardized schema. For each image, we recorded the instrument type, readout design, candidate units, and the valid interval of reading values; any value within this interval is considered correct.

We recruited 10 qualiﬁed annotators and assigned tasks aligned with their professional backgrounds. Each image was independently labeled by one annotator and veriﬁed by another; disagreements were adjudicated by a third annotator. Another independent round of review was conducted to verify the correctness of annotation, including the numerical intervals and the unit. We have also conducted a preliminary analysis on prompt sensitivity (with details in Sec 3.3) and ﬁnd very little impact on overall results, so we leave most of the collected prompts unchanged.

#### 2.4 Data synthesis framework

[Figure 7]

Figure 3: Left: A hybrid measuring instrument synthesis framework. Right: Examples of synthetic images.

As shown in Figure 3, we develop a modular and scalable synthesis framework that treats each instrument as a generator registered under a uniﬁed interface. A global registry maps instrument names to generators; each generator returns a rendered image together with a standardized label schema comprising the numeric value, unit and readout design. This uniform contract enables plug-and-play additions.

For each sample, the framework randomizes the number and type of scales, measurement readout numbers, ranges/units, materials, lighting, backgrounds, and camera pose, while enforcing semantic validity. Figure 3 illustrates the resulting diversity along four axes.

We provide two complementary back-ends under the same interface:

• 2D programmatic rendering. A prompt template speciﬁes instrument types and reading constraints, and prescribes the code interface and preferred libraries. LLMs then draft the code accordingly. We verify the code before registering the generator.

Model Real-world subset Synthetic subset Ovr Val Unit Dial Dig Lin Com Ovr Val Unit Dial Dig Lin Com

Gemini-2.5-Pro 30.2 30.7 96.2 31.5 80.2 21.9 3.8 26.3 26.8 93.1 18.3 70.0 40.0 15.0 Qwen3-VL-235B 22.6 23.0 95.7 23.5 64.6 15.2 19.0 19.6 94.4 14.1 60.0 26.3 GPT-5-Mini 22.0 22.4 95.2 20.8 70.8 16.9 2.9 17.9 18.6 93.2 12.0 56.7 28.3 Gemini-2.5-Flash 20.2 21.1 93.4 20.5 65.6 13.0 1.0 18.1 19.0 91.7 11.9 75.0 25.7 GPT-5 19.8 19.9 96.0 18.3 66.7 15.2 2.9 16.9 17.5 94.3 9.7 48.3 31.7 1.7 Qwen3-VL-8B 15.3 15.8 94.0 14.5 53.1 11.3 0.0 11.4 11.6 92.4 8.0 25.0 19.3 0.0 Qwen2.5-VL-7B 14.6 15.0 93.4 13.8 49.0 11.4 0.0 10.9 11.5 88.5 5.7 33.3 21.7 0.0 Qwen2.5-VL-72B 14.5 92.1 12.2 55.2 12.2 0.0 11.7 12.0 92.3 43.3 21.0 0.0 Claude-Opus-4.1 14.3 14.9 94.5 14.8 38.5 11.1 0.0 13.3 14.1 93.1 6.4 45.0 27.0 0.0 InternVL3.5-38B 12.9 13.6 89.8 12.1 51.6 7.7 0.0 12.6 15.4 78.5 6.3 41.7 25.3 0.0 Claude-Sonnet-4 12.6 13.1 89.9 15.0 20.8 9.1 0.0 11.0 11.5 92.8 5.1 26.7 25.0 0.0 LLaMA-4-maverick 12.2 12.9 91.6 12.1 44.8 7.2 0.0 12.1 13.2 89.7 6.3 50.0 21.7 0.0 Qwen2.5-VL-32B 11.7 12.0 94.6 9.0 51.6 9.7 0.0 10.5 10.7 96.0 5.3 28.3 22.0 0.0 LLaMA-4-scout 10.9 11.4 90.6 8.2 54.2 8.0 0.0 9.1 10.2 86.4 5.5 20.0 17.7 0.0 Mistral-medium-3.1 10.6 11.2 93.4 7.0 57.3 8.3 0.0 8.5 8.8 91.6 3.7 23.3 19.3 0.0 InternVL3.5-8B 9.7 10.9 84.0 10.4 30.5 5.5 0.0 7.7 8.4 84.6 3.5 26.7 16.0 0.0 Mistral-small-3.2 8.5 9.7 81.3 7.9 32.3 5.8 0.0 6.5 8.0 80.5 3.2 5.0 16.3 0.0 Grok-4 7.5 7.7 80.5 6.5 24.0 7.5 0.0 6.2 6.4 71.6 3.3 25.0 10.3 1.7

Table 2: Performance on real and synthetic images. We report accuracy (%) for each model: overall (Ovr), value (Val), unit (Unit), and by readout type—Dial, Digital (Dig), Linear (Lin), Composite (Com).

• 3D physical rendering. We leverage existing Blender3 assets and write code to randomize backgrounds, instrument readings (e.g., pointer angles, scale ranges) and camera pose to produce photorealistic images and narrow the sim-to-real gap. (See appendix for details)

We implement 39 distinct appearances spanning 16 instrument types. For benchmarking purpose, we independently generate 30 images per appearance, totaling 1,170 synthetic images. As illustrated in Figure 3, the images vary along four axes: multi-style (2D vs. photorealistic 3D), multi-class (dial, linear, composite, digital), multi-orientation (rotations/tilts and imaging perturbations), and multi-scale (ranges/units and dual scales), providing broad coverage for robust reading models.

### 3 Evaluation Results

We present a systematic evaluation of various VLMs on MeasureBench: 8 proprietary and 10 open-weight models. The evaluated model families include GPT (OpenAI, 2025a), Claude (Anthropic, 2025), Gemini (Gemini Team, 2025), Mistral (Mistral AI, 2025), Grok (xAI, 2025), Qwen-VL (Bai et al., 2025), InternVL3 (Zhu et al., 2025), and LLaMA-4 (Meta AI, 2025). All evaluations are conducted using FlagEvalMM (He et al., 2025), a ﬂexible and comprehensive framework for multimodal model evaluation.

#### 3.1 Main results

Table 2 reports results on MeasureBench. The best-performing Gemini 2.5 Pro reaches only 30.2% overall accuracy on real images and 26.3% on synthetic images, showing that reading measuring instruments remains a challenging ﬁne-grained vision task for current VLMs. A few more observations:

Value reading is the bottleneck. Across models, recognizing (or inferring) the unit is consistently above 90% accurate on both real and synthetic sets, while value accuracy is much lower (e.g., Gemini 2.5 Pro: 96.2% unit vs. 30.7% value on real images). This suggests strong OCR or object recognition capabilities from current VLMs for unit prediction. However, the low performance shows that estimating the numerical value often requires precise localization of pointers, ticks, and scales.

Diﬀerent readout designs are not equally challenging. Table 2 displays decomposed accuracy metrics by instrument type. Digital displays are much easier (e.g., up to 80.2% on real images for Gemini 2.5 Pro), reﬂecting

3https://www.blender.org/

reliance on OCR. Dial and linear instruments remain challenging (typically 10-32%), as they require needle localization or reading tick marks under clutter, highlights, and distortion. Composite instruments are by far the most challenging: they require combining multiple readout designs, reading each component correctly, and performing the corresponding numerical calculations.

Larger models may not always read better. Table 2 shows that GPT-5-Mini marginally outperforms GPT-5, while Qwen2.5-VL-7B performs on par with Qwen2.5-VL-72B and outperforms Qwen2.5-VL-32B. Through case studies (more in Appendix), we observe that while a portion of GPT-5-Mini’s correct answers can attribute to successful guessing, GPT-5 genuinely erred in image recognition on a diﬀerent subset of problems. Within the Qwen family, overall performance does not monotonically improve with larger LLMs when the visual encoder remains unchanged.4 This implies that larger language backbones do not contribute to better ﬁne-grained perception, while sometimes the language prior may incorrectly bias the predicted values. For instance, Qwen2.5-VL-72B has shown a much stronger preference in predicting “10:10” on clock images (73% of real images), as described in Appendix E.

Real vs. synthetic. The gap between real and synthetic is modest but consistent: most VLMs achieve generally lower accuracy on synthetic images than real images (e.g., Gemini-2.5-Pro 30.2 to 26.3, GPT-5 19.8 to 16.9), showing that synthetic scenes remain a genuine challenge rather than an easier abstraction. The drop is mainly driven by value accuracy, while unit accuracies remain similar, suggesting that numeric extraction is the primary failure mode.

[Figure 10]

Figure 4: Model accuracies on real images by instrument category (categories with ≥20 samples).

4Qwen2.5-VL-7B, Qwen2.5-VL-32B and Qwen2.5-VL-72B use a ViT with the same number of parameters (Bai et al., 2025).

Category-wise performance varies widely. Figure 4 reveals substantial spread across instrument categories. Categories with a high proportion of digital readouts (e.g., electricity meters) tend to achieve higher accuracy, whereas categories dominated by multi-needle dials (e.g., clocks, water meters) are challenging for all models. Single-needle dials with sparse ticks (e.g., speedometer) are generally easier, and linear gauges (e.g., rulers, measuring cylinders) are easier overall than dials.

#### 3.2 To think, or not to think?

Inference-time “thinking” has been widely adopted to improve large language models (LLMs) on complex text-based reasoning. We ask whether this also holds for VLMs on MeasureBench, which demands ﬁne-grained visual perception coupled with numerical reasoning. We compare a couple of hybrid reasoning models under a nothinking setting (reasoning tokens set to 0) against a thinking setting (maximum of 10,240 reasoning tokens). The study covers six models: InternVL3.5-8B/38B, Qwen3-VL-8B/235B, Claude 4.1 Opus, and Gemini 2.5 Flash.

[Figure 12]

As shown in Figure 5, enabling thinking yields very little improvement, sometimes even degrades performance. While thinking often boosts text-only reasoning, it does not appear to help VLMs attend to the most relevant image regions or to enhance ﬁne-grained visual perception on MeasureBench. This conforms to recent ﬁndings on the utility of test-time thinking for visual problems (FlagEval Team, 2025).

Figure 5: Performance and eﬃciency analysis of various large vision-language models. The accuracy against the average token count is plotted to show the performance-cost trade-oﬀ.

- Figure 5 further relates accuracy to the average number of reasoning tokens consumed per sample. Although thinking increases token usage, the increment is modest (a few hundreds up to 1k-2k tokens), yet accuracy gains remain limited. Instrument reading primarily requires precise visual interpretation rather than extended chain-of-thought, so extended text-based reasoning is ineﬀective to improve performance on this task.

#### 3.3 Prompt sensitivity analysis

We investigate whether prompting style aﬀects instrument reading by comparing two styles:

- • Speciﬁc: explicitly names the instrument type (e.g., “What is the reading of this ammeter?”).
- • General: uses a uniform query (e.g., “What is the reading of the instrument?”), requiring the model to infer the instrument type and the quantity being measured.

A manual review of prompts yields three categories: besides speciﬁc (9.36%) and general (80.11%), the remaining prompts are immutable (10.53%) as essential information prevents rewriting (e.g., “What is the high pressure according to the sphygmomanometer?”). We exclude them and rewrite the other two sets into the two styles. We also explore the eﬀect of adding a prompt suﬃx: “Provide an answer as precise as possible”.

As shown in Table 3, prompting with speciﬁc instrument names only has a moderately positive impact on unit accuracy but no real diﬀerence on total accuracy. Meanwhile, explicitly asking VLMs to provide as precise answers does not yield consistent impact on performance either. Hence we retain the original simple prompts for main benchmark evaluation without additional prompt engineering.

Model Precise Total Acc. Unit Acc.

Gen. Spec. Gen. Spec. No 17.7 88.0 90.0

GPT-4.1-Mini Yes 15.2 15.5 88.8 91.4

No 9.8 9.6 84.5 90.2 InternVL3.5-8B Yes 11.6 10.8 84.2 89.6

No 13.5 14.9 90.1 90.7 Qwen2.5-VL-72B Yes 14.0 13.4 91.5 92.8

Table 3: Model performance under diﬀerent prompt settings.

[Figure 14]

- Figure 6: Case studies. Text in green marks statements consistent with the image; yellow marks contradictions.

#### 3.4 Case studies

- Figure 6 shows two typical examples from our benchmark: a measuring cylinder and an ammeter. In each panel, text in green highlights denote statements consistent with the image, whereas text in yellow denotes claims that are contradicted by the visual evidence.

What VLMs get right: Models generally know the task. They identify the instrument, locate the indicator (meniscus/needle), infer the unit and major tick spacing, and try to interpolate to a ﬁnal value. This shows mission awareness and an active search for the pointer.

Where they fail: Most errors arise from small perceptual mistakes that dominate the numeric outcome: (i) Pointer localization: one minor tick left/right changes the reading (e.g., 4.4 vs. 4.5 A). (ii) Indicator interpretation: wrong minor-tick count or reading the wrong edge of the meniscus.

Right answer, wrong reasoning. We observe frequent error cancellation. In the cylinder example (Gemini-2.5pro), an incorrect subdivision story coincidentally oﬀsets a later mistake, yielding the correct number. Such cases inﬂate accuracy if only the ﬁnal answer is scored.

#### 3.5 Impact of detailed instructional prompts

We investigate whether providing step-by-step reading instructions tailored to diﬀerent instrument types can improve model performance. The instruction includes design-level common rules shared by instruments of

the same readout design and instrument-speciﬁc guidance. As shown in Table 4, even with explicit guidance on how to read the instruments, the performance gain on real-world MeasureBench is very limited. Adding in-context examples does not seem to help either, suggesting that the bottleneck lies in ﬁne-grained visual perception rather than a lack of procedural knowledge.

Model Overall Value Unit

Gemini-2.5-Pro 31.8(+1.6) 32.2(+1.5) 96.8(+0.6) GPT-5 19.9(+0.1) 20.1(+0.2) 97.5(+1.5) Qwen2.5-VL-7B 14.6(+0.0) 15.5(+0.5) 93.2(-0.2)

- Table 4: Impact of more detailed instructional prompts on MeasureBench (real-world images). Numbers in parentheses indicate the change compared to the default prompt.

3.6 Generalization on general benchmarks

To assess potential negative transfer from reinforcement ﬁnetuning on synthetic measurement data, we evaluate Qwen2.5-VL-7B before and after GRPO training on popular general-purpose benchmarks. As shown

- in Table 5, GRPO training on synthetic data yields comparable performance on these benchmarks with no degradation, indicating that the model retains its general capabilities.

Model MMMU MMMU-Pro MathVista TextVQA

Qwen2.5-VL-7B 52.44 37.75 69.40 84.74 + GRPO 54.33 37.86 69.00 84.24

Table 5: Qwen2.5-VL-7B results on general benchmarks before and after GRPO training on synthetic measurement data.

3.7 Comparison with supervised ﬁne-tuning

We compare GRPO with supervised ﬁne-tuning (SFT) on the same synthetic dataset. We experiment with two SFT response formats: (i) direct answer only (e.g., “4.3cm”, “6.5L”), and (ii) rationale + ﬁnal answer, where GPT-5.2 is used to generate step-by-step rationales given the image and the ground-truth readout. As shown

- in Table 6, while SFT substantially improves in-domain synthetic performance, it degrades accuracy on realworld images, indicating severe overﬁtting to synthetic patterns. In contrast, GRPO improves on both synthetic and real-world subsets, demonstrating better generalization.

- Table 6: Comparison of GRPO and SFT training on Qwen2.5-VL-7B. SFT overﬁts to synthetic patterns, degrading real-world performance, while GRPO improves both.

Real-world Synthetic Ovr Val Unit Ovr Val Unit

Model

Qwen2.5-VL-7B 14.6 15.0 93.4 10.9 11.5 88.5 + GRPO 19.7 20.4 92.3 35.2 35.6 96.7 + answer SFT 12.5 13.4 89.5 29.7 30.9 96.8 + rationale SFT 8.8 9.1 91.3 21.7 22.0 95.8

#### 3.8 Would earlier special-purpose systems work?

Prior to the emergence of general-purpose VLMs, there exist earlier domain-speciﬁc computer vision systems (Shu et al., 2023; Reitsma et al., 2024) designed for gauge reading where model checkpoints are still available. These systems manually design specialized pipelines, e.g., detecting the gauge region, localizing the pointer, recognizing scale marks, and reading text via OCR. We have veriﬁed that the systems are performing as expected on their test sets, before evaluating both systems on a relevant subset (dial meters) of MeasureBench. The results show very little generalization across detection, pointer localization, or text reading on our benchmark data which may diﬀer a lot from their training images. As pipeline systems, they tend

to fail on the new images at diﬀerent components including gauge detection, number recognition (OCR), and pointer segmentation.

- Table 7 shows that the generalization of pointer value detection is inferior to that of general-purpose VLMs. The special-purpose neural network components (Shu et al., 2023) more or less overﬁt their training data, resulting in failures to detect pointers or scale marks on most out-of-distribution datasets. Meanwhile, the accuracy of OCR-based unit recognition (Reitsma et al., 2024) is signiﬁcantly lower than that of VLMs.

Real-world Synthetic Ovr Val Unit Ovr Val Unit

Model

Reitsma et al. Reitsma et al. (2024) 8.5 11.9 17.8 12.5 12.5 17.5 Shu et al. Shu et al. (2023) N/A 4.2 N/A N/A 0.0 N/A Gemini-2.5-Pro 30.2 30.7 96.2 26.3 26.8 93.1 Qwen3-VL-8B 15.3 15.8 94.0 11.4 11.6 92.4

Table 7: Model accuracy (%) for VLMs and prior special-purpose systems; “N/A” = failed on all examples.

### 4 Can training with synthetic data help?

The data synthesis pipeline provides diverse instrument images, which raises the question of whether training on synthetic data can improve performance on real-world images. To investigate this, we generated 100 samples for each of the 39 instruments (3,900 image–question pairs) and used them for model training. The task format especially suits reinforcement learning via assigning a positive reward on correct reading results. We adapt the GRPO algorithm (Shao et al., 2024) to conduct reinforcement ﬁnetuning (RFT) on Qwen2.5VL-3B and Qwen2.5-VL-7B. Models are trained using the verl library Sheng et al. (2024), training details are listed in appendix.

#### 4.1 Reward design

To stay consistent with the scoring used in our evaluation, we employ a rule-based reward aligned with the evaluation method.5 Let I = [l,r] and u denote the ground-truth interval and unit, and let p˜ be the model’s textual prediction from which we extract the numeric value yˆ and unit uˆ. F is the response pattern “<think>.*</think>.*Final Answer.*” Deﬁne the indicators

call = I{yˆ ∈ I ∧ uˆ = u} (1) cfmt = I{p˜ matches the schema F}, (2)

With a weight α = 0.9, the reward is deﬁned as

Reval = α call + (1 − α)cfmt. (3)

#### 4.2 Results and analysis

The evaluation results of Qwen2.5-VL series models with RFT are shown in Table 8. We get signiﬁcant performance boost on the in-domain synthetic image test set where the overall accuracy increased by more than threefold (e.g., Qwen2.5-VL-3B: 8.4% to 31.5%). Moreover, the model exhibited enhanced generalization to out-of-distribution (OOD) real-world images, with accuracy rising notably (e.g., Qwen2.5-VL-7B: 14.6% to 19.7%).

We further analyze the per-instrument accuracy changes on Qwen2.5-VL-7B. As shown in Figure 7, The top-10 gains occur primarily for simpler instruments, such as single-needle dials (e.g., tachometers), digital readouts (e.g., electricity meters) and linear gauges with sparse ticks. This pattern suggests that RFT on synthetic data is most eﬀective for instruments with low layout complexity, whereas composite reading design remain challenging.

5We have also tried a soft variant that gives partial reward to predicted values that are close to the ground-truth interval, but not observing much diﬀerence. We discuss more details in the Appendix.

###### Model Overall Value Unit

Qwen2.5-VL-7B + No RFT (real) 14.6 15.0 93.4 GRPO (real) 19.7 (+34.9%) 20.4 (+36.0%) 92.3 (-1.2%) No RFT (synth) 10.9 11.5 88.5 GRPO (synth) 35.2 (+222.9%) 35.6 (+209.6%) 96.7 (+9.3%)

Qwen2.5-VL-3B + No RFT (real) 10.5 10.8 89.3 GRPO (real) 12.7 (+21.0%) 13.8 (+27.8%) 89.0 (-0.3%) No RFT (synth) 8.4 9.1 89.9 GRPO (synth) 31.5 (+275.0%) 32.4 (+256.0%) 95.7 (+6.5%)

Table 8: Results of Qwen2.5-VL series with GRPO on real-world and synthetic subsets.

[Figure 18]

- Figure 7: Top-10 instrument categories (≥ 20 samples) with the largest accuracy gains after RFT on Qwen2.5-VL-7B.

[Figure 19]

Figure 8: Numerical output distributions in the [0, 100] range.

As shown in Figure 8, we compare numerical distributions between model predictions and ground truth. The ground-truth readings are relatively smooth, with no pronounced peaks. In contrast, Qwen2.5-VL-7B exhibits clear spikes at round multiples of ten (e.g., 10, 20), indicating a strong language-model prior that favors such values over the visual evidence. After RFT, these peaks are reduced and the distribution becomes smoother and closer to that of a stronger model (e.g., Gemini-2.5-Pro), indicating that RFT helps mitigate this prior bias.

In general, these results show potential from more data curation for VLM training, but also leaving a question on whether we should instead pursuit better model architectures and visual encoding schemes that would make a future VLM genuinely reasoning from detailed visual cues and generalizing over unseen types of instruments.

### 5 Related work

VLMs & VLM Benchmarks Vision–Language Models (VLMs) have made rapid progress in recent years. Early systems such as LLaVA (Liu et al., 2023) and InstructBLIP (Dai et al., 2023) pioneered vision instruction tuning, while families like Qwen-VL (Bai et al., 2023), InternVL (Chen et al., 2024), and GPT-4o (OpenAI, 2024) demonstrated strong general multimodal understanding. More recently, models augmented with reinforcement learning and veriﬁable rewards (e.g., OpenAI o3 (OpenAI, 2025b), Gemini 2.5 Pro (Gemini Team, 2025), Claude Opus 4 (Anthropic, 2025), Qwen3-VL (Bai et al., 2025)) exhibit improved stepwise reasoning and planning. To assess these capabilities, a broad suite of benchmarks has emerged. General-purpose evaluations (e.g., MMBench (Liu et al., 2024), MM-Vet (Yu et al., 2024), Seed-Bench (Li et al., 2023b;a)) target holistic multimodal competence; knowledge-intensive suites (e.g., MMMU (Yue et al., 2024; 2025), ScienceQA (Lu et al., 2022)) emphasize academic problem solving; math-centric sets (e.g., MathVision (Wang et al., 2024), MathVerse (Zhang et al., 2024)) probe visual mathematical reasoning; and perception-focused tests (e.g., CV-Bench (Tong et al., 2024), BLIND (Rahmanzadehgervi et al., 2024)) stress ﬁne-grained visual understanding. More specialized studies on ﬁne-grained reading report persistent weaknesses: SalBench (Dahou et al., 2025) high-

lights diﬃculties with low-level perceptual cues, while BlindTest (Rahmanzadehgervi et al., 2024), SRBench (Stogiannidis et al., 2025), and VisOnlyQA (Kamoi et al., 2025) expose brittle shape, geometry, and spatial reasoning. Despite this progress, relatively less attention has been paid to instrument reading, which requires precise localized visual perception coupled with light numerical computation (e.g., inferring tick intervals, decimal placement, and unit normalization).

Measuring Instruments Reading Reading measuring instruments is challenging because it integrates ﬁnegrained visual perception, text reading, and visuospatial reasoning. Numerous computer vision methods target speciﬁc families of tools such as rulers (Pan et al., 2025), clocks (Yang et al., 2022; Saxena et al., 2025), water meters (Van et al., 2025), pressure gauges (Reitsma et al., 2024), and other analog dials (Howells et al., 2021; Salomon et al., 2022; Shu et al., 2023; Leon-Alcazar et al., 2024). Typical pipelines combine detection/segmentation of scales and pointers, geometric rectiﬁcation, and OCR or tick-interval estimation to map visuals to numeric values and units. However, these approaches are narrowly tailored on moderatescale training/validation data henceforth generalize poorly across device types, design variations, viewpoints, glare/occlusion, and unit ambiguity. More recently, VLMs have been applied to instrument reading: GPT-4o (OpenAI, 2024) reports preliminary ability on industrial gauges, and CAD2DMD-SET (Valente et al., 2025) evaluates several VLMs on digital measurement devices. Yet current evaluations remain fragmented: they cover limited device diversity, emphasize categorical correctness over calibrated numeric error, and seldom assess unit normalization, tolerance bands, or robustness stressors.

### 6 Conclusions and Discussion

We introduced MeasureBench, a comprehensive benchmark with both real-world and synthetic subsets for evaluating vision–language models (VLMs) on instrument reading. Our analyses reveal a persistent limitation of current VLMs: diﬃculty with ﬁne-grained visual cues and precise visual–numeric correspondences, leading to errors in value estimation and unit normalization. The proposed synthetic data generation pipeline serves both as a source of controlled benchmarks and as an eﬀective means of training data augmentation. We also explored reinforcement ﬁnetuning with GRPO. Preliminary results suggest that even small amounts of targeted synthetic data can yield measurable gains that transfer to real-world settings, but only to a moderate extent. We hope this work could help future VLM development with more comprehensive training data curation or better visual representation modeling to enable stronger capabilities in ﬁne-grained understanding, geometric alignment, and spatial reasoning.

### Authors

Fenfen Lin∗, Yesheng Liu∗, Haiyu Xu∗, Chen Yue∗, Zheqi He†, Mingxuan Zhao, Miguel Hu Chen, Jiakang Liu, JG Yao, Xi Yang

### References

Anthropic. Claude opus 4 & claude sonnet 4 — system card. https://www.anthropic.com/

claude-4-system-card, 2025.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24185– 24198, 2024.

†Project lead; ∗ equally contributed to this work. correspondance to: zqhe at baai.ac.cn

Yasser Dahou, Ngoc Dung Huynh, Phuc H. Le-Khac, Wamiq Reyaz Para, Ankit Singh, and Sanath Narayan. Salbench: Vision-language models can’t see the obvious. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025. URL https://salbench.github.io.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in neural information processing systems, 36:49250–49267, 2023.

FlagEval Team. FlagEval ﬁndings report: A preliminary evaluation of large reasoning models on automatically veriﬁable textual and visual questions, 2025. URL https://arxiv.org/abs/2509.17177.

Google Gemini Team. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. URL https://arxiv.org/ abs/2507.06261.

Zheqi He, Yesheng Liu, Jing-Shu Zheng, Xuejing Li, Jin-Ge Yao, Bowen Qin, Richeng Xuan, and Xi Yang. Flagevalmm: A ﬂexible framework for comprehensive multimodal model evaluation. In Pushkar Mishra, Smaranda Muresan, and Tao Yu (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pp. 51–61, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-253-4. URL https://aclanthology.org/2025. acl-demo.6/.

Ben Howells, James Charles, and Roberto Cipolla. Real-time analogue gauge transcription on mobile phone. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2369–2377, 2021.

Juan Izquierdo-Domenech, Jordi Linares-Pellicer, Carlos Aliaga-Torro, and Isabel Ferri-Molla. Towards robust industrial control interpretation through comparative analysis of vision–language models. Machines, 13(9): 759, 2025.

Ryo Kamoi, Yusen Zhang, Sarkar Snigdha Sarathi Das, Ranran Haoran Zhang, and Rui Zhang. VisonlyQA: Large vision language models still struggle with visual perception of geometric information. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=PYHwlyu2fa.

Juan Leon-Alcazar, Yazeed Alnumay, Cheng Zheng, Hassane Trigui, Sahejad Patel, and Bernard Ghanem. Learning to read analog gauges from synthetic data. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 8616–8625, 2024.

Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench-2: Benchmarking multimodal large language models. arXiv preprint arXiv:2311.17092, 2023a.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking

multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023b. Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang,

Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pp. 216–233. Springer, 2024.

Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS), 2022.

Ahmed Masry, Do Long, Jia Qing Tan, Shaﬁq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pp. 2263–2279, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.177. URL https://aclanthology.org/2022.findings-acl.177.

Dalius Matuzevičius. Rulers2023: an annotated dataset of synthetic and real images for ruler detection using deep learning. Electronics, 12(24):4924, 2023.

Meta AI. Llama 4 — models. https://www.llama.com/models/llama-4/, 2025. Oﬃcial Llama 4 model overview.

Mistral AI. Mistral medium 3. https://mistral.ai/news/mistral-medium-3, 2025. Oﬃcial announcement; Medium 3 series.

OpenAI. Gpt-4o system card. https://openai.com/index/gpt-4o-system-card/, May 2024. Accessed:

2025-09-25. OpenAI. Introducing gpt-5, August 2025a. URL https://openai.com/index/introducing-gpt-5/. OpenAI. Introducing openai o3 and o4-mini. https://openai.com/index/introducing-o3-and-o4-mini/,

April 2025b. Accessed: 2025-09-25. Yimu Pan, Manas Mehta, Gwen Sincerbeaux, Jeﬀery A Goldstein, Alison D Gernand, and James Z Wang. Reading a ruler in the wild. arXiv preprint arXiv:2507.07077, 2025. Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, and Others. Humanity’s last exam, 2025. URL https://arxiv.org/abs/2501.14249. Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. Vision language models are blind. arXiv preprint arXiv:2407.06581, 2024.

Maurits Reitsma, Julian Keller, Kenneth Blomqvist, and Roland Siegwart. Under pressure: learning-based analog gauge reading in the wild. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 14–20. IEEE, 2024.

Gabriel Salomon, Rayson Laroca, and David Menotti. Image-based automatic dial meter reading in unconstrained scenarios. Measurement, 204:112025, 2022.

Rohit Saxena, Aryo Pradipta Gema, and Pasquale Minervini. Lost in time: Clock and calendar understanding challenges in multimodal llms. arXiv preprint arXiv:2502.05092, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridﬂow: A ﬂexible and eﬃcient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Yan Shu, Shaohui Liu, Honglei Xu, and Feng Jiang. Read pointer meters based on a human-like alignment and recognition algorithm. In CCF National Conference of Computer Applications, pp. 162–178. Springer, 2023.

Amanpreet Singh, Vivek Natarjan, Meet Shah, Yu Jiang, Xinlei Chen, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 8317–8326, 2019.

Ilias Stogiannidis, Steven McDonagh, and Sotirios A Tsaftaris. Mind the gap: Benchmarking spatial reasoning in vision-language models. arXiv preprint arXiv:2503.19707, 2025.

Liyan Tang, Grace Kim, Xinyu Zhao, Thom Lake, Wenxuan Ding, Fangcong Yin, Prasann Singhal, Manya Wadhwa, Zeyu Leo Liu, Zayne Sprague, Ramya Namuduri, Bodun Hu, Juan Diego Rodriguez, Puyuan Peng, and Greg Durrett. Chartmuseum: Testing visual reasoning capabilities of large vision-language models, 2025. URL https://arxiv.org/abs/2505.13444.

Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms, 2024.

João Valente, Atabak Dehban, and Rodrigo Ventura. Cad2dmd-set: Synthetic generation tool of digital measurement device cad model datasets for ﬁne-tuning large vision-language models. arXiv preprint arXiv:2508.21732, 2025.

Bay Nguyen Van, Anh Nguyen, Kiet Tran Trung, Thien Ho Huong, Ha Duong Thi Hong, Hau Nguyen Trung, and Vinh Truong Hoang. Water meter reading based on text recognition techniques and deep learning. IEEE Access, 2025.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview. net/forum?id=QWTCcxMpPA.

xAI. Grok 4 — model announcement. https://x.ai/news/grok-4, 2025. Charig Yang, Weidi Xie, and Andrew Zisserman. It’s about time: Analog clock reading in the wild. In Proceedings

of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2508–2517, 2022.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In International conference on machine learning. PMLR, 2024.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multidiscipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of ACL, 2025.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, KaiWei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint arXiv:2403.14624, 2024.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

### A Additional Details

#### A.1 Training details

We employ reinforcement ﬁnetuning (RFT) on the synthesis datasets with 3900 images by reinforcement learning framework verl. Following Deepseek-R1, we employ GRPO with a format reward function to optimize the model to output the thinking process within “<think>...</think>". Training is performed on 8×H100 GPUs for 15 epochs with a global batch size of 128 and a learning rate of 1 × 10−6, and a rollout number of 8.

#### A.2 Soft-margin Reward

We also conduct experiments with a soft-margin reward that grants partial credit to predictions near the target interval. For the numeric component of the answer, deﬁne the distance to the interval as

(4)

0, yˆ ∈ [l,r], min(|yˆ − l|, |yˆ − r|), otherwise,

d(ˆy,I) =

and the margin

(5)

r − l, r > l, 0.05l, otherwise,

m(I) =

with a small constant ε > 0. We deﬁne a linearly decaying partial-credit score

ssm(ˆy;I) = 21 max 0, 1 − md((ˆIy,I)+)ε . (6)

The soft-margin reward replaces the value term by taking the better of exact correctness and soft credit, while keeping the formatting term unchanged:

Rsoft = α max call, ssm(ˆy;I) + (1 − α)cfmt. (7)

This keeps rewards consistent with evaluation when the answer is exact, while giving informative feedback to near-miss predictions.

Model/Dataset Overall Value Unit Qwen2.5-VL-7B (Real-world) 14.6 15.0 93.4 Qwen2.5-VL-7B+GRPO (Real-world) 19.7 (+34.9%) 20.4 (+36.0%) 92.3 (-1.2%) Qwen2.5-VL-7B+GRPO-soft (Real-world) 19.6 (+34.2%) 20.4 (+36.0%) 91.9 (-1.6%) Qwen2.5-VL-7B (Synthetic) 10.9 11.5 88.5 Qwen2.5-VL-7B+GRPO (Synthetic) 35.2 (+222.9%) 35.6 (+209.6%) 96.7 (+9.3%) Qwen2.5-VL-7B+GRPO-soft (Synthetic) 35.3 (+223.9%) 35.6 (+209.6%) 98.0 (+10.7%)

Table 9: Results of Qwen2.5-VL-7B variants with GRPO on real-world and synthetic subsets.

As shown in Table 9, the soft-margin reward yields comparable performance to the original hard-margin reward after RFT on both real-world and synthetic subsets. This suggests that the original reward design is already eﬀective, while the soft-margin variant provides an alternative that may be more suitable in scenarios where near-miss predictions are common.

### B Examples of synthesized images

Here we provide additional examples of synthesized images of measuring instruments generated by our framework. Each generator in our framework is expected to render an image of an instrument with similar appearance along with random readout. In Figure 9, 2D images are rendered by oﬄine-only libraries like Pillow, NumPy and Matplotlib, 3D images are rendered by Blender which is more realistic.

[Figure 25]

Figure 9: Additional examples of synthetic measuring instruments generated by our pipeline.

### C 3D modeling with Blender

To construct a large collection of measurement-related 3D assets, we use Blender (v4.2) in combination with publicly available online repositories. The procedure was as follows.

[Figure 27]

Figure 10: Examples of augmentation strategies applied during 3D model acquisition and preparation with Blender (v4.2).

#### C.1 Asset retrieval

We integrate the BlenderKit plugin into Blender to access free 3D assets, including models, HDRs, and materials. For categories underrepresented in BlenderKit (e.g., cylinder, hygrometer), we also retrieved models from Sketchfab. The queries included watches, clocks, scales and rulers, thermometers, covering both pointerbased and linear-scale instruments.

#### C.2 Model normalization

- 1. Pointer-based models (e.g., clocks, scales): In many assets, the pointer was not initially aligned with the zero position. We manually rotated the pointer to zero and reset its transformations (rotation along the x, y, z axes set to 0).
- 2. Linear-scale models (e.g., thermometers): For these, we determined the minimum-maximum mapping on the scale and adjusted the geometry proportionally so that the linear transformations of pointer correctly represented measurement values.

#### C.3 Contextual scene augmentation

Some models only represented the measurement instrument itself, which led to unrealistic renderings when the pointer indicated a nonzero value. To improve semantic consistency, we augmented scenes with additional objects:

- • Scales: To avoid showing a dial reading 1 kg with an empty plate, we placed an additional object (e.g., a fruit model, such as dragon fruit) on the weighing surface.
- • Rulers: Since rulers measure relative length, we included a reference object (a pen). The pen was rescaled and positioned alongside the ruler, allowing queries such as "How long is the pen?" to be grounded in the rendered image.

These contextual additions ensured that pointer readings were visually consistent with the surrounding scene, enhancing dataset realism, and reducing ambiguity for vision-language evaluation.

- C.4 Pointer rotation control Pointer manipulation was automated with Blender’s Python API.

• Clocks: For clocks, rotation angles were computed directly from the target hour, minute, and (optionally) second values:

- 1 second_angle = math.radians(target_second * 6)

- 2 minute_angle = math.radians(target_minute * 6 + target_second * 0.1)

- 3 hour_angle = math.radians((target_hour %

The axis of rotation varied across diﬀerent models (i.e. whether Oxy, Oxz, or Oyz). For example, a clock’s hour hand can be controlled with: hour_hand.rotation_euler = (0, 0, -hour_angle). However, depending on the model, the rotation angle might be applied to the ﬁrst or second component of the Euler tuple rather than to the third.

• Other dials (e.g., hygrometers): For these, the degree of pointer rotation depends on the speciﬁc model geometry. We ﬁrst check for the maximum rotation angle (max.rot.deg) that corresponded to the maximum scale value, and set pointer positions linearly:

- 1 max_rot = math.radians(max_rot_deg)

- 2 rot_z = min_rot + (humidity -min_humidity) / (max_humidity -min_humidity)

- 3 * (max_rot -min_rot) This approach is generalized to other instruments with linear or semi-linear dial mappings.

If the geometry of the model used a nonstandard orientation, we rotated the entire object to align it with the desired axis.

- C.5 Camera alignment

Since the dial panels of many models were not centered at the origin, we applied oﬀsets to position the camera such that it directly faces the dial. Camera distance and angle were tuned empirically to maximize legibility of the dial face and pointer. For small-scale instruments, shorter distances and narrower angle ranges provided clearer renderings, whereas larger instruments beneﬁted from wider perspectives.

- C.6 Lighting and environment HDRs

To ensure consistent illumination across renderings, we used two strategies depending on the dataset requirements:

- • HDR environment maps: For most models, we initialized scenes with background environment maps (.exr ﬁles), either using Blender’s built-in HDRIs or downloading additional ones via BlenderKit. These provided realistic lighting and surface reﬂections. HDRs were ﬁrst manually conﬁgured and later automated using Python.
- • Direct light sources: For cases where a clean background was preferred, we disabled HDRs and instead added light objects from diﬀerent positions (e.g., point lights or area lights). This clearly illuminated the dial while leaving the background neutral.

- C.7 Rendering execution

Scripts were executed either directly within Blender’s Scripting panel or externally via Python (importing the bpy module) in an IDE such as Visual Studio Code. This ﬂexibility enabled large-scale automated rendering of models across diﬀerent instrument categories.

Figure 10 provides illustrative examples of the augmentation strategies described above.

### D Extended Analysis

We provide further analysis of model behavior on speciﬁc challenging cases, as well as statistical distributions of numerical outputs.

- D.1 Reading of complex measuring instruments

[Figure 30]

Figure 11: Results comparison on an electricity meter

Complex measuring instruments with composite readout designs or multiple dials and pointers remain highly challenging for current VLMs: an error at any step of ﬁne-grained visual perception or reading interpretation will propagate to an incorrect ﬁnal result. Figure 11 shows a multi-dial electricity meter with ﬁve dials, each with a pointer. To read the meter, one should note the position of each pointer and record the numbers from left to right, remembering that adjacent dials rotate in opposite directions. If a pointer is between two numbers, the lower number is recorded, unless it is between 9 and 0, in which case 9 is recorded. We present the results of three models, although all of them correctly describe the reading procedure, they still struggle to localize the pointer positions, misreading almost all pointers and thus producing incorrect ﬁnal readings.

- D.2 Example of correct “guessing”

We observed that some correct answers may arise from guessing rather than accurate visual understanding. Figure 12 shows an ammeter whose pointer indicates a reading of 20 A. Both GPT-5 and GPT-5-Mini provide detailed reasoning steps before giving their ﬁnal answers. GPT-5 incorrectly estimates the reading as 26 A, whereas GPT-5-Mini is clearly uncertain and essentially guesses 20 A, which happens to be correct.

### E The “10:10” phenomenon

We found an interesting phenomenon that many models tend to answer “10:10” when reading clock times, regardless of the actual time shown in the image. During the real-world data collection, we deliberately avoid including images with “10:10”, and in the synthesis process,clock times are uniformly sampled. As a result, there are few ground-truth of “10:10” in our benchmark.

[Figure 32]

- Figure 12: An example where GPT-5 answers incorrectly while GPT-5-Mini guesses the correct reading.

Model Real-world Synthetic Qwen2.5-VL-72B-Instruct 72.88% 50.74% GPT-5-Mini 29.66% 7.78% Claude-Sonnet-4 26.27% 16.30% Qwen2.5-VL-7B-Instruct 23.73% 15.56% Qwen2.5-VL-32B-Instruct 21.19% 8.89% GPT-5 20.34% 6.30% Mistral-medium-3.1 16.95% 4.07% InternVL3.5-38B-thinking 16.10% 12.96% Claude-Opus-4.1 13.56% 9.63% InternVL3.5-8B-thinking 12.71% 7.04% Claude-Opus-4.1-thinking 12.71% 10.37% Qwen3-VL-235b-instruct 12.71% 12.96% InternVL3.5-38B 11.86% 10.00% Qwen2.5-VL-3B-Instruct 11.86% 3.70% Gemini-2.5-Pro 11.86% 3.33% Qwen3-VL-8B 7.63% 9.63% Gemini-2.5-Flash 4.24% 1.11% InternVL3.5-8B 3.39% 6.30% Qwen2.5-VL-7B-GRPO 3.39% 1.11% Grok-4 3.39% 4.81% Gemini-2.5-Flash-thinking 2.54% 1.48% LLaMA-4-maverick 0.85% 1.11% LLaMA-4-scout 0.00% 1.85%

Table 10: Proportion of "10:10" responses on clock images in MeasureBench.

However, as shown in Table 10, we calculate the proportion of string “10:10” in the models’s answers on clock images. It is surprising that the powerful open-source model Qwen2.5-VL-72B-Instruct outputs “10:10” for 72.88% of real-world clock images and 50.74% of synthetic clock images. The frontier commercial model

GPT-5 also predicts “10:10” in more than 20% of its answers on real-world clock images. This bias likely stems from training data, where clocks are frequently depicted at “10:10” for aesthetic reasons in advertisements and product listings. To further verify this, we examine the answers of Qwen2.5-VL-7B with RFT training on our synthetic dataset, where clock times follow a uniform distribution. The RFT-trained model predicts “10:10” on only 3.39% of real-world clock images, whereas the original Qwen2.5-VL-7B does so on 23.73% of images, indicating that training with a more uniform distribution can eﬀectively mitigate this bias.

### F Spikes distributions at integers

[Figure 34]

[Figure 35]

- Figure 13: The numeric spikes distribution of InternVL3.5 series on real-world subset.

Figure 14: The numeric spikes distribution of Qwen2.5-VL series on real-world subset.

We further analyze the distribution of numeric outputs from diﬀerent models. Figure 13 and Figure 14 show the spikes at integer values for InternVL3.5 series and Qwen2.5-VL series respectively. We can observe that both models exhibit signiﬁcant spikes at multiples of ten, the same model series have similar distribution patterns, and thinking mode can not mitigate these spikes. This may be attributed to the models’ training data, where round numbers are more frequently represented.

