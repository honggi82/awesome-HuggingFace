# arXiv:2511.02243v1[cs.AI]4Nov2025

WHEN MODALITIES CONFLICT: HOW UNIMODAL REASONING UNCERTAINTY GOVERNS PREFERENCE DYNAMICS IN MLLMS

Zhuoran Zhang1,2 Tengyue Wang2,4 Xilin Gong2,6 Yang Shi1 Haotian Wang5 Di Wang2,3 Lijie Hu7

- 1Peking University
- 2Provable Responsible AI and Data Analytics (PRADA) Lab 3King Abdullah University of Science and Technology 4South China University of Technology 5Tsinghua University 6University of Georgia 7MBZUAI

ABSTRACT

Multimodal large language models (MLLMs) must resolve conflicts when different modalities provide contradictory information, a process we term modality following. Prior work measured this behavior only with coarse dataset-level statistics, overlooking the influence of models’ confidence in unimodal reasoning. In this paper, we introduce a new framework that decomposes modality following into two fundamental factors: relative reasoning uncertainty ( the case-specific confidence gap between unimodal predictions) and inherent modality preference( a model’s stable bias when uncertainties are balanced). To validate this framework, we construct a controllable dataset that systematically varies the reasoning difficulty of visual and textual inputs. Using entropy as a fine-grained uncertainty metric, we uncover a universal law: the probability of following a modality decreases monotonically as its relative uncertainty increases. At the relative difficulty level where the model tends to follow both modalities with comparable probability what we call the balance point, a practical indicator of the model’s inherent preference. Unlike traditional macro-level ratios, this measure offers a more principled and less confounded way to characterize modality bias, disentangling it from unimodal capabilities and dataset artifacts. Further, by probing layerwise predictions, we reveal the internal mechanism of oscillation: in ambiguous regions near the balance point, models vacillate between modalities across layers, explaining externally observed indecision. Together, these findings establish relative uncertainty and inherent preference as the two governing principles of modality following, offering both a quantitative framework and mechanistic insight into how MLLMs resolve conflicting information.

1 INTRODUCTION

Multimodal large language models (MLLMs) (Achiam et al., 2023; Team et al., 2023; Wang et al., 2024; Yin et al., 2024; OpenAI et al., 2024) demonstrate powerful capabilities by processing information from various sources, like images and text, making them vital in applications ranging from web navigation (OpenAI, 2025) to aiding visually impaired users. However, a critical challenge arises when these modalities present conflicting information. For example, an image might show a blue car, while the accompanying text describes it as red. In such cases, the MLLM must resolve the conflict, leading to an observable behavior we term modality following: the model’s final output aligns with the information from one modality over the other.

Prior studies (Zhang et al., 2025; Deng et al., 2025) have typically examined this phenomenon using coarse, dataset-level statistic: the ratio of text-following versus vision-following cases on a given set of conflicting inputs. This approach, however, often attempts to neutralize the model’s unimodal capabilities by filtering for cases where the model can correctly answer based on either modality alone. This overlooks a crucial factor: the model’s confidence in each of its unimodal predictions. For the same instance, one model may produce the correct answer with high confidence while another does so with low confidence. Even within a single model, two different instances can elicit correct unimodal answers but with vastly different certainty levels. Such variations in underlying confidence directly influence the model’s final choice in multimodal settings and, consequently, shape the aggregate statistics of modality-following behavior.

To truly understand the modality-following process, we propose that the static, dataset-level following statistics are emergent properties of two distinct underlying factors: (1) the relative reasoning uncertainty between the two modalities on a case-by-case basis, measured under unimodal inputs, which reflects the model’s confidence gap between text-only and vision-only reasoning, and (2) a more stable, inherent modality preference, which we define as the model’s intrinsic leaning toward one modality when the reasoning uncertainties from both are perceived as equal. This leads to our central hypothesis:

## An MLLM’s modality-following behavior is a dynamic process governed by the interplay between the relative reasoning uncertainty of the conflicting modalities and the model’s own inherent preference.

In simpler terms, a model’s decision to follow the text depends on whether the text’s reasoning advantage (i.e., its low relative uncertainty compared to the image) is significant enough to overcome the model’s potential inherent preference for vision.

We quantified the model’s perceived uncertainty for each unimodal case using the output entropy of its answer token, where a higher value indicates lower confidence (Shannon, 1948; Farquhar et al., 2024; Zhang et al., 2024a; Cao & Ou, 2025).Our overall analysis process is shown in Figure 1. To validate the hypothesis, we constructed a controllable toy dataset that allows us to systematically and independently manipulate the reasoning difficulty of visual and textual inputs, thereby inducing varying levels of uncertainty in unimodal reasoning. The relationship between these two uncertainty scores was then used to define the relative uncertainty, forming the central axis for our analysis.

Our first goal was to verify if relative uncertainty indeed governs the model’s final choice. By analyzing the model’s outputs across our benchmark, we uncovered a clear and predictable pattern. As we systematically increased the reasoning uncertainty of one modality relative to the other, the model’s probability of following that modality showed a consistent monotonic decrease. This finding confirms that modality following is not a fixed attribute but a fluid behavior that predictably shifts with the relative difficulty of unimodal inputs.

However, we observed that a model does not necessarily follow the modality with the lower relative uncertainty. Instead, each model possesses a unique threshold—a subjective balance point of uncertainty that it is willing to tolerate. This balance point reveals the model’s inherent preference. For example, a model with a strong inherent preference for vision might only follow the text if the text is significantly easier to process than the image.

Having established this behavioral relationship, we then sought to understand the internal mechanism behind it. Why does a model hesitate or average its choices when the relative uncertainty is near its subjective balance point? To explore this, we categorized conflict scenarios into two types. In a clear region, where one modality is significantly less uncertain (i.e., much easier) than the other, the model quickly and stably commits to the easier modality in its early processing layers. In contrast, in the ambiguous region where both modalities have a similarly high or low level of uncertainty close to the model’s balance point, the model will hesitate. This is visible internally as “oscillations”, where the model’s top prediction repeatedly switches between the answer suggested by text and the one by vision across its layers. This internal oscillation provides a mechanistic explanation for the externally observed behavior of averaged following in uncertain situations. In summary, this paper makes three key contributions:

- • We propose a new framework that decomposes the observable “modality following” behavior into two core components: case-specific relative reasoning uncertainty and a model’s stable inherent modality preference.
- • Using a novel controllable dataset, we empirically discover a fundamental law: a model’s probability of following a modality monotonically decreases as its relative reasoning uncertainty increases. We show how a model’s inherent preference can be quantified as the balance point on this curve.
- • We uncover the internal mechanism of “oscillation” within the model’s layers, explaining why models hesitate and average their choices in ambiguous scenarios, thus linking internal dynamics to external behavior.

- 2 DEFINING CONFLICTING INPUTS AND QUANTIFYING MODALITY FOLLOWING

Conflicting Inputs. We define a conflicting input as a triplet (I,T,Q) consisting of an image I, a textual description T, and a question Q, such that the unimodal predictions of the MLLM Mθ disagree:

Yv = Mθ(Q,I) ̸= Yt = Mθ(Q,T). Here, Yv and Yt denote the predictions when the model relies solely on the visual or textual modality, respectively. For example in Figure 1 (a), consider the question Q = “What is the color of the square?”. If the image I shows a red square, while the text T states “The color of the square is the same as a morpho butterfly’s wings”, then the image supports the answer “red” whereas the text suggests “blue”. This forms a concrete instance of a conflicting input triplet (I,T,Q). This setting requires the model to resolve contradictory cues and implicitly decide which modality to follow.

Macro-level Metrics for Modality Following. Given a conflicting input x = (I,T,Q), the multimodal prediction is Ym = Mθ(x). We categorize the outcome as vision-following if Ym = Yv, text-following if Ym = Yt, and other otherwise. To quantify the aggregate modality-following behavior on a dataset, we adopt the traditional approach of calculating following ratios. We define the text-following ratio (TFR) and vision-following ratio (VFR) as:

TFR = |{x : Ym = Yt}|

|{x : Ym ∈ {Yv,Yt}}|

, VFR = 1 − TFR.

These ratios offer a simple, macro-level statistic of a model’s aggregate behavior. In subsequent sections, we will deconstruct how these statistics emerge from a deeper interplay between casespecific uncertainty and a model’s inherent preference, which these ratios alone cannot capture.

- 3 PREPARING FOR THE ANALYSIS: A CONTROLLABLE DATASET AND AN UNCERTAINTY METRIC

To systematically investigate our central hypothesis: that modality following is governed by relative uncertainty and inherent preference, we must first establish a controlled experimental setup. This section details the two essential preparations for our analysis: (1) the construction of a novel dataset with independently controllable difficulty levels for both vision and text, and (2) the validation of entropy as the uncertainty metric, to precisely quantify the model’s perceived reasoning difficulty in a fine-grained, modality-comparable manner.

- 3.1 CONSTRUCTING A DATASET WITH CONTROLLABLE DIFFICULTY

Existing benchmarks lack the ability to systematically vary the reasoning difficulty of each modality independently. To overcome this, we built a toy dataset where each multimodal instance is defined by a task type T and two integer-based design tiers, dv and dt, which control the complexity of the visual and textual inputs, respectively.

We use the color recognition task as an example. As shown in Figure 1(a), the visual design tier (dv) modulates perceptual difficulty by adding distractors, shrinking the target object, or introducing oc-

clusions. A low dv might feature a single, clear red square, while a high dv might present it as a

(b) Unimodal Uncertainty Measure

(a) Unimodal Input with dynamic reasoning difficulty

[Figure 1]

[Figure 2]

[Figure 3]

Question Whatcoloristhesquare? Q 𝐻𝑣

[Figure 4]

[Figure 5]

[Figure 6]

| |
|---|

+

[Figure 7]

#### MLLM

Vision Input (Color = Red)

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

|[Figure 13]<br><br>[Figure 14]|
|---|

|[Figure 15]<br><br>[Figure 16]|
|---|

|[Figure 17]<br><br>[Figure 18]|
|---|

𝐻𝑡

Blue Q

+

(c) Preference Dynamics along Relative Uncertainty

𝑑𝑣 = 0 𝑑𝑣 = 1 𝑑𝑣 = 2

Textual Input (Color = Blue)

2 𝐻𝑡 − 𝐻𝑣 𝐻𝑡 + 𝐻𝑣

Conflict Input

Δ𝐻rel =

[Figure 19]

“The square is Blue.”

- 𝑑𝑡 = 0
- 𝑑𝑡 = 1
- 𝑑𝑡 = 2

[Figure 20]

[Figure 21]

[Figure 22]

“The color of square is same as a sapphire; a

| |
|---|

+ Blue+Q

Text-easier Vision-easier Text-prefer Vision-prefer

sapphire is Blue.” “The square has the same color as a morpho butterfly’s wings.”

- Figure 1: Overview of the analytical framework. (a) We create inputs with independently controllable visual (dv) and textual (dt) difficulty. (b) We measure the model’s perceived uncertainty for each modality via output entropy (Hv, Ht). (c) We then use the relative uncertainty (∆Hrel) to analyze the model’s choice when faced with a conflict.

[Figure 23]

- Figure 2: Unimodal Entropy Trends Across Difficulty Tiers. Average unimodal entropy for text (left) and vision (right) as a function of our designed difficulty tiers. Across all models, entropy consistently increases with difficulty, validating its use as a proxy for model-perceived uncertainty and revealing differences in model capabilities.

small, partially obscured object among many other colorful shapes. Similarly, the textual design tier (dt) controls reasoning complexity. A low dt provides a direct (but conflicting) statement (e.g., “The square is blue”), while a high dt requires multi-hop relational reasoning (e.g., “The square shares its color with a morpho butterfly’s wings”). We ensure that the conflicting color mentioned in text never appears among visual distractors, so each modality provides information independently. By systematically pairing different levels of dv and dt, we generate a structured landscape of conflict cases that spans a wide and predictable range of relative difficulty. Further details are in Appendix B.1.

- 3.2 QUANTIFYING PERCEIVED UNCERTAINTY WITH ENTROPY

Entropy as proxy of perceived uncertainty. While design tiers provide a human-interpretable notion of difficulty, our analysis requires a model-centric metric that reflects the model’s own perceived uncertainty. For this purpose, we employ the Entropy of the model’s output distribution over the answer token (Shannon, 1948; Cao & Ou, 2025). Given a unimodal input x (either vision-only or text-only), for example, consider a vision-only input where the question is “What is the color of

the square?” and the image shows a red square. Its uncertainty is:

H(x) = −

p(y | x) log p(y | x),

y∈V

where V is the token vocabulary. A low entropy value indicates a confident, sharp prediction (e.g., the probability for “red” is high, and near zero for other tokens), whereas a high entropy value would suggest that the model also considers alternative tokens (e.g., “orange,” “brown”), revealing greater uncertainty about its own prediction. Since the output is always in the same token space, entropy serves as a unified and comparable measure of perceived uncertainty across both modalities, which we denote as H(v) for vision and H(t) for text.

Analysis of Unimodal Entropy Trends. To validate that entropy reliably captures our designed difficulty, we measured it across different models and tiers, with the results presented in Figure 2. The data provides strong empirical support for our methodology through three key observations. First, entropy consistently increases with higher design tiers (dv,dt), proving it aligns with our intended difficulty structure. This trend is especially clear in the vision modality, where for instance, the LLaVA-v1.6-7B model’s entropy climbs steadily from approximately 0.25 at the lowest difficulty tier to over 1.5 at the highest. Second, the entropy values for both text and vision span a broad and comparable dynamic range from near-zero to over 1.75, which is crucial for creating conflict scenarios with diverse relative uncertainties. Third, and critically, the differences in entropy across models correspond to their known capabilities. The Qwen2.5-VL model, for example, consistently exhibits the lowest entropy, reflecting its strong performance, while we also observe expected scaling trends within model families, such as the LLaVA-v1.5-13B model showing generally lower visual uncertainty than its 7B counterpart.

Conclusion: (1) We construct a novel dataset that allows for the systematic and independent control of reasoning difficulty across visual and textual modalities. (2) Output token entropy is a robust and reliable proxy for a model’s perceived unimodal uncertainty, establishing it as a sound foundation for our analysis.

[Figure 24]

[Figure 25]

(a) Overall macro-level performance (b) Relative uncertainty distribution

- Figure 3: Macro-level modality-following ratios and relative uncertainty distributions of model performance on the dataset.
- 4 MODALITY FOLLOWING IS SHAPED BY RELATIVE UNCERTAINTY

Contradictory Behaviors at the Macro Level. As a first step, we evaluate the modality-following behavior of six MLLMs using the text-following ratio (TFR), as defined in Section 2. The types of MLLMs covers LLaVA1.5 Family (Liu et al., 2024a), LLaVA1.6 Family (Li et al., 2024) and QwenVL family (Wang et al., 2024; Yang et al., 2024; Bai et al., 2025). For this analysis, we focus on the subset of instances where the model answers correctly in both the vision-only and text-only settings. Figure 3a reveals stark, seemingly arbitrary differences between model families. The LLaVA series consistently exhibits a high TFR, appearing strongly text-following. In contrast,

[Figure 26]

(a) TRP decreases monotonically with relative uncertainty (∆Hrel). Each model’s unique balance point (where its curve crosses the 0.5 probability line) quantifies its inherent preference.

[Figure 27]

(b) The monotonic law remains robust when data is split into low-entropy (solid lines) and high-entropy (dashed lines) subsets.

- Figure 4: The relationship between relative unimodal uncertainty (∆Hrel, x-axis) and the probability of following the text modality (Text Preference Ratio, y-axis) for various models.

the Qwen-VL series is more vision-following. This raises a puzzle: why do models exhibit such divergent and seemingly fixed preferences when evaluated on the same dataset?

A Finer Lens: Relative Unimodal Uncertainty. The core flaw in macro-level statistics like TFR is that they ignore the model’s case-by-case reasoning confidence. To capture this, we introduce relative unimodal uncertainty (∆Hrel). For a given conflicting input x = (I,T,Q), we first decouple its components to measure the unimodal uncertainties. We calculate the text-only entropy, H(t), by providing only the text and the question (T,Q) to the model. Similarly, we calculate the vision-only entropy, H(v), by providing only the image and the question (I,Q). The relative uncertainty is the normalized difference between these two values:

2 H(t)(x) − H(v)(x) H(t)(x) + H(v)(x)

∆Hrel(x) =

.

Here, H(t)(x) and H(v)(x) refer to the unimodal entropies derived from the components of the multimodal input x. This metric, ∆Hrel, thus quantifies the model’s perceived confidence gap for each specific input. It is a direct manifestation of the model’s unimodal capabilities, shaped by its architecture and training data. A negative value indicates the model is more confident in the text, while a positive value means it is more confident in the vision. When we plot the distribution of ∆Hrel for the correctly solved cases (Figure 3b), a new puzzle emerges. Despite their different macro-level behaviors, most models face a similar distribution skewed towards negative values, meaning the dataset is, on average, easier for them to process through text. This deepens the mystery: if the underlying difficulty distribution is similar for most models, why are their final choices so different?

A Unified Monotonic Law. The answer emerges when we shift our perspective from aggregate statistics to the dynamic relationship between uncertainty and choice. By plotting the probability of a model following the text modality against the corresponding ∆Hrel for each case, the apparent chaos resolves into a single, unified pattern, as shown in Figure 4a. For all six models, regardless of architecture or scale, the curve shows a smooth, monotonic decrease. In other words, as text becomes harder relative to vision (i.e., as ∆Hrel increases), the probability that the model follows the text steadily and predictably decreases. This discovery directly confirms our central hypothesis from the Introduction: modality following is not a fixed trait but a dynamic behavior governed by relative reasoning uncertainty.

Quantifying Inherent Preference via the Balance Point. While all models obey this monotonic law, their curves are positioned differently along the axis. This leads to our second key insight. We

define the balance point as the ∆Hrel value at which the model is equally likely to follow either modality (a 50% text-following probability). This balance point provides a principled, quantitative measure of the model’s inherent modality preference—the concept we introduced in the Introduction as the model’s intrinsic leaning when reasoning difficulty is equalized. A balance point below zero indicates an inherent vision preference (as text must be significantly easier to be treated as equal), while a point above zero indicates an inherent text preference. This finally allows us to disentangle a model’s fluid, in-the-moment decision-making from its stable, underlying biases.

Reconciling Macro-Level Contradictions. Our framework, which separates unimodal capability (reflected in the ∆Hrel distribution) from inherent preference (the balance point), can now fully explain the apparent contradictions from our initial macro-level analysis. Consider Qwen2-VL, which appears more vision-following than Qwen2.5-VL based on its VFR. Our analysis reveals this is largely a dataset artifact. Qwen2-VL’s stronger visual capabilities on this specific dataset mean that more data points simply fall into the ”vision-is-easier” (positive ∆Hrel) region, mechanically inflating its vision-following stats. However, Qwen2.5-VL has a balance point further to the left (more negative), revealing a stronger inherent vision preference, as it continues to trust vision even when text is substantially easier. Similarly, the difference between LLaVA and Qwen models is not just about capability. While both face a dataset where text is often easier, Qwen models possess a clear inherent vision preference (negative balance point), whereas LLaVA models have a neutral or text-leaning preference (balance point near or above zero). It is this crucial difference in their inherent preference that drives their divergent behaviors, a nuance entirely missed by macro-level metrics.

Robustness and Generality. To test the generality of our findings, we verified that the monotonic law remains stable across different conditions. We split the data into high- and low-entropy subsets (based on the median total entropy). As shown in Figure 4b, both subsets preserve the same monotonic decline, with only minor shifts in balance points: in high-entropy cases, the balance point moves closer to the center, consistent with the intuition that an already uncertain modality is more easily swayed by relative difficulty in the other. Furthermore, evaluations on additional benchmarks, including our attribute-recognition dataset and tasks from the MC2 benchmark, consistently revealed the same monotonic pattern (see Appendix C). This confirms that the relationship between relative uncertainty and modality following is a robust and general principle.

Takeaways: (1) Seemingly arbitrary macro-level following behaviors can be explained by a single, unified principle: the probability of following a modality monotonically decreases as its relative reasoning uncertainty increases. (2) A model’s inherent preference can be quantified as the “balance point” on the relative uncertainty axis, separating it from the confounding effects of unimodal capability and dataset distribution. (3) Traditional macro-level metrics (like TFR/VFR) are misleading because they conflate these two distinct factors: the model’s capabilities and its inherent preference. Our framework successfully disentangles them.

- 5 THE INTERNAL MECHANISM: OSCILLATION IN THE FACE OF AMBIGUITY

Having established a robust behavioral law that modality following is a dynamic function of relative uncertainty, we now turn to the underlying mechanism. Why does a model hesitate and produce averaged following behavior when the relative uncertainty is close to its inherent balance point? In this section, we peer inside the model’s layer-by-layer reasoning process to reveal the internal dynamics of its decision-making. Our analysis demonstrates that the model’s external hesitation is a direct consequence of internal oscillations between the conflicting choices.

Probing Layer-wise Predictions in Ambiguous vs. Clear Regions. To quantify the model’s internal decision process, we conducted two analyses. First, we defined distinct reasoning scenarios. A case is in the ambiguous region if its relative uncertainty ∆Hrel is within a 0.5 radius of the model’s balance point; otherwise, it is in the clear region, where one modality is significantly easier. Second, we tracked the model’s top-1 prediction for the answer token at each layer using a LogitLens-style technique (nostalgebraist, 2020; Zhang et al., 2024b). Finally, to quantify this internal struggle, we define and count the number of oscillations. An oscillation is counted whenever the model’s layer-wise top-1 prediction switches from a vision-supported answer to a text-supported answer, or

[Figure 28]

- Figure 5: A comparison of the average number of concept oscillations for different models. Across all models, the number of oscillations is significantly higher in the ambiguous region (patterned bars) than in the clear region (solid bars).

[Figure 29]

(a) Logit Difference Heatmap Across Model Layers and Relative Uncertainty.

|[Figure 30]<br><br>[Figure 31]|
|---|

“The rectangle is blue.”

“The rectangle's color is the same as a pentagon. The pentagon is blue.”

“The rectangle's color is the same as a peacock's neck.”

- 𝑑𝑡 = 0
- 𝑑𝑡 = 1
- 𝑑𝑡 = 2

[Figure 32]

Vision Input Text Input

(b) Case Study: Impact of Text Uncertainty on Layerwise Confidence Dynamics.

- Figure 6: Visualization of the Model’s Internal Decision-Making Dynamics. In these visualizations, the x-axis represents the model’s layers. The y-axis is the logit difference, calculated as the logits of the text answer minus the logits of the vision answer (logit(Yt) − logit(Yv)).

vice-versa, regardless of any intermittent predictions of irrelevant tokens. For instance, a sequence of layer-wise predictions like ‘vision → irrelevant → text‘ counts as a single oscillation. This robust definition captures the number of times the model vacillates between the two primary conflicting concepts. To ensure our analysis captures true semantic conflict, we also designed a control group with irrelevant conflict, where the text describes a different object with a conflicting attribute (e.g., for a red square, the text becomes “The triangle is blue”). This maintains sentence structure while removing the direct conflict about the target object.

The results shown in Figure 5 reveal that the irrelevant conflict group consistently shows a very low number of oscillations (e.g., 0.35 for LLaVA-1.5-7B), confirming that the struggle is not due to mere sentence structure but to the semantic contradiction itself. More importantly, across all models, the ambiguous region with conflict exhibits significantly more oscillations than the clear region. For LLaVA-1.6-7B, the oscillation count in the ambiguous region (1.43) is nearly double that of the clear region (0.71), providing strong statistical evidence that models vacillate when faced with choices of similar perceived difficulty.

Visualizing Indecision with Logit Difference Heatmaps. To further investigate this internal struggle, we examine the difference in logits between the text-supported answer and the visionsupported answer across all layers. Figure 6a presents a heatmap of this logit difference. The x-axis represents the model’s layers, and the y-axis represents the relative uncertainty ∆Hrel. The heatmap provides two key insights. First, near the center of the y-axis (the ambiguous region), the logit difference remains close to zero for many layers (indicated by the white color), meaning the model is highly uncertain. This numerical indecision is the direct cause of the oscillations. Second, towards the extremes of the y-axis (the clear regions), the color deepens to solid red or blue in the early-tomid layers. This shows that when one modality is clearly easier, the model quickly and confidently commits to its corresponding answer, leading to stable processing.

A Case Study: The Dynamics of Conflict in a Single Image. Finally, we return to a concrete example to demonstrate our findings in action. Figure 6b plots the layer-wise logit difference for a single visual input paired with three text prompts of increasing reasoning difficulty (dt = 0,1,2). By manipulating dt, we effectively place the model into three distinct regions on the relative uncertainty spectrum, revealing its dramatically different internal states. The easy text (dt = 0) places the model in the text-clear region, and its trajectory (the blue line) shows a rapid, stable commitment to the text modality. Conversely, the hard text (dt = 2) pushes the case into the vision-clear region, where the red line decisively commits to vision. Most importantly, the intermediate difficulty text (dt = 1) creates an ambiguous region case; its trajectory (the gray line) visualizes the internal hesitation and oscillation by hovering near the zero-line decision boundary. This single example encapsulates our central thesis: controllable input difficulty (dt) shapes relative uncertainty, which in turn determines the model’s internal state and its final, observable choice.

- 6 RELATED WORK

Processing and Characterizing Conflicting Information. A significant body of research has focused on characterizing how Multimodal Large Language Models (MLLMs) behave when faced with conflicting inputs. Various benchmarks have been developed to probe this phenomenon, revealing a complex and often inconsistent landscape of modality preferences. A frequently reported observation is that many models exhibit a “blind faith” in text, systematically ignoring visual information Deng et al. (2025). However, this tendency is not universal, as other studies demonstrate that preferences can vary significantly across different models and scenarios Zhang et al. (2025); Liu et al. (2024b). Further work with benchmarks like MMIR has focused on the model’s ability to detect and reason about such inconsistencies (Yan et al., 2025). The lack of a consistent principle to explain these varied and often contradictory macro-level observations is a key motivation for our work. Our primary contribution is to move beyond dataset-level statistics by proposing a unifying framework. We explain this apparent variability as an emergent property of two core factors: case-specific relative reasoning uncertainty and a model’s stable inherent preference.

Explaining and Interpreting Conflict Resolution. Another line of research seeks to explain the underlying causes of modality preference. Some studies focus on external factors that can steer a model’s behavior, such as the order of inputs (Deng et al., 2025) or the use of instructional prompts. Others delve deeper, attributing the behavior to internal factors like inconsistencies within the model’s learned knowledge representations Zhu et al. (2024); Golovanevsky et al. (2025). A third approach uses attribution methods, such as those based on Shapley values, to quantify the relative influence of each modality on the final decision (Alishahi et al., 2019; Parcalabescu & Frank, 2022; 2024). While these approaches identify potential causes and influencing factors, they do not fully reveal the dynamic, layer-by-layer computational process through which a model resolves ambiguity. Motivated by this gap, our work provides this missing mechanistic link. We introduce the concept of internal “oscillations” as direct, observable evidence of the conflict resolution process, demonstrating how our high-level framework is physically manifested in the model’s computational dynamics and explains why models hesitate under uncertainty.

- 7 CONCLUSION

Prior investigations of modality following have typically relied on coarse dataset-level statistics, often ignoring how differences in unimodal uncertainty shape aggregate outcomes. Without explicitly accounting for or aligning uncertainty across modalities, such analyses risk conflating a model’s capabilities with its underlying biases. We reframed modality following in MLLMs as a dynamic process shaped jointly by relative reasoning uncertainty and inherent modality preference. Across models and datasets, we uncovered a robust law: the likelihood of following a modality monotonically decreases as its relative uncertainty grows, with the balance point offering a principled measure of inherent preference. Probing layer-wise predictions further revealed that in ambiguous regions near this balance point, models exhibit strong oscillations between modalities, directly explaining their external hesitation. This framework thus disentangles capability from preference and provides a clearer lens for understanding and improving MLLM decision dynamics.

REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Alibaba Cloud / QwenLM. Qwen-Plus (model card / api). https://www. alibabacloud.com/help/en/model-studio/models (model listing) / https://qwen.readthedocs.io (Qwen docs), 2025. Official model page / API reference for Qwen family (model name: “qwen-plus”). Accessed: 2025-09-24.

Afra Alishahi, Grzegorz Chrupała, and Tal Linzen. Analyzing and interpreting neural networks for nlp: A report on the first blackboxnlp workshop. Natural Language Engineering, 25(4):543–557, 2019.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Guiming Cao and Yuming Ou. Ame: Aligned manifold entropy for robust vision-language distillation, 2025. URL https://arxiv.org/abs/2508.08644.

Ailin Deng, Tri Cao, Zhirui Chen, and Bryan Hooi. Words or vision: Do vision-language models have blind faith in text?, 2025. URL https://arxiv.org/abs/2503.02199.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, 2024.

Michal Golovanevsky, William Rudman, Michael Lepori, Amir Bar, Ritambhara Singh, and Carsten Eickhoff. Pixels versus priors: Controlling knowledge priors in vision-language models through visual counterfacts. arXiv preprint arXiv:2505.17127, 2025.

Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Fei-Fei Li, C. Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proc. of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 198–207, 2017. URL https://arxiv.org/abs/1612. 06890. Dataset and generation code: https://github.com/facebookresearch/ clevr-dataset-gen.

Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 26296–26306, June 2024a.

Xiaoyuan Liu, Wenxuan Wang, Youliang Yuan, Jen tse Huang, Qiuzhi Liu, Pinjia He, and Zhaopeng Tu. Insight over sight? exploring the vision-knowledge conflicts in multimodal llms, 2024b. URL https://arxiv.org/abs/2410.08145.

nostalgebraist. interpreting gpt: the logit lens. https://www.lesswrong.com/posts/ AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens, 2020.

OpenAI. Introducing operator. https://openai.com/index/ introducing-operator/, January 2025. Published January 23, 2025; accessed 2025-0514.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Sim´on Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David M´ely, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cer´on Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774.

Letitia Parcalabescu and Anette Frank. Mm-shap: A performance-agnostic metric for measuring multimodal contributions in vision and language models & tasks. arXiv preprint arXiv:2212.08158, 2022.

Letitia Parcalabescu and Anette Frank. Do vision & language decoders use images and text equally? how self-consistent are their explanations? arXiv preprint arXiv:2404.18624, 2024.

Claude E Shannon. A mathematical theory of communication. The Bell system technical journal, 27(3):379–423, 1948.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

Qianqi Yan, Yue Fan, Hongquan Li, Shan Jiang, Yang Zhao, Xinze Guan, Ching-Chen Kuo, and Xin Eric Wang. Multimodal inconsistency reasoning (mmir): A new benchmark for multimodal reasoning models, 2025. URL https://arxiv.org/abs/2502.16033.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. National Science Review, pp. nwae403, 11 2024. ISSN 2095-5138.

Ruiyang Zhang, Hu Zhang, and Zhedong Zheng. Vl-uncertainty: Detecting hallucination in large vision-language model via uncertainty estimation, 2024a. URL https://arxiv.org/abs/ 2411.11919.

Yu Zhang, Jinlong Ma, Yongshuai Hou, Xuefeng Bai, Kehai Chen, Yang Xiang, Jun Yu, and Min Zhang. Evaluating and steering modality preferences in multimodal large language model, 2025. URL https://arxiv.org/abs/2505.20977.

Zhuoran Zhang, Yongxiang Li, Zijian Kan, Keyuan Cheng, Lijie Hu, and Di Wang. Locate-then-edit for multi-hop factual recall under knowledge editing. arXiv preprint arXiv:2410.06331, 2024b.

Tinghui Zhu, Qin Liu, Fei Wang, Zhengzhong Tu, and Muhao Chen. Unraveling cross-modality knowledge conflict in large vision-language models. arXiv preprint arXiv:2410.03659, 2024.

- A THE USE OF LARGE LANGUAGE MODELS (LLMS)

During the preparation of this paper, we used large language models (LLMs) solely as generalpurpose writing assistants. Specifically, LLMs were employed to help refine the clarity, grammar, and readability of our drafts, as well as to suggest alternative phrasings in English. Importantly, all conceptual contributions including the design of research questions, development of methods, execution of experiments, and interpretation of results were conceived and carried out entirely by the authors. The authors carefully reviewed and edited all text suggested by LLMs to ensure accuracy and originality, and we take full responsibility for the final content of the paper.

- B INFORMATION CONFLICT DATASET GENERATION DETIALS

To investigate the external performance and internal mechanisms of multimodal models when dealing with conflicts between image and text information, we constructed two datasets. The first is Color Recognition Dataset, which requires the model to identify the color of geometric shapes placed on a white canvas. The second is Attribution Recognition Dataset, adapted and filtered from the CLEVR(Johnson et al., 2017) dataset, whose task is to identify the material and shape of three-dimensional objects. Both datasets contain multiple task groups. Each group provides images with increasing visual complexity and text descriptions that contradict the image information while exhibiting increasing textual reasoning complexity. By systematically controlling the visual perception complexity (dv) and the textual reasoning complexity (dt), this design constructs conflict scenarios with diverse visual-textual difficulty combinations in a systematic manner.

- B.1 DATASET OVERVIEW

The Color Recognition Dataset consists of 400 groups, each containing 14 images and questions with 3 different types of conflict descriptions. Images with difficulty levels 0–4 are 800×600 pixels, while those with levels 5–13 are 224×224 pixels. The text is divided into three different types, with an average length of 22.7 words. In each group, the same image answer color can be derived from any image information, while the same text answer color which is different from the image answer, can be obtained from any conflict description in the text. The distribution of image answer and text answer is as follows:

- • Image answer Colors: Red(67), Yellow(67), Blue(67), Green(66), Purple(66), Orange(67)

- • Text answer Colors: Red(67), Yellow(66), Blue(67), Green(67), Purple(66), Orange(67)

The Shape subset and the Material subset of the Attribution Recognition Dataset each contain 300 groups. Each group includes 4 images and questions with 3 different types of conflict descriptions. All images are 480×320 pixels, while the text is divided into five different types, with an average length of 30.0 words. In each group, the same image answer attribute can be derived from any image information, while the same text answer attribute which is different from the image answer can be obtained from any conflict description in the text. The distribution of image answer and text answer is as follows:

- • Image answer Shapes: Sphere(108), Cube(100), Cylinder(92)

- • Text answer Shapes: Sphere(100), Cube(92), Cylinder(108)

- • Image answer Materials: Metal(160), Rubber(140)

- • Text answer Materials: Metal(140), Rubber(160)

- B.2 IMAGE GENERATION OF COLOR RECOGNITION DATASET

For each set of 14 images with a progressive difficulty gradient in the Color Recognition Dataset, we used the Python PIL library for rendering. The following is the generation pipeline.

- 1. Initialization: A target shape (e.g., Circle) is randomly selected.
- 2. Color Assignment:

- • Visual Answer Color: One color is randomly assigned to the target shape.
- • Textual Answer Color: A different color is randomly selected as the conflicting textual statement.

- 3. Distractor Generation: Distractor shapes are randomly chosen from the set excluding the target shape. Their colors are randomly selected from the set excluding both the visual and textual answer colors.
- 4. Difficulty Tiers (dv = 0 to 13): Fourteen progressive difficulty levels are defined by target size, number of distractors and occlusion. Parameters are specified in Table1.

Table 1: Visual Difficulty (dv) Tiers Specification

|Difficulty(dv)<br><br>|Target Size<br><br>|# Distractors|Occlusion Rule|
|---|---|---|---|
|0|80-200 pixels<br><br>|0|No occlusion|
|1<br><br>|80-200 pixels<br><br>|1|No occlusion|
|2|80-200 pixels<br><br>|2|No occlusion|
|3|80-200 pixels|3<br><br>|No occlusion|
|4<br><br>|80-200 pixels|4<br><br>|No occlusion|
|5<br><br>|20%-40% of image|7<br><br>|50% occlusion rate|
|6|20%-40% of image|10<br><br>|80% occlusion rate|
|7<br><br>|5%-10% of image|7|50% occlusion rate|
|8<br><br>|5%-10% of image<br><br>|11|80% occlusion rate|
|9<br><br>|4%-6% of image|20<br><br>|30% occlusion rate|
|10<br><br>|4%-6% of image<br><br>|30<br><br>|60% occlusion rate|
|11<br><br>|4%-6% of image<br><br>|40|50% occlusion rate|
|12|4%-6% of image<br><br>|55<br><br>|60% occlusion rate|
|13<br><br>|4%-6% of image|70<br><br>|70% occlusion rate|

Note 1: ”Occlusion rate” refers to the proportion of distractors that visually overlap the target. Different rates for odd/even tiers introduce finer-grained difficulty variation.

- B.3 IMAGE SELECTION OF ATTRIBUTION RECOGNITION DATASET

All images in the Attribution Recognition Dataset were curated from the CLEVR dataset, which contains objects defined by three geometric shapes (cube, sphere, cylinder), two materials (rubber, metal), and eight colors. For each target attribute corresponding to the subset, our selection procedure began by forming all possible attribute–color pairs via the Cartesian product. For each unique pair, we identified images from the CLEVR validation set containing exactly one object matching that specific combination. The selected images were then assigned a difficulty level based on scene complexity, with a fixed number of images sampled per level to construct the final task groups.Table2 shows the various difficulty levels of the pictures.

Table 2: Difficulty levels for image selection

Difficulty(dv) Number of objects in scene Target object size

- 0 3–4 objects large
- 1 6–8 objects large
- 2 6–8 objects small
- 3 ≥10 objects small

- B.4 TEXTUAL MODALITY CONSTRUCTION

The conflict text issues between the Color Recognition Dataset and the Attribution Recognition Dataset share many similarities in terms of structure and pipeline construction. In both cases, we gradually increase the complexity of the textual modality by increasing the number of reasoning steps and converting explicit reasoning into implicit reasoning. The questions within the same group share a fixed target shape with the images of that group, inquire an attribute depending on the

dataset they belong to, and utilize an identical text answer that contradicts the image information. Each textual problem follows the format of: [Conflict Description] + [Question] + [Command].

- • Question: What {attribute} is the {target shape}?

- • Command: Please use one word to answer this question.

For each group, we generate 3 types of conflict description for Color Recognition Dataset and 4 for Attribution Recognition Dataset with increasing difficulty. The Table3 below lists each type and a concise description, where A denotes the target object, T denotes the text answer, B/S1/S2 represent randomly selected objects absent from the image, D represents a real-world instance unambiguously possessing attribute T, and Pos1/Pos2 denote a pair of opposite spatial relations Left and Right.

Table 3: Question types and descriptions (descriptions only)

Difficulty(dt) Type Description x Original No interference description.

- 0 Direct The A is T.
- 1 Indirect simple The A’s {attribute} is the same as a B. The B is T.

- 2 Indirect The A’s {attribute} is the same as a D.
- 3 Space(Attribution Recognition Dataset only)

There is a T S1, on the Pos1 of the S1 is a S2. The A’s {attribute} is the same as the object Pos2 to the S2.

Robustness Processing: To prevent models from solving tasks via superficial pattern matching, texts in Color Recognition Dataset for dt ≥ 0 were paraphrased using Qwen-Plus(Alibaba Cloud / QwenLM, 2025). This process preserved core semantics, reasoning structure, and key information tokens while varying sentence structure, prepositional phrases, and lexical choices.

Control Group Setup:For ablation studies, two types of control data were constructed:

- • Text-Irrelevant: The target shape ‘A‘ in conflict description only is replaced with a randomly chosen non-target shape (e.g., if target is ‘circle‘, replace with ‘triangle‘ or ‘rectangle‘).
- • Image-Irrelevant: The target shape ‘A‘ in the entire text is replaced with a shape never present in the images (‘star‘, ‘cone‘, ‘frustum‘), maintaining the correspondence between the question and the text description while severing the connection with the image.

Rewrite Questions Task

====SYSTEM==== You are a conservative paraphrasing assistant specialized in subtle wording changes. Your goal is to rewrite a single question sentence while preserving *all* facts, *all* explicit instructions, and the exact multi-hop reasoning structure (number of inference steps and intermediate referents). Make only minor wording, grammar, punctuation, and token-count adjustments; do NOT add, remove, or transform factual content or the logical chain.

====USER==== Field type: {FIELD TYPE} Original question: {ORIGINAL QUESTION} Rewrite Instructions (STRICT):

- 1. Output exactly one rewritten question sentence (no explanation, no notes, no extra punctuation before/after).
- 2. Preserve *all* factual propositions and named referents. Do not add or remove facts.
- 3. Preserve the multi-hop reasoning structure:

- - If the original is a single-step (direct), keep it single-step.
- - If it is implicit multi-step (indirect), keep it implicit and do not make steps explicit.
- - If it is explicit multi-hop (indirect simple), keep the same explicit chain of premises and the same number of hops.

- 4. Preserve any explicit answering instruction exactly (e.g., ”Please use one word to answer this question.”).
- 5. Do not change the identity of entities (e.g., ”hexagon”, ”pine tree”, ”circle”) or the target attribute (e.g., ”color”).
- 6. Only rewrite wording, punctuation, and sentence flow to be more natural or shorter, and optionally reduce/increase token count slightly. You can use near-synonyms with very high similarity.
- 7. Avoid introducing pronouns that obscure referents; keep clarity of which object each premise references.
- 8. If the original contains multiple sentences that together form the multi-hop chain, you may combine or split them only if you exactly preserve the same premises and hop order. Output: the single rewritten question sentence (no extra text).

- B.5 ILLUSTRATIVE SAMPLES FROM THE DATASET

To provide a more intuitive understanding of our image-text conflict dataset, we have sampled several image-question pairs from the Color Recognition Dataset and the Attribution Recognition Dataset subsets and presented them in Figure7, Figure8 and Figure9.

- C CURVE OF ALL REMAIN DATASETS

## Group41 Difficulty0

[Figure 33]

Original: Question: What color is the triangle? Command: Please use one word to answer this question. Vision-based Answer: Yellow Text-based Answer:

## Group41 Difficulty3

[Figure 34]

Direct: The triangle is blue. Question: What color is the triangle? Command: Please use one word to answer this question. Vision-based Answer: Yellow Text-based Answer: Blue

## Group41 Difficulty6

[Figure 35]

Indirect simple: The triangle’s color is the same as a pentagon. The pentagon is blue. Question: What color is the triangle? Command: Please use one word to answer this question. Vision-based Answer: Yellow Text-based Answer: Blue

## Group41 Difficulty15

[Figure 36]

Indirect: The triangle’s color is the same as a mailbox in the US. Question: What color is the triangle? Command: Please use one word to answer this question. Vision-based Answer: Yellow Text-based Answer: Blue

- Figure 7: A selection of image-text pairings from a group in the Color Recognition Dataset. The text highlighted in red indicates the descriptions and answers that conflict with the image information.

## Group193 Difficulty0

[Figure 37]

Direct: The cyan rubber object is a cylinder. Question: What is the shape of the cyan rubber object? Command: Please answer with one word. Vision-based Answer: sphere Text-based Answer: cylinder

## Group193 Difficulty2

[Figure 38]

Indirect: The cyan rubber object’s shape is the same as a log. Question: What is the shape of the cyan rubber object? Command: Please answer with one word. Vision-based Answer: sphere Text-based Answer: cylinder

- Figure 8: A selection of image-text pairings from a group in the Shape subset of the Attribution Recognition Dataset. The text highlighted in red indicates the descriptions and answers that conflict with the image information.

Group79 Difficulty1

[Figure 39]

Indirect simple: The Frustum is rubber, blue cube’s material is the same as the Frustum.

Question: What is the material of the blue cube? Command: Please use one word to answer this question. Vision-based Answer: metal Text-based Answer: rubber

Group79 Difficulty3

[Figure 40]

Space: There is a rubber cone, the right of the cone is a wood frustum. The blue cube’s material is the same as the object left to the wood frustum.

Question: What is the material of the blue cube? Command: Please use one word to answer this question. Vision-based Answer: metal Text-based Answer: rubber

- Figure 9: A selection of image-text pairings from a group in the Material subset of the Attribution Recognition Dataset. The text highlighted in red indicates the descriptions and answers that conflict with the image information.

[Figure 41]

(a) Curve of Color Recognition Task in MC2 Datasets.

[Figure 42]

(c) Curve of Attribution Recognition Task in MC2 Datasets.

[Figure 43]

(e) Curve of Attribution Recognition Task in Our Dataset.

[Figure 44]

(b) Curve of Object Recognition Task in MC2 Datasets.

[Figure 45]

(d) Curve of Position Reasoning Task in MC2 Datasets.

[Figure 46]

(f) Curve of Color Recognition Task in Our Dataset with prompts after rewriting.

- Figure 10: Relative uncertainty versus text-following ratio (TFR) curves across multiple datasets, including Color Recognition, Object Recognition, Attribution Recognition, and Position Reasoning from the MC2 benchmark, our CLEVR-derived Attribution Recognition dataset introduced in Section B.1, and the Color Recognition dataset after prompt diversification with Qwen, which introduced in B.4. Across all datasets and models, we consistently observe a monotonic decrease in TFR as relative uncertainty increases, confirming the robustness of the law. Meanwhile, the locations of the balance points vary significantly across datasets due to differences in textual and visual characteristics, which affect the resulting unimodal entropy distributions. These shifts in balance points reflect each model’s inherent preference toward the specific type of data.

