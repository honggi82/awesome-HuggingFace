## arXiv:2307.15771v1[cs.LG]28Jul2023

[Figure 1]

2023-8-1

# The Hydra Effect: Emergent Self-repair in Language Model Computations

###### Thomas McGrath1, Matthew Rahtz1, János Kramár1, Vladimir Mikulik1 and Shane Legg1

1Google DeepMind

We investigate the internal structure of language model computations using causal analysis and demonstrate two motifs: (1) a form of adaptive computation where ablations of one attention layer of a language model cause another layer to compensate (which we term the Hydra effect) and (2) a counterbalancing function of late MLP layers that act to downregulate the maximum-likelihood token. Our ablation studies demonstrate that language model layers are typically relatively loosely coupled (ablations to one layer only affect a small number of downstream layers). Surprisingly, these effects occur even in language models trained without any form of dropout. We analyse these effects in the context of factual recall and consider their implications for circuit-level attribution in language models.

[Figure 2]

Figure 1 | Diagram of our protocol for investigating network self-repair and illustrative results. The blue line indicates the effect on output logits for each layer for the maximum-likelihood continuation of the prompt shown in the title. Faint red lines show direct effects following ablation of at a single layer indicated by dashed vertical line (attention layer 18 in this case) using patches from different prompts and the solid red line indicates the mean across patches. See Section 2 for details.

### 1. Introduction

Ablation studies are a vital tool in our attempts to understand the internal computations of neural networks: by ablating components of a trained network at inference time and studying the downstream effects of these ablations we hope to be able to map the network’s computational structure and attribute responsibility among different components. In order to interpret the results of interventions on neural networks we need to understand how network computations respond to the types of interventions we typically perform. A natural expectation is that ablating important components will substantially degrade model performance (Morcos et al., 2018) and may cause cascading failures that break the network. We demonstrate that the situation in large language models (LLMs) is substantially more complex: LLMs exhibit not just redundancy but actively self-repairing computations. When one layer of attention heads is ablated, another later layer appears to take over its function. We call this

© 2023 DeepMind. All rights reserved

the Hydra effect: when one set of heads is cut off, other heads grow in importance1. We present these results in Section 2.

The Hydra effect (referred to in (Wang et al., 2022) as backup behaviour) complicates our understanding of what it means for a network component to be important because two natural-seeming measures of importance (unembedding and ablation-based measures) become much less correlated than we would naïvely expect. These impact measurements correspond to the direct and total effect, which we introduce in a short, self-contained primer on causal inference in neural networks in Section 3 before performing a more comprehensive quantitative analysis of the Hydra effect in Section 4. Finally we discuss related work in Section 5 and close by hypothesising possible causes for our findings and discuss their implications for future causal analyses of language model computations in Section 6.

### 2. Self-repair and the Hydra effect

##### 2.1. The Transformer architecture for autoregressive language modelling

We want to analyse the computational structure of large autoregressive language models with an decoder-only Transformer architecture. In this work we use a 7 billion parameter model from the Chinchilla family (meaning that the architecture and training setup is identical to that proposed in (Hoffmann et al., 2022) but the model is approximately 7 billion parameters and the training dataset size is scaled down appropriately). An autoregressive language model maps a sequence of input tokens 𝑥≤𝑡 = (𝑥1, . . . , 𝑥𝑡) of length 𝑡 to a probability distribution over the next token 𝑥𝑡+1 using a function 𝑓𝜃

𝑝(𝑥𝑡+1|𝑥≤𝑡) = 𝑓𝜃(𝑥≤𝑡) (1) = softmax (𝜋𝑡(𝑥≤𝑡)) , (2)

where the pre-softmax values 𝜋 are called the logits. The function 𝑓𝜃 is a standard Transformer architecture comprised of 𝐿 layers

𝜋𝑡 = RMSNorm(𝑧𝑡𝐿)𝑊𝑈 (3) 𝑧𝑡𝑙 = 𝑧𝑡𝑙−1 + 𝑎𝑡𝑙 + 𝑚𝑡𝑙 (4) 𝑎𝑡𝑙 = Attn(𝑧𝑙≤−𝑡1) (5)

𝑚𝑡𝑙 = MLP(𝑧𝑡𝑙−1), (6)

where RMSNorm(·) is an RMSNorm normalisation layer, 𝑊𝑈 an unembedding matrix Attn(·) an attention layer (Bahdanau et al., 2014; Vaswani et al., 2017) and MLP(·) a two-layer perceptron. The dependence of these functions on the model parameters 𝜃 is left implicit. In common with much of the literature on mechanistic interpretability (e.g. Elhage et al. (2021)) we refer to the series of residual activations 𝑧𝑖𝑙, 𝑖 = 1, . . . , 𝑡 as the residual stream. For more details on the Transformer architecture in language modelling and the specific details of Chinchilla language models see (Hoffmann et al., 2022; Phuong and Hutter, 2022). As an additional notational shorthand we will typically denote the dependence of network activations on inputs by skipping the repeated function composition and simply writing 𝑧𝑡𝑙(𝑥≤𝑡) (or 𝑎𝑡𝑙(𝑥≤𝑡, 𝑚𝑡𝑙(𝑥≤𝑡)) to denote the activations at layer 𝑙, position 𝑡 due to input string 𝑥≤𝑡 rather than writing the full recursive function composition implied by Equations 3-6.

1Although evocative, we recognise that the use of the name ‘Hydra’ is not completely mythologically accurate: sometimes only one head grows in importance, and as we show in Section 4 the total effect decreases on average, in contrast with the behaviour of the mythological Hydra.

##### 2.2. Using the Counterfact dataset to elicit factual recall

The Counterfact dataset, introduced in (Wang et al., 2022), is a collection of factual statements originally intended to evaluate the efficacy of model editing. The dataset comprises a series of prompts formed by combining tuples of subject, relation, and object (𝑠, 𝑟, 𝑜∗, 𝑜𝑐), where 𝑠 is the subject (e.g. Honus Wagner), 𝑟 the relation (e.g. “professionally plays the sport of”), 𝑜∗ the true object (e.g. baseball), and 𝑜𝑐 some counterfactual claim that makes sense in context. We are interested in the way that language models store and relate factual knowledge so we only use the concatentation of 𝑠 and 𝑟 to form our prompts. This produces prompts whose completion requires factual knowledge that Chinchilla 7B can complete answer correctly.

##### 2.3. Measuring importance by unembedding

One way to assess the effect of a neural network layer is to attempt to map its outputs onto the network output logits. We discuss approaches that use a learned probe in Section 5, however in this work we use the model’s own unembedding mechanism 𝑢 to compute effects. This approach, often referred to as the logit lens, was introduced in (nostalgebraist, 2020) and has also been used in subsequent interpretability research (Dar et al., 2022; Geva et al., 2022a). The GPT-2 model used in the original logit lens analysis had a LayerNorm (Ba et al., 2016) prior to unembedding, however in the Chinchilla model we analyse the unembedding function is an RMSNorm (Zhang and Sennrich,

- 2019) followed by an unembedding matrix 𝑊𝑈: 𝑢(𝑧𝑙) = RMSNorm(𝑧𝑙)𝑊𝑈. (7)

RMSNorm is a simplification of LayerNorm referred to as RMSNorm Zhang and Sennrich (2019) which dispenses with centring and the learned bias term:

###### ∑︁𝑑

1

𝑧 𝜎(𝑧)

𝑧2𝑖 (8)

RMSNorm(𝑧) =

𝐺; 𝜎(𝑧) =

𝑑

𝑖=1

where 𝑧𝑖 is the 𝑖-th element of the vector 𝑧 and 𝐺 is a learned diagonal matrix. In the remainder of this paper, unembedding refers to computing the logit lens distribution 𝜋˜𝑡 = 𝑢(𝑧𝑡𝑙) = RMSNorm(𝑧𝑡𝑙)𝑊𝑈 using the model’s final RMSNorm layer.

##### 2.3.1. Impact metric

Our unembedding-based impact measurement Δunembed will be measured in terms of model logits over the vocabulary 𝑉. Whenever we deal with logits, either from inference by the complete model or from unembedding, we will first centre the logits:

###### ∑︁𝑉

1

𝜋ˆ𝑡 = 𝜋𝑡 − 𝜇𝜋; 𝜇𝜋 =

[𝜋𝑡]𝑗, (9)

𝑉

𝑗=1

where [𝜋𝑡]𝑗 indicates the logit of the 𝑗-th vocabulary element. We measure the impact of an attention layer 𝑎𝑙 on the logit of the maximum-likelihood token at the final token position 𝑡:

Δunembed,𝑙 = 𝑢ˆ(𝑎𝑡𝑙)𝑖; 𝑖 = argmax

[𝜋𝑡]𝑗, (10)

𝑗

where 𝑢ˆ denotes the centred version of the logits obtained by unembedding as in Equation 9. We also repeat the procedure for the MLP layers 𝑚𝑙, 𝑙 = 1, . . . , 𝐿. We fix the RMSNorm normalisation

factor 𝜎 to the value attained in the forward pass, i.e. 𝜎 = 𝜎(𝑧𝑡𝐿) where 𝐿 denotes the final layer of the network. Fixing the normalisation factor means that the effect of each layer output on the logits is linear, and corresponds to the unembedding that is actually carried out during the model’s forward pass. As we will show in Section 3, unembedding in this way corresponds to the direct effect from causal inference.

##### 2.4. Measuring importance by ablation

An alternative approach to investigating a layer’s function is to ablate it by replacing its output with some other value, for example replacing 𝑎𝑡𝑙 with an identically-shaped vector of zeros during network inference would ablate 𝑎𝑡𝑙. This approach has appealing parallels to standard methodologies in neuroscience and has been suggested as a gold standard for evidence in interpretability research (Leavitt and Morcos, 2020). We naturally expect that ablating components that are important for a given input will lead to degraded performance on that input (indeed this seems like a reasonable definition of importance) and if a network’s internal mechanisms generalise then the ablation will also lead to degraded performance on other similar inputs.

##### 2.4.1. Intervention notation & impact metric

In keeping with standard notation in causality (Glymour et al., 2016; Pearl, 2009) we indicate replacing one set of activations 𝑎𝑡𝑙 with another using the do(·) operation. Here we need to introduce some additional notation: we refer to specific nodes in the model’s compute graph using capital letters and their actual realisation on a set of inputs with lowercase. For example: 𝐴𝑡𝑙 refers to the attention output at layer 𝑙 and position 𝑡, whereas 𝑎𝑡𝑙(𝑥≤𝑡) refers to the value of this vector when the model is evaluated on inputs 𝑥≤𝑡 (when the dependence on inputs is omitted for brevity it should be either clear from context or not relevant). If we have inputs 𝑥≤𝑡 that result in logits 𝜋𝑡(𝑥≤𝑡) we would write the value of 𝜋𝑡(𝑥≤𝑡) following an intervention on 𝐴𝑡𝑙 as

𝜋𝑡 𝑥≤𝑡|do(𝐴𝑡𝑙 = 𝑎˜𝑡𝑙) = 𝜋𝑡 𝑥≤𝑡|do 𝐴𝑡𝑙 = 𝑎𝑡𝑙(𝑥′≤𝑡) (11) for some alternative input 𝑥′

≤𝑡 (see Appendix A for details of how alternative inputs can be chosen). As with the unembedding impact measure Δunembed we measure the impact of ablation Δablate on the centred logit 𝜋ˆ of the maximum-likelihood token 𝑖 for a given input 𝑥≤𝑡 (see Section 2.3.1). To compute Δablate,𝑙 of attention layer 𝑙, token position 𝑡 we instead compare the centred logit of the maximum-likelihood token 𝑖 before and after intervention:

Δablate,𝑙 = 𝜋 ˆ𝑡(𝑥≤𝑡|do(𝐴𝑡𝑙 = 𝑎˜𝑡𝑙)) − 𝜋ˆ𝑡(𝑥≤𝑡) 𝑖 . (12)

As we will show in Section 3, Equation 12 corresponds to measuring the total effect of 𝐴𝑡𝑙 in the context 𝑥≤𝑡. We used resample ablation (Chan et al., 2022) with patches from 15 alternative prompts from the dataset to provide ablation activations 𝑎˜𝑡𝑙 (see Appendix A for more details).

##### 2.5. Ablation-based and unembedding-based importance measures disagree in most layers

We calculate the importance of each layer according to both ablation and unembedding-based measures of importance for all layers of Chinchilla 7B for all prompts in the Counterfact dataset. For each prompt we calculate Δunembed,𝑙 and Δablate,𝑙 at the final token position for every attention and MLP layer.

These results are shown in Figure 2, which shows average and per-prompt results, demonstrating a substantial disagreement between Δunembed and Δablate. This is surprising as we would expect that

[Figure 3]

[Figure 4]

(a) (b)

[Figure 5]

[Figure 6]

(c) (d)

- Figure 2 | Measurements of (a) unembedding-based and (b) ablation-based impact for MLP and attention layers for a 7B parameter language model on the Counterfact dataset. Grey lines indicate per-prompt data, blue line indicates average across prompts. (c) Comparison of unembedding- and ablation-based impact measures across all layers showing low correlation and Δunembed > Δablate for most prompts and layers, contrary to expectations. (d) Quantification of correlation between Δunembed and Δablate and proportion of prompts where Δunembed > Δablate across layers.

ablating a layer not only destroys its impact on the logits from unembedding but potentially also breaks further downstream network components, so we would expect that the ablation-based measure would be greater than or equal to the unembedding measure. As Figure 2 shows, this is far from the case. We now demonstrate that the lack of correlation between ablation and unembedding measures of component importance at all but the late layers of the network is due to downstream changes in layer outputs that counteract the effect of ablations.

##### 2.5.1. Methodology

In order to understand the mechanism behind the difference in results from ablation and unembedding methods we propose a simple methodology that allows us to localise changes in network computation. First, for each input 𝑥≤𝑡 we compute attention and MLP outputs 𝑎𝑡𝑙(𝑥≤𝑡) and 𝑚𝑡𝑙(𝑥≤𝑡) for all layers

[Figure 7]

[Figure 8]

- Figure 3 | Example results of self-healing computations in Chinchilla 7B showing per-layer impact on final logits before and after ablation.

𝑙. We refer to this as the clean run, matching the terminology from (Meng et al., 2022). We then compute the unembedding-based impact Δunembed,𝑙 as defined in Section 2.3.1. This gives us a perlayer measure of the impact of each layer on the maximum-likelihood logit in the clean run. We then ablate a specific attention or MLP layer 𝑘 using resample ablation (see Appendix A). We refer to this

- as the layer 𝑘 ablation run. We now compute Δunembed,𝑙 for each layer 𝑙 and layer 𝑘 ablation run. We denote a specific unembedding layer 𝑙 and ablation layer 𝑘 by Δ˜unembed𝑘 ,𝑙:

Δ˜unembed𝑘 ,𝑙 = 𝑢 𝑎𝑡𝑙 | do(𝐴𝑡𝑘 = 𝑎˜𝑡𝑘) . (13) Because the transformer network is a feedforward network, if a readout layer 𝑙 is not causally downstream of the ablation layer 𝑘 then Δ˜unembed𝑘 ,𝑙 = Δunembed,𝑙. If 𝑘 = 𝑙 then Δ˜unembed𝑘 ,𝑙 ≈ 0 because the output of that layer will be ablated (the approximate equality is due to the fact that the resample ablation is stochastic and so may not fully zero out the centred logit of the maximum-likelihood token).

This methodology, depicted in Figure 1, measures the impact of each layer on the maximum-likelihood token before and after ablation. This allows us to determine how individual layers react to a given ablation.

##### 2.5.2. Results

Sample results for attention ablation are shown in Figure 1 and Figure 3 showing ablations of layers with high Δunembed. Several observations stand out from this data:

Resample ablations work but are noisy: Resample ablations successfully reduce Δunembed of the ablated layer to approximately zero when averaged across patches, although there is substantial

variance between the effect of each patch. This demonstrates that resample ablations can provide true ablations but require a reasonably large sample size to prevent excessive noise.

The Hydra effect occurs in networks trained without dropout: In both cases an attention layer downstream of the ablated layer (layer 20 in the examples shown) substantially increases its impact in

the ablated network compared to the intact network, i.e. Δ˜unembed𝑚 ,𝑙 > Δunembed,𝑙 for some unembedding layer 𝑙 > 𝑚. These are examples of the Hydra effect: we cut off some attention heads, but others grow

their effect to (partially) compensate. The Chinchilla-like language model we study here was trained entirely without dropout, stochastic depth, or layer dropout.

Downstream effects of ablation on attention layers are localised: The effects of attention ablation on downstream layers are localised: apart from the subsequent attention layer involved in the Hydra effect, other attention layers have Δunembed almost entirely unchanged in the ablated network. This does not necessarily imply that the features that would have been created by the ablated layer are unused in later layers, however (although it is consistent with this hypothesis). It is possible that the Hydra effect is not just compensating for the impact of the ablated layer on logits, but is also replacing the missing features that the ablated layer would have provided. More detailed multiple ablation studies would be needed to distinguish between these hypotheses.

Downstream MLPs may perform erasure/memory management: The shape of the unembedding impact curve across layers for the MLP layers remains very similar, but the impact of many of the layers is attenuated. This suggests that these MLPs are performing an erasure/memory-management role: when the attention layer has a high positive impact they have a high negative impact and when the attention layer’s Δunembed is reduced theirs is similarly attenuated.

Although these observations are surprising, the evidence we have presented here is largely anecdotal. Before expanding our analysis to the full Counterfact dataset and analysing all layers in Section 4 we first introduce basic tools of causal inference and reframe our analyses in terms of concepts from causality in Section 3.

### 3. Neural networks as causal models: the compute graph is the causal graph

This section introduces structural causal models, causal graphs and the idea of interventions. The central idea of this section is that we can consider the internal structure of neural networks as structural causal models and use tools from causality to analyse their internal workings. One thing we are not claiming is that neural networks are naturally forming causal models of their training data, or that networks learn to perform causal reasoning - the degree to which this happens in practice or forms the basis for successful generalisation is still not well understood.

##### Causal models

- Definition 3.1 (Causal model). A structural causal model 𝑀 is a tuple 𝑀 = ⟨𝑈,𝑉, 𝐹, 𝑃(𝑢)⟩ where:

1. 𝑈 and 𝑉 are sets of variables called the exogenous and endogenous variables respectively. We follow the standard convention of writing a random variable in uppercase and a specific realisation of that variable in lowercase, so 𝑢 is a realisation of the exogenous variable 𝑈.

next token

tokens

endogenous variables

exogenous variables

A series of

- Figure 4 | Viewing a transformer network as a structural causal model.

- 2. 𝐹 is a set of deterministic functions where 𝑓𝑖 ∈ 𝐹 determines the value of 𝑉𝑖 from a subset of variables Pa𝑖 ∈ 𝑈 ∪ 𝑉 \ {𝑉𝑖}, i.e. 𝑣𝑖 = 𝑓𝑖(pa𝑖).
- 3. 𝑃(𝑢) is a probability distribution over exogenous variables 𝑈.

In a structural causal model the exogenous variables 𝑈 represent randomly-set external conditions (the ‘context’) and the endogenous variables 𝑉 follow deterministically from the context 𝑢. Given a setting of exogenous variables 𝑈 = 𝑢, the value of an endogenous variable 𝑉𝑖 is fully determined. We write the value 𝑉𝑖 takes in this situation as 𝑉𝑖(𝑢) to reflect this rather than write the full set of functions relating 𝑉𝑖 to 𝑈 for conciseness and generality.

We can visualise structural causal models as directed graphs where an edge 𝑋𝑖 → 𝑋𝑗 (for 𝑋𝑖, 𝑋𝑗 ∈ 𝑈 ∪ 𝑉) exists if 𝑋𝑖 ∈ pa𝑖, i.e. if 𝑋𝑖 is one of the variables that directly determines the value of 𝑋𝑗. The important factor distinguishing 𝑈 from 𝑉 is that variables in 𝑈 have no in-edges: there is nothing apart from the probability distribution 𝑃(𝑢) that determines their value. Variables in 𝑉, on the other hand, can eventually trace their ancestry back to one or more values in 𝑈 (although in large models it may take many edges to do so).

Interventions & counterfactuals in causal models In a causal model an intervention do(𝑍 = 𝑧′) alters the value of an endogenous variable 𝑍 from whatever value it would have taken based on its parents 𝑧 = 𝑓𝑍(pa𝑍) to the new value 𝑧′. This entails a change in the causal model 𝑀 to a new intervened-upon model 𝑀𝑍 where 𝑀𝑍 is identical to 𝑀 except that 𝑓𝑍(pa𝑍) = 𝑧′ regardless of the values taken by pa𝑍: the function 𝑓𝑍 has been replaced by a constant function returning 𝑧′. In terms of the causal graph this leads to a removal of in-edges to 𝑍 (as it no longer depends on other elements of the graph) and the addition of a new intervention edge (see Figure 5 for some examples). We express this intervention using the do-operator (Pearl, 2009), where 𝑌(𝑢 | do(𝑍 = 𝑧′)) denotes the value that 𝑌 takes in the modified model 𝑀𝑍 given the context 𝑢.

[Figure 9]

- Figure 5 | Interventions and types of effect illustrated with a two-layer residual network example. The total effect involves intervening on a node and allowing the changes due to that intervention to propagate arbitrarily, whereas the direct effect only allows those changes to propagate via the direct path between intervention and output. The indirect effect is the complement of the direct effect: effects are allowed to flow via all but the direct path.

Treating language models as causal models Given these preliminaries the correspondence between neural networks and structural causal models is hopefully clear: the input data 𝑥 correspond to the exogenous variables 𝑈 and the network’s activations (𝑧, 𝑎, 𝑚) and outputs 𝜋 correspond to the endogenous variables 𝑉. In autoregressive transformer-based language models causal masking ensures that activations at input position 𝑡1 are never causal parents of activations at input position 𝑡2 if 𝑡1 > 𝑡2. For the remainder of this work we will assume a standard Transformer architecture, as shown in Figure 4. Intervening on neural networks is identical to intervening in a SCM: we set the nodes to their intervention values and propagate the effects of this intervention forwards. Although for concreteness we focus on language modelling using transformers, the general framework of causal analysis is applicable to deterministic feedforward neural networks more generally - all that changes is the structure of the graph. Stochastic networks such as VAEs or diffusion models can be incorporated by adding additional exogenous nodes corresponding to sources of noise.

##### 3.1. Direct, indirect, and total effects

We now have the notation and concepts needed to define and compute several important types of effects in causal graphs: the total, direct, and indirect effects. When we define the different types of effect in a causal model we will always consider the effects in a given context (i.e. a given setting of exogenous variables 𝑢).

- Definition 3.2 (Total effect). The total effect of altering the value of a variable 𝑍 from 𝑍 = 𝑧 to 𝑍 = 𝑧′ on another variable 𝑌 in a context 𝑢 is given by

𝑇𝐸(𝑧 → 𝑧′,𝑌, 𝑢) = 𝑌(𝑢|do(𝑍 = 𝑧′)) − 𝑌(𝑢|do(𝑍 = 𝑧)). (14)

Total effect corresponds to ablation-based impact If we choose 𝑌 = 𝜋ˆ𝑡 (i.e. the random variable we measure the effect of the intervention on is the centred logit of the maximum-likelihood token 𝑖 at

the final token position 𝑡) and the exogenous variables are the prompt 𝑢 = 𝑥≤𝑡 then we the total effect of an intervention to change activations 𝐴𝑡𝑙 from their ‘natural’ 𝑎𝑡𝑙 = 𝑎𝑡𝑙(𝑥≤𝑡) to an intervened-upon value 𝑎˜𝑡𝑙 = 𝑎𝑡𝑙(𝑥′

≤𝑡) is

𝑇𝐸(𝑎𝑡𝑙 → 𝑎˜𝑡𝑙, [𝜋ˆ𝑡]𝑖, 𝑥≤𝑡) = [𝜋ˆ𝑡(𝑥≤𝑡|do(𝐴𝑡𝑙 = 𝑎˜𝑡𝑙))]𝑖 − [𝜋ˆ𝑡(𝑥≤𝑡|do(𝐴𝑡𝑙 = 𝑎𝑡𝑙))]𝑖 (15) = [𝜋ˆ𝑡(𝑥≤𝑡|do(𝐴𝑡𝑙 = 𝑎˜𝑡𝑙)) − 𝜋ˆ𝑡(𝑥≤𝑡|do(𝐴𝑡𝑙 = 𝑎𝑡𝑙))]𝑖 (16) = [𝜋ˆ𝑡(𝑥≤𝑡|do(𝐴𝑡𝑙 = 𝑎˜𝑡𝑙)) − 𝜋ˆ𝑡(𝑥≤𝑡)]𝑖 (17) = Δablate,𝑙(𝑥≤𝑡). (18)

where we go from Equation 16 to 17 by using the fact that the intervention do(𝐴𝑡𝑙 = 𝑎𝑡𝑙) doesn’t change 𝐴𝑡𝑙 from the value it would have taken in the context 𝑥≤𝑡, as in Section 2.4.1. This shows that our ablation-based impact measure corresponds to measuring the total effect due to a change from the natural activation distribution to one sampled from the ablation distribution. The total effect of ablation (knockout) measures the importance of a unit in a given inference: if we were to ablate it, how much would performance suffer if the effects of the ablation were to cascade through the network.

- Definition 3.3 (Direct effect). The direct effect of altering 𝑍 = 𝑧 → 𝑍 = 𝑧′ is given by 𝐷𝐸(𝑧 → 𝑧′,𝑌, 𝑢) = 𝑌 (𝑢|do(𝑍 = 𝑧′, 𝑀 = 𝑚(𝑢))) − 𝑌(𝑢|do(𝑍 = 𝑧)) (19)

i.e. the effect of intervening to set 𝑍 = 𝑧′ and then resetting all other variables 𝑀 = 𝑉 \ {𝑍,𝑌} to the value they would have obtained in the context 𝑢. As with the total effect, if 𝑧 = 𝑧(𝑢) then the direct effect reduces to

𝐷𝐸(𝑧 → 𝑧′,𝑌, 𝑢) = 𝑌 (𝑢|do(𝑍 = 𝑧′, 𝑀 = 𝑚(𝑢))) − 𝑌(𝑢). (20)

The direct effect measures how much changing the unit’s value would affect the output if all other units’ values were held constant. Because of this, in a language model only units connected via a residual path to the output (i.e. at the same token position) can have a direct effect - all other units must have their effect mediated by at least one attention head in order to cross between token positions. The residual structure of our language models means that effects on logits are additive (up to a normalisation constant introduced by RMSNorm), and so every change in logits eventually occurs due to a direct effect.

Unembedding-based impact with RMSnorm held constant approximates direct effect To see the relation between the unembedding-based impact Δunembed,𝑙 and the direct effect of an attention variable 𝐴𝑡𝑙 on the logits 𝜋ˆ𝑡 in context 𝑥≤𝑡 we first rewrite the architecture defined by Equations 3-6 in the ‘unrolled view’ introduced by Veit et al. (2016) (for more details on this mathematical perspective on transformers see (Elhage et al., 2021)):

∑︁𝐿

𝑚𝑡𝑙(𝑥≤𝑡) + 𝑎𝑡𝐿(𝑥≤𝑡), (21)

𝑧𝑡𝐿(𝑥≤𝑡) =

𝑙=1

where 𝐿 is the final layer of the network and we leave the dependence of activations at layer 𝐿 on earlier layers implicit. To get the logits we simply unembed 𝑧𝑡𝐿:

𝜋𝑡(𝑥≤𝑡) = RMSNorm(𝑧𝑡𝐿)𝑊𝑈 (22)

𝑧𝑡𝐿(𝑥≤𝑡) 𝜎(𝑧𝑡𝐿)

𝐺𝑊𝑈 (23)

=

###### ∑︁𝐿

1 𝜎(𝑧𝑡𝐿)

𝑚𝑡𝑗(𝑥≤𝑡) + 𝑎𝑡𝑗(𝑥≤𝑡) 𝐺𝑊𝑈, (24)

=

𝑗=1

where 𝐺 is the RMSNorm gain matrix and 𝑊𝑈 is the unembedding matrix (see Sections 2.1 and 2.4.1). Equation 24 demonstrates that the logits (and thus the centred logits 𝜋ˆ) are linear in the layer outputs

so long as the RMSNorm scale factor 𝜎(𝑧𝑡𝐿) is held constant. Now we can compute the direct effect of ablating attention layer output 𝑎𝑡𝑙 on the logits (we omit centring as in Equation 9 here for brevity, but it is trivial to include):

𝐷𝐸(𝑎𝑡𝑙 → 𝑎˜𝑡𝑙, 𝜋𝑡, 𝑢) = 𝜋𝑡(𝑥≤𝑡|do(𝐴𝑡𝑙 = 𝑎˜𝑡𝑙, 𝑀 = 𝑀(𝑥≤𝑡))) − 𝜋𝑡(𝑥≤𝑡) 𝑖 (25)

𝑚𝑡𝑗(𝑥≤𝑡) + 𝑎𝑡𝑗(𝑥≤𝑡) 𝐺𝑊𝑈  𝑖

= 

∑︁𝐿

∑︁𝐿

1 𝜎(𝑧𝑡𝐿)

𝑎 ˜𝑡𝑙 + 𝑚𝑡𝑙(𝑥≤𝑡) +

𝑚𝑡𝑗(𝑥≤𝑡) + 𝑎𝑡𝑗(𝑥≤𝑡) −

 

𝑗=1

𝑗≠𝑙

(26)

𝑎 ˜𝑡𝑙 − 𝑎𝑡𝑙(𝑥≤𝑡) 𝜎(𝑧𝑡𝐿)

(27)

=

𝐺𝑊𝑈

𝑖

= 𝑢(𝑎˜𝑡𝑙)𝑖 − 𝑢(𝑎𝑡𝑙(𝑥≤𝑡))𝑖. (28)

The only difference (up to centring) between the direct effect computed above and Δunembed,𝑙 is the inclusion of the impact of the ablation on the maximum-likelihood token 𝑢(𝑎˜𝑡𝑙)𝑖. This factor is typically negligible if the source of resample ablations are chosen correctly (otherwise the ablation would still be substantially increasing the probability of the maximum-likelihood token) and is zero in the case of zero ablations, in which case Δunembed,𝑙 = 𝐷𝐸(𝑎𝑡𝑙 → 𝑎˜𝑡𝑙, 𝜋ˆ𝑡, 𝑢).

- Definition 3.4 (Indirect effect). The indirect effect of altering 𝑧 → 𝑧′ is given by

𝐼𝐸(𝑧 → 𝑧′,𝑌, 𝑢) = 𝑌(𝑢|do(𝑍 = 𝑧, 𝑀 = 𝑚˜) − 𝑌(𝑢|do(𝑍 = 𝑧)); 𝑚˜ = 𝑚(𝑥′≤𝑡) (29)

which is the effect of setting the variables 𝑀 to their ablation values while also resetting 𝑍 to its default value 𝑧.

Indirect effect measures the effect that a unit has via downstream units, i.e. variables that are on the path from 𝑍 to 𝑌 (we say a variable 𝑀 satisfying this criterion mediates the relationship between 𝑍 and 𝑌). Units inside a circuit that is important in the current context will have high indirect effect, whereas units at the terminus of the circuit will have high direct effect. When we don’t specify the mediating variables we assume that all variables between the intervened variables and the output variables are changing.

##### 3.2. Challenges and opportunities in intervening in neural networks

The difficulties and affordances involved in performing causal analysis on neural networks are almost the opposite of those involved in most real-world causal analysis: we know the causal model with complete certainty (down to the level of individual parameters), can perform arbitrarily long chains of interventions rapidly and can read the value of all variables simultaneously. On the other hand, our causal models involved enormous numbers of individual parameters, and these parameters have no obvious meaning attached to them (with the exception of input and output variables). Painstaking analysis often suggests meanings for individual neurons or clusters of neurons (Bau et al., 2018; Carter et al., 2019; Goh et al., 2021; Hilton et al., 2020) but the correct unit of analysis still remains unclear (Morcos et al., 2018). A recent line of work on a phenomenon known as superposition has begun to shed some light on how directions in activation space relate to meaningful units of analysis (Elhage et al., 2022) but this work has yet to reach any definitive conclusions that allow us to decide how to group neurons to reduce the size of the graph. For this reason we work with ablations at the level of individual layers, while acknowledging that this level of analysis is still likely to be too coarse to capture all the relevant phenomena.

[Figure 10]

- Figure 6 | Network motifs that cause total and direct effect to be uncorrelated, either by self-repair or erasure. An arrow-ended line 𝑋 → 𝑌 indicates that 𝑋 increases 𝑌 whereas a bar-ended line indicates that 𝑋 inhibits 𝑌.

- 3.3. Toy model and motif for the Hydra effect

We now turn the tools of causal inference that we have just introduced to our findings on self-repair by studying a simplified toy model of the phenomena we observed in Section 2. These toy models both use a simple causal model involving a single exogenous variable 𝑢 and endogenous variables 𝑥(𝑢) = 𝑢 and 𝑦:

𝑦 = 𝑥(𝑢) + 𝑓 (𝑥, 𝑢). (30)

If we intervene to set 𝑥 = 0, what functions 𝑓 will ensure that the total effect 𝑇𝐸(𝑥, 𝑦) is zero? Two simple possibilities stand out:

𝑓 (𝑥, 𝑢) = −𝑥 (Erasure) (31) which will ensure that 𝑦 = 0 regardless of the input variable 𝑢, or

𝑓 (𝑥, 𝑢) =

0 if 𝑥 = 𝑢 𝑢 otherwise

(Self-repair). (32)

In both cases the output variable 𝑦 is kept unchanged regardless of the inner workings of the model, but in the erasure case it is clamped to zero whereas in the self-repair case it is fixed at 𝑢. Although these simple circuits are stylised in order to illustrate a point, they turn out to be surprisingly good models of phenomena we observe in real language models.

- 4. Quantifying erasure and the Hydra Effect

- 4.1. Methodology

For a given context 𝑢 we can measure the total compensatory effect following an ablation 𝑎˜𝑚 by summing the effects of the downstream changes in the network:

###### ∑︁𝐿

∑︁𝐿

𝐶𝐸(𝑎˜𝑚, 𝑢) =

###### Δ𝐷𝐸(𝑎𝑙, 𝑢, 𝑎˜𝑚)

###### Δ𝐷𝐸(𝑚𝑙, 𝑢, 𝑎˜𝑚)

, (33)

+

𝑙=𝑚+1

𝑙=𝑚

Downstream effect on MLPs

Downstream effect on Attns

where Δ𝐷𝐸 is the difference in direct effects between the ablated and unablated networks: Δ𝐷𝐸(𝑧𝑙, 𝑢, ˜𝑧𝑚) = 𝐷𝐸ablated(𝑧𝑙, 𝑢, ˜𝑧𝑚) − 𝐷𝐸(𝑧𝑙, 𝑢), (34)

where 𝐷𝐸ablated(𝑧𝑙, 𝑢, ˜𝑧𝑚) = Δ˜unembed𝑘 ,𝑙 is the direct effect of layer 𝑙 following an ablation at layer 𝑚 in context 𝑢. The starting index of the downstream MLPs and attention layers differs because MLP

layers follow attention layers in our model. The compensatory effect for MLP ablations 𝐶𝐸(𝑚˜𝑚, 𝑢) is identical to Equation 33 except that the MLP sum index starts at 𝑚 + 1. We generate a dataset of direct and compensatory effects by computing 𝐷𝐸(𝑎𝑙, 𝑢), 𝐷𝐸(𝑚𝑙, 𝑢), 𝐶𝐸(𝑎𝑙, 𝑢) and 𝐶𝐸(𝑚𝑙, 𝑢) for all layers 𝑙 and all 1,209 contexts 𝑢 in the Counterfact dataset, i.e. for every prompt we compute the direct and compensatory effects of every possible layer ablation.

##### 4.2. Results

[Figure 11]

[Figure 12]

(a) (b)

[Figure 13]

[Figure 14]

(c) (d)

- Figure 7 | Results from carrying out ablations at all layers across the entire Counterfact dataset. (a) Direct effects against compensatory effect for attention layers, with colourmap indicated network depth. Middle layers show strong correlations whereas early and late layers do not. (b) Relation between direct and compensatory effect at the layer with the highest correlation, which occurs at layer 23 where compensation explains 92% of the variance in changes between the intact and ablated network. (c) Same as (a) but for MLP layers. (d) Summary of variance explained (solid blue line) and slope (dashed black line) of a linear regression between direct and compensatory effects at each attention layer. Red line marks layer shown in subfigure (c).

Results from full quantification of the compensatory effects across the full Counterfact dataset are

shown in Figure 7, with data further broken out to individual layers in Figure 8. We highlight the following observations from these results:

Direct and compensatory effects are only well-correlated at intermediate layers: early layer ablations have large total effects but almost no direct effect (Figure 8a, c, c.f. Figure 2) whereas very late layers only have non-negligible direct effect (which makes sense as there are few downstream layers for them to have an effect on).

[Figure 15]

- Figure 8 | Per-layer compensation results for a selection of ablation layers. The balance between the effect due to the attention and MLP layers shifts from attention to MLP as depth increases.

The compensatory response is almost entirely responsible for changes in direct effect at later layers. Compensatory response is very highly correlated at intermediate-late layers (see Figure 8b, d), suggesting that the network’s response to ablations at these layers is almost entirely driven by the Hydra effect and a decrease in MLP erasure. Layer 23 (of 32) marks the high point of this phenomenon, with 92% of the variance in Δ𝐷𝐸 in downstream layers being explained by the Hydra effect.

The compensatory response does not fully restore the output: fitting a linear regression between direct effect and compensatory response gives a slope of less than one at all layers past layer 13 (see Figure 7). This implies that these ablations do have a nonzero total effect, but one that is much smaller than it would have been without the Hydra effect and responses from erasure MLPs.

The balance between the Hydra effect and reduction in MLP erasure shifts with network depth Figure 8 shows Δ𝐷𝐸 for attention and MLP layers separately at different layers in the network. In early layers changes in attention direct effect play a considerably larger role, whereas by layer 22 the balance has shifted such that the the MLP response is considerably more predictive and at layer 23 almost all of the response is occuring in the erasure MLPs.

### 5. Related Work

The use of techniques from causal inference to analyse neural networks has been used in a range of cases, including the causal tracing method for locating factual knowledge (Meng et al., 2022), mediation analyses for gender bias (Vig et al., 2020a,b) and analysis of syntactic agreement (Finlayson

- et al., 2021). There is also a recent line of work on constructing causal abstractions of neural network computations (Geiger et al., 2021, 2022, 2023a,b; Massidda et al., 2022). The use of ablations as a way of validating hypotheses about mechanisms in neural networks has been previously suggested (Chan et al., 2022; Leavitt and Morcos, 2020; Morcos et al., 2018), although our findings caution against straightforwardly interpreting low effectiveness of ablations as meaning that a network component is unimportant.

Earlier work on residual networks (of which decoder-only transformers are a special case) determined that for image classification networks, residual networks behave like ensembles of shallower networks (Veit et al., 2016). This work introduced both the ‘unravelled view’ of residual networks that we make use of and experimented with ablating network layers at test time, determining that most effective paths through residual networks are short and layers often do not depend on one another.

The idea of interpreting neural networks in terms of their internal mechanisms or circuits Olah et al. (2020) (often referred to as mechanistic interpretability) is relatively recent. Earlier work on vision models (Olah et al., 2018) identified human-understandable neurons and simple circuits (for instance curve detectors (Cammarata et al., 2020)). Subsequent work on transformers has identified ‘induction circuits’ responsible for simple instances of in-context learning (Elhage et al., 2021; Olsson

- et al., 2022), as well as a mechanism for indirect object identification (Wang et al., 2022) and the mechanism underlying the ‘grokking’ phenomenon (Chughtai et al., 2023; Nanda et al., 2023; Power et al., 2022).

The use of probes to analyse neural network training originated in (Alain and Bengio, 2016), and has been widely used since. In the context of transformer language models the ‘logit lens’ approach, which involves using the model’s own unembedding matrix to decode the residual stream, has been applied to early GPT models (nostalgebraist, 2020). In order to better align with the model’s true

outputs Belrose et al. (2023) use a learned affine unembedding rather than the model’s unembedding matrix and are also able to perform causal interventions using a learned ‘causal basis’. Geva et al. (2022b) and follow-on work (Dar et al., 2022; Geva et al., 2022a) analyse MLP layers by unembedding specific subspaces of their outputs. Sparse probing has been used to empirically study the superposition phenomenon (Elhage et al., 2022) in large language models (Gurnee et al., 2023) and has been used to understand concept learning in deep reinforcement learning agents (Forde et al., 2022; McGrath et al., 2022).

### 6. Conclusion

Findings In this work we have investigated the computational structure of language models during factual recall by performing detailed ablation studies. We found that networks exhibit surprising self-repair properties: knockout of an attention layer causes another attention layer to increase its effect in order to compensate. We term this new motif the Hydra effect. We also found that late-layer MLPs appear to have a negative-feedback/erasure effect: late layer MLPs often act to reduce the probability of the maximum-likelihood token, and this reduction is attenuated when attention layers promoting that token are knocked out. We find that these effects are approximately linear, and that

- at middle layers (where these effects are strongest) the Hydra effect and reduction in MLP effects collectively act to restore approximately 70% of the reduction in token logits.

Implications for interpretability research These findings corroborate earlier work on neural network computations in GPT-2 Small (Wang et al., 2022) which reported a similar effect that the authors term ‘backup heads’. The authors of (Wang et al., 2022) hypothesised that dropout (Srivastava et al., 2014) was responsible for self-repair behaviour, which we disprove as the model we study (Chinchilla 7B) was trained without any form of dropout or stochastic depth. The occurrence of this motif across tasks and models suggests that it may be an instance of universality (Olah et al.,

- 2020). Our original motivation for this work was performing automated ablation studies using an algorithm similar to (Conmy et al., 2023), which led to us investigating the changes in network computation under repeated ablations. The Hydra effect poses a challenge to automating ablations: if we prioritise network components for ablation according to their total effect, we will be using a measure that does not fully reflect the computational structure of the intact network. Fortunately, the fact that the compensatory effect is typically less than 100% means that automated ablations will still have some signal to work with. The Hydra effect and erasure MLPs also have implications for attributing responsibility for a network’s output to individual network components: is the responsible component the attention layer that has the effect in the intact network, or the circuits that act to compensate following ablation? The framework of actual causality (Halpern, 2016) may be a useful way to approach this question.

Our findings also suggest that attempting to assign semantics to MLP neurons may be more complicated than otherwise expected: erasure MLPs may have no clear semantics in terms of the model’s input, as they are responding to the language model’s internal computations. Finally, our findings also have implications for work that attempts to understand language models by unembedding the output of individual layers (e.g. (Geva et al., 2022b)) - this corresponds to an assumption that the direct effect is the only meaningful effect of a layer. The existence of erasure MLPs poses a challenge to this approach to interpretability: if the output of an attention layer or MLP is guaranteed to be partially undone by an erasure MLP, it’s no longer straightforward to interpret that output in terms of its direct effects on logits: the effect of the mediating MLPs should also be considered. Our findings also provide context for earlier ablation studies (such as (Morcos et al., 2018)): it is not enough simply to measure the total effect of an ablation without investigating downstream changes in the

network, and more important network components are more likely to be robust to ablation.

Implications for language modelling research From the perspective of language modelling the Hydra effect is surprising: it confers robustness to a kind of ablation that the model will never experience at inference time and so appears to be a waste of parameters. If this is the case, what benefit does it confer? One possibility (drawing on the analytical framework of Tinbergen’s four questions (Tinbergen, 1963)) is that the Hydra effect confers no benefit at inference time, but is beneficial in the context of training. If gradient descent were to occasionally break network components then a kind of ‘natural dropout’ would occur during training. In this case it would be beneficial for networks to be robust to layers failing. We emphasise that this is conjecture, however, and would need further research.

Possible extensions Although we have identified two new motifs, we have not investigated more deeply than individual layers (for instance looking at the level of individual attention heads or directions in activation space). Achieving a greater level of precision is a natural next step and would unlock deeper understanding of the mechanisms at play here. Some questions that could be answered with a finer-grained understanding of how this kind of redundancy operates include:

- 1. How much does the Hydra effect occur across the entire training distribution? Does sequence length play any role?
- 2. What are the Hydra effect attention heads responding to the presence/absence of in the residual stream?
- 3. Do the same downstream heads act as Hydra effect replacement heads across multiple contexts?
- 4. What causes the Hydra effect? Is the natural dropout hypothesis correct or is some other phenomenon responsible (superposition has been suggested as an alternative explanation).
- 5. Is there a Hydra effect for features rather than direct effect on logits?
- 6. What are the erasure heads responding to in the residual stream? Do they have a ‘recalibration’ effect when a wider range of tokens is considered?
- 7. If we account for redundancy/Hydra effect, can we probe network structure by using targeted ablations?

Acknowledgements We would like to thank Joe Halpern, Avraham Ruderman, Tom Lieberum, Chris Olah, Zhengdong Wang, Tim Genewein and Neel Nanda for helpful discussions.

### References

G. Alain and Y. Bengio. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644, 2016.

J. L. Ba, J. R. Kiros, and G. E. Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016. D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and

translate. arXiv preprint arXiv:1409.0473, 2014.

D. Bau, J.-Y. Zhu, H. Strobelt, B. Zhou, J. B. Tenenbaum, W. T. Freeman, and A. Torralba. Gan dissection: Visualizing and understanding generative adversarial networks. arXiv preprint arXiv:1811.10597, 2018.

N. Belrose, Z. Furman, L. Smith, D. Halawi, I. Ostrovsky, L. McKinney, S. Biderman, and J. Steinhardt. Eliciting latent predictions from transformers with the tuned lens. arXiv preprint arXiv:2303.08112, 2023.

N. Cammarata, G. Goh, S. Carter, L. Schubert, M. Petrov, and C. Olah. Curve detectors. Distill, 5(6): e00024–003, 2020.

- S. Carter, Z. Armstrong, L. Schubert, I. Johnson, and C. Olah. Activation atlas. Distill, 2019. doi: 10.23915/distill.00015. https://distill.pub/2019/activation-atlas.

L. Chan, A. Garriga-Alonso, N. Goldwosky-Dill, R. Greenblatt, J. Nitishinskaya, A. Radhakrishnan,

- B. Shlegeris, and N. Thomas. Causal scrubbing, a method for rigorously testing interpretability hypotheses. AI Alignment Forum, 2022. https://www.alignmentforum.org/posts/ JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing.

- B. Chughtai, L. Chan, and N. Nanda. A toy model of universality: Reverse engineering how networks learn group operations. arXiv preprint arXiv:2302.03025, 2023.

A. Conmy, A. N. Mavor-Parker, A. Lynch, S. Heimersheim, and A. Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. arXiv preprint arXiv:2304.14997, 2023.

G. Dar, M. Geva, A. Gupta, and J. Berant. Analyzing transformers in embedding space. arXiv preprint arXiv:2209.02535, 2022.

- T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer. Llm. int8 (): 8-bit matrix multiplication for transformers at scale. arXiv preprint arXiv:2208.07339, 2022.

N. Elhage, N. Nanda, C. Olsson, T. Henighan, N. Joseph, B. Mann, A. Askell, Y. Bai, A. Chen, T. Conerly, N. DasSarma, D. Drain, D. Ganguli, Z. Hatfield-Dodds, D. Hernandez, A. Jones, J. Kernion, L. Lovitt, K. Ndousse, D. Amodei, T. Brown, J. Clark, J. Kaplan, S. McCandlish, and C. Olah. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021. https://transformer-

- circuits.pub/2021/framework/index.html.

N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, S. Kravec, Z. Hatfield-Dodds, R. Lasenby, D. Drain, C. Chen, R. Grosse, S. McCandlish, J. Kaplan, D. Amodei, M. Wattenberg, and C. Olah. Toy models of superposition. Transformer Circuits Thread, 2022. https://transformer-

- circuits.pub/2022/toy_model/index.html.

M. Finlayson, A. Mueller, S. Gehrmann, S. Shieber, T. Linzen, and Y. Belinkov. Causal analysis of syntactic agreement mechanisms in neural language models. arXiv preprint arXiv:2106.06087, 2021.

J. Z. Forde, C. Lovering, G. Konidaris, E. Pavlick, and M. L. Littman. Where, when & which concepts does alphazero learn? lessons from the game of hex. In AAAI Workshop on Reinforcement Learning in Games, volume 2, 2022.

A. Geiger, H. Lu, T. Icard, and C. Potts. Causal abstractions of neural networks. Advances in Neural Information Processing Systems, 34:9574–9586, 2021.

A. Geiger, Z. Wu, H. Lu, J. Rozner, E. Kreiss, T. Icard, N. Goodman, and C. Potts. Inducing causal structure for interpretable neural networks. In International Conference on Machine Learning, pages 7324–7338. PMLR, 2022.

A. Geiger, C. Potts, and T. Icard. Causal abstraction for faithful model interpretation. arXiv preprint arXiv:2301.04709, 2023a.

A. Geiger, Z. Wu, C. Potts, T. Icard, and N. D. Goodman. Finding alignments between interpretable causal variables and distributed neural representations. arXiv preprint arXiv:2303.02536, 2023b.

M. Geva, A. Caciularu, G. Dar, P. Roit, S. Sadde, M. Shlain, B. Tamir, and Y. Goldberg. Lm-debugger: An interactive tool for inspection and intervention in transformer-based language models. arXiv preprint arXiv:2204.12130, 2022a.

M. Geva, A. Caciularu, K. R. Wang, and Y. Goldberg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. arXiv preprint arXiv:2203.14680, 2022b.

- M. Glymour, J. Pearl, and N. P. Jewell. Causal inference in statistics: A primer. John Wiley & Sons, 2016.

G. Goh, N. C. †, C. V. †, S. Carter, M. Petrov, L. Schubert, A. Radford, and C. Olah. Multimodal neurons in artificial neural networks. Distill, 2021. doi: 10.23915/distill.00030. https://distill.pub/2021/multimodal-neurons.

W. Gurnee, N. Nanda, M. Pauly, K. Harvey, D. Troitskii, and D. Bertsimas. Finding neurons in a

haystack: Case studies with sparse probing. arXiv preprint arXiv:2305.01610, 2023. J. Y. Halpern. Actual causality. MiT Press, 2016. J. Hilton, N. Cammarata, S. Carter, G. Goh, and C. Olah. Understanding rl vision. Distill, 2020. doi:

10.23915/distill.00029. https://distill.pub/2020/understanding-rl-vision.

- J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

- M. L. Leavitt and A. Morcos. Towards falsifiable interpretability research. arXiv preprint arXiv:2010.12016, 2020.

R. Massidda, A. Geiger, T. Icard, and D. Bacciu. Causal abstraction with soft interventions. arXiv preprint arXiv:2211.12270, 2022.

T. McGrath, A. Kapishnikov, N. Tomašev, A. Pearce, M. Wattenberg, D. Hassabis, B. Kim, U. Paquet, and V. Kramnik. Acquisition of chess knowledge in alphazero. Proceedings of the National Academy of Sciences, 119(47):e2206625119, 2022.

K. Meng, D. Bau, A. Andonian, and Y. Belinkov. Locating and editing factual associations in gpt. Advances in Neural Information Processing Systems, 35:17359–17372, 2022.

A. S. Morcos, D. G. Barrett, N. C. Rabinowitz, and M. Botvinick. On the importance of single directions for generalization. arXiv preprint arXiv:1803.06959, 2018.

- N. Nanda, L. Chan, T. Liberum, J. Smith, and J. Steinhardt. Progress measures for grokking via mechanistic interpretability. arXiv preprint arXiv:2301.05217, 2023.

#### nostalgebraist. interpreting gpt: the logit lens, 2020. URL https://www.lesswrong.com/posts/ AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens.

- C. Olah, A. Satyanarayan, I. Johnson, S. Carter, L. Schubert, K. Ye, and A. Mordvintsev. The building blocks of interpretability. Distill, 3(3):e10, 2018.

- C. Olah, N. Cammarata, L. Schubert, G. Goh, M. Petrov, and S. Carter. Zoom in: An introduction to circuits. Distill, 2020. doi: 10.23915/distill.00024.001. https://distill.pub/2020/circuits/zoom-in.

C. Olsson, N. Elhage, N. Nanda, N. Joseph, N. DasSarma, T. Henighan, B. Mann, A. Askell, Y. Bai, A. Chen, et al. In-context learning and induction heads. arXiv preprint arXiv:2209.11895, 2022. J. Pearl. Causality. Cambridge university press, 2009.

- M. Phuong and M. Hutter. Formal algorithms for transformers. arXiv preprint arXiv:2207.09238, 2022.

A. Power, Y. Burda, H. Edwards, I. Babuschkin, and V. Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv preprint arXiv:2201.02177, 2022.

- N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1): 1929–1958, 2014.

- N. Tinbergen. On aims and methods of ethology. Zeitschrift für tierpsychologie, 20(4):410–433, 1963.

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

- A. Veit, M. J. Wilber, and S. Belongie. Residual networks behave like ensembles of relatively shallow networks. Advances in neural information processing systems, 29, 2016.

J. Vig, S. Gehrmann, Y. Belinkov, S. Qian, D. Nevo, S. Sakenis, J. Huang, Y. Singer, and S. Shieber. Causal mediation analysis for interpreting neural nlp: The case of gender bias. arXiv preprint arXiv:2004.12265, 2020a.

- J. Vig, S. Gehrmann, Y. Belinkov, S. Qian, D. Nevo, Y. Singer, and S. Shieber. Investigating gender bias in language models using causal mediation analysis. Advances in neural information processing systems, 33:12388–12401, 2020b.
- K. Wang, A. Variengien, A. Conmy, B. Shlegeris, and J. Steinhardt. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small. arXiv preprint arXiv:2211.00593, 2022.

- B. Zhang and R. Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.

### A. Choice of intervention distribution

When we intervene on a neural network’s activations with the do-operator, we are setting some subset of these activations to a new value in the forward pass and allowing the effect of these changes to propagate forward. Both the corrupted run and the patching operation in the causal tracing method (Meng et al., 2022) are examples of interventions. Practically we can accomplish these interventions via PyTorch hooks, JAX’s Harvest functionality, or passing values for intervened-upon tensors in Tensorflow’s feed-dict. When we intervene to set some network activation 𝑍 to an ‘ablation’ value ˜𝑧, what value should we use? Four main possibilities have been suggested:

Zero ablation : ˜𝑧 = 0, (35) Mean ablation : ˜𝑧 = 𝔼𝑢∼𝑃(𝑢) [𝑍(𝑢)] , (36) Noise ablation : ˜𝑧 = 𝑍(𝑢 + 𝜖), 𝜖 ∼ N(0, 𝜎2), (37)

Resample ablation : ˜𝑧 = 𝑍(𝑢˜), 𝑢˜ ∼ 𝑃(𝑢). (38)

Of these, zero-ablation is the simplest to implement but is typically out-of-distribution for networks trained without some form of layer dropout or stochastic depth. Noise ablation was used extensively in causal tracing (Meng et al., 2022). Resample ablation (as introduced by Chan et al. (2022)) is more complicated to implement but is probably the most principled, as every ablation is a naturallyoccurring set of activations, and so is more likely to respect properties of network activations such as emergent outliers (Dettmers et al., 2022). Resample ablation also has the appealing property that by specifying the distribution of inputs 𝑃(𝑢) we can control for properties of the input that might otherwise confound our results. To get meaningful results from sample ablations it is necessary to use an average of sample ablations across multiple samples from the same dataset, i.e. a Monte-Carlo approximation to:

𝑉𝑍(𝑢) = ∫ 𝑉(𝑢|𝑑𝑜(𝑍 = ˜𝑧)) 𝑝(˜𝑧) 𝑑˜𝑧, (39)

where 𝑝(˜𝑧) = ∫ 𝑝(𝑍(𝑢))𝑑𝑢 is the probability of getting activation values ˜𝑧. Note that mean ablation and resample ablation are quite different: mean ablation ablates with the average activation, whereas resample activation averages the effects of different ablations. See (Chan et al., 2022) for an extended discussion of these methodological details.

