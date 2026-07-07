# arXiv:2502.14372v1[quant-ph]20Feb2025

## Discovering highly efficient low-weight quantum error-correcting codes with reinforcement learning

Austin Yubo He1, 2, ∗ and Zi-Wen Liu3, †

1The University of Texas at Austin, Austin, TX 78712, USA 2University of California, Berkeley, Berkeley, CA 94720, USA 3Yau Mathematical Sciences Center, Tsinghua University, Beijing, 100084, China (Dated: February 21, 2025)

The realization of scalable fault-tolerant quantum computing is expected to hinge on quantum error-correcting codes. In the quest for more efficient quantum fault tolerance, a critical code parameter is the weight of measurements that extract information about errors to enable error correction: as higher measurement weights require higher implementation costs and introduce more errors, it is important in code design to optimize measurement weight. This underlies the surging interest in quantum low-density parity-check (qLDPC) codes, the study of which has primarily focused on the asymptotic (large-code-limit) properties. In this work, we introduce a versatile and computationally efficient approach to stabilizer code weight reduction based on reinforcement learning (RL), which produces new low-weight codes that substantially outperform the state of the art in practically relevant parameter regimes, extending significantly beyond previously accessible small distances. For example, our approach demonstrates savings in physical qubit overhead compared to existing results by 1 to 2 orders of magnitude for weight 6 codes and brings the overhead into a feasible range for near-future experiments. We also investigate the interplay between code parameters using our RL framework, offering new insights into the potential efficiency and power of practically viable coding strategies. Overall, our results demonstrate how RL can effectively advance the crucial yet challenging problem of quantum code discovery and thereby facilitating a faster path to the practical implementation of fault-tolerant quantum technologies.

#### I. INTRODUCTION

Quantum information processing offers promising potential for revolutionary advantages over conventional methods in computation and various other types of technologies [1–4]. However, a fundamental obstacle stands in the way: quantum systems and their manipulation are inherently prone to a wide variety of noise and errors, necessitating efficient fault tolerance strategies that maintain the protection of quantum information when all components may be faulty, in order to make quantum advantages practically scalable and unlock their full potential. Quantum error-correcting (QEC) codes provide a pathway to efficient fault tolerance, positioning them as a pivotal field of research in quantum information [5–8]. Furthermore, they have recently made a profound impact on physics [9–12], underscoring their fundamental importance.

Stabilizer codes serve as a canonical framework for QEC codes, enabling logical qubits to be encoded in more physical qubits in a way that information about errors needed for correction can be inferred through measuring certain parity-check (stabilizer) operators [6]. Since measurements of higher weight (the size of nontrivial support) typically require larger circuits and qubit overhead, which can make experimental executions significantly more difficult and introduce more errors, it is crucial to

∗ austinhe@utexas.edu † zwliu0@tsinghua.edu.cn

minimize the check weight in code design for practical quantum computing. Note that lower check weight is also a key motivation behind the interest in various generalized coding schemes such as subsystem codes [13, 14] and dynamical codes [15, 16].

In line with this, constraining the check weight (as well as the degree) to be asymptotically O(1)—that is, bounded by a constant as the code length grows—gives rise to the so-called quantum low-density parity-check (qLDPC) codes which have attracted intensive interest as a promising scheme for fault tolerance with low overhead [17–20]. In particular, general qLDPC codes can achieve substantially better code parameters and fault tolerance efficiency [18, 21, 22] than those with geometric connectivity constraints including the surface code, which has long been regarded the leading scheme for implementing fault tolerance [9, 23–26]. The need for substantial long-range connectivity to overcome geometric barriers for code parameters [27–31] poses a significant obstacle to capitalizing on the advantages of qLDPC codes. However, recent remarkable advances in quantum computing with the reconfigurable atom array platform [32] bring hope for alleviating this difficulty, potentially establishing qLDPC code-based fault tolerance schemes as mainstream. Driven largely by theoretical interest, intensive study has been devoted to the constructions of qLDPC code families with desirable asymptotic parameters in the infinite code length limit (see e.g. Ref. [18] for a slightly outdated review). Notably, there has been a recent surge of breakthroughs in achieving asymptotically ‘good’ qLDPC code families that simultaneously attain optimal scalings of both code rate

and distance [33–35]. However, in practically relevant finite-size regimes, these asymptotic code constructions are not expected to exhibit good parameters and usually feature stabilizers with relatively high (though finite) weights too demanding for actual implementation. The optimization of finite-scale code design requires specialized approaches yet has received little attention, despite its evident importance for the practical development of quantum hardware. In particular, recent experimental progress [26, 32] suggests that QEC codes with a distance of several tens are crucial in the coming years for the development of fault-tolerant hardware, whereas existing code design approaches (ranging from e.g. greedy algorithms, constraint satisfaction, exhaustive search, evolutionary algorithms, to reinforcement learning [36–43]) generally struggle to exceed single-digit distances.

In the pursuit of optimizing QEC codes, an effective strategy is known as weight reduction, which involves algorithms that aim to decrease check weight while maintaining other code properties like rate and distance, at the cost of physical qubit overhead. Hastings first proposed a weight reduction method for CSS codes [44, 45] primarily focusing on the asymptotic setting, reducing the check weight and degree to O(1). Subsequent work by Sabo et al. [46] extended the idea to finite-size regimes for product codes, demonstrating a modified weight reduction method that can be applied with significantly lower qubit overhead on relatively small codes and better performance when implemented on a cluster state architecture using GKP qubits [47]. Nevertheless, as we shall demonstrate, their method is still far from optimal, typically entailing qubit overhead that can be significantly improved especially larger-size regimes that are crucial for future applications.

In this work, we present a remarkably effective and general scheme for discovering low-weight QEC codes based on a novel reinforcement learning (RL) [48] scheme for weight reduction. This addresses the previously recognized difficulty of learning relatively large qLDPC codes as explained in, e.g., Ref. [37]. Our results reveal an essential insight that decreasing weight with distance constraints is a significantly more approachable problem compared to increasing distance with weight constraints, especially for learning methods. Specifically, a major obstacle that restricts the scale of previous code design approaches is that distance is a high-complexity property depending on the hamming weight of all non-trivial logical operators, and must be computed through exhaustive search or sampling [49], while weight can be directly calculated from the parity check matrix. Note that several works [37–41, 50–52] have explored the use of RL in QEC code design from various other perspectives including distance, threshold, logical error rate, decoding, and error adaptation, while this work highlights the prominence of check weight.

We specifically demonstrate the effectiveness of our approach using hypergraph product codes as base codes. Notably, we find that our RL model consistently achieves

Policy:|π(as)θ

Action: +/- edge

State: H Reward: d,w,q

Environment

FIG. 1. An illustration of our RL scheme. The RL agent (left) maintains a policy network that, given the state of the Tanner graph, selects an action of adding or removing an edge. The environment (right) updates the graph accordingly and returns a reward based on the code’s new distance and weight. This reward signal is then used to update the policy network, guiding the agent toward better code designs.

significantly smaller physical qubit overhead—up to 73x in the best case—compared to previous weight reduction methods. Moreover, it learns to design codes with distances up to 4x more than previous approaches applying RL methods to code design. Altogether, our RL-based scheme enables the discovery of numerous new low-weight codes with high distances of up to d ≈ 35 that requires significantly less physical qubits than existing constructions. This brings the efficiency of codes into a feasible range for near-term quantum devices with up to a few thousand physical qubits, and represents the first demonstration of RL-designed codes with parameters sufficiently good to be practically useful.

#### II. REINFORCEMENT LEARNING FRAMEWORK FOR WEIGHT REDUCTION

Reinforcement learning (RL) is a widely used machine learning paradigm in which an agent interacts with an environment by selecting actions and receiving feedback in the form of reward signals. The agent iteratively refines its decision-making policy to maximize cumulative rewards [48]. This paradigm has proven successful in a number of problems in physics [53–57], as well as problems with exceptionally large and complex decision spaces [58] and thus naturally lends itself to the combinatorial nature of code design.

As illustrated in Fig. 1, the environment of our weight reduction framework is defined by the Tanner graph of a stabilizer code. The agent can remove or add edges in the Tanner graph, and its objective is to minimize the maximum degree of variable and check nodes while preserving the code distance.

We utilize the Proximal Policy Optimization (PPO) algorithm with action masking to efficiently explore the action space [59]. PPO balances exploration and exploitation by constraining policy updates, ensuring stable

learning. The objective function is LCLIP(θ) = Et min rt(θ)Aˆt, clip(rt(θ),1 − ϵ,1 + ϵ)Aˆt ,

(1) where rt(θ) = π

θ(at|st)

πθold(at|st) is the probability ratio, Aˆt is the advantage estimate, and ϵ controls the clipping range.

Then we design the reward function to guide our Reinforcement Learning (RL) agent in optimizing Tanner graphs for stabilizer codes. The function balances node degree reduction with code distance preservation, ensuring robust error-correcting capabilities. We define local reward functions Rv(κv) and Rc(κc) for variable and check nodes, respectively, by assigning fixed rewards for degrees up to Vmax, and an exponential penalty for degrees above Vmax:



C1, κv = 1, C2, κv = 2, .



Rv(κv) =

(2)

CV

, κv = Vmax, exp −λ(κv − Vmax) , κv > Vmax,



max

where λ controls the severity of the penalty for large degrees, and the constants C1,C2,...,CV

reflect the relative desirability of each integer degree within the allowable range. A similar function Rc(κc) can be defined for check nodes. In our demonstration using hypergraph product codes we have Vmax = 3. The gap between the values of C3 and C2 was set to be close, with both around 0.7 to 1.0, while C1 was set to be a smaller value around 0.1 to 0.5. The precise values were adjusted empirically based on training results to encourage degree distributions to avoid under-utilizing check operators.

max

We min-max normalize each local reward Rv(κv),Rc(κc) and the distance-based terms d,∆d to fall in the interval [0,1]. Specifically, for any quantity X that lies in [Xmin,Xmax], the min-max normalized version X is defined as

X − Xmin Xmax − Xmin

, X ∈ [0,1]. (3)

X =

The normalized quantities Rv(κv), Rc(κc), d, ∆d are defined analogously. The minima and maxima are determined from known bounds of the code distance [dmin,dmax], and viable degree distributions for Rv(κv),Rc(κc).

The reward is formulated as R = α

Rc(κc) + β d − δ ∆d, (4)

Rv(κv) +

v∈V

c∈C

where V and C are variable and check nodes, κv and κc are their degrees, d is the code distance, and ∆d = dnew − dprev is the change in distance. The weights α, β, and δ balance degree reduction, distance preservation, and penalize distance reductions. The weights sum to 1 to ensure the reward is between 0 and 1.

To enforce weight constraints on parity checks, we employ action masking:

1, if action a is permissible in state s, 0, otherwise,

M(a|s) =

(5) We exclude all adding operations from nodes with degree greater than the maximum desired weight, and all deleting operations from nodes with degree 1 or the minimum desired weight. Masked actions are assigned zero probability, and the remaining action probabilities are re-normalized to form a valid probability distribution. This preserves the theoretical properties of PPO, such as the trust region constraint and still ensures stable policy updates [60]. Most importantly, this allows us to restrict the agent to only focus on codes within the target weight, which enhances learning efficiency without compromising the algorithm’s convergence guarantees. The mask can be adapted to codes with a variety of constraints on their structure. For example, the masked property for general stabilizer codes is symplectic orthogonality of H, while for CSS codes it is the orthogonality between HX and HZ. We can also extend this to various important refined constraints are as needed, such as k-orthogonality which induces codes with transversal non-Clifford gates, and geometric locality or connectivity constraints that ubiquitously arise in physical and experimental scenarios.

In Fig. 2a, we showcase samples of reward curves for our RL policy. The initial reward for codes with lower rate is higher, as there are more checks so low weight codewords are less likely to exist at random, while higher rate codes have less checks so low weight codewords are more likely to exist at random [61]. All three displayed codes still converge to rewards in a similar range, although the learning process is more stochastic for codes with higher maximum distances. The evolution of weight, degree, and distance parameters over one episode of training on the three example codes are shown in Fig. 2b, and the area explored over 10 episodes in the state space of tanner graphs is shown in Fig. 2c. We see that as maximum degree and weight are gradually reduced, distance also decreases, then plateaus and rapidly increases. After this point both degree and weight remain in range and the policy begins to tweak distance. It is interesting that despite the ∆d term penalizing decreases in distance, the policy does not attempt to reduce the weight and degree in a more slow and careful way to perfectly preserve d at each step. We provide two interpretations, either the policy is taking the path of least resistance and simply finds the problem easier to approach this way, or it could be learning from fluctuations in distance. This term both penalizes decreases and rewards increases in distance, so it may be pursuing the maximum possible reward in this setting, which is when a code with minimum d code is modified into a code with maximum d in one step, while the maximum weights and degrees are below 6 and 3. Regardless, the policy still produces codes

[Figure 1]

[Figure 2]

[Figure 3]

(a) (b) (c)

- FIG. 2. Reinforcement learning-driven code design. (a) Training trajectories of codes with varying parameters averaged over 3 runs. (b) Evolution of parameters in the three example codes throughout a single episode.(c) Exploration of 10 episodes (represented by different colors) over PCA decomposition of state space.

that effectively minimize weight reduction overhead.

#### III. MAIN RESULTS

We now showcase the representative code discovery results achieved through our RL weight reduction scheme and discuss important comparisons with existing results. Additional information including more complete data and numerous auxiliary results and illustrations can be found in the Supplementary Information.

We extend the standard [[n,k,d]] notation for QEC codes (n,k are the number of physical and logical qubits respectively and d is the code distance) to [[n,k,d]](w,q), where w denotes the check weight and q denotes qubit degree.

A. Overview of code discovery

Here we primarily focus on the following setting: we apply our RL agent to hypergraph product codes, and aim to reduce to a maximum weight of six and qubit degree of three, matching the parameters produced in Ref. [46]. These are favorable parameters for practical implementation and also allow us to make direct comparisons. Note that our method can be adapted for any given weight or degree, and we also show examples for maximum weight eight and degree four. Furthermore, one can easily apply our method to general stabilizer codes settings and specialize to certain types including CSS codes, product codes, k-orthogonal codes and so on as needed, as addressed in Sec. II by adapting the action masking logic accordingly.

Hypergraph product codes [62] provide an elegant framework for constructing quantum codes from classi-

cal ones and serve as a prototypical model in the study of qLDPC codes with desirable code parameters and FT properties [18, 62–64]. More explicitly, given parity check matrices of classical codes H1 and H2, define

1 ⊗ H2T , (6) HZ = In

HX = H1 ⊗ In

Ir

2

1 ⊗ H2 H1T ⊗ Ir

, (7)

2

represent the X and Z check matrices of a CSS code respectively. Here H1 and H2 are the check matrices of the original classical codes, and I denotes identity matrices of appropriate dimensions. This construction ensures orthogonality between HX and HZ for any pair of linear classical codes and therefore produces a valid CSS stabilizer code. In this work we use the same classical code for H1 and H2.

We generate new codes by executing our RL scheme on hypergraph product base codes constructed from all classical codes with n ≤ 30 from the best known linear codes database in the GUAVA package in GAP [65, 66], which we collectively refer to as HGP-30. Several examples of codes with particularly large distances beyond this range will also be included. All results were produced using an i7-13700HX CPU and RTX-4060 GPU. With more extensive training it is feasible to further optimize code parameters or scale to larger sized codes [67].

We visualize the code parameter combinations of the base and new codes produced by our RL-based weight reduction scheme with a parallel coordinates plot – see Fig. 3. This illustrates how our RL-based weight reduction scheme modifies code parameters. Initially, the w and q parameters can be quite large, and while most codes have relatively balanced weight and degree, there exist a number of codes which are highly skewed. After our RL-based weight reduction, n tends to increase while w and q decrease, and the k and d values are unchanged. Unsurprisingly, k and d exhibit an inverse re-

[Figure 4]

- FIG. 3. Parallel coordinates plot comparing hypergraph product base codes (blue) and RL-optimized codes (red) after weight reduction. For each color, 475 codes (including 10 high-distance ones beyond the HGP-30 regime) with varying parameters are shown. Each vertical axis is normalized to the maximum observed value for that parameter, and each line traces a single code’s parameters across all axes.

lation, as codes with higher k tend to have lower d, and vice versa. Our method uncovers a broad spectrum of new low-weight and degree codes with considerably better code parameters compared to known constructions, as we will demonstrate explicitly through suitable comparisons now.

B. Weight reduction comparisons

We first compare the effectiveness of our RL-based method and existing weight reduction methods. In Fig. 4, we illustrate the direct comparisons between the parameters of codes discovered by our RL policy and earlier weight reduction methods. In the low-rate and low-distance regime, Hastings’ early method [45] require thousands of qubits. The state-of-the-art (labeled by SOTA) results from Sabo et al. [46] lower the qubit overhead by more than one order of magnitude compared to Ref. [45] but some overhead remains, while our RL method does not need overhead at all in this regime. For larger codes, the overhead of our method exhibits roughly 1–2 orders of magnitude lower overhead on (6,3) codes (see also Fig. 6 for further information). We also compare the overheads for codes reduced to weight and degree (8,4), which are significantly smaller at moderate to high rate codes. The qubit overhead for (6,3) and (8,4) are not related in a straightforward way, for example, the qubit overhead when reducing to (6,3) is increasing

[Figure 5]

FIG. 4. Comparisons of codes discovered by our RL-based scheme and existing weight reduction methods. (top) Comparison with Hastings [45] (data taken from Ref. [46]) and SOTA results from Sabo et al. [46]. (bottom) Comparisons with SOTA results on all hypergraph product codes constructed from n ≤ 30 classical codes. Explicit code parameters are shown in Table I, II.

almost monotonously at (0 < k < 600), while for (8,4) it shows decreasing trends at many ranges of k values (0 < k < 200), (200 < k < 300), (350 < k < 450). This provides a glimpse into the trade-off of w, q against n, k, d, although a more thorough exploration of different w,q combinations would be needed to fully map these trade-offs. These points are not necessarily optimal, and the inefficiencies are unknown, so the true relationship between the upper bounds on n,k,d,w,q could also turn out to be vastly different.

Then in Fig. 5 we depict the overhead factors (the ratio of the qubit number after and before weight reduction) for various combinations of code parameters, highlighting the clear advantages in efficiency of our RL-based method. As shown, the overhead of RL weight reduced codes tends to gradually increase with code rate k/n, while for Ref. [46], the overhead peaks around moderate rates (0.1 < k/n < 0.40) and sharply decreases after this point. Our RL method exhibits lower qubit overhead for all 465 codes considered, with the greatest difference at low to moderate rates. The maximum observed reduction in qubit overhead among the examples is about 73x for a [[1109,9,14]](8,13) code. Further analysis shows that

[Figure 6]

[Figure 7]

[Figure 8]

- FIG. 5. Breakdown of overhead factors shown by heatmaps at varying n, k, d parameters. The top and bottom rows correspond to codes discovered by our RL weight-reduction scheme and Sabo et al.’s method, respectively. Gradients are binned for ease of visualization and not exact representations of overhead factors, as seen in the varying scales.

[Figure 9]

(a) (b) (c)

[Figure 10]

[Figure 11]

- FIG. 6. Comparisons against SOTA on weight reduction and RL code design. (a) Comparison of qubit overheads of weight reduction to (6,3) between RL and SOTA on all hypergraph product codes constructed from classical codes with n ≤ 30. (b) Rate vs. relative distance for weight-reduced vs. hypergraph product base codes. (c) Comparisons of code parameters against various alternative RL methods for code design. Ten additional (6,3) codes (labeled by diamonds) beyond the HGP-30 regime were produced using our RL model to display examples of particularly high distance codes.

our RL agent tends to have the highest overhead on d = 4 codes, and the overhead factor decreases as distance increases (mostly because this means k/n is decreasing). We also note that the areas with highest overhead factor locally tend to be around the frontier of k/n vs d, which is evident from Fig. 5 and Fig. 10 in Supplementary Information. Overall, this suggests that for our RL agent, the bottleneck primarily arises from k/n instead of d, which bodes well for the performance of our RL agent at larger distances and code sizes.

Overall, our RL agent significantly reduces the qubit overhead needed to achieve a target weight and degree, including codes relevant in the near-term parameter regime. To exemplify the effectiveness of our RL policy, in Fig. 6a we show our RL policy tends to re-

quire an overhead of 103 to 104 physical qubits, whereas Ref. [46] requires an overhead of 104 to 105 physical qubits, representing 1–2 orders of magnitude’s improvement over SOTA. In particular, our method exhibits vastly smaller overhead at moderate to large code sizes. This is especially clear in an example beginning with a [[2500,100,16]](29,19) base code: our RL agent produces a [[6100,100,16]](6,3) code, which is within reach for near-term hardware [68], whereas the SOTA method in Ref. [46] yields a [[144772,100,55]](6,3) code with a substantially more severe qubit count requirement. Notice that despite the large overhead, the approach of Ref. [46] can also increase distance. This may potentially produce codes interesting in their own right, although up to orders of magnitude larger than the original code. The codes

discovered by our RL agent tend to have better overhead to distance scaling as shown in Fig. 6b. For example, our RL agent finds a [[2257,25,15]](6,3) code from a [[1525,25,15]](12,15) base code, while for the same n and k, the method given in Ref. [46] produces a [[2257,25,13]](6,3) code from a [[277,25,6]](6,10) base code. Other codes with similar k and d produced by Ref. [46] require even larger overhead to obtain comparable d and are shown in Table III. This comparison may be conservative as the method given in Ref. [46] can produce vastly different results for base codes with the same parameters, and even permutations of the same code, while our RL agent is running on relatively modest computational resources. To conclude, the codes discovered by our method are expected to exhibit even larger advantages at larger distances, although at the price of increasing computational demands [67].

C. RL code design comparisons

Check weight constraints have scarcely been considered in previous studies applying numerical methods to code design, especially for RL, largely because they have focused on codes with few physical qubits, or with relatively constrained state spaces. Here we make specific comparisons with the most recent works by Olle et al. [37] and Mauron et al. [38]. When making the comparisons, we still apply the restriction of maximum weight 6 and degree 3, although we are able to achieve even better parameters if we relax these constraints, at the price of the code performing worse in practice.

Olle et al. [37] find codes up to d ≤ 5 and n ≤ 25. A circuit level representation and Knill-Laflamme condition based reward function is used, and requires tracking all error operators with Hamming weight equal to the target distance. This approach circumvents the sparsity of a distance reward, but is exponentially memory intensive, and they create a road map which requires a GPU with approximately 80GB of VRAM to find codes with d = 10. Mauron et al. [38] find codes with up to d ≤ 9 and n < 25. A tensor network representation is used with a distancebased reward function which is not restricted by memory requirements, making it possible to design codes with larger d, but still were unable to surpass d = 10.

We compare the codes discovered by our RL agent with those from Refs. [37, 38] along with surface code in Fig. 6c. All methods can produce codes that outperform the surface code in terms of rate and distance. While other methods are restricted to d ≤ 9, our RL scheme can design codes in a significantly larger distance regime, discovering codes up to d = 40 and ones outperforming the surface code with d = 35. Note that we did not run a comprehensive sweep outside the HGP-30 regime due to computational constraints. Although the rates are comparable at small code lengths, our RL agent discovers codes with a maximum number of logical qubits k = 900, while previous works have been limited to codes

with single digit k. Also, these are the first RL designed codes with parameters large enough to be relevant to devices in the approaching Megaquop regime [68], where it is widely believed that relatively high distance codes with at least d ⪆ 20 will be necessary to reach logical error rates on the order of 10−6 [22, 26, 68].

D. Additional discussions and results

In Fig. 7a, we present pairwise scatter plots for the parameters of the base hypergraph product codes and RL weight reduced codes, which provide abundant insights into the behavior of the RL scheme and codes. Prior to weight reduction, the w and q values correlate with n,k,d, while the correlations vanish after our RL agent applies weight reduction and w and q become constant as seen in Fig. 7b. The data points for d and n make curves that are well described by d = O(√n) for each k value, as shown in Fig. 11a and Fig. 11b in Supplementary Information. This confirms that finite-size scaling aligns with known asymptotic bounds on hypergraph product codes [62], and demonstrates more fine-grained details of d vs. n scaling and the effects of weight reduction.

It is also natural to ask how the behaviors of our scheme change if some loss of distance is allowed, which is also considered in the original formulation of weight reduction in Ref. [44]. In Fig. 8, we showcase 6 examples with varying k and d values to demonstrate how overhead can be further reduced at the cost of a loss in distance. The required overhead tends to increase with rate, although the [[1429,49,12]](14,11) and [[1476,36,14]](12,15) codes are an exception, possibly because the difference in d is 2, instead of 1 as in other codes, and thus more influential on the overhead. This further hints at the trade-off between n, k, d at finite sizes, and may be of interest for near-term experiments when there are hard constraints on n. Also, it is possible that applying weight reduction to a larger code while allowing a small decrease in distance may yield better parameters than directly applying to a code with the target parameters.

On a last note, the creation of linearly dependent stabilizers (corresponding to meta-checks) when using product constructions has been overlooked in the context of weight reduction. In both methods we consider k to be fixed. However, after removing linearly dependent rows, k can be increased by a multiplicative factor without affecting other code parameters [69]. This applies to all the product codes considered, and our weight reduction procedure does not tend to prevent the formation of resulting linear dependencies. For instance, the [[6100,100,16]](6,3) code discovered by our RL policy has true rank 5683 and thus it actually has parameters [[6100,417,16]](6,3). This was not directly factored into our reward due to computational constraints as it slows down training and is mostly a feature of product code constructions that does not apply to weight reduction or code design in general, but the true k is generically even better if we take meta-

[Figure 12]

[Figure 13]

(a) (b)

- FIG. 7. Pairwise scatter plots for the parameters of the HGP-30 base codes and RL codes. (a) Hypergraph product code with parameters n, k, d, w, q before weight reduction, (b) RL codes after weight reduction. Each subplot compares two parameters (off-diagonal) while the diagonal entries depict individual parameter distributions.

[Figure 14]

- FIG. 8. Minimum qubit overheads under relaxed distance constraints. Permitting the distance d to decrease can reduce the total qubit overhead. Code parameters are shown in Table IV

for designing low-weight stabilizer QEC codes based on a highly efficient RL-based algorithm for weight reduction, which takes the effective route of starting with a code with the target distance and then optimizing weight, rather than designing a code from scratch. This method enables the discovery of an abundance of new low-weight codes that extend well beyond the previously accessible code parameter regime even with modest computational resources. In particular, while existing numerical methods generally stagnate at single-digit distance codes, our approach systematically generates efficient low-weight codes with distances in the tens—a regime expected to be crucial for experimental developments in the coming years. As low-weight QEC codes are critical components of fault-tolerant quantum computing, our findings pave the way for more feasible implementation of high-performance QEC, potentially bringing fault tolerance closer to realization in the near future.

Moreover, from the machine learning perspective, we have demonstrated that RL is particularly well-suited for stabilizer code design problems and, notably, far more scalable than previously thought. Our simple model is able to circumvent previous obstacles in designing codes with high distance and low weight, and exhibit great potential for further scalability, as discussed. Our current results are produced by running on 16 cores, and with about a thousand cores we expect to be able to design product codes with a few million qubits and non-product codes with a few thousand qubits.

checks into account.

#### IV. CONCLUSION AND OUTLOOK

In this work, we introduced a powerful new scheme

The main bottleneck in these estimates is the distance calculation. An avenue for future work could be using the spectral gap as part of the reward, which is commonly used to reason about distance in expander based constructions (although mostly in the asymptotic setting) The spectral gap is computable in polynomial time, as opposed to distance which is NP-hard [49], and highly parallelizable. This is a good heuristic to identify codes likely to have high distance, but does not fully replace the distance calculation. With further tuning we expect the efficiency of the algorithm and thus the accessible code sizes to improve further.

It should be noted that different choices of reward functions, model architectures, learning algorithms, hyperparameters, and learning representations, etc., often lead to vastly different outcomes [70]. It is likely that further tuning to this framework and scaling up computational resources [67] can lead to improvements of the results. Furthermore, as noted in Sec. II, our RL model can be adapted to codes with a variety of constraints on their structure, including geometry constraints and logical gates or symmetries, which would be worthy of further study. In particular, previous RL code design has focused solely on the memory perspective. As a key challenge for fault tolerance, it is crucial to address the discovery of codes with desired logical operations (Ref. [71] has

studied the identification of logical operations for a given code using machine learning). Additionally, this framework can naturally be extended to design qudit codes, by modifying the action space to take an additional parameter. Other future directions we consider particularly important include investigating the compatibility of our high-performance RL-designed codes with different experimental platforms and architectures, further analyzing their fault tolerance properties, and establishing a more complete understanding of the tradeoffs between distance, weight and other code parameters.

To conclude, our work highlights a promising new avenue where artificial intelligence can advance quantum computing through QEC code design. We anticipate vast opportunities for future work using artificial intelligence to discover codes and fault tolerance strategies at finite sizes, which hold great promise in uncovering new constructions that far surpass human-designed ones and accelerate the realization of scalable quantum technologies.

#### ACKNOWLEDGMENTS

ZWL is supported in part by a startup funding from YMSC, Tsinghua University, and NSFC under Grant No. 12475023.

- [1] M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum Information: 10th Anniversary Edition (Cambridge University Press, 2010).
- [2] P. W. Shor, Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer, SIAM Journal on Computing 26, 1484–1509 (1997).
- [3] S. Lloyd, Universal Quantum Simulators, Science 273, 1073 (1996).
- [4] V. Giovannetti, S. Lloyd, and L. Maccone, Advances in quantum metrology, Nature Photonics 5, 222–229 (2011).
- [5] P. W. Shor, Scheme for reducing decoherence in quantum computer memory, Phys. Rev. A 52, R2493 (1995).
- [6] D. Gottesman, Stabilizer Codes and Quantum Error Correction, Ph.D. thesis, California Institute of Technology, Pasadena, CA (1997).
- [7] P. W. Shor, Fault-tolerant quantum computation, in Proceedings of 37th Conference on Foundations of Computer Science (1996) pp. 56–65.
- [8] E. T. Campbell, B. M. Terhal, and C. Vuillot, Roads towards fault-tolerant universal quantum computation, Nature 549, 172–179 (2017).
- [9] A. Kitaev, Fault-tolerant quantum computation by anyons, Annals of Physics 303, 2–30 (2003).
- [10] B. Zeng, X. Chen, D.-L. Zhou, and X.-G. Wen, Quantum information meets quantum matter – from quantum entanglement to topological phase in many-body systems

(2018), arXiv:1508.02595 [cond-mat.str-el].

- [11] A. Almheiri, X. Dong, and D. Harlow, Bulk locality and quantum error correction in AdS/CFT, Journal of High Energy Physics 2015, 163 (2015).

- [12] J. Yi, W. Ye, D. Gottesman, and Z.-W. Liu, Complexity and order in approximate quantum error-correcting codes, Nature Physics 20, 1798–1803 (2024).
- [13] D. Poulin, Stabilizer formalism for operator quantum error correction, Phys. Rev. Lett. 95, 230504 (2005).
- [14] P. Aliferis and A. W. Cross, Subsystem fault tolerance with the bacon-shor code, Phys. Rev. Lett. 98, 220502

(2007).

- [15] M. B. Hastings and J. Haah, Dynamically Generated Logical Qubits, Quantum 5, 564 (2021).
- [16] X. Fu and D. Gottesman, Error correction in dynamical codes (2024), arXiv:2403.04163 [quant-ph].
- [17] D. Gottesman, Fault-tolerant quantum computation with constant overhead, Quantum Info. Comput. 14, 1338–1372 (2014).
- [18] N. P. Breuckmann and J. N. Eberhardt, Quantum lowdensity parity-check codes, PRX Quantum 2, 040101

- (2021).

[19] L. Z. Cohen, I. H. Kim, S. D. Bartlett, and B. J. Brown, Low-overhead fault-tolerant quantum computing using long-range connectivity, Science Advances 8, eabn1717

- (2022).

- [20] A. deMarti iOlius, I. E. Martinez, J. Roffe, and J. E. Martinez, An almost-linear time decoding algorithm for quantum ldpc codes under circuit-level noise (2024), arXiv:2409.01440 [quant-ph].
- [21] S. Bravyi, A. W. Cross, J. M. Gambetta, D. Maslov, P. Rall, and T. J. Yoder, High-threshold and lowoverhead fault-tolerant quantum memory, Nature 627, 778–782 (2024).

- [22] Q. Xu, J. P. Bonilla Ataides, C. A. Pattison, N. Raveendran, D. Bluvstein, J. Wurtz, B. Vasic´, M. D. Lukin, L. Jiang, and H. Zhou, Constant-overhead fault-tolerant quantum computation with reconfigurable atom arrays, Nature Physics 20, 1084–1090 (2024).
- [23] R. Raussendorf and J. Harrington, Fault-tolerant quantum computation with high threshold in two dimensions, Phys. Rev. Lett. 98, 190504 (2007).
- [24] A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. N. Cleland, Surface codes: Towards practical large-scale quantum computation, Phys. Rev. A 86, 032324 (2012).
- [25] Google Quantum AI, Suppressing quantum errors by scaling a surface code logical qubit, Nature 614, 676

- (2023).

[26] Google Quantum AI and Collaborators, Quantum error correction below the surface code threshold, Nature

- (2024).

- [27] S. Bravyi and B. Terhal, A no-go theorem for a twodimensional self-correcting quantum memory based on stabilizer codes, New Journal of Physics 11, 043029

(2009).

- [28] S. Bravyi, D. Poulin, and B. Terhal, Tradeoffs for reliable quantum information storage in 2D systems, Phys. Rev. Lett. 104, 050503 (2010).
- [29] N. Baspin and A. Krishna, Quantifying nonlocality: How outperforming local quantum codes is expensive, Phys. Rev. Lett. 129, 050505 (2022).
- [30] N. Baspin and A. Krishna, Connectivity constrains quantum codes, Quantum 6, 711 (2022).
- [31] N. Baspin, V. Guruswami, A. Krishna, and R. Li, Improved rate-distance trade-offs for quantum codes with restricted connectivity (2023), arXiv:2307.03283 [quantph].
- [32] D. Bluvstein, S. J. Evered, A. A. Geim, S. H. Li,

- H. Zhou, T. Manovitz, S. Ebadi, M. Cain, M. Kalinowski, D. Hangleiter, J. P. Bonilla Ataides, N. Maskara,
- I. Cong, X. Gao, P. Sales Rodriguez, T. Karolyshyn, G. Semeghini, M. J. Gullans, M. Greiner, V. Vuleti´c, and M. D. Lukin, Logical quantum processor based on reconfigurable atom arrays, Nature 626, 58–65 (2023).

- [33] P. Panteleev and G. Kalachev, Asymptotically good quantum and locally testable classical LDPC codes

(2022), arXiv:2111.03654 [cs.IT].

- [34] A. Leverrier and G. Zemor, Quantum Tanner codes, in 2022 IEEE 63rd Annual Symposium on Foundations of Computer Science (FOCS) (IEEE, 2022) p. 872–883.
- [35] I. Dinur, M.-H. Hsieh, T.-C. Lin, and T. Vidick, Good quantum LDPC codes with linear time decoders (2022), arXiv:2206.07750 [quant-ph].
- [36] M. Tremblay, G. Duclos-Cianci, and S. Kourtis, Finiterate sparse quantum codes aplenty, Quantum 7, 985

(2023).

- [37] J. Olle, R. Zen, M. Puviani, and F. Marquardt, Simultaneous discovery of quantum error correction codes and encoders with a noise-aware reinforcement learning agent, npj Quantum Information 10, 126 (2024).
- [38] C. Mauron, T. Farrelly, and T. M. Stace, Optimization of tensor network codes with reinforcement learning, New Journal of Physics 26, 023024 (2024).
- [39] V. P. Su, C. Cao, H.-Y. Hu, Y. Yanay, C. Tahan, and B. Swingle, Discovery of optimal quantum error correcting codes via reinforcement learning (2023), arXiv:2305.06378 [quant-ph].
- [40] H. P. Nautrup, N. Delfosse, V. Dunjko, H. J. Briegel, and

- N. Friis, Optimizing Quantum Error Correction Codes with Reinforcement Learning, Quantum 3, 215 (2019).
- [41] Y. Zeng, Z.-Y. Zhou, E. Rinaldi, C. Gneiting, and F. Nori, Approximate autonomous quantum error correction with reinforcement learning, Phys. Rev. Lett. 131, 050601 (2023).
- [42] D. Tandeitnik and T. Guerreiro, Evolving quantum circuits, Quantum Inf. Process. 23, 109 (2024).
- [43] M. Webster and D. Browne, Engineering quantum error correction codes using evolutionary algorithms (2024), arXiv:2409.13017 [quant-ph].
- [44] M. B. Hastings, Weight reduction for quantum codes (), arXiv:1611.03790 [quant-ph].
- [45] M. B. Hastings, On quantum weight reduction (), arXiv:2102.10030 [quant-ph].
- [46] E. Sabo, L. G. Gunderman, B. Ide, M. Vasmer, and G. Dauphinais, Weight-reduced stabilizer codes with lower overhead, PRX Quantum 5, 040302 (2024).
- [47] I. Tzitrin, T. Matsuura, R. N. Alexander, G. Dauphinais, J. E. Bourassa, K. K. Sabapathy, N. C. Menicucci, and I. Dhand, Fault-tolerant quantum computation with static linear optics, PRX Quantum 2, 040353 (2021).
- [48] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. (The MIT Press, 2018).
- [49] U. Kapshikar and S. Kundu, On the hardness of the minimum distance problem of quantum codes, IEEE Transactions on Information Theory 69, 6293–6302 (2023).
- [50] R. Sweke, M. S. Kesselring, E. P. L. van Nieuwenburg, and J. Eisert, Reinforcement learning decoders for faulttolerant quantum computation, Machine Learning: Science and Technology 2, 025005 (2020).
- [51] V. Sivak, M. Newman, and P. Klimov, Optimization of decoder priors for accurate quantum error correction

(2024), arXiv:2406.02700 [quant-ph].

- [52] B. C. A. Freire, N. Delfosse, and A. Leverrier, Optimizing hypergraph product codes with random walks, simulated annealing and reinforcement learning (2025), arXiv:2501.09622 [quant-ph].
- [53] S. Sgroi, G. M. Palma, and M. Paternostro, Reinforcement learning approach to nonequilibrium quantum thermodynamics, Phys. Rev. Lett. 126, 020601 (2021).
- [54] A. A. Melnikov, P. Sekatski, and N. Sangouard, Setting up experimental bell tests with reinforcement learning, Phys. Rev. Lett. 125, 160401 (2020).
- [55] A. Bolens and M. Heyl, Reinforcement learning for digital quantum simulation, Phys. Rev. Lett. 127, 110502

(2021).

- [56] F. Boccardo and O. Pierre-Louis, Reinforcement learning with thermal fluctuations at the nanoscale, Phys. Rev. E 110, L023301 (2024).
- [57] S. Borah, B. Sarma, M. Kewming, G. J. Milburn, and J. Twamley, Measurement-based feedback quantum control with deep reinforcement learning for a double-well nonlinear potential, Phys. Rev. Lett. 127, 190403 (2021).
- [58] D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. van den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe, J. Nham, N. Kalchbrenner, I. Sutskever, T. Lillicrap, M. Leach, K. Kavukcuoglu, T. Graepel, and D. Hassabis, Mastering the game of go with deep neural networks and tree search, Nature 529, 484 (2016).
- [59] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, Proximal policy optimization algorithms

(2017), arXiv:1707.06347 [cs.LG].

- [60] S. Huang and S. Onta˜no´n, A closer look at invalid action masking in policy gradient algorithms, The International FLAIRS Conference Proceedings 35 (2022).
- [61] T. A. Brun, Quantum Error Correction (Cambridge University Press, 2013).
- [62] J.-P. Tillich and G. Zemor, Quantum LDPC codes with positive rate and minimum distance proportional to n1/2, in 2009 IEEE International Symposium on Information Theory (2009) pp. 799–803.
- [63] A. Krishna and D. Poulin, Fault-tolerant gates on hypergraph product codes, Phys. Rev. X 11, 011023 (2021).
- [64] N. Connolly, V. Londe, A. Leverrier, and N. Delfosse, Fast erasure decoder for hypergraph product codes, Quantum 8, 1450 (2024).
- [65] R. Baart, T. Boothby, J. Cramwinckel, J. Fields, D. Joyner, R. Miller, E. Minkes, E. Roijackers, L. Ruscio, and C. Tjhai, GUAVA, a gap package for computing with error-correcting codes, Version 3.19, https://gap-packages.github.io/guava (2024).

- [66] GAP – Groups, Algorithms, and Programming, Version 4.14.0, The GAP Group (2024).
- [67] J. Hilton, J. Tang, and J. Schulman, Scaling laws for single-agent reinforcement learning (2023), arXiv:2301.13442 [cs.LG].
- [68] J. Preskill, Beyond NISQ: The Megaquop Machine

(2024), Q2B24 Silicon Valley.

- [69] D. Ostrev, D. Orsucci, F. La´zaro, and B. Matuz, Classical product code constructions for quantum CalderbankShor-Steane codes, Quantum 8, 1420 (2024).
- [70] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, Deep reinforcement learning that matters (2019), arXiv:1709.06560 [cs.LG].
- [71] H. Chen, M. Vasmer, N. P. Breuckmann, and E. Grant, Automated discovery of logical gates for quantum error correction, Quantum Information and Computation 22, 947–964 (2022).

#### SUPPLEMENTARY INFORMATION

The Supplementary Information includes additional data and diagrams that support and enrich the findings presented in the main text.

A. Data tables

- TABLE I. Some examples of Comparison of Hastings, Sabo et al. [46], and our RL weight reduction methods. Results using Hastings’ method are obtained from Ref. [46]. Data is also shown in Fig. 4

|Base Code|Hastings [44] Sabo et al. [46] RL(6,3)<br><br>|
|---|---|
|[[45, 9, 3]](8,3) [[74, 4, 4]](6,4) [[65, 9, 4]](8,3) [[58, 16, 3]](8,3)|[[2892, 9, 5]](6,6) [[65, 9, 4]](6,3) [[45, 9, 3]](6,3) [[7466, 4, 6]](6,8) [[100, 4, 4]](6,3) [[74, 4, 4]](6,3) [[6844, 9, 5]](6,8) [[89, 9, 4]](6,3) [[65, 9, 4]](6,3) [[5085, 16, 3]](6,8) [[136, 16, 4]](6,3) [[58, 16, 3]](6,3)<br><br>|

- TABLE II. Comparison of RL(6,3), RL(8,4), against SOTA on hypergraph product codes constructed from n = 30 classical codes. Data is also shown in Fig. 4

|Base Code|RL(6,3) RL(8,4) SOTA<br><br>|
|---|---|
|[[1741, 1, 30]](4,2) [[1684, 4, 20]](6,19) [[1629, 9, 16]](8,15) [[1576, 16, 16]](10,15) [[1525, 25, 15]](12,15) [[1476, 36, 14]](12,15) [[1429, 49, 12]](14,11) [[1384, 64, 12]](16,15) [[1341, 81, 12]](18,15) [[1300, 100, 11]](20,14) [[1261, 121, 10]](18,13) [[1224, 144, 9]](22,13) [[1189, 169, 8]](24,11) [[1156, 196, 8]](22,11) [[1125, 225, 8]](26,11) [[1096, 256, 7]](24,11) [[1069, 289, 6]](24,7) [[1044, 324, 6]](36,9) [[1021, 361, 6]](36,9) [[1000, 400, 5]](38,8) [[981, 441, 4]](32,5) [[964, 484, 4]](32,5) [[949, 529, 4]](28,5) [[936, 576, 4]](30,5) [[925, 625, 3]](32,5) [[916, 676, 2]](54,1) [[909, 729, 2]](56,1) [[904, 784, 2]](58,1) [[901, 841, 2]](60,1) [[900, 900, 1]](2,1)|[[1741, 1, 30]](4,2) [[1741, 1, 30]](4,2) [[1741, 1, 30]](4,2) [[1802, 4, 20]](6,3) [[1741, 4, 20]](8,4) [[7444, 4, 36]](6,3) [[1745, 9, 16]](6,3) [[1745, 9, 16]](8,4) [[9389, 9, 29]](6,3) [[1930, 16, 16]](6,3) [[1690, 16, 16]](8,4) [[15496, 16, 33]](6,3) [[2125, 25, 15]](6,3) [[1997, 25, 15]](8,4) [[23557, 25, 33]](6,3) [[2330, 36, 14]](6,3) [[2066, 36, 14]](8,4) [[33300, 36, 36]](6,3) [[2137, 49, 12]](6,3) [[1765, 49, 12]](8,4) [[27637, 49, 28]](6,3) [[2624, 64, 12]](6,3) [[1954, 64, 12]](8,4) [[41504, 64, 33]](6,3) [[3161, 81, 12]](6,3) [[2153, 81, 12]](8,4) [[52853, 81, 33]](6,3) [[3412, 100, 11]](6,3) [[2228, 100, 11]](8,4) [[59908, 100, 35]](6,3) [[3673, 121, 10]](6,3) [[2173, 121, 10]](8,4) [[70373, 121, 32]](6,3) [[3944, 144, 9]](6,3) [[2120, 144, 9]](8,4) [[73800, 144, 28]](6,3) [[4225, 169, 8]](6,3) [[2069, 169, 8]](8,4) [[44785, 169, 22]](6,3) [[4706, 196, 8]](6,3) [[2146, 196, 8]](8,4) [[51940, 196, 23]](6,3) [[5417, 225, 8]](6,3) [[2493, 225, 8]](8,4) [[66346, 225, 23]](6,3) [[6626, 256, 7]](6,3) [[2440, 256, 7]](8,4) [[63496, 256, 22]](6,3) [[6529, 289, 6]](6,3) [[2389, 289, 6]](8,4) [[40757, 289, 15]](6,3) [[6660, 324, 7]](6,3) [[2612, 324, 6]](8,4) [[54612, 324, 18]](6,3) [[8761, 361, 6]](6,3) [[2845, 361, 6]](8,4) [[56293, 361, 20]](6,3) [[6928, 400, 5]](6,3) [[2792, 400, 5]](8,4) [[50128, 400, 20]](6,3) [[7301, 441, 4]](6,3) [[2465, 441, 4]](8,4) [[10733, 441, 7]](6,3) [[7684, 484, 4]](6,3) [[2692, 484, 4]](8,4) [[14692, 484, 7]](6,3) [[9649, 529, 4]](6,3) [[2785, 529, 4]](8,4) [[12277, 529, 7]](6,3) [[12456, 576, 4]](6,3) [[3026, 576, 4]](8,4) [[17960, 576, 11]](6,3) [[6925, 625, 3]](6,3) [[3277, 625, 3]](8,4) [[15625, 625, 9]](6,3) [[1396, 676, 2]](6,3) [[1396, 676, 2]](8,4) [[2294, 676, 2]](6,2) [[1469, 729, 2]](6,3) [[1469, 729, 2]](8,4) [[3809, 729, 2]](6,2) [[1544, 784, 2]](6,3) [[1544, 784, 2]](8,4) [[3920, 784, 2]](6,2) [[1621, 841, 2]](6,3) [[1621, 841, 2]](8,4) [[4033, 841, 2]](6,2) [[900, 900, 1]](2,1) [[900, 900, 1]](2,1) [[900, 900, 1]](2,1)<br><br>|

- TABLE III. Some examples of comparisons between a RL produced code and various codes produced by the method from Sabo et al. [46] with different base codes. On each code the highest distance over 100 tries was taken for the method by Ref. [46].

Reduced Code Base Code RL(6,3) [[2257, 25, 15]](6,3) [[1525, 25, 15]](12,15)

[[2257, 25, 13]](6,3) [[3457, 25, 15]](6,3) [[4337, 25, 17]](6,3) [[4153, 25, 16]](6,3) [[4337, 25, 16]](6,3) [[4525, 25, 17]](6,3)

[[277, 25, 6]](6,10) [[325, 25, 7]](12,7) [[377, 25, 9]](12,7) [[433, 25, 8]](10,7) [[493, 25, 8]](10,7) [[557, 25, 8]](10,7)

Sabo et al. [46]

TABLE IV. Qubit overheads when allowing small reductions in distance at w = 6, q = 3. Data is also shown in Fig. 8

Base Code d d − 1 d − 2 d − 3 [[1525, 25, 15]](12,15) 600 293 112 0 [[1476, 36, 14]](12,15) 854 590 342 110 [[1429, 49, 12]](14,11) 708 456 108 0 [[1282, 64, 12]](16,15) 1240 1098 826 570 [[1241, 81, 11]](18,14) 1764 1320 912 660

[[1108, 100, 10]](16,13) 2304 1392 1120 864

B. Figures

Spectral properties of the codes produced by our RL agent are demonstrated in Fig. 9a and Fig. 9b. Pareto fronts of our (6,3) codes in terms of n,k,d,k/n,d/n are shown in Fig. 10. Further analysis of weight reduction trade-offs are made through a meta analysis of n,k,d in Fig. 11, with individual regressions for n vs. d at different k values shown in Fig. 12, Fig. 13, as well as an extrapolation for possible code parameters obtainable by our RL framework up to n = 20,000 in Fig. 14.

[Figure 15]

[Figure 16]

(a) (b)

##### FIG. 9. (a) Spectral gap evolution. Our RL agent causes the Tanner graphs to rapidly lose their expander properties, with

the gap λ1 − λ2 stabilizing around ≤ 0.5. This happens simultaneously to reduction of weight and degree in the Tanner graph. (b) Tanner graph eigenvalues evolution. The eigenvalues begin concentrated around +1 and −1, and spread out quickly. It is interesting that the codes our agent finds are significantly less structured and tend to have nearly random eigenvalue distributions. This suggests that our agent finds codes largely outside the realm of theoretical constructions, which often tend to rely on expansion-related arguments, although this is also at the cost of worse performance on message-passing-based decoding.

[Figure 17]

##### FIG. 10. Pareto fronts of n, k, d parameters. The plotted Pareto fronts show locally optimal (n, k, d) codes found by our RL agent. We observe there is considerable opportunity to improve k/n and d/n in the weight-reduction setting, especially since hypergraph product code codes cannot reach certain theoretical bounds. Note: The 10 additional points discussed in the main text are omitted from these plots.

[Figure 18]

[Figure 19]

(a) (b)

[Figure 20]

[Figure 21]

(c) (d)

- FIG. 11. (a) Meta-analysis of square-root regressions on RL n vs. d at various k values. Each curve shows how the best-fit n vs. d relationship changes at unique k. After weight reduction, the scaling factor of d to n decreases with k, eventually flattening at high k. In contrast, hypergraph product codes prior to weight reduction maintain a roughly constant scaling factor for all k. (b) Meta-analysis of linear regressions on RL n vs. k at various d values. Each line shows how n scales with k for fixed d. The coefficient tends to decrease both before and after weight reduction (c) Meta-analysis of square-root regressions for hypergraph product codes on n vs. d for varying k. Before weight reduction, the hypergraph product codes do not maintain low w, q; hence their scaling factors remain fairly constant across different k values. These points are also optimal for hypergraph product codes in terms of n, k, d but still exhibit growth in weight and degree not seen in RL-generated codes. (d) Meta-analysis of linear regressions for hypergraph product codes on n vs. k for varying d. Coefficients show a decreasing trend, although beginning at larger values than (b).

[Figure 22]

##### FIG. 12. Regressions of n and d for hypergraph product codes (no weight reduction). The coefficient in the n vs. d regressions remains mostly constant as k varies, and there is an observable auto-correlation in the placements of these points. At large k, both the range of d and the associated r2 values tend to drop, reflecting increased uncertainty at high code rates.

[Figure 23]

##### FIG. 13. Regressions of n and d for RL-generated codes (after weight reduction). Once weight and degree are reduced, the coefficients typically decrease. At large k, both the range of d and the associated r2 values tend to drop, reflecting increased uncertainty at high code rates. Also, some inefficiencies in the RL optimization process can improve the r2 fit for certain k values.

[Figure 24]

##### FIG. 14. Extrapolations of n vs. d regressions to n = 20, 000 for RL-generated (6,3) codes from the HGP-30 regime. Error bars shown in Fig. 13 are omitted for ease of visualization.

