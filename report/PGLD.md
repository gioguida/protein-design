# PGLD extension: forward KL / Jeffreys divergence library design

Status: planning
Owner: [you]
Related: PGLD paper (Sussex et al.), Section 4.3/5.3 (reverse-KL sampling from generative models), Section 5.4 (C05 influenza case study), DPO antibody design project

## 1. Problem

PGLD currently only supports reverse KL. The Mean-Entropy objective (Eq. 5/8 in the paper) is

$$J(\theta) = \mathbb{E}_{s \sim \pi_\theta}[\log p_{\text{model}}(s)] + \lambda H(\pi_\theta)$$

which is fine because $p_{\text{model}}$ is only ever evaluated at sequences we sampled ourselves. Forward KL needs the opposite:

$$J(\theta) = -\mathbb{E}_{s \sim p_{\text{model}}}[\log \pi_\theta(s)]$$

Now $\pi_\theta$ gets evaluated at sequences sampled from $p_{\text{model}}$, not from itself. If a template's IUPAC codes are hard (position 5 really is only "A"), any $p_{\text{model}}$ sample that doesn't fit any of the $M$ templates gives $\pi_\theta(s) = 0$, $\log = -\infty$. Not just a bad gradient, the scalar reward itself is undefined. This is why Weinstein et al. (Variational Synthesis) fit forward KL via EM on a static pre-sampled dataset instead of a REINFORCE loop.

## 2. Fix

Soften every mixed base so no entry is exactly $0$. E.g. "A" $= (1, 0, 0, 0)$ becomes $(1 - \varepsilon, \varepsilon/3, \varepsilon/3, \varepsilon/3)$, same idea for all $15$ IUPAC codes. Then $\pi_\theta(s) > 0$ everywhere, forward KL is always finite.

Important: this is a small change to Algorithm 1, not a new algorithm. Looking at the existing REINFORCE terms (Eq. 2 - template choice, template design, nucleotide sampling), they're already always finite, because they're log-probs of an event that was just sampled (you can't sample something with probability $0$). The $-\infty$ problem only lives inside the reward function $R(S)$, and only for objectives where $S$ is external data rather than samples from $\pi_\theta$.

So the plug-in is:

- Reverse KL (existing): sample $\theta \sim \mathrm{Cat}(\mathrm{softmax}(\psi))$, sample $S \sim \pi_\theta$, $R = \mathrm{mean}(\log p_{\text{model}}(s)) + \lambda H(\pi_\theta)$
- Forward KL (new): sample $\theta \sim \mathrm{Cat}(\mathrm{softmax}(\psi))$ (same line), $R = \mathrm{mean}_{s \in S_{\text{ext}}} \log \pi_\theta(s)$, where $S_{\text{ext}}$ is a fixed batch pre-sampled from $p_{\text{model}}$. Needs soft bases so this is finite for every $s \in S_{\text{ext}}$.
- Jeffreys: $R(\beta) = (1 - \beta)R_{\text{reverse}} + \beta R_{\text{forward}}$. $\beta$ sweeps mode-seeking <-> mode-covering.

Implementation note: the forward term's expectation is over $p_{\text{model}}$, which doesn't depend on $\psi$. So $\mathrm{d}/\mathrm{d}\psi$ of that term is a plain autodiff gradient (ordinary cross-entropy), not a REINFORCE estimator. Only the reverse term needs the score-function trick. Keep them as two separate backward passes summed together - should have lower variance than routing everything through REINFORCE.

## 3. Bias, made precise

For $s$ inside a template's true (hard) support, softening shifts $\log \pi_\theta(s)$ by $\mathcal{O}(L\varepsilon) \to 0$ as $\varepsilon \to 0$.

For $s$ mismatching the best-fitting template at $d$ positions, $\log \pi_\theta(s) \approx -d \log(1/\varepsilon) + \mathrm{const}$. Instead of a hard wall at $-\infty$ you get a smooth penalty that scales with Hamming distance to the nearest template. This is actually useful for optimization (same idea as label smoothing), not just a necessary evil.

TODO: write this up as a short lemma, same style as Theorem 1 in the paper (continuous relaxation optimum matches discrete optimum). Also worth an ablation on annealing $\varepsilon$ down over training (start loose, tighten late), same idea as Gumbel-softmax temperature schedules.

## 4. Known limitation, expect this, don't rediscover it

Weinstein et al. (2022) prove a lower bound: because each template samples each position independently, there's some $\eta > 0$ such that $\mathrm{KL}(p \| q_\theta) > \eta$ for all $M$ and $\theta$, whenever $p$ has correlations across positions. More templates shrink the gap but can't close it. Any PLM will have such correlations.

So pure forward KL should plateau. Good reason to expect Jeffreys (swept beta) to be the practically useful endpoint, not forward KL alone.

Sanity check before trusting anything downstream: sweep M, confirm the forward-KL gap shrinks the way their theory predicts. If it doesn't, something in the implementation is wrong, not the theory.

## 5. Redoing Section 5.4 (C05 influenza) via distribution matching

Currently 5.4 maximizes Mean-Entropy, i.e. reverse-KL projection onto a Boltzmann target $p^*(s) \propto \exp(f(s)/T)$, where $f = f_{\mathrm{H1N1}} + f_{\mathrm{H3N2}}$ from the DDMS-trained binding classifiers. Two ways to redo it:

Track A - ablation, same target. Keep $p^*(s) \propto \exp(f(s)/T)$ exactly as is, compare reverse-KL (current 5.4 result), forward-KL, and Jeffreys projections onto the same target. Isolates "does divergence choice matter" from "does the target change." Cleanest thing to put in a paper - same fitness models, same eval pipeline (theoretical diversity, edit distance histograms, expected hits in-silico).

Track B - new target, our DPO model. Use the DPO-tuned CDRH3 generative model (checkpoints already exist: evo_dpo, just_dpo) as $p_{\text{model}}$ directly instead of a Boltzmann reweighting of a discriminative predictor. Design a library that forward/Jeffreys-matches it. This connects the two projects: DPO aligns a generative model toward high-affinity CDRH3s -> PGLD synthesizes a physically realizable stochastic library approximating that model -> yeast display validates. Stronger paper narrative than "PGLD supports one more divergence."

Do A before B. A is the correctness/ablation check, B is the payoff. Don't commit to a real synthesis + yeast display round until in-silico results from A and B are in - synthesis and screening is the expensive step, don't spend it validating an ablation.

## 6. Plan

1. Implement soft IUPAC bases. $\varepsilon$ as a hyperparameter. Discrete-sampling loop in Algorithm 1 unchanged.
2. Write the bias lemma (in-support $\mathcal{O}(L\varepsilon)$, off-support $\mathcal{O}(d \log(1/\varepsilon))$).
3. Implement forward-KL reward and Jeffreys blend $R(\beta)$. Reuse existing marginal-gain / equal-share credit assignment for multi-template scaling (Section 3.2) - should carry over unchanged, still a black-box $R(S)$.
4. Validate against the Weinstein factorization bound. Sweep M, check the gap behaves as predicted.
5. Reproduce Section 5.3 with the new objective (IgLM/Trastuzumab, ESM2/C05). Add forward-KL and Jeffreys-KL PGLD next to Variational Synthesis and reverse-KL PGLD. Same metrics already in the paper: MMD V-stat, per-position TVD, likelihood CDF, early-stop %. Cheapest test, do this before touching C05 wet-lab pipeline.
6. Sweep $\beta$ on the influenza landscape in-silico (Pareto plot, forward vs reverse KL, same style as Fig. 7b/3a).
7. Track A in-silico.
8. Track B in-silico.
9. Decide with Scott which track (or both) merits real synthesis + yeast display, based on in-silico diversity/coverage numbers.
10. Write up as an extension section after 5.3, "generalizing to arbitrary divergences," with C05 pipeline as the applied result. Not a new paper.

## 7. Related work

- Weinstein et al., AISTATS 2022 - Variational Synthesis, forward KL via EM, the factorization lower bound $\eta$ (already ref [46] in the paper).
- Weinstein et al., Nature Biotechnology 2024/2026 - manufacturing-aware generative architectures, successor work. Check if they handle correlated positions / multi-base blocks, relevant to point 4 above (already ref [47]).
- BOND (Sessa et al., 2024) - $\beta$-weighted Jeffreys divergence for LLM alignment via distillation. Closest ML analogue to what we're doing here. Steal their framing and $\beta$ schedule if it's sane.
- Wake-sleep algorithm (Hinton et al., 1995) - historical precedent for alternating forward-KL (wake) / reverse-KL (sleep) fitting of a generative/recognition pair. Good for intro framing.
- Li, "Mode-Seeking Divergences: Theory and Applications to GANs" (2023) and "Adaptive Symmetrization of the KL Divergence" (2026) - general theory on which f-divergences are mode-seeking vs mode-covering. Relevant if we generalize past Jeffreys to a full f-divergence family later.
- Label smoothing (Szegedy et al., 2016) - the soft-base trick is a domain-specific instance of this. Useful for the bias write-up.
- Zhu et al. 2024 (Science Advances), Yang et al. 2023 (Decoil) - already in the paper's related work. Check if either has attempted forward-KL-style fitting to a target model rather than reward maximization - most likely direct competitor if so.

## 8. Open questions for Scott

- Priority: Track A or Track B first? Both doable, B is more publishable but depends on DPO checkpoints being in a state we trust.
- Does he want a formal bias bound or is an empirical $\varepsilon$-sweep enough for this paper?
- Given the factorization lower bound, any appetite for extending the template parameterization (arbitrary-codon blocks, Appendix A of the paper) to reduce $\eta$, or is that out of scope for this detour?