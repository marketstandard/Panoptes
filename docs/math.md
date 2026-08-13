# Mathematical definitions

This document defines the calculations exposed by Panoptes. Every runtime output should link back to the relevant definition and state whether the value is calibrated evidence or descriptive context.

## Notation

- \(i\): token or segment index.
- \(n\): number of eligible watermark decisions.
- \(G\): number of observed green-list decisions.
- \(\gamma\): expected green-list fraction under the null hypothesis.
- \(z\): standardized normal statistic.
- \(p\): one-sided probability of evidence at least as extreme under the null.
- \(q\): false-discovery-corrected p-value.
- \(x\): standardized feature vector for a segment or document.
- \(\mu_m\): feature centroid for source family \(m\).
- \(\Sigma\): shrinkage covariance estimate.

## Known watermark statistic

For a green-list watermark with null green fraction \(\gamma\):

\[
z = \frac{G - \gamma n}{\sqrt{n\gamma(1-\gamma)}}.
\]

The one-sided p-value is:

\[
p = 1-\Phi(z),
\]

where \(\Phi\) is the standard normal cumulative distribution function. For small samples, a Binomial exact test may be used instead.

The green-rate confidence interval should use a binomial interval suitable for small counts, such as Wilson or exact intervals. The interval communicates uncertainty in signal strength without turning a p-value into a posterior probability.

## Editing and dilution model

Heavy editing can remove some watermark-bearing tokens while leaving others. Panoptes may model the retained signal fraction \(\rho\) with a Beta-Binomial process:

\[
G \mid \rho \sim \mathrm{Binomial}(n, \gamma + \rho(1-\gamma)).
\]

The model estimates whether the surviving evidence is compatible with dilution. It is interpretive and does not prove that editing occurred.

## False-discovery correction

When multiple schemes and windows are tested, Panoptes applies Benjamini-Hochberg correction. For ordered p-values \(p_{(1)} \le \cdots \le p_{(m)}\), the adjusted value for rank \(i\) is derived from:

\[
q_{(i)} = \min_{j \ge i}\left\{\frac{m}{j}p_{(j)}\right\}.
\]

Summary views use corrected evidence. Technical mode can show both p-values and q-values.

## Calibrated posterior

A detector score \(s\) is mapped through a calibration function \(g\) estimated on an **independent calibration set**:

\[
p = P(Y=1\mid s,x)=g(s;\theta_{\text{cohort}}).
\]

Calibration happens before the posterior is reported. Expected calibration error is a diagnostic of \(g\); it does not discount or rescale the posterior.

The likelihood ratio that can be combined with a user-declared prior uses the calibration-cohort prevalence \(\pi\):

\[
\mathrm{LR}=\frac{p}{1-p}\cdot\frac{1-\pi}{\pi},\qquad O_1 = O_0 \times \mathrm{LR}.
\]

When \(\pi=1/2\), this reduces to \(p/(1-p)\). Detector evidence (LR) and posterior probability are not the same quantity.

When detector outputs are correlated, Panoptes compares three evidence models rather than assuming segment independence: naive summation of segment log-likelihood ratios, correlation-aware shrinkage by effective sample size, and a document-level estimate.

## Source-family geometry

For source family \(m\), Mahalanobis distance is:

\[
d_m^2 = (x-\mu_m)^T\Sigma^{-1}(x-\mu_m).
\]

Distances are converted through a regularized multinomial logistic calibrator. This yields a conditional distribution:

\[
P(m \mid \text{AI, supported candidates, cohort}).
\]

The unknown score is separately calibrated using held-out generator families. If no candidate has sufficient support, Panoptes returns `unknown`.

## Conformal knownness

Conformal calibration chooses a nonconformity threshold \(t_\alpha\) on held-out known-family examples. A new input is known-set compatible when its score does not exceed the threshold. This provides finite-sample coverage under exchangeability assumptions, but it can degrade under domain shift.

## Segment evidence and change points

For each segment \(j\), Panoptes computes local detector evidence \(e_j\). A cumulative log-evidence series is:

\[
S_k = \sum_{j\le k} e_j.
\]

Change-point detection looks for large changes in \(S_k\) or the underlying segment distribution. Detected boundaries are hypotheses, not ground truth; uncertainty intervals should be shown when available.

## Calibration quality

Panoptes reports:

- Brier score: mean squared probability error;
- expected calibration error: average gap between predicted confidence and empirical frequency (a diagnostic, not a posterior discount);
- calibration slope and intercept: systematic probability distortion;
- reliability-bin range: observed frequency interval around the reported probability;
- conformal coverage: empirical fraction of true labels included on held-out data;
- selective risk at declared coverage levels;
- TPR at 0.1%, 1%, and 5% FPR.

Metrics are reported separately for prose and code, and by language/domain where sample size permits.
