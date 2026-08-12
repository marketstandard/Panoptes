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

A detector score \(s\) is mapped through a calibration function \(g\) estimated on held-out examples:

\[
P(Y=1\mid s,x)=g(s;\theta_{\text{cohort}}).
\]

Prior odds \(O_0=P(Y=1)/P(Y=0)\) can be adjusted explicitly. When evidence is represented as a calibrated likelihood ratio \(LR\), posterior odds are:

\[
O_1 = O_0 \times LR.
\]

When detector outputs are correlated, Panoptes uses a held-out meta-logistic model rather than multiplying raw scores.

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
- expected calibration error: average gap between predicted confidence and empirical frequency;
- reliability-bin range: observed frequency interval around the reported probability;
- conformal coverage: empirical fraction of true labels included on held-out data;
- TPR at fixed FPR: true-positive rate at a selected false-positive operating point.

Metrics are reported separately for prose and code, and by language/domain where sample size permits.
