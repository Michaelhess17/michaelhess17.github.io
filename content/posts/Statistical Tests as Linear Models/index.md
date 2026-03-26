---
title: "Common Statistical Tests Are Just Linear Models"
date: 2026-03-25
draft: false
tags: ["math", "research", "tools", "tutorials"]
categories: ["math", "research", "tools", "tutorials"]
---

{{< katex >}}

{{< button href="https://colab.research.google.com/drive/1m6lWUMbTA_Sq6YHepwlOnyx_DZo8HUT9" target="_blank" rel="noopener noreferrer" >}}Open on Colab{{< /button >}}

> This is a Python translation of the original R-based document “Common
> statistical tests are linear models (or: how to teach stats)”. All
> code chunks use Python (pandas, numpy, seaborn/matplotlib,
> statsmodels).

> Original R source: https://github.com/lindeloev/tests-as-linear

------------------------------------------------------------------------

### What this notebook is doing

Many “named tests” (t-tests, correlations, ANOVA, chi-square, …) are special cases of a small number of model families:
- **Linear models (OLS)** for continuous outcomes: $y = \beta_0 + \beta_1 x + \cdots$
- **Generalized linear models (GLMs)** for counts/proportions (e.g., Poisson with a log link)
- **Rank-based tests** can often be understood as running the *same model* after transforming $y$ to ranks (this is a useful mental model, but the p-values are typically only an approximation).

Throughout, we fit the explicit model in `statsmodels` and compare it to SciPy’s dedicated test to show how the same structure reappears.

**A very helpful cheat sheet of what we'll be talking about:**
{{% figure caption="" src="linear_tests_cheat_sheet.png" %}}

## Setup


```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

sns.set(context="notebook", style="whitegrid")
np.random.seed(40)
```



Helper functions used throughout.


```python
def rnorm_fixed(N, mu=0.0, sd=1.0):
    """Generate N normal values with approximately given mean and SD."""
    x = np.random.normal(size=N)
    x = (x - x.mean()) / x.std(ddof=1)
    return x * sd + mu


def signed_rank(x):
    """Signed ranks: rank by |x|, then apply sign."""
    x = np.asarray(x)
    ranks = stats.rankdata(np.abs(x), method="average")
    return np.sign(x) * ranks


def print_df(df, decimals=4):
    """Pretty-print a DataFrame rounded to 'decimals'."""
    display(np.round(df, decimals))


def plot_intercept_only(y, title, y_label="y", beta0=None, ci=0.95):
    y = np.asarray(y)
    if beta0 is None:
        beta0 = float(np.mean(y))

    # 95% CI for the mean (t interval)
    n = len(y)
    if n > 1 and ci is not None:
        alpha = 1.0 - float(ci)
        se = float(np.std(y, ddof=1) / np.sqrt(n))
        tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        lower = beta0 - tcrit * se
        upper = beta0 + tcrit * se
    else:
        lower = upper = None

    jitter = np.random.uniform(-0.06, 0.06, size=len(y))
    x = np.zeros_like(y, dtype=float) + jitter

    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    ax.scatter(x, y, color="black", alpha=0.85)
    if lower is not None:
        ax.axhspan(lower, upper, color="red", alpha=0.15, label=r"95\% CI")
    ax.axhline(beta0, color="red", linewidth=2, label=r"$\hat{\beta}_0$")
    ax.set_xticks([])
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best")
    plt.show()


def _pointplot_mean_ci(ax, df, x_col, y_col, color="red", ci=95):
    """Draw group means with a CI errorbar in a seaborn-version-tolerant way."""
    try:
        # seaborn >= 0.12
        return sns.pointplot(
            data=df,
            x=x_col,
            y=y_col,
            estimator=np.mean,
            errorbar=("ci", ci),
            color=color,
            ax=ax,
        )
    except TypeError:
        # older seaborn
        return sns.pointplot(
            data=df,
            x=x_col,
            y=y_col,
            estimator=np.mean,
            ci=ci,
            color=color,
            ax=ax,
        )


def plot_two_groups(df, x_col, y_col, title, ci=95):
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    sns.stripplot(data=df, x=x_col, y=y_col, color="black", alpha=0.8, jitter=0.15, ax=ax)
    _pointplot_mean_ci(ax=ax, df=df, x_col=x_col, y_col=y_col, color="red", ci=ci)
    ax.set_title(title)
    plt.show()


def plot_one_way(df, x_col, y_col, title, ci=95):
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    sns.stripplot(data=df, x=x_col, y=y_col, color="black", alpha=0.75, jitter=0.15, ax=ax)
    _pointplot_mean_ci(ax=ax, df=df, x_col=x_col, y_col=y_col, color="red", ci=ci)
    ax.set_title(title)
    plt.show()
```

Basic “toy” data used by several sections.

Notes:
- The vectors `x`, `y`, `y2` are convenient placeholders used later for examples (e.g., paired tests use `y` vs `y2`).
- The *correlation* section uses its **own** correlated dataset (`x_corr`, `y_corr`) so the scatterplots visibly show a relationship (as in the original Rmd).
- Because these are simulated datasets, exact numerical p-values will change if you change the random seed or sample size.


```python
# Wide-ish
y  = np.concatenate([
    np.random.normal(size=15),
    np.exp(np.random.normal(size=15)),
    np.random.uniform(-3, 0, size=20)
])
x  = rnorm_fixed(50, mu=0.0, sd=1.0)         # for correlation
y2 = rnorm_fixed(50, mu=0.5, sd=1.5)         # second group

# Long format
value = np.concatenate([y, y2])
group = np.array(["y1"] * 50 + ["y2"] * 50)
D_long = pd.DataFrame({"value": value, "group": group})
```


## Pearson and Spearman correlation

### Linear-model view

Model:

$$ y = \beta_0 + \beta_1 x, \hspace{1em} H_0: \beta_1 = 0 $$

The model is trying to learn whether $y$ meaningfully varies with $x$. The null hypothesis is that $\beta_1 = 0$ (no linear association), and the alternative is that $\beta_1 \neq 0$.



```python
# Dedicated correlated data for this section 
mu = np.array([0.9, 0.9])
sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
xy_corr = np.random.multivariate_normal(mean=mu, cov=sigma, size=30)
x_corr = 0.5 * xy_corr[:, 0] + 0.2
y_corr = 0.5 * xy_corr[:, 1] + 0.4

# Built-in Pearson
pearson = stats.pearsonr(x_corr, y_corr)

# Linear model y ~ x
df_corr = pd.DataFrame({"x": x_corr, "y": y_corr})
model_lin = smf.ols("y ~ x", data=df_corr).fit()

# Standardized to recover r as slope
df_corr_std = df_corr.assign(
    x_std=(df_corr["x"] - df_corr["x"].mean()) / df_corr["x"].std(ddof=1),
    y_std=(df_corr["y"] - df_corr["y"].mean()) / df_corr["y"].std(ddof=1),
)
model_std = smf.ols("y_std ~ x_std", data=df_corr_std).fit()

res = pd.DataFrame({
    "model": ["scipy.pearsonr", "OLS_std", "OLS_raw"],
    "r_or_beta1": [
        pearson.statistic,
        model_std.params["x_std"],
        model_lin.params["x"],
    ],
    "p": [
        pearson.pvalue,
        model_std.pvalues["x_std"],
        model_lin.pvalues["x"],
    ]
})
print_df(res)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>r_or_beta1</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>scipy.pearsonr</td>
      <td>0.8159</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OLS_std</td>
      <td>0.8159</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>OLS_raw</td>
      <td>0.7306</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
plt.figure(figsize=(5, 4.2))
sns.regplot(
    x="x",
    y="y",
    data=df_corr,
    ci=95,
    scatter_kws={"color": "black"},
    line_kws={"color": "red"},
    label="Data",
)
plt.xlabel("x")
plt.ylabel("y")
plt.hlines(
    y=model_lin.params.Intercept,
    xmin=df_corr["x"].min(),
    xmax=df_corr["x"].max(),
    linestyles="dashed",
    label=r"$\beta_0$",
    color="blue",
)
plt.title("Pearson Correlation: y vs x with Linear Fit")
data_handle = mlines.Line2D([], [], color="black", marker="o", linestyle="None",
                            markersize=8, alpha=0.9, label="Data Points")
fit_handle = mlines.Line2D([], [], color="crimson", label="Linear Fit")
ci_handle = mpatches.Patch(color="crimson", alpha=0.15, label=r"95\% CI")
beta_0_handle = mlines.Line2D([], [], color="blue", linestyle="--", label=r"$\beta_0$")

plt.legend(handles=[data_handle, fit_handle, ci_handle, beta_0_handle])
plt.grid(True)
plt.show()
```


    
![png](index_nbconvert_files/index_nbconvert_8_0.png)
    



### Spearman correlation as Pearson on ranks

Model:

$$ \operatorname{rank}(y) = \beta_0 + \beta_1 \operatorname{rank}(x) $$

Spearman’s $\rho$ is (conceptually) Pearson correlation applied to **rank-transformed** data. A rank transform replaces values with their order (1 = smallest, 2 = next-smallest, …).

In this notebook we show this by:
- Computing `scipy.spearmanr(x, y)`, and
- Fitting an OLS line to $(\operatorname{rank}(x), \operatorname{rank}(y))$.

The slope from the rank-OLS fit lines up with $\rho$; small differences in p-values can occur because different implementations use slightly different approximations/tie handling.


```python
# Built-in Spearman on the same correlated sample
spearman = stats.spearmanr(x_corr, y_corr)

# Linear model on ranks
df_rank = pd.DataFrame({
    "rx": stats.rankdata(x_corr),
    "ry": stats.rankdata(y_corr),
})
model_spear = smf.ols("ry ~ rx", data=df_rank).fit()

res = pd.DataFrame({
    "model": ["scipy.spearmanr", "OLS_rank"],
    "rho_or_beta1": [
        spearman.statistic,
        model_spear.params["rx"],
    ],
    "p": [
        spearman.pvalue,
        model_spear.pvalues["rx"],
    ]
})
print_df(res)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>rho_or_beta1</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>scipy.spearmanr</td>
      <td>0.8367</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OLS_rank</td>
      <td>0.8367</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
plt.figure(figsize=(5, 4.2))
sns.regplot(x="rx", y="ry", data=df_rank, ci=95, scatter_kws={"color": "black"},  # Color for the scatter plot points
    line_kws={"color": "red"}, label="Data")
plt.xlabel("Rank of x")
plt.ylabel("Rank of y")
plt.hlines(y=model_spear.params.Intercept, xmin=df_rank["rx"].min(), xmax=df_rank["rx"].max(), linestyles="dashed", label=r"$\beta_0$", color="blue")
plt.title("Spearman Correlation: Ranks of y vs Ranks of x with Linear Fit")
data_handle = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                             markersize=8, alpha=0.9, label='Data Points')
fit_handle = mlines.Line2D([], [], color='crimson', label='Linear Fit')
ci_handle = mpatches.Patch(color='crimson', alpha=0.15, label=r'95\% CI')
beta_0_handle = mlines.Line2D([], [], color='blue', linestyle='--', label=r'$\beta_0$')
plt.legend(handles=[data_handle, fit_handle, ci_handle, beta_0_handle])
plt.grid(True)
plt.show()
```


    
![png](index_nbconvert_files/index_nbconvert_11_0.png)
    



## One mean: one-sample t-test and Wilcoxon signed-rank

### Linear-model view

One-sample t-test model:

$$ y = \beta_0, \hspace{1em} H_0: \beta_0 = 0 $$

Here, $\beta_0$ is the population mean (estimated by the sample mean). In OLS, the intercept-only model `y ~ 1` estimates $\hat\beta_0 = \bar y$, and the usual t-test on the intercept matches the classical one-sample t-test.

Wilcoxon signed-rank (approximate) model:

$$ y_{signed \hspace{0.2em} rank} = \beta_0 $$

This rank-based view helps connect Wilcoxon to the same linear-model *shape*, but the classical Wilcoxon p-value is not computed from the same t/F reference distribution as OLS, so expect small discrepancies—especially for small $n$, ties, or many zeros.


```python
# Data for this section
y1 = rnorm_fixed(20, mu=0.5, sd=0.6)

# One-sample t-test
t_res = stats.ttest_1samp(y1, popmean=0.0)

# Intercept-only linear model
df_t1 = pd.DataFrame({"y": y1})
model_t1 = smf.ols("y ~ 1", data=df_t1).fit()

res = pd.DataFrame({
    "model": ["scipy.ttest_1samp", "OLS_intercept"],
    "estimate": [y1.mean(), model_t1.params["Intercept"]],
    "t": [t_res.statistic, model_t1.tvalues["Intercept"]],
    "p": [t_res.pvalue, model_t1.pvalues["Intercept"]],
})
print_df(res)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>estimate</th>
      <th>t</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>scipy.ttest_1samp</td>
      <td>0.5</td>
      <td>3.7268</td>
      <td>0.0014</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OLS_intercept</td>
      <td>0.5</td>
      <td>3.7268</td>
      <td>0.0014</td>
    </tr>
  </tbody>
</table>
</div>



```python
plot_intercept_only(
    y1,
    title="One-sample t-test: intercept-only linear model",
    y_label="y",
    beta0=model_t1.params["Intercept"],
)
```


    
![png](index_nbconvert_files/index_nbconvert_14_0.png)
    


Wilcoxon signed-rank equivalence (approximate).

Important nuance: Wilcoxon is often described as “non-parametric”, but it’s helpful (pedagogically) to view it as a **test on signed ranks**. In this notebook we fit an intercept-only OLS model to signed ranks to highlight the shared structure.

This is a conceptual approximation: Wilcoxon’s classical p-value is based on a discrete rank-sum null distribution, while the OLS model uses a t/F reference distribution. They can be close for moderate/large $n$, but they are not guaranteed to match exactly.


```python
# Wilcoxon signed-rank test
w_stat, w_p = stats.wilcoxon(y1)

# Linear model on signed ranks
sr = signed_rank(y1)
df_w = pd.DataFrame({"sr": sr})
model_w = smf.ols("sr ~ 1", data=df_w).fit()

res = pd.DataFrame({
    "model": ["scipy.wilcoxon", "OLS_signed_rank"],
    "p": [w_p, model_w.pvalues["Intercept"]],
    "mean_signed_rank": [np.nan, model_w.params["Intercept"]],
})
print_df(res)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>p</th>
      <th>mean_signed_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>scipy.wilcoxon</td>
      <td>0.0023</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OLS_signed_rank</td>
      <td>0.0014</td>
      <td>7.8</td>
    </tr>
  </tbody>
</table>
</div>



```python
plot_intercept_only(
    sr,
    title="Wilcoxon signed-rank (approx): intercept-only linear model on signed ranks",
    y_label="signed rank",
    beta0=model_w.params["Intercept"],
)
```


    
![png](index_nbconvert_files/index_nbconvert_17_0.png)
    



## Paired samples t-test and Wilcoxon matched pairs

Paired t-test on differences:

$$ y_2 - y_1 = \beta_0, \hspace{1em} H_0: \beta_0 = 0 $$

The key idea is **non-independence**: each $y_2$ is paired with a specific $y_1$. The paired t-test reduces the problem to a one-sample test on the within-pair differences $d_i = y_{2,i} - y_{1,i}$.

In the linear-model view, this is again an intercept-only model on $d$: `diff ~ 1`.


```python
# Paired data y, y2 (50 each) already defined
paired_t = stats.ttest_rel(y, y2)

df_pair = pd.DataFrame({"diff": y - y2})
model_pair = smf.ols("diff ~ 1", data=df_pair).fit()

res = pd.DataFrame({
    "model": ["scipy.ttest_rel", "OLS_diff"],
    "mean_diff": [(y - y2).mean(), model_pair.params["Intercept"]],
    "t": [paired_t.statistic, model_pair.tvalues["Intercept"]],
    "p": [paired_t.pvalue, model_pair.pvalues["Intercept"]],
})
print_df(res)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>mean_diff</th>
      <th>t</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>scipy.ttest_rel</td>
      <td>-0.5689</td>
      <td>-1.6019</td>
      <td>0.1156</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OLS_diff</td>
      <td>-0.5689</td>
      <td>-1.6019</td>
      <td>0.1156</td>
    </tr>
  </tbody>
</table>
</div>



```python
plot_intercept_only(
    df_pair["diff"],
    title="Paired t-test: intercept-only linear model on differences",
    y_label="y - y2",
    beta0=model_pair.params["Intercept"],
)
```


    
![png](index_nbconvert_files/index_nbconvert_20_0.png)
    



Wilcoxon matched-pairs (approximate) as intercept-only on signed ranks
of differences.

As with the one-sample Wilcoxon: we rank-transform the paired differences (using signed ranks) and fit an intercept-only OLS model to show the shared structure.
The classical Wilcoxon matched-pairs test uses a discrete rank-based null distribution, so its p-values won’t generally be identical to OLS/t-test p-values.


```python
diff = y - y2
w_stat, w_p = stats.wilcoxon(diff)

sr = signed_rank(diff)
df_wpair = pd.DataFrame({"sr": sr})
model_wpair = smf.ols("sr ~ 1", data=df_wpair).fit()

res = pd.DataFrame({
    "model": ["scipy.wilcoxon", "OLS_signed_rank_diff"],
    "p": [w_p, model_wpair.pvalues["Intercept"]],
    "mean_signed_rank_diff": [np.nan, model_wpair.params["Intercept"]],
})
print_df(res)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>p</th>
      <th>mean_signed_rank_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>scipy.wilcoxon</td>
      <td>0.0951</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OLS_signed_rank_diff</td>
      <td>0.0942</td>
      <td>-6.94</td>
    </tr>
  </tbody>
</table>
</div>



```python
sr_diff = signed_rank(diff)
plot_intercept_only(
    sr_diff,
    title="Wilcoxon matched pairs (approx): intercept-only linear model on signed ranks of differences",
    y_label="signed rank",
    beta0=model_wpair.params["Intercept"],
)
```


    
![png](index_nbconvert_files/index_nbconvert_23_0.png)
    



## Two means: independent t-test and Mann–Whitney U

### Dummy coding and independent t-test

Model (two groups):

$$ y_i = \beta_0 + \beta_1 x_i, \quad x_i \in \{0,1\}, \quad H_0: \beta_1 = 0 $$

Think of $x_i$ as an indicator (“dummy variable”):
- If $x_i = 0$ (group A), the model predicts $y = \beta_0$ → the group A mean.
- If $x_i = 1$ (group B), the model predicts $y = \beta_0 + \beta_1$ → the group B mean.
So $\beta_1$ is the **mean difference** (B minus A). This is why an independent two-sample t-test is a special case of linear regression.


```python
N = 20
yA = rnorm_fixed(N, mu=0.3, sd=0.3)
yB = rnorm_fixed(N, mu=1.3, sd=0.3)

df_t2 = pd.DataFrame({
    "y": np.concatenate([yA, yB]),
    "group": np.array(["A"] * N + ["B"] * N),
})
df_t2["group_B"] = (df_t2["group"] == "B").astype(int)

# Classical independent t-test (equal var)
t_ind = stats.ttest_ind(yA, yB, equal_var=True)

# OLS with dummy
model_t2 = smf.ols("y ~ group_B", data=df_t2).fit()

res = pd.DataFrame({
    "model": ["scipy.ttest_ind", "OLS_dummy"],
    "mean_A": [yA.mean(), model_t2.params["Intercept"]],
    "mean_diff_B_minus_A": [yB.mean() - yA.mean(), model_t2.params["group_B"]],
    "t": [t_ind.statistic, model_t2.tvalues["group_B"]],
    "p": [t_ind.pvalue, model_t2.pvalues["group_B"]],
})
print_df(res)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>mean_A</th>
      <th>mean_diff_B_minus_A</th>
      <th>t</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>scipy.ttest_ind</td>
      <td>0.3</td>
      <td>1.0</td>
      <td>-10.5409</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OLS_dummy</td>
      <td>0.3</td>
      <td>1.0</td>
      <td>10.5409</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
plot_two_groups(
    df_t2,
    x_col="group",
    y_col="y",
    title="Independent t-test: dummy-coded OLS fits group means",
)
```


    
![png](index_nbconvert_files/index_nbconvert_26_0.png)
    



### Mann–Whitney U as linear model on ranks

Approximate model (ANOVA/t-test on ranks):

$$ \operatorname{rank}(y_i) = \beta_0 + \beta_1 x_i, \quad x_i \in \{0,1\}, \quad H_0: \beta_1 = 0 $$

A simple way to demystify Mann–Whitney is: replace $y$ by its ranks and then do the same two-group model. In that view, $\beta_1$ is a **mean rank difference**.

Caveat: the classical Mann–Whitney U p-value is computed from a U-statistic (exact/asymptotic, with tie corrections), not from an OLS t-test. The rank-OLS model is mainly here to expose the shared “one coefficient = group difference” structure.


```python
u_stat, u_p = stats.mannwhitneyu(yA, yB, alternative="two-sided")

df_mw = df_t2.copy()
df_mw["rank_y"] = stats.rankdata(df_mw["y"])
model_mw = smf.ols("rank_y ~ group_B", data=df_mw).fit()

res = pd.DataFrame({
    "model": ["scipy.mannwhitneyu", "OLS_rank"],
    "U_or_beta1": [u_stat, model_mw.params["group_B"]],
    "p": [u_p, model_mw.pvalues["group_B"]],
})
print_df(res)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>U_or_beta1</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>scipy.mannwhitneyu</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OLS_rank</td>
      <td>19.6</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
plot_two_groups(
    df_mw,
    x_col="group",
    y_col="rank_y",
    title="Mann–Whitney U (approx): OLS fits mean ranks by group",
)
```


    
![png](index_nbconvert_files/index_nbconvert_29_0.png)
    



## Welch’s t-test

Same linear mean structure, different group variances.  
We approximate Welch’s t-test using `statsmodels` WLS with
group-specific weights.


```python
# Welch's t-test
t_welch = stats.ttest_ind(yA, yB, equal_var=False)

# Approximate GLS via WLS with group-dependent variances
varA = yA.var(ddof=1)
varB = yB.var(ddof=1)
df_welch = df_t2.copy()
df_welch["var"] = np.where(df_welch["group"] == "A", varA, varB)
weights = 1.0 / df_welch["var"]

model_welch = smf.wls("y ~ group_B", data=df_welch, weights=weights).fit()

res = pd.DataFrame({
    "model": ["scipy.ttest_ind (Welch)", "WLS_group_var"],
    "mean_A": [yA.mean(), model_welch.params["Intercept"]],
    "mean_diff_B_minus_A": [yB.mean() - yA.mean(), model_welch.params["group_B"]],
    "t_or_t_like": [t_welch.statistic, model_welch.tvalues["group_B"]],
    "p": [t_welch.pvalue, model_welch.pvalues["group_B"]],
})
print_df(res)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>mean_A</th>
      <th>mean_diff_B_minus_A</th>
      <th>t_or_t_like</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>scipy.ttest_ind (Welch)</td>
      <td>0.3</td>
      <td>1.0</td>
      <td>-10.5409</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WLS_group_var</td>
      <td>0.3</td>
      <td>1.0</td>
      <td>10.5409</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
plot_two_groups(
    df_t2,
    x_col="group",
    y_col="y",
    title="Welch’s t-test: same mean structure, different variances",
)
```


    
![png](index_nbconvert_files/index_nbconvert_32_0.png)
    



## One-way ANOVA and Kruskal–Wallis

### One-way ANOVA as linear model

In `statsmodels`, formulas look like `y ~ x1 + x2` and are interpreted as “model **y** as a function of **x1** and **x2**”.

- The **left-hand side** (`value`) is the outcome/response column in the DataFrame.
- The **right-hand side** (`C(group)`) lists predictors (features).
- The `C(...)` wrapper means “treat this variable as **categorical**” (a factor), even if it’s stored as strings or integers.

So `value ~ C(group)` means: “estimate a separate mean for each level of `group`.” Under the hood, this is just ordinary regression with dummy variables. For three groups `a`, `b`, `c` (with `a` as the reference level), this corresponds to a model like:

$$ value_i = \beta_0 + \beta_1\,\mathbb{1}(group_i=b) + \beta_2\,\mathbb{1}(group_i=c) + \varepsilon_i. $$

Interpretation:
- Group `a` mean is $\beta_0$
- Group `b` mean is $\beta_0 + \beta_1$
- Group `c` mean is $\beta_0 + \beta_2$

The **one-way ANOVA F-test** is a single joint test of whether `group` matters at all, i.e. whether *any* group mean differs. In this dummy-variable view, that’s:

$$ H_0: \beta_1 = \beta_2 = 0 \quad (\text{all group means equal}). $$


```python
N = 30
a = rnorm_fixed(N, mu=0.0, sd=1.0)
b = rnorm_fixed(N, mu=1.0, sd=1.0)
c = rnorm_fixed(N, mu=-0.5, sd=1.0)

df_anova = pd.DataFrame({
    "value": np.concatenate([a, b, c]),
    "group": np.array(["a"] * N + ["b"] * N + ["c"] * N),
})

# Classical one-way ANOVA (F-test)
model_anova = smf.ols("value ~ C(group)", data=df_anova).fit()
anova_table = sm.stats.anova_lm(model_anova, typ=2)
print_df(anova_table)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(group)</th>
      <td>35.0</td>
      <td>2.0</td>
      <td>17.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>87.0</td>
      <td>87.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



```python
plot_one_way(
    df_anova,
    x_col="group",
    y_col="value",
    title="One-way ANOVA: OLS fits one mean per group",
)
```


    
![png](index_nbconvert_files/index_nbconvert_35_0.png)
    


### Kruskal–Wallis as ANOVA on ranks

Kruskal–Wallis is commonly taught as a “non-parametric one-way ANOVA”. A useful way to connect it to linear models is:
- Replace the outcome by its ranks: $\operatorname{rank}(y)$
- Fit the same one-way ANOVA/OLS model to those ranks

As above: the model structure matches, but **the classical Kruskal–Wallis p-value** is based on a $\chi^2$ reference distribution, while ANOVA uses an $F$ reference distribution. For that reason, the p-values won’t generally be identical (we also report a simple large-sample $F\to\chi^2$ rescaling to show the connection).


```python
# Kruskal–Wallis test
kw_stat, kw_p = stats.kruskal(a, b, c)

df_kw = df_anova.copy()
df_kw["rank_value"] = stats.rankdata(df_kw["value"])

model_kw = smf.ols("rank_value ~ C(group)", data=df_kw).fit()
anova_kw = sm.stats.anova_lm(model_kw, typ=2)

# OLS-on-ranks uses an F reference distribution; Kruskal–Wallis uses a chi-square reference.
# For large df_resid, (k-1)*F is often closer to the chi-square scale used by Kruskal–Wallis.
k = df_kw["group"].nunique()
F_group = float(anova_kw.loc["C(group)", "F"])
p_F = float(anova_kw.loc["C(group)", "PR(>F)"])
chi2_approx = (k - 1) * F_group
p_chi2_approx = stats.chi2.sf(chi2_approx, df=k - 1)

res = pd.DataFrame({
    "test": ["scipy.kruskal", "OLS_rank (ANOVA F)", "OLS_rank (F→chi2 approx)"],
    "stat": [kw_stat, F_group, chi2_approx],
    "df": [k - 1, (k - 1), (k - 1)],
    "p": [kw_p, p_F, p_chi2_approx],
})
print_df(res)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test</th>
      <th>stat</th>
      <th>df</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>scipy.kruskal</td>
      <td>25.1168</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OLS_rank (ANOVA F)</td>
      <td>17.1028</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>OLS_rank (F→chi2 approx)</td>
      <td>34.2055</td>
      <td>2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
plot_one_way(
    df_kw,
    x_col="group",
    y_col="rank_value",
    title="Kruskal–Wallis (approx): ANOVA/OLS on ranks fits mean ranks per group",
)
```


    
![png](index_nbconvert_files/index_nbconvert_38_0.png)
    



## Two-way ANOVA



```python
# Extend df_anova with a second factor: mood
df2 = df_anova.copy()
df2["mood"] = np.tile(["happy", "sad"], len(df2) // 2)

# Full two-way ANOVA (with interaction)
model_tw = smf.ols("value ~ C(group) * C(mood)", data=df2).fit()
anova_tw = sm.stats.anova_lm(model_tw, typ=2)
print_df(anova_tw)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(group)</th>
      <td>35.0000</td>
      <td>2.0</td>
      <td>17.1880</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>C(mood)</th>
      <td>1.0611</td>
      <td>1.0</td>
      <td>1.0422</td>
      <td>0.3102</td>
    </tr>
    <tr>
      <th>C(group):C(mood)</th>
      <td>0.4141</td>
      <td>2.0</td>
      <td>0.2033</td>
      <td>0.8164</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>85.5248</td>
      <td>84.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



```python
fig, ax = plt.subplots(figsize=(5, 4.2))
try:
    # seaborn >= 0.12
    sns.pointplot(
        data=df2,
        x="group",
        y="value",
        hue="mood",
        estimator=np.mean,
        errorbar=("ci", 95),
        dodge=True,
        ax=ax,
    )
except TypeError:
    # older seaborn
    sns.pointplot(
        data=df2,
        x="group",
        y="value",
        hue="mood",
        estimator=np.mean,
        ci=95,
        dodge=True,
        ax=ax,
    )
ax.set_title("Two-way ANOVA: group and mood means (with interaction)")
plt.show()
```


    
![png](index_nbconvert_files/index_nbconvert_41_0.png)
    



## ANCOVA

ANCOVA is “ANOVA + a continuous covariate”. In formula form we fit:

$$ value = \beta_0 + \text{(group effects)} + \beta_{age}\,age + \varepsilon. $$

In code this is: `value ~ C(group) + age`.
- `C(group)` means “treat `group` as categorical” (so we estimate an adjusted mean for each group).
- `age` is a numeric predictor (a slope shared across groups in this simple model).

### How to interpret the tests shown

We do nested-model comparisons:

1) **Does age matter after adjusting for group?**  
We compare the full model `value ~ C(group) + age` to `value ~ C(group)`. If the reported `Pr(>F)` is small, then `age` explains variation in `value` *beyond* what group alone explains.

2) **Does group matter after adjusting for age?**  
We compare the full model `value ~ C(group) + age` to `value ~ age`. If the reported `Pr(>F)` is small, then groups differ in their *adjusted means* (differences in `value` that remain even after controlling for `age`).

3) **Do groups have different age slopes? (interaction)**  
Add an interaction term and compare `value ~ C(group) + age` to `value ~ C(group) * age`.
Here, `C(group) * age` expands to `C(group) + age + C(group):age`. The interaction test is:

$$ H_0: \text{all } C(group):age \text{ coefficients} = 0 \quad (\text{same slope in every group}). $$

If the interaction p-value is small, the relationship between `age` and `value` depends on group (non-parallel lines).


```python
df3 = df_anova.copy()
df3["age"] = df3["value"] + rnorm_fixed(len(df3), sd=3.0) + 15 # Age is correlated with value

# Full ANCOVA
full_anc = smf.ols("value ~ C(group) + age", data=df3).fit()

# Test main effect of age
null_age = smf.ols("value ~ C(group)", data=df3).fit()
anova_age = sm.stats.anova_lm(null_age, full_anc)
print_df(anova_age)

# Test main effect of group
null_group = smf.ols("value ~ age", data=df3).fit()
anova_group = sm.stats.anova_lm(null_group, full_anc)
print_df(anova_group)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>df_resid</th>
      <th>ssr</th>
      <th>df_diff</th>
      <th>ss_diff</th>
      <th>F</th>
      <th>Pr(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>87.0</td>
      <td>87.000</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>86.0</td>
      <td>78.362</td>
      <td>1.0</td>
      <td>8.638</td>
      <td>9.48</td>
      <td>0.0028</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>df_resid</th>
      <th>ssr</th>
      <th>df_diff</th>
      <th>ss_diff</th>
      <th>F</th>
      <th>Pr(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>88.0</td>
      <td>110.6936</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>86.0</td>
      <td>78.3620</td>
      <td>2.0</td>
      <td>32.3316</td>
      <td>17.7415</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Interaction test: do slopes differ by group?
full_anc_int = smf.ols("value ~ C(group) * age", data=df3).fit()
anova_int_anc = sm.stats.anova_lm(full_anc, full_anc_int)
print_df(anova_int_anc)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>df_resid</th>
      <th>ssr</th>
      <th>df_diff</th>
      <th>ss_diff</th>
      <th>F</th>
      <th>Pr(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>86.0</td>
      <td>78.3620</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>84.0</td>
      <td>78.3516</td>
      <td>2.0</td>
      <td>0.0104</td>
      <td>0.0056</td>
      <td>0.9944</td>
    </tr>
  </tbody>
</table>
</div>



```python
df3_plot = df3.copy()
df3_plot["pred"] = full_anc.predict(df3_plot)

groups = sorted(df3_plot["group"].unique())
palette = dict(zip(groups, sns.color_palette(n_colors=len(groups))))

fig, ax = plt.subplots(figsize=(5, 4.2))
sns.scatterplot(
    data=df3_plot,
    x="age",
    y="value",
    hue="group",
    palette=palette,
    alpha=0.8,
    ax=ax,
 )

# Add fitted line + 95% CI band for the mean prediction in each group
for g, sub in df3_plot.sort_values("age").groupby("group"):
    pred_frame = full_anc.get_prediction(sub).summary_frame(alpha=0.05)
    ax.plot(sub["age"], pred_frame["mean"], color=palette[g], linewidth=2)
    ax.fill_between(
        sub["age"],
        pred_frame["mean_ci_lower"],
        pred_frame["mean_ci_upper"],
        color=palette[g],
        alpha=0.15,
        linewidth=0,
    )

ax.set_title("ANCOVA: linear fit per group (value ~ age + group)")
plt.show()
```


    
![png](index_nbconvert_files/index_nbconvert_45_0.png)
    



## Proportions: chi-square as log-linear model

Here we’re working with **counts** in categories (e.g., number of observations in each mood level). Two common ways to analyze this are:

1) A classical **chi-square test** (from SciPy), and
2) A **log-linear model** (a Poisson GLM with a log link) which is a regression way to model counts.

### Goodness of fit

SciPy’s call `stats.chisquare(counts)` performs a chi-square *goodness-of-fit* test. By default, it tests whether the expected counts are **equal across categories** (i.e., uniform proportions) unless you pass an explicit `f_exp=` expected vector.

- **Test statistic**: $\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}$, where $O_i$ are observed counts and $E_i$ are expected counts.
- **Degrees of freedom**: typically $k-1$ for $k$ categories when the expected proportions are fully specified.
- **p-value**: `pvalue` is $P(\chi^2_{df} \ge \chi^2_{obs} \mid H_0)$.

### The Poisson GLM output you see

The model `counts ~ C(mood)` is a **Poisson GLM** with a log link:

$$ \log(\mathbb{E}[counts\mid mood]) = \beta_0 + \beta_{meh}\,\mathbb{1}(mood=meh) + \beta_{sad}\,\mathbb{1}(mood=sad). $$

Interpretation of coefficients:
- `Intercept` corresponds to the baseline category (here the first mood level, e.g. `happy`). On the original count scale: $\mathbb{E}[count\mid happy] = \exp(\beta_0)$.
- `C(mood)[T.meh]` is a **log ratio** relative to baseline: $\exp(\beta_{meh}) = \frac{\mathbb{E}[count\mid meh]}{\mathbb{E}[count\mid happy]}$. Same idea for `sad`.

Why does it say `Df Residuals: 0` and `Deviance ~ 0`? Because with 3 categories you have **3 observations** (one per mood) and the model `counts ~ C(mood)` estimates **3 means** (one per mood). That’s a *saturated* model: it can fit the 3 counts essentially perfectly, leaving 0 residual degrees of freedom.

### Connecting the chi-square test to the GLM

To get a single chi-square-style test from the GLM side (comparable to `stats.chisquare`), compare:
- a **null** model with equal expected counts across categories: `counts ~ 1`, vs
- the **full** model allowing different means by category: `counts ~ C(mood)`.

The likelihood-ratio statistic $2(\ell_{full}-\ell_{null})$ is asymptotically $\chi^2$ with $df = k-1$.


```python
mood = np.array(["happy", "sad", "meh"])
counts = np.array([11, 14, 32])

df_g = pd.DataFrame({"mood": mood, "counts": counts})

# Chi-square goodness-of-fit
chi_g = stats.chisquare(counts)

# Poisson log-linear: counts ~ mood
model_g = smf.glm(
    "counts ~ C(mood)",
    data=df_g,
    family=sm.families.Poisson()
).fit()

print("chisquare:", chi_g)
print(model_g.summary())
```

    chisquare: Power_divergenceResult(statistic=np.float64(13.578947368421053), pvalue=np.float64(0.0011255610188435387))
                     Generalized Linear Model Regression Results                  
    ==============================================================================
    Dep. Variable:                 counts   No. Observations:                    3
    Model:                            GLM   Df Residuals:                        0
    Model Family:                 Poisson   Df Model:                            2
    Link Function:                    Log   Scale:                          1.0000
    Method:                          IRLS   Log-Likelihood:                -7.0243
    Date:                Wed, 25 Mar 2026   Deviance:                   2.2204e-16
    Time:                        21:39:54   Pearson chi2:                 9.59e-26
    No. Iterations:                     4   Pseudo R-squ. (CS):             0.9859
    Covariance Type:            nonrobust                                         
    ==================================================================================
                         coef    std err          z      P>|z|      [0.025      0.975]
    ----------------------------------------------------------------------------------
    Intercept          2.3979      0.302      7.953      0.000       1.807       2.989
    C(mood)[T.meh]     1.0678      0.350      3.055      0.002       0.383       1.753
    C(mood)[T.sad]     0.2412      0.403      0.599      0.549      -0.549       1.031
    ==================================================================================


    /home/michael/Documents/michaelhess17.github.io/content/posts/Statistical Tests as Linear Models/.pixi/envs/default/lib/python3.14/site-packages/statsmodels/regression/_tools.py:121: RuntimeWarning: divide by zero encountered in scalar divide
      scale = np.dot(wresid, wresid) / df_resid
    /home/michael/Documents/michaelhess17.github.io/content/posts/Statistical Tests as Linear Models/.pixi/envs/default/lib/python3.14/site-packages/statsmodels/genmod/generalized_linear_model.py:1342: PerfectSeparationWarning: Perfect separation or prediction detected, parameter may not be identified
      warnings.warn(msg, category=PerfectSeparationWarning)



```python
# Likelihood-ratio test: equal proportions (null) vs different by mood (full)
model_g_null = smf.glm(
    "counts ~ 1",
    data=df_g,
    family=sm.families.Poisson()
).fit()

# statsmodels' GLMResults may not expose compare_lr_test in all versions,
# so we compute the LRT directly from log-likelihoods.
lr_stat = 2.0 * (model_g.llf - model_g_null.llf)
lr_df = int(model_g.df_model - model_g_null.df_model)
lr_p = stats.chi2.sf(lr_stat, df=lr_df)

print("GLM LRT (counts ~ 1 vs counts ~ C(mood)):")
print("  LR stat:", lr_stat)
print("  df:", lr_df)
print("  p:", lr_p)
```

    GLM LRT (counts ~ 1 vs counts ~ C(mood)):
      LR stat: 12.788355384999704
      df: 2
      p: 0.0016712595880694237



## Notes

### How to read the results tables

In each section we usually show two things side-by-side:
- A **dedicated test** from SciPy (e.g. `ttest_1samp`, `wilcoxon`, `kruskal`, `chi2_contingency`)
- An explicit **model fit** in `statsmodels` (OLS/GLM), chosen to match the same mean structure

When an equivalence is exact (e.g. one-sample t-test ↔ intercept-only OLS; two-sample t-test ↔ dummy-coded OLS; ANOVA ↔ OLS with categorical predictors), the test statistics and p-values should match up to numerical tolerance.

When we say “approximate” (many rank-based examples), the goal is *conceptual unity*: the same linear-model shape appears after transforming the outcome to ranks. Classical rank tests use different null distributions and tie/zero handling, so expect small (sometimes not-so-small) p-value differences.

### About “non-parametric” tests

A useful teaching simplification is: many common “non-parametric” tests behave like their parametric counterparts **on rank-transformed data**. This does *not* mean the procedures are identical, nor that assumptions disappear; it is a way to understand what the test is sensitive to.

### Data in this notebook

All datasets here are simulated toy examples to illustrate model structure. If you change the random seed, sample sizes, or distributional choices, you’ll change the numeric outputs—but the modeling relationships should stay the same.
