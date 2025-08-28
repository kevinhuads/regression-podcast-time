# Core numerical / data-manipulation libraries
import numpy as np     
import pandas as pd   
import shap

import os, math, json, warnings, itertools, joblib, textwrap, requests
from dotenv import load_dotenv

# Statistical test used inside cramers_v
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf

# Plotting the LightGBM learning curve
import matplotlib.pyplot as plt        
import matplotlib.ticker as ticker    
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import seaborn as sns


from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

SEED = 3

# Seaborn custom theme
sns.set_theme(
    style="darkgrid",           
    rc={
        "figure.facecolor": "#0d1b2a",
        "axes.facecolor":   "#0d1b2a",
        "axes.edgecolor":   "#cccccc",
        "grid.color":       "#2a3f5f",
        "axes.labelcolor":  "#ffffff",
        "text.color":       "#ffffff",
        "xtick.color":      "#ffffff",
        "ytick.color":      "#ffffff",
    },
    palette="deep"               
)

# --------------------------- EDA Notebook ------------------------------

# Customize tables ------

dtype_palette = {
    "int64"         : "blue",
    "float64"       : "green",
    "object"        : "darkorange",
    "bool"          : "purple",
    "category"      : "teal",
    "datetime64[ns]": "brown"
}

# --- make all text white ---
plt.rcParams.update({
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
})

def colour_dtype(val):
    """Return a CSS 'color' style based on a dtype name."""
    return f"color: {dtype_palette.get(str(val), 'black')};"

# Value-based gradient for the missing column
def colour_gradient(val, cmap_name, vmin, vmax):
    """Return a CSS 'color' style using a colormap scaled to [vmin, vmax]."""
    if pd.isna(val):
        return ""
    norm  = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgb   = plt.get_cmap(cmap_name)(norm(val))[:3]        
    r,g,b = (int(255*c) for c in rgb)
    return f"color: rgb({r}, {g}, {b});"
    
# Function to calculate Cramér's V ---------
def cramers_v(x, y):
    """Compute Cramér's V for association between two categorical variables."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))
    
# Compute the Eta squared after the Anova test --------
def compute_eta_squared(df, x, y):
    """Compute eta-squared (effect size) from one-way ANOVA for x→y."""
    groups = [g[y].values for _, g in df[[x, y]].dropna().groupby(x)]
    grand_mean = df[y].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = sum((df[y] - grand_mean)**2)
    return ss_between / ss_total if ss_total != 0 else float('nan')
    

#plot quantile regression
def plot_quantile_regression_band(df,x, y, qs=(0.1, 0.5, 0.9),
    n_grid=100, ax=None,scatter_kw=None,band_kw=None,line_kw=None,palette="deep"):
    """Plot quantile regression lines and a shaded band for y ~ x."""
    # Prep
    qs = sorted(qs)
    q_low, q_high = qs[0], qs[-1]
    scatter_kw, band_kw, line_kw = scatter_kw or {}, band_kw or {}, line_kw or {}
    col_scatter, col_band = sns.color_palette(palette, 4)[0], sns.color_palette(palette, 4)[3]

    # Axis handle
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Fit quantile regressions
    formula = f"{y} ~ {x}"
    mod = smf.quantreg(formula, df)
    res_dict = {q: mod.fit(q=q) for q in qs}

    # Prediction grid
    x_range = np.linspace(df[x].min(), df[x].max(), n_grid)
    df_pred = pd.DataFrame({x: x_range})
    for q, res in res_dict.items():
        df_pred[f"p{q:.2f}"] = res.predict(df_pred)

    # Plotting
    # Scatter layer
    ax.scatter(df[x],df[y],s=25,alpha=0.35,
               color=scatter_kw.pop("color", col_scatter),
               label=scatter_kw.pop("label", "Observed"),
               **scatter_kw,)

    # Shaded band (between lowest and highest quantile requested)
    ax.fill_between(
        df_pred[x],
        df_pred[f"p{q_low:.2f}"],
        df_pred[f"p{q_high:.2f}"],
        color=band_kw.pop("color", col_band),
        alpha=band_kw.pop("alpha", 0.25),
        label=band_kw.pop("label", f"{q_low:.1%}–{q_high:.1%} band"),
        **band_kw,
    )

    # Quantile lines
    for q in qs:
        style = {"lw": 2}
        if np.isclose(q, 0.50):  # highlight median
            style["ls"] = line_kw.pop("ls", "--")
            style["color"] = line_kw.pop("color", "white")
            style["label"] = line_kw.pop("label", "Median (0.5)")
        else:
            style["color"] = line_kw.get("color", col_band)
            style["label"] = line_kw.get("label", "")
        style.update(line_kw)
        ax.plot(df_pred[x], df_pred[f"p{q:.2f}"], **style)

    # Cosmetics
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    ax.set_title(
        f"{y.replace('_', ' ').title()} vs. {x.replace('_', ' ').title()} "
        f"with {q_low:.1%}–{q_high:.1%} quantile band"
    )
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig, ax, res_dict

# --------------------------- Final Analysis Notebook ------------------------------

### Get Kaggle Results ---

def get_kaggle_results(user, key, path, overwrite, comp, models_names):
    """Fetch Kaggle submission scores for given models and cache them to CSV."""
    if not os.path.exists(path) or overwrite:

        session = requests.Session()
        session.auth = (user, key)

        url = f"https://www.kaggle.com/api/v1/competitions/submissions/list/{comp}?page=1"
        r = session.get(url, timeout=30)

        if r.status_code == 401:
            raise SystemExit("401 Unauthorized: username/key are wrong.")
        if r.status_code == 403:
            raise SystemExit("403 Forbidden: join the competition and accept the rules on Kaggle, then retry.")
        if r.status_code == 404:
            raise SystemExit(f"404 Not Found: competition slug '{comp}' is wrong.")
        r.raise_for_status()

        perfs = pd.DataFrame(r.json())

        cols = [c for c in perfs.columns if c.lower() in {
            "ref","description","publicscore","privatescore","date","status","filename"
        }]
        perfs = perfs[cols]
        perfs = perfs.drop(["ref","date","description","status"], axis = 1)

        perfs = pd.DataFrame([perfs.loc[perfs["fileName"] == f"predictions_{model}.csv"][["publicScore","privateScore"]].iloc[0] for model in models_names + ['stack']])
        perfs.index = models_names + ['stack']
        perfs[["publicScore","privateScore"]] = perfs[["publicScore","privateScore"]].astype(float)
        perfs.to_csv(path)

    else:
        perfs = pd.read_csv(path,index_col=0)
    return perfs


### RMSE Plot ---

# Create a new figure and axes
def new_fig_ax(figsize=(10, 6)):
    """Create and return a new (fig, ax) with the given figsize."""
    return plt.subplots(figsize=figsize)

# Compute padded y-limits from arrays
def auto_ylim(ax, *ys, pad_frac=0.10, min_pad=0.5):
    """Set y-limits with padding inferred from provided series."""
    vals = [v for arr in ys if arr is not None for v in (arr if hasattr(arr, "__iter__") else [arr])]
    if not vals:
        return
    y_min, y_max = min(vals), max(vals)
    pad = pad_frac * (y_max - y_min) if y_max > y_min else min_pad
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.margins(x=0.10)

# Annotate centered text with vertical offset
def annotate_centered(ax, x, y, text="stack", dy=15, color="white", fontsize=12, weight="bold"):
    """Annotate a point with centered text offset vertically (in points)."""
    ax.annotate(text, xy=(x, y), xytext=(0, dy), textcoords="offset points",
                ha="center", color=color, fontsize=fontsize, weight=weight)

# Plot one series plus highlighted stack point
def plot_series_with_stack(ax, x_labels, y, stack_label, *,
                           base_label, base_color="C0", base_alpha=1.0,
                           marker="o", s_model=125, edgecolor="white", lw_model=0.8,
                           stack_color="red", stack_alpha=1.0, s_stack=250, lw_stack=1.2,
                           zorder_stack=5, stack_label_text=None):
    """Scatter a series and highlight the 'stack' label; return its y-value."""
    d = dict(zip(x_labels, y))
    base_x = [x for x in x_labels if x != stack_label]
    ax.scatter(base_x, [d[x] for x in base_x],
               s=s_model, marker=marker, c=base_color,
               edgecolors=edgecolor, linewidths=lw_model,
               alpha=base_alpha, label=base_label)
    ax.scatter([stack_label], [d[stack_label]],
               s=s_stack, marker=marker, c=stack_color,
               edgecolors="white", linewidths=lw_stack,
               alpha=stack_alpha, zorder=zorder_stack,
               label=stack_label_text or f"{base_label} (stack)")
    return d[stack_label]

# Overlay hollow markers across all labels
def overlay_hollow(ax, x_labels, y, *, label="CV RMSE",
                   marker="X", size=150, edgecolor="white",
                   lw=1.6, alpha=0.30, zorder=4):
    """Overlay hollow markers across labels for comparison."""
    ax.scatter(x_labels, y, s=size, marker=marker, facecolors="none",
               edgecolors=edgecolor, linewidths=lw, alpha=alpha,
               label=label, zorder=zorder)

# Apply title, labels, legend, layout
def finalize(ax, *, title, ylabel, xlabel="model", title_color="w", legend_kw=None):
    """Apply title/labels/legend to an axis and tighten layout."""
    ax.set_title(title, color=title_color, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend_kw is None:
        legend_kw = dict(frameon=False, labelspacing=1.05)
    ax.legend(**legend_kw)
    plt.tight_layout()

# Plot RMSE with public/private + CV overlay
def plot_cv_rmse(x_labels, public, private, cv_rmse, stack_label,
                 *, base_color="C0", stack_color="red",
                 private_alpha=0.30, cv_alpha=0.30,
                 s_model=125, s_stack=250):
    """Plot Public/Private RMSE with CV overlay and highlight the stack model."""
    fig, ax = new_fig_ax()

    y_stack_pub = plot_series_with_stack(
        ax, x_labels, public, stack_label,
        base_label="Public (base)", base_color=base_color, base_alpha=1.0,
        s_model=s_model, stack_color=stack_color, stack_alpha=1.0, s_stack=s_stack
    )
    plot_series_with_stack(
        ax, x_labels, private, stack_label,
        base_label="Private (base)", base_color=base_color, base_alpha=private_alpha,
        s_model=s_model, stack_color=stack_color, stack_alpha=private_alpha, s_stack=s_stack
    )

    overlay_hollow(ax, x_labels, cv_rmse, label="CV RMSE", alpha=cv_alpha)
    annotate_centered(ax, stack_label, y_stack_pub, dy=15)
    auto_ylim(ax, public, private, cv_rmse)
    finalize(ax, title="Model RMSE - Public, Private & CV", ylabel="RMSE")
    return fig, ax

# Plot CV R² with highlighted stack point
def plot_cv_r2(x_labels, cv_r2, stack_label,
               *, base_color="C0", stack_color="red",
               s_model=125, s_stack=250):
    """Plot CV R² by model and highlight the stack model."""
    fig, ax = new_fig_ax()

    y_stack = plot_series_with_stack(
        ax, x_labels, cv_r2, stack_label,
        base_label="CV R²", base_color=base_color, base_alpha=1.0,
        s_model=s_model, stack_color=stack_color, stack_alpha=1.0, s_stack=s_stack
    )

    annotate_centered(ax, stack_label, y_stack, dy=-25)
    auto_ylim(ax, cv_r2)
    finalize(ax, title="Model CV R²", ylabel="R²")
    return fig, ax

### Customize Heatmap ----

def set_bold(label, ticks_list):
    """Bold, resize, and recolor a tick label that matches the target text."""
    for lbl in ticks_list:
        if lbl.get_text() == label:
            lbl.set_fontweight("bold")
            lbl.set_size(14)
            lbl.set_color("white")

## barplot 

def parent_of(feature, X_cat, X_num):
    """Return original column for an encoded feature name."""
    name = str(feature)
    if "__" in name:
        name = name.split("__")[-1]          # drop prefixes like 'prep__cat__'
    if name in X_cat or name in X_num:
        return name                           # passthrough numeric/cat
    for col in list(X_cat) + list(X_num):     # one-hot: '<col>_<category>'
        if name.startswith(f"{col}_"):
            return col
    return None


def make_color_map(keys, cmap="tab20"):
    """Map each key (parent column) to a stable color."""
    import numpy as np
    from matplotlib import cm
    uniq = [k for k in dict.fromkeys([k for k in keys if k is not None])]
    colors = cm.get_cmap(cmap, max(1, len(uniq)))(np.arange(len(uniq)))
    return {k: colors[i] for i, k in enumerate(uniq)}


def barh_percent(df, x_col, y_col, title=None, top_n=15, cmap=None,
                 color_map=None, color_by=None):
    """
    Horizontal % bar chart (top-N). Highest at top. Right-end annotations.
    - color_map: dict {key -> color}
    - color_by: column used to look up color_map (defaults to x_col)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm

    d = (df[[x_col, y_col] + ([color_by] if color_by and color_by != x_col else [])]
         .dropna(subset=[x_col, y_col])
         .sort_values(y_col, ascending=False)
         .head(top_n)
         .copy())

    total = float(df[y_col].sum())
    d["_pct"] = 100 * d[y_col] / total if total else 0.0

    # Colors
    default_colors = cm.get_cmap(cmap, len(d))(np.arange(len(d))) if cmap else None
    key_col = color_by or x_col
    if color_map:
        keys = d[key_col].astype(str).tolist()
        colors = [color_map.get(k, (default_colors[i] if default_colors is not None else None))
                  for i, k in enumerate(keys)]
    else:
        colors = default_colors

    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(d))))
    bars = ax.barh(np.arange(len(d)), d["_pct"].to_numpy(), color=colors)

    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d[x_col].astype(str).to_list())
    ax.set_xlabel("Share of total MDI (%)")
    if title: ax.set_title(title)

    xmax = max(100.0, float(d["_pct"].max()) * 1.10)
    ax.set_xlim(0, xmax)

    for rect, pct in zip(bars, d["_pct"].to_numpy()):
        ax.text(rect.get_width() + xmax * 0.01,
                rect.get_y() + rect.get_height() / 2,
                f"{pct:.1f}%", va="center", ha="left")

    ax.invert_yaxis()  # highest at the top
    fig.tight_layout()
    return ax


# SHAP -----------

def to_numeric_codes(df: pd.DataFrame, X_num, X_cat):
    """Return a numeric-coded copy: floats for X_num, int codes for X_cat."""
    Z = df.copy()
    if X_cat:
        Z[X_cat] = Z[X_cat].apply(lambda s: s.astype("category").cat.codes.astype("int64"))
    if X_num:
        Z[X_num] = Z[X_num].astype("float64")
    return Z

def from_numeric_codes(Z, template: pd.DataFrame, X_num, X_cat):
    """
    Rebuild a DataFrame like `template` from a numeric-coded array/DataFrame Z.
    Uses template’s categorical dtypes (categories/order) and original column order.
    """
    cols = list(template.columns)
    if not isinstance(Z, pd.DataFrame):
        Z = pd.DataFrame(Z, columns=cols)

    out = {}
    # restore categoricals using template's dtype (keeps categories & order)
    for c in X_cat:
        dtype = template[c].dtype
        out[c] = pd.Categorical.from_codes(Z[c].astype("int64").to_numpy(), dtype=dtype)
    # restore numerics
    for c in X_num:
        out[c] = Z[c].astype("float64").to_numpy()

    df = pd.DataFrame(out, columns=cols)
    # enforce dtypes explicitly
    for c in X_cat:
        df[c] = df[c].astype(template[c].dtype)
    for c in X_num:
        df[c] = df[c].astype("float64")
    return df



def contains_ohe(est) -> bool:
    """Recursively detect whether an estimator/transformer contains a OneHotEncoder."""
    # direct hit
    if isinstance(est, OneHotEncoder):
        return True
    
    # inside a Pipeline
    if isinstance(est, Pipeline):
        return any(contains_ohe(step) for _, step in est.steps)
    
    # inside a ColumnTransformer (before or after fit)
    if isinstance(est, ColumnTransformer):
        transformers = getattr(est, "transformers_", getattr(est, "transformers", []))
        for _, trans, _ in transformers:
            if trans in ("drop", "passthrough"):
                continue
            if contains_ohe(trans):
                return True
    return False



def compute_shap_payload(models, model_name, X, N_SAMPLE, X_num, X_cat,n_bg = 200, max_evals = 2000):
    """
    Compute SHAP Explanation for a regression Pipeline on raw features with categoricals.
    Returns a payload dict ready to cache (includes sv + sample indices + columns + meta).
    """
    # rows to explain (exact indices passed in)
    
    search = models[model_name]
    best = search.best_estimator_
    pre  = best.named_steps["pre"]
    
    has_ohe = contains_ohe(pre)

    # Decide whether we can use TreeExplainer (only when there's no OHE and the model is supported)
    can_use_tree = False
    tree_explainer = None
    if not has_ohe:
        try:
            # If this raises (e.g., InvalidModelError), we’ll fall back to the generic path
            tree_explainer = shap.TreeExplainer(model)
            can_use_tree = True
        except Exception:
            can_use_tree = False

    if has_ohe or not can_use_tree:
        X_sample_idx = X.sample(n=min(N_SAMPLE, len(X)), random_state=SEED).index
        X_sample = X.loc[X_sample_idx]


        background_raw = shap.utils.sample(X, n_bg, random_state=SEED)
        bg_num = to_numeric_codes(background_raw, X_num, X_cat)

        clust = shap.utils.hclust(bg_num.values, metric="correlation")
        masker_obj = shap.maskers.Partition(bg_num, max_samples=n_bg, clustering=clust)

        # model wrapper: accept numeric-coded, rebuild raw, call pipeline.predict
        def f_num(Z):
            raw = from_numeric_codes(Z, template=X, X_num=X_num, X_cat=X_cat)
            return best.predict(raw)  # regression → 1D

        explainer = shap.Explainer(f_num, masker_obj, algorithm="partition")

        # compute SHAP
        X_sample_num = to_numeric_codes(X_sample, X_num, X_cat)
        sv = explainer(X_sample_num, max_evals=max_evals)
    else:
        # preprocess once; we’ll pass the raw matrix to SHAP
        pre.set_output(transform="pandas")
        model = best.named_steps[model_name]
        X_pp = pre.transform(X)        
        X_sample = X_pp.sample(n=1000, random_state=SEED)
        feature_names = X_pp.columns        
        explainer = shap.TreeExplainer(model)
        sv = explainer(X_sample, check_additivity=False)   

    return sv

### SHAP Analysis -------

def _safe_feature_names(sv):
    """Prefer sv.feature_names; fall back to columns in sv.data; else generic."""
    names = getattr(sv, "feature_names", None)
    if names is not None:
        return list(names)
    data = getattr(sv, "data", None)
    if hasattr(data, "columns"):
        return list(data.columns)
    n_features = np.asarray(sv.values).shape[-1]
    return [f"f{j}" for j in range(n_features)]

def summarize_explanation(sv):
    """
    Reduce a SHAP Explanation to per-feature summary vectors (single vector per model):
      - imp_abs_mean: mean absolute SHAP (global importance, unnormalized)
      - net_effect: mean signed SHAP (direction on average)
      - pos_rate: fraction of positive SHAP values among *all* values
    Handles shapes (n, p) and (n, K, p) by averaging across samples (and outputs if present).
    """
    vals = np.asarray(sv.values)  # (n, p) or (n, K, p)
    if vals.ndim == 2:
        # (n, p)
        imp_abs_mean = np.nanmean(np.abs(vals), axis=0)
        net_effect   = np.nanmean(vals, axis=0)
        pos_rate     = np.mean(vals > 0, axis=0)
        nz_mask      = np.abs(vals) > 0
        
    elif vals.ndim == 3:
        # (n, K, p) -> average over samples and outputs
        imp_abs_mean = np.nanmean(np.abs(vals), axis=(0, 1))
        net_effect   = np.nanmean(vals, axis=(0, 1))
        pos_rate     = np.mean(vals > 0, axis=(0, 1))
        nz_mask      = np.abs(vals) > 0
    else:
        raise ValueError(f"Unexpected sv.values ndim={vals.ndim}")
    feat_names = _safe_feature_names(sv)
    return imp_abs_mean, net_effect, pos_rate, feat_names

def normalize_series(s, min_total=1e-12):
    """L1-normalize a nonnegative importance vector; return zeros if tiny sum."""
    total = s.sum()
    if not np.isfinite(total) or abs(total) < min_total:
        return s.copy() * 0.0
    return s / total

### Plot Residuals -----

def plot_residuals(df, y, m, color_col="C0", cmap=None, show_legend=False):
    """Plot Predicted vs True and Residuals vs Predictions side by side."""
    if isinstance(cmap, (list, tuple)):
        cmap = mcolors.ListedColormap(list(cmap))

    # if cmap is None and color_col looks like a binary label vector,
    if cmap is None:
        is_array_like = hasattr(color_col, "__len__") and not isinstance(color_col, (str, bytes))
        if is_array_like:
            vals = [v for v in color_col if v is not None]
            sample = vals[:50]
            looks_like_explicit_colors = len(sample) > 0 and all(mcolors.is_color_like(v) for v in sample)
            if not looks_like_explicit_colors:
                uniques = set([v for v in color_col if v is not None])
                if len(uniques) == 2:
                    cmap = mcolors.ListedColormap(["#1f77b4", "#d62728"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    supt = fig.suptitle(f"Model: {m}", fontsize=18, fontweight="bold", y=0.92)

    # Left plot: Predicted vs True
    ax = axes[0]
    sc0 = ax.scatter(df[y], df[m], c=color_col, cmap=cmap, s=8, alpha=0.6)
    if show_legend:
        handles, labels = sc0.legend_elements()
        legend_title = getattr(color_col, "name", None)

        dummy = mlines.Line2D([], [], linestyle="none", marker="", label=legend_title)
        handles = [dummy] + handles
        labels = [legend_title] + labels

        y_title = supt.get_position()[1]  # align legend vertically with the suptitle
        fig.legend(
            handles, labels,
            loc="center right",             # keep it right-aligned
            bbox_to_anchor=(1.0, y_title),  # at the same y as the title
            ncol=len(labels),
            markerscale=1.5,
            handlelength=0,
            handletextpad=1
        )
    mn = min(df[y].min(), df[m].min())
    mx = max(df[y].max(), df[m].max())
    ax.plot([mn, mx], [mn, mx], color="white", linestyle="--")
    ax.set_xlabel("True y")
    ax.set_ylabel("Predicted ŷ")
    ax.set_title("Predicted vs True")

    # Right plot: Residuals vs Predictions
    ax = axes[1]
    ax.scatter(df[m], df[f"res_{m}"], c=color_col, cmap=cmap, s=8, alpha=0.6)
    ax.axhline(0, color="white", linestyle="--")
    ax.set_xlabel("Predicted ŷ")
    ax.set_ylabel("Residual (y - ŷ)")
    ax.set_title("Residuals vs Predictions")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    
# Plot aggregated SHAP Values

def shap_bar_agg(df_plot):
    """
    Plot a horizontal stacked bar chart of SHAP feature importances
    aggregated across models, ordered by average importance.
    
    Parameters
    ----------
    df_plot : pandas.DataFrame
        DataFrame with features as columns and models as rows.
    """
    # Order features by their average importance across ALL models (descending)
    feature_order = df_plot.mean(axis=0).sort_values(ascending=False).index
    df_plot = df_plot[feature_order]

    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    fig, ax = plt.subplots(figsize=(12, 9))

    # Horizontal stacked bar plot (models on Y, features as colors)
    df_plot.plot(kind="barh", stacked=True, ax=ax, width=0.7)
    ax.set_title("SHAP Importances by Model (in Percentage)")

    # Remove x-axis ticks and labels (preserved)
    ax.set_xticks([])
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Legend at the bottom (preserved)
    leg = ax.legend(
        loc="upper center",bbox_to_anchor=(0.5, -0.03), 
        ncol=min(len(df_plot.columns), 4),frameon=False
    )

    # Annotate only when there's enough space (preserved)
    values = df_plot.values
    row_totals = values.sum(axis=1)
    left_edges = np.c_[np.zeros(len(df_plot)), np.cumsum(values, axis=1)]

    for i in range(values.shape[0]):  # loop over models (rows)
        total = row_totals[i]
        for j in range(values.shape[1]):  # loop over features (columns)
            w = values[i, j]
            if w / total >= 0.04:   # annotate only if wide enough
                x_center = left_edges[i, j] + w / 2
                y_center = i - 0.05
                ax.text(
                    x_center, y_center, f"{w:.2f}",
                    ha="center", va="center",fontweight="bold",fontsize=14
                )

    fig.subplots_adjust(left=0.0, right=1, top=0.8, bottom=0.05)
    plt.show()
    
# Custom RMSE plots -----------

# Add annotations on the RMSE plot
def add_text(ax, x = 0.1,y = 0.9, percent = False, ha = "right"):
    """Add a range annotation box (absolute or percent) to an axis."""
    ymin, ymax = ax.get_ylim()
    if percent:
        range_ = f"{100*ymin:.2f}% - {100*ymax:.2f}%"
    else:
        range_ = f"{ymin:.2f} - {ymax:.2f}"
    text = f"Range : {range_}"
    
    ax.text(
        x, y, text,
        transform=ax.transAxes, ha=ha, va="top", color="white",
        bbox=dict(boxstyle="square,pad=0.5", facecolor="none", edgecolor="white", linewidth=1.0)
    )

# Customize the placement of the annotation whether it's RMSE or R2
def add_text_score(ax_top, ax_bottom, score):
    """Place range annotations appropriately for RMSE or R² panels."""
    for ax in (ax_top, ax_bottom):
        if score == "rmse": 
            add_text(ax, x = 0.96,y = 0.92, percent = False, ha = "right")
        else:  # r2
            add_text(ax, x = 0.04,y = 0.92, percent = True, ha = "left")
    

