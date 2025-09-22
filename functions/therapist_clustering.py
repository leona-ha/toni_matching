import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import gower

import numpy as np
import pandas as pd
from pandas.api import types as pdt


def repair_therapist_age(
    df,
    therapist_id_col="therapist_id",
    age_col="t0_age_t",
    exp_col="t0_experience_t",
    *,
    min_age=23, max_age=80,
    prior_start_age=32,      # used only when imputing low ages
    min_gap=18, max_gap=40,  # enforce plausible (age - experience) on imputed values
    out_prior="t0_age_t_fixed",
    out_emp="t0_age_t_fixed_emp"
):
    """
    Returns ONLY rows where age was fixed (age < min_age), with both fixed age columns.
    No merge, therapist-level unique rows.
    """
    # therapist-level uniques (avoid dyad weighting)
    t = df[[therapist_id_col, age_col, exp_col]].drop_duplicates(therapist_id_col).copy()
    a = pd.to_numeric(t[age_col], errors="coerce")
    e = pd.to_numeric(t[exp_col], errors="coerce")  # experience assumed complete

    # mask: fix ONLY when recorded age exists and is implausibly low
    low_age = a.notna() & (a < min_age)
    if not low_age.any():
        return pd.DataFrame(columns=[therapist_id_col, out_prior, out_emp])

    # ---- prior-based fix (only low ages) ----
    ap = a.copy()
    ap.loc[low_age] = e.loc[low_age] + prior_start_age
    ap.loc[low_age] = np.maximum(ap.loc[low_age], e.loc[low_age] + min_gap)
    ap.loc[low_age] = np.minimum(ap.loc[low_age], e.loc[low_age] + max_gap)
    ap.loc[low_age] = ap.loc[low_age].clip(min_age, max_age)

    # ---- empirical fix (mean age per integer exp year), only low ages ----
    exp_year = e.round().astype(int)
    plausible = a.notna() & a.between(min_age, max_age)
    age_by_exp = a[plausible].groupby(exp_year[plausible]).median()
    pred = exp_year.map(age_by_exp.to_dict()).astype(float)           # may be NaN for unseen exp_years
    pred = pred.fillna(e + prior_start_age)                           # fallback if no mapping
    pred = pred.clip(min_age, max_age)
    pred = np.maximum(pred, e + min_gap)
    pred = np.minimum(pred, e + max_gap)
    pred = pred.clip(min_age, max_age)

    pe = a.copy()
    pe.loc[low_age] = pred.loc[low_age]

    # return ONLY fixed rows
    fixed_rows = t.loc[low_age, [therapist_id_col]].copy()
    fixed_rows[out_prior] = ap.loc[low_age].to_numpy(dtype=float)
    fixed_rows[out_emp]   = pe.loc[low_age].to_numpy(dtype=float)
    
    return fixed_rows.reset_index(drop=True)



import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from pandas.api import types as pdt
import gower

def cluster_therapists_gower_kmedoids(
    df_all: pd.DataFrame,
    therapist_numeric: list,
    therapist_categorical: list,
    therapist_id_col: str = "therapist_id",
    patient_id_col: str = "patient_id",
    outcome_t0_col: str = "t0_distress_sum",
    date_col: str ="u_registrationDate_p",
    use_train_only: bool = True,
    K_grid = (3, 4, 8, 10, 12),
    min_dyads_per_cluster: int = 20,
    random_state: int = 42,
    feature_group_weights: dict | None = None,
    *,
    method: str = "famd_kmeans",
    famd_components: int = 8,
    kmeans_n_init: int = 20,
    # NEW:
    outcome_delta_col:  str | None = None,   # e.g., "delta_distress_T1" (higher=better)
    shrink_m: float = 50.0                   # EB shrinkage strength (pseudo-n)
):
    """Cluster therapists.

    method="gower_pam": Gower distance + PAM (medoids).
    method="famd_kmeans": FAMD embedding (mixed data) + KMeans.

    Returns common structure; for famd_kmeans, distance_matrix is Euclidean in the
    embedding and gower_feature_weights=None.
    """
    # local imports needed by this function (keeps it self-contained)
    from pandas.api import types as pdt
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, pairwise_distances

    # -----------------------------
    # 1) Therapist table (T)
    # -----------------------------
    df = df_all.copy()
    if use_train_only and (date_col is not None) and (date_col in df.columns):
        df = df.sort_values(date_col)
        idx_keep = []
        for _, g in df.groupby(therapist_id_col, sort=False):
            n = len(g)
            if n >= 3:
                idx_keep.extend(g.index[:-1])
            elif n == 2:
                idx_keep.extend(g.index[[0]])
            else:
                idx_keep.extend(g.index)
        df_caseload = df.loc[idx_keep].copy()
    else:
        df_caseload = df

    caseload = (
        df_caseload.groupby(therapist_id_col, as_index=False)
        .agg(n_patients=(patient_id_col, "nunique"),
             mean_t0_distress=(outcome_t0_col, "mean"))
    )

    have_cols = [c for c in therapist_numeric + therapist_categorical if c in df_all.columns]
    T = (
        df_all[[therapist_id_col] + have_cols]
        .sort_values(therapist_id_col)
        .drop_duplicates(subset=[therapist_id_col])
        .merge(caseload, on=therapist_id_col, how="left")
    )
    T["n_patients"] = T["n_patients"].fillna(0).astype(int)

    # --- Optional: therapist-level average improvement (train window to avoid leakage) ---
    if (outcome_delta_col is not None) and (outcome_delta_col in df_caseload.columns):
        gb = df_caseload.groupby(therapist_id_col, as_index=False)
        try:
            delta_g = gb.agg(mean_delta_t1=(outcome_delta_col, "mean"),
                             n_delta=(outcome_delta_col, "size"))
        except TypeError:
            tmp = gb[outcome_delta_col].agg(["mean", "size"]).reset_index()
            tmp = tmp.rename(columns={"mean": "mean_delta_t1", "size": "n_delta"})
            delta_g = tmp

        mu0 = float(delta_g["mean_delta_t1"].mean())  # global mean for shrinkage
        delta_g["mean_delta_t1_shrunk"] = (
            (delta_g["mean_delta_t1"] * delta_g["n_delta"] + mu0 * shrink_m)
            / (delta_g["n_delta"] + shrink_m)
        )
        T = T.merge(delta_g, on=therapist_id_col, how="left")
    else:
        # keep columns around (for cards), but we'll *not* use them as numeric features
        T = T.assign(mean_delta_t1=np.nan, n_delta=0, mean_delta_t1_shrunk=np.nan)

    # unique dyads per therapist
    base_df = df_caseload if (use_train_only and (date_col is not None) and (date_col in df_all.columns)) else df_all
    dyads_unique = base_df[[therapist_id_col, patient_id_col]].drop_duplicates()
    dyads_per_tid = dyads_unique.groupby(therapist_id_col).size().rename("n_dyads_tid").reset_index()
    T = T.merge(dyads_per_tid, on=therapist_id_col, how="left")
    T["n_dyads_tid"] = T["n_dyads_tid"].fillna(0).astype(int)

    # -----------------------------------------
    # 2) Preprocess columns (robust to missing delta)
    # -----------------------------------------
    # base numeric columns (always considered)
    num_cols = [c for c in therapist_numeric if c in T.columns] \
        + [c for c in ["mean_t0_distress", "n_patients"] if c in T.columns]

    # include delta-derived columns *only if* an outcome delta was actually provided
    if outcome_delta_col is not None:
        num_cols += [c for c in ["mean_delta_t1_shrunk", "mean_delta_t1"] if c in T.columns]

    cat_cols_all = [c for c in therapist_categorical if c in T.columns]

    # treat binary 0/1 as numeric; others as categorical
    true_cat, bin_as_num = [], []
    for c in cat_cols_all:
        s = T[c]
        if pdt.is_numeric_dtype(s) and set(pd.unique(s.dropna())) <= {0, 1}:
            bin_as_num.append(c)
        else:
            true_cat.append(c)
    num_cols = list(dict.fromkeys(num_cols + bin_as_num))
    cat_cols = true_cat

    X = T.copy()

    # IMPORTANT: drop numeric columns that are all-NaN before imputation to avoid sklearn dropping them silently
    num_cols = [c for c in num_cols if (c in X.columns and not X[c].isna().all())]

    # If absolutely no usable features remain, fail fast with a helpful error
    if (len(num_cols) + len(cat_cols)) == 0:
        raise ValueError("No usable features left for clustering after preprocessing (all numeric were NaN and no categorical provided).")

    if num_cols:
        imp = SimpleImputer(strategy="median")
        X[num_cols] = imp.fit_transform(X[num_cols])
        X[num_cols] = StandardScaler().fit_transform(X[num_cols])

    for c in cat_cols:
        X[c] = X[c].astype("category")

    # --- Local fallback if helper isn't in scope ---
    try:
        _merge_small_clusters  # noqa: F821
        merge_small_clusters_fn = _merge_small_clusters
    except NameError:
        def merge_small_clusters_fn(labels: np.ndarray,
                                    D: np.ndarray,
                                    dyads_by_row: np.ndarray,
                                    min_dyads_per_cluster: int) -> np.ndarray:
            z = labels.copy()

            def _medoid_index(idx_arr: np.ndarray) -> int:
                if len(idx_arr) == 1:
                    return int(idx_arr[0])
                subD = D[np.ix_(idx_arr, idx_arr)]
                return int(idx_arr[np.argmin(subD.sum(axis=1))])

            while True:
                clusters = np.unique(z)
                if len(clusters) == 1:
                    break
                sums = {c: int(dyads_by_row[z == c].sum()) for c in clusters}
                failing = [c for c, s in sums.items() if s < min_dyads_per_cluster]
                if not failing:
                    break

                medoid_idx = {c: _medoid_index(np.where(z == c)[0]) for c in clusters}
                c_small = min(failing, key=lambda c: sums[c])
                i_small = medoid_idx[c_small]
                candidates = [c for c in clusters if c != c_small]
                target = min(candidates, key=lambda c: D[i_small, medoid_idx[c]])
                z[z == c_small] = target

            return z

    # ---------- METHOD A: Gower + PAM ----------
    if method == "gower_pam":
        try:
            import gower
        except Exception as e:
            raise ImportError("method='gower_pam' requires the 'gower' package. Install via `pip install gower`.") from e

        Xp = X[num_cols + cat_cols].copy()
        if Xp.shape[1] == 0:
            raise ValueError("No features to compute Gower distance on (Xp has 0 columns).")

        w = np.ones(Xp.shape[1], dtype=float)
        if feature_group_weights:
            for pref, total_w in feature_group_weights.items():
                idxs = [i for i, col in enumerate(Xp.columns) if str(col).startswith(pref)]
                if idxs:
                    per = total_w / len(idxs)
                    for i in idxs:
                        w[i] = per

        cat_mask = np.array([False]*len(num_cols) + [True]*len(cat_cols), dtype=bool)
        D = gower.gower_matrix(Xp, weight=w, cat_features=cat_mask)

        best = None; best_sil = -1.0
        tid_series = T[therapist_id_col].reset_index(drop=True)
        dyads_by_tid = dict(zip(tid_series.tolist(), T["n_dyads_tid"].tolist()))

        for K in K_grid:
            labels, medoids = _pam_kmedoids(D, K, n_init=5, max_iter=100, random_state=random_state)
            ok = True
            if (min_dyads_per_cluster is not None) and (min_dyads_per_cluster > 0):
                for k in range(K):
                    members = np.where(labels == k)[0]
                    dyads_k = sum(dyads_by_tid[tid_series.iloc[i]] for i in members)
                    if dyads_k < min_dyads_per_cluster:
                        ok = False
                        break
            if not ok:
                continue

            sil = silhouette_score(D, labels, metric="precomputed")
            if sil > best_sil:
                best_sil, best = sil, (labels, medoids)

        if best is None:
            for K in K_grid:
                labels, medoids = _pam_kmedoids(D, K, n_init=10, max_iter=100, random_state=random_state)
                sil = silhouette_score(D, labels, metric="precomputed")
                if sil > best_sil:
                    best_sil, best = sil, (labels, medoids)

        labels, medoids = best
        distance_matrix = D  # needed for merging step

        # Enforce min dyads per cluster by merging (if requested)
        if (min_dyads_per_cluster is not None) and (min_dyads_per_cluster > 0):
            dyads_by_row = T["n_dyads_tid"].to_numpy()
            labels = merge_small_clusters_fn(labels, distance_matrix, dyads_by_row, min_dyads_per_cluster)

        # Recompute medoids under the FINAL labeling
        final_medoids = []
        for c in sorted(np.unique(labels)):
            idx = np.where(labels == c)[0]
            if len(idx) == 1:
                final_medoids.append(int(idx[0]))
            else:
                subD = distance_matrix[np.ix_(idx, idx)]
                final_medoids.append(int(idx[np.argmin(subD.sum(axis=1))]))

        T["therapist_group"] = labels
        km_or_medoids = _SimpleKMedoids(np.array(final_medoids, dtype=int))
        gower_weights_out = w
        embedding_famd = None

    # ---------- METHOD B: FAMD + KMeans ----------
    elif method == "famd_kmeans":
        try:
            import prince
        except Exception as e:
            raise ImportError("method='famd_kmeans' requires the 'prince' package. Install via `pip install prince`.") from e

        Xf = X[num_cols + cat_cols].copy()
        if Xf.shape[1] == 0:
            raise ValueError("No features to fit FAMD on (Xf has 0 columns).")

        famd = prince.FAMD(n_components=famd_components, random_state=random_state)
        Z_df = famd.fit_transform(Xf)        # DataFrame
        Z = Z_df.to_numpy(copy=True)

        best = None; best_sil = -1.0
        dyads_by_row = T["n_dyads_tid"].to_numpy()

        for K in K_grid:
            km = KMeans(n_clusters=K, n_init=kmeans_n_init, random_state=random_state).fit(Z)
            labels = km.labels_

            ok = True
            if (min_dyads_per_cluster is not None) and (min_dyads_per_cluster > 0):
                for k in np.unique(labels):
                    members = np.where(labels == k)[0]
                    dyads_k = int(dyads_by_row[members].sum())
                    if dyads_k < min_dyads_per_cluster:
                        ok = False
                        break
            if not ok:
                continue

            sil = silhouette_score(Z, labels)
            if sil > best_sil:
                best_sil, best = sil, (labels, km.cluster_centers_)

        # Fallback if every K violated the constraint: pick best silhouette anyway
        if best is None:
            best_sil = -1.0
            for K in K_grid:
                km = KMeans(n_clusters=K, n_init=kmeans_n_init, random_state=random_state).fit(Z)
                sil = silhouette_score(Z, km.labels_)
                if sil > best_sil:
                    best_sil, best = sil, (km.labels_, km.cluster_centers_)

        labels, centers = best

        distance_matrix = pairwise_distances(Z, metric="euclidean")

        # Enforce min dyads per cluster by merging (if requested)
        if (min_dyads_per_cluster is not None) and (min_dyads_per_cluster > 0):
            dyads_by_row = T["n_dyads_tid"].to_numpy()
            labels = merge_small_clusters_fn(labels, distance_matrix, dyads_by_row, min_dyads_per_cluster)

        # Recompute medoid-like exemplars under the FINAL labeling (nearest to centroid)
        final_medoids = []
        for c in sorted(np.unique(labels)):
            idx = np.where(labels == c)[0]
            cvec = Z[idx].mean(axis=0, keepdims=True)
            d = np.linalg.norm(Z[idx] - cvec, axis=1)
            final_medoids.append(int(idx[np.argmin(d)]))

        T["therapist_group"] = labels
        km_or_medoids = _SimpleKMedoids(np.array(final_medoids, dtype=int))
        gower_weights_out = None
        embedding_famd = Z

    else:
        raise ValueError("Unknown method. Use 'gower_pam' or 'famd_kmeans'.")

    # ------------------------------
    # 4) Attach back + cluster cards
    # ------------------------------
    df_with_groups = df_all.merge(T[[therapist_id_col, "therapist_group"]], on=therapist_id_col, how="left")

    cards_rows = []
    overall_mean = T[[c for c in num_cols if c in T.columns]].mean(numeric_only=True)
    overall_sd   = T[[c for c in num_cols if c in T.columns]].std(ddof=0, numeric_only=True).replace(0, np.nan)

    medoid_ids = {}
    try:
        for k, idx in enumerate(km_or_medoids.medoid_indices_):
            medoid_ids[k] = T.iloc[idx][therapist_id_col]
    except Exception:
        pass

    for k in sorted(T["therapist_group"].unique()):
        g = T[T["therapist_group"] == k]
        size_k = len(g)
        dyads_k = int(g["n_dyads_tid"].sum())
        medoid_id = medoid_ids.get(k, g.iloc[0][therapist_id_col])

        # top numeric standardized deltas
        num_deltas = {}
        for c in num_cols:
            if c not in g.columns:
                continue
            dlt = (g[c].mean() - overall_mean.get(c, 0.0)) / (overall_sd.get(c, np.nan))
            if np.isfinite(dlt):
                num_deltas[c] = float(dlt)
        top_num = sorted(num_deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        # top categorical enrichment (level with biggest prevalence shift)
        cat_list = []
        for c in cat_cols:
            pk = g[c].astype(str).value_counts(normalize=True, dropna=False)
            p0 = T[c].astype(str).value_counts(normalize=True, dropna=False)
            best_lvl, best_delta = None, 0.0
            for lvl, p in pk.items():
                d = p - p0.get(lvl, 0.0)
                if abs(d) > abs(best_delta):
                    best_lvl, best_delta = lvl, float(d)
            cat_list.append((c, best_lvl, best_delta))
        top_cat = sorted(cat_list, key=lambda x: abs(x[2]), reverse=True)[:5]

        cards_rows.append({
            "cluster": int(k),
            "n_therapists": int(size_k),
            "n_dyads": dyads_k,
            "medoid_example": medoid_id,
            "top_numeric_deltas": top_num,
            "top_categorical_deltas": top_cat,
        })

    cards = pd.DataFrame(cards_rows).sort_values("cluster").reset_index(drop=True)

    return {
        "df_with_groups": df_with_groups,
        "therapists_table": T,
        "cluster_cards": cards,
        "kmedoids_model": km_or_medoids,      # shim with .medoid_indices_
        "distance_matrix": distance_matrix,   # Gower (method A) or Euclidean in FAMD space (method B)
        "processed_features": X[num_cols + cat_cols].copy(),
        "gower_feature_weights": gower_weights_out,
        "embedding_famd": embedding_famd,     # np.array or None
        "method": method,
    }
