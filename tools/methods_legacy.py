import os
import pandas as pd
import numpy as np
import uuid
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tools.registry import registry
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# Standalone tool implementations (plugins)
# ==========================================

@registry.register(requires_confirmation=True)
def clean_data(context):
    """
    Robust data cleaning tool.

    Parameters:
    - action_type:
        'standardize_missing' : Normalize NaN/inf/empty strings/common missing tokens only; no imputation or drops
        'impute'              : Impute missing values
        'drop'                : Drop rows or columns with missing values
        'plot_safe'           : Final plot-safe cleanup: drop rows with NaN/inf in selected numeric columns

    - columns:
        'all' or a list of column names

    - strategy:
        If action_type='impute':
            'mean', 'median', 'mode', 'interpolate', 'constant'
        If action_type='drop':
            'rows', 'cols'
        If action_type='standardize_missing' or 'plot_safe':
            May be omitted

    Optional parameters:
    - missing_col_threshold:
        Used when action_type='drop', strategy='cols'.
        Default 0.4: drop columns with missing rate > 40%.

    - constant_value:
        Used when strategy='constant'.

    - numeric_parse_threshold:
        Default 0.85.
        For object/string columns, coerce to numeric only if >=85% of non-missing values parse as numeric.

    - save:
        Default True: write back parquet.

    Returns:
    - dict:
        {
            "status": "ok" | "warning" | "blocked" | "failed",
            "message": str,
            "details": {...},
            "audit": {...}
        }
    """
    import numpy as np
    import pandas as pd

    MISSING_TOKENS = {
        "",
        " ",
        "na",
        "n/a",
        "nan",
        "null",
        "none",
        "missing",
        "unknown",
        "unk",
        "?",
        "-",
        "--",
        ".",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }

    def _select_columns(df, cols):
        if cols is None or cols == "all":
            return df.columns.tolist()

        if isinstance(cols, str):
            cols = [cols]

        if not isinstance(cols, list):
            raise ValueError("columns must be 'all', a column name, or a list of column names.")

        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in dataset: {missing_cols}")

        return cols

    def _normalize_missing_tokens(df):
        df = df.copy()

        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                def normalize_value(x):
                    if isinstance(x, str):
                        lx = x.strip().lower()
                        if lx in MISSING_TOKENS:
                            return np.nan
                        return x.strip()
                    return x

                df[col] = df[col].map(normalize_value)

        return df

    def _replace_infinite_values(df):
        df = df.copy()
        return df.replace([np.inf, -np.inf], np.nan)

    def _try_convert_numeric_object_columns(df, cols, threshold=0.85):
        """
        Coerce object/string columns to numeric only when most values look numeric.
        Avoid destroying categorical variables such as sex/major/group.
        """
        df = df.copy()
        conversions = []

        for col in cols:
            if col not in df.columns:
                continue

            s = df[col]

            if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
                continue

            non_missing = s.dropna()

            if len(non_missing) == 0:
                continue

            coerced = pd.to_numeric(non_missing, errors="coerce")
            numeric_rate = float(coerced.notna().mean())

            if numeric_rate >= threshold:
                before_dtype = str(df[col].dtype)
                df[col] = pd.to_numeric(df[col], errors="coerce")
                after_dtype = str(df[col].dtype)

                conversions.append({
                    "column": col,
                    "from_dtype": before_dtype,
                    "to_dtype": after_dtype,
                    "numeric_parse_rate": numeric_rate,
                })

        return df, conversions

    def _infer_column_kind(s):
        if pd.api.types.is_bool_dtype(s):
            return "categorical"

        if pd.api.types.is_numeric_dtype(s):
            return "numeric"

        if pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"

        if pd.api.types.is_categorical_dtype(s):
            return "categorical"

        return "categorical"

    def _safe_mode(s):
        mode_values = s.dropna().mode()
        if len(mode_values) == 0:
            return np.nan
        return mode_values.iloc[0]

    def _impute_numeric(s, strategy, constant_value=None):
        if s.dropna().empty:
            fill_value = 0 if constant_value is None else constant_value
            return s.fillna(fill_value), fill_value

        if strategy == "mean":
            fill_value = s.mean()
            return s.fillna(fill_value), fill_value

        if strategy == "median":
            fill_value = s.median()
            return s.fillna(fill_value), fill_value

        if strategy == "mode":
            fill_value = _safe_mode(s)
            if pd.isna(fill_value):
                fill_value = 0
            return s.fillna(fill_value), fill_value

        if strategy == "constant":
            fill_value = 0 if constant_value is None else constant_value
            return s.fillna(fill_value), fill_value

        if strategy == "interpolate":
            out = s.interpolate(method="linear", limit_direction="both")

            if out.isna().any():
                fallback = s.median() if not s.dropna().empty else 0
                out = out.fillna(fallback)

            return out, "linear_interpolation_with_median_or_zero_fallback"

        raise ValueError(f"Unsupported numeric imputation strategy: {strategy}")

    def _impute_categorical(s, strategy, constant_value=None):
        """
        For categorical columns, mean/median/interpolate are meaningless.
        Fall back to mode.
        """
        if strategy in {"mean", "median", "interpolate"}:
            strategy = "mode"

        if strategy == "mode":
            fill_value = _safe_mode(s)
            if pd.isna(fill_value):
                fill_value = "Unknown"
            return s.fillna(fill_value), fill_value

        if strategy == "constant":
            fill_value = "Unknown" if constant_value is None else constant_value
            return s.fillna(fill_value), fill_value

        raise ValueError(f"Unsupported categorical imputation strategy: {strategy}")

    def _impute_datetime(s, strategy, constant_value=None):
        if s.dropna().empty:
            if constant_value is None:
                return s, None

            fill_value = pd.to_datetime(constant_value, errors="coerce")
            return s.fillna(fill_value), fill_value

        if strategy in {"mean", "median"}:
            numeric_time = s.dropna().astype("int64")
            median_time = int(np.median(numeric_time))
            fill_value = pd.to_datetime(median_time)
            return s.fillna(fill_value), fill_value

        if strategy == "mode":
            fill_value = _safe_mode(s)
            return s.fillna(fill_value), fill_value

        if strategy == "constant":
            fill_value = pd.to_datetime(constant_value, errors="coerce")
            return s.fillna(fill_value), fill_value

        if strategy == "interpolate":
            numeric = s.astype("int64")
            numeric = pd.Series(numeric, index=s.index)
            numeric = numeric.replace(np.iinfo("int64").min, np.nan)
            numeric = numeric.interpolate(limit_direction="both")

            if numeric.isna().any():
                fallback = int(np.median(s.dropna().astype("int64")))
                numeric = numeric.fillna(fallback)

            out = pd.to_datetime(numeric)
            return out, "datetime_interpolation_with_fallback"

        raise ValueError(f"Unsupported datetime imputation strategy: {strategy}")

    def _drop_high_missing_columns(df, cols, threshold):
        df = df.copy()
        existing_cols = [c for c in cols if c in df.columns]

        missing_rates = df[existing_cols].isna().mean()
        to_drop = missing_rates[missing_rates > threshold].index.tolist()

        df = df.drop(columns=to_drop)

        return df, to_drop

    def _remove_nonfinite_numeric_rows(df, cols):
        """
        Plot-safe cleanup before plotting.
        Only for selected numeric columns.
        Drop rows that still contain NaN/inf in those columns.
        """
        df = df.copy()

        existing_cols = [c for c in cols if c in df.columns]
        numeric_cols = [
            c for c in existing_cols
            if pd.api.types.is_numeric_dtype(df[c])
        ]

        if not numeric_cols:
            return df, 0

        before = len(df)

        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        finite_mask = np.isfinite(df[numeric_cols].to_numpy(dtype=float)).all(axis=1)
        df = df.loc[finite_mask].copy()

        after = len(df)

        return df, before - after

    def _count_total_inf(df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return 0

        return int(np.isinf(df[numeric_cols].to_numpy(dtype=float)).sum())

    try:
        df = context.load_df()

        if df is None or not isinstance(df, pd.DataFrame):
            return {
                "status": "failed",
                "error_code": "INVALID_DATAFRAME",
                "message": "context.load_df() did not return a valid pandas DataFrame.",
                "recoverable": False,
            }

        action = context.get_arg("action_type", "standardize_missing")
        cols_arg = context.get_arg("columns", "all")
        strategy = context.get_arg("strategy", "median")

        missing_col_threshold = float(context.get_arg("missing_col_threshold", 0.4))
        constant_value = context.get_arg("constant_value", None)
        numeric_parse_threshold = float(context.get_arg("numeric_parse_threshold", 0.85))
        save = bool(context.get_arg("save", True))

        original_shape = df.shape
        original_columns = df.columns.tolist()
        original_missing = int(df.isna().sum().sum())
        original_inf = _count_total_inf(df)

        cols = _select_columns(df, cols_arg)

        audit = {
            "action_type": action,
            "strategy": strategy,
            "requested_columns": cols_arg,
            "resolved_columns": cols,
            "original_shape": original_shape,
            "original_missing": original_missing,
            "original_inf": original_inf,
            "standardized_missing_tokens": False,
            "converted_inf_to_nan": False,
            "numeric_conversions": [],
            "imputations": [],
            "dropped_rows": 0,
            "dropped_columns": [],
            "plot_safe_rows_removed": 0,
            "warnings": [],
        }

        # 1. Normalize common missing tokens globally.
        df = _normalize_missing_tokens(df)
        audit["standardized_missing_tokens"] = True

        # 2. Replace inf with NaN globally.
        df = _replace_infinite_values(df)
        audit["converted_inf_to_nan"] = True

        # 3. Conservatively coerce numeric-like object columns.
        df, conversions = _try_convert_numeric_object_columns(
            df=df,
            cols=cols,
            threshold=numeric_parse_threshold,
        )
        audit["numeric_conversions"] = conversions

        # Column list may still be valid; refreshed after column drops.
        cols = [c for c in cols if c in df.columns]

        # 4. Apply the requested action.
        if action == "standardize_missing":
            # No extra operation.
            pass

        elif action == "impute":
            for col in cols:
                if col not in df.columns:
                    continue

                before_missing = int(df[col].isna().sum())

                if before_missing == 0:
                    continue

                kind = _infer_column_kind(df[col])

                if kind == "numeric":
                    df[col], fill_value = _impute_numeric(
                        df[col],
                        strategy=strategy,
                        constant_value=constant_value,
                    )

                elif kind == "datetime":
                    df[col], fill_value = _impute_datetime(
                        df[col],
                        strategy=strategy,
                        constant_value=constant_value,
                    )

                else:
                    df[col], fill_value = _impute_categorical(
                        df[col],
                        strategy=strategy,
                        constant_value=constant_value,
                    )

                after_missing = int(df[col].isna().sum())

                audit["imputations"].append({
                    "column": col,
                    "kind": kind,
                    "strategy_requested": strategy,
                    "fill_value": None if pd.isna(fill_value) else str(fill_value),
                    "missing_before": before_missing,
                    "missing_after": after_missing,
                })

        elif action == "drop":
            if strategy == "rows":
                before = len(df)
                df = df.dropna(subset=cols)
                after = len(df)
                audit["dropped_rows"] = int(before - after)

            elif strategy == "cols":
                df, dropped_cols = _drop_high_missing_columns(
                    df=df,
                    cols=cols,
                    threshold=missing_col_threshold,
                )
                audit["dropped_columns"] = dropped_cols

            else:
                return {
                    "status": "blocked",
                    "error_code": "UNSUPPORTED_DROP_STRATEGY",
                    "message": f"Unsupported drop strategy: {strategy}",
                    "recoverable": True,
                    "suggested_next_actions": [
                        "Use strategy='rows' or strategy='cols'."
                    ],
                    "details": {
                        "action_type": action,
                        "strategy": strategy,
                    },
                }

        elif action == "plot_safe":
            df, removed = _remove_nonfinite_numeric_rows(df, cols)
            audit["plot_safe_rows_removed"] = int(removed)

        else:
            return {
                "status": "blocked",
                "error_code": "UNSUPPORTED_CLEANING_ACTION",
                "message": f"Unsupported cleaning action: {action}",
                "recoverable": True,
                "suggested_next_actions": [
                    "Use action_type='standardize_missing', 'impute', 'drop', or 'plot_safe'."
                ],
                "details": {
                    "action_type": action,
                },
            }

        # 5. Final inf guard.
        df = _replace_infinite_values(df)

        # 6. Post-clean metrics.
        final_shape = df.shape
        final_columns = df.columns.tolist()
        total_missing_after = int(df.isna().sum().sum())
        total_inf_after = _count_total_inf(df)

        remaining_selected_cols = [c for c in cols if c in df.columns]

        selected_missing_after = {}
        selected_inf_after = {}

        for col in remaining_selected_cols:
            selected_missing_after[col] = int(df[col].isna().sum())

            if pd.api.types.is_numeric_dtype(df[col]):
                arr = df[col].to_numpy(dtype=float)
                selected_inf_after[col] = int(np.isinf(arr).sum())
            else:
                selected_inf_after[col] = 0

        # 7. Determine status.
        status = "ok"
        message = "✅ Cleaning completed."

        if total_inf_after > 0:
            status = "warning"
            audit["warnings"].append(
                f"After cleaning, numeric columns still have {total_inf_after} inf/-inf values."
            )

        if action in {"impute", "plot_safe"}:
            unresolved_selected_missing = {
                c: n for c, n in selected_missing_after.items()
                if n > 0
            }

            if unresolved_selected_missing:
                status = "warning"
                audit["warnings"].append(
                    f"After cleaning, selected columns still have missing values: {unresolved_selected_missing}"
                )

        if len(df) == 0:
            status = "warning"
            audit["warnings"].append(
                "Dataset has 0 rows after cleaning; downstream analysis/plots may fail."
            )

        # 8. Write parquet.
        if save:
            df.to_parquet(context.file_path, index=False)

        return {
            "status": status,
            "message": message,
            "recoverable": False,
            "details": {
                "action_type": action,
                "strategy": strategy,
                "original_shape": original_shape,
                "final_shape": final_shape,
                "original_columns": original_columns,
                "final_columns": final_columns,
                "total_missing_before": original_missing,
                "total_missing_after": total_missing_after,
                "total_inf_before": original_inf,
                "total_inf_after": total_inf_after,
                "selected_missing_after": selected_missing_after,
                "selected_inf_after": selected_inf_after,
            },
            "audit": audit,
        }

    except Exception as e:
        return {
            "status": "failed",
            "error_code": "CLEAN_DATA_EXCEPTION",
            "message": f"❌ Cleaning failed: {str(e)}",
            "recoverable": True,
            "details": {
                "exception_type": type(e).__name__,
            },
            "suggested_next_actions": [
                "Run inspect_dataset first.",
                "Check whether selected columns exist.",
                "Try action_type='standardize_missing' before imputation or plotting.",
            ],
        }

@registry.register()
def get_summary_stats(context):
    """
    Safe operation: descriptive statistics for the dataset.
    """
    df = context.load_df()
    return df.describe().to_dict()

def run_inspect_dataset(action, workspace_dir):
    return {
        "shape": {"rows": workspace_dir.shape[0], "columns": workspace_dir.shape[1]},
        "dtypes": workspace_dir.dtypes.astype(str).to_dict(),
        "total_missing": int(workspace_dir.isnull().sum().sum())
    }


def run_summarize_columns(action, workspace_dir):
    cols = action.arguments.get("columns", [])
    missing_cols = [c for c in cols if c not in workspace_dir.columns]
    if missing_cols:
        raise ValueError(f"Columns not in dataset: {missing_cols}")
    return {"summary": workspace_dir[cols].describe().to_dict()}


def run_linear_regression(action, workspace_dir):
    outcome = action.arguments.get("outcome")
    predictors = action.arguments.get("predictors", [])
    formula = f"{outcome} ~ {' + '.join(predictors)}"
    model = smf.ols(formula=formula, data=workspace_dir).fit()
    return {
        "formula": formula,
        "r_squared": round(model.rsquared, 4),
        "adj_r_squared": round(model.rsquared_adj, 4),
        "f_pvalue": round(model.f_pvalue, 6),
        "coefficients": model.params.round(4).to_dict(),
        "p_values": model.pvalues.round(6).to_dict()
    }


def run_t_test(action, workspace_dir):
    group_col = action.arguments.get("group_column")
    value_col = action.arguments.get("value_column")
    if group_col not in workspace_dir.columns or value_col not in workspace_dir.columns:
        raise ValueError(f"Column {group_col} or {value_col}  not found.")

    groups = workspace_dir[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(f"t-test requires exactly two groups; found:{groups}")

    group1_data = workspace_dir[workspace_dir[group_col] == groups[0]][value_col].dropna()
    group2_data = workspace_dir[workspace_dir[group_col] == groups[1]][value_col].dropna()
    t_stat, p_val = stats.ttest_ind(group1_data, group2_data)

    return {
        "group_1": str(groups[0]),
        "group_1_mean": round(group1_data.mean(), 4),
        "group_2": str(groups[1]),
        "group_2_mean": round(group2_data.mean(), 4),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_val, 6),
        "significant": p_val < 0.05
    }


def run_generate_scatterplot(action, workspace_dir):
    # Use uuid for output filename (could incorporate action_id)
    import uuid
    x_col = action.arguments.get("x_column")
    y_col = action.arguments.get("y_column")
    if x_col not in workspace_dir.columns or y_col not in workspace_dir.columns:
        raise ValueError(f"Column {x_col} or {y_col}  not found.")

    output_dir = "artifacts"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"scatter_{uuid.uuid4().hex[:8]}.png")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=workspace_dir, x=x_col, y=y_col)
    plt.title(f"Scatter plot of {y_col} vs {x_col}")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    return {
        "message": "Successfully generated scatter plot.",
        "artifact_path": plot_path,
        "x_axis": x_col,
        "y_axis": y_col
    }


@registry.register()
def get_correlation_matrix(context):
    """
    Pearson correlation matrix for all numeric columns.
    No arguments required.
    """
    df = context.load_df()
    # Auto-select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return "Error: fewer than 2 numeric columns; cannot compute correlation."

    corr_matrix = numeric_df.corr().round(3)
    return corr_matrix.to_dict()


@registry.register()
def run_independent_t_test(context):
    """
    Independent samples t-test: compare means of a continuous variable between two groups.
    Required arguments:
    - target_col: numeric outcome (e.g. 'salary')
    - group_col: grouping column (e.g. 'gender')
    - group1_val: label for group 1 (e.g. 'Male')
    - group2_val: label for group 2 (e.g. 'Female')
    """
    df = context.load_df()

    # Uses get_arg for parameter access.
    target_col = context.get_arg("target_col")
    group_col = context.get_arg("group_col")
    group1_val = context.get_arg("group1_val")
    group2_val = context.get_arg("group2_val")

    if not all([target_col, group_col, group1_val, group2_val]):
        return "Error: missing required arguments target_col, group_col, group1_val, group2_val."

    try:
        group1_data = df[df[group_col] == group1_val][target_col].dropna()
        group2_data = df[df[group_col] == group2_val][target_col].dropna()
    except KeyError as e:
        return f"Error: column not found {str(e)}."

    if len(group1_data) < 2 or len(group2_data) < 2:
        return "Error: one or more groups have fewer than 2 observations."

    t_stat, p_val = stats.ttest_ind(group1_data, group2_data, equal_var=False)

    return {
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_val, 4),
        "significant_at_0_05": bool(p_val < 0.05),  # Boolean for easier LLM consumption
        "group1_mean": round(group1_data.mean(), 4),
        "group2_mean": round(group2_data.mean(), 4)
    }


# Code execution is high-risk and requires human confirmation.
@registry.register(requires_confirmation=True)
def generate_chart(context):
    """
    Visualization: executes Python plotting code from the model.
    Arguments:
    - code: Python source string with full plotting logic.

    Strict rules for the code argument:
    1. Use `df` (pandas DataFrame for the current dataset).
    2. Use `save_path` (absolute path where the chart must be saved).
    3. Pre-imported: `plt`, `sns`, `pd`, `np`.
    4. End with `plt.savefig(save_path, bbox_inches='tight')`.
    5. Never call plt.show().
    """
    df = context.load_df()
    code_string = context.get_arg("code")

    if not code_string:
        return "Error: no plotting code provided."

    image_name = f"chart_{uuid.uuid4().hex[:8]}.png"
    save_path = os.path.join(context.workspace_dir, image_name)

    print(f"DEBUG: Saving chart to: {save_path}")

    # Sandbox env passed into exec()
    local_env = {
        "df": df,
        "save_path": save_path,
        "plt": plt,
        "sns": sns,
        "pd": pd,
        "np": np,
        "sm": sm
    }

    try:
        # Execute LLM-generated Python dynamically
        # In production, run inside an isolated container.

        #### Before exec:
        print("\n" + "!" * 30)
        print("🤖 Generated plotting code:")
        print(code_string)
        print("!" * 30 + "\n")

        exec(code_string, {}, local_env)

        # Close matplotlib figures
        plt.close('all')

        if os.path.exists(save_path):
            return f"Success! Chart saved as: {image_name}。Tell the user the chart is ready."
        else:
            return "Execution finished but image missing; ensure plt.savefig(save_path)."

    except Exception as e:
        plt.close('all')
        return f"Plotting code crashed: {str(e)}。Check logic (e.g. column names) and retry with fixes."

@registry.register()
def run_multiple_regression(context):
    """
    Multiple linear regression (OLS).
    Joint effect of predictors X on outcome Y.
    Parameters:
    - target_col: Outcome column Y
    - feature_cols: Predictor column names (list of strings)
    """
    df = context.load_df()
    target_col = context.get_arg("target_col")
    feature_cols = context.get_arg("feature_cols")

    if not target_col or not feature_cols or not isinstance(feature_cols, list):
        return "Error: invalid arguments; provide target_col (str) and feature_cols (list)."

    try:
        # Drop rows with NA in selected columns
        analysis_df = df[[target_col] + feature_cols].dropna()
        if len(analysis_df) < len(feature_cols) + 2:
            return "Error: too few complete rows to fit regression."

        Y = analysis_df[target_col]
        X = analysis_df[feature_cols]

        # One-hot encode categoricals
        X = pd.get_dummies(X, drop_first=True)
        # Add intercept
        X = sm.add_constant(X)

        # Fit OLS
        model = sm.OLS(Y, X.astype(float)).fit()

        # Return key statistics
        return {
            "R_squared": round(model.rsquared, 4),
            "Adj_R_squared": round(model.rsquared_adj, 4),
            "F_statistic_p_value": round(model.f_pvalue, 5),
            "Coefficients": model.params.round(4).to_dict(),
            "P_values": model.pvalues.round(4).to_dict()
        }
    except Exception as e:
        return f"Regression failed: {str(e)}. Check for non-convertible column types."


@registry.register()
def run_standard_bootstrap(context):
    """
    Standard bootstrap (Standard Independent Bootstrap) 95% CI for the mean of a numeric column.
    i.i.d. assumed; do not use if data are serially correlated.
    Parameters:
    - target_col: Numeric column to evaluate
    - iterations: Bootstrap iterations (default 1000)
    """
    df = context.load_df()
    target_col = context.get_arg("target_col")
    iterations = context.get_arg("iterations", 1000)

    if not target_col or target_col not in df.columns:
        return "Error: target column missing or not in dataset."

    data = df[target_col].dropna().values
    n = len(data)

    if n < 30:
        return f"Warning: sample size ({n}) is small; bootstrap may be unstable."

    try:
        # Bootstrap resampling with replacement; random index matrix (iterations, n)
        bootstrap_indices = np.random.randint(0, n, size=(iterations, n))
        bootstrap_samples = data[bootstrap_indices]

        # Per-resample means
        bootstrap_means = np.mean(bootstrap_samples, axis=1)

        # Percentile CI
        lower_bound = np.percentile(bootstrap_means, 2.5)
        upper_bound = np.percentile(bootstrap_means, 97.5)

        return {
            "method": "Standard Independent Bootstrap",
            "iterations": iterations,
            "sample_size": n,
            "original_mean": round(float(np.mean(data)), 4),
            "95_CI_lower": round(float(lower_bound), 4),
            "95_CI_upper": round(float(upper_bound), 4)
        }
    except Exception as e:
        return f"Bootstrap failed: {str(e)}"


@registry.register()
def run_anova(context):
    """
    One-way ANOVA.
    Tests whether group means differ.
    Parameters:
    - target_col: Numeric outcome column
    - group_col: Categorical grouping column
    """
    df = context.load_df()
    target_col = context.get_arg("target_col")
    group_col = context.get_arg("group_col")

    try:
        # Model formula: Y ~ C(Group)
        model_formula = f"{target_col} ~ C({group_col})"
        model = ols(model_formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        return {
            "F_statistic": round(anova_table['F'][0], 4),
            "p_value": round(anova_table['PR(>F)'][0], 5),
            "significant_at_0_05": bool(anova_table['PR(>F)'][0] < 0.05),
            "df_group": int(anova_table['df'][0]),
            "df_resid": int(anova_table['df'][1])
        }
    except Exception as e:
        return f"ANOVA failed: {str(e)}"


@registry.register()
def run_chi_square(context):
    """
    Chi-square test of independence.
    Tests association between two categorical variables.
    Parameters:
    - col1: First categorical column
    - col2: Second categorical column
    """
    df = context.load_df()
    col1 = context.get_arg("col1")
    col2 = context.get_arg("col2")

    try:
        # Contingency table
        contingency_table = pd.crosstab(df[col1], df[col2])
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        return {
            "chi2_statistic": round(chi2, 4),
            "p_value": round(p, 5),
            "degrees_of_freedom": dof,
            "significant_at_0_05": bool(p < 0.05)
        }
    except Exception as e:
        return f"Chi-square failed: {str(e)}"


@registry.register()
def run_logistic_regression(context):
    """
    Logistic regression.
    Predicts binary outcome probabilities.
    Parameters:
    - target_col: Binary outcome 0/1
    - feature_cols: list of predictor columns
    """
    df = context.load_df()
    target_col = context.get_arg("target_col")
    feature_cols = context.get_arg("feature_cols")

    try:
        Y = df[target_col]
        X = sm.add_constant(df[feature_cols])
        model = sm.Logit(Y.astype(float), X.astype(float)).fit(disp=0)

        return {
            "Pseudo_R_squared": round(model.prsquared, 4),
            "AIC": round(model.aic, 2),
            "Coefficients": model.params.to_dict(),
            "P_values": model.pvalues.to_dict()
        }
    except Exception as e:
        return f"Logistic regression failed: {str(e)}"


# @registry.register()
# def smart_impute_data(context):
#     """
#     (Commented-out) Preferred missing-value handling vs row deletion.
#     Operates on in-memory DataFrame; independent of source file format.
#
#     Use when data has missing values and you want multivariate imputation.
#     """
#     df = context.load_df()
#     target_col = context.get_arg("target_col")
#
#     try:
#         # 1. Drop rows missing the target column
#         if target_col and target_col in df.columns:
#             initial_len = len(df)
#             df = df.dropna(subset=[target_col])
#             dropped_count = initial_len - len(df)
#
#         # 2. Numeric columns for imputation
#         num_df = df.select_dtypes(include=[np.number])
#         if num_df.isnull().sum().sum() == 0:
#             return "No numeric missing values left to impute."
#
#         # 3. Iterative imputer with RandomForestRegressor base estimator
#         imputer = IterativeImputer(
#             estimator=RandomForestRegressor(n_estimators=10, random_state=42),
#             max_iter=10,
#             random_state=42
#         )
#
#         imputed_values = imputer.fit_transform(num_df)
#
#         df.loc[:, num_df.columns] = imputed_values
#
#         context.save_df(df)
#
#         report = (
#             f"Imputation complete:\n- Target column [{target_col}] dropped missing rows: {dropped_count}\n"
#             f"- Other numeric columns imputed with iterative RF regression."
#         )
#         return report
#
#     except Exception as e:
#         return f"Smart imputation failed: {str(e)}"


@registry.register(requires_confirmation=False)  # Pure computation; no human gate
def regression_diagnostics(context):
    """
    Deep regression diagnostics including VIF and Breusch-Pagan heteroskedasticity test.
    Use before/after formal regression to assess reliability.

    Arguments:
    - target_col: Outcome column Y.
    - feature_cols: Predictor columns (numeric).
    """
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.stats.api as sms

    df = context.load_df()
    y_col = context.get_arg("target_col")
    x_cols = context.get_arg("feature_cols")

    if not y_col or not x_cols:
        return "Error: target_col and feature_cols are required."

    try:
        # Drop NA in selected columns
        analysis_df = df[[y_col] + x_cols].dropna()
        X = analysis_df[x_cols]
        X = sm.add_constant(X)  # Add constant
        y = analysis_df[y_col]

        # 1. Multicollinearity (VIF)
        vif_data = []
        for i in range(X.shape[1]):
            # VIF excluding intercept
            if X.columns[i] != 'const':
                vif = variance_inflation_factor(X.values, i)
                vif_data.append(f"  - {X.columns[i]}: VIF = {vif:.2f}")

        # 2. Breusch-Pagan heteroskedasticity test
        model = sm.OLS(y, X).fit()
        # bp_test returns (LM stat, LM p-value, F stat, F p-value)
        bp_test = sms.het_breuschpagan(model.resid, model.model.exog)
        bp_pvalue = bp_test[1]

        # Build diagnostic report
        report = "[Regression diagnostics report]\n"
        report += "1. Multicollinearity (VIF):\n"
        report += "\n".join(vif_data) + "\n"
        report += "   (Note: VIF > 10 suggests strong multicollinearity.)\n\n"

        report += "2. Breusch-Pagan heteroskedasticity:\n"
        report += f"   - P-value = {bp_pvalue:.4f}\n"
        if bp_pvalue < 0.05:
            report += "   (Warning: P < 0.05 suggests heteroskedasticity; OLS SE may be unreliable; consider robust SE.)\n"
        else:
            report += "   (OK: P >= 0.05, no significant heteroskedasticity.)\n"

        return report

    except Exception as e:
        return f"Diagnostics failed: {str(e)}"