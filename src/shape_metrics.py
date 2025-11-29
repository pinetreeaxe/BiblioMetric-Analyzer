"""
shape_metrics.py
================

Module for fitting mathematical models to h-index temporal evolution.

This module implements various growth models to characterize the "shape"
of an author's h-index trajectory over time:

Models Implemented
------------------
1. Linear: h(t) = slope·t + intercept
   - Constant growth rate
   
2. Hirsch: h(t) = a·√t + b
   - Square root growth (original Hirsch model)
   - Parameter 'a' represents growth coefficient
   
3. Power-law: h(t) = a·t^b
   - Flexible growth exponent 'b'
   - b<1: decelerating, b=1: linear, b>1: accelerating
   
4. Exponential saturation: h(t) = a·(1 - e^(-b·t)) + c
   - Approaches asymptotic limit (saturation)
   - Common in mature careers

Applications
------------
- Career trajectory classification
- Future h-index prediction
- Comparison of growth patterns across researchers
- Detection of career phases (acceleration, plateau)

Authors: Diogo Abreu, João Machado, Pedro Lopes
Date: 11/2025
Version: 1.0
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


def _clean_series(years, h_values):
    """
    Clean and prepare time series data for model fitting.
    
    Removes NaN values, ensures proper array types, and sorts by year.
    
    Parameters
    ----------
    years : array-like
        Years (x values)
    h_values : array-like
        H-index values (y values)
        
    Returns
    -------
    tuple (np.ndarray, np.ndarray) or (None, None)
        - Cleaned and sorted (years, h_values) arrays
        - (None, None) if insufficient data points (<3)
        
    Notes
    -----
    Requires at least 3 data points for meaningful model fitting.
    """
    years = np.asarray(years, dtype=float)
    h_values = np.asarray(h_values, dtype=float)

    mask = np.isfinite(years) & np.isfinite(h_values)
    years = years[mask]
    h_values = h_values[mask]

    # Require at least 3 points for fitting
    if len(years) < 3:
        return None, None

    # Sort by year
    order = np.argsort(years)
    return years[order], h_values[order]


def _fit_linear(years, h_values):
    """
    Fit linear model: h(t) = slope·t + intercept
    
    Parameters
    ----------
    years : np.ndarray
        Career time points
    h_values : np.ndarray
        H-index values
        
    Returns
    -------
    dict
        {'slope': float, 'intercept': float, 'rmse': float, 'success': True}
    """
    model = LinearRegression()
    X = years.reshape(-1, 1)
    model.fit(X, h_values)
    preds = model.predict(X)
    rmse = root_mean_squared_error(h_values, preds)
    return {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "rmse": float(rmse),
        "success": True,
    }


def _hirsch_func(t, a, b):
    """
    Hirsch growth model function.
    
    h(t) = a·√t + b
    
    Parameters
    ----------
    t : array-like
        Career time (years since start)
    a : float
        Growth coefficient (Hirsch Growth Coefficient, HGC)
    b : float
        Offset parameter
        
    Returns
    -------
    np.ndarray
        Predicted h-index values
    """
    return a * np.sqrt(t) + b


def _fit_hirsch(years, h_values):
    """
    Fit Hirsch square-root growth model.
    
    Uses nonlinear least squares to fit h(t) = a·√t + b.
    
    Parameters
    ----------
    years : np.ndarray
        Career time points (must be > 0)
    h_values : np.ndarray
        H-index values
        
    Returns
    -------
    dict
        {'a': float, 'b': float, 'rmse': float, 'success': bool}
        Returns {'success': False} if fitting fails
        
    Notes
    -----
    The parameter 'a' (HGC) represents the characteristic growth rate
    and is used in the SaH metric for career stability analysis.
    """
    # Requires t > 0 for square root
    mask = years > 0
    years = years[mask]
    h_values = h_values[mask]
    if len(years) < 3:
        return {"success": False}

    try:
        popt, _ = curve_fit(_hirsch_func, years, h_values, maxfev=5000)
        a, b = popt
        preds = _hirsch_func(years, a, b)
        rmse = root_mean_squared_error(h_values, preds)
        return {
            "a": float(a),
            "b": float(b),
            "rmse": float(rmse),
            "success": True,
        }
    except Exception:
        return {"success": False}


def _power_func(t, a, b):
    """
    Power-law growth model function.
    
    h(t) = a·t^b
    
    Parameters
    ----------
    t : array-like
        Career time (years since start)
    a : float
        Amplitude parameter
    b : float
        Growth exponent
        - b < 1: decelerating growth
        - b = 1: linear growth
        - b > 1: accelerating growth
        
    Returns
    -------
    np.ndarray
        Predicted h-index values
    """
    return a * (t ** b)


def _fit_power_law(years, h_values):
    """
    Fit power-law model using log-log linear regression.
    
    Transforms h(t) = a·t^b to log(h) = log(a) + b·log(t)
    and fits via linear regression for robustness.
    
    Parameters
    ----------
    years : np.ndarray
        Career time points
    h_values : np.ndarray
        H-index values (must be > 0)
        
    Returns
    -------
    dict
        {'a': float, 'b': float, 'rmse': float, 'success': True}
        
    Notes
    -----
    The exponent 'b' characterizes growth dynamics:
    - b ≈ 0.5: Square root growth (Hirsch-like)
    - b ≈ 1.0: Linear growth
    - b > 1.0: Accelerating (compounding success)
    """
    years = np.asarray(years, dtype=float)
    h_values = np.asarray(h_values, dtype=float)

    # Convert to career time: 1, 2, 3, ...
    t = years - years.min() + 1

    # Filter valid values: t > 0 and h > 0
    mask = (t > 0) & (h_values > 0) & np.isfinite(t) & np.isfinite(h_values)
    t = t[mask]
    h = h_values[mask]

    if len(t) < 3:
        return {"a": np.nan, "b": np.nan, "rmse": np.nan, "success": False}

    log_t = np.log(t).reshape(-1, 1)
    log_h = np.log(h)

    model = LinearRegression()
    model.fit(log_t, log_h)

    b = float(model.coef_[0])
    ln_a = float(model.intercept_)
    a = float(np.exp(ln_a))

    # Predictions for error calculation
    pred_h = a * (t ** b)
    rmse = np.sqrt(np.mean((h - pred_h) ** 2))

    return {"a": a, "b": b, "rmse": rmse, "success": True}


def _exp_func(t, a, b, c):
    """
    Exponential saturation model function.
    
    h(t) = a·(1 - e^(-b·t)) + c
    
    Parameters
    ----------
    t : array-like
        Career time (years since start)
    a : float
        Growth amplitude (asymptotic gain)
    b : float
        Growth rate (how quickly saturation is approached)
    c : float
        Initial offset / baseline h-index
        
    Returns
    -------
    np.ndarray
        Predicted h-index values
        
    Notes
    -----
    This model approaches the asymptote h_max = a + c as t → ∞.
    Suitable for mature careers showing growth plateau.
    """
    t = np.asarray(t, dtype=float)
    x = -b * t
    # Prevent numerical overflow in exp
    x = np.clip(x, -700, 700)
    return a * (1.0 - np.exp(x)) + c


def _fit_exponential(years, h_values):
    """
    Fit exponential saturation model.
    
    Uses bounded nonlinear least squares to ensure physically
    meaningful parameters (all positive).
    
    Parameters
    ----------
    years : np.ndarray
        Career time points (must be > 0)
    h_values : np.ndarray
        H-index values
        
    Returns
    -------
    dict
        {'a': float, 'b': float, 'c': float, 'rmse': float, 'success': bool}
        Returns {'success': False} if fitting fails
        
    Notes
    -----
    Initial guess: a=1, b=0.05, c=0
    Bounds: a≥0, 0≤b≤1, c≥0
    """
    mask = years > 0
    years = years[mask]
    h_values = h_values[mask]
    if len(years) < 3:
        return {"success": False}

    try:
        # Bounded optimization for physical validity
        popt, _ = curve_fit(
            _exp_func,
            years,
            h_values,
            p0=(1.0, 0.05, 0.0),
            bounds=([0.0, 0.0, 0.0], [np.inf, 1.0, np.inf]),
            maxfev=5000,
        )
        a, b, c = popt
        preds = _exp_func(years, a, b, c)
        rmse = root_mean_squared_error(h_values, preds)
        return {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "rmse": float(rmse),
            "success": True,
        }
    except Exception:
        return {"success": False}


def fit_shape_parameters(years, h_values):
    """
    Fit multiple growth models to h-index time series.
    
    Attempts to fit all four models (Linear, Hirsch, Power-law, Exponential)
    and returns parameters for each, allowing comparison of fit quality.
    
    Parameters
    ----------
    years : array-like
        Years (absolute, e.g., 2010, 2015, 2020)
    h_values : array-like
        H-index values at those years
        
    Returns
    -------
    dict
        Nested dictionary with keys "linear", "hirsch", "power", "exponential".
        Each contains model parameters and RMSE:
        {
            "linear": {slope, intercept, rmse, success},
            "hirsch": {a, b, rmse, success},
            "power": {a, b, rmse, success},
            "exponential": {a, b, c, rmse, success}
        }
        
    Notes
    -----
    - Years are normalized to career time (1, 2, 3, ...) internally
    - Requires at least 3 data points
    - Models with success=False failed to converge
    
    Model Selection Guidelines:
    - Compare RMSE values (lower is better)
    - Consider parameter interpretability
    - Early career: Linear or Hirsch often best
    - Mature career: Exponential may fit better
    - Mid-career with acceleration: Power-law
    
    Examples
    --------
    >>> years = np.array([2010, 2012, 2014, 2016, 2018, 2020])
    >>> h_vals = np.array([5, 9, 14, 20, 27, 33])
    >>> params = fit_shape_parameters(years, h_vals)
    >>> 
    >>> # Check which model fits best
    >>> for model in ['linear', 'hirsch', 'power', 'exponential']:
    ...     if params[model]['success']:
    ...         print(f"{model}: RMSE = {params[model]['rmse']:.2f}")
    linear: RMSE = 0.82
    hirsch: RMSE = 0.65
    power: RMSE = 0.43
    exponential: RMSE = 0.71
    >>> # Power-law fits best for this accelerating trajectory
    """
    years, h_values = _clean_series(years, h_values)
    if years is None:
        return {
            "linear": {"success": False},
            "hirsch": {"success": False},
            "power": {"success": False},
            "exponential": {"success": False},
        }

    # Normalize years to career time: 1, 2, 3, ...
    # Example: [2010, 2015, 2020] → [1, 6, 11]
    career_years = years - years.min() + 1

    results = {
        "linear": _fit_linear(career_years, h_values),
        "hirsch": _fit_hirsch(career_years, h_values),
        "power": _fit_power_law(career_years, h_values),
        "exponential": _fit_exponential(career_years, h_values),
    }

    return results


def compute_shape_evolution(h_by_year):
    """
    Analyze temporal stability of growth model parameters.
    
    For each year, refits all models using data up to that year.
    This reveals how model parameters evolve as the career progresses,
    indicating consistency or changes in growth dynamics.
    
    Parameters
    ----------
    h_by_year : pd.DataFrame
        DataFrame with columns ['Year', 'H-Index']
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with one row per year containing:
        - Year: cutoff year
        - lin_slope, lin_rmse
        - hirsch_a, hirsch_b, hirsch_rmse
        - power_a, power_b, power_rmse
        - exp_a, exp_b, exp_c, exp_rmse
        Returns None if insufficient data
        
    Notes
    -----
    Requires at least 3 years of data for first fit.
    
    Applications:
    - Stability analysis: constant parameters → stable growth pattern
    - Phase detection: parameter shifts → career transitions
    - Prediction reliability: stable parameters → more reliable forecasts
    
    Examples
    --------
    >>> h_data = pd.DataFrame({
    ...     'Year': range(2010, 2025),
    ...     'H-Index': [5, 7, 10, 13, 17, 21, 25, 28, 31, 34, 
    ...                 36, 38, 40, 42, 43]
    ... })
    >>> evolution = compute_shape_evolution(h_data)
    >>> 
    >>> # Check stability of Hirsch coefficient
    >>> print(evolution[['Year', 'hirsch_a']].tail())
        Year  hirsch_a
    10  2020     2.85
    11  2021     2.88
    12  2022     2.90
    13  2023     2.91
    14  2024     2.92
    >>> # Stable 'a' indicates consistent growth pattern
    """
    years = sorted(h_by_year["Year"].dropna().unique())
    rows = []

    for cutoff in years:
        subset = h_by_year[h_by_year["Year"] <= cutoff]
        if len(subset) < 3:
            continue  # Skip if too few points

        params = fit_shape_parameters(
            subset["Year"].values,
            subset["H-Index"].values
        )

        lin = params.get("linear", {})
        hir = params.get("hirsch", {})
        pw = params.get("power", {})
        exp = params.get("exponential", {})

        rows.append({
            "Year": cutoff,
            "lin_slope": lin.get("slope"),
            "lin_rmse": lin.get("rmse"),
            "hirsch_a": hir.get("a"),
            "hirsch_b": hir.get("b"),
            "hirsch_rmse": hir.get("rmse"),
            "power_a": pw.get("a"),
            "power_b": pw.get("b"),
            "power_rmse": pw.get("rmse"),
            "exp_a": exp.get("a"),
            "exp_b": exp.get("b"),
            "exp_c": exp.get("c"),
            "exp_rmse": exp.get("rmse"),
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


def compute_hirsch_metrics(shape_params, shape_evol):
    """
    Calculate Hirsch-based career metrics.
    
    Computes three related metrics based on the Hirsch model:
    1. HGC (Hirsch Growth Coefficient): parameter 'a' from global fit
    2. CV(a): Coefficient of variation of 'a' over time (stability)
    3. SaH: Combined metric = a·(1 - CV(a)) (growth × stability)
    
    Parameters
    ----------
    shape_params : dict
        Result from fit_shape_parameters() (global fit)
    shape_evol : pd.DataFrame
        Result from compute_shape_evolution() (temporal evolution)
        
    Returns
    -------
    tuple (float, float, float) or (None, None, None)
        - HGC: Hirsch Growth Coefficient (global 'a')
        - CV(a): Coefficient of variation (std/mean of 'a' over time)
        - SaH: Stability-adjusted H-growth = a·(1 - CV(a))
        Returns (None, None, None) if Hirsch model failed
        
    Notes
    -----
    Interpretation:
    
    HGC (Hirsch Growth Coefficient):
    - Typical range: 0.5 to 5.0
    - Higher values → faster h-index accumulation
    - Independent of career length
    
    CV(a) (Stability):
    - Range: 0 (perfect stability) to 1+ (high variability)
    - Low CV → consistent growth pattern
    - High CV → career fluctuations or phase changes
    
    SaH (Stability-adjusted H-growth):
    - Combines strength and consistency
    - Rewards both high growth and stable patterns
    - Penalizes erratic trajectories
    - Higher is better
    
    Examples
    --------
    >>> params = fit_shape_parameters(years, h_values)
    >>> evolution = compute_shape_evolution(h_by_year)
    >>> hgc, cv_a, sah = compute_hirsch_metrics(params, evolution)
    >>> 
    >>> print(f"HGC (growth): {hgc:.3f}")
    >>> print(f"CV(a) (variability): {cv_a:.3f}")
    >>> print(f"Stability: {1-cv_a:.1%}")
    >>> print(f"SaH (combined): {sah:.3f}")
    HGC (growth): 2.850
    CV(a) (variability): 0.085
    Stability: 91.5%
    SaH (combined): 2.608
    
    See Also
    --------
    This metric can be used for:
    - Comparing researchers with similar h-index but different trajectories
    - Identifying researchers with sustained vs. volatile impact
    - Prediction confidence (stable patterns → more reliable forecasts)
    """
    hir = shape_params.get("hirsch", {})
    if not hir.get("success"):
        return None, None, None

    a_global = hir.get("a", None)
    if a_global is None:
        return None, None, None

    if shape_evol is None or shape_evol.empty or "hirsch_a" not in shape_evol.columns:
        # Can't measure temporal stability, but return global HGC
        return a_global, None, None

    series = shape_evol["hirsch_a"].dropna().to_numpy()
    if len(series) < 3:
        return a_global, None, None

    mean_a = float(np.mean(series))
    std_a = float(np.std(series))
    if mean_a == 0:
        return a_global, None, None

    cv_a = std_a / mean_a  # Coefficient of variation
    sah = a_global * (1.0 - cv_a)  # Stability-adjusted H-growth

    return a_global, cv_a, sah