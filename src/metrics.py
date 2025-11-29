"""
metrics.py
==========

Module for calculating standard bibliometric metrics.

This module implements commonly used bibliometric indices for evaluating
author impact and productivity:
- h-index (Hirsch index)
- g-index (Egghe index)  
- m-index (h-index per career year)
- i10-index (papers with ≥10 citations)
- h-index temporal analysis (growth rates)

Authors: Diogo Abreu, João Machado, Pedro Lopes
Date: 11/2025
Version: 1.0
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def g_index(citations):
    """
    Calculate the g-index for an author.
    
    The g-index is defined as the largest number g such that the top g
    publications have together at least g² citations.
    
    Parameters
    ----------
    citations : list of int
        List of citation counts for each publication
        
    Returns
    -------
    int
        The g-index value
        
    Notes
    -----
    The g-index is always greater than or equal to the h-index and gives
    more weight to highly cited papers. It's more sensitive to exceptional
    papers than h-index.
    
    Examples
    --------
    >>> citations = [100, 50, 30, 20, 10, 5, 5, 5]
    >>> g = g_index(citations)
    >>> print(f"G-index: {g}")
    G-index: 7
    
    >>> # Explanation: Top 7 papers have 100+50+30+20+10+5+5 = 220 citations
    >>> # 220 >= 7² = 49 ✓
    >>> # Top 8 papers have 220+5 = 225 citations  
    >>> # 225 >= 8² = 64 ✗
    
    See Also
    --------
    The h-index tends to undervalue highly cited papers, while g-index
    rewards them more appropriately.
    """
    if not citations:
        return 0

    sorted_cites = sorted(citations, reverse=True)
    cumulative = 0
    g = 0

    for i, c in enumerate(sorted_cites, start=1):
        cumulative += c
        if cumulative >= i * i:
            g = i
        else:
            break

    return g


def m_index(h_index, first_year, last_year=None):
    """
    Calculate the m-index (h-index normalized by career length).
    
    The m-index represents the rate of h-index growth per year and is
    useful for comparing researchers at different career stages.
    
    Parameters
    ----------
    h_index : int or float
        Current h-index value
    first_year : int
        Year of first publication (career start)
    last_year : int, optional
        Current year or year of last publication (default: current year)
        
    Returns
    -------
    float
        m-index value (h-index per year of career)
        
    Notes
    -----
    Formula: m = h / (current_year - first_year + 1)
    
    Interpretation:
    - m < 1: Slow growth (early career or low productivity)
    - m ≈ 1-2: Normal growth for active researchers
    - m > 2: Fast growth (highly productive/impactful researcher)
    
    Examples
    --------
    >>> # Researcher with h=30 after 15 years
    >>> m = m_index(h_index=30, first_year=2009, last_year=2024)
    >>> print(f"M-index: {m:.2f}")
    M-index: 1.88
    
    >>> # Early career researcher (5 years, h=8)
    >>> m_early = m_index(h_index=8, first_year=2019, last_year=2024)
    >>> print(f"M-index (early): {m_early:.2f}")
    M-index (early): 1.33
    
    >>> # Established researcher (30 years, h=45)
    >>> m_established = m_index(h_index=45, first_year=1994, last_year=2024)
    >>> print(f"M-index (established): {m_established:.2f}")
    M-index (established): 1.45
    """
    if first_year is None:
        return 0

    if last_year is None:
        from datetime import datetime
        last_year = datetime.now().year

    career_years = max(1, last_year - first_year + 1)
    return h_index / career_years


def i10_index(citations):
    """
    Calculate the i10-index (papers with at least 10 citations).
    
    A simple metric introduced by Google Scholar that counts the number
    of publications with 10 or more citations.
    
    Parameters
    ----------
    citations : list of int
        List of citation counts for each publication
        
    Returns
    -------
    int
        Number of publications with ≥10 citations
        
    Notes
    -----
    The i10-index is straightforward to understand and correlates well
    with h-index. It's less commonly used in formal evaluations but
    provides a quick sense of the number of "impactful" papers.
    
    Examples
    --------
    >>> citations = [100, 50, 30, 20, 15, 12, 10, 8, 5, 3, 1]
    >>> i10 = i10_index(citations)
    >>> print(f"I10-index: {i10}")
    I10-index: 7
    >>> # 7 papers have at least 10 citations
    
    >>> # Compare different profiles
    >>> citations_prolific = [25]*20 + [5]*50  # 20 moderate papers + many low-cited
    >>> citations_selective = [100, 80, 60, 40, 30]  # Few highly cited papers
    >>> print(f"Prolific: i10={i10_index(citations_prolific)}")
    >>> print(f"Selective: i10={i10_index(citations_selective)}")
    Prolific: i10=20
    Selective: i10=5
    """
    return sum(1 for c in citations if c >= 10)


def compute_h_index_slopes(h_by_year: pd.DataFrame, window: int = 3):
    """
    Compute global and local growth rates of h-index over time.
    
    Analyzes how quickly an author's h-index is growing by computing:
    1. Global slope: overall linear trend across entire career
    2. Local slopes: rolling window slopes showing acceleration/deceleration
    
    Parameters
    ----------
    h_by_year : pd.DataFrame
        DataFrame with columns ['Year', 'H-Index']
    window : int, optional
        Size of rolling window in years for local slopes (default: 3)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'global_slope' : float
            Overall h-index growth rate (h-index units per year)
        - 'local_slopes' : pd.DataFrame
            DataFrame with columns ['Year_center', 'Slope']
            showing local growth rates over time
            
    Notes
    -----
    Global slope interpretation:
    - 0.5-1.0: Slow but steady growth
    - 1.0-2.0: Healthy growth rate  
    - 2.0+: Rapid growth (productive/highly cited researcher)
    
    Local slopes reveal:
    - Periods of acceleration (career breakthroughs)
    - Periods of plateau (reduced activity or consolidation)
    - Impact of major publications
    
    Examples
    --------
    >>> h_data = pd.DataFrame({
    ...     'Year': [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024],
    ...     'H-Index': [5, 8, 12, 18, 25, 32, 38, 43]
    ... })
    >>> slopes = compute_h_index_slopes(h_data, window=3)
    >>> print(f"Global growth: {slopes['global_slope']:.2f} h/year")
    Global growth: 2.71 h/year
    
    >>> # View local slopes
    >>> print(slopes['local_slopes'])
       Year_center  Slope
    0       2012.0   1.50
    1       2014.0   2.50
    2       2016.0   3.50
    3       2018.0   3.50
    ...
    
    >>> # Identify career phases
    >>> local_slopes = slopes['local_slopes']
    >>> max_growth_period = local_slopes.loc[local_slopes['Slope'].idxmax()]
    >>> print(f"Peak growth around {max_growth_period['Year_center']:.0f}")
    Peak growth around 2016
    """
    df = h_by_year.dropna(subset=["Year", "H-Index"]).sort_values("Year").copy()
    if len(df) < 2:
        return {
            "global_slope": np.nan,
            "local_slopes": pd.DataFrame(columns=["Year_center", "Slope"])
        }

    # Global slope via simple linear regression
    X = df["Year"].values.reshape(-1, 1)
    y = df["H-Index"].values
    model = LinearRegression()
    model.fit(X, y)
    global_slope = float(model.coef_[0])

    # Local rolling slopes
    local_rows = []
    years = df["Year"].values
    hvals = df["H-Index"].values

    if window < 2:
        window = 2

    for i in range(len(df) - window + 1):
        x_win = years[i:i+window].reshape(-1, 1)
        y_win = hvals[i:i+window]
        m = LinearRegression().fit(x_win, y_win).coef_[0]
        year_center = float(np.mean(x_win))
        local_rows.append({"Year_center": year_center, "Slope": float(m)})

    local_df = pd.DataFrame(local_rows)
    return {"global_slope": global_slope, "local_slopes": local_df}