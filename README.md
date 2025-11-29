# Comprehensive Bibliometric Analysis System

A Python-based toolkit for collecting, analyzing, and visualizing academic publication metrics from Scopus API. Features h-index tracking, growth model fitting, and interactive dashboards for bibliometric research.

## 🎯 Overview

This project provides a complete workflow for:
- Automated data collection from Scopus API
- H-index temporal evolution analysis
- Growth pattern modeling (Linear, Power-law, Hirsch, Exponential)
- Novel stability metrics (HGC, SaH)
- Interactive visualization dashboards
- Future impact prediction

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Architecture](#-architecture)
- [Metrics & Models](#-metrics--models)
- [Output Files](#-output-files)
- [API Quota Management](#-api-quota-management)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

## ✨ Features

### Data Collection
- **Automated publication retrieval** from Scopus API
- **Citation tracking** for all publications
- **Complete metadata** extraction (DOI, authors, journals, citations)
- **Quota monitoring** to manage API limits

### Analysis Tools
- **Standard metrics**: h-index, g-index, m-index, i10-index
- **Growth models**: Linear, Hirsch (√t), Power-law (t^b), Exponential saturation
- **Novel metrics**: 
  - HGC (Hirsch Growth Coefficient)
  - CV(a) (Growth stability)
  - SaH (Stability-adjusted H-growth)
- **Slope analysis**: Global and local h-index growth rates

### Visualization
- **Interactive Streamlit dashboard**
- **Time series plots** with forecast overlays
- **Model comparison** charts
- **Parameter evolution** tracking
- **Publication tables** with sortable columns

### Prediction
- **Multiple regression models**: Linear, Polynomial, Spline, Random Forest
- **Growth-based forecasting**: Power-law, Exponential, Hirsch
- **Train/test validation** with adjustable cutoffs
- **Accuracy metrics**: RMSE, MAE

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Scopus API credentials ([Get them here](https://dev.elsevier.com/))

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/Digazz19/PI-H-Index.git
cd PI-H-Index

# Install required packages
pip install -r requirements.txt
```

### Requirements.txt
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
plotly>=5.17.0
requests>=2.31.0
python-dotenv>=1.0.0
tqdm>=4.66.0
```

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
SCOPUS_KEY=your_api_key_here
SCOPUS_INST_TOKEN=your_institution_token_here
```

**Getting API Credentials:**
1. Register at [Elsevier Developer Portal](https://dev.elsevier.com/)
2. Create an application to get your API Key
3. Obtain Institution Token from your library (if applicable)
4. You'll need to talk with Elsevier Support Team to give you a REFEID token within your API Key to be able to run option 4 without errors

## 🏃 Quick Start

### Step 1: Collect Publications

```bash
python cli.py
```

Choose option **1** and enter:
- Scopus Author ID (e.g., `7201540270`)
- Author name (e.g., `João M. Fernandes`)

This creates: `info/{author_id}/{Author_Name}.csv` and `.json`

### Step 2: Fetch Citations

Choose option **2** from the CLI menu. This retrieves citation data for all publications.

⚠️ **Note**: This can take significant time for highly cited authors.

### Step 3: Compute H-Index Timeline

Choose option **3** to calculate year-by-year h-index values.

Creates: `{Author_Name}_h_index_timeline.json`

### Step 4: Launch Dashboard

```bash
streamlit run dashboard.py
```

Open your browser at: `http://localhost:8501`

## 📖 Usage Guide

### Command-Line Interface (cli.py)

The CLI provides 5 main options:

1. **Fetch publications** - Initial data collection
2. **Fetch citations** - Retrieve citing articles for all publications
3. **Compute h-index timeline** - Calculate temporal evolution
4. **Check API quota** - Monitor remaining API calls
5. **Exit**

**Typical Workflow:**
```
Option 1 → Option 2 → Option 3 → streamlit run dashboard.py
```

### Dashboard Features

#### 1. General Statistics
- Publications count
- Total citations
- H-index, G-index, M-index, I10-index

#### 2. H-Index Evolution
- Historical timeline
- Power-law forecast overlay
- Interactive zoom and pan

#### 3. Slope Analysis
- Global growth rate (h-index/year)
- Local rolling window slopes
- Career acceleration/deceleration detection

#### 4. Model Comparison
- Multiple regression models
- Growth-based models
- Train/test split validation
- RMSE and MAE metrics

#### 5. Shape Parameters
- Model fit quality comparison
- Parameter values for all models
- Temporal parameter evolution
- Stability analysis

#### 6. Novel Metrics
- **HGC**: Growth coefficient from Hirsch model
- **CV(a)**: Coefficient of variation (stability measure)
- **SaH**: Combined growth × stability metric

## 🏗️ Architecture

### Module Structure

```
PI-H-Index/
├── cli.py                    # Command-line interface
├── dashboard.py              # Streamlit visualization
├── api_client.py             # Scopus API communication
├── data_processing.py        # Data collection & organization
├── metrics.py                # Standard bibliometric indices
├── shape_metrics.py          # Growth model fitting
├── prediction_models.py      # Forecasting functions
├── .env                      # API credentials (create this)
├── requirements.txt          # Python dependencies
└── info/                     # Data directory
    └── {author_id}/
        ├── {Author_Name}.csv
        ├── {Author_Name}.json
        ├── {Author_Name}_h_index_timeline.json
        └── cited_by_data/
            └── {eid}_citations.json
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| `api_client.py` | Scopus API requests, quota management |
| `data_processing.py` | Publication & citation data collection |
| `metrics.py` | Standard indices (g, m, i10, slopes) |
| `shape_metrics.py` | Growth model fitting, stability metrics |
| `prediction_models.py` | Power-law forecasting |
| `cli.py` | Interactive data collection interface |
| `dashboard.py` | Web-based visualization & analysis |

## 📊 Metrics & Models

### Standard Bibliometric Indices

**H-Index**: Largest h where h papers have ≥h citations each  
**G-Index**: Largest g where top g papers have ≥g² total citations  
**M-Index**: H-index per career year (h-index / years_active)  
**I10-Index**: Number of papers with ≥10 citations

### Growth Models

#### 1. Linear Model
```
h(t) = slope·t + intercept
```
Constant growth rate throughout career.

#### 2. Hirsch Model (Square Root)
```
h(t) = a·√t + b
```
- Parameter `a` = Hirsch Growth Coefficient (HGC)
- Reflects original Hirsch growth assumption

#### 3. Power-Law Model
```
h(t) = a·t^b
```
- `b < 1`: Decelerating growth
- `b = 1`: Linear growth
- `b > 1`: Accelerating growth

#### 4. Exponential Saturation
```
h(t) = a·(1 - e^(-b·t)) + c
```
- Asymptotic limit: h_max = a + c
- Common in mature careers

### Novel Stability Metrics

**HGC (Hirsch Growth Coefficient)**: Parameter `a` from Hirsch model  
- Typical range: 4.0 to 10.0
- Higher = faster h-index accumulation

**CV(a) (Coefficient of Variation)**: std(a) / mean(a) over time  
- Range: 0 (perfect stability) to 1+ (high variability)
- Low CV = consistent growth pattern

**SaH (Stability-adjusted H-growth)**: a·(1 - CV(a))  
- Combines growth strength with consistency
- Rewards stable, sustained impact

## 📁 Output Files

### Directory Structure
```
info/
└── 7201540270/
    ├── João_M._Fernandes.csv              # Publications table
    ├── João_M._Fernandes.json              # Complete metadata
    ├── João_M._Fernandes_h_index_timeline.json  # Year-by-year h-index
    └── cited_by_data/
        ├── 2-s2.0-123456789_citations.json
        └── 2-s2.0-987654321_citations.json

Predictions/
└── 7201540270_h_index_predictions.csv     # Forecast results

quota_status.json                           # API usage tracking
```

### File Formats

**Publications CSV**: Title, Year, Citations, DOI, Journal, Authors, etc.

**H-Index Timeline JSON**:
```json
{
  "2010": 5,
  "2011": 6,
  "2012": 8,
  ...
}
```

**Citations JSON**:
```json
[
  {"eid": "2-s2.0-999", "date": "2020-05-15"},
  {"eid": "2-s2.0-888", "date": "2021-03-22"}
]
```

## 🔒 API Quota Management

### Scopus API Limits
- Typically **20,000 requests per week**
- Rate limit headers tracked automatically
- Quota status saved to `quota_status.json`

### Request Costs
- **Publication search**: ~1 request per 25 results
- **Abstract details**: 1 request per publication
- **Citation search**: ~1 request per 25 citing articles

### Monitoring Quota

```python
# Check current quota status
python cli.py
# Choose option 4
```

Or programmatically:
```python
from api_client import check_quota_via_dummy_request
check_quota_via_dummy_request()
```

### Best Practices
1. Check quota before large data collections
2. Run citation fetching overnight for prolific authors
3. Use incremental updates (skips existing files)
4. Monitor `quota_status.json` regularly

## 🐛 Troubleshooting

### Common Issues

**"API key or Institution token not found"**
- Ensure `.env` file exists in project root
- Check credentials are correct (no extra spaces)

**"No CSV files found in /info"**
- Run CLI option 1 first to collect publications
- Check `info/{author_id}/` directory was created

**"H-Index timeline file not found"**
- Run CLI option 3 to compute timeline
- Requires citation data (option 2) to be collected first

**"Insufficient data to fit model"**
- Need at least 3 years of h-index data
- Check author has enough publication history

**Slow citation fetching**
- Normal for highly cited authors (100+ citations per paper)
- Let process run overnight if needed
- Consider testing with smaller author first

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional growth models (logistic, Gompertz)
- Confidence intervals for predictions
- Batch processing for multiple authors
- Export to LaTeX/PDF reports
- Field-normalized metrics
- Collaboration network analysis

**How to contribute:**
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@software{pi_h_index_2025,
  author = {Abreu, Diogo and Machado, João and Lopes, Pedro},
  title = {pinetreeaxe/BiblioMetric-Analyzer: Comprehensive Bibliometric Analysis System},
  year = {2025},
  month = {11},
  version = {1.0},
  url = {https://github.com/pinetreeaxe/BiblioMetric-Analyzer}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👥 Authors

- **Diogo Abreu**
- **João Machado** 
- **Pedro Lopes**

## 🙏 Acknowledgments

- Elsevier Scopus API for data access
- Hirsch (2005) for the original h-index concept
- Egghe (2006) for the g-index formulation

## 📧 Contact

For questions or support, please open an issue on GitHub or contact the authors.

***

**Note**: This tool is for research purposes. Respect Scopus API terms of service and usage limits. Not affiliated with Elsevier or Scopus.

***
