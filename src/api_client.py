"""
api_client.py
=============


BiblioMetric-Analyzer Scopus API communication module.


Standardized API functions for CLI Options 1-4:
- 🔍 fetch_all_publications(): CLI Option 1 core (pagination + progress)
- 📄 get_abstract_details(): Publication metadata enrichment
- 🚀 get_citation_years_for_article(): Single article citations (w/ progress)
- 🤫 get_citation_years_for_article_silent(): Batch citations (CLI Option 2)
- ⚠️  check_quota_via_dummy_request(): CLI Option 4 quota monitoring


Required Configuration (.env)
-----------------------------
SCOPUS_KEY=<your_api_key>
SCOPUS_INST_TOKEN=<your_institution_token>

Get credentials: https://dev.elsevier.com/

Scopus Quota (20k requests/week):
- Option 1: ~1 req/25 pubs + 1 req/pub (abstracts)
- Option 2: ~1 req/25 citations per publication
- Option 4: 1 dummy request (quota check)


Authors: Diogo Abreu, João Machado, Pedro Lopes
Date: 11/2025
Version: 2.0 (BiblioMetric-Analyzer Standardized)
"""


import os
import requests
import json
import time
import datetime
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION - BiblioMetric-Analyzer Endpoints
# ============================================================
SEARCH_URL = "https://api.elsevier.com/content/search/scopus"
ABSTRACT_URL = "https://api.elsevier.com/content/abstract/scopus_id/"

# Load credentials
load_dotenv()
API_KEY = os.getenv("SCOPUS_KEY", "").strip()
INST_TOKEN = os.getenv("SCOPUS_INST_TOKEN", "").strip()

if not API_KEY or not INST_TOKEN:
    print("❌ BiblioMetric-Analyzer: Missing SCOPUS_KEY or SCOPUS_INST_TOKEN in .env")
    print("   👉 Get credentials: https://dev.elsevier.com/")
    exit(1)


# ============================================================
# QUOTA MANAGEMENT (CLI Option 4)
# ============================================================
def save_quota_info(response, filename='quota_status.json'):
    """
    Save Scopus API quota status (automatic after every request).
    
    Extracts rate limit headers and saves to JSON for CLI Option 4 display.
    
    Parameters
    ----------
    response : requests.Response
        Scopus API response object
    filename : str, optional
        Quota file location (default: 'quota_status.json')
        
    File Format
    -----------
    {
        "X-RateLimit-Limit": "20000",
        "X-RateLimit-Remaining": "18543", 
        "X-RateLimit-Reset": "1735632000",
        "X-RateLimit-Reset-Time": "2024-12-30 00:00:00",
        "Timestamp": "2025-11-29 22:35:12"
    }
    
    BiblioMetric-Analyzer Integration
    ---------------------------------
    Every API call → save_quota_info() → CLI Option 4 reads latest status
    
    Examples
    --------
    >>> response = requests.get(SEARCH_URL, headers=headers, params=params)
    >>> save_quota_info(response)  # Auto-called after every API request
    """
    quota_info = {
        "X-RateLimit-Limit": response.headers.get("X-RateLimit-Limit"),
        "X-RateLimit-Remaining": response.headers.get("X-RateLimit-Remaining"),
        "X-RateLimit-Reset": response.headers.get("X-RateLimit-Reset"),
        "Timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Human-readable reset time
    reset_epoch = quota_info["X-RateLimit-Reset"]
    if reset_epoch and reset_epoch.isdigit():
        reset_time = datetime.datetime.fromtimestamp(int(reset_epoch))
        quota_info["X-RateLimit-Reset-Time"] = reset_time.strftime('%Y-%m-%d %H:%M:%S')
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(quota_info, f, indent=2, ensure_ascii=False)


def print_quota_info_file(filename='quota_status.json'):
    """
    Display quota status from saved file (CLI Option 4 helper).
    
    Parameters
    ----------
    filename : str, optional
        Quota status file
        
    BiblioMetric-Analyzer Output
    ----------------------------
    ⚠️  API Quota: 18,543/20,000 remaining (resets 2025-12-30)
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            quota_info = json.load(f)
        
        limit = quota_info.get('X-RateLimit-Limit', 'N/A')
        remaining = quota_info.get('X-RateLimit-Remaining', 'N/A')
        reset_time = quota_info.get('X-RateLimit-Reset-Time', 'N/A')
        
        print(f"⚠️  API Quota: {remaining}/{limit} remaining")
        print(f"   ⏰ Resets: {reset_time}")
        
    except Exception as ex:
        print(f"⚠️  Could not read quota file: {ex}")


def check_quota_via_dummy_request():
    """
    CLI Option 4: Check quota status with minimal API call.
    
    BiblioMetric-Analyzer standardized quota display:
    1. Makes dummy "ALL(test)" request (1 result)
    2. Updates quota_status.json 
    3. Displays remaining quota + reset time
    4. Usage tip for large authors
    
    Examples
    --------
    === 📊 BiblioMetric-Analyzer Menu ===
    ⚠️  4. Check/refresh API quota info
    Choose an option: 4
    
    ⚠️  Checking API quota status...
    ⚠️  API Quota: 18,543/20,000 remaining
       ⏰ Resets: 2025-12-30 00:00:00
    👉 Tip: Check quota before Options 1-2 for prolific authors!
    """
    headers = {
        "X-ELS-APIKey": API_KEY,
        "X-ELS-Insttoken": INST_TOKEN,
        "Accept": "application/json"
    }
    params = {"query": "ALL(test)", "count": 1}
    
    print("\n⚠️  Checking API quota status...")
    r = requests.get(SEARCH_URL, headers=headers, params=params)
    save_quota_info(r)
    print_quota_info_file()
    print("\n👉 Tip: Check quota before large runs (Options 1-2)!")


# ============================================================
# CORE API FUNCTIONS (CLI Options 1-2)
# ============================================================
def fetch_all_publications(author_id, max_results=5000, suppress_progress=False):
    """
    CLI Option 1 Core: Fetch ALL author publications with pagination.
    
    BiblioMetric-Analyzer standardized publication fetcher:
    1. 🔍 AU-ID({author_id}) query → COMPLETE view
    2. 📊 Automatic pagination (25/page → up to 5k max)
    3. Progress messages (suppressible for CLI batch mode)
    4. Real-time quota updates
    
    Returns
    -------
    list of dict
        Raw Scopus entries → collect_publication_data() input format
        
    Performance
    -----------
    Michael Karin (734 pubs): ~30 requests (25/page) → ~2 minutes
    Prolific authors (>1k): Monitor quota (Option 4 first!)
    
    Examples
    --------
    🔍 Searching for publications of AU-ID(36040291000)...
    Retrieved 25 results so far...
    Retrieved 50 results so far...
    ...
    Retrieved 734 results so far.
    """
    headers = {
        "X-ELS-APIKey": API_KEY,
        "X-ELS-Insttoken": INST_TOKEN,
        "Accept": "application/json"
    }
    start = 0
    count = 25
    all_entries = []
    
    if not suppress_progress:
        print(f"\n🔍 Searching for publications of AU-ID({author_id})...\n")
    
    while True:
        params = {
            "query": f"AU-ID({author_id})",
            "count": count,
            "start": start,
            "view": "COMPLETE"
        }
        r = requests.get(SEARCH_URL, headers=headers, params=params)
        save_quota_info(r)
        
        if r.status_code != 200:
            if not suppress_progress:
                print(f"⚠️  API Error {r.status_code}: {r.text[:100]}")
            break
        
        data = r.json().get("search-results", {})
        entries = data.get("entry", [])
        
        if not entries:
            break
        
        all_entries.extend(entries)
        if not suppress_progress:
            print(f"Retrieved {len(all_entries):,} results so far...")
        
        start += count
        if start >= max_results:
            if not suppress_progress:
                print(f"⚠️  Maximum {max_results:,} results reached.")
            break
    
    if not suppress_progress:
        print(f"✅ Retrieved {len(all_entries):,} total publications.")
    
    return all_entries


def get_abstract_details(eid):
    """
    Enrich publication with volume/issue/pages/OA status (CLI Option 1).
    
    Fetches FULL abstract record for EID → extracts journal metadata.
    
    Returns
    -------
    tuple (dict, bool)
        ({'Volume': '45', 'Issue': '3', 'Open Access': 'Yes'}, True)
        
    BiblioMetric-Analyzer Integration
    ---------------------------------
    collect_publication_data() → for each pub → get_abstract_details(eid)
    
    Examples
    --------
    >>> details, success = get_abstract_details("2-s2.0-85047404838")
    >>> print(f"Vol {details['Volume']} | OA: {details['Open Access']}")
    Vol 45 | OA: Yes
    """
    headers = {
        "X-ELS-APIKey": API_KEY,
        "X-ELS-Insttoken": INST_TOKEN,
        "Accept": "application/json"
    }
    url = f"{ABSTRACT_URL}{eid}?view=FULL"
    r = requests.get(url, headers=headers)
    save_quota_info(r)
    
    if r.status_code != 200:
        return {}, False
    
    abs_data = r.json().get("abstracts-retrieval-response", {})
    core = abs_data.get("coredata", {})
    
    # Open Access status
    open_access = "Yes" if core.get("openaccessFlag", False) else "No"
    
    # Volume/Issue
    volume = core.get("prism:volume", "")
    issue = core.get("prism:issueIdentifier", "")
    
    # Page parsing
    page_range = core.get("prism:pageRange", "")
    page_start, page_end, page_count = "", "", ""
    if page_range and "-" in page_range:
        parts = page_range.split("-", 1)
        page_start = parts[0].strip()
        page_end = parts[1].strip()
        try:
            page_count = str(max(1, abs(int(page_end) - int(page_start))))
        except ValueError:
            page_count = ""
    
    return {
        "Volume": volume,
        "Issue": issue,
        "Page start": page_start,
        "Page end": page_end,
        "Page count": page_count,
        "Open Access": open_access
    }, True


def get_citation_years_for_article(REFEID, author_folder, max_results=10000):
    """
    SINGLE article citation fetcher WITH progress bar (standalone use).
    
    BiblioMetric-Analyzer citation pagination + real-time stats:
    📥 2-s2.0-85047404838: 2,283 citations | Fetching: 100%| 92/92 [01:28, 1.65s/p]
    
    Parameters
    ----------
    REFEID : str
        Target article EID
    author_folder : str
        info/{author_id}/ path for cited_by_data/
    max_results : int
        Citation cap (default: 10k)
        
    Notes
    -----
    - Progress: citations/sec + pages/sec
    - Auto-pagination (25/page)
    - Saves: cited_by_data/{REFEID}_citations.json
    
    Examples
    --------
    >>> get_citation_years_for_article("2-s2.0-85047404838", "info/36040291000")
    📥 2-s2.0-85047404838: 2,283 citations | Fetching: 100%| 92/92 [01:28<00:00, 1.65s/pages, cit/s=25.6]
    ✅ Saved: info/36040291000/cited_by_data/2-s2.0-85047404838_citations.json
    """
    from tqdm import tqdm
    
    headers = {
        "X-ELS-APIKey": API_KEY,
        "X-ELS-Insttoken": INST_TOKEN,
        "Accept": "application/json"
    }
    
    # Count total citations
    count_params = {"query": f"REFEID({REFEID})", "view": "STANDARD", "count": 1, "start": 0}
    count_response = requests.get(SEARCH_URL, headers=headers, params=count_params)
    total_citations = 0
    if count_response.status_code == 200:
        total_str = count_response.json().get("search-results", {}).get("opensearch:totalResults", "0")
        total_citations = int(total_str) if total_str.isdigit() else 0
    
    print(f"📥 {REFEID}: {total_citations:,} citations", end=" | ")
    
    citing_entries = []
    start = 0
    count = 25
    total_pages = (total_citations + count - 1) // count
    
    with tqdm(total=total_pages, desc="Fetching", unit="pages", ncols=100) as pbar:
        while True:
            params = {"query": f"REFEID({REFEID})", "view": "STANDARD", "start": start, "count": count}
            r = requests.get(SEARCH_URL, headers=headers, params=params)
            save_quota_info(r)
            
            if r.status_code != 200:
                pbar.set_postfix({"status": f"ERROR {r.status_code}"})
                break
            
            data = r.json()
            entries = data.get("search-results", {}).get("entry", [])
            if not entries:
                break
            
            page_citations = 0
            for entry in entries:
                eid = entry.get("eid", "")
                cover_date = entry.get("prism:coverDate", "")
                if eid and cover_date:
                    citing_entries.append({"eid": eid, "date": cover_date})
                    page_citations += 1
            
            elapsed = pbar.format_dict['elapsed'].total_seconds()
            cit_per_sec = page_citations / (elapsed + 0.001)
            pbar.set_postfix({
                "cit/s": f"{cit_per_sec:.1f}",
                "speed": f"{page_citations} cit/page"
            })
            
            start += count
            pbar.update(1)
            if start >= total_citations or start >= max_results:
                break
    
    # Save with sanitized filename
    safe_refeid = REFEID.replace("/", "_").replace(":", "_")
    cited_by_folder = os.path.join(author_folder, "cited_by_data")
    os.makedirs(cited_by_folder, exist_ok=True)
    out_path = os.path.join(cited_by_folder, f"{safe_refeid}_citations.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(citing_entries, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Saved: {out_path}")


def get_citation_years_for_article_silent(REFEID, author_folder, max_results=10000):
    """
    SILENT citation fetcher for CLI Option 2 batch processing.
    
    Identical to get_citation_years_for_article() but NO console output.
    Perfect for get_all_articles_that_cited_the_author() outer progress.
    
    Returns
    -------
    int
        Citation count (for outer progress bar)
        
    BiblioMetric-Analyzer Dual Progress
    -----------------------------------
    📥 [45/635] 2-s2.0-850474048 | Liver.. 📊 PROGRESS: 7%| cites=2.3k, new=45
         ↑ silent inner calls          ↑ outer progress manages display
    
    Examples
    --------
    >>> citations = get_citation_years_for_article_silent("2-s2.0-85047404838", "info/36040291000")
    >>> print(f"✅ {citations:,} citations")  # Outer display only
    ✅ 2,283 citations
    """
    headers = {
        "X-ELS-APIKey": API_KEY,
        "X-ELS-Insttoken": INST_TOKEN,
        "Accept": "application/json"
    }
    
    # Silent total count
    count_params = {"query": f"REFEID({REFEID})", "view": "STANDARD", "count": 1, "start": 0}
    count_response = requests.get(SEARCH_URL, headers=headers, params=count_params)
    total_citations = 0
    if count_response.status_code == 200:
        total_str = count_response.json().get("search-results", {}).get("opensearch:totalResults", "0")
        total_citations = int(total_str) if total_str.isdigit() else 0
    
    citing_entries = []
    start = 0
    count = 25
    
    while True:
        params = {"query": f"REFEID({REFEID})", "view": "STANDARD", "start": start, "count": count}
        r = requests.get(SEARCH_URL, headers=headers, params=params)
        save_quota_info(r)
        
        if r.status_code != 200:
            break
        
        data = r.json()
        entries = data.get("search-results", {}).get("entry", [])
        if not entries:
            break
        
        for entry in entries:
            eid = entry.get("eid", "")
            cover_date = entry.get("prism:coverDate", "")
            if eid and cover_date:
                citing_entries.append({"eid": eid, "date": cover_date})
        
        start += count
        if start >= total_citations or start >= max_results:
            break
    
    # Silent save
    safe_refeid = REFEID.replace("/", "_").replace(":", "_")
    cited_by_folder = os.path.join(author_folder, "cited_by_data")
    os.makedirs(cited_by_folder, exist_ok=True)
    out_path = os.path.join(cited_by_folder, f"{safe_refeid}_citations.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(citing_entries, f, indent=4, ensure_ascii=False)
    
    return len(citing_entries)
