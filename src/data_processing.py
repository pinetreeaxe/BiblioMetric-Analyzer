"""
data_processing.py
==================


Module for processing and organizing data collected from Scopus API
for BiblioMetric-Analyzer.


This module provides standardized functions for:
- 🔍 Collecting publication data with progress bars (Option 1)
- 🚀 Managing citation data collection (Option 2 support)
- 📈 Computing h-index evolution timelines (Option 3)
- 📊 Loading and organizing stored citation data


Dependencies
------------
- api_client: Scopus API communication + silent citation fetching
- json, os: File handling and data persistence
- tqdm: Progress bars with real-time statistics


Authors: Diogo Abreu, João Machado, Pedro Lopes
Date: 11/2025
Version: 2.0 (BiblioMetric-Analyzer Standardized)
"""


import os
import json
from tqdm import tqdm
from api_client import (
    fetch_all_publications, 
    get_abstract_details, 
    get_citation_years_for_article_silent
)


def collect_publication_data(author_id, author_name, total_pubs=None, suppress_fetch_progress=False):
    """
    Collect complete publication data for BiblioMetric-Analyzer (Option 1).
    
    Standardized CLI Option 1 workflow:
    1. 🔍 Fetch all publications via Scopus API (pagination)
    2. 📊 Enrich each with abstract metadata (volume/issue/pages/OA)
    3. 💾 Returns structured data ready for CSV/JSON export
    
    Parameters
    ----------
    author_id : str
        Scopus Author ID (e.g., "36040291000")
    author_name : str
        Author's full name for consistent file naming
    total_pubs : int, optional
        Pre-fetched total count for accurate progress bar
    suppress_fetch_progress : bool, optional
        Hide fetch_all_publications spam (CLI mode)
        
    Returns
    -------
    list of dict
        Standardized publication records. Each dict contains:
        - Authors: Short semicolon-separated names
        - Author full names: "Name (authid)" format  
        - Author(s) ID: Semicolon-separated authids
        - Title: Publication title
        - Year: YYYY from coverDate
        - Source title: Journal/conference name
        - Volume, Issue: From abstract metadata
        - Page start/end/count: Parsed from pageRange
        - Cited by: Citation count
        - DOI: Digital Object Identifier
        - Link: Scopus inward URL
        - Document Type: Article/Conference/etc.
        - Open Access: "Yes"/"No"
        - Source: "Scopus"
        - EID: Scopus Electronic ID (key field)
        
    BiblioMetric-Analyzer Workflow
    ------------------------------
    CLI Option 1 → collect_publication_data() → info/{author_id}/{name}.json
    CLI Option 2 → get_all_articles_that_cited_the_author() → cited_by_data/
    
    Performance Notes
    -----------------
    - ~1 API call per publication (abstract retrieval)
    - Slow for prolific authors (>100 pubs): ~2-5min/100 pubs
    - Graceful error handling: continues on individual failures
    - Real-time quota updates via api_client.save_quota_info()
    
    Examples
    --------
    >>> pubs = collect_publication_data("36040291000", "Michael Karin", 734)
    📊 Processing 734 publications...
    Processing pubs: 100%|██████████| 734/734 [04:32<00:00, 2.7pub/s]
    📈 Processing stats:
       • Total publications: 734
       • Successfully processed: 728
       • Errors: 6
    
    >>> print(pubs[0])
    {
        'Authors': 'Karin M; Johnson G',
        'Author full names': 'Karin M (36040291000); Johnson G (1234567890)',
        'Title': 'Liver Cancer Initiation Requires p53 Inhibition',
        'Year': '2018',
        'EID': '2-s2.0-85047404838',
        ...
    }
    """
    # Phase 1: Fetch raw publications (suppress CLI spam)
    entries = fetch_all_publications(author_id, suppress_progress=suppress_fetch_progress)
    
    if not entries:
        print("❌ No publications found for this author.")
        return []
    
    # Phase 2: Pre-flight summary
    if total_pubs:
        print(f"📊 Processing {total_pubs:,} publications...")
    
    publications = []
    errors = 0
    
    # Phase 3: Enrich with progress bar
    for i, entry in enumerate(tqdm(entries, desc="🔍 Processing pubs", unit="pub")):
        try:
            # Extract core metadata
            eid = entry.get("eid", "")
            title = entry.get("dc:title", "N/A")
            year = entry.get("prism:coverDate", "N/A")[:4]
            cited = int(entry.get("citedby-count", "0"))
            doi = entry.get("prism:doi", "")
            source_title = entry.get("prism:publicationName", "N/A")
            doc_type = entry.get("subtypeDescription", "N/A")
            
            # Parse authors (standardized format)
            authors = entry.get("author", [])
            authors_short = "; ".join(a.get("authname", "") for a in authors)
            authors_full = "; ".join(
                f"{a.get('authname', '')} ({a.get('authid', 'N/A')})" for a in authors
            )
            author_ids = "; ".join(a.get("authid", "") for a in authors)
            
            # Phase 3b: Abstract metadata (volume/pages/OA)
            abs_data, success = get_abstract_details(eid)
            
            publications.append({
                "Authors": authors_short,
                "Author full names": authors_full,
                "Author(s) ID": author_ids,
                "Title": title,
                "Year": year,
                "Source title": source_title,
                "Volume": abs_data.get("Volume", ""),
                "Issue": abs_data.get("Issue", ""),
                "Page start": abs_data.get("Page start", ""),
                "Page end": abs_data.get("Page end", ""),
                "Page count": abs_data.get("Page count", ""),
                "Cited by": cited,
                "DOI": doi,
                "Link": f"https://www.scopus.com/inward/record.uri?eid={eid}",
                "Document Type": doc_type,
                "Open Access": abs_data.get("Open Access", "No"),
                "Source": "Scopus",
                "EID": eid  # Critical for citation lookup
            })
            
        except Exception as ex:
            errors += 1
            tqdm.write(f"⚠️ Error pub {i+1}: {str(ex)[:50]}")
    
    # Phase 4: Final statistics
    if errors > 0:
        tqdm.write(f"⚠️ Completed with {errors} errors")
    
    print(f"\n📈 Processing stats:")
    print(f"   • Total publications: {len(entries):,}")
    print(f"   • Successfully processed: {len(publications):,}")
    print(f"   • Errors: {errors:,}")
    
    return publications


def get_all_articles_that_cited_the_author(author_id, author_name, total_pubs=None):
    """
    Fetch citations for ALL author publications (CLI Option 2).
    
    BiblioMetric-Analyzer standardized dual-progress workflow:
    1. 📊 Pre-flight: Count existing/missing citation files
    2. 🚀 Single clean progress bar (current pub + stats)
    3. Silent API calls via get_citation_years_for_article_silent()
    4. 📈 Final statistics with completion summary
    
    Parameters
    ----------
    author_id : str
        Scopus Author ID
    author_name : str
        Author name for status display
    total_pubs : int, optional
        Total publications (pre-flight info)
        
    File Structure Created
    ----------------------
    info/{author_id}/cited_by_data/
    ├── 2-s2.0-85047404838_citations.json  ← 2,283 citing articles
    ├── 2-s2.0-85044512447_citations.json  ← 1,892 citing articles
    └── ...
    
    Progress Bar Features
    ---------------------
    📥 [45/635] 2-s2.0-850474048 | Liver Cancer.. 📊 PROGRESS: 7%| 45/635 [2h15m<27h, 15.2s/pub, cites=2.3k, new=45, skip=99]
    
    Notes
    -----
    - Skips existing citation files (resume capability)
    - Real-time citation counts per publication
    - Single-line clean progress (no overlap)
    - Graceful error handling per publication
    
    Examples
    --------
    >>> get_all_articles_that_cited_the_author("36040291000", "Michael Karin", 734)
    📊 Found 734 publications | Existing: 99 | New: 635
    🚀 Fetching 635/734 citations
    📥 [1/635] 2-s2.0-850511927 | New mito.. 📊 PROGRESS: 0%| 1/635 [00:57, cites=968, new=1]
    📈 COMPLETE!
       • Total: 734 | New: 635 | Skip: 99 | Errors: 0
    """
    safe_name = author_name.replace(" ", "_")
    author_folder = os.path.join("info", author_id)
    json_path = os.path.join(author_folder, f"{safe_name}.json")
    cited_by_folder = os.path.join(author_folder, "cited_by_data")
    os.makedirs(cited_by_folder, exist_ok=True)
    
    with open(json_path, "r", encoding="utf-8") as f:
        publications = json.load(f)
    
    # Pre-flight: Filter missing citations only
    publications_to_process = []
    for pub in publications:
        refeid = pub.get("EID", "")
        out_path = os.path.join(cited_by_folder, f"{refeid.replace('/', '_').replace(':', '_')}_citations.json")
        if refeid and not os.path.exists(out_path):
            publications_to_process.append(pub)
    
    total_to_process = len(publications_to_process)
    if total_to_process == 0:
        print("✅ All citations already fetched!")
        return
    
    print(f"\n🚀 Fetching {total_to_process:,}/{len(publications):,} citations\n")
    
    new_fetched = 0
    skipped = len(publications) - total_to_process
    errors = 0
    
    # Single clean progress bar (current article + stats)
    with tqdm(total=total_to_process, desc="📊 PROGRESS", 
             unit="pub", leave=True, ncols=140, position=0) as pbar:
        
        for i, pub in enumerate(publications_to_process):
            refeid = pub.get("EID", "")
            title_short = (pub.get("Title", "N/A")[:25] + "..") if len(pub.get("Title", "")) > 25 else pub.get("Title", "N/A")
            
            # Live current article status
            pbar.set_description(f"📥 [{i+1}/{total_to_process}] {refeid[:16]}")
            
            try:
                citations_count = get_citation_years_for_article_silent(refeid, author_folder)
                new_fetched += 1
                pbar.set_postfix({
                    "cites": f"{citations_count:,}", 
                    "new": new_fetched, 
                    "skip": skipped, 
                    "err": errors
                })
            except Exception as ex:
                errors += 1
                pbar.set_postfix({
                    "error": str(ex)[:20], 
                    "new": new_fetched, 
                    "err": errors
                })
            
            pbar.update(1)
    
    # Final standardized summary
    print(f"\n📈 COMPLETE!")
    print(f"   • Total publications: {len(publications):,}")
    print(f"   • New citations: {new_fetched:,}")
    print(f"   • Skipped (existing): {skipped:,}")
    print(f"   • Errors: {errors:,}")
    print(f"✅ All citations for {author_name} processed!")


def load_all_citations(author_folder):
    """
    Load ALL citation data for h-index computation (Option 3 support).
    
    Parses cited_by_data/*_citations.json files → extracts years → 
    {pub_eid: [2018, 2019, 2020, ...]} format for h-index calculation.
    
    Parameters
    ----------
    author_folder : str
        Path: info/{author_id}/ (must contain cited_by_data/)
        
    Returns
    -------
    dict
        {pub_eid: [citation_years]} format for compute_h_index_for_year()
        
    File Processing
    ---------------
    cited_by_data/2-s2.0-85047404838_citations.json → ["2018-05-15", ...]
    ↓ parse years
    "2-s2.0-85047404838": [2018, 2018, 2019, 2020, ...]
    
    Notes
    -----
    - Silent date parsing (skips invalid YYYY-MM-DD)
    - EID keys preserve underscores from filename
    - Empty dict if no cited_by_data/ folder
    
    Examples
    --------
    >>> citations = load_all_citations("info/36040291000")
    >>> pub = "2-s2.0-85047404838"
    >>> print(f"{pub}: {len(citations[pub])} citations over {len(set(citations[pub]))} years")
    2-s2.0-85047404838: 2283 citations over 7 years
    """
    citations_data = {}
    cited_by_folder = os.path.join(author_folder, "cited_by_data")
    
    if not os.path.exists(cited_by_folder):
        print(f"📂 Citations folder not found: {cited_by_folder}")
        return citations_data
    
    for filename in os.listdir(cited_by_folder):
        if filename.endswith("_citations.json"):
            pub_eid = filename.replace("_citations.json", "")
            
            try:
                with open(os.path.join(cited_by_folder, filename), "r", encoding="utf-8") as f:
                    citing_entries = json.load(f)
                
                years = []
                for entry in citing_entries:
                    date_str = entry.get("date", "")
                    if date_str and len(date_str) >= 4:
                        try:
                            year = int(date_str[:4])
                            years.append(year)
                        except ValueError:
                            continue
                
                citations_data[pub_eid] = years
            except Exception as ex:
                print(f"⚠️ Skip {filename}: {ex}")
    
    return citations_data


def compute_h_index_for_year(citations_data, year):
    """
    Core h-index algorithm for specific year.
    
    Counts citations ≤ year per publication → sorts → finds max h where
    h papers have ≥h citations (classic h-index definition).
    
    Parameters
    ----------
    citations_data : dict
        From load_all_citations(): {pub_eid: [years]}
    year : int
        Compute h-index up to this year
        
    Returns
    -------
    int
        h-index value
        
    Algorithm
    ---------
    1. [pub1: 3 cites≤2015, pub2: 2, pub3: 2, ...]
    2. Sort descending: [12, 8, 5, 3, 2, 2, 1, ...]
    3. h=5 ✓ (5th paper has ≥5 cites)
    4. h=6 ✗ (6th paper has 2<6 cites)
    
    Examples
    --------
    >>> citations = {"pub1": [2010,2015], "pub2": [2012,2015], "pub3": [2015]}
    >>> compute_h_index_for_year(citations, 2015)
    2  # pub1:2, pub2:2, pub3:1 → sorted [2,2,1] → h=2
    """
    citation_counts = []
    
    for pub_eid, years in citations_data.items():
        count = sum(1 for y in years if y <= year)
        citation_counts.append(count)
    
    citation_counts.sort(reverse=True)
    
    h = 0
    for i, count in enumerate(citation_counts, start=1):
        if count >= i:
            h = i
        else:
            break
    
    return h


def h_index_over_time(author_folder):
    """
    Complete h-index timeline computation (CLI Option 3).
    
    1. 📊 Load all citations → citations_data dict
    2. 📈 Compute h-index for every citation year
    3. 💾 Returns {year: h_index} for visualization
    
    Parameters
    ----------
    author_folder : str
        info/{author_id}/ path
        
    Returns
    -------
    dict
        {year: h_index} → {2010: 5, 2011: 6, 2012: 8, ...}
        
    BiblioMetric-Analyzer Integration
    ---------------------------------
    CLI Option 3 → h_index_over_time() → {name}_h_index_timeline.json
    Dashboard ← JSON timeline visualization
    
    Notes
    -----
    - Only years with citations (no empty years)
    - Monotonic: h-index never decreases
    - Fast: O(P*C) where P=publications, C=citations total
    
    Examples
    --------
    >>> timeline = h_index_over_time("info/36040291000")
    >>> print(f"Career h-index: {max(timeline.values())} in {max(timeline)}")
    Career h-index: 42 in 2024
    """
    citations_data = load_all_citations(author_folder)
    
    if not citations_data:
        print("❌ No citation data found.")
        return {}
    
    # Collect unique citation years
    all_years = set()
    for years in citations_data.values():
        all_years.update(years)
    
    if not all_years:
        print("❌ No citation years found.")
        return {}
    
    # Compute timeline
    year_h_index = {}
    for year in sorted(all_years):
        h = compute_h_index_for_year(citations_data, year)
        year_h_index[year] = h
    
    return year_h_index
