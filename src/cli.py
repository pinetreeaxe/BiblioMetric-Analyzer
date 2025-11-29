"""
cli.py
======

Command-line interface for the BiblioMetric-Analyzer system.

This module provides an interactive text-based menu for:
- Fetching author publications from Scopus (Option 1)
- Collecting citation data with SINGLE CLEAN progress bar (Option 2) 
- Computing h-index timelines with statistics (Option 3)
- Checking API quota status (Option 4)
- Clean exit (Option 5)

The CLI is the primary interface for data collection workflows,
complementing the Streamlit dashboard used for visualization.

Workflow
--------
Typical usage sequence:
1. Fetch publications (Option 1) → 📊 creates CSV/JSON files
2. Fetch citations (Option 2) → 🚀 SINGLE clean progress bar
3. Compute h-index timeline (Option 3) → 📈 creates timeline JSON
4. Check quota before large runs (Option 4) → ⚠️ quota monitoring
5. View results: streamlit run dashboard.py

File Structure Created
----------------------
info/
└── {author_id}/
    ├── {Author_Name}.csv                 ← Option 1
    ├── {Author_Name}.json                ← Option 1  
    ├── {Author_Name}_h_index_timeline.json ← Option 3
    └── cited_by_data/
        ├── {eid}_citations.json          ← Option 2
        ├── ...

Authors: Diogo Abreu, João Machado, Pedro Lopes
Date: 11/2025
Version: 2.1 (SINGLE CLEAN PROGRESS BAR)
"""

import os
import json
import pandas as pd
import requests
from tqdm import tqdm
from data_processing import (
    collect_publication_data,
    get_all_articles_that_cited_the_author,
    h_index_over_time
)
from api_client import check_quota_via_dummy_request, API_KEY, INST_TOKEN, SEARCH_URL
from api_client import get_citation_years_for_article_silent  # NEW: Direct silent API call


def show_menu():
    """Display the standardized BiblioMetric-Analyzer menu."""
    print("\n=== 📊 BiblioMetric-Analyzer Menu ===")
    print("🔍 1. Fetch publications for an author")
    print("🚀 2. Fetch citations for all articles") 
    print("📈 3. Compute h-index timeline for an author")
    print("⚠️  4. Check/refresh API quota info")
    print("❌ 5. Exit")
    return input("Choose an option: ").strip()


def option_2_fetch_all_citations():
    """🚀 Option 2: Fetch citations for ALL publications - SINGLE CLEAN BAR"""
    author_id = input("Enter Scopus Author ID: ").strip()
    author_name = input("Enter author name: ").strip()
    
    safe_name = author_name.replace(" ", "_")
    author_folder = os.path.join("info", author_id)
    json_path = os.path.join(author_folder, f"{safe_name}.json")
    
    # Pre-flight check
    if not os.path.exists(json_path):
        print(f"❌ Publications file not found: {json_path}")
        print("👉 Run Option 1 first!")
        input("Press Enter to continue...")
        return
    
    with open(json_path, "r", encoding="utf-8") as f:
        publications = json.load(f)
    
    # Count missing citations ONLY
    cited_by_folder = os.path.join(author_folder, "cited_by_data")
    os.makedirs(cited_by_folder, exist_ok=True)
    
    publications_to_process = []
    for pub in publications:
        refeid = pub.get("EID", "")
        if refeid:
            safe_eid = refeid.replace('/', '_').replace(':', '_')
            out_path = os.path.join(cited_by_folder, f"{safe_eid}_citations.json")
            if not os.path.exists(out_path):
                publications_to_process.append(pub)
    
    total_to_process = len(publications_to_process)
    total_pubs = len(publications)
    
    if total_to_process == 0:
        print("✅ All citations already fetched!")
        input("Press Enter to continue...")
        return
    
    print(f"\n📊 Found {total_pubs:,} publications")
    print(f"📊 Existing citations: {total_pubs - total_to_process:,}")
    print(f"📊 New citations to fetch: {total_to_process:,}\n")
    
    print(f"🚀 Fetching {total_to_process:,}/{total_pubs:,} citations")
    
    # =====================================================
    # SINGLE CLEAN PROGRESS BAR - NO NESTING/OVERLAP
    # =====================================================
    new_fetched = 0
    total_citations = 0
    errors = 0
    
    with tqdm(total=total_to_process, 
             desc="📊 CITATIONS", 
             unit="pub", 
             leave=True, 
             ncols=140,
             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
        
        for i, pub in enumerate(publications_to_process):
            refeid = pub.get("EID", "")
            title = pub.get("Title", "N/A")[:35]  # Truncated title
            
            # SINGLE LINE STATUS: Current pub + preview
            pbar.set_description(f"📥 [{i+1:>3}/{total_to_process}] {refeid[:20]} | {title}")
            
            try:
                # SILENT API CALL - NO INNER PROGRESS BARS
                citations_count = get_citation_years_for_article_silent(refeid, author_folder)
                new_fetched += 1
                total_citations += citations_count
                
                # LIVE STATS: citations, new files, errors, rate
                pbar.set_postfix({
                    "cites": f"{citations_count:,}",
                    "new": new_fetched,
                    "err": errors,
                    "total": f"{total_citations:,}"
                })
                
            except Exception as ex:
                errors += 1
                pbar.set_postfix({
                    "error": str(ex)[:20],
                    "new": new_fetched,
                    "err": errors
                })
            
            pbar.update(1)
    
    # FINAL CLEAN SUMMARY
    print(f"\n{'='*70}")
    print(f"📈 CITATIONS COMPLETE!")
    print(f"   • Total publications: {total_pubs:,}")
    print(f"   • New citation files created: {new_fetched:,}")
    print(f"   • Skipped (existing): {total_pubs - total_to_process:,}")
    print(f"   • Total citations fetched: {total_citations:,}")
    print(f"   • Errors: {errors:,}")
    print(f"✅ Ready for h-index analysis (Option 3)!")
    print(f"📁 Data saved: {cited_by_folder}")
    print(f"{'='*70}\n")
    
    input("Press Enter to continue...")


def main():
    """Main CLI loop with standardized emoji-styled output."""
    print("=== 📊 BiblioMetric-Analyzer CLI ===")
    while True:
        choice = show_menu()
        
        if choice == "1":
            # Option 1: Fetch publications - STANDARDIZED
            author_id = input("Enter Scopus Author ID (e.g., 36040291000): ").strip()
            author_name = input("Enter author name (e.g., Michael Karin): ").strip()
            
            print(f"\n🔍 Fetching publications for AU-ID({author_id})...")
            
            # Pre-flight total count
            headers = {
                "X-ELS-APIKey": API_KEY,
                "X-ELS-Insttoken": INST_TOKEN,
                "Accept": "application/json"
            }
            count_params = {
                "query": f"AU-ID({author_id})",
                "count": 1,
                "start": 0,
                "view": "STANDARD"
            }
            count_response = requests.get(SEARCH_URL, headers=headers, params=count_params)
            
            total_pubs = 0
            if count_response.status_code == 200:
                total_str = count_response.json().get("search-results", {}).get("opensearch:totalResults", "0")
                total_pubs = int(total_str) if total_str.isdigit() else 0
                print(f"📊 Found {total_pubs:,} total publications")
            else:
                print("⚠️ Could not fetch total count, proceeding...")
            
            # Fetch with progress bar
            results = collect_publication_data(author_id, author_name, total_pubs, suppress_fetch_progress=True)
            
            if results:
                df = pd.DataFrame(results)
                print(f"\n✅ Publications processed: {len(df):,}")
                
                safe_name = author_name.replace(" ", "_")
                target_dir = os.path.join("info", author_id)
                os.makedirs(target_dir, exist_ok=True)
                
                csv_path = os.path.join(target_dir, f"{safe_name}.csv")
                json_path = os.path.join(target_dir, f"{safe_name}.json")
                
                print("💾 Saving data to CSV and JSON files...")
                df.to_csv(csv_path, index=False)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                
                print(f"\n✅ Data saved to:")
                print(f"   • {csv_path}")
                print(f"   • {json_path}")
            else:
                print("❌ No publications found.")
                
        elif choice == "2":
            # Option 2: NEW SINGLE CLEAN PROGRESS BAR
            option_2_fetch_all_citations()
            
        elif choice == "3":
            # Option 3: H-Index Timeline - STANDARDIZED
            author_id = input("Enter Scopus Author ID: ").strip()
            author_name = input("Enter author name: ").strip()
            
            safe_name = author_name.replace(" ", "_")
            target_dir = os.path.join("info", author_id)
            cited_by_folder = os.path.join(target_dir, "cited_by_data")
            
            if not os.path.exists(cited_by_folder):
                print(f"\n❌ Citation data not found at {cited_by_folder}")
                print("   👉 Run Option 2 (Fetch citations) first.")
            else:
                # Count citation files for pre-flight
                citation_files = [f for f in os.listdir(cited_by_folder) 
                                if f.endswith("_citations.json")]
                total_citations_files = len(citation_files)
                
                print(f"\n📊 Found {total_citations_files:,} citation files")
                print("📈 Computing h-index timeline...")
                
                h_index_timeline = h_index_over_time(target_dir)
                
                if h_index_timeline:
                    print(f"\n📈 H-Index Timeline ({len(h_index_timeline)} years):")
                    print("   Year | H-Index")
                    print("   " + "-" * 20)
                    for year, h_value in sorted(h_index_timeline.items()):
                        print(f"   {year:>4} | {h_value:>7}")
                    
                    h_index_path = os.path.join(target_dir, f"{safe_name}_h_index_timeline.json")
                    with open(h_index_path, "w", encoding="utf-8") as f:
                        json.dump(h_index_timeline, f, indent=4, ensure_ascii=False)
                    
                    print(f"\n✅ H-index timeline saved to:")
                    print(f"   • {h_index_path}")
                else:
                    print("❌ No citation data available to calculate h-index.")
                
        elif choice == "4":
            # Option 4: Check Quota - STANDARDIZED
            print("\n⚠️  Checking API quota status...")
            check_quota_via_dummy_request()
            print("\n👉 Tip: Check quota before running Options 1-2 for large authors!")
            
        elif choice == "5":
            # Option 5: Exit - STANDARDIZED
            print("\n👋 Thanks for using BiblioMetric-Analyzer!")
            print("   💡 Next: streamlit run dashboard.py")
            break
            
        else:
            print("\n❌ Invalid option. Please choose 1-5.")


if __name__ == "__main__":
    main()
