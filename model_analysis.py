# -*- coding: utf-8 -*-
"""
model_analysis.py
-----------------
Comprehensive analysis of the SuperalloyDigger text-mining model.

Run from the repository root:
    python model_analysis.py

The script requires no additional dependencies beyond the Python standard
library and produces a structured report covering:
  1. Pipeline architecture overview
  2. NER dictionary / pattern statistics
  3. Pattern-matching demonstration on the bundled sample corpus
  4. Source-code metrics (lines of code per module)
"""

import ast
import configparser
import os
import re
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DICT_PATH = os.path.join(REPO_ROOT, "pipeline", "dictionary.ini")
HEA_DICT_PATH = os.path.join(REPO_ROOT, "HEA_use_case", "dictionary_HEAs.ini")
SAMPLE_TXT = os.path.join(
    REPO_ROOT, "input_txt", "10.1016-j.msea.2014.09.074.txt"
)

PROPERTIES = ["solvus", "solidus", "liquidus", "density"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def separator(char="=", width=72):
    print(char * width)


def heading(title):
    separator()
    print(f"  {title}")
    separator()


def subheading(title):
    print()
    print(f"── {title}")
    print("  " + "─" * (len(title) + 2))


def load_dict_section(path):
    """Load a dictionary .ini file and return a plain dict of evaluated values."""
    cp = configparser.RawConfigParser()
    cp.read(path, "UTF-8")
    result = {}
    for key in cp.options("DICTIONARY"):
        try:
            result[key] = ast.literal_eval(cp.get("DICTIONARY", key))
        except Exception:
            result[key] = cp.get("DICTIONARY", key)
    return result


def count_loc(filepath):
    """Count non-empty, non-comment lines in a Python file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        code = [
            l for l in lines
            if l.strip() and not l.strip().startswith("#")
        ]
        return len(lines), len(code)
    except OSError:
        return 0, 0


# ---------------------------------------------------------------------------
# Section 1 – Pipeline Architecture
# ---------------------------------------------------------------------------

ARCHITECTURE = [
    ("Article Download", [
        "XML/TXT batch download via Elsevier Scopus & ScienceDirect APIs",
        "HTML scraping for non-Elsevier publishers via CrossRef DOI lookup",
    ]),
    ("Corpus Pre-processing", [
        "Text normalisation (Unicode cleanup, abbreviation expansion)",
        "Alloy-name token merging (multi-word → hyphenated tokens)",
        "Temperature/unit canonicalisation (e.g. '1300 C' → '1300°C')",
    ]),
    ("Sentence Classification", [
        "Dictionary-driven keyword filter",
        "Keeps sentences that mention target properties (solvus, density, …)",
    ]),
    ("Named Entity Recognition (Rule-based)", [
        "Alloy names: 6 compiled regex patterns covering notation styles",
        "Property specifiers: keyword lists per property",
        "Property values: unit-anchored numeric regex per property",
        "Chemical elements: symbol list + full-name variants",
    ]),
    ("Relation Extraction – Text", [
        "Algorithm 1 (Relation_extraciton_orig): position/distance scoring",
        "Algorithm 2 (Relation_extraciton_dp): NLTK dependency parsing",
        "Output: (alloy_name, property_specifier, value) triples",
    ]),
    ("Table Parsing", [
        "XML tables: BeautifulSoup parser for Elsevier XML",
        "HTML tables: web-scraped via DOI → publisher-specific extraction",
        "Header detection, column/row analysis, cell normalisation",
    ]),
    ("Dependency Parser", [
        "Resolves linkage between composition and property fragments",
        "Handles multi-sentence composition+property dependencies",
    ]),
    ("Output Compilation", [
        "Excel (.xls/.xlsx) via openpyxl / xlwt",
        "MongoDB document structure (table.py entity classes)",
        "Fields: DOI, alloy_name, element, fraction, property, value, unit",
    ]),
]


def section_architecture():
    heading("1. PIPELINE ARCHITECTURE")
    for stage, details in ARCHITECTURE:
        print(f"\n  [{stage}]")
        for detail in details:
            print(f"    • {detail}")


# ---------------------------------------------------------------------------
# Section 2 – NER / Dictionary Statistics
# ---------------------------------------------------------------------------

def section_ner_stats():
    heading("2. NER DICTIONARY STATISTICS  (pipeline/dictionary.ini)")

    if not os.path.exists(DICT_PATH):
        print(f"  ✗ File not found: {DICT_PATH}")
        return

    d = load_dict_section(DICT_PATH)

    subheading("2a. Text Pre-processing Rules")
    print(f"  replace_word entries       : {len(d.get('replace_word', {}))}")
    print(f"  alloy_to_replace patterns  : {len(d.get('alloy_to_replace', {}))}")
    print(f"  paras_to_replace patterns  : {len(d.get('paras_to_replace', {}))}")

    subheading("2b. Alloy-Name Recognition")
    awt = d.get("alloy_writing_type", [])
    abt = d.get("alloy_blank_type", [])
    print(f"  alloy_writing_type patterns: {len(awt)}")
    for i, pat in enumerate(awt, 1):
        print(f"    [{i}] {pat}")
    print(f"  alloy_blank_type patterns  : {len(abt)}")
    for i, pat in enumerate(abt, 1):
        print(f"    [{i}] {pat}")

    subheading("2c. Property Specifier Keywords")
    pwt = d.get("prop_writing_type", {})
    for prop, kws in pwt.items():
        print(f"  {prop:10s}: {kws}")

    subheading("2d. Property Value Regex (text)")
    vwt = d.get("value_wt", {})
    for prop, patterns in vwt.items():
        print(f"  {prop}  ({len(patterns)} patterns)")
        for pat in patterns:
            print(f"    {pat}")

    subheading("2e. Chemical Element List")
    ele = d.get("ele_list", [])
    print(f"  Elements tracked ({len(ele)}): {', '.join(ele)}")

    subheading("2f. Table Extraction Patterns")
    print(f"  table_e_pattern (element):  {d.get('table_e_pattern', '')[:80]}…")
    print(f"  table_ratio_pattern:        {d.get('table_ratio_pattern', '')[:80]}…")
    tup = d.get("table_units", [])
    print(f"  table_units ({len(tup)}):         {tup}")
    tnp = d.get("table_number_pattern", {})
    for prop, pat in tnp.items():
        print(f"  table_number_pattern[{prop}]: {pat}")

    subheading("2g. Ambiguity / Noise Suppression")
    op = d.get("other_phase", {})
    oq = d.get("other_quality", {})
    for prop in PROPERTIES:
        phases = op.get(prop, [])
        qualities = oq.get(prop, [])
        print(f"  {prop}  – excluded phases: {len(phases)}  "
              f"| excluded qualifiers: {len(qualities)}")

    subheading("2h. Element Full-name → Symbol Mappings")
    e2a = d.get("ele_to_abr", {})
    print(f"  Mappings: {len(e2a)}")
    for full, sym in list(e2a.items())[:12]:
        print(f"    {full:<14} → {sym}")
    if len(e2a) > 12:
        print(f"    … and {len(e2a) - 12} more")

    # --- HEA extension ---
    if os.path.exists(HEA_DICT_PATH):
        print()
        print("  HEA extension (HEA_use_case/dictionary_HEAs.ini)")
        hea = load_dict_section(HEA_DICT_PATH)
        hea_pwt = hea.get("prop_writing_type", {})
        for prop, kws in hea_pwt.items():
            print(f"    {prop:10s}: {kws}")
        hea_ele = hea.get("ele_list", [])
        print(f"    HEA element list ({len(hea_ele)}): {', '.join(hea_ele)}")


# ---------------------------------------------------------------------------
# Section 3 – Pattern Matching Demo on Sample Corpus
# ---------------------------------------------------------------------------

def match_alloy_names(text, patterns):
    """Return unique alloy-name candidates found by the regex patterns."""
    found = set()
    # word-level scan
    words = re.split(r'\s+', text)
    for word in words:
        for pat in patterns:
            if re.search(pat, word):
                found.add(word)
                break
    # blank-type (context) patterns
    return sorted(found)


def match_property_values(text, value_patterns):
    """Return matched numeric tokens for each property."""
    results = {}
    words = re.split(r'\s+', text)
    for prop, patterns in value_patterns.items():
        hits = set()
        for word in words:
            for pat in patterns:
                if re.search(pat, word):
                    hits.add(word)
                    break
        results[prop] = sorted(hits)
    return results


def section_pattern_demo():
    heading("3. PATTERN-MATCHING DEMO  (sample corpus)")

    if not os.path.exists(SAMPLE_TXT):
        print(f"  ✗ Sample file not found: {SAMPLE_TXT}")
        return
    if not os.path.exists(DICT_PATH):
        print(f"  ✗ Dictionary not found: {DICT_PATH}")
        return

    with open(SAMPLE_TXT, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    # Basic normalisation matching what pre_processor.py does
    d = load_dict_section(DICT_PATH)

    text = raw
    for old, new in d.get("replace_word", {}).items():
        text = text.replace(old, new)

    subheading("3a. Corpus statistics")
    all_sentences = re.split(r'(?<=[.!?])\s+', text)
    print(f"  Characters            : {len(text):,}")
    print(f"  Approximate sentences : {len(all_sentences):,}")

    # Sentences containing each property keyword
    print()
    print("  Sentences mentioning each property keyword:")
    prop_kws = d.get("prop_writing_type", {})
    for prop, kws in prop_kws.items():
        hits = [s for s in all_sentences if any(kw in s for kw in kws)]
        print(f"    {prop:10s} ({kws[0]:20s} …): {len(hits):3d} sentence(s)")

    subheading("3b. Alloy-name candidates extracted")
    awt = d.get("alloy_writing_type", [])
    alloys = match_alloy_names(text, awt)
    # Filter obviously non-alloy tokens (very short, all-lowercase words)
    alloys_filtered = [
        a for a in alloys
        if len(a) >= 2 and re.search(r'[A-Z]', a)
    ]
    print(f"  Unique candidates: {len(alloys_filtered)}")
    for a in alloys_filtered[:30]:
        print(f"    {a}")
    if len(alloys_filtered) > 30:
        print(f"    … and {len(alloys_filtered) - 30} more")

    subheading("3c. Property-value candidates extracted")
    vwt = d.get("value_wt", {})
    pv = match_property_values(text, vwt)
    for prop, hits in pv.items():
        print(f"  {prop:10s}: {len(hits):3d} hit(s)  {hits[:10]}")

    subheading("3d. Chemical element mentions in corpus")
    ele = d.get("ele_list", [])
    words_in_text = set(re.split(r'\W+', text))
    found_ele = [e for e in ele if e in words_in_text]
    print(f"  Elements present: {found_ele}")

    subheading("3e. Sample target sentences (solvus / solidus / liquidus / density)")
    for prop in PROPERTIES:
        kws = prop_kws.get(prop, [prop])
        hits = [s for s in all_sentences if any(kw in s for kw in kws)]
        if hits:
            print(f"\n  ─ {prop} (first match) ─")
            print(f"  {hits[0][:300].strip()}")
        else:
            print(f"\n  ─ {prop}: no match in this file ─")


# ---------------------------------------------------------------------------
# Section 4 – Source-Code Metrics
# ---------------------------------------------------------------------------

MODULES = {
    "pipeline": [
        "main.py",
        "class_modified.py",
        "text_with_table.py",
        "Relation_extraciton_dp.py",
        "Relation_extraciton_orig.py",
        "output_modified_triple.py",
        "table_info_html.py",
        "html_parser.py",
        "get_tifo_from_html.py",
        "Phrase_parse.py",
        "T_pre_processor.py",
        "pre_processor.py",
        "sentence_positioner.py",
        "get_full_text.py",
        "other_journals.py",
        "get_all_attributes.py",
        "dictionary.py",
        "table.py",
        "file_io.py",
        "log_wp.py",
    ],
    "text_extractor": [
        "Relation_extraciton.py",
        "Phrase_parse.py",
        "T_pre_processor.py",
        "pre_processor.py",
        "sentence_positioner.py",
        "get_full_text.py",
        "get_all_attributes.py",
        "dictionary.py",
        "file_io.py",
        "log_wp.py",
    ],
    "word embedding": [
        "load_word2vec.py",
        "main.py",
        "__init__.py",
    ],
    "table_extractor/elsevier_xml": [
        "class_modified.py",
        "dictionary.py",
        "table.py",
        "log_wp.py",
    ],
}


def section_code_metrics():
    heading("4. SOURCE-CODE METRICS")

    fmt = "  {:<40s} {:>8s} {:>8s}"
    print(fmt.format("File", "Total", "Code"))
    print("  " + "-" * 58)

    grand_total, grand_code = 0, 0
    for module, files in MODULES.items():
        subtotal, subcode = 0, 0
        print(f"\n  [{module}]")
        for fname in files:
            fpath = os.path.join(REPO_ROOT, module, fname)
            total, code = count_loc(fpath)
            subtotal += total
            subcode += code
            if total:
                print(fmt.format(f"  {fname}", str(total), str(code)))
        print(fmt.format(f"  Subtotal", str(subtotal), str(subcode)))
        grand_total += subtotal
        grand_code += subcode

    print()
    print(fmt.format("GRAND TOTAL", str(grand_total), str(grand_code)))

    subheading("Dependency summary (requirements.txt)")
    req_path = os.path.join(REPO_ROOT, "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path) as fh:
            reqs = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
        cats = {
            "NLP / ML": ["nltk", "gensim", "torch", "ChemDataExtractor"],
            "Data":     ["pandas", "numpy", "scipy", "openpyxl", "xlrd", "xlwt"],
            "Web":      ["beautifulsoup4", "requests"],
            "Database": ["pymongo"],
            "Utility":  ["Unidecode", "matplotlib", "seaborn"],
        }
        for cat, pkgs in cats.items():
            matched = [r for r in reqs if any(r.lower().startswith(p.lower()) for p in pkgs)]
            print(f"  {cat:12s}: {', '.join(matched)}")
    else:
        print(f"  requirements.txt not found at {req_path}")


# ---------------------------------------------------------------------------
# Section 5 – Model Design Notes
# ---------------------------------------------------------------------------

def section_design_notes():
    heading("5. MODEL DESIGN NOTES")

    notes = [
        ("Strengths",
         [
             "Fully rule-based NER — zero labelled training data required",
             "Pre-trained Word2Vec on domain corpus (≈9,000 superalloy articles)",
             "Multi-format input: XML (Elsevier), HTML (other publishers), plain text",
             "Configurable via .ini files — easily extended to new alloy families "
             "(see HEA_use_case/)",
             "Dependency-parsing fallback for multi-sentence composition–property links",
         ]),
        ("Limitations / Known Gaps",
         [
             "Rule coverage is English-only and domain-specific",
             "Regex patterns may miss non-standard unit spellings or formatting",
             "Word2Vec model binary not included in repo — must be trained separately",
             "No probabilistic confidence score on extracted triples",
             "Windows-style path separators (\\) hard-coded in several files "
             "(text_with_table.py mkdir, etc.) may break on Linux/macOS",
             "dictionary.ini table_units had two implicit string-concatenation bugs "
             "('kPa''Pa' → 'kPaPa', 'gr''mg' → 'grmg') — fixed in this PR",
         ]),
        ("Extensibility",
         [
             "Add new property: extend dictionary.ini sections "
             "(prop_writing_type, value_wt, other_quality, …)",
             "Add new publisher format: implement new class in html_parser.py "
             "or table_info_html.py",
             "New alloy family: provide a custom dictionary_<family>.ini "
             "(cf. HEA_use_case/)",
         ]),
    ]

    for title, points in notes:
        subheading(title)
        for pt in points:
            print(f"  • {pt}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    separator("*")
    print("  SuperalloyDigger – Model Analysis Report")
    print(f"  Repository root: {REPO_ROOT}")
    separator("*")

    section_architecture()
    section_ner_stats()
    section_pattern_demo()
    section_code_metrics()
    section_design_notes()

    separator("*")
    print("  Analysis complete.")
    separator("*")


if __name__ == "__main__":
    main()
