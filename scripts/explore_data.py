import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "diagrams")

# ── Load datasets ──────────────────────────────────────────────

ieee = pd.read_csv(os.path.join(DATA_DIR, "IEEE.csv"))
wos = pd.read_excel(os.path.join(DATA_DIR, "web_of_science.xls"))

# Normalize column names to combine both datasets
ieee_norm = ieee.rename(columns={
    "Document Title": "title",
    "Authors": "authors",
    "Abstract": "abstract",
    "Author Keywords": "keywords",
    "Publication Year": "year",
    "Article Citation Count": "citations",
    "Publication Title": "venue",
    "Document Identifier": "doc_type",
}).assign(source="IEEE")

wos_norm = wos.rename(columns={
    "Article Title": "title",
    "Authors": "authors",
    "Abstract": "abstract",
    "Author Keywords": "keywords",
    "Publication Year": "year",
    "Times Cited, All Databases": "citations",
    "Source Title": "venue",
    "Document Type": "doc_type",
}).assign(source="WoS")

cols = ["title", "abstract", "keywords", "year", "citations", "venue", "doc_type", "source"]
df = pd.concat([ieee_norm[cols], wos_norm[cols]], ignore_index=True)
df["citations"] = pd.to_numeric(df["citations"], errors="coerce").fillna(0).astype(int)
df["year"] = pd.to_numeric(df["year"], errors="coerce")

print(f"Combined dataset: {len(df)} papers ({df['year'].min():.0f}–{df['year'].max():.0f})")
print(f"  IEEE: {len(ieee_norm)}, Web of Science: {len(wos_norm)}")
print(f"  Papers with abstracts: {df['abstract'].notna().sum()}")
print(f"  Papers with keywords: {df['keywords'].notna().sum()}")
print()

# ── 1. Publication trend by year ───────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
year_counts = df.groupby(["year", "source"]).size().unstack(fill_value=0)
year_counts.plot(kind="bar", stacked=True, ax=ax, color=["#1f77b4", "#ff7f0e"])
ax.set_title("Publications per Year by Source")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Papers")
ax.legend(title="Source")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "publications_per_year.png"), dpi=150)
print("Saved: publications_per_year.png")

# ── 2. Top cited papers ───────────────────────────────────────

print("\n── Top 10 Most Cited Papers ──")
top_cited = df.nlargest(10, "citations")[["title", "year", "citations", "source"]]
for i, row in top_cited.iterrows():
    print(f"  [{row['source']}] ({row['year']:.0f}, {row['citations']} cites) {row['title'][:90]}")

# ── 3. Document type distribution ─────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, src in zip(axes, ["IEEE", "WoS"]):
    subset = df[df["source"] == src]
    counts = subset["doc_type"].dropna().value_counts().head(6)
    if counts.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        counts.plot(kind="barh", ax=ax, color="#2ca02c")
    ax.set_title(f"{src} — Document Types")
    ax.set_xlabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "document_types.png"), dpi=150)
print("\nSaved: document_types.png")

# ── 4. Top venues ─────────────────────────────────────────────

print("\n── Top 10 Venues ──")
top_venues = df["venue"].value_counts().head(10)
for venue, count in top_venues.items():
    print(f"  {count:4d}  {venue[:80]}")

# ── 5. Keyword frequency ──────────────────────────────────────

all_keywords = df["keywords"].dropna().str.lower().str.split(r"\s*;\s*").explode().str.strip()
all_keywords = all_keywords[all_keywords != ""]
kw_counts = all_keywords.value_counts().head(20)

fig, ax = plt.subplots(figsize=(10, 6))
kw_counts.plot(kind="barh", ax=ax, color="#d62728")
ax.set_title("Top 20 Author Keywords")
ax.set_xlabel("Frequency")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_keywords.png"), dpi=150)
print("\nSaved: top_keywords.png")

# ── 6. TF-IDF on abstracts ────────────────────────────────────

abstracts = df["abstract"].dropna().tolist()
print(f"\n── TF-IDF Analysis on {len(abstracts)} abstracts ──")

tfidf = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(abstracts)

feature_names = tfidf.get_feature_names_out()
mean_scores = tfidf_matrix.mean(axis=0).A1
top_indices = mean_scores.argsort()[::-1][:30]

print("\nTop 30 TF-IDF terms (unigrams + bigrams):")
for i, idx in enumerate(top_indices):
    print(f"  {i+1:2d}. {feature_names[idx]:<35s} {mean_scores[idx]:.4f}")

fig, ax = plt.subplots(figsize=(10, 8))
top_terms = [feature_names[i] for i in top_indices]
top_scores = [mean_scores[i] for i in top_indices]
ax.barh(top_terms[::-1], top_scores[::-1], color="#9467bd")
ax.set_title("Top 30 TF-IDF Terms Across All Abstracts")
ax.set_xlabel("Mean TF-IDF Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tfidf_top_terms.png"), dpi=150)
print("\nSaved: tfidf_top_terms.png")

# ── 7. Citation stats ─────────────────────────────────────────

print("\n── Citation Statistics ──")
for src in ["IEEE", "WoS"]:
    subset = df[df["source"] == src]
    cited = subset[subset["citations"] > 0]
    print(f"  {src}: median={cited['citations'].median():.0f}, "
          f"mean={cited['citations'].mean():.1f}, "
          f"max={cited['citations'].max()}, "
          f"papers with citations={len(cited)}/{len(subset)}")

# ── 8. Year vs citation scatter ───────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
for src, color in [("IEEE", "#1f77b4"), ("WoS", "#ff7f0e")]:
    subset = df[(df["source"] == src) & (df["citations"] > 0)]
    ax.scatter(subset["year"], subset["citations"], alpha=0.4, label=src, color=color, s=20)
ax.set_title("Citations vs Publication Year")
ax.set_xlabel("Year")
ax.set_ylabel("Citations")
ax.set_yscale("log")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "citations_vs_year.png"), dpi=150)
print("\nSaved: citations_vs_year.png")

print("\n── Done. All charts saved to diagrams/ ──")
