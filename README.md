# text-graph

`text-graph` turns a ChatGPT data export into an offline corpus you can browse, search, cluster, and analyze locally.

It rebuilds branch-aware conversations, writes readable transcripts, persists a SQLite index, emits graph JSON, and produces a navigator UI with topic clusters, category communities, bridge conversations, keyword hotspots, and model-comparison views.

The stack is intentionally hybrid:

- A deterministic heuristic layer extracts baseline categories, topics, keywords, and graph edges.
- A local ML.NET classifier can refine categorization.
- Local hash or ONNX embeddings can drive semantic similarity and clustering.
- Higher-order corpus analysis such as communities, clusters, topic labels, keyword co-occurrence, and category hierarchy is then built deterministically on top of those signals.

## Get Your ChatGPT Export

Before using this tool, export your ChatGPT data from `Settings -> Data Controls -> Export Data`, then download the ZIP from the email OpenAI sends you and extract the included `conversations.json`.

Useful details for users:

- On the web, the current path is `Profile icon -> Settings -> Data Controls -> Export Data -> Confirm export`.
- On iOS and Android, the path is `Settings -> Data Controls -> Export Data`.
- You may need access to the email inbox or phone number tied to the account to verify ownership.
- OpenAI sends the download link to the account email address, and the link expires after 24 hours.
- Large exports can take some time to generate.
- Organization-managed accounts may have export controls restricted or managed by an administrator.

## What This Repo Produces

Given a normalized message file, the tool can generate:

- branch-aware text transcripts
- corpus and conversation graph JSON
- a SQLite database with conversations, chunks, categories, topics, keywords, embeddings, clusters, communities, and co-occurrence tables
- `graph/navigator.html` for interactive browsing
- `graph/insights.md` and CSV summaries for clusters, communities, and bridge conversations
- optional comparison reports for heuristic vs ML.NET categorization and hash vs ONNX embeddings

## Quick Start

### 1. Prerequisites

- Windows and PowerShell are the intended environment for the included scripts
- .NET 9 SDK

Optional local-model support is documented in [models/README.md](models/README.md).

### 2. Export and extract your ChatGPT archive

After OpenAI emails the ZIP, extract it to a local folder and locate the exported conversation data.

### 3. Convert the export into `normalized-messages.json`

The console app expects a normalized flat message file, not the raw ChatGPT export directly.

This repo includes [`convert-chat.ps1`](convert-chat.ps1) as a starter conversion script. Today it is repo-local and expects you to adjust it for your export:

- set `$root` near the bottom of the script to your extracted export folder
- if your export contains a single `conversations.json` instead of files matching `conversations-*.json`, update the file filter accordingly

The script inventories the extracted files and writes `normalized-messages.json`.

### 4. Build the project

```powershell
dotnet build .\ChatGptExportGraphBuilder.csproj
```

### 5. Run the baseline pipeline

This path works without any local model files:

```powershell
dotnet run --project .\ChatGptExportGraphBuilder.csproj -- --input .\normalized-messages.json --output .\out --db .\out\chatdump.db --execution-profile heuristic
```

### 6. Open the generated artifacts

Start with:

- `out\graph\navigator.html`
- `out\graph\insights.md`
- `out\chatdump.db`

## How The Analysis Works

### Heuristic baseline

The heuristic layer builds a weighted corpus from titles, user messages, assistant replies, code, and execution output. It assigns a baseline category from a fixed term map, extracts weighted keywords and phrases, derives topic candidates, chunks text for indexing, and emits graph edges for categories, topics, and keywords.

### Local categorization

The ML.NET path trains a multiclass text classifier from the heuristic-labeled corpus and can run in `mlnet` or `hybrid` mode. In hybrid mode, ML.NET is preferred when available, but heuristic categories remain as a second opinion and alternate facets.

### Local similarity and clustering

The embedding layer uses either:

- hashing embeddings for a lightweight, dependency-free fallback
- ONNX embeddings for stronger local semantic similarity when model files are present

Pairwise cosine similarity builds a `similar_to` graph. Connected components in that graph become topic clusters. Cluster labels are deterministic by default, with optional Ollama labeling if you explicitly enable it.

### Grouping and community analysis

Each conversation can carry primary, secondary, and tertiary category facets. Those facet signatures are grouped and merged with weighted Jaccard similarity into category communities. This is deterministic corpus grouping built on top of the selected category outputs.

### Topic and keyword analysis

Keywords are weighted from conversation titles, structured text, plain text, code, and tool output, then filtered for noise and overlap. Topics are built from the strongest title phrases, keywords, and grounded category terms. Across the whole corpus, the tool also computes keyword co-occurrence hotspots and category hierarchy links.

## Execution Modes

The project supports several named profiles:

- `heuristic`: heuristic categories, hash embeddings, character chunking
- `mlnet`: ML.NET categories, hash embeddings, tokenizer-aware chunking
- `hybrid`: ML.NET with heuristic fallback, hash embeddings, tokenizer-aware chunking
- `onnx`: hybrid categories with ONNX embeddings and tokenizer-aware chunking

Explicit provider switches can override the profile defaults. See [models/README.md](models/README.md) for the full matrix, GPU notes, tokenizer options, and comparison commands.

## Output Artifacts Worth Opening First

- `graph/navigator.html`
- `graph/insights.md`
- `graph/conversation-catalog.csv`
- `graph/bridge-conversations.csv`
- `graph/cluster-overview.csv`
- `graph/category-community-overview.csv`
- `graph/corpus-index.json`
- `graph/conversation-graph.json`

If you enable comparison flags, also inspect:

- `graph/category-comparison.json`
- `graph/embedding-comparison.json`
- `graph/perspective-summary.json`

## Notes

- Without local model files, the tool still works with heuristic categorization and hash embeddings.
- If `models/classification/conversation-category.zip` exists, the ML.NET classifier can be used automatically.
- If `models/embeddings/model.onnx` and `models/embeddings/vocab.txt` exist, ONNX embeddings can be used automatically.
- The PowerShell scripts and examples are Windows-oriented, but the core app is a .NET console pipeline.
- Detailed local model setup, CUDA/DirectML notes, tokenizer behavior, and full comparison commands live in [models/README.md](models/README.md).
