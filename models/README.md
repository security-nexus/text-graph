# Local Models


Installation prerequsites:
https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_576.57_windows.exe
https://developer.download.nvidia.com/compute/cudnn/9.20.0/local_installers/cudnn_9.20.0_windows_x86_64.exe
https://github.com/microsoft/onnxruntime
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2


Default layout:

```text
models/
  embeddings/
    model.onnx
    vocab.txt
  classification/
    conversation-category.zip
    training-metrics.json
```

Train the local ML.NET classifier from the normalized export:

```powershell
dotnet run --project .\ChatGptExportGraphBuilder.csproj -- --input .\normalized-messages.json --train-classifier
```

Run the heuristic-only baseline:

```powershell
dotnet run --project .\ChatGptExportGraphBuilder.csproj -- --input .\normalized-messages.json --output .\out-heuristic --db .\heuristic.db --category-provider heuristic --embedding-provider hash
```

Run indexing with auto-discovered local models:

```powershell
dotnet run --project .\ChatGptExportGraphBuilder.csproj -- --input .\normalized-messages.json --output .\out-auto --db .\chatdump.db --category-provider hybrid --embedding-provider auto
```

Run heuristic analysis with Ollama cluster labeling:

```powershell
dotnet run --project .\ChatGptExportGraphBuilder.csproj -- --input .\normalized-messages.json --output .\out-cluster-labeled --db .\chatdump-cluster-labeled.db --category-provider hybrid --embedding-provider auto --analysis-provider heuristic --cluster-labeler ollama --ollama-model phi3:mini
```

Run ML.NET + ONNX with both comparison reports enabled:

```powershell
dotnet run --project .\ChatGptExportGraphBuilder.csproj -- --input .\normalized-messages.json --output .\out-compare-models --db .\chatdump-compare-models.db --category-provider hybrid --embedding-provider auto --compare-models --hash-similarity-threshold 0.72 --onnx-similarity-threshold 0.60
```

Run only the category comparison:

```powershell
dotnet run --project .\ChatGptExportGraphBuilder.csproj -- --input .\normalized-messages.json --output .\out-compare-categories --db .\chatdump-compare-categories.db --category-provider hybrid --embedding-provider auto --compare-categories
```

Run only the embedding comparison:

```powershell
dotnet run --project .\ChatGptExportGraphBuilder.csproj -- --input .\normalized-messages.json --output .\out-compare-embeddings --db .\chatdump-compare-embeddings.db --category-provider hybrid --embedding-provider auto --compare-embeddings --hash-similarity-threshold 0.72 --onnx-similarity-threshold 0.60
```

Force a specific provider combination:

```powershell
dotnet run --project .\ChatGptExportGraphBuilder.csproj -- --input .\normalized-messages.json --output .\out-mlnet-onnx --db .\mlnet-onnx.db --category-provider mlnet --embedding-provider onnx
```

Run token-aware chunking explicitly:

```powershell
dotnet run --project .\ChatGptExportGraphBuilder.csproj -- --input .\normalized-messages.json --output .\out-tokenizer --db .\chatdump-tokenizer.db --chunking-provider tokenizer --chunk-max-tokens 256 --chunk-overlap-tokens 32 --category-provider hybrid --embedding-provider auto
```

Run the named ML.NET profile:

```powershell
dotnet run --project .\ChatGptExportGraphBuilder.csproj -- --input .\normalized-messages.json --output .\out-profile-mlnet --db .\chatdump-profile-mlnet.db --execution-profile mlnet
```

Run the named ONNX profile with automatic CUDA/DirectML/CPU fallback:

```powershell
dotnet run --project .\ChatGptExportGraphBuilder.csproj -- --input .\normalized-messages.json --output .\out-profile-onnx --db .\chatdump-profile-onnx.db --execution-profile onnx
```

Run the ONNX profile with explicit CUDA on device 0:

```powershell
dotnet run --project .\ChatGptExportGraphBuilder.csproj -- --input .\normalized-messages.json --output .\out-profile-onnx-cuda --db .\chatdump-profile-onnx-cuda.db --execution-profile onnx --onnx-execution-provider cuda --cuda-device-id 0
```

Run the ONNX profile with explicit DirectML on adapter 0:

```powershell
dotnet build .\ChatGptExportGraphBuilder.csproj -p:OnnxRuntimeFlavor=DirectML
dotnet run --no-build --project .\ChatGptExportGraphBuilder.csproj -- --input .\normalized-messages.json --output .\out-profile-onnx-dml --db .\chatdump-profile-onnx-dml.db --execution-profile onnx --onnx-execution-provider directml --directml-device-id 0
```

Compare legacy WordPiece ONNX vs Microsoft tokenizer ONNX:

```powershell
dotnet run --project .\ChatGptExportGraphBuilder.csproj -- --input .\normalized-messages.json --output .\out-compare-onnx-tokenizers --db .\chatdump-compare-onnx-tokenizers.db --execution-profile onnx --compare-onnx-tokenizers
```

Run the full product matrix with clustering enabled:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-full-product.ps1
```

Rerun the matrix without rebuilding:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-full-product.ps1 -SkipBuild
```

Provider switches:

- `--execution-profile custom|heuristic|mlnet|hybrid|onnx`
- `--category-provider heuristic|mlnet|hybrid`
- `--chunking-provider auto|char|tokenizer|mltokenizer`
- `--chunk-max-tokens 256`
- `--chunk-overlap-tokens 32`
- `--onnx-tokenizer legacy|mltokenizer`
- `--onnx-execution-provider cpu|cuda|directml|auto`
- `--cuda-device-id 0`
- `--directml-device-id 0`
- `--embedding-provider hash|onnx|auto`
- `--analysis-provider heuristic|ollama`
- `--cluster-labeler heuristic|ollama|auto`
- `--compare-categories` writes `graph/category-comparison.json`
- `--compare-embeddings` writes `graph/similarity-hash.json`, `graph/similarity-onnx.json`, and `graph/embedding-comparison.json`
- `--compare-onnx-tokenizers` writes `graph/similarity-onnx-legacy.json`, `graph/similarity-onnx-mltokenizer.json`, and `graph/embedding-comparison-onnx-tokenizers.json`
- `--compare-models` enables both comparison reports together
- `--hash-similarity-threshold` and `--onnx-similarity-threshold` calibrate each embedding perspective separately
- ONNX defaults to `0.60` when no provider-specific threshold is supplied; hash defaults to the shared `0.72` baseline
- `--use-ollama` remains as a compatibility shortcut that enables both Ollama analysis and Ollama cluster labeling

Chunking notes:

- `--chunking-provider auto` uses `Microsoft.ML.Tokenizers` when `models/embeddings/vocab.txt` is present; otherwise it falls back to the legacy character chunker.
- Token-aware chunks are stored in SQLite `conversation_chunks.token_count`.
- Transcript and branch headers now include `ChunkCount` and `ChunkTokens`.
- Token-aware chunking currently uses the BERT-compatible vocab that also feeds the ONNX embedding path.

Execution profile notes:

- `heuristic` prefers heuristic categories, hash embeddings, and the character chunker.
- `mlnet` prefers ML.NET categories, hash embeddings, and `Microsoft.ML.Tokenizers` chunking.
- `hybrid` prefers ML.NET with heuristic fallback, hash embeddings, and `Microsoft.ML.Tokenizers` chunking.
- `onnx` prefers hybrid categories, ONNX embeddings, `Microsoft.ML.Tokenizers` chunking, the legacy ONNX tokenizer, and `auto` ONNX execution-provider selection unless you override it.
- Explicit provider flags still override the profile defaults.

ONNX runtime flavor notes:

- The native ONNX runtime package is selected at build time with `-p:OnnxRuntimeFlavor=Gpu|DirectML`.
- The project now defaults to `Gpu`, which brings in the CUDA execution provider and CPU fallback.
- Use `-p:OnnxRuntimeFlavor=DirectML` when you want a DirectML-native build instead.
- A single build should use one native ONNX runtime flavor at a time; the packages ship different native `onnxruntime.dll` payloads.

Execution provider notes:

- `--onnx-execution-provider auto` tries CUDA first, then DirectML, then CPU.
- `--onnx-execution-provider cuda` requires the GPU native build and working CUDA runtime DLLs on the process path.
- `--onnx-execution-provider directml` requires the DirectML-native build and a supported Windows adapter.
- `--cuda-device-id` selects the CUDA device index passed to `AppendExecutionProvider_CUDA`.
- `--directml-device-id` selects the GPU adapter index passed to `AppendExecutionProvider_DML`.
- The active embedding provider description shows whether the session used `cuda:<id>`, `directml:<id>`, `cpu`, or `cpu-fallback`.

Full product matrix notes:

- `run-full-product.ps1` runs `hash`, `mlnet`, `onnx`, and `hybrid` profiles against the same normalized export.
- The script forces tokenizer-based chunking for all profiles so the comparison is not distorted by different chunkers.
- Clustering is enabled in every run with `--cluster-labeler heuristic`.
- The ONNX and hybrid paths use `--onnx-execution-provider cuda`, so ONNX embeddings and ONNX comparison passes use the CUDA provider when available.
- The script writes per-profile logs under `out-full-product\summary\`.

Generated graph artifacts to inspect first:

- `graph/corpus-index.json`
- `graph/conversation-graph.json`
- `graph/category-comparison.json`
- `graph/embedding-comparison.json`
- `graph/embedding-comparison-onnx-tokenizers.json`
- `graph/perspective-summary.json`
- `graph/perspective-summary.md`

Navigator and extraction artifacts:

- `graph/navigator.html`
- `graph/insights.md`
- `graph/insights.json`
- `graph/conversation-catalog.csv`
- `graph/bridge-conversations.csv`
- `graph/cluster-overview.csv`
- `graph/category-community-overview.csv`

Full product summary artifacts:

- `out-full-product\summary\full-product-report.json`
- `out-full-product\summary\full-product-report.md`
- `out-full-product\summary\full-product-disagreements.csv`
- `out-full-product\summary\full-product-hotpaths.csv`
- `out-full-product\summary\hash.log`
- `out-full-product\summary\mlnet.log`
- `out-full-product\summary\onnx.log`
- `out-full-product\summary\hybrid.log`

If `models/embeddings/model.onnx` and `models/embeddings/vocab.txt` exist, the ONNX embedding provider is used automatically when `--embedding-provider auto`.
If `models/classification/conversation-category.zip` exists, the ML.NET classifier is used automatically when `--category-provider hybrid` or `--category-provider mlnet`.
If `--cluster-labeler heuristic` is used, clusters still exist, but their labels come from deterministic fallback logic instead of Ollama.
