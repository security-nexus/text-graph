using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Data.Sqlite;

// ChatGptExportGraphBuilder.cs
// .NET 9 console program
// Purpose:
// 1) Load flat normalized ChatGPT export JSON (array of messages)
// 2) Reconstruct branch-aware conversations
// 3) Emit readable text transcripts
// 4) Persist conversations/messages/topics/keywords/graph edges to SQLite
// 5) Emit graph JSON for downstream visualization
//
// Build:
//   dotnet new console -n ChatDumpGraph -f net9.0
//   copy this file over Program.cs
//   dotnet add package Microsoft.Data.Sqlite
//   dotnet build
//
// Run:
//   dotnet run -- --input normalized-messages.json --output out --db out/chatdump.db
//
// Optional local model:
//   set CHATDUMP_OLLAMA_MODEL=phi3:mini
//   set CHATDUMP_OLLAMA_URL=http://localhost:11434
//   dotnet run -- --input normalized-messages.json --output out --db out/chatdump.db --use-ollama

var arguments = AppArgs.Parse(args);
var conversationChunker = ConversationChunkerFactory.Create(arguments);
if (arguments.TrainClassifierOnly)
{
    var flatTrainingMessages = await FlatMessageLoader.LoadAsync(arguments.InputPath);
    var trainingConversationSet = ConversationRebuilder.Rebuild(flatTrainingMessages);

    Console.WriteLine($"Loaded {flatTrainingMessages.Count:N0} flat messages.");
    Console.WriteLine($"Rebuilt {trainingConversationSet.Conversations.Count:N0} conversations.");

    var heuristicAnalyses = await ConversationAnalysisPipeline.AnalyzeAsync(trainingConversationSet, new HeuristicConversationAnalyzer(conversationChunker));
    var trainingResult = LocalModelTrainer.TrainClassifier(heuristicAnalyses, arguments);

    Console.WriteLine("Classifier training complete.");
    Console.WriteLine($"Model:   {trainingResult.ModelPath}");
    Console.WriteLine($"Metrics: {trainingResult.MetricsPath}");
    Console.WriteLine($"Macro:   {trainingResult.MacroAccuracy.ToString("0.###", CultureInfo.InvariantCulture)}");
    Console.WriteLine($"Micro:   {trainingResult.MicroAccuracy.ToString("0.###", CultureInfo.InvariantCulture)}");
    return;
}

Directory.CreateDirectory(arguments.OutputDirectory);
Directory.CreateDirectory(Path.Combine(arguments.OutputDirectory, "transcripts"));
Directory.CreateDirectory(Path.Combine(arguments.OutputDirectory, "branches"));
Directory.CreateDirectory(Path.Combine(arguments.OutputDirectory, "graph"));

var flatMessages = await FlatMessageLoader.LoadAsync(arguments.InputPath);
var conversationSet = ConversationRebuilder.Rebuild(flatMessages);

IConversationAnalyzer analyzer = arguments.UseOllamaAnalysis
    ? new OllamaConversationAnalyzer(arguments.OllamaBaseUrl, arguments.OllamaModel, conversationChunker)
    : new HeuristicConversationAnalyzer(conversationChunker);

Console.WriteLine($"Loaded {flatMessages.Count:N0} flat messages.");
Console.WriteLine($"Rebuilt {conversationSet.Conversations.Count:N0} conversations.");
Console.WriteLine($"Execution profile: {arguments.ExecutionProfileMode}");
Console.WriteLine($"Conversation analyzer: {(arguments.UseOllamaAnalysis ? $"ollama:{arguments.OllamaModel}" : "heuristic")}");
Console.WriteLine($"Cluster labeler mode: {arguments.ClusterLabelerMode}");
Console.WriteLine($"Chunking provider: {conversationChunker.Description}");

var analyzed = await ConversationAnalysisPipeline.AnalyzeAsync(conversationSet, analyzer);
var indexed = await CorpusIndexingPipeline.BuildAsync(analyzed, arguments);
TranscriptWriter.Write(indexed.Conversations, arguments.OutputDirectory);
CorpusGraphWriter.Write(indexed, Path.Combine(arguments.OutputDirectory, "graph"));
await CategoryComparisonWriter.WriteIfEnabledAsync(analyzed, arguments, Path.Combine(arguments.OutputDirectory, "graph"));
await EmbeddingComparisonWriter.WriteIfEnabledAsync(analyzed, arguments, Path.Combine(arguments.OutputDirectory, "graph"));
await PerspectiveSummaryWriter.WriteIfAvailableAsync(Path.Combine(arguments.OutputDirectory, "graph"));
CorpusDatabaseWriter.Write(indexed, arguments.DatabasePath);
await NavigatorWriter.WriteAsync(indexed, arguments.OutputDirectory, Path.Combine(arguments.OutputDirectory, "graph"));

Console.WriteLine("Done.");
Console.WriteLine($"Transcripts: {Path.Combine(arguments.OutputDirectory, "transcripts")}");
Console.WriteLine($"Branches:    {Path.Combine(arguments.OutputDirectory, "branches")}");
Console.WriteLine($"Graph:       {Path.Combine(arguments.OutputDirectory, "graph")}");
Console.WriteLine($"SQLite DB:   {arguments.DatabasePath}");

// -----------------------------
// Argument parsing
// -----------------------------

sealed record AppArgs(
    string InputPath,
    string OutputDirectory,
    string DatabasePath,
    string ExecutionProfileMode,
    string ConversationAnalyzerMode,
    string ClusterLabelerMode,
    string OllamaBaseUrl,
    string OllamaModel,
    string ModelsDirectory,
    bool TrainClassifierOnly,
    string ClassifierMetricsPath,
    string ChunkingProviderMode,
    int ChunkMaxTokens,
    int ChunkOverlapTokens,
    string OnnxTokenizerMode,
    string OnnxExecutionProviderMode,
    int CudaDeviceId,
    int DirectMlDeviceId,
    string? EmbeddingModelPath,
    string? EmbeddingVocabularyPath,
    string? MlNetModelPath,
    string CategoryProviderMode,
    string EmbeddingProviderMode,
    int EmbeddingMaxTokens,
    int HashEmbeddingDimensions,
    double SimilarityThreshold,
    bool HasExplicitSimilarityThreshold,
    double? HashSimilarityThreshold,
    double? OnnxSimilarityThreshold,
    int MaxSimilarNeighbors,
    int KeywordCooccurrenceKeywords,
    int KeywordCooccurrenceMinimum,
    bool CompareCategories,
    bool CompareEmbeddings,
    bool CompareOnnxTokenizers)
{
    public bool UseOllamaAnalysis => string.Equals(ConversationAnalyzerMode, "ollama", StringComparison.OrdinalIgnoreCase);

    public static AppArgs Parse(string[] args)
    {
        string? input = null;
        string output = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "out"));
        string? db = null;
        string executionProfileMode = "custom";
        string conversationAnalyzerMode = "heuristic";
        string clusterLabelerMode = "heuristic";
        string ollamaBaseUrl = Environment.GetEnvironmentVariable("CHATDUMP_OLLAMA_URL") ?? "http://localhost:11434";
        string ollamaModel = Environment.GetEnvironmentVariable("CHATDUMP_OLLAMA_MODEL") ?? "phi3:mini";
        string modelsDirectory = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "models"));
        bool trainClassifierOnly = false;
        string? classifierMetricsPath = null;
        string chunkingProviderMode = "auto";
        int chunkMaxTokens = 256;
        int chunkOverlapTokens = 32;
        string onnxTokenizerMode = "legacy";
        string onnxExecutionProviderMode = "cpu";
        int cudaDeviceId = 0;
        int directMlDeviceId = 0;
        string? embeddingModelPath = null;
        string? embeddingVocabularyPath = null;
        string? mlNetModelPath = null;
        string categoryProviderMode = "hybrid";
        string embeddingProviderMode = "auto";
        int embeddingMaxTokens = 256;
        int hashEmbeddingDimensions = 384;
        double similarityThreshold = 0.72;
        bool hasExplicitSimilarityThreshold = false;
        double? hashSimilarityThreshold = null;
        double? onnxSimilarityThreshold = null;
        int maxSimilarNeighbors = 4;
        int keywordCooccurrenceKeywords = 6;
        int keywordCooccurrenceMinimum = 2;
        bool compareCategories = false;
        bool compareEmbeddings = false;
        bool compareOnnxTokenizers = false;
        bool explicitCategoryProvider = false;
        bool explicitEmbeddingProvider = false;
        bool explicitChunkingProvider = false;
        bool explicitAnalysisProvider = false;
        bool explicitClusterLabeler = false;
        bool explicitOnnxTokenizer = false;
        bool explicitOnnxExecutionProvider = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--input":
                case "-i":
                    input = RequireValue(args, ref i);
                    break;
                case "--output":
                case "-o":
                    output = Path.GetFullPath(RequireValue(args, ref i));
                    break;
                case "--db":
                    db = Path.GetFullPath(RequireValue(args, ref i));
                    break;
                case "--execution-profile":
                    executionProfileMode = RequireValue(args, ref i).Trim().ToLowerInvariant();
                    break;
                case "--use-ollama":
                    conversationAnalyzerMode = "ollama";
                    clusterLabelerMode = "ollama";
                    explicitAnalysisProvider = true;
                    explicitClusterLabeler = true;
                    break;
                case "--analysis-provider":
                    conversationAnalyzerMode = RequireValue(args, ref i).Trim().ToLowerInvariant();
                    explicitAnalysisProvider = true;
                    break;
                case "--cluster-labeler":
                    clusterLabelerMode = RequireValue(args, ref i).Trim().ToLowerInvariant();
                    explicitClusterLabeler = true;
                    break;
                case "--ollama-url":
                    ollamaBaseUrl = RequireValue(args, ref i);
                    break;
                case "--ollama-model":
                    ollamaModel = RequireValue(args, ref i);
                    break;
                case "--models-dir":
                    modelsDirectory = Path.GetFullPath(RequireValue(args, ref i));
                    break;
                case "--train-classifier":
                    trainClassifierOnly = true;
                    break;
                case "--classifier-report":
                    classifierMetricsPath = Path.GetFullPath(RequireValue(args, ref i));
                    break;
                case "--chunking-provider":
                    chunkingProviderMode = RequireValue(args, ref i).Trim().ToLowerInvariant();
                    explicitChunkingProvider = true;
                    break;
                case "--chunk-max-tokens":
                    chunkMaxTokens = int.Parse(RequireValue(args, ref i), CultureInfo.InvariantCulture);
                    break;
                case "--chunk-overlap-tokens":
                    chunkOverlapTokens = int.Parse(RequireValue(args, ref i), CultureInfo.InvariantCulture);
                    break;
                case "--onnx-tokenizer":
                    onnxTokenizerMode = RequireValue(args, ref i).Trim().ToLowerInvariant();
                    explicitOnnxTokenizer = true;
                    break;
                case "--onnx-execution-provider":
                    onnxExecutionProviderMode = RequireValue(args, ref i).Trim().ToLowerInvariant();
                    explicitOnnxExecutionProvider = true;
                    break;
                case "--cuda-device-id":
                    cudaDeviceId = int.Parse(RequireValue(args, ref i), CultureInfo.InvariantCulture);
                    break;
                case "--directml-device-id":
                    directMlDeviceId = int.Parse(RequireValue(args, ref i), CultureInfo.InvariantCulture);
                    break;
                case "--embedding-model":
                    embeddingModelPath = Path.GetFullPath(RequireValue(args, ref i));
                    break;
                case "--embedding-vocab":
                    embeddingVocabularyPath = Path.GetFullPath(RequireValue(args, ref i));
                    break;
                case "--mlnet-model":
                    mlNetModelPath = Path.GetFullPath(RequireValue(args, ref i));
                    break;
                case "--category-provider":
                    categoryProviderMode = RequireValue(args, ref i).Trim().ToLowerInvariant();
                    explicitCategoryProvider = true;
                    break;
                case "--embedding-provider":
                    embeddingProviderMode = RequireValue(args, ref i).Trim().ToLowerInvariant();
                    explicitEmbeddingProvider = true;
                    break;
                case "--embedding-max-tokens":
                    embeddingMaxTokens = int.Parse(RequireValue(args, ref i), CultureInfo.InvariantCulture);
                    break;
                case "--hash-embedding-dim":
                    hashEmbeddingDimensions = int.Parse(RequireValue(args, ref i), CultureInfo.InvariantCulture);
                    break;
                case "--similarity-threshold":
                    similarityThreshold = double.Parse(RequireValue(args, ref i), CultureInfo.InvariantCulture);
                    hasExplicitSimilarityThreshold = true;
                    break;
                case "--hash-similarity-threshold":
                    hashSimilarityThreshold = double.Parse(RequireValue(args, ref i), CultureInfo.InvariantCulture);
                    break;
                case "--onnx-similarity-threshold":
                    onnxSimilarityThreshold = double.Parse(RequireValue(args, ref i), CultureInfo.InvariantCulture);
                    break;
                case "--max-similar-neighbors":
                    maxSimilarNeighbors = int.Parse(RequireValue(args, ref i), CultureInfo.InvariantCulture);
                    break;
                case "--cooccurrence-keywords":
                    keywordCooccurrenceKeywords = int.Parse(RequireValue(args, ref i), CultureInfo.InvariantCulture);
                    break;
                case "--cooccurrence-min":
                    keywordCooccurrenceMinimum = int.Parse(RequireValue(args, ref i), CultureInfo.InvariantCulture);
                    break;
                case "--compare-categories":
                    compareCategories = true;
                    break;
                case "--compare-embeddings":
                    compareEmbeddings = true;
                    break;
                case "--compare-onnx-tokenizers":
                    compareOnnxTokenizers = true;
                    break;
                case "--compare-models":
                    compareCategories = true;
                    compareEmbeddings = true;
                    break;
                case "--help":
                case "-h":
                    PrintUsageAndExit();
                    break;
                default:
                    throw new ArgumentException($"Unknown argument: {args[i]}");
            }
        }

        if (string.IsNullOrWhiteSpace(input))
        {
            PrintUsageAndExit("Missing required --input argument.");
        }

        if (categoryProviderMode is not ("heuristic" or "mlnet" or "hybrid"))
        {
            PrintUsageAndExit($"Unsupported --category-provider value: {categoryProviderMode}");
        }

        if (executionProfileMode is not ("custom" or "heuristic" or "mlnet" or "hybrid" or "onnx"))
        {
            PrintUsageAndExit($"Unsupported --execution-profile value: {executionProfileMode}");
        }

        if (chunkingProviderMode is not ("auto" or "char" or "tokenizer" or "mltokenizer"))
        {
            PrintUsageAndExit($"Unsupported --chunking-provider value: {chunkingProviderMode}");
        }

        if (embeddingProviderMode is not ("auto" or "hash" or "onnx"))
        {
            PrintUsageAndExit($"Unsupported --embedding-provider value: {embeddingProviderMode}");
        }

        if (conversationAnalyzerMode is not ("heuristic" or "ollama"))
        {
            PrintUsageAndExit($"Unsupported --analysis-provider value: {conversationAnalyzerMode}");
        }

        if (clusterLabelerMode is not ("heuristic" or "ollama" or "auto"))
        {
            PrintUsageAndExit($"Unsupported --cluster-labeler value: {clusterLabelerMode}");
        }

        if (onnxTokenizerMode is not ("legacy" or "mltokenizer"))
        {
            PrintUsageAndExit($"Unsupported --onnx-tokenizer value: {onnxTokenizerMode}");
        }

        if (onnxExecutionProviderMode is not ("cpu" or "cuda" or "directml" or "auto"))
        {
            PrintUsageAndExit($"Unsupported --onnx-execution-provider value: {onnxExecutionProviderMode}");
        }

        (categoryProviderMode, embeddingProviderMode, chunkingProviderMode, onnxTokenizerMode, onnxExecutionProviderMode, conversationAnalyzerMode, clusterLabelerMode) =
            ApplyExecutionProfileDefaults(
                executionProfileMode,
                explicitCategoryProvider,
                explicitEmbeddingProvider,
                explicitChunkingProvider,
                explicitOnnxTokenizer,
                explicitOnnxExecutionProvider,
                explicitAnalysisProvider,
                explicitClusterLabeler,
                categoryProviderMode,
                embeddingProviderMode,
                chunkingProviderMode,
                onnxTokenizerMode,
                onnxExecutionProviderMode,
                conversationAnalyzerMode,
                clusterLabelerMode);

        if (string.Equals(clusterLabelerMode, "auto", StringComparison.OrdinalIgnoreCase))
        {
            clusterLabelerMode = conversationAnalyzerMode;
        }

        string defaultEmbeddingModelPath = Path.Combine(modelsDirectory, "embeddings", "model.onnx");
        string defaultEmbeddingVocabularyPath = Path.Combine(modelsDirectory, "embeddings", "vocab.txt");
        string defaultClassifierModelPath = Path.Combine(modelsDirectory, "classification", "conversation-category.zip");
        string defaultClassifierMetricsPath = Path.Combine(modelsDirectory, "classification", "training-metrics.json");

        embeddingModelPath ??= File.Exists(defaultEmbeddingModelPath) ? defaultEmbeddingModelPath : null;
        embeddingVocabularyPath ??= File.Exists(defaultEmbeddingVocabularyPath) ? defaultEmbeddingVocabularyPath : null;

        if (string.IsNullOrWhiteSpace(mlNetModelPath))
        {
            if (trainClassifierOnly)
            {
                mlNetModelPath = defaultClassifierModelPath;
            }
            else if (File.Exists(defaultClassifierModelPath))
            {
                mlNetModelPath = defaultClassifierModelPath;
            }
        }

        classifierMetricsPath ??= defaultClassifierMetricsPath;
        db ??= Path.Combine(output, "chatdump.db");
        return new AppArgs(
            Path.GetFullPath(input!),
            output,
            db,
            executionProfileMode,
            conversationAnalyzerMode,
            clusterLabelerMode,
            ollamaBaseUrl,
            ollamaModel,
            modelsDirectory,
            trainClassifierOnly,
            classifierMetricsPath,
            chunkingProviderMode,
            chunkMaxTokens,
            chunkOverlapTokens,
            onnxTokenizerMode,
            onnxExecutionProviderMode,
            cudaDeviceId,
            directMlDeviceId,
            embeddingModelPath,
            embeddingVocabularyPath,
            mlNetModelPath,
            categoryProviderMode,
            embeddingProviderMode,
            embeddingMaxTokens,
            hashEmbeddingDimensions,
            similarityThreshold,
            hasExplicitSimilarityThreshold,
            hashSimilarityThreshold,
            onnxSimilarityThreshold,
            maxSimilarNeighbors,
            keywordCooccurrenceKeywords,
            keywordCooccurrenceMinimum,
            compareCategories,
            compareEmbeddings,
            compareOnnxTokenizers);
    }

    public double GetSimilarityThresholdForProvider(string? providerDescription)
    {
        if (!string.IsNullOrWhiteSpace(providerDescription))
        {
            if (providerDescription.IndexOf("onnx:", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                if (OnnxSimilarityThreshold.HasValue)
                {
                    return OnnxSimilarityThreshold.Value;
                }

                return HasExplicitSimilarityThreshold ? SimilarityThreshold : 0.60;
            }

            if (providerDescription.StartsWith("hash:", StringComparison.OrdinalIgnoreCase))
            {
                return HashSimilarityThreshold ?? SimilarityThreshold;
            }
        }

        return SimilarityThreshold;
    }

    private static string RequireValue(string[] args, ref int index)
    {
        if (index + 1 >= args.Length)
            throw new ArgumentException($"Missing value for {args[index]}");
        index++;
        return args[index];
    }

    private static void PrintUsageAndExit(string? error = null)
    {
        if (!string.IsNullOrWhiteSpace(error))
            Console.Error.WriteLine(error);

        Console.WriteLine(@"Usage:
  dotnet run -- --input normalized-messages.json [--output out] [--db out/chatdump.db]
      [--use-ollama] [--analysis-provider heuristic|ollama] [--cluster-labeler heuristic|ollama|auto]
      [--ollama-url http://localhost:11434] [--ollama-model phi3:mini]
      [--execution-profile custom|heuristic|mlnet|hybrid|onnx]
      [--models-dir models] [--train-classifier] [--classifier-report models\classification\training-metrics.json]
      [--chunking-provider auto|char|tokenizer|mltokenizer] [--chunk-max-tokens 256] [--chunk-overlap-tokens 32]
      [--onnx-tokenizer legacy|mltokenizer]
      [--onnx-execution-provider cpu|cuda|directml|auto] [--cuda-device-id 0] [--directml-device-id 0]
      [--embedding-model models\all-MiniLM-L6-v2.onnx] [--embedding-vocab models\vocab.txt]
      [--mlnet-model models\conversation-category.zip] [--category-provider hybrid] [--embedding-provider auto]
      [--embedding-max-tokens 256]
      [--hash-embedding-dim 384] [--similarity-threshold 0.72] [--hash-similarity-threshold 0.72]
      [--onnx-similarity-threshold 0.60] [--max-similar-neighbors 4]
      [--cooccurrence-keywords 6] [--cooccurrence-min 2]
      [--compare-categories] [--compare-embeddings] [--compare-onnx-tokenizers] [--compare-models]

Notes:
  --use-ollama is a compatibility shortcut that enables both the Ollama conversation analyzer and Ollama cluster labels.
  --execution-profile provides named provider bundles. Explicit provider flags still win over the profile defaults.
  --analysis-provider = heuristic | ollama. This controls per-conversation analysis and summaries.
  --cluster-labeler = heuristic | ollama | auto. heuristic uses deterministic fallback labels; auto follows --analysis-provider.
  --models-dir defaults to .\models. If model files exist there, they are auto-discovered.
  --train-classifier trains a local ML.NET category model from the normalized export and exits.
  --chunking-provider = auto | char | tokenizer | mltokenizer. auto uses Microsoft.ML.Tokenizers with the embedding vocab when available.
  --chunk-max-tokens and --chunk-overlap-tokens control token-aware chunk windows for transcript/index chunks.
  --onnx-tokenizer = legacy | mltokenizer. This controls tokenizer behavior inside the ONNX embedding path.
  --onnx-execution-provider = cpu | cuda | directml | auto. auto tries CUDA, then DirectML, then CPU.
  --cuda-device-id selects the CUDA adapter index when CUDA is enabled.
  --directml-device-id selects the DirectML adapter index when DirectML is enabled.
  The native ONNX runtime flavor is selected at build time with -p:OnnxRuntimeFlavor=Gpu|DirectML (default Gpu).
  --embedding-model + --embedding-vocab enable local ONNX embeddings for clustering/similarity.
  --category-provider = heuristic | mlnet | hybrid. hybrid prefers ML.NET and keeps heuristic as a second opinion.
  --embedding-provider = auto | hash | onnx. auto uses ONNX when available, otherwise hash.
  --hash-similarity-threshold and --onnx-similarity-threshold override the shared similarity threshold per provider.
  Without explicit overrides, ONNX uses a calibrated default of 0.60 and hash uses the shared default of 0.72.
  --compare-categories writes a heuristic-vs-ML.NET category comparison report.
  --compare-embeddings writes extra graph files that compare hash vs ONNX similarity edges.
  --compare-onnx-tokenizers writes extra graph files that compare legacy WordPiece ONNX vs Microsoft tokenizer ONNX.
  --compare-models enables both comparison reports.
  Without ONNX model files, the corpus index falls back to a local hashing embedding.
  --mlnet-model points to a pre-trained ML.NET multiclass classifier zip for category prediction.
  Input must be the flattened JSON array generated from the export, with fields such as:
    ConversationId, Title, NodeId, ParentNodeId, ChildNodeIds, Role, CreateTime, ContentType, Text, AttachmentsJson, AggregateCode, ExecutionOutput
");
        Environment.Exit(string.IsNullOrWhiteSpace(error) ? 0 : 1);
    }

    private static (string CategoryProviderMode, string EmbeddingProviderMode, string ChunkingProviderMode, string OnnxTokenizerMode, string OnnxExecutionProviderMode, string ConversationAnalyzerMode, string ClusterLabelerMode)
        ApplyExecutionProfileDefaults(
            string executionProfileMode,
            bool explicitCategoryProvider,
            bool explicitEmbeddingProvider,
            bool explicitChunkingProvider,
            bool explicitOnnxTokenizer,
            bool explicitOnnxExecutionProvider,
            bool explicitAnalysisProvider,
            bool explicitClusterLabeler,
            string categoryProviderMode,
            string embeddingProviderMode,
            string chunkingProviderMode,
            string onnxTokenizerMode,
            string onnxExecutionProviderMode,
            string conversationAnalyzerMode,
            string clusterLabelerMode)
    {
        if (string.Equals(executionProfileMode, "custom", StringComparison.OrdinalIgnoreCase))
        {
            return (categoryProviderMode, embeddingProviderMode, chunkingProviderMode, onnxTokenizerMode, onnxExecutionProviderMode, conversationAnalyzerMode, clusterLabelerMode);
        }

        if (!explicitAnalysisProvider)
        {
            conversationAnalyzerMode = "heuristic";
        }

        if (!explicitClusterLabeler)
        {
            clusterLabelerMode = "heuristic";
        }

        switch (executionProfileMode)
        {
            case "heuristic":
                if (!explicitCategoryProvider)
                    categoryProviderMode = "heuristic";
                if (!explicitEmbeddingProvider)
                    embeddingProviderMode = "hash";
                if (!explicitChunkingProvider)
                    chunkingProviderMode = "char";
                break;

            case "mlnet":
                if (!explicitCategoryProvider)
                    categoryProviderMode = "mlnet";
                if (!explicitEmbeddingProvider)
                    embeddingProviderMode = "hash";
                if (!explicitChunkingProvider)
                    chunkingProviderMode = "mltokenizer";
                break;

            case "hybrid":
                if (!explicitCategoryProvider)
                    categoryProviderMode = "hybrid";
                if (!explicitEmbeddingProvider)
                    embeddingProviderMode = "hash";
                if (!explicitChunkingProvider)
                    chunkingProviderMode = "mltokenizer";
                break;

            case "onnx":
                if (!explicitCategoryProvider)
                    categoryProviderMode = "hybrid";
                if (!explicitEmbeddingProvider)
                    embeddingProviderMode = "onnx";
                if (!explicitChunkingProvider)
                    chunkingProviderMode = "mltokenizer";
                if (!explicitOnnxTokenizer)
                    onnxTokenizerMode = "legacy";
                if (!explicitOnnxExecutionProvider)
                    onnxExecutionProviderMode = "auto";
                break;
        }

        return (categoryProviderMode, embeddingProviderMode, chunkingProviderMode, onnxTokenizerMode, onnxExecutionProviderMode, conversationAnalyzerMode, clusterLabelerMode);
    }
}

// -----------------------------
// Input schema
// -----------------------------

sealed class FlatMessage
{
    public string? SourceFile { get; set; }
    public string? ConversationId { get; set; }
    public string? Title { get; set; }
    public double? ConversationTime { get; set; }
    public string? CurrentNode { get; set; }
    public string NodeId { get; set; } = string.Empty;
    public string? ParentNodeId { get; set; }
    public string? ChildNodeIds { get; set; }
    public string? Role { get; set; }
    public string? AuthorName { get; set; }
    public double? CreateTime { get; set; }
    public double? UpdateTime { get; set; }
    public string? Channel { get; set; }
    public string? Recipient { get; set; }
    public string? ContentType { get; set; }
    public string? Text { get; set; }
    public string? AttachmentsJson { get; set; }
    public string? AggregateCode { get; set; }
    public string? ExecutionOutput { get; set; }

    [JsonIgnore]
    public IReadOnlyList<string> Children => string.IsNullOrWhiteSpace(ChildNodeIds)
        ? Array.Empty<string>()
        : ChildNodeIds.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
}

static class FlatMessageLoader
{
    public static async Task<List<FlatMessage>> LoadAsync(string path)
    {
        await using var fs = File.OpenRead(path);
        var data = await JsonSerializer.DeserializeAsync<List<FlatMessage>>(fs, JsonOptions.Default);
        return data ?? new List<FlatMessage>();
    }
}

// -----------------------------
// Domain model
// -----------------------------

sealed class ConversationSet
{
    public required List<Conversation> Conversations { get; init; }
    public required Dictionary<string, MessageNode> AllNodes { get; init; }
}

sealed class Conversation
{
    public string ConversationId { get; init; } = string.Empty;
    public string Title { get; init; } = "(untitled)";
    public double? ConversationTime { get; init; }
    public string? CurrentNodeId { get; init; }
    public string? SourceFile { get; init; }
    public required Dictionary<string, MessageNode> Nodes { get; init; }

    public IEnumerable<MessageNode> MessageNodes => Nodes.Values
        .Where(n => n.Flat.Role is not null || !string.IsNullOrWhiteSpace(n.Flat.Text) || !string.IsNullOrWhiteSpace(n.Flat.AggregateCode))
        .OrderBy(n => n.Flat.CreateTime ?? double.MaxValue)
        .ThenBy(n => n.Flat.NodeId, StringComparer.Ordinal);

    public IReadOnlyList<MessageNode> Roots => Nodes.Values
        .Where(n => n.Parent is null)
        .OrderBy(n => n.Flat.CreateTime ?? double.MaxValue)
        .ThenBy(n => n.Flat.NodeId, StringComparer.Ordinal)
        .ToList();

    public MessageNode? CurrentNode => !string.IsNullOrWhiteSpace(CurrentNodeId) && Nodes.TryGetValue(CurrentNodeId!, out var current)
        ? current
        : null;
}

sealed class MessageNode
{
    public required FlatMessage Flat { get; init; }
    public MessageNode? Parent { get; set; }
    public List<MessageNode> Children { get; } = new();
    public string Id => Flat.NodeId;
}

sealed record ConversationAnalysis(
    Conversation Conversation,
    string Category,
    string? SecondaryCategory,
    string? TertiaryCategory,
    IReadOnlyList<string> Topics,
    IReadOnlyList<KeywordScore> Keywords,
    IReadOnlyList<ConversationTextChunk> Chunks,
    string? ModelSummary,
    IReadOnlyList<GraphEdge> GraphEdges,
    IReadOnlyList<CategoryPrediction>? CategoryPredictions = null,
    ConversationEmbedding? Embedding = null,
    string? TopicLabel = null,
    string? CategorySource = null,
    string? TopicClusterId = null,
    string? TopicClusterLabel = null,
    string? TopicClusterSummary = null,
    string? CategoryCommunityId = null,
    string? CategoryCommunityLabel = null);

sealed record KeywordScore(string Keyword, double Score);
sealed record ConversationTextChunk(string Kind, string Text, int? TokenCount = null);
sealed record GraphEdge(string FromId, string ToId, string EdgeType, double Weight, string ScopeId = "conversation", string? MetadataJson = null);

static class AnalysisDefaults
{
    public const string DefaultCategory = "Uncategorized";
}

// -----------------------------
// Rebuild graph from flat rows
// -----------------------------

static class ConversationRebuilder
{
    public static ConversationSet Rebuild(List<FlatMessage> flatMessages)
    {
        var allNodes = new Dictionary<string, MessageNode>(StringComparer.Ordinal);
        var grouped = flatMessages
            .Where(m => !string.IsNullOrWhiteSpace(m.ConversationId) && !string.IsNullOrWhiteSpace(m.NodeId))
            .GroupBy(m => m.ConversationId!, StringComparer.Ordinal)
            .OrderBy(g => g.Min(x => x.ConversationTime ?? x.CreateTime ?? double.MaxValue));

        var conversations = new List<Conversation>();

        foreach (var group in grouped)
        {
            var nodes = group.ToDictionary(
                m => m.NodeId,
                m => new MessageNode { Flat = m },
                StringComparer.Ordinal);

            foreach (var node in nodes.Values)
            {
                var parentId = node.Flat.ParentNodeId;
                if (!string.IsNullOrWhiteSpace(parentId) && nodes.TryGetValue(parentId, out var parent))
                {
                    node.Parent = parent;
                    parent.Children.Add(node);
                }
            }

            foreach (var node in nodes.Values)
            {
                allNodes[node.Id] = node;
            }

            var first = group.First();
            conversations.Add(new Conversation
            {
                ConversationId = group.Key,
                Title = string.IsNullOrWhiteSpace(first.Title) ? "(untitled)" : first.Title!,
                ConversationTime = first.ConversationTime,
                CurrentNodeId = first.CurrentNode,
                SourceFile = first.SourceFile,
                Nodes = nodes
            });
        }

        return new ConversationSet { Conversations = conversations, AllNodes = allNodes };
    }
}

// -----------------------------
// Analysis
// -----------------------------

interface IConversationAnalyzer
{
    Task<AnalysisResult> AnalyzeAsync(Conversation conversation, CancellationToken cancellationToken = default);
}

sealed record AnalysisResult(
    string Category,
    IReadOnlyList<string> Topics,
    IReadOnlyList<KeywordScore> Keywords,
    string? ModelSummary,
    IReadOnlyList<ConversationTextChunk> Chunks,
    IReadOnlyList<GraphEdge> GraphEdges);

static class ConversationAnalysisPipeline
{
    public static async Task<List<ConversationAnalysis>> AnalyzeAsync(ConversationSet set, IConversationAnalyzer analyzer, CancellationToken cancellationToken = default)
    {
        var results = new List<ConversationAnalysis>(capacity: set.Conversations.Count);

        foreach (var conversation in set.Conversations)
        {
            var analyzed = await analyzer.AnalyzeAsync(conversation, cancellationToken);
            results.Add(new ConversationAnalysis(
                conversation,
                analyzed.Category,
                null,
                null,
                analyzed.Topics,
                analyzed.Keywords,
                analyzed.Chunks,
                analyzed.ModelSummary,
                analyzed.GraphEdges));
        }

        return results;
    }
}

sealed class HeuristicConversationAnalyzer : IConversationAnalyzer
{
    private readonly IConversationChunker _chunker;

    private static readonly HashSet<string> StopWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "the","and","for","that","with","this","from","have","your","into","about","there","would","could","should","what",
        "when","where","which","while","will","want","need","using","used","then","than","they","them","their","does",
        "did","how","why","can","you","our","are","not","but","all","was","were","has","had","its","let","lets",
        "also","just","like","more","some","such","over","under","into","onto","each","than","very","true","false",
        "null","text","code","json","file","files","data","output","message","messages","conversation","conversations"
    };

    private static readonly (string Category, string[] Terms)[] CategoryMap =
    {
        ("Identity/PKI", new[] { "certificate", "certutil", "ca", "adcs", "enrollment", "ldap", "ocsp", "crl", "template", "aia", "cdp", "x509" }),
        ("Observability/Telemetry", new[] { "grafana", "loki", "tempo", "otel", "etw", "metrics", "trace", "tracing", "alloy", "prometheus", "kql", "logql" }),
        ("C#/.NET", new[] { "c#", "dotnet", "roslyn", "blazor", "asp.net", "entityframework", "linq", "visual studio" }),
        ("Infrastructure/Cloud", new[] { "azure", "kubernetes", "docker", "terraform", "vm", "openstack", "oracle cloud", "cloud" }),
        ("Security", new[] { "security", "acl", "sddl", "kerberos", "delegation", "credential", "attack", "threat", "auth" }),
        ("Creative/Visualization", new[] { "matplotlib", "plotly", "visualization", "graph", "render", "logo", "image", "3d" }),
        ("PowerShell", new[] { "powershell", "script", "cmdlet", "psobject", "where-object", "convertfrom-json" }),
        ("General Engineering", new[] { "architecture", "design", "build", "implementation", "refactor", "parser", "database" }),
    };

    public HeuristicConversationAnalyzer(IConversationChunker? chunker = null)
    {
        _chunker = chunker ?? new CharacterConversationChunker();
    }

    public Task<AnalysisResult> AnalyzeAsync(Conversation conversation, CancellationToken cancellationToken = default)
    {
        var mainPath = ConversationTraversal.GetCurrentPath(conversation);
        var allMessages = conversation.MessageNodes.ToList();
        var textCorpus = BuildCorpus(allMessages);
        var keywordScores = ConversationIndexText.ExtractKeywords(conversation, 18);
        var category = Classify(textCorpus);
        var topics = ConversationIndexText.SelectTopics(conversation.Title, keywordScores, 8);

        var chunks = allMessages
            .Where(m => !string.IsNullOrWhiteSpace(m.Flat.Text) || !string.IsNullOrWhiteSpace(m.Flat.AggregateCode) || !string.IsNullOrWhiteSpace(m.Flat.ExecutionOutput))
            .SelectMany(m => _chunker.ChunkMessage(m))
            .ToList();

        var graphEdges = new List<GraphEdge>();
        foreach (var node in conversation.Nodes.Values)
        {
            foreach (var child in node.Children)
            {
                graphEdges.Add(new GraphEdge(node.Id, child.Id, "reply", 1.0));
            }
        }

        foreach (var keyword in keywordScores.Take(10))
        {
            graphEdges.Add(new GraphEdge(conversation.ConversationId, $"kw:{keyword.Keyword}", "has_keyword", keyword.Score));
        }

        graphEdges.Add(new GraphEdge(conversation.ConversationId, $"cat:{category}", "has_category", 1.0));
        foreach (var topic in topics)
        {
            graphEdges.Add(new GraphEdge(conversation.ConversationId, $"topic:{topic}", "has_topic", 1.0));
        }

        return Task.FromResult(new AnalysisResult(category, topics, keywordScores, null, chunks, graphEdges));
    }

    private static string BuildCorpus(IEnumerable<MessageNode> nodes)
    {
        var sb = new StringBuilder();
        foreach (var node in nodes.OrderBy(n => n.Flat.CreateTime ?? double.MaxValue))
        {
            if (!string.IsNullOrWhiteSpace(node.Flat.Text))
            {
                sb.AppendLine(node.Flat.Text);
            }
            if (!string.IsNullOrWhiteSpace(node.Flat.AggregateCode))
            {
                sb.AppendLine(node.Flat.AggregateCode);
            }
            if (!string.IsNullOrWhiteSpace(node.Flat.ExecutionOutput))
            {
                sb.AppendLine(node.Flat.ExecutionOutput);
            }
        }
        return sb.ToString();
    }

    private static string Classify(string corpus)
    {
        var lower = corpus.ToLowerInvariant();
        var best = (Category: AnalysisDefaults.DefaultCategory, Score: 0);
        foreach (var rule in CategoryMap)
        {
            int score = rule.Terms.Sum(t => lower.Contains(t, StringComparison.OrdinalIgnoreCase) ? 1 : 0);
            if (score > best.Score)
            {
                best = (rule.Category, score);
            }
        }
        return best.Score == 0 ? AnalysisDefaults.DefaultCategory : best.Category;
    }

    private static IReadOnlyList<KeywordScore> ExtractKeywords(string corpus, int take)
    {
        var tokens = Tokenize(corpus);
        var counts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var token in tokens)
        {
            counts.TryGetValue(token, out int count);
            counts[token] = count + 1;
        }

        int max = Math.Max(1, counts.Values.DefaultIfEmpty(1).Max());
        return counts
            .Select(kvp => new KeywordScore(kvp.Key, Math.Round((double)kvp.Value / max, 4)))
            .OrderByDescending(k => k.Score)
            .ThenBy(k => k.Keyword, StringComparer.OrdinalIgnoreCase)
            .Take(take)
            .ToList();
    }

    private static IEnumerable<string> Tokenize(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (char ch in text)
        {
            sb.Append(char.IsLetterOrDigit(ch) || ch is '+' or '#' or '.' or '-' or '_' ? char.ToLowerInvariant(ch) : ' ');
        }

        foreach (var token in sb.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (token.Length < 3) continue;
            if (StopWords.Contains(token)) continue;
            if (token.All(char.IsDigit)) continue;
            yield return token;
        }
    }
}

sealed class OllamaConversationAnalyzer : IConversationAnalyzer
{
    private readonly HeuristicConversationAnalyzer _fallback;
    private readonly HttpClient _http = new() { Timeout = TimeSpan.FromMinutes(5) };
    private readonly string _baseUrl;
    private readonly string _model;

    public OllamaConversationAnalyzer(string baseUrl, string model, IConversationChunker? chunker = null)
    {
        _fallback = new HeuristicConversationAnalyzer(chunker);
        _baseUrl = baseUrl.TrimEnd('/');
        _model = model;
    }

    public async Task<AnalysisResult> AnalyzeAsync(Conversation conversation, CancellationToken cancellationToken = default)
    {
        var baseline = await _fallback.AnalyzeAsync(conversation, cancellationToken);

        try
        {
            var corpus = ConversationTraversal.GetCurrentPath(conversation)
                .Where(n => !string.IsNullOrWhiteSpace(n.Flat.Text))
                .Select(n => $"[{n.Flat.Role}] {n.Flat.Text}")
                .JoinAsSingleString("\n\n");

            if (string.IsNullOrWhiteSpace(corpus))
                return baseline;

            var prompt = $$"""
You are analyzing one conversation transcript.
Return strict JSON with this exact schema:
{
  "category": "string",
  "topics": ["string"],
  "keywords": [{"keyword":"string","score":0.0}],
  "summary": "string"
}

Rules:
- Pick one concise category.
- Topics: 3 to 6 items.
- Keywords: 5 to 12 items, lower-case.
- Scores between 0 and 1.
- No markdown. JSON only.

Transcript:
{{corpus}}
""";

            var req = new
            {
                model = _model,
                prompt,
                stream = false,
                format = "json"
            };

            var response = await _http.PostAsJsonAsync($"{_baseUrl}/api/generate", req, cancellationToken);
            response.EnsureSuccessStatusCode();
            var payload = await response.Content.ReadFromJsonAsync<OllamaGenerateResponse>(JsonOptions.Default, cancellationToken);
            if (payload is null || string.IsNullOrWhiteSpace(payload.Response))
                return baseline;

            var local = JsonSerializer.Deserialize<LocalModelResult>(payload.Response, JsonOptions.Default);
            if (local is null)
                return baseline;

            var mergedEdges = baseline.GraphEdges.ToList();
            if (!string.IsNullOrWhiteSpace(local.Category))
                mergedEdges.Add(new GraphEdge(conversation.ConversationId, $"modelcat:{local.Category}", "has_model_category", 1.0));

            foreach (var t in local.Topics ?? Enumerable.Empty<string>())
                mergedEdges.Add(new GraphEdge(conversation.ConversationId, $"modeltopic:{t}", "has_model_topic", 1.0));

            return baseline with
            {
                Category = string.IsNullOrWhiteSpace(local.Category) ? baseline.Category : local.Category,
                Topics = local.Topics?.Where(x => !string.IsNullOrWhiteSpace(x)).Distinct(StringComparer.OrdinalIgnoreCase).ToList() ?? baseline.Topics,
                Keywords = local.Keywords?.Where(k => !string.IsNullOrWhiteSpace(k.Keyword)).ToList() ?? baseline.Keywords,
                ModelSummary = local.Summary,
                GraphEdges = mergedEdges
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[warn] Local model analysis failed for {conversation.ConversationId}: {ex.Message}");
            return baseline;
        }
    }

    private sealed class OllamaGenerateResponse
    {
        [JsonPropertyName("response")]
        public string? Response { get; set; }
    }

    private sealed class LocalModelResult
    {
        [JsonPropertyName("category")]
        public string? Category { get; set; }
        [JsonPropertyName("topics")]
        public List<string>? Topics { get; set; }
        [JsonPropertyName("keywords")]
        public List<KeywordScore>? Keywords { get; set; }
        [JsonPropertyName("summary")]
        public string? Summary { get; set; }
    }
}

// -----------------------------
// Traversal and transcript rendering
// -----------------------------

static class ConversationTraversal
{
    public static IReadOnlyList<MessageNode> GetCurrentPath(Conversation conversation)
    {
        var current = conversation.CurrentNode;
        if (current is null)
        {
            // Fallback to latest leaf.
            current = conversation.Nodes.Values
                .Where(n => n.Children.Count == 0)
                .OrderByDescending(n => n.Flat.CreateTime ?? double.MinValue)
                .FirstOrDefault();
        }

        if (current is null)
            return Array.Empty<MessageNode>();

        var stack = new Stack<MessageNode>();
        var cursor = current;
        while (cursor is not null)
        {
            stack.Push(cursor);
            cursor = cursor.Parent;
        }

        return stack
            .Where(n => n.Flat.Role is not null || !string.IsNullOrWhiteSpace(n.Flat.Text) || !string.IsNullOrWhiteSpace(n.Flat.AggregateCode) || !string.IsNullOrWhiteSpace(n.Flat.ExecutionOutput))
            .ToList();
    }

    public static IReadOnlyList<IReadOnlyList<MessageNode>> GetLeafPaths(Conversation conversation)
    {
        var leaves = conversation.Nodes.Values
            .Where(n => n.Children.Count == 0)
            .OrderBy(n => n.Flat.CreateTime ?? double.MaxValue)
            .ThenBy(n => n.Id, StringComparer.Ordinal)
            .ToList();

        var result = new List<IReadOnlyList<MessageNode>>();
        foreach (var leaf in leaves)
        {
            var stack = new Stack<MessageNode>();
            var cursor = leaf;
            while (cursor is not null)
            {
                stack.Push(cursor);
                cursor = cursor.Parent;
            }

            result.Add(stack
                .Where(n => n.Flat.Role is not null || !string.IsNullOrWhiteSpace(n.Flat.Text) || !string.IsNullOrWhiteSpace(n.Flat.AggregateCode) || !string.IsNullOrWhiteSpace(n.Flat.ExecutionOutput))
                .ToList());
        }

        return result;
    }
}

static class TranscriptWriter
{
    public static void Write(IEnumerable<ConversationAnalysis> analyses, string outputDirectory)
    {
        string transcriptDir = Path.Combine(outputDirectory, "transcripts");
        string branchDir = Path.Combine(outputDirectory, "branches");

        foreach (var analysis in analyses)
        {
            var safe = FileNameSafe(analysis.Conversation.Title, analysis.Conversation.ConversationId);
            var transcriptPath = Path.Combine(transcriptDir, safe + ".txt");
            var mainText = RenderConversationMainline(analysis);
            File.WriteAllText(transcriptPath, mainText, new UTF8Encoding(false));

            var paths = ConversationTraversal.GetLeafPaths(analysis.Conversation);
            for (int i = 0; i < paths.Count; i++)
            {
                var branchPath = Path.Combine(branchDir, safe + $"__branch-{i + 1:000}.txt");
                File.WriteAllText(branchPath, RenderBranch(analysis, paths[i], i + 1), new UTF8Encoding(false));
            }
        }
    }

    private static string RenderConversationMainline(ConversationAnalysis analysis)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Title: {analysis.Conversation.Title}");
        sb.AppendLine($"ConversationId: {analysis.Conversation.ConversationId}");
        sb.AppendLine($"SourceFile: {analysis.Conversation.SourceFile}");
        sb.AppendLine($"Created: {FormatUnix(analysis.Conversation.ConversationTime)}");
        sb.AppendLine($"CurrentNode: {analysis.Conversation.CurrentNodeId}");
        sb.AppendLine($"Category: {analysis.Category}");
        if (!string.IsNullOrWhiteSpace(analysis.SecondaryCategory))
            sb.AppendLine($"SecondaryCategory: {analysis.SecondaryCategory}");
        if (!string.IsNullOrWhiteSpace(analysis.TertiaryCategory))
            sb.AppendLine($"TertiaryCategory: {analysis.TertiaryCategory}");
        if (!string.IsNullOrWhiteSpace(analysis.CategorySource))
            sb.AppendLine($"CategorySource: {analysis.CategorySource}");
        if (!string.IsNullOrWhiteSpace(analysis.CategoryCommunityId))
            sb.AppendLine($"CategoryCommunity: {analysis.CategoryCommunityId}");
        if (!string.IsNullOrWhiteSpace(analysis.CategoryCommunityLabel))
            sb.AppendLine($"CategoryCommunityLabel: {analysis.CategoryCommunityLabel}");
        if (analysis.Chunks.Count > 0)
            sb.AppendLine($"ChunkCount: {analysis.Chunks.Count}");
        if (TryGetChunkTokenTotal(analysis, out int mainTokenTotal))
            sb.AppendLine($"ChunkTokens: {mainTokenTotal}");
        sb.AppendLine($"Topics: {string.Join(", ", analysis.Topics)}");
        if (!string.IsNullOrWhiteSpace(analysis.TopicLabel))
            sb.AppendLine($"TopicLabel: {analysis.TopicLabel}");
        if (!string.IsNullOrWhiteSpace(analysis.TopicClusterId))
            sb.AppendLine($"TopicCluster: {analysis.TopicClusterId}");
        if (!string.IsNullOrWhiteSpace(analysis.TopicClusterLabel))
            sb.AppendLine($"TopicClusterLabel: {analysis.TopicClusterLabel}");
        if (!string.IsNullOrWhiteSpace(analysis.TopicClusterSummary))
            sb.AppendLine($"TopicClusterSummary: {analysis.TopicClusterSummary}");
        sb.AppendLine($"Keywords: {string.Join(", ", analysis.Keywords.Select(k => $"{k.Keyword}:{k.Score.ToString("0.###", CultureInfo.InvariantCulture)}"))}");
        if (analysis.CategoryPredictions is { Count: > 0 } categoryPredictions)
        {
            sb.AppendLine($"CategoryCandidates: {string.Join(", ", categoryPredictions.Select(p => $"{p.Category}:{p.Score.ToString("0.###", CultureInfo.InvariantCulture)} [{p.Source}{(p.IsSelected ? ",selected" : string.Empty)}]"))}");
        }
        if (!string.IsNullOrWhiteSpace(analysis.ModelSummary))
            sb.AppendLine($"ModelSummary: {analysis.ModelSummary}");
        sb.AppendLine(new string('=', 80));
        sb.AppendLine();

        var path = ConversationTraversal.GetCurrentPath(analysis.Conversation);
        foreach (var node in path)
        {
            AppendNode(sb, node);
        }

        return sb.ToString();
    }

    private static string RenderBranch(ConversationAnalysis analysis, IReadOnlyList<MessageNode> path, int branchNumber)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Title: {analysis.Conversation.Title}");
        sb.AppendLine($"ConversationId: {analysis.Conversation.ConversationId}");
        sb.AppendLine($"Branch: {branchNumber}");
        sb.AppendLine($"Category: {analysis.Category}");
        if (!string.IsNullOrWhiteSpace(analysis.SecondaryCategory))
            sb.AppendLine($"SecondaryCategory: {analysis.SecondaryCategory}");
        if (!string.IsNullOrWhiteSpace(analysis.TertiaryCategory))
            sb.AppendLine($"TertiaryCategory: {analysis.TertiaryCategory}");
        if (!string.IsNullOrWhiteSpace(analysis.CategoryCommunityId))
            sb.AppendLine($"CategoryCommunity: {analysis.CategoryCommunityId}");
        if (!string.IsNullOrWhiteSpace(analysis.CategoryCommunityLabel))
            sb.AppendLine($"CategoryCommunityLabel: {analysis.CategoryCommunityLabel}");
        if (analysis.Chunks.Count > 0)
            sb.AppendLine($"ChunkCount: {analysis.Chunks.Count}");
        if (TryGetChunkTokenTotal(analysis, out int branchTokenTotal))
            sb.AppendLine($"ChunkTokens: {branchTokenTotal}");
        sb.AppendLine($"Topics: {string.Join(", ", analysis.Topics)}");
        if (!string.IsNullOrWhiteSpace(analysis.TopicClusterId))
            sb.AppendLine($"TopicCluster: {analysis.TopicClusterId}");
        if (!string.IsNullOrWhiteSpace(analysis.TopicClusterLabel))
            sb.AppendLine($"TopicClusterLabel: {analysis.TopicClusterLabel}");
        sb.AppendLine(new string('-', 80));
        sb.AppendLine();

        foreach (var node in path)
        {
            AppendNode(sb, node);
        }

        return sb.ToString();
    }

    private static void AppendNode(StringBuilder sb, MessageNode node)
    {
        var role = node.Flat.Role ?? "unknown";
        sb.AppendLine($"[{role}] Node={node.Id} Parent={node.Flat.ParentNodeId} Time={FormatUnix(node.Flat.CreateTime)} ContentType={node.Flat.ContentType}");
        if (!string.IsNullOrWhiteSpace(node.Flat.Text))
        {
            sb.AppendLine(node.Flat.Text!.TrimEnd());
        }
        if (!string.IsNullOrWhiteSpace(node.Flat.AggregateCode))
        {
            sb.AppendLine();
            sb.AppendLine("```code");
            sb.AppendLine(node.Flat.AggregateCode!.TrimEnd());
            sb.AppendLine("```");
        }
        if (!string.IsNullOrWhiteSpace(node.Flat.ExecutionOutput))
        {
            sb.AppendLine();
            sb.AppendLine("```execution_output");
            sb.AppendLine(node.Flat.ExecutionOutput!.TrimEnd());
            sb.AppendLine("```");
        }
        if (!string.IsNullOrWhiteSpace(node.Flat.AttachmentsJson))
        {
            sb.AppendLine();
            sb.AppendLine("Attachments:");
            sb.AppendLine(node.Flat.AttachmentsJson!.TrimEnd());
        }
        sb.AppendLine();
        sb.AppendLine(new string('-', 80));
        sb.AppendLine();
    }

    private static bool TryGetChunkTokenTotal(ConversationAnalysis analysis, out int total)
    {
        total = 0;
        bool any = false;
        foreach (var chunk in analysis.Chunks)
        {
            if (!chunk.TokenCount.HasValue)
            {
                continue;
            }

            total += chunk.TokenCount.Value;
            any = true;
        }

        return any;
    }

    private static string FileNameSafe(string title, string conversationId)
    {
        string combined = string.IsNullOrWhiteSpace(title) ? conversationId : $"{title}__{conversationId}";
        foreach (char c in Path.GetInvalidFileNameChars())
            combined = combined.Replace(c, '_');
        if (combined.Length > 140)
            combined = combined.Substring(0, 140);
        return combined;
    }

    private static string FormatUnix(double? unix)
    {
        if (unix is null)
            return "(null)";
        try
        {
            return DateTimeOffset.FromUnixTimeMilliseconds((long)(unix.Value * 1000)).ToString("u");
        }
        catch
        {
            return unix.Value.ToString(CultureInfo.InvariantCulture);
        }
    }
}

// -----------------------------
// Graph outputs
// -----------------------------

static class GraphWriter
{
    public static void Write(IReadOnlyList<ConversationAnalysis> analyses, string graphDirectory)
    {
        var graph = new
        {
            nodes = BuildNodes(analyses),
            edges = analyses.SelectMany(a => a.GraphEdges).ToList()
        };

        File.WriteAllText(Path.Combine(graphDirectory, "conversation-graph.json"), JsonSerializer.Serialize(graph, JsonOptions.WriteIndented), new UTF8Encoding(false));
        File.WriteAllText(Path.Combine(graphDirectory, "conversation-graph.dot"), BuildDot(analyses), new UTF8Encoding(false));
    }

    private static List<object> BuildNodes(IReadOnlyList<ConversationAnalysis> analyses)
    {
        var nodes = new List<object>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        foreach (var analysis in analyses)
        {
            if (seen.Add(analysis.Conversation.ConversationId))
            {
                nodes.Add(new { id = analysis.Conversation.ConversationId, label = analysis.Conversation.Title, kind = "conversation" });
            }

            foreach (var message in analysis.Conversation.Nodes.Values)
            {
                if (seen.Add(message.Id))
                {
                    nodes.Add(new { id = message.Id, label = Preview(message.Flat.Text ?? message.Flat.AggregateCode ?? message.Flat.ExecutionOutput), kind = message.Flat.Role ?? "message" });
                }
            }

            foreach (var keyword in analysis.Keywords)
            {
                string id = $"kw:{keyword.Keyword}";
                if (seen.Add(id)) nodes.Add(new { id, label = keyword.Keyword, kind = "keyword" });
            }

            foreach (var topic in analysis.Topics)
            {
                string id = $"topic:{topic}";
                if (seen.Add(id)) nodes.Add(new { id, label = topic, kind = "topic" });
            }

            string cat = $"cat:{analysis.Category}";
            if (seen.Add(cat)) nodes.Add(new { id = cat, label = analysis.Category, kind = "category" });
        }

        return nodes;
    }

    private static string BuildDot(IReadOnlyList<ConversationAnalysis> analyses)
    {
        var sb = new StringBuilder();
        sb.AppendLine("digraph ChatDump {");
        sb.AppendLine("  rankdir=LR;");

        foreach (var analysis in analyses)
        {
            var convId = Escape(analysis.Conversation.ConversationId);
            sb.AppendLine($"  \"{convId}\" [shape=box,label=\"{Escape(analysis.Conversation.Title)}\"];");
            foreach (var edge in analysis.GraphEdges)
            {
                sb.AppendLine($"  \"{Escape(edge.FromId)}\" -> \"{Escape(edge.ToId)}\" [label=\"{Escape(edge.EdgeType)}:{edge.Weight.ToString("0.###", CultureInfo.InvariantCulture)}\"];");
            }
        }

        sb.AppendLine("}");
        return sb.ToString();
    }

    private static string Preview(string? text)
    {
        if (string.IsNullOrWhiteSpace(text)) return string.Empty;
        var s = text.Replace("\r", " ").Replace("\n", " ").Trim();
        return s.Length > 80 ? s.Substring(0, 80) + "…" : s;
    }

    private static string Escape(string input) => input.Replace("\\", "\\\\").Replace("\"", "\\\"");
}

// -----------------------------
// SQLite
// -----------------------------

static class DatabaseWriter
{
    public static void Write(IReadOnlyList<ConversationAnalysis> analyses, string databasePath)
    {
        var directory = Path.GetDirectoryName(databasePath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        using var connection = new SqliteConnection($"Data Source={databasePath}");
        connection.Open();

        using var transaction = connection.BeginTransaction();
        CreateSchema(connection, transaction);
        ClearExisting(connection, transaction);

        foreach (var analysis in analyses)
        {
            InsertConversation(connection, transaction, analysis);
            foreach (var node in analysis.Conversation.MessageNodes)
            {
                InsertMessage(connection, transaction, analysis.Conversation, node);
            }
            foreach (var keyword in analysis.Keywords)
            {
                InsertKeyword(connection, transaction, analysis.Conversation.ConversationId, keyword);
            }
            foreach (var topic in analysis.Topics)
            {
                InsertTopic(connection, transaction, analysis.Conversation.ConversationId, topic, analysis.Category);
            }
            foreach (var chunk in analysis.Chunks)
            {
                InsertChunk(connection, transaction, analysis.Conversation.ConversationId, chunk);
            }
            foreach (var edge in analysis.GraphEdges)
            {
                InsertEdge(connection, transaction, analysis.Conversation.ConversationId, edge);
            }
        }

        transaction.Commit();
    }

    private static void CreateSchema(SqliteConnection connection, SqliteTransaction transaction)
    {
        var sql = @"
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_file TEXT NULL,
    created_utc TEXT NULL,
    current_node_id TEXT NULL,
    category TEXT NOT NULL,
    model_summary TEXT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    node_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    parent_node_id TEXT NULL,
    role TEXT NULL,
    author_name TEXT NULL,
    created_utc TEXT NULL,
    content_type TEXT NULL,
    channel TEXT NULL,
    recipient TEXT NULL,
    text_body TEXT NULL,
    attachments_json TEXT NULL,
    aggregate_code TEXT NULL,
    execution_output TEXT NULL,
    child_node_ids TEXT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS conversation_keywords (
    conversation_id TEXT NOT NULL,
    keyword TEXT NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY(conversation_id, keyword),
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS conversation_topics (
    conversation_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    category TEXT NOT NULL,
    PRIMARY KEY(conversation_id, topic),
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS conversation_chunks (
    conversation_id TEXT NOT NULL,
    chunk_hash TEXT NOT NULL,
    kind TEXT NOT NULL,
    text_body TEXT NOT NULL,
    token_count INTEGER NULL,
    PRIMARY KEY(conversation_id, chunk_hash),
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
);

CREATE TABLE IF NOT EXISTS graph_edges (
    conversation_id TEXT NOT NULL,
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY(conversation_id, from_id, to_id, edge_type),
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
);

CREATE INDEX IF NOT EXISTS ix_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS ix_messages_parent_node_id ON messages(parent_node_id);
CREATE INDEX IF NOT EXISTS ix_keywords_keyword ON conversation_keywords(keyword);
CREATE INDEX IF NOT EXISTS ix_topics_topic ON conversation_topics(topic);
CREATE INDEX IF NOT EXISTS ix_edges_from_to ON graph_edges(from_id, to_id);
";
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    private static void ClearExisting(SqliteConnection connection, SqliteTransaction transaction)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
DELETE FROM graph_edges;
DELETE FROM conversation_chunks;
DELETE FROM conversation_topics;
DELETE FROM conversation_keywords;
DELETE FROM messages;
DELETE FROM conversations;";
        cmd.ExecuteNonQuery();
    }

    private static void InsertConversation(SqliteConnection connection, SqliteTransaction transaction, ConversationAnalysis analysis)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT INTO conversations(conversation_id, title, source_file, created_utc, current_node_id, category, model_summary)
VALUES ($conversation_id, $title, $source_file, $created_utc, $current_node_id, $category, $model_summary);";
        cmd.Parameters.AddWithValue("$conversation_id", analysis.Conversation.ConversationId);
        cmd.Parameters.AddWithValue("$title", analysis.Conversation.Title);
        cmd.Parameters.AddWithValue("$source_file", (object?)analysis.Conversation.SourceFile ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$created_utc", (object?)FormatUnix(analysis.Conversation.ConversationTime) ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$current_node_id", (object?)analysis.Conversation.CurrentNodeId ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$category", analysis.Category);
        cmd.Parameters.AddWithValue("$model_summary", (object?)analysis.ModelSummary ?? DBNull.Value);
        cmd.ExecuteNonQuery();
    }

    private static void InsertMessage(SqliteConnection connection, SqliteTransaction transaction, Conversation conversation, MessageNode node)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT INTO messages(node_id, conversation_id, parent_node_id, role, author_name, created_utc, content_type, channel, recipient, text_body, attachments_json, aggregate_code, execution_output, child_node_ids)
VALUES ($node_id, $conversation_id, $parent_node_id, $role, $author_name, $created_utc, $content_type, $channel, $recipient, $text_body, $attachments_json, $aggregate_code, $execution_output, $child_node_ids);";
        cmd.Parameters.AddWithValue("$node_id", node.Id);
        cmd.Parameters.AddWithValue("$conversation_id", conversation.ConversationId);
        cmd.Parameters.AddWithValue("$parent_node_id", (object?)node.Flat.ParentNodeId ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$role", (object?)node.Flat.Role ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$author_name", (object?)node.Flat.AuthorName ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$created_utc", (object?)FormatUnix(node.Flat.CreateTime) ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$content_type", (object?)node.Flat.ContentType ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$channel", (object?)node.Flat.Channel ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$recipient", (object?)node.Flat.Recipient ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$text_body", (object?)node.Flat.Text ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$attachments_json", (object?)node.Flat.AttachmentsJson ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$aggregate_code", (object?)node.Flat.AggregateCode ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$execution_output", (object?)node.Flat.ExecutionOutput ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$child_node_ids", (object?)node.Flat.ChildNodeIds ?? DBNull.Value);
        cmd.ExecuteNonQuery();
    }

    private static void InsertKeyword(SqliteConnection connection, SqliteTransaction transaction, string conversationId, KeywordScore keyword)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT INTO conversation_keywords(conversation_id, keyword, score)
VALUES ($conversation_id, $keyword, $score);";
        cmd.Parameters.AddWithValue("$conversation_id", conversationId);
        cmd.Parameters.AddWithValue("$keyword", keyword.Keyword);
        cmd.Parameters.AddWithValue("$score", keyword.Score);
        cmd.ExecuteNonQuery();
    }

    private static void InsertTopic(SqliteConnection connection, SqliteTransaction transaction, string conversationId, string topic, string category)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT INTO conversation_topics(conversation_id, topic, category)
VALUES ($conversation_id, $topic, $category);";
        cmd.Parameters.AddWithValue("$conversation_id", conversationId);
        cmd.Parameters.AddWithValue("$topic", topic);
        cmd.Parameters.AddWithValue("$category", category);
        cmd.ExecuteNonQuery();
    }

    private static void InsertChunk(SqliteConnection connection, SqliteTransaction transaction, string conversationId, ConversationTextChunk chunk)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT OR IGNORE INTO conversation_chunks(conversation_id, chunk_hash, kind, text_body, token_count)
VALUES ($conversation_id, $chunk_hash, $kind, $text_body, $token_count);";
        cmd.Parameters.AddWithValue("$conversation_id", conversationId);
        cmd.Parameters.AddWithValue("$chunk_hash", Sha256(chunk.Kind + "\n" + chunk.Text));
        cmd.Parameters.AddWithValue("$kind", chunk.Kind);
        cmd.Parameters.AddWithValue("$text_body", chunk.Text);
        cmd.Parameters.AddWithValue("$token_count", (object?)chunk.TokenCount ?? DBNull.Value);
        cmd.ExecuteNonQuery();
    }

    private static void InsertEdge(SqliteConnection connection, SqliteTransaction transaction, string conversationId, GraphEdge edge)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT INTO graph_edges(conversation_id, from_id, to_id, edge_type, weight)
VALUES ($conversation_id, $from_id, $to_id, $edge_type, $weight);";
        cmd.Parameters.AddWithValue("$conversation_id", conversationId);
        cmd.Parameters.AddWithValue("$from_id", edge.FromId);
        cmd.Parameters.AddWithValue("$to_id", edge.ToId);
        cmd.Parameters.AddWithValue("$edge_type", edge.EdgeType);
        cmd.Parameters.AddWithValue("$weight", edge.Weight);
        cmd.ExecuteNonQuery();
    }

    private static string? FormatUnix(double? unix)
    {
        if (unix is null)
            return null;
        try
        {
            return DateTimeOffset.FromUnixTimeMilliseconds((long)(unix.Value * 1000)).UtcDateTime.ToString("u");
        }
        catch
        {
            return unix.Value.ToString(CultureInfo.InvariantCulture);
        }
    }

    private static string Sha256(string value)
    {
        using var sha = SHA256.Create();
        var bytes = sha.ComputeHash(Encoding.UTF8.GetBytes(value));
        return Convert.ToHexString(bytes);
    }
}

// -----------------------------
// Shared helpers
// -----------------------------

static class JsonOptions
{
    public static readonly JsonSerializerOptions Default = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true
    };

    public static readonly JsonSerializerOptions WriteIndented = new(Default)
    {
        WriteIndented = true
    };
}

static class EnumerableExtensions
{
    public static string JoinAsSingleString(this IEnumerable<string> values, string separator)
        => string.Join(separator, values);
}
