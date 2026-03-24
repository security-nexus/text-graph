using System.Globalization;
using System.Net.Http.Json;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Data.Sqlite;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

sealed record CategoryPrediction(string Category, double Score, string Source, bool IsSelected = false);
sealed record ConversationEmbedding(string Provider, string Model, float[] Values)
{
    public int Dimension => Values.Length;
}
sealed record TopicCluster(
    string ClusterId,
    string Label,
    string? Summary,
    string PrimaryCategory,
    IReadOnlyList<string> ConversationIds,
    IReadOnlyList<string> RepresentativeKeywords);
sealed record CategoryCommunity(
    string CommunityId,
    string Label,
    string? Summary,
    string PrimaryCategory,
    IReadOnlyList<string> Categories,
    IReadOnlyList<string> ConversationIds,
    IReadOnlyList<string> RepresentativeKeywords);
sealed record KeywordCooccurrence(string LeftKeyword, string RightKeyword, int Count, double Weight);
sealed record CategoryHierarchyLink(string ParentCategory, string ChildCategory, int ConversationCount);
sealed record CorpusIndex(
    IReadOnlyList<ConversationAnalysis> Conversations,
    IReadOnlyList<TopicCluster> TopicClusters,
    IReadOnlyList<CategoryCommunity> CategoryCommunities,
    IReadOnlyList<KeywordCooccurrence> KeywordCooccurrences,
    IReadOnlyList<CategoryHierarchyLink> CategoryHierarchy,
    IReadOnlyList<GraphEdge> GlobalEdges);

static class CorpusIndexingPipeline
{
    private const string GlobalScopeId = "global";
    private const string FallbackCategory = "Uncategorized";
    private const int CategoryFacetCount = 3;
    private const double MinimumAlternateCategoryScore = 0.11;

    public static async Task<CorpusIndex> BuildAsync(
        IReadOnlyList<ConversationAnalysis> analyses,
        AppArgs arguments,
        CancellationToken cancellationToken = default)
    {
        using IEmbeddingProvider embeddingProvider = CreateEmbeddingProvider(arguments);
        using ICategoryPredictor categoryPredictor = CreateCategoryPredictor(arguments);
        string resolvedClusterLabelerMode = ResolveClusterLabelerMode(arguments);
        using IClusterLabeler? clusterLabeler = string.Equals(resolvedClusterLabelerMode, "ollama", StringComparison.OrdinalIgnoreCase)
            ? new OllamaClusterLabeler(arguments.OllamaBaseUrl, arguments.OllamaModel)
            : null;
        double similarityThreshold = arguments.GetSimilarityThresholdForProvider(embeddingProvider.Description);

        Console.WriteLine($"Embedding provider: {embeddingProvider.Description}");
        Console.WriteLine($"Similarity threshold: {similarityThreshold.ToString("0.###", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"Category predictor: {categoryPredictor.Description}");
        Console.WriteLine($"Cluster labeler: {(clusterLabeler is null ? "heuristic" : $"ollama:{arguments.OllamaModel}")}");

        var enriched = new List<ConversationAnalysis>(analyses.Count);
        foreach (var analysis in analyses)
        {
            cancellationToken.ThrowIfCancellationRequested();

            string corpus = ConversationIndexText.BuildCorpus(analysis.Conversation);
            var embedding = embeddingProvider.Embed(analysis.Conversation.ConversationId, corpus);
            var predictions = BuildCategoryPredictions(analysis, corpus, categoryPredictor, arguments);
            var selectedCategory = predictions.First(p => p.IsSelected);
            var categoryFacets = SelectCategoryFacets(predictions, CategoryFacetCount);
            var mergedTopics = MergeTopics(analysis, selectedCategory.Category);
            var topicLabel = BuildTopicLabel(analysis, mergedTopics);
            var edges = BuildConversationEdges(analysis, categoryFacets, mergedTopics, predictions, topicLabel);

            enriched.Add(analysis with
            {
                Category = selectedCategory.Category,
                SecondaryCategory = categoryFacets.Secondary?.Category,
                TertiaryCategory = categoryFacets.Tertiary?.Category,
                Topics = mergedTopics,
                GraphEdges = edges,
                CategoryPredictions = predictions,
                Embedding = embedding,
                TopicLabel = topicLabel,
                CategorySource = selectedCategory.Source
            });
        }

        var similarityEdges = SimilarityGraphBuilder.Build(
            enriched,
            similarityThreshold,
            arguments.MaxSimilarNeighbors);

        var clusters = await TopicClusterBuilder.BuildAsync(enriched, similarityEdges, clusterLabeler, cancellationToken);

        var clusterAssignments = clusters
            .SelectMany(cluster => cluster.ConversationIds.Select(conversationId => (conversationId, cluster)))
            .ToDictionary(x => x.conversationId, x => x.cluster, StringComparer.Ordinal);

        WriteClusterDiagnostics(clusters);

        var clustered = enriched
            .Select(analysis => analysis with
            {
                TopicClusterId = clusterAssignments.GetValueOrDefault(analysis.Conversation.ConversationId)?.ClusterId,
                TopicClusterLabel = clusterAssignments.GetValueOrDefault(analysis.Conversation.ConversationId)?.Label,
                TopicClusterSummary = clusterAssignments.GetValueOrDefault(analysis.Conversation.ConversationId)?.Summary
            })
            .ToList();

        var categoryCommunities = CategoryCommunityBuilder.Build(clustered);
        var categoryCommunityAssignments = categoryCommunities
            .SelectMany(community => community.ConversationIds.Select(conversationId => (conversationId, community)))
            .ToDictionary(x => x.conversationId, x => x.community, StringComparer.Ordinal);

        WriteCategoryCommunityDiagnostics(categoryCommunities);

        var communityClustered = clustered
            .Select(analysis => analysis with
            {
                CategoryCommunityId = categoryCommunityAssignments.GetValueOrDefault(analysis.Conversation.ConversationId)?.CommunityId,
                CategoryCommunityLabel = categoryCommunityAssignments.GetValueOrDefault(analysis.Conversation.ConversationId)?.Label
            })
            .ToList();

        var keywordCooccurrences = KeywordCooccurrenceBuilder.Build(
            communityClustered,
            arguments.KeywordCooccurrenceKeywords,
            arguments.KeywordCooccurrenceMinimum);

        var categoryHierarchy = CategoryHierarchyBuilder.Build(communityClustered);

        var globalEdges = DeduplicateEdges(
            similarityEdges
                .Concat(BuildClusterEdges(clusters))
                .Concat(BuildCategoryCommunityEdges(categoryCommunities))
                .Concat(KeywordCooccurrenceBuilder.BuildEdges(keywordCooccurrences))
                .Concat(CategoryHierarchyBuilder.BuildEdges(categoryHierarchy))
                .ToList());

        return new CorpusIndex(communityClustered, clusters, categoryCommunities, keywordCooccurrences, categoryHierarchy, globalEdges);
    }

    private static string ResolveClusterLabelerMode(AppArgs arguments)
        => string.Equals(arguments.ClusterLabelerMode, "auto", StringComparison.OrdinalIgnoreCase)
            ? arguments.ConversationAnalyzerMode
            : arguments.ClusterLabelerMode;

    private static void WriteClusterDiagnostics(IReadOnlyList<TopicCluster> clusters)
    {
        var sizes = clusters.Select(cluster => cluster.ConversationIds.Count).OrderBy(size => size).ToList();
        int singletonCount = sizes.Count(size => size == 1);
        int pairCount = sizes.Count(size => size == 2);
        int threePlusCount = sizes.Count(size => size >= 3);
        int maxSize = sizes.Count == 0 ? 0 : sizes[^1];
        double averageSize = sizes.Count == 0 ? 0 : sizes.Average();

        Console.WriteLine($"Topic clusters: {clusters.Count:N0} total; {singletonCount:N0} singletons; {pairCount:N0} pairs; {threePlusCount:N0} size>=3; max {maxSize:N0}; avg {averageSize.ToString("0.##", CultureInfo.InvariantCulture)}");
    }

    private static void WriteCategoryCommunityDiagnostics(IReadOnlyList<CategoryCommunity> communities)
    {
        var sizes = communities.Select(community => community.ConversationIds.Count).OrderBy(size => size).ToList();
        int singletonCount = sizes.Count(size => size == 1);
        int pairCount = sizes.Count(size => size == 2);
        int threePlusCount = sizes.Count(size => size >= 3);
        int maxSize = sizes.Count == 0 ? 0 : sizes[^1];
        double averageSize = sizes.Count == 0 ? 0 : sizes.Average();

        Console.WriteLine($"Category communities: {communities.Count:N0} total; {singletonCount:N0} singletons; {pairCount:N0} pairs; {threePlusCount:N0} size>=3; max {maxSize:N0}; avg {averageSize.ToString("0.##", CultureInfo.InvariantCulture)}");
    }

    private static IEmbeddingProvider CreateEmbeddingProvider(AppArgs arguments)
    {
        var fallback = new HashingEmbeddingProvider(arguments.HashEmbeddingDimensions);
        if (string.Equals(arguments.EmbeddingProviderMode, "hash", StringComparison.OrdinalIgnoreCase))
        {
            return fallback;
        }

        if (string.IsNullOrWhiteSpace(arguments.EmbeddingModelPath) || string.IsNullOrWhiteSpace(arguments.EmbeddingVocabularyPath))
        {
            return fallback;
        }

        if (!File.Exists(arguments.EmbeddingModelPath))
        {
            Console.Error.WriteLine($"[warn] ONNX embedding model not found: {arguments.EmbeddingModelPath}. Falling back to hashing embeddings.");
            return fallback;
        }

        if (LocalModelFileInspector.IsGitLfsPointer(arguments.EmbeddingModelPath))
        {
            Console.Error.WriteLine($"[warn] ONNX embedding model is a Git LFS pointer, not a hydrated binary: {arguments.EmbeddingModelPath}. Falling back to hashing embeddings.");
            return fallback;
        }

        if (!File.Exists(arguments.EmbeddingVocabularyPath))
        {
            Console.Error.WriteLine($"[warn] Embedding vocabulary not found: {arguments.EmbeddingVocabularyPath}. Falling back to hashing embeddings.");
            return fallback;
        }

        try
        {
            return new FallbackEmbeddingProvider(
                CreateOnnxEmbeddingProvider(arguments),
                fallback);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[warn] Failed to initialize ONNX embedding provider: {ex.Message}. Falling back to hashing embeddings.");
            return fallback;
        }
    }

    internal static IEmbeddingProvider CreateOnnxEmbeddingProvider(AppArgs arguments, string? tokenizerModeOverride = null)
        => new OnnxEmbeddingProvider(
            arguments.EmbeddingModelPath!,
            arguments.EmbeddingVocabularyPath!,
            arguments.EmbeddingMaxTokens,
            tokenizerModeOverride ?? arguments.OnnxTokenizerMode,
            arguments.OnnxExecutionProviderMode,
            arguments.CudaDeviceId,
            arguments.DirectMlDeviceId);

    private static ICategoryPredictor CreateCategoryPredictor(AppArgs arguments)
    {
        if (string.Equals(arguments.CategoryProviderMode, "heuristic", StringComparison.OrdinalIgnoreCase))
        {
            return new NullCategoryPredictor();
        }

        ICategoryPredictor? predictor = TryCreateMlNetCategoryPredictor(arguments);
        return predictor ?? new NullCategoryPredictor();
    }

    internal static ICategoryPredictor? TryCreateMlNetCategoryPredictor(AppArgs arguments, bool writeWarnings = true)
    {
        if (string.IsNullOrWhiteSpace(arguments.MlNetModelPath))
        {
            return null;
        }

        if (!File.Exists(arguments.MlNetModelPath))
        {
            if (writeWarnings)
            {
                Console.Error.WriteLine($"[warn] ML.NET model not found: {arguments.MlNetModelPath}. Using baseline categories only.");
            }

            return null;
        }

        if (LocalModelFileInspector.IsGitLfsPointer(arguments.MlNetModelPath))
        {
            if (writeWarnings)
            {
                Console.Error.WriteLine($"[warn] ML.NET model path is a Git LFS pointer, not a hydrated binary: {arguments.MlNetModelPath}. Using baseline categories only.");
            }

            return null;
        }

        try
        {
            return new MlNetCategoryPredictor(arguments.MlNetModelPath);
        }
        catch (Exception ex)
        {
            if (writeWarnings)
            {
                Console.Error.WriteLine($"[warn] Failed to initialize ML.NET classifier: {ex.Message}. Using baseline categories only.");
            }

            return null;
        }
    }

    private static List<CategoryPrediction> BuildCategoryPredictions(
        ConversationAnalysis analysis,
        string corpus,
        ICategoryPredictor categoryPredictor,
        AppArgs arguments)
    {
        string baselineCategory = NormalizeCategory(analysis.Category);
        string baselineSource = arguments.UseOllamaAnalysis ? "ollama" : "heuristic";
        var predictions = new List<CategoryPrediction>();
        if (!string.Equals(arguments.CategoryProviderMode, "mlnet", StringComparison.OrdinalIgnoreCase))
        {
            predictions.Add(new CategoryPrediction(baselineCategory, 0.75, baselineSource));
        }

        foreach (var mlPrediction in categoryPredictor.PredictTopK(corpus, 5))
        {
            if (!string.IsNullOrWhiteSpace(mlPrediction.Category))
            {
                predictions.Add(mlPrediction with { Category = NormalizeCategory(mlPrediction.Category) });
            }
        }

        if (predictions.Count == 0)
        {
            predictions.Add(new CategoryPrediction(baselineCategory, 0.75, baselineSource));
        }

        var deduped = predictions
            .GroupBy(p => p.Category, StringComparer.OrdinalIgnoreCase)
            .Select(group => group
                .OrderByDescending(p => p.Source.Equals("mlnet", StringComparison.OrdinalIgnoreCase))
                .ThenByDescending(p => p.Score)
                .First())
            .ToList();

        CategoryPrediction selected = arguments.CategoryProviderMode switch
        {
            "heuristic" => deduped
                .FirstOrDefault(p => p.Source.Equals("heuristic", StringComparison.OrdinalIgnoreCase)
                    || p.Source.Equals("ollama", StringComparison.OrdinalIgnoreCase))
                ?? deduped.First(),
            "mlnet" => deduped
                .FirstOrDefault(p => p.Source.Equals("mlnet", StringComparison.OrdinalIgnoreCase))
                ?? deduped.First(),
            _ => deduped
                .FirstOrDefault(p => p.Source.Equals("mlnet", StringComparison.OrdinalIgnoreCase))
                ?? deduped.First()
        };

        return deduped
            .Select(p => p with
            {
                IsSelected = string.Equals(p.Category, selected.Category, StringComparison.OrdinalIgnoreCase)
                    && string.Equals(p.Source, selected.Source, StringComparison.OrdinalIgnoreCase)
            })
            .OrderByDescending(p => p.IsSelected)
            .ThenByDescending(p => p.Score)
            .ThenBy(p => p.Category, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static CategoryFacetSelection SelectCategoryFacets(
        IReadOnlyList<CategoryPrediction> predictions,
        int maxCategories)
    {
        var ranked = predictions
            .Where(prediction => !string.IsNullOrWhiteSpace(prediction.Category))
            .DistinctBy(prediction => prediction.Category, StringComparer.OrdinalIgnoreCase)
            .Take(Math.Max(1, maxCategories))
            .ToList();

        CategoryPrediction primary = ranked.FirstOrDefault(prediction => prediction.IsSelected)
            ?? ranked.FirstOrDefault()
            ?? new CategoryPrediction(FallbackCategory, 1.0, "heuristic", IsSelected: true);

        var alternates = ranked
            .Where(prediction =>
                !string.Equals(prediction.Category, primary.Category, StringComparison.OrdinalIgnoreCase)
                && prediction.Score >= MinimumAlternateCategoryScore)
            .ToList();

        return new CategoryFacetSelection(
            primary,
            alternates.ElementAtOrDefault(0),
            alternates.ElementAtOrDefault(1));
    }

    private static IReadOnlyList<string> MergeTopics(ConversationAnalysis analysis, string selectedCategory)
    {
        var topics = new List<string>();
        topics.AddRange(analysis.Topics);
        topics.AddRange(analysis.Keywords.Take(6).Select(k => k.Keyword));

        string categoryLeaf = CategoryHierarchyBuilder.GetLeafCategory(selectedCategory);
        if (!string.IsNullOrWhiteSpace(categoryLeaf)
            && !string.Equals(categoryLeaf, FallbackCategory, StringComparison.OrdinalIgnoreCase)
            && CategoryLeafIsGrounded(analysis, categoryLeaf))
        {
            topics.Add(categoryLeaf);
        }

        return ConversationIndexText.SelectDiverseTerms(
            topics
            .Where(topic => !string.IsNullOrWhiteSpace(topic))
            .Select(topic => topic.Trim()),
            8);
    }

    private static bool CategoryLeafIsGrounded(ConversationAnalysis analysis, string categoryLeaf)
    {
        var leafTokens = ConversationIndexText.Tokenize(categoryLeaf).ToArray();
        if (leafTokens.Length == 0)
        {
            return false;
        }

        var extractedTerms = analysis.Topics
            .Concat(analysis.Keywords.Select(keyword => keyword.Keyword))
            .SelectMany(ConversationIndexText.Tokenize)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        return leafTokens.Any(extractedTerms.Contains);
    }

    private static string BuildTopicLabel(ConversationAnalysis analysis, IReadOnlyList<string> mergedTopics)
    {
        if (!string.IsNullOrWhiteSpace(analysis.TopicLabel))
        {
            return analysis.TopicLabel!;
        }

        if (mergedTopics.Count > 0)
        {
            return HumanizeTerms(mergedTopics.Take(3));
        }

        if (analysis.Keywords.Count > 0)
        {
            return HumanizeTerms(analysis.Keywords.Take(3).Select(k => k.Keyword));
        }

        if (!string.IsNullOrWhiteSpace(analysis.Conversation.Title) && !string.Equals(analysis.Conversation.Title, "(untitled)", StringComparison.Ordinal))
        {
            return analysis.Conversation.Title;
        }

        return analysis.Category;
    }

    private static IReadOnlyList<GraphEdge> BuildConversationEdges(
        ConversationAnalysis analysis,
        CategoryFacetSelection categoryFacets,
        IReadOnlyList<string> mergedTopics,
        IReadOnlyList<CategoryPrediction> predictions,
        string topicLabel)
    {
        var edges = analysis.GraphEdges
            .Where(edge => edge.EdgeType is not "has_category"
                and not "has_secondary_category"
                and not "has_tertiary_category"
                and not "has_topic"
                and not "has_model_category"
                and not "has_model_topic"
                and not "candidate_category"
                and not "has_topic_label")
            .ToList();

        string conversationId = analysis.Conversation.ConversationId;
        edges.Add(new GraphEdge(conversationId, $"cat:{categoryFacets.Primary.Category}", "has_category", Math.Max(1.0, categoryFacets.Primary.Score)));

        if (categoryFacets.Secondary is not null)
        {
            edges.Add(new GraphEdge(
                conversationId,
                $"cat:{categoryFacets.Secondary.Category}",
                "has_secondary_category",
                categoryFacets.Secondary.Score));
        }

        if (categoryFacets.Tertiary is not null)
        {
            edges.Add(new GraphEdge(
                conversationId,
                $"cat:{categoryFacets.Tertiary.Category}",
                "has_tertiary_category",
                categoryFacets.Tertiary.Score));
        }

        foreach (var prediction in predictions.Where(prediction =>
                     !prediction.IsSelected
                     && !string.Equals(prediction.Category, categoryFacets.Secondary?.Category, StringComparison.OrdinalIgnoreCase)
                     && !string.Equals(prediction.Category, categoryFacets.Tertiary?.Category, StringComparison.OrdinalIgnoreCase)))
        {
            edges.Add(new GraphEdge(conversationId, $"cat:{prediction.Category}", "candidate_category", prediction.Score));
        }

        foreach (var topic in mergedTopics)
        {
            edges.Add(new GraphEdge(conversationId, $"topic:{topic}", "has_topic", 1.0));
        }

        if (!string.IsNullOrWhiteSpace(topicLabel))
        {
            edges.Add(new GraphEdge(conversationId, $"topiclabel:{topicLabel}", "has_topic_label", 1.0));
        }

        return DeduplicateEdges(edges);
    }

    private static IReadOnlyList<GraphEdge> BuildClusterEdges(IReadOnlyList<TopicCluster> clusters)
    {
        var edges = new List<GraphEdge>();
        foreach (var cluster in clusters)
        {
            foreach (var conversationId in cluster.ConversationIds)
            {
                edges.Add(new GraphEdge(conversationId, cluster.ClusterId, "belongs_to_cluster", 1.0, GlobalScopeId));
            }

            if (!string.IsNullOrWhiteSpace(cluster.PrimaryCategory))
            {
                edges.Add(new GraphEdge(cluster.ClusterId, $"cat:{cluster.PrimaryCategory}", "cluster_category", 1.0, GlobalScopeId));
            }

            foreach (var keyword in cluster.RepresentativeKeywords.Take(5))
            {
                edges.Add(new GraphEdge(cluster.ClusterId, $"kw:{keyword}", "cluster_keyword", 1.0, GlobalScopeId));
            }
        }

        return edges;
    }

    private static IReadOnlyList<GraphEdge> BuildCategoryCommunityEdges(IReadOnlyList<CategoryCommunity> communities)
    {
        var edges = new List<GraphEdge>();
        foreach (var community in communities)
        {
            foreach (var conversationId in community.ConversationIds)
            {
                edges.Add(new GraphEdge(conversationId, community.CommunityId, "belongs_to_category_community", 1.0, GlobalScopeId));
            }

            foreach (var category in community.Categories.Take(5))
            {
                edges.Add(new GraphEdge(community.CommunityId, $"cat:{category}", "community_category", 1.0, GlobalScopeId));
            }

            foreach (var keyword in community.RepresentativeKeywords.Take(5))
            {
                edges.Add(new GraphEdge(community.CommunityId, $"kw:{keyword}", "community_keyword", 1.0, GlobalScopeId));
            }
        }

        return edges;
    }

    private static List<GraphEdge> DeduplicateEdges(IEnumerable<GraphEdge> edges)
    {
        return edges
            .GroupBy(edge => (edge.ScopeId, edge.FromId, edge.ToId, edge.EdgeType), StringTupleComparer.Ordinal)
            .Select(group => group
                .OrderByDescending(edge => edge.Weight)
                .ThenBy(edge => edge.MetadataJson, StringComparer.Ordinal)
                .First())
            .ToList();
    }

    private static string NormalizeCategory(string? category)
        => string.IsNullOrWhiteSpace(category) ? FallbackCategory : category.Trim();

    private static string HumanizeTerms(IEnumerable<string> values)
        => string.Join(" / ", values.Select(HumanizeTerm).Where(value => !string.IsNullOrWhiteSpace(value)));

    private static string HumanizeTerm(string value)
    {
        string normalized = value.Replace('_', ' ').Replace('-', ' ').Trim();
        if (normalized.Length == 0)
        {
            return normalized;
        }

        return CultureInfo.InvariantCulture.TextInfo.ToTitleCase(normalized);
    }

    private sealed record CategoryFacetSelection(
        CategoryPrediction Primary,
        CategoryPrediction? Secondary,
        CategoryPrediction? Tertiary);
}

static class SimilarityGraphBuilder
{
    private const string GlobalScopeId = "global";

    public static IReadOnlyList<GraphEdge> Build(
        IReadOnlyList<ConversationAnalysis> analyses,
        double similarityThreshold,
        int maxSimilarNeighbors)
    {
        var candidates = new Dictionary<string, List<(string NeighborId, double Score)>>(StringComparer.Ordinal);
        foreach (var analysis in analyses)
        {
            candidates[analysis.Conversation.ConversationId] = new List<(string NeighborId, double Score)>();
        }

        for (int i = 0; i < analyses.Count; i++)
        {
            var left = analyses[i];
            if (left.Embedding is null || left.Embedding.Dimension == 0)
            {
                continue;
            }

            for (int j = i + 1; j < analyses.Count; j++)
            {
                var right = analyses[j];
                if (right.Embedding is null || right.Embedding.Dimension == 0)
                {
                    continue;
                }

                double similarity = CosineSimilarity(left.Embedding.Values, right.Embedding.Values);
                if (similarity < similarityThreshold)
                {
                    continue;
                }

                candidates[left.Conversation.ConversationId].Add((right.Conversation.ConversationId, similarity));
                candidates[right.Conversation.ConversationId].Add((left.Conversation.ConversationId, similarity));
            }
        }

        var selectedPairs = new Dictionary<(string LeftId, string RightId), double>(ConversationPairComparer.Ordinal);
        foreach (var conversation in candidates)
        {
            foreach (var neighbor in conversation.Value
                .OrderByDescending(item => item.Score)
                .Take(Math.Max(1, maxSimilarNeighbors)))
            {
                var pair = CanonicalPair(conversation.Key, neighbor.NeighborId);
                selectedPairs[pair] = selectedPairs.TryGetValue(pair, out double existing)
                    ? Math.Max(existing, neighbor.Score)
                    : neighbor.Score;
            }
        }

        return selectedPairs
            .OrderByDescending(pair => pair.Value)
            .ThenBy(pair => pair.Key.LeftId, StringComparer.Ordinal)
            .ThenBy(pair => pair.Key.RightId, StringComparer.Ordinal)
            .Select(pair => new GraphEdge(pair.Key.LeftId, pair.Key.RightId, "similar_to", Math.Round(pair.Value, 4), GlobalScopeId))
            .ToList();
    }

    private static (string LeftId, string RightId) CanonicalPair(string leftId, string rightId)
        => StringComparer.Ordinal.Compare(leftId, rightId) <= 0
            ? (leftId, rightId)
            : (rightId, leftId);

    private static double CosineSimilarity(float[] left, float[] right)
    {
        int length = Math.Min(left.Length, right.Length);
        if (length == 0)
        {
            return 0;
        }

        double dot = 0;
        double leftNorm = 0;
        double rightNorm = 0;
        for (int i = 0; i < length; i++)
        {
            dot += left[i] * right[i];
            leftNorm += left[i] * left[i];
            rightNorm += right[i] * right[i];
        }

        if (leftNorm == 0 || rightNorm == 0)
        {
            return 0;
        }

        return dot / (Math.Sqrt(leftNorm) * Math.Sqrt(rightNorm));
    }
}

static class CategoryComparisonWriter
{
    public static Task WriteIfEnabledAsync(
        IReadOnlyList<ConversationAnalysis> analyses,
        AppArgs arguments,
        string graphDirectory,
        CancellationToken cancellationToken = default)
    {
        if (!arguments.CompareCategories)
        {
            return Task.CompletedTask;
        }

        using var predictor = CorpusIndexingPipeline.TryCreateMlNetCategoryPredictor(arguments, writeWarnings: false);
        if (predictor is null)
        {
            Console.Error.WriteLine("[warn] Category comparison skipped because no usable ML.NET model was configured.");
            return Task.CompletedTask;
        }

        Directory.CreateDirectory(graphDirectory);

        string baselineSource = arguments.UseOllamaAnalysis ? "ollama" : "heuristic";
        var comparisons = new List<CategoryComparisonRow>(analyses.Count);
        foreach (var analysis in analyses)
        {
            cancellationToken.ThrowIfCancellationRequested();

            string corpus = ConversationIndexText.BuildCorpus(analysis.Conversation);
            string baselineCategory = NormalizeCategory(analysis.Category);
            var mlPrediction = predictor.Predict(corpus);
            string? mlNetCategory = string.IsNullOrWhiteSpace(mlPrediction?.Category)
                ? null
                : NormalizeCategory(mlPrediction.Category);

            bool agrees = mlNetCategory is not null
                && string.Equals(baselineCategory, mlNetCategory, StringComparison.OrdinalIgnoreCase);

            string selectedSource;
            string selectedCategory;
            if (string.Equals(arguments.CategoryProviderMode, "heuristic", StringComparison.OrdinalIgnoreCase) || mlNetCategory is null)
            {
                selectedSource = baselineSource;
                selectedCategory = baselineCategory;
            }
            else
            {
                selectedSource = "mlnet";
                selectedCategory = mlNetCategory;
            }

            comparisons.Add(new CategoryComparisonRow(
                analysis.Conversation.ConversationId,
                analysis.Conversation.Title,
                baselineSource,
                baselineCategory,
                mlNetCategory,
                mlPrediction?.Score,
                agrees,
                selectedSource,
                selectedCategory));
        }

        int mlNetCount = comparisons.Count(row => row.MlNetCategory is not null);
        int agreementCount = comparisons.Count(row => row.Agrees);
        var disagreements = comparisons
            .Where(row => row.MlNetCategory is not null && !row.Agrees)
            .OrderBy(row => row.Title, StringComparer.OrdinalIgnoreCase)
            .Take(100)
            .ToList();

        var report = new CategoryComparisonReport(
            arguments.CategoryProviderMode,
            baselineSource,
            analyses.Count,
            mlNetCount,
            agreementCount,
            mlNetCount == 0 ? 0 : Math.Round((double)agreementCount / mlNetCount, 4),
            comparisons.Count(row => row.MlNetCategory is not null && !row.Agrees),
            disagreements,
            comparisons);

        File.WriteAllText(
            Path.Combine(graphDirectory, "category-comparison.json"),
            JsonSerializer.Serialize(report, JsonOptions.WriteIndented),
            new UTF8Encoding(false));

        Console.WriteLine(
            $"Category comparison: {baselineSource} vs ML.NET agreements={agreementCount:N0}/{mlNetCount:N0} " +
            $"({(mlNetCount == 0 ? 0 : (double)agreementCount / mlNetCount).ToString("P1", CultureInfo.InvariantCulture)})");

        return Task.CompletedTask;
    }

    private static string NormalizeCategory(string? category)
        => string.IsNullOrWhiteSpace(category) ? "Uncategorized" : category.Trim();

    private sealed record CategoryComparisonRow(
        string ConversationId,
        string Title,
        string BaselineSource,
        string BaselineCategory,
        string? MlNetCategory,
        double? MlNetScore,
        bool Agrees,
        string SelectedSource,
        string SelectedCategory);

    private sealed record CategoryComparisonReport(
        string CategoryProviderMode,
        string BaselineSource,
        int ConversationCount,
        int MlNetPredictionCount,
        int AgreementCount,
        double AgreementRate,
        int DisagreementCount,
        IReadOnlyList<CategoryComparisonRow> Disagreements,
        IReadOnlyList<CategoryComparisonRow> Conversations);
}

static class EmbeddingComparisonWriter
{
    public static Task WriteIfEnabledAsync(
        IReadOnlyList<ConversationAnalysis> analyses,
        AppArgs arguments,
        string graphDirectory,
        CancellationToken cancellationToken = default)
    {
        if (!arguments.CompareEmbeddings && !arguments.CompareOnnxTokenizers)
        {
            return Task.CompletedTask;
        }

        using var activeOnnxProvider = TryCreateOnnxProvider(arguments);
        if (activeOnnxProvider is null)
        {
            Console.Error.WriteLine("[warn] Embedding comparison skipped because no usable ONNX model was configured.");
            return Task.CompletedTask;
        }

        Directory.CreateDirectory(graphDirectory);

        var corpora = analyses.ToDictionary(
            analysis => analysis.Conversation.ConversationId,
            analysis => ConversationIndexText.BuildCorpus(analysis.Conversation),
            StringComparer.Ordinal);

        if (arguments.CompareEmbeddings)
        {
            using var hashProvider = new HashingEmbeddingProvider(arguments.HashEmbeddingDimensions);
            var hashSnapshot = BuildSnapshot(analyses, corpora, hashProvider, arguments, cancellationToken);
            var onnxSnapshot = BuildSnapshot(analyses, corpora, activeOnnxProvider, arguments, cancellationToken);
            var comparison = BuildComparison(analyses, hashSnapshot, onnxSnapshot, arguments);

            File.WriteAllText(
                Path.Combine(graphDirectory, "similarity-hash.json"),
                JsonSerializer.Serialize(hashSnapshot, JsonOptions.WriteIndented),
                new UTF8Encoding(false));

            File.WriteAllText(
                Path.Combine(graphDirectory, "similarity-onnx.json"),
                JsonSerializer.Serialize(onnxSnapshot, JsonOptions.WriteIndented),
                new UTF8Encoding(false));

            File.WriteAllText(
                Path.Combine(graphDirectory, "embedding-comparison.json"),
                JsonSerializer.Serialize(comparison, JsonOptions.WriteIndented),
                new UTF8Encoding(false));

            Console.WriteLine(
                $"Embedding comparison: {hashSnapshot.Provider} threshold={hashSnapshot.Threshold.ToString("0.###", CultureInfo.InvariantCulture)} edges={hashSnapshot.EdgeCount:N0}, " +
                $"{onnxSnapshot.Provider} threshold={onnxSnapshot.Threshold.ToString("0.###", CultureInfo.InvariantCulture)} edges={onnxSnapshot.EdgeCount:N0}, overlap={comparison.OverlapEdgeCount:N0}, " +
                $"jaccard={comparison.EdgeJaccard.ToString("0.###", CultureInfo.InvariantCulture)}");
        }

        if (arguments.CompareOnnxTokenizers)
        {
            using var legacyProvider = TryCreateOnnxProvider(arguments, "legacy");
            using var mlTokenizerProvider = TryCreateOnnxProvider(arguments, "mltokenizer");
            if (legacyProvider is null || mlTokenizerProvider is null)
            {
                Console.Error.WriteLine("[warn] ONNX tokenizer comparison skipped because one of the ONNX tokenizer providers could not be initialized.");
                return Task.CompletedTask;
            }

            var legacySnapshot = BuildSnapshot(analyses, corpora, legacyProvider, arguments, cancellationToken);
            var mlTokenizerSnapshot = BuildSnapshot(analyses, corpora, mlTokenizerProvider, arguments, cancellationToken);
            var tokenizerComparison = BuildComparison(analyses, legacySnapshot, mlTokenizerSnapshot, arguments);

            File.WriteAllText(
                Path.Combine(graphDirectory, "similarity-onnx-legacy.json"),
                JsonSerializer.Serialize(legacySnapshot, JsonOptions.WriteIndented),
                new UTF8Encoding(false));

            File.WriteAllText(
                Path.Combine(graphDirectory, "similarity-onnx-mltokenizer.json"),
                JsonSerializer.Serialize(mlTokenizerSnapshot, JsonOptions.WriteIndented),
                new UTF8Encoding(false));

            File.WriteAllText(
                Path.Combine(graphDirectory, "embedding-comparison-onnx-tokenizers.json"),
                JsonSerializer.Serialize(tokenizerComparison, JsonOptions.WriteIndented),
                new UTF8Encoding(false));

            Console.WriteLine(
                $"ONNX tokenizer comparison: {legacySnapshot.Provider} edges={legacySnapshot.EdgeCount:N0}, " +
                $"{mlTokenizerSnapshot.Provider} edges={mlTokenizerSnapshot.EdgeCount:N0}, overlap={tokenizerComparison.OverlapEdgeCount:N0}, " +
                $"jaccard={tokenizerComparison.EdgeJaccard.ToString("0.###", CultureInfo.InvariantCulture)}");
        }

        return Task.CompletedTask;
    }

    private static ProviderSimilaritySnapshot BuildSnapshot(
        IReadOnlyList<ConversationAnalysis> analyses,
        IReadOnlyDictionary<string, string> corpora,
        IEmbeddingProvider provider,
        AppArgs arguments,
        CancellationToken cancellationToken)
    {
        var embedded = new List<ConversationAnalysis>(analyses.Count);
        foreach (var analysis in analyses)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string corpus = corpora[analysis.Conversation.ConversationId];
            var embedding = provider.Embed(analysis.Conversation.ConversationId, corpus);
            embedded.Add(analysis with { Embedding = embedding });
        }

        double threshold = arguments.GetSimilarityThresholdForProvider(provider.Description);

        var edges = SimilarityGraphBuilder.Build(
            embedded,
            threshold,
            arguments.MaxSimilarNeighbors);

        var titles = analyses.ToDictionary(
            analysis => analysis.Conversation.ConversationId,
            analysis => analysis.Conversation.Title,
            StringComparer.Ordinal);

        var neighbors = BuildNeighborSets(titles, edges);
        var conversations = analyses
            .Select(analysis => new ConversationNeighborSet(
                analysis.Conversation.ConversationId,
                analysis.Conversation.Title,
                neighbors.GetValueOrDefault(analysis.Conversation.ConversationId) ?? Array.Empty<NeighborScore>()))
            .ToList();

        return new ProviderSimilaritySnapshot(provider.Description, threshold, edges.Count, edges, conversations);
    }

    private static EmbeddingComparisonReport BuildComparison(
        IReadOnlyList<ConversationAnalysis> analyses,
        ProviderSimilaritySnapshot baseline,
        ProviderSimilaritySnapshot candidate,
        AppArgs arguments)
    {
        var baselinePairs = BuildEdgeSet(baseline.Edges);
        var candidatePairs = BuildEdgeSet(candidate.Edges);
        int overlap = baselinePairs.Intersect(candidatePairs).Count();
        int union = baselinePairs.Union(candidatePairs).Count();

        var baselineNeighborMap = baseline.Conversations.ToDictionary(
            conversation => conversation.ConversationId,
            conversation => conversation.Neighbors,
            StringComparer.Ordinal);

        var candidateNeighborMap = candidate.Conversations.ToDictionary(
            conversation => conversation.ConversationId,
            conversation => conversation.Neighbors,
            StringComparer.Ordinal);

        var allDifferences = analyses
            .Select(analysis => BuildConversationComparison(
                analysis.Conversation.ConversationId,
                analysis.Conversation.Title,
                baselineNeighborMap.GetValueOrDefault(analysis.Conversation.ConversationId) ?? Array.Empty<NeighborScore>(),
                candidateNeighborMap.GetValueOrDefault(analysis.Conversation.ConversationId) ?? Array.Empty<NeighborScore>()))
            .Where(item => item.BaselineOnlyNeighbors.Count > 0 || item.CandidateOnlyNeighbors.Count > 0)
            .OrderByDescending(item => item.BaselineOnlyNeighbors.Count + item.CandidateOnlyNeighbors.Count)
            .ThenBy(item => item.SharedNeighborCount)
            .ThenBy(item => item.Title, StringComparer.OrdinalIgnoreCase)
            .ToList();

        var differences = allDifferences
            .Take(50)
            .ToList();

        return new EmbeddingComparisonReport(
            baseline.Provider,
            candidate.Provider,
            analyses.Count,
            baseline.Threshold,
            candidate.Threshold,
            arguments.MaxSimilarNeighbors,
            baseline.EdgeCount,
            candidate.EdgeCount,
            overlap,
            union == 0 ? 1.0 : Math.Round((double)overlap / union, 4),
            allDifferences.Count,
            differences);
    }

    private static ConversationNeighborComparison BuildConversationComparison(
        string conversationId,
        string title,
        IReadOnlyList<NeighborScore> baselineNeighbors,
        IReadOnlyList<NeighborScore> candidateNeighbors)
    {
        var baselineMap = baselineNeighbors.ToDictionary(neighbor => neighbor.ConversationId, neighbor => neighbor, StringComparer.Ordinal);
        var candidateMap = candidateNeighbors.ToDictionary(neighbor => neighbor.ConversationId, neighbor => neighbor, StringComparer.Ordinal);

        var baselineOnly = baselineNeighbors
            .Where(neighbor => !candidateMap.ContainsKey(neighbor.ConversationId))
            .ToList();

        var candidateOnly = candidateNeighbors
            .Where(neighbor => !baselineMap.ContainsKey(neighbor.ConversationId))
            .ToList();

        int shared = baselineNeighbors.Count(neighbor => candidateMap.ContainsKey(neighbor.ConversationId));

        return new ConversationNeighborComparison(
            conversationId,
            title,
            baselineNeighbors.Count,
            candidateNeighbors.Count,
            shared,
            baselineOnly,
            candidateOnly,
            baselineNeighbors,
            candidateNeighbors);
    }

    private static HashSet<string> BuildEdgeSet(IReadOnlyList<GraphEdge> edges)
    {
        var pairs = new HashSet<string>(StringComparer.Ordinal);
        foreach (var edge in edges.Where(edge => edge.EdgeType == "similar_to"))
        {
            string left = StringComparer.Ordinal.Compare(edge.FromId, edge.ToId) <= 0 ? edge.FromId : edge.ToId;
            string right = StringComparer.Ordinal.Compare(edge.FromId, edge.ToId) <= 0 ? edge.ToId : edge.FromId;
            pairs.Add($"{left}|{right}");
        }

        return pairs;
    }

    private static Dictionary<string, IReadOnlyList<NeighborScore>> BuildNeighborSets(
        IReadOnlyDictionary<string, string> titles,
        IReadOnlyList<GraphEdge> edges)
    {
        var neighbors = titles.Keys.ToDictionary(
            conversationId => conversationId,
            _ => new List<NeighborScore>(),
            StringComparer.Ordinal);

        foreach (var edge in edges.Where(edge => edge.EdgeType == "similar_to"))
        {
            if (neighbors.TryGetValue(edge.FromId, out var fromNeighbors))
            {
                fromNeighbors.Add(new NeighborScore(
                    edge.ToId,
                    titles.GetValueOrDefault(edge.ToId) ?? edge.ToId,
                    edge.Weight));
            }

            if (neighbors.TryGetValue(edge.ToId, out var toNeighbors))
            {
                toNeighbors.Add(new NeighborScore(
                    edge.FromId,
                    titles.GetValueOrDefault(edge.FromId) ?? edge.FromId,
                    edge.Weight));
            }
        }

        return neighbors.ToDictionary(
            pair => pair.Key,
            pair => (IReadOnlyList<NeighborScore>)pair.Value
                .OrderByDescending(neighbor => neighbor.Score)
                .ThenBy(neighbor => neighbor.Title, StringComparer.OrdinalIgnoreCase)
                .ToList(),
            StringComparer.Ordinal);
    }

    private static IEmbeddingProvider? TryCreateOnnxProvider(AppArgs arguments, string? tokenizerModeOverride = null)
    {
        if (string.IsNullOrWhiteSpace(arguments.EmbeddingModelPath) || string.IsNullOrWhiteSpace(arguments.EmbeddingVocabularyPath))
        {
            return null;
        }

        if (!File.Exists(arguments.EmbeddingModelPath) || !File.Exists(arguments.EmbeddingVocabularyPath))
        {
            return null;
        }

        if (LocalModelFileInspector.IsGitLfsPointer(arguments.EmbeddingModelPath))
        {
            return null;
        }

        try
        {
            return CorpusIndexingPipeline.CreateOnnxEmbeddingProvider(arguments, tokenizerModeOverride);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[warn] Embedding comparison could not initialize the ONNX model: {ex.Message}");
            return null;
        }
    }

    private sealed record ProviderSimilaritySnapshot(
        string Provider,
        double Threshold,
        int EdgeCount,
        IReadOnlyList<GraphEdge> Edges,
        IReadOnlyList<ConversationNeighborSet> Conversations);

    private sealed record ConversationNeighborSet(
        string ConversationId,
        string Title,
        IReadOnlyList<NeighborScore> Neighbors);

    private sealed record NeighborScore(
        string ConversationId,
        string Title,
        double Score);

    private sealed record ConversationNeighborComparison(
        string ConversationId,
        string Title,
        int BaselineNeighborCount,
        int CandidateNeighborCount,
        int SharedNeighborCount,
        IReadOnlyList<NeighborScore> BaselineOnlyNeighbors,
        IReadOnlyList<NeighborScore> CandidateOnlyNeighbors,
        IReadOnlyList<NeighborScore> BaselineNeighbors,
        IReadOnlyList<NeighborScore> CandidateNeighbors);

    private sealed record EmbeddingComparisonReport(
        string BaselineProvider,
        string CandidateProvider,
        int ConversationCount,
        double BaselineThreshold,
        double CandidateThreshold,
        int MaxSimilarNeighbors,
        int BaselineEdgeCount,
        int CandidateEdgeCount,
        int OverlapEdgeCount,
        double EdgeJaccard,
        int ConversationsWithDifferences,
        IReadOnlyList<ConversationNeighborComparison> TopDifferences);
}

static class PerspectiveSummaryWriter
{
    public static async Task WriteIfAvailableAsync(string graphDirectory)
    {
        string categoryComparisonPath = Path.Combine(graphDirectory, "category-comparison.json");
        string hashSimilarityPath = Path.Combine(graphDirectory, "similarity-hash.json");
        string onnxSimilarityPath = Path.Combine(graphDirectory, "similarity-onnx.json");

        CategoryComparisonFile? categoryComparison = await ReadJsonAsync<CategoryComparisonFile>(categoryComparisonPath);
        SimilaritySnapshotFile? hashSnapshot = await ReadJsonAsync<SimilaritySnapshotFile>(hashSimilarityPath);
        SimilaritySnapshotFile? onnxSnapshot = await ReadJsonAsync<SimilaritySnapshotFile>(onnxSimilarityPath);

        if (categoryComparison is null && hashSnapshot is null && onnxSnapshot is null)
        {
            return;
        }

        var allConversationIds = new HashSet<string>(StringComparer.Ordinal);
        foreach (var row in categoryComparison?.Conversations ?? Array.Empty<CategoryComparisonConversation>())
        {
            allConversationIds.Add(row.ConversationId);
        }

        foreach (var row in hashSnapshot?.Conversations ?? Array.Empty<SimilarityConversation>())
        {
            allConversationIds.Add(row.ConversationId);
        }

        foreach (var row in onnxSnapshot?.Conversations ?? Array.Empty<SimilarityConversation>())
        {
            allConversationIds.Add(row.ConversationId);
        }

        var categoryMap = (categoryComparison?.Conversations ?? Array.Empty<CategoryComparisonConversation>())
            .ToDictionary(row => row.ConversationId, StringComparer.Ordinal);
        var hashMap = (hashSnapshot?.Conversations ?? Array.Empty<SimilarityConversation>())
            .ToDictionary(row => row.ConversationId, StringComparer.Ordinal);
        var onnxMap = (onnxSnapshot?.Conversations ?? Array.Empty<SimilarityConversation>())
            .ToDictionary(row => row.ConversationId, StringComparer.Ordinal);

        var rows = allConversationIds
            .Select(conversationId => BuildRow(
                conversationId,
                categoryMap.GetValueOrDefault(conversationId),
                hashMap.GetValueOrDefault(conversationId),
                onnxMap.GetValueOrDefault(conversationId)))
            .OrderByDescending(row => row.PerspectiveScore)
            .ThenByDescending(row => row.CategoryDisagrees ? 1 : 0)
            .ThenByDescending(row => row.EmbeddingDifferenceCount)
            .ThenBy(row => row.Title, StringComparer.OrdinalIgnoreCase)
            .ToList();

        int categoryDisagreements = rows.Count(row => row.CategoryDisagrees);
        int embeddingDifferences = rows.Count(row => row.EmbeddingDifferenceCount > 0);
        int strongSignals = rows.Count(row => row.CategoryDisagrees || row.EmbeddingDifferenceCount >= 4);

        var report = new PerspectiveSummaryReport(
            rows.Count,
            categoryComparison?.CategoryProviderMode,
            categoryComparison?.BaselineSource,
            hashSnapshot?.Provider,
            hashSnapshot?.Threshold,
            onnxSnapshot?.Provider,
            onnxSnapshot?.Threshold,
            categoryDisagreements,
            embeddingDifferences,
            strongSignals,
            rows.Take(100).ToList(),
            rows);

        File.WriteAllText(
            Path.Combine(graphDirectory, "perspective-summary.json"),
            JsonSerializer.Serialize(report, JsonOptions.WriteIndented),
            new UTF8Encoding(false));

        File.WriteAllText(
            Path.Combine(graphDirectory, "perspective-summary.md"),
            BuildMarkdown(report),
            new UTF8Encoding(false));

        Console.WriteLine(
            $"Perspective summary: category disagreements={categoryDisagreements:N0}, " +
            $"embedding divergences={embeddingDifferences:N0}, strong signals={strongSignals:N0}");
    }

    private static PerspectiveSummaryRow BuildRow(
        string conversationId,
        CategoryComparisonConversation? category,
        SimilarityConversation? hash,
        SimilarityConversation? onnx)
    {
        string title = category?.Title
            ?? hash?.Title
            ?? onnx?.Title
            ?? conversationId;

        var hashNeighbors = hash?.Neighbors ?? Array.Empty<SimilarityNeighbor>();
        var onnxNeighbors = onnx?.Neighbors ?? Array.Empty<SimilarityNeighbor>();
        var onnxNeighborIds = onnxNeighbors.Select(neighbor => neighbor.ConversationId).ToHashSet(StringComparer.Ordinal);
        var hashNeighborIds = hashNeighbors.Select(neighbor => neighbor.ConversationId).ToHashSet(StringComparer.Ordinal);

        var hashOnly = hashNeighbors
            .Where(neighbor => !onnxNeighborIds.Contains(neighbor.ConversationId))
            .Take(5)
            .ToList();

        var onnxOnly = onnxNeighbors
            .Where(neighbor => !hashNeighborIds.Contains(neighbor.ConversationId))
            .Take(5)
            .ToList();

        int shared = hashNeighbors.Count(neighbor => onnxNeighborIds.Contains(neighbor.ConversationId));
        int embeddingDifferenceCount = (hashNeighbors.Count - shared) + (onnxNeighbors.Count - shared);
        bool categoryDisagrees = category is not null
            && !string.IsNullOrWhiteSpace(category.MlNetCategory)
            && !category.Agrees;
        int perspectiveScore = (categoryDisagrees ? 100 : 0) + embeddingDifferenceCount;

        return new PerspectiveSummaryRow(
            conversationId,
            title,
            category?.SelectedSource,
            category?.SelectedCategory,
            category?.BaselineCategory,
            category?.MlNetCategory,
            categoryDisagrees,
            hashNeighbors.Count,
            onnxNeighbors.Count,
            shared,
            hashNeighbors.Count - shared,
            onnxNeighbors.Count - shared,
            embeddingDifferenceCount,
            perspectiveScore,
            hashOnly.Select(neighbor => new PerspectiveNeighbor(neighbor.ConversationId, neighbor.Title, neighbor.Score)).ToList(),
            onnxOnly.Select(neighbor => new PerspectiveNeighbor(neighbor.ConversationId, neighbor.Title, neighbor.Score)).ToList());
    }

    private static string BuildMarkdown(PerspectiveSummaryReport report)
    {
        var sb = new StringBuilder();
        sb.AppendLine("# Perspective Summary");
        sb.AppendLine();
        sb.AppendLine("## Overview");
        sb.AppendLine();
        sb.AppendLine($"- Conversations: {report.ConversationCount:N0}");
        sb.AppendLine($"- Category disagreements: {report.CategoryDisagreementCount:N0}");
        sb.AppendLine($"- Embedding divergences: {report.EmbeddingDivergenceCount:N0}");
        sb.AppendLine($"- Strong signals: {report.StrongSignalCount:N0}");

        if (!string.IsNullOrWhiteSpace(report.HashProvider) || !string.IsNullOrWhiteSpace(report.OnnxProvider))
        {
            sb.AppendLine();
            sb.AppendLine("## Embedding Perspectives");
            sb.AppendLine();
            if (!string.IsNullOrWhiteSpace(report.HashProvider))
            {
                sb.AppendLine($"- Hash: `{report.HashProvider}` at threshold `{report.HashThreshold?.ToString("0.###", CultureInfo.InvariantCulture) ?? "n/a"}`");
            }

            if (!string.IsNullOrWhiteSpace(report.OnnxProvider))
            {
                sb.AppendLine($"- ONNX: `{report.OnnxProvider}` at threshold `{report.OnnxThreshold?.ToString("0.###", CultureInfo.InvariantCulture) ?? "n/a"}`");
            }
        }

        sb.AppendLine();
        sb.AppendLine("## Top Signals");
        sb.AppendLine();
        int rank = 1;
        foreach (var row in report.TopConversations.Take(25))
        {
            sb.AppendLine($"{rank}. {row.Title} (`{row.ConversationId}`)");
            if (!string.IsNullOrWhiteSpace(row.SelectedCategory))
            {
                sb.AppendLine($"   Selected category: `{row.SelectedCategory}` via `{row.SelectedSource ?? "unknown"}`");
            }

            if (row.CategoryDisagrees)
            {
                sb.AppendLine($"   Category disagreement: heuristic `{row.BaselineCategory}` vs ML.NET `{row.MlNetCategory}`");
            }

            sb.AppendLine($"   Embeddings: hash `{row.HashNeighborCount}`, onnx `{row.OnnxNeighborCount}`, shared `{row.SharedNeighborCount}`, hash-only `{row.HashOnlyNeighborCount}`, onnx-only `{row.OnnxOnlyNeighborCount}`");

            if (row.HashOnlyExamples.Count > 0)
            {
                sb.AppendLine($"   Hash-only examples: {string.Join("; ", row.HashOnlyExamples.Select(FormatNeighbor))}");
            }

            if (row.OnnxOnlyExamples.Count > 0)
            {
                sb.AppendLine($"   ONNX-only examples: {string.Join("; ", row.OnnxOnlyExamples.Select(FormatNeighbor))}");
            }

            rank++;
        }

        return sb.ToString();
    }

    private static string FormatNeighbor(PerspectiveNeighbor neighbor)
        => $"{neighbor.Title} ({neighbor.Score.ToString("0.###", CultureInfo.InvariantCulture)})";

    private static async Task<T?> ReadJsonAsync<T>(string path)
    {
        if (!File.Exists(path))
        {
            return default;
        }

        await using var stream = File.OpenRead(path);
        return await JsonSerializer.DeserializeAsync<T>(stream, JsonOptions.Default);
    }

    private sealed record CategoryComparisonFile(
        string CategoryProviderMode,
        string BaselineSource,
        IReadOnlyList<CategoryComparisonConversation> Conversations);

    private sealed record CategoryComparisonConversation(
        string ConversationId,
        string Title,
        string BaselineSource,
        string BaselineCategory,
        string? MlNetCategory,
        double? MlNetScore,
        bool Agrees,
        string SelectedSource,
        string SelectedCategory);

    private sealed record SimilaritySnapshotFile(
        string Provider,
        double Threshold,
        IReadOnlyList<SimilarityConversation> Conversations);

    private sealed record SimilarityConversation(
        string ConversationId,
        string Title,
        IReadOnlyList<SimilarityNeighbor> Neighbors);

    private sealed record SimilarityNeighbor(
        string ConversationId,
        string Title,
        double Score);

    private sealed record PerspectiveSummaryReport(
        int ConversationCount,
        string? CategoryProviderMode,
        string? CategoryBaselineSource,
        string? HashProvider,
        double? HashThreshold,
        string? OnnxProvider,
        double? OnnxThreshold,
        int CategoryDisagreementCount,
        int EmbeddingDivergenceCount,
        int StrongSignalCount,
        IReadOnlyList<PerspectiveSummaryRow> TopConversations,
        IReadOnlyList<PerspectiveSummaryRow> Conversations);

    private sealed record PerspectiveSummaryRow(
        string ConversationId,
        string Title,
        string? SelectedSource,
        string? SelectedCategory,
        string? BaselineCategory,
        string? MlNetCategory,
        bool CategoryDisagrees,
        int HashNeighborCount,
        int OnnxNeighborCount,
        int SharedNeighborCount,
        int HashOnlyNeighborCount,
        int OnnxOnlyNeighborCount,
        int EmbeddingDifferenceCount,
        int PerspectiveScore,
        IReadOnlyList<PerspectiveNeighbor> HashOnlyExamples,
        IReadOnlyList<PerspectiveNeighbor> OnnxOnlyExamples);

    private sealed record PerspectiveNeighbor(
        string ConversationId,
        string Title,
        double Score);
}

static class TopicClusterBuilder
{
    public static async Task<IReadOnlyList<TopicCluster>> BuildAsync(
        IReadOnlyList<ConversationAnalysis> analyses,
        IReadOnlyList<GraphEdge> similarityEdges,
        IClusterLabeler? clusterLabeler,
        CancellationToken cancellationToken = default)
    {
        var byConversationId = analyses.ToDictionary(analysis => analysis.Conversation.ConversationId, StringComparer.Ordinal);
        var adjacency = analyses.ToDictionary(
            analysis => analysis.Conversation.ConversationId,
            _ => new HashSet<string>(StringComparer.Ordinal),
            StringComparer.Ordinal);

        foreach (var edge in similarityEdges.Where(edge => edge.EdgeType == "similar_to"))
        {
            if (adjacency.TryGetValue(edge.FromId, out var left))
            {
                left.Add(edge.ToId);
            }

            if (adjacency.TryGetValue(edge.ToId, out var right))
            {
                right.Add(edge.FromId);
            }
        }

        var visited = new HashSet<string>(StringComparer.Ordinal);
        var clusters = new List<TopicCluster>();
        int clusterNumber = 1;

        foreach (var analysis in analyses.OrderBy(analysis => analysis.Conversation.ConversationId, StringComparer.Ordinal))
        {
            string conversationId = analysis.Conversation.ConversationId;
            if (!visited.Add(conversationId))
            {
                continue;
            }

            var queue = new Queue<string>();
            var componentIds = new List<string>();
            queue.Enqueue(conversationId);

            while (queue.Count > 0)
            {
                string current = queue.Dequeue();
                componentIds.Add(current);

                foreach (var neighbor in adjacency[current])
                {
                    if (visited.Add(neighbor))
                    {
                        queue.Enqueue(neighbor);
                    }
                }
            }

            componentIds.Sort(StringComparer.Ordinal);
            var component = componentIds.Select(id => byConversationId[id]).ToList();
            var representativeKeywords = component
                .SelectMany(item => item.Keywords.Take(6))
                .GroupBy(item => item.Keyword, StringComparer.OrdinalIgnoreCase)
                .Select(group => new KeywordScore(group.First().Keyword, group.Sum(item => item.Score)))
                .OrderByDescending(item => item.Score)
                .ThenBy(item => item.Keyword, StringComparer.OrdinalIgnoreCase)
                .Take(8)
                .Select(item => item.Keyword)
                .ToList();

            string primaryCategory = component
                .GroupBy(item => item.Category, StringComparer.OrdinalIgnoreCase)
                .OrderByDescending(group => group.Count())
                .ThenBy(group => group.Key, StringComparer.OrdinalIgnoreCase)
                .Select(group => group.Key)
                .FirstOrDefault() ?? "Uncategorized";

            string fallbackLabel = BuildFallbackLabel(component, representativeKeywords, primaryCategory, clusterNumber);
            string? fallbackSummary = BuildFallbackSummary(component, representativeKeywords, primaryCategory);

            string label = fallbackLabel;
            string? summary = fallbackSummary;
            if (clusterLabeler is not null)
            {
                var labeled = await clusterLabeler.LabelAsync(
                    new ClusterLabelRequest(
                        component.Select(item => item.Conversation.Title).Where(title => !string.IsNullOrWhiteSpace(title)).Distinct(StringComparer.OrdinalIgnoreCase).Take(6).ToList(),
                        representativeKeywords,
                        component.Select(item => item.Category).Distinct(StringComparer.OrdinalIgnoreCase).Take(4).ToList(),
                        component.Select(item => item.ModelSummary).Where(summary => !string.IsNullOrWhiteSpace(summary)).Take(4).Cast<string>().ToList()),
                    cancellationToken);

                if (!string.IsNullOrWhiteSpace(labeled?.Label))
                {
                    label = labeled.Label!;
                }

                if (!string.IsNullOrWhiteSpace(labeled?.Summary))
                {
                    summary = labeled.Summary;
                }
            }

            clusters.Add(new TopicCluster(
                $"cluster:{clusterNumber:0000}",
                label,
                summary,
                primaryCategory,
                componentIds,
                representativeKeywords));

            clusterNumber++;
        }

        return clusters;
    }

    private static string BuildFallbackLabel(
        IReadOnlyList<ConversationAnalysis> component,
        IReadOnlyList<string> representativeKeywords,
        string primaryCategory,
        int clusterNumber)
    {
        if (component.Count == 1)
        {
            var item = component[0];
            if (!string.IsNullOrWhiteSpace(item.TopicLabel))
            {
                return item.TopicLabel!;
            }

            if (!string.IsNullOrWhiteSpace(item.Conversation.Title) && !string.Equals(item.Conversation.Title, "(untitled)", StringComparison.Ordinal))
            {
                return item.Conversation.Title;
            }
        }

        if (representativeKeywords.Count > 0)
        {
            return string.Join(" / ", representativeKeywords.Take(3).Select(keyword => CultureInfo.InvariantCulture.TextInfo.ToTitleCase(keyword.Replace('_', ' ').Replace('-', ' '))));
        }

        if (!string.IsNullOrWhiteSpace(primaryCategory))
        {
            return primaryCategory;
        }

        return $"Cluster {clusterNumber:0000}";
    }

    private static string? BuildFallbackSummary(
        IReadOnlyList<ConversationAnalysis> component,
        IReadOnlyList<string> representativeKeywords,
        string primaryCategory)
    {
        var summary = component
            .Select(item => item.ModelSummary)
            .FirstOrDefault(value => !string.IsNullOrWhiteSpace(value));

        if (!string.IsNullOrWhiteSpace(summary))
        {
            return summary;
        }

        if (representativeKeywords.Count == 0)
        {
            return primaryCategory;
        }

        return $"{primaryCategory}: {string.Join(", ", representativeKeywords.Take(5))}".Trim().TrimEnd(':');
    }
}

static class CategoryCommunityBuilder
{
    private const int MinimumAnchorSize = 3;
    private const double PrimaryWeight = 1.0;
    private const double SecondaryWeight = 0.68;
    private const double TertiaryWeight = 0.44;
    private const double MinimumMergeSimilarity = 0.46;

    public static IReadOnlyList<CategoryCommunity> Build(IReadOnlyList<ConversationAnalysis> analyses)
    {
        var signatureGroups = analyses
            .GroupBy(BuildSignatureKey, StringComparer.Ordinal)
            .Select(group => new WorkingSignatureGroup(group.Key, group.ToList()))
            .OrderByDescending(group => group.Analyses.Count)
            .ThenBy(group => group.SignatureKey, StringComparer.OrdinalIgnoreCase)
            .ToList();

        var communities = new List<WorkingCategoryCommunity>();
        var anchors = signatureGroups
            .Where(group => group.Analyses.Count >= MinimumAnchorSize)
            .ToList();

        if (anchors.Count == 0)
        {
            anchors = signatureGroups.ToList();
        }

        foreach (var anchor in anchors)
        {
            communities.Add(new WorkingCategoryCommunity(anchor));
        }

        foreach (var group in signatureGroups.Except(anchors))
        {
            var best = communities
                .Select(community => new
                {
                    Community = community,
                    Score = ScoreGroupAgainstCommunity(group.Profile, group.PrimaryCategory, community)
                })
                .OrderByDescending(item => item.Score)
                .ThenByDescending(item => item.Community.Members.Count)
                .ThenBy(item => item.Community.SeedSignatureKey, StringComparer.OrdinalIgnoreCase)
                .FirstOrDefault();

            if (best is not null && best.Score >= MinimumMergeSimilarity)
            {
                best.Community.Add(group);
            }
            else
            {
                communities.Add(new WorkingCategoryCommunity(group));
            }
        }

        return communities
            .OrderByDescending(community => community.Members.Count)
            .ThenBy(community => community.GetLabel(), StringComparer.OrdinalIgnoreCase)
            .Select((community, index) => community.ToRecord(index + 1))
            .ToList();
    }

    private static string BuildSignatureKey(ConversationAnalysis analysis)
    {
        var categories = new[] { analysis.Category, analysis.SecondaryCategory, analysis.TertiaryCategory }
            .Where(category => !string.IsNullOrWhiteSpace(category))
            .Select(category => category!)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
        return categories.Count == 0 ? AnalysisDefaults.DefaultCategory : string.Join(" | ", categories);
    }

    private static Dictionary<string, double> BuildFacetProfile(ConversationAnalysis analysis)
    {
        var weights = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
        AddWeight(weights, analysis.Category, PrimaryWeight);
        AddWeight(weights, analysis.SecondaryCategory, SecondaryWeight);
        AddWeight(weights, analysis.TertiaryCategory, TertiaryWeight);
        return weights;
    }

    private static void AddWeight(Dictionary<string, double> weights, string? category, double weight)
    {
        if (string.IsNullOrWhiteSpace(category))
        {
            return;
        }

        string key = category.Trim();
        double existing = weights.GetValueOrDefault(key);
        weights[key] = Math.Max(existing, weight);
    }

    private static double ScoreGroupAgainstCommunity(
        IReadOnlyDictionary<string, double> groupProfile,
        string? groupPrimaryCategory,
        WorkingCategoryCommunity community)
    {
        var communityProfile = community.GetNormalizedProfile();
        double score = WeightedJaccard(groupProfile, communityProfile);
        if (!string.IsNullOrWhiteSpace(groupPrimaryCategory)
            && string.Equals(groupPrimaryCategory, community.GetPrimaryCategory(), StringComparison.OrdinalIgnoreCase))
        {
            score += 0.05;
        }

        return Math.Min(1.0, score);
    }

    private static double WeightedJaccard(
        IReadOnlyDictionary<string, double> left,
        IReadOnlyDictionary<string, double> right)
    {
        var keys = left.Keys
            .Concat(right.Keys)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        double numerator = 0;
        double denominator = 0;
        foreach (var key in keys)
        {
            double leftValue = left.GetValueOrDefault(key);
            double rightValue = right.GetValueOrDefault(key);
            numerator += Math.Min(leftValue, rightValue);
            denominator += Math.Max(leftValue, rightValue);
        }

        return denominator <= 0 ? 0 : numerator / denominator;
    }

    private sealed class WorkingSignatureGroup
    {
        public WorkingSignatureGroup(string signatureKey, List<ConversationAnalysis> analyses)
        {
            SignatureKey = signatureKey;
            Analyses = analyses;
            Profile = analyses.Count == 0
                ? new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase)
                : BuildFacetProfile(analyses[0]);
            PrimaryCategory = analyses
                .Select(analysis => analysis.Category)
                .FirstOrDefault(category => !string.IsNullOrWhiteSpace(category));
        }

        public string SignatureKey { get; }
        public List<ConversationAnalysis> Analyses { get; }
        public Dictionary<string, double> Profile { get; }
        public string? PrimaryCategory { get; }
    }

    private sealed class WorkingCategoryCommunity
    {
        private readonly Dictionary<string, double> _categoryWeightTotals = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, int> _primaryCategoryCounts = new(StringComparer.OrdinalIgnoreCase);

        public WorkingCategoryCommunity(WorkingSignatureGroup seed)
        {
            SeedSignatureKey = seed.SignatureKey;
            Members = new List<ConversationAnalysis>();
            Add(seed);
        }

        public string SeedSignatureKey { get; }
        public List<ConversationAnalysis> Members { get; }

        public void Add(WorkingSignatureGroup group)
        {
            foreach (var analysis in group.Analyses)
            {
                Members.Add(analysis);

                foreach (var pair in BuildFacetProfile(analysis))
                {
                    _categoryWeightTotals[pair.Key] = _categoryWeightTotals.GetValueOrDefault(pair.Key) + pair.Value;
                }

                if (!string.IsNullOrWhiteSpace(analysis.Category))
                {
                    _primaryCategoryCounts[analysis.Category] = _primaryCategoryCounts.GetValueOrDefault(analysis.Category) + 1;
                }
            }
        }

        public Dictionary<string, double> GetNormalizedProfile()
        {
            double divisor = Math.Max(1, Members.Count);
            return _categoryWeightTotals.ToDictionary(
                pair => pair.Key,
                pair => Math.Round(pair.Value / divisor, 4),
                StringComparer.OrdinalIgnoreCase);
        }

        public string GetPrimaryCategory()
            => _primaryCategoryCounts
                .OrderByDescending(pair => pair.Value)
                .ThenBy(pair => pair.Key, StringComparer.OrdinalIgnoreCase)
                .Select(pair => pair.Key)
                .FirstOrDefault()
                ?? AnalysisDefaults.DefaultCategory;

        public string GetLabel()
        {
            var categories = GetTopCategories().Take(3).ToList();
            return categories.Count switch
            {
                0 => AnalysisDefaults.DefaultCategory,
                1 => categories[0],
                _ => string.Join(" + ", categories)
            };
        }

        public string? GetSummary()
        {
            var categories = GetTopCategories().Take(4).ToList();
            if (categories.Count == 0)
            {
                return AnalysisDefaults.DefaultCategory;
            }

            string primaryCategory = GetPrimaryCategory();
            string supporting = string.Join(", ", categories.Skip(1).Take(3));
            if (string.IsNullOrWhiteSpace(supporting))
            {
                return $"{primaryCategory} centered across {Members.Count} conversations.";
            }

            return $"{primaryCategory} centered with {supporting} across {Members.Count} conversations.";
        }

        public CategoryCommunity ToRecord(int number)
        {
            var categories = GetTopCategories().Take(6).ToList();
            var representativeKeywords = Members
                .SelectMany(item => item.Keywords.Take(8))
                .GroupBy(item => item.Keyword, StringComparer.OrdinalIgnoreCase)
                .Select(group => new KeywordScore(group.First().Keyword, group.Sum(item => item.Score)))
                .OrderByDescending(item => item.Score)
                .ThenBy(item => item.Keyword, StringComparer.OrdinalIgnoreCase)
                .Take(10)
                .Select(item => item.Keyword)
                .ToList();

            return new CategoryCommunity(
                $"catcomm:{number:0000}",
                GetLabel(),
                GetSummary(),
                GetPrimaryCategory(),
                categories,
                Members
                    .Select(member => member.Conversation.ConversationId)
                    .OrderBy(id => id, StringComparer.Ordinal)
                    .ToList(),
                representativeKeywords);
        }

        private IReadOnlyList<string> GetTopCategories()
            => _categoryWeightTotals
                .OrderByDescending(pair => pair.Value)
                .ThenBy(pair => _primaryCategoryCounts.GetValueOrDefault(pair.Key))
                .ThenBy(pair => pair.Key, StringComparer.OrdinalIgnoreCase)
                .Select(pair => pair.Key)
                .ToList();
    }
}

static class KeywordCooccurrenceBuilder
{
    private const string GlobalScopeId = "global";

    public static IReadOnlyList<KeywordCooccurrence> Build(
        IReadOnlyList<ConversationAnalysis> analyses,
        int keywordsPerConversation,
        int minimumCooccurrence)
    {
        var counts = new Dictionary<(string LeftId, string RightId), int>(ConversationPairComparer.OrdinalIgnoreCase);

        foreach (var analysis in analyses)
        {
            var keywords = analysis.Keywords
                .Take(Math.Max(2, keywordsPerConversation))
                .Select(keyword => keyword.Keyword)
                .Where(keyword => !string.IsNullOrWhiteSpace(keyword))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .OrderBy(keyword => keyword, StringComparer.OrdinalIgnoreCase)
                .ToList();

            for (int i = 0; i < keywords.Count; i++)
            {
                for (int j = i + 1; j < keywords.Count; j++)
                {
                    var pair = (keywords[i], keywords[j]);
                    counts[pair] = counts.TryGetValue(pair, out int existing) ? existing + 1 : 1;
                }
            }
        }

        int maxCount = Math.Max(1, counts.Values.DefaultIfEmpty(1).Max());
        return counts
            .Where(item => item.Value >= Math.Max(1, minimumCooccurrence))
            .OrderByDescending(item => item.Value)
            .ThenBy(item => item.Key.LeftId, StringComparer.OrdinalIgnoreCase)
            .ThenBy(item => item.Key.RightId, StringComparer.OrdinalIgnoreCase)
            .Select(item => new KeywordCooccurrence(
                item.Key.LeftId,
                item.Key.RightId,
                item.Value,
                Math.Round((double)item.Value / maxCount, 4)))
            .ToList();
    }

    public static IReadOnlyList<GraphEdge> BuildEdges(IReadOnlyList<KeywordCooccurrence> keywordCooccurrences)
        => keywordCooccurrences
            .Select(item => new GraphEdge($"kw:{item.LeftKeyword}", $"kw:{item.RightKeyword}", "co_occurs_with", item.Weight, GlobalScopeId))
            .ToList();
}

static class CategoryHierarchyBuilder
{
    private const string GlobalScopeId = "global";

    public static IReadOnlyList<CategoryHierarchyLink> Build(IReadOnlyList<ConversationAnalysis> analyses)
    {
        var counts = new Dictionary<(string LeftId, string RightId), int>(ConversationPairComparer.OrdinalIgnoreCase);
        foreach (var analysis in analyses)
        {
            var segments = analysis.Category
                .Split('/', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

            for (int i = 0; i < segments.Length - 1; i++)
            {
                string parent = string.Join("/", segments.Take(i + 1));
                string child = string.Join("/", segments.Take(i + 2));
                var key = (parent, child);
                counts[key] = counts.TryGetValue(key, out int existing) ? existing + 1 : 1;
            }
        }

        return counts
            .OrderByDescending(item => item.Value)
            .ThenBy(item => item.Key.LeftId, StringComparer.OrdinalIgnoreCase)
            .ThenBy(item => item.Key.RightId, StringComparer.OrdinalIgnoreCase)
            .Select(item => new CategoryHierarchyLink(item.Key.LeftId, item.Key.RightId, item.Value))
            .ToList();
    }

    public static IReadOnlyList<GraphEdge> BuildEdges(IReadOnlyList<CategoryHierarchyLink> hierarchy)
        => hierarchy
            .Select(item => new GraphEdge($"cat:{item.ParentCategory}", $"cat:{item.ChildCategory}", "category_parent", item.ConversationCount, GlobalScopeId))
            .ToList();

    public static string GetLeafCategory(string category)
    {
        var segments = category
            .Split('/', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return segments.Length == 0 ? category : segments[^1];
    }
}

static class ConversationIndexText
{
    private static readonly HashSet<string> StopWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "the","and","for","that","with","this","from","have","your","into","about","there","would","could","should","what",
        "when","where","which","while","will","want","need","using","used","then","than","they","them","their","does",
        "did","how","why","can","you","our","are","not","but","all","was","were","has","had","its","let","lets",
        "also","just","like","more","some","such","over","under","onto","each","very","true","false","null","text",
        "code","json","file","files","data","output","message","messages","conversation","conversations",
        "procedure","procedures","issue","issues","problem","problems","check","checks","checking","guide","guides",
        "outline","overview","example","examples","request","requests","step","steps","troubleshooting","troubleshoot",
        "any","one","might","across","below","brief","known","eventually","value","values"
    };

    private static readonly HashSet<string> ShortTechnicalTokens = new(StringComparer.OrdinalIgnoreCase)
    {
        "ad","ai","ca","cd","ci","db","dc","id","ip","ml","os","ps","qa","ui","ux","vm","c#","f#"
    };

    private static readonly HashSet<string> NoiseTokens = new(StringComparer.OrdinalIgnoreCase)
    {
        "---","----","###","####","#####","```","~~~","nbsp","quot","amp","lt","gt"
    };

    public static string BuildCorpus(Conversation conversation)
    {
        var sb = new StringBuilder();
        foreach (var node in conversation.MessageNodes.OrderBy(node => node.Flat.CreateTime ?? double.MaxValue))
        {
            AppendIfPresent(sb, node.Flat.Text);
            AppendIfPresent(sb, node.Flat.AggregateCode);
            AppendIfPresent(sb, node.Flat.ExecutionOutput);
        }

        return sb.ToString();
    }

    public static IReadOnlyList<KeywordScore> ExtractKeywords(Conversation conversation, int take)
    {
        var scores = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
        AddWeightedTerms(scores, conversation.Title, 6.0, allowPhrases: true, forceStructured: true);

        foreach (var node in conversation.MessageNodes.OrderBy(node => node.Flat.CreateTime ?? double.MaxValue))
        {
            double textWeight = ResolveTextWeight(node.Flat.Role);
            AddWeightedTerms(scores, node.Flat.Text, textWeight, allowPhrases: true, forceStructured: false);
            AddWeightedTerms(scores, node.Flat.AggregateCode, 0.35, allowPhrases: false, forceStructured: false);
            AddWeightedTerms(scores, node.Flat.ExecutionOutput, 0.20, allowPhrases: false, forceStructured: false);
        }

        var ranked = scores
            .Select(kvp => new ScoredTerm(kvp.Key, kvp.Value, CountWords(kvp.Key)))
            .OrderByDescending(item => item.Score * (item.WordCount > 1 ? 1.75 : 1.0))
            .ThenByDescending(item => item.WordCount)
            .ThenBy(item => item.Term, StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (ranked.Count == 0)
        {
            return Array.Empty<KeywordScore>();
        }

        double maxScore = ranked[0].Score;
        return SelectDiverseTerms(ranked, Math.Max(6, take))
            .Select(item => new KeywordScore(item.Term, Math.Round(item.Score / maxScore, 4)))
            .ToList();
    }

    public static IReadOnlyList<string> SelectTopics(string title, IReadOnlyList<KeywordScore> keywords, int take)
    {
        var keywordScores = keywords.ToDictionary(keyword => keyword.Keyword, keyword => keyword.Score, StringComparer.OrdinalIgnoreCase);
        var titleTokens = Tokenize(title).ToList();
        var prioritizedTitlePhrases = BuildPhrases(titleTokens)
            .Select((phrase, index) => new
            {
                Phrase = phrase,
                Index = index,
                Score = Tokenize(phrase).Sum(token => keywordScores.GetValueOrDefault(token))
            })
            .OrderByDescending(item => item.Score)
            .ThenBy(item => item.Index)
            .Select(item => item.Phrase)
            .Take(1);

        var prioritizedTitleTokens = titleTokens
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderByDescending(token => keywordScores.GetValueOrDefault(token))
            .ThenBy(token => token, StringComparer.OrdinalIgnoreCase);

        var ordered = keywords
            .OrderByDescending(keyword => keyword.Score * (CountWords(keyword.Keyword) > 1 ? 2.1 : 1.0))
            .ThenByDescending(keyword => CountWords(keyword.Keyword))
            .ThenBy(keyword => keyword.Keyword, StringComparer.OrdinalIgnoreCase)
            .Select(keyword => keyword.Keyword);

        return SelectDiverseTerms(
            prioritizedTitlePhrases
                .Concat(prioritizedTitleTokens)
                .Concat(ordered),
            Math.Max(4, take));
    }

    public static IReadOnlyList<string> SelectDiverseTerms(IEnumerable<string> candidates, int take)
    {
        var selected = new List<string>(Math.Max(1, take));
        foreach (var candidate in candidates)
        {
            if (string.IsNullOrWhiteSpace(candidate))
            {
                continue;
            }

            string normalized = candidate.Trim();
            if (selected.Any(existing => TermsOverlap(existing, normalized)))
            {
                continue;
            }

            selected.Add(normalized);
            if (selected.Count >= take)
            {
                break;
            }
        }

        return selected;
    }

    public static bool TermsOverlap(string left, string right)
    {
        if (string.Equals(left, right, StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        var leftTokens = Tokenize(left).Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
        var rightTokens = Tokenize(right).Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
        if (leftTokens.Length == 0 || rightTokens.Length == 0)
        {
            return false;
        }

        int overlap = leftTokens.Intersect(rightTokens, StringComparer.OrdinalIgnoreCase).Count();
        return overlap == leftTokens.Length || overlap == rightTokens.Length;
    }

    public static IEnumerable<string> Tokenize(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            yield break;
        }

        foreach (var rawLine in text.Replace("\r\n", "\n").Split('\n'))
        {
            string line = NormalizeMarkdownLine(rawLine);
            if (line.Length == 0)
            {
                continue;
            }

            foreach (var token in TokenizeLine(line))
            {
                yield return token;
            }
        }
    }

    private static double ResolveTextWeight(string? role)
        => role?.Trim().ToLowerInvariant() switch
        {
            "user" => 2.0,
            "assistant" => 1.2,
            "tool" => 0.9,
            "system" => 0.0,
            _ => 0.8
        };

    private static void AddWeightedTerms(
        Dictionary<string, double> scores,
        string? text,
        double weight,
        bool allowPhrases,
        bool forceStructured)
    {
        if (weight <= 0 || string.IsNullOrWhiteSpace(text))
        {
            return;
        }

        foreach (var rawLine in text.Replace("\r\n", "\n").Split('\n'))
        {
            string line = NormalizeMarkdownLine(rawLine);
            if (line.Length == 0)
            {
                continue;
            }

            var tokens = TokenizeLine(line).ToList();
            if (tokens.Count == 0)
            {
                continue;
            }

            bool structured = forceStructured || IsStructuredLine(rawLine, line);
            double lineWeight = structured ? weight * 1.15 : weight;
            foreach (var token in tokens)
            {
                AddScore(scores, token, lineWeight);
            }

            if (!allowPhrases || !structured)
            {
                continue;
            }

            foreach (var phrase in BuildPhrases(tokens))
            {
                AddScore(scores, phrase, lineWeight * 1.75);
            }
        }
    }

    private static void AddScore(Dictionary<string, double> scores, string term, double weight)
        => scores[term] = scores.TryGetValue(term, out double existing) ? existing + weight : weight;

    private static IEnumerable<string> BuildPhrases(IReadOnlyList<string> tokens)
    {
        if (tokens.Count < 2)
        {
            yield break;
        }

        int limit = Math.Min(tokens.Count - 1, 7);
        for (int i = 0; i < limit; i++)
        {
            string left = tokens[i];
            string right = tokens[i + 1];
            if (StopWords.Contains(left) || StopWords.Contains(right))
            {
                continue;
            }

            yield return $"{left} {right}";
        }
    }

    private static IEnumerable<string> TokenizeLine(string line)
    {
        var current = new StringBuilder(line.Length);
        for (int i = 0; i < line.Length; i++)
        {
            char ch = line[i];
            if (char.IsLetterOrDigit(ch) || ch is '+' or '#' or '.' or '-' or '_')
            {
                current.Append(ch);
                continue;
            }

            if (current.Length > 0)
            {
                string? token = NormalizeToken(current.ToString());
                if (!string.IsNullOrWhiteSpace(token))
                {
                    yield return token;
                }

                current.Clear();
            }
        }

        if (current.Length > 0)
        {
            string? token = NormalizeToken(current.ToString());
            if (!string.IsNullOrWhiteSpace(token))
            {
                yield return token;
            }
        }
    }

    private static string NormalizeMarkdownLine(string rawLine)
    {
        string line = rawLine.Trim();
        if (line.Length == 0)
        {
            return string.Empty;
        }

        if (line.StartsWith("```", StringComparison.Ordinal) || line.StartsWith("~~~", StringComparison.Ordinal))
        {
            return string.Empty;
        }

        while (line.StartsWith('#') || line.StartsWith('>'))
        {
            line = line[1..].TrimStart();
        }

        if (line.StartsWith("- ", StringComparison.Ordinal)
            || line.StartsWith("* ", StringComparison.Ordinal)
            || line.StartsWith("+ ", StringComparison.Ordinal))
        {
            line = line[2..].TrimStart();
        }

        line = line
            .Replace("**", " ", StringComparison.Ordinal)
            .Replace("__", " ", StringComparison.Ordinal)
            .Replace("`", " ", StringComparison.Ordinal)
            .Trim();

        int index = 0;
        bool sawDigit = false;
        while (index < line.Length && (char.IsDigit(line[index]) || line[index] == '.'))
        {
            sawDigit |= char.IsDigit(line[index]);
            index++;
        }

        if (sawDigit && index < line.Length && (line[index] == ')' || line[index] == ':'))
        {
            line = line[(index + 1)..].TrimStart();
        }
        else if (sawDigit && index < line.Length && char.IsWhiteSpace(line[index]))
        {
            line = line[index..].TrimStart();
        }

        return line.Trim();
    }

    private static bool IsStructuredLine(string rawLine, string normalizedLine)
    {
        string trimmed = rawLine.TrimStart();
        if (trimmed.StartsWith("#", StringComparison.Ordinal)
            || trimmed.StartsWith("- ", StringComparison.Ordinal)
            || trimmed.StartsWith("* ", StringComparison.Ordinal)
            || trimmed.StartsWith("+ ", StringComparison.Ordinal))
        {
            return true;
        }

        int index = 0;
        while (index < trimmed.Length && char.IsDigit(trimmed[index]))
        {
            index++;
        }

        if (index > 0 && index < trimmed.Length && (trimmed[index] == '.' || trimmed[index] == ')'))
        {
            return true;
        }

        return normalizedLine.Length <= 96;
    }

    private static string? NormalizeToken(string rawToken)
    {
        string raw = rawToken.Trim();
        if (raw.Length == 0)
        {
            return null;
        }

        string normalized = raw.ToLowerInvariant();
        normalized = normalized switch
        {
            "c#" => "csharp",
            "f#" => "fsharp",
            ".net" => "dotnet",
            _ => normalized
        };

        normalized = normalized.Trim('.', '-', '_', '#', '+');
        if (normalized.Length == 0)
        {
            return null;
        }

        if (normalized.StartsWith("http", StringComparison.OrdinalIgnoreCase) || normalized.Contains("://", StringComparison.Ordinal))
        {
            return null;
        }

        if (NoiseTokens.Contains(normalized))
        {
            return null;
        }

        if (!normalized.Any(char.IsLetterOrDigit) || normalized.All(char.IsDigit))
        {
            return null;
        }

        if (normalized.Length > 48)
        {
            return null;
        }

        if (normalized.Length < 3 && !IsAllowedShortToken(raw, normalized))
        {
            return null;
        }

        if (StopWords.Contains(normalized))
        {
            return null;
        }

        return normalized;
    }

    private static bool IsAllowedShortToken(string raw, string normalized)
    {
        if (ShortTechnicalTokens.Contains(normalized))
        {
            return true;
        }

        var letters = raw.Where(char.IsLetter).ToArray();
        return letters.Length >= 2
            && letters.Length <= 4
            && letters.All(char.IsUpper);
    }

    private static int CountWords(string term)
        => term.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Length;

    private static IReadOnlyList<ScoredTerm> SelectDiverseTerms(IReadOnlyList<ScoredTerm> ranked, int take)
    {
        var selected = new List<ScoredTerm>(Math.Max(1, take));
        foreach (var item in ranked)
        {
            if (selected.Any(existing => TermsOverlap(existing.Term, item.Term)))
            {
                continue;
            }

            selected.Add(item);
            if (selected.Count >= take)
            {
                break;
            }
        }

        return selected;
    }

    private static void AppendIfPresent(StringBuilder sb, string? value)
    {
        if (!string.IsNullOrWhiteSpace(value))
        {
            sb.AppendLine(value);
        }
    }

    private sealed record ScoredTerm(string Term, double Score, int WordCount);
}

static class LocalModelFileInspector
{
    public static bool IsGitLfsPointer(string path)
    {
        using var reader = File.OpenText(path);
        string? firstLine = reader.ReadLine();
        string? secondLine = reader.ReadLine();

        return string.Equals(firstLine, "version https://git-lfs.github.com/spec/v1", StringComparison.Ordinal)
            && secondLine is not null
            && secondLine.StartsWith("oid sha256:", StringComparison.Ordinal);
    }
}

interface IEmbeddingProvider : IDisposable
{
    string Description { get; }
    ConversationEmbedding Embed(string conversationId, string text);
}

sealed class HashingEmbeddingProvider : IEmbeddingProvider
{
    private readonly int _dimensions;

    public HashingEmbeddingProvider(int dimensions)
    {
        _dimensions = Math.Max(32, dimensions);
    }

    public string Description => $"hash:{_dimensions}";

    public ConversationEmbedding Embed(string conversationId, string text)
    {
        var vector = new float[_dimensions];
        foreach (var token in ConversationIndexText.Tokenize(text))
        {
            uint hash = StableHash(token);
            int index = (int)(hash % (uint)_dimensions);
            float sign = ((hash >> 1) & 1) == 0 ? 1f : -1f;
            vector[index] += sign;
        }

        NormalizeInPlace(vector);
        return new ConversationEmbedding("hash", $"hash-{_dimensions}", vector);
    }

    public void Dispose()
    {
    }

    private static uint StableHash(string value)
    {
        const uint offset = 2166136261;
        const uint prime = 16777619;
        uint hash = offset;
        foreach (char ch in value)
        {
            hash ^= ch;
            hash *= prime;
        }

        return hash;
    }

    private static void NormalizeInPlace(float[] values)
    {
        double norm = 0;
        foreach (float value in values)
        {
            norm += value * value;
        }

        if (norm == 0)
        {
            return;
        }

        float scale = (float)(1.0 / Math.Sqrt(norm));
        for (int i = 0; i < values.Length; i++)
        {
            values[i] *= scale;
        }
    }
}

sealed class FallbackEmbeddingProvider : IEmbeddingProvider
{
    private readonly IEmbeddingProvider _primary;
    private readonly IEmbeddingProvider _fallback;
    private bool _warned;

    public FallbackEmbeddingProvider(IEmbeddingProvider primary, IEmbeddingProvider fallback)
    {
        _primary = primary;
        _fallback = fallback;
    }

    public string Description => $"{_primary.Description} -> {_fallback.Description}";

    public ConversationEmbedding Embed(string conversationId, string text)
    {
        try
        {
            return _primary.Embed(conversationId, text);
        }
        catch (Exception ex)
        {
            if (!_warned)
            {
                Console.Error.WriteLine($"[warn] ONNX embedding inference failed: {ex.Message}. Using fallback hashing embeddings.");
                _warned = true;
            }

            return _fallback.Embed(conversationId, text);
        }
    }

    public void Dispose()
    {
        _primary.Dispose();
        _fallback.Dispose();
    }
}

sealed class OnnxEmbeddingProvider : IEmbeddingProvider
{
    private readonly string _modelPath;
    private readonly int _maxTokens;
    private readonly InferenceSession _session;
    private readonly IOnnxTextTokenizer _tokenizer;
    private readonly string _executionProviderDescription;

    public OnnxEmbeddingProvider(
        string modelPath,
        string vocabularyPath,
        int maxTokens,
        string tokenizerMode,
        string executionProviderMode,
        int cudaDeviceId,
        int directMlDeviceId)
    {
        _modelPath = modelPath;
        _maxTokens = Math.Max(16, maxTokens);
        _tokenizer = CreateTokenizer(vocabularyPath, tokenizerMode);
        (_session, _executionProviderDescription) = CreateSession(modelPath, executionProviderMode, cudaDeviceId, directMlDeviceId);
    }

    public string Description => $"onnx:{Path.GetFileName(_modelPath)}[{_tokenizer.Description},{_executionProviderDescription}]";

    public ConversationEmbedding Embed(string conversationId, string text)
    {
        var encoded = _tokenizer.Encode(text, _maxTokens);

        using var results = _session.Run(BuildInputs(encoded));
        var vector = ExtractEmbedding(results, encoded.AttentionMask);
        return new ConversationEmbedding("onnx", Path.GetFileNameWithoutExtension(_modelPath), vector);
    }

    public void Dispose()
    {
        _session.Dispose();
    }

    private IReadOnlyList<NamedOnnxValue> BuildInputs(OnnxTokenizedText encoded)
    {
        var inputs = new List<NamedOnnxValue>();

        if (_session.InputMetadata.TryGetValue("input_ids", out var inputIdsMetadata))
        {
            inputs.Add(CreateTensorValue("input_ids", encoded.InputIds, inputIdsMetadata));
        }

        if (_session.InputMetadata.TryGetValue("attention_mask", out var attentionMaskMetadata))
        {
            inputs.Add(CreateTensorValue("attention_mask", encoded.AttentionMask, attentionMaskMetadata));
        }

        if (_session.InputMetadata.TryGetValue("token_type_ids", out var tokenTypeMetadata))
        {
            inputs.Add(CreateTensorValue("token_type_ids", encoded.TokenTypeIds, tokenTypeMetadata));
        }

        return inputs;
    }

    private static IOnnxTextTokenizer CreateTokenizer(string vocabularyPath, string tokenizerMode)
        => tokenizerMode switch
        {
            "legacy" => new LegacyOnnxTextTokenizer(vocabularyPath),
            "mltokenizer" => new MlTokenizerOnnxTextTokenizer(vocabularyPath),
            _ => throw new NotSupportedException($"Unsupported ONNX tokenizer mode: {tokenizerMode}")
        };

    private static (InferenceSession Session, string Description) CreateSession(
        string modelPath,
        string executionProviderMode,
        int cudaDeviceId,
        int directMlDeviceId)
    {
        if (string.Equals(executionProviderMode, "cpu", StringComparison.OrdinalIgnoreCase))
        {
            return (new InferenceSession(modelPath), "cpu");
        }

        if (string.Equals(executionProviderMode, "cuda", StringComparison.OrdinalIgnoreCase))
        {
            return (CreateCudaSession(modelPath, cudaDeviceId), $"cuda:{cudaDeviceId}");
        }

        if (string.Equals(executionProviderMode, "directml", StringComparison.OrdinalIgnoreCase))
        {
            return (CreateDirectMlSession(modelPath, directMlDeviceId), $"directml:{directMlDeviceId}");
        }

        if (string.Equals(executionProviderMode, "auto", StringComparison.OrdinalIgnoreCase))
        {
            try
            {
                return (CreateCudaSession(modelPath, cudaDeviceId), $"cuda:{cudaDeviceId}");
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[warn] CUDA initialization failed: {ex.Message}. Trying DirectML next.");
            }

            try
            {
                return (CreateDirectMlSession(modelPath, directMlDeviceId), $"directml:{directMlDeviceId}");
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[warn] DirectML initialization failed: {ex.Message}. Falling back to CPU ONNX execution.");
                return (new InferenceSession(modelPath), "cpu-fallback");
            }
        }

        throw new NotSupportedException($"Unsupported ONNX execution provider mode: {executionProviderMode}");
    }

    private static InferenceSession CreateCudaSession(string modelPath, int cudaDeviceId)
    {
        var options = new SessionOptions();
        options.AppendExecutionProvider_CUDA(cudaDeviceId);
        return new InferenceSession(modelPath, options);
    }

    private static InferenceSession CreateDirectMlSession(string modelPath, int directMlDeviceId)
    {
        var options = new SessionOptions();
        options.AppendExecutionProvider_DML(directMlDeviceId);
        return new InferenceSession(modelPath, options);
    }

    private static NamedOnnxValue CreateTensorValue(string name, long[] values, NodeMetadata metadata)
    {
        if (metadata.ElementType == typeof(long))
        {
            var tensor = new DenseTensor<long>(new[] { 1, values.Length });
            for (int i = 0; i < values.Length; i++)
            {
                tensor[0, i] = values[i];
            }

            return NamedOnnxValue.CreateFromTensor(name, tensor);
        }

        if (metadata.ElementType == typeof(int))
        {
            var tensor = new DenseTensor<int>(new[] { 1, values.Length });
            for (int i = 0; i < values.Length; i++)
            {
                tensor[0, i] = (int)values[i];
            }

            return NamedOnnxValue.CreateFromTensor(name, tensor);
        }

        throw new NotSupportedException($"Unsupported ONNX tensor input type for {name}: {metadata.ElementType}");
    }

    private static float[] ExtractEmbedding(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
        long[] attentionMask)
    {
        DisposableNamedOnnxValue output = results
            .FirstOrDefault(result => string.Equals(result.Name, "sentence_embedding", StringComparison.OrdinalIgnoreCase))
            ?? results.FirstOrDefault(result => string.Equals(result.Name, "embeddings", StringComparison.OrdinalIgnoreCase))
            ?? results.FirstOrDefault(result => string.Equals(result.Name, "last_hidden_state", StringComparison.OrdinalIgnoreCase))
            ?? results.First();

        var tensor = output.AsTensor<float>();
        return tensor.Rank switch
        {
            2 => Normalize(ReadRow(tensor)),
            3 => Normalize(MeanPool(tensor, attentionMask)),
            _ => throw new NotSupportedException($"Unsupported embedding tensor rank: {tensor.Rank}")
        };
    }

    private static float[] ReadRow(Tensor<float> tensor)
    {
        int width = tensor.Dimensions[^1];
        var vector = new float[width];
        for (int i = 0; i < width; i++)
        {
            vector[i] = tensor[0, i];
        }

        return vector;
    }

    private static float[] MeanPool(Tensor<float> tensor, long[] attentionMask)
    {
        int sequenceLength = tensor.Dimensions[1];
        int hiddenSize = tensor.Dimensions[2];
        var vector = new float[hiddenSize];
        double count = 0;

        for (int tokenIndex = 0; tokenIndex < sequenceLength && tokenIndex < attentionMask.Length; tokenIndex++)
        {
            if (attentionMask[tokenIndex] == 0)
            {
                continue;
            }

            count++;
            for (int hiddenIndex = 0; hiddenIndex < hiddenSize; hiddenIndex++)
            {
                vector[hiddenIndex] += tensor[0, tokenIndex, hiddenIndex];
            }
        }

        if (count > 0)
        {
            float divisor = (float)count;
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] /= divisor;
            }
        }

        return vector;
    }

    private static float[] Normalize(float[] values)
    {
        double norm = 0;
        foreach (float value in values)
        {
            norm += value * value;
        }

        if (norm == 0)
        {
            return values;
        }

        float scale = (float)(1.0 / Math.Sqrt(norm));
        for (int i = 0; i < values.Length; i++)
        {
            values[i] *= scale;
        }

        return values;
    }
}

sealed record OnnxTokenizedText(long[] InputIds, long[] AttentionMask, long[] TokenTypeIds, int TokenCount);

interface IOnnxTextTokenizer
{
    string Description { get; }
    OnnxTokenizedText Encode(string text, int maxTokens);
}

sealed class LegacyOnnxTextTokenizer : IOnnxTextTokenizer
{
    private readonly BertWordPieceTokenizer _tokenizer;

    public LegacyOnnxTextTokenizer(string vocabularyPath)
    {
        _tokenizer = new BertWordPieceTokenizer(vocabularyPath);
    }

    public string Description => "legacy-wordpiece";

    public OnnxTokenizedText Encode(string text, int maxTokens)
    {
        var encoded = _tokenizer.Encode(text, maxTokens);
        return new OnnxTokenizedText(encoded.InputIds, encoded.AttentionMask, encoded.TokenTypeIds, encoded.TokenCount);
    }
}

sealed class MlTokenizerOnnxTextTokenizer : IOnnxTextTokenizer
{
    private readonly string _vocabularyPath;
    private readonly BertTokenizer _tokenizer;

    public MlTokenizerOnnxTextTokenizer(string vocabularyPath)
    {
        _vocabularyPath = vocabularyPath;
        _tokenizer = BertTokenizer.Create(vocabularyPath);
    }

    public string Description => $"mltokenizer:{Path.GetFileName(_vocabularyPath)}";

    public OnnxTokenizedText Encode(string text, int maxTokens)
    {
        int targetLength = Math.Max(8, maxTokens);
        var tokenIds = _tokenizer.EncodeToIds(
            text ?? string.Empty,
            targetLength,
            addSpecialTokens: true,
            normalizedText: out _,
            charsConsumed: out _,
            considerPreTokenization: true,
            considerNormalization: false);

        var inputIds = new long[targetLength];
        var attentionMask = new long[targetLength];
        var tokenTypeIds = new long[targetLength];

        int count = Math.Min(tokenIds.Count, targetLength);
        for (int i = 0; i < count; i++)
        {
            inputIds[i] = tokenIds[i];
            attentionMask[i] = 1;
        }

        for (int i = count; i < targetLength; i++)
        {
            inputIds[i] = _tokenizer.PaddingTokenId;
        }

        return new OnnxTokenizedText(inputIds, attentionMask, tokenTypeIds, count);
    }
}

sealed class BertWordPieceTokenizer
{
    private readonly Dictionary<string, int> _vocabulary;
    private readonly int _padTokenId;
    private readonly int _unknownTokenId;

    public BertWordPieceTokenizer(string vocabularyPath)
    {
        _vocabulary = File.ReadLines(vocabularyPath)
            .Select((token, index) => (token: token.Trim(), index))
            .Where(item => item.token.Length > 0)
            .ToDictionary(item => item.token, item => item.index, StringComparer.Ordinal);

        _padTokenId = ResolveSpecialToken("[PAD]", "[pad]");
        _unknownTokenId = ResolveSpecialToken("[UNK]", "[unk]");
    }

    public TokenizedText Encode(string text, int maxTokens)
    {
        int targetLength = Math.Max(8, maxTokens);
        var tokens = new List<string> { "[CLS]" };

        foreach (var token in BasicTokenize(text))
        {
            foreach (var piece in WordPieceTokenize(token))
            {
                if (tokens.Count >= targetLength - 1)
                {
                    break;
                }

                tokens.Add(piece);
            }

            if (tokens.Count >= targetLength - 1)
            {
                break;
            }
        }

        tokens.Add("[SEP]");

        var inputIds = new long[targetLength];
        var attentionMask = new long[targetLength];
        var tokenTypeIds = new long[targetLength];

        int count = Math.Min(tokens.Count, targetLength);
        for (int i = 0; i < count; i++)
        {
            inputIds[i] = ResolveTokenId(tokens[i]);
            attentionMask[i] = 1;
        }

        for (int i = count; i < targetLength; i++)
        {
            inputIds[i] = _padTokenId;
        }

        return new TokenizedText(inputIds, attentionMask, tokenTypeIds, count);
    }

    private int ResolveSpecialToken(string preferred, string alternate)
    {
        if (_vocabulary.TryGetValue(preferred, out int preferredId))
        {
            return preferredId;
        }

        if (_vocabulary.TryGetValue(alternate, out int alternateId))
        {
            return alternateId;
        }

        return 0;
    }

    private int ResolveTokenId(string token)
        => _vocabulary.TryGetValue(token, out int tokenId)
            ? tokenId
            : _unknownTokenId;

    private IEnumerable<string> BasicTokenize(string text)
    {
        var sb = new StringBuilder();
        foreach (char ch in text.Normalize(NormalizationForm.FormKC).ToLowerInvariant())
        {
            if (char.IsWhiteSpace(ch))
            {
                if (sb.Length > 0)
                {
                    yield return sb.ToString();
                    sb.Clear();
                }

                continue;
            }

            if (char.IsLetterOrDigit(ch))
            {
                sb.Append(ch);
                continue;
            }

            if (sb.Length > 0)
            {
                yield return sb.ToString();
                sb.Clear();
            }

            yield return ch.ToString();
        }

        if (sb.Length > 0)
        {
            yield return sb.ToString();
        }
    }

    private IEnumerable<string> WordPieceTokenize(string token)
    {
        if (_vocabulary.ContainsKey(token))
        {
            yield return token;
            yield break;
        }

        int start = 0;
        while (start < token.Length)
        {
            string? current = null;
            int end = token.Length;
            while (start < end)
            {
                string candidate = token.Substring(start, end - start);
                if (start > 0)
                {
                    candidate = "##" + candidate;
                }

                if (_vocabulary.ContainsKey(candidate))
                {
                    current = candidate;
                    break;
                }

                end--;
            }

            if (current is null)
            {
                yield return "[UNK]";
                yield break;
            }

            yield return current;
            start = end;
        }
    }

    public sealed record TokenizedText(long[] InputIds, long[] AttentionMask, long[] TokenTypeIds, int TokenCount);
}

interface ICategoryPredictor : IDisposable
{
    string Description { get; }
    CategoryPrediction? Predict(string text);
    IReadOnlyList<CategoryPrediction> PredictTopK(string text, int maxResults);
}

sealed class NullCategoryPredictor : ICategoryPredictor
{
    public string Description => "baseline-only";

    public CategoryPrediction? Predict(string text) => null;

    public IReadOnlyList<CategoryPrediction> PredictTopK(string text, int maxResults) => Array.Empty<CategoryPrediction>();

    public void Dispose()
    {
    }
}

sealed class MlNetCategoryPredictor : ICategoryPredictor
{
    private readonly MLContext _context = new(seed: 17);
    private readonly PredictionEngine<TextClassificationInput, TextClassificationPrediction> _predictionEngine;
    private readonly string _modelPath;
    private readonly IReadOnlyList<string> _scoreLabels;

    public MlNetCategoryPredictor(string modelPath)
    {
        _modelPath = modelPath;
        using var stream = File.OpenRead(modelPath);
        ITransformer model = _context.Model.Load(stream, out _);
        _predictionEngine = _context.Model.CreatePredictionEngine<TextClassificationInput, TextClassificationPrediction>(model);

        var inputSchema = _context.Data.LoadFromEnumerable(Array.Empty<TextClassificationInput>()).Schema;
        var outputSchema = model.GetOutputSchema(inputSchema);
        _scoreLabels = TryReadScoreLabels(outputSchema);
    }

    public string Description => $"mlnet:{Path.GetFileName(_modelPath)}";

    public CategoryPrediction? Predict(string text)
        => PredictTopK(text, 1).FirstOrDefault();

    public IReadOnlyList<CategoryPrediction> PredictTopK(string text, int maxResults)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return Array.Empty<CategoryPrediction>();
        }

        var prediction = _predictionEngine.Predict(new TextClassificationInput { Text = text });
        if (prediction.Score is { Length: > 0 } scores
            && _scoreLabels.Count == scores.Length)
        {
            var probabilities = Softmax(scores);
            return _scoreLabels
                .Select((label, index) => new CategoryPrediction(
                    label.Trim(),
                    Math.Round(probabilities[index], 4),
                    "mlnet"))
                .Where(prediction => !string.IsNullOrWhiteSpace(prediction.Category))
                .OrderByDescending(prediction => prediction.Score)
                .ThenBy(prediction => prediction.Category, StringComparer.OrdinalIgnoreCase)
                .Take(Math.Max(1, maxResults))
                .ToList();
        }

        if (string.IsNullOrWhiteSpace(prediction.PredictedLabel))
        {
            return Array.Empty<CategoryPrediction>();
        }

        double score = prediction.Score is { Length: > 0 }
            ? Math.Round(Math.Clamp(prediction.Score.Max(), 0f, 1f), 4)
            : 1.0;

        return new[]
        {
            new CategoryPrediction(prediction.PredictedLabel.Trim(), score, "mlnet")
        };
    }

    public void Dispose()
    {
    }

    private static IReadOnlyList<string> TryReadScoreLabels(DataViewSchema outputSchema)
    {
        var scoreColumn = outputSchema.FirstOrDefault(column => string.Equals(column.Name, "Score", StringComparison.Ordinal));
        if (string.IsNullOrWhiteSpace(scoreColumn.Name))
        {
            return Array.Empty<string>();
        }

        var slotNamesColumn = scoreColumn.Annotations.Schema
            .FirstOrDefault(column => string.Equals(column.Name, "SlotNames", StringComparison.Ordinal));
        if (string.IsNullOrWhiteSpace(slotNamesColumn.Name))
        {
            return Array.Empty<string>();
        }

        VBuffer<ReadOnlyMemory<char>> slotNames = default;
        scoreColumn.Annotations.GetValue("SlotNames", ref slotNames);

        return slotNames.DenseValues()
            .Select(value => value.ToString())
            .Where(value => !string.IsNullOrWhiteSpace(value))
            .ToArray();
    }

    private static double[] Softmax(IReadOnlyList<float> scores)
    {
        if (scores.Count == 0)
        {
            return Array.Empty<double>();
        }

        double max = scores.Max();
        var exps = new double[scores.Count];
        double sum = 0;

        for (int i = 0; i < scores.Count; i++)
        {
            exps[i] = Math.Exp(scores[i] - max);
            sum += exps[i];
        }

        if (sum <= 0)
        {
            return Enumerable.Repeat(1.0 / scores.Count, scores.Count).ToArray();
        }

        for (int i = 0; i < exps.Length; i++)
        {
            exps[i] /= sum;
        }

        return exps;
    }

    private sealed class TextClassificationInput
    {
        public string Label { get; set; } = string.Empty;
        public string Text { get; set; } = string.Empty;
    }

    private sealed class TextClassificationPrediction
    {
        [ColumnName("PredictedLabel")]
        public string? PredictedLabel { get; set; }

        [ColumnName("Score")]
        public float[]? Score { get; set; }
    }
}

interface IClusterLabeler : IDisposable
{
    Task<ClusterLabelResult?> LabelAsync(ClusterLabelRequest request, CancellationToken cancellationToken = default);
}

sealed record ClusterLabelRequest(
    IReadOnlyList<string> Titles,
    IReadOnlyList<string> Keywords,
    IReadOnlyList<string> Categories,
    IReadOnlyList<string> Summaries);

sealed record ClusterLabelResult(string? Label, string? Summary);

sealed class OllamaClusterLabeler : IClusterLabeler
{
    private readonly HttpClient _http = new() { Timeout = TimeSpan.FromMinutes(3) };
    private readonly string _baseUrl;
    private readonly string _model;

    public OllamaClusterLabeler(string baseUrl, string model)
    {
        _baseUrl = baseUrl.TrimEnd('/');
        _model = model;
    }

    public async Task<ClusterLabelResult?> LabelAsync(ClusterLabelRequest request, CancellationToken cancellationToken = default)
    {
        try
        {
            string prompt = $$"""
You are labeling a cluster of related conversations.
Return strict JSON with this schema:
{
  "label": "string",
  "summary": "string"
}

Rules:
- label: 2 to 6 words, concrete noun phrase.
- summary: one concise sentence.
- Use the metadata only. No markdown.

Titles:
{{string.Join("\n", request.Titles.Select(title => "- " + title))}}

Categories:
{{string.Join("\n", request.Categories.Select(category => "- " + category))}}

Keywords:
{{string.Join(", ", request.Keywords)}}

Summaries:
{{string.Join("\n", request.Summaries.Select(summary => "- " + summary))}}
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
            {
                return null;
            }

            var parsed = JsonSerializer.Deserialize<ClusterLabelResult>(payload.Response, JsonOptions.Default);
            return parsed;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[warn] Cluster labeling via Ollama failed: {ex.Message}");
            return null;
        }
    }

    public void Dispose()
    {
        _http.Dispose();
    }

    private sealed class OllamaGenerateResponse
    {
        public string? Response { get; set; }
    }
}

static class CorpusGraphWriter
{
    public static void Write(CorpusIndex index, string graphDirectory)
    {
        var allEdges = index.Conversations
            .SelectMany(conversation => conversation.GraphEdges)
            .Concat(index.GlobalEdges)
            .ToList();

        var nodes = BuildNodes(index, allEdges);
        var graph = new
        {
            nodes,
            edges = allEdges
        };

        var corpusIndex = new
        {
            conversations = index.Conversations.Select(analysis => new
            {
                analysis.Conversation.ConversationId,
                analysis.Conversation.Title,
                analysis.Category,
                PrimaryCategory = analysis.Category,
                analysis.SecondaryCategory,
                analysis.TertiaryCategory,
                analysis.CategorySource,
                analysis.CategoryCommunityId,
                analysis.CategoryCommunityLabel,
                analysis.TopicLabel,
                analysis.TopicClusterId,
                Summary = analysis.ModelSummary,
                Topics = analysis.Topics,
                Keywords = analysis.Keywords,
                CategoryPredictions = analysis.CategoryPredictions
            }),
            topicClusters = index.TopicClusters,
            categoryCommunities = index.CategoryCommunities,
            keywordCooccurrences = index.KeywordCooccurrences,
            categoryHierarchy = index.CategoryHierarchy,
            graph
        };

        File.WriteAllText(
            Path.Combine(graphDirectory, "conversation-graph.json"),
            JsonSerializer.Serialize(graph, JsonOptions.WriteIndented),
            new UTF8Encoding(false));

        File.WriteAllText(
            Path.Combine(graphDirectory, "corpus-index.json"),
            JsonSerializer.Serialize(corpusIndex, JsonOptions.WriteIndented),
            new UTF8Encoding(false));

        File.WriteAllText(
            Path.Combine(graphDirectory, "conversation-graph.dot"),
            BuildDot(nodes, allEdges),
            new UTF8Encoding(false));
    }

    private static IReadOnlyList<GraphNode> BuildNodes(CorpusIndex index, IReadOnlyList<GraphEdge> allEdges)
    {
        var nodes = new Dictionary<string, GraphNode>(StringComparer.Ordinal);

        foreach (var analysis in index.Conversations)
        {
            Add(nodes, new GraphNode(analysis.Conversation.ConversationId, analysis.Conversation.Title, "conversation"));

            foreach (var message in analysis.Conversation.Nodes.Values)
            {
                Add(nodes, new GraphNode(
                    message.Id,
                    Preview(message.Flat.Text ?? message.Flat.AggregateCode ?? message.Flat.ExecutionOutput),
                    message.Flat.Role ?? "message"));
            }

            foreach (var keyword in analysis.Keywords)
            {
                Add(nodes, new GraphNode($"kw:{keyword.Keyword}", keyword.Keyword, "keyword"));
            }

            foreach (var topic in analysis.Topics)
            {
                Add(nodes, new GraphNode($"topic:{topic}", topic, "topic"));
            }

            Add(nodes, new GraphNode($"cat:{analysis.Category}", analysis.Category, "category"));
            if (!string.IsNullOrWhiteSpace(analysis.SecondaryCategory))
            {
                Add(nodes, new GraphNode($"cat:{analysis.SecondaryCategory}", analysis.SecondaryCategory, "category"));
            }

            if (!string.IsNullOrWhiteSpace(analysis.TertiaryCategory))
            {
                Add(nodes, new GraphNode($"cat:{analysis.TertiaryCategory}", analysis.TertiaryCategory, "category"));
            }

            if (!string.IsNullOrWhiteSpace(analysis.TopicLabel))
            {
                Add(nodes, new GraphNode($"topiclabel:{analysis.TopicLabel}", analysis.TopicLabel, "topic_label"));
            }

            if (!string.IsNullOrWhiteSpace(analysis.CategoryCommunityId))
            {
                Add(nodes, new GraphNode(
                    analysis.CategoryCommunityId,
                    analysis.CategoryCommunityLabel ?? analysis.CategoryCommunityId,
                    "category_community"));
            }
        }

        foreach (var cluster in index.TopicClusters)
        {
            Add(nodes, new GraphNode(cluster.ClusterId, cluster.Label, "topic_cluster"));
            Add(nodes, new GraphNode($"cat:{cluster.PrimaryCategory}", cluster.PrimaryCategory, "category"));

            foreach (var keyword in cluster.RepresentativeKeywords)
            {
                Add(nodes, new GraphNode($"kw:{keyword}", keyword, "keyword"));
            }
        }

        foreach (var community in index.CategoryCommunities)
        {
            Add(nodes, new GraphNode(community.CommunityId, community.Label, "category_community"));

            foreach (var category in community.Categories)
            {
                Add(nodes, new GraphNode($"cat:{category}", category, "category"));
            }

            foreach (var keyword in community.RepresentativeKeywords)
            {
                Add(nodes, new GraphNode($"kw:{keyword}", keyword, "keyword"));
            }
        }

        foreach (var hierarchy in index.CategoryHierarchy)
        {
            Add(nodes, new GraphNode($"cat:{hierarchy.ParentCategory}", hierarchy.ParentCategory, "category"));
            Add(nodes, new GraphNode($"cat:{hierarchy.ChildCategory}", hierarchy.ChildCategory, "category"));
        }

        foreach (var edge in allEdges)
        {
            if (!nodes.ContainsKey(edge.FromId))
            {
                Add(nodes, InferNode(edge.FromId));
            }

            if (!nodes.ContainsKey(edge.ToId))
            {
                Add(nodes, InferNode(edge.ToId));
            }
        }

        return nodes.Values
            .OrderBy(node => node.Kind, StringComparer.Ordinal)
            .ThenBy(node => node.Label, StringComparer.OrdinalIgnoreCase)
            .ThenBy(node => node.Id, StringComparer.Ordinal)
            .ToList();
    }

    private static string BuildDot(IReadOnlyList<GraphNode> nodes, IReadOnlyList<GraphEdge> edges)
    {
        var sb = new StringBuilder();
        sb.AppendLine("digraph ChatDump {");
        sb.AppendLine("  rankdir=LR;");

        foreach (var node in nodes)
        {
            sb.AppendLine($"  \"{Escape(node.Id)}\" [shape={DotShape(node.Kind)},label=\"{Escape(node.Label)}\"];");
        }

        foreach (var edge in edges)
        {
            sb.AppendLine($"  \"{Escape(edge.FromId)}\" -> \"{Escape(edge.ToId)}\" [label=\"{Escape(edge.EdgeType)}:{edge.Weight.ToString("0.###", CultureInfo.InvariantCulture)}\"];");
        }

        sb.AppendLine("}");
        return sb.ToString();
    }

    private static void Add(Dictionary<string, GraphNode> nodes, GraphNode node)
    {
        nodes[node.Id] = node;
    }

    private static GraphNode InferNode(string id)
    {
        if (id.StartsWith("kw:", StringComparison.Ordinal))
        {
            return new GraphNode(id, id[3..], "keyword");
        }

        if (id.StartsWith("topic:", StringComparison.Ordinal))
        {
            return new GraphNode(id, id[6..], "topic");
        }

        if (id.StartsWith("topiclabel:", StringComparison.Ordinal))
        {
            return new GraphNode(id, id["topiclabel:".Length..], "topic_label");
        }

        if (id.StartsWith("cat:", StringComparison.Ordinal))
        {
            return new GraphNode(id, id[4..], "category");
        }

        if (id.StartsWith("cluster:", StringComparison.Ordinal))
        {
            return new GraphNode(id, id, "topic_cluster");
        }

        if (id.StartsWith("catcomm:", StringComparison.Ordinal))
        {
            return new GraphNode(id, id, "category_community");
        }

        return new GraphNode(id, id, "node");
    }

    private static string DotShape(string kind) => kind switch
    {
        "conversation" => "box",
        "topic_cluster" => "oval",
        "category_community" => "octagon",
        "keyword" => "diamond",
        "category" => "hexagon",
        "topic" => "ellipse",
        _ => "plaintext"
    };

    private static string Preview(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        string normalized = text.Replace("\r", " ").Replace("\n", " ").Trim();
        return normalized.Length > 80 ? normalized[..80] + "..." : normalized;
    }

    private static string Escape(string value)
        => value.Replace("\\", "\\\\").Replace("\"", "\\\"");

    private sealed record GraphNode(string Id, string Label, string Kind);
}

static class CorpusDatabaseWriter
{
    public static void Write(CorpusIndex index, string databasePath)
    {
        var directory = Path.GetDirectoryName(databasePath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        using var connection = new SqliteConnection($"Data Source={databasePath}");
        connection.Open();

        using var transaction = connection.BeginTransaction();
        RecreateSchema(connection, transaction);

        foreach (var analysis in index.Conversations)
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
                InsertTopic(connection, transaction, analysis.Conversation.ConversationId, topic, analysis.Category, analysis.TopicLabel);
            }

            foreach (var chunk in analysis.Chunks)
            {
                InsertChunk(connection, transaction, analysis.Conversation.ConversationId, chunk);
            }

            if (analysis.CategoryPredictions is not null)
            {
                for (int i = 0; i < analysis.CategoryPredictions.Count; i++)
                {
                    InsertCategoryPrediction(
                        connection,
                        transaction,
                        analysis.Conversation.ConversationId,
                        analysis.CategoryPredictions[i],
                        i + 1);
                }
            }

            if (analysis.Embedding is not null)
            {
                InsertEmbedding(connection, transaction, analysis.Conversation.ConversationId, analysis.Embedding);
            }

            foreach (var edge in analysis.GraphEdges)
            {
                InsertEdge(connection, transaction, analysis.Conversation.ConversationId, edge);
            }
        }

        foreach (var cluster in index.TopicClusters)
        {
            InsertCluster(connection, transaction, cluster);
            foreach (var conversationId in cluster.ConversationIds)
            {
                InsertClusterMember(connection, transaction, cluster.ClusterId, conversationId);
            }
        }

        foreach (var community in index.CategoryCommunities)
        {
            InsertCategoryCommunity(connection, transaction, community);
            foreach (var conversationId in community.ConversationIds)
            {
                InsertCategoryCommunityMember(connection, transaction, community.CommunityId, conversationId);
            }
        }

        foreach (var pair in index.KeywordCooccurrences)
        {
            InsertKeywordCooccurrence(connection, transaction, pair);
        }

        foreach (var link in index.CategoryHierarchy)
        {
            InsertCategoryHierarchy(connection, transaction, link);
        }

        foreach (var edge in index.GlobalEdges)
        {
            InsertEdge(connection, transaction, edge.ScopeId, edge);
        }

        transaction.Commit();
    }

    private static void RecreateSchema(SqliteConnection connection, SqliteTransaction transaction)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
DROP TABLE IF EXISTS graph_edges;
DROP TABLE IF EXISTS category_hierarchy;
DROP TABLE IF EXISTS keyword_cooccurrence;
DROP TABLE IF EXISTS category_community_members;
DROP TABLE IF EXISTS category_communities;
DROP TABLE IF EXISTS topic_cluster_members;
DROP TABLE IF EXISTS topic_clusters;
DROP TABLE IF EXISTS conversation_embeddings;
DROP TABLE IF EXISTS conversation_category_predictions;
DROP TABLE IF EXISTS conversation_chunks;
DROP TABLE IF EXISTS conversation_topics;
DROP TABLE IF EXISTS conversation_keywords;
DROP TABLE IF EXISTS messages;
DROP TABLE IF EXISTS conversations;

CREATE TABLE conversations (
    conversation_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_file TEXT NULL,
    created_utc TEXT NULL,
    current_node_id TEXT NULL,
    category TEXT NOT NULL,
    secondary_category TEXT NULL,
    tertiary_category TEXT NULL,
    category_source TEXT NULL,
    category_community_id TEXT NULL,
    category_community_label TEXT NULL,
    topic_label TEXT NULL,
    topic_cluster_id TEXT NULL,
    embedding_provider TEXT NULL,
    embedding_model TEXT NULL,
    model_summary TEXT NULL
);

CREATE TABLE messages (
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
    child_node_ids TEXT NULL
);

CREATE TABLE conversation_keywords (
    conversation_id TEXT NOT NULL,
    keyword TEXT NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY(conversation_id, keyword)
);

CREATE TABLE conversation_topics (
    conversation_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    category TEXT NOT NULL,
    topic_label TEXT NULL,
    PRIMARY KEY(conversation_id, topic)
);

CREATE TABLE conversation_chunks (
    conversation_id TEXT NOT NULL,
    chunk_hash TEXT NOT NULL,
    kind TEXT NOT NULL,
    text_body TEXT NOT NULL,
    token_count INTEGER NULL,
    PRIMARY KEY(conversation_id, chunk_hash)
);

CREATE TABLE conversation_category_predictions (
    conversation_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    category TEXT NOT NULL,
    score REAL NOT NULL,
    source TEXT NOT NULL,
    is_selected INTEGER NOT NULL,
    PRIMARY KEY(conversation_id, rank)
);

CREATE TABLE conversation_embeddings (
    conversation_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    vector_json TEXT NOT NULL
);

CREATE TABLE topic_clusters (
    cluster_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    summary TEXT NULL,
    primary_category TEXT NOT NULL,
    conversation_count INTEGER NOT NULL,
    representative_keywords_json TEXT NOT NULL
);

CREATE TABLE topic_cluster_members (
    cluster_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    PRIMARY KEY(cluster_id, conversation_id)
);

CREATE TABLE category_communities (
    community_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    summary TEXT NULL,
    primary_category TEXT NOT NULL,
    conversation_count INTEGER NOT NULL,
    categories_json TEXT NOT NULL,
    representative_keywords_json TEXT NOT NULL
);

CREATE TABLE category_community_members (
    community_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    PRIMARY KEY(community_id, conversation_id)
);

CREATE TABLE keyword_cooccurrence (
    left_keyword TEXT NOT NULL,
    right_keyword TEXT NOT NULL,
    pair_count INTEGER NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY(left_keyword, right_keyword)
);

CREATE TABLE category_hierarchy (
    parent_category TEXT NOT NULL,
    child_category TEXT NOT NULL,
    conversation_count INTEGER NOT NULL,
    PRIMARY KEY(parent_category, child_category)
);

CREATE TABLE graph_edges (
    scope_id TEXT NOT NULL,
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL NOT NULL,
    metadata_json TEXT NULL,
    PRIMARY KEY(scope_id, from_id, to_id, edge_type)
);

CREATE INDEX ix_messages_conversation_id ON messages(conversation_id);
CREATE INDEX ix_messages_parent_node_id ON messages(parent_node_id);
CREATE INDEX ix_keywords_keyword ON conversation_keywords(keyword);
CREATE INDEX ix_topics_topic ON conversation_topics(topic);
CREATE INDEX ix_topic_cluster_members_conversation_id ON topic_cluster_members(conversation_id);
CREATE INDEX ix_category_community_members_conversation_id ON category_community_members(conversation_id);
CREATE INDEX ix_graph_edges_from_to ON graph_edges(from_id, to_id);
";
        cmd.ExecuteNonQuery();
    }

    private static void InsertConversation(SqliteConnection connection, SqliteTransaction transaction, ConversationAnalysis analysis)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT INTO conversations(
    conversation_id, title, source_file, created_utc, current_node_id, category, secondary_category, tertiary_category, category_source, category_community_id, category_community_label,
    topic_label, topic_cluster_id, embedding_provider, embedding_model, model_summary)
VALUES (
    $conversation_id, $title, $source_file, $created_utc, $current_node_id, $category, $secondary_category, $tertiary_category, $category_source, $category_community_id, $category_community_label,
    $topic_label, $topic_cluster_id, $embedding_provider, $embedding_model, $model_summary);";
        cmd.Parameters.AddWithValue("$conversation_id", analysis.Conversation.ConversationId);
        cmd.Parameters.AddWithValue("$title", analysis.Conversation.Title);
        cmd.Parameters.AddWithValue("$source_file", (object?)analysis.Conversation.SourceFile ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$created_utc", (object?)FormatUnix(analysis.Conversation.ConversationTime) ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$current_node_id", (object?)analysis.Conversation.CurrentNodeId ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$category", analysis.Category);
        cmd.Parameters.AddWithValue("$secondary_category", (object?)analysis.SecondaryCategory ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$tertiary_category", (object?)analysis.TertiaryCategory ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$category_source", (object?)analysis.CategorySource ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$category_community_id", (object?)analysis.CategoryCommunityId ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$category_community_label", (object?)analysis.CategoryCommunityLabel ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$topic_label", (object?)analysis.TopicLabel ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$topic_cluster_id", (object?)analysis.TopicClusterId ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$embedding_provider", (object?)analysis.Embedding?.Provider ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$embedding_model", (object?)analysis.Embedding?.Model ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$model_summary", (object?)analysis.ModelSummary ?? DBNull.Value);
        cmd.ExecuteNonQuery();
    }

    private static void InsertMessage(SqliteConnection connection, SqliteTransaction transaction, Conversation conversation, MessageNode node)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT INTO messages(
    node_id, conversation_id, parent_node_id, role, author_name, created_utc, content_type, channel, recipient,
    text_body, attachments_json, aggregate_code, execution_output, child_node_ids)
VALUES (
    $node_id, $conversation_id, $parent_node_id, $role, $author_name, $created_utc, $content_type, $channel, $recipient,
    $text_body, $attachments_json, $aggregate_code, $execution_output, $child_node_ids);";
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

    private static void InsertTopic(
        SqliteConnection connection,
        SqliteTransaction transaction,
        string conversationId,
        string topic,
        string category,
        string? topicLabel)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT INTO conversation_topics(conversation_id, topic, category, topic_label)
VALUES ($conversation_id, $topic, $category, $topic_label);";
        cmd.Parameters.AddWithValue("$conversation_id", conversationId);
        cmd.Parameters.AddWithValue("$topic", topic);
        cmd.Parameters.AddWithValue("$category", category);
        cmd.Parameters.AddWithValue("$topic_label", (object?)topicLabel ?? DBNull.Value);
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

    private static void InsertCategoryPrediction(
        SqliteConnection connection,
        SqliteTransaction transaction,
        string conversationId,
        CategoryPrediction prediction,
        int rank)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT INTO conversation_category_predictions(conversation_id, rank, category, score, source, is_selected)
VALUES ($conversation_id, $rank, $category, $score, $source, $is_selected);";
        cmd.Parameters.AddWithValue("$conversation_id", conversationId);
        cmd.Parameters.AddWithValue("$rank", rank);
        cmd.Parameters.AddWithValue("$category", prediction.Category);
        cmd.Parameters.AddWithValue("$score", prediction.Score);
        cmd.Parameters.AddWithValue("$source", prediction.Source);
        cmd.Parameters.AddWithValue("$is_selected", prediction.IsSelected ? 1 : 0);
        cmd.ExecuteNonQuery();
    }

    private static void InsertEmbedding(SqliteConnection connection, SqliteTransaction transaction, string conversationId, ConversationEmbedding embedding)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT INTO conversation_embeddings(conversation_id, provider, model, dimension, vector_json)
VALUES ($conversation_id, $provider, $model, $dimension, $vector_json);";
        cmd.Parameters.AddWithValue("$conversation_id", conversationId);
        cmd.Parameters.AddWithValue("$provider", embedding.Provider);
        cmd.Parameters.AddWithValue("$model", embedding.Model);
        cmd.Parameters.AddWithValue("$dimension", embedding.Dimension);
        cmd.Parameters.AddWithValue("$vector_json", JsonSerializer.Serialize(embedding.Values));
        cmd.ExecuteNonQuery();
    }

    private static void InsertCluster(SqliteConnection connection, SqliteTransaction transaction, TopicCluster cluster)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT INTO topic_clusters(cluster_id, label, summary, primary_category, conversation_count, representative_keywords_json)
VALUES ($cluster_id, $label, $summary, $primary_category, $conversation_count, $representative_keywords_json);";
        cmd.Parameters.AddWithValue("$cluster_id", cluster.ClusterId);
        cmd.Parameters.AddWithValue("$label", cluster.Label);
        cmd.Parameters.AddWithValue("$summary", (object?)cluster.Summary ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$primary_category", cluster.PrimaryCategory);
        cmd.Parameters.AddWithValue("$conversation_count", cluster.ConversationIds.Count);
        cmd.Parameters.AddWithValue("$representative_keywords_json", JsonSerializer.Serialize(cluster.RepresentativeKeywords));
        cmd.ExecuteNonQuery();
    }

    private static void InsertClusterMember(SqliteConnection connection, SqliteTransaction transaction, string clusterId, string conversationId)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT INTO topic_cluster_members(cluster_id, conversation_id)
VALUES ($cluster_id, $conversation_id);";
        cmd.Parameters.AddWithValue("$cluster_id", clusterId);
        cmd.Parameters.AddWithValue("$conversation_id", conversationId);
        cmd.ExecuteNonQuery();
    }

    private static void InsertCategoryCommunity(SqliteConnection connection, SqliteTransaction transaction, CategoryCommunity community)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT INTO category_communities(community_id, label, summary, primary_category, conversation_count, categories_json, representative_keywords_json)
VALUES ($community_id, $label, $summary, $primary_category, $conversation_count, $categories_json, $representative_keywords_json);";
        cmd.Parameters.AddWithValue("$community_id", community.CommunityId);
        cmd.Parameters.AddWithValue("$label", community.Label);
        cmd.Parameters.AddWithValue("$summary", (object?)community.Summary ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$primary_category", community.PrimaryCategory);
        cmd.Parameters.AddWithValue("$conversation_count", community.ConversationIds.Count);
        cmd.Parameters.AddWithValue("$categories_json", JsonSerializer.Serialize(community.Categories));
        cmd.Parameters.AddWithValue("$representative_keywords_json", JsonSerializer.Serialize(community.RepresentativeKeywords));
        cmd.ExecuteNonQuery();
    }

    private static void InsertCategoryCommunityMember(SqliteConnection connection, SqliteTransaction transaction, string communityId, string conversationId)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT INTO category_community_members(community_id, conversation_id)
VALUES ($community_id, $conversation_id);";
        cmd.Parameters.AddWithValue("$community_id", communityId);
        cmd.Parameters.AddWithValue("$conversation_id", conversationId);
        cmd.ExecuteNonQuery();
    }

    private static void InsertKeywordCooccurrence(SqliteConnection connection, SqliteTransaction transaction, KeywordCooccurrence pair)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT INTO keyword_cooccurrence(left_keyword, right_keyword, pair_count, weight)
VALUES ($left_keyword, $right_keyword, $pair_count, $weight);";
        cmd.Parameters.AddWithValue("$left_keyword", pair.LeftKeyword);
        cmd.Parameters.AddWithValue("$right_keyword", pair.RightKeyword);
        cmd.Parameters.AddWithValue("$pair_count", pair.Count);
        cmd.Parameters.AddWithValue("$weight", pair.Weight);
        cmd.ExecuteNonQuery();
    }

    private static void InsertCategoryHierarchy(SqliteConnection connection, SqliteTransaction transaction, CategoryHierarchyLink link)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT INTO category_hierarchy(parent_category, child_category, conversation_count)
VALUES ($parent_category, $child_category, $conversation_count);";
        cmd.Parameters.AddWithValue("$parent_category", link.ParentCategory);
        cmd.Parameters.AddWithValue("$child_category", link.ChildCategory);
        cmd.Parameters.AddWithValue("$conversation_count", link.ConversationCount);
        cmd.ExecuteNonQuery();
    }

    private static void InsertEdge(SqliteConnection connection, SqliteTransaction transaction, string scopeId, GraphEdge edge)
    {
        using var cmd = connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
INSERT INTO graph_edges(scope_id, from_id, to_id, edge_type, weight, metadata_json)
VALUES ($scope_id, $from_id, $to_id, $edge_type, $weight, $metadata_json);";
        cmd.Parameters.AddWithValue("$scope_id", scopeId);
        cmd.Parameters.AddWithValue("$from_id", edge.FromId);
        cmd.Parameters.AddWithValue("$to_id", edge.ToId);
        cmd.Parameters.AddWithValue("$edge_type", edge.EdgeType);
        cmd.Parameters.AddWithValue("$weight", edge.Weight);
        cmd.Parameters.AddWithValue("$metadata_json", (object?)edge.MetadataJson ?? DBNull.Value);
        cmd.ExecuteNonQuery();
    }

    private static string? FormatUnix(double? unix)
    {
        if (unix is null)
        {
            return null;
        }

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

sealed class StringTupleComparer : IEqualityComparer<(string ScopeId, string FromId, string ToId, string EdgeType)>
{
    public static readonly StringTupleComparer Ordinal = new(StringComparer.Ordinal);

    private readonly StringComparer _comparer;

    private StringTupleComparer(StringComparer comparer)
    {
        _comparer = comparer;
    }

    public bool Equals((string ScopeId, string FromId, string ToId, string EdgeType) x, (string ScopeId, string FromId, string ToId, string EdgeType) y)
        => _comparer.Equals(x.ScopeId, y.ScopeId)
            && _comparer.Equals(x.FromId, y.FromId)
            && _comparer.Equals(x.ToId, y.ToId)
            && _comparer.Equals(x.EdgeType, y.EdgeType);

    public int GetHashCode((string ScopeId, string FromId, string ToId, string EdgeType) obj)
        => HashCode.Combine(
            _comparer.GetHashCode(obj.ScopeId),
            _comparer.GetHashCode(obj.FromId),
            _comparer.GetHashCode(obj.ToId),
            _comparer.GetHashCode(obj.EdgeType));
}

sealed class ConversationPairComparer : IEqualityComparer<(string LeftId, string RightId)>
{
    public static readonly ConversationPairComparer Ordinal = new(StringComparer.Ordinal);
    public static readonly ConversationPairComparer OrdinalIgnoreCase = new(StringComparer.OrdinalIgnoreCase);

    private readonly StringComparer _comparer;

    private ConversationPairComparer(StringComparer comparer)
    {
        _comparer = comparer;
    }

    public bool Equals((string LeftId, string RightId) x, (string LeftId, string RightId) y)
        => _comparer.Equals(x.LeftId, y.LeftId) && _comparer.Equals(x.RightId, y.RightId);

    public int GetHashCode((string LeftId, string RightId) obj)
        => HashCode.Combine(_comparer.GetHashCode(obj.LeftId), _comparer.GetHashCode(obj.RightId));
}
