using System.Globalization;
using System.Text;
using System.Text.Json;

static class NavigatorWriter
{
    public static async Task WriteAsync(CorpusIndex index, string outputDirectory, string graphDirectory)
    {
        Directory.CreateDirectory(graphDirectory);

        var categoryComparison = await ReadJsonAsync<NavigatorCategoryComparisonFile>(
            Path.Combine(graphDirectory, "category-comparison.json"));
        var perspectiveSummary = await ReadJsonAsync<NavigatorPerspectiveSummaryFile>(
            Path.Combine(graphDirectory, "perspective-summary.json"));

        var categoryComparisonMap = (categoryComparison?.Conversations ?? Array.Empty<NavigatorCategoryComparisonConversation>())
            .ToDictionary(row => row.ConversationId, StringComparer.Ordinal);
        var perspectiveMap = (perspectiveSummary?.Conversations ?? Array.Empty<NavigatorPerspectiveConversation>())
            .ToDictionary(row => row.ConversationId, StringComparer.Ordinal);
        var clusterLookup = index.TopicClusters.ToDictionary(cluster => cluster.ClusterId, StringComparer.Ordinal);
        var activeNeighborMap = BuildActiveNeighborMap(index);

        var conversations = index.Conversations
            .Select(analysis => BuildConversationRow(
                analysis,
                clusterLookup,
                activeNeighborMap.GetValueOrDefault(analysis.Conversation.ConversationId),
                categoryComparisonMap.GetValueOrDefault(analysis.Conversation.ConversationId),
                perspectiveMap.GetValueOrDefault(analysis.Conversation.ConversationId)))
            .OrderByDescending(row => row.PerspectiveScore)
            .ThenByDescending(row => row.BridgeScore)
            .ThenBy(row => row.Title, StringComparer.OrdinalIgnoreCase)
            .ToList();

        var summary = BuildSummary(index, conversations, perspectiveSummary);
        var insights = BuildInsights(index, conversations);
        var bundle = new NavigatorBundle(summary, insights, conversations);

        File.WriteAllText(
            Path.Combine(graphDirectory, "insights.json"),
            JsonSerializer.Serialize(new NavigatorInsightReport(summary, insights), JsonOptions.WriteIndented),
            new UTF8Encoding(false));

        File.WriteAllText(
            Path.Combine(graphDirectory, "insights.md"),
            BuildInsightsMarkdown(summary, insights),
            new UTF8Encoding(false));

        File.WriteAllText(
            Path.Combine(graphDirectory, "conversation-catalog.csv"),
            BuildConversationCatalogCsv(conversations),
            new UTF8Encoding(false));

        File.WriteAllText(
            Path.Combine(graphDirectory, "bridge-conversations.csv"),
            BuildBridgeConversationCsv(insights.BridgeConversations),
            new UTF8Encoding(false));

        File.WriteAllText(
            Path.Combine(graphDirectory, "cluster-overview.csv"),
            BuildClusterOverviewCsv(insights.TopClusters),
            new UTF8Encoding(false));

        File.WriteAllText(
            Path.Combine(graphDirectory, "category-community-overview.csv"),
            BuildCategoryCommunityOverviewCsv(insights.TopCategoryCommunities),
            new UTF8Encoding(false));

        File.WriteAllText(
            Path.Combine(graphDirectory, "navigator-data.js"),
            "window.CHATDUMP_NAVIGATOR_DATA = " + JsonSerializer.Serialize(bundle, JsonOptions.WriteIndented) + ";",
            new UTF8Encoding(false));

        File.WriteAllText(Path.Combine(graphDirectory, "navigator.html"), NavigatorHtml, new UTF8Encoding(false));
        File.WriteAllText(Path.Combine(graphDirectory, "navigator.css"), NavigatorCss, new UTF8Encoding(false));
        File.WriteAllText(Path.Combine(graphDirectory, "navigator.js"), NavigatorJs, new UTF8Encoding(false));

        Console.WriteLine($"Navigator:   {Path.Combine(graphDirectory, "navigator.html")}");
        Console.WriteLine($"Insights:    {Path.Combine(graphDirectory, "insights.md")}");
    }

    private static IReadOnlyDictionary<string, IReadOnlyList<NavigatorNeighbor>> BuildActiveNeighborMap(CorpusIndex index)
    {
        var titleLookup = index.Conversations.ToDictionary(
            analysis => analysis.Conversation.ConversationId,
            analysis => analysis.Conversation.Title,
            StringComparer.Ordinal);
        var categoryLookup = index.Conversations.ToDictionary(
            analysis => analysis.Conversation.ConversationId,
            analysis => analysis.Category,
            StringComparer.Ordinal);
        var clusterLookup = index.Conversations.ToDictionary(
            analysis => analysis.Conversation.ConversationId,
            analysis => analysis.TopicClusterId,
            StringComparer.Ordinal);

        var neighbors = index.Conversations.ToDictionary(
            analysis => analysis.Conversation.ConversationId,
            _ => new List<NavigatorNeighbor>(),
            StringComparer.Ordinal);

        foreach (var edge in index.GlobalEdges.Where(edge => edge.EdgeType == "similar_to"))
        {
            AddNeighbor(neighbors, edge.FromId, edge.ToId, edge.Weight, titleLookup, categoryLookup, clusterLookup);
            AddNeighbor(neighbors, edge.ToId, edge.FromId, edge.Weight, titleLookup, categoryLookup, clusterLookup);
        }

        return neighbors.ToDictionary(
            pair => pair.Key,
            pair => (IReadOnlyList<NavigatorNeighbor>)pair.Value
                .OrderByDescending(neighbor => neighbor.Score)
                .ThenBy(neighbor => neighbor.Title, StringComparer.OrdinalIgnoreCase)
                .Take(12)
                .ToList(),
            StringComparer.Ordinal);
    }

    private static void AddNeighbor(
        Dictionary<string, List<NavigatorNeighbor>> neighbors,
        string sourceId,
        string targetId,
        double score,
        IReadOnlyDictionary<string, string> titleLookup,
        IReadOnlyDictionary<string, string> categoryLookup,
        IReadOnlyDictionary<string, string?> clusterLookup)
    {
        if (!neighbors.TryGetValue(sourceId, out var list))
        {
            return;
        }

        list.Add(new NavigatorNeighbor(
            targetId,
            titleLookup.GetValueOrDefault(targetId) ?? targetId,
            Math.Round(score, 4),
            categoryLookup.GetValueOrDefault(targetId) ?? AnalysisDefaults.DefaultCategory,
            clusterLookup.GetValueOrDefault(targetId)));
    }

    private static NavigatorConversationRow BuildConversationRow(
        ConversationAnalysis analysis,
        IReadOnlyDictionary<string, TopicCluster> clusterLookup,
        IReadOnlyList<NavigatorNeighbor>? activeNeighbors,
        NavigatorCategoryComparisonConversation? categoryComparison,
        NavigatorPerspectiveConversation? perspective)
    {
        string safe = BuildSafeFileName(analysis.Conversation.Title, analysis.Conversation.ConversationId);
        string transcriptPath = $"../transcripts/{EncodeHrefSegment(safe)}.txt";
        int branchCount = ConversationTraversal.GetLeafPaths(analysis.Conversation).Count;
        string? clusterLabel = null;
        string? clusterSummary = null;
        if (!string.IsNullOrWhiteSpace(analysis.TopicClusterId)
            && clusterLookup.TryGetValue(analysis.TopicClusterId, out var cluster))
        {
            clusterLabel = cluster.Label;
            clusterSummary = cluster.Summary;
        }

        var keywords = analysis.Keywords
            .Take(18)
            .Select(keyword => new NavigatorKeyword(keyword.Keyword, Math.Round(keyword.Score, 4)))
            .ToList();

        var distinctFacetCategories = new[]
            {
                analysis.Category,
                analysis.SecondaryCategory,
                analysis.TertiaryCategory
            }
            .Where(category => !string.IsNullOrWhiteSpace(category))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        var neighbors = activeNeighbors ?? Array.Empty<NavigatorNeighbor>();
        var crossCategoryNeighbors = neighbors
            .Where(neighbor => !string.Equals(neighbor.PrimaryCategory, analysis.Category, StringComparison.OrdinalIgnoreCase))
            .ToList();
        var neighborCategories = crossCategoryNeighbors
            .Select(neighbor => neighbor.PrimaryCategory)
            .Where(category => !string.IsNullOrWhiteSpace(category))
            .Select(category => category!)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(category => category, StringComparer.OrdinalIgnoreCase)
            .ToList();

        int bridgeScore = Math.Max(0, (distinctFacetCategories.Count - 1) * 4)
            + (neighborCategories.Count * 2)
            + crossCategoryNeighbors.Count;

        bool categoryDisagrees = perspective?.CategoryDisagrees
            ?? (categoryComparison is not null && !categoryComparison.Agrees && !string.IsNullOrWhiteSpace(categoryComparison.MlNetCategory));
        int embeddingDifferenceCount = perspective?.EmbeddingDifferenceCount ?? 0;
        int perspectiveScore = perspective?.PerspectiveScore ?? ((categoryDisagrees ? 100 : 0) + embeddingDifferenceCount);
        bool strongSignal = categoryDisagrees || embeddingDifferenceCount >= 4;

        string searchText = string.Join(
            " ",
            new[]
            {
                analysis.Conversation.Title,
                analysis.Category,
                analysis.SecondaryCategory,
                analysis.TertiaryCategory,
                analysis.CategoryCommunityLabel,
                analysis.TopicLabel,
                clusterLabel,
                analysis.ModelSummary,
                string.Join(" ", analysis.Topics),
                string.Join(" ", keywords.Select(keyword => keyword.Keyword))
            }
            .Where(value => !string.IsNullOrWhiteSpace(value)))
            .ToLowerInvariant();

        return new NavigatorConversationRow(
            analysis.Conversation.ConversationId,
            analysis.Conversation.Title,
            FormatUnix(analysis.Conversation.ConversationTime),
            analysis.Conversation.SourceFile,
            branchCount,
            analysis.Category,
            analysis.SecondaryCategory,
            analysis.TertiaryCategory,
            analysis.CategorySource,
            analysis.CategoryCommunityId,
            analysis.CategoryCommunityLabel,
            analysis.TopicClusterId,
            clusterLabel,
            clusterSummary,
            analysis.TopicLabel,
            analysis.ModelSummary,
            analysis.Topics.ToList(),
            keywords,
            analysis.CategoryPredictions is { } categoryPredictions
                ? categoryPredictions.Take(8).ToList()
                : Array.Empty<CategoryPrediction>(),
            transcriptPath,
            searchText,
            categoryDisagrees,
            categoryComparison?.BaselineCategory,
            categoryComparison?.MlNetCategory,
            categoryComparison?.SelectedSource,
            categoryComparison?.SelectedCategory,
            perspectiveScore,
            embeddingDifferenceCount,
            bridgeScore,
            crossCategoryNeighbors.Count,
            neighborCategories,
            neighbors.ToList(),
            perspective is { HashOnlyExamples: { } hashOnlyExamples }
                ? hashOnlyExamples.Select(example => new NavigatorNeighbor(example.ConversationId, example.Title, example.Score, null, null)).ToList()
                : Array.Empty<NavigatorNeighbor>(),
            perspective is { OnnxOnlyExamples: { } onnxOnlyExamples }
                ? onnxOnlyExamples.Select(example => new NavigatorNeighbor(example.ConversationId, example.Title, example.Score, null, null)).ToList()
                : Array.Empty<NavigatorNeighbor>(),
            strongSignal);
    }

    private static NavigatorSummary BuildSummary(
        CorpusIndex index,
        IReadOnlyList<NavigatorConversationRow> conversations,
        NavigatorPerspectiveSummaryFile? perspectiveSummary)
    {
        var activeEmbedding = index.Conversations
            .Select(conversation => conversation.Embedding)
            .FirstOrDefault(embedding => embedding is not null);

        int similarityEdgeCount = index.GlobalEdges.Count(edge => edge.EdgeType == "similar_to");
        int categoryCount = conversations
            .SelectMany(conversation => new[] { conversation.PrimaryCategory, conversation.SecondaryCategory, conversation.TertiaryCategory })
            .Where(category => !string.IsNullOrWhiteSpace(category))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Count();

        return new NavigatorSummary(
            DateTimeOffset.UtcNow.ToString("u", CultureInfo.InvariantCulture),
            conversations.Count,
            index.TopicClusters.Count,
            index.CategoryCommunities.Count,
            categoryCount,
            similarityEdgeCount,
            index.KeywordCooccurrences.Count,
            conversations.Count(conversation => !string.IsNullOrWhiteSpace(conversation.SecondaryCategory)),
            conversations.Count(conversation => !string.IsNullOrWhiteSpace(conversation.TertiaryCategory)),
            perspectiveSummary?.CategoryDisagreementCount ?? conversations.Count(conversation => conversation.CategoryDisagrees),
            perspectiveSummary?.EmbeddingDivergenceCount ?? conversations.Count(conversation => conversation.EmbeddingDifferenceCount > 0),
            perspectiveSummary?.StrongSignalCount ?? conversations.Count(conversation => conversation.StrongSignal),
            activeEmbedding?.Provider,
            activeEmbedding?.Model);
    }

    private static NavigatorInsights BuildInsights(CorpusIndex index, IReadOnlyList<NavigatorConversationRow> conversations)
    {
        var categoryInsights = conversations
            .SelectMany(conversation => new[]
            {
                (category: conversation.PrimaryCategory, facet: "primary", conversation),
                (category: conversation.SecondaryCategory, facet: "secondary", conversation),
                (category: conversation.TertiaryCategory, facet: "tertiary", conversation)
            })
            .Where(item => !string.IsNullOrWhiteSpace(item.category))
            .GroupBy(item => item.category!, StringComparer.OrdinalIgnoreCase)
            .Select(group =>
            {
                var grouped = group.ToList();
                var examples = grouped
                    .Select(item => item.conversation)
                    .DistinctBy(conversation => conversation.ConversationId, StringComparer.Ordinal)
                    .OrderByDescending(conversation => conversation.PerspectiveScore)
                    .ThenByDescending(conversation => conversation.BridgeScore)
                    .ThenBy(conversation => conversation.Title, StringComparer.OrdinalIgnoreCase)
                    .Take(3)
                    .Select(conversation => conversation.Title)
                    .ToList();

                var topTopics = grouped
                    .Where(item => string.Equals(item.facet, "primary", StringComparison.Ordinal))
                    .SelectMany(item => item.conversation.Topics)
                    .GroupBy(topic => topic, StringComparer.OrdinalIgnoreCase)
                    .OrderByDescending(topic => topic.Count())
                    .ThenBy(topic => topic.Key, StringComparer.OrdinalIgnoreCase)
                    .Take(5)
                    .Select(topic => topic.Key)
                    .ToList();

                return new NavigatorCategoryInsight(
                    group.Key,
                    grouped.Count(item => string.Equals(item.facet, "primary", StringComparison.Ordinal)),
                    grouped.Select(item => item.conversation.ConversationId).Distinct(StringComparer.Ordinal).Count(),
                    grouped.Count(item => string.Equals(item.facet, "secondary", StringComparison.Ordinal)),
                    grouped.Count(item => string.Equals(item.facet, "tertiary", StringComparison.Ordinal)),
                    grouped.Select(item => item.conversation.TopicClusterId).Where(id => !string.IsNullOrWhiteSpace(id)).Distinct(StringComparer.Ordinal).Count(),
                    topTopics,
                    examples);
            })
            .OrderByDescending(item => item.AnyFacetCount)
            .ThenBy(item => item.Category, StringComparer.OrdinalIgnoreCase)
            .Take(20)
            .ToList();

        var categoryBlends = conversations
            .SelectMany(BuildCategoryBlendPairs)
            .GroupBy(item => (item.LeftCategory, item.RightCategory), NavigatorCategoryBlendComparer.OrdinalIgnoreCase)
            .Select(group => new NavigatorCategoryBlend(
                group.Key.LeftCategory,
                group.Key.RightCategory,
                group.Count(),
                group.Select(item => item.Title)
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .Take(3)
                    .ToList()))
            .OrderByDescending(item => item.Count)
            .ThenBy(item => item.LeftCategory, StringComparer.OrdinalIgnoreCase)
            .ThenBy(item => item.RightCategory, StringComparer.OrdinalIgnoreCase)
            .Take(25)
            .ToList();

        var clusterInsights = index.TopicClusters
            .OrderByDescending(cluster => cluster.ConversationIds.Count)
            .ThenBy(cluster => cluster.Label, StringComparer.OrdinalIgnoreCase)
            .Take(40)
            .Select(cluster => new NavigatorClusterInsight(
                cluster.ClusterId,
                cluster.Label,
                cluster.PrimaryCategory,
                cluster.ConversationIds.Count,
                cluster.Summary,
                cluster.RepresentativeKeywords.Take(8).ToList(),
                cluster.ConversationIds
                    .Select(id => conversations.FirstOrDefault(conversation => string.Equals(conversation.ConversationId, id, StringComparison.Ordinal)))
                    .Where(conversation => conversation is not null)
                    .Select(conversation => conversation!)
                    .OrderByDescending(conversation => conversation.PerspectiveScore)
                    .ThenBy(conversation => conversation.Title, StringComparer.OrdinalIgnoreCase)
                    .Take(3)
                    .Select(conversation => conversation.Title)
                    .ToList()))
            .ToList();

        var categoryCommunityInsights = index.CategoryCommunities
            .OrderByDescending(community => community.ConversationIds.Count)
            .ThenBy(community => community.Label, StringComparer.OrdinalIgnoreCase)
            .Take(40)
            .Select(community => new NavigatorCategoryCommunityInsight(
                community.CommunityId,
                community.Label,
                community.PrimaryCategory,
                community.ConversationIds.Count,
                community.Summary,
                community.Categories.Take(8).ToList(),
                community.ConversationIds
                    .Select(id => conversations.FirstOrDefault(conversation => string.Equals(conversation.ConversationId, id, StringComparison.Ordinal)))
                    .Where(conversation => conversation is not null)
                    .Select(conversation => conversation!)
                    .OrderByDescending(conversation => conversation.PerspectiveScore)
                    .ThenBy(conversation => conversation.Title, StringComparer.OrdinalIgnoreCase)
                    .Take(3)
                    .Select(conversation => conversation.Title)
                    .ToList()))
            .ToList();

        var bridgeConversations = conversations
            .Where(conversation => conversation.BridgeScore > 0)
            .OrderByDescending(conversation => conversation.BridgeScore)
            .ThenByDescending(conversation => conversation.CrossCategoryNeighborCount)
            .ThenByDescending(conversation => conversation.PerspectiveScore)
            .ThenBy(conversation => conversation.Title, StringComparer.OrdinalIgnoreCase)
            .Take(100)
            .Select(conversation => new NavigatorBridgeConversation(
                conversation.ConversationId,
                conversation.Title,
                conversation.PrimaryCategory,
                conversation.SecondaryCategory,
                conversation.TertiaryCategory,
                conversation.TopicLabel,
                conversation.TopicClusterLabel,
                conversation.BridgeScore,
                conversation.CrossCategoryNeighborCount,
                conversation.NeighborCategories,
                conversation.TranscriptPath))
            .ToList();

        var crossCategorySimilarities = index.GlobalEdges
            .Where(edge => edge.EdgeType == "similar_to")
            .Select(edge => (edge, left: conversations.FirstOrDefault(c => c.ConversationId == edge.FromId), right: conversations.FirstOrDefault(c => c.ConversationId == edge.ToId)))
            .Where(item => item.left is not null
                && item.right is not null
                && !string.Equals(item.left!.PrimaryCategory, item.right!.PrimaryCategory, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(item => item.edge.Weight)
            .ThenBy(item => item.left!.Title, StringComparer.OrdinalIgnoreCase)
            .Take(100)
            .Select(item => new NavigatorCrossCategorySimilarity(
                item.left!.ConversationId,
                item.left.Title,
                item.left.PrimaryCategory,
                item.right!.ConversationId,
                item.right.Title,
                item.right.PrimaryCategory,
                Math.Round(item.edge.Weight, 4)))
            .ToList();

        var keywordHotspots = index.KeywordCooccurrences
            .OrderByDescending(pair => pair.Weight)
            .ThenByDescending(pair => pair.Count)
            .ThenBy(pair => pair.LeftKeyword, StringComparer.OrdinalIgnoreCase)
            .Take(50)
            .Select(pair => new NavigatorKeywordHotspot(pair.LeftKeyword, pair.RightKeyword, pair.Count, Math.Round(pair.Weight, 4)))
            .ToList();

        var disagreements = conversations
            .Where(conversation => conversation.CategoryDisagrees)
            .OrderByDescending(conversation => conversation.PerspectiveScore)
            .ThenBy(conversation => conversation.Title, StringComparer.OrdinalIgnoreCase)
            .Take(50)
            .Select(conversation => new NavigatorDisagreement(
                conversation.ConversationId,
                conversation.Title,
                conversation.BaselineCategory,
                conversation.MlNetCategory,
                conversation.SelectedCategory,
                conversation.TranscriptPath))
            .ToList();

        var strongSignals = conversations
            .Where(conversation => conversation.StrongSignal)
            .OrderByDescending(conversation => conversation.PerspectiveScore)
            .ThenByDescending(conversation => conversation.BridgeScore)
            .ThenBy(conversation => conversation.Title, StringComparer.OrdinalIgnoreCase)
            .Take(50)
            .Select(conversation => new NavigatorStrongSignal(
                conversation.ConversationId,
                conversation.Title,
                conversation.SelectedCategory ?? conversation.PrimaryCategory,
                conversation.PerspectiveScore,
                conversation.EmbeddingDifferenceCount,
                conversation.CategoryDisagrees,
                conversation.TranscriptPath))
            .ToList();

        return new NavigatorInsights(
            categoryInsights,
            categoryBlends,
            clusterInsights,
            categoryCommunityInsights,
            bridgeConversations,
            crossCategorySimilarities,
            keywordHotspots,
            disagreements,
            strongSignals);
    }

    private static IEnumerable<(string LeftCategory, string RightCategory, string Title)> BuildCategoryBlendPairs(NavigatorConversationRow conversation)
    {
        var categories = new[] { conversation.PrimaryCategory, conversation.SecondaryCategory, conversation.TertiaryCategory }
            .Where(category => !string.IsNullOrWhiteSpace(category))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(category => category, StringComparer.OrdinalIgnoreCase)
            .ToList();

        for (int i = 0; i < categories.Count; i++)
        {
            for (int j = i + 1; j < categories.Count; j++)
            {
                yield return (categories[i]!, categories[j]!, conversation.Title);
            }
        }
    }

    private static string BuildInsightsMarkdown(NavigatorSummary summary, NavigatorInsights insights)
    {
        var sb = new StringBuilder();
        sb.AppendLine("# Corpus Insights");
        sb.AppendLine();
        sb.AppendLine("## Overview");
        sb.AppendLine();
        sb.AppendLine($"- Conversations: {summary.ConversationCount:N0}");
        sb.AppendLine($"- Topic clusters: {summary.ClusterCount:N0}");
        sb.AppendLine($"- Category communities: {summary.CategoryCommunityCount:N0}");
        sb.AppendLine($"- Distinct categories across all facets: {summary.CategoryCount:N0}");
        sb.AppendLine($"- Similarity edges: {summary.SimilarityEdgeCount:N0}");
        sb.AppendLine($"- Keyword hotspots: {summary.KeywordHotspotCount:N0}");
        sb.AppendLine($"- With secondary category: {summary.WithSecondaryCategoryCount:N0}");
        sb.AppendLine($"- With tertiary category: {summary.WithTertiaryCategoryCount:N0}");
        sb.AppendLine($"- Category disagreements: {summary.CategoryDisagreementCount:N0}");
        sb.AppendLine($"- Strong signals: {summary.StrongSignalCount:N0}");

        sb.AppendLine();
        sb.AppendLine("## Top Categories");
        sb.AppendLine();
        int rank = 1;
        foreach (var category in insights.TopCategories.Take(12))
        {
            sb.AppendLine($"{rank}. {category.Category} - primary {category.PrimaryCount:N0}, any facet {category.AnyFacetCount:N0}");
            rank++;
        }

        if (insights.BridgeConversations.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("## Bridge Conversations");
            sb.AppendLine();
            rank = 1;
            foreach (var bridge in insights.BridgeConversations.Take(15))
            {
                sb.AppendLine($"{rank}. {bridge.Title} - bridge score {bridge.BridgeScore:N0}, cross-category neighbors {bridge.CrossCategoryNeighborCount:N0}");
                rank++;
            }
        }

        if (insights.StrongSignals.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("## Strong Signals");
            sb.AppendLine();
            rank = 1;
            foreach (var signal in insights.StrongSignals.Take(15))
            {
                sb.AppendLine($"{rank}. {signal.Title} - perspective {signal.PerspectiveScore:N0}, embedding difference {signal.EmbeddingDifferenceCount:N0}");
                rank++;
            }
        }

        if (insights.TopClusters.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("## Largest Clusters");
            sb.AppendLine();
            rank = 1;
            foreach (var cluster in insights.TopClusters.Take(12))
            {
                sb.AppendLine($"{rank}. {cluster.Label} - {cluster.ConversationCount:N0} conversations [{cluster.PrimaryCategory}]");
                rank++;
            }
        }

        if (insights.TopCategoryCommunities.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("## Largest Category Communities");
            sb.AppendLine();
            rank = 1;
            foreach (var community in insights.TopCategoryCommunities.Take(12))
            {
                sb.AppendLine($"{rank}. {community.Label} - {community.ConversationCount:N0} conversations [{community.PrimaryCategory}]");
                rank++;
            }
        }

        sb.AppendLine();
        sb.AppendLine("## Exports");
        sb.AppendLine();
        sb.AppendLine("- `conversation-catalog.csv`");
        sb.AppendLine("- `bridge-conversations.csv`");
        sb.AppendLine("- `cluster-overview.csv`");
        sb.AppendLine("- `category-community-overview.csv`");
        sb.AppendLine("- `navigator.html`");

        return sb.ToString();
    }

    private static string BuildConversationCatalogCsv(IReadOnlyList<NavigatorConversationRow> conversations)
    {
        var sb = new StringBuilder();
        sb.AppendLine("ConversationId,Title,CreatedUtc,SourceFile,BranchCount,PrimaryCategory,SecondaryCategory,TertiaryCategory,CategorySource,CategoryCommunityId,CategoryCommunityLabel,ClusterId,ClusterLabel,TopicLabel,Topics,Keywords,CategoryDisagrees,PerspectiveScore,BridgeScore,TranscriptPath");
        foreach (var conversation in conversations)
        {
            sb.AppendLine(string.Join(",",
                EscapeCsv(conversation.ConversationId),
                EscapeCsv(conversation.Title),
                EscapeCsv(conversation.CreatedUtc),
                EscapeCsv(conversation.SourceFile),
                EscapeCsv(conversation.BranchCount.ToString(CultureInfo.InvariantCulture)),
                EscapeCsv(conversation.PrimaryCategory),
                EscapeCsv(conversation.SecondaryCategory),
                EscapeCsv(conversation.TertiaryCategory),
                EscapeCsv(conversation.CategorySource),
                EscapeCsv(conversation.CategoryCommunityId),
                EscapeCsv(conversation.CategoryCommunityLabel),
                EscapeCsv(conversation.TopicClusterId),
                EscapeCsv(conversation.TopicClusterLabel),
                EscapeCsv(conversation.TopicLabel),
                EscapeCsv(string.Join(" | ", conversation.Topics)),
                EscapeCsv(string.Join(" | ", conversation.Keywords.Select(keyword => keyword.Keyword))),
                EscapeCsv(conversation.CategoryDisagrees ? "true" : "false"),
                EscapeCsv(conversation.PerspectiveScore.ToString(CultureInfo.InvariantCulture)),
                EscapeCsv(conversation.BridgeScore.ToString(CultureInfo.InvariantCulture)),
                EscapeCsv(conversation.TranscriptPath)));
        }

        return sb.ToString();
    }

    private static string BuildCategoryCommunityOverviewCsv(IReadOnlyList<NavigatorCategoryCommunityInsight> communities)
    {
        var sb = new StringBuilder();
        sb.AppendLine("CommunityId,Label,PrimaryCategory,ConversationCount,Summary,Categories,Examples");
        foreach (var community in communities)
        {
            sb.AppendLine(string.Join(",",
                EscapeCsv(community.CommunityId),
                EscapeCsv(community.Label),
                EscapeCsv(community.PrimaryCategory),
                EscapeCsv(community.ConversationCount.ToString(CultureInfo.InvariantCulture)),
                EscapeCsv(community.Summary),
                EscapeCsv(string.Join(" | ", community.Categories)),
                EscapeCsv(string.Join(" | ", community.ExampleTitles))));
        }

        return sb.ToString();
    }

    private static string BuildBridgeConversationCsv(IReadOnlyList<NavigatorBridgeConversation> bridges)
    {
        var sb = new StringBuilder();
        sb.AppendLine("ConversationId,Title,PrimaryCategory,SecondaryCategory,TertiaryCategory,TopicLabel,ClusterLabel,BridgeScore,CrossCategoryNeighborCount,NeighborCategories,TranscriptPath");
        foreach (var bridge in bridges)
        {
            sb.AppendLine(string.Join(",",
                EscapeCsv(bridge.ConversationId),
                EscapeCsv(bridge.Title),
                EscapeCsv(bridge.PrimaryCategory),
                EscapeCsv(bridge.SecondaryCategory),
                EscapeCsv(bridge.TertiaryCategory),
                EscapeCsv(bridge.TopicLabel),
                EscapeCsv(bridge.TopicClusterLabel),
                EscapeCsv(bridge.BridgeScore.ToString(CultureInfo.InvariantCulture)),
                EscapeCsv(bridge.CrossCategoryNeighborCount.ToString(CultureInfo.InvariantCulture)),
                EscapeCsv(string.Join(" | ", bridge.NeighborCategories)),
                EscapeCsv(bridge.TranscriptPath)));
        }

        return sb.ToString();
    }

    private static string BuildClusterOverviewCsv(IReadOnlyList<NavigatorClusterInsight> clusters)
    {
        var sb = new StringBuilder();
        sb.AppendLine("ClusterId,Label,PrimaryCategory,ConversationCount,Summary,Keywords,Examples");
        foreach (var cluster in clusters)
        {
            sb.AppendLine(string.Join(",",
                EscapeCsv(cluster.ClusterId),
                EscapeCsv(cluster.Label),
                EscapeCsv(cluster.PrimaryCategory),
                EscapeCsv(cluster.ConversationCount.ToString(CultureInfo.InvariantCulture)),
                EscapeCsv(cluster.Summary),
                EscapeCsv(string.Join(" | ", cluster.Keywords)),
                EscapeCsv(string.Join(" | ", cluster.ExampleTitles))));
        }

        return sb.ToString();
    }

    private static string EscapeCsv(string? value)
    {
        string text = value ?? string.Empty;
        if (text.Contains('"'))
        {
            text = text.Replace("\"", "\"\"");
        }

        return "\"" + text + "\"";
    }

    private static string BuildSafeFileName(string title, string conversationId)
    {
        string combined = string.IsNullOrWhiteSpace(title) ? conversationId : $"{title}__{conversationId}";
        foreach (char ch in Path.GetInvalidFileNameChars())
        {
            combined = combined.Replace(ch, '_');
        }

        if (combined.Length > 140)
        {
            combined = combined[..140];
        }

        return combined;
    }

    private static string EncodeHrefSegment(string segment)
        => Uri.EscapeDataString(segment).Replace("%2F", "/");

    private static string? FormatUnix(double? unix)
    {
        if (unix is null)
        {
            return null;
        }

        try
        {
            return DateTimeOffset.FromUnixTimeMilliseconds((long)(unix.Value * 1000)).ToString("u", CultureInfo.InvariantCulture);
        }
        catch
        {
            return unix.Value.ToString(CultureInfo.InvariantCulture);
        }
    }

    private static async Task<T?> ReadJsonAsync<T>(string path)
    {
        if (!File.Exists(path))
        {
            return default;
        }

        await using var stream = File.OpenRead(path);
        return await JsonSerializer.DeserializeAsync<T>(stream, JsonOptions.Default);
    }

    private const string NavigatorHtml = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ChatDump Navigator</title>
  <link rel="stylesheet" href="navigator.css">
</head>
<body>
  <div class="shell">
    <header class="masthead">
      <div>
        <p class="eyebrow">Local Corpus Navigator</p>
        <h1>Conversations, signals, and extractable slices</h1>
        <p class="subhead">Filter the corpus, inspect bridge conversations, review disagreement hotspots, and export the current selection.</p>
      </div>
      <div class="header-links">
        <a href="insights.md">Insights</a>
        <a href="insights.json">Insights JSON</a>
        <a href="conversation-catalog.csv">Catalog CSV</a>
        <a href="bridge-conversations.csv">Bridge CSV</a>
        <a href="category-community-overview.csv">Community CSV</a>
      </div>
    </header>

    <section class="summary-grid" id="summary-grid"></section>

    <section class="workspace">
      <aside class="panel filters">
        <h2>Filters</h2>
        <label><span>Search</span><input id="search-box" type="search" placeholder="title, topic, keyword, category"></label>
        <label><span>Primary Category</span><select id="primary-category"></select></label>
        <label><span>Any Facet Category</span><select id="facet-category"></select></label>
        <label><span>Category Community</span><select id="category-community-filter"></select></label>
        <label><span>Cluster</span><select id="cluster-filter"></select></label>
        <label>
          <span>Sort</span>
          <select id="sort-mode">
            <option value="perspective">Perspective</option>
            <option value="bridge">Bridge</option>
            <option value="title">Title</option>
            <option value="created">Created</option>
          </select>
        </label>
        <label class="toggle"><input id="only-disagreements" type="checkbox"> <span>Only category disagreements</span></label>
        <label class="toggle"><input id="only-strong-signals" type="checkbox"> <span>Only strong signals</span></label>
        <label class="toggle"><input id="only-bridges" type="checkbox"> <span>Only bridge conversations</span></label>
        <div class="extract-bar">
          <button id="export-json" type="button">Export JSON</button>
          <button id="export-csv" type="button">Export CSV</button>
        </div>
        <div class="insight-stack">
          <section><h3>Top Categories</h3><div id="insight-categories" class="compact-list"></div></section>
          <section><h3>Category Communities</h3><div id="insight-communities" class="compact-list"></div></section>
          <section><h3>Bridge Queue</h3><div id="insight-bridges" class="compact-list"></div></section>
          <section><h3>Strong Signals</h3><div id="insight-signals" class="compact-list"></div></section>
        </div>
      </aside>

      <main class="panel catalog">
        <div class="catalog-header">
          <div>
            <h2>Conversation Catalog</h2>
            <p id="result-count" class="muted"></p>
          </div>
        </div>
        <div id="conversation-list" class="conversation-list"></div>
      </main>

      <aside class="panel detail">
        <div id="detail-panel" class="detail-panel"></div>
      </aside>
    </section>
  </div>

  <script src="navigator-data.js"></script>
  <script src="navigator.js"></script>
</body>
</html>
""";

    private const string NavigatorCss = """
:root{--bg:#f4efe6;--panel:#fffaf2;--panel-strong:#f7f0e4;--ink:#1b2528;--muted:#5f6c70;--line:#d8cdbf;--accent:#0f6d70;--accent-soft:#d7efef;--signal:#9a3d2b;--signal-soft:#f4ddd6;--bridge:#6c4d8a;--bridge-soft:#ece2f5;--shadow:0 18px 40px rgba(27,37,40,.08)}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at top left, rgba(15,109,112,.10), transparent 22rem),radial-gradient(circle at top right, rgba(154,61,43,.08), transparent 24rem),var(--bg);color:var(--ink);font-family:"Aptos","Segoe UI",sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.shell{max-width:1680px;margin:0 auto;padding:24px}
.masthead{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:20px}
.eyebrow{margin:0 0 8px;text-transform:uppercase;letter-spacing:.16em;color:var(--accent);font-size:.78rem;font-weight:700}
.masthead h1{margin:0;font-family:"Bahnschrift","Segoe UI",sans-serif;font-size:2rem;line-height:1.05}
.subhead{margin:10px 0 0;color:var(--muted);max-width:72ch}
.header-links{display:flex;gap:12px;flex-wrap:wrap}
.header-links a{padding:10px 14px;border:1px solid var(--line);border-radius:999px;background:rgba(255,255,255,.6)}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:20px}
.summary-card,.panel{background:var(--panel);border:1px solid var(--line);box-shadow:var(--shadow)}
.summary-card{border-radius:20px;padding:16px 18px}
.summary-card .label{display:block;color:var(--muted);font-size:.82rem;text-transform:uppercase;letter-spacing:.08em}
.summary-card .value{display:block;margin-top:8px;font-size:1.6rem;font-weight:700}.summary-card .meta{display:block;margin-top:6px;color:var(--muted);font-size:.88rem}
.workspace{display:grid;grid-template-columns:320px minmax(0,1fr) 440px;gap:16px}
.panel{border-radius:24px;padding:18px}.panel h2,.panel h3{margin:0 0 12px;font-family:"Bahnschrift","Segoe UI",sans-serif}
.filters label{display:block;margin-bottom:12px}.filters label span{display:block;margin-bottom:6px;font-size:.88rem;color:var(--muted)}
input,select,button{width:100%;font:inherit;border:1px solid var(--line);border-radius:14px;padding:11px 12px;background:#fffdfa;color:var(--ink)}
button{cursor:pointer;background:var(--ink);color:#fff;border-color:var(--ink)}button:hover{filter:brightness(1.05)}
.toggle{display:flex;gap:10px;align-items:center;margin:10px 0}.toggle input{width:auto}.toggle span{margin:0;color:var(--ink)}
.extract-bar{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:16px 0 20px}.insight-stack section+section{margin-top:18px}
.compact-list{display:flex;flex-direction:column;gap:8px}
.compact-item{padding:10px 12px;border-radius:16px;background:var(--panel-strong);border:1px solid var(--line)}
.compact-item strong{display:block}.compact-item span{display:block;color:var(--muted);font-size:.88rem;margin-top:4px}
.catalog-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}.muted{color:var(--muted);margin:0}
.conversation-list{display:flex;flex-direction:column;gap:10px;max-height:calc(100vh - 260px);overflow:auto;padding-right:4px}
.conversation-card{border:1px solid var(--line);border-radius:18px;padding:14px;background:#fffdfa;cursor:pointer}
.conversation-card.active{border-color:var(--accent);box-shadow:0 0 0 2px rgba(15,109,112,.16)}.conversation-card h3{margin:0 0 8px;font-size:1rem}
.conversation-card p{margin:8px 0 0;color:var(--muted);font-size:.92rem}
.tag-row{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.tag{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border-radius:999px;font-size:.82rem;border:1px solid var(--line);background:#fff}
.tag.primary{background:var(--accent-soft);border-color:rgba(15,109,112,.22)}.tag.signal{background:var(--signal-soft);border-color:rgba(154,61,43,.18)}.tag.bridge{background:var(--bridge-soft);border-color:rgba(108,77,138,.18)}
.metric-row{display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;color:var(--muted);font-size:.86rem}
.detail-panel{display:flex;flex-direction:column;gap:16px;max-height:calc(100vh - 260px);overflow:auto;padding-right:6px}.detail-empty{color:var(--muted)}
.detail-card{padding:14px;border:1px solid var(--line);border-radius:18px;background:#fffdfa}.detail-card h3{margin:0 0 8px}.detail-card p{margin:8px 0 0}
.link-list{display:flex;flex-wrap:wrap;gap:10px}.link-list a{padding:8px 10px;border-radius:999px;border:1px solid var(--line);background:#fff}
.neighbor-list{display:flex;flex-direction:column;gap:8px}.neighbor{display:flex;justify-content:space-between;gap:12px;padding:9px 10px;border-radius:14px;background:var(--panel-strong)}.neighbor .meta{color:var(--muted);font-size:.85rem}
@media (max-width:1320px){.workspace{grid-template-columns:300px minmax(0,1fr)}.detail{grid-column:1/-1}.detail-panel{max-height:none}}
@media (max-width:860px){.shell{padding:16px}.masthead,.workspace{display:block}.panel{margin-top:14px}.conversation-list{max-height:none}}
""";

    private const string NavigatorJs = """
(function(){
  const data = window.CHATDUMP_NAVIGATOR_DATA || { Summary:{}, Insights:{}, Conversations:[] };
  const conversations = data.Conversations || [];
  const state = { search:"", primaryCategory:"", facetCategory:"", categoryCommunityId:"", clusterId:"", onlyDisagreements:false, onlyStrongSignals:false, onlyBridges:false, sortMode:"perspective", selectedId: conversations[0] ? conversations[0].ConversationId : null };
  const $ = (id) => document.getElementById(id);
  init();

  function init(){
    renderSummary();
    populateFilters();
    bindEvents();
    renderInsights();
    renderAll();
  }

  function bindEvents(){
    $("search-box").addEventListener("input", e => { state.search = e.target.value.trim().toLowerCase(); renderAll(); });
    $("primary-category").addEventListener("change", e => { state.primaryCategory = e.target.value; renderAll(); });
    $("facet-category").addEventListener("change", e => { state.facetCategory = e.target.value; renderAll(); });
    $("category-community-filter").addEventListener("change", e => { state.categoryCommunityId = e.target.value; renderAll(); });
    $("cluster-filter").addEventListener("change", e => { state.clusterId = e.target.value; renderAll(); });
    $("sort-mode").addEventListener("change", e => { state.sortMode = e.target.value; renderAll(); });
    $("only-disagreements").addEventListener("change", e => { state.onlyDisagreements = e.target.checked; renderAll(); });
    $("only-strong-signals").addEventListener("change", e => { state.onlyStrongSignals = e.target.checked; renderAll(); });
    $("only-bridges").addEventListener("change", e => { state.onlyBridges = e.target.checked; renderAll(); });
    $("export-json").addEventListener("click", exportFilteredJson);
    $("export-csv").addEventListener("click", exportFilteredCsv);
  }

  function populateFilters(){
    populateSelect($("primary-category"), unique(conversations.map(row => row.PrimaryCategory)));
    populateSelect($("facet-category"), unique(conversations.flatMap(row => [row.PrimaryCategory, row.SecondaryCategory, row.TertiaryCategory]).filter(Boolean)));
    populateSelect($("category-community-filter"), uniqueObjects(conversations.filter(row => row.CategoryCommunityId && row.CategoryCommunityLabel).map(row => ({ value: row.CategoryCommunityId, label: row.CategoryCommunityLabel }))));
    populateSelect($("cluster-filter"), uniqueObjects(conversations.filter(row => row.TopicClusterId && row.TopicClusterLabel).map(row => ({ value: row.TopicClusterId, label: row.TopicClusterLabel }))));
  }

  function renderSummary(){
    const summary = data.Summary || {};
    const cards = [
      ["Conversations", summary.ConversationCount || 0, "indexed corpus"],
      ["Clusters", summary.ClusterCount || 0, "topic components"],
      ["Communities", summary.CategoryCommunityCount || 0, "category communities"],
      ["Categories", summary.CategoryCount || 0, "across all facets"],
      ["Similarity", summary.SimilarityEdgeCount || 0, "active neighbor edges"],
      ["Disagreements", summary.CategoryDisagreementCount || 0, "heuristic vs ML.NET"],
      ["Strong Signals", summary.StrongSignalCount || 0, "multi-perspective hotspots"],
      ["Secondary Facets", summary.WithSecondaryCategoryCount || 0, "conversations with secondary category"]
    ];
    $("summary-grid").innerHTML = cards.map(card => `<article class="summary-card"><span class="label">${escapeHtml(card[0])}</span><span class="value">${escapeHtml(String(card[1]))}</span><span class="meta">${escapeHtml(card[2])}</span></article>`).join("");
  }

  function renderInsights(){
    const insights = data.Insights || {};
    $("insight-categories").innerHTML = (insights.TopCategories || []).slice(0,8).map(item => compact(item.Category, item.AnyFacetCount + " conversations")).join("");
    $("insight-communities").innerHTML = (insights.TopCategoryCommunities || []).slice(0,8).map(item => compact(item.Label, item.ConversationCount + " conversations")).join("");
    $("insight-bridges").innerHTML = (insights.BridgeConversations || []).slice(0,8).map(item => compact(item.Title, item.PrimaryCategory + " | bridge " + item.BridgeScore)).join("");
    $("insight-signals").innerHTML = (insights.StrongSignals || []).slice(0,8).map(item => compact(item.Title, item.SelectedCategory + " | perspective " + item.PerspectiveScore)).join("");
  }

  function renderAll(){
    const filtered = getFilteredConversations();
    if (!filtered.some(row => row.ConversationId === state.selectedId)) {
      state.selectedId = filtered[0] ? filtered[0].ConversationId : null;
    }
    $("result-count").textContent = filtered.length + " of " + conversations.length + " conversations";
    renderList(filtered);
    renderDetail(filtered.find(row => row.ConversationId === state.selectedId) || null);
  }

  function renderList(rows){
    $("conversation-list").innerHTML = rows.map(row => {
      const active = row.ConversationId === state.selectedId ? " active" : "";
      const tags = [badge(row.PrimaryCategory,"primary"), row.SecondaryCategory ? badge(row.SecondaryCategory,"") : "", row.TertiaryCategory ? badge(row.TertiaryCategory,"") : "", row.CategoryCommunityLabel ? badge(row.CategoryCommunityLabel,"") : "", row.StrongSignal ? badge("Strong signal","signal") : "", row.BridgeScore > 0 ? badge("Bridge " + row.BridgeScore,"bridge") : "", row.TopicClusterLabel ? badge(row.TopicClusterLabel,"") : ""].join("");
      const summary = row.TopicLabel || row.Summary || (row.Topics || []).slice(0,4).join(", ");
      return `<article class="conversation-card${active}" data-id="${escapeHtml(row.ConversationId)}"><h3>${escapeHtml(row.Title)}</h3><div class="tag-row">${tags}</div><p>${escapeHtml(summary)}</p><div class="metric-row"><span>Perspective ${escapeHtml(String(row.PerspectiveScore))}</span><span>Neighbors ${escapeHtml(String((row.ActiveNeighbors || []).length))}</span><span>Branches ${escapeHtml(String(row.BranchCount))}</span></div></article>`;
    }).join("");
    Array.from(document.querySelectorAll(".conversation-card")).forEach(node => node.addEventListener("click", () => { state.selectedId = node.getAttribute("data-id"); renderAll(); }));
  }

  function renderDetail(row){
    if (!row) {
      $("detail-panel").innerHTML = '<p class="detail-empty">No conversations match the current filter.</p>';
      return;
    }
    $("detail-panel").innerHTML = `
      <section class="detail-card">
        <h3>${escapeHtml(row.Title)}</h3>
        <div class="tag-row">${badge(row.PrimaryCategory,"primary")}${row.SecondaryCategory ? badge(row.SecondaryCategory,"") : ""}${row.TertiaryCategory ? badge(row.TertiaryCategory,"") : ""}${row.CategoryCommunityLabel ? badge(row.CategoryCommunityLabel,"") : ""}${row.StrongSignal ? badge("Strong signal","signal") : ""}${row.BridgeScore > 0 ? badge("Bridge " + row.BridgeScore,"bridge") : ""}</div>
        <p>${escapeHtml(row.Summary || row.TopicLabel || "")}</p>
        <div class="metric-row"><span>Perspective ${escapeHtml(String(row.PerspectiveScore))}</span><span>Embedding difference ${escapeHtml(String(row.EmbeddingDifferenceCount))}</span><span>Cross-category neighbors ${escapeHtml(String(row.CrossCategoryNeighborCount))}</span><span>Created ${escapeHtml(row.CreatedUtc || "n/a")}</span></div>
        <div class="link-list"><a href="${escapeHtml(row.TranscriptPath)}">Open transcript</a></div>
      </section>
      ${row.CategoryCommunityLabel ? `<section class="detail-card"><h3>Category Community</h3><p><strong>${escapeHtml(row.CategoryCommunityLabel)}</strong>${row.CategoryCommunityId ? ` <span class="muted">(${escapeHtml(row.CategoryCommunityId)})</span>` : ""}</p></section>` : ""}
      <section class="detail-card"><h3>Topics</h3><div class="tag-row">${(row.Topics || []).map(topic => badge(topic,"")).join("")}</div></section>
      <section class="detail-card"><h3>Keywords</h3><div class="tag-row">${(row.Keywords || []).map(keyword => badge(keyword.Keyword + " " + keyword.Score,"")).join("")}</div></section>
      <section class="detail-card"><h3>Category Predictions</h3><div class="tag-row">${(row.CategoryPredictions || []).map(prediction => badge(prediction.Category + " " + prediction.Score + (prediction.IsSelected ? " selected" : ""), prediction.IsSelected ? "primary" : "")).join("")}</div>${row.CategoryDisagrees ? `<p>Baseline <strong>${escapeHtml(row.BaselineCategory || "n/a")}</strong> vs ML.NET <strong>${escapeHtml(row.MlNetCategory || "n/a")}</strong>.</p>` : ""}</section>
      <section class="detail-card"><h3>Active Similar Neighbors</h3><div class="neighbor-list">${renderNeighbors(row.ActiveNeighbors || [])}</div></section>
      ${(row.HashOnlyExamples || []).length || (row.OnnxOnlyExamples || []).length ? `<section class="detail-card"><h3>Perspective Deltas</h3>${(row.HashOnlyExamples || []).length ? `<p><strong>Hash-only</strong></p><div class="neighbor-list">${renderNeighbors(row.HashOnlyExamples)}</div>` : ""}${(row.OnnxOnlyExamples || []).length ? `<p><strong>ONNX-only</strong></p><div class="neighbor-list">${renderNeighbors(row.OnnxOnlyExamples)}</div>` : ""}</section>` : ""}
    `;
  }

  function renderNeighbors(neighbors){
    if (!neighbors.length) return '<div class="neighbor"><div>No neighbors in this slice.</div></div>';
    return neighbors.map(neighbor => `<div class="neighbor"><div><strong>${escapeHtml(neighbor.Title)}</strong><div class="meta">${escapeHtml([neighbor.PrimaryCategory, neighbor.TopicClusterId].filter(Boolean).join(" | "))}</div></div><div>${escapeHtml(String(neighbor.Score))}</div></div>`).join("");
  }

  function getFilteredConversations(){
    const rows = conversations.filter(row => {
      if (state.search && !(row.SearchText || "").includes(state.search)) return false;
      if (state.primaryCategory && row.PrimaryCategory !== state.primaryCategory) return false;
      if (state.facetCategory && ![row.PrimaryCategory, row.SecondaryCategory, row.TertiaryCategory].filter(Boolean).includes(state.facetCategory)) return false;
      if (state.categoryCommunityId && row.CategoryCommunityId !== state.categoryCommunityId) return false;
      if (state.clusterId && row.TopicClusterId !== state.clusterId) return false;
      if (state.onlyDisagreements && !row.CategoryDisagrees) return false;
      if (state.onlyStrongSignals && !row.StrongSignal) return false;
      if (state.onlyBridges && !(row.BridgeScore > 0)) return false;
      return true;
    });
    rows.sort((left, right) => {
      switch (state.sortMode) {
        case "bridge": return compareNumbers(right.BridgeScore, left.BridgeScore) || compareNumbers(right.PerspectiveScore, left.PerspectiveScore) || compareText(left.Title, right.Title);
        case "title": return compareText(left.Title, right.Title);
        case "created": return compareText(right.CreatedUtc || "", left.CreatedUtc || "") || compareText(left.Title, right.Title);
        default: return compareNumbers(right.PerspectiveScore, left.PerspectiveScore) || compareNumbers(right.BridgeScore, left.BridgeScore) || compareText(left.Title, right.Title);
      }
    });
    return rows;
  }

  function exportFilteredJson(){ download("navigator-selection.json", JSON.stringify(getFilteredConversations(), null, 2), "application/json"); }
  function exportFilteredCsv(){
    const rows = getFilteredConversations();
    const lines = [["ConversationId","Title","PrimaryCategory","SecondaryCategory","TertiaryCategory","CategoryCommunityLabel","ClusterLabel","TopicLabel","PerspectiveScore","BridgeScore","CategoryDisagrees","TranscriptPath"].join(",")].concat(rows.map(row => [csv(row.ConversationId),csv(row.Title),csv(row.PrimaryCategory),csv(row.SecondaryCategory || ""),csv(row.TertiaryCategory || ""),csv(row.CategoryCommunityLabel || ""),csv(row.TopicClusterLabel || ""),csv(row.TopicLabel || ""),csv(String(row.PerspectiveScore)),csv(String(row.BridgeScore)),csv(row.CategoryDisagrees ? "true" : "false"),csv(row.TranscriptPath)].join(",")));
    download("navigator-selection.csv", lines.join("\\n"), "text/csv");
  }
  function download(name, content, mime){ const blob = new Blob([content], { type: mime }); const url = URL.createObjectURL(blob); const link = document.createElement("a"); link.href = url; link.download = name; document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url); }
  function unique(values){ return Array.from(new Set(values.filter(Boolean))).sort(compareText); }
  function uniqueObjects(values){ return Array.from(new Map(values.map(value => [value.value, value])).values()).sort((left, right) => compareText(left.label, right.label)); }
  function populateSelect(select, values){ select.innerHTML = ['<option value="">All</option>'].concat(values.map(value => typeof value === "string" ? `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>` : `<option value="${escapeHtml(value.value)}">${escapeHtml(value.label)}</option>`)).join(""); }
  function compact(title, meta){ return `<div class="compact-item"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(meta)}</span></div>`; }
  function badge(text, kind){ return `<span class="tag ${kind || ""}">${escapeHtml(text)}</span>`; }
  function escapeHtml(value){ return String(value ?? "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }
  function csv(value){ return '"' + String(value ?? "").replace(/"/g, '""') + '"'; }
  function compareText(left, right){ return String(left || "").localeCompare(String(right || ""), undefined, { sensitivity:"base" }); }
  function compareNumbers(left, right){ return (left || 0) - (right || 0); }
}());
""";

    private sealed record NavigatorBundle(
        NavigatorSummary Summary,
        NavigatorInsights Insights,
        IReadOnlyList<NavigatorConversationRow> Conversations);

    private sealed record NavigatorInsightReport(
        NavigatorSummary Summary,
        NavigatorInsights Insights);

    private sealed record NavigatorSummary(
        string GeneratedUtc,
        int ConversationCount,
        int ClusterCount,
        int CategoryCommunityCount,
        int CategoryCount,
        int SimilarityEdgeCount,
        int KeywordHotspotCount,
        int WithSecondaryCategoryCount,
        int WithTertiaryCategoryCount,
        int CategoryDisagreementCount,
        int EmbeddingDivergenceCount,
        int StrongSignalCount,
        string? ActiveEmbeddingProvider,
        string? ActiveEmbeddingModel);

    private sealed record NavigatorInsights(
        IReadOnlyList<NavigatorCategoryInsight> TopCategories,
        IReadOnlyList<NavigatorCategoryBlend> TopCategoryBlends,
        IReadOnlyList<NavigatorClusterInsight> TopClusters,
        IReadOnlyList<NavigatorCategoryCommunityInsight> TopCategoryCommunities,
        IReadOnlyList<NavigatorBridgeConversation> BridgeConversations,
        IReadOnlyList<NavigatorCrossCategorySimilarity> CrossCategorySimilarities,
        IReadOnlyList<NavigatorKeywordHotspot> KeywordHotspots,
        IReadOnlyList<NavigatorDisagreement> Disagreements,
        IReadOnlyList<NavigatorStrongSignal> StrongSignals);

    private sealed record NavigatorConversationRow(
        string ConversationId,
        string Title,
        string? CreatedUtc,
        string? SourceFile,
        int BranchCount,
        string PrimaryCategory,
        string? SecondaryCategory,
        string? TertiaryCategory,
        string? CategorySource,
        string? CategoryCommunityId,
        string? CategoryCommunityLabel,
        string? TopicClusterId,
        string? TopicClusterLabel,
        string? TopicClusterSummary,
        string? TopicLabel,
        string? Summary,
        IReadOnlyList<string> Topics,
        IReadOnlyList<NavigatorKeyword> Keywords,
        IReadOnlyList<CategoryPrediction> CategoryPredictions,
        string TranscriptPath,
        string SearchText,
        bool CategoryDisagrees,
        string? BaselineCategory,
        string? MlNetCategory,
        string? SelectedSource,
        string? SelectedCategory,
        int PerspectiveScore,
        int EmbeddingDifferenceCount,
        int BridgeScore,
        int CrossCategoryNeighborCount,
        IReadOnlyList<string> NeighborCategories,
        IReadOnlyList<NavigatorNeighbor> ActiveNeighbors,
        IReadOnlyList<NavigatorNeighbor> HashOnlyExamples,
        IReadOnlyList<NavigatorNeighbor> OnnxOnlyExamples,
        bool StrongSignal);

    private sealed record NavigatorKeyword(string Keyword, double Score);

    private sealed record NavigatorNeighbor(
        string ConversationId,
        string Title,
        double Score,
        string? PrimaryCategory,
        string? TopicClusterId);

    private sealed record NavigatorCategoryInsight(
        string Category,
        int PrimaryCount,
        int AnyFacetCount,
        int SecondaryCount,
        int TertiaryCount,
        int ClusterCount,
        IReadOnlyList<string> TopTopics,
        IReadOnlyList<string> ExampleTitles);

    private sealed record NavigatorCategoryBlend(
        string LeftCategory,
        string RightCategory,
        int Count,
        IReadOnlyList<string> ExampleTitles);

    private sealed record NavigatorClusterInsight(
        string ClusterId,
        string Label,
        string PrimaryCategory,
        int ConversationCount,
        string? Summary,
        IReadOnlyList<string> Keywords,
        IReadOnlyList<string> ExampleTitles);

    private sealed record NavigatorCategoryCommunityInsight(
        string CommunityId,
        string Label,
        string PrimaryCategory,
        int ConversationCount,
        string? Summary,
        IReadOnlyList<string> Categories,
        IReadOnlyList<string> ExampleTitles);

    private sealed record NavigatorBridgeConversation(
        string ConversationId,
        string Title,
        string PrimaryCategory,
        string? SecondaryCategory,
        string? TertiaryCategory,
        string? TopicLabel,
        string? TopicClusterLabel,
        int BridgeScore,
        int CrossCategoryNeighborCount,
        IReadOnlyList<string> NeighborCategories,
        string TranscriptPath);

    private sealed record NavigatorCrossCategorySimilarity(
        string LeftConversationId,
        string LeftTitle,
        string LeftCategory,
        string RightConversationId,
        string RightTitle,
        string RightCategory,
        double Score);

    private sealed record NavigatorKeywordHotspot(
        string LeftKeyword,
        string RightKeyword,
        int Count,
        double Weight);

    private sealed record NavigatorDisagreement(
        string ConversationId,
        string Title,
        string? BaselineCategory,
        string? MlNetCategory,
        string? SelectedCategory,
        string TranscriptPath);

    private sealed record NavigatorStrongSignal(
        string ConversationId,
        string Title,
        string SelectedCategory,
        int PerspectiveScore,
        int EmbeddingDifferenceCount,
        bool CategoryDisagrees,
        string TranscriptPath);

    private sealed record NavigatorCategoryComparisonFile(
        string CategoryProviderMode,
        string BaselineSource,
        IReadOnlyList<NavigatorCategoryComparisonConversation> Conversations);

    private sealed record NavigatorCategoryComparisonConversation(
        string ConversationId,
        string Title,
        string BaselineSource,
        string BaselineCategory,
        string? MlNetCategory,
        double? MlNetScore,
        bool Agrees,
        string SelectedSource,
        string SelectedCategory);

    private sealed record NavigatorPerspectiveSummaryFile(
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
        IReadOnlyList<NavigatorPerspectiveConversation> Conversations);

    private sealed record NavigatorPerspectiveConversation(
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
        IReadOnlyList<NavigatorPerspectiveNeighbor> HashOnlyExamples,
        IReadOnlyList<NavigatorPerspectiveNeighbor> OnnxOnlyExamples);

    private sealed record NavigatorPerspectiveNeighbor(
        string ConversationId,
        string Title,
        double Score);

    private sealed class NavigatorCategoryBlendComparer : IEqualityComparer<(string LeftCategory, string RightCategory)>
    {
        public static readonly NavigatorCategoryBlendComparer OrdinalIgnoreCase = new(StringComparer.OrdinalIgnoreCase);

        private readonly StringComparer _comparer;

        private NavigatorCategoryBlendComparer(StringComparer comparer)
        {
            _comparer = comparer;
        }

        public bool Equals((string LeftCategory, string RightCategory) x, (string LeftCategory, string RightCategory) y)
            => _comparer.Equals(x.LeftCategory, y.LeftCategory) && _comparer.Equals(x.RightCategory, y.RightCategory);

        public int GetHashCode((string LeftCategory, string RightCategory) obj)
            => HashCode.Combine(_comparer.GetHashCode(obj.LeftCategory), _comparer.GetHashCode(obj.RightCategory));
    }
}
