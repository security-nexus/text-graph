using System.Text;
using System.Text.Json;
using Microsoft.ML;

sealed record ClassifierTrainingResult(
    string ModelPath,
    string MetricsPath,
    int ExampleCount,
    int LabelCount,
    double MacroAccuracy,
    double MicroAccuracy);

static class LocalModelTrainer
{
    public static ClassifierTrainingResult TrainClassifier(
        IReadOnlyList<ConversationAnalysis> analyses,
        AppArgs arguments)
    {
        if (string.IsNullOrWhiteSpace(arguments.MlNetModelPath))
        {
            throw new InvalidOperationException("No ML.NET model path is configured. Set --mlnet-model or --models-dir.");
        }

        var examples = BuildExamples(analyses);
        if (examples.Count < 20)
        {
            throw new InvalidOperationException($"Need at least 20 labeled conversations to train a useful classifier. Found {examples.Count}.");
        }

        var labelDistribution = examples
            .GroupBy(example => example.Label, StringComparer.OrdinalIgnoreCase)
            .OrderByDescending(group => group.Count())
            .ThenBy(group => group.Key, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => group.Count(), StringComparer.OrdinalIgnoreCase);

        if (labelDistribution.Count < 2)
        {
            throw new InvalidOperationException("Need at least two distinct labels to train the classifier.");
        }

        string modelPath = arguments.MlNetModelPath;
        string metricsPath = arguments.ClassifierMetricsPath;
        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);
        Directory.CreateDirectory(Path.GetDirectoryName(metricsPath)!);

        var ml = new MLContext(seed: 17);
        var allData = ml.Data.LoadFromEnumerable(examples);
        var split = ml.Data.TrainTestSplit(allData, testFraction: 0.2, seed: 17);

        var corePipeline = ml.Transforms.Conversion.MapValueToKey("Label")
            .Append(ml.Transforms.Text.FeaturizeText("Features", nameof(ClassifierTrainingRow.Text)))
            .Append(ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                labelColumnName: "Label",
                featureColumnName: "Features"));

        var evaluationModel = corePipeline.Fit(split.TrainSet);
        var scored = evaluationModel.Transform(split.TestSet);
        var metrics = ml.MulticlassClassification.Evaluate(scored, labelColumnName: "Label");

        var finalPipeline = corePipeline
            .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        var finalModel = finalPipeline.Fit(allData);

        using (var output = File.Create(modelPath))
        {
            ml.Model.Save(finalModel, allData.Schema, output);
        }

        var report = new
        {
            generatedUtc = DateTimeOffset.UtcNow,
            inputPath = Path.GetFullPath(arguments.InputPath),
            modelPath = Path.GetFullPath(modelPath),
            exampleCount = examples.Count,
            labelCount = labelDistribution.Count,
            macroAccuracy = Math.Round(metrics.MacroAccuracy, 4),
            microAccuracy = Math.Round(metrics.MicroAccuracy, 4),
            logLoss = Math.Round(metrics.LogLoss, 4),
            topKAccuracy = Math.Round(metrics.TopKAccuracy, 4),
            labelDistribution
        };

        File.WriteAllText(metricsPath, JsonSerializer.Serialize(report, JsonOptions.WriteIndented), new UTF8Encoding(false));

        return new ClassifierTrainingResult(
            modelPath,
            metricsPath,
            examples.Count,
            labelDistribution.Count,
            metrics.MacroAccuracy,
            metrics.MicroAccuracy);
    }

    private static List<ClassifierTrainingRow> BuildExamples(IReadOnlyList<ConversationAnalysis> analyses)
    {
        var examples = analyses
            .Select(analysis => new ClassifierTrainingRow
            {
                Label = NormalizeLabel(analysis.Category),
                Text = BuildTrainingText(analysis)
            })
            .Where(example => !string.IsNullOrWhiteSpace(example.Text))
            .ToList();

        var initialCounts = examples
            .GroupBy(example => example.Label, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => group.Count(), StringComparer.OrdinalIgnoreCase);

        foreach (var example in examples)
        {
            if (initialCounts.TryGetValue(example.Label, out int count) && count < 4)
            {
                example.Label = AnalysisDefaults.DefaultCategory;
            }
        }

        return examples
            .GroupBy(example => $"{example.Label}\u001f{example.Text}", StringComparer.OrdinalIgnoreCase)
            .Select(group => group.First())
            .ToList();
    }

    private static string NormalizeLabel(string? label)
        => string.IsNullOrWhiteSpace(label) ? AnalysisDefaults.DefaultCategory : label.Trim();

    private static string BuildTrainingText(ConversationAnalysis analysis)
    {
        var sb = new StringBuilder();
        if (!string.IsNullOrWhiteSpace(analysis.Conversation.Title) && !string.Equals(analysis.Conversation.Title, "(untitled)", StringComparison.Ordinal))
        {
            sb.AppendLine(analysis.Conversation.Title);
        }

        if (analysis.Topics.Count > 0)
        {
            sb.AppendLine(string.Join(", ", analysis.Topics));
        }

        if (analysis.Keywords.Count > 0)
        {
            sb.AppendLine(string.Join(", ", analysis.Keywords.Take(12).Select(keyword => keyword.Keyword)));
        }

        sb.AppendLine(ConversationIndexText.BuildCorpus(analysis.Conversation));
        return sb.ToString().Trim();
    }

    private sealed class ClassifierTrainingRow
    {
        public string Label { get; set; } = string.Empty;
        public string Text { get; set; } = string.Empty;
    }
}
