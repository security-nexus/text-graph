using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Tokenizers;

interface IConversationChunker
{
    string Description { get; }
    IReadOnlyList<ConversationTextChunk> ChunkMessage(MessageNode node);
}

static class ConversationChunkerFactory
{
    public static IConversationChunker Create(AppArgs arguments)
    {
        if (string.Equals(arguments.ChunkingProviderMode, "char", StringComparison.OrdinalIgnoreCase))
        {
            return new CharacterConversationChunker();
        }

        bool wantsMlTokenizer =
            string.Equals(arguments.ChunkingProviderMode, "tokenizer", StringComparison.OrdinalIgnoreCase)
            || string.Equals(arguments.ChunkingProviderMode, "mltokenizer", StringComparison.OrdinalIgnoreCase)
            || string.Equals(arguments.ChunkingProviderMode, "auto", StringComparison.OrdinalIgnoreCase);

        if (wantsMlTokenizer
            && !string.IsNullOrWhiteSpace(arguments.EmbeddingVocabularyPath)
            && File.Exists(arguments.EmbeddingVocabularyPath))
        {
            return new BertConversationChunker(
                arguments.EmbeddingVocabularyPath,
                arguments.ChunkMaxTokens,
                arguments.ChunkOverlapTokens);
        }

        if (string.Equals(arguments.ChunkingProviderMode, "tokenizer", StringComparison.OrdinalIgnoreCase)
            || string.Equals(arguments.ChunkingProviderMode, "mltokenizer", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                "--chunking-provider mltokenizer requires --embedding-vocab or models/embeddings/vocab.txt.");
        }

        return new CharacterConversationChunker();
    }
}

sealed class CharacterConversationChunker : IConversationChunker
{
    private readonly int _maxChars;

    public CharacterConversationChunker(int maxChars = 1200)
    {
        _maxChars = Math.Max(200, maxChars);
    }

    public string Description => $"char:{_maxChars}";

    public IReadOnlyList<ConversationTextChunk> ChunkMessage(MessageNode node)
    {
        var chunks = new List<ConversationTextChunk>();
        AddChunks(chunks, "text", node.Flat.Text);
        AddChunks(chunks, "code", node.Flat.AggregateCode);
        AddChunks(chunks, "execution_output", node.Flat.ExecutionOutput);
        return chunks;
    }

    private void AddChunks(List<ConversationTextChunk> chunks, string kind, string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return;
        }

        var normalized = NormalizeText(text);
        for (int i = 0; i < normalized.Length; i += _maxChars)
        {
            int take = Math.Min(_maxChars, normalized.Length - i);
            chunks.Add(new ConversationTextChunk(kind, normalized.Substring(i, take)));
        }
    }

    private static string NormalizeText(string text) => text.Replace("\r\n", "\n");
}

sealed class BertConversationChunker : IConversationChunker
{
    private readonly string _vocabularyPath;
    private readonly BertTokenizer _tokenizer;
    private readonly int _maxTokens;
    private readonly int _overlapTokens;

    public BertConversationChunker(string vocabularyPath, int maxTokens, int overlapTokens)
    {
        _vocabularyPath = vocabularyPath;
        _tokenizer = BertTokenizer.Create(vocabularyPath);
        _maxTokens = Math.Max(16, maxTokens);
        _overlapTokens = Math.Clamp(overlapTokens, 0, _maxTokens - 1);
    }

    public string Description => $"tokenizer:{Path.GetFileName(_vocabularyPath)}:{_maxTokens}/{_overlapTokens}";

    public IReadOnlyList<ConversationTextChunk> ChunkMessage(MessageNode node)
    {
        var chunks = new List<ConversationTextChunk>();
        AddChunks(chunks, "text", node.Flat.Text);
        AddChunks(chunks, "code", node.Flat.AggregateCode);
        AddChunks(chunks, "execution_output", node.Flat.ExecutionOutput);
        return chunks;
    }

    private void AddChunks(List<ConversationTextChunk> chunks, string kind, string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return;
        }

        foreach (var chunk in ChunkText(kind, NormalizeText(text)))
        {
            chunks.Add(chunk);
        }
    }

    private IEnumerable<ConversationTextChunk> ChunkText(string kind, string text)
    {
        int cursor = 0;
        while (cursor < text.Length)
        {
            var remaining = text[cursor..];
            var tokenIds = _tokenizer.EncodeToIds(remaining, _maxTokens, out _, out int charsConsumed, true, false);
            int take = Math.Min(Math.Max(charsConsumed, 0), remaining.Length);

            if (take == 0)
            {
                yield return new ConversationTextChunk(kind, remaining, tokenIds.Count);
                yield break;
            }

            var slice = remaining[..take].Trim();
            if (slice.Length > 0)
            {
                yield return new ConversationTextChunk(kind, slice, tokenIds.Count);
            }

            if (cursor + take >= text.Length)
            {
                yield break;
            }

            int nextCursor = cursor + take;
            if (_overlapTokens > 0)
            {
                int overlapStart = _tokenizer.GetIndexByTokenCountFromEnd(slice, _overlapTokens, out _, out _, true, false);
                if (overlapStart > 0 && overlapStart < slice.Length)
                {
                    nextCursor = cursor + overlapStart;
                }
            }

            if (nextCursor <= cursor)
            {
                nextCursor = cursor + take;
            }

            cursor = nextCursor;
        }
    }

    private static string NormalizeText(string text) => text.Replace("\r\n", "\n");
}
