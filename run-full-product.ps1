param(
    [string]$Project = ".\ChatGptExportGraphBuilder.csproj",
    [string]$InputFile = ".\normalized-messages.json",
    [string]$OutputRoot = ".\out-full-product",
    [string]$CudaCudnnBin = "F:\Program Files\NVIDIA\CUDNN\v9.2\bin\12.9\x64",
    [switch]$SkipBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-PathSegment {
    param([string]$Segment)

    if ([string]::IsNullOrWhiteSpace($Segment)) {
        return
    }

    if (-not (Test-Path $Segment)) {
        throw "Required path does not exist: $Segment"
    }

    $existing = $env:PATH -split ';' | ForEach-Object { $_.Trim() }
    if (-not ($existing | Where-Object { $_ -eq $Segment })) {
        $env:PATH = "$Segment;$env:PATH"
    }
}

function Invoke-Dotnet {
    param(
        [string]$WorkingDirectory,
        [string]$LogPath,
        [string[]]$Arguments
    )

    Write-Host ("dotnet " + ($Arguments -join " "))
    $stdoutPath = [System.IO.Path]::GetTempFileName()
    $stderrPath = [System.IO.Path]::GetTempFileName()

    try {
        $process = Start-Process `
            -FilePath "dotnet" `
            -ArgumentList $Arguments `
            -WorkingDirectory $WorkingDirectory `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath

        $stdoutLines = if (Test-Path $stdoutPath) { Get-Content -Path $stdoutPath } else { @() }
        $stderrLines = if (Test-Path $stderrPath) { Get-Content -Path $stderrPath } else { @() }
        $allLines = @($stdoutLines) + @($stderrLines)

        $allLines | Set-Content -Path $LogPath -Encoding UTF8
        foreach ($line in $allLines) {
            Write-Host $line
        }

        if ($process.ExitCode -ne 0) {
            throw "dotnet command failed. See $LogPath"
        }
    }
    finally {
        Remove-Item -Path $stdoutPath -ErrorAction SilentlyContinue
        Remove-Item -Path $stderrPath -ErrorAction SilentlyContinue
    }

    if ($process.ExitCode -ne 0) {
        throw "dotnet command failed. See $LogPath"
    }
}

function Get-DoubleValue {
    param($Value)

    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        return 0.0
    }

    return [double]::Parse([string]$Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Get-FirstValue {
    param([System.Collections.IEnumerable]$Values)

    foreach ($value in $Values) {
        if ($null -ne $value -and -not [string]::IsNullOrWhiteSpace([string]$value)) {
            return [string]$value
        }
    }

    return $null
}

function Get-MapString {
    param([System.Collections.IDictionary]$Map)

    return (($Map.GetEnumerator() | Sort-Object Name | ForEach-Object { "{0}={1}" -f $_.Name, $_.Value }) -join "; ")
}

function Import-RunData {
    param(
        [string]$ProfileName,
        [string]$RootPath
    )

    $graphDir = Join-Path $RootPath "graph"
    $catalogPath = Join-Path $graphDir "conversation-catalog.csv"
    $insightsPath = Join-Path $graphDir "insights.json"

    if (-not (Test-Path $catalogPath)) {
        throw "Missing catalog for profile '$ProfileName': $catalogPath"
    }

    if (-not (Test-Path $insightsPath)) {
        throw "Missing insights for profile '$ProfileName': $insightsPath"
    }

    $catalog = Import-Csv $catalogPath
    $insights = Get-Content $insightsPath -Raw | ConvertFrom-Json
    $byId = @{}
    foreach ($row in $catalog) {
        $byId[$row.ConversationId] = $row
    }

    return [pscustomobject]@{
        ProfileName = $ProfileName
        RootPath = $RootPath
        GraphPath = $graphDir
        Catalog = $catalog
        ById = $byId
        Insights = $insights
    }
}

$projectPath = (Resolve-Path $Project).Path
$inputPath = (Resolve-Path $InputFile).Path
$outputRootPath = [System.IO.Path]::GetFullPath($OutputRoot)
$summaryPath = Join-Path $outputRootPath "summary"

New-Item -ItemType Directory -Path $outputRootPath -Force | Out-Null
New-Item -ItemType Directory -Path $summaryPath -Force | Out-Null

$env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = "1"
$env:DOTNET_CLI_HOME = (Resolve-Path ".").Path
Ensure-PathSegment $CudaCudnnBin

$buildLog = Join-Path $summaryPath "build.log"
if (-not $SkipBuild) {
    Invoke-Dotnet -WorkingDirectory (Get-Location).Path -LogPath $buildLog -Arguments @("clean", $projectPath)
    Invoke-Dotnet -WorkingDirectory (Get-Location).Path -LogPath $buildLog -Arguments @("build", $projectPath)
}

$runs = @(
    @{
        Name = "hash"
        Arguments = @("--execution-profile", "heuristic", "--cluster-labeler", "heuristic", "--chunking-provider", "tokenizer")
    },
    @{
        Name = "mlnet"
        Arguments = @("--execution-profile", "mlnet", "--cluster-labeler", "heuristic", "--chunking-provider", "tokenizer")
    },
    @{
        Name = "onnx"
        Arguments = @("--execution-profile", "onnx", "--cluster-labeler", "heuristic", "--chunking-provider", "tokenizer", "--onnx-execution-provider", "cuda", "--compare-models", "--compare-onnx-tokenizers")
    },
    @{
        Name = "hybrid"
        Arguments = @("--execution-profile", "hybrid", "--cluster-labeler", "heuristic", "--chunking-provider", "tokenizer", "--onnx-execution-provider", "cuda", "--compare-models")
    }
)

$runOutputs = @{}
foreach ($run in $runs) {
    $profileName = $run.Name
    $profileOutput = Join-Path $outputRootPath $profileName
    $profileDb = Join-Path $outputRootPath ("{0}.db" -f $profileName)
    $profileLog = Join-Path $summaryPath ("{0}.log" -f $profileName)

    New-Item -ItemType Directory -Path $profileOutput -Force | Out-Null

    $arguments = @(
        "run",
        "--no-build",
        "--project", $projectPath,
        "--",
        "--input", $inputPath,
        "--output", $profileOutput,
        "--db", $profileDb
    ) + $run.Arguments

    Invoke-Dotnet -WorkingDirectory (Get-Location).Path -LogPath $profileLog -Arguments $arguments
    $runOutputs[$profileName] = Import-RunData -ProfileName $profileName -RootPath $profileOutput
}

$allConversationIds = @(
    $runOutputs.Values |
        ForEach-Object { $_.ById.Keys } |
        Sort-Object -Unique
)

$conversationRows = @(
foreach ($conversationId in $allConversationIds) {
    $rowsByProfile = [ordered]@{}
    foreach ($profileName in ($runOutputs.Keys | Sort-Object)) {
        if ($runOutputs[$profileName].ById.ContainsKey($conversationId)) {
            $rowsByProfile[$profileName] = $runOutputs[$profileName].ById[$conversationId]
        }
    }

    $title = Get-FirstValue ($rowsByProfile.Values | ForEach-Object { $_.Title })
    $transcriptPath = Get-FirstValue ($rowsByProfile.Values | ForEach-Object { $_.TranscriptPath })
    $distinctCategories = @(
        $rowsByProfile.Values |
            ForEach-Object { $_.PrimaryCategory } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            Sort-Object -Unique
    )
    $distinctTopicLabels = @(
        $rowsByProfile.Values |
            ForEach-Object { $_.TopicLabel } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            Sort-Object -Unique
    )
    $distinctClusterLabels = @(
        $rowsByProfile.Values |
            ForEach-Object { $_.ClusterLabel } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            Sort-Object -Unique
    )

    $bridgeScores = [ordered]@{}
    $perspectiveScores = [ordered]@{}
    $categoriesByProfile = [ordered]@{}
    $topicLabelsByProfile = [ordered]@{}

    foreach ($profileName in $rowsByProfile.Keys) {
        $row = $rowsByProfile[$profileName]
        $bridgeScores[$profileName] = Get-DoubleValue $row.BridgeScore
        $perspectiveScores[$profileName] = Get-DoubleValue $row.PerspectiveScore
        $categoriesByProfile[$profileName] = $row.PrimaryCategory
        $topicLabelsByProfile[$profileName] = $row.TopicLabel
    }

    $maxBridgeScore = ($bridgeScores.Values | Measure-Object -Maximum).Maximum
    $avgBridgeScore = ($bridgeScores.Values | Measure-Object -Average).Average
    $maxPerspectiveScore = ($perspectiveScores.Values | Measure-Object -Maximum).Maximum
    if ($null -eq $avgBridgeScore) {
        $avgBridgeScore = 0
    }

    [pscustomobject]@{
        ConversationId = $conversationId
        Title = $title
        TranscriptPath = $transcriptPath
        ProfilesPresent = $rowsByProfile.Count
        DistinctCategoryCount = $distinctCategories.Count
        DistinctTopicLabelCount = $distinctTopicLabels.Count
        DistinctClusterLabelCount = $distinctClusterLabels.Count
        CategoryDisagrees = ($distinctCategories.Count -gt 1)
        MaxBridgeScore = [math]::Round([double]$maxBridgeScore, 3)
        AvgBridgeScore = [math]::Round([double]$avgBridgeScore, 3)
        MaxPerspectiveScore = [math]::Round([double]$maxPerspectiveScore, 3)
        CategoriesByProfile = $categoriesByProfile
        TopicLabelsByProfile = $topicLabelsByProfile
        BridgeScoresByProfile = $bridgeScores
        PerspectiveScoresByProfile = $perspectiveScores
        Categories = ($distinctCategories -join " | ")
        TopicLabels = ($distinctTopicLabels -join " | ")
        ClusterLabels = ($distinctClusterLabels -join " | ")
        CategoryMap = (Get-MapString $categoriesByProfile)
        TopicLabelMap = (Get-MapString $topicLabelsByProfile)
        BridgeScoreMap = (Get-MapString $bridgeScores)
        PerspectiveScoreMap = (Get-MapString $perspectiveScores)
    }
}
)

$categoryDisagreements = @(
    $conversationRows |
        Where-Object { $_.CategoryDisagrees } |
        Sort-Object `
            @{ Expression = "MaxBridgeScore"; Descending = $true }, `
            @{ Expression = "MaxPerspectiveScore"; Descending = $true }, `
            @{ Expression = "DistinctCategoryCount"; Descending = $true }, `
            @{ Expression = "Title"; Descending = $false }
)

$hotpaths = @(
    $conversationRows |
        Where-Object {
            $_.MaxBridgeScore -gt 0 -or
            $_.MaxPerspectiveScore -gt 0 -or
            $_.CategoryDisagrees
        } |
        Sort-Object `
            @{ Expression = "MaxBridgeScore"; Descending = $true }, `
            @{ Expression = "DistinctCategoryCount"; Descending = $true }, `
            @{ Expression = "MaxPerspectiveScore"; Descending = $true }, `
            @{ Expression = "Title"; Descending = $false } |
        Select-Object -First 75
)

$profiles = @($runOutputs.Keys | Sort-Object)
$pairwiseCategoryAgreement = @(
foreach ($left in $profiles) {
    foreach ($right in $profiles) {
        if ([string]::CompareOrdinal($left, $right) -ge 0) {
            continue
        }

        $leftRows = $runOutputs[$left].ById
        $rightRows = $runOutputs[$right].ById
        $shared = $allConversationIds | Where-Object { $leftRows.ContainsKey($_) -and $rightRows.ContainsKey($_) }
        $matches = 0
        $differences = 0

        foreach ($conversationId in $shared) {
            if ($leftRows[$conversationId].PrimaryCategory -eq $rightRows[$conversationId].PrimaryCategory) {
                $matches++
            }
            else {
                $differences++
            }
        }

        $total = $matches + $differences
        [pscustomobject]@{
            LeftProfile = $left
            RightProfile = $right
            SharedConversationCount = $total
            MatchingCategoryCount = $matches
            DifferentCategoryCount = $differences
            MatchRate = if ($total -eq 0) { 1.0 } else { [math]::Round(($matches / $total), 4) }
        }
    }
}
)

$profileSummaries = @(
foreach ($profileName in $profiles) {
    $summary = $runOutputs[$profileName].Insights.Summary
    [pscustomobject]@{
        Profile = $profileName
        ConversationCount = $summary.ConversationCount
        ClusterCount = $summary.ClusterCount
        CategoryCommunityCount = $summary.CategoryCommunityCount
        SimilarityEdgeCount = $summary.SimilarityEdgeCount
        CategoryDisagreementCount = $summary.CategoryDisagreementCount
        EmbeddingDivergenceCount = $summary.EmbeddingDivergenceCount
        StrongSignalCount = $summary.StrongSignalCount
        ActiveEmbeddingProvider = $summary.ActiveEmbeddingProvider
        ActiveEmbeddingModel = $summary.ActiveEmbeddingModel
        OutputDirectory = $runOutputs[$profileName].RootPath
    }
}
)

$topBridgesByProfile = @(
foreach ($profileName in $profiles) {
    [pscustomobject]@{
        Profile = $profileName
        Conversations = @(
            $runOutputs[$profileName].Catalog |
                Sort-Object `
                    @{ Expression = { Get-DoubleValue $_.BridgeScore }; Descending = $true }, `
                    @{ Expression = "Title"; Descending = $false } |
                Select-Object -First 15 `
                    ConversationId,
                    Title,
                    PrimaryCategory,
                    SecondaryCategory,
                    TertiaryCategory,
                    BridgeScore,
                    PerspectiveScore,
                    ClusterLabel,
                    TopicLabel
        )
    }
}
)

$report = [ordered]@{
    GeneratedUtc = [DateTime]::UtcNow.ToString("u")
    InputPath = $inputPath
    OutputRoot = $outputRootPath
    CudaCudnnBin = $CudaCudnnBin
    Profiles = $profileSummaries
    PairwiseCategoryAgreement = $pairwiseCategoryAgreement
    CategoryDisagreementCount = $categoryDisagreements.Count
    HotpathCount = $hotpaths.Count
    CategoryDisagreements = @($categoryDisagreements | Select-Object -First 75)
    Hotpaths = @($hotpaths)
    TopBridgeConversationsByProfile = $topBridgesByProfile
}

$jsonPath = Join-Path $summaryPath "full-product-report.json"
$markdownPath = Join-Path $summaryPath "full-product-report.md"
$disagreementCsvPath = Join-Path $summaryPath "full-product-disagreements.csv"
$hotpathCsvPath = Join-Path $summaryPath "full-product-hotpaths.csv"

$report | ConvertTo-Json -Depth 10 | Set-Content -Path $jsonPath -Encoding UTF8

$categoryDisagreements |
    Select-Object `
        ConversationId,
        Title,
        DistinctCategoryCount,
        Categories,
        MaxBridgeScore,
        MaxPerspectiveScore,
        CategoryMap,
        TopicLabelMap,
        TranscriptPath |
    Export-Csv -Path $disagreementCsvPath -NoTypeInformation -Encoding UTF8

$hotpaths |
    Select-Object `
        ConversationId,
        Title,
        MaxBridgeScore,
        AvgBridgeScore,
        MaxPerspectiveScore,
        DistinctCategoryCount,
        Categories,
        CategoryMap,
        TopicLabelMap,
        BridgeScoreMap,
        TranscriptPath |
    Export-Csv -Path $hotpathCsvPath -NoTypeInformation -Encoding UTF8

$sb = [System.Text.StringBuilder]::new()
[void]$sb.AppendLine("# Full Product Report")
[void]$sb.AppendLine()
[void]$sb.AppendLine(('Generated: {0}' -f [DateTime]::UtcNow.ToString("u")))
[void]$sb.AppendLine(('Input: `{0}`' -f $inputPath))
[void]$sb.AppendLine(('Output root: `{0}`' -f $outputRootPath))
[void]$sb.AppendLine()
[void]$sb.AppendLine("## Profiles")
[void]$sb.AppendLine()
foreach ($profile in $profileSummaries) {
    [void]$sb.AppendLine(('* `{0}`: embedding `{1}/{2}`, clusters `{3}`, communities `{4}`, similarity edges `{5}`, strong signals `{6}`' -f `
        $profile.Profile,
        $profile.ActiveEmbeddingProvider,
        $profile.ActiveEmbeddingModel,
        $profile.ClusterCount,
        $profile.CategoryCommunityCount,
        $profile.SimilarityEdgeCount,
        $profile.StrongSignalCount))
}
[void]$sb.AppendLine()
[void]$sb.AppendLine("## Pairwise Category Agreement")
[void]$sb.AppendLine()
foreach ($row in $pairwiseCategoryAgreement | Sort-Object MatchRate, LeftProfile, RightProfile) {
    [void]$sb.AppendLine(('* `{0}` vs `{1}`: {2:P1} match ({3}/{4})' -f `
        $row.LeftProfile,
        $row.RightProfile,
        $row.MatchRate,
        $row.MatchingCategoryCount,
        $row.SharedConversationCount))
}
[void]$sb.AppendLine()
[void]$sb.AppendLine("## Category Disagreements")
[void]$sb.AppendLine()
if ($categoryDisagreements.Count -eq 0) {
    [void]$sb.AppendLine("* No category disagreements were found across the selected profiles.")
}
else {
    foreach ($row in $categoryDisagreements | Select-Object -First 20) {
        [void]$sb.AppendLine(('* `{0}`: `{1}` | bridge `{2}` | categories `{3}`' -f `
            $row.Title,
            $row.ConversationId,
            $row.MaxBridgeScore,
            $row.CategoryMap))
    }
}
[void]$sb.AppendLine()
[void]$sb.AppendLine("## Hotpaths")
[void]$sb.AppendLine()
foreach ($row in $hotpaths | Select-Object -First 20) {
    [void]$sb.AppendLine(('* `{0}`: bridge `{1}`, perspective `{2}`, categories `{3}`' -f `
        $row.Title,
        $row.MaxBridgeScore,
        $row.MaxPerspectiveScore,
        $row.CategoryMap))
}
[void]$sb.AppendLine()
[void]$sb.AppendLine("## Top Bridge Conversations By Profile")
[void]$sb.AppendLine()
foreach ($profile in $topBridgesByProfile) {
    [void]$sb.AppendLine(("### {0}" -f $profile.Profile))
    [void]$sb.AppendLine()
    foreach ($conversation in $profile.Conversations | Select-Object -First 10) {
        [void]$sb.AppendLine(('* `{0}`: bridge `{1}` | `{2}` | `{3}`' -f `
            $conversation.Title,
            $conversation.BridgeScore,
            $conversation.PrimaryCategory,
            $conversation.TopicLabel))
    }
    [void]$sb.AppendLine()
}
[void]$sb.AppendLine("## Files")
[void]$sb.AppendLine()
[void]$sb.AppendLine(('* JSON report: `{0}`' -f $jsonPath))
[void]$sb.AppendLine(('* Markdown report: `{0}`' -f $markdownPath))
[void]$sb.AppendLine(('* Disagreement CSV: `{0}`' -f $disagreementCsvPath))
[void]$sb.AppendLine(('* Hotpath CSV: `{0}`' -f $hotpathCsvPath))

$sb.ToString() | Set-Content -Path $markdownPath -Encoding UTF8

Write-Host ""
Write-Host "Full product run complete."
Write-Host ("JSON report:        {0}" -f $jsonPath)
Write-Host ("Markdown report:    {0}" -f $markdownPath)
Write-Host ("Disagreement CSV:   {0}" -f $disagreementCsvPath)
Write-Host ("Hotpath CSV:        {0}" -f $hotpathCsvPath)
