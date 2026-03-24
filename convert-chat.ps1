Set-StrictMode -Version Latest
$ErrorActionPreference =  3

function Get-FileKind {
    param([string]$Path)

    $fs = [System.IO.File]::OpenRead($Path)
    try {
        $buf = New-Object byte[] 8
        [void]$fs.Read($buf, 0, $buf.Length)

        if ($buf[0] -eq 0x89 -and $buf[1] -eq 0x50 -and $buf[2] -eq 0x4E -and $buf[3] -eq 0x47) {
            return 'png'
        }

        if ($buf[0] -eq 0x5B -or $buf[0] -eq 0x7B) {
            return 'json'
        }

        return 'binary'
    }
    finally {
        $fs.Dispose()
    }
}

function Test-Property {
    param(
        [Parameter(Mandatory)] $Object,
        [Parameter(Mandatory)][string] $Name
    )

    if ($null -eq $Object) { return $false }

    return $null -ne $Object.PSObject.Properties[$Name]
}

function Get-PropertyValue {
    param(
        [Parameter(Mandatory)] $Object,
        [Parameter(Mandatory)][string] $Name,
        $Default = $null
    )

    if ($null -eq $Object) { return $Default }

    $prop = $Object.PSObject.Properties[$Name]
    if ($null -eq $prop) { return $Default }

    return $prop.Value
}

function Get-NestedPropertyValue {
    param(
        [Parameter(Mandatory)] $Object,
        [Parameter(Mandatory)][string[]] $Path,
        $Default = $null
    )

    $current = $Object

    foreach ($segment in $Path) {
        if ($null -eq $current) { return $Default }

        $prop = $current.PSObject.Properties[$segment]
        if ($null -eq $prop) { return $Default }

        $current = $prop.Value
    }

    return $current
}

function Get-BEUInt32 {
    param([byte[]]$Bytes)
    return ($Bytes[0] -shl 24) -bor ($Bytes[1] -shl 16) -bor ($Bytes[2] -shl 8) -bor $Bytes[3]
}

function Get-PngTextChunks {
    param([string]$Path)

    $result = [System.Collections.Generic.List[object]]::new()
    $fs = [System.IO.File]::OpenRead($Path)
    $br = [System.IO.BinaryReader]::new($fs)

    try {
        [void]$br.ReadBytes(8)

        while ($fs.Position -lt $fs.Length) {
            $lenBytes = $br.ReadBytes(4)
            if ($lenBytes.Length -lt 4) { break }

            $length = Get-BEUInt32 $lenBytes
            $type = [System.Text.Encoding]::ASCII.GetString($br.ReadBytes(4))
            $data = $br.ReadBytes($length)
            [void]$br.ReadBytes(4)

            if ($type -eq 'tEXt') {
                $nullIndex = [Array]::IndexOf($data, [byte]0)
                if ($nullIndex -gt 0) {
                    $key = [System.Text.Encoding]::Latin1.GetString($data, 0, $nullIndex)
                    $value = [System.Text.Encoding]::Latin1.GetString(
                        $data,
                        $nullIndex + 1,
                        $data.Length - $nullIndex - 1
                    )

                    $result.Add([pscustomobject]@{
                        ChunkType = $type
                        Key       = $key
                        Value     = $value
                    })
                }
            }

            if ($type -eq 'IEND') { break }
        }
    }
    finally {
        $br.Dispose()
        $fs.Dispose()
    }

    return $result
}


function Convert-ContentPartsToText {
    param($Content)

    if ($null -eq $Content) { return $null }

    $parts = @()

    $contentType = Get-PropertyValue $Content 'content_type'
    $contentParts = Get-PropertyValue $Content 'parts'
    $contentText = Get-PropertyValue $Content 'text'

    if ($contentType -eq 'text' -and $null -ne $contentParts) {
        foreach ($p in $contentParts) {
            if ($p -is [string]) {
                $parts += $p
            }
        }
    }
    elseif ($null -ne $contentParts) {
        foreach ($p in $contentParts) {
            if ($p -is [string]) {
                $parts += $p
                continue
            }

            $partType = Get-PropertyValue $p 'content_type'
            if ($partType -eq 'image_asset_pointer') {
                $assetPointer = Get-PropertyValue $p 'asset_pointer'
                $parts += "[image_asset_pointer:$assetPointer]"
            }
            elseif ($null -ne $partType) {
                $parts += "[part:$partType]"
            }
            else {
                $parts += ($p | ConvertTo-Json -Compress -Depth 20)
            }
        }
    }
    elseif ($null -ne $contentText) {
        return [string]$contentText
    }

    return ($parts -join "`n").Trim()
}


function Get-NormalizedMessagesFromConversationFile {
    param([string]$Path)

    $json = Get-Content -Raw -Encoding UTF8 -Path $Path | ConvertFrom-Json

    foreach ($conv in $json) {
        $convId = Get-PropertyValue $conv 'conversation_id'
        if ([string]::IsNullOrWhiteSpace($convId)) {
            $convId = Get-PropertyValue $conv 'id'
        }

        $title = Get-PropertyValue $conv 'title'
        $currentNode = Get-PropertyValue $conv 'current_node'
        $mapping = Get-PropertyValue $conv 'mapping'

        if ($null -eq $mapping) { continue }

        foreach ($prop in $mapping.PSObject.Properties) {
            $nodeId = $prop.Name
            $node = $prop.Value
            $msg = Get-PropertyValue $node 'message'

            if ($null -eq $msg) { continue }

            $content = Get-PropertyValue $msg 'content'
            $author = Get-PropertyValue $msg 'author'

            $text = Convert-ContentPartsToText -Content $content

            $attachments = $null
            $aggregateCode = $null
            $executionOutput = $null

            $attachmentsObj = Get-NestedPropertyValue $msg @('metadata', 'attachments')
            if ($null -ne $attachmentsObj) {
                try {
                    $attachments = $attachmentsObj | ConvertTo-Json -Compress -Depth 20
                } catch {}
            }

            $aggregateCodeObj = Get-NestedPropertyValue $msg @('metadata', 'aggregate_result', 'code')
            if ($null -ne $aggregateCodeObj) {
                $aggregateCode = [string]$aggregateCodeObj
            }

            $contentType = Get-PropertyValue $content 'content_type'
            $contentText = Get-PropertyValue $content 'text'
            if ($contentType -eq 'execution_output' -and $null -ne $contentText) {
                $executionOutput = [string]$contentText
            }

            [pscustomobject]@{
                SourceFile       = [System.IO.Path]::GetFileName($Path)
                ConversationId   = $convId
                Title            = $title
                ConversationTime = Get-PropertyValue $conv 'create_time'
                CurrentNode      = $currentNode
                NodeId           = $nodeId
                ParentNodeId     = Get-PropertyValue $node 'parent'
                ChildNodeIds     = @((Get-PropertyValue $node 'children' @())) -join ','
                Role             = Get-PropertyValue $author 'role'
                AuthorName       = Get-PropertyValue $author 'name'
                CreateTime       = Get-PropertyValue $msg 'create_time'
                UpdateTime       = Get-PropertyValue $msg 'update_time'
                Channel          = Get-PropertyValue $msg 'channel'
                Recipient        = Get-PropertyValue $msg 'recipient'
                ContentType      = $contentType
                Text             = $text
                AttachmentsJson  = $attachments
                AggregateCode    = $aggregateCode
                ExecutionOutput  = $executionOutput
            }
        }
    }
}
# ----- MAIN -----

$root = "D:\Downloads\7026-03-03-06-30-32-thiso\"

# 1) Inventory files
$inventory = foreach ($f in Get-ChildItem -File $root) {
    $kind = Get-FileKind -Path $f.FullName
    $pngText = $null

   <# if ($kind -eq 'png') {
        $pngText = Get-PngTextChunks -Path $f.FullName
    }#>
    try{
        [pscustomobject]@{
            Name        = $f.Name
            FullName    = $f.FullName
            Length      = $f.Length
            Kind        = $kind
            Sha256      = (Get-FileHash -Algorithm SHA256 -Path $f.FullName).Hash
            PngTextJson = if ($pngText) { $pngText | ConvertTo-Json -Compress -Depth 10 } else { $null }
        }
    } catch {$f}
}

$inventory | Sort-Object Kind, Name | Tee-Object -FilePath "$root\inventory.json" | Out-Null

# 2) Normalize all conversation messages
$allMessages = foreach ($jsonFile in Get-ChildItem -File $root -Filter "conversations-*.json") {
    Get-NormalizedMessagesFromConversationFile -Path $jsonFile.FullName
}

# 3) Save flat JSONL-like JSON
$allMessages | Sort-Object CreateTime |
    ConvertTo-Json -Depth 20 |
    Set-Content -Encoding UTF8 "$root\normalized-messages.json"

# 4) Example filters
$allMessages |
    Where-Object { $_.Text -match 'matplotlib|plot|chart' -or $_.AggregateCode -match 'matplotlib' } |
    Sort-Object CreateTime |
    Select-Object SourceFile, ConversationId, Title, Role, CreateTime, ContentType, Text |
    Export-Csv "$root\matplotlib-related.csv" -NoTypeInformation -Encoding UTF8

$allMessages |
    Where-Object { $_.AttachmentsJson } |
    Select-Object SourceFile, ConversationId, Title, Role, CreateTime, AttachmentsJson |
    Export-Csv "$root\messages-with-attachments.csv" -NoTypeInformation -Encoding UTF8