<#
Read-only ME3 report feed. Run on the laptop; no local server/port is created.
Ctrl+C closes the SSH feed. Generated client files stay under var/results.
#>
[CmdletBinding()]
param(
    [ValidatePattern('^(?!-)[A-Za-z0-9_.@-]+$')]
    [Parameter(Mandatory=$true)]
    [string]$SshHost,
    [ValidatePattern('^/[A-Za-z0-9_./-]+$')]
    [Parameter(Mandatory=$true)]
    [string]$RemoteProject,
    [string]$Workspace = (Join-Path $env:USERPROFILE 'Downloads\ME3'),
    [ValidateRange(1,60)]
    [int]$Interval = 2,
    [switch]$NoOpen
)
$ErrorActionPreference = 'Stop'
$directory = Join-Path $Workspace 'var\results'
[IO.Directory]::CreateDirectory($directory) | Out-Null
$utf8 = [Text.UTF8Encoding]::new($false)
$htmlPath = Join-Path $directory 'me3-dashboard.html'
$jsonPath = Join-Path $directory 'me3-dashboard.json'
$opened = $false
$previousEncoding = [Console]::OutputEncoding
$previousOutputEncoding = $OutputEncoding
$command = "cd '$RemoteProject' && exec var/cache/venv/bin/python -B -u -m me3_voice.dashboard stream --interval $Interval"
function Write-Artifact([string]$Path, [string]$Content) {
    # Replace only this watcher's generated presentation files.
    $temporary = "$Path.tmp"
    [IO.File]::WriteAllText($temporary, $Content, $utf8)
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}
try {
    [Console]::OutputEncoding = $utf8
    $OutputEncoding = $utf8
    Write-Host "Connecting to $SshHost. This streams reports, not microphone audio."
    Write-Host "Browser feed folder: $directory"
    Write-Host 'In the browser open Diagnostics, choose Connect report feed, then select that folder. Ctrl+C stops the feed.'
    & ssh -T -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -- $SshHost $command | ForEach-Object {
        $line = [string]$_
        if (-not $line.StartsWith('{')) { return }
        try { $record = $line | ConvertFrom-Json } catch {
            Write-Warning 'An incomplete feed record was ignored.'
            return
        }
        if ($record.snapshot.schema_version -ne 1 -or $null -eq $record.snapshot.reports) {
            Write-Warning 'Unsupported dashboard snapshot; keeping previous data.'
            return
        }
        if ($record.type -eq 'bootstrap' -and $record.html -is [string]) {
            Write-Artifact $htmlPath $record.html
        } elseif ($record.type -ne 'snapshot') { return }
        # Keep the original normalized JSON: PowerShell 5.1's serializer can truncate deep fields.
        # The server also sends this ASCII-escaped compact string for a lossless local write.
        if ($record.snapshot_json -isnot [string]) {
            Write-Warning 'Missing serialized snapshot; update the remote ME3 source.'
            return
        }
        Write-Artifact $jsonPath $record.snapshot_json
        if (-not $opened -and (Test-Path -LiteralPath $htmlPath)) {
            $opened = $true
            Write-Host "Dashboard ready: $htmlPath"
            if (-not $NoOpen) { Start-Process -FilePath $htmlPath }
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "SSH feed ended with exit code $LASTEXITCODE. Previous data remains on disk." }
} finally {
    [Console]::OutputEncoding = $previousEncoding
    $OutputEncoding = $previousOutputEncoding
    Write-Host 'Report feed stopped. The browser will mark the last data as stale.'
}
