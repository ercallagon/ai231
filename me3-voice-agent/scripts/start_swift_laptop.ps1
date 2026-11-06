# Run locally on Windows. The HPC bridge and VS Code port forward must already be running.
[CmdletBinding()]
param(
    [uri]$SwiftUrl = 'http://127.0.0.1:8765/'
)

$ErrorActionPreference = 'Stop'
if (-not $SwiftUrl.IsAbsoluteUri -or $SwiftUrl.Scheme -ne 'http' -or
    -not $SwiftUrl.IsLoopback -or $SwiftUrl.UserInfo -or $SwiftUrl.Query -or
    $SwiftUrl.Fragment -or $SwiftUrl.AbsolutePath -ne '/') {
    throw 'SwiftUrl must be an HTTP loopback root address without credentials, query or fragment.'
}

if (-not (Get-Process -Name Spotify -ErrorAction SilentlyContinue)) {
    try {
        Start-Process -FilePath 'spotify:'
    } catch {
        throw 'Install the Spotify desktop app and sign in once with your authorized Premium account.'
    }
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if (Get-Process -Name Spotify -ErrorAction SilentlyContinue) { break }
        Start-Sleep -Milliseconds 500
    }
    if (-not (Get-Process -Name Spotify -ErrorAction SilentlyContinue)) {
        throw 'Spotify desktop did not start. Install it and sign in once before launching this test.'
    }
}

# Opening the receiver does not start a song. Swift issues playback after a voice request.
Write-Host 'Spotify receiver is running. Use the same account that authorized ME3.'
Start-Process -FilePath $SwiftUrl.AbsoluteUri
Write-Host 'Swift opened. Allow microphone access and select Start conversation.'
