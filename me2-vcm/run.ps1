param([Parameter(ValueFromRemainingArguments=$true)][string[]]$TaskArgs)
$ErrorActionPreference = "Stop"
if (-not $TaskArgs) { $TaskArgs = @("verify") }
py "$PSScriptRoot/scripts/reproduce.py" @TaskArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
