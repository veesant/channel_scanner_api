<#
.SYNOPSIS
  Manual test runner for scanners/morning_sine_scanner.py (Yahoo or Polygon).

.EXAMPLE
  .\scripts\run_morning_sine_local.ps1 -Tickers AAPL,MSFT

.EXAMPLE
  .\scripts\run_morning_sine_local.ps1 -TickersFile data\nasdaq_01.txt -OnlyMatches

.EXAMPLE
  $env:POLYGON_API_KEY = "your_key"
  .\scripts\run_morning_sine_local.ps1 -DataSource polygon -Tickers AAPL -Out out_poly.json
#>

[CmdletBinding()]
param(
    [string[]] $Tickers = @("AAPL"),
    [string] $TickersFile = "",
    [ValidateSet("yfinance", "polygon", "auto")]
    [string] $DataSource = "yfinance",
    [string] $Interval = "15m",
    [int] $LookbackDays = 7,
    [string] $PremarketStart = "04:00",
    [string] $PremarketEnd = "09:00",
    [ValidateSet("today_only", "latest_available")]
    [string] $PremarketDayPolicy = "latest_available",
    [switch] $OnlyMatches,
    [switch] $PolygonFallbackYfinance,
    [double] $MinVwapTouchRatio = 0.0,
    [string] $Out = "morning_sine_test.json"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Error "python not found on PATH"
}

$argsList = @(
    "scanners/morning_sine_scanner.py",
    "--data-source", $DataSource,
    "--interval", $Interval,
    "--lookback_days", "$LookbackDays",
    "--premarket-start", $PremarketStart,
    "--premarket-end", $PremarketEnd,
    "--premarket-day-policy", $PremarketDayPolicy,
    "--min-vwap-touch-ratio", "$MinVwapTouchRatio",
    "--out", $Out
)

if ($TickersFile) {
    $argsList += @("--tickers_file", $TickersFile)
}
elseif ($Tickers -and $Tickers.Count -gt 0) {
    $argsList += @("--tickers") + $Tickers
}

if ($OnlyMatches) { $argsList += "--only_matches" }
if ($PolygonFallbackYfinance) { $argsList += "--polygon-fallback-yfinance" }

Write-Host "Repo: $Root"
Write-Host "Command: python $($argsList -join ' ')"
& python @argsList
exit $LASTEXITCODE
