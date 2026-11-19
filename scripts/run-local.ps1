[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$PrismArguments,
    [string]$Provider = "",
    [string]$Model = "",
    [switch]$NoVision,
    [string]$ReasoningEffort = "",
    [switch]$VisionAudit,
    [switch]$SkipSourceOCR
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent

# Optional local model discovery from user-level ~/.pi/agent/models.json when
# PRISM_T_LLM_BASE_URL is not already configured or -Provider/-Model is specified.
$piModelsPath = Join-Path $env:USERPROFILE ".pi\agent\models.json"
if ((-not [string]::IsNullOrWhiteSpace($Provider) -or -not [string]::IsNullOrWhiteSpace($Model) -or [string]::IsNullOrWhiteSpace($env:PRISM_T_LLM_BASE_URL)) -and (Test-Path -LiteralPath $piModelsPath)) {
    $piModels = Get-Content -LiteralPath $piModelsPath -Raw | ConvertFrom-Json
    $providerConfig = $null
    if (-not [string]::IsNullOrWhiteSpace($Provider)) {
        $providerConfig = $piModels.providers.$Provider
        if ($null -eq $providerConfig) { throw "Provider '$Provider' was not found in $piModelsPath" }
    } elseif ($piModels.providers) {
        foreach ($pProp in $piModels.providers.PSObject.Properties) {
            if ($pProp.Value.baseUrl -and $pProp.Value.models.Count -gt 0) {
                $providerConfig = $pProp.Value
                break
            }
        }
    }
    if ($null -ne $providerConfig) {
        $modelConfig = if (-not [string]::IsNullOrWhiteSpace($Model)) {
            $providerConfig.models | Where-Object { $_.id -eq $Model } | Select-Object -First 1
        } else {
            $providerConfig.models | Select-Object -First 1
        }
        if ($null -eq $modelConfig) { throw "Model '$Model' was not found for provider '$Provider'" }
        $env:PRISM_T_LLM_BASE_URL = [string]$providerConfig.baseUrl
        $env:PRISM_T_LLM_MODEL = [string]$modelConfig.id
        if ($NoVision) {
            $env:PRISM_T_LLM_VISION = "false"
        } elseif ([string]::IsNullOrWhiteSpace($env:PRISM_T_LLM_VISION)) {
            $env:PRISM_T_LLM_VISION = if ($modelConfig.input -contains "image") { "true" } else { "false" }
        }
        $credential = [string]$providerConfig.apiKey
        $environmentCredential = if ($credential) { [Environment]::GetEnvironmentVariable($credential) } else { $null }
        $env:PRISM_T_LLM_API_KEY = if ($environmentCredential) { $environmentCredential } else { $credential }
    }
} elseif ($NoVision) {
    $env:PRISM_T_LLM_VISION = "false"
}

if (-not [string]::IsNullOrWhiteSpace($ReasoningEffort)) {
    $env:PRISM_T_LLM_REASONING_EFFORT = $ReasoningEffort
}
if ($VisionAudit) {
    $env:PRISM_T_LLM_VISION_AUDIT = "true"
}
if ($SkipSourceOCR) {
    $env:PRISM_T_SKIP_SOURCE_OCR = "true"
}
if (-not $PrismArguments) { $PrismArguments = @("serve") }
Push-Location $repoRoot
try {
    $ErrorActionPreference = "Continue"
    & (Join-Path $repoRoot "prismtd.exe") @PrismArguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
