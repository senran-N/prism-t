[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Listen = "127.0.0.1:3090",
    [switch]$NoOpen,
    [switch]$Rebuild,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent

function Get-BrowserBaseUrl([string]$ListenAddr) {
    if ($ListenAddr -match '^(?:0\.0\.0\.0|\[?::\]?):(\d+)$') {
        return "http://127.0.0.1:$($Matches[1])"
    }
    if ($ListenAddr -match '^:(\d+)$') {
        return "http://127.0.0.1:$($Matches[1])"
    }
    return "http://$ListenAddr"
}

$studioUrl = Get-BrowserBaseUrl $Listen
$alreadyRunning = $false
try {
    $null = Invoke-RestMethod -Uri "$studioUrl/api/v1/health" -TimeoutSec 2
    $alreadyRunning = $true
} catch {
    $alreadyRunning = $false
}

if ($alreadyRunning) {
    Write-Host "[prism-t] Web Studio is already running at $studioUrl"
    if (-not $NoOpen) {
        Start-Process $studioUrl | Out-Null
    }
    exit 0
}

Push-Location $repoRoot
try {
    $webIndex = Join-Path $repoRoot "web\dist\index.html"
    if ($Rebuild -or -not (Test-Path -LiteralPath $webIndex)) {
        Write-Host "[prism-t] Building Web Studio frontend..."
        & npm --prefix web run build
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    $exePath = Join-Path $repoRoot "prismtd.exe"
    if ($Rebuild -or -not (Test-Path -LiteralPath $exePath)) {
        Write-Host "[prism-t] Building prismtd.exe..."
        $env:CGO_ENABLED = "0"
        & go build -o prismtd.exe ./cmd/prismtd
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    $dataDir = if ($env:PRISM_T_DATA_DIR) { $env:PRISM_T_DATA_DIR } else { Join-Path $repoRoot "data" }
    $settingsFile = Join-Path $dataDir "settings.json"
    $piModelsPath = Join-Path $env:USERPROFILE ".pi\agent\models.json"
    if (-not (Test-Path -LiteralPath $settingsFile) -and [string]::IsNullOrWhiteSpace($env:PRISM_T_LLM_BASE_URL) -and (Test-Path -LiteralPath $piModelsPath)) {
        try {
            $piModels = Get-Content -LiteralPath $piModelsPath -Raw | ConvertFrom-Json
            $prov = $null
            $pickedModel = $null
            if ($piModels.providers) {
                foreach ($pProp in $piModels.providers.PSObject.Properties) {
                    if ($pProp.Value.baseUrl -and $pProp.Value.models.Count -gt 0) {
                        $prov = $pProp.Value
                        $pickedModel = $pProp.Value.models[0]
                        break
                    }
                }
            }
            if ($prov -and $pickedModel) {
                $env:PRISM_T_LLM_BASE_URL = [string]$prov.baseUrl
                $env:PRISM_T_LLM_MODEL = [string]$pickedModel.id
                $cred = [string]$prov.apiKey
                $envCred = if ($cred) { [Environment]::GetEnvironmentVariable($cred) } else { $null }
                $env:PRISM_T_LLM_API_KEY = if ($envCred) { $envCred } else { $cred }
                if ([string]::IsNullOrWhiteSpace($env:PRISM_T_LLM_VISION)) {
                    $env:PRISM_T_LLM_VISION = if ($pickedModel.input -contains "image") { "true" } else { "false" }
                }
                if ([string]::IsNullOrWhiteSpace($env:PRISM_T_LLM_REASONING_EFFORT)) {
                    $env:PRISM_T_LLM_REASONING_EFFORT = "none"
                }
            }
        } catch {
            Write-Warning "[prism-t] Skipped optional local model discovery: $_"
        }
    }

    $serveArgs = @("serve", "--listen", $Listen)
    if (-not $NoOpen) {
        $serveArgs += "--open"
    }
    if ($ExtraArgs) {
        $serveArgs += $ExtraArgs
    }

    Write-Host "[prism-t] Launching Web Studio at $studioUrl ..."
    & $exePath @serveArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
