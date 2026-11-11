param(
    [string]$QaCases = "..\prepared_data\experiments\final_experiment_input_470.csv",
    [string]$Chunks = "..\prepared_data\experiments\combined_chunks_internal_external.csv",
    [int]$TopK = 5,
    [int]$Limit = 0,
    [string]$ModelName = "gpt-5.4-mini",
    [double]$Temperature = 0.0,
    [int]$MaxTokens = 1200,
    [double]$RequestDelaySeconds = 0.0,
    [string]$ModelVariants = "llm_only,static_rag,wag_only,hybrid_rag_wag,hybrid_rag_wag_verifier",
    [string]$OutputStem = "prepared_step6",
    [switch]$Mock,
    [switch]$Resume,
    [switch]$RunAblation
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )
    python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "python $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

Push-Location $Root
try {
    $ModelOut = "outputs/model_outputs/${OutputStem}_model_outputs.jsonl"
    $VerifierOut = "outputs/verifier_outputs/${OutputStem}_verifier_outputs.jsonl"
    $MetricsOut = "outputs/evaluation_results/${OutputStem}_metrics.csv"
    $ErrorsOut = "outputs/evaluation_results/${OutputStem}_error_analysis_cases.csv"

    $modelArgs = @(
        "scripts/01_run_five_models.py",
        "--qa-cases", $QaCases,
        "--chunks", $Chunks,
        "--output", $ModelOut,
        "--top-k", [string]$TopK,
        "--model-name", $ModelName,
        "--temperature", [string]$Temperature,
        "--max-tokens", [string]$MaxTokens,
        "--request-delay-seconds", [string]$RequestDelaySeconds,
        "--model-variants", $ModelVariants
    )
    if ($Mock) {
        $modelArgs += "--mock"
    }
    if ($Resume) {
        $modelArgs += "--resume"
    }
    if ($Limit -gt 0) {
        $modelArgs += @("--limit", [string]$Limit)
    }

    Invoke-CheckedPython -Arguments $modelArgs
    Invoke-CheckedPython -Arguments @("scripts/02_run_verifier.py", "--model-outputs", $ModelOut, "--output", $VerifierOut)
    Invoke-CheckedPython -Arguments @("scripts/03_evaluate_results.py", "--model-outputs", $ModelOut, "--verifier-outputs", $VerifierOut, "--output", $MetricsOut)
    Invoke-CheckedPython -Arguments @("scripts/05_error_analysis.py", "--model-outputs", $ModelOut, "--verifier-outputs", $VerifierOut, "--output", $ErrorsOut)

    if ($RunAblation) {
        $ablationArgs = @(
            "scripts/04_run_ablation_study.py",
            "--qa-cases", $QaCases,
            "--chunks", $Chunks,
            "--top-k", [string]$TopK,
            "--model-name", $ModelName,
            "--temperature", [string]$Temperature,
            "--max-tokens", [string]$MaxTokens,
            "--request-delay-seconds", [string]$RequestDelaySeconds
        )
        if ($Mock) {
            $ablationArgs += "--mock"
        }
        if ($Resume) {
            $ablationArgs += "--resume"
        }
        if ($Limit -gt 0) {
            $ablationArgs += @("--limit", [string]$Limit)
        }
        Invoke-CheckedPython -Arguments $ablationArgs
    }
}
finally {
    Pop-Location
}
