# Windows-only QA fallback when the bundled DOCX renderer has no LibreOffice.
# Export this report read-only with a new hidden Word instance, then use Poppler.
param([Parameter(Mandatory=$true)][string]$PopplerDirectory)
$ErrorActionPreference = 'Stop'
$reportDirectory = [IO.Path]::GetFullPath($PSScriptRoot)
$aitkenDirectory = [IO.Path]::GetFullPath((Join-Path $reportDirectory '..'))
$renderDirectory = [IO.Path]::GetFullPath((Join-Path $aitkenDirectory 'artifacts/comparison_20260912/word_render'))
if (-not $renderDirectory.StartsWith($aitkenDirectory + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'QA output escaped Aitken'
}
$null = New-Item -ItemType Directory -Path $renderDirectory -Force
$sourceReport = Join-Path $reportDirectory 'London_PM25_Comparison_and_Error_Analysis.docx'
$pdfReport = Join-Path $renderDirectory 'London_PM25_Comparison_and_Error_Analysis.pdf'
$sourceHash = (Get-FileHash -LiteralPath $sourceReport -Algorithm SHA256).Hash
$comparisonWord = $null
$comparisonDocument = $null
try {
    $comparisonWord = New-Object -ComObject Word.Application
    $comparisonWord.Visible = $false
    $comparisonWord.DisplayAlerts = 0
    $comparisonWord.AutomationSecurity = 3
    $comparisonDocument = $comparisonWord.Documents.Open($sourceReport, $false, $true)
    $comparisonDocument.ExportAsFixedFormat($pdfReport, 17)
} finally {
    if ($null -ne $comparisonDocument) {
        $comparisonDocument.Close(0)
        $null = [Runtime.InteropServices.Marshal]::FinalReleaseComObject($comparisonDocument)
    }
    if ($null -ne $comparisonWord) {
        $comparisonWord.Quit()
        $null = [Runtime.InteropServices.Marshal]::FinalReleaseComObject($comparisonWord)
    }
}
if ($sourceHash -ne (Get-FileHash -LiteralPath $sourceReport -Algorithm SHA256).Hash) {
    throw 'Read-only rendering modified the source DOCX'
}
& (Join-Path $PopplerDirectory 'pdfinfo.exe') $pdfReport
if ($LASTEXITCODE -ne 0) { throw 'PDF inspection failed' }
& (Join-Path $PopplerDirectory 'pdftoppm.exe') -png -scale-to 1600 $pdfReport (Join-Path $renderDirectory 'page')
if ($LASTEXITCODE -ne 0) { throw 'Page rendering failed' }
$pdfTextExecutable = Join-Path $PopplerDirectory 'pdftotext.exe'
if (Test-Path -LiteralPath $pdfTextExecutable) {
    & $pdfTextExecutable -layout $pdfReport (Join-Path $renderDirectory 'report.txt')
    if ($LASTEXITCODE -ne 0) { throw 'PDF text extraction failed' }
}
Write-Output "Verified source unchanged. QA output: $renderDirectory"
