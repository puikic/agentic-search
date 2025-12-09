# MS MARCO Passage Ranking Dataset Download Script (PowerShell)
# For NDCG-based retrieval training

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Downloading MS MARCO Passage Ranking Dataset" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Create data directory
$dataDir = "data"
if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir | Out-Null
}
Set-Location $dataDir

# 1. Download passage collection
Write-Host "" -ForegroundColor Yellow
Write-Host "[1/4] Downloading passage collection (8.8M passages, ~1GB)..." -ForegroundColor Yellow
$collectionFile = "collection.tar.gz"
if (-not (Test-Path $collectionFile)) {
    Invoke-WebRequest -Uri "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz" -OutFile $collectionFile
}
Write-Host "[OK] collection.tar.gz downloaded" -ForegroundColor Green

# 2. Download queries
Write-Host "" -ForegroundColor Yellow
Write-Host "[2/4] Downloading queries (~42MB)..." -ForegroundColor Yellow
$queriesFile = "queries.tar.gz"
if (-not (Test-Path $queriesFile)) {
    Invoke-WebRequest -Uri "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz" -OutFile $queriesFile
}
Write-Host "[OK] queries.tar.gz downloaded" -ForegroundColor Green

# 3. Download qrels (relevance labels) - KEY for NDCG!
Write-Host "" -ForegroundColor Yellow
Write-Host "[3/4] Downloading qrels (relevance labels for NDCG)..." -ForegroundColor Yellow
$qrelsTrain = "qrels.train.tsv"
if (-not (Test-Path $qrelsTrain)) {
    Invoke-WebRequest -Uri "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv" -OutFile $qrelsTrain
}
$qrelsDev = "qrels.dev.tsv"
if (-not (Test-Path $qrelsDev)) {
    Invoke-WebRequest -Uri "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv" -OutFile $qrelsDev
}
Write-Host "[OK] qrels.train.tsv / qrels.dev.tsv downloaded" -ForegroundColor Green

# 4. Skip large optional file
Write-Host "" -ForegroundColor Yellow
Write-Host "[4/4] Skipping top1000.dev (~2.5GB, optional)..." -ForegroundColor Yellow
Write-Host "     Download manually if needed" -ForegroundColor Gray

Write-Host "" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Download Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "" -ForegroundColor White
Write-Host "Files:" -ForegroundColor White
Write-Host "  collection.tar.gz - 8,841,823 passages (corpus)" -ForegroundColor White
Write-Host "  queries.tar.gz    - train/dev queries" -ForegroundColor White
Write-Host "  qrels.train.tsv   - relevance labels (532,761 pairs)" -ForegroundColor White
Write-Host "  qrels.dev.tsv     - dev relevance labels (59,273 pairs)" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "Next Steps:" -ForegroundColor White
Write-Host "  1. Extract: tar -xzf collection.tar.gz" -ForegroundColor White
Write-Host "     (Use 7-Zip or Git Bash on Windows)" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "  2. Process data: python prepare_marco_data.py" -ForegroundColor White
Write-Host "" -ForegroundColor White

Set-Location ..
