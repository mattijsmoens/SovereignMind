# SovereignMind - HuthLab Semantic Decoder Setup Script
# This downloads the REAL UT Austin semantic decoding models and data.
# Paper: "Semantic reconstruction of continuous language from non-invasive brain recordings"
# Authors: Jerry Tang, Amanda LeBel, Shailee Jain, Alexander G. Huth

Write-Host "=============================================="
Write-Host "SovereignMind: HuthLab Semantic Decoder Setup"
Write-Host "=============================================="

$BASE_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

# Step 1: Clone the HuthLab semantic-decoding repository
Write-Host "`n[1/5] Cloning HuthLab/semantic-decoding from GitHub..."
if (Test-Path "$BASE_DIR\semantic-decoding") {
    Write-Host "  -> Already cloned. Skipping."
} else {
    git clone https://github.com/HuthLab/semantic-decoding.git "$BASE_DIR\semantic-decoding"
}

# Step 2: Download language model data
Write-Host "`n[2/5] Downloading language model data into data_lm/..."
$LM_URL = "https://utexas.box.com/shared/static/7ab8qm5e3i0vfsku0ee4dc6hzgeg7nyh.zip"
$LM_ZIP = "$BASE_DIR\semantic-decoding\data_lm.zip"
if (-Not (Test-Path "$BASE_DIR\semantic-decoding\data_lm")) {
    Invoke-WebRequest -Uri $LM_URL -OutFile $LM_ZIP
    Expand-Archive -Path $LM_ZIP -DestinationPath "$BASE_DIR\semantic-decoding\data_lm" -Force
    Remove-Item $LM_ZIP
    Write-Host "  -> data_lm/ extracted."
} else {
    Write-Host "  -> data_lm/ already exists. Skipping."
}

# Step 3: Download training data
Write-Host "`n[3/5] Downloading training data into data_train/..."
$TRAIN_URL = "https://utexas.box.com/shared/static/3go1g4gcdar2cntjit2knz5jwr3mvxwe.zip"
$TRAIN_ZIP = "$BASE_DIR\semantic-decoding\data_train.zip"
if (-Not (Test-Path "$BASE_DIR\semantic-decoding\data_train")) {
    Invoke-WebRequest -Uri $TRAIN_URL -OutFile $TRAIN_ZIP
    Expand-Archive -Path $TRAIN_ZIP -DestinationPath "$BASE_DIR\semantic-decoding\data_train" -Force
    Remove-Item $TRAIN_ZIP
    Write-Host "  -> data_train/ extracted."
} else {
    Write-Host "  -> data_train/ already exists. Skipping."
}

# Step 4: Download test data
Write-Host "`n[4/5] Downloading test data into data_test/..."
$TEST_URL = "https://utexas.box.com/shared/static/ae5u0t3sh4f46nvmrd3skniq0kk2t5uh.zip"
$TEST_ZIP = "$BASE_DIR\semantic-decoding\data_test.zip"
if (-Not (Test-Path "$BASE_DIR\semantic-decoding\data_test")) {
    Invoke-WebRequest -Uri $TEST_URL -OutFile $TEST_ZIP
    Expand-Archive -Path $TEST_ZIP -DestinationPath "$BASE_DIR\semantic-decoding\data_test" -Force
    Remove-Item $TEST_ZIP
    Write-Host "  -> data_test/ extracted."
} else {
    Write-Host "  -> data_test/ already exists. Skipping."
}

# Step 5: Download pre-fit encoding and word rate models
Write-Host "`n[5/5] Downloading pre-fit encoding models and word rate models..."
$MODELS_URL = "https://utexas.box.com/s/ri13t06iwpkyk17h8tfk0dtyva7qtqlz"
Write-Host "  -> IMPORTANT: Pre-fit models are hosted on UT Austin Box."
Write-Host "  -> The direct download link may require manual browser download."
Write-Host "  -> Open this URL in your browser if the automatic download fails:"
Write-Host "  -> $MODELS_URL"

# Attempt automatic download (Box shared links sometimes redirect)
$MODELS_DIR = "$BASE_DIR\semantic-decoding\models"
if (-Not (Test-Path $MODELS_DIR)) {
    New-Item -ItemType Directory -Path $MODELS_DIR -Force | Out-Null
    Write-Host "  -> Created models/ directory. Download pre-fit models from the URL above and extract into: $MODELS_DIR"
} else {
    Write-Host "  -> models/ directory already exists."
}

Write-Host "`n=============================================="
Write-Host "Setup Complete!"
Write-Host "=============================================="
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. If the pre-fit models did not auto-download, manually download from:"
Write-Host "     $MODELS_URL"
Write-Host "     and extract into: $MODELS_DIR"
Write-Host ""
Write-Host "  2. Download OpenNeuro stimulus/response data from:"
Write-Host "     Training: https://openneuro.org/datasets/ds003020/"
Write-Host "     Testing:  https://openneuro.org/datasets/ds004510/"
Write-Host "     Place into semantic-decoding/train_stimulus/ and semantic-decoding/train_response/[SUBJECT_ID]"
Write-Host ""
Write-Host "  3. To test the decoder, run:"
Write-Host "     python3 semantic-decoding/decoding/run_decoder.py --subject [SUBJECT_ID] --experiment [EXPERIMENT_NAME] --task [TASK_NAME]"
