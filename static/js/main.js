/**
 * MusangKing Classification System - Frontend JavaScript
 * Handles image upload, processing, and results display
 */

// DOM Elements
const uploadArea = document.getElementById('upload-area');
const imageInput = document.getElementById('image-input');
const uploadPreview = document.getElementById('upload-preview');
const previewImage = document.getElementById('preview-image');
const clearBtn = document.getElementById('clear-btn');
const processBtn = document.getElementById('process-btn');

const gammaSlider = document.getElementById('gamma-slider');
const gammaValue = document.getElementById('gamma-value');
const kSlider = document.getElementById('k-slider');
const kValue = document.getElementById('k-value');

const originalContainer = document.getElementById('original-container');
const originalImage = document.getElementById('original-image');
const maskContainer = document.getElementById('mask-container');
const maskImage = document.getElementById('mask-image');
const processingSteps = document.getElementById('processing-steps');

const classificationResult = document.getElementById('classification-result');
const classificationBadge = document.getElementById('classification-badge');
const classificationText = document.getElementById('classification-text');
const parametersSection = document.getElementById('parameters-section');
const emptyState = document.getElementById('empty-state');

// State
let currentFilename = null;

// ============================================
// Upload Handling
// ============================================

// Click to upload
uploadArea.addEventListener('click', () => {
    imageInput.click();
});

// File input change
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileUpload(file);
    }
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileUpload(file);
    }
});

// Clear button
clearBtn.addEventListener('click', () => {
    resetUpload();
});

// Handle file upload
async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('image', file);

    try {
        // Show preview immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadArea.style.display = 'none';
            uploadPreview.style.display = 'block';
        };
        reader.readAsDataURL(file);

        // Upload to server
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            currentFilename = data.filename;
            processBtn.disabled = false;

            // Show original image in visualization panel
            showOriginalImage(data.filepath);
        } else {
            alert('Upload failed: ' + data.error);
            resetUpload();
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Upload failed. Please try again.');
        resetUpload();
    }
}

// Show original image
function showOriginalImage(filepath) {
    originalImage.src = filepath;
    originalImage.style.display = 'block';
    originalContainer.querySelector('.placeholder').style.display = 'none';
}

// Reset upload
function resetUpload() {
    imageInput.value = '';
    previewImage.src = '';
    uploadArea.style.display = 'block';
    uploadPreview.style.display = 'none';
    processBtn.disabled = true;
    currentFilename = null;

    // Reset visualization
    originalImage.style.display = 'none';
    originalContainer.querySelector('.placeholder').style.display = 'flex';
    maskImage.style.display = 'none';
    maskContainer.querySelector('.placeholder').style.display = 'flex';

    // Reset results
    hideResults();
}

// ============================================
// Slider Controls
// ============================================

gammaSlider.addEventListener('input', () => {
    gammaValue.textContent = gammaSlider.value;
});

kSlider.addEventListener('input', () => {
    kValue.textContent = kSlider.value;
});

// ============================================
// Processing
// ============================================

processBtn.addEventListener('click', async () => {
    if (!currentFilename) return;

    // Show loading state
    setProcessing(true);

    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: currentFilename,
                gamma: parseFloat(gammaSlider.value),
                k_value: parseInt(kSlider.value)
            })
        });

        const data = await response.json();

        if (data.success) {
            // Show processed mask
            showProcessedMask(data.mask_path);

            // Show results
            showResults(data);
        } else {
            alert('Processing failed: ' + data.error);
        }
    } catch (error) {
        console.error('Processing error:', error);
        alert('Processing failed. Please try again.');
    } finally {
        setProcessing(false);
    }
});

// Set processing state
function setProcessing(isProcessing) {
    const btnText = processBtn.querySelector('.btn-text');
    const btnLoader = processBtn.querySelector('.btn-loader');

    if (isProcessing) {
        btnText.textContent = 'Processing...';
        btnLoader.style.display = 'block';
        processBtn.disabled = true;
        processingSteps.style.display = 'flex';
        animateProcessingSteps();
    } else {
        btnText.textContent = 'Process Image';
        btnLoader.style.display = 'none';
        processBtn.disabled = false;
    }
}

// Animate processing steps
function animateProcessingSteps() {
    const steps = processingSteps.querySelectorAll('.step');
    let currentStep = 0;

    const interval = setInterval(() => {
        steps.forEach((step, index) => {
            step.classList.remove('active', 'completed');
            if (index < currentStep) {
                step.classList.add('completed');
            } else if (index === currentStep) {
                step.classList.add('active');
            }
        });

        currentStep++;

        if (currentStep > steps.length) {
            clearInterval(interval);
            steps.forEach(step => step.classList.add('completed'));
        }
    }, 400);
}

// Show processed mask
function showProcessedMask(filepath) {
    maskImage.src = filepath + '?' + Date.now(); // Cache bust
    maskImage.style.display = 'block';
    maskContainer.querySelector('.placeholder').style.display = 'none';
}

// ============================================
// Results Display
// ============================================

function showResults(data) {
    // Hide empty state
    emptyState.style.display = 'none';

    // Show classification
    classificationResult.style.display = 'block';
    classificationBadge.className = 'classification-badge ' + data.classification_class;
    classificationText.textContent = data.classification;

    // Update attributes table
    document.getElementById('attr-compactness').textContent = data.features.compactness;
    document.getElementById('attr-smoothness').textContent = data.features.smoothness;
    document.getElementById('attr-mean-hue').textContent = data.features.mean_hue + '°';

    // Show parameters
    parametersSection.style.display = 'block';
    document.getElementById('param-gamma').textContent = data.parameters.gamma;
    document.getElementById('param-k').textContent = data.parameters.k_value;
}

function hideResults() {
    emptyState.style.display = 'flex';
    classificationResult.style.display = 'none';
    parametersSection.style.display = 'none';
    processingSteps.style.display = 'none';

    // Reset attributes
    document.getElementById('attr-compactness').textContent = '--';
    document.getElementById('attr-smoothness').textContent = '--';
    document.getElementById('attr-mean-hue').textContent = '--';
}

// ============================================
// Initialize
// ============================================

console.log('MusangKing Classification System loaded');
