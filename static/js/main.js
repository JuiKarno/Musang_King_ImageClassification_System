/**
 * MusangKing Classification System - Frontend JavaScript
 * Simplified for Hybrid Segmentation (K-Means + AI Fallback)
 */

// DOM Elements
const uploadArea = document.getElementById('upload-area');
const imageInput = document.getElementById('image-input');
const uploadPreview = document.getElementById('upload-preview');
const previewImage = document.getElementById('preview-image');
const clearBtn = document.getElementById('clear-btn');
const processBtn = document.getElementById('process-btn');

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

uploadArea.addEventListener('click', () => {
    imageInput.click();
});

imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileUpload(file);
    }
});

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

clearBtn.addEventListener('click', () => {
    resetUpload();
});

async function handleFileUpload(file) {
    const formData = new FormData();
    formData.append('image', file);

    try {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadArea.style.display = 'none';
            uploadPreview.style.display = 'block';

            // Also show in original container
            if (originalImage && originalContainer) {
                originalImage.src = e.target.result;
                originalImage.style.display = 'block';
                originalContainer.querySelector('.placeholder').style.display = 'none';
            }
        };
        reader.readAsDataURL(file);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.filename) {
            currentFilename = data.filename;
            processBtn.disabled = false;
        } else {
            alert('Upload failed: ' + (data.error || 'Unknown error'));
            resetUpload();
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Upload failed. Please try again.');
        resetUpload();
    }
}

function resetUpload() {
    imageInput.value = '';
    previewImage.src = '';
    uploadArea.style.display = 'block';
    uploadPreview.style.display = 'none';
    processBtn.disabled = true;
    currentFilename = null;

    // Reset visualization
    if (originalImage && originalContainer) {
        originalImage.style.display = 'none';
        originalContainer.querySelector('.placeholder').style.display = 'flex';
    }
    if (maskImage && maskContainer) {
        maskImage.style.display = 'none';
        maskContainer.querySelector('.placeholder').style.display = 'flex';
    }

    // Reset results
    hideResults();
}

// ============================================
// Processing
// ============================================

processBtn.addEventListener('click', async () => {
    if (!currentFilename) return;

    setProcessing(true);

    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: currentFilename
            })
        });

        const data = await response.json();

        if (data.success) {
            // Show mask (Overlay preference for better visual)
            if (maskImage && maskContainer) {
                maskImage.src = (data.overlay_path || data.mask_path) + '?' + Date.now();
                maskImage.style.display = 'block';
                maskContainer.querySelector('.placeholder').style.display = 'none';
            }

            // PIPELINE VISUALIZATION (Modal Popup)
            const toggleBtn = document.getElementById('toggle-pipeline-btn');
            const modal = document.getElementById('pipeline-modal');
            const closeBtn = document.querySelector('.close-modal');

            if (data.pipeline && toggleBtn && modal) {
                toggleBtn.style.display = 'inline-block';

                // Populate step images
                if (document.getElementById('step-gamma')) document.getElementById('step-gamma').src = data.pipeline.gamma;
                if (document.getElementById('step-lab')) document.getElementById('step-lab').src = data.pipeline.lab;
                if (document.getElementById('step-kmeans')) document.getElementById('step-kmeans').src = data.pipeline.kmeans;
                if (document.getElementById('step-morph')) document.getElementById('step-morph').src = data.pipeline.morphology;

                // Open Modal
                toggleBtn.onclick = function () {
                    modal.style.display = "flex";
                }

                // Close Modal
                closeBtn.onclick = function () {
                    modal.style.display = "none";
                }

                // Close if clicked outside
                window.onclick = function (event) {
                    if (event.target == modal) {
                        modal.style.display = "none";
                    }
                }
            }

            // Show results
            showResults(data);
        } else {
            alert('Processing failed: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Processing error:', error);
        alert('Processing failed. Please try again.');
    } finally {
        setProcessing(false);
    }
});

function setProcessing(isProcessing) {
    const btnText = processBtn.querySelector('.btn-text');
    const btnLoader = processBtn.querySelector('.btn-loader');

    if (isProcessing) {
        btnText.textContent = 'Processing...';
        btnLoader.style.display = 'block';
        processBtn.disabled = true;
    } else {
        btnText.textContent = 'Process Image';
        btnLoader.style.display = 'none';
        processBtn.disabled = false;
    }
}

// ============================================
// Results Display
// ============================================

function showResults(data) {
    // Hide empty state
    emptyState.style.display = 'none';

    // Show classification with confidence
    classificationResult.style.display = 'block';
    classificationBadge.className = 'classification-badge ' + data.classification_class;

    if (data.confidence && data.confidence > 0) {
        classificationText.textContent = data.classification + ' (' + data.confidence.toFixed(1) + '%)';
    } else {
        classificationText.textContent = data.classification;
    }

    // Update attributes table
    document.getElementById('attr-compactness').textContent = data.features.compactness;
    document.getElementById('attr-smoothness').textContent = data.features.smoothness;
    document.getElementById('attr-mean-hue').textContent = data.features.mean_hue + '°';

    // Show method used
    parametersSection.style.display = 'block';

    const methodElement = document.getElementById('method-used');
    if (methodElement && data.method_used) {
        methodElement.textContent = data.method_used;
        methodElement.className = 'method-badge ' + (data.method_used.includes('K-Means') ? 'primary' : 'fallback');
    }

    // Show method reference section below images
    const methodReference = document.getElementById('method-reference');
    const pipelineMethod = document.getElementById('pipeline-method');
    const referenceDetails = document.getElementById('reference-details');

    if (methodReference && data.method_used) {
        methodReference.style.display = 'block';
        pipelineMethod.textContent = data.method_used;

        if (data.method_used.includes('K-Means')) {
            pipelineMethod.className = 'method-badge primary';
            referenceDetails.textContent = 'Primary: K-Means Clustering (Proposed Method)';
        } else if (data.method_used.includes('Hybrid')) {
            pipelineMethod.className = 'method-badge primary';
            referenceDetails.textContent = 'Hybrid Strategy: AI (Prediction) + K-Means (Analysis)';
        } else {
            pipelineMethod.className = 'method-badge fallback';
            referenceDetails.textContent = 'K-Means failed (Empty Mask). Switched to AI Fallback (rembg).';
        }
    }
}

function hideResults() {
    emptyState.style.display = 'flex';
    classificationResult.style.display = 'none';
    parametersSection.style.display = 'none';

    const methodReference = document.getElementById('method-reference');
    if (methodReference) {
        methodReference.style.display = 'none';
    }

    // Hide pipeline toggle and grid
    const toggleBtn = document.getElementById('toggle-pipeline-btn');
    const pipelineDiv = document.getElementById('pipeline-steps');
    if (toggleBtn) toggleBtn.style.display = 'none';
    if (pipelineDiv) pipelineDiv.style.display = 'none';

    // Reset attributes
    document.getElementById('attr-compactness').textContent = '--';
    document.getElementById('attr-smoothness').textContent = '--';
    document.getElementById('attr-mean-hue').textContent = '--';
}

// ============================================
// Initialize
// ============================================

console.log('MusangKing Hybrid System');
console.log('Varieties: Musang King, Black Thorn, Udang Merah');
console.log('Ripeness: Mature, Immature, Defective');
