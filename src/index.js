import addOnUISdk from "https://new.express.adobe.com/static/add-on-sdk/sdk.js";

const appState = {
    brandData: {
        logo: null,
        colors: {
            primary: '#000000',
            secondary: '#FFFFFF',
            accent: '#FF0000'
        },
        font: 'Arial',
        references: [],
        description: '',
        productImage: null // For poster generation product_launch mode
    },
    generatedAssets: [],
    generatedPoster: null, // Store poster generation result
    currentSection: 'brand-upload',
    generationMode: 'standard' // 'standard' or 'poster'
};

function showSection(sectionId) {
    try {
        console.log(`🔄 Showing section: ${sectionId}`);
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });
        const targetSection = document.getElementById(`${sectionId}-section`);
        if (targetSection) {
            targetSection.classList.add('active');
            appState.currentSection = sectionId;
            console.log(`✓ Section ${sectionId} is now visible`);
        } else {
            console.error(`❌ Section ${sectionId}-section not found!`);
        }
    } catch (error) {
        console.error('❌ Error in showSection:', error);
    }
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(e);
        reader.readAsDataURL(file);
    });
}

function validateFileSize(file, maxSizeMB = 5) {
    const maxSize = maxSizeMB * 1024 * 1024;
    return file.size <= maxSize;
}

function validateFileType(file, allowedTypes) {
    return allowedTypes.includes(file.type);
}

function showError(elementId, message) {
    try {
        const errorElement = document.getElementById(elementId);
        if (errorElement) {
            errorElement.textContent = message;
        }
    } catch (error) {
        console.error('Error in showError:', error);
    }
}

function clearError(elementId) {
    showError(elementId, '');
}

function setButtonLoading(buttonId, isLoading) {
    try {
        const button = document.getElementById(buttonId);
        if (!button) return;
        
        const textSpan = button.querySelector('.btn-text');
        const loaderSpan = button.querySelector('.btn-loader');
        
        if (isLoading) {
            if (textSpan) textSpan.style.display = 'none';
            if (loaderSpan) loaderSpan.style.display = 'flex';
            button.disabled = true;
        } else {
            if (textSpan) textSpan.style.display = 'block';
            if (loaderSpan) loaderSpan.style.display = 'none';
            button.disabled = false;
        }
    } catch (error) {
        console.error('Error in setButtonLoading:', error);
    }
}

function updateProgress(percentage, message) {
    try {
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        
        if (progressFill) {
            progressFill.style.width = `${percentage}%`;
        }
        if (progressText) {
            progressText.textContent = message;
        }
    } catch (error) {
        console.error('Error in updateProgress:', error);
    }
}

function initLogoUpload() {
    try {
        console.log('🔧 Initializing logo upload...');
        
        const dropZone = document.getElementById('logo-drop-zone');
        const fileInput = document.getElementById('logo-upload');
        const preview = document.getElementById('logo-preview');
        const previewImg = document.getElementById('logo-preview-img');
        const removeBtn = document.getElementById('logo-remove');
        const uploadPrompt = dropZone?.querySelector('.upload-prompt');

        if (!dropZone || !fileInput) {
            console.error('❌ Logo upload elements not found!');
            return;
        }

        dropZone.addEventListener('click', (e) => {
            if (e.target !== removeBtn && !e.target.closest('.remove-btn')) {
                fileInput.click();
            }
        });

        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) {
                await handleLogoFile(file);
            }
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', async (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) {
                await handleLogoFile(file);
            }
        });

        if (removeBtn) {
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                appState.brandData.logo = null;
                preview.style.display = 'none';
                uploadPrompt.style.display = 'block';
                fileInput.value = '';
                clearError('logo-error');
                validateBrandForm();
            });
        }

        async function handleLogoFile(file) {
            try {
                console.log('⚙️ Processing logo file:', file.name);
                clearError('logo-error');

                const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/svg+xml'];
                if (!validateFileType(file, allowedTypes)) {
                    showError('logo-error', 'Please upload a PNG, JPG, or SVG file');
                    return;
                }

                if (!validateFileSize(file, 5)) {
                    showError('logo-error', 'File size must be less than 5MB');
                    return;
                }

                const base64 = await fileToBase64(file);
                
                appState.brandData.logo = {
                    name: file.name,
                    type: file.type,
                    data: base64
                };

                previewImg.src = base64;
                uploadPrompt.style.display = 'none';
                preview.style.display = 'block';

                console.log('✓ Logo uploaded');
                validateBrandForm();
            } catch (error) {
                console.error('❌ Logo upload error:', error);
                showError('logo-error', 'Failed to process logo. Please try again.');
            }
        }
        
        console.log('✓ Logo upload initialized');
    } catch (error) {
        console.error('❌ Error in initLogoUpload:', error);
    }
}

function initColorPickers() {
    try {
        const colorInputs = [
            { colorId: 'primary-color', hexId: 'primary-color-hex', key: 'primary' },
            { colorId: 'secondary-color', hexId: 'secondary-color-hex', key: 'secondary' },
            { colorId: 'accent-color', hexId: 'accent-color-hex', key: 'accent' }
        ];

        colorInputs.forEach(({ colorId, hexId, key }) => {
            const colorInput = document.getElementById(colorId);
            const hexInput = document.getElementById(hexId);

            if (!colorInput || !hexInput) return;

            colorInput.addEventListener('input', (e) => {
                const color = e.target.value;
                hexInput.value = color;
                appState.brandData.colors[key] = color;
            });

            hexInput.addEventListener('input', (e) => {
                let hex = e.target.value;
                if (hex.startsWith('#') && /^#[0-9A-Fa-f]{6}$/.test(hex)) {
                    colorInput.value = hex;
                    appState.brandData.colors[key] = hex;
                }
            });

            hexInput.addEventListener('blur', (e) => {
                let hex = e.target.value;
                if (!hex.startsWith('#')) {
                    hex = '#' + hex;
                }
                if (!/^#[0-9A-Fa-f]{6}$/.test(hex)) {
                    hexInput.value = appState.brandData.colors[key];
                } else {
                    hexInput.value = hex.toUpperCase();
                    colorInput.value = hex;
                    appState.brandData.colors[key] = hex;
                }
            });
        });
    } catch (error) {
        console.error('❌ Error in initColorPickers:', error);
    }
}

function initFontSelector() {
    try {
        const fontSelect = document.getElementById('font-family');
        if (fontSelect) {
            fontSelect.addEventListener('change', (e) => {
                appState.brandData.font = e.target.value;
            });
        }
    } catch (error) {
        console.error('❌ Error in initFontSelector:', error);
    }
}

function initReferenceUpload() {
    try {
        const dropZone = document.getElementById('reference-drop-zone');
        const fileInput = document.getElementById('reference-upload');
        const prompt = document.getElementById('reference-prompt');
        const preview = document.getElementById('reference-preview');
        const previewImg = document.getElementById('reference-preview-img');

        if (!dropZone || !fileInput || !prompt || !preview || !previewImg) {
            console.error('❌ Reference upload elements not found!');
            return;
        }

        dropZone.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) await handleReferenceFile(file);
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', async (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) await handleReferenceFile(file);
        });

        async function handleReferenceFile(file) {
            clearError('reference-error');

            const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
            
            if (!validateFileType(file, allowedTypes)) {
                showError('reference-error', 'Invalid file type. Use PNG or JPG.');
                return;
            }

            if (!validateFileSize(file, 5)) {
                showError('reference-error', 'File too large (max 5MB)');
                return;
            }

            try {
                const base64 = await fileToBase64(file);
                const referenceId = Date.now() + Math.random();
                
                // Store single reference (replace if exists)
                appState.brandData.references = [{
                    id: referenceId,
                    name: file.name,
                    data: base64
                }];

                showReferencePreview(base64);
                updateGenerationModeIndicator();
                validateBrandForm();
                
                console.log('✅ Reference image uploaded');
            } catch (error) {
                showError('reference-error', `Failed to process ${file.name}`);
                console.error('Reference upload error:', error);
            }
        }

        function showReferencePreview(imageData) {
            if (prompt && preview && previewImg) {
                prompt.style.display = 'none';
                preview.style.display = 'block';
                previewImg.src = imageData;
            }
        }

        function removeReference() {
            appState.brandData.references = [];
            
            if (prompt && preview) {
                prompt.style.display = 'block';
                preview.style.display = 'none';
            }
            
            // Clear file input
            fileInput.value = '';
            
            updateGenerationModeIndicator();
            validateBrandForm();
            clearError('reference-error');
        }

        // Remove button handler
        const removeBtn = document.getElementById('reference-remove');
        if (removeBtn) {
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                removeReference();
            });
        }
        
    } catch (error) {
        console.error('❌ Error in initReferenceUpload:', error);
    }
}

function initProductImageUpload() {
    try {
        const dropZone = document.getElementById('product-drop-zone');
        const fileInput = document.getElementById('product-upload');
        const preview = document.getElementById('product-preview');
        const previewImg = document.getElementById('product-preview-img');
        const removeBtn = document.getElementById('product-remove');

        if (!dropZone || !fileInput || !preview || !previewImg || !removeBtn) return;

        dropZone.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (file) await handleProductFile(file);
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', async (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) await handleProductFile(file);
        });

        removeBtn.addEventListener('click', () => {
            removeProductImage();
        });

        async function handleProductFile(file) {
            clearError('product-error');

            const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
            if (!validateFileType(file, allowedTypes)) {
                showError('product-error', 'Invalid file type. Use PNG or JPG.');
                return;
            }

            if (!validateFileSize(file, 5)) {
                showError('product-error', 'File too large (max 5MB)');
                return;
            }

            try {
                const base64 = await fileToBase64(file);
                appState.brandData.productImage = {
                    name: file.name,
                    data: base64
                };

                // Show preview
                previewImg.src = base64;
                preview.style.display = 'block';
                dropZone.querySelector('.upload-prompt').style.display = 'none';

                console.log('✅ Product image uploaded');
            } catch (error) {
                showError('product-error', `Failed to process ${file.name}`);
                console.error('Product upload error:', error);
            }
        }

        function removeProductImage() {
            appState.brandData.productImage = null;
            fileInput.value = '';
            preview.style.display = 'none';
            dropZone.querySelector('.upload-prompt').style.display = 'block';
            clearError('product-error');
        }
    } catch (error) {
        console.error('❌ Error in initProductImageUpload:', error);
    }
}

function initBrandDescription() {
    try {
        const textarea = document.getElementById('brand-description');
        const charCount = document.querySelector('.char-count');

        if (textarea && charCount) {
            textarea.addEventListener('input', (e) => {
                const length = e.target.value.length;
                charCount.textContent = `${length} / 500`;
                appState.brandData.description = e.target.value;
            });
        }
    } catch (error) {
        console.error('❌ Error in initBrandDescription:', error);
    }
}

function validateBrandForm() {
    try {
        const processBtn = document.getElementById('process-brand-btn');
        if (processBtn) {
            // Only require reference image for poster generation
            const hasReference = appState.brandData.references && appState.brandData.references.length > 0;
            processBtn.disabled = !hasReference;
        }
    } catch (error) {
        console.error('❌ Error in validateBrandForm:', error);
    }
}

function initProcessBrand() {
    try {
        const processBtn = document.getElementById('process-brand-btn');
        if (!processBtn) return;
        
        // Initialize Edit Layers button in main UI (always visible)
        const editLayersMainBtn = document.getElementById('edit-layers-main-btn');
        if (editLayersMainBtn) {
            editLayersMainBtn.addEventListener('click', () => {
                if (appState.generationMode === 'poster' && appState.generatedPoster && appState.generatedPoster.layers) {
                    showSection('layer-editor');
                    initLayerCanvas();
                } else {
                    alert('Please generate a poster first to edit layers.');
                }
            });
        }

        processBtn.addEventListener('click', async () => {
            try {
                // Get prompt from first screen
                const promptInput = document.getElementById('generation-prompt');
                const prompt = promptInput ? promptInput.value.trim() : '';
                
                if (!prompt) {
                    showError('prompt-error', 'Please describe what you want to create');
                    if (promptInput) promptInput.focus();
                    return;
                }
                
                clearError('prompt-error');
                console.log('🚀 Starting poster generation...');
                setButtonLoading('process-brand-btn', true);
                
                // Validate reference image
                if (!appState.brandData.references || appState.brandData.references.length === 0) {
                    showError('reference-error', 'Please upload a reference image');
                    setButtonLoading('process-brand-btn', false);
                    return;
                }
                
                // Update generation mode
                appState.generationMode = 'poster';
                
                // Start generation directly
                const generationConfig = {
                    prompt: prompt,
                    brandData: appState.brandData,
                    outputType: 'static',
                    frameCount: 1
                };
                
                // Show progress
                const progressContainer = document.getElementById('generation-progress');
                if (progressContainer) progressContainer.style.display = 'block';
                
                // Generate poster
                await simulateGeneration(generationConfig);
                
                // Show results
                showSection('results');
                displayResults();
                
                console.log('✅ Poster generation complete');
                
            } catch (error) {
                console.error('❌ Generation error:', error);
                alert('Generation failed: ' + error.message);
            } finally {
                setButtonLoading('process-brand-btn', false);
                const progressContainer = document.getElementById('generation-progress');
                if (progressContainer) progressContainer.style.display = 'none';
                updateProgress(0, 'Initializing...');
            }
        });
    } catch (error) {
        console.error('❌ Error in initProcessBrand:', error);
    }
}

function updateBrandSummary() {
    try {
        console.log('📋 Updating brand summary...');
        
        // Update generation mode indicator
        updateGenerationModeIndicator();
        
        console.log('✓ Brand summary updated');
    } catch (error) {
        console.error('❌ Brand summary error:', error);
    }
}

function updateGenerationModeIndicator() {
    try {
        const modeIndicator = document.getElementById('generation-mode-indicator');
        const modeTitle = document.getElementById('mode-title');
        const modeDescription = document.getElementById('mode-description');
        
        if (!modeIndicator || !modeTitle || !modeDescription) return;
        
        const hasReferences = appState.brandData.references && appState.brandData.references.length > 0;
        
        if (hasReferences) {
            const hasProductImage = appState.brandData.productImage !== null;
            
            modeIndicator.style.display = 'block';
            modeTitle.textContent = 'Poster Generation Mode';
            modeDescription.textContent = 
                hasProductImage 
                    ? 'BrandDiffusion will create a marketing poster with background, subject, and your product image as the hero product.'
                    : 'BrandDiffusion will generate a complete marketing poster with background, subject, and AI-generated hero product.';
        } else {
            modeIndicator.style.display = 'none';
        }
    } catch (error) {
        console.error('❌ Error updating mode indicator:', error);
    }
}

function initGenerationInterface() {
    try {
        console.log('🎨 Initializing generation interface...');
        
        const backBtn = document.getElementById('back-to-brand');
        const outputTypeRadios = document.querySelectorAll('input[name="output-type"]');
        const reelOptions = document.getElementById('reel-options');
        const audioOptions = document.getElementById('audio-options');
        const frameCountSlider = document.getElementById('frame-count');
        const sliderValue = document.querySelector('.slider-value');
        const includeAudioCheckbox = document.getElementById('include-audio');
        const audioConfig = document.getElementById('audio-config');

        if (backBtn) {
            backBtn.addEventListener('click', () => {
                showSection('brand-upload');
            });
        }

        if (outputTypeRadios) {
            outputTypeRadios.forEach(radio => {
                radio.addEventListener('change', (e) => {
                    const isReel = e.target.value === 'reel';
                    if (reelOptions) reelOptions.style.display = isReel ? 'block' : 'none';
                    if (audioOptions) audioOptions.style.display = isReel ? 'block' : 'none';
                });
            });
        }

        if (frameCountSlider && sliderValue) {
            frameCountSlider.addEventListener('input', (e) => {
                sliderValue.textContent = `${e.target.value} frames`;
            });
        }

        if (includeAudioCheckbox && audioConfig) {
            includeAudioCheckbox.addEventListener('change', (e) => {
                audioConfig.style.display = e.target.checked ? 'block' : 'none';
            });
        }
        
        console.log('✓ Generation interface initialized');
    } catch (error) {
        console.error('❌ Error in initGenerationInterface:', error);
    }
}

function initGeneration() {
    try {
        console.log('⚡ Initializing generation...');
        
        const generateBtn = document.getElementById('generate-btn');
        const promptInput = document.getElementById('generation-prompt');
        const progressContainer = document.getElementById('generation-progress');

        if (!generateBtn || !promptInput) {
            console.warn('Generate button or prompt input not found');
            return;
        }
        
        // Edit Layers button in main UI - always visible, works after generation
        const editLayersMainBtn = document.getElementById('edit-layers-main-btn');
        if (editLayersMainBtn) {
            // Button is always visible in the main UI (brand-upload section)
            editLayersMainBtn.addEventListener('click', () => {
                if (appState.generationMode === 'poster' && appState.generatedPoster && appState.generatedPoster.layers) {
                    showSection('layer-editor');
                    initLayerCanvas();
                } else {
                    alert('Please generate a poster first to edit layers.');
                }
            });
        }

        generateBtn.addEventListener('click', async () => {
            try {
                const prompt = promptInput.value.trim();

                if (!prompt) {
                    showError('prompt-error', 'Please describe what you want to create');
                    return;
                }

                clearError('prompt-error');
                setButtonLoading('generate-btn', true);
                if (progressContainer) progressContainer.style.display = 'block';

                const outputType = document.querySelector('input[name="output-type"]:checked')?.value || 'static';
                const frameCount = parseInt(document.getElementById('frame-count')?.value || '1');

                const generationConfig = {
                    prompt,
                    brandData: appState.brandData,
                    outputType,
                    frameCount: outputType === 'reel' ? frameCount : 1
                };

                await simulateGeneration(generationConfig);
                
                console.log('🎬 Generation complete, showing results...');
                showSection('results');
                displayResults();

            } catch (error) {
                console.error('❌ Generation error:', error);
                alert('Generation failed: ' + error.message);
            } finally {
                setButtonLoading('generate-btn', false);
                if (progressContainer) progressContainer.style.display = 'none';
                updateProgress(0, 'Initializing...');
            }
        });
        
        console.log('✓ Generation initialized');
    } catch (error) {
        console.error('❌ Error in initGeneration:', error);
    }
}

async function simulateGeneration(config) {
    try {
        // Check if we have reference images - use poster generation if available
        const hasReferences = config.brandData.references && config.brandData.references.length > 0;
        
        if (hasReferences) {
            // Use BrandDiffusion poster generation
            appState.generationMode = 'poster';
            return await generatePoster(config);
        } else {
            // Use standard generation
            appState.generationMode = 'standard';
            return await generateStandardAssets(config);
        }
    } catch (error) {
        console.error('❌ Generation error:', error);
        throw error;
    }
}

async function generatePoster(config) {
    try {
        updateProgress(10, 'Initializing poster generation...');
        console.log('🎨 Generating poster with BrandDiffusion...');
        
        // Prepare reference images
        const referenceImages = config.brandData.references.map(ref => ref.data);
        
        // Prepare request body
        const requestBody = {
            prompt: config.prompt,
            referenceImages: referenceImages
        };
        
        // Add logo if available
        if (config.brandData.logo && config.brandData.logo.data) {
            requestBody.logo = config.brandData.logo.data;
        }
        
        // Add product image if available (for product_launch mode)
        if (config.brandData.productImage && config.brandData.productImage.data) {
            requestBody.productImage = config.brandData.productImage.data;
        }
        
        updateProgress(20, 'Connecting to BrandDiffusion backend...');
        
        // Call poster generation endpoint
        const response = await fetch('http://localhost:8000/api/generate-poster', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || error.message || `Backend error: ${response.status}`);
        }
        
        updateProgress(40, 'Analyzing references and generating assets...');
        
        const data = await response.json();
        console.log('✅ Received poster response from backend:', data);
        
        if (!data.success) {
            throw new Error(data.message || 'Poster generation failed');
        }
        
        updateProgress(80, 'Compositing final poster...');
        
        // Store poster data
        appState.generatedPoster = {
            composite: data.composite,
            layers: data.layers || {},
            metadata: data.metadata || {}
        };
        
        // Also store composite as an asset for display
        if (data.composite) {
            appState.generatedAssets = [{
                id: 'poster_composite',
                type: 'image',
                url: data.composite,
                prompt: config.prompt
            }];
        }
        
        updateProgress(100, 'Poster generated successfully!');
        await new Promise(resolve => setTimeout(resolve, 500));
        
        console.log('✅ Poster generated successfully');
        console.log('   Layers:', Object.keys(appState.generatedPoster.layers));
        console.log('   Metadata:', appState.generatedPoster.metadata);
        
    } catch (error) {
        console.error('❌ Poster generation error:', error);
        throw error;
    }
}

async function generateStandardAssets(config) {
    try {
        updateProgress(10, 'Connecting to AI backend...');
        console.log('📤 Sending generation request to backend...');
        
        // Call Python backend (original endpoint)
        const response = await fetch('http://localhost:8000/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: config.prompt,
                brandData: config.brandData,
                outputType: config.outputType,
                frameCount: config.frameCount,
                audio: config.audio
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `Backend error: ${response.status}`);
        }
        
        updateProgress(40, 'Processing brand identity...');
        
        const data = await response.json();
        console.log('✅ Received response from backend:', data);
        
        if (!data.success) {
            throw new Error(data.message || 'Generation failed');
        }
        
        updateProgress(80, 'Finalizing assets...');
        
        // Store generated assets
        appState.generatedAssets = data.assets;
        appState.generatedPoster = null; // Clear poster data
        
        updateProgress(100, 'Complete!');
        await new Promise(resolve => setTimeout(resolve, 500));
        
        console.log(`✅ Generated ${appState.generatedAssets.length} assets`);
        
    } catch (error) {
        console.error('❌ Generation error:', error);
        throw error;
    }
}


function createPlaceholderImage() {
    try {
        const canvas = document.createElement('canvas');
        canvas.width = 800;
        canvas.height = 600;
        const ctx = canvas.getContext('2d');
        
        const gradient = ctx.createLinearGradient(0, 0, 800, 600);
        gradient.addColorStop(0, appState.brandData.colors.primary);
        gradient.addColorStop(1, appState.brandData.colors.accent);
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 800, 600);
        
        ctx.fillStyle = appState.brandData.colors.secondary;
        ctx.font = 'bold 48px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Brand Content', 400, 300);
        
        return canvas.toDataURL('image/png');
    } catch (error) {
        console.error('Error creating placeholder:', error);
        return '';
    }
}

function initResults() {
    try {
        console.log('📊 Initializing results...');
        
        const addToDocBtn = document.getElementById('add-to-document-btn');
        const generateAgainBtn = document.getElementById('generate-again-btn');

        if (addToDocBtn) {
            addToDocBtn.addEventListener('click', async () => {
                setButtonLoading('add-to-document-btn', true);
                
                try {
                    console.log('📤 Adding to document...');
                    
                    // Check if we have poster or standard assets
                    if (appState.generationMode === 'poster' && appState.generatedPoster && appState.generatedPoster.composite) {
                        // Add poster composite to document
                        await addOnUISdk.instance.emit("addAssetsToDocument", {
                            assets: [{
                                id: 'poster_composite',
                                type: 'image',
                                url: appState.generatedPoster.composite,
                                prompt: appState.generatedPoster.metadata?.generated_copy?.headline || 'Generated Poster'
                            }]
                        });
                        console.log('✅ Poster added to document');
                    } else if (appState.generatedAssets && appState.generatedAssets.length > 0) {
                        // Add standard assets to document
                        await addOnUISdk.instance.emit("addAssetsToDocument", {
                            assets: appState.generatedAssets
                        });
                        console.log(`✅ Added ${appState.generatedAssets.length} asset(s) to document`);
                    } else {
                        alert('No assets to add. Please generate content first.');
                    }
                } catch (error) {
                    console.error('Add to document error:', error);
                    alert('Failed to add assets: ' + error.message);
                } finally {
                    setButtonLoading('add-to-document-btn', false);
                }
            });
        }

        if (generateAgainBtn) {
            generateAgainBtn.addEventListener('click', () => {
                showSection('generation');
            });
        }
        
        // Edit Layers button (in results section)
        const editLayersBtn = document.getElementById('edit-layers-btn');
        if (editLayersBtn) {
            editLayersBtn.addEventListener('click', () => {
                openLayerEditor();
            });
        }
        
        // Edit Layers button (in main UI - generation section)
        const editLayersMainBtn = document.getElementById('edit-layers-main-btn');
        if (editLayersMainBtn) {
            editLayersMainBtn.addEventListener('click', () => {
                openLayerEditor();
            });
        }
        
        // Function to open layer editor
        function openLayerEditor() {
            if (appState.generationMode === 'poster' && appState.generatedPoster && appState.generatedPoster.layers) {
                showSection('layer-editor');
                initLayerCanvas();
            } else {
                alert('Please generate a poster first to edit layers.');
            }
        }
        window.openLayerEditor = openLayerEditor;
        
        console.log('✓ Results initialized');
    } catch (error) {
        console.error('❌ Error in initResults:', error);
    }
}

function displayResults() {
    try {
        console.log('📊 Displaying results...');
        
        const resultsGrid = document.getElementById('results-grid');
        
        if (!resultsGrid) {
            console.error('❌ Results grid not found!');
            return;
        }
        
        resultsGrid.innerHTML = '';

        // Check if we have poster data
        if (appState.generationMode === 'poster' && appState.generatedPoster) {
            displayPosterResults(resultsGrid);
        } else if (appState.generatedAssets.length === 0) {
            resultsGrid.innerHTML = '<p style="color: #666; padding: 2rem; text-align: center;">No assets generated.</p>';
        } else {
            displayStandardResults(resultsGrid);
        }
        
        console.log(`✓ Displayed results (mode: ${appState.generationMode})`);
    } catch (error) {
        console.error('❌ Error in displayResults:', error);
    }
}

function displayPosterResults(resultsGrid) {
    const poster = appState.generatedPoster;
    
    // Show/hide Edit Layers button based on poster availability
    const editLayersBtn = document.getElementById('edit-layers-btn');
    if (editLayersBtn) {
        editLayersBtn.style.display = poster && poster.layers ? 'block' : 'none';
    }
    
    // Display composite
    if (poster.composite) {
        const compositeItem = document.createElement('div');
        compositeItem.className = 'result-item';
        compositeItem.style.marginBottom = '2rem';
        
        const title = document.createElement('h3');
        title.textContent = 'Final Poster';
        title.style.marginBottom = '1rem';
        title.style.color = '#333';
        
        const img = document.createElement('img');
        img.src = poster.composite;
        img.alt = 'Generated poster';
        img.style.width = '100%';
        img.style.height = 'auto';
        img.style.display = 'block';
        img.style.borderRadius = '8px';
        img.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
        
        compositeItem.appendChild(title);
        compositeItem.appendChild(img);
        resultsGrid.appendChild(compositeItem);
    }
    
    // Display metadata if available
    if (poster.metadata && poster.metadata.generated_copy) {
        const copyInfo = document.createElement('div');
        copyInfo.style.marginTop = '1rem';
        copyInfo.style.padding = '1rem';
        copyInfo.style.background = '#f5f5f5';
        copyInfo.style.borderRadius = '8px';
        
        const copy = poster.metadata.generated_copy;
        copyInfo.innerHTML = `
            <h4 style="margin: 0 0 0.5rem 0; color: #333;">Generated Copy</h4>
            <p style="margin: 0.25rem 0; color: #666;"><strong>Headline:</strong> ${copy.headline || 'N/A'}</p>
            <p style="margin: 0.25rem 0; color: #666;"><strong>Subheadline:</strong> ${copy.subheadline || 'N/A'}</p>
            <p style="margin: 0.25rem 0; color: #666;"><strong>CTA:</strong> ${copy.cta || 'N/A'}</p>
        `;
        resultsGrid.appendChild(copyInfo);
    }
    
    // Display layers if available (optional)
    if (poster.layers && Object.keys(poster.layers).length > 0) {
        const layersTitle = document.createElement('h3');
        layersTitle.textContent = 'Layers (for editing)';
        layersTitle.style.marginTop = '2rem';
        layersTitle.style.marginBottom = '1rem';
        layersTitle.style.color = '#333';
        resultsGrid.appendChild(layersTitle);
        
        const layersGrid = document.createElement('div');
        layersGrid.style.display = 'grid';
        layersGrid.style.gridTemplateColumns = 'repeat(auto-fill, minmax(200px, 1fr))';
        layersGrid.style.gap = '1rem';
        
        Object.entries(poster.layers).forEach(([name, layer]) => {
            const layerItem = document.createElement('div');
            layerItem.style.padding = '0.5rem';
            layerItem.style.background = '#fff';
            layerItem.style.borderRadius = '4px';
            layerItem.style.border = '1px solid #ddd';
            
            const layerLabel = document.createElement('div');
            layerLabel.textContent = name;
            layerLabel.style.fontSize = '0.875rem';
            layerLabel.style.fontWeight = 'bold';
            layerLabel.style.marginBottom = '0.5rem';
            layerLabel.style.color = '#666';
            
            const layerImg = document.createElement('img');
            layerImg.src = layer.data || layer;
            layerImg.alt = name;
            layerImg.style.width = '100%';
            layerImg.style.height = 'auto';
            layerImg.style.display = 'block';
            layerImg.style.borderRadius = '4px';
            
            layerItem.appendChild(layerLabel);
            layerItem.appendChild(layerImg);
            layersGrid.appendChild(layerItem);
        });
        
        resultsGrid.appendChild(layersGrid);
    }
}

function displayStandardResults(resultsGrid) {
    appState.generatedAssets.forEach((asset, index) => {
        const item = document.createElement('div');
        item.className = 'result-item';
        item.innerHTML = `<img src="${asset.url}" alt="Generated content ${index + 1}" style="width: 100%; height: auto; display: block;">`;
        resultsGrid.appendChild(item);
    });
}

function init() {
    try {
        console.log('🚀 Brand AI Generator - Initializing...');

        initLogoUpload();
        initReferenceUpload();
        initProductImageUpload(); // Initialize product image upload for poster generation
        initProcessBrand();
        initGenerationInterface();
        initGeneration();
        initResults();
        initLayerEditor();
        
        // Update validation on load
        validateBrandForm();

        showSection('brand-upload');

        console.log('✅ Brand AI Generator initialized successfully');
    } catch (error) {
        console.error('❌ FATAL: Error in init():', error);
    }
}

// ============================================================================
// 🎨 LAYER EDITOR
// ============================================================================

let layerEditorState = {
    layers: {},
    textLayers: [],
    selectedTextLayer: null,
    canvas: null,
    ctx: null
};

function initLayerEditor() {
    try {
        console.log('🎨 Initializing layer editor...');
        
        // Back button
        const backBtn = document.getElementById('back-to-results');
        if (backBtn) {
            backBtn.addEventListener('click', () => {
                showSection('results');
            });
        }
        
        // Text controls
        const textSizeSlider = document.getElementById('text-size');
        const textSizeValue = document.getElementById('text-size-value');
        if (textSizeSlider && textSizeValue) {
            textSizeSlider.addEventListener('input', (e) => {
                textSizeValue.textContent = `${e.target.value}px`;
            });
        }
        
        // Color input sync
        const colorInput = document.getElementById('text-color');
        const hexInput = document.getElementById('text-color-hex');
        if (colorInput && hexInput) {
            colorInput.addEventListener('input', (e) => {
                hexInput.value = e.target.value.toUpperCase();
            });
            hexInput.addEventListener('input', (e) => {
                let hex = e.target.value;
                if (!hex.startsWith('#')) hex = '#' + hex;
                if (/^#[0-9A-Fa-f]{6}$/.test(hex)) {
                    colorInput.value = hex;
                }
            });
        }
        
        // Add text button
        const addTextBtn = document.getElementById('add-text-btn');
        if (addTextBtn) {
            addTextBtn.addEventListener('click', () => {
                addTextLayer();
                // Clear selection after adding
                layerEditorState.selectedTextLayer = null;
                layerEditorState.textLayers.forEach(layer => layer.selected = false);
                updateLayersList();
            });
        }
        
        // Real-time updates for text controls
        const textControls = ['text-content', 'text-font', 'text-size', 'text-color', 'text-x', 'text-y'];
        textControls.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.addEventListener('input', () => {
                    if (layerEditorState.selectedTextLayer !== null) {
                        const layer = layerEditorState.textLayers[layerEditorState.selectedTextLayer];
                        if (layer) {
                            const textContent = document.getElementById('text-content');
                            const textFont = document.getElementById('text-font');
                            const textSize = document.getElementById('text-size');
                            const textColor = document.getElementById('text-color');
                            const textX = document.getElementById('text-x');
                            const textY = document.getElementById('text-y');
                            const textSizeValue = document.getElementById('text-size-value');
                            
                            if (textContent) layer.text = textContent.value.trim();
                            if (textFont) layer.font = textFont.value;
                            if (textSize) {
                                layer.size = parseInt(textSize.value);
                                if (textSizeValue) textSizeValue.textContent = `${layer.size}px`;
                            }
                            if (textColor) layer.color = textColor.value;
                            if (textX) layer.x = parseInt(textX.value);
                            if (textY) layer.y = parseInt(textY.value);
                            
                            renderCanvas();
                            updateLayersList();
                        }
                    }
                });
            }
        });
        
        // Export button
        const exportBtn = document.getElementById('export-canvas-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', exportCanvas);
        }
        
        // Reset button
        const resetBtn = document.getElementById('reset-canvas-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                if (confirm('Reset all changes? This will remove all custom text layers.')) {
                    layerEditorState.textLayers = [];
                    initLayerCanvas();
                }
            });
        }
        
        console.log('✓ Layer editor initialized');
    } catch (error) {
        console.error('❌ Error in initLayerEditor:', error);
    }
}

function initLayerCanvas() {
    try {
        const canvas = document.getElementById('layer-canvas');
        if (!canvas) return;
        
        layerEditorState.canvas = canvas;
        layerEditorState.ctx = canvas.getContext('2d');
        const ctx = layerEditorState.ctx;
        
        // Set canvas size (1280x800 for poster)
        canvas.width = 1280;
        canvas.height = 800;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Load layers from poster data
        const poster = appState.generatedPoster;
        if (!poster || !poster.layers) {
            console.error('No poster layers available');
            return;
        }
        
        layerEditorState.layers = {};
        
        // Load layer images
        const layerOrder = ['background', 'subject', 'hero', 'logo'];
        const layerPromises = [];
        
        layerOrder.forEach(layerName => {
            if (poster.layers[layerName]) {
                const layerData = poster.layers[layerName];
                const img = new Image();
                img.crossOrigin = 'anonymous';
                
                const promise = new Promise((resolve) => {
                    img.onload = () => {
                        layerEditorState.layers[layerName] = {
                            image: img,
                            visible: true,
                            name: layerName
                        };
                        resolve();
                    };
                    img.onerror = () => {
                        console.warn(`Failed to load layer: ${layerName}`);
                        resolve();
                    };
                    img.src = layerData.data || layerData;
                });
                
                layerPromises.push(promise);
            }
        });
        
        // Render after all images load
        Promise.all(layerPromises).then(() => {
            renderCanvas();
            updateLayersList();
        });
        
    } catch (error) {
        console.error('❌ Error in initLayerCanvas:', error);
    }
}

function renderCanvas() {
    const canvas = layerEditorState.canvas;
    const ctx = layerEditorState.ctx;
    
    if (!canvas || !ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw layers in order
    const layerOrder = ['background', 'subject', 'hero', 'logo'];
    layerOrder.forEach(layerName => {
        const layer = layerEditorState.layers[layerName];
        if (layer && layer.visible && layer.image) {
            ctx.drawImage(layer.image, 0, 0, canvas.width, canvas.height);
        }
    });
    
    // Draw text layers
    layerEditorState.textLayers.forEach(textLayer => {
        if (textLayer.visible) {
            ctx.save();
            ctx.font = `${textLayer.size}px ${textLayer.font}`;
            ctx.fillStyle = textLayer.color;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText(textLayer.text, textLayer.x, textLayer.y);
            ctx.restore();
        }
    });
}

function updateLayersList() {
    const layersList = document.getElementById('layers-list');
    if (!layersList) return;
    
    layersList.innerHTML = '';
    
    // Add image layers
    Object.values(layerEditorState.layers).forEach(layer => {
        const item = document.createElement('div');
        item.className = 'layer-item';
        item.innerHTML = `
            <div class="layer-item-info">
                <div class="layer-item-name">${layer.name.charAt(0).toUpperCase() + layer.name.slice(1)}</div>
                <div class="layer-item-type">Image Layer</div>
            </div>
            <button class="layer-item-toggle ${layer.visible ? 'active' : ''}" 
                    data-layer="${layer.name}"></button>
        `;
        
        const toggle = item.querySelector('.layer-item-toggle');
        toggle.addEventListener('click', (e) => {
            e.stopPropagation();
            layer.visible = !layer.visible;
            toggle.classList.toggle('active');
            renderCanvas();
        });
        
        layersList.appendChild(item);
    });
    
    // Add text layers
    layerEditorState.textLayers.forEach((textLayer, index) => {
        const item = document.createElement('div');
        item.className = `text-layer-item ${textLayer.selected ? 'selected' : ''}`;
        item.innerHTML = `
            <div class="text-layer-preview" style="font-family: ${textLayer.font}; font-size: ${textLayer.size}px; color: ${textLayer.color};">
                ${textLayer.text}
            </div>
            <div class="text-layer-actions">
                <button class="btn btn-icon btn-secondary" onclick="editTextLayer(${index})">Edit</button>
                <button class="btn btn-icon btn-secondary" onclick="deleteTextLayer(${index})">Delete</button>
            </div>
        `;
        
        item.addEventListener('click', () => {
            selectTextLayer(index);
        });
        
        layersList.appendChild(item);
    });
}

function addTextLayer() {
    const textContent = document.getElementById('text-content');
    const textFont = document.getElementById('text-font');
    const textSize = document.getElementById('text-size');
    const textColor = document.getElementById('text-color');
    const textX = document.getElementById('text-x');
    const textY = document.getElementById('text-y');
    
    if (!textContent || !textContent.value.trim()) {
        alert('Please enter some text');
        return;
    }
    
    const newTextLayer = {
        text: textContent.value.trim(),
        font: textFont ? textFont.value : 'Arial',
        size: textSize ? parseInt(textSize.value) : 48,
        color: textColor ? textColor.value : '#000000',
        x: textX ? parseInt(textX.value) : 100,
        y: textY ? parseInt(textY.value) : 100,
        visible: true,
        selected: false
    };
    
    layerEditorState.textLayers.push(newTextLayer);
    
    // Clear input
    if (textContent) textContent.value = '';
    
    renderCanvas();
    updateLayersList();
    
    console.log('✅ Text layer added');
}

function selectTextLayer(index) {
    layerEditorState.textLayers.forEach((layer, i) => {
        layer.selected = (i === index);
    });
    
    const selectedLayer = layerEditorState.textLayers[index];
    if (selectedLayer) {
        // Update controls
        const textContent = document.getElementById('text-content');
        const textFont = document.getElementById('text-font');
        const textSize = document.getElementById('text-size');
        const textColor = document.getElementById('text-color');
        const textColorHex = document.getElementById('text-color-hex');
        const textX = document.getElementById('text-x');
        const textY = document.getElementById('text-y');
        const textSizeValue = document.getElementById('text-size-value');
        
        if (textContent) textContent.value = selectedLayer.text;
        if (textFont) textFont.value = selectedLayer.font;
        if (textSize) textSize.value = selectedLayer.size;
        if (textColor) textColor.value = selectedLayer.color;
        if (textColorHex) textColorHex.value = selectedLayer.color.toUpperCase();
        if (textX) textX.value = selectedLayer.x;
        if (textY) textY.value = selectedLayer.y;
        if (textSizeValue) textSizeValue.textContent = `${selectedLayer.size}px`;
    }
    
    layerEditorState.selectedTextLayer = index;
    updateLayersList();
}

function editTextLayer(index) {
    selectTextLayer(index);
    
    // Update text layer when controls change
    const updateLayer = () => {
        const layer = layerEditorState.textLayers[index];
        if (!layer) return;
        
        const textContent = document.getElementById('text-content');
        const textFont = document.getElementById('text-font');
        const textSize = document.getElementById('text-size');
        const textColor = document.getElementById('text-color');
        const textX = document.getElementById('text-x');
        const textY = document.getElementById('text-y');
        const textSizeValue = document.getElementById('text-size-value');
        
        if (textContent) layer.text = textContent.value.trim();
        if (textFont) layer.font = textFont.value;
        if (textSize) {
            layer.size = parseInt(textSize.value);
            if (textSizeValue) textSizeValue.textContent = `${layer.size}px`;
        }
        if (textColor) layer.color = textColor.value;
        if (textX) layer.x = parseInt(textX.value);
        if (textY) layer.y = parseInt(textY.value);
        
        renderCanvas();
        updateLayersList();
    };
    
    // Add event listeners to controls (remove old ones first)
    const controlIds = ['text-content', 'text-font', 'text-size', 'text-color', 'text-x', 'text-y'];
    controlIds.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            // Clone and replace to remove all listeners
            const newEl = el.cloneNode(true);
            el.parentNode.replaceChild(newEl, el);
            newEl.addEventListener('input', updateLayer);
            newEl.addEventListener('change', updateLayer);
        }
    });
}

function deleteTextLayer(index) {
    if (confirm('Delete this text layer?')) {
        layerEditorState.textLayers.splice(index, 1);
        layerEditorState.selectedTextLayer = null;
        renderCanvas();
        updateLayersList();
    }
}

function exportCanvas() {
    const canvas = layerEditorState.canvas;
    if (!canvas) return;
    
    try {
        // Convert canvas to base64
        const dataURL = canvas.toDataURL('image/png');
        
        // Update appState with edited poster
        appState.generatedPoster.composite = dataURL;
        
        // Show success message
        alert('Poster exported! You can now add it to your document.');
        
        // Go back to results
        showSection('results');
        displayResults();
        
        console.log('✅ Canvas exported');
    } catch (error) {
        console.error('❌ Export error:', error);
        alert('Failed to export poster: ' + error.message);
    }
}

// Make functions globally accessible for onclick handlers
window.editTextLayer = editTextLayer;
window.deleteTextLayer = deleteTextLayer;

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        addOnUISdk.ready.then(() => {
            console.log('✓ Adobe Express SDK ready');
            init();
        }).catch(error => {
            console.error('❌ SDK initialization failed:', error);
        });
    });
} else {
    addOnUISdk.ready.then(() => {
        console.log('✓ Adobe Express SDK ready');
        init();
    }).catch(error => {
        console.error('❌ SDK initialization failed:', error);
    });
}
