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
        description: ''
    },
    generatedAssets: [],
    currentSection: 'brand-upload'
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
        const previewsContainer = document.getElementById('reference-previews');

        if (!dropZone || !fileInput || !previewsContainer) return;

        dropZone.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', async (e) => {
            const files = Array.from(e.target.files);
            await handleReferenceFiles(files);
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
            const files = Array.from(e.dataTransfer.files);
            await handleReferenceFiles(files);
        });

        async function handleReferenceFiles(files) {
            clearError('reference-error');

            const currentCount = appState.brandData.references.length;
            const newCount = currentCount + files.length;
            
            if (newCount > 5) {
                showError('reference-error', 'Maximum 5 reference images allowed');
                return;
            }

            const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];

            for (const file of files) {
                if (!validateFileType(file, allowedTypes)) {
                    showError('reference-error', `${file.name}: Invalid file type. Use PNG or JPG.`);
                    continue;
                }

                if (!validateFileSize(file, 5)) {
                    showError('reference-error', `${file.name}: File too large (max 5MB)`);
                    continue;
                }

                try {
                    const base64 = await fileToBase64(file);
                    const referenceId = Date.now() + Math.random();
                    
                    appState.brandData.references.push({
                        id: referenceId,
                        name: file.name,
                        data: base64
                    });

                    addReferencePreview(referenceId, base64);
                } catch (error) {
                    showError('reference-error', `Failed to process ${file.name}`);
                    console.error('Reference upload error:', error);
                }
            }
        }

        function addReferencePreview(id, imageData) {
            const item = document.createElement('div');
            item.className = 'reference-item';
            item.dataset.id = id;
            
            item.innerHTML = `
                <img src="${imageData}" alt="Reference">
                <button class="remove-btn" data-id="${id}">×</button>
            `;

            const removeBtn = item.querySelector('.remove-btn');
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                removeReference(id);
            });

            previewsContainer.appendChild(item);
        }

        function removeReference(id) {
            appState.brandData.references = appState.brandData.references.filter(
                ref => ref.id !== id
            );
            
            const item = previewsContainer.querySelector(`[data-id="${id}"]`);
            if (item) {
                item.remove();
            }
            
            clearError('reference-error');
        }
    } catch (error) {
        console.error('❌ Error in initReferenceUpload:', error);
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
            const hasLogo = appState.brandData.logo !== null;
            processBtn.disabled = !hasLogo;
        }
    } catch (error) {
        console.error('❌ Error in validateBrandForm:', error);
    }
}

function initProcessBrand() {
    try {
        const processBtn = document.getElementById('process-brand-btn');
        if (!processBtn) return;

        processBtn.addEventListener('click', async () => {
            try {
                console.log('🔄 Processing brand assets...');
                setButtonLoading('process-brand-btn', true);

                await new Promise(resolve => setTimeout(resolve, 1500));
                
                updateBrandSummary();
                
                console.log('📱 Switching to generation section...');
                showSection('generation');
                console.log('✓ Section switched successfully');
                
            } catch (error) {
                console.error('❌ Brand processing error:', error);
                alert('Failed to process brand assets: ' + error.message);
            } finally {
                setButtonLoading('process-brand-btn', false);
            }
        });
    } catch (error) {
        console.error('❌ Error in initProcessBrand:', error);
    }
}

function updateBrandSummary() {
    try {
        console.log('📋 Updating brand summary...');
        
        const activeLogo = document.getElementById('active-logo');
        const activePrimary = document.getElementById('active-primary');
        const activeSecondary = document.getElementById('active-secondary');
        const activeAccent = document.getElementById('active-accent');
        
        if (appState.brandData.logo && activeLogo) {
            activeLogo.src = appState.brandData.logo.data;
        }
        
        if (activePrimary) {
            activePrimary.style.backgroundColor = appState.brandData.colors.primary;
        }
        
        if (activeSecondary) {
            activeSecondary.style.backgroundColor = appState.brandData.colors.secondary;
        }
        
        if (activeAccent) {
            activeAccent.style.backgroundColor = appState.brandData.colors.accent;
        }
        
        console.log('✓ Brand summary updated');
    } catch (error) {
        console.error('❌ Brand summary error:', error);
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
        updateProgress(20, 'Processing brand identity...');
        await new Promise(resolve => setTimeout(resolve, 1000));

        updateProgress(50, 'Generating brand-consistent visuals...');
        await new Promise(resolve => setTimeout(resolve, 2000));

        updateProgress(75, 'Optimizing layouts...');
        await new Promise(resolve => setTimeout(resolve, 1000));

        updateProgress(100, 'Complete!');
        await new Promise(resolve => setTimeout(resolve, 500));

        const placeholderImage = appState.brandData.logo?.data || createPlaceholderImage();
        
        appState.generatedAssets = Array.from({ length: config.frameCount }, (_, i) => ({
            id: Date.now() + i,
            type: 'image',
            url: placeholderImage
        }));
        
        console.log('✓ Generated assets:', appState.generatedAssets.length);
    } catch (error) {
        console.error('❌ Error in simulateGeneration:', error);
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
                    console.log('📤 Mock: Adding to document...');
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    alert('Assets would be added to document (document sandbox not connected yet)');
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

        if (appState.generatedAssets.length === 0) {
            resultsGrid.innerHTML = '<p style="color: #666; padding: 2rem; text-align: center;">No assets generated.</p>';
            return;
        }

        appState.generatedAssets.forEach((asset, index) => {
            const item = document.createElement('div');
            item.className = 'result-item';
            item.innerHTML = `<img src="${asset.url}" alt="Generated content ${index + 1}" style="width: 100%; height: auto; display: block;">`;
            resultsGrid.appendChild(item);
        });
        
        console.log(`✓ Displayed ${appState.generatedAssets.length} results`);
    } catch (error) {
        console.error('❌ Error in displayResults:', error);
    }
}

function init() {
    try {
        console.log('🚀 Brand AI Generator - Initializing...');

        initLogoUpload();
        initColorPickers();
        initFontSelector();
        initReferenceUpload();
        initBrandDescription();
        initProcessBrand();
        initGenerationInterface();
        initGeneration();
        initResults();

        showSection('brand-upload');

        console.log('✅ Brand AI Generator initialized successfully');
    } catch (error) {
        console.error('❌ FATAL: Error in init():', error);
    }
}

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
