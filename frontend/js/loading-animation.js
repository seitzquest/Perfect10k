/**
 * Loading Animation System for Perfect10k
 * Provides visual feedback during semantic data generation
 */

class LoadingAnimationManager {
    constructor() {
        this.currentAnimation = null;
        this.loadingContainer = null;
        this.createLoadingContainer();
    }

    createLoadingContainer() {
        // Create loading overlay
        this.loadingContainer = document.createElement('div');
        this.loadingContainer.id = 'semantic-loading-overlay';
        this.loadingContainer.className = 'loading-overlay hidden';
        
        this.loadingContainer.innerHTML = `
            <div class="loading-content">
                <div class="loading-header">
                    <div class="loading-spinner">
                        <div class="flowing-dot"></div>
                        <div class="flowing-dot"></div>
                        <div class="flowing-dot"></div>
                        <div class="flowing-dot"></div>
                    </div>
                    <div class="loading-text">
                        <h4 id="loading-title">Initializing Route Planning</h4>
                        <p id="loading-description">Preparing your personalized route...</p>
                    </div>
                </div>
                <div class="loading-progress">
                    <div class="progress-bar">
                        <div id="progress-fill" class="progress-fill"></div>
                    </div>
                    <span id="progress-text">0%</span>
                </div>
                <div class="loading-phases">
                    <div class="phase" id="phase-graph-loading">
                        <span class="phase-icon">🗺️</span>
                        <span class="phase-text">Loading street network</span>
                        <span class="phase-status">⏳</span>
                    </div>
                    <div class="phase" id="phase-route-initialization">
                        <span class="phase-icon">🎯</span>
                        <span class="phase-text">Initializing route parameters</span>
                        <span class="phase-status">⏳</span>
                    </div>
                    <div class="phase" id="phase-semantic-analysis">
                        <span class="phase-icon">🌿</span>
                        <span class="phase-text">Analyzing natural features</span>
                        <span class="phase-status">⏳</span>
                    </div>
                </div>
                <div class="loading-details">
                    <small id="loading-details-text">This may take a few moments for new areas...</small>
                </div>
            </div>
        `;
        
        document.body.appendChild(this.loadingContainer);
        this.addLoadingStyles();
    }

    addLoadingStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .loading-overlay {
                position: fixed;
                top: 20px;
                right: 20px;
                width: 280px;
                z-index: 10000;
                pointer-events: auto;
            }
            
            .loading-overlay.hidden {
                display: none;
            }
            
            .loading-content {
                background: white;
                padding: 14px 16px;
                border-radius: 10px;
                box-shadow: 0 3px 12px rgba(0,0,0,0.12);
                border: 1px solid rgba(0,0,0,0.08);
                backdrop-filter: blur(10px);
            }
            
            .loading-header {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 6px;
            }
            
            
            .loading-spinner {
                position: relative;
                width: 52px;
                height: 18px;
                flex-shrink: 0;
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 6px;
            }
            
            .flowing-dot {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: var(--primary, #626F47);
                animation: simple-blink 1.5s ease-in-out infinite;
            }
            
            .flowing-dot:nth-child(1) {
                animation-delay: 0s;
            }
            
            .flowing-dot:nth-child(2) {
                animation-delay: 0.25s;
            }
            
            .flowing-dot:nth-child(3) {
                animation-delay: 0.5s;
            }
            
            .flowing-dot:nth-child(4) {
                animation-delay: 0.75s;
            }
            
            @keyframes simple-blink {
                0%, 80%, 100% { 
                    opacity: 0.3;
                }
                40% { 
                    opacity: 1.0;
                }
            }
            
            .loading-text {
                flex: 1;
                text-align: left;
            }
            
            .loading-text h4 {
                margin: 0 0 1px 0;
                color: #333;
                font-size: 0.9em;
                font-weight: 600;
            }
            
            .loading-text p {
                margin: 0;
                color: #666;
                font-size: 0.7em;
                line-height: 1.2;
            }
            
            .loading-progress {
                margin: 10px 0;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .progress-bar {
                flex: 1;
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
                overflow: hidden;
            }
            
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #2196F3, #4CAF50);
                border-radius: 3px;
                width: 0%;
                transition: width 0.3s ease;
            }
            
            #progress-text {
                font-size: 0.7em;
                color: #666;
                min-width: 30px;
            }
            
            .loading-phases {
                margin: 15px 0 10px 0;
            }
            
            .phase {
                display: flex;
                align-items: center;
                padding: 4px 0;
                gap: 8px;
                font-size: 0.75em;
            }
            
            .phase-icon {
                font-size: 1em;
                width: 16px;
                text-align: center;
            }
            
            .phase-text {
                flex: 1;
                color: #333;
            }
            
            .phase-status {
                font-size: 0.9em;
            }
            
            .phase.completed .phase-status {
                color: #4CAF50;
            }
            
            .phase.completed .phase-text {
                color: #666;
                text-decoration: line-through;
            }
            
            .loading-details {
                margin-top: 10px;
                padding-top: 10px;
                border-top: 1px solid #eee;
            }
            
            .loading-details small {
                color: #888;
                font-size: 0.7em;
                line-height: 1.2;
            }
        `;
        document.head.appendChild(style);
    }

    showLoading(options = {}) {
        const {
            title = "Generating Route Candidates",
            description = "Analyzing natural features for your perfect route...",
            showDetails = true
        } = options;

        // Update text
        document.getElementById('loading-title').textContent = title;
        document.getElementById('loading-description').textContent = description;
        
        if (showDetails) {
            document.getElementById('loading-details-text').textContent = 
                "This may take a few moments for new areas...";
        }

        // Reset phases
        this.resetPhases();
        
        // Show overlay
        this.loadingContainer.classList.remove('hidden');
        
        
        // Start progress animation
        this.animateProgress(0);
    }

    hideLoading() {
        this.loadingContainer.classList.add('hidden');
        if (this.currentAnimation) {
            clearInterval(this.currentAnimation);
            this.currentAnimation = null;
        }
        
        // Restore sections for next time (in case they were hidden for minimalistic view)
        this.restoreFullLoadingView();
    }
    
    restoreFullLoadingView() {
        // Show all sections again for future non-minimalistic loading
        const progressSection = this.loadingContainer.querySelector('.loading-progress');
        const phasesSection = this.loadingContainer.querySelector('.loading-phases');
        const detailsSection = this.loadingContainer.querySelector('.loading-details');
        
        if (progressSection) progressSection.style.display = '';
        if (phasesSection) phasesSection.style.display = '';
        if (detailsSection) detailsSection.style.display = '';
    }

    removeBackgroundNotification() {
        // Stub method for API compatibility - no background notifications in current design
        // This prevents errors when API client tries to call this method
    }

    updateProgress(phases = [], totalDuration = 0) {
        const phaseMapping = {
            'graph_loading': 'phase-graph-loading',
            'route_initialization': 'phase-route-initialization', 
            'semantic_analysis': 'phase-semantic-analysis'
        };

        let completedPhases = 0;
        
        phases.forEach(phase => {
            const elementId = phaseMapping[phase.phase];
            if (elementId) {
                const element = document.getElementById(elementId);
                if (element) {
                    const statusEl = element.querySelector('.phase-status');
                    const textEl = element.querySelector('.phase-text');
                    
                    if (phase.completed) {
                        element.classList.add('completed');
                        statusEl.textContent = '✅';
                        
                        if (phase.duration_ms) {
                            textEl.textContent += ` (${phase.duration_ms}ms)`;
                        }
                        
                        if (phase.was_cached) {
                            textEl.textContent += ' - cached';
                        }
                        
                        completedPhases++;
                    } else {
                        statusEl.textContent = '⏳';
                    }
                }
            }
        });

        // Update progress bar
        const progress = (completedPhases / 3) * 100;
        this.animateProgress(progress);
        
        // Update details if we have timing info
        if (totalDuration > 0) {
            const detailsEl = document.getElementById('loading-details-text');
            if (totalDuration < 1000) {
                detailsEl.textContent = `Completed in ${totalDuration}ms (lightning fast!)`;
            } else {
                detailsEl.textContent = `Completed in ${(totalDuration/1000).toFixed(1)}s`;
            }
        }
    }
    
    updateJobProgress(jobStatus) {
        if (!jobStatus) return;
        
        // Update progress bar based on job progress
        if (jobStatus.progress !== undefined) {
            this.animateProgress(jobStatus.progress);
        }
        
        // Update current phase text
        if (jobStatus.current_phase) {
            const descriptionEl = document.getElementById('loading-description');
            if (descriptionEl) {
                descriptionEl.textContent = jobStatus.current_phase;
            }
        }
        
        // Update phases if provided
        if (jobStatus.phases && Array.isArray(jobStatus.phases)) {
            this.updateProgress(jobStatus.phases);
        }
        
        // Update details text with job status
        const detailsEl = document.getElementById('loading-details-text');
        if (detailsEl) {
            if (jobStatus.status === 'running') {
                detailsEl.textContent = jobStatus.current_phase || 'Processing in background...';
            } else if (jobStatus.status === 'pending') {
                detailsEl.textContent = 'Job queued, starting soon...';
            } else if (jobStatus.status === 'completed') {
                detailsEl.textContent = 'Completed successfully!';
            } else if (jobStatus.status === 'failed') {
                detailsEl.textContent = `Error: ${jobStatus.error || 'Job failed'}`;
            }
        }
    }
    
    showAsyncJobProgress(jobId) {
        // Show minimalistic loading for async jobs
        this.showMinimalisticLoading();
    }
    
    showMinimalisticLoading() {
        // Show only the header with minimal content
        this.loadingContainer.classList.remove('hidden');
        
        // Hide progress bar, phases, and details for minimalistic view
        const progressSection = this.loadingContainer.querySelector('.loading-progress');
        const phasesSection = this.loadingContainer.querySelector('.loading-phases');
        const detailsSection = this.loadingContainer.querySelector('.loading-details');
        
        if (progressSection) progressSection.style.display = 'none';
        if (phasesSection) phasesSection.style.display = 'none';
        if (detailsSection) detailsSection.style.display = 'none';
        
        // Update header content to be minimal
        const titleEl = document.getElementById('loading-title');
        const descriptionEl = document.getElementById('loading-description');
        
        if (titleEl) titleEl.textContent = "Processing Route";
        if (descriptionEl) descriptionEl.textContent = "Loading street network";
        
    }

    resetPhases() {
        ['phase-graph-loading', 'phase-route-initialization', 'phase-semantic-analysis'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.classList.remove('completed');
                element.querySelector('.phase-status').textContent = '⏳';
                const textEl = element.querySelector('.phase-text');
                // Reset text to original
                const originalTexts = {
                    'phase-graph-loading': 'Loading street network',
                    'phase-route-initialization': 'Initializing route parameters',
                    'phase-semantic-analysis': 'Analyzing natural features'
                };
                textEl.textContent = originalTexts[id];
            }
        });
    }

    animateProgress(targetPercent) {
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        
        if (progressFill && progressText) {
            progressFill.style.width = targetPercent + '%';
            progressText.textContent = Math.round(targetPercent) + '%';
        }
    }

    // Integration with the existing API
    async makeApiRequestWithLoading(url, options = {}) {
        const { 
            method = 'POST', 
            body = null,
            showLoading = true,
            loadingOptions = {}
        } = options;

        if (showLoading) {
            this.showLoading(loadingOptions);
        }

        try {
            const response = await fetch(url, {
                method,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                body: body ? JSON.stringify(body) : null
            });

            const data = await response.json();

            // Update loading progress if we have phase information
            if (data.loading_phases) {
                this.updateProgress(data.loading_phases, data.performance?.total_duration_ms);
                
                // Keep loading visible briefly to show completion
                setTimeout(() => {
                    if (showLoading) {
                        this.hideLoading();
                    }
                }, 1000);
            } else {
                if (showLoading) {
                    this.hideLoading();
                }
            }

            return data;
        } catch (error) {
            if (showLoading) {
                this.hideLoading();
                this.removeBackgroundNotification();
            }
            throw error;
        }
    }
}

// Global instance
window.loadingManager = new LoadingAnimationManager();

// Example usage for start-session API call:
/*
async function startRouteWithLoading(lat, lon, preference = "scenic nature") {
    try {
        const result = await window.loadingManager.makeApiRequestWithLoading('/api/start-session', {
            method: 'POST',
            body: {
                lat: lat,
                lon: lon,
                preference: preference,
                target_distance: 8000
            },
            loadingOptions: {
                title: "Finding Perfect Route",
                description: "Analyzing natural features and generating personalized candidates..."
            }
        });
        
        console.log('Route started:', result);
        // Handle the result (update map, show candidates, etc.)
        
    } catch (error) {
        console.error('Failed to start route:', error);
        // Handle error
    }
}
*/