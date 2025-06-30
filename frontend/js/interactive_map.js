/**
 * Interactive Map Editor for Perfect10k
 * Handles step-by-step interactive route building
 */

class InteractiveMapEditor {
    constructor(containerId) {
        this.containerId = containerId;
        this.map = null;
        this.userLocation = null;
        this.currentRoute = null;
        this.candidates = [];
        this.routeBuilding = false;
        this.sessionId = null;
        this.waypoints = [];
        this.candidateMarkers = [];
        this.routeLayers = [];
        this.userSetZoom = false;
        this.feelingLuckyMode = false;
        this.semanticOverlays = null;
        
        this.initialize();
    }
    
    /**
     * Initialize the map
     */
    initialize() {
        // Initialize the map centered on Cologne, Germany (where we have data)
        this.map = L.map(this.containerId, {
            center: [50.924, 7.004],
            zoom: 13,
            zoomControl: false
        });
        
        // Add colorful CartoDB Voyager style tile layer - modern and vibrant
        L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 20
        }).addTo(this.map);
        
        // Zoom controls disabled - using custom controls instead
        
        // Set up event handlers
        this.setupEventHandlers();
        
        // Track user zoom interactions
        this.map.on('zoomstart', () => {
            this.userSetZoom = true;
        });
        
        // Set up feeling lucky mode listeners
        this.setupFeelingLuckyMode();
        
        // Initialize semantic overlays
        this.initializeSemanticOverlays();
        
        console.log('Interactive map editor initialized');
    }
    
    /**
     * Set up feeling lucky mode
     */
    setupFeelingLuckyMode() {
        // Desktop checkbox
        const feelingLuckyCheckbox = document.getElementById('feelingLucky');
        if (feelingLuckyCheckbox) {
            feelingLuckyCheckbox.addEventListener('change', (e) => {
                this.feelingLuckyMode = e.target.checked;
                
                // Sync with mobile checkbox
                const mobileCheckbox = document.getElementById('feelingLuckyMobile');
                if (mobileCheckbox) {
                    mobileCheckbox.checked = this.feelingLuckyMode;
                }
                
                console.log('Feeling lucky mode:', this.feelingLuckyMode);
            });
        }
        
        // Mobile checkbox
        const feelingLuckyMobileCheckbox = document.getElementById('feelingLuckyMobile');
        if (feelingLuckyMobileCheckbox) {
            feelingLuckyMobileCheckbox.addEventListener('change', (e) => {
                this.feelingLuckyMode = e.target.checked;
                
                // Sync with desktop checkbox
                const desktopCheckbox = document.getElementById('feelingLucky');
                if (desktopCheckbox) {
                    desktopCheckbox.checked = this.feelingLuckyMode;
                }
                
                console.log('Feeling lucky mode:', this.feelingLuckyMode);
            });
        }
    }
    
    /**
     * Initialize semantic overlays system
     */
    initializeSemanticOverlays() {
        try {
            // Initialize the semantic overlays manager
            this.semanticOverlays = new SemanticOverlaysManager(this.map, window.apiClient);
            
            // Set up desktop toggle buttons
            const overlayTypes = ['forests', 'rivers', 'lakes', 'scoring'];
            overlayTypes.forEach(overlayType => {
                const toggleBtn = document.getElementById(`overlay-toggle-${overlayType}`);
                if (toggleBtn) {
                    // Special handling for scoring overlay
                    if (overlayType === 'scoring') {
                        // Initially disable scoring button
                        toggleBtn.disabled = true;
                        toggleBtn.title = 'Start a route first to visualize algorithm scoring';
                        toggleBtn.classList.add('disabled');
                    }
                    
                    toggleBtn.addEventListener('click', () => {
                        // Prevent clicking on disabled scoring button
                        if (overlayType === 'scoring' && toggleBtn.disabled) {
                            this.showMessage('Please start a route first to visualize algorithm scoring', 'info');
                            return;
                        }
                        
                        this.semanticOverlays.toggleOverlay(overlayType);
                    });
                }
                
                // Set up mobile checkboxes
                const checkbox = document.getElementById(`overlay-checkbox-${overlayType}`);
                if (checkbox) {
                    checkbox.addEventListener('change', (e) => {
                        if (e.target.checked) {
                            this.semanticOverlays.showOverlay(overlayType);
                        } else {
                            this.semanticOverlays.hideOverlay(overlayType);
                        }
                        
                        // Sync desktop button state
                        const desktopBtn = document.getElementById(`overlay-toggle-${overlayType}`);
                        if (desktopBtn) {
                            desktopBtn.classList.toggle('active', e.target.checked);
                        }
                    });
                }
            });
            
            // Set up scoring overlay controls
            this.setupScoringOverlayControls();
            
            console.log('Semantic overlays initialized');
            
        } catch (error) {
            console.error('Failed to initialize semantic overlays:', error);
        }
    }
    
    /**
     * Set up scoring overlay specific controls
     */
    setupScoringOverlayControls() {
        // Show/hide scoring controls when overlay is toggled
        const scoringControls = document.getElementById('scoring-controls');
        const scoringTypeSelect = document.getElementById('scoring-type-select');
        
        if (scoringControls) {
            // Monitor scoring overlay state changes
            const checkScoringState = () => {
                if (this.semanticOverlays) {
                    const isVisible = this.semanticOverlays.overlayStates.scoring;
                    scoringControls.style.display = isVisible ? 'block' : 'none';
                }
            };
            
            // Check every 500ms for state changes (simple but effective)
            setInterval(checkScoringState, 500);
        }
        
        // Handle score type changes
        if (scoringTypeSelect) {
            scoringTypeSelect.addEventListener('change', async (e) => {
                const newScoreType = e.target.value;
                if (this.semanticOverlays) {
                    try {
                        await this.semanticOverlays.changeScoringType(newScoreType);
                    } catch (error) {
                        console.error('Failed to change scoring type:', error);
                        this.showMessage('Failed to change scoring type: ' + error.message, 'error');
                    }
                }
            });
        }
    }
    
    /**
     * Enable scoring overlay button when session is active
     */
    enableScoringOverlay() {
        const scoringBtn = document.getElementById('overlay-toggle-scoring');
        if (scoringBtn) {
            scoringBtn.disabled = false;
            scoringBtn.title = 'Visualize algorithm scoring';
            scoringBtn.classList.remove('disabled');
            }
    }
    
    /**
     * Disable scoring overlay button when no session
     */
    disableScoringOverlay() {
        const scoringBtn = document.getElementById('overlay-toggle-scoring');
        if (scoringBtn) {
            // First hide the overlay if it's currently visible
            if (this.semanticOverlays && this.semanticOverlays.overlayStates.scoring) {
                this.semanticOverlays.hideOverlay('scoring');
            }
            
            // Update the overlay state to false
            if (this.semanticOverlays) {
                this.semanticOverlays.overlayStates.scoring = false;
                // Use the semantic overlay's own UI update method
                this.semanticOverlays.updateToggleUI('scoring', false);
            }
            
            // Then disable the button
            scoringBtn.disabled = true;
            scoringBtn.title = 'Start a route first to visualize algorithm scoring';
            scoringBtn.classList.add('disabled');
            
            // Force remove active state visually (in case updateToggleUI didn't work)
            scoringBtn.classList.remove('active');
            scoringBtn.setAttribute('aria-pressed', 'false');
            
            console.log('Scoring overlay disabled - button classes:', scoringBtn.className);
        }
    }
    
    /**
     * Set up map event handlers
     */
    setupEventHandlers() {
        // Click to start or add waypoints
        this.map.on('click', (e) => {
            this.handleMapClick(e.latlng.lat, e.latlng.lng);
        });
        
        // Map controls
        document.getElementById('zoomIn')?.addEventListener('click', () => {
            this.map.zoomIn();
        });
        
        document.getElementById('zoomOut')?.addEventListener('click', () => {
            this.map.zoomOut();
        });
        
        document.getElementById('centerRoute')?.addEventListener('click', () => {
            this.centerOnRoute();
        });
        
        document.getElementById('fullscreen')?.addEventListener('click', () => {
            this.toggleFullscreen();
        });
    }
    
    /**
     * Handle map clicks - select start location if none set or update existing location
     */
    async handleMapClick(lat, lon) {
        // Only allow location changes in initial state or if no route is being built
        if (!this.routeBuilding && window.perfect10kApp && window.perfect10kApp.appState !== 'completed') {
            // Set or update start location by clicking on map
            this.setLocationAndUnblur(lat, lon);
            
            // Notify app that location was set
            window.perfect10kApp.setLocation(lat, lon);
        }
    }
    
    /**
     * Start a new interactive routing session
     */
    async startRoutingSession(lat, lon) {
        try {
            // Show loading animation immediately
            if (window.loadingManager) {
                window.loadingManager.showMinimalisticLoading();
                
                // Update the text to match our specific use case
                const titleEl = document.getElementById('loading-title');
                const descriptionEl = document.getElementById('loading-description');
                if (titleEl) titleEl.textContent = "Finding Perfect Route";
                if (descriptionEl) descriptionEl.textContent = "Analyzing natural features for your perfect route...";
            }
            
            // Get user preferences
            const preferences = document.getElementById('preferencesInput').value || 
                               document.getElementById('preferencesInputMobile')?.value || 
                               "I love scenic nature paths through parks with quiet walkways near water";
            
            const steps = parseInt(document.getElementById('stepsInput').value) || 10000;
            const targetDistance = steps * 0.8; // Convert steps to meters
            
            console.log('Starting session with:', { lat, lon, preferences, targetDistance });
            
            // Start session via API
            const response = await window.apiClient.startSession(lat, lon, preferences, targetDistance);
            
            console.log('Session start response:', response);
            
            // Check if response is valid before accessing properties
            if (!response) {
                throw new Error('No response received from server');
            }
            
            // The new API returns data directly without a "success" wrapper
            if (response.session_id && response.candidates) {
                this.sessionId = response.session_id;
                console.log('Interactive map session ID set to:', this.sessionId);
                console.log('API client session ID:', window.apiClient.currentSession);
                
                // Set session ID for scoring overlay
                if (this.semanticOverlays) {
                    this.semanticOverlays.setCurrentSession(this.sessionId);
                }
                
                // Enable scoring overlay button now that we have a session
                this.enableScoringOverlay();
                this.waypoints = [{lat, lon}];
                
                // Clear existing route display (but keep start marker) - but don't reset routeBuilding
                this.clearRouteLayers();
                
                // Now set route building to true AFTER clearing
                this.routeBuilding = true;
                
                // Show candidates
                this.showCandidates(response.candidates);
                
                // Update UI
                this.updateRouteStats(response.route_stats);
                this.showRouteStats();
                
                // Update mobile UI to show distance instead of location controls
                this.updateMobileDistanceDisplay(response.route_stats.current_distance / 1000, response.route_stats.progress * 100);
                
                // Hide loading animation
                if (window.loadingManager) {
                    window.loadingManager.hideLoading();
                }
                
                // Route planning started - candidates displayed
            } else {
                this.showMessage(`Failed to start session: ${response.message || 'Unknown error'}`, 'error');
                // Hide loading animation on error
                if (window.loadingManager) {
                    window.loadingManager.hideLoading();
                }
            }
            
        } catch (error) {
            console.error('Failed to start routing session:', error);
            this.showMessage(`Failed to start routing: ${error.message}`, 'error');
            // Hide loading animation on error
            if (window.loadingManager) {
                window.loadingManager.hideLoading();
            }
        }
    }
    
    /**
     * Add waypoint to current route
     */
    async addWaypoint(nodeId) {
        try {
            // Immediately close all popups for instant feedback
            this.map.closePopup();
            
            // Show loading animation immediately
            if (window.loadingManager) {
                window.loadingManager.showMinimalisticLoading();
            }
            
            
            const response = await window.apiClient.addWaypoint(nodeId);
            
            if (response.waypoint_added) {
                // Add to waypoints (use waypoint_added data)
                const waypoint = response.waypoint_added;
                this.waypoints.push({
                    lat: waypoint.lat,
                    lon: waypoint.lon,
                    nodeId: waypoint.node_id
                });
                
                // Clear previous candidates
                this.clearCandidates();
                
                // Show updated route path (use actual route coordinates)
                if (response.route_coordinates && response.route_coordinates.length > 0) {
                    this.showRouteProgress(response.route_coordinates);
                } else {
                    // Log warning when route coordinates are missing
                    console.warn('Route coordinates not available in response - route may not follow proper network edges');
                    console.log('Response data:', response);
                    
                    // Don't show route progress without proper coordinates to avoid "air connections"
                    // The route will be shown when proper coordinates are available
                }
                
                // Show new candidates if any
                if (response.candidates && response.candidates.length > 0) {
                    this.showCandidates(response.candidates);
                }
                
                // Update stats
                this.updateRouteStats(response.route_stats);
                this.showRouteStats();
                
            } else {
                this.showMessage(`Failed to add waypoint: ${response.message || 'Invalid response format'}`, 'error');
                console.log('Unexpected response format:', response);
            }
            
        } catch (error) {
            console.error('Failed to add waypoint:', error);
            this.showMessage(`Failed to add waypoint: ${error.message}`, 'error');
        } finally {
            // Hide all loading indicators
            if (window.loadingManager) {
                window.loadingManager.hideLoading();
            }
        }
    }
    
    /**
     * Finalize route with selected destination
     */
    async finalizeRoute(nodeId) {
        try {
            // Immediately close all popups for instant feedback
            this.map.closePopup();
            
            // Show loading animation immediately
            if (window.loadingManager) {
                window.loadingManager.showMinimalisticLoading();
            }
            
            
            const response = await window.apiClient.finalizeRoute(nodeId);
            
            if (response.route_completed) {
                // Clear candidates
                this.clearCandidates();
                
                // Show final route
                this.showFinalRoute({
                    coordinates: response.route_coordinates
                });
                
                // Update final stats
                this.updateFinalRouteStats(response.route_stats);
                
                // End route building mode
                this.routeBuilding = false;
                
                // Set app state to completed
                if (window.perfect10kApp) {
                    window.perfect10kApp.onRouteCompleted();
                }
                
                // Update mobile UI - keep showing distance for completed route
                this.updateMobileDistanceDisplay(response.route_stats.total_distance / 1000, 100);
                
                // Route completed successfully
                
            } else {
                this.showMessage(`Failed to finalize route: ${response.message || 'Invalid response format'}`, 'error');
                console.log('Unexpected finalize response:', response);
            }
            
        } catch (error) {
            console.error('Failed to finalize route:', error);
            this.showMessage(`Failed to finalize route: ${error.message}`, 'error');
        } finally {
            // Hide all loading indicators
            if (window.loadingManager) {
                window.loadingManager.hideLoading();
            }
        }
    }
    
    /**
     * Show candidate destinations on map
     */
    showCandidates(candidates) {
        this.clearCandidates();
        this.candidates = candidates;
        
        // If feeling lucky mode is enabled, skip showing candidates and instantly process
        if (this.feelingLuckyMode && candidates.length > 0) {
            this.processLuckyRoute(candidates);
            return;
        }
        
        candidates.forEach((candidate, index) => {
            const marker = L.marker([candidate.lat, candidate.lon], {
                icon: L.divIcon({
                    className: 'candidate-marker',
                    html: `
                        <div class="candidate-marker-content">
                            <div class="candidate-number">${index + 1}</div>
                            <div class="candidate-distance">${(candidate.distance_from_current / 1000).toFixed(1)}km</div>
                        </div>
                    `,
                    iconSize: [60, 60],
                    iconAnchor: [30, 30]
                })
            }).addTo(this.map);
            
            // Add popup with actions
            const semanticScoresHtml = this.generateSemanticScoresHtml(candidate);
            const popupContent = `
                <div class="candidate-popup">
                    <h4>Candidate ${index + 1}</h4>
                    <p><strong>Distance:</strong> ${(candidate.distance_from_current / 1000).toFixed(1)}km</p>
                    <div class="semantic-scores">
                        <p><strong>Location Features:</strong></p>
                        ${semanticScoresHtml}
                        <p class="semantic-summary"><strong>Why this spot:</strong> ${candidate.explanation || 'Basic walkable area'}</p>
                        ${candidate.semantic_details ? `<p class="semantic-details">${candidate.semantic_details}</p>` : ''}
                    </div>
                    <div class="candidate-actions">
                        <button class="btn btn-light-green btn-sm" onclick="window.interactiveMap.addWaypoint(${candidate.node_id})">
                            Add Waypoint
                        </button>
                        <button class="btn btn-primary btn-sm" onclick="window.interactiveMap.finalizeRoute(${candidate.node_id})">
                            Final Destination
                        </button>
                    </div>
                </div>
            `;
            
            marker.bindPopup(popupContent);
            this.candidateMarkers.push(marker);
        });
        
        // Only fit map to show all candidates if user hasn't manually set zoom
        if (candidates.length > 0 && !this.userSetZoom) {
            const group = new L.featureGroup([...this.candidateMarkers]);
            this.map.fitBounds(group.getBounds().pad(0.1));
        }
    }
    
    /**
     * Process lucky route generation - instantly build complete route
     */
    processLuckyRoute(candidates) {
        if (!this.feelingLuckyMode || candidates.length === 0) return;
        
        // Get current route stats to determine if we should finalize
        const currentStats = this.getCurrentRouteStats();
        const targetDistance = this.getTargetDistance();
        const currentDistance = currentStats ? currentStats.current_distance : 0;
        const progressPercent = currentDistance / targetDistance;
        
        // If we're close to target distance (80%+), finalize instantly
        // Otherwise, add waypoint and continue
        const shouldFinalize = progressPercent >= 0.8;
        
        // Select random candidate
        const randomIndex = Math.floor(Math.random() * candidates.length);
        const selectedCandidate = candidates[randomIndex];
        
        console.log(`Feeling lucky: ${shouldFinalize ? 'finalizing route' : 'adding waypoint'}`);
        
        // Instantly add waypoint or finalize
        if (shouldFinalize) {
            this.finalizeRoute(selectedCandidate.node_id);
        } else {
            this.addWaypoint(selectedCandidate.node_id);
        }
    }
    
    
    /**
     * Get current route statistics
     */
    getCurrentRouteStats() {
        // Try to extract current stats from the UI
        const distanceElement = document.getElementById('routeDistance');
        if (distanceElement) {
            const distanceText = distanceElement.textContent;
            const distance = parseFloat(distanceText) * 1000; // Convert km to meters
            return { current_distance: distance };
        }
        return null;
    }
    
    /**
     * Get target distance from settings
     */
    getTargetDistance() {
        const stepsInput = document.getElementById('stepsInput') || document.getElementById('stepsInputMobile');
        if (stepsInput) {
            const steps = parseInt(stepsInput.value) || 10000;
            return steps * 0.8; // Convert steps to meters
        }
        return 8000; // Default 10k steps = 8km
    }
    
    /**
     * Generate HTML for semantic scores display
     */
    generateSemanticScoresHtml(candidate) {
        if (!candidate.feature_scores || Object.keys(candidate.feature_scores).length === 0) {
            return '<p class="no-scores">No semantic data available</p>';
        }
        
        const featureScores = candidate.feature_scores;
        const overallScore = candidate.value_score || 0;
        
        // Create score bars for each feature
        let html = '<div class="score-breakdown">';
        
        const featureDisplayMap = {
            'close_to_forest': { label: 'Forest Access', icon: '🌲', color: '#228B22' },
            'close_to_water': { label: 'Water Features', icon: '🌊', color: '#0077BE' },
            'close_to_park': { label: 'Parks & Gardens', icon: '🏞️', color: '#228B22' },
            'path_quality': { label: 'Path Quality', icon: '🚶', color: '#8B4513' },
            'intersection_density': { label: 'Connectivity', icon: '🛤️', color: '#696969' },
            'elevation_variety': { label: 'Terrain Variety', icon: '⛰️', color: '#8B4513' }
        };
        
        // Show features that have scores
        Object.keys(featureScores).forEach(featureKey => {
            const featureInfo = featureDisplayMap[featureKey];
            if (featureInfo) {
                const score = featureScores[featureKey] || 0;
                const percentage = Math.round(score * 100);
                const scoreClass = this.getScoreClass(score);
                
                html += `
                    <div class="score-item">
                        <span class="score-icon">${featureInfo.icon}</span>
                        <span class="score-label">${featureInfo.label}:</span>
                        <div class="score-bar">
                            <div class="score-fill ${scoreClass}" style="width: ${percentage}%; background-color: ${featureInfo.color}"></div>
                            <span class="score-text">${percentage}%</span>
                        </div>
                    </div>
                `;
            }
        });
        
        // Overall score
        const overallPercentage = Math.round(overallScore * 100);
        const overallClass = this.getScoreClass(overallScore);
        
        html += `
            <div class="score-item overall-score">
                <span class="score-icon">⭐</span>
                <span class="score-label"><strong>Overall Score:</strong></span>
                <div class="score-bar">
                    <div class="score-fill ${overallClass}" style="width: ${overallPercentage}%; background-color: #626F47"></div>
                    <span class="score-text"><strong>${overallPercentage}%</strong></span>
                </div>
            </div>
        `;
        
        html += '</div>';
        return html;
    }
    
    /**
     * Get CSS class for score level
     */
    getScoreClass(score) {
        if (score >= 0.7) return 'score-high';
        if (score >= 0.4) return 'score-medium';
        if (score >= 0.1) return 'score-low';
        return 'score-none';
    }
    
    /**
     * Clear candidate markers
     */
    clearCandidates() {
        this.candidateMarkers.forEach(marker => {
            this.map.removeLayer(marker);
        });
        this.candidateMarkers = [];
    }
    
    /**
     * Add starting point marker
     */
    addStartMarker(lat, lon) {
        // Remove existing start marker if any
        this.routeLayers = this.routeLayers.filter(layer => {
            if (layer.options && layer.options.isStartMarker) {
                this.map.removeLayer(layer);
                return false;
            }
            return true;
        });
        
        const startMarker = L.marker([lat, lon], {
            icon: L.divIcon({
                className: 'start-marker',
                html: `
                    <div class="start-marker-content">
                        <div class="start-icon">⌂</div>
                        <div class="start-label">Start</div>
                    </div>
                `,
                iconSize: [50, 50],
                iconAnchor: [25, 25]
            }),
            isStartMarker: true
        }).addTo(this.map);
        
        startMarker.bindPopup('<strong>Starting Point</strong><br>Click Start to begin route planning!');
        this.routeLayers.push(startMarker);
    }
    
    /**
     * Show route progress (partial route so far)
     */
    showRouteProgress(routePath) {
        // Remove previous route progress
        this.routeLayers.forEach(layer => {
            if (layer instanceof L.Polyline) {
                this.map.removeLayer(layer);
            }
        });
        
        if (routePath && routePath.length > 1) {
            // Handle both coordinate formats: {lat, lon} objects and [lat, lon] arrays
            const coordinates = routePath.map(point => {
                if (Array.isArray(point)) {
                    // Already in [lat, lon] format
                    return point;
                } else {
                    // Convert from {lat, lon} object to [lat, lon] array
                    return [point.lat, point.lon];
                }
            });
            
            const routeLine = L.polyline(coordinates, {
                color: '#626F47',
                weight: 4,
                opacity: 0.7,
                dashArray: '8, 4'
            }).addTo(this.map);
            
            this.routeLayers.push(routeLine);
            
            // Only fit map to show the route progress if user hasn't manually set zoom
            if (!this.userSetZoom) {
                this.map.fitBounds(routeLine.getBounds().pad(0.1));
            }
        }
    }
    
    /**
     * Show final completed route
     */
    showFinalRoute(routeData) {
        // Clear all existing route layers
        this.clearRoute();
        
        // Add the completed route
        if (routeData.coordinates && routeData.coordinates.length > 1) {
            // Convert from {lat, lon} objects to [lat, lon] arrays for Leaflet
            const coordinates = routeData.coordinates.map(point => [point.lat, point.lon]);
            
            const routeLine = L.polyline(coordinates, {
                color: '#626F47',
                weight: 5,
                opacity: 0.8
            }).addTo(this.map);
            
            this.routeLayers.push(routeLine);
            
            // Store current route for export
            this.currentRoute = {
                data: routeData,
                layer: routeLine
            };
            
            // Only fit map to route if user hasn't manually set zoom
            if (!this.userSetZoom) {
                this.map.fitBounds(routeLine.getBounds().pad(0.05));
            }
        }
    }
    
    /**
     * Clear only route layers without affecting route building state
     */
    clearRouteLayers() {
        this.routeLayers.forEach(layer => {
            // Only remove if it's not a start marker
            if (layer.options && !layer.options.isStartMarker) {
                this.map.removeLayer(layer);
            }
        });
        this.routeLayers = this.routeLayers.filter(layer => 
            layer.options && layer.options.isStartMarker
        );
        this.currentRoute = null;
    }

    /**
     * Clear all route-related layers (but preserve start marker) and reset state
     */
    clearRoute() {
        this.clearRouteLayers();
        this.routeBuilding = false;
        this.userSetZoom = false;
        this.sessionId = null;
        
        // Disable scoring overlay button when route is cleared
        this.disableScoringOverlay();
        
        // Reset mobile UI to show location controls
        const mobileDistanceDisplay = document.getElementById('mobileDistanceDisplay');
        const mobileLocationControl = document.getElementById('mobileLocationControl');
        
        if (mobileDistanceDisplay && mobileLocationControl) {
            mobileDistanceDisplay.style.display = 'none';
            mobileLocationControl.style.display = 'flex';
        }
        
        // Reset app state when route is cleared
        if (window.perfect10kApp) {
            window.perfect10kApp.setAppState('initial');
        }
    }
    
    /**
     * Update route building statistics
     */
    updateRouteStats(stats) {
        document.getElementById('routeDistance').textContent = `${(stats.current_distance / 1000).toFixed(1)} km`;
        
        // Calculate meaningful progress: current_distance / target_distance
        const targetDistance = this.getTargetDistance(); // in meters
        const currentDistance = stats.current_distance || 0; // in meters
        const progress = Math.min(currentDistance / targetDistance, 1.0); // Cap at 100%
        const progressPercent = Math.round(progress * 100);
        
        document.getElementById('routeProgress').textContent = `${progressPercent}%`;
        
        // Update mobile distance display
        this.updateMobileDistanceDisplay(stats.current_distance / 1000, progressPercent);
    }
    
    /**
     * Update final route statistics
     */
    updateFinalRouteStats(stats) {
        document.getElementById('routeDistance').textContent = `${(stats.total_distance / 1000).toFixed(1)} km`;
        document.getElementById('routeProgress').textContent = '100%';
        document.getElementById('routeArea').textContent = `${Math.round(stats.area || 0).toLocaleString()} m²`;
        document.getElementById('routeNodes').textContent = stats.total_nodes || 0;
        document.getElementById('routeConvexity').textContent = `${Math.round((stats.convexity || 0) * 100)}%`;
        document.getElementById('routeScore').textContent = `${Math.round((stats.score || 0) * 100)}%`;
        document.getElementById('routeConflicts').textContent = stats.conflicts || 0;
        
        // Update mobile distance display with final stats
        this.updateMobileDistanceDisplay(stats.total_distance / 1000, 100);
    }
    
    /**
     * Show route stats panel
     */
    showRouteStats() {
        document.getElementById('routeStats').style.display = 'block';
    }
    
    /**
     * Update mobile distance display and handle UI transitions
     */
    updateMobileDistanceDisplay(distanceKm, progressPercent) {
        const mobileDistanceDisplay = document.getElementById('mobileDistanceDisplay');
        const mobileLocationControl = document.getElementById('mobileLocationControl');
        const mobileRouteDistance = document.getElementById('mobileRouteDistance');
        const mobileBatteryFill = document.getElementById('mobileBatteryFill');
        
        if (mobileDistanceDisplay && mobileLocationControl && mobileRouteDistance && mobileBatteryFill) {
            // Update distance value
            mobileRouteDistance.textContent = `${distanceKm.toFixed(1)} km`;
            
            // Update battery fill width and color
            this.updateBatteryProgress(mobileBatteryFill, progressPercent);
            
            // Show distance display and hide location controls when route is being built or completed
            const hasActiveRoute = this.routeBuilding || this.currentRoute;
            
            if (hasActiveRoute) {
                mobileDistanceDisplay.style.display = 'flex';
                mobileLocationControl.style.display = 'none';
            } else {
                mobileDistanceDisplay.style.display = 'none';
                mobileLocationControl.style.display = 'flex';
            }
        }
    }
    
    /**
     * Update battery fill element based on progress percentage
     */
    updateBatteryProgress(fillElement, progressPercent) {
        if (!fillElement) return;
        
        // Remove all existing progress attributes
        fillElement.removeAttribute('data-progress');
        fillElement.removeAttribute('data-progress-range');
        
        const progress = Math.round(progressPercent);
        
        // Set the width of the fill based on progress
        fillElement.style.width = `${Math.min(progress, 100)}%`;
        
        // Set the appropriate color based on progress range
        if (progress === 100) {
            fillElement.setAttribute('data-progress', '100');
        } else if (progress >= 90) {
            fillElement.setAttribute('data-progress-range', '90-100');
        } else if (progress >= 75) {
            fillElement.setAttribute('data-progress-range', '75-90');
        } else if (progress >= 50) {
            fillElement.setAttribute('data-progress-range', '50-75');
        } else if (progress >= 25) {
            fillElement.setAttribute('data-progress-range', '25-50');
        } else if (progress > 0) {
            fillElement.setAttribute('data-progress-range', '0-25');
        } else {
            fillElement.setAttribute('data-progress', '0');
        }
    }
    
    /**
     * Request user's current location
     */
    requestUserLocation() {
        if (!navigator.geolocation) {
            this.showMessage('Geolocation not supported by this browser', 'error');
            return;
        }
        
        // Getting user location
        
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                
                this.userLocation = { lat, lon };
                this.setLocationAndUnblur(lat, lon);
                
                // Show success and set location
                if (window.perfect10kApp) {
                    window.perfect10kApp.setLocation(lat, lon);
                    window.perfect10kApp.showMessage('Location found', 'success');
                }
            },
            (error) => {
                console.warn('Location access failed:', error);
                if (window.perfect10kApp) {
                    window.perfect10kApp.showMessage('Could not get location. Please try again or enter manually.', 'error');
                }
            },
            { timeout: 10000, enableHighAccuracy: true }
        );
    }
    
    /**
     * Set map location and update inputs
     */
    setLocationAndUnblur(lat, lon) {
        // Only set zoom if user hasn't manually adjusted it, otherwise just pan to location
        if (this.userSetZoom) {
            this.map.panTo([lat, lon]);
        } else {
            this.map.setView([lat, lon], 15);
        }
        
        // Update location inputs
        const locationText = `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
        document.getElementById('locationInput').value = locationText;
        
        const locationInputMobile = document.getElementById('locationInputMobile');
        if (locationInputMobile) {
            locationInputMobile.value = locationText;
        }
        
        this.userLocation = { lat, lon };
    }
    
    /**
     * Center map on current route
     */
    centerOnRoute() {
        if (this.currentRoute && this.currentRoute.layer) {
            this.map.fitBounds(this.currentRoute.layer.getBounds().pad(0.05));
        } else if (this.routeLayers.length > 0) {
            const group = new L.featureGroup(this.routeLayers);
            this.map.fitBounds(group.getBounds().pad(0.1));
        }
    }
    
    /**
     * Toggle fullscreen mode
     */
    toggleFullscreen() {
        const mapContainer = document.querySelector('.map-container');
        
        if (!document.fullscreenElement) {
            mapContainer.requestFullscreen().then(() => {
                setTimeout(() => this.map.invalidateSize(), 100);
            });
        } else {
            document.exitFullscreen().then(() => {
                setTimeout(() => this.map.invalidateSize(), 100);
            });
        }
    }
    
    /**
     * Show status message
     */
    showMessage(text, type = 'info') {
        const statusElement = document.getElementById('statusText');
        if (statusElement) {
            statusElement.textContent = text;
            
            // Add color class based on type
            statusElement.className = `status-${type}`;
            
            // Clear after 5 seconds for non-error messages
            if (type !== 'error') {
                setTimeout(() => {
                    if (statusElement.textContent === text) {
                        statusElement.textContent = 'Ready to plan your route';
                        statusElement.className = '';
                    }
                }, 5000);
            }
        }
        
        console.log(`[${type.toUpperCase()}] ${text}`);
    }
    
    /**
     * Get current map state for debugging
     */
    getMapState() {
        return {
            center: this.map.getCenter(),
            zoom: this.map.getZoom(),
            hasRoute: !!this.currentRoute,
            routeBuilding: this.routeBuilding,
            candidatesCount: this.candidates.length,
            waypointsCount: this.waypoints.length,
            semanticOverlays: this.semanticOverlays ? this.semanticOverlays.getOverlayStats() : null
        };
    }
    
    /**
     * Debug function to test mobile UI
     */
    testMobileUI() {
        console.log('Testing mobile UI...');
        const mobileDistanceDisplay = document.getElementById('mobileDistanceDisplay');
        const mobileLocationControl = document.getElementById('mobileLocationControl');
        
        if (mobileDistanceDisplay && mobileLocationControl) {
            console.log('Elements found, testing visibility toggle...');
            // Force show distance display
            mobileDistanceDisplay.style.display = 'flex';
            mobileLocationControl.style.display = 'none';
            
            // Update values
            const mobileRouteDistance = document.getElementById('mobileRouteDistance');
            const mobileRouteProgress = document.getElementById('mobileRouteProgress');
            if (mobileRouteDistance && mobileRouteProgress) {
                mobileRouteDistance.textContent = '2.5 km';
                mobileRouteProgress.textContent = '75%';
                console.log('Updated distance and progress values');
            }
        } else {
            console.error('Mobile UI elements not found:', {
                mobileDistanceDisplay: !!mobileDistanceDisplay,
                mobileLocationControl: !!mobileLocationControl
            });
        }
    }
    
}

// Initialize the interactive map when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.interactiveMap = new InteractiveMapEditor('map');
});