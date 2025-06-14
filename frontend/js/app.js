/**
 * Perfect10k Frontend Application
 * Main application logic and UI coordination
 */

class Perfect10kApp {
    constructor() {
        this.mapEditor = null;
        this.currentSession = null;
        this.planningInProgress = false;
        this.editHistory = [];
        this.appState = 'initial'; // 'initial', 'building', 'completed'
        
        this.initialize();
    }
    
    /**
     * Initialize the application
     */
    initialize() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupApp());
        } else {
            this.setupApp();
        }
    }
    
    /**
     * Show loading bar with optional text
     */
    showLoading(text = 'Loading...') {
        const loadingBar = document.getElementById('loadingBar');
        if (loadingBar) {
            loadingBar.classList.remove('hidden');
        }
        
        // Update status text if provided
        this.updateStatus(text, 'info');
    }
    
    /**
     * Hide loading bar
     */
    hideLoading() {
        const loadingBar = document.getElementById('loadingBar');
        if (loadingBar) {
            loadingBar.classList.add('hidden');
        }
    }
    
    /**
     * Set up the main application
     */
    setupApp() {
        // Wait for interactive map to be available
        if (window.interactiveMap) {
            this.mapEditor = window.interactiveMap;
        } else {
            // Fallback: create our own instance
            this.mapEditor = new InteractiveMapEditor('map');
        }
        
        // Set up event handlers
        this.setupEventHandlers();
        
        // Set up steps slider
        this.setupStepsSlider();
        
        // Check backend connectivity
        this.checkBackendHealth();
        
        // Update session info
        this.updateSessionInfo();
        
        // Hide route stats initially
        document.getElementById('routeStats').style.display = 'none';
        
        // Set initial app state
        this.setAppState('initial');
        
        // Initialize buttons as disabled since no location is set
        this.updateStartButton('disabled');
        
        // Try to get user location automatically
        this.checkAndRequestLocation();
        
        console.log('Perfect10k Interactive application initialized');
    }
    
    /**
     * Set up steps slider functionality
     */
    setupStepsSlider() {
        const stepsInput = document.getElementById('stepsInput');
        const stepsValue = document.getElementById('stepsValue');
        
        if (stepsInput && stepsValue) {
            stepsInput.addEventListener('input', (e) => {
                const steps = parseInt(e.target.value);
                stepsValue.textContent = steps.toLocaleString();
                
                // Sync with mobile slider
                const stepsInputMobile = document.getElementById('stepsInputMobile');
                const stepsValueMobile = document.getElementById('stepsValueMobile');
                if (stepsInputMobile && stepsValueMobile) {
                    stepsInputMobile.value = steps;
                    stepsValueMobile.textContent = steps.toLocaleString();
                }
            });
        }
        
        // Mobile steps slider
        const stepsInputMobile = document.getElementById('stepsInputMobile');
        const stepsValueMobile = document.getElementById('stepsValueMobile');
        
        if (stepsInputMobile && stepsValueMobile) {
            stepsInputMobile.addEventListener('input', (e) => {
                const steps = parseInt(e.target.value);
                stepsValueMobile.textContent = steps.toLocaleString();
                
                // Sync with desktop slider
                if (stepsInput && stepsValue) {
                    stepsInput.value = steps;
                    stepsValue.textContent = steps.toLocaleString();
                }
            });
        }
        
        // Sync preferences between mobile and desktop
        const preferencesInput = document.getElementById('preferencesInput');
        const preferencesInputMobile = document.getElementById('preferencesInputMobile');
        
        if (preferencesInput && preferencesInputMobile) {
            preferencesInput.addEventListener('input', (e) => {
                preferencesInputMobile.value = e.target.value;
            });
            
            preferencesInputMobile.addEventListener('input', (e) => {
                preferencesInput.value = e.target.value;
            });
        }
    }
    
    /**
     * Convert steps to kilometers using standard conversion
     */
    convertStepsToKm(steps) {
        const metersPerStep = 0.8; // Average step length in meters
        return (steps * metersPerStep) / 1000; // Convert to kilometers
    }
    
    /**
     * Set up all event handlers
     */
    setupEventHandlers() {
        // Start/Restart button (desktop)
        document.getElementById('startRoute').addEventListener('click', () => {
            this.handleStartRestart();
        });
        
        // Mobile start/restart button
        const startRouteMobile = document.getElementById('startRouteMobile');
        if (startRouteMobile) {
            startRouteMobile.addEventListener('click', () => {
                this.handleStartRestart();
            });
        }
        
        // Settings toggle for mobile
        const settingsToggle = document.getElementById('settingsToggleMobile');
        if (settingsToggle) {
            settingsToggle.addEventListener('click', () => {
                this.toggleSettingsOverlay();
            });
        }
        
        // Settings close button for mobile
        const closeSettings = document.getElementById('closeSettingsMobile');
        if (closeSettings) {
            closeSettings.addEventListener('click', () => {
                this.hideSettingsOverlay();
            });
        }
        
        // Location input and current location
        document.getElementById('useCurrentLocation').addEventListener('click', () => {
            this.showLoading('Getting your location...');
            this.mapEditor.requestUserLocation();
        });
        
        // Mobile current location button
        const useCurrentLocationMobile = document.getElementById('useCurrentLocationMobile');
        if (useCurrentLocationMobile) {
            useCurrentLocationMobile.addEventListener('click', () => {
                this.showLoading('Getting your location...');
                this.mapEditor.requestUserLocation();
            });
        }
        
        // Mobile download button in bottom bar
        const downloadToggleMobile = document.getElementById('downloadToggleMobile');
        if (downloadToggleMobile) {
            downloadToggleMobile.addEventListener('click', () => {
                this.exportRoute();
            });
        }
        
        // Location input - handle both keypress and input events for mobile compatibility
        const locationInput = document.getElementById('locationInput');
        locationInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' || e.keyCode === 13) {
                e.preventDefault();
                this.handleLocationInput();
            }
        });
        
        // Clear location input on focus for easy entry
        locationInput.addEventListener('focus', (e) => {
            e.target.select(); // Select all text for easy replacement
        });
        
        // Mobile location input
        const locationInputMobile = document.getElementById('locationInputMobile');
        if (locationInputMobile) {
            locationInputMobile.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' || e.keyCode === 13) {
                    e.preventDefault();
                    this.handleLocationInputMobile();
                }
            });
            
            // Clear mobile location input on focus for easy entry
            locationInputMobile.addEventListener('focus', (e) => {
                e.target.select(); // Select all text for easy replacement
            });
            
            // Also handle input event for better mobile support
            locationInputMobile.addEventListener('input', (e) => {
                // Sync with desktop input
                locationInput.value = e.target.value;
            });
        }
        
        // Route actions - desktop only (mobile is handled above)
        document.getElementById('exportRoute').addEventListener('click', () => {
            this.exportRoute();
        });
        
        
        // Network status monitoring
        window.addEventListener('networkStatusChanged', (e) => {
            this.handleNetworkStatusChange(e.detail.status);
        });
    }
    
    /**
     * Start interactive routing (new approach)
     */
    async startInteractiveRouting() {
        try {
            this.showLoading('Loading area map...');
            
            const locationInput = document.getElementById('locationInput').value.trim();
            
            if (locationInput) {
                // Parse and use specified location
                const location = await this.parseLocation(locationInput);
                if (!location) {
                    this.hideLoading();
                    this.showMessage('Invalid location format', 'error');
                    return;
                }
                
                await this.mapEditor.startRoutingSession(location.lat, location.lon);
            } else {
                // Use current location or prompt user to click on map
                if (this.mapEditor.userLocation) {
                    await this.mapEditor.startRoutingSession(
                        this.mapEditor.userLocation.lat, 
                        this.mapEditor.userLocation.lon
                    );
                } else {
                    this.hideLoading();
                    // Do nothing - user needs to set a location first
                    return;
                }
            }
            
        } catch (error) {
            this.hideLoading();
            console.error('Interactive routing failed:', error);
            this.showMessage(`Routing failed: ${error.message}`, 'error');
        }
    }
    
    /**
     * Execute route planning
     */
    async executeRoutePlanning(request) {
        // Call the API
        const response = await window.apiClient.planRoute(request);
        
        // Display the route
        await this.mapEditor.displayRoute(response, 'final');
        
        // Update session info
        this.currentSession = window.apiClient.currentSession;
        this.updateSessionInfo();
        
        // Update route statistics including algorithm performance
        this.updateRouteStatistics(response);
        
        // Update app state and mobile UI
        this.onRouteCompleted();
        this.hideSettingsOverlay(); // Hide settings after generating route
        
        this.showMessage(response.message, 'success');
    }
    
    /**
     * Check and auto-request user location on app load
     */
    async checkAndRequestLocation() {
        if (navigator.geolocation) {
            try {
                // Try to get location - more aggressive settings for mobile
                const position = await new Promise((resolve, reject) => {
                    navigator.geolocation.getCurrentPosition(resolve, reject, {
                        timeout: 10000, // Longer timeout for mobile
                        enableHighAccuracy: true, // More accurate for mobile
                        maximumAge: 60000 // 1 minute cache
                    });
                });
                
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                console.log('Auto-detected location:', lat, lon);
                this.setLocation(lat, lon);
                
            } catch (error) {
                console.log('Location detection failed:', error.message);
                // Just log the error, don't show overlay - let user manually set location
            }
        } else {
            console.log('Geolocation not supported');
            // No geolocation support - don't show overlay, let user set manually
        }
    }
    
    /**
     * Set location and hide overlay
     */
    setLocation(lat, lon) {
        // Don't allow location changes in completed state
        if (this.appState === 'completed') {
            console.log('Location change blocked - route is completed');
            return;
        }
        
        this.mapEditor.setLocationAndUnblur(lat, lon);
        this.hideLocationOverlay();
        
        // Add start marker immediately
        this.mapEditor.addStartMarker(lat, lon);
        
        // Update button to show Start is available
        this.updateStartButton('start');
        
        // Ensure we stay in initial state when location is set
        if (this.appState !== 'initial') {
            this.setAppState('initial');
        }
    }
    
    /**
     * Show location selection overlay (deprecated - overlay removed)
     */
    showLocationOverlay() {
        // No longer used - overlay removed for better UX
    }
    
    /**
     * Hide location selection overlay (deprecated - overlay removed)
     */
    hideLocationOverlay() {
        // No longer used - overlay removed for better UX
    }
    
    /**
     * Handle Start/Restart button click
     */
    async handleStartRestart() {
        const button = document.getElementById('startRoute');
        
        // Don't process if button is disabled
        if (button.disabled) {
            return;
        }
        
        if (button.textContent.includes('Start')) {
            // Starting new route
            if (this.mapEditor.userLocation) {
                await this.startInteractiveRouting();
                this.updateStartButton('restart');
                document.getElementById('routeStats').style.display = 'block';
                this.setAppState('building');
            } else {
                // No location set - show message in status
                this.showMessage('Please set a location first by tapping the map or using current location', 'warning');
            }
        } else {
            // Restarting - reset everything
            this.restartRoute();
            
            // If we have a location, immediately show that we can start again
            if (this.mapEditor.userLocation) {
                this.setAppState('initial');
                this.updateStartButton('start');
            } else {
                this.updateStartButton('disabled');
            }
        }
    }
    
    /**
     * Update Start button text and style (both desktop and mobile)
     */
    updateStartButton(mode) {
        const button = document.getElementById('startRoute');
        const mobileButton = document.getElementById('startRouteMobile');
        
        if (mode === 'disabled') {
            // Desktop button - disabled state
            button.innerHTML = '📍 Select Location First';
            button.className = 'btn btn-outline btn-large';
            button.disabled = true;
            
            // Mobile button - disabled state
            if (mobileButton) {
                mobileButton.className = 'btn-start-mobile';
                mobileButton.title = 'Select Location First';
                mobileButton.disabled = true;
                mobileButton.innerHTML = `
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polygon points="5,3 19,12 5,21"></polygon>
                    </svg>
                `;
            }
        } else if (mode === 'start') {
            // Desktop button
            button.innerHTML = '📍 Start';
            button.className = 'btn btn-primary btn-large';
            button.disabled = false;
            
            // Mobile button - right arrow icon
            if (mobileButton) {
                mobileButton.className = 'btn-start-mobile';
                mobileButton.title = 'Start Route';
                mobileButton.disabled = false;
                mobileButton.innerHTML = `
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polygon points="5,3 19,12 5,21"></polygon>
                    </svg>
                `;
            }
        } else {
            // Desktop button
            button.innerHTML = '🔄 Restart';
            button.className = 'btn btn-outline btn-large';
            button.disabled = false;
            
            // Mobile button - restart icon
            if (mobileButton) {
                mobileButton.className = 'btn-start-mobile restart';
                mobileButton.title = 'Restart Route';
                mobileButton.disabled = false;
                mobileButton.innerHTML = `
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M23 4v6h-6"></path>
                        <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                    </svg>
                `;
            }
        }
    }
    
    /**
     * Restart route planning
     */
    restartRoute() {
        // Clear current route and reset state
        this.mapEditor.clearRoute();
        this.mapEditor.clearCandidates();
        this.mapEditor.routeBuilding = false;
        this.mapEditor.waypoints = [];
        this.mapEditor.sessionId = null;
        
        // Hide route stats
        document.getElementById('routeStats').style.display = 'none';
        
        // Clear API session
        window.apiClient.clearSession();
        
        // Reset button
        this.updateStartButton('start');
        
        // Set app state back to initial
        this.setAppState('initial');
        
        // Keep current location and start marker - don't clear them
        // This allows user to quickly restart with same location or click to change
    }
    
    /**
     * Update route statistics display with comprehensive metrics
     */
    updateRouteStatistics(response) {
        const stats = response.route_info || {};
        const debug = response.debug_info || {};
        
        // Basic route stats
        document.getElementById('routeDistance').textContent = `${(stats.distance / 1000).toFixed(1)} km`;
        document.getElementById('routeArea').textContent = `${Math.round(stats.area || 0).toLocaleString()} m²`;
        document.getElementById('routeConvexity').textContent = `${Math.round((stats.convexity || 0) * 100)}%`;
        document.getElementById('routeConflicts').textContent = `${stats.conflicts || 0}`;
        document.getElementById('routeScore').textContent = `${Math.round((stats.score || 0) * 100)}%`;
        
        // Algorithm performance stats
        if (debug.step_results) {
            this.updateAlgorithmStats(debug.step_results, stats);
        }
        
        // Distance accuracy
        const targetDistance = parseInt(document.getElementById('stepsInput').value) * 0.8; // Convert steps to meters
        const actualDistance = stats.distance || 0;
        this.updateDistanceAccuracy(actualDistance, targetDistance);
        
        // Show stats panel
        document.getElementById('routeStats').style.display = 'block';
    }
    
    /**
     * Update algorithm performance statistics
     */
    updateAlgorithmStats(stepResults, routeStats) {
        // Step 1: Coarse Search
        const step1 = stepResults.step1 || {};
        const zones = step1.zones || 0;
        const corridors = step1.corridors || 0;
        document.getElementById('step1Value').textContent = `${zones} zones, ${corridors} corridors`;
        document.getElementById('step1Value').className = zones > 0 ? 'step-value success' : 'step-value warning';
        
        // Step 2: Initial Cycle
        const step2 = stepResults.step2 || {};
        const nodes = step2.nodes || 0;
        const distanceRatio = step2.distance_ratio || 0;
        const step2Status = distanceRatio > 0.5 && distanceRatio < 2.0 ? 'success' : 'warning';
        document.getElementById('step2Value').textContent = `${nodes} nodes, ${(distanceRatio * 100).toFixed(0)}% target`;
        document.getElementById('step2Value').className = `step-value ${step2Status}`;
        
        // Step 2.5: Pruning
        const step25 = stepResults.step2_5 || {};
        const originalDistance = step25.original_distance || 0;
        const prunedDistance = step25.pruned_distance || 0;
        const reduction = step25.distance_reduction || 0;
        const step25Status = prunedDistance > 0 ? 'success' : 'error';
        document.getElementById('step25Value').textContent = `${(reduction * 100).toFixed(0)}% reduction`;
        document.getElementById('step25Value').className = `step-value ${step25Status}`;
        
        // Step 3: Optimization
        const step3 = stepResults.step3 || {};
        const finalConflicts = step3.conflicts || 0;
        const finalArea = step3.area || 0;
        const step3Status = finalConflicts === 0 && finalArea > 0 ? 'success' : finalConflicts > 0 ? 'warning' : 'error';
        document.getElementById('step3Value').textContent = `${finalConflicts} conflicts, ${Math.round(finalArea).toLocaleString()} m²`;
        document.getElementById('step3Value').className = `step-value ${step3Status}`;
    }
    
    /**
     * Update distance accuracy indicator
     */
    updateDistanceAccuracy(actualDistance, targetDistance) {
        if (targetDistance <= 0) return;
        
        const ratio = actualDistance / targetDistance;
        const error = Math.abs(ratio - 1.0);
        
        // Calculate accuracy percentage (100% = perfect, 0% = terrible)
        let accuracy = Math.max(0, 100 - (error * 100));
        
        // Update accuracy bar
        document.getElementById('accuracyFill').style.width = `${accuracy}%`;
        
        // Update accuracy text
        const errorPercent = (error * 100).toFixed(1);
        let statusText = '';
        if (error < 0.05) {
            statusText = `Excellent (${errorPercent}% error)`;
        } else if (error < 0.15) {
            statusText = `Good (${errorPercent}% error)`;
        } else if (error < 0.3) {
            statusText = `Fair (${errorPercent}% error)`;
        } else {
            statusText = `Poor (${errorPercent}% error)`;
        }
        
        document.getElementById('accuracyText').textContent = statusText;
    }
    
    
    /**
     * Parse location input (coordinates or address)
     */
    async parseLocation(input) {
        // Try to parse as coordinates first (lat, lon)
        const coordMatch = input.match(/(-?\d+\.?\d*),\s*(-?\d+\.?\d*)/);
        if (coordMatch) {
            return {
                lat: parseFloat(coordMatch[1]),
                lon: parseFloat(coordMatch[2])
            };
        }
        
        // If not coordinates, treat as address and geocode
        try {
            // Simple geocoding using Nominatim (OpenStreetMap)
            const encodedAddress = encodeURIComponent(input);
            const response = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodedAddress}&limit=1`);
            const data = await response.json();
            
            if (data.length > 0) {
                return {
                    lat: parseFloat(data[0].lat),
                    lon: parseFloat(data[0].lon)
                };
            }
        } catch (error) {
            console.error('Geocoding failed:', error);
        }
        
        return null;
    }
    
    /**
     * Handle location input
     */
    async handleLocationInput() {
        const locationInput = document.getElementById('locationInput');
        const location = await this.parseLocation(locationInput.value);
        
        if (location) {
            this.setLocation(location.lat, location.lon);
        }
    }
    
    /**
     * Handle mobile location input
     */
    async handleLocationInputMobile() {
        const locationInputMobile = document.getElementById('locationInputMobile');
        const location = await this.parseLocation(locationInputMobile.value);
        
        if (location) {
            // Sync with desktop input
            const locationInput = document.getElementById('locationInput');
            locationInput.value = locationInputMobile.value;
            
            this.setLocation(location.lat, location.lon);
        }
    }
    
    /**
     * Export route as GPX
     */
    async exportRoute() {
        if (!this.mapEditor.currentRoute) {
            this.showMessage('No route to export', 'warning');
            return;
        }
        
        try {
            // Create GPX content
            const gpxContent = this.createGPXContent(this.mapEditor.currentRoute.data);
            
            // Download file
            const blob = new Blob([gpxContent], { type: 'application/gpx+xml' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `perfect10k-route-${new Date().toISOString().split('T')[0]}.gpx`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            this.showMessage('Route exported successfully', 'success');
            
        } catch (error) {
            this.showMessage(`Export failed: ${error.message}`, 'error');
        }
    }
    
    /**
     * Create GPX content from route data
     */
    createGPXContent(routeData) {
        const timestamp = new Date().toISOString();
        
        let trackPoints = '';
        routeData.coordinates.forEach(coord => {
            trackPoints += `        <trkpt lat="${coord[0]}" lon="${coord[1]}"></trkpt>\n`;
        });
        
        return `<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Perfect10k" xmlns="http://www.topografix.com/GPX/1/1">
    <metadata>
        <name>Perfect10k Route</name>
        <desc>Generated by Perfect10k route planner</desc>
        <time>${timestamp}</time>
    </metadata>
    <trk>
        <name>Perfect10k Route</name>
        <desc>Distance: ${(routeData.distance / 1000).toFixed(1)}km, Score: ${(routeData.value_score * 100).toFixed(0)}%</desc>
        <trkseg>
${trackPoints}        </trkseg>
    </trk>
</gpx>`;
    }
    
    /**
     * Check backend health
     */
    async checkBackendHealth() {
        try {
            await window.apiClient.getHealthStatus();
            // Connection successful - no need to show message
        } catch (error) {
            console.error('Backend connection failed:', error);
        }
    }
    
    /**
     * Update session information display
     */
    updateSessionInfo() {
        const sessionElement = document.getElementById('sessionId');
        const sessionInfo = window.apiClient.getSessionInfo();
        
        if (sessionInfo.sessionId) {
            sessionElement.textContent = `Session: ${sessionInfo.sessionId.substring(0, 8)}...`;
            sessionElement.classList.remove('hidden');
        } else {
            sessionElement.classList.add('hidden');
        }
    }
    
    /**
     * Handle network status changes
     */
    handleNetworkStatusChange(status) {
        const networkElement = document.getElementById('networkStatus');
        
        switch (status) {
            case 'online':
                networkElement.style.color = 'var(--success)';
                break;
            case 'degraded':
                networkElement.style.color = 'var(--warning)';
                break;
            case 'offline':
                networkElement.style.color = 'var(--error)';
                break;
        }
    }
    
    /**
     * Set button state (enabled/disabled with loading text)
     */
    setButtonState(buttonId, enabled, text = null) {
        const button = document.getElementById(buttonId);
        button.disabled = !enabled;
        
        if (text) {
            button.textContent = text;
        }
    }
    
    /**
     * Show status message
     */
    showMessage(text, type = 'info') {
        this.mapEditor.showMessage(text, type);
    }
    
    /**
     * Show loading bar
     */
    showLoading() {
        document.getElementById('loadingBar').classList.remove('hidden');
    }
    
    /**
     * Hide loading bar
     */
    hideLoading() {
        document.getElementById('loadingBar').classList.add('hidden');
    }
    
    /**
     * Set application state and update UI visibility
     */
    setAppState(newState) {
        // Remove old state class
        document.body.classList.remove(`app-state-${this.appState}`);
        
        // Update state
        this.appState = newState;
        
        // Add new state class
        document.body.classList.add(`app-state-${newState}`);
        
        console.log(`App state changed to: ${newState}`);
    }
    
    /**
     * Handle route completion - update state to completed
     */
    onRouteCompleted() {
        this.setAppState('completed');
        
        // Show route stats if not already visible
        document.getElementById('routeStats').style.display = 'block';
    }
    
    /**
     * Enable/disable route planning button
     */
    enableRoutePlanning(enabled) {
        const planButton = document.getElementById('planRoute');
        const planButtonMobile = document.getElementById('planRouteMobile');
        
        if (planButton) {
            planButton.disabled = !enabled;
        }
        if (planButtonMobile) {
            planButtonMobile.disabled = !enabled;
        }
    }
    
    /**
     * Show settings overlay (mobile)
     */
    showSettingsOverlay() {
        const overlay = document.getElementById('settingsOverlay');
        if (overlay) {
            overlay.classList.add('open');
        }
    }
    
    /**
     * Hide settings overlay (mobile)
     */
    hideSettingsOverlay() {
        const overlay = document.getElementById('settingsOverlay');
        if (overlay) {
            overlay.classList.remove('open');
        }
    }
    
    /**
     * Toggle settings overlay (mobile)
     */
    toggleSettingsOverlay() {
        const overlay = document.getElementById('settingsOverlay');
        if (overlay) {
            overlay.classList.toggle('open');
        }
    }
    
    /**
     * Add edit operation to history
     */
    addEditToHistory(operation, type, details) {
        const editEntry = {
            operation: operation,
            type: type,
            details: details,
            timestamp: new Date().toISOString()
        };
        
        this.editHistory.push(editEntry);
        console.log('Edit added to history:', editEntry);
    }
    
    /**
     * Get edit history
     */
    getEditHistory() {
        return this.editHistory;
    }
    
    /**
     * Clear edit history
     */
    clearEditHistory() {
        this.editHistory = [];
    }
    
    /**
     * Utility delay function
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    /**
     * Get application state
     */
    getAppState() {
        return {
            hasRoute: !!this.mapEditor.currentRoute,
            currentSession: this.currentSession,
            editHistory: this.editHistory.length,
            networkStatus: window.apiClient.networkStatus,
            mapState: this.mapEditor.getMapState()
        };
    }
    
    /**
     * Show map planning state with blur and overlay
     */
    showMapPlanningState() {
        const mapElement = document.getElementById("map");
        mapElement.classList.add("planning");
        
        // Show planning overlay
        const planningOverlay = document.createElement("div");
        planningOverlay.id = "planningOverlay";
        planningOverlay.className = "planning-overlay";
        planningOverlay.innerHTML = `
            <div class="loading-spinner"></div>
            <span>Planning route...</span>
        `;
        
        const mapContainer = document.querySelector(".map-container");
        mapContainer.appendChild(planningOverlay);
    }
    
    /**
     * Hide map planning state
     */
    hideMapPlanningState() {
        const mapElement = document.getElementById("map");
        mapElement.classList.remove("planning");
        
        // Remove planning overlay
        const planningOverlay = document.getElementById("planningOverlay");
        if (planningOverlay) {
            planningOverlay.remove();
        }
    }
}

// Initialize the application when the page loads
const app = new Perfect10kApp();

// Make app globally available for debugging
window.perfect10kApp = app;

// Debug function to test mobile UI changes
window.testMobileUIChanges = function(progress = 85) {
    console.log('Testing mobile UI battery with progress:', progress + '%');
    
    const mobileDistanceDisplay = document.getElementById('mobileDistanceDisplay');
    const mobileLocationControl = document.getElementById('mobileLocationControl');
    const mobileRouteDistance = document.getElementById('mobileRouteDistance');
    const mobileBatteryFill = document.getElementById('mobileBatteryFill');
    
    console.log('Found elements:', {
        mobileDistanceDisplay: !!mobileDistanceDisplay,
        mobileLocationControl: !!mobileLocationControl,
        mobileRouteDistance: !!mobileRouteDistance,
        mobileBatteryFill: !!mobileBatteryFill
    });
    
    if (mobileDistanceDisplay && mobileLocationControl) {
        console.log('Switching to battery display...');
        mobileDistanceDisplay.style.display = 'flex';
        mobileLocationControl.style.display = 'none';
        
        if (mobileRouteDistance && mobileBatteryFill) {
            mobileRouteDistance.textContent = '3.2 km';
            window.interactiveMap.updateBatteryProgress(mobileBatteryFill, progress);
            console.log('Updated battery with progress:', progress + '%');
        }
    } else {
        console.error('Elements not found!');
    }
};

// Test different progress levels
window.testProgressColors = function() {
    const progressLevels = [0, 15, 35, 65, 80, 95, 100];
    let index = 0;
    
    function showNext() {
        if (index < progressLevels.length) {
            window.testMobileUIChanges(progressLevels[index]);
            console.log(`Showing progress: ${progressLevels[index]}%`);
            index++;
            setTimeout(showNext, 2000); // Show each for 2 seconds
        }
    }
    
    showNext();
};

// Debug function to test switching back
window.testMobileUIReset = function() {
    console.log('Resetting mobile UI...');
    
    const mobileDistanceDisplay = document.getElementById('mobileDistanceDisplay');
    const mobileLocationControl = document.getElementById('mobileLocationControl');
    
    if (mobileDistanceDisplay && mobileLocationControl) {
        mobileDistanceDisplay.style.display = 'none';
        mobileLocationControl.style.display = 'flex';
        console.log('Reset to location control');
    }
};
