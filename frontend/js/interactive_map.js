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
        
        this.initialize();
    }
    
    /**
     * Initialize the map
     */
    initialize() {
        // Initialize the map centered on San Francisco
        this.map = L.map(this.containerId, {
            center: [37.7749, -122.4194],
            zoom: 13,
            zoomControl: false
        });
        
        // Add minimal CartoDB style tile layer
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
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
            // Show loading indicator
            if (window.perfect10kApp) {
                window.perfect10kApp.showLoading();
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
            
            // The new API returns data directly without a "success" wrapper
            if (response.session_id && response.candidates) {
                this.sessionId = response.session_id;
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
                
                // Route planning started - candidates displayed
            } else {
                this.showMessage(`Failed to start session: ${response.message || 'Unknown error'}`, 'error');
            }
            
        } catch (error) {
            console.error('Failed to start routing session:', error);
            this.showMessage(`Failed to start routing: ${error.message}`, 'error');
        } finally {
            // Hide loading indicator
            if (window.perfect10kApp) {
                window.perfect10kApp.hideLoading();
            }
        }
    }
    
    /**
     * Add waypoint to current route
     */
    async addWaypoint(nodeId) {
        try {
            // Show loading indicator
            if (window.perfect10kApp) {
                window.perfect10kApp.showLoading();
            }
            
            const response = await window.apiClient.addWaypoint(nodeId);
            
            if (response.success) {
                // Add to waypoints (get location from current_path)
                if (response.current_path && response.current_path.length > 0) {
                    const lastPoint = response.current_path[response.current_path.length - 1];
                    this.waypoints.push({
                        lat: lastPoint.lat,
                        lon: lastPoint.lon,
                        nodeId: nodeId
                    });
                }
                
                // Clear previous candidates
                this.clearCandidates();
                
                // Show updated route path
                this.showRouteProgress(response.current_path);
                
                // Show new candidates if any
                if (response.candidates && response.candidates.length > 0) {
                    this.showCandidates(response.candidates);
                }
                
                // Update stats
                this.updateRouteStats(response.route_stats);
                
            } else {
                this.showMessage(`Failed to add waypoint: ${response.message}`, 'error');
            }
            
        } catch (error) {
            console.error('Failed to add waypoint:', error);
            this.showMessage(`Failed to add waypoint: ${error.message}`, 'error');
        } finally {
            // Hide loading indicator
            if (window.perfect10kApp) {
                window.perfect10kApp.hideLoading();
            }
        }
    }
    
    /**
     * Finalize route with selected destination
     */
    async finalizeRoute(nodeId) {
        try {
            // Show loading indicator
            if (window.perfect10kApp) {
                window.perfect10kApp.showLoading();
            }
            
            const response = await window.apiClient.finalizeRoute(nodeId);
            
            if (response.success) {
                // Clear candidates
                this.clearCandidates();
                
                // Show final route
                this.showFinalRoute({
                    coordinates: response.completed_route
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
                this.showMessage(`Failed to finalize route: ${response.message}`, 'error');
            }
            
        } catch (error) {
            console.error('Failed to finalize route:', error);
            this.showMessage(`Failed to finalize route: ${error.message}`, 'error');
        } finally {
            // Hide loading indicator
            if (window.perfect10kApp) {
                window.perfect10kApp.hideLoading();
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
                            <div class="candidate-distance">${(candidate.distance / 1000).toFixed(1)}km</div>
                        </div>
                    `,
                    iconSize: [60, 60],
                    iconAnchor: [30, 30]
                })
            }).addTo(this.map);
            
            // Add popup with actions
            const popupContent = `
                <div class="candidate-popup">
                    <h4>Candidate ${index + 1}</h4>
                    <p><strong>Distance:</strong> ${(candidate.distance / 1000).toFixed(1)}km</p>
                    <p><strong>Why this spot:</strong> ${candidate.explanation || 'Basic walkable area'}</p>
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
            // Convert from {lat, lon} objects to [lat, lon] arrays for Leaflet
            const coordinates = routePath.map(point => [point.lat, point.lon]);
            
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
        document.getElementById('routeProgress').textContent = `${Math.round(stats.progress * 100)}%`;
        document.getElementById('estimatedDistance').textContent = `${(stats.estimated_final_distance / 1000).toFixed(1)} km`;
        document.getElementById('waypointsCount').textContent = stats.waypoints_count;
        
        // Update mobile distance display
        this.updateMobileDistanceDisplay(stats.current_distance / 1000, stats.progress * 100);
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
            if (window.perfect10kApp) {
                window.perfect10kApp.hideLoading();
            }
            return;
        }
        
        // Getting user location
        
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                
                this.userLocation = { lat, lon };
                this.setLocationAndUnblur(lat, lon);
                
                // Hide loading and show success
                if (window.perfect10kApp) {
                    window.perfect10kApp.hideLoading();
                    window.perfect10kApp.setLocation(lat, lon);
                    window.perfect10kApp.showMessage('Location found', 'success');
                }
            },
            (error) => {
                console.warn('Location access failed:', error);
                if (window.perfect10kApp) {
                    window.perfect10kApp.hideLoading();
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
            waypointsCount: this.waypoints.length
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