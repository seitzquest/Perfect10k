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
        
        // Add tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.map);
        
        // Add zoom controls in custom position
        L.control.zoom({
            position: 'topright'
        }).addTo(this.map);
        
        // Set up event handlers
        this.setupEventHandlers();
        
        console.log('Interactive map editor initialized');
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
                this.routeBuilding = true;
                
                // Clear existing route display (but keep start marker)
                this.clearRoute();
                
                // Show candidates
                this.showCandidates(response.candidates);
                
                // Update UI
                this.updateRouteStats(response.route_stats);
                this.showRouteStats();
                
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
                    <p><strong>Score:</strong> ${(candidate.value_score * 100).toFixed(0)}%</p>
                    <div class="candidate-actions">
                        <button class="btn btn-primary btn-sm" onclick="window.interactiveMap.addWaypoint(${candidate.node_id})">
                            📍 Add Waypoint
                        </button>
                        <button class="btn btn-success btn-sm" onclick="window.interactiveMap.finalizeRoute(${candidate.node_id})">
                            🏁 Final Destination
                        </button>
                    </div>
                </div>
            `;
            
            marker.bindPopup(popupContent);
            this.candidateMarkers.push(marker);
        });
        
        // Fit map to show all candidates
        if (candidates.length > 0) {
            const group = new L.featureGroup([...this.candidateMarkers]);
            this.map.fitBounds(group.getBounds().pad(0.1));
        }
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
                        <div class="start-icon">🏠</div>
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
                color: '#10B981',
                weight: 4,
                opacity: 0.8,
                dashArray: '10, 5'
            }).addTo(this.map);
            
            this.routeLayers.push(routeLine);
            
            // Fit map to show the route progress
            this.map.fitBounds(routeLine.getBounds().pad(0.1));
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
                color: '#3B82F6',
                weight: 5,
                opacity: 0.9
            }).addTo(this.map);
            
            this.routeLayers.push(routeLine);
            
            // Store current route for export
            this.currentRoute = {
                data: routeData,
                layer: routeLine
            };
            
            // Fit map to route
            this.map.fitBounds(routeLine.getBounds().pad(0.05));
        }
    }
    
    /**
     * Clear all route-related layers (but preserve start marker)
     */
    clearRoute() {
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
    }
    
    /**
     * Show route stats panel
     */
    showRouteStats() {
        document.getElementById('routeStats').style.display = 'block';
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
                
                // Notify app that location was set
                if (window.perfect10kApp) {
                    window.perfect10kApp.setLocation(lat, lon);
                }
            },
            (error) => {
                console.warn('Location access failed:', error);
            },
            { timeout: 10000, enableHighAccuracy: true }
        );
    }
    
    /**
     * Set map location and update inputs
     */
    setLocationAndUnblur(lat, lon) {
        this.map.setView([lat, lon], 15);
        
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
    
}

// Initialize the interactive map when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.interactiveMap = new InteractiveMapEditor('map');
});