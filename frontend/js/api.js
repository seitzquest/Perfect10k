/**
 * API Client for Perfect10k Backend
 * Handles all communication with the 4-step route planning API
 */

class ApiClient {
    constructor(baseUrl = '') {
        // Auto-detect backend URL based on current protocol and host
        if (!baseUrl) {
            const protocol = window.location.protocol;
            const hostname = window.location.hostname;
            const port = protocol === 'https:' ? '8000' : '8000';
            this.baseUrl = `${protocol}//${hostname}:${port}`;
        } else {
            this.baseUrl = baseUrl;
        }
        this.currentSession = null;
        this.networkStatus = 'online';
        this.requestQueue = [];
        this.retryAttempts = 3;
        this.retryDelay = 1000;
        
        // Set up network status monitoring
        this.setupNetworkMonitoring();
    }
    
    /**
     * Start a new interactive routing session
     */
    async startSession(lat, lon, preference = "scenic parks and nature", targetDistance = 8000) {
        const requestData = {
            lat: lat,
            lon: lon,
            preference: preference,
            target_distance: targetDistance
        };
        
        try {
            const response = await this.makeRequest('/api/start-session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            // Store session ID
            this.currentSession = response.session_id;
            
            return response;
            
        } catch (error) {
            console.error('Session start failed:', error);
            throw new Error(`Session start failed: ${error.message}`);
        }
    }
    
    /**
     * Add a waypoint to the current route
     */
    async addWaypoint(nodeId) {
        if (!this.currentSession) {
            throw new Error('No active session');
        }
        
        const requestData = {
            session_id: this.currentSession,
            node_id: nodeId
        };
        
        try {
            const response = await this.makeRequest('/api/add-waypoint', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            return response;
            
        } catch (error) {
            console.error('Add waypoint failed:', error);
            throw new Error(`Add waypoint failed: ${error.message}`);
        }
    }
    
    /**
     * Finalize the route with a destination
     */
    async finalizeRoute(finalNodeId) {
        if (!this.currentSession) {
            throw new Error('No active session');
        }
        
        const requestData = {
            session_id: this.currentSession,
            final_node_id: finalNodeId
        };
        
        try {
            const response = await this.makeRequest('/api/finalize-route', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            return response;
            
        } catch (error) {
            console.error('Finalize route failed:', error);
            throw new Error(`Finalize route failed: ${error.message}`);
        }
    }
    
    /**
     * Get current route status
     */
    async getRouteStatus() {
        if (!this.currentSession) {
            throw new Error('No active session');
        }
        
        try {
            const response = await this.makeRequest(`/api/route-status/${this.currentSession}`);
            return response;
            
        } catch (error) {
            console.error('Get route status failed:', error);
            throw new Error(`Get route status failed: ${error.message}`);
        }
    }
    
    /**
     * Plan a route using the 4-step algorithm (LEGACY - for backwards compatibility)
     */
    async planRoute(request) {
        const requestData = {
            lat: request.lat,
            lon: request.lon,
            preference: request.preference || "scenic parks and nature",
            target_distance: request.targetDistance || 8000,
            algorithm: "4-step-convex-hull",
            length_preference: request.lengthPreference || "exact",
            debug_mode: request.debugMode || false
        };
        
        try {
            const response = await this.makeRequest('/api/plan-route', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            // Store session ID if provided
            if (response.session_id) {
                this.currentSession = response.session_id;
            }
            
            return response;
            
        } catch (error) {
            console.error('Route planning failed:', error);
            throw new Error(`Route planning failed: ${error.message}`);
        }
    }
    
    /**
     * Edit a node in the current route
     */
    async editNode(operation, nodeId, position = null) {
        if (!this.currentSession) {
            throw new Error('No active route session');
        }
        
        const requestData = {
            operation: operation, // 'add' or 'remove'
            node_id: nodeId,
            route_id: this.currentSession,
            position: position // For add operations, where to insert
        };
        
        try {
            const response = await this.makeRequest('/api/edit-node', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            return response;
            
        } catch (error) {
            console.error('Node edit failed:', error);
            throw new Error(`Node edit failed: ${error.message}`);
        }
    }
    
    /**
     * Get available nodes near a location for adding to route
     */
    async getAvailableNodes(lat, lon, radius = 200) {
        if (!this.currentSession) {
            throw new Error('No active route session');
        }
        
        try {
            const response = await this.makeRequest(
                `/api/available-nodes/${this.currentSession}?lat=${lat}&lon=${lon}&radius=${radius}`
            );
            
            return response;
            
        } catch (error) {
            console.error('Failed to get available nodes:', error);
            throw new Error(`Failed to get nodes: ${error.message}`);
        }
    }
    
    /**
     * Get nearby nodes for the current route (for edit mode)
     */
    async getNearbyNodes(radius = 200) {
        if (!this.currentSession) {
            throw new Error('No active route session');
        }
        
        try {
            const response = await this.makeRequest(
                `/api/nearby-nodes/${this.currentSession}?radius=${radius}`
            );
            
            return response;
            
        } catch (error) {
            console.error('Failed to get nearby nodes:', error);
            throw new Error(`Failed to get nearby nodes: ${error.message}`);
        }
    }
    
    /**
     * Find nearest OSM node to a specific location
     */
    async findNearestNode(lat, lng, radius = 100) {
        if (!this.currentSession) {
            throw new Error('No active route session');
        }
        
        try {
            const response = await this.makeRequest(
                `/api/find-nearest-node/${this.currentSession}?lat=${lat}&lon=${lng}&radius=${radius}`
            );
            
            return response;
            
        } catch (error) {
            console.error('Failed to find nearest node:', error);
            throw new Error(`Failed to find nearest node: ${error.message}`);
        }
    }
    
    /**
     * Get detailed route information including full network path
     */
    async getRouteDetails() {
        if (!this.currentSession) {
            throw new Error('No active route session');
        }
        
        try {
            const response = await this.makeRequest(`/api/route-details/${this.currentSession}`);
            return response;
            
        } catch (error) {
            console.error('Failed to get route details:', error);
            throw new Error(`Route details failed: ${error.message}`);
        }
    }
    
    /**
     * Get visualization data for the current session
     */
    async getVisualizationData() {
        if (!this.currentSession) {
            throw new Error('No active route session');
        }
        
        try {
            const response = await this.makeRequest(`/api/visualization/${this.currentSession}`);
            return response;
            
        } catch (error) {
            console.error('Failed to get visualization data:', error);
            throw new Error(`Visualization data failed: ${error.message}`);
        }
    }
    
    /**
     * Get health status of the backend
     */
    async getHealthStatus() {
        try {
            const response = await this.makeRequest('/health');
            return response;
            
        } catch (error) {
            console.error('Health check failed:', error);
            return { status: 'unhealthy', error: error.message };
        }
    }
    
    /**
     * Get semantic overlays for specified area and feature types
     */
    async getSemanticOverlays(lat, lon, radiusKm = 2.0, featureTypes = ["forests", "rivers", "lakes"], useCache = true) {
        const requestData = {
            lat: lat,
            lon: lon,
            radius_km: radiusKm,
            feature_types: featureTypes,
            use_cache: useCache
        };
        
        try {
            const response = await this.makeRequest('/api/semantic-overlays', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            return response;
            
        } catch (error) {
            console.error('Get semantic overlays failed:', error);
            throw new Error(`Get semantic overlays failed: ${error.message}`);
        }
    }
    
    /**
     * Get single semantic overlay for a specific feature type
     */
    async getSingleSemanticOverlay(featureType, lat, lon, radiusKm = 2.0, useCache = true) {
        try {
            const response = await this.makeRequest(
                `/api/semantic-overlays/${featureType}?lat=${lat}&lon=${lon}&radius_km=${radiusKm}&use_cache=${useCache}`
            );
            
            return response;
            
        } catch (error) {
            console.error(`Get ${featureType} overlay failed:`, error);
            throw new Error(`Get ${featureType} overlay failed: ${error.message}`);
        }
    }
    
    /**
     * Get information about available semantic overlay types
     */
    async getSemanticOverlaysInfo() {
        try {
            const response = await this.makeRequest('/api/semantic-overlays-info');
            return response;
            
        } catch (error) {
            console.error('Get semantic overlays info failed:', error);
            throw new Error(`Get semantic overlays info failed: ${error.message}`);
        }
    }
    
    /**
     * Clear semantic overlay cache
     */
    async clearSemanticOverlayCache(olderThanHours = null) {
        const requestData = olderThanHours ? { older_than_hours: olderThanHours } : {};
        
        try {
            const response = await this.makeRequest('/api/semantic-overlays/clear-cache', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            return response;
            
        } catch (error) {
            console.error('Clear semantic overlay cache failed:', error);
            throw new Error(`Clear semantic overlay cache failed: ${error.message}`);
        }
    }
    
    /**
     * Core request method with retry logic and error handling
     */
    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        
        for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
            try {
                const response = await fetch(url, {
                    ...options,
                    timeout: 30000 // 30 second timeout
                });
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                this.updateNetworkStatus('online');
                return data;
                
            } catch (error) {
                console.warn(`Request attempt ${attempt} failed:`, error.message);
                
                if (attempt === this.retryAttempts) {
                    this.updateNetworkStatus('offline');
                    throw error;
                }
                
                // Wait before retry
                await this.delay(this.retryDelay * attempt);
            }
        }
    }
    
    /**
     * Set up network status monitoring
     */
    setupNetworkMonitoring() {
        // Listen for online/offline events
        window.addEventListener('online', () => {
            this.updateNetworkStatus('online');
            this.processQueuedRequests();
        });
        
        window.addEventListener('offline', () => {
            this.updateNetworkStatus('offline');
        });
        
        // Periodic health check
        setInterval(() => {
            if (navigator.onLine) {
                this.getHealthStatus().then(status => {
                    this.updateNetworkStatus(status.status === 'healthy' ? 'online' : 'degraded');
                }).catch(() => {
                    this.updateNetworkStatus('offline');
                });
            }
        }, 30000); // Check every 30 seconds
    }
    
    /**
     * Update network status and UI indicator
     */
    updateNetworkStatus(status) {
        this.networkStatus = status;
        
        const statusElement = document.getElementById('networkStatus');
        if (statusElement) {
            statusElement.textContent = status === 'online' ? '●' : 
                                      status === 'degraded' ? '◐' : '○';
            statusElement.className = `network-status ${status}`;
            statusElement.title = `Network: ${status}`;
        }
        
        // Dispatch custom event for other components
        window.dispatchEvent(new CustomEvent('networkStatusChanged', {
            detail: { status }
        }));
    }
    
    /**
     * Queue requests when offline
     */
    queueRequest(request) {
        this.requestQueue.push(request);
    }
    
    /**
     * Process queued requests when back online
     */
    async processQueuedRequests() {
        if (this.networkStatus !== 'online' || this.requestQueue.length === 0) {
            return;
        }
        
        const queue = [...this.requestQueue];
        this.requestQueue = [];
        
        for (const request of queue) {
            try {
                await request();
            } catch (error) {
                console.error('Queued request failed:', error);
                // Re-queue failed requests
                this.requestQueue.push(request);
            }
        }
    }
    
    /**
     * Utility method for delays
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    /**
     * Get current session information
     */
    getSessionInfo() {
        return {
            sessionId: this.currentSession,
            networkStatus: this.networkStatus,
            queuedRequests: this.requestQueue.length
        };
    }
    
    /**
     * Clear current session
     */
    clearSession() {
        this.currentSession = null;
    }
}

// Create global API client instance
window.apiClient = new ApiClient();

// Export for module usage if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ApiClient;
}