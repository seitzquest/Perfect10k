/**
 * API Client for Perfect10k Backend
 * Handles all communication with the 4-step route planning API
 */

class ApiClient {
    constructor(baseUrl = '') {
        if (!baseUrl) {
            // Always use relative paths - backend serves frontend and handles API
            // In development: backend serves frontend on :8000
            // In production: nginx proxies everything to backend on :8000
            this.baseUrl = '';
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
     * Start a new interactive routing session with intelligent async/sync routing
     */
    async startSession(lat, lon, preference = "scenic parks and nature", targetDistance = 8000) {
        try {
            // Use hybrid async endpoint that automatically chooses sync for cache hits
            console.log('Using hybrid async endpoint for optimal performance');
            try {
                return await this.startSessionAsync(lat, lon, preference, targetDistance);
            } catch (asyncError) {
                console.warn('Hybrid async endpoint failed, falling back to pure sync:', asyncError.message);
                // Fallback to sync version if hybrid async fails
                return await this.startSessionSync(lat, lon, preference, targetDistance);
            }
            
        } catch (error) {
            console.error('Session start failed:', error);
            throw error;
        }
    }
    
    /**
     * Start a new interactive routing session using async endpoint (for long-running requests)
     */
    async startSessionAsync(lat, lon, preference = "scenic parks and nature", targetDistance = 8000) {
        const requestData = {
            lat: lat,
            lon: lon,
            preference: preference,
            target_distance: targetDistance
        };
        
        try {
            // Use async endpoint for better timeout handling
            const asyncResponse = await this.makeRequest('/api/start-session-async', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData),
                timeout: 120000  // 2 minute timeout for async endpoint setup on slow hardware
            });
            
            if (asyncResponse.status === 'completed') {
                // Immediate response (either from cache hit or sync processing)
                this.currentSession = asyncResponse.job_id || asyncResponse.session_id;
                console.log(`✅ Immediate response received (${asyncResponse.response_type || 'cached'})`);
                return asyncResponse.result;
            } else if (asyncResponse.status === 'processing') {
                // Background job started, poll for completion
                if (window.loadingManager) {
                    window.loadingManager.showAsyncJobProgress(asyncResponse.job_id, {
                        title: "Loading Area Data",
                        description: `Processing ${preference} routes... This may take several minutes for new areas.`,
                        estimatedTime: asyncResponse.estimated_completion_ms || 120000
                    });
                }
                
                // Poll for completion
                const result = await this.pollJobUntilComplete(asyncResponse.job_id);
                if (result && result.session_id) {
                    this.currentSession = result.session_id;
                }
                return result;
            } else {
                throw new Error(`Unexpected async response status: ${asyncResponse.status}`);
            }
            
        } catch (error) {
            console.error('Async session start failed:', error);
            throw error;
        }
    }
    
    /**
     * Start a new interactive routing session synchronously (legacy/fallback)
     */
    async startSessionSync(lat, lon, preference = "scenic parks and nature", targetDistance = 8000) {
        const requestData = {
            lat: lat,
            lon: lon,
            preference: preference,
            target_distance: targetDistance
        };
        
        try {
            // Use loading manager if available
            if (window.loadingManager) {
                const response = await window.loadingManager.makeApiRequestWithLoading(
                    `${this.baseUrl}/api/start-session`, 
                    {
                        method: 'POST',
                        body: requestData,
                        minimalistic: true,
                        loadingOptions: {
                            title: "Finding Perfect Route",
                            description: `Analyzing natural features for ${preference} routes... This may take several minutes for new areas. The app will wait up to 30 minutes for first-time area loading.`
                        }
                    }
                );
                
                // Store session ID
                this.currentSession = response.session_id;
                
                // Log performance info for debugging
                if (response.performance) {
                    console.log('Route generation performance:', response.performance);
                }
                
                return response;
            } else {
                // Fallback to regular request if loading manager not available
                const response = await this.makeRequest('/api/start-session', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestData),
                    timeout: 1800000 // 30 minute timeout for first-time area loading
                });
                
                // Store session ID
                this.currentSession = response.session_id;
                
                return response;
            }
            
        } catch (error) {
            console.error('Session start failed:', error);
            throw new Error(`Session start failed: ${error.message}`);
        }
    }
    
    /**
     * Poll job status until completion with exponential backoff
     */
    async pollJobUntilComplete(jobId) {
        const maxAttempts = 300; // Reduced from 900 to prevent resource exhaustion
        let attempts = 0;
        let consecutiveErrors = 0;
        const maxConsecutiveErrors = 5;
        
        // Show async job progress if loading manager available
        if (window.loadingManager) {
            window.loadingManager.showAsyncJobProgress(jobId);
        }
        
        while (attempts < maxAttempts) {
            try {
                const status = await this.getJobStatus(jobId);
                
                // Reset error count on success
                consecutiveErrors = 0;
                
                // Update loading progress with job status
                if (window.loadingManager) {
                    window.loadingManager.updateJobProgress(status);
                }
                
                if (status.status === 'completed') {
                    window.loadingManager?.hideLoading();
                    window.loadingManager?.removeBackgroundNotification();
                    return status.result;
                } else if (status.status === 'failed') {
                    window.loadingManager?.hideLoading();
                    window.loadingManager?.removeBackgroundNotification();
                    throw new Error(status.error || 'Job failed');
                }
                
                // Adaptive polling with exponential backoff
                const baseDelay = 1000;
                const backoffFactor = Math.min(1.5, 1 + (attempts / 100)); // Gradual backoff
                const pollDelay = Math.min(baseDelay * backoffFactor, 5000); // Cap at 5 seconds
                
                await this.delay(pollDelay);
                attempts++;
                
            } catch (error) {
                console.error('Job polling error:', error);
                consecutiveErrors++;
                
                // Circuit breaker: fail fast after consecutive errors
                if (consecutiveErrors >= maxConsecutiveErrors) {
                    window.loadingManager?.hideLoading();
                    window.loadingManager?.removeBackgroundNotification();
                    throw new Error(`Job polling failed after ${maxConsecutiveErrors} consecutive errors: ${error.message}`);
                }
                
                // Update loading with error status
                if (window.loadingManager) {
                    window.loadingManager.updateJobProgress({
                        status: 'running',
                        current_phase: 'Retrying connection...',
                        progress: Math.min(90, (attempts / maxAttempts) * 100)
                    });
                }
                
                // Exponential backoff for errors
                const errorDelay = Math.min(2000 * Math.pow(1.5, consecutiveErrors), 10000); // Cap at 10 seconds
                await this.delay(errorDelay);
                attempts++;
            }
        }
        
        // Timeout
        window.loadingManager?.hideLoading();
        window.loadingManager?.removeBackgroundNotification();
        throw new Error('Job timed out after 5 minutes');
    }
    
    /**
     * Get job status
     */
    async getJobStatus(jobId) {
        try {
            const response = await this.makeRequest(`/api/job-status/${jobId}`);
            return response;
        } catch (error) {
            console.error('Get job status failed:', error);
            throw new Error(`Get job status failed: ${error.message}`);
        }
    }
    
    /**
     * Add a waypoint to the current route
     */
    async addWaypoint(nodeId) {
        if (!this.currentSession) {
            console.error('No active session found. Current session:', this.currentSession);
            throw new Error('No active session - please start a route first');
        }
        
        console.log('Adding waypoint with session:', this.currentSession);
        
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
     * Clear semantic candidate cache to get fresh probabilistic results
     */
    async clearSemanticCache() {
        try {
            // Use loading manager if available for visual feedback
            if (window.loadingManager) {
                const response = await window.loadingManager.makeApiRequestWithLoading(
                    `${this.baseUrl}/api/clear-semantic-cache`, 
                    {
                        method: 'POST',
                        loadingOptions: {
                            title: "Refreshing Candidate Cache",
                            description: "Clearing cached data for fresh route variations...",
                            showDetails: false
                        }
                    }
                );
                return response;
            } else {
                // Fallback without loading animation
                const response = await this.makeRequest('/api/clear-semantic-cache', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                return response;
            }
            
        } catch (error) {
            console.error('Clear semantic cache failed:', error);
            throw new Error(`Clear semantic cache failed: ${error.message}`);
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
     * Score multiple locations based on semantic feature proximity
     */
    async scoreLocationsSemantics(locations, propertyNames = ["forests", "rivers", "lakes"], ensureLoadedRadius = 2.0) {
        const requestData = {
            locations: locations,
            property_names: propertyNames,
            ensure_loaded_radius: ensureLoadedRadius
        };
        
        try {
            const response = await this.makeRequest('/api/semantic-scoring', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            return response;
            
        } catch (error) {
            console.error('Score locations semantics failed:', error);
            throw new Error(`Score locations semantics failed: ${error.message}`);
        }
    }
    
    /**
     * Score a single location based on semantic feature proximity
     */
    async scoreSingleLocationSemantics(lat, lon, propertyNames = ["forests", "rivers", "lakes"], ensureLoadedRadius = 2.0) {
        try {
            const response = await this.makeRequest(
                `/api/semantic-scoring/single?lat=${lat}&lon=${lon}&property_names=${propertyNames.join(',')}&ensure_loaded_radius=${ensureLoadedRadius}`
            );
            
            return response;
            
        } catch (error) {
            console.error('Score single location semantics failed:', error);
            throw new Error(`Score single location semantics failed: ${error.message}`);
        }
    }
    
    /**
     * Get semantic properties information
     */
    async getSemanticProperties() {
        try {
            const response = await this.makeRequest('/api/semantic-properties');
            return response;
            
        } catch (error) {
            console.error('Get semantic properties failed:', error);
            throw new Error(`Get semantic properties failed: ${error.message}`);
        }
    }
    
    /**
     * Update semantic property configuration
     */
    async updateSemanticProperty(propertyName, updates) {
        try {
            const response = await this.makeRequest(`/api/semantic-properties/${propertyName}/update`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(updates)
            });
            
            return response;
            
        } catch (error) {
            console.error(`Update semantic property ${propertyName} failed:`, error);
            throw new Error(`Update semantic property failed: ${error.message}`);
        }
    }
    
    /**
     * Core request method with retry logic and circuit breaker
     */
    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const timeout = options.timeout || 600000; // Reduced to 10 minutes to prevent resource exhaustion
        
        // Initialize circuit breaker state if not exists
        if (!this.circuitBreaker) {
            this.circuitBreaker = {
                failures: 0,
                lastFailureTime: 0,
                state: 'closed' // closed, open, half-open
            };
        }
        
        // Check circuit breaker
        const now = Date.now();
        if (this.circuitBreaker.state === 'open') {
            if (now - this.circuitBreaker.lastFailureTime > 60000) { // 1 minute cooldown
                this.circuitBreaker.state = 'half-open';
                console.log('Circuit breaker half-open, attempting request');
            } else {
                throw new Error('Circuit breaker is open. Service temporarily unavailable.');
            }
        }
        
        for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
            try {
                // Create abort controller for timeout
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), timeout);
                
                const response = await fetch(url, {
                    ...options,
                    signal: controller.signal
                });
                
                clearTimeout(timeoutId);
                
                if (!response.ok) {
                    let errorData = {};
                    try {
                        errorData = await response.json();
                    } catch (parseError) {
                        // Server returned non-JSON response (likely HTML error page)
                        if (response.status === 504) {
                            throw new Error(`Server timeout (504): The route generation is taking longer than expected. Please try again or try a different location.`);
                        }
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                this.updateNetworkStatus('online');
                
                // Reset circuit breaker on success
                this.circuitBreaker.failures = 0;
                this.circuitBreaker.state = 'closed';
                
                return data;
                
            } catch (error) {
                console.warn(`Request attempt ${attempt} failed:`, error.message);
                
                // Handle abort/timeout errors
                if (error.name === 'AbortError') {
                    throw new Error(`Request timeout: The server is taking longer than expected. For new areas, this can take several minutes. Please try the location again or try a different area.`);
                }
                
                // Update circuit breaker
                this.circuitBreaker.failures++;
                this.circuitBreaker.lastFailureTime = now;
                
                if (this.circuitBreaker.failures >= 3) {
                    this.circuitBreaker.state = 'open';
                    console.warn('Circuit breaker opened due to repeated failures');
                }
                
                if (attempt === this.retryAttempts) {
                    this.updateNetworkStatus('offline');
                    throw error;
                }
                
                // Exponential backoff for retries
                const backoffDelay = Math.min(this.retryDelay * Math.pow(2, attempt - 1), 10000);
                await this.delay(backoffDelay);
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