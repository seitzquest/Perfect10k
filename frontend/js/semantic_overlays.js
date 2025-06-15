/**
 * Semantic Overlays Manager for Perfect10k
 * Handles display and management of forest, river, and lake overlays from OSM data
 */

class SemanticOverlaysManager {
    constructor(map, apiClient) {
        this.map = map;
        this.apiClient = apiClient;
        
        // Layer groups for each overlay type
        this.overlayLayers = {
            forests: L.layerGroup(),
            rivers: L.layerGroup(),
            lakes: L.layerGroup()
        };
        
        // Track overlay visibility state
        this.overlayStates = {
            forests: false,
            rivers: false,
            lakes: false
        };
        
        // Cache for overlay data to avoid repeated requests
        this.overlayCache = new Map();
        
        // Current map bounds for overlay loading
        this.currentBounds = null;
        this.loadingOverlays = new Set();
        
        // Configuration for each overlay type
        this.overlayConfigs = {
            forests: {
                name: 'Forests',
                icon: '🌲',
                description: 'Parks, forests, and wooded areas',
                defaultStyle: {
                    color: '#228B22',
                    fillColor: '#32CD32',
                    fillOpacity: 0.3,
                    weight: 1
                }
            },
            rivers: {
                name: 'Rivers',
                icon: '🌊',
                description: 'Rivers, streams, and waterways',
                defaultStyle: {
                    color: '#0077BE',
                    weight: 2,
                    opacity: 0.8
                }
            },
            lakes: {
                name: 'Lakes',
                icon: '🏞️',
                description: 'Lakes, ponds, and water bodies',
                defaultStyle: {
                    color: '#0077BE',
                    fillColor: '#87CEEB',
                    fillOpacity: 0.4,
                    weight: 1
                }
            }
        };
        
        this.initialize();
    }
    
    /**
     * Initialize the overlays manager
     */
    initialize() {
        // Add layer groups to map (initially empty and hidden)
        Object.values(this.overlayLayers).forEach(layerGroup => {
            layerGroup.addTo(this.map);
        });
        
        // Set up event handlers
        this.setupEventHandlers();
        
        console.log('Semantic overlays manager initialized');
    }
    
    /**
     * Set up map event handlers
     */
    setupEventHandlers() {
        // Listen for map movement to load overlays for new areas
        this.map.on('moveend', () => {
            this.onMapMoved();
        });
        
        // Listen for zoom changes to adjust overlay visibility
        this.map.on('zoomend', () => {
            this.onMapZoomed();
        });
    }
    
    /**
     * Handle map movement - load overlays for new visible area
     */
    async onMapMoved() {
        const newBounds = this.map.getBounds();
        
        // Only reload if we've moved significantly
        if (this.shouldReloadOverlays(newBounds)) {
            this.currentBounds = newBounds;
            await this.loadVisibleOverlays();
        }
    }
    
    /**
     * Handle map zoom changes
     */
    onMapZoomed() {
        const zoom = this.map.getZoom();
        
        // Hide overlays on very low zoom levels for performance
        if (zoom < 12) {
            this.hideAllOverlays();
        } else {
            // Restore previously enabled overlays
            this.restoreVisibleOverlays();
        }
    }
    
    /**
     * Check if overlays should be reloaded based on map movement
     */
    shouldReloadOverlays(newBounds) {
        if (!this.currentBounds) {
            return true;
        }
        
        // Calculate if we've moved more than 50% outside current bounds
        const currentCenter = this.currentBounds.getCenter();
        const newCenter = newBounds.getCenter();
        const distance = currentCenter.distanceTo(newCenter);
        
        // Reload if moved more than 1km
        return distance > 1000;
    }
    
    /**
     * Toggle overlay visibility
     */
    async toggleOverlay(overlayType) {
        if (!this.overlayConfigs[overlayType]) {
            console.error(`Unknown overlay type: ${overlayType}`);
            return;
        }
        
        const isCurrentlyVisible = this.overlayStates[overlayType];
        
        if (isCurrentlyVisible) {
            // Hide overlay
            this.hideOverlay(overlayType);
        } else {
            // Show overlay - load data if needed
            await this.showOverlay(overlayType);
        }
        
        // Update UI toggle state
        this.updateToggleUI(overlayType, !isCurrentlyVisible);
    }
    
    /**
     * Show a specific overlay
     */
    async showOverlay(overlayType) {
        if (this.loadingOverlays.has(overlayType)) {
            return; // Already loading
        }
        
        try {
            this.loadingOverlays.add(overlayType);
            this.showLoadingIndicator(overlayType);
            
            // Load overlay data
            await this.loadOverlayData(overlayType);
            
            // Show the layer
            this.overlayLayers[overlayType].addTo(this.map);
            this.overlayStates[overlayType] = true;
            
            console.log(`${overlayType} overlay shown`);
            
        } catch (error) {
            console.error(`Failed to show ${overlayType} overlay:`, error);
            this.showErrorMessage(`Failed to load ${overlayType} overlay`);
        } finally {
            this.loadingOverlays.delete(overlayType);
            this.hideLoadingIndicator(overlayType);
        }
    }
    
    /**
     * Hide a specific overlay
     */
    hideOverlay(overlayType) {
        this.map.removeLayer(this.overlayLayers[overlayType]);
        this.overlayStates[overlayType] = false;
        console.log(`${overlayType} overlay hidden`);
    }
    
    /**
     * Load overlay data from backend
     */
    async loadOverlayData(overlayType) {
        const center = this.map.getCenter();
        const zoom = this.map.getZoom();
        
        // Calculate appropriate radius based on zoom level
        const radius = this.calculateRadiusForZoom(zoom);
        
        // Check cache first
        const cacheKey = `${overlayType}_${center.lat.toFixed(4)}_${center.lng.toFixed(4)}_${radius}`;
        
        if (this.overlayCache.has(cacheKey)) {
            const cachedData = this.overlayCache.get(cacheKey);
            this.renderOverlayData(overlayType, cachedData);
            return;
        }
        
        // Fetch from backend
        const response = await this.apiClient.getSingleSemanticOverlay(
            overlayType, 
            center.lat, 
            center.lng, 
            radius
        );
        
        if (response.success && response.data) {
            // Cache the data
            this.overlayCache.set(cacheKey, response.data);
            
            // Render on map
            this.renderOverlayData(overlayType, response.data);
        } else {
            throw new Error('Invalid response from backend');
        }
    }
    
    /**
     * Render overlay data on the map
     */
    renderOverlayData(overlayType, data) {
        const layerGroup = this.overlayLayers[overlayType];
        const config = this.overlayConfigs[overlayType];
        
        // Clear existing data
        layerGroup.clearLayers();
        
        if (!data.features || data.features.length === 0) {
            console.log(`No ${overlayType} features found in current area`);
            return;
        }
        
        // Get style configuration
        const style = data.style || config.defaultStyle;
        
        // Create GeoJSON layer
        const geoJsonLayer = L.geoJSON(data, {
            style: (feature) => ({
                ...style,
                // Add interactive styling
                className: `semantic-overlay-${overlayType}`
            }),
            onEachFeature: (feature, layer) => {
                this.setupFeatureInteraction(layer, feature, overlayType);
            }
        });
        
        // Add to layer group
        layerGroup.addLayer(geoJsonLayer);
        
        console.log(`Rendered ${data.features.length} ${overlayType} features`);
    }
    
    /**
     * Set up feature interaction (popups, etc.)
     */
    setupFeatureInteraction(layer, feature, overlayType) {
        const config = this.overlayConfigs[overlayType];
        const properties = feature.properties || {};
        const tags = properties.tags || {};
        
        // Get the original style for this layer
        const originalStyle = layer.options || config.defaultStyle;
        
        // Create popup content
        let popupContent = `
            <div class="overlay-popup">
                <h4>${config.icon} ${config.name}</h4>
        `;
        
        // Add feature name if available
        if (tags.name) {
            popupContent += `<p><strong>Name:</strong> ${tags.name}</p>`;
        }
        
        // Add type-specific information
        if (overlayType === 'forests') {
            if (tags.leisure) popupContent += `<p><strong>Type:</strong> ${tags.leisure}</p>`;
            if (tags.natural) popupContent += `<p><strong>Natural:</strong> ${tags.natural}</p>`;
        } else if (overlayType === 'rivers') {
            if (tags.waterway) popupContent += `<p><strong>Waterway:</strong> ${tags.waterway}</p>`;
            if (tags.width) popupContent += `<p><strong>Width:</strong> ${tags.width}</p>`;
        } else if (overlayType === 'lakes') {
            if (tags.water) popupContent += `<p><strong>Type:</strong> ${tags.water}</p>`;
            if (tags.natural) popupContent += `<p><strong>Natural:</strong> ${tags.natural}</p>`;
        }
        
        popupContent += `</div>`;
        
        // Bind popup
        layer.bindPopup(popupContent);
        
        // Add hover effects
        layer.on('mouseover', () => {
            layer.setStyle({
                weight: originalStyle.weight ? originalStyle.weight + 1 : 2,
                opacity: 1,
                fillOpacity: originalStyle.fillOpacity ? Math.min(originalStyle.fillOpacity + 0.2, 1) : 0.6
            });
        });
        
        layer.on('mouseout', () => {
            layer.setStyle(originalStyle);
        });
    }
    
    /**
     * Calculate appropriate radius based on zoom level
     */
    calculateRadiusForZoom(zoom) {
        if (zoom >= 16) return 0.5;   // Very detailed view
        if (zoom >= 14) return 1.0;   // Neighborhood view
        if (zoom >= 12) return 2.0;   // District view
        if (zoom >= 10) return 4.0;   // City view
        return 8.0;                   // Regional view
    }
    
    /**
     * Load all currently visible overlays
     */
    async loadVisibleOverlays() {
        const promises = [];
        
        for (const [overlayType, isVisible] of Object.entries(this.overlayStates)) {
            if (isVisible && !this.loadingOverlays.has(overlayType)) {
                promises.push(this.loadOverlayData(overlayType));
            }
        }
        
        if (promises.length > 0) {
            try {
                await Promise.all(promises);
            } catch (error) {
                console.error('Failed to load visible overlays:', error);
            }
        }
    }
    
    /**
     * Hide all overlays (used during low zoom)
     */
    hideAllOverlays() {
        Object.keys(this.overlayLayers).forEach(overlayType => {
            this.map.removeLayer(this.overlayLayers[overlayType]);
        });
    }
    
    /**
     * Restore previously visible overlays
     */
    restoreVisibleOverlays() {
        Object.entries(this.overlayStates).forEach(([overlayType, isVisible]) => {
            if (isVisible) {
                this.overlayLayers[overlayType].addTo(this.map);
            }
        });
    }
    
    /**
     * Update toggle UI state
     */
    updateToggleUI(overlayType, isVisible) {
        const toggleButton = document.getElementById(`overlay-toggle-${overlayType}`);
        if (toggleButton) {
            toggleButton.classList.toggle('active', isVisible);
            toggleButton.setAttribute('aria-pressed', isVisible.toString());
        }
        
        // Update checkbox if present
        const checkbox = document.getElementById(`overlay-checkbox-${overlayType}`);
        if (checkbox) {
            checkbox.checked = isVisible;
        }
    }
    
    /**
     * Show loading indicator for overlay
     */
    showLoadingIndicator(overlayType) {
        const toggleButton = document.getElementById(`overlay-toggle-${overlayType}`);
        if (toggleButton) {
            toggleButton.classList.add('loading');
            toggleButton.disabled = true;
        }
    }
    
    /**
     * Hide loading indicator for overlay
     */
    hideLoadingIndicator(overlayType) {
        const toggleButton = document.getElementById(`overlay-toggle-${overlayType}`);
        if (toggleButton) {
            toggleButton.classList.remove('loading');
            toggleButton.disabled = false;
        }
    }
    
    /**
     * Show error message
     */
    showErrorMessage(message) {
        // Integration with existing app messaging system
        if (window.interactiveMap && typeof window.interactiveMap.showMessage === 'function') {
            window.interactiveMap.showMessage(message, 'error');
        } else {
            console.error(message);
        }
    }
    
    /**
     * Get overlay statistics
     */
    getOverlayStats() {
        const stats = {};
        
        Object.entries(this.overlayLayers).forEach(([overlayType, layerGroup]) => {
            let featureCount = 0;
            layerGroup.eachLayer(layer => {
                if (layer instanceof L.GeoJSON) {
                    layer.eachLayer(() => featureCount++);
                }
            });
            
            stats[overlayType] = {
                visible: this.overlayStates[overlayType],
                featureCount: featureCount,
                loading: this.loadingOverlays.has(overlayType)
            };
        });
        
        return {
            overlayStats: stats,
            cacheSize: this.overlayCache.size,
            currentBounds: this.currentBounds
        };
    }
    
    /**
     * Clear overlay cache
     */
    clearCache() {
        this.overlayCache.clear();
        console.log('Overlay cache cleared');
    }
    
    /**
     * Refresh all visible overlays
     */
    async refreshVisibleOverlays() {
        this.clearCache();
        await this.loadVisibleOverlays();
    }
    
    /**
     * Get configuration for a specific overlay type
     */
    getOverlayConfig(overlayType) {
        return this.overlayConfigs[overlayType];
    }
    
    /**
     * Get all available overlay types
     */
    getAvailableOverlayTypes() {
        return Object.keys(this.overlayConfigs);
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SemanticOverlaysManager;
}