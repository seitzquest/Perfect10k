// Map functionality for Perfect10k

class MapManager {
    constructor() {
        this.map = null;
        this.currentRoute = null;
        this.routeLayer = null;
        this.markersLayer = null;
        this.selectedLocation = null;
        this.isPlanning = false;
        
        this.init();
    }

    init() {
        this.initializeMap();
        this.setupEventListeners();
    }

    initializeMap() {
        // Initialize Leaflet map
        this.map = L.map('map').setView([49.807880, 8.989109], 13); // Default to Odenwald

        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 18
        }).addTo(this.map);

        // Create layer groups
        this.markersLayer = L.layerGroup().addTo(this.map);
        this.routeLayer = L.layerGroup().addTo(this.map);

        // Add map click handler
        this.map.on('click', (e) => this.handleMapClick(e));

        // Add loading indicator
        this.showMapLoading(false);
    }

    setupEventListeners() {
        // Use current location button
        document.getElementById('useCurrentLocation')?.addEventListener('click', () => {
            this.useCurrentLocation();
        });

        // Location input
        const locationInput = document.getElementById('locationInput');
        if (locationInput) {
            locationInput.addEventListener('input', utils.debounce((e) => {
                this.searchLocation(e.target.value);
            }, 500));
        }
    }

    async useCurrentLocation() {
        const button = document.getElementById('useCurrentLocation');
        
        try {
            utils.setLoading(button, true);
            
            const position = await utils.getCurrentPosition();
            const lat = position.coords.latitude;
            const lng = position.coords.longitude;
            
            this.setLocation(lat, lng);
            utils.toast.show('Current location set', 'success');
            
        } catch (error) {
            console.error('Geolocation error:', error);
            
            let message = 'Could not get your location';
            if (error.code === 1) {
                message = 'Location access denied. Please enable location services.';
            } else if (error.code === 2) {
                message = 'Location unavailable. Please try again.';
            } else if (error.code === 3) {
                message = 'Location request timed out. Please try again.';
            }
            
            utils.toast.show(message, 'error');
        } finally {
            utils.setLoading(button, false);
        }
    }

    setLocation(lat, lng, address = null) {
        this.selectedLocation = { lat, lng, address };
        
        // Update map view
        this.map.setView([lat, lng], 15);
        
        // Clear existing markers
        this.markersLayer.clearLayers();
        
        // Add location marker
        const marker = this.createCustomMarker([lat, lng], 'start', '📍');
        this.markersLayer.addLayer(marker);
        
        // Update input
        const locationInput = document.getElementById('locationInput');
        const coordsDisplay = document.getElementById('coordsDisplay');
        const coordinates = document.getElementById('locationCoordinates');
        
        if (locationInput) {
            locationInput.value = address || `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
        }
        
        if (coordsDisplay) {
            coordsDisplay.textContent = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
        }
        
        if (coordinates) {
            coordinates.classList.remove('hidden');
        }
    }

    handleMapClick(e) {
        const lat = e.latlng.lat;
        const lng = e.latlng.lng;
        
        this.setLocation(lat, lng);
        
        // Reverse geocoding could be added here
        // For now, we'll just use coordinates
    }

    async searchLocation(query) {
        if (!query || query.length < 3) return;

        try {
            // Use Nominatim for geocoding (you might want to use a different service)
            const response = await fetch(
                `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=5`
            );
            
            const results = await response.json();
            
            if (results.length > 0) {
                const first = results[0];
                const lat = parseFloat(first.lat);
                const lng = parseFloat(first.lon);
                
                this.setLocation(lat, lng, first.display_name);
            }
            
        } catch (error) {
            console.warn('Geocoding failed:', error);
        }
    }

    createCustomMarker(latlng, type, icon) {
        const iconHtml = `
            <div class="custom-marker marker-${type}">
                ${icon}
            </div>
        `;
        
        const customIcon = L.divIcon({
            html: iconHtml,
            className: 'custom-marker-container',
            iconSize: [32, 32],
            iconAnchor: [16, 16]
        });
        
        return L.marker(latlng, { icon: customIcon });
    }

    showMapLoading(show = true) {
        const loading = document.getElementById('mapLoading');
        if (loading) {
            loading.classList.toggle('hidden', !show);
        }
    }

    showPlanningOverlay(show = true) {
        let overlay = document.querySelector('.route-planning-overlay');
        
        if (show && !overlay) {
            overlay = document.createElement('div');
            overlay.className = 'route-planning-overlay';
            overlay.innerHTML = `
                <div class="planning-animation"></div>
                <p class="planning-text">Planning your perfect route...</p>
                <p class="planning-subtext">Using AI to find places you'll love</p>
            `;
            document.getElementById('map').appendChild(overlay);
        } else if (!show && overlay) {
            overlay.remove();
        }
    }

    async planRoute(preferences) {
        if (!this.selectedLocation) {
            utils.toast.show('Please select a starting location', 'warning');
            return null;
        }

        if (!auth.requireAuth()) {
            return null;
        }

        this.isPlanning = true;
        this.showPlanningOverlay(true);

        try {
            const routeRequest = {
                latitude: this.selectedLocation.lat,
                longitude: this.selectedLocation.lng,
                targetDistance: parseFloat(preferences.distance) * 1000, // Convert to meters
                tolerance: parseFloat(preferences.tolerance) * 1000,
                preferenceQuery: preferences.query || null,
                avoidRoads: preferences.avoidRoads
            };

            const route = await api.planRoute(routeRequest);
            
            this.displayRoute(route);
            return route;

        } catch (error) {
            utils.handleError(error, 'route planning');
            return null;
        } finally {
            this.isPlanning = false;
            this.showPlanningOverlay(false);
        }
    }

    displayRoute(route) {
        this.currentRoute = route;
        
        // Clear existing route
        this.routeLayer.clearLayers();
        
        if (!route.pathCoordinates || route.pathCoordinates.length === 0) {
            utils.toast.show('No route coordinates received', 'warning');
            return;
        }

        // Convert coordinates format
        const latLngs = route.pathCoordinates.map(coord => [coord[0], coord[1]]);
        
        // Create route polyline
        const routeLine = L.polyline(latLngs, {
            color: '#0ea5e9',
            weight: 4,
            opacity: 0.8,
            smoothFactor: 1
        });
        
        this.routeLayer.addLayer(routeLine);
        
        // Add start/end markers
        if (latLngs.length > 0) {
            const startMarker = this.createCustomMarker(latLngs[0], 'start', '🏁');
            const endMarker = this.createCustomMarker(latLngs[latLngs.length - 1], 'end', '🏁');
            
            this.markersLayer.addLayer(startMarker);
            this.markersLayer.addLayer(endMarker);
        }
        
        // Add place markers
        if (route.matchedPlaces) {
            route.matchedPlaces.forEach((place, index) => {
                const placeMarker = this.createCustomMarker(
                    [place.latitude, place.longitude], 
                    'place', 
                    this.getPlaceIcon(place.type)
                );
                
                const popupContent = `
                    <div class="popup-header">${place.name}</div>
                    <div class="popup-type">${place.type}</div>
                    <div class="popup-description">Similarity: ${Math.round(place.similarityScore * 100)}%</div>
                `;
                
                placeMarker.bindPopup(popupContent);
                this.markersLayer.addLayer(placeMarker);
            });
        }
        
        // Fit map to route bounds
        this.map.fitBounds(routeLine.getBounds(), { padding: [20, 20] });
        
        utils.toast.show('Route planned successfully!', 'success');
    }

    getPlaceIcon(placeType) {
        const icons = {
            park: '🌳',
            lake: '🏞️',
            river: '🌊',
            cafe: '☕',
            restaurant: '🍽️',
            museum: '🏛️',
            library: '📚',
            playground: '🛝',
            beach: '🏖️',
            mountain: '⛰️',
            trail: '🥾',
            garden: '🌺',
            viewpoint: '👁️',
            bridge: '🌉',
            historic_site: '🏛️',
            shopping_center: '🛍️',
            church: '⛪',
            stadium: '🏟️',
            hospital: '🏥'
        };
        
        return icons[placeType] || '📍';
    }

    clearRoute() {
        this.currentRoute = null;
        this.routeLayer.clearLayers();
        
        // Keep location marker but remove route-specific markers
        this.markersLayer.clearLayers();
        
        if (this.selectedLocation) {
            const marker = this.createCustomMarker(
                [this.selectedLocation.lat, this.selectedLocation.lng], 
                'start', 
                '📍'
            );
            this.markersLayer.addLayer(marker);
        }
    }

    async exportGPX() {
        if (!this.currentRoute) {
            utils.toast.show('No route to export', 'warning');
            return;
        }

        try {
            const blob = await api.exportRouteGPX(this.currentRoute.id);
            
            // Create download link
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `perfect10k-route-${this.currentRoute.id}.gpx`;
            
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            utils.toast.show('Route exported successfully', 'success');
            
        } catch (error) {
            utils.handleError(error, 'GPX export');
        }
    }

    getSelectedLocation() {
        return this.selectedLocation;
    }

    getCurrentRoute() {
        return this.currentRoute;
    }
}

// Initialize map manager
const mapManager = new MapManager();

// Export for use in other scripts
window.mapManager = mapManager;