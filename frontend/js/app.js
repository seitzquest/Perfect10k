// Main application logic for Perfect10k

class Perfect10kApp {
    constructor() {
        this.currentRoute = null;
        this.preferences = [];
        this.routeHistory = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadInitialData();
        
        // Show welcome message
        if (auth.isAuthenticated) {
            this.loadUserData();
        }
    }

    setupEventListeners() {
        // Route planning form
        const routeForm = document.getElementById('routeForm');
        if (routeForm) {
            routeForm.addEventListener('submit', (e) => this.handleRoutePlanning(e));
        }

        // Route action buttons
        document.getElementById('exportGpxBtn')?.addEventListener('click', () => {
            mapManager.exportGPX();
        });

        document.getElementById('saveRouteBtn')?.addEventListener('click', () => {
            this.saveCurrentRoute();
        });

        document.getElementById('newRouteBtn')?.addEventListener('click', () => {
            this.startNewRoute();
        });

        // Preference form
        const preferenceForm = document.getElementById('preferenceForm');
        if (preferenceForm) {
            preferenceForm.addEventListener('submit', (e) => this.handleAddPreference(e));
        }

        // Page change events
        window.addEventListener('pageChange', (e) => {
            this.handlePageChange(e.detail.pageId);
        });

        // Auth state changes
        window.addEventListener('authStateChanged', () => {
            this.handleAuthStateChange();
        });
    }

    async loadInitialData() {
        try {
            // Load place types for future use
            if (auth.isAuthenticated) {
                this.placeTypes = await api.getPlaceTypes();
            }
        } catch (error) {
            console.warn('Failed to load initial data:', error);
        }
    }

    async loadUserData() {
        if (!auth.isAuthenticated) return;

        try {
            // Load user preferences
            await this.loadPreferences();
            
            // Load route history
            await this.loadRouteHistory();
            
        } catch (error) {
            console.warn('Failed to load user data:', error);
        }
    }

    async handleRoutePlanning(event) {
        event.preventDefault();

        if (!auth.requireAuth()) {
            return;
        }

        const form = event.target;
        const submitBtn = form.querySelector('button[type="submit"]');

        // Get form data
        const formData = utils.getFormData(form);
        const preferences = {
            distance: formData.distanceInput || 8,
            tolerance: formData.toleranceInput || 1,
            query: formData.preferencesInput?.trim() || '',
            avoidRoads: formData.avoidRoads === true
        };

        // Validate location
        if (!mapManager.getSelectedLocation()) {
            utils.toast.show('Please select a starting location on the map', 'warning');
            return;
        }

        try {
            utils.setLoading(submitBtn, true);

            // Plan route using map manager
            const route = await mapManager.planRoute(preferences);

            if (route) {
                this.currentRoute = route;
                this.showRouteResults(route);
            }

        } catch (error) {
            utils.handleError(error, 'route planning');
        } finally {
            utils.setLoading(submitBtn, false);
        }
    }

    showRouteResults(route) {
        const resultsContainer = document.getElementById('routeResults');
        const routeForm = document.getElementById('routeForm');

        if (resultsContainer) {
            // Update stats
            document.getElementById('routeDistance').textContent = utils.formatDistance(route.actualDistance);
            document.getElementById('routeElevation').textContent = route.elevationGain ? 
                `${Math.round(route.elevationGain)} m` : 'N/A';
            document.getElementById('routePlaces').textContent = route.matchedPlaces ? 
                route.matchedPlaces.length : '0';

            // Show matched places
            this.displayMatchedPlaces(route.matchedPlaces || []);

            // Show results and hide form
            resultsContainer.classList.remove('hidden');
            routeForm.classList.add('hidden');
        }
    }

    displayMatchedPlaces(places) {
        const container = document.getElementById('matchedPlaces');
        if (!container) return;

        if (places.length === 0) {
            container.innerHTML = '<p class="text-muted">No specific places matched your preferences</p>';
            return;
        }

        container.innerHTML = `
            <h4>Places You'll Love</h4>
            ${places.map(place => `
                <div class="place-item">
                    <div class="place-icon">${mapManager.getPlaceIcon(place.type)}</div>
                    <div class="place-info">
                        <div class="place-name">${place.name}</div>
                        <div class="place-type">${place.type.replace('_', ' ')}</div>
                    </div>
                    <div class="place-similarity">${Math.round(place.similarityScore * 100)}%</div>
                </div>
            `).join('')}
        `;
    }

    startNewRoute() {
        // Clear current route
        this.currentRoute = null;
        mapManager.clearRoute();

        // Show form and hide results
        document.getElementById('routeForm').classList.remove('hidden');
        document.getElementById('routeResults').classList.add('hidden');

        // Clear form
        const form = document.getElementById('routeForm');
        form.reset();
        
        // Reset distance and tolerance to defaults
        document.getElementById('distanceInput').value = '8';
        document.getElementById('toleranceInput').value = '1';
    }

    async saveCurrentRoute() {
        if (!this.currentRoute) {
            utils.toast.show('No route to save', 'warning');
            return;
        }

        if (!auth.requireAuth()) {
            return;
        }

        // Route is already saved by the backend when planning
        // Just show confirmation and refresh history
        utils.toast.show('Route saved successfully', 'success');
        
        // Refresh route history if on history page
        if (utils.pageManager.currentPage === 'history') {
            await this.loadRouteHistory();
        }
    }

    async handleAddPreference(event) {
        event.preventDefault();

        if (!auth.requireAuth()) {
            return;
        }

        const form = event.target;
        const submitBtn = form.querySelector('button[type="submit"]');
        const input = form.querySelector('#newPreferenceInput');

        const description = input.value.trim();
        if (!description) {
            utils.toast.show('Please enter a preference description', 'warning');
            return;
        }

        try {
            utils.setLoading(submitBtn, true);

            const preference = await api.createPreference(description);
            
            utils.toast.show('Preference added successfully', 'success');
            form.reset();
            
            // Refresh preferences list
            await this.loadPreferences();

        } catch (error) {
            utils.handleError(error, 'adding preference');
        } finally {
            utils.setLoading(submitBtn, false);
        }
    }

    async loadPreferences() {
        if (!auth.isAuthenticated) return;

        try {
            this.preferences = await api.getUserPreferences();
            this.displayPreferences();
        } catch (error) {
            console.warn('Failed to load preferences:', error);
        }
    }

    displayPreferences() {
        const container = document.getElementById('preferencesList');
        if (!container) return;

        if (this.preferences.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">💭</div>
                    <h3>No preferences yet</h3>
                    <p>Add your first preference to help us find places you'll love</p>
                </div>
            `;
            return;
        }

        container.innerHTML = this.preferences.map(pref => `
            <div class="preference-item" data-id="${pref.id}">
                <div class="item-header">
                    <div class="item-title">${pref.description}</div>
                    <div class="item-date">${utils.formatDate(pref.createdAt)}</div>
                </div>
                <div class="item-actions">
                    <button class="btn btn-small ${pref.isActive ? 'btn-success' : 'btn-secondary'}" 
                            onclick="app.togglePreference('${pref.id}')">
                        ${pref.isActive ? 'Active' : 'Inactive'}
                    </button>
                    <button class="btn btn-small btn-outline" 
                            onclick="app.editPreference('${pref.id}')">
                        Edit
                    </button>
                    <button class="btn btn-small btn-error" 
                            onclick="app.deletePreference('${pref.id}')">
                        Delete
                    </button>
                </div>
            </div>
        `).join('');
    }

    async togglePreference(preferenceId) {
        try {
            await api.togglePreference(preferenceId);
            await this.loadPreferences();
            utils.toast.show('Preference updated', 'success');
        } catch (error) {
            utils.handleError(error, 'updating preference');
        }
    }

    async deletePreference(preferenceId) {
        if (!confirm('Are you sure you want to delete this preference?')) {
            return;
        }

        try {
            await api.deletePreference(preferenceId);
            await this.loadPreferences();
            utils.toast.show('Preference deleted', 'success');
        } catch (error) {
            utils.handleError(error, 'deleting preference');
        }
    }

    async loadRouteHistory() {
        if (!auth.isAuthenticated) return;

        try {
            const history = await api.getRouteHistory(1, 20);
            this.routeHistory = history.routes;
            this.displayRouteHistory();
        } catch (error) {
            console.warn('Failed to load route history:', error);
        }
    }

    displayRouteHistory() {
        const container = document.getElementById('historyList');
        if (!container) return;

        if (this.routeHistory.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">🗺️</div>
                    <h3>No routes yet</h3>
                    <p>Start planning routes to see your history here</p>
                </div>
            `;
            return;
        }

        container.innerHTML = this.routeHistory.map(route => `
            <div class="history-item" data-id="${route.id}">
                <div class="item-header">
                    <div class="item-title">
                        ${utils.formatDistance(route.actualDistance)} route
                        ${route.elevationGain ? `• ${Math.round(route.elevationGain)}m elevation` : ''}
                    </div>
                    <div class="item-date">${utils.formatDate(route.createdAt)}</div>
                </div>
                ${route.matchedPlaces && route.matchedPlaces.length > 0 ? `
                    <div class="item-description">
                        Visited: ${route.matchedPlaces.slice(0, 3).map(p => p.name).join(', ')}
                        ${route.matchedPlaces.length > 3 ? ` and ${route.matchedPlaces.length - 3} more` : ''}
                    </div>
                ` : ''}
                <div class="item-actions">
                    <button class="btn btn-small btn-outline" 
                            onclick="app.viewRoute('${route.id}')">
                        View
                    </button>
                    <button class="btn btn-small btn-outline" 
                            onclick="app.exportRoute('${route.id}')">
                        Export
                    </button>
                    <button class="btn btn-small btn-error" 
                            onclick="app.deleteRoute('${route.id}')">
                        Delete
                    </button>
                </div>
            </div>
        `).join('');
    }

    async deleteRoute(routeId) {
        if (!confirm('Are you sure you want to delete this route?')) {
            return;
        }

        try {
            await api.deleteRoute(routeId);
            await this.loadRouteHistory();
            utils.toast.show('Route deleted', 'success');
        } catch (error) {
            utils.handleError(error, 'deleting route');
        }
    }

    async exportRoute(routeId) {
        try {
            const blob = await api.exportRouteGPX(routeId);
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `perfect10k-route-${routeId}.gpx`;
            
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            utils.toast.show('Route exported successfully', 'success');
            
        } catch (error) {
            utils.handleError(error, 'exporting route');
        }
    }

    handlePageChange(pageId) {
        switch (pageId) {
            case 'planner':
                // Already handled by map initialization
                break;
                
            case 'history':
                if (auth.isAuthenticated) {
                    this.loadRouteHistory();
                }
                break;
                
            case 'preferences':
                if (auth.isAuthenticated) {
                    this.loadPreferences();
                }
                break;
                
            case 'profile':
                if (auth.isAuthenticated) {
                    this.displayProfile();
                }
                break;
        }
    }

    displayProfile() {
        const container = document.getElementById('profileInfo');
        if (!container || !auth.currentUser) return;

        container.innerHTML = `
            <div class="profile-card">
                <h3>Account Information</h3>
                <div class="profile-field">
                    <label>Email:</label>
                    <span>${auth.currentUser.email}</span>
                </div>
                <div class="profile-field">
                    <label>Member since:</label>
                    <span>${utils.formatDate(auth.currentUser.createdAt)}</span>
                </div>
                <div class="profile-field">
                    <label>Status:</label>
                    <span class="status ${auth.currentUser.isActive ? 'active' : 'inactive'}">
                        ${auth.currentUser.isActive ? 'Active' : 'Inactive'}
                    </span>
                </div>
            </div>
            
            <div class="profile-stats">
                <h3>Your Stats</h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">${this.routeHistory.length}</div>
                        <div class="stat-label">Routes Planned</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${this.preferences.length}</div>
                        <div class="stat-label">Preferences</div>
                    </div>
                </div>
            </div>
        `;
    }

    handleAuthStateChange() {
        if (auth.isAuthenticated) {
            this.loadUserData();
        } else {
            this.preferences = [];
            this.routeHistory = [];
            this.currentRoute = null;
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new Perfect10kApp();
});

// Handle authentication state changes
window.addEventListener('storage', (e) => {
    if (e.key === 'auth') {
        // Auth state changed in another tab
        location.reload();
    }
});