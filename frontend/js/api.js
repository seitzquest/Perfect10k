// API service for Perfect10k frontend

class APIService {
    constructor(baseURL = '/api') {
        this.baseURL = baseURL;
        this.token = null;
        this.loadToken();
    }

    loadToken() {
        const authData = utils.storage.get('auth');
        if (authData && authData.token) {
            this.token = authData.token;
        }
    }

    setToken(token) {
        this.token = token;
        const authData = utils.storage.get('auth') || {};
        authData.token = token;
        utils.storage.set('auth', authData);
    }

    clearToken() {
        this.token = null;
        utils.storage.remove('auth');
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
            ...options
        };

        // Add authorization header if token exists
        if (this.token) {
            defaultOptions.headers.Authorization = `Bearer ${this.token}`;
        }

        try {
            const response = await fetch(url, defaultOptions);
            
            // Handle 401 Unauthorized
            if (response.status === 401) {
                this.clearToken();
                window.location.reload();
                throw new Error('Authentication required');
            }

            // Parse response
            let data;
            const contentType = response.headers.get('content-type');
            
            if (contentType && contentType.includes('application/json')) {
                data = await response.json();
            } else {
                data = await response.text();
            }

            if (!response.ok) {
                throw new Error(data.error || data.message || `HTTP ${response.status}`);
            }

            return data;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    // Authentication endpoints
    async login(email, password) {
        const formData = new FormData();
        formData.append('username', email);
        formData.append('password', password);

        const response = await this.request('/auth/login', {
            method: 'POST',
            headers: {}, // Remove Content-Type to let browser set it for FormData
            body: formData
        });

        if (response.access_token) {
            this.setToken(response.access_token);
            
            // Store user data
            const authData = {
                token: response.access_token,
                user: response.user,
                expiresAt: Date.now() + (response.expires_in * 1000)
            };
            utils.storage.set('auth', authData);
        }

        return response;
    }

    async register(email, password) {
        const response = await this.request('/auth/register', {
            method: 'POST',
            body: JSON.stringify({ email, password })
        });

        if (response.access_token) {
            this.setToken(response.access_token);
            
            // Store user data
            const authData = {
                token: response.access_token,
                user: response.user,
                expiresAt: Date.now() + (response.expires_in * 1000)
            };
            utils.storage.set('auth', authData);
        }

        return response;
    }

    async logout() {
        try {
            await this.request('/auth/logout', { method: 'POST' });
        } catch (error) {
            console.warn('Logout request failed:', error);
        } finally {
            this.clearToken();
        }
    }

    async getCurrentUser() {
        return this.request('/auth/me');
    }

    // Route planning endpoints
    async planRoute(routeRequest) {
        return this.request('/routes/plan', {
            method: 'POST',
            body: JSON.stringify(routeRequest)
        });
    }

    async getRouteHistory(page = 1, size = 20) {
        return this.request(`/routes/history?page=${page}&size=${size}`);
    }

    async getRoute(routeId) {
        return this.request(`/routes/${routeId}`);
    }

    async deleteRoute(routeId) {
        return this.request(`/routes/${routeId}`, { method: 'DELETE' });
    }

    async exportRouteGPX(routeId) {
        const response = await fetch(`${this.baseURL}/routes/${routeId}/export/gpx`, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${this.token}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to export route');
        }

        return response.blob();
    }

    // Places endpoints
    async searchPlaces(searchRequest) {
        return this.request('/places/search', {
            method: 'POST',
            body: JSON.stringify(searchRequest)
        });
    }

    async getNearbyPlaces(latitude, longitude, radius = 5000, placeTypes = null, limit = 50) {
        let url = `/places/nearby?latitude=${latitude}&longitude=${longitude}&radius=${radius}&limit=${limit}`;
        
        if (placeTypes && placeTypes.length > 0) {
            url += `&place_types=${placeTypes.join(',')}`;
        }

        return this.request(url);
    }

    async refreshPlacesCache(latitude, longitude, radius = 10000) {
        return this.request('/places/refresh', {
            method: 'POST',
            body: JSON.stringify({ latitude, longitude, radius })
        });
    }

    async getPlaceTypes() {
        return this.request('/places/types');
    }

    // Preferences endpoints
    async createPreference(description, weight = 1.0) {
        return this.request('/preferences', {
            method: 'POST',
            body: JSON.stringify({ description, weight })
        });
    }

    async getUserPreferences() {
        return this.request('/preferences');
    }

    async updatePreference(preferenceId, description, weight = 1.0) {
        return this.request(`/preferences/${preferenceId}`, {
            method: 'PUT',
            body: JSON.stringify({ description, weight })
        });
    }

    async deletePreference(preferenceId) {
        return this.request(`/preferences/${preferenceId}`, { method: 'DELETE' });
    }

    async togglePreference(preferenceId) {
        return this.request(`/preferences/${preferenceId}/toggle`, { method: 'POST' });
    }

    // Health check
    async healthCheck() {
        return this.request('/health');
    }

    // Utility methods
    isAuthenticated() {
        const authData = utils.storage.get('auth');
        if (!authData || !authData.token) {
            return false;
        }

        // Check if token is expired
        if (authData.expiresAt && Date.now() > authData.expiresAt) {
            this.clearToken();
            return false;
        }

        return true;
    }

    getCurrentUser() {
        const authData = utils.storage.get('auth');
        return authData ? authData.user : null;
    }
}

// Create global API instance
const api = new APIService();

// Export for use in other scripts
window.api = api;