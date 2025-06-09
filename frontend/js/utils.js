// Utility functions for Perfect10k frontend

// Toast notification system
class ToastManager {
    constructor() {
        this.container = document.getElementById('toastContainer');
        this.toasts = new Map();
    }

    show(message, type = 'info', duration = 4000) {
        const id = Date.now() + Math.random();
        const toast = this.createToast(id, message, type);
        
        this.container.appendChild(toast);
        this.toasts.set(id, toast);

        // Auto remove after duration
        if (duration > 0) {
            setTimeout(() => this.remove(id), duration);
        }

        return id;
    }

    createToast(id, message, type) {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.dataset.id = id;

        const icons = {
            success: '✓',
            error: '✕',
            warning: '⚠',
            info: 'ℹ'
        };

        const titles = {
            success: 'Success',
            error: 'Error',
            warning: 'Warning',
            info: 'Info'
        };

        toast.innerHTML = `
            <div class="toast-header">
                <div class="toast-title">${icons[type]} ${titles[type]}</div>
                <button class="toast-close" onclick="toast.remove(${id})">&times;</button>
            </div>
            <div class="toast-message">${message}</div>
        `;

        return toast;
    }

    remove(id) {
        const toast = this.toasts.get(id);
        if (toast) {
            toast.style.animation = 'slideOut 0.3s ease forwards';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
                this.toasts.delete(id);
            }, 300);
        }
    }

    clear() {
        this.toasts.forEach((toast, id) => this.remove(id));
    }
}

// Global toast instance
const toast = new ToastManager();

// Local Storage utilities
const storage = {
    get(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (error) {
            console.warn('Error reading from localStorage:', error);
            return defaultValue;
        }
    },

    set(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (error) {
            console.warn('Error writing to localStorage:', error);
        }
    },

    remove(key) {
        try {
            localStorage.removeItem(key);
        } catch (error) {
            console.warn('Error removing from localStorage:', error);
        }
    },

    clear() {
        try {
            localStorage.clear();
        } catch (error) {
            console.warn('Error clearing localStorage:', error);
        }
    }
};

// Debounce function
function debounce(func, wait, immediate = false) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func(...args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func(...args);
    };
}

// Throttle function
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Format distance
function formatDistance(meters) {
    if (meters < 1000) {
        return `${Math.round(meters)} m`;
    }
    return `${(meters / 1000).toFixed(1)} km`;
}

// Format duration
function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
}

// Format date
function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = now - date;
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
        return 'Today';
    } else if (diffDays === 1) {
        return 'Yesterday';
    } else if (diffDays < 7) {
        return `${diffDays} days ago`;
    } else {
        return date.toLocaleDateString();
    }
}

// Validate email
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

// Generate UUID (simple version)
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// Get user location
function getCurrentPosition(options = {}) {
    return new Promise((resolve, reject) => {
        if (!navigator.geolocation) {
            reject(new Error('Geolocation is not supported by this browser'));
            return;
        }

        const defaultOptions = {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 300000 // 5 minutes
        };

        navigator.geolocation.getCurrentPosition(
            resolve,
            reject,
            { ...defaultOptions, ...options }
        );
    });
}

// Calculate distance between two points (Haversine formula)
function calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // Radius of the Earth in kilometers
    const dLat = toRadians(lat2 - lat1);
    const dLon = toRadians(lon2 - lon1);
    const a = 
        Math.sin(dLat/2) * Math.sin(dLat/2) +
        Math.cos(toRadians(lat1)) * Math.cos(toRadians(lat2)) * 
        Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c * 1000; // Distance in meters
}

function toRadians(degrees) {
    return degrees * (Math.PI / 180);
}

// Modal utilities
class ModalManager {
    constructor() {
        this.currentModal = null;
        this.setupEventListeners();
    }

    open(modalId) {
        this.close(); // Close any existing modal
        
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.remove('hidden');
            this.currentModal = modal;
            document.body.style.overflow = 'hidden';
            
            // Focus on first input
            const firstInput = modal.querySelector('input, textarea, select');
            if (firstInput) {
                setTimeout(() => firstInput.focus(), 100);
            }
        }
    }

    close() {
        if (this.currentModal) {
            this.currentModal.classList.add('hidden');
            this.currentModal = null;
            document.body.style.overflow = '';
        }
    }

    setupEventListeners() {
        // Close modal when clicking outside
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                this.close();
            }
        });

        // Close modal on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.currentModal) {
                this.close();
            }
        });

        // Handle close buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal-close')) {
                this.close();
            }
        });
    }
}

// Global modal instance
const modal = new ModalManager();

// Form utilities
function getFormData(form) {
    const formData = new FormData(form);
    const data = {};
    
    for (let [key, value] of formData.entries()) {
        // Handle checkboxes
        if (form.querySelector(`[name="${key}"]`).type === 'checkbox') {
            data[key] = form.querySelector(`[name="${key}"]`).checked;
        } else {
            data[key] = value;
        }
    }
    
    return data;
}

function setFormData(form, data) {
    for (let [key, value] of Object.entries(data)) {
        const field = form.querySelector(`[name="${key}"]`);
        if (field) {
            if (field.type === 'checkbox') {
                field.checked = value;
            } else {
                field.value = value;
            }
        }
    }
}

function clearForm(form) {
    form.reset();
    // Clear any custom validation states
    form.querySelectorAll('.input').forEach(input => {
        input.classList.remove('error', 'success');
    });
}

// Loading state utilities
function setLoading(element, loading = true) {
    if (loading) {
        element.classList.add('loading');
        element.disabled = true;
        
        const text = element.querySelector('.btn-text');
        const loadingText = element.querySelector('.btn-loading');
        
        if (text) text.classList.add('hidden');
        if (loadingText) loadingText.classList.remove('hidden');
    } else {
        element.classList.remove('loading');
        element.disabled = false;
        
        const text = element.querySelector('.btn-text');
        const loadingText = element.querySelector('.btn-loading');
        
        if (text) text.classList.remove('hidden');
        if (loadingText) loadingText.classList.add('hidden');
    }
}

// Page navigation
class PageManager {
    constructor() {
        this.currentPage = 'planner';
        this.setupEventListeners();
    }

    showPage(pageId) {
        // Hide all pages
        document.querySelectorAll('.page').forEach(page => {
            page.classList.remove('active');
        });

        // Show target page
        const targetPage = document.getElementById(pageId + 'Page');
        if (targetPage) {
            targetPage.classList.add('active');
            this.currentPage = pageId;
        }

        // Update navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });

        const activeLink = document.querySelector(`.nav-link[data-page="${pageId}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }

        // Trigger page-specific initialization
        this.onPageChange(pageId);
    }

    onPageChange(pageId) {
        // Trigger events for page-specific functionality
        window.dispatchEvent(new CustomEvent('pageChange', { detail: { pageId } }));
    }

    setupEventListeners() {
        document.addEventListener('click', (e) => {
            if (e.target.matches('.nav-link[data-page]')) {
                e.preventDefault();
                const pageId = e.target.dataset.page;
                this.showPage(pageId);
            }
        });
    }
}

// Global page manager
const pageManager = new PageManager();

// Error handling
function handleError(error, context = '') {
    console.error(`Error in ${context}:`, error);
    
    let message = 'An unexpected error occurred';
    
    if (error.response) {
        // API error
        message = error.response.data?.error || error.response.data?.message || `Server error (${error.response.status})`;
    } else if (error.message) {
        message = error.message;
    }
    
    toast.show(message, 'error');
}

// Export for use in other scripts
window.utils = {
    toast,
    storage,
    modal,
    pageManager,
    debounce,
    throttle,
    formatDistance,
    formatDuration,
    formatDate,
    isValidEmail,
    generateUUID,
    getCurrentPosition,
    calculateDistance,
    getFormData,
    setFormData,
    clearForm,
    setLoading,
    handleError
};