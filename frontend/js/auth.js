// Authentication management for Perfect10k

class AuthManager {
    constructor() {
        this.isAuthenticated = false;
        this.currentUser = null;
        this.init();
    }

    init() {
        this.checkAuthState();
        this.setupEventListeners();
        this.updateUI();
    }

    checkAuthState() {
        this.isAuthenticated = api.isAuthenticated();
        this.currentUser = api.getCurrentUser();
    }

    setupEventListeners() {
        // Login form
        const loginForm = document.getElementById('loginForm');
        if (loginForm) {
            loginForm.addEventListener('submit', (e) => this.handleLogin(e));
        }

        // Register form
        const registerForm = document.getElementById('registerForm');
        if (registerForm) {
            registerForm.addEventListener('submit', (e) => this.handleRegister(e));
        }

        // Navigation buttons
        const loginBtn = document.getElementById('loginBtn');
        const registerBtn = document.getElementById('registerBtn');
        const logoutBtn = document.getElementById('logoutBtn');

        if (loginBtn) {
            loginBtn.addEventListener('click', () => utils.modal.open('loginModal'));
        }

        if (registerBtn) {
            registerBtn.addEventListener('click', () => utils.modal.open('registerModal'));
        }

        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => this.handleLogout());
        }

        // Modal close buttons
        document.getElementById('closeLoginModal')?.addEventListener('click', () => {
            utils.modal.close();
        });

        document.getElementById('closeRegisterModal')?.addEventListener('click', () => {
            utils.modal.close();
        });
    }

    async handleLogin(event) {
        event.preventDefault();
        
        const form = event.target;
        const submitBtn = form.querySelector('button[type="submit"]');
        const emailInput = form.querySelector('#loginEmail');
        const passwordInput = form.querySelector('#loginPassword');

        // Validate inputs
        if (!emailInput.value || !passwordInput.value) {
            utils.toast.show('Please fill in all fields', 'error');
            return;
        }

        if (!utils.isValidEmail(emailInput.value)) {
            utils.toast.show('Please enter a valid email address', 'error');
            return;
        }

        try {
            utils.setLoading(submitBtn, true);

            const response = await api.login(emailInput.value, passwordInput.value);
            
            this.isAuthenticated = true;
            this.currentUser = response.user;
            
            utils.toast.show('Login successful!', 'success');
            utils.modal.close();
            this.updateUI();
            
            // Clear form
            utils.clearForm(form);

        } catch (error) {
            utils.handleError(error, 'login');
        } finally {
            utils.setLoading(submitBtn, false);
        }
    }

    async handleRegister(event) {
        event.preventDefault();
        
        const form = event.target;
        const submitBtn = form.querySelector('button[type="submit"]');
        const emailInput = form.querySelector('#registerEmail');
        const passwordInput = form.querySelector('#registerPassword');
        const confirmPasswordInput = form.querySelector('#confirmPassword');

        // Validate inputs
        if (!emailInput.value || !passwordInput.value || !confirmPasswordInput.value) {
            utils.toast.show('Please fill in all fields', 'error');
            return;
        }

        if (!utils.isValidEmail(emailInput.value)) {
            utils.toast.show('Please enter a valid email address', 'error');
            return;
        }

        if (passwordInput.value !== confirmPasswordInput.value) {
            utils.toast.show('Passwords do not match', 'error');
            return;
        }

        if (passwordInput.value.length < 6) {
            utils.toast.show('Password must be at least 6 characters', 'error');
            return;
        }

        try {
            utils.setLoading(submitBtn, true);

            const response = await api.register(emailInput.value, passwordInput.value);
            
            this.isAuthenticated = true;
            this.currentUser = response.user;
            
            utils.toast.show('Registration successful!', 'success');
            utils.modal.close();
            this.updateUI();
            
            // Clear form
            utils.clearForm(form);

        } catch (error) {
            utils.handleError(error, 'registration');
        } finally {
            utils.setLoading(submitBtn, false);
        }
    }

    async handleLogout() {
        try {
            await api.logout();
            
            this.isAuthenticated = false;
            this.currentUser = null;
            
            utils.toast.show('Logged out successfully', 'success');
            this.updateUI();
            
            // Redirect to login if on protected page
            utils.pageManager.showPage('planner');

        } catch (error) {
            utils.handleError(error, 'logout');
        }
    }

    updateUI() {
        const navAuth = document.getElementById('navAuth');
        const navUser = document.getElementById('navUser');
        const userEmail = document.getElementById('userEmail');
        const navMenu = document.getElementById('navMenu');

        if (this.isAuthenticated && this.currentUser) {
            // Show authenticated state
            navAuth.classList.add('hidden');
            navUser.classList.remove('hidden');
            navMenu.style.display = 'flex';
            
            if (userEmail) {
                userEmail.textContent = this.currentUser.email;
            }

            // Enable protected navigation
            document.querySelectorAll('.nav-link').forEach(link => {
                link.style.pointerEvents = 'auto';
                link.style.opacity = '1';
            });

        } else {
            // Show unauthenticated state
            navAuth.classList.remove('hidden');
            navUser.classList.add('hidden');
            navMenu.style.display = 'none';

            // Disable protected navigation except planner
            document.querySelectorAll('.nav-link').forEach(link => {
                if (link.dataset.page !== 'planner') {
                    link.style.pointerEvents = 'none';
                    link.style.opacity = '0.5';
                }
            });

            // Show login prompt on planner page
            this.showLoginPrompt();
        }
    }

    showLoginPrompt() {
        const plannerPage = document.getElementById('plannerPage');
        const existingPrompt = plannerPage.querySelector('.login-prompt');
        
        if (existingPrompt) {
            existingPrompt.remove();
        }

        const prompt = document.createElement('div');
        prompt.className = 'login-prompt';
        prompt.innerHTML = `
            <div class="login-prompt-content">
                <h3>Welcome to Perfect10k</h3>
                <p>Please log in to start planning your perfect walking routes with AI-powered place matching.</p>
                <div class="login-prompt-actions">
                    <button class="btn btn-primary" onclick="utils.modal.open('loginModal')">Login</button>
                    <button class="btn btn-outline" onclick="utils.modal.open('registerModal')">Register</button>
                </div>
            </div>
        `;

        // Add styles for the prompt
        const style = document.createElement('style');
        style.textContent = `
            .login-prompt {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                border-radius: var(--radius-lg);
                box-shadow: var(--shadow-xl);
                padding: var(--space-8);
                text-align: center;
                max-width: 400px;
                z-index: 100;
            }
            
            .login-prompt-content h3 {
                color: var(--primary-600);
                margin-bottom: var(--space-4);
            }
            
            .login-prompt-content p {
                color: var(--gray-600);
                margin-bottom: var(--space-6);
            }
            
            .login-prompt-actions {
                display: flex;
                gap: var(--space-3);
                justify-content: center;
            }
        `;
        
        if (!document.querySelector('style[data-login-prompt]')) {
            style.setAttribute('data-login-prompt', 'true');
            document.head.appendChild(style);
        }

        plannerPage.appendChild(prompt);
    }

    requireAuth(callback) {
        if (!this.isAuthenticated) {
            utils.toast.show('Please log in to access this feature', 'warning');
            utils.modal.open('loginModal');
            return false;
        }
        
        if (callback) {
            callback();
        }
        return true;
    }

    // Auto-refresh token before expiration
    startTokenRefresh() {
        const authData = utils.storage.get('auth');
        if (!authData || !authData.expiresAt) return;

        const refreshTime = authData.expiresAt - Date.now() - (5 * 60 * 1000); // 5 minutes before expiry
        
        if (refreshTime > 0) {
            setTimeout(async () => {
                try {
                    const response = await api.request('/auth/refresh', { method: 'POST' });
                    if (response.access_token) {
                        api.setToken(response.access_token);
                        const newAuthData = {
                            ...authData,
                            token: response.access_token,
                            expiresAt: Date.now() + (response.expires_in * 1000)
                        };
                        utils.storage.set('auth', newAuthData);
                        this.startTokenRefresh(); // Schedule next refresh
                    }
                } catch (error) {
                    console.warn('Token refresh failed:', error);
                    this.handleLogout();
                }
            }, refreshTime);
        }
    }
}

// Initialize auth manager
const auth = new AuthManager();

// Start token refresh if authenticated
if (auth.isAuthenticated) {
    auth.startTokenRefresh();
}

// Export for use in other scripts
window.auth = auth;