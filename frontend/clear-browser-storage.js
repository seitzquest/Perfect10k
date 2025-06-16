/**
 * Clear Browser Storage Script
 * Run this in browser console to clear all local data
 */

// Clear all localStorage
console.log('Clearing localStorage...');
localStorage.clear();

// Clear all sessionStorage  
console.log('Clearing sessionStorage...');
sessionStorage.clear();

// Clear all IndexedDB databases
console.log('Clearing IndexedDB...');
if ('indexedDB' in window) {
    indexedDB.databases().then(databases => {
        databases.forEach(db => {
            indexedDB.deleteDatabase(db.name);
        });
    }).catch(e => console.log('IndexedDB clear failed:', e));
}

// Clear all cookies for current domain
console.log('Clearing cookies...');
document.cookie.split(";").forEach(function(c) { 
    document.cookie = c.replace(/^ +/, "").replace(/=.*/, "=;expires=" + new Date().toUTCString() + ";path=/"); 
});

// Clear service worker cache if available
console.log('Clearing service worker caches...');
if ('caches' in window) {
    caches.keys().then(cacheNames => {
        cacheNames.forEach(cacheName => {
            caches.delete(cacheName);
        });
    }).catch(e => console.log('Cache clear failed:', e));
}

console.log('✅ All browser storage cleared! Refresh the page to start fresh.');