/**
 * Computational Physics ISP - Main JavaScript
 * Adam Field - Worcester Polytechnic Institute
 */

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('Computational Physics ISP Dashboard Loaded');
    
    initNavigation();
    initScrollEffects();
    initAnimations();
    updateDateTime();
});

// ============================================
// Navigation
// ============================================

function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    // Smooth scroll to sections
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            
            // Only handle internal links
            if (href.startsWith('#')) {
                e.preventDefault();
                
                const targetId = href.slice(1);
                const targetSection = document.getElementById(targetId);
                
                if (targetSection) {
                    const navHeight = document.querySelector('.main-nav').offsetHeight;
                    const targetPosition = targetSection.offsetTop - navHeight - 20;
                    
                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });
                }
            }
        });
    });
    
    // Update active nav link on scroll
    window.addEventListener('scroll', updateActiveNav);
}

function updateActiveNav() {
    const sections = document.querySelectorAll('.section');
    const navLinks = document.querySelectorAll('.nav-link');
    const navHeight = document.querySelector('.main-nav').offsetHeight;
    
    let currentSection = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop - navHeight - 100;
        const sectionHeight = section.offsetHeight;
        
        if (window.pageYOffset >= sectionTop && 
            window.pageYOffset < sectionTop + sectionHeight) {
            currentSection = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === '#' + currentSection) {
            link.classList.add('active');
        }
    });
}

// ============================================
// Scroll Effects
// ============================================

function initScrollEffects() {
    // Fade in elements on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe cards and project cards
    document.querySelectorAll('.card, .project-card, .note-item, .chapter-item').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

// ============================================
// Animations
// ============================================

function initAnimations() {
    // Add pulse effect to status badges on hover
    const statusBadges = document.querySelectorAll('.status-badge');
    
    statusBadges.forEach(badge => {
        badge.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.1)';
        });
        
        badge.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
        });
    });
    
    // Add ripple effect to buttons
    const buttons = document.querySelectorAll('.btn:not(.btn-disabled)');
    
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.classList.add('ripple');
            
            this.appendChild(ripple);
            
            setTimeout(() => ripple.remove(), 600);
        });
    });
}

// ============================================
// Utility Functions
// ============================================

function updateDateTime() {
    // Update any dynamic date/time elements
    const dateElements = document.querySelectorAll('[data-dynamic-date]');
    
    dateElements.forEach(el => {
        const now = new Date();
        el.textContent = now.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    });
}

// ============================================
// Progress Tracking (for future use)
// ============================================

function updateProgress() {
    // Calculate textbook progress
    const chapters = document.querySelectorAll('.chapter-item');
    const completed = document.querySelectorAll('.chapter-item.completed').length;
    const inProgress = document.querySelectorAll('.chapter-item.in-progress').length;
    const total = chapters.length;
    
    console.log(`Textbook Progress: ${completed}/${total} complete, ${inProgress} in progress`);
    
    // Could update a progress bar here in the future
}

// ============================================
// Notes Management (for future implementation)
// ============================================

function addNote(date, topic, tag) {
    // This would be implemented when adding dynamic note creation
    console.log('Note added:', { date, topic, tag });
}

function loadNotes() {
    // Load notes from localStorage or backend
    const notes = localStorage.getItem('courseNotes');
    if (notes) {
        return JSON.parse(notes);
    }
    return [];
}

function saveNotes(notes) {
    // Save notes to localStorage
    localStorage.setItem('courseNotes', JSON.stringify(notes));
}

// ============================================
// Keyboard Shortcuts
// ============================================

document.addEventListener('keydown', function(e) {
    // Alt + 1-4 for quick navigation
    if (e.altKey) {
        switch(e.key) {
            case '1':
                document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' });
                break;
            case '2':
                document.getElementById('projects')?.scrollIntoView({ behavior: 'smooth' });
                break;
            case '3':
                document.getElementById('notes')?.scrollIntoView({ behavior: 'smooth' });
                break;
            case '4':
                document.getElementById('resources')?.scrollIntoView({ behavior: 'smooth' });
                break;
        }
    }
});

// ============================================
// Easter Egg: Konami Code
// ============================================

let konamiCode = [];
const konamiPattern = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];

document.addEventListener('keydown', function(e) {
    konamiCode.push(e.key);
    konamiCode = konamiCode.slice(-konamiPattern.length);
    
    if (konamiCode.join('') === konamiPattern.join('')) {
        activateEasterEgg();
    }
});

function activateEasterEgg() {
    console.log('🎮 Konami Code activated!');
    
    // Add fun animation or effect
    document.body.style.animation = 'rainbow 2s linear';
    
    setTimeout(() => {
        document.body.style.animation = '';
    }, 2000);
}

// ============================================
// Export functions for potential use
// ============================================

window.cpISP = {
    updateProgress,
    addNote,
    loadNotes,
    saveNotes
};