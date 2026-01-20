// Filter functionality with smooth animations
document.getElementById('filter').addEventListener('change', function() {
    const filter = this.value;
    const cards = document.querySelectorAll('.card');
    
    cards.forEach((card, index) => {
        const shouldShow = filter === 'all' || card.dataset.topic === filter;
        
        if (shouldShow) {
            // Show the card with animation
            card.style.display = 'inline-block';
            // Small delay for staggered animation effect
            setTimeout(() => {
                card.classList.add('card-visible');
                card.classList.remove('card-hidden');
            }, index * 50); // 50ms delay between each card
        } else {
            // Hide the card with animation
            card.classList.add('card-hidden');
            card.classList.remove('card-visible');
            // Hide completely after animation finishes
            setTimeout(() => {
                if (card.classList.contains('card-hidden')) {
                    card.style.display = 'none';
                }
            }, 300); // Match CSS transition duration
        }
    });
    
    // Update select width after filter change
    adjustSelectWidth(this);
});

// Keyboard accessibility for cards
document.querySelectorAll('.card').forEach(card => {
    card.setAttribute('tabindex', '0');
    card.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            const link = this.querySelector('a');
            if (link) {
                window.location.href = link.href;
            }
        }
    });
});

function adjustSelectWidth(selectElement) {
    const measurer = document.getElementById('measurer');
    const selectedOption = selectElement.options[selectElement.selectedIndex];
    const text = selectedOption ? selectedOption.text : '';
    
    // Copy styles to measurer
    const computedStyle = window.getComputedStyle(selectElement);
    measurer.style.font = computedStyle.font;
    measurer.style.fontSize = computedStyle.fontSize;
    measurer.style.fontFamily = computedStyle.fontFamily;
    measurer.style.fontWeight = computedStyle.fontWeight;
    measurer.style.padding = computedStyle.padding;
    
    // Measure the text
    measurer.textContent = text;
    
    // Get container width for percentage calculation
    const container = selectElement.closest('.filter-section') || selectElement.closest('.container') || document.body;
    const containerWidth = container.offsetWidth;
    
    // Calculate text width with extra space for dropdown arrow
    const textWidth = measurer.offsetWidth;
    const extraSpace = 20; // Consistent space for dropdown arrow and padding
    const totalRequiredWidth = textWidth + extraSpace;
    
    // Convert to percentage of container width
    let widthPercentage = (totalRequiredWidth / containerWidth) * 100;
    
    // Apply responsive constraints based on screen size
    const screenWidth = window.innerWidth;
    let maxPercent;
    
    if (screenWidth < 480) { // Mobile phones
        maxPercent = 90;
    } else if (screenWidth < 768) { // Tablets
        maxPercent = 75;
    } else { // Desktop
        maxPercent = 65;
    }
    
    // Constrain the percentage within reasonable bounds
    widthPercentage = Math.min(maxPercent, widthPercentage);
    
    selectElement.style.width = widthPercentage + '%';
}

// Initialize all dynamic selects
function initializeDynamicSelects() {
    const selects = document.querySelectorAll('.filter-select');
    
    selects.forEach(select => {
        // Set initial width
        adjustSelectWidth(select);
        
        // Add change event listener for width adjustment (filter change already handles this)
        if (select.id !== 'filter') {
            select.addEventListener('change', function() {
                adjustSelectWidth(this);
            });
        }
    });
}

// Debounced resize handler for better performance
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Handle window resize to recalculate widths
function handleResize() {
    const selects = document.querySelectorAll('.filter-select');
    selects.forEach(select => {
        adjustSelectWidth(select);
    });
}

// Improved alternative using viewport units for ultra-responsive behavior
function adjustSelectWidthViewport(selectElement) {
    const measurer = document.getElementById('measurer');
    const selectedOption = selectElement.options[selectElement.selectedIndex];
    const text = selectedOption ? selectedOption.text : '';
    
    const computedStyle = window.getComputedStyle(selectElement);
    measurer.style.font = computedStyle.font;
    measurer.textContent = text;
    
    const textWidth = measurer.offsetWidth;
    const viewportWidth = window.innerWidth;
    const extraSpace = 20; // Consistent with main function
    
    // Calculate as viewport width units (vw)
    let vwWidth = ((textWidth + extraSpace) / viewportWidth) * 100;
    
    // Constrain viewport-based width with improved ranges
    const screenWidth = window.innerWidth;
    let maxVw;
    
    if (screenWidth < 480) {
        maxVw = 85;
    } else if (screenWidth < 768) {
        maxVw = 70;
    } else {
        maxVw = 50;
    }
    
    vwWidth = Math.min(maxVw, vwWidth);
    selectElement.style.width = vwWidth + 'vw';
}

// Orientation change handler for mobile devices
function handleOrientationChange() {
    // Small delay to allow for orientation change to complete
    setTimeout(handleResize, 150);
}

// Performance monitoring (optional - remove in production)
function logPerformance(functionName, startTime) {
    const endTime = performance.now();
    if (endTime - startTime > 16) { // Log if takes longer than 1 frame (16ms)
        console.log(`${functionName} took ${endTime - startTime} milliseconds`);
    }
}

// Enhanced initialization with error handling and card animation setup
document.addEventListener('DOMContentLoaded', function() {
    try {
        // Check if required elements exist
        const filterSelect = document.getElementById('filter');
        const measurer = document.getElementById('measurer');
        
        if (!filterSelect) {
            console.warn('Filter select element not found');
            return;
        }
        
        if (!measurer) {
            console.warn('Hidden measurer element not found');
            return;
        }
        
        // Initialize all cards as visible (for "All" filter default state)
        const cards = document.querySelectorAll('.card');
        cards.forEach((card, index) => {
            // Add initial animation with slight delay for loading effect
            setTimeout(() => {
                card.classList.add('card-visible');
            }, index * 100);
        });
        
        // Initialize dynamic selects
        initializeDynamicSelects();
        
        // Add event listeners with error handling
        const debouncedResize = debounce(handleResize, 250);
        
        window.addEventListener('resize', debouncedResize);
        window.addEventListener('orientationchange', handleOrientationChange);
        
        // Optional: Log successful initialization
        console.log('Filter and dynamic select functionality initialized successfully');
        
    } catch (error) {
        console.error('Error initializing filter functionality:', error);
    }
});

// Utility function to manually trigger width recalculation (useful for dynamic content)
function recalculateAllSelectWidths() {
    const selects = document.querySelectorAll('.filter-select');
    selects.forEach(select => {
        adjustSelectWidth(select);
    });
}

// Export function for potential use in other scripts
window.recalculateSelectWidths = recalculateAllSelectWidths;