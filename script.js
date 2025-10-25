// DOM elements
const body = document.body;
const themeToggle = document.getElementById('theme-toggle');
const leftToggle = document.getElementById('left-toggle');
const rightToggle = document.getElementById('right-toggle');
const leftSidebar = document.querySelector('.left-sidebar');
const rightSidebar = document.querySelector('.right-sidebar');
const weekHeaders = document.querySelectorAll('.week-header');
const topics = document.querySelectorAll('.topics li');
const contentDiv = document.getElementById('content');

// Initialize theme
const savedTheme = localStorage.getItem('theme') || 'light';
body.setAttribute('data-theme', savedTheme);
updateThemeIcon();

// Theme toggle
themeToggle.addEventListener('click', () => {
    const currentTheme = body.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    body.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon();
});

function updateThemeIcon() {
    const icon = themeToggle.querySelector('i');
    icon.className = body.getAttribute('data-theme') === 'light' ? 'fas fa-moon' : 'fas fa-sun';
}

// Accordion functionality
weekHeaders.forEach(header => {
    header.addEventListener('click', () => {
        const week = header.parentElement;
        week.classList.toggle('expanded');
    });
});

// Topic selection
topics.forEach(topic => {
    topic.addEventListener('click', () => {
        // Remove active class from all topics
        topics.forEach(t => t.classList.remove('active'));
        // Add active class to clicked topic
        topic.classList.add('active');
        // Load content
        const file = topic.getAttribute('data-file');
        loadContent(file);
        // Close sidebar on mobile after selection
        if (window.innerWidth <= 768) {
            leftSidebar.classList.remove('open');
        }
    });
});

// Ensure MathJax is loaded
function ensureMathJax() {
    return new Promise((resolve) => {
        if (window.MathJax && window.MathJax.typesetPromise) {
            resolve();
        } else {
            const check = () => {
                if (window.MathJax && window.MathJax.typesetPromise) {
                    resolve();
                } else {
                    setTimeout(check, 100);
                }
            };
            check();
        }
    });
}

// Load Markdown content
async function loadContent(file) {
    try {
        const response = await fetch(file);
        if (!response.ok) {
            throw new Error('File not found');
        }
        const markdown = await response.text();
        const html = marked.parse(markdown);
        contentDiv.innerHTML = html;
        // Wait for MathJax and render math expressions
        await ensureMathJax();
        await window.MathJax.typesetPromise([contentDiv]);
        // Highlight code blocks
        Prism.highlightAll();
    } catch (error) {
        contentDiv.innerHTML = `<div class="welcome"><h1>Error</h1><p>Unable to load content: ${error.message}</p></div>`;
    }
}

// Responsive sidebar toggles
leftToggle.addEventListener('click', () => {
    leftSidebar.classList.toggle('open');
    rightSidebar.classList.remove('open'); // Close other sidebar
});

rightToggle.addEventListener('click', () => {
    rightSidebar.classList.toggle('open');
    leftSidebar.classList.remove('open'); // Close other sidebar
});

// Close sidebars when clicking outside on mobile
document.addEventListener('click', (e) => {
    if (window.innerWidth <= 768) {
        if (!leftSidebar.contains(e.target) && !leftToggle.contains(e.target)) {
            leftSidebar.classList.remove('open');
        }
        if (!rightSidebar.contains(e.target) && !rightToggle.contains(e.target)) {
            rightSidebar.classList.remove('open');
        }
    }
});

// Handle window resize
window.addEventListener('resize', () => {
    if (window.innerWidth > 768) {
        leftSidebar.classList.remove('open');
        rightSidebar.classList.remove('open');
    }
});
