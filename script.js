// DOM Elements
const postsContainer = document.getElementById('postsContainer');
const closeBtn = document.getElementById('closeBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const queryString = window.location.search;
const urlParams = new URLSearchParams(queryString);

let allPosts = [];

// ANIMATION-ONLY FUNCTIONS
function animateCardsIn(cards, staggerDelay = 100) {
    cards.forEach((card, index) => {
        // Start hidden
        card.classList.remove('card-visible');
        card.classList.add('card-hidden');
        
        // Animate in with stagger
        setTimeout(() => {
            card.classList.remove('card-hidden');
            card.classList.add('card-visible');
        }, index * staggerDelay);
    });
}

function animateCardsOut(cards, callback) {
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.classList.remove('card-visible');
            card.classList.add('card-hidden');
        }, index * 50);
    });
    
    // Execute callback after all animations complete
    setTimeout(callback, cards.length * 50 + 300);
}
// END ANIMATION-ONLY FUNCTIONS

// Updated date parsing function for Safari compatibility
function parseDate(dateString) {
    const parts = dateString.split('-');
    const year = parseInt(parts[0]);
    const month = parseInt(parts[1]) - 1;  // Months are 0-indexed
    const day = parseInt(parts[2]);
    return new Date(year, month, day);
}

function formatDate(dateString) {
    const date = parseDate(dateString);
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    return date.toLocaleDateString(undefined, options);
}

// TABLE PROCESSING FUNCTION
function processMarkdownTables(text) {
    const lines = text.split('\n');
    const result = [];
    let i = 0;
    
    while (i < lines.length) {
        const line = lines[i];
        
        // Check if this line looks like a table row (starts and ends with |, or has | in it)
        if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
            // Potential table start - look for separator row
            if (i + 1 < lines.length) {
                const nextLine = lines[i + 1].trim();
                // Check if next line is a separator (contains only |, -, :, and spaces)
                if (nextLine.startsWith('|') && /^[\|\-:\s]+$/.test(nextLine)) {
                    // This is a table! Parse it
                    const tableLines = [line];
                    let j = i + 1;
                    
                    // Collect all table lines
                    while (j < lines.length && lines[j].trim().startsWith('|') && lines[j].trim().endsWith('|')) {
                        tableLines.push(lines[j]);
                        j++;
                    }
                    
                    // Parse the table
                    const tableHtml = parseTable(tableLines);
                    result.push(tableHtml);
                    
                    i = j; // Skip past the table
                    continue;
                }
            }
        }
        
        result.push(line);
        i++;
    }
    
    return result.join('\n');
}

function parseTable(tableLines) {
    if (tableLines.length < 2) return tableLines.join('\n');
    
    // Parse header row
    const headerRow = parseTableRow(tableLines[0]);
    
    // Parse separator row to get alignments
    const separatorRow = tableLines[1].trim();
    const alignments = parseSeparatorRow(separatorRow);
    
    // Parse data rows
    const dataRows = tableLines.slice(2).map(line => parseTableRow(line));
    
    // Build HTML table - NO internal newlines
    let html = '<table class="markdown-table">';
    
    // Header
    html += '<thead><tr>';
    headerRow.forEach((cell, idx) => {
        const align = alignments[idx] || 'left';
        html += `<th style="text-align: ${align}">${cell.trim()}</th>`;
    });
    html += '</tr></thead>';
    
    // Body
    if (dataRows.length > 0) {
        html += '<tbody>';
        dataRows.forEach(row => {
            html += '<tr>';
            row.forEach((cell, idx) => {
                const align = alignments[idx] || 'left';
                html += `<td style="text-align: ${align}">${cell.trim()}</td>`;
            });
            html += '</tr>';
        });
        html += '</tbody>';
    }
    
    html += '</table>';
    return html;
}

function parseTableRow(line) {
    // Remove leading and trailing |, then split by |
    let trimmed = line.trim();
    if (trimmed.startsWith('|')) trimmed = trimmed.slice(1);
    if (trimmed.endsWith('|')) trimmed = trimmed.slice(0, -1);
    
    // Split by | but be careful with escaped pipes or pipes in LaTeX
    // For now, simple split - LaTeX should already be protected by placeholders
    return trimmed.split('|');
}

function parseSeparatorRow(line) {
    const cells = parseTableRow(line);
    return cells.map(cell => {
        const trimmed = cell.trim();
        const leftColon = trimmed.startsWith(':');
        const rightColon = trimmed.endsWith(':');
        
        if (leftColon && rightColon) return 'center';
        if (rightColon) return 'right';
        return 'left';
    });
}

function renderMarkdown(content) {
    // Step 0: Extract and placeholder code blocks FIRST (highest priority)
    const codeBlocks = [];
    let html = content.replace(/```([\w-]*)\s*\n?([\s\S]*?)```/g, (match, lang, code) => {
        // Remove only the very first newline if it exists
        let processedCode = code;
        if (processedCode.startsWith('\n')) {
            processedCode = processedCode.substring(1);
        }
        
        // Remove trailing whitespace but preserve internal formatting
        processedCode = processedCode.replace(/\s+$/, '');
        
        // Convert tabs to 4 spaces for consistent rendering
        processedCode = processedCode.replace(/\t/g, '    ');
        
        // Escape HTML entities but preserve whitespace and newlines exactly
        const escapedCode = processedCode
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
            
        const codeBlockHtml = lang 
            ? `<pre><code class="language-${lang}">${escapedCode}</code></pre>`
            : `<pre><code>${escapedCode}</code></pre>`;
            
        // Store the processed code block and return a placeholder
        const placeholder = `__CODEBLOCK_${codeBlocks.length}__`;
        codeBlocks.push(codeBlockHtml);
        return placeholder;
    });

    // Step 1: Extract and placeholder inline code (to protect code containers in backticks)
    const inlineCode = [];
    html = html.replace(/`(.*?)`/gim, (match, code) => {
        const escapedCode = code
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        const inlineHtml = `<code>${escapedCode}</code>`;
        const placeholder = `__INLINECODE_${inlineCode.length}__`;
        inlineCode.push(inlineHtml);
        return placeholder;
    });

    // Step 2: Extract and placeholder code containers (after inline code is protected)
    const codeContainers = [];
    html = html.replace(/\[codeContainer\]\(([^)]+)\)/g, (match, scriptPath) => {
        const placeholder = `__CODECONTAINER_${codeContainers.length}__`;
        codeContainers.push(scriptPath);
        return placeholder;
    });

    // Step 3: Extract and placeholder LaTeX blocks to protect them from processing
    const latexBlocks = [];
    
    // Handle $$ blocks first (display/block math)
    html = html.replace(/\$\$([\s\S]*?)\$\$/g, (match, latex) => {
        const processedLatex = latex.trim();
        const latexHtml = `<div class="latex-display">$$${processedLatex}$$</div>`;
        const placeholder = `__LATEXBLOCK_${latexBlocks.length}__`;
        latexBlocks.push(latexHtml);
        return placeholder;
    });
    
    // Handle inline $ LaTeX (but avoid matching $$ which we already processed)
    html = html.replace(/(?<!\$)\$(?!\$)((?:[^$]|\\\$)+?)\$(?!\$)/g, (match, latex) => {
        const processedLatex = latex.trim();
        const latexHtml = `<span class="latex-inline">$${processedLatex}$</span>`;
        const placeholder = `__LATEXBLOCK_${latexBlocks.length}__`;
        latexBlocks.push(latexHtml);
        return placeholder;
    });

    // Step 4: Process tables and protect them with placeholders
    const tables = [];
    html = processMarkdownTables(html);
    html = html.replace(/<table class="markdown-table">.*?<\/table>/gs, (match) => {
        const placeholder = `__TABLE_${tables.length}__`;
        tables.push(match);
        return placeholder;
    });

    // Step 5: Process the rest of the markdown (without code blocks, inline code, code containers, LaTeX, or tables)
    html = html
        // Headers
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^# (.*$)/gim, '<h1>$1</h1>')
        // Images (must be before links)
        .replace(/!\[([^\]]+)\]\(([^)]+)\)/gim, '<img src="$2" alt="$1">')
        // Links
        .replace(/\[([^\]]+)\]\(([^)]+)\)/gim, '<a href="$2">$1</a>')
        // Bold and italic
        .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/gim, '<em>$1</em>')
        // Blockquotes
        .replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>');

    // Step 6: Handle lists
    // First, convert markdown unordered lists to ordered list items
    html = html.replace(/^- (.*$)/gim, '<li>$1</li>');
    
    // Also handle existing numbered lists
    html = html.replace(/^\d+\. (.*$)/gim, '<li>$1</li>');
    
    // Split into lines and process list items
    const lines = html.split('\n');
    const processedLines = [];
    let inList = false;
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const isListItem = line.trim().match(/^<li>.*<\/li>$/);
        
        if (isListItem && !inList) {
            // Starting a new list
            processedLines.push('<ol>');
            processedLines.push(line);
            inList = true;
        } else if (isListItem && inList) {
            // Continue current list
            processedLines.push(line);
        } else if (!isListItem && inList) {
            // End current list
            processedLines.push('</ol>');
            processedLines.push(line);
            inList = false;
        } else {
            // Regular line
            processedLines.push(line);
        }
    }
    
    // Close list if we ended while in a list
    if (inList) {
        processedLines.push('</ol>');
    }
    
    html = processedLines.join('\n');

    // Step 7: Process paragraphs and line breaks (but not inside code blocks, LaTeX, or tables)
    html = html
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n(?!<li>|<\/ol>|__TABLE_)/g, '<br>')
        .replace(/(<ol>)\n/g, '$1')
        .replace(/(<\/li>)\n/g, '$1');

    // Step 8: Wrap in paragraphs and clean up
    html = `<p>${html}</p>`
        .replace(/<p><\/p>/g, '')
        .replace(/<p>(<\/?(?:pre|h\d|blockquote|ol|div|__TABLE_)[^>]*>)/g, '$1')
        .replace(/(<\/?(?:pre|h\d|blockquote|ol|div|__TABLE_)[^>]*>)<\/p>/g, '$1');

    // Step 9: Restore the protected code blocks
    codeBlocks.forEach((codeBlock, index) => {
        html = html.replace(`__CODEBLOCK_${index}__`, codeBlock);
    });

    // Step 12: Restore the protected tables
    tables.forEach((table, index) => {
        html = html.replace(`__TABLE_${index}__`, table);
    });
    
    // Step 11: Restore the protected LaTeX blocks
    latexBlocks.forEach((latexBlock, index) => {
        html = html.replace(`__LATEXBLOCK_${index}__`, latexBlock);
    });

    // Step 12: Create code container divs with data attributes for script loading
    codeContainers.forEach((scriptPath, index) => {
        const containerHtml = `<div class="code-container" data-script="${scriptPath}" id="codeContainer-${index}"></div>`;
        html = html.replace(`__CODECONTAINER_${index}__`, containerHtml);
    });

    // Step 13: Restore inline code (last, after all other processing)
    inlineCode.forEach((code, index) => {
        html = html.replace(`__INLINECODE_${index}__`, code);
    });

    return html;
}

const markdownFiles = [
    "./nbody_comparison/web/index.md",
    "./textbook_notes/chapter_0/notes-1.md",
    "./textbook_notes/chapter_0/exercises-1.md",
    "./fem_1d_benchmark/web/fem_1d_benchmark_project.md",
    "./textbook_notes/chapter_0/sobolev_fem_foundations.md"
];

function parseFrontMatter(content) {
    if (!content.startsWith('---')) {
        console.error('Front matter not found at start of file');
        return {
            metadata: { title: 'Untitled', date: new Date().toISOString().split('T')[0], tags: [], snippet: '' },
            content: content
        };
    }
    
    const frontMatterEnd = content.indexOf('---', 3);
    if (frontMatterEnd === -1) {
        console.error('Front matter not properly closed');
        return {
            metadata: { title: 'Untitled', date: new Date().toISOString().split('T')[0], tags: [], snippet: '' },
            content: content
        };
    }
    
    const frontMatter = content.slice(3, frontMatterEnd).trim();
    const contentBody = content.slice(frontMatterEnd + 3).trim();
    
    const metadata = {};
    frontMatter.split('\n').forEach(line => {
        const colonIndex = line.indexOf(':');
        if (colonIndex > 0) {
            const key = line.slice(0, colonIndex).trim();
            const value = line.slice(colonIndex + 1).trim();
            metadata[key] = value.replace(/^['"](.*)['"]$/, '$1');
        }
    });
    
    if (!metadata.title) metadata.title = 'Untitled Post';
    if (!metadata.date) metadata.date = new Date().toISOString().split('T')[0];
    if (!metadata.tags) metadata.tags = '';
    if (!metadata.snippet) {
        metadata.snippet = contentBody.substring(0, 100) + (contentBody.length > 100 ? '...' : '');
    }
    
    return { metadata, content: contentBody };
}

async function loadMarkdownFiles() {
    const posts = [];
    let loadedCount = 0;
    
    for (const file of markdownFiles) {
        try {
            const absolutePath = new URL(file, window.location.origin).href;
            console.log(`Fetching: ${absolutePath}`);
            loadingIndicator.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Loading ${loadedCount + 1}/${markdownFiles.length} posts...`;

            const response = await fetch(file);
            if (!response.ok) {
                throw new Error(`Failed to fetch ${file}: ${response.status} ${response.statusText}`);
            }
            
            const text = await response.text();
            const { metadata, content } = parseFrontMatter(text);
            
            const tags = metadata.tags 
                ? metadata.tags.split(',').map(tag => tag.trim()) 
                : [];
            
            posts.push({
                title: metadata.title,
                date: metadata.date,
                tags: tags,
                snippet: metadata.snippet,
                content: content
            });
            
            loadedCount++;
        } catch (error) {
            console.error(`Error loading ${file}:`, error);
            posts.push({
                title: `Error: ${file}`,
                date: new Date().toISOString().split('T')[0],
                tags: ['error'],
                snippet: `Failed to load this post: ${error.message}`,
                content: `# Error Loading Post\n\nCould not load the post from ${file}.\n\nError: ${error.message}`
            });
            loadedCount++;
        }
    }
    
    loadingIndicator.style.display = 'none';
    return posts;
}

// ONLY modification: Added animation wrapper around your original function
function renderPosts(posts) {
    // Check if we should animate (if there are existing cards)
    const existingCards = Array.from(postsContainer.querySelectorAll('.card'));
    const shouldAnimate = existingCards.length > 0;
    
    if (shouldAnimate) {
        // Animate out existing cards first, then render new ones
        animateCardsOut(existingCards, () => {
            renderPostsOriginal(posts);
            // Animate in the new cards
            const newCards = Array.from(postsContainer.querySelectorAll('.card'));
            setTimeout(() => animateCardsIn(newCards, 100), 50);
        });
    } else {
        // First load - render normally then animate in
        renderPostsOriginal(posts);
        const newCards = Array.from(postsContainer.querySelectorAll('.card'));
        setTimeout(() => animateCardsIn(newCards, 100), 50);
    }
}

// Your EXACT original renderPosts function, renamed but unchanged
// Your EXACT original renderPosts function, renamed but unchanged
function renderPostsOriginal(posts) {
    postsContainer.innerHTML = '';
    
    if (posts.length === 0) {
        postsContainer.innerHTML = '<div class="error">No blog posts found. Please check your markdown files.</div>';
        return;
    }
    
    posts.forEach(post => {
        const postElement = document.createElement('div');
        postElement.className = 'card';
        // Add data-topic attribute with first tag (or empty string if no tags)
        postElement.dataset.topic = post.tags.length > 0 ? post.tags[0] : '';
        
        postElement.innerHTML = `
            <div class="post-header">
                <div class="post-title">${post.title}</div>
                <div class="post-date">${formatDate(post.date)}</div>
            </div>
            <div class="post-snippet">
                ${post.snippet}
            </div>
            <div class="post-footer">
                <div class="post-tags">
                    ${post.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
                <div class="read-more">Click to read →</div>
            </div>
        `;
        
        postElement.addEventListener('click', () => {
            showFullPost(post);
            urlParams.set('blog', post.title);
            const newUrl = window.location.pathname + '?' + urlParams.toString();
            window.history.pushState({ path: newUrl }, '', newUrl);
            document.title = post.title;
        });

        if (urlParams.get('blog')==post.title) {
            showFullPost(post)
        }
        
        postsContainer.appendChild(postElement);
    });
}

// Load scripts for code containers after rendering
async function loadCodeContainerScripts(container) {
    const codeContainers = container.querySelectorAll('.code-container');
    
    for (const codeContainer of codeContainers) {
        const scriptPath = codeContainer.dataset.script;
        if (scriptPath) {
            try {
                // Create a new script element
                const script = document.createElement('script');
                script.src = scriptPath;
                script.async = true;
                
                // Store reference to the container globally so the script can access it
                const containerId = codeContainer.id;
                window.currentCodeContainer = codeContainer;
                window.currentCodeContainerId = containerId;
                
                // Load the script
                await new Promise((resolve, reject) => {
                    script.onload = resolve;
                    script.onerror = () => reject(new Error(`Failed to load script: ${scriptPath}`));
                    document.head.appendChild(script);
                });
                
                console.log(`Successfully loaded script: ${scriptPath}`);
            } catch (error) {
                console.error(`Error loading script for code container:`, error);
                codeContainer.innerHTML = `<div class="error">Failed to load interactive content: ${error.message}</div>`;
            }
        }
    }
}

function showFullPost(post) {
    const postFull = document.createElement('div');
    postFull.className = 'post-full';
    postFull.id = 'postFull';
    postFull.innerHTML = `
        <div class="post-full-content">
            <div class="post-full-header">
                <h1 class="post-full-title">${post.title}</h1>
                <div class="post-full-meta">
                    <span><i class="far fa-calendar"></i> ${formatDate(post.date)}</span>
                    <span><i class="far fa-clock"></i> ${Math.ceil(post.content.length / 1200)} min read</span>
                </div>
            </div>
            <div class="post-full-body">
                ${renderMarkdown(post.content)}
            </div>
        </div>
    `;
    
    document.body.appendChild(postFull);
    
    closeBtn.classList.add('visible');
    
    setTimeout(async () => {
        postFull.classList.add('active');
        if (window.Prism) {
            Prism.highlightAll();
        } else {
            console.warn("Prism not loaded. Syntax highlighting disabled.");
        }
        // Render LaTeX with MathJax
        if (window.MathJax && window.MathJax.typesetPromise) {
            console.log('Rendering MathJax...');
            MathJax.typesetPromise().then(() => {
                console.log('MathJax rendering complete');
            }).catch((err) => {
                console.error('MathJax rendering error:', err);
            });
        } else {
            console.warn("MathJax not loaded or not ready. LaTeX rendering disabled.");
            console.log('MathJax object:', window.MathJax);
        }
        
        // Load code container scripts after everything is rendered
        await loadCodeContainerScripts(postFull);
    }, 10);
    
    document.body.style.overflow = 'hidden';
    
    document.addEventListener('keydown', handleEscapeKey);
}

function closeFullPost() {
    const postFull = document.getElementById('postFull');
    if (postFull) {
        postFull.classList.remove('active');
        
        setTimeout(() => {
            postFull.remove();
        }, 300);
    }
    
    closeBtn.classList.remove('visible');
    
    document.body.style.overflow = '';
    
    document.removeEventListener('keydown', handleEscapeKey);

    urlParams.delete('blog')
    const newUrl = window.location.pathname + urlParams.toString();
    window.history.pushState({ path: newUrl }, '', newUrl);
    document.title = "Adam's Blog Page";
}

function handleEscapeKey(event) {
    if (event.key === 'Escape') {
        closeFullPost();
    }
}

function sortOldestFirst(posts) {
    return [...posts].sort((a, b) => {
        const dateComparison = parseDate(a.date) - parseDate(b.date);
        if (dateComparison === 0) {
            return a.title.localeCompare(b.title);
        }
        return dateComparison;
    });
}

closeBtn.addEventListener('click', closeFullPost);

loadMarkdownFiles().then(posts => {
    allPosts = posts;
    renderPosts(allPosts);
}).catch(error => {
    console.error('Error loading posts:', error);
    loadingIndicator.innerHTML = `<div class="error">Error loading posts: ${error.message}</div>`;
});