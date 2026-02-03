// FEM Benchmark - Full Dashboard Iframe Embed
// Alternative to inline charts - embeds the complete standalone dashboard

(function() {
    const container = window.currentCodeContainer || document.getElementById(window.currentCodeContainerId);
    
    if (!container) {
        console.error('Code container not found');
        return;
    }

    // Get the correct path to benchmark_results.html
    // This assumes the script is in fem_1d_benchmark/web/
    const dashboardPath = './benchmark_results.html';

    // Create styled iframe container
    container.innerHTML = `
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; 
                    border-radius: 15px; 
                    margin: 20px 0;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
            <div style="background: white; 
                        border-radius: 10px; 
                        overflow: hidden;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="background: #667eea; 
                            color: white; 
                            padding: 15px 20px; 
                            font-size: 18px; 
                            font-weight: 600;">
                    📊 Interactive Benchmark Dashboard
                    <span style="float: right; font-size: 14px; font-weight: normal; opacity: 0.9;">
                        Full-screen available
                    </span>
                </div>
                <iframe src="${dashboardPath}" 
                        style="width: 100%; 
                               height: 900px; 
                               border: none; 
                               display: block;
                               background: white;">
                </iframe>
            </div>
            <div style="margin-top: 15px; 
                        text-align: center;">
                <a href="${dashboardPath}" 
                   target="_blank"
                   style="display: inline-block;
                          background: white;
                          color: #667eea;
                          padding: 10px 20px;
                          border-radius: 8px;
                          text-decoration: none;
                          font-weight: 600;
                          box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                          transition: transform 0.2s, box-shadow 0.2s;">
                    🔗 Open Dashboard in New Tab
                </a>
            </div>
        </div>
    `;

    // Add hover effect to the link
    const link = container.querySelector('a');
    link.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-2px)';
        this.style.boxShadow = '0 4px 10px rgba(0,0,0,0.15)';
    });
    link.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0)';
        this.style.boxShadow = '0 2px 5px rgba(0,0,0,0.1)';
    });
})();