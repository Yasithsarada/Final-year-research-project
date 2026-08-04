import base64
import os
from pathlib import Path

# Paths
WORKSPACE_DIR = Path("C:/Users/Yasith/Desktop/reasearch v2")
MARKDOWN_PATH = Path("C:/Users/Yasith/.gemini/antigravity-ide/brain/4c0286ce-b033-4a3a-af96-eaad93e7199e/system_overview.md")
OUTPUT_HTML_PATH = WORKSPACE_DIR / "system_overview.html"

def generate_html():
    if not MARKDOWN_PATH.exists():
        print(f"Error: Markdown file not found at {MARKDOWN_PATH}")
        return

    # Read markdown content
    markdown_content = MARKDOWN_PATH.read_text(encoding="utf-8")
    
    # Base64 encode the markdown content for safe injection into JavaScript
    b64_content = base64.b64encode(markdown_content.encode("utf-8")).decode("utf-8")

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>System Overview & Diagrammatic Representation</title>
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    
    <!-- Markdown Parser -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    
    <!-- Mermaid Diagrams -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>

    <style>
        :root {
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --bg-base: #f8fafc;
            --bg-card: #ffffff;
            --text-main: #0f172a;
            --text-muted: #475569;
            --border: #e2e8f0;
            --code-bg: #f1f5f9;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
            line-height: 1.6;
            margin: 0;
            padding: 0;
        }

        .action-bar {
            position: sticky;
            top: 0;
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 100;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        .brand {
            font-weight: 700;
            color: #1e293b;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn-print {
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s ease;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        }

        .btn-print:hover {
            background-color: var(--primary-hover);
            transform: translateY(-1px);
        }

        .container {
            max-width: 900px;
            margin: 40px auto;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
            padding: 48px;
        }

        /* Markdown Styling styling */
        .markdown-body h1 {
            font-size: 2.2rem;
            font-weight: 800;
            border-bottom: 2px solid var(--border);
            padding-bottom: 12px;
            margin-top: 0;
            margin-bottom: 24px;
            color: #0f172a;
        }

        .markdown-body h2 {
            font-size: 1.5rem;
            font-weight: 700;
            margin-top: 32px;
            margin-bottom: 16px;
            color: #1e293b;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }

        .markdown-body h3 {
            font-size: 1.25rem;
            font-weight: 600;
            margin-top: 24px;
            margin-bottom: 12px;
            color: #334155;
        }

        .markdown-body p {
            margin-bottom: 16px;
            color: var(--text-muted);
        }

        .markdown-body a {
            color: var(--primary);
            text-decoration: none;
            font-weight: 500;
        }

        .markdown-body a:hover {
            text-decoration: underline;
        }

        .markdown-body table {
            width: 100%;
            border-collapse: collapse;
            margin: 24px 0;
        }

        .markdown-body th {
            background-color: #f8fafc;
            font-weight: 600;
            border: 1px solid var(--border);
            padding: 10px 14px;
            text-align: left;
            color: #1e293b;
        }

        .markdown-body td {
            border: 1px solid var(--border);
            padding: 10px 14px;
            color: var(--text-muted);
        }

        .markdown-body tr:nth-child(even) {
            background-color: #fdfefe;
        }

        .markdown-body code {
            font-family: 'JetBrains Mono', monospace;
            background-color: var(--code-bg);
            padding: 3px 6px;
            border-radius: 4px;
            font-size: 0.875em;
            color: #0f172a;
        }

        .markdown-body pre {
            background-color: #0f172a;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
        }

        .markdown-body pre code {
            background-color: transparent;
            padding: 0;
            color: #f8fafc;
            font-size: 0.9rem;
        }

        /* Diagram containers */
        .mermaid-container {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 24px;
            margin: 24px 0;
            display: flex;
            justify-content: center;
            overflow-x: auto;
            box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.02);
        }

        .mermaid {
            width: 100%;
            text-align: center;
        }

        hr {
            border: 0;
            height: 1px;
            background: var(--border);
            margin: 32px 0;
        }

        /* Print styles */
        @media print {
            body {
                background-color: white;
                color: black;
                font-size: 11pt;
            }

            .action-bar {
                display: none !important;
            }

            .container {
                max-width: 100%;
                margin: 0;
                padding: 0;
                border: none;
                box-shadow: none;
            }

            .markdown-body h1, 
            .markdown-body h2, 
            .markdown-body h3 {
                page-break-after: avoid;
            }

            .mermaid-container, 
            .markdown-body table, 
            .markdown-body pre {
                page-break-inside: avoid;
            }

            @page {
                margin: 1.5cm 2cm;
            }
        }
    </style>
</head>
<body>

    <div class="action-bar">
        <div class="brand">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--primary);"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
            System Overview Export
        </div>
        <button class="btn-print" onclick="window.print()">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9V2h12v7"></path><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path><rect x="6" y="14" width="12" height="8"></rect></svg>
            Save as PDF / Print
        </button>
    </div>

    <div class="container">
        <div id="content" class="markdown-body">
            <!-- Rendered Markdown goes here -->
        </div>
    </div>

    <script>
        // Retrieve and decode base64 encoded markdown content
        const b64Data = "{b64_content}";
        const markdownText = decodeURIComponent(escape(atob(b64Data)));

        // Setup custom renderer for marked to output diagrams in container
        const renderer = new marked.Renderer();
        
        // Handle code blocks (especially mermaid)
        renderer.code = function(code, language) {
            if (language === 'mermaid') {
                return `<div class="mermaid-container"><div class="mermaid">${code}</div></div>`;
            }
            return `<pre><code class="language-${language || ''}">${escapeHtml(code)}</code></pre>`;
        };

        function escapeHtml(text) {
            return text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        marked.setOptions({
            renderer: renderer,
            gfm: true,
            breaks: true
        });

        // Render markdown to HTML
        document.getElementById('content').innerHTML = marked.parse(markdownText);

        // Initialize Mermaid
        mermaid.initialize({
            startOnLoad: true,
            theme: 'default',
            securityLevel: 'loose',
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true
            }
        });
    </script>
</body>
</html>
""".replace("{b64_content}", b64_content)

    OUTPUT_HTML_PATH.write_text(html_template, encoding="utf-8")
    print(f"Success: HTML exported to {OUTPUT_HTML_PATH}")

if __name__ == "__main__":
    generate_html()
