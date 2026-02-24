import re

def update_file(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Tailwind script replacement
    if 'tailwind.config' not in content and filename.endswith('.html'):
        script_to_add = """    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        background: 'oklch(0.15 0 0)',
                        foreground: 'oklch(0.98 0 0)',
                        card: 'oklch(0.2 0 0)',
                        cardForeground: 'oklch(0.98 0 0)',
                        popover: 'oklch(0.2 0 0)',
                        popoverForeground: 'oklch(0.98 0 0)',
                        primary: {
                            DEFAULT: 'oklch(0.55 0.25 300)',
                            foreground: 'oklch(0.98 0 0)'
                        },
                        secondary: {
                            DEFAULT: 'oklch(0.25 0 0)',
                            foreground: 'oklch(0.98 0 0)'
                        },
                        muted: {
                            DEFAULT: 'oklch(0.25 0 0)',
                            foreground: 'oklch(0.7 0 0)'
                        },
                        accent: {
                            DEFAULT: 'oklch(0.3 0.1 300)',
                            foreground: 'oklch(0.98 0 0)'
                        },
                        destructive: {
                            DEFAULT: 'oklch(0.4 0.15 25)',
                            foreground: 'oklch(0.98 0 0)'
                        },
                        border: 'oklch(0.25 0 0)',
                        input: 'oklch(0.25 0 0)',
                        ring: 'oklch(0.55 0.25 300)'
                    },
                    borderRadius: {
                        sm: 'calc(0.5rem - 4px)',
                        md: 'calc(0.5rem - 2px)',
                        lg: '0.5rem',
                        xl: 'calc(0.5rem + 4px)',
                        '2xl': 'calc(0.5rem + 8px)',
                        '3xl': 'calc(0.5rem + 12px)'
                    }
                }
            }
        }
    </script>"""
        content = content.replace('<script src="https://unpkg.com/lucide@latest"></script>',
                                '<script src="https://unpkg.com/lucide@latest"></script>\n' + script_to_add)

    # Dictionary of class replacements
    replacements = {
        r'bg-neutral-950/95': 'bg-background/95',
        r'bg-neutral-950': 'bg-background',
        r'text-neutral-200': 'text-foreground',
        r'bg-neutral-900/50': 'bg-card/50',
        r'bg-neutral-900': 'bg-card',
        r'border-neutral-800/50': 'border-border/50',
        r'border-neutral-800': 'border-border',
        r'bg-neutral-800': 'bg-secondary',
        r'hover:bg-neutral-700': 'hover:bg-secondary/80',
        r'text-neutral-500': 'text-muted-foreground',
        r'text-neutral-400': 'text-muted-foreground',
        r'text-neutral-600': 'text-muted-foreground',
        r'text-neutral-700': 'text-muted-foreground',
        
        r'bg-indigo-600/20': 'bg-primary/20',
        r'bg-indigo-600': 'bg-primary',
        r'hover:bg-indigo-500': 'hover:bg-primary/90',
        r'text-indigo-500': 'text-primary',
        r'text-indigo-400': 'text-primary',
        r'shadow-indigo-600/20': 'shadow-primary/20',
        r'shadow-indigo-500/10': 'shadow-primary/10',
        r'focus:border-indigo-500': 'focus:border-primary',
        r'focus:ring-indigo-500': 'focus:ring-ring',

        r'text-rose-500': 'text-destructive',
        r'hover:text-rose-500': 'hover:text-destructive',
        r'bg-rose-950/40': 'bg-destructive/10',
        r'border-rose-900/50': 'border-destructive/20',
        r'hover:bg-rose-900/40': 'hover:bg-destructive/20',
        r'hover:text-rose-400': 'hover:text-destructive/80',
        r'hover:bg-rose-500/20': 'hover:bg-destructive/20',
        
        # Explicit color values in JS ui.js
        r'rgba\(99, 102, 241,': 'rgba(139, 92, 246,', # Using a standard purple since OKLCH string in rgba won't work easily here, wait, better use exact css var
    }

    for pattern, repl in replacements.items():
        content = re.sub(r'\b' + pattern + r'\b', repl, content)

    with open(filename, 'w') as f:
        f.write(content)

update_file("index.html")
update_file("js/ui.js")
print("Done")
