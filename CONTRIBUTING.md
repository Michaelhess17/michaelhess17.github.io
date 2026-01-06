# Contributing Guide

This document provides guidance for working with and contributing to this Hugo site.

## Table of Contents

- [Development Workflow](#development-workflow)
- [Content Guidelines](#content-guidelines)
- [Code Style](#code-style)
- [Testing Changes](#testing-changes)
- [Deployment](#deployment)

## Development Workflow

### Setting Up Your Environment

1. **Clone the repository with submodules**:
   ```bash
   git clone --recursive https://github.com/Michaelhess17/michaelhess17.github.io.git
   cd michaelhess17.github.io
   ```

2. **Install Hugo Extended** (v0.87.0 or later):
   - macOS: `brew install hugo`
   - Ubuntu/Debian: `sudo apt-get install hugo`
   - Or download from [Hugo Releases](https://github.com/gohugoio/hugo/releases)

3. **Start the development server**:
   ```bash
   hugo server -D
   ```
   Visit `http://localhost:1313/` to see your changes live.

### Making Changes

1. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines below

3. **Test locally** using the Hugo development server

4. **Commit your changes** with clear, descriptive messages:
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

5. **Push your branch** and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

## Content Guidelines

### Creating New Blog Posts

Use Hugo's `new` command to create posts from the archetype template:

```bash
hugo new posts/my-new-post/index.md
```

This creates a new directory under `content/posts/` with an `index.md` file.

### Post Frontmatter

Each post should include proper frontmatter. Here's a complete example:

```yaml
+++
title = "My Awesome Post"
date = 2024-01-15
draft = false
description = "A brief description for SEO and social sharing"
tags = ["neuroscience", "machine-learning"]
categories = ["Research"]
series = ["My Series"]
showTableOfContents = true
showReadingTime = true
+++
```

### Frontmatter Fields

- **Required**:
  - `title`: Post title
  - `date`: Publication date (YYYY-MM-DD format)
  - `draft`: Set to `false` when ready to publish

- **Recommended**:
  - `description`: SEO description (150-160 characters)
  - `tags`: Relevant tags (2-5 tags recommended)
  - `categories`: Broader categorization

- **Optional**:
  - `series`: Group related posts
  - `showTableOfContents`: Override default TOC setting
  - `showReadingTime`: Override default reading time display
  - `featured`: Path to featured/hero image

### Content Organization

```
content/
├── _index.md           # Homepage content
└── posts/              # Blog posts
    ├── post-name-1/
    │   ├── index.md    # Post content
    │   └── featured.jpg # Hero image (optional)
    └── post-name-2/
        └── index.md
```

### Writing Style

- Use clear, concise language
- Break content into sections with headings (##, ###)
- Include code examples in fenced code blocks with language specified:
  ````markdown
  ```python
  def hello_world():
      print("Hello, world!")
  ```
  ````
- Add images to the post directory and reference relatively:
  ```markdown
  ![Alt text](./image.jpg)
  ```

## Code Style

### Configuration Files

- **TOML files**: Use consistent indentation (2 spaces)
- **Comments**: Add clear comments for all configuration sections
- **Organization**: Group related settings together

### Custom CSS

- **Comments**: Document the purpose of each CSS section
- **Organization**: Group related styles together
- **Specificity**: Use the minimum specificity needed
- **Format**: Follow consistent formatting (see `assets/css/custom.css`)

### Markdown

- Use ATX-style headings (`##` instead of underlines)
- Add blank lines between paragraphs and sections
- Use meaningful link text (not "click here")
- Add alt text to all images

## Testing Changes

### Local Testing

1. **Start the development server**:
   ```bash
   hugo server -D
   ```

2. **Check different pages**:
   - Homepage
   - Individual blog posts
   - Category/tag archive pages
   - Navigation menus

3. **Test responsive design**:
   - Resize browser window
   - Test on mobile device or emulator
   - Check at different breakpoints

4. **Verify content**:
   - Check for typos and formatting
   - Ensure images load correctly
   - Test internal and external links
   - Verify code blocks render properly

### Build Testing

Build the production version to catch any build errors:

```bash
hugo --minify
```

Check the `public/` directory to ensure files are generated correctly.

### Accessibility

- Use semantic HTML
- Provide alt text for images
- Ensure sufficient color contrast
- Test with keyboard navigation

## Deployment

### Automatic Deployment

The site automatically deploys to GitHub Pages when changes are merged to the `main` branch:

1. GitHub Actions workflow triggers (`.github/workflows/hugo.yml`)
2. Hugo builds the site with minification
3. Built files deploy to GitHub Pages
4. Site is live at https://michaelhess17.github.io/

### Manual Deployment

If needed, you can manually trigger deployment:

1. Go to the [Actions tab](https://github.com/Michaelhess17/michaelhess17.github.io/actions)
2. Select "Deploy Hugo site to Pages" workflow
3. Click "Run workflow"

### Deployment Checklist

Before merging to `main`:

- [ ] All content is reviewed and proofread
- [ ] Images are optimized (compressed, appropriate size)
- [ ] Draft status is set to `false`
- [ ] Local build succeeds without errors
- [ ] Links are tested and functional
- [ ] Changes are tested locally
- [ ] Commit messages are clear and descriptive

## Theme Updates

The Blowfish theme is included as a Git submodule. To update it:

```bash
cd themes/blowfish
git pull origin main
cd ../..
git add themes/blowfish
git commit -m "Update Blowfish theme to latest version"
```

## Troubleshooting

### Submodule Issues

If the theme isn't loading:
```bash
git submodule update --init --recursive
```

### Cache Issues

Clear Hugo's cache if you see stale content:
```bash
rm -rf resources/
hugo server -D
```

### Build Errors

1. Check Hugo version: `hugo version` (should show "extended")
2. Verify all frontmatter is valid TOML/YAML
3. Check for unclosed HTML tags or shortcodes
4. Review error messages for specific file/line numbers

## Getting Help

- **Hugo Documentation**: https://gohugo.io/documentation/
- **Blowfish Theme Docs**: https://blowfish.page/docs/
- **Open an Issue**: For bugs or feature requests

## Questions?

If you have questions about contributing, feel free to reach out to [michael.hess@emory.edu](mailto:michael.hess@emory.edu).
