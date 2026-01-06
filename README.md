# Michael Hess - Personal Portfolio Website

This repository contains the source code for Michael Hess's personal portfolio website, hosted at [https://michaelhess17.github.io/](https://michaelhess17.github.io/).

## About

This is a static website built with [Hugo](https://gohugo.io/) and the [Blowfish](https://blowfish.page/) theme. The site showcases research, blog posts, and professional information for Michael Hess, a Ph.D. candidate in Neuroscience at Emory University.

## Technologies Used

- **Static Site Generator**: Hugo (v0.87.0+)
- **Theme**: Blowfish (included as a Git submodule)
- **Hosting**: GitHub Pages
- **CI/CD**: GitHub Actions
- **Styling**: Custom CSS overrides

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── hugo.yml          # GitHub Actions workflow for deployment
├── archetypes/
│   └── default.md            # Template for new content
├── assets/
│   ├── css/
│   │   └── custom.css        # Custom CSS overrides for the theme
│   ├── abstract-swirls.jpg   # Background image
│   └── bear_shark.jpg        # Profile image
├── config/
│   └── _default/             # Hugo configuration files
│       ├── languages.en.toml # English language configuration
│       ├── markup.toml       # Markdown rendering settings
│       ├── menus.en.toml     # Site navigation menus
│       ├── module.toml       # Hugo module configuration
│       └── params.toml       # Theme parameters
├── content/
│   ├── _index.md             # Homepage content
│   └── posts/                # Blog posts and articles
├── resources/                # Hugo generated resources (cached)
├── themes/
│   └── blowfish/             # Blowfish theme (Git submodule)
├── hugo.toml                 # Main Hugo configuration
└── README.md                 # This file
```

## Prerequisites

To build and run this site locally, you'll need:

- **Hugo Extended** (v0.87.0 or later)
  - Download from [Hugo Releases](https://github.com/gohugoio/hugo/releases)
  - Must be the "extended" version for SCSS/SASS support
- **Git** for cloning the repository and managing submodules
- **Dart Sass** (optional, for local development)

## Getting Started

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/Michaelhess17/michaelhess17.github.io.git
cd michaelhess17.github.io
```

**Note**: The `--recursive` flag is important to clone the Blowfish theme submodule.

If you already cloned without `--recursive`, initialize the submodule:

```bash
git submodule update --init --recursive
```

### 2. Install Hugo

#### macOS (using Homebrew)
```bash
brew install hugo
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install hugo
```

#### Windows (using Chocolatey)
```bash
choco install hugo-extended
```

Or download directly from the [Hugo releases page](https://github.com/gohugoio/hugo/releases).

### 3. Run the Development Server

```bash
hugo server -D
```

The `-D` flag includes draft content. The site will be available at `http://localhost:1313/`.

The server supports live reload, so changes to content or configuration will automatically refresh in your browser.

## Building the Site

To build the static site for production:

```bash
hugo --minify
```

This generates the static files in the `public/` directory. The `--minify` flag optimizes the output by minifying HTML, CSS, and JavaScript.

## Deployment

The site is automatically deployed to GitHub Pages using GitHub Actions whenever changes are pushed to the `main` branch.

### Deployment Workflow

1. Push changes to the `main` branch
2. GitHub Actions workflow (`.github/workflows/hugo.yml`) is triggered
3. Hugo builds the site with minification
4. Built files are deployed to the `gh-pages` branch
5. GitHub Pages serves the site from the `gh-pages` branch

The workflow includes:
- Installing Hugo Extended (v0.128.0)
- Installing Dart Sass
- Building the site with minification
- Deploying to GitHub Pages

## Creating New Content

### Create a New Blog Post

```bash
hugo new posts/my-new-post/index.md
```

This creates a new post using the archetype template. Edit the frontmatter and content in the generated file.

### Frontmatter Example

```yaml
+++
title = "My New Post"
date = 2024-01-01
draft = false
tags = ["tag1", "tag2"]
categories = ["category1"]
+++
```

## Configuration

### Main Configuration (`hugo.toml`)

- Site metadata (title, base URL, language)
- Author information and social links
- Taxonomies (tags, categories, series)
- Sitemap settings
- Related content configuration

### Theme Configuration (`config/_default/params.toml`)

- Color scheme and appearance
- Layout settings for different page types
- Table of contents settings
- Social sharing options
- Analytics configuration

### Custom Styling (`assets/css/custom.css`)

Custom CSS overrides for the Blowfish theme, including:
- Full-width content layout modifications
- Prose and article width constraints removal
- Table of contents positioning
- Responsive spacing and padding

## Customization

### Changing the Color Scheme

Edit `config/_default/params.toml`:

```toml
colorScheme = "slate"  # Options: blowfish, avocado, ocean, fire, slate, etc.
defaultAppearance = "dark"  # Options: light, dark
```

### Modifying Navigation Menus

Edit `config/_default/menus.en.toml` to add or remove menu items.

### Updating Social Links

Edit `hugo.toml` in the `[params.author]` section to update social media links.

## Theme Documentation

For detailed theme customization options, refer to the [Blowfish documentation](https://blowfish.page/docs/).

## Troubleshooting

### Submodule Issues

If the theme isn't loading:

```bash
git submodule update --init --recursive
```

### Build Errors

Ensure you're using Hugo Extended (not the standard version):

```bash
hugo version
```

Should show "extended" in the output.

### Local Development Issues

Clear Hugo's cache:

```bash
rm -rf resources/
hugo server -D
```

## Contributing

This is a personal portfolio site. If you notice any issues or have suggestions, feel free to open an issue.

## License

Content and code in this repository are for portfolio purposes. The Blowfish theme is licensed under the MIT License.

## Contact

- **Author**: Michael Hess
- **Email**: michael.hess@emory.edu
- **GitHub**: [@michaelhess17](https://github.com/michaelhess17)
- **LinkedIn**: [michael-hess-654245155](https://www.linkedin.com/in/michael-hess-654245155/)
