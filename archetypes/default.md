+++
# ==============================================================================
# Default Content Archetype
# ==============================================================================
# This template is used when creating new content with 'hugo new'
# Example: hugo new posts/my-post/index.md
#
# Template variables (filled in automatically):
# - .Date: Current date/time
# - .File.ContentBaseName: Filename without extension (with spaces)
# ==============================================================================

# Publication date (automatically set to current date/time)
date = '{{ .Date }}'

# Draft status - set to false when ready to publish
# Draft posts are not included in production builds unless --buildDrafts is used
draft = true

# Post title (automatically derived from filename)
# Hyphens are replaced with spaces and title-cased
# Example: "my-new-post" becomes "My New Post"
title = '{{ replace .File.ContentBaseName "-" " " | title }}'

# ==============================================================================
# Optional Frontmatter Fields (uncomment to use)
# ==============================================================================
# description = "Brief description of the post for SEO and previews"
# tags = ["tag1", "tag2"]
# categories = ["category1"]
# series = ["series-name"]
# author = "Michael Hess"
# showAuthor = true
# showDate = true
# showReadingTime = true
# showTableOfContents = true
+++

<!-- Your content goes here -->

<!-- Use <!--more--> to mark where the summary ends -->
