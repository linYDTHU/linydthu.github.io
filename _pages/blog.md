---
layout: default
permalink: /blog/
title: Blog
nav: true
nav_order: 2
pagination:
  enabled: true
  collection: posts
  permalink: /page/:num/
  per_page: 5
  sort_field: date
  sort_reverse: true
  trail:
    before: 1 # The number of links before the current page
    after: 3 # The number of links after the current page
---

<link rel="stylesheet" href="{{ '/assets/css/research-blog.css' | relative_url }}?v=20261114-review2">
<div class="post research-blog-page">
  <header class="post-header">
    <h1 class="post-title">{{ site.blog_name }}</h1>
  </header>
  <div class="research-posts">
    {% if page.pagination.enabled %}
      {% assign postlist = paginator.posts %}
    {% else %}
      {% assign postlist = site.posts %}
    {% endif %}
    {% for item in postlist %}
      {% include research-post.liquid item=item heading="h2" %}
    {% endfor %}
  </div>
  {% if page.pagination.enabled %}
    {% include pagination.liquid %}
  {% endif %}
</div>
