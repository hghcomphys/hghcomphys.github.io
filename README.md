# Personal site 
The code repository for my personal [website](https://hghcomphys.github.io/).


## Features

- [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) theme 
    1. Installs Ruby dependencies listed in the `Gemfile`.
        ```bash
        pixi run bundle install
        ```
        
    2. Start a local Jekyll website server using the Ruby environment
        ```bash
        pixi run bundle exec jekyll serve
        ```

[Pixi](https://pixi.prefix.dev/latest/installation/) is used to manege project's dependencies.


- Google analytics \
Analytics are disabled by default in development. To enable when testing/building locally be sure to set `JEKYLL_ENV=production` to force the environment to production.

- Using different [layouts](https://mmistakes.github.io/minimal-mistakes/docs/layouts/#home-page-layout)

- [Disqus](https://disqus.com/) comments provider

- add _LaTeX_ support to Markdown

- Include videos
    ```
    {% include video id="IaU-Y2aWQnc" provider="youtube" %}
    ```
