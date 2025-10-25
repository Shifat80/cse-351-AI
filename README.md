# AI Study Documentation Website

A modern, responsive study documentation website for AI and Machine Learning courses, built with plain HTML, CSS, and JavaScript.

## Features

- **Three-column layout**: Left sidebar for topics, main content area, right sidebar for profile
- **Accordion-style navigation**: Weeks expand to show topics
- **Dynamic content loading**: Markdown files are fetched and rendered on-demand
- **Dark/Light mode toggle**: Switch between themes with smooth transitions
- **Responsive design**: Sidebars collapse into toggle menus on smaller screens
- **Modern design**: Clean, minimal styling with rounded corners and subtle shadows

## Structure

```
/
├── index.html          # Main HTML file
├── style.css           # Stylesheet with CSS variables
├── script.js           # JavaScript for interactivity
├── README.md           # This file
└── Content/            # Markdown content files
    ├── week1/
    │   └── ai_course_notes_week-1.md
    ├── week2/
    │   └── ai_course_notes_week_2.md
    └── ...
```

## Deployment to GitHub Pages

1. **Create a GitHub repository**:
   - Go to GitHub and create a new repository
   - Name it something like `ai-study-docs`
   - Make it public (required for GitHub Pages)

2. **Upload your files**:
   - Clone the repository to your local machine
   - Copy all files (`index.html`, `style.css`, `script.js`, `Content/` folder, `README.md`) to the repository root
   - Commit and push the changes

3. **Enable GitHub Pages**:
   - Go to your repository on GitHub
   - Click on "Settings" tab
   - Scroll down to "Pages" section
   - Under "Source", select "Deploy from a branch"
   - Select "main" branch and "/ (root)" folder
   - Click "Save"

4. **Access your site**:
   - GitHub will provide a URL like `https://yourusername.github.io/ai-study-docs/`
   - It may take a few minutes for the site to be live

## Customization

- **Profile Information**: Edit the profile section in `index.html` with your photo, name, bio, and social links
- **Content**: Add or modify Markdown files in the `Content/` folder structure
- **Styling**: Modify `style.css` to change colors, fonts, or layout
- **Navigation**: Update the navigation structure in `index.html` to match your content

## Technologies Used

- **HTML5**: Semantic markup
- **CSS3**: Flexbox layout, CSS variables, responsive design
- **JavaScript (ES6+)**: DOM manipulation, fetch API
- **Marked.js**: Markdown parsing and rendering
- **Font Awesome**: Icons for UI elements

## Browser Support

- Modern browsers (Chrome, Firefox, Safari, Edge)
- Responsive design works on mobile devices

## License

This project is open source. Feel free to use and modify as needed.