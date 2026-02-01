// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "About",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-publications",
          title: "Publications",
          description: "Publications in reversed chronological order. Refer to my [Google Scholar](https://scholar.google.com/citations?user=jg2atr8AAAAJ&amp;hl=zh-CN) profile.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "news-started-mphil-program-in-the-department-of-electrical-and-electronic-engineering-the-university-of-hong-kong-advised-by-prof-xiaojuan-qi",
          title: 'Started MPhil program in the Department of Electrical and Electronic Engineering, The University...',
          description: "",
          section: "News",},{id: "news-one-paper-symmetricdiffusers-was-accepted-to-iclr-2025-as-oral-213-11672-top-1-8",
          title: 'One paper SymmetricDiffusers was accepted to ICLR 2025 as Oral (213/11672, Top 1.8%)!...',
          description: "",
          section: "News",},{id: "news-one-paper-diffugts-was-accepted-to-cvpr-2025",
          title: 'One paper DiffuGTS was accepted to CVPR 2025!',
          description: "",
          section: "News",},{id: "news-received-aistats-2025-best-reviewer-award",
          title: 'Received AISTATS 2025 Best Reviewer Award! 🎉',
          description: "",
          section: "News",},{id: "news-joined-kling-team-kuaishou-technology-as-a-research-intern",
          title: 'Joined KLING Team, Kuaishou Technology as a Research Intern.',
          description: "",
          section: "News",},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%79%64%6C%69%6E%37%31%38@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-x',
        title: 'X',
        section: 'Socials',
        handler: () => {
          window.open("https://twitter.com/ydlin718", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/linYDTHU", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=jg2atr8AAAAJ", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/donglin-yang-473635212", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
