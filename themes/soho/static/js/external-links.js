document.addEventListener("DOMContentLoaded", function() {
    const externalLinkSymbol = '↗'; // Customize as needed
    const links = document.querySelectorAll('a');
  
    links.forEach(link => {
      const href = link.getAttribute('href');
      if (href && (href.startsWith('http') || href.startsWith('//')) && !href.includes(window.location.hostname)) {
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
        //const symbolSpan = document.createElement('span');
        //symbolSpan.classList.add('external-link-icon');
        //symbolSpan.innerHTML = externalLinkSymbol;
        //link.appendChild(symbolSpan);
      }
    });
  });
  