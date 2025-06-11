window.addEventListener('resize', function () {
    const sidebar = parent.document.querySelector('[data-testid="stSidebar"]');
    const content = parent.document.querySelector('.main');

    if (window.innerWidth < 768) {
        if (sidebar && content) {
            sidebar.style.width = '100%';
            content.style.display = 'none';
        }
    } else {
        if (sidebar && content) {
            sidebar.style.width = 'auto';
            content.style.display = 'block';
        }
    }
});
