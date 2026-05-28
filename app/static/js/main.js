document.addEventListener('DOMContentLoaded', function () {
    const tables = document.querySelectorAll('table');
    tables.forEach(function (table) {
        table.addEventListener('mouseover', function (event) {
            const row = event.target.closest('tr');
            if (row && row.parentElement.tagName.toLowerCase() !== 'thead') {
                row.style.backgroundColor = '#f8faff';
            }
        });
        table.addEventListener('mouseout', function (event) {
            const row = event.target.closest('tr');
            if (row) {
                row.style.backgroundColor = '';
            }
        });
    });
});
