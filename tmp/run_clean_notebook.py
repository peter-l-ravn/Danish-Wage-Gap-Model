import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
path = Path('Create shock clean.ipynb')
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
client = NotebookClient(nb, timeout=300, kernel_name='python3', resources={'metadata': {'path': str(Path.cwd())}})
client.execute()
nbformat.write(nb, path)
figures = sum('image/png' in o.get('data', {}) for c in nb.cells for o in c.get('outputs', []))
print(f'Notebook ran successfully: {len(nb.cells)} cells, {figures} rendered figures')
import base64
for i,cell in enumerate(nb.cells):
    for out in cell.get('outputs',[]):
        png=out.get('data',{}).get('image/png')
        if png:
            Path(f'tmp/shock_preview_{i}.png').write_bytes(base64.b64decode(png))
