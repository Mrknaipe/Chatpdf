import os

def tree(dir_path, prefix=""):
    lines = []
    entries = list(os.scandir(dir_path))
    entries.sort(key=lambda e: e.name)
    
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        lines.append(prefix + connector + entry.name)
        
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            lines.extend(tree(entry.path, prefix + extension))
    
    return lines

arbo = "\n".join(tree("CHATPDFRAG"))
print(arbo)
