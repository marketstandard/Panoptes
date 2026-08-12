```python
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)
        
    dir_path = sys.argv[1]
    
    if not os.path.isdir(dir_path):
        print(f"Error: The directory '{dir_path}' does not exist or is not a directory.")
        sys.exit(1)

    files_with_sizes = []
    
    for entry in os.scandir(dir_path):
        if entry.is_file(follow_symlinks=False):
            try:
                files_with_sizes.append((entry.name, entry.stat().st_size))
            except OSError:
                pass
                
    files_with_sizes.sort(key=lambda x: x[1], reverse=True)
    
    for name, size in files_with_sizes[:5]:
        size_mb = size / (1024 * 1024)
        print(f"{name} - {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
```