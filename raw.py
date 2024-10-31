import os
import sys

def get_total_size():
    site_packages = next(p for p in sys.path if 'site-packages' in p)
    total_size = 0
    for dirpath, _, filenames in os.walk(site_packages):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size/(1024 ** 2)

print(f"Total size of installed libraries: {get_total_size(): .2f}MB")