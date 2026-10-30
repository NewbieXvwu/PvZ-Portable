"""Prepared task193 adapter install for the existing SHA-aware task105 importer."""
import argparse
from pathlib import Path
import shutil


def install(root, source):
    importer = root / 'scripts/import_llm_run.py'
    text = importer.read_text()
    start = text.index('def main_with(args)')
    end = text.index('\ndef main(', start)
    text = text[:start] + '''def main_with(args) -> int:
    from pvz_live_import import main_with as live_import
    return live_import(args, sys.modules[__name__])

''' + text[end:]
    shutil.copyfile(source, root / 'scripts/pvz_live_import.py')
    importer.write_text(text)
    batch = root / 'scripts/task105_import_batch.py'
    text = batch.read_text()
    start = text.index('def main()')
    end = text.index("\nif __name__ == '__main__':", start)
    text = text[:start] + """def main():
    from pvz_live_import import batch_main
    return batch_main(ROOT, ENGINE)

""" + text[end:]
    batch.write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    install(args.root, args.source)
