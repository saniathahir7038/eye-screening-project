"""Download pinned SLID files or reuse a verified local archive, then extract."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile
import zlib

ROOT = Path(__file__).resolve().parent
COMMIT = 'b1c518b42d9d04873afbe6864258bd3bbba1cd28'
ZIP_SHA256 = 'cf74278dbd0c1782e58db834e8826a87f4038c007a5a4133b26131cf1b2a9607'


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (destination / member.filename).resolve()
            if (not target.is_relative_to(destination) or ':' in member.filename
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f'Unsafe archive member: {member.filename}')
        z.extractall(destination)


def extraction_complete(archive, destination):
    """Check actual extracted bytes; a marker alone cannot prove completeness."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if member.is_dir():
                continue
            target = (destination / member.filename).resolve()
            if (not target.is_relative_to(destination) or not target.is_file()
                    or target.stat().st_size != member.file_size):
                return False
            crc = 0
            with target.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    crc = zlib.crc32(chunk, crc)
            if crc != member.CRC:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Reuse a local ZIP matching the pinned SHA-256')
    args = parser.parse_args()
    if args.archive and (not args.archive.is_file() or sha256(args.archive) != ZIP_SHA256):
        raise ValueError('Supplied archive is missing, incomplete or has an incorrect SHA-256.')
    repo = ROOT / 'data/SLID'
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        subprocess.run(['git', 'clone', 'https://github.com/xumingyu-hub/SLID.git', str(repo)],
                       env={**os.environ, 'GIT_LFS_SKIP_SMUDGE': '1'}, check=True)
    subprocess.run(['git', '-C', str(repo), 'checkout', COMMIT], check=True)
    archive = args.archive or repo / 'Original_Slit-lamp_Images.zip'
    if not archive.exists() or sha256(archive) != ZIP_SHA256:
        subprocess.run(['git', '-C', str(repo), 'lfs', 'pull'], check=True, timeout=180)
    if sha256(archive) != ZIP_SHA256:
        raise ValueError('SLID ZIP checksum mismatch or unresolved Git LFS pointer.')
    destination = ROOT / 'data/images'
    marker = destination / '.extracted.json'
    if not extraction_complete(archive, destination):
        safe_extract(archive, destination)
    marker.write_text(json.dumps({'zip_sha256': ZIP_SHA256}), encoding='utf-8')
    provenance = {'repository': 'https://github.com/xumingyu-hub/SLID', 'commit': COMMIT,
                  'zip_sha256': ZIP_SHA256, 'zip_bytes': archive.stat().st_size,
                  'annotations_sha256': sha256(repo / 'Annotations.csv')}
    (ROOT / 'data/provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
