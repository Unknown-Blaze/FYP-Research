import argparse
import sys
from pathlib import Path
import torch


def read_fasta(fp: Path):
    header = None
    seq_chunks = []
    with fp.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, ''.join(seq_chunks)


def main():
    p = argparse.ArgumentParser(description='Generate peptide monomer PDBs using ESMFold (fast).')
    p.add_argument('--input', required=True, help='Input FASTA file (peptide_unique.fasta)')
    p.add_argument('--out', required=True, help='Output directory for PDB files')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max-length', type=int, default=600, help='Max AA length supported by model')
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from esm import pretrained
    except Exception as e:
        print('ERROR: esm not installed. Install with:', file=sys.stderr)
        print('  pip install esm torch --extra-index-url https://download.pytorch.org/whl/cu118', file=sys.stderr)
        sys.exit(1)

    print('Loading ESMFold model ...', flush=True)
    model = pretrained.esmfold_v1().eval().to(args.device)

    fasta = Path(args.input)
    total = 0
    skipped = 0
    existing = 0

    with torch.no_grad():
        for header, seq in read_fasta(fasta):
            total += 1
            # Build a safe base name
            safe = header.replace(' ', '_')
            base = f"{safe}"
            pdb_out = out_dir / f"{base}_rank_001_esmfold.pdb"
            if pdb_out.exists():
                existing += 1
                continue
            if len(seq) == 0 or len(seq) > args.max_length:
                skipped += 1
                continue
            try:
                pdb_string = model.infer_pdb(seq)
                pdb_out.write_text(pdb_string)
            except Exception as ex:
                skipped += 1
                continue

    print(f"Done. total={total} written={total-existing-skipped} existing={existing} skipped={skipped}")


if __name__ == '__main__':
    main()
