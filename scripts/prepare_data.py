#!/usr/bin/env python
"""
Data preparation pipeline for DensityResNet.

Aggregates NWChem RT-TDDFT data into training-ready format:
1. Parse restart files -> density_series.npy
2. Extract overlap matrix -> overlap.npy
3. Sync field data -> field.npy
4. Validate all outputs

Usage:
    python scripts/prepare_data.py --restart-dir perm/ --nwchem-out h2_plus.out --output-dir data/
    python scripts/prepare_data.py --restart-dir perm/ --nwchem-out h2_plus.out --field-file field.dat --output-dir data/
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import glob
import os
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare training data from NWChem RT-TDDFT output'
    )
    parser.add_argument(
        '--restart-dir', type=str, required=True,
        help='Directory containing NWChem restart files'
    )
    parser.add_argument(
        '--nwchem-out', type=str, default=None,
        help='NWChem output file (for overlap matrix extraction)'
    )
    parser.add_argument(
        '--field-file', type=str, default=None,
        help='External field data file (optional)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data',
        help='Output directory for processed files'
    )
    parser.add_argument(
        '--n-electrons', type=float, nargs='+', default=None,
        help='Expected electrons per spin channel (for validation)'
    )
    parser.add_argument(
        '--skip-validation', action='store_true',
        help='Skip data validation step'
    )
    return parser.parse_args()


def aggregate_restart_files(directory, output_file):
    """Aggregate restart files into density_series.npy"""
    import torch

    # Filter for files ending in numeric digits
    file_pattern = os.path.join(directory, "*rt_restart*")
    files = glob.glob(file_pattern)
    files = [f for f in files if re.search(r'\.\d+$', f)]
    files.sort(key=lambda x: int(re.search(r'\.(\d+)$', x).group(1)))

    if not files:
        raise FileNotFoundError(f"No restart files found in {directory}")

    print(f"Found {len(files)} restart files. Processing...")

    rho_list = []
    times = []

    for i, fp in enumerate(files):
        t, rho = parse_nwchem_restart(fp)
        rho_list.append(rho)
        times.append(t)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(files)}...")

    # Stack time steps: (Time, Spins, N, N)
    rho_series = torch.stack(rho_list)
    time_series = np.array(times)

    # Save density series
    np.save(output_file, rho_series.numpy())

    # Save timestamps separately
    time_file = str(output_file).replace('.npy', '_times.npy')
    np.save(time_file, time_series)

    print(f"  Shape: {rho_series.shape} (Time, Spins, N_basis, N_basis)")
    print(f"  Saved: {output_file}")
    print(f"  Times: {time_file}")

    return rho_series.numpy(), time_series


def parse_nwchem_restart(filepath):
    """Parse a single NWChem RT-TDDFT restart file for Open-Shell systems."""
    import torch

    with open(filepath, 'r') as f:
        content = f.read().split()

    # Extract metadata
    nbf_idx = content.index('nbf_ao')
    n_basis = int(content[nbf_idx + 1])

    nmats_idx = content.index('nmats')
    n_mats = int(content[nmats_idx + 1])

    t_idx = content.index('t')
    time = float(content[t_idx + 1])

    # Extract data stream (after 'checksum' value)
    checksum_idx = content.index('checksum')
    data_start_idx = checksum_idx + 2
    raw_data = np.array(content[data_start_idx:], dtype=np.float64)

    # Validate size
    floats_per_matrix = (n_basis * n_basis) * 2
    expected_total = n_mats * floats_per_matrix

    if raw_data.size != expected_total:
        raise ValueError(
            f"Size mismatch in {filepath}. "
            f"Expected {expected_total} floats, got {raw_data.size}"
        )

    # Process matrices (Alpha & Beta)
    spins_list = []

    for m in range(n_mats):
        start = m * floats_per_matrix
        end = start + floats_per_matrix
        matrix_data = raw_data[start:end]

        # Interleaved: Even->Real, Odd->Imag
        real_part = matrix_data[0::2].reshape(n_basis, n_basis)
        imag_part = matrix_data[1::2].reshape(n_basis, n_basis)

        rho_real = torch.tensor(real_part, dtype=torch.float64)
        rho_imag = torch.tensor(imag_part, dtype=torch.float64)
        rho_complex = torch.complex(rho_real, rho_imag)
        spins_list.append(rho_complex)

    # Stack Spins: (2, N, N)
    rho_combined = torch.stack(spins_list)

    return time, rho_combined


def parse_overlap_matrix(filename, output_file):
    """Parse overlap matrix from NWChem output."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find the overlap matrix section
    start_idx = None
    for i, line in enumerate(lines):
        if 'Overlap matrix' in line:
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("Could not find 'Overlap matrix' in output file")

    # Skip header and dashes
    data_start = start_idx + 2

    # Parse all matrix elements
    elements = {}
    max_row = 0
    max_col = 0

    for i in range(data_start, len(lines)):
        line = lines[i].strip()

        # Stop at empty line or next section
        if not line or '=' in line or 'Matrix' in line or '---' in line:
            if elements:
                break

        parts = line.split()
        if len(parts) >= 3:
            try:
                row = int(parts[0]) - 1
                col = int(parts[1]) - 1
                value = float(parts[2])

                elements[(row, col)] = value
                max_row = max(max_row, row)
                max_col = max(max_col, col)
            except (ValueError, IndexError):
                continue

    if not elements:
        raise ValueError("No matrix elements found in overlap section")

    # Construct matrix
    nbf = max(max_row, max_col) + 1
    S = np.zeros((nbf, nbf))

    for (row, col), value in elements.items():
        S[row, col] = value

    # Symmetrize if needed
    if not np.allclose(S, S.T, atol=1e-10):
        S = (S + S.T) / 2

    np.save(output_file, S)
    print(f"  Shape: {S.shape}")
    print(f"  Trace: {np.trace(S):.6f} (expected: {nbf})")
    print(f"  Saved: {output_file}")

    return S


def sync_field_data(density_times, field_file, output_file, tolerance=1e-5):
    """Sync field data to density timestamps."""
    # Load field data
    field_data = np.array([])
    if field_file is not None and Path(field_file).exists():
        try:
            field_data = np.loadtxt(field_file)
        except Exception:
            field_data = np.array([])

    matched_rows = []

    if field_data.size == 0:
        print("  No external field data found. Using zero field.")
        for t in density_times:
            matched_rows.append([t, 0.0, 0.0, 0.0])
    else:
        field_times = field_data[:, 0]
        for target_t in density_times:
            idx = (np.abs(field_times - target_t)).argmin()
            diff = abs(field_times[idx] - target_t)

            if diff < tolerance:
                matched_rows.append(field_data[idx])
            else:
                print(f"  Warning: No matching field for t={target_t:.6f}")
                matched_rows.append([target_t, 0.0, 0.0, 0.0])

    matched_array = np.array(matched_rows)

    # Save field values only (E_x, E_y, E_z)
    field_values = matched_array[:, 1:4].astype(np.float32)
    np.save(output_file, field_values)

    print(f"  Shape: {field_values.shape} (timesteps, 3)")
    print(f"  Field range: [{field_values.min():.6e}, {field_values.max():.6e}]")
    print(f"  Saved: {output_file}")

    return field_values


def validate_data(density_series, overlap_matrix, field_data, n_electrons=None):
    """Validate the prepared data."""
    from data.preprocessing import validate_density_data

    print("\nValidating data...")

    issues = validate_density_data(density_series, overlap_matrix, field_data)

    if issues:
        print("  Warnings:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  All validations passed!")

    # Additional checks
    n_steps, n_spin, nbf, _ = density_series.shape

    print(f"\n  Data summary:")
    print(f"    Timesteps: {n_steps}")
    print(f"    Spin channels: {n_spin}")
    print(f"    Basis functions: {nbf}")

    # Check trace conservation
    traces = []
    for t in range(min(n_steps, 100)):
        for s in range(n_spin):
            rho = density_series[t, s]
            trace = np.trace(rho @ overlap_matrix).real
            traces.append((s, trace))

    alpha_traces = [t[1] for t in traces if t[0] == 0]
    beta_traces = [t[1] for t in traces if t[0] == 1]

    print(f"\n  Trace(rho*S) statistics (first 100 steps):")
    print(f"    Alpha: {np.mean(alpha_traces):.6f} +/- {np.std(alpha_traces):.2e}")
    if beta_traces:
        print(f"    Beta:  {np.mean(beta_traces):.6f} +/- {np.std(beta_traces):.2e}")

    if n_electrons is not None:
        print(f"\n  Expected electrons: {n_electrons}")
        if abs(np.mean(alpha_traces) - n_electrons[0]) > 0.01:
            print(f"    Warning: Alpha trace differs from expected!")
        if len(n_electrons) > 1 and beta_traces:
            if abs(np.mean(beta_traces) - n_electrons[1]) > 0.01:
                print(f"    Warning: Beta trace differs from expected!")


def main():
    args = parse_args()

    print("=" * 60)
    print("DensityResNet Data Preparation")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Step 1: Aggregate density matrices
    print("\n" + "-" * 60)
    print("Step 1: Aggregating density matrices...")
    density_file = output_dir / 'density_series.npy'
    density_series, density_times = aggregate_restart_files(
        args.restart_dir,
        density_file
    )

    # Step 2: Extract overlap matrix
    overlap_matrix = None
    if args.nwchem_out is not None:
        print("\n" + "-" * 60)
        print("Step 2: Extracting overlap matrix...")
        overlap_file = output_dir / 'overlap.npy'
        overlap_matrix = parse_overlap_matrix(args.nwchem_out, overlap_file)
    else:
        print("\n" + "-" * 60)
        print("Step 2: Skipping overlap extraction (no NWChem output provided)")
        print("  Note: You'll need to provide overlap.npy separately")

    # Step 3: Sync field data
    print("\n" + "-" * 60)
    print("Step 3: Synchronizing field data...")
    field_file_out = output_dir / 'field.npy'
    field_data = sync_field_data(
        density_times,
        args.field_file,
        field_file_out
    )

    # Step 4: Validation
    if not args.skip_validation and overlap_matrix is not None:
        print("\n" + "-" * 60)
        print("Step 4: Validating data...")
        validate_data(
            density_series,
            overlap_matrix,
            field_data,
            n_electrons=args.n_electrons
        )

    # Summary
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {output_dir}/density_series.npy")
    if overlap_matrix is not None:
        print(f"  - {output_dir}/overlap.npy")
    print(f"  - {output_dir}/field.npy")
    print(f"\nNext steps:")
    print(f"  1. Create/update config file with these paths")
    print(f"  2. Run: python scripts/train.py --config your_config.json")


if __name__ == '__main__':
    main()
