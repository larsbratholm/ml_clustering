#!/usr/bin/env python2

import sys
import fml
import numpy as np
from fml.math import manhattan_distance, l2_distance
import fml.clustering
import time
import argparse

def parse_first_frame(lines):
    V = list()

    charges = list()
    atom_names = list()
    residues = list()
    coordinates = list()
    for i, line in enumerate(lines):
        if line.startswith("TER") or line.startswith("END"):
            remaining_lines = lines[i+1:]
            break
        if line.startswith("ATOM"):
            atom_names.append(line[12:16].strip())
            if "N" in atom_names[-1]:
                charges.append(7)
            elif "O" in atom_names[-1]:
                charges.append(8)
            elif "S" in atom_names[-1]:
                charges.append(16)
            elif "C" in atom_names[-1]:
                charges.append(6)
            else:
                # ignore H. could be added.
                break
            residues.append(int(line[22:26]))
            coordinates.append(np.asarray([line[30:38], line[38:46], line[46:54]], dtype=float))
    charges = np.asarray(charges, dtype=int)
    atom_names = np.asarray(atom_names)
    residues = np.asarray(residues, dtype=int)
    coordinates = np.asarray(coordinates, dtype=float)
    return charges, atom_names, residues, coordinates, remaining_lines

charge_to_atype = {1:"H", 6:"C", 7:"N", 8:"O", 16:"S"}

def parse_frame(lines):
    coordinates = []
    for i, line in enumerate(lines):
        if line.startswith("TER"):
            break
        if line.startswith("ATOM"):
            coordinates.append(np.asarray([line[30:38], line[38:46], line[46:54]], dtype=float))
    return coordinates

def parse_trajectory(lines):
    all_coordinates = []
    coordinates = []
    for i, line in enumerate(lines):
        if line.startswith("TER"):
            all_coordinates.append(np.asarray(coordinates))
            coordinates = []
            continue
        if line.startswith("ATOM"):
            coordinates.append(np.asarray([line[30:38], line[38:46], line[46:54]], dtype=float))
    return all_coordinates

def parse_args():
    description = ""
    epilog = ""

    parser = argparse.ArgumentParser(
            description = description,
            formatter_class = argparse.RawDescriptionHelpFormatter,
            epilog = epilog)

    parser.add_argument('-i', '--input', help='Structures in pdb format.\n \
                                               Assume same header and atom order in all files.\n \
                                               If only one file is given, assume it to be a trajectory. \
                                               ', action='store', type=str, nargs='+', default=[])
    parser.add_argument('-m', '--method', help='Method to use.',
                                                choices = ['bob','cm','ucm','acm','uacm'], action='store', default='ucm')
    parser.add_argument('-o', '--output', help='Output distances to file', action='store', default = "-")
    parser.add_argument('-e', '--exponents', action='store', type=int, default=[1,1,1], choices = (1,2), nargs='+')
    parser.add_argument('-c', '--cutoff', action='store', type=float, default=6)
    parser.add_argument('--ca', action='store_true', default=False)


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def create_mols(args):
    t = time.time()
    filenames = args.input
    all_coordinates = []
    if len(filenames) == 1:
        # trajectory
        with open(filenames[0]) as f:
            lines = f.readlines()
            charges, atom_names, residues, coordinates, remaining_lines = parse_first_frame(lines)
            all_coordinates = parse_trajectory(remaining_lines)
            all_coordinates = [coordinates] + all_coordinates
    else:
        with open(filenames[0]) as f:
            lines = f.readlines()
        charges, atom_names, residues, coordinates, _ = parse_first_frame(lines)
        all_coordinates.append(coordinates)
        for filename in filenames[1:]:
            with open(filename) as f:
                lines = f.readlines()
            all_coordinates.append(parse_frame(lines))
    atomtypes = np.asarray([charge_to_atype[charge] for charge in charges])
    print "Parsed input in %.1f seconds" % (time.time() - t)

    # check if some of the structures have the same coordinates
    #for i, coord_i in enumerate(all_coordinates):
    #    for j, coord_j in enumerate(all_coordinates):
    #        if j <= i:
    #            continue
    #        if np.allclose(coord_i[:50], coord_j[:50]):
    #            print filenames[i], filenames[j]

    t = time.time()
    mols = np.empty(len(all_coordinates), dtype=object)
    for i in range(mols.size):
        mol = fml.Molecule()
        mol.natoms = charges.size
        mol.atomtypes = atomtypes
        mol.atomnames = atom_names
        mol.nuclear_charges = charges
        mol.coordinates = all_coordinates[i]
        mol.residues = residues
        generate_descriptor(mol, args)
        mols[i] = mol
    print "Generated descriptors in %.1f seconds" % (time.time() - t)
    return mols

def generate_descriptor(mol, args):
    if args.method == "bob":
        mol.generate_bob(size = mol.natoms, asize = dict(zip(*np.unique(mol.atomtypes, return_counts = True))))
    elif args.method == "cm":
        mol.generate_coulomb_matrix(size = mol.natoms)
    elif args.method == "ucm":
        mol.generate_unsorted_coulomb_matrix(size = mol.natoms)
    elif args.method == "acm":
        # empirical fit
        size = int(0.5 + 3*args.cutoff**2.1)
        mol.generate_atomic_coulomb_matrix(size = size, cutoff = args.cutoff)
    elif args.method == "uacm":
        mol.generate_atomic_coulomb_matrix(size = mol.natoms, cutoff = args.cutoff)

def generate_distances(mols, args):
    t = time.time()
    if args.method in ["bob", "cm", "ucm"]:
        X = np.asarray([mol.descriptor for mol in mols], dtype=float)
        if args.exponents[0] == 1:
            D = manhattan_distance(X.T,X.T)
        else:
            # prevent overflow in l2
            X = X[:,(X > 1.5).all(0)]
            D = l2_distance(X.T,X.T)
    else:
        #X = np.asarray([np.concatenate(mol.local_descriptor) for mol in mols])
        #D = manhattan_distance(X.T, X.T)
        # initialize
        D = np.zeros((len(mols), len(mols)), dtype=float)
        resids = np.unique(mols[0].residues)
        for res in resids:
            if args.ca:
                indices = np.where((mols[0].residues == res) & (mols[0].atomnames == "CA"))[0]
            else:
                indices = np.where(mols[0].residues == res)[0]
            D_res = np.zeros((len(mols), len(mols)), dtype=float)
            for i in indices:
                X = np.asarray([mol.local_descriptor[i] for mol in mols])
                if args.exponents[2] == 1:
                    D_res += manhattan_distance(X.T, X.T)**args.exponents[1]
                else:
                    D_res += l2_distance(X.T, X.T)**args.exponents[1]
            if args.exponents[1] == 2:
                np.sqrt(D_res, out=D_res)
            D += D_res**args.exponents[0]
    if args.exponents[0] == 2:
        np.sqrt(D, out=D)

    print "Generated distances in %.1f seconds" % (time.time() - t)
    return D

def write_output(D, args):
    n = D.shape[0]
    if args.output == "-":
        f = sys.stdout
    else:
        f = open(args.output, "w")
    for i in range(1,n):
        for j in range(i):
            f.write("%d %d %.6f\n" % (i, j, D[i,j]))
    f.close()

if __name__ == "__main__":
    args = parse_args()
    mols = create_mols(args)
    if args.method == "acm":
        out = args.output
        for i in (1,2):
            for j in (1,2):
                for k in (1,2):
                    args.exponents = [i,j,k]
                    args.output = out + "%d_%d_%d.txt" % (i,j,k)
                    D = generate_distances(mols, args)
                    write_output(D, args)
    else:
        D = generate_distances(mols, args)
        write_output(D, args)
