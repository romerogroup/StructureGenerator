#!/usr/bin/env python3

__author__='Pedram Tavadze'
__date__= "02-07-2020"

import numpy as np
import pychemia
import pyxtal.crystal as crystal
import pyxtal.symmetry as symmetry
import re
from multiprocessing import Pool
import itertools
import argparse
import string
import warnings
warnings.filterwarnings("ignore")

oxidations = {  'H':  (-1,1) ,
                'He': () ,
                'Li': (1,) ,
                'Be': (2,) ,
                'B':  (3,) ,
                #'C':  (-4, -3, -2, -1, 1, 2, 3, 4) ,
                'C':  (-4, 4) ,
                'N':  (-3, 3, 5) ,
                'O':  (-2,) ,
                'F':  (-1,) ,
                'Ne': () ,
                'Na': (1,) ,
                'Mg': (2,) ,
                'Al': (3,) ,
                'Si': (-4,4) ,
                'P':  (-3, 3, 5) ,
                'S':  (-2, 2, 4, 6) ,
                'Cl': (-1, 1, 3, 5, 7) ,
                'Ar': () ,
                'K':  (1,) ,
                'Ca': (2,) ,
                'Sc': (3,) ,
                'Ti': (4,) ,
                'V':  (5,) ,
                'Cr': (3,6) ,
                'Mn': (2,4,7) ,
                'Fe': (2, 3, 6) ,
                'Co': (2, 3) ,
                'Ni': (2,) ,
                'Cu': (2,) ,
                'Zn': (2,) ,
                'Ga': (3,) ,
                'Ge': (-4,2,4) ,
                'As': (-3, 3, 5) ,
                'Se': (-2, 2, 4, 6) ,
                'Br': (-1, 1, 3, 5, 7) ,
                'Kr': () ,
                'Rb': (1,) ,
                'Sr': (2,) ,
                'Y' : (3,) ,
                'Zr': (4,) ,
                'Nb': (5,) ,
                'Mo': (4, 6) ,
                'Tc': () ,
                'Ru': (2, 3, 4) ,
                'Rh': (3,) ,
                'Pd': (2,4) ,
                'Ag': (1,) ,
                'Cd': (2,) ,
                'In': (3,) ,
                'Sn': (-4,2,4) ,
                'Sb': (-3,3,5) ,
                'Te': (-2,2,4,6) ,
                'I' : (-1, 1, 3, 5,7) ,
                'Xe': () ,
                'Cs': (1,) ,
                'Ba': (2,) ,
                'La': (3,) ,
                'Ce': (2, 3, 4) ,
                'Pr': (2, 3, 4, 5) ,
                'Nd': (2, 3, 4) ,
                'Pm': (2, 3) ,
                'Sm': (2, 3) ,
                'Eu': (2, 3) ,
                'Gd': (1, 2, 3) ,
                'Tb': (1, 2, 3, 4) ,
                'Dy': (2, 3, 4) ,
                'Ho': (2, 3) ,
                'Er': (2, 3) ,
                'Tm': (2, 3) ,
                'Yb': (2, 3) ,
                'Lu': (2, 3) ,
                'Hf': (4,) ,
                'Ta': (5,) ,
                'W' : (4,6) ,
                'Re': (4,) ,
                'Os': (4,) ,
                'Ir': (3, 4) ,
                'Pt': (2, 4) ,
                'Au': (3,) ,
                'Hg': (1, 2) ,
                'Tl': (1, 3) ,
                'Pb': (2, 4) ,
                'Bi': (3,) ,
                'Po': (-2, 2, 4) ,
                'At': (-1, 1) ,
                'Rn': () ,
                'Fr': (1,) ,
                'Ra': (2,) ,
                'Ac': (3,) ,
                'Th': (1, 2, 3, 4) ,
                'Pa': (3, 4, 5) ,
                'U' : () ,
                'Np': () ,
                'Pu': (2, 3, 4, 5, 6, 7) ,
                'Am': (2, 3, 4, 5, 6, 7) ,
                'Cm': (3, 4, 6) ,
                'Bk': (3, 4) ,
                'Cf': (2, 3, 4) ,
                'Es': (2, 3, 4) ,
                'Fm': (2, 3) ,
                'Md': (2, 3) ,
                'No': (2, 3) ,
                'Lr': (3,) ,
                'Rf': (4,) ,
                'Db': (5,) ,
                'Sg': (6,) ,
                'Bh': (7,) ,
                'Hs': (8,) ,
                'Mt': () ,
                'Ds': () ,
                'Rg': () ,
                'Cn': () ,
                'Nh': () ,
                'Fl': () ,
                'Mc': () ,
                'Lv': () ,
                'Ts': () ,
                'Og': () }

# oxidations = {  'H':  (-1,0,1),
#                 'He': (0,) ,
#                 'Li': (-1,0,1,) ,
#                 'Be': (0,1,2) ,
#                 'B':  (-5,-1,0,1,2,3) ,
#                 'C':  (-4, -3, -2, -1, 0, 1, 2, 3, 4) ,
#                 'N':  (-3, -2, -1, -3, 0, 1, 2, 3, 4, 5) ,
#                 'O':  (-2,-1,0,1,2) ,
#                 'F':  (-1,0) ,
#                 'Ne': (0,) ,
#                 'Na': (-1,0,1) ,
#                 'Mg': (0,1,2) ,
#                 'Al': (-2,-1,0,1,2,3) ,
#                 'Si': (-4,-3,-2,-1,0,1,2,3,4) ,
#                 'P':  (-3,-2,-1,0,1,2,3,4,5) ,
#                 'S':  (-2,-1,0,1,2,3,4,5,6) ,
#                 'Cl': (-1,0,1,2,3,4,5,6,7) ,
#                 'Ar': (0,) ,
#                 'K':  (-1,0,1) ,
#                 'Ca': (0,1,2) ,
#                 'Sc': (0,1,2,3) ,
#                 'Ti': (-2,-1,0,1,2,3,4) ,
#                 'V':  (-3,-1,0,1,2,3,4,5) ,
#                 'Cr': (-4,-2,-1,0,1,2,3,4,5,6) ,
#                 'Mn': (-3,-2,-1,0,1,2,3,4,5,6,7) ,
#                 'Fe': (-4,-2,-1,0,1,2, 3, 4,5,6) ,
#                 'Co': (-3,-1,0,1,2, 3,4,5) ,
#                 'Ni': (-2,-1,0,1, 2,3,4) ,
#                 'Cu': (-2,0,1,2,3,4) ,
#                 'Zn': (-2,0,1,2) ,
#                 'Ga': (-5,-4,-2,-1,0,1,2,3) ,
#                 'Ge': (-4,-3,-2,-1,0,1,2,3,4) ,
#                 'As': (-3, -2,-1,0,1,2, 3, 4, 5) ,
#                 'Se': (-2, -1,0, 1, 2, 3, 4, 5, 6) ,
#                 'Br': (-1, 0, 1, 3, 4, 5, 7) ,
#                 'Kr': (0,2) ,
#                 'Rb': (-1,0,1) ,
#                 'Sr': (0,1,2) ,
#                 'Y' : (0,1,2,3) ,
#                 'Zr': (-2,0,1,2,3,4) ,
#                 'Nb': (-3,-1,0,1,2,3,4,5) ,
#                 'Mo': (-4,-2,-1,0,1,2,3,4,5,6) ,
#                 'Tc': (-3,-1,0,1,2,3,4,5,6,7) ,
#                 'Ru': (-4,-2,0,1,2,3,4,5,6,7,8) ,
#                 'Rh': (-3,-1,0,1,2,3,4,5,6) ,
#                 'Pd': (0,1,2,3,4,5,6) ,
#                 'Ag': (-2,-1,0,1,2,3,4) ,
#                 'Cd': (-2,0,1,2) ,
#                 'In': (-5,-2,-1,0,1,2,3) ,
#                 'Sn': (-4,-3,-2,-1,0,1,2,3,4) ,
#                 'Sb': (-3,-2,-1,0,1,2,3,4,5) ,
#                 'Te': (-2,-1,0,1,2,3,4,5,6) ,
#                 'I' : (-1, 0, 1, 3, 4, 5,6, 7) ,
#                 'Xe': (0,2,4,6,8) ,
#                 'Cs': (-1,0,1) ,
#                 'Ba': (0,1,2) ,
#                 'Hf': (-2,0,1,2,3,4) ,
#                 'Ta': (-3,-1,0,1,2,3,4,5) ,
#                 'W' : (-4,-2,-1,0,1,2,3,4,5,6) ,
#                 'Re': (-3,-1,0,1,2,3,4,5,6,7) ,
#                 'Os': (-4,-2,-1,0,1,2,3,4,5,6,7,8) ,
#                 'Ir': (-3,-1, 0,1,2,3,4,5,6,7,8,9) ,
#                 'Pt': (-3, -2 , -1, 0,1,2,3,4,5,6) ,
#                 'Au': (-3,-2,-1,0,1,2,3,5) ,
#                 'Hg': (-2,0,1, 2) ,
#                 'Tl': (-5, -2, -1, 0, 1, 2, 3) ,
#                 'Pb': (-4, -2 ,-1, 0, 1, 2, 3, 4) ,
#                 'Bi': (-3,-2,-1,0,1,2,3,4,5) ,
#                 'Po': (-2, 0, 2, 4, 5, 6) ,
#                 'At': (-1, 0, 1, 3, 5, 7) ,
#                 'Rn': (0,2,6) ,
#                 'Fr': (0,1) ,
#                 'Ra': (0,2) ,
#                 'La': (0,1,2,3) ,
#                 'Ce': (0,2, 3, 4) ,
#                 'Pr': (0,2, 3, 4) ,
#                 'Nd': (0,2, 3, 4) ,
#                 'Pm': (0, 2, 3) ,
#                 'Sm': (0, 2, 3) ,
#                 'Eu': (0, 2, 3) ,
#                 'Gd': (0, 1, 2, 3) ,
#                 'Tb': (0, 1, 2, 3, 4) ,
#                 'Dy': (0, 2, 3, 4) ,
#                 'Ho': (0, 2, 3) ,
#                 'Er': (0, 2, 3) ,
#                 'Tm': (0, 2, 3) ,
#                 'Yb': (0, 2, 3) ,
#                 'Lu': (0, 2, 3) ,
#                 'Ac': (0,2,3) ,
#                 'Th': (0, 1, 2, 3, 4) ,
#                 'Pa': (0, 2,3, 4, 5) ,
#                 'U' : (0,1,2,3,4,5,6) ,
#                 'Np': (0,2,3,4,5,6,7) ,
#                 'Pu': (0,1,2, 3, 4, 5, 6, 7,8) ,
#                 'Am': (0,2, 3, 4, 5, 6, 7,8) ,
#                 'Cm': (0,2,3, 4, 6) ,
#                 'Bk': (0,2,3, 4) ,
#                 'Cf': (0,2, 3, 4) ,
#                 'Es': (0,2, 3, 4) ,
#                 'Fm': (0, 2, 3) ,
#                 'Md': (0, 2, 3) ,
#                 'No': (0, 2, 3) ,
#                 'Lr': (0, 3)}

bad_list = ['','H','Hs','Am','Bh','Cm','Sg','Bk','Cf','Es','Rf','Fm','Md','No',
                             'Lr','Cn','Db','Mt','Ds','Rg','Nh','Fl','Mc','Lv','Ts','Og'] # elements from this list will not be chosen


def get_wyckoff_positions(spg):
    wykcoff_string = re.findall("\[[xyz,\'\s*0-9\/+-]*\]",symmetry.wyckoff_df.iloc[spg]['0'])
    n = len(wykcoff_string) - 1

    ret = {}
    for iwyckof in wykcoff_string:
        multiplicity = len(re.findall("\'([xyz,\s\/0-9+-]*)\'",iwyckof))
        letter = string.ascii_lowercase[n]
        n-=1
        ret[str(multiplicity)+letter] = re.findall("\'([xyz,\s\/0-9+-]*)\'",iwyckof)
    return ret



def generator(args):
    elements = args['elements']
    composition = args['composition']
    spg = args['spg']
    wyckoff_position = args['wyckoff_position']
    generation_info = args['generation_information']
    cs = crystal.random_crystal(int(spg),
                                species=elements,
                                numIons=composition,
                                factor=1)   
    if not cs.valid:
        return None
    
    if wyckoff_position in list(np.unique(['%d%s'% (s.wp.multiplicity,s.wp.letter) for s in cs.wyckoff_sites])):
        cell = cs.struct.lattice.matrix
        reduced = cs.struct.frac_coords
        symbols = [ele.value for ele in cs.struct.species]
        structure = pychemia.Structure(cell=cell, symbols=symbols, reduced=reduced)
        sym = pychemia.crystal.CrystalSymmetry(structure)
        properties = { 'pretty_formula':structure.get_composition().sorted_formula(sortby='electroneg'),
                       'elements':elements,
                       'generation_information':generation_info,
                       'requested_spg':spg,
                       'spacegroup':{'number': sym.number(symprec=1e-3),
                                     'symbol': sym.symbol(symprec=1e-3),
                                     'spglib_wyckoffs' :sym.get_symmetry_dataset()['wyckoffs'],
                                     'wyckoffs':list(np.unique(['%d%s'% (s.wp.multiplicity,s.wp.letter) for s in cs.wyckoff_sites])),
                                     }}
        
        print(str(properties['spacegroup']['number'])+'-'+str(structure.nspecies)+'-'+properties['pretty_formula']+'.vasp')
        if spg != sym.number(symprec=1e-3):
            
            print('++++++++++++++++++++++++++++++++++++++++++++')
            print('++++++++++++++++++++++++++++++++++++++++++++')
            print('The requested space group %d not equal to the generated space group %d'%(spg,sym.number(symprec=1e-3)))
            cs.print_all()
            print(cs.frac_coords.round(5))
            print(structure.reduced)
            print("--------------------------------------------")
            print('elements',elements)
            print('compisotion',composition)
            print('spg',spg)
            return None

        return (structure,properties)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-spg','--space_group',
                        dest="space_groups",
                        nargs='+',
                        default=None,
                        type=int,
                        help="All the space groups that you want to create structures with, e.g. --spg 169 170")
    parser.add_argument('-e','--elements',
                        dest='elements',
                        nargs='+',
                        default=['all'],
                        help='Which elements you want create structures with, e.g. --elements C O Al, default: --elements all')
    parser.add_argument('-me','--must_elements',
                        dest='must_elements',
                        nargs='+',
                        default=None,
                        help='Which element you want to be in the structure all the time, e.g. --must_elements C ,e.g. -me Al')

    parser.add_argument('-nspc','--nspecies',
                        dest='nspecies',
                        type=int,
                        default=3,
                        help='Number of species, binary, ternary, etc')
    parser.add_argument('-w','--wyckoffs',
                        dest='wyckoffs',
                        nargs='+',
                        default='all',
                        help='The Wyckoff positions that you want to be present in the structure, e.g. --wyckoffs 6a, default: --wyckoffs all')
    parser.add_argument('-fmt','--format',
                        dest='fmt',
                        nargs='+',
                        default=['poscar'],
                        help='The output format for the structures. options: POSCAR, cif, mongodb, e.g. --fomrat PosCAR, default: --format poscar')
    parser.add_argument('-np','--nprocess',
                        dest='nparal',
                        type=int,
                        default=1,
                        help='Number of processors used for this operation, e.g. --nprocess 4 ,default: --nprocess 1 ')
    parser.add_argument('-v','--verbos',
                        dest='verbose',
                        type=str,
                        default='flase',
                        help='If you want progress update, e.g. -verbose true, default: -v false ')
    parser.add_argument('-nb','--nbatch',
                        dest='nbatch',
                        type=int,
                        default=1000,
                        help='If parallel is selected, the structures will get created with batches, e.g. -nbatch 1000, default: -nb 100')
    parser.add_argument('-m','--max_stoichiometry',
                        dest='max_stoichiometry',
                        type=int,
                        default=10,
                        help="Max limit of stoichiometry for each atom in unit cell, default: -m 10")
    parser.add_argument('-db','--database_name',
                        dest='db_name',
                        type=str,
                        default='generated_structures',
                        help="The name of the mongodb you want to save on. This is only relevant if fmt is chosen to be mongodb")
    
    
    args = parser.parse_args()


    args.fmt = [ifmt.lower() for ifmt in args.fmt]

    if  'mongodb' in args.fmt:
        db = pychemia.db.get_database({'name':args.db_name})

    oxidations_rev = {}
    for i in range(-5,10):
        oxidations_rev[i] = []
    
    if args.elements[0] != 'all':
        new_oxidations = {}
        for ielement in args.elements:
            new_oxidations[ielement] = oxidations[ielement]
        oxidations = new_oxidations

    
        
    for ielement in oxidations:
        if ielement in bad_list or pychemia.utils.periodic.atomic_number(ielement) > 83 : # or pychemia.utils.periodic.block(ielement) == 'f' This is from having one metal at least
            continue
        for iox in oxidations[ielement]:
            oxidations_rev[iox].append(ielement)
    
    # if args.must_element is None:
    #     args.must_element = pychemia.utils.periodic.atomic_symbols
    # else :
    #     args.must_element = [args.must_element]
    if args.elements[0].lower() == 'all':
        args.elements = pychemia.utils.periodic.atomic_symbols

    pool_args = []
    for ispg in args.space_groups:
        if args.wyckoffs == 'all':
            wyckoff_list = get_wyckoff_positions(ispg)
        else :
            wyckoff_list = args.wyckoffs
        for iposition in wyckoff_list:

            multiplicity = int(iposition[0])
            letter = iposition[1]
            if multiplicity > max_stoichiometry:
                continue
            for ielement in args.elements:#pychemia.utils.periodic.atomic_symbols:

                if ielement in bad_list:
                    continue
                print(ispg,iposition,ielement)
                for z_oxidation in pychemia.utils.periodic.oxidation_state(ielement):

                    needed_oxidation = -1*z_oxidation*multiplicity
                    for iox in itertools.combinations_with_replacement(range(-5,10),args.nspecies-1):
                        for istoch in itertools.combinations_with_replacement(range(1,args.max_stoichiometry),args.nspecies-1):
                            oxidation = 0
                            generation_info = []

                            for ispc in range(args.nspecies-1):
                                oxidation += iox[ispc]*istoch[ispc]
                                generation_info.append([iox[ispc],istoch[ispc]])
                            if oxidation==needed_oxidation:
                                
                                candidates = []
                                composition = []

                                for ispc in range(args.nspecies-1):
                                    candidates.append(oxidations_rev[iox[ispc]])
                                    composition.append(istoch[ispc])
                                composition.append(multiplicity)
                                for iele in itertools.product(*tuple(candidates)):
                                    elements = [x for x in iele]
                                    
                                    if args.must_elements is not None:
                                        if not all([x in elements for x in args.must_elements]):
                                            continue
                                    if args.elements is not None:
                                        if not any([x in elements for x in args.elements]):
                                            continue
                                    elements.append(ielement)
                                    generation_info.append([z_oxidation,multiplicity])
                                    
                                    pool_args.append({'elements':elements,
                                                      'composition':composition,
                                                      'generation_information':generation_info,
                                                      'spg':ispg,
                                                      'wyckoff_position':iposition})
                                    if len(pool_args) > args.nbatch:
                                        pool = Pool(args.nparal)
                                        results = pool.map(generator,pool_args)
                                        pool.close()
                                        pool_args=[]
                                        for ientry in results:
                                            if ientry is None :
                                                continue
                                            else:
                                                structure,properties = ientry
                                                if 'mongodb' in args.fmt:
                                                    entry_id = '%d_%s_%d_%s' % (istrc,properties['spacegroup']['number'],structure.nspecies, properties['pretty_formula'])
                                                    db.insert(structure=structure,properties=properties,entry_id=entry_id)
                                                elif 'poscar' in args.fmt:
                                                    pychemia.code.vasp.write_poscar(structure=structure,
                                                                                    filepath=str(properties['spacegroup']['number'])+'-'+str(structure.nspecies)+'-'+properties['pretty_formula']+'.vasp')
                                                    rf = open(str(properties['spacegroup']['number'])+'-'+str(structure.nspecies)+'-'+properties['pretty_formula']+'.vasp','r')
                                                    lines = rf.readlines()
                                                    rf.close()
                                                    comment = ''
                                                    elements = properties['elements']
                                                    generation_information = properties['generation_information']
                                                    for ispc in range(len(elements)):
                                                        comment += "(oxidation=%d, stoch=%d, element=%s, requested spg=%d) "%(generation_info[ispc][0],
                                                                                                                              generation_info[ispc][1],
                                                                                                                              elements[ispc],
                                                                                                                              properties['requested_spg'])
                                                    comment += '\n'
                                                    lines[0] = comment
                                                    wf = open(str(properties['spacegroup']['number'])+'-'+str(structure.nspecies)+'-'+properties['pretty_formula']+'.vasp','w')
                                                    wf.writelines(lines)
                                                    wf.close()
                                                elif 'cif' in args.fmt:
                                                    print('cif option is not implemented yet')
