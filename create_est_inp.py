import pandas as pd
import json
import sys
from typing import List, Dict, Tuple
import numpy as np

def make_monomers(charge_method: str = 'Z2',ligand_residue_name = "MOL") -> Tuple[List[str], str, List[str], List[str]]:
    """
    Creates XYZs for QM and MM regions; gets charge of QM protein region.
    
    Ligand Logic:
    1. Look for 'MOL' in 'PDB_RES' column.
    2. If not found, exclude Protein (from with_HL), Water, and Ions.
    这个with_HL.dat文件中存储的QM原子列表是基于蛋白质的, 并不包含配体。配体整个都被直接纳入QM区域
    """
    
    # 1. 加载数据
    try:
        df = pd.read_csv('dataframe.csv', index_col='CX_PDB_ID')
        with open('with_HL.dat', 'r') as dictfile:
            with_HL = json.load(dictfile)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Key error in dataframe (missing column?): {e}")
        sys.exit(1)

    # 2. 确定切断的键数量 (用于后续 QM Protein 组装)
    num_bonds_broken = 0
    for key in with_HL.keys():
        if key.startswith('Q1_'):
            try:
                num = int(key.split('_')[-1])
                if num > num_bonds_broken:
                    num_bonds_broken = num
            except:
                pass

    # 3. 计算 QM 区域电荷 (c_QM)
    total_charge = 0.0
    # 遍历 qm_pro 区域的原子 ID
    for x in with_HL['QM']:
        x = int(x) # 确保 ID 格式统一
        if x in df.index and ligand_residue_name not in df.loc[x, 'PDB_RES']: # 要把配体的形式电荷统计排除在外
            # 直接读取 'q' 列的数值进行累加\
            charge = float(df.loc[x, 'FORMAL_Q'])
            total_charge += charge
     # 遍历 qm_lig 区域的原子 ID (使用模糊查询)
    ligand_charge = df[df['PDB_RES'].str.contains(ligand_residue_name)]['q'].sum()
    total_charge += ligand_charge
    print("计算得到的配体电荷为",ligand_charge)
    # 将累加的浮点数电荷四舍五入为最接近的整数，作为 QM 区域的总电荷
    total_charge = str(int(round(total_charge)))

    # 4. 获取 QM 区域的所有的原子坐标
    qm_indices = []
    qm_indices.extend(with_HL['QM'])
    for bond in range(1, num_bonds_broken + 1):
        hl_key = f'HL_{bond}'
        if hl_key in with_HL: #把封端的氢原子也纳入QM区域进行计算
            qm_indices.extend(with_HL[hl_key])
    qm_indices.sort()
    qm_indices = list(set(qm_indices))  # 去重
    qm_atoms = get_xyz_from_df(df, qm_indices)
    
    # 6. 获取 MM 环境点电荷 (mm_env)
    if charge_method == 'Z2':
        mm_env = Z2(df, with_HL, num_bonds_broken)
    elif charge_method == 'BRCD':
        mm_env = bal_RC_array(charge_method, df, with_HL, num_bonds_broken)
    # 默认回退到 Z2
    else:
        mm_env = Z2(df, with_HL, num_bonds_broken)
    return  qm_atoms,total_charge, mm_env, len(qm_atoms)

def get_xyz_from_df(df: pd.DataFrame, atoms: List[int]) -> List[str]:
    """
    从 DataFrame 中提取指定原子的 XYZ 坐标，并格式化为 "Element X Y Z"
    """
    xyz_lines = []
    for idx in atoms:
        if idx in df.index:
            # 获取元素
            if 'ELEMENT' in df.columns:
                element = str(df.at[idx, 'ELEMENT']).strip()
            else:
                # 尝试从 AT_LABEL 解析 (例如 N3 -> N)
                raw_label = str(df.at[idx, 'AT_LABEL'])
                element = ''.join([i for i in raw_label if not i.isdigit() and i not in ['.', '+', '-']])
            
            x = df.at[idx, 'X']
            y = df.at[idx, 'Y']
            z = df.at[idx, 'Z']
            xyz_lines.append(f"{element:<3} {x:>12.6f} {y:>12.6f} {z:>12.6f}\n")
        else:
            print(f"出现重大错误{idx}不在csv文件中")
            sys.exit(1)
    return xyz_lines
def SEE_atoms(num_bonds_broken:int, with_HL:Dict[str,List[int]]) -> List[str]:
    """
    Returns all M1, M2, and M3 atoms

    Parameters
    ----------
    num_bonds_broken: int
        number of bonds broken that need to be capped
    with_HL: Dict[str, List[int]]
        Dictionary with key of the region and value with a list of atoms in that region

    Returns
    -------
    MM_for_array: List[str]
        list of the charges along with xyz coords
    """
    MM_atoms = []
    for bond in range(1,num_bonds_broken+1):
        if f'M1_{bond}' in with_HL and f'M2_{bond}' in with_HL and f'M3_{bond}' in with_HL:
            MM_atoms += (with_HL[f'M1_{bond}'] + with_HL[f'M2_{bond}'] + with_HL[f'M3_{bond}'])
        elif f'M1_{bond}' in with_HL and f'M2_{bond}' in with_HL and f'M3_{bond}' not in with_HL:
            MM_atoms += (with_HL[f'M1_{bond}'] + with_HL[f'M2_{bond}'])
        else:
            MM_atoms += (with_HL[f'M1_{bond}'])
    return MM_atoms
def Z1_atoms_charge(num_bonds_broken:int, with_HL: Dict[str,List[int]]) -> List[str]:
    """
    returns the M2 and M3 atoms 

    Parameters
    ----------
    num_bonds_broken: int
        number of bonds broken that need to be capped
    with_HL: Dict[str, List[int]]
        Dictionary with key of the region and value with a list of atoms in that region

    Returns
    -------
    MM_atoms: List[int]
        list of the MM atoms to keep
    """
    MM_atoms = []
    for bond in range(1,num_bonds_broken+1):
        if f'M2_{bond}' in with_HL.keys():
            MM_atoms += with_HL[f'M2_{bond}'] 
        if f'M3_{bond}' in with_HL.keys():
            MM_atoms += with_HL[f'M3_{bond}']
    return MM_atoms
def Z2_atoms_charge(num_bonds_broken:int, with_HL:Dict[str,List[int]]) -> List[str]:
    """
    returns the M3 atoms 

    Parameters
    ----------
    num_bonds_broken: int
        number of bonds broken that need to be capped
    with_HL: Dict[str, List[int]]
        Dictionary with key of the region and value with a list of atoms in that region

    Returns
    -------
    MM_atoms: List[int]
        list of the MM atoms to keep
    """
    MM_atoms = []
    for bond in range(1,num_bonds_broken+1):
        if f'M3_{bond}' in with_HL.keys():
            MM_atoms += with_HL[f'M3_{bond}']
    return MM_atoms

def bal_redist_charges(num_bonds_broken:int, MM_for_array:List[str], MM_atoms:List[str], charge_method:str, df:pd.DataFrame, with_HL:Dict[str,List[int]]) -> List[str]:
    """
    Takes in charge array without redistributed charges, 
    Redistibutes charges, 
    Creates external charge array for BRC, BRCD, and BRC2

    Parameters
    ----------
    num_bonds_broken: int
        number of bonds broken that need to be capped
    MM_for_array: List[str]
        List of charges of each MM atom along with their XYZ coordinates
    MM_atoms: MM_atoms: List[str]
       List of MM atoms 
    charge_method: str
        redistribution scheme
    df: pd.DataFrame
        pandas dataframe containing all atoms in the system
    with_HL: Dict[str, List[int]]
        Dictionary with key of the region and value with a list of atoms in that region

    Returns
    -------
    MM_for_array: List[str]
        list of the charges along with xyz coords
    """
    for bond in range(1,num_bonds_broken+1):
        # get coordinates of M1 and M2 atoms
        M1_atom = with_HL[f'M1_{bond}']
        M1atom_xyz = [df.at[M1_atom[0], 'X'], df.at[M1_atom[0], 'Y'], df.at[M1_atom[0],'Z']]
        M1atom_charge = df.at[M1_atom[0], 'q']
        M1atom_residue = df.at[M1_atom[0], 'PDB_RES']
        if df.at[M1_atom[0], 'PDB_AT'] == 'CA':     
            # get original integer charge of residue
            MM_group_charges = np.where(df['PDB_RES'] == str(M1atom_residue), df[['q']].sum(axis=1),0)
            MM_group_charge = np.sum(MM_group_charges)
            # sum the charges on atoms in that residue and also in MM
            MMdf = df.loc[df.index.isin(MM_atoms)]
            MM_resi_charges = np.where(MMdf['PDB_RES'] == str(M1atom_residue), MMdf[['q']].sum(axis=1),0)
            MM_resi_charge = np.sum(MM_resi_charges)
            q_purple = MM_group_charge - MM_resi_charge
            M1_bal_charge = M1atom_charge + q_purple
        elif df.at[M1_atom[0], 'PDB_AT'] == 'C':
            # get original integer charge of residue
            QM_group_charges = np.where(df['PDB_RES'] == str(M1atom_residue), df[['q']].sum(axis=1),0)
            QM_group_charge = np.sum(QM_group_charges)
            # sum the charges on atoms in that residue and also in MM
            MMdf = df.loc[df.index.isin(MM_atoms)]
            MM_resi_charges = np.where(MMdf['PDB_RES'] == str(M1atom_residue), MMdf[['q']].sum(axis=1),0)
            q_purple = np.sum(MM_resi_charges)
            M1_bal_charge = M1atom_charge - q_purple
        M2_atoms = with_HL[f'M2_{bond}']
        num_M2 = len(M2_atoms)
        redist_charge = float(M1_bal_charge) / float(num_M2)
        for atom in M2_atoms:
            midpoint_xyz = []
            M2atom_charge = df.at[atom, 'q']
            M2atom_xyz = [df.at[atom, 'X'], df.at[atom, 'Y'], df.at[atom,'Z']]
            # appending M2 atoms to array with new charge for RCD only
            if charge_method == 'BRCD':
                rcd_m2_charge = float(M2atom_charge) - float(redist_charge)
                MM_for_array.append(f'{float(rcd_m2_charge):.6f}')
                MM_for_array.append(f'{float(M2atom_xyz[0]):.3f}')
                MM_for_array.append(f'{float(M2atom_xyz[1]):.3f}')
                MM_for_array.append(f'{float(M2atom_xyz[2]):.3f}\n')
            if charge_method == 'BRC2':
                rcd_m2_charge = float(M2atom_charge) + float(redist_charge)
                MM_for_array.append(f'{float(rcd_m2_charge):.6f}')
                MM_for_array.append(f'{float(M2atom_xyz[0]):.3f}')
                MM_for_array.append(f'{float(M2atom_xyz[1]):.3f}')
                MM_for_array.append(f'{float(M2atom_xyz[2]):.3f}\n')
            if charge_method == 'BRC':
                MM_for_array.append(f'{float(redist_charge):.6f}')
            elif charge_method == 'BRCD':
                MM_for_array.append(f'{float(2*redist_charge):.6f}') # double redistributed charge for RCD scheme
            # calculate midpoints of M1 and M2
            if charge_method == 'BRC' or charge_method == 'BRCD':
                for n,x in enumerate(M2atom_xyz):
                    coord = (float(M1atom_xyz[n]) + float(x))/2
                    midpoint_xyz.append(f'{coord:.3f}')
                    if n == 0 or n == 1:
                        MM_for_array.append(f'{coord:.3f}')
                    else:
                        MM_for_array.append(f'{coord:.3f}\n')
    return MM_for_array

def bal_RC_array(charge_method:str, df:pd.DataFrame, with_HL:Dict[str,List[int]], num_bonds_broken:int) -> List[str]:
    """
    Create external charge array for BRC, BRCD, BRC2

    Parameters
    ----------
    charge_method: str
        redistribution scheme
    df: pd.DataFrame
        pandas dataframe containing all atoms in the system
    with_HL: Dict[str, List[int]]
        Dictionary with key of the region and value with a list of atoms in that region
    num_bonds_broken: int
        number of bonds broken that need to be capped

    Returns
    -------
    MM_for_array: List[str]
        list of the charges along with xyz coords
    """
    #print(df)
    if charge_method == 'BRC':
        BRC_ext = Z1_atoms_charge(num_bonds_broken, with_HL) + with_HL['MM']
    elif charge_method == 'BRCD' or charge_method == 'BRC2':
        BRC_ext = Z2_atoms_charge(num_bonds_broken, with_HL) + with_HL['MM']
    ext_df = df.loc[df.index.isin(BRC_ext)]
    # add fourth water point
    ext_with_wat = BRC_ext
    for idx in ext_df.index:
        if ext_df.loc[idx, 'MOL2_RES'] == 'WAT' and ext_df.loc[idx, 'MOL2_AT'] == 'OW':
            df_idx = ext_df.loc[idx, 'MOL2_ID'] + 3.5
            ext_with_wat.append(df_idx)
    ext_wat_df = df.loc[df.index.isin(ext_with_wat)]
    ext_wat_df.to_csv('MMdf.csv')
    ext_wat_df = ext_wat_df[['q', 'X', 'Y', 'Z']]
    MM_for_array = []
    for x in ext_wat_df.values.tolist():
        for n,y in enumerate(x):
            if n < 3:
                MM_for_array.append(str(y))
            else:
                MM_for_array.append(str(y)+'\n')
    MM_atoms =  SEE_atoms(num_bonds_broken, with_HL) + with_HL['MM']
    mm_env = bal_redist_charges(num_bonds_broken, MM_for_array, MM_atoms, charge_method, df, with_HL)
    return mm_env

def Z2(df: pd.DataFrame, with_HL: Dict[str, List[int]], num_bonds_broken: int) -> List[str]:
    """
    Z2 方法实现：M1/M2 层电荷归零，其余 MM 电荷保留。
    """
    MM_for_array = []
    
    # 1. 识别需要归零的原子 (M1 和 M2 层)
    z2_zero_indices = []
    for i in range(1, num_bonds_broken + 1):
        m1_key = f'M1_{i}'
        m2_key = f'M2_{i}'
        if m1_key in with_HL:
            z2_zero_indices.extend(with_HL[m1_key])
        if m2_key in with_HL:
            z2_zero_indices.extend(with_HL[m2_key])
            
    # 2. 识别保留电荷的原子
    mm_all_indices = with_HL.get('MM', [])
    z2_charge_indices = [x for x in mm_all_indices if x not in z2_zero_indices]

    ext_list = z2_zero_indices + z2_charge_indices
    
    # 3. 处理显式水分子 (WAT) - 将不在 MM 列表中的水也加入背景电荷
    # 如果系统包含离子，且不在 QM 区域，它们通常也应该作为 MM 电荷的一部分
    # 这里简单地将所有非配体、非 QM 的剩余原子视为环境电荷（包括水和离子）
    # 但为了稳健，我们先只加 WAT，避免重复添加
    
    existing_mm_set = set(ext_list)
    wat_indices = []
    
    # 查找水分子行
    if 'PDB_RES' in df.columns:
         wat_df = df[df['PDB_RES'].isin(['WAT', 'HOH', 'TIP3', 'SOL'])]
         for idx in wat_df.index:
             if idx not in existing_mm_set:
                 wat_indices.append(idx)

    final_indices = ext_list + wat_indices
    
    # 4. 构建输出
    target_df = df.loc[df.index.isin(final_indices)].copy()
    
    # 将 M1/M2 区域电荷设为 0
    target_df.loc[target_df.index.isin(z2_zero_indices), 'q'] = 0.0

    for idx in final_indices:
        if idx in target_df.index:
            row = target_df.loc[idx]
            MM_for_array.append(f"{row['q']:.6f} {row['X']:.6f} {row['Y']:.6f} {row['Z']:.6f}\n")
            
    return MM_for_array
