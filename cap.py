import os
import math 
import json
import csv
import re  # 引入正则模块处理不规则空格
import parmed as pmd
from parmed.charmm import CharmmParameterSet
from pathlib import Path 
from typing import Dict, List, Tuple, Optional
from parmed.topologyobjects import AtomType

def get_boundary_bonds_from_struct(Q1_idx: int, M1_idx: int, struct: pmd.Structure, ff_type: str, params: CharmmParameterSet = None) -> Tuple[str,str,str, List[str]]: 
    """
    获取真实的原子类型用于查找参数。
    修改：不再返回单一的 QH_bond 字符串，而是返回 Q1 的类型，以便在 calc_C_HL 中动态匹配氢原子。
    """
    Q1_atom = struct.atoms[Q1_idx]
    M1_atom = struct.atoms[M1_idx]

    # 去除可能存在的空格
    Q1_at = (Q1_atom.type if Q1_atom.type else Q1_atom.name).strip()
    M1_at = (M1_atom.type if M1_atom.type else M1_atom.name).strip()

    # 构建 Q1-M1 键对字符串 (用于后续查找)
    QM_bond = f'{Q1_at}-{M1_at}'
    QM_bond_perm = f'{M1_at}-{Q1_at}'

    # 我们不再在这里硬编码 "XC-H1"，而是只传递 Q1 的类型
    # 因为具体的氢原子类型(HC, H1, HA)取决于 parm10.dat 里定义了什么
    
    return QM_bond, QM_bond_perm, Q1_at, None

def parse_parm_line(line: str) -> Optional[Tuple[str, str, float]]:
    """
    解析 parm.dat 的一行。
    返回: (Type1, Type2, Req) 或 None
    """
    # 忽略注释和空行
    if line.startswith('[') or not line.strip():
        return None
        
    # 移除行内注释 (Amber文件通常用 ! 或 # 注释，但也可能没有)
    content = line.split('!')[0].split('#')[0].strip()
    
    # 预处理：有些行写成 "C -C" 或 "CT-CT"，统一把中间的 " -" 或 "- " 变成 "-"
    # 这样 "C -C" 变成 "C-C"
    content = content.replace(' -', '-').replace('- ', '-')
    
    # 按空白字符分割
    parts = content.split()
    
    # 检查是否是有效的键参数行
    # 通常格式: AT1-AT2  Force  Req  [Description]
    # 或者是: AT1 AT2 Force Req (如果不带连字符)
    
    if len(parts) < 3:
        return None
        
    bond_pair = parts[0]
    
    # 验证第一部分是否包含连字符 (这是 parm10.dat 的特征)
    if '-' not in bond_pair:
        return None
        
    t1, t2 = bond_pair.split('-')
    
    try:
        # 在 parm10.dat 中，Req (平衡键长) 是第3个数据块 (Index 2)
        # Index 0: Type-Type
        # Index 1: Force Constant
        # Index 2: Req
        req = float(parts[2])
        return (t1, t2, req)
    except (ValueError, IndexError):
        return None

def calc_C_HL(QM_bond:str, QM_bond_perm:str, Q1_at:str, _unused, ff_type:str, 
              Q1_atom=None, M1_atom=None, params:CharmmParameterSet = None, PARM_PATH:str= None) -> float:
    """
    计算缩放比例 C_HL。
    逻辑更新：
    1. 健壮地解析文件。
    2. 查找 Q1-M1 的 R0。
    3. 查找 Q1-H (尝试 HC, H1, HA, H4, H5) 的 R0。
    """
    R0_Q1M1 = None
    R0_Q1HL = None
    
    # Amber 力场中常见的氢原子类型优先级列表
    # HC: 脂肪族 H (最常见)
    # H1: 脂肪族 H (带拉电子基团)
    # HA: 芳香族 H
    # H4, H5: 特定芳香族 H
    candidate_h_types = ['HC', 'H1', 'HA', 'H4', 'H5', 'H', 'HO', 'HS']

    if ff_type == 'amber':
        try:
            with open(PARM_PATH, 'r') as parmfile:
                for line in parmfile:
                    result = parse_parm_line(line)
                    if not result:
                        continue
                        
                    t1, t2, req = result
                    
                    # 1. 检查是否匹配 QM 键 (Q1-M1)
                    current_bond = f"{t1}-{t2}"
                    current_bond_rev = f"{t2}-{t1}"
                    
                    if current_bond == QM_bond or current_bond == QM_bond_perm or \
                       current_bond_rev == QM_bond or current_bond_rev == QM_bond_perm:
                        R0_Q1M1 = req
                    
                    # 2. 检查是否匹配 Q1-H 键
                    # 我们只关心其中一个是 Q1，另一个在候选氢列表中
                    if R0_Q1HL is None: # 如果还没找到，继续找
                        if t1 == Q1_at and t2 in candidate_h_types:
                            R0_Q1HL = req # 找到了一个匹配的
                        elif t2 == Q1_at and t1 in candidate_h_types:
                            R0_Q1HL = req

                    # 如果两个都找到了，可以提前退出吗？
                    # 最好不要，因为可能文件后面有更精确的定义，但 parm.dat 通常不重复。
                    # 为了效率，如果都找到了就退出。
                    if R0_Q1M1 is not None and R0_Q1HL is not None:
                        break
                        
        except Exception as e:
            print(f"[Warning] 读取 parm 文件出错: {e}")
            pass
            
    # 如果找不到参数，使用默认值，并打印警告
    if R0_Q1M1 is None or R0_Q1HL is None or R0_Q1M1 == 0:
        if Q1_atom is not None and M1_atom is not None:
            print("-" * 60)
            print(f"[Warning] 无法找到键参数，将使用默认缩放比例 0.72")
            print(f"  涉及原子 1 (QM边界): {Q1_at} (Res: {Q1_atom.residue.name})")
            print(f"  涉及原子 2 (MM边界): {(M1_atom.type if M1_atom.type else M1_atom.name)} (Res: {M1_atom.residue.name})")
            print(f"  查找状态: R0({Q1_at}-MM)={'Found ('+str(R0_Q1M1)+')' if R0_Q1M1 else 'Missing'} | R0({Q1_at}-H)={'Found ('+str(R0_Q1HL)+')' if R0_Q1HL else 'Missing'}")
            print("-" * 60)
        return 0.72 # 默认比例
    
    ratio = R0_Q1HL / R0_Q1M1
    print(f"[Info] 计算缩放比例成功: R_QM={R0_Q1M1}, R_QH={R0_Q1HL}, Ratio={ratio:.4f}")
    return ratio

def append_hl_to_csv(new_atoms_data: List[dict]):
    """
    将封端氢原子的信息追加到 dataframe.csv
    """
    csv_file = 'dataframe.csv'
    # 如果文件不存在，可以考虑创建一个带 header 的，但这里保持原逻辑
    if not os.path.exists(csv_file):
        return

    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for atom in new_atoms_data:
            writer.writerow([
                atom['id'],          # CX_PDB_ID
                atom['name'],        # PDB_AT
                atom['res_full'],    # PDB_RES
                f"{atom['x']:.3f}",  # X
                f"{atom['y']:.3f}",  # Y
                f"{atom['z']:.3f}",  # Z
                "0.0000",            # q
                0,                   # FORMAL_Q
                "H"                  # ELEMENT
            ])

def cap(no_HL:Dict[str,List[int]], num_broken_bonds:int, CAPPED_PDB_PATH:str, ff_type:str, params:CharmmParameterSet = None, PARM_PATH:str=None) -> int:
    """
    执行封端、保存 PDB 并更新 CSV
    """
    prmtop_file = "production.top"
    nc_file = "temp_full_centered.nc"

    # 1. 载入拓扑和坐标
    # 增加健壮性检查
    if not os.path.exists(prmtop_file):
        print(f"[Error] 找不到拓扑文件 {prmtop_file}")
        return 1
        
    struct = pmd.load_file(prmtop_file)
    
    if os.path.exists(str(nc_file)):
        traj = pmd.load_file(str(nc_file))
        try:
            frame_coords = traj.coordinates[0]
        except (IndexError, TypeError):
            frame_coords = traj.coordinates
        struct.coordinates = frame_coords
    else:
        print(f"[Warning] 找不到坐标文件 {nc_file}，将使用拓扑中的坐标")
    
    
    new_atoms = [] 
    hl_csv_data = []
    with_HL = no_HL.copy()
    
    # 用于构建 Mask 的真实索引列表 (1-based index)
    new_atom_indices = []

    # 获取当前 PDB 最大原子序号
    last_serial = 0
    for a in struct.atoms:
        if a.number > last_serial: last_serial = a.number
    if last_serial == 0: last_serial = len(struct.atoms)

    for bond in range(1, num_broken_bonds+1):
        # 获取索引 (0-based)
        if f'Q1_{bond}' not in no_HL: continue
        
        Q1_idx = int(no_HL[f'Q1_{bond}'][0]) - 1
        M1_idx = int(no_HL[f'M1_{bond}'][0]) - 1
        
        Q1_atom = struct.atoms[Q1_idx]
        M1_atom = struct.atoms[M1_idx]
        
        Q1_coords = [Q1_atom.xx, Q1_atom.xy, Q1_atom.xz]
        M1_coords = [M1_atom.xx, M1_atom.xy, M1_atom.xz]

        # 获取 Q1 类型，不再获取错误的 XC-H1
        qm_bond_str, qm_perm, q1_type, _ = get_boundary_bonds_from_struct(Q1_idx, M1_idx, struct, ff_type, params)
        
        # 计算比例
        if ff_type == 'amber':
            C_HL = calc_C_HL(qm_bond_str, qm_perm, q1_type, None, ff_type, 
                           Q1_atom=Q1_atom, M1_atom=M1_atom, PARM_PATH=PARM_PATH)
        else:
            C_HL = 0.72
        
        # 计算 HL 坐标
        R_Q1M1 = math.dist(Q1_coords, M1_coords)
        R_Q1HL = C_HL * R_Q1M1
        
        # 防止除零错误
        if R_Q1M1 < 0.001:
            R_ratio = 0.72
        else:
            R_ratio = (R_Q1HL / R_Q1M1)
        
        HL_coords = []
        for n, x in enumerate(Q1_coords):
            HL_coords.append((1-R_ratio)*x + R_ratio*M1_coords[n])
            
        # 创建原子对象
        last_serial += 1
        new_atom = pmd.Atom(name='H', type='H', atomic_number=1, mass=1.008)
        if not hasattr(new_atom.atom_type, 'rmin'):
            dummy_type = AtomType(name='H', number=1, mass=1.008, atomic_number=1)
            dummy_type.rmin = 0.6000
            dummy_type.epsilon = 0.0157
            new_atom.atom_type = dummy_type
        new_atom.xx, new_atom.xy, new_atom.xz = HL_coords
        new_atom.number = last_serial
        
        # 目标残基 (Q1 所在的残基)
        target_residue = Q1_atom.residue
        
        # 确保添加到正确的残基对象
        struct.add_atom(new_atom, resname=target_residue.name, resnum=target_residue.number, chain=target_residue.chain)
        
        added_residue = struct.residues[-1]
        
        if added_residue is not target_residue:
            new_atom.residue = target_residue
            target_residue.add_atom(new_atom)
            struct.residues.remove(added_residue)
        
        # 记录 Index (ParmEd mask 使用 1-based index)
        current_atom_idx = len(struct.atoms)
        new_atom_indices.append(current_atom_idx)
        
        with_HL[f'HL_{bond}'] = [last_serial]

        hl_csv_data.append({
            'id': last_serial,
            'name': f'HL{bond}',
            'res_full': f"{target_residue.name}{target_residue.number}",
            'x': HL_coords[0], 'y': HL_coords[1], 'z': HL_coords[2]
        })

    # 构建 Mask
    qm_mask_list = list(map(str, with_HL['QM']))
    
    new_mask_list = list(map(str, new_atom_indices))
    
    full_mask_str = "@" + ",".join(qm_mask_list)
    if new_mask_list:
         full_mask_str += "," + ",".join(new_mask_list)

    print(f"[Cap] 生成的 Mask 长度: QM={len(qm_mask_list)}, New={len(new_mask_list)}")
    
    # 保存文件
    struct[full_mask_str].save(CAPPED_PDB_PATH, overwrite=True)
    
    # 写入 CSV
    append_hl_to_csv(hl_csv_data)

    # 保存更新后的字典
    with open('with_HL.dat', 'w+') as wfile:
        json.dump(with_HL, wfile)
    
    print(f"[Cap] 保存封端后的PDB文件 {CAPPED_PDB_PATH}; 共有 {len(new_atom_indices)} link atoms 载入 CSV")
    
    return 0

def run_cap(ff_type:str,pdb_id:str = None) -> int:    
    dict_path = 'dictionary.dat' if os.path.exists('dictionary.dat') else 'pre-dictionary.dat'
    print(f"[Cap] 使用字典: {dict_path}")
        
    with open(dict_path, 'r') as dictfile:
        no_HL = json.load(dictfile)
    
    # 解析切断键数量
    num_broken_bonds = 0
    for key in no_HL.keys():
        if key.startswith('Q1_'):
            try:
                n = int(key.split('_')[-1])
                if n > num_broken_bonds: num_broken_bonds = n
            except: pass
    
    CAPPED_PDB_PATH = f'{pdb_id}_QM_capped.pdb'
        
    if ff_type =='amber':
        # 这里你可以根据需要修改 PARM_PATH 的路径
        path_to_env = os.environ.get('AMBERHOME', '')
        # 如果当前目录下有 parm10.dat，优先使用
        if os.path.exists('parm10.dat'):
            PARM_PATH = 'parm10.dat'
        else:
            PARM_PATH = f'{path_to_env}/dat/leap/parm/parm10.dat' 
            
        print(f"[Cap] 读取 Amber 参数文件: {PARM_PATH}")
        return cap(no_HL, num_broken_bonds, CAPPED_PDB_PATH, ff_type='amber', PARM_PATH=PARM_PATH)
    
    return 0