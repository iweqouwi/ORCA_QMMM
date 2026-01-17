import parmed as pmd
import json
import os
from typing import List, Set

def run_cut_protein_parmed(cutoff: float, region_cutoff_dist: float, ligand_residue_name: str) -> int:
    """
    利用 production.top 获取键合信息, temp_full_centered.nc是轨迹文件, 取第一帧
    并将蛋白质骨架切割位点调整为 Alpha碳(CA) - 羰基碳(C) 之间。
    
    修改点：MM区域现在被定义为 [距离配体 region_cutoff_dist 内的完整残基] 减去 [QM原子]。
    """
    prmtop_file = "production.top"
    nc_file = "temp_full_centered.nc"
    if not os.path.exists(prmtop_file):
        raise FileNotFoundError(f"未找到拓扑文件 {prmtop_file}，无法获取精确键合信息。")

    print(f"[Cut] 正在加载拓扑文件 {prmtop_file} 和坐标文件 {nc_file} ...")
    
    # 1. 载入拓扑和坐标
    struct = pmd.load_file(prmtop_file)
    traj = pmd.load_file(str(nc_file))
    
    try:
        frame_coords = traj.coordinates[0]
    except (IndexError, TypeError):
        frame_coords = traj.coordinates
    struct.coordinates = frame_coords

    # 定义中心（配体）
    center_mask = f":{ligand_residue_name}"
    view = struct.view[center_mask]
    center_atoms = [a for a in view.atoms]
    if not center_atoms:
        raise ValueError(f"在结构中未找到配体残基: {ligand_residue_name}")

    # 2. 初步选择 QM 区域 (基于完整残基)
    print(f"[Cut] 正在根据 cutoff={cutoff}A 选择初始 QM 残基...")
    selection_mask = f"{center_mask}<:{cutoff}"
    initial_selection = struct.view[selection_mask]
    
    initial_qm_residues = set()
    for atom in initial_selection.atoms:
        initial_qm_residues.add(atom.residue)
        
    for atom in center_atoms:
        initial_qm_residues.add(atom.residue)

    # 3. 构建初始 QM 原子集合
    qm_atom_set = set()
    for res in initial_qm_residues:
        for atom in res.atoms:
            qm_atom_set.add(atom)
            
    # =========================================================================
    # 4. 调整边界：将切割位点从肽键(C-N)移动到 (CA-C)
    # =========================================================================
    print("[Cut] 正在调整蛋白质骨架切割位置 (Target: CA-C bond)...")
    
    atoms_to_remove = set()
    atoms_to_add = set()

    for atom in qm_atom_set:
        # 情况 A: 检查 C 端边界 (QM -> MM), 移除 C 和 O
        if atom.name == 'C': 
            is_boundary = False
            for partner in atom.bond_partners:
                if partner.name == 'N' and partner not in qm_atom_set:
                    is_boundary = True
                    break
            
            if is_boundary:
                atoms_to_remove.add(atom)
                for partner in atom.bond_partners:
                    if partner.element_name == 'O' or partner.name.startswith('O'):
                        atoms_to_remove.add(partner)

        # 情况 B: 检查 N 端边界 (MM -> QM), 拉入前一个残基的 C 和 O
        if atom.name == 'N':
            for partner in atom.bond_partners:
                if partner.name == 'C' and partner not in qm_atom_set:
                    atoms_to_add.add(partner)
                    for c_partner in partner.bond_partners:
                        if c_partner.name == 'O':
                            atoms_to_add.add(c_partner)

    if atoms_to_add:
        print(f"[Cut] N端扩展: 将 {len(atoms_to_add)} 个骨架原子(C/O) 纳入 QM 区域。")
        qm_atom_set.update(atoms_to_add)
    
    if atoms_to_remove:
        print(f"[Cut] C端收缩: 将 {len(atoms_to_remove)} 个骨架原子(C/O) 移出 QM 区域。")
        qm_atom_set.difference_update(atoms_to_remove)

    # =========================================================================

    # 5. 识别最终的边界键 (QM-MM Boundary)
    boundary_bonds = []
    for atom in qm_atom_set:
        for bonded_atom in atom.bond_partners:
            if bonded_atom not in qm_atom_set:
                boundary_bonds.append((atom, bonded_atom))

    print(f"[Cut] 最终 QM 区域包含 {len(qm_atom_set)} 个原子。")
    print(f"[Cut] 识别到 {len(boundary_bonds)} 处切断的化学键。")

    # 6. 生成 pre-dictionary.dat (重点修改区域)
    bond_dict = {}
    
    # 6.1 处理 QM ID
    qm_ids = sorted([a.idx + 1 for a in qm_atom_set])
    qm_idx_set = set(a.idx for a in qm_atom_set) # 用于快速查找
    
    # 6.2 处理 MM ID (新逻辑)
    print(f"[Cut] 正在根据 region_cutoff_dist={region_cutoff_dist}A 筛选 MM 区域...")
    
    # 步骤 A: 找到所有在 region_cutoff_dist 范围内的原子
    mm_search_mask = f"{center_mask}<:{region_cutoff_dist}"
    mm_nearby_view = struct.view[mm_search_mask]
    
    # 步骤 B: 获取涉及的完整残基 (set去重)
    # 只要残基中有任何原子在范围内，整个残基都被视为候选对象
    mm_candidate_residues = set()
    for atom in mm_nearby_view.atoms:
        mm_candidate_residues.add(atom.residue)
        
    # 步骤 C: 筛选属于这些残基、但不在 QM 区域的原子
    mm_ids_list = []
    
    # 注意：这里我们遍历候选残基的所有原子，确保残基完整性
    for res in mm_candidate_residues:
        for atom in res.atoms:
            # 核心判断：如果该原子不在 QM 集合中，它就是 MM 原子
            if atom.idx not in qm_idx_set:
                mm_ids_list.append(atom.idx + 1)
    
    # 排序并去重 (虽然逻辑上应该无重复，但为了保险)
    mm_ids = sorted(list(set(mm_ids_list)))
    
    print(f"[Cut] 最终 MM 区域包含 {len(mm_ids)} 个原子 (Active MM).")

    bond_dict['QM'] = qm_ids
    bond_dict['MM'] = mm_ids
    
    # 填充边界信息 Q1, M1, M2, M3
    for i, (q1, m1) in enumerate(boundary_bonds):
        n = i + 1
        bond_dict[f'Q1_{n}'] = [q1.idx + 1]
        bond_dict[f'M1_{n}'] = [m1.idx + 1]
        
        # M2: 与 M1 相连，且不是 Q1
        m2_atoms = [a for a in m1.bond_partners if a != q1]
        bond_dict[f'M2_{n}'] = [a.idx + 1 for a in m2_atoms]
        
        # M3: 与 M2 相连，且不是 M1
        m3_ids = []
        for m2 in m2_atoms:
            for m3 in m2.bond_partners:
                if m3 != m1:
                    m3_ids.append(m3.idx + 1)
        bond_dict[f'M3_{n}'] = list(set(m3_ids))

    with open('pre-dictionary.dat', 'w') as f:
        json.dump(bond_dict, f, indent=2)

    # 7. 保存 QM PDB
    if qm_ids:
        qm_mask_str = "@" + ",".join(map(str, qm_ids))
        struct[qm_mask_str].save("QM.pdb", overwrite=True)
        
        # 可选：如果你也想保存 Active MM 的结构用于检查，可以取消下面注释
        # if mm_ids:
        #     mm_mask_str = "@" + ",".join(map(str, mm_ids))
        #     struct[mm_mask_str].save("MM_active.pdb", overwrite=True)
    else:
        print("[Error] QM 区域为空！")

    return len(qm_ids)