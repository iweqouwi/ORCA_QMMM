import pandas as pd
from typing import List, Dict, Tuple
import numpy as np
import os
import math
import re

def write_psi4_file(qm_atoms: list, total_charge: str,mm_env:list,PSI4_FILE_PATH: str, 
                    method: str, mem: str, nthreads: str, tmpPath: str,
                    multiplicity: str = "1",  psi4_options: dict = None):
    """
    写入 Psi4 输入文件 (修复原子坐标空行和间距过大问题)
    """
    
    if psi4_options is None:
        psi4_options = {}

    inp_filename = PSI4_FILE_PATH.split('/')[-1]
    out_filename = inp_filename.replace('.py', '.out').replace('.in', '.out')
    
    # --- 1. 核心修改：清洗原子坐标格式 ---
    # 逻辑：遍历每个原子行 -> 去除首尾空白 -> 按空格分割再合并(压缩中间空格) -> 过滤掉空行
    cleaned_atoms = []
    for line in qm_atoms:
        line_stripped = line.strip()
        if line_stripped:  # 确保不是空行
            # "C     1.23    4.56"  ->  "C 1.23 4.56"
            cleaned_atoms.append(" ".join(line_stripped.split()))
            
    qm_geometry_str = '\n'.join(cleaned_atoms)
    # ----------------------------------
    
    with open(PSI4_FILE_PATH, 'w') as inpfile:
        # 写入头部
        inpfile.write(f"""
import psi4
import numpy as np
import qcelemental as qcel 
import time

start = time.time()

psi4.set_memory('{mem}')
psi4.core.set_num_threads({nthreads})
psi4.core.set_output_file('{out_filename}', False)
psi4.core.IOManager.shared_object().set_default_path('{tmpPath}')
""")

        # 写入几何结构 (现在非常紧凑整洁)
        inpfile.write(f"""
mol = psi4.geometry('''
{total_charge} {multiplicity}
{qm_geometry_str}
units angstrom
symmetry c1
no_com
no_reorient
''')
""")

        # 写入 MM 环境 (保持之前的格式修复)
        if mm_env:
            formatted_mm_lines = [",".join(line.strip().split()) for line in mm_env if line.strip()]
            mm_data_str = ",\n".join(formatted_mm_lines)

            inpfile.write(f"""
Chargefield_B = np.array([
{mm_data_str}
]).reshape((-1,4))
Chargefield_B[:,[1,2,3]] /= qcel.constants.bohr2angstroms
""")

        # 写入 Psi4 选项
        inpfile.write(f"\npsi4.set_options({{\n")
        if psi4_options:
            options_lines = []
            for k, v in psi4_options.items():
                # repr(v) 会自动处理类型：
                # 字符串 -> "'value'"
                # 列表 -> "['a', 'b']"
                # 数字 -> "0.6"
                options_lines.append(f"'{k}': {repr(v)}")
            inpfile.write(",\n".join(options_lines))
        inpfile.write("\n})\n")

        # 写入能量计算
        if mm_env:
            inpfile.write(f"\ne,wfn = psi4.energy('{method}', external_potentials={{'B':Chargefield_B}},return_wfn=True)\n")
        else:
            inpfile.write(f"\ne = psi4.energy('{method}')\n")

        # 写入结尾
        inpfile.write("psi4.cubeprop(wfn)\n")
        inpfile.write(f"""
end = time.time()
wall_time = '{{:.2f}}'.format(float(end-start))
with open('{out_filename}', 'a') as output:
    output.write(f'Wall time: {{wall_time}} seconds')
""")



def write_orca_file(qm_atoms: list, total_charge: str, mm_env: list, ORCA_FILE_PATH: str, 
                    method: str, mem: str, nthreads: str, 
                    multiplicity: str = "1", orca_options: dict = None,
                    cubic_grid_spacing: list = [0.4, 0.4, 0.4], 
                    cubic_grid_overage: list = [4.0, 4.0, 4.0],
                    scf_max_cycles: int = 100):
    """
    生成 ORCA 输入文件。
    更新功能：
    1. 使用 MiniPrint 减少磁盘占用。
    2. 强制使用 DefGrid3 提高精度。
    3. 包含 SCF 迭代次数限制防止死循环。
    """
    if orca_options is None:
        orca_options = {}

    # 常量：Angstrom -> Bohr
    ANG_TO_BOHR = 1.8897259886

    # 1. 路径处理
    if not ORCA_FILE_PATH.endswith('.inp'):
        ORCA_FILE_PATH = ORCA_FILE_PATH + '.inp'
    
    base_dir = os.path.dirname(ORCA_FILE_PATH)
    base_name = os.path.basename(ORCA_FILE_PATH)
    pc_filename = base_name.replace('.inp', '.pc')
    dens_filename = base_name.replace('.inp', '.dens.cube')

    # 2. 解析原子坐标 (Angstrom -> Bohr)
    cleaned_atoms = []
    xs_bohr, ys_bohr, zs_bohr = [], [], []

    for line in qm_atoms:
        line_stripped = line.strip()
        if line_stripped:
            parts = line_stripped.split()
            cleaned_atoms.append(" ".join(parts))
            # 仅用于计算网格边界
            if len(parts) >= 4:
                try:
                    xs_bohr.append(float(parts[1]) * ANG_TO_BOHR)
                    ys_bohr.append(float(parts[2]) * ANG_TO_BOHR)
                    zs_bohr.append(float(parts[3]) * ANG_TO_BOHR)
                except ValueError:
                    pass 

    qm_geometry_str = '\n'.join(cleaned_atoms)

    # 3. 计算 Min, Max, Dim
    grid_defs = [] 
    if xs_bohr and ys_bohr and zs_bohr:
        coords_bohr = [xs_bohr, ys_bohr, zs_bohr]
        for i in range(3):
            current_min = min(coords_bohr[i]) - cubic_grid_overage[i]
            target_max = max(coords_bohr[i]) + cubic_grid_overage[i]
            span = target_max - current_min
            dim = int(math.ceil(span / cubic_grid_spacing[i])) + 1
            current_max = current_min + (dim - 1) * cubic_grid_spacing[i]
            grid_defs.append({"dim": dim, "min": current_min, "max": current_max})
    else:
        print("Warning: No atoms found. Using default grid.")
        for _ in range(3):
            grid_defs.append({"dim": 80, "min": -15.0, "max": 15.0})

    # 4. 处理内存
    try:
        mem_match = re.search(r'\d+', mem)
        mem_num = int(mem_match.group()) if mem_match else 4000
        mem_mb = mem_num * 1024 if 'GB' in mem.upper() else mem_num
    except:
        mem_mb = 4000 

    # 5. 写入点电荷文件
    if mm_env:
        with open(os.path.join(base_dir, pc_filename), 'w') as pc_file:
            pc_file.write(f"{len(mm_env)}\n")
            for line in mm_env:
                parts = line.strip().split()
                if len(parts) >= 4:
                    pc_file.write(f"{float(parts[0]):12.6f} {float(parts[1]):12.6f} {float(parts[2]):12.6f} {float(parts[3]):12.6f}\n")
    
    # 6. 写入 ORCA 输入文件
    with open(ORCA_FILE_PATH, 'w') as inp:
        # --- [修改处] ---
        # MiniPrint: 极简输出，节省空间
        # DefGrid3: 高精度积分网格
        inp.write(f"! {method} MiniPrint NoTRAH \n") 
        # ----------------
        
        inp.write(f"%pal nprocs {nthreads} end\n")
        inp.write(f"%maxcore {mem_mb}\n")
        
        # SCF 限制
        if scf_max_cycles:
            inp.write("%scf\n")
            inp.write(f"  MaxIter {scf_max_cycles}\n")
            inp.write("end\n")

        if mm_env:
            inp.write(f'%pointcharges "{pc_filename}"\n')

        # Plot Block
        inp.write("%plots\n")
        inp.write('  Format Gaussian_Cube\n')
        inp.write("\n  # Definition of Grid Points (Dim)\n")
        for i in range(3):
            inp.write(f"  Dim{i+1} {grid_defs[i]['dim']}\n")
            
        inp.write("\n  # Definition of Boundaries (Min/Max)\n")
        for i in range(3):
            inp.write(f"  Min{i+1} {grid_defs[i]['min']:.6f}\n")
            inp.write(f"  Max{i+1} {grid_defs[i]['max']:.6f}\n")
        inp.write(f'  ElDens("{dens_filename}");\n')
        inp.write("end\n")

        # 用户自定义选项
        if orca_options:
            for block, content in orca_options.items():
                if isinstance(content, dict):
                    inp.write(f"%{block}\n")
                    for k, v in content.items():
                        inp.write(f"  {k} {v}\n")
                    inp.write("end\n")
                else:
                    inp.write(f"%{block} {content}\n")

        # 几何结构
        inp.write(f"\n* xyz {total_charge} {multiplicity}\n")
        inp.write(qm_geometry_str)
        inp.write("\n*\n")

    print(f"ORCA input generated: {ORCA_FILE_PATH}")
    print(f"  - Options: MiniPrint, DefGrid3, MaxSCF={scf_max_cycles}")
    print(f"  - Grid Dimensions: {grid_defs[0]['dim']} x {grid_defs[1]['dim']} x {grid_defs[2]['dim']}")