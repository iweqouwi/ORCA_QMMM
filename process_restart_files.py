import os
import gzip
import shutil
import subprocess
import csv
import sys  # 新增: 用于退出程序
from pathlib import Path
import parmed as pmd
from pymol import cmd, stored
import numpy as np
from scipy.spatial.distance import cdist # 建议显式导入

# ==========================================
# 新增功能: 自动查找配体残基名称
# ==========================================
def find_ligand_resname(top_file):
    """
    解析拓扑文件，排除标准氨基酸、水和离子，查找配体残基名称。
    """
    print(f"[*] 正在分析拓扑文件以自动识别配体: {top_file}")
    
    try:
        struct = pmd.load_file(str(top_file))
    except Exception as e:
        print(f"[-] 致命错误: 无法通过 ParmEd 加载拓扑文件: {e}")
        sys.exit(1)

    # 定义标准排除列表 (大写)
    standard_residues = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'HID', 'HIE', 'HIP', 'CYX', 'CYM', 'ASH', 'GLH', 'LYN', 'ARN', # Amber variants
        'ACE', 'NME', 'NHE', # Caps
        'WAT', 'HOH', 'TIP3', 'TP3', 'TIP4P', 'T4P', # Water
        'NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'LI', 'CS', 'RB', 'IB', # Ions
        'CIO', 'IB', 'K+', 'NA+', 'CL-', 'MG2+', 'CA2+' # Common Ion variants
    }

    # 1. 遍历收集所有非标准残基
    found_candidates = []
    
    for residue in struct.residues:
        rname = residue.name.upper()
        # 排除标准残基
        if rname not in standard_residues:
            # 记录尚未记录的残基名 (保持出现顺序)
            if rname not in found_candidates:
                found_candidates.append(rname)

    # 2. 核心检查：如果完全没有非标准残基，直接退出
    if not found_candidates:
        print(f"[-] 致命错误: 在拓扑文件 {top_file} 中未发现任何配体（非标准残基）！")
        print("[-] 系统只包含标准氨基酸、水或离子。程序终止。")
        sys.exit(1)

    # 打印所有找到的候选者
    print(f"[*] 扫描到的非标准残基候选列表: {found_candidates}")

    # 3. 定义优先级列表 (越靠前优先级越高)
    priority_list = ['MOL', 'LIG', 'UNK', 'DRG', 'INH']
    
    selected_ligand = None

    # 检查是否有高优先级名称
    for p_name in priority_list:
        if p_name in found_candidates:
            selected_ligand = p_name
            print(f"[+] 命中常用配体名称，优先选择: {selected_ligand}")
            break
    
    # 4. 如果没有命中优先级列表，选择第一个
    if selected_ligand is None:
        selected_ligand = found_candidates[0]
        print(f"[*] 未发现常用配体名称 (MOL/LIG等)，默认选择列表第一个: {selected_ligand}")
    
    return selected_ligand

# ==========================================
# 新增功能: 调用 Sander 进行 最小化-淬火-最小化
# ==========================================
def run_sander_optimization(top_file, input_coord, output_coord, n_threads=1):
    """
    使用 Sander 引擎执行优化。
    """
    top_file = Path(top_file).resolve()
    input_coord = Path(input_coord).resolve()
    output_coord = Path(output_coord).resolve()
    work_dir = output_coord.parent

    # 确定运行命令
    if n_threads > 1:
        base_cmd = ["mpirun", "-np", str(n_threads), "sander.MPI"]
        print(f"[*] 正在启动 Sander 优化流程 (并行模式: {n_threads} 核心)...")
    else:
        base_cmd = ["sander"]
        print(f"[*] 正在启动 Sander 优化流程 (串行模式)...")

    prefix = output_coord.stem
    min1_rst = work_dir / f"{prefix}_step1_min.rst"
    quench_rst = work_dir / f"{prefix}_step2_quench.rst"

    # --- Step 1: 初始最小化 ---
    in_min1 = work_dir / "sander_step1_min.in"
    with open(in_min1, 'w') as f:
        f.write(f"""Initial Minimization
 &cntrl
   imin=1, maxcyc=100, ncyc=10,
   ntb=1,          ! Constant Volume
   cut=10.0,       ! 非键截断
   ntpr=5,         ! 打印频率
   ntc=2, ntf=2,
 /
""")

    # --- Step 2: 100K 淬火 ---
    in_quench = work_dir / "sander_step2_quench.in"
    with open(in_quench, 'w') as f:
        f.write(f"""Quench to 100K
 &cntrl
   imin=0, nstlim=1000, dt=0.002,
   ntx=1, irest=0,
   ntt=3, gamma_ln=2.0, temp0=100.0, tempi=100.0,
   ntb=1,
   cut=10.0, ntpr=100,
   ntc=2, ntf=2,
 /
""")

    # --- Step 3: 最终最小化 ---
    in_min2 = work_dir / "sander_step3_min.in"
    with open(in_min2, 'w') as f:
        f.write(f"""Final Minimization
 &cntrl
   imin=1, maxcyc=100, ncyc=10,
   drms=0.1,       ! 收敛判据
   ntb=1, cut=10.0, ntpr=10,
   ntc=2, ntf=2,
 /
""")

    try:
        # 运行 Step 1
        cmd1 = base_cmd + [
            "-O",
            "-i", str(in_min1), "-p", str(top_file), "-c", str(input_coord),
            "-r", str(min1_rst), "-o", str(work_dir / "step1.out")
        ]
        subprocess.run(cmd1, check=True, capture_output=True)

        # 运行 Step 2
        cmd2 = base_cmd + [
            "-O",
            "-i", str(in_quench), "-p", str(top_file), "-c", str(min1_rst),
            "-r", str(quench_rst), "-o", str(work_dir / "step2.out")
        ]
        subprocess.run(cmd2, check=True, capture_output=True)

        # 运行 Step 3
        cmd3 = base_cmd + [
            "-O",
            "-i", str(in_min2), "-p", str(top_file), "-c", str(quench_rst),
            "-r", str(output_coord), "-o", str(work_dir / "step3.out")
        ]
        subprocess.run(cmd3, check=True, capture_output=True)
        
        print(f"[+] Sander 优化完成。最终坐标: {output_coord}")

    except subprocess.CalledProcessError as e:
        print(f"[-] Sander 运行失败! Return Code: {e.returncode}")
        if n_threads > 1:
            print("[-] 提示: 并行运行时失败。请确保系统中安装了 'mpirun' 和 'sander.MPI'。")
        if e.stderr:
            print(f"[-] Stderr: {e.stderr.decode('utf-8')}")
        raise
    finally:
        for f in [in_min1, in_quench, in_min2, min1_rst, quench_rst]:
            if f.exists():
                f.unlink()

def get_amber_formal_charge(atom):
    """
    根据提供的 OpenMM (SPICE数据库) 脚本中的 residues 列表逻辑进行硬编码。
    """
    resn = atom.residue.name.upper()
    name = atom.name.upper()

    if name == 'OXT': return -1
    if resn == 'ASP' and name == 'OD2': return -1
    if resn == 'GLU' and name == 'OE2': return -1
    if resn == 'CYM' and name == 'SG': return -1 
    if resn == 'CYX' and name == 'SG': return 0  # 修正: 二硫键硫原子为中性
    if resn == 'LYS' and name == 'NZ': return 1
    if resn == 'ARG' and name == 'NH2': return 1
    if resn == 'HIP' and name == 'ND1': return 1
    return 0

def decompress_top(gz_path, top_path):
    if not top_path.exists():
        print(f"[*] Decompressing {gz_path} ...")
        with gzip.open(gz_path, 'rb') as f_in, open(top_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def run_cpptraj_center_ligand(top_file, rst_file, output_coord, ligand_resname):
    """
    强制将配体居中，并重组溶剂盒子。
    """
    ligand_mask = f":{ligand_resname}"
    print(f"[*] 正在以配体 {ligand_mask} 为中心进行重组...")
    
    top_file = Path(top_file).resolve()
    rst_file = Path(rst_file).resolve()
    output_coord = Path(output_coord).resolve()
    
    work_dir = output_coord.parent
    
    cpptraj_in_content = f"""
        parm {top_file}
        trajin {rst_file}
        center {ligand_mask} origin mass
        image origin center
        trajout {output_coord} restart
        run
    """
    
    in_file = work_dir / "temp_center_lig.in"
    
    with open(in_file, "w") as f:
        f.write(cpptraj_in_content)
    
    try:
        subprocess.run(
            ["cpptraj", "-i", str(in_file)], 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"[+] 居中处理完成，中间坐标文件已保存至: {output_coord}")
        
    except subprocess.CalledProcessError as e:
        print(f"[-] cpptraj 处理失败 (Centering)。任务目录: {work_dir}")
        print(f"[-] Return Code: {e.returncode}")
        print("[-] Cpptraj STDERR:", e.stderr)
        raise
        
    finally:
        if in_file.exists():
            in_file.unlink()

# ==========================================
# 新增功能: 专门用于去除溶剂并保存干燥PDB
# ==========================================
def run_cpptraj_strip_solvent(top_file, rst_file, output_pdb):
    """
    使用 cpptraj 去除水和离子，保存为干燥的 PDB 文件 (含整个蛋白和配体)。
    """
    print(f"[*] 正在去除溶剂并保存干复合物 PDB: {output_pdb}")
    
    top_file = Path(top_file).resolve()
    rst_file = Path(rst_file).resolve()
    output_pdb = Path(output_pdb).resolve()
    work_dir = output_pdb.parent
    
    # 定义标准溶剂和离子 Mask (Amber 常用)
    solvent_mask = ":WAT,HOH,TIP3,TIP4P,Na+,Cl-,K+,Mg+,Ca+,Zn+,Cs+,Li+,Rb+,CIO,IB,MG,CL,NA"
    
    cpptraj_in_content = f"""
        parm {top_file}
        trajin {rst_file}
        strip {solvent_mask}
        trajout {output_pdb} pdb
        run
    """
    
    in_file = work_dir / "temp_strip_solvent.in"
    
    with open(in_file, "w") as f:
        f.write(cpptraj_in_content)
    
    try:
        subprocess.run(
            ["cpptraj", "-i", str(in_file)], 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"[+] 去水处理完成，文件已保存。")
        
    except subprocess.CalledProcessError as e:
        print(f"[-] cpptraj 处理失败 (Stripping)。")
        print("[-] Cpptraj STDERR:", e.stderr)
        raise
    finally:
        if in_file.exists():
            in_file.unlink()

def filter_and_save_data(nc_file, parent_dir):
    """
    加载拓扑和轨迹，不进行切割，直接将全量信息写入 CSV。
    注：region_cutoff_dist 和 ligand_resname 参数虽保留但在此函数中不再起作用。
    """
    parent_path = Path(parent_dir)
    nc_path = Path(nc_file)
    top_path = parent_path / "production.top"
    
    # 1. 加载拓扑和轨迹
    print(f"[*] 正在加载原始拓扑: {top_path}")
    try:
        struct_orig = pmd.load_file(str(top_path))
        traj = pmd.load_file(str(nc_path))
    except Exception as e:
        print(f"[-] 文件加载失败: {e}")
        sys.exit(1)

    # 2. 同步坐标 (将轨迹的第一帧坐标赋给拓扑对象)
    print("[*] 正在同步轨迹坐标...")
    try:
        # 尝试获取第一帧，如果 traj 本身就是一帧则直接使用
        frame_coords = traj.coordinates[0]
    except (IndexError, TypeError):
        frame_coords = traj.coordinates

    if len(struct_orig.atoms) != len(frame_coords):
        raise ValueError(f"错误: 拓扑原子数 ({len(struct_orig.atoms)}) 与坐标原子数 ({len(frame_coords)}) 不匹配！")
        
    struct_orig.coordinates = frame_coords
    
    # 处理 Box 信息 (如果有)
    if hasattr(traj, 'box') and traj.box is not None:
        try:
            struct_orig.box = traj.box[0]
        except:
            struct_orig.box = traj.box

    # 3. 生成全量 CSV
    csv_file = parent_path / "dataframe.csv"
    print(f"[*] 正在生成全量 CSV 文件: {csv_file}")
    print(f"    -> 总原子数: {len(struct_orig.atoms)}")

    headers = ["CX_PDB_ID", "PDB_AT", "PDB_RES", "X", "Y", "Z", "q", "FORMAL_Q", "ELEMENT"]

    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            # 直接遍历原始拓扑的所有原子
            for i, atom in enumerate(struct_orig.atoms):
                # 构建残基标识字符串 (如: ALA12)
                # 注意：这里使用的是原始拓扑中的残基编号
                pdb_res_full = f"{atom.residue.name}{atom.residue.number}"
                
                # 获取坐标 (ParmEd 单位通常为 Angstrom)
                x, y, z = f"{atom.xx:.3f}", f"{atom.xy:.3f}", f"{atom.xz:.3f}"
                
                # 获取电荷
                charge_val = atom.charge 
                
                # 处理元素推断逻辑
                element = ""
                if hasattr(atom, 'element_name') and atom.element_name:
                    element = atom.element_name.upper()
                if not element and atom.name:
                    # 备用推断逻辑：从原子名推断 (如 CA -> C, H1 -> H)
                    element = atom.name[0].upper()
                    if element.isdigit() and len(atom.name) > 1:
                        element = atom.name[1].upper()
                
                # 写入行
                writer.writerow([
                    i + 1,                          # CX_PDB_ID (从1开始的原子序号)
                    atom.name,                      # PDB_AT
                    pdb_res_full,                   # PDB_RES
                    x, y, z,                        # X, Y, Z
                    f"{charge_val:.4f}",            # q
                    get_amber_formal_charge(atom),  # FORMAL_Q (需确保此辅助函数在外部已定义)
                    element                         # ELEMENT
                ])
                
        print(f"[+] 全量写入 CSV 处理成功。")

    except IOError as e:
        print(f"[-] CSV 写入失败: {e}")
        sys.exit(1)

def process_restart(parent_dir, pdb_id=None, sander_threads=1):
    parent_path = Path(parent_dir)
    
    gz_file = parent_path / "production.top.gz"
    top_file = parent_path / "production.top"
    rst_file = parent_path / "production.rst"
    
    # 1. 解压拓扑
    if gz_file.is_file():
        decompress_top(gz_file, top_file)
    if not top_file.exists():
        print(f"[-] 错误: 找不到拓扑文件 {top_file}")
        sys.exit(1)

    # 2. 自动查找配体残基名称
    # 假设该函数已定义在当前命名空间
    ligand_resname = find_ligand_resname(top_file)
    
    # 定义中间文件路径
    temp_centered = parent_path / "temp_full_centered.nc"
    temp_quenched = parent_path / "temp_full_quenched.rst"

    # 3. 居中处理
    run_cpptraj_center_ligand(top_file, rst_file, temp_centered, ligand_resname)
    
    # 4. 运行 Sander (传递线程数)
    # run_sander_optimization(top_file, temp_centered, temp_quenched, n_threads=sander_threads)

    # 5. 保存去水后的完整复合物 PDB
    dry_pdb_file = parent_path / f"{pdb_id}_dry.pdb"
    run_cpptraj_strip_solvent(top_file, temp_centered, dry_pdb_file)
    # run_cpptraj_strip_solvent(top_file, temp_quenched, dry_pdb_file)

    # 6. 筛选并保存数据
    filter_and_save_data(temp_centered, parent_dir)
    # filter_and_save_data(temp_quenched, parent_dir, region_cutoff_dist, ligand_resname)

    return ligand_resname