import sys
import traceback
import os
import argparse
import shutil
import tempfile  # 新增：用于创建临时目录
import time      # 新增：用于随机休眠
import random    # 新增：用于随机休眠
from typing import Dict, Any
from pathlib import Path

# 假设这些是你自定义的模块
from process_restart_files import process_restart
from cut_protein import run_cut_protein_parmed 
from create_est_inp import make_monomers
from cap import run_cap
from write_out_files import write_orca_file

# --- 辅助函数：将源目录的所有内容复制到目标目录 ---
def copy_inputs_to_local(source_dir: Path, target_dir: Path):
    if not source_dir.exists():
        raise FileNotFoundError(f"源目录不存在: {source_dir}")
    
    # 遍历源目录复制文件
    for item in source_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, target_dir / item.name)
        elif item.is_dir():
            # 如果有子文件夹需要复制，使用 copytree
            shutil.copytree(item, target_dir / item.name, dirs_exist_ok=True)

def process_single_directory(work_dir: str, config: Dict[str, Any], output_dir: Path = None, md_threads: int = 1):
    original_cwd = os.getcwd()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file_obj = None
    
    # 1. 解析原始路径（共享存储路径）
    source_path = Path(work_dir).resolve() # 这是输入文件所在的共享存储路径
    
    # 确定最终结果存放路径
    # 如果指定了 output_dir，结果存到 output_dir/文件夹名
    # 如果没指定，结果通常需要存回 source_path (原地更新模式)
    if output_dir:
        final_dest_path = output_dir / source_path.name
    else:
        final_dest_path = source_path

    # 2. 设定本地暂存盘根目录
    # SLURM系统通常会在作业里设置 TMPDIR 或 SLURM_TMPDIR 环境变量指向本地SSD
    # 如果没有，默认为 /tmp
    local_scratch_root = os.environ.get("SLURM_TMPDIR", os.environ.get("TMPDIR", "/tmp")) 
    # 使用 TemporaryDirectory 自动管理创建和清理
    local_work_dir_name = f"job_{source_path.name}_{int(time.time())}_{random.randint(1000,9999)}"
    local_work_path = Path(local_scratch_root) / local_work_dir_name
    
    try:
        # --- 步骤 A: 错峰启动 (防止并发复制风暴) ---
        # 让任务随机等待 1-10 秒，避免 256 个任务同时读硬盘
        time.sleep(random.uniform(1, 10))

        # --- 步骤 B: 创建本地环境并从共享存储拷入文件 ---
        local_work_path.mkdir(parents=True, exist_ok=True)
        
        # 将 source_path 下的所有输入文件拷贝到 local_work_path
        copy_inputs_to_local(source_path, local_work_path)
        print(f"[{source_path.name}] 输入文件已复制到本地暂存盘: {local_work_path}")

        # --- 步骤 C: 切换到本地目录开始计算 ---
        os.chdir(local_work_path)

        # --- 步骤 d: 记录原始 pdb id 以便后续使用
        original_pdb_id = source_path.name

        # 开启日志 (日志现在写在本地磁盘，飞快)
        log_file_path = local_work_path / "log.out"
        log_file_obj = open(log_file_path, 'w', buffering=1)
        sys.stdout = log_file_obj
        sys.stderr = log_file_obj 
        
        print(f"作业运行节点: {os.uname().nodename}")
        print(f"本地工作目录 (Local Scratch): {local_work_path}")
        print(f"原始数据目录 (Shared Storage): {source_path}")
        print("=== 开始计算任务 ===")
        
        # --- 核心计算流程 (完全不变，但此时 IO 都在本地 SSD 上) ---
        ligand_residue_name = process_restart(str(local_work_path), pdb_id = original_pdb_id, sander_threads=md_threads)
        run_cut_protein_parmed( cutoff=5.0, region_cutoff_dist=12.0,  ligand_residue_name = ligand_residue_name)
        run_cap(ff_type='amber',pdb_id=original_pdb_id)
        qm_atoms, total_charge, mm_env, QM_capped_num_atoms = make_monomers('Z2', ligand_residue_name)
        
        cx_inp_filename = f'{source_path.name}_{QM_capped_num_atoms}' # 注意文件名可能要基于 source 名字
        
        write_orca_file(
            qm_atoms, 
            total_charge, 
            mm_env, 
            ORCA_FILE_PATH=cx_inp_filename,
            method=config['method'],
            mem=config['mem'],
            nthreads=config['nthreads'],
            cubic_grid_spacing = config['cubic_grid_spacing'],
            cubic_grid_overage = config['cubic_grid_overage']
        )
        print("=== 计算任务结束 ===")

        # --- 步骤 D: 停止日志 ---
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file_obj.close() 
        log_file_obj = None 
        
        print(f"[{source_path.name}] 计算完成，开始将结果从本地回传至共享存储...")

        # --- 步骤 E: 结果回传 (Copy Out) ---
        transfer_suffixes = ['QM_capped.pdb','log.out','inp','pc','dry.pdb'] 
        
        # 确保目标目录存在
        final_dest_path.mkdir(parents=True, exist_ok=True)

        files_copied_count = 0
        for file_obj in local_work_path.iterdir():
            if file_obj.is_file():
                should_copy = False
                for suffix in transfer_suffixes:
                    if file_obj.name.endswith(suffix):
                        should_copy = True
                        break
                
                if should_copy:
                    try:
                        # 复制回共享存储
                        shutil.copy2(file_obj, final_dest_path / file_obj.name)
                        files_copied_count += 1
                    except Exception as e:
                        print(f" [Error] 回传文件 {file_obj.name} 失败: {e}")

        print(f"[{source_path.name}] 处理全部完成。回传文件数: {files_copied_count}")
        return True, work_dir
    
    except Exception as e:
        # 异常处理：即使失败，也要尝试把 log 拷出来，否则不知道哪里错了
        if log_file_obj and not log_file_obj.closed:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file_obj.close()
            log_file_obj = None
        
        print(f"\n[Error] 处理目录 {source_path.name} 时失败: {e}")
        traceback.print_exc()

        # 尝试抢救日志文件
        try:
            local_log = local_work_path / "log.out"
            if local_log.exists():
                final_dest_path.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_log, final_dest_path / "log_failed.out")
                print(f"已将错误日志保存至: {final_dest_path / 'log_failed.out'}")
        except:
            pass

        return False, f"{work_dir}: {str(e)}"
        
    finally:
        # --- 步骤 F: 无论成功失败，必须清理本地临时目录 ---
        os.chdir(original_cwd)
        if sys.stdout != original_stdout:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        if log_file_obj and not log_file_obj.closed:
            log_file_obj.close()
            
        # 暴力删除本地临时目录（相当于原来的 clean_working_directory，但更彻底）
        if local_work_path.exists():
            try:
                shutil.rmtree(local_work_path)
                print(f"本地临时清理完毕: {local_work_path}")
            except Exception as e:
                print(f"[Warning] 清理本地目录失败: {e}")

if __name__ == "__main__":
    # --- 配置区域 ---
    options = {
        'method': 'r2scan-3c',
        'mem': '64GB', 
        'nthreads': '32',
        # tmpPath 如果是给 Psi4 用的，最好也指向本地暂存盘
        # 这里的 tmpPath 可以动态修改为 os.environ.get("SLURM_TMPDIR", "/tmp")
        'tmpPath': os.environ.get("SLURM_TMPDIR", '/tmp'), 
        'psi4_options': {
            'scf_type': 'df',
            'cubeprop_tasks': ['density'],
            'cubic_grid_spacing': [0.4, 0.4, 0.4],
            'cubic_grid_overage': [4.0, 4.0, 4.0],
        },
        'cubic_grid_spacing': [0.4, 0.4, 0.4],
        'cubic_grid_overage': [4.0, 4.0, 4.0],
    }

    parser = argparse.ArgumentParser(description="处理单个复合物文件夹")
    parser.add_argument("--dir", type=str, default= "/ampha/tenant/fyust/private/user/houkun/Process_protein/parameter_restart_files_MD/1a99", help="指定要处理的单个复合物文件夹路径")
    parser.add_argument("--output", type=str, default="orca_files", help="输出目录路径")
    parser.add_argument("--md_cpu", type=int, default=1, help="用于Sander MD模拟的CPU核心数")
    
    args = parser.parse_args()
    
    # 确保 output 也是绝对路径
    output_path = Path(args.output).resolve() if args.output else None
    
    print(f"目标源目录: {args.dir}")
    if output_path:
        print(f"最终输出目录: {output_path}")

    success, msg = process_single_directory(
        work_dir=args.dir,
        config=options,
        output_dir=output_path,
        md_threads=args.md_cpu
    )

    if success:
        sys.exit(0)
    else:
        sys.exit(1)