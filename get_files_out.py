import sys
import traceback
import os
import argparse
import shutil
import multiprocessing
from typing import Dict, Any
from pathlib import Path

# --- 保持原本的 import 不变 ---
from process_restart_files import process_restart
from cut_protein import run_cut_protein_parmed 
from move_M3s import move_m3s
from create_est_inp import make_monomers
from cap import run_cap
from write_out_files import write_orca_file

def clean_working_directory(work_path: Path, keep_extensions: list):
    """
    清理工作目录，只保留指定扩展名的文件
    """
    print(f"清理工作目录: {work_path}")
    
    # 获取要保留的文件模式
    keep_patterns = keep_extensions
    
    # 遍历工作目录中的所有文件
    for item in work_path.iterdir():
        if item.is_file():
            # 检查文件是否在保留列表中
            keep = False
            for pattern in keep_patterns:
                if item.name.endswith(pattern):
                    keep = True
                    break
            
            # 如果不是要保留的文件，删除它
            if not keep:
                try:
                    item.unlink()
                except Exception as e:
                    print(f"删除文件 {item.name} 时出错: {e}")
    print("工作目录清理完成。")

def process_single_directory(work_dir: str, config: Dict[str, Any], output_dir: Path = None, task_id: int = 0):
    """
    针对单个文件夹的处理逻辑
    """
    # 保存原始目录（尽管多进程中对主进程影响较小，但为了安全起见）
    original_cwd = os.getcwd()
    
    try:
        print(f"任务 {task_id}: 开始处理目录 {work_dir}")
        
        # 0. 获取绝对路径
        work_path = Path(work_dir).resolve()
        
        # --- 关键修改：切换当前工作目录到子文件夹 ---
        os.chdir(work_path)
        # ----------------------------------------

        log_file = work_path / "log.out"
        
        # 重定向输出到日志文件
        # 注意：这里我们使用 'w' 模式，确保覆盖旧日志
        sys.stdout = open(log_file, 'w')
        sys.stderr = sys.stdout # 将错误也重定向到同一个文件
        
        print(f"当前工作目录已切换至: {os.getcwd()}")
        
        # 1. 处理重启拓扑文件
        # 因为已经切换了目录，这里传 '.' 或者绝对路径都可以，建议传绝对路径
        process_restart(str(work_path), region_cutoff_dist="14.0")
        
        # 3. 切割QM区域
        # 注意：如果 run_cut_protein 内部依赖相对路径，现在已经在目录下，可能需要调整参数
        # 但通常传入绝对路径是安全的：
        run_cut_protein_parmed(str(work_path / "complex.pdb"), cutoff="4.0")
        
        # 5. 封端口
        run_cap(ff_type='amber')
        
        # 6. 静电嵌入处理
        qm_atoms, total_charge, mm_env, QM_capped_num_atoms = make_monomers('Z2')
        
        # 7. 写入psi4/orca代码
        cx_inp_filename = f'{work_path.name}_{QM_capped_num_atoms}.py'
        
        # write_psi4_file(
        #     qm_atoms, 
        #     total_charge, 
        #     mm_env, 
        #     cx_inp_filename,
        #     method=config['method'],
        #     mem=config['mem'],
        #     nthreads=config['nthreads'],
        #     tmpPath=config['tmpPath'],
        #     psi4_options=config['psi4_options']
        # )
        write_orca_file(
            qm_atoms, 
            total_charge, 
            mm_env, 
            cx_inp_filename,
            method=config['method'],
            mem=config['mem'],
            nthreads=config['nthreads'],
        )

        # 8. 如果指定了output目录，复制生成的.py文件
        if output_dir is not None:
            try:
                # 此时我们已经在 work_path 里面了，可以直接用文件名，或者用绝对路径
                source_file = work_path / cx_inp_filename
                if source_file.exists():
                    # 确保目标目录存在（多进程中尽量避免并在，虽然 makedirs 是线程安全的）
                    output_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = output_dir / cx_inp_filename
                    shutil.copy2(source_file, dest_file)
                    print(f"已将文件复制到: {dest_file}")
                else:
                    print(f"警告: 源文件 {source_file} 不存在，无法复制")
            except Exception as e:
                print(f"复制文件到output目录时出错: {e}")
        
        # 9. 清理工作目录
        keep_extensions = ['complex.pdb', 'QM_capped.pdb', '.top.gz', '.rst','log.out', '.py','inp','pc','.csv']
        clean_working_directory(work_path, keep_extensions)
        
        print(f"任务 {task_id}: 成功完成目录 {work_dir} 的处理")
        
        # 恢复标准输出
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
        return True, work_dir
    
    except Exception as e:
        # 发生错误时，尝试记录到控制台
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        print(f"任务 {task_id}: 处理目录 {work_dir} 时出错: {e}")
        # traceback.print_exc() # 可以取消注释查看详细堆栈
        
        return False, f"{work_dir}: {str(e)}"
        
    finally:
        # 无论成功失败，切回原来的目录
        os.chdir(original_cwd)
        # 确保文件句柄关闭
        if 'sys.stdout' in locals() and sys.stdout != sys.__stdout__:
            try:
                sys.stdout.close()
            except:
                pass
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

def batch_process_directories(parent_dir: str, config: Dict[str, Any], output_dir: Path = None, max_workers: int = None):
    """
    批量处理父目录下的所有子目录，使用进程池并行处理
    """
    parent_path = Path(parent_dir).resolve()
    
    if not parent_path.exists() or not parent_path.is_dir():
        print(f"错误: 父目录不存在或不是目录: {parent_dir}")
        return
    
    # 获取所有子目录
    subdirectories = [d for d in parent_path.iterdir() if d.is_dir()]
    
    if not subdirectories:
        print(f"警告: 父目录下没有找到任何子目录: {parent_dir}")
        return
    
    print(f"找到 {len(subdirectories)} 个子目录需要处理")
    
    # 设置进程池大小
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()  # 使用所有CPU核心
    print(f"使用进程池，最大工作进程数: {max_workers}")
    
    # 创建参数列表，为每个任务分配一个ID
    tasks = [(str(subdir), config, output_dir, i) for i, subdir in enumerate(subdirectories)]
    
    # 使用进程池并行处理
    success_count = 0
    failure_count = 0
    
    with multiprocessing.Pool(processes=max_workers) as pool:
        # 使用starmap并行处理所有目录
        results = []
        for task_args in tasks:
            result = pool.apply_async(process_single_directory, task_args)
            results.append(result)
        
        # 收集结果
        for i, result in enumerate(results):
            try:
                success, msg = result.get(timeout=3600)  # 设置超时时间为1小时
                if success:
                    success_count += 1
                    print(f"任务 {i}: 成功 - {msg}")
                else:
                    failure_count += 1
                    print(f"任务 {i}: 失败 - {msg}")
            except multiprocessing.TimeoutError:
                failure_count += 1
                print(f"任务 {i}: 超时 - {tasks[i][0]}")
            except Exception as e:
                failure_count += 1
                print(f"任务 {i}: 异常 - {e}")
    
    # 输出统计信息
    print(f"\n处理完成!")
    print(f"成功: {success_count}")
    print(f"失败: {failure_count}")
    print(f"总计: {len(subdirectories)}")

if __name__ == "__main__":
    # --- 1. 定义计算参数 (Configuration) ---
    options = {
        'method': 'r2scan-3c',
        'mem': '64GB', 
        'nthreads': '32',
        'tmpPath': '/ampha/tenant/fyust/private/user/houkun/psi4_scratch',
        'psi4_options': {
            # 'basis_set': '',
            'scf_type': 'df',
            'cubeprop_tasks': ['density'],
            'cubic_grid_spacing': [0.4, 0.4, 0.4],
            'cubic_grid_overage': [4.0, 4.0, 4.0],
        }
    }

    # --- 2. 解析命令行参数 (Argument Parsing) ---
    parser = argparse.ArgumentParser(description="批量处理指定父目录下的所有子文件夹")
    # 添加位置参数 parent_dir
    parser.add_argument("--parent", type=str, default= "parameter_restart_files_MD",help="包含各个体系文件夹的父目录路径")
    # 添加output参数
    parser.add_argument("--output", type=str, default = "cal_files",help="输出目录路径，用于保存生成的.py文件")
    # 添加进程数参数
    parser.add_argument("--processes", type=int, default=64,
                       help="并行处理的进程数，默认为CPU核心数")
    args = parser.parse_args()
    
    # 将output路径转换为Path对象
    output_path = Path(args.output).resolve() if args.output else None
    
    # 批量处理所有目录
    batch_process_directories(
        parent_dir=args.parent,
        config=options,
        output_dir=output_path,
        max_workers=args.processes
    )