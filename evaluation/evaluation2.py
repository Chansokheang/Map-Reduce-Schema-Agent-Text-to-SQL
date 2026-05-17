import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
import pdb
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        
def load_json(dir):
    with open(dir, 'r', encoding='utf8') as j:
        contents = json.loads(j.read())
    return contents

def result_callback(result):
    exec_result.append(result)


def execute_sql(predicted_sql,ground_truth, db_path):
    import os

    if not os.path.exists(db_path):
        print("not exist...")
        raise FileNotFoundError(f"Database file not found at {db_path}")
    
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res



def execute_model(predicted_sql,ground_truth, db_place, idx, meta_time_out):
    try:
        res = func_timeout(meta_time_out, execute_sql,
                                  args=(predicted_sql, ground_truth, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f'timeout',)]
        res = 0
    except Exception as e:
        result = [(f'error',)]  # possibly len(query) > 512 or not executable
        res = 0
    # result = str(set([ret[0] for ret in result]))
    result = {'sql_idx': idx, 'res': res}
 
    return result


def package_sqls(sql_path, db_root_path, mode='gpt', data_mode='dataset', file_name=None):
    """
    Package SQL queries for execution
    
    Args:
        sql_path: Base path for SQL files
        db_root_path: Path to database files
        mode: Mode for SQL processing (gpt or gt)
        data_mode: Dataset mode (test or dev)
        file_name: Optional specific file name to use (overrides default)
    """
    clean_sqls = []
    db_path_list = []
    
    # Determine file path based on arguments
    if mode == 'gpt':
        if file_name:
            # Use specified file name if provided
            file_path = f"{sql_path}{file_name}"
        else:
            # Default file
            file_path = f"{sql_path}claude.json"
            
        print("Checking file path:", file_path)
        sql_data = json.load(open(file_path, 'r', encoding='utf8'))
        
        for idx, sql_str in sql_data.items():
            delimiter = '\t----- bird -----\t'
            sql, db_name = sql_str.split(delimiter)
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    elif mode == 'gt':
        # Ground truth SQL file path
        if data_mode == 'test':
            file_path = sql_path + 'test' + '_gold.sql'
        else:
            file_path = sql_path + 'dev' + '_gold.sql'
            
        print("Using ground truth file:", file_path)
        sqls = open(file_path)
        sql_txt = sqls.readlines()
        
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split('\t')
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    return clean_sqls, db_path_list

def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i,sql_pair in enumerate(sqls):

        predicted_sql, ground_truth = sql_pair
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out), callback=result_callback)
    pool.close()
    pool.join()

def sort_results(list_of_dicts):
  return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_acc_by_diff(exec_results, diff_json_path, start_idx=0, end_idx=None):
    # pdb.set_trace()
    num_queries = len(exec_results)
    results = [res['res'] for res in exec_results]
    contents = load_json(diff_json_path)

    # Apply range filter to contents
    if end_idx is not None:
        contents = contents[start_idx:end_idx]
    else:
        contents = contents[start_idx:]

    simple_results, moderate_results, challenging_results = [], [], []

    # Only iterate over the number of results we have (handles partial evaluation)
    for i in range(min(len(contents), num_queries)):
        content = contents[i]
        if content['difficulty'] == 'simple':
            simple_results.append(exec_results[i])

        if content['difficulty'] == 'moderate':
            moderate_results.append(exec_results[i])

        if content['difficulty'] == 'challenging':
            challenging_results.append(exec_results[i])

    # Handle empty categories gracefully
    simple_acc = sum([res['res'] for res in simple_results])/len(simple_results) if simple_results else 0
    moderate_acc = sum([res['res'] for res in moderate_results])/len(moderate_results) if moderate_results else 0
    challenging_acc = sum([res['res'] for res in challenging_results])/len(challenging_results) if challenging_results else 0
    all_acc = sum(results)/num_queries if num_queries > 0 else 0
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists



def print_data(score_lists,count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))

    print('======================================    ACCURACY    =====================================')
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy', *score_lists))


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description="Evaluate SQL prediction accuracy")
    args_parser.add_argument('--predicted_sql_path', type=str, required=True, default='',
                           help="Path to directory containing predicted SQL files")
    args_parser.add_argument('--ground_truth_path', type=str, required=True, default='',
                           help="Path to directory containing ground truth SQL files")
    args_parser.add_argument('--data_mode', type=str, required=True, default='dev',
                           help="Dataset mode (dev or test)")
    args_parser.add_argument('--db_root_path', type=str, required=True, default='',
                           help="Path to database files")
    args_parser.add_argument('--num_cpus', type=int, default=1,
                           help="Number of CPUs to use for parallel execution")
    args_parser.add_argument('--meta_time_out', type=float, default=30.0,
                           help="Timeout for SQL execution in seconds")
    args_parser.add_argument('--mode_gt', type=str, default='gt',
                           help="Mode for ground truth SQL processing")
    args_parser.add_argument('--mode_predict', type=str, default='gpt',
                           help="Mode for predicted SQL processing")
    args_parser.add_argument('--difficulty',type=str,default='simple',
                           help="Difficulty level")
    args_parser.add_argument('--diff_json_path',type=str,default='',
                           help="Path to difficulty json file")
    args_parser.add_argument('--file_name', type=str, default=None,
                           help="Specific prediction file name to evaluate (e.g., 'groq.json', 'claude.json')")
    args_parser.add_argument('--start', type=int, default=None,
                           help="Start index for evaluation range (0-indexed)")
    args_parser.add_argument('--end', type=int, default=None,
                           help="End index for evaluation range (exclusive)")
    args = args_parser.parse_args()
    exec_result = [] 
    
    # Process predicted queries with optional file name
    pred_queries, db_paths = package_sqls(
        args.predicted_sql_path, 
        args.db_root_path, 
        mode=args.mode_predict,
        data_mode=args.data_mode,
        file_name=args.file_name
    )
    
    # Generate ground truth SQLs
    gt_queries, db_paths_gt = package_sqls(
        args.ground_truth_path, 
        args.db_root_path, 
        mode='gt',
        data_mode=args.data_mode
    )

    # Apply range filter if specified
    start_idx = args.start if args.start is not None else 0
    end_idx = args.end if args.end is not None else len(gt_queries)

    # Slice ground truth to match the range
    gt_queries = gt_queries[start_idx:end_idx]
    db_paths_gt = db_paths_gt[start_idx:end_idx]

    # Filter predicted queries to match the range (predicted file uses string indices)
    if args.start is not None or args.end is not None:
        filtered_pred = []
        filtered_db_paths = []
        for i in range(start_idx, min(end_idx, len(pred_queries))):
            filtered_pred.append(pred_queries[i])
            filtered_db_paths.append(db_paths[i])
        pred_queries = filtered_pred
        db_paths = filtered_db_paths
        print(f"Evaluating range: [{start_idx}, {end_idx})")

    # Verify we have matching number of queries
    if len(pred_queries) != len(gt_queries):
        print(f"Warning: Number of predicted queries ({len(pred_queries)}) doesn't match number of ground truth queries ({len(gt_queries)})")

    query_pairs = list(zip(pred_queries, gt_queries))
    run_sqls_parallel(query_pairs, db_places=db_paths, num_cpus=args.num_cpus, meta_time_out=args.meta_time_out)
    exec_result = sort_results(exec_result)
    
    print('Starting calculation...')
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = \
        compute_acc_by_diff(exec_result, args.diff_json_path, start_idx, end_idx)
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    print_data(score_lists, count_lists)
    print('===========================================================================================')
    print(f"Finished evaluation{' for ' + args.file_name if args.file_name else ''}")
    