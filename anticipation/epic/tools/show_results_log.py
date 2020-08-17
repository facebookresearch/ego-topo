import glob
import argparse
import os
import csv
import numpy as np
import json


def get_results(flog, epoch=None):
    with open(flog, "r") as f:
        rows = f.readlines()
       #  rows = json.loads(data)
    
    row = json.loads(rows[-1])
    # print(row)
    if row["mode"] == "val" and (epoch is None or row["epoch"] == epoch):
        print("{} on {} epoch".format(flog, row["epoch"]))
        return row
    else:
        print("{} no val {} epoch".format(flog, row["epoch"]) )
        return None

    # results = dict()
    # for row in rows:
    #     if "=" in row:
    #         name, num = row.split(" = ")
    #         results[name] = float(num)
        
    # return results


def simplify(names):
    if len(names) == 0:
        return []
    
    for i in range(len(os.path.dirname(names[0])) + 2):
        flag = True
        for j in range(1, len(names)):
            if names[j][i] != names[0][i]:
                flag = False
                break
        if not flag:
            break
    
    ret = [n[i:] for n in names]

    return ret


def filtering(flogs):
    return [flog for flog in flogs if "gen-lfb" not in os.path.dirname(flog)]


def filter_latest(flogs):
    dir_to_name = dict()
    for flog in flogs:
        d = os.path.dirname(flog)
        if d in dir_to_name:
            dir_to_name[d] = max(dir_to_name[d], flog)
        else:
            dir_to_name[d] = flog
    
    return [dir_to_name[k] for k in dir_to_name]


def aggregate_multi_run(results, names, fieldnames):
    aggre_results, aggre_names = [], []
    aggre = dict()
    for r, name in zip(results, names):
        n = name[name.find("/") + 1:]
        if n not in aggre:
            aggre[n] = []
        aggre[n].append(r)
    
    for k, runs in aggre.items():
        name = "avg {}: ".format(len(runs)) + k
        # avg, std = dict(), dict()
        result = dict() 
        for f in fieldnames:
            q = [r[f] for r in runs]
            result[f + " avg"] = np.mean(q)
            result[f + " std"] = np.std(q)
            # result[f + " max-min"] = np.max(q) - np.min(q)
        
        aggre_results.append(result)
        aggre_names.append("Runs {}: ".format(len(runs)) + k)
        # aggre_results.append(std)
        # aggre_names.append("std {}: ".format(len(runs)) + k)

    return aggre_results, aggre_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="show test results")
    parser.add_argument("--log-dir", type=str, nargs="+")
    parser.add_argument("--save-path", type=str, default="work_dir/results/output.csv")
    parser.add_argument("--multi-run", action='store_true')
    parser.add_argument("--epoch", type=int)

    args = parser.parse_args()

    flogs = []
    for log_dir in args.log_dir:
        flogs.extend(glob.glob("{}/**/*.log.json".format(log_dir), recursive=True))
    
    flogs = filter_latest(flogs)

    names = []
    results = []
    for flog in flogs:
        result = get_results(flog, args.epoch)
        if result is not None:
            names.append(os.path.dirname(flog))
            results.append(result)
    names = simplify(names)
    fieldnames = [key for key in results[0].keys() if "mAP" in key]
    if args.multi_run:
        results, names = aggregate_multi_run(results, names, fieldnames)
        # print(results, names)
        updated_fieldnames = []
        for mode in ["avg", "std"]:
            for r in fieldnames:
            # for mode in ["avg", "std"]: #["avg", "std", "max-min"]:
                updated_fieldnames.append(r + " " + mode)
        fieldnames = updated_fieldnames
    # fieldnames = ["model"] + list(results[0].keys())
    fieldnames = ["model"] + fieldnames
    for r, name in zip(results, names):
        r["model"] = name
    results = sorted(results, key=lambda x:x["model"])
    with open(args.save_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    for r in results:
        print(r)

        