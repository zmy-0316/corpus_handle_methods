from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import json
import numpy as np
import argparse
import os

def load_data(args):
    with open(args.input_data_path, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    datas=[]
    for sample in texts:
        if all(field in sample for field in args.CONTENT_FIELD_NAME):
            combined_content = " ".join([sample[field] for field in args.CONTENT_FIELD_NAME])
            datas.append(combined_content)
        else:
            print(f"Warning: JSON object is missing one or more fields: {args.CONTENT_FIELD_NAME}")

    return datas,texts

def DBSCAN_SIM(args,datas,texts):
    vectorizer = TfidfVectorizer()
    scaler = StandardScaler()
    dbscan = DBSCAN(eps=args.eps, min_samples=args.min_samples)
    X = vectorizer.fit_transform(datas)
    X_scaled = scaler.fit_transform(X.toarray())
    clusters1 = dbscan.fit_predict(X_scaled)
    unique_values= np.unique(clusters1)

    if -1 not in unique_values:
        clusters = {i: [] for i in range(len(unique_values))}
    else:
        clusters = {i: [] for i in range(-1, len(unique_values) - 1, 1)}

    for data, cluster_id in zip(texts, clusters1):
        clusters[cluster_id].append(data)

    if args.record_mode:
        all_result = []
        for cluster_id, cluster_texts in clusters.items():
            result = {"cluster_id": cluster_id,
                      "datas": cluster_texts
                      }
            all_result.append(result)
        output_path=os.path.join(args.output_file,'dbscan_result.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_result, f, ensure_ascii=False, indent=4)
    else:
        for cluster, group in clusters.items():
            with open(f'{args.output_file}cluster_{cluster}.json', 'w',encoding="utf-8") as f:
                json.dump(group, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path',default='D:\\PYcharm_project\\pythonProject\\Corpus_Quality_Optimization\\corpus\\cot_data_cn.json',type=str, help="json file path for input data")
    parser.add_argument('--output_file', default="./DBSCAN_result/", type=str,help="file path for output data")
    parser.add_argument('--record_mode', default=True, type=bool,help="Whether to write the results to a json file")
    parser.add_argument('--CONTENT_FIELD_NAME', choices=["instruction", "input"], type=str,help='field name of content in json')
    parser.add_argument('--eps', default=0.5, type=float, help='DBSCAN parameters')
    parser.add_argument('--min_samples', default=2, type=int, help='DBSCAN parameters')
    args = parser.parse_args()
    datas,texts=load_data(args)
    DBSCAN_SIM(args,datas,texts)
