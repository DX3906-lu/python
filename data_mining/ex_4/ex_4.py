import numpy as np
import pandas as pd
from itertools import combinations

def load_transactions(data_source="default"):
    if data_source == "default":
        transactions = [
            [1, 3, 4],
            [2, 3, 5],
            [1, 2, 3, 5],
            [2, 5]
        ]
    else:
        df = pd.read_csv(data_source, header=None)
        transactions = df.values.tolist()
    
    return [set(transaction) for transaction in transactions]

def create_single_item_candidates(transactions):
    single_items = set()
    for transaction in transactions:
        for item in transaction:
            single_items.add(frozenset([item])) 
    return list(single_items)

def calculate_support(candidates, transactions):
    support_dict = {}
    total_transactions = len(transactions)
    
    for candidate in candidates:
        count = 0
        for transaction in transactions:
            if candidate.issubset(transaction): 
                count += 1
        support = count / total_transactions 
        support_dict[candidate] = support
    
    return support_dict

def filter_frequent_itemsets(candidates, transactions, min_support):
    support_dict = calculate_support(candidates, transactions)
    frequent_itemsets = [itemset for itemset, support in support_dict.items() if support >= min_support]
    frequent_support_dict = {itemset: support for itemset, support in support_dict.items() if itemset in frequent_itemsets}
    return frequent_itemsets, frequent_support_dict

def generate_k_item_candidates(frequent_k_minus_1_itemsets, k):
    k_item_candidates = set()
    len_frequent = len(frequent_k_minus_1_itemsets)
    for i in range(len_frequent):
        for j in range(i + 1, len_frequent):
            itemset_i = sorted(list(frequent_k_minus_1_itemsets[i]))[:k-2]
            itemset_j = sorted(list(frequent_k_minus_1_itemsets[j]))[:k-2]
            
            if itemset_i == itemset_j:  
                merged_itemset = frequent_k_minus_1_itemsets[i] | frequent_k_minus_1_itemsets[j]
                if len(merged_itemset) == k:  
                    k_item_candidates.add(merged_itemset)
    
    return list(k_item_candidates)


def apriori(transactions, min_support=0.5):
    single_item_candidates = create_single_item_candidates(transactions)
    frequent_1_itemsets, support_dict = filter_frequent_itemsets(single_item_candidates, transactions, min_support)
    
    all_frequent_itemsets = [frequent_1_itemsets] 
    k = 2  
    
    while len(all_frequent_itemsets[k-2]) > 0:
        k_item_candidates = generate_k_item_candidates(all_frequent_itemsets[k-2], k)
        if not k_item_candidates:  
            break
        
        frequent_k_itemsets, k_support_dict = filter_frequent_itemsets(k_item_candidates, transactions, min_support)
        if frequent_k_itemsets: 
            all_frequent_itemsets.append(frequent_k_itemsets)
            support_dict.update(k_support_dict)  
        else:
            break
        
        k += 1 
    
    return all_frequent_itemsets, support_dict

def generate_all_non_empty_subsets(itemset):
    item_list = list(itemset)
    subsets = []
    for length in range(1, len(item_list)):
        for subset_tuple in combinations(item_list, length):
            subsets.append(frozenset(subset_tuple))
    return subsets

def generate_association_rules(all_frequent_itemsets, support_dict, min_confidence=0.7):
    association_rules = []
    for k in range(1, len(all_frequent_itemsets)):
        for freq_itemset in all_frequent_itemsets[k]:
            subsets = generate_all_non_empty_subsets(freq_itemset)
            for antecedent in subsets:
                consequent = freq_itemset - antecedent
                if not consequent:
                    continue
                freq_support = support_dict[freq_itemset]
                antecedent_support = support_dict[antecedent]
                confidence = freq_support / antecedent_support

                if confidence >= min_confidence:
                    association_rules.append((
                        set(antecedent), 
                        set(consequent),  
                        round(freq_support, 4), 
                        round(confidence, 4) 
                    ))
    
    return association_rules

def save_results(all_frequent_itemsets, association_rules, support_dict):
    frequent_items_list = []
    for k, itemsets in enumerate(all_frequent_itemsets, start=1):
        for itemset in itemsets:
            frequent_items_list.append({
                "项集长度": k,
                "频繁项集": str(set(itemset)),
                "支持度": support_dict[itemset]
            })
    frequent_df = pd.DataFrame(frequent_items_list)
    frequent_df.to_csv("./data_mining/ex_4/output/frequent_itemsets.csv", index=False, encoding="utf-8-sig")
    print("频繁项集已保存到 frequent_itemsets.csv")
    
    # 保存关联规则
    rules_list = []
    for rule in association_rules:
        antecedent, consequent, support, confidence = rule
        rules_list.append({
            "前件": str(antecedent),
            "后件": str(consequent),
            "支持度": support,
            "置信度": confidence
        })
    rules_df = pd.DataFrame(rules_list)
    rules_df.to_csv("./data_mining/ex_4/output/association_rules.csv", index=False, encoding="utf-8-sig")
    print("关联规则已保存到 association_rules.csv")

if __name__ == "__main__":
    transactions = load_transactions(data_source="default")
    print("加载的交易数据：")
    for i, trans in enumerate(transactions, start=1):
        print(f"交易{i}: {trans}")
    print(f"总交易数：{len(transactions)}")
    print("-" * 50)
    
    min_support = 0.5
    all_frequent_itemsets, support_dict = apriori(transactions, min_support=min_support)
    
    print(f"最小支持度={min_support} 下的频繁项集：")
    for k, itemsets in enumerate(all_frequent_itemsets, start=1):
        print(f"\n{k}-项频繁项集：")
        for itemset in itemsets:
            print(f"项集：{set(itemset)}, 支持度：{support_dict[itemset]:.4f}")
    print("-" * 50)
    
    min_confidence = 0.7
    association_rules = generate_association_rules(all_frequent_itemsets, support_dict, min_confidence=min_confidence)
    
    print(f"最小置信度={min_confidence} 下的关联规则：")
    if association_rules:
        for i, rule in enumerate(association_rules, start=1):
            antecedent, consequent, support, confidence = rule
            print(f"规则{i}: {antecedent} → {consequent}, 支持度：{support}, 置信度：{confidence}")
    else:
        print("未找到满足最小置信度的关联规则")
    print("-" * 50)
    
    save_results(all_frequent_itemsets, association_rules, support_dict)