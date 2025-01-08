import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from rapidfuzz.distance import Levenshtein
from copy import deepcopy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef
from sklearn.model_selection import KFold


##############################################################################
# 1) Load config from a JSON file
##############################################################################
def load_config_from_json(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

##############################################################################
# 2a) Repeated-Merge Handling with merges_dict
##############################################################################
def resolve_intermediate_label(label: str, merges_dict: Dict[str, str]) -> str:
    """
    Follows merges_dict to find the final category label for 'label'.
    If 'label' was merged multiple times, we chase down the chain.
    """
    while label in merges_dict:
        label = merges_dict[label]
    return label


##############################################################################
# 2b) Optional Category-Merging Logic
##############################################################################
def compute_intra_category_similarities(activities: pd.DataFrame, vectorizer: TfidfVectorizer) -> Dict[str, float]:
    """
    Compute the average intra-category similarity for each category (TF-IDF + cosine).
    For single-entry categories, returns NaN.
    """
    intra_category_similarities = {}
    for category, group in activities.groupby('category'):
        activity_matrix = vectorizer.fit_transform(group['activity_name'])
        if activity_matrix.shape[0] > 1:
            sims = cosine_similarity(activity_matrix)
            tri_upper = sims[np.triu_indices_from(sims, k=1)]
            intra_similarity = np.mean(tri_upper)
        else:
            intra_similarity = np.nan
        intra_category_similarities[category] = intra_similarity
    return intra_category_similarities


def merge_categories(
    activities: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    intra_category_similarities: Dict[str, float],
    merges_dict: Dict[str, str],
    similarity_threshold: float = 0.2,
    size_threshold: int = 100
) -> List[dict]:
    """
    Identify pairs of categories to merge based on:
      - Pairwise similarity of merged activity descriptions (TF-IDF & cosine).
      - Intra-category similarity checks.
      - Category size checks.
    Also updates merges_dict so we can track intermediate merges.
    """
    category_representation = activities.groupby('category')['activity_name'].apply(lambda x: ' '.join(x))
    category_sizes = activities.groupby('category').size()

    cat_matrix = vectorizer.fit_transform(category_representation)
    sims = cosine_similarity(cat_matrix)
    categories = category_representation.index.tolist()

    merge_candidates = []
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            cat_i, cat_j = categories[i], categories[j]
            similarity_score = sims[i, j]
            intra_sim_1 = intra_category_similarities.get(cat_i, np.nan)
            intra_sim_2 = intra_category_similarities.get(cat_j, np.nan)
            size_1 = category_sizes[cat_i]
            size_2 = category_sizes[cat_j]

            if (
                (similarity_score > similarity_threshold and (np.isnan(intra_sim_1) or np.isnan(intra_sim_2)))
                or
                (similarity_score > intra_sim_1 and similarity_score > intra_sim_2
                 and (size_1 < size_threshold or size_2 < size_threshold))
            ):
                merge_candidates.append({
                    'category_1': cat_i,
                    'category_2': cat_j,
                    'similarity_score': similarity_score,
                    'intra_similarity_category_1': intra_sim_1,
                    'intra_similarity_category_2': intra_sim_2,
                    'size_category_1': size_1,
                    'size_category_2': size_2
                })

    merge_candidates = sorted(merge_candidates, key=lambda x: x['similarity_score'], reverse=True)
    return merge_candidates


def iterative_merge_categories(
    activities: pd.DataFrame,
    merges_dict: Dict[str, str],
    similarity_threshold: float = 0.2,
    size_threshold: int = 100
) -> pd.DataFrame:
    """
    Iteratively merges categories until no suitable merge candidate remains.
    We store merges in merges_dict so we can resolve intermediate merges later.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    while True:
        intra_cat_sims = compute_intra_category_similarities(activities, vectorizer)
        merge_candidates = merge_categories(
            activities, vectorizer, intra_cat_sims, merges_dict,
            similarity_threshold=similarity_threshold,
            size_threshold=size_threshold
        )
        if not merge_candidates:
            break

        best_candidate = merge_candidates[0]
        cat1, cat2 = best_candidate['category_1'], best_candidate['category_2']
        new_category = f"{cat1}/{cat2}"
        # Update merges_dict for the old labels
        merges_dict[cat1] = new_category
        merges_dict[cat2] = new_category

        # If new_category itself is later merged, we'll set merges_dict[new_category] too.
        activities.loc[activities['category'] == cat1, 'category'] = new_category
        activities.loc[activities['category'] == cat2, 'category'] = new_category
        print(f"Merged: {cat1} and {cat2} into {new_category}")

    return activities


##############################################################################
# 3) Preprocessing & Similarity
##############################################################################
def preprocess_text(text: str) -> str:
    return text.lower().strip()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.map(preprocess_text)

def jaccard_similarity(str1: str, str2: str) -> float:
    set1 = set(str1.split())
    set2 = set(str2.split())
    union = set1 | set2
    if not union:
        return 1.0
    intersection = set1 & set2
    return len(intersection) / len(union)

def levenshtein_similarity(str1: str, str2: str) -> float:
    distance = Levenshtein.distance(str1, str2)
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance / max_len)

def compute_similarity(
    train_items: List[str],
    input_items: List[str],
    vectorizer_name: str,
    similarity_type: str
) -> np.ndarray:
    vectorizers = {
        'tfidf': TfidfVectorizer(stop_words='english'),
        'count': CountVectorizer(stop_words='english'),
        'hashing': HashingVectorizer(n_features=2**12, alternate_sign=False)
    }
    if vectorizer_name not in vectorizers:
        raise ValueError(f"Vectorizer '{vectorizer_name}' is not supported.")
    vectorizer = vectorizers[vectorizer_name]

    train_matrix = vectorizer.fit_transform(train_items)
    input_matrix = vectorizer.transform(input_items)

    if similarity_type == 'cosine':
        return cosine_similarity(input_matrix, train_matrix)

    elif similarity_type == 'jaccard':
        return np.array([
            [jaccard_similarity(inp, t) for t in train_items]
            for inp in input_items
        ])

    elif similarity_type == 'levenshtein':
        return np.array([
            [levenshtein_similarity(inp, t) for t in train_items]
            for inp in input_items
        ])

    else:
        raise ValueError(f"Similarity type '{similarity_type}' is not supported.")


##############################################################################
# 4) Handling Unknown Predicted Labels in Metric Computation
##############################################################################
def filter_unseen_labels(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Remove rows where y_pred is not in y_true to avoid the scikit-learn warning
    and exclude those from metric calculations.
    """
    valid_labels = set(y_true)
    mask = np.array([pred in valid_labels for pred in y_pred], dtype=bool)
    return y_true[mask], y_pred[mask]


def compute_performance_metric(y_true, y_pred, metric_name: str) -> float:
    """
    Filters out predictions where y_pred is not in y_true,
    then computes balanced_accuracy, kappa, or mcc.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true_f, y_pred_f = filter_unseen_labels(y_true, y_pred)
    if len(y_true_f) == 0:
        return float('nan')

    if metric_name == 'balanced_accuracy':
        return balanced_accuracy_score(y_true_f, y_pred_f)
    elif metric_name == 'kappa':
        return cohen_kappa_score(y_true_f, y_pred_f)
    elif metric_name == 'mcc':
        return matthews_corrcoef(y_true_f, y_pred_f)
    else:
        raise ValueError(f"Unsupported metric_name: {metric_name}")


def compute_all_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Also excludes unknown predictions before computing the three metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true_f, y_pred_f = filter_unseen_labels(y_true, y_pred)
    if len(y_true_f) == 0:
        return {
            'balanced_accuracy': float('nan'),
            'kappa': float('nan'),
            'mcc': float('nan')
        }

    return {
        'balanced_accuracy': balanced_accuracy_score(y_true_f, y_pred_f),
        'kappa': cohen_kappa_score(y_true_f, y_pred_f),
        'mcc': matthews_corrcoef(y_true_f, y_pred_f)
    }


##############################################################################
# 5) evaluate_ensemble, compute_predictions_for_input, etc.
##############################################################################
def evaluate_ensemble(
    precomputed_results: Dict,
    train_set: pd.DataFrame,
    input_items: List[str],
    vectorizer_names: List[str],
    similarity_types: List[str],
    ensemble_method: str,
    model_types: List[str]
) -> pd.DataFrame:
    activity_names = train_set['activity_name'].tolist()
    categories = train_set['category'].tolist()

    similarity_matrices = [
        precomputed_results[(v, s, m)]
        for v in vectorizer_names
        for s in similarity_types
        for m in model_types
    ]
    if len(similarity_matrices) == 1:
        ensemble_method = 'mean'

    if ensemble_method == 'voting':
        votes = np.zeros(similarity_matrices[0].shape, dtype=int)
        for mat in similarity_matrices:
            row_argmax = np.argmax(mat, axis=1)
            for i, idx in enumerate(row_argmax):
                votes[i, idx] += 1
        ensemble_matrix = votes / len(similarity_matrices)
    else:
        # 'mean'
        if len(similarity_matrices) == 1:
            ensemble_matrix = similarity_matrices[0]
        else:
            ensemble_matrix = sum(similarity_matrices) / len(similarity_matrices)

    best_indices = np.argmax(ensemble_matrix, axis=1)
    predictions = pd.DataFrame({
        'item': input_items,
        'predicted_activity': [activity_names[idx] for idx in best_indices],
        'predicted_category': [categories[idx] for idx in best_indices],
        'similarity_score': np.max(ensemble_matrix, axis=1)
    })
    return predictions


def compute_predictions_for_input(
    train_set: pd.DataFrame,
    input_items: List[str],
    best_config: Dict
) -> pd.DataFrame:
    train_items_activity = train_set['activity_name'].tolist()
    train_items_category = train_set['category'].tolist()
    train_items_original_category = train_set['original_category'].tolist()

    # Recompute similarity with the best config
    similarity_matrices = []
    for vec in best_config['vectorizers']:
        for sim in best_config['similarities']:
            for mt in best_config['model_types']:
                train_items = train_items_activity if mt == 'activity' else train_items_category
                mat = compute_similarity(train_items, input_items, vec, sim)
                similarity_matrices.append(mat)

    if len(similarity_matrices) == 1:
        best_config['ensemble'] = 'mean'  # override if just 1 matrix

    if best_config['ensemble'] == 'voting':
        votes = np.zeros(similarity_matrices[0].shape, dtype=int)
        for mat in similarity_matrices:
            row_argmax = np.argmax(mat, axis=1)
            for i, idx in enumerate(row_argmax):
                votes[i, idx] += 1
        ensemble_matrix = votes / len(similarity_matrices)
    else:
        if len(similarity_matrices) == 1:
            ensemble_matrix = similarity_matrices[0]
        else:
            ensemble_matrix = sum(similarity_matrices) / len(similarity_matrices)

    best_indices = np.argmax(ensemble_matrix, axis=1)
    predictions = pd.DataFrame({
        'item': input_items,
        'predicted_activity': [train_items_activity[idx] for idx in best_indices],
        'predicted_category': [train_items_original_category[idx] for idx in best_indices], #Use the original categories
        'similarity_score': np.max(ensemble_matrix, axis=1)
    })
    return predictions


def precompute_similarities(
    train_set: pd.DataFrame,
    input_items: List[str],
    vectorizer_options: List[str],
    similarity_options: List[str],
    model_type_options: List[str]
) -> Dict:
    results = {}
    for vec in vectorizer_options:
        for sim in similarity_options:
            for mt in model_type_options:
                if mt == 'activity':
                    train_items = train_set['activity_name'].tolist()
                else:
                    train_items = train_set['category'].tolist()
                print(f"Precomputing: {vec}, {sim}, {mt}")
                sim_mat = compute_similarity(train_items, input_items, vec, sim)
                results[(vec, sim, mt)] = sim_mat
    return results

##############################################################################
# 6) Grid Search with 4-Fold CV
##############################################################################

def grid_search_best_model_cv(
    activities: pd.DataFrame,
    input_items: List[str],
    vectorizer_options: List[str],
    similarity_options: List[str],
    model_type_options: List[str],
    ensemble_options: List[str],
    performance_metric: str = 'balanced_accuracy',
    n_splits: int = 4,
    random_state: int = 42
) -> Tuple[Dict, float, pd.DataFrame, Dict[str, float]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    best_config = None
    best_main_score = -np.inf
    best_other_metrics = {'balanced_accuracy': -np.inf, 'kappa': -np.inf, 'mcc': -np.inf}

    for v in vectorizer_options:
        for s in similarity_options:
            for e in ensemble_options:
                for m in model_type_options:
                    fold_main_scores = []
                    fold_bal = []
                    fold_kap = []
                    fold_mcc = []

                    for train_idx, val_idx in kf.split(activities):
                        train_set = activities.iloc[train_idx].copy()
                        val_set = activities.iloc[val_idx].copy()

                        # Precompute
                        precomp = precompute_similarities(
                            train_set,
                            val_set['activity_name'].tolist(),
                            [v],
                            [s],
                            [m]
                        )
                        preds = evaluate_ensemble(
                            precomp,
                            train_set,
                            val_set['activity_name'].tolist(),
                            [v],
                            [s],
                            e,
                            [m]
                        )

                        y_true = val_set['category'].values
                        y_pred = preds['predicted_category'].values

                        main_score = compute_performance_metric(y_true, y_pred, performance_metric)
                        fold_main_scores.append(main_score)

                        all_m = compute_all_metrics(y_true, y_pred)
                        fold_bal.append(all_m['balanced_accuracy'])
                        fold_kap.append(all_m['kappa'])
                        fold_mcc.append(all_m['mcc'])

                    avg_main = np.mean(fold_main_scores)
                    avg_bal = np.mean(fold_bal)
                    avg_kap = np.mean(fold_kap)
                    avg_mcc = np.mean(fold_mcc)

                    if avg_main > best_main_score:
                        best_main_score = avg_main
                        best_config = {
                            'vectorizers': [v],
                            'similarities': [s],
                            'ensemble': e,
                            'model_types': [m]
                        }
                        best_other_metrics = {
                            'balanced_accuracy': avg_bal,
                            'kappa': avg_kap,
                            'mcc': avg_mcc
                        }

    return best_config, best_main_score, best_other_metrics

##############################################################################
# 7) Plotting
##############################################################################
def plot_category_distribution(activities: pd.DataFrame, title: str, save_path: str = None):
    plt.figure(figsize=(25, 12))
    counts = activities['category'].value_counts()
    counts.plot(kind='bar')
    plt.title(title)
    plt.ylabel("Count")
    plt.xlabel("Category")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.5)
    if save_path:
        plt.savefig(save_path)
    plt.close()

##############################################################################
# 8) Main Execution
##############################################################################
if __name__ == "__main__":
    # 1) Load config
    config = load_config_from_json("config.json")

    output_file_name = config.get("output_file_name", "best_predictions.csv")
    merge_categories_option = config["merge_categories_option"]
    group_small_categories = config["group_small_categories"]
    size_cutoff_for_other = config["size_cutoff_for_other"]
    use_prespecified_config = config["use_prespecified_config"]
    performance_metric = config["performance_metric"]

    activities_csv = config["activities_csv"]
    input_data_csv = config["input_data_csv"]

    prespecified_config = config["prespecified_config"]

    vectorizer_options = config["vectorizer_options"]
    similarity_options = config["similarity_options"]
    model_type_options = config["model_type_options"]
    ensemble_options = config["ensemble_options"]

    # 2) Read data
    activities = pd.read_csv(activities_csv)
    input_data = pd.read_csv(input_data_csv)

    # 3) Preprocess
    activities = preprocess_data(activities)
    input_data = preprocess_data(input_data)
    activities['original_category'] = deepcopy(activities['category'])
    
    # 4) Plot distribution BEFORE merges
    plot_category_distribution(activities, "Category Distribution (Before Merging)", "category_distribution_before.png")

    # 5) Optionally do merges with merges_dict to track repeated merges
    merges_dict = {}
    if merge_categories_option:
        activities = iterative_merge_categories(activities, merges_dict)

    # 6) Group small categories
    if group_small_categories:
        counts = activities['category'].value_counts()
        small_cats = counts[counts < size_cutoff_for_other].index
        if len(small_cats) > 0:
            activities.loc[activities['category'].isin(small_cats), 'category'] = 'other'

    # 7) Save final merged dataset
    activities.to_csv("final_merged_activities.csv", index=False)

    # 8) Plot distribution AFTER merges
    if merge_categories_option or group_small_categories:
        plot_category_distribution(activities, "Category Distribution (After Merging/Grouping)", "category_distribution_after.png")

    # 9) Build final->original map
    final_to_original_map = (
        activities.groupby('category')['original_category']
        .apply(lambda grp: sorted(set(grp)))
        .to_dict()
    )

    # 10) Either prespecified config or do 4-fold CV
    if use_prespecified_config:
        best_config = prespecified_config
        best_score = np.nan
        best_other_metrics = {'balanced_accuracy': np.nan, 'kappa': np.nan, 'mcc': np.nan}
    else:
        best_config, best_score, best_other_metrics = grid_search_best_model_cv(
            activities,
            input_data['item'].tolist(),
            vectorizer_options,
            similarity_options,
            model_type_options,
            ensemble_options,
            performance_metric=performance_metric,
            n_splits=4,
            random_state=42
        )
            # Print results
        print(
            f"Best Configuration: {best_config} -> "
            f"Tuning metric={performance_metric}={best_score:.4f}, "
            "Other metrics="
            f"(balanced_accuracy={best_other_metrics['balanced_accuracy']:.4f}, "
            f"kappa={best_other_metrics['kappa']:.4f}, "
            f"mcc={best_other_metrics['mcc']:.4f})"
        )

    predictions = compute_predictions_for_input(activities, input_data['item'].tolist(), best_config)


    # 13) Save predictions
    predictions.to_csv(output_file_name, index=False)
    print(f"Predictions saved to {output_file_name}")
