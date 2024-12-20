
import json
import sys
import numpy as np
from utils import compute_faithfulness_percentage_score, compute_completeness_percentage_score
from scipy.stats import pearsonr, spearmanr
import scipy.stats as ss

def main(frank_result_path, realsumm_result_path):

    # Load Frank data from the specified path
    frank_results = []
    for line in open(frank_result_path, 'r'):
        line = json.loads(line) # Parse each line in the Frank result file as JSON
        frank_results.append(line) # Append each parsed result to the list

    faithfulness_eval(frank_results)

    # Load RealSumm data from the specified path
    realsumm_results = []
    for line in open(realsumm_result_path, 'r'):
        line = json.loads(line) # Parse each line in the RealSumm result file as JSON
        realsumm_results.append(line) # Append each parsed result to the list
        
    completeness_and_conciseness_eval(realsumm_results)


def faithfulness_eval(results):
    '''
    A function to evaluate the results from FineSurE on faithfulness at the three different levels
    '''

    # Initialize dictionaries to store results for each model and for the overall evaluation
    model_wise_results = {}
    conv_ids = []
    task_keys = []
    summary_keys = []
    full_results = {
        # general factaulity (sentence-level)¸¸
        'gt_faithfulness_binary_labels': [],
        'pred_faithfulness_binary_labels': [],
        # general factaulity (summary-level)
        'gt_faithfulness_scores': [],
        'pred_faithfulness_scores': [],
    }

    cnt_success_inference = 0

    # Iterate through the results for each document
    for result in results:
        conv_id = result['doc_id']
        dataset_name = result['source']
        model = result['model']

        # Initialize model-wise result dictionary if not already done
        if model not in model_wise_results:
            model_wise_results[model] = {
                'gt_faithfulness_binary_labels': [],
                'pred_faithfulness_binary_labels': [],
                'gt_faithfulness_scores': [],
                'pred_faithfulness_scores': [],
            }

        # Append conversation ID concatenated with the model name for unique identification
        conv_ids.append(conv_id + model)

        # Extract ground truth and predicted faithfulness labels
        gt_faithfulness_binary_labels = get_aggregate_gt_labels(result['raw_annotations'], key="factuality_labels")
        pred_faithfulness_binary_labels = result['pred_faithfulness_labels']

        # Skip this result if there's a mismatch in the number of labels
        if len(gt_faithfulness_binary_labels) != len(pred_faithfulness_binary_labels):
            # failure cases
            continue
        
        # Filter out any invalid or 'None' labels from ground truth and predicted labels
        _gt_faithfulness_binary_labels, _pred_faithfulness_binary_labels = [], []
        for idx, item in enumerate(gt_faithfulness_binary_labels):
            # exception handler
            if item != 'None':
                _gt_faithfulness_binary_labels.append(gt_faithfulness_binary_labels[idx])
                _pred_faithfulness_binary_labels.append(pred_faithfulness_binary_labels[idx])
        
        # Convert the filtered labels to numpy arrays
        gt_faithfulness_binary_labels, pred_faithfulness_binary_labels = np.array(_gt_faithfulness_binary_labels), np.array(_pred_faithfulness_binary_labels)
        
        # Skip if there are no valid labels after filtering
        if len(gt_faithfulness_binary_labels) == 0:
            # failure cases
            continue

        key = dataset_name + '-' + conv_id + '-' + model
        for sentence_id in range(len(gt_faithfulness_binary_labels)):
            task_keys.append(key + '-' + str(sentence_id + 1)) # Create task-specific keys
        summary_keys.append(key)

        cnt_success_inference += 1
  
        #### compute summary-level
        gt_faithfulness_score = compute_faithfulness_percentage_score(gt_faithfulness_binary_labels) 
        pred_faithfulness_score = compute_faithfulness_percentage_score(pred_faithfulness_binary_labels) 

        # Store the results for the current document/model combination
        full_results['gt_faithfulness_binary_labels'].extend(gt_faithfulness_binary_labels)
        full_results['pred_faithfulness_binary_labels'].extend(pred_faithfulness_binary_labels)
        full_results['gt_faithfulness_scores'].append(gt_faithfulness_score)
        full_results['pred_faithfulness_scores'].append(pred_faithfulness_score)

        # Store the results in the model-wise dictionary
        model_wise_results[model]['gt_faithfulness_binary_labels'].extend(gt_faithfulness_binary_labels)
        model_wise_results[model]['pred_faithfulness_binary_labels'].extend(pred_faithfulness_binary_labels)
        model_wise_results[model]['gt_faithfulness_scores'].append(gt_faithfulness_score)
        model_wise_results[model]['pred_faithfulness_scores'].append(pred_faithfulness_score)

    print('[Faithfulness Evaluation]')

    print('* Sentence-level')
    bAcc = balancedAcc(full_results['gt_faithfulness_binary_labels'], full_results['pred_faithfulness_binary_labels'])
    print('\t-Balanced Accuracy:', '{:.1%}'.format(bAcc))

    # Compute and print Pearson and Spearman correlations for summary-level scores
    pearson_corr = pearsonr(full_results['gt_faithfulness_scores'], full_results['pred_faithfulness_scores'])
    spearman_corr = spearmanr(full_results['gt_faithfulness_scores'], full_results['pred_faithfulness_scores'])

    print('* Summary-level:')
    print("\t-Pearson:", pearson_corr)
    print("\t-Spearman:", spearman_corr)

    # System-level evaluation: Compute rank correlation between models
    print('* System-level:')
    # model-wise ranking 
    _rank_correlation = rank_correlation(model_wise_results, key="faithfulness_scores")
    print("\t-Rank Correlation:", _rank_correlation)

    # Compute and print success ratio
    success_rate = cnt_success_inference / len(conv_ids)
    print('* Success ratio', '{:.1%}'.format(success_rate))


def completeness_and_conciseness_eval(results):
    '''
    A function to evaluate the results from FineSurE on faithfulness at the three different levels
    '''

    # Initialize dictionaries to store results for each model and for the overall evaluation
    model_wise_results = {}
    conv_ids = []
    task_keys = []
    summary_keys = []
    full_results = {
        'gt_completeness_scores': [],
        'pred_completeness_scores': [],
        'gt_conciseness_scores': [],
        'pred_conciseness_scores': [],
    }

    cnt_success_inference = 0

    # Iterate through the results for each document
    for result in results:
        conv_id = result['doc_id']
        dataset_name = result['source']
        model = result['model']

        # Initialize model-wise result dictionary if not already done
        if model not in model_wise_results:
            model_wise_results[model] = {
                'gt_completeness_scores': [],
                'pred_completeness_scores': [],
                'gt_conciseness_scores': [],
                'pred_conciseness_scores': [],
            }
    
        conv_ids.append(conv_id + model)

        # Extract ground truth and predicted labels
        gt_alignment_labels = get_aggregate_gt_labels(result['raw_annotations'], key="key_fact_labels")
        gt_sentence_line_numbers = get_aggregate_gt_labels(result['raw_annotations'], key="sentence_labels")
        pred_alignment_labels = result['pred_alignment_labels']
        pred_sentence_line_numbers = result['pred_sentence_line_numbers']

        # Skip if the number of ground truth labels doesn't match the predicted labels
        if len(gt_alignment_labels) != len(pred_alignment_labels):
            continue

        # Filter out invalid or 'None' labels from ground truth and predicted alignment labels
        _gt_alignment_labels, _pred_alignment_labels = [], []
        for idx, item in enumerate(gt_alignment_labels):
            if item != 'None':
                _gt_alignment_labels.append(gt_alignment_labels[idx])
                _pred_alignment_labels.append(pred_alignment_labels[idx])
        gt_alignment_labels, pred_alignment_labels = np.array(_gt_alignment_labels), np.array(_pred_alignment_labels)

        # Generate a unique key for the current document, model, and dataset combination
        key = dataset_name + '-' + conv_id + '-' + model
        for key_fact_id in range(len(gt_alignment_labels)):
            task_keys.append(key + '-' + str(key_fact_id + 1))
        summary_keys.append(key)

        cnt_success_inference += 1

        # Compute completeness score as the percentage of key facts aligned
        gt_completeness_score = compute_completeness_percentage_score(gt_alignment_labels)
        pred_completeness_score = compute_completeness_percentage_score(pred_alignment_labels)

       # Compute conciseness score based on matching sentence line numbers
        _pred_sentence_line_numbers = []
        for idx in range(len(gt_sentence_line_numbers)):
            if (idx +1) in pred_sentence_line_numbers:
                _pred_sentence_line_numbers.append(1.0)
            else:
                _pred_sentence_line_numbers.append(0.0)
        pred_sentence_line_numbers = _pred_sentence_line_numbers

      
        assert len(gt_sentence_line_numbers) == len(pred_sentence_line_numbers)

        # Compute conciseness score as the proportion of sentences that align
        gt_conciseness_score = sum(gt_sentence_line_numbers) / len(gt_sentence_line_numbers)
        pred_conciseness_score = sum(pred_sentence_line_numbers) / len(pred_sentence_line_numbers)

        # Store the results for the current document/model combination
        full_results['gt_completeness_scores'].append(gt_completeness_score)
        full_results['pred_completeness_scores'].append(pred_completeness_score)
        full_results['gt_conciseness_scores'].append(gt_conciseness_score)
        full_results['pred_conciseness_scores'].append(pred_conciseness_score)

        # Store the results in the model-wise dictionary
        model_wise_results[model]['gt_completeness_scores'].append(gt_completeness_score)
        model_wise_results[model]['pred_completeness_scores'].append(pred_completeness_score)
        model_wise_results[model]['gt_conciseness_scores'].append(gt_conciseness_score)
        model_wise_results[model]['pred_conciseness_scores'].append(pred_conciseness_score)

    print('\n[Completeness Evaluation]')

     # Compute and print Pearson and Spearman correlations for completeness scores
    pearson_corr = pearsonr(full_results['gt_completeness_scores'], full_results['pred_completeness_scores'])
    spearman_corr = spearmanr(full_results['gt_completeness_scores'], full_results['pred_completeness_scores'])

    print('* Summary-level:')
    print("\t-Pearson:", pearson_corr)
    print("\t-Spearman:", spearman_corr)

    # System-level evaluation: Compute rank correlation between models
    print('* System-level:')
    # model-wise ranking 
    _rank_correlation = rank_correlation(model_wise_results, key="completeness_scores")
    print("\t-Rank Correlation:", _rank_correlation)

    print('\n[Conciseness Evaluation]')

    # Compute and print Pearson and Spearman correlations for conciseness scores
    pearson_corr = pearsonr(full_results['gt_conciseness_scores'], full_results['pred_conciseness_scores'])
    spearman_corr = spearmanr(full_results['gt_conciseness_scores'], full_results['pred_conciseness_scores'])

    print('* Summary-level:')
    print("\t-Pearson:", pearson_corr)
    print("\t-Spearman:", spearman_corr)

    # System-level evaluation: Compute rank correlation between models
    print('* System-level:')
    # model-wise ranking 
    _rank_correlation = rank_correlation(model_wise_results, key="conciseness_scores")
    print("\t-Rank Correlation:", _rank_correlation)

    # Compute and print success ratio
    success_rate = cnt_success_inference / len(conv_ids)
    print('\n* Success ratio', '{:.1%}'.format(success_rate))
  

def get_aggregate_gt_labels(raw_annotations, key):
    '''
    A function to generate the aggregated human labels from three annotators
    Args:
        - raw_annotations: the raw annotations from three annotators
        - key: the annotation type ('xxx:' )
    Returns:
        - final_labels: the aggregated labels by majority voting
    '''

    # if there are four annotators, we remove the last
    if key == "sentence_labels" and "3" in raw_annotations:
        del raw_annotations["3"]

    merged_gt_labels = []

    # Iterate over the raw annotations dictionary to extract the labels for the specified key
    for worker_id, annotation in raw_annotations.items():
        gt_labels = annotation[key]    
        merged_gt_labels.append(gt_labels)

    final_labels = [] # List to store the final aggregated labels
    merged_gt_labels = np.array(merged_gt_labels)
    num_labels = len(merged_gt_labels[-1])

    # Iterate over each sentence index to perform majority voting
    for sent_idx in range(num_labels):
        _column = merged_gt_labels[:, sent_idx]
        _column = [float(item) for item in _column if item != 'None']

        # If there's no valid label (all are 'None'), append 'None' to the final labels
        if len(_column) <= 1:
            final_labels.append('None')
        else:
            final_label = max(set(_column), key = _column.count)
            final_labels.append(float(final_label))

    assert len(final_labels) == num_labels

    return final_labels # Return the list of aggregated labels


def balancedAcc(gt, pred):
    '''
    A function to compute the balanced accuracy
    Args:
        - gt: ground truth labels
        - pred: predicted labels
    Return:
        - balanced accuracy
    '''
    ones, zeros = [], []

    # Iterate over the ground truth labels to separate predictions based on the ground truth
    for idx in range(len(gt)):
        if gt[idx] == 1.0:
            ones.append(pred[idx])
        elif gt[idx] == 0.0:
            zeros.append(pred[idx])

    # Calculate accuracy for the error class (ones)
    error_acc = sum(ones) / len(ones)

    # Calculate accuracy for the non-error class (zeros)
    non_error_acc =  1.0 - sum(zeros) / len(zeros)

    return (error_acc + non_error_acc) / 2.0


def rank_correlation(model_wise_results, key, min_number=5):
    '''
    A function to compute the balanced accuracy
    Args:
        - model_wise_results: evaluation results per model in dict
        - key: evaluation dimension
        - min_number: the minimum number of examples to be included in the evaluation
    Return:
        - rank correlation with p value
    '''

    model_list =  model_wise_results.keys()

    models = []
    gt_errors = []
    pred_errors = []

    # Iterate over each model and collect ground truth and predicted errors
    for model_name in model_list:
        models.append(model_name)

        # Compute the average of the ground truth and predicted errors for the current model
        gt_error, pred_error = np.mean(model_wise_results[model_name]['gt_' + key]), np.mean(model_wise_results[model_name]['pred_' + key])

        # Only include models with sufficient data (greater than or equal to `min_number` examples)
        if len(model_wise_results[model_name]['gt_' + key]) >= min_number:
            gt_errors.append(gt_error)
            pred_errors.append(pred_error)

    # Convert the errors to numpy arrays for easier manipulation
    pred_errors = np.array(pred_errors) 
    gt_errors = np.array(gt_errors) 

     # Compute the ranks of the predicted and ground truth errors
    estimated_rank = ss.rankdata(pred_errors)
    human_rank = ss.rankdata(gt_errors)
    #print("models:", models)
    #print('gt ' + key + ':', gt_errors)
    #print('pred ' + key + ':', pred_errors )
    #print('gt rank ' + key + ':', human_rank)
    #print('pred rank ' + key + ':', estimated_rank)

    # Calculate Spearman's rank correlation coefficient
    spearman_corr = spearmanr(estimated_rank, human_rank)

    return spearman_corr



if __name__ == "__main__":

    '''
    Runnining Command:
        cd CodeRelease/reproduce
        python reproduce-main-results.py results/frank-result-by-gpt4-w-finesure.json results/realsumm-result-by-gpt4-w-finesure.json
    '''

    frank_result_path = sys.argv[1]
    realsumm_result_path = sys.argv[2]
    
    main(frank_result_path, realsumm_result_path)


