def get_sentences_with_labels(sentences, raw_asessment_results, rounded_asessment_results):
    """Get sentences, each with its own raw label and rounded label

    Args:
        sentences (list): list of sentences
        raw_asessment_results (list): list of float numbers
        rounded_asessment_results (list): list of int numbers

    Returns:
        list: list of dicts, each with a sentence and its raw and rounded labels
    """
    sents_with_label = []
    for i in range(len(sentences)):
        sent = {
            "text": sentences[i],
            "raw_label": raw_asessment_results[i],
            "rounded_label": rounded_asessment_results[i],
        }
        sents_with_label.append(sent)
    return sents_with_label

def assess_against_threshold(raw_asessment_results, sentences, threshold):
    """Assess against a threshold and return assessment with individual sentences labeled

    Args:
        raw_asessment_results (list of list): list of labels the model predicts for each sentence
        sentences (list): list of sentences fed into BERT
        threshold (integer): threshold for a particular criteria

    Returns:
        dict: final result and sentences with labels for 1 criteria
    """
    raw_asessment_results = [res[0] for res in raw_asessment_results]
    rounded_asessment_results = [int(round(res)) for res in raw_asessment_results]
    final_result = 1 if rounded_asessment_results.count(1) >= threshold else 0

    return {
        "final_result": final_result,
        "sentences": get_sentences_with_labels(sentences, raw_asessment_results, rounded_asessment_results)
    }
