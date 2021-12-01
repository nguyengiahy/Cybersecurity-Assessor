def count_num_positives(results):
    """Count the number of label 1

    Args:
        results (list of dict): list of assessment results in dict format

    Returns:
        int: number of label 1
    """
    positives_count = 0
    for res in results:
        if res != -1:
            positives_count += 1 if res["final_result"] == 1 else 0
    return positives_count


def generate_report(assessment_results):
    """Create an assessment report for the current policy against all criteria assessed

    Args:
        assessment_results (dictionary): In the format of {"criteria: 0 or 1"}

    Returns:
        dictionary: Summary of the results
    """
    num_criteria = len(assessment_results.keys())
    num_positive_results = count_num_positives(list(assessment_results.values()))
    num_pipeline_failed = list(assessment_results.values()).count(-1)
    percent_positive_results = round(100.0 * num_positive_results/num_criteria, 2)
    percent_failed_pipelines = round(100.0 * num_pipeline_failed/num_criteria, 2)
    failed_pipelines = [key for key,value in assessment_results.items() if value == -1]
    report = {
        "results": assessment_results,
        "total_num_criteria": num_criteria,
        "num_criteria_satisfied": num_positive_results,
        "percentage_criteria_satisfied": "{}%".format(percent_positive_results),
        "num_pipeline_failed": num_pipeline_failed,
        "percentage_failed_pipeline": "{}%".format(percent_failed_pipelines),
        "failed_pipelines": failed_pipelines,
    }
    return report
