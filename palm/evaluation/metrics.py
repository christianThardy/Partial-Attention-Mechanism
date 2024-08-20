from rouge import Rouge
from bert_score import score


def calculate_rouge(reference, hypothesis):
    # Initialize ROUGE metric object
    rouge = Rouge()

    # Calculate ROUGE scores between the hypothesis and reference texts
    scores = rouge.get_scores(hypothesis, reference)[0]

    # Return F1 score of the ROUGE-L metric (longest common subsequence)
    return scores['rouge-l']['f']


def calculate_bert_score(reference, hypothesis):
    # Calculate BERTScore F1 score between the hypothesis and reference texts
    _, _, f1 = score([hypothesis], [reference], lang="en")
    # Return F1 score as a scalar value
    return f1.item()


def evaluate_generations(reference, palm_output, baseline_output):
    # Calculate ROUGE-L and BERT scores for the PALM model output
    palm_rouge = calculate_rouge(reference, palm_output)
    baseline_rouge = calculate_rouge(reference, baseline_output)
    
    palm_bert_score = calculate_bert_score(reference, palm_output)
    baseline_bert_score = calculate_bert_score(reference, baseline_output)
    
    # Return a dictionary containing evaluation scores for both models
    return {
        "palm_rouge": palm_rouge,
        "baseline_rouge": baseline_rouge,
        "palm_bert_score": palm_bert_score,
        "baseline_bert_score": baseline_bert_score
    }


def evaluate_information_extraction(true_info, palm_output, baseline_output):
    # Helper function to extract answers from the model output text
    def extract_answers(output):
        lines = output.split('\n')
        answers = [line.split(': ', 1)[1] if ': ' in line else '' for line in lines if line.strip().startswith(('1.', '2.', '3.', '4.', '5.'))]
        return answers

    # Extract answers from PALM and baseline outputs
    palm_answers = extract_answers(palm_output)
    baseline_answers = extract_answers(baseline_output)

    # Calculate the number of correct answers for PALM and baseline outputs
    palm_correct = sum(1 for true, pred in zip(true_info, palm_answers) if true.lower() in pred.lower())
    baseline_correct = sum(1 for true, pred in zip(true_info, baseline_answers) if true.lower() in pred.lower())

    # Return a dictionary containing accuracy metrics for both models
    return {
        "palm_accuracy": palm_correct / len(true_info),
        "baseline_accuracy": baseline_correct / len(true_info)
    }
