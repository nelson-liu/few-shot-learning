import argparse
import numpy as np
import json
from collections import Counter
from tqdm import tqdm
from transformers import GPT2Tokenizer
from utils import complete_gpt3, setup_gpt3
import random

def main(model, train_data_path, test_data_path, seed, shots, batch_size, estimate_num_tokens, output_path):
    # Load the train data
    train_instances = []
    num_questions = 0
    print(f"Loading train data from {train_data_path}")
    with open(train_data_path) as f:
        train_data = json.load(f)
        for article in train_data["data"]:
            title = article["title"]
            for paragraph in article["paragraphs"]:
                instance = {}
                context = " ".join(paragraph["context"].split())
                instance["title"] = title
                instance["context"] = context
                instance["qas"] = []
                for qas in paragraph["qas"]:
                    num_questions += 1
                    instance["qas"].append({
                        "question": " ".join(qas["question"].split()),
                        "answer": Counter([answer["text"] for answer in qas["answers"]]).most_common(1)[0][0],
                        "qid": qas["id"]
                    })
                train_instances.append(instance)

    assert shots <= num_questions

    # Load the test data
    test_instances = []
    print(f"Loading test data from {test_data_path}")
    with open(test_data_path) as f:
        test_data = json.load(f)
        for article in test_data["data"]:
            title = article["title"]
            for paragraph in article["paragraphs"]:
                instance = {}
                context = " ".join(paragraph["context"].split())
                instance["title"] = title
                instance["context"] = context
                instance["qas"] = []
                for qas in paragraph["qas"]:
                    instance["qas"].append({
                        "question": " ".join(qas["question"].split()),
                        "answer": Counter([answer["text"] for answer in qas["answers"]]).most_common(1)[0][0],
                        "qid": qas["id"]
                    })
                test_instances.append(instance)

    # Set the seed
    np.random.seed(seed)
    random.seed(seed)

    # Get instances for training
    in_context_instances = []
    while sum([len(instance["qas"]) for instance in in_context_instances]) != shots:
        # Select a random instance
        current_instance = random.choice(train_instances)

        # Filter out questions that are already in the in_context_instances
        in_context_qids = set()
        for instance in in_context_instances:
            in_context_qids.update([qa["qid"] for qa in instance["qas"]])
        current_instance["qas"] = [qa for qa in current_instance["qas"] if qa["qid"] not in in_context_qids]

        # if this current instance has more questions than we need, subsample the questions
        num_questions_needed = shots - sum([len(instance["qas"]) for instance in in_context_instances])
        if num_questions_needed < len(current_instance["qas"]):
            current_instance["qas"] = random.sample(current_instance["qas"], k=num_questions_needed)
        in_context_instances.append(current_instance)

    assert sum([len(instance["qas"]) for instance in in_context_instances]) == shots
    print(f"Got {sum([len(instance['qas']) for instance in in_context_instances])} instances for prompting")

    # Construct the prompt prefix for any in-context examples, if applicable
    if in_context_instances:
        instances_to_prompt_strings = []
        for instance in in_context_instances:
            title = instance["title"]
            context = instance["context"]
            qas_as_strings = []
            for qa in instance["qas"]:
                question = qa["question"]
                answer = qa["answer"]
                qas_as_strings.append(f"Q: {question}\n\nA: {answer}")
            qas_string = "\n".join(qas_as_strings)
            instances_to_prompt_strings.append(f"Title: {title}\nBackground: {context}\n\n"
                                               f"{qas_string}\n\n")
        prompt_prefix = "".join(instances_to_prompt_strings)
    else:
        prompt_prefix = ""

    # Construct the prompts
    prompts = []
    prompt_qids = []
    if estimate_num_tokens:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        num_tokens = 0
    print("Building prompts")
    for instance in tqdm(test_instances):
        title = instance["title"]
        context = instance["context"]
        for qa in instance["qas"]:
            question = qa["question"]
            prompt_qids.append(qa["qid"])
            prompt = (prompt_prefix +
                      f"Title: {title}\nBackground: {context}\n\n" +
                      f"Q: {question}\n\nA:")
            if estimate_num_tokens:
                num_tokens += len(tokenizer.encode_plus(prompt)["input_ids"])
            prompts.append(prompt)

    if estimate_num_tokens:
        print(f"Prompt total GPT-2 token count: {num_tokens}")
        return

    # Get the responses from the API
    print(f"getting raw resp for {len(prompts)} prompts")
    raw_resp_test = get_model_response(prompts, model, batch_size, num_tokens_to_predict=20)

    # Write the output to output_path
    assert len(prompts) == len(prompt_qids)
    answers = {}
    for qid, response in zip(prompt_qids, raw_resp_test):
        answers[qid] = response["text"]
    with open(output_path, "w") as f:
        json.dump(answers, f)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_model_response(prompts, model, batch_size, num_tokens_to_predict):
    setup_gpt3()
    all_raw_answers = []

    print("First 3 prompts:")
    print(prompts[:3])

    chunked_prompts = list(chunks(prompts, batch_size))
    for chunk_id, test_chunk_prompts in tqdm(enumerate(chunked_prompts)):
        resp = complete_gpt3(test_chunk_prompts,
                             l=num_tokens_to_predict,
                             model_name=model,
                             temp=0)
        for answer in resp['choices']:
            all_raw_answers.append(answer)
    return all_raw_answers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--model', type=str, required=True, help='name of model, e.g., davinci')
    parser.add_argument('--train-data-path', type=str, help='Path to training data')
    parser.add_argument('--test-data-path', type=str, help='Path to test data')
    parser.add_argument('--seed', type=int, required=True, help='Seeds for the training set')
    parser.add_argument('--shots', type=int, required=True, help='Num training examples to use')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size for model queries.')
    parser.add_argument('--estimate-num-tokens', action="store_true", help='Try to estimate the number of tokens to use')
    parser.add_argument('--output-path', type=str,
                        help='Write predictions to this path.')
    args = parser.parse_args()
    main(args.model, args.train_data_path, args.test_data_path, args.seed, args.shots, args.batch_size, args.estimate_num_tokens, args.output_path)
