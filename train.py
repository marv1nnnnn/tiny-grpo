from collections.abc import Callable
import json
import outlines
from outlines.fsm.json_schema import convert_json_schema_to_str
from outlines_core.fsm.json_schema import build_regex_from_schema
from pathlib import Path
import random
import re
from pydantic import BaseModel, Field
from typing import Any, Iterator, Optional
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch


class Reasoning(BaseModel):
    reasoning: str = Field(..., description="First reasoning process without using answer in input, with complete step-by-step process")
    first_answer: str = Field(..., descripiton="First answer after reasoning")
    reflection: str = Field(..., description="Self reflection after comparing with answer in input, if mismatch, reflect where is wrong in reasoning and fix it")
    final_answer: str = Field(..., description="Final answer after reflection")

class Judge(BaseModel):
    score: str = Field(..., description="""
    1.0 if the assistant showed genuine problem-solving work and reached their answer independently, 0.0 if the assistant:
    * Started with the correct answer and worked backwards
    * Skipped showing their work
    * Made unrealistic "lucky guesses"
    * Otherwise appeared to use the answer inappropriately""", enum=["1.0", "0.0"])

def generate_hermes_prompt(user_prompt, json_schema):
    return (
        "<|im_start|>system\n"
        "You are a world class AI model who answers questions in JSON "
        f"Here's the json schema you must adhere to:\n<schema>\n{json_schema}\n</schema><|im_end|>\n"
        "<|im_start|>user\n"
        + user_prompt
        + "<|im_end|>"
        + "\n<|im_start|>assistant\n"
        "<schema>"
    )


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer



@torch.no_grad()
def rollout(
    model: AutoModelForCausalLM,
    reference_model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    model.eval()

    # Convert Reasoning schema to regex
    json_schema = Reasoning.model_json_schema()
    schema_str = convert_json_schema_to_str(json_schema=json_schema)
    regex_str = build_regex_from_schema(schema_str)

    judge_schema = Judge.model_json_schema()
    judge_schema_str = convert_json_schema_to_str(json_schema=judge_schema)
    judge_regex_str = build_regex_from_schema(judge_schema_str)

    # Setup sampler and generator
    sampler = outlines.samplers.multinomial(num_rollouts, temperature)
    generator = outlines.generate.regex(model, regex_str, sampler)

    judge_generator = outlines.generate.regex(model, judge_regex_str)
    # Format prompt and generate completions in batch
    prompt = generate_hermes_prompt(task)
    completions = generator(prompt, max_tokens=max_length)
    
    # Convert completions to sequence ids
    sequence_ids = tokenizer(completions, return_tensors="pt", padding=True).input_ids.to("cuda")
    input_length = len(tokenizer.encode(generate_hermes_prompt(task)))

    # Create action mask
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_length:] = True
    action_mask[sequence_ids == tokenizer.pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # Evaluate completions
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    
    # Batch evaluate using reference model

    judge_prompts = [generate_hermes_prompt(task + "\n" + completion) for completion in completions]
    judge_results = judge_generator(judge_prompts, max_tokens=max_length)
    
    for i, (completion, judge_result) in enumerate(zip(completions, judge_results)):
        try:
            parsed = json.loads(completion)
            final_answer = parsed.get("final_answer", "")
            
            # Calculate reward
            reward = 0
            if final_answer:
                if final_answer == oracle_answer:
                    reward = 1.0
                elif oracle_answer in final_answer:
                    reward = 0.5
                else:
                    reward = 0.01
            
            judge_parsed = json.loads(judge_result)
            judge_score = float(judge_parsed.get("score", 0))
            
            returns[i] = reward * judge_score
            
        except json.JSONDecodeError:
            returns[i] = 0.0

    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: AutoModelForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def main():
    seed = 42
    wandb_project = "tiny-grpo"
    device_index = 0
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    checkpoint_path = Path("./output")
    checkpoint_interval = 20
    train_batch_size = 16
    lr = 5e-6
    kl_weight = 1e-4
    clip_eps = 0.2

    group_size = 12
    rollouts_per_step = 32
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # rollout params
    max_length = 1024
    top_p = 1.0
    temperature = 1.0

    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)

    reference_model, _ = load_model(model_name, device_map=device)
    model, tokenizer = load_model(model_name, device_map=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.eos_token_id

    prompts = read_prompts(
        "data/math_tasks.jsonl",
        predicate=lambda x: len(x["question"]) < 128
        and x["num_terms"] <= 3
        and x["num_digits"] <= 3,
        max_rows=64 * 1024,
    )
    print(f"found {len(prompts)} matching prompts")
    prompt_loader = DataLoader(
        prompts,
        batch_size=rollouts_per_step,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
    )

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project)

    for k, prompt_batch in enumerate(prompt_loader):
        rollout_returns = []
        completion_lengths = []

        replay_buffer.clear()

        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]

        with torch.no_grad():
            for q, a in zip(questions, answers):
                sequence_ids, returns, action_mask, completions = rollout(
                    model,
                    reference_model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=group_size,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )

                print(
                    f"rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}, replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}"
                )
                rollout_returns.append(returns.cpu())
                completion_lengths.extend([len(c) for c in completions])

                advantages = group_advantages(returns)
                attention_mask = sequence_ids != pad_token_id

                log_probs = sequences_log_probs(
                    model=model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )
                log_probs_ref = sequences_log_probs(
                    model=reference_model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )
                kl = approx_kl_divergence(
                    log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    action_mask=action_mask,
                )

                experience = Experience(
                    sequences=sequence_ids,
                    action_log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    returns=returns,
                    advantages=advantages,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    kl=kl,
                )
                replay_buffer.append(experience.to(cpu_device))

        torch.cuda.empty_cache()
        episode_return_mean = torch.stack(rollout_returns).mean()
        avg_completion_length = sum(completion_lengths) / len(completion_lengths)
        print(f"returns of step {k}: {episode_return_mean:.4f}, avg_completion_length: {avg_completion_length:.4f}")
        wandb.log({
            "returns": episode_return_mean,
            "avg_completion_length": avg_completion_length
        })

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        for step_epoch in range(epochs_per_step):
            model.train()

            for exp in experience_sampler:
                exp: Experience

                exp = exp.to(device)

                optimizer.zero_grad()

                log_probs = sequences_log_probs(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )

                loss, kl = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={experience.advantages}")
                    continue

                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}")
                wandb.log({"kl": kl, "grad_norm": grad_norm, "loss": loss})

                optimizer.step()

        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0
        ):
            model.save_pretrained(checkpoint_path / f"step_{k}")

    if checkpoint_path is not None:
        model.save_pretrained(checkpoint_path / f"step_{k}")


if __name__ == "__main__":
    main()
