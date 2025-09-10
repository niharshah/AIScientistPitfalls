from dataclasses import dataclass

import jsonschema
from dataclasses_json import DataClassJsonMixin

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType
import copy
import random
import backoff
import logging
from typing import Callable
import os
# Input your OPENAI_API_KEY
os.environ.get("OPENAI_API_KEY")

@backoff.on_predicate(
    wait_gen=backoff.expo,
    max_value=60,
    factor=1.5,
)
def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception], *args, **kwargs
):
    try:
        return create_fn(*args, **kwargs)
    except retry_exceptions as e:
        return False


def opt_messages_to_list(
    system_message: str | None, user_message: str | None
) -> list[dict[str, str]]:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    """Convert a prompt into markdown format"""
    try:

        if prompt is None:
            return ""

        if isinstance(prompt, str):
            return prompt.strip() + "\n"

        if isinstance(prompt, list):
            # Handle empty list case
            if not prompt:
                return ""
            # Special handling for multi-modal messages
            if all(isinstance(item, dict) and "type" in item for item in prompt):
                # For multi-modal messages, just pass through without modification
                return prompt

            try:
                result = "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])
                return result
            except Exception as e:
                raise

        if isinstance(prompt, dict):
            # Check if this is a single multi-modal message
            if "type" in prompt:
                return prompt

            # Regular dict processing
            try:
                out = []
                header_prefix = "#" * _header_depth
                for k, v in prompt.items():
                    out.append(f"{header_prefix} {k}\n")
                    out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
                return "\n".join(out)
            except Exception as e:

                raise

        raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    except Exception as e:
        raise


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
        }

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }



import json
import logging
import time
from funcy import notnone, once, select_values
import openai
from rich import print


# _client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

@once
def _setup_openai_client():
    global _client
    _client = openai.OpenAI(max_retries=0)


def openai_query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    # _setup_openai_client()
    _client = openai.OpenAI(max_retries=0,   api_key=os.environ.get("OPENAI_API_KEY"))
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message)

    # print(messages)
    # print(json.dumps(messages[0]['content'], indent=2))


    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model to use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    t0 = time.time()
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            # print(f"[cyan]Raw func call response: {choice}[/cyan]")
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:

            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info



def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
    }

    # Handle models with beta limitations
    # ref: https://platform.openai.com/docs/guides/reasoning/beta-limitations
    if model.startswith("o1"):
        if system_message and user_message is None:
            user_message = system_message
        elif system_message is None and user_message:
            pass
        elif system_message and user_message:
            system_message["Main Instructions"] = {}
            system_message["Main Instructions"] |= user_message
            user_message = system_message
        system_message = None
        # model_kwargs["temperature"] = 0.5
        model_kwargs["reasoning_effort"] = "high"
        model_kwargs["max_completion_tokens"] = 100000  # max_tokens
        # remove 'temperature' from model_kwargs
        model_kwargs.pop("temperature", None)
    else:
        model_kwargs["max_tokens"] = max_tokens

    query_func = openai_query
    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=compile_prompt_to_md(system_message) if system_message else None,
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output


node_selection_spec = FunctionSpec(
    name="select_best_implementation",
    description="Select the best implementation based on comprehensive analysis",
    json_schema={
        "type": "object",
        "properties": {
            "selected_id": {
                "type": "string",
                "description": "ID of the selected best implementation",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed explanation of why this implementation was chosen",
            },
        },
        "required": ["selected_id", "reasoning"],
    },
)

def get_best_node(nodes, only_good=True, use_val_metric_only=False):
        """Return the best solution found so far."""

        if use_val_metric_only:
            return max(nodes, key=lambda n: n['metric'])

        if len(nodes) == 1:
            return nodes[0]

        # Create evaluation prompt for LLM
        prompt = {
            "Introduction": (
                "You are an experienced AI researcher evaluating different implementations "
                "of an experiment to select the best one. You should consider all aspects "
                "including performance metrics, training dynamics, generated plots quality."
            ),
            "Task": (
                "Select the best implementation from the candidates below, considering all available evidence."
                "Avoid relying too heavily on the validation loss alone, because "
                "it may not be directly comparable across different objective functions or training details. "
                "If there are multiple validation losses (e.g., when evaluating multiple datasets), "
                "consider all of them and select the implementation that performs best overall."
            ),
            "Candidates": "",
        }
        # Gather info about each node
        for node in nodes:
            if not node['is_seed_node']:
                candidate_info = (
                    f"ID: {node['id']}\n" f"Metric: {str(node['metric'])}\n"
                    if node['metric']
                    else (
                        "N/A\n" f"Training Analysis: {node['analysis']}\n"
                        if hasattr(node, "analysis")
                        else (
                            "N/A\n" f"VLM Feedback: {node['vlm_feedback_summary']}\n"
                            if hasattr(node, "vlm_feedback_summary")
                            else "N/A\n"
                        )
                    )
                )
                prompt["Candidates"] += candidate_info
        # print(prompt)
        prompt = "\n\n".join([f"{key}:\n{value}" for key, value in prompt.items()])
        # return
        try:
            selection = openai_query(
                system_message=prompt,
                user_message=None,
                func_spec=node_selection_spec,
                model="gpt-4o",
                temperature=0.3,
            )

            selected_node = next(
                (node for node in nodes if str(node['id']) == selection[0]["selected_id"]),
                None,
            )
            if selected_node:
                return selected_node
            else:
                return max(nodes, key=lambda n: n['metric'])

        except Exception as e:
            return max(nodes, key=lambda n: n['metric'])
        

if __name__ == "__main__":
    path = "./control/journal{i}.json"

    results = []
    for i in range(1,21):
        with open(path.format(i=i), 'r') as f:
            journal = json.load(f)
            for j in range(10):
                ### copy and shuffle the  journal['nodes'] list
                nodes_copy = copy.deepcopy(journal['nodes'])  # 深拷贝避免修改原数据
                random.shuffle(nodes_copy)  # 原地打乱列表

                best_node = get_best_node(journal['nodes'])
                index = best_node['ranking']
                results.append(index)

