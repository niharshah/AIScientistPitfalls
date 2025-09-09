import argparse
import json
import os.path as osp
import re
import traceback
from typing import Any, Dict, List
import pathlib    
import sys

sys.path.append(osp.join(osp.dirname(__file__), ".."))
from ai_scientist.llm import (
    AVAILABLE_LLMS,
    create_client,
    get_response_from_llm,
)

from ai_scientist.tools.semantic_scholar import SemanticScholarSearchTool
from ai_scientist.tools.base_tool import BaseTool

# Create tool instances
semantic_scholar_tool = SemanticScholarSearchTool()

# Define tools at the top of the file
tools = [
    semantic_scholar_tool,
    {
        "name": "FinalizeIdea",
        "description": """Finalize your idea by providing the idea details.

The IDEA JSON should include the following fields:
- "Name": A short descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A catchy and informative title for the proposal.
- "Short Hypothesis": A concise statement of the main hypothesis or research question. Clarify the need for this specific direction, ensure this is the best setting to investigate this idea, and there are not obvious other simpler ways to answer the question.
- "Related Work": A brief discussion of the most relevant related work and how the proposal clearly distinguishes from it, and is not a trivial extension.
- "Abstract": An abstract that summarizes the proposal in conference format (approximately 250 words).
- "Experiments": A list of experiments that would be conducted to validate the proposal. Ensure these are simple and feasible. Be specific in exactly how you would test the hypothesis, and detail precise algorithmic changes. Include the evaluation metrics you would use.
- "Risk Factors and Limitations": A list of potential risks and limitations of the proposal.""",
    },
]

# Create a tools dictionary for easy lookup
tools_dict = {tool.name: tool for tool in tools if isinstance(tool, BaseTool)}

# Create a string with the tool descriptions
tool_descriptions = "\n\n".join(
    (
        f"- **{tool.name}**: {tool.description}"
        if isinstance(tool, BaseTool)
        else f"- **{tool['name']}**: {tool['description']}"
    )
    for tool in tools
)

# Extract tool names for the prompt
tool_names = [
    f'"{tool.name}"' if isinstance(tool, BaseTool) else f'"{tool["name"]}"'
    for tool in tools
]
tool_names_str = ", ".join(tool_names)

system_prompt = f"""You are an experienced AI researcher who aims to propose high-impact research ideas resembling exciting grant proposals. Feel free to propose any novel ideas or experiments; make sure they are novel. Be very creative and think out of the box. Each proposal should stem from a simple and elegant question, observation, or hypothesis about the topic. For example, they could involve very interesting and simple interventions or investigations that explore new possibilities or challenge existing assumptions. Clearly clarify how the proposal distinguishes from the existing literature.

Ensure that the proposal does not require resources beyond what an academic lab could afford. These proposals should lead to papers that are publishable at top ML conferences.

You have access to the following tools:

{tool_descriptions}

Respond in the following format:

ACTION:
<The action to take, exactly one of {tool_names_str}>

ARGUMENTS:
<If ACTION is "SearchSemanticScholar", provide the search query as {{"query": "your search query"}}. If ACTION is "FinalizeIdea", provide the idea details as {{"idea": {{ ... }}}} with the IDEA JSON specified below.>

If you choose to finalize your idea, provide the IDEA JSON in the arguments:

IDEA JSON:
```json
{{
    "Name": "...",
    "Title": "...",
    "Short Hypothesis": "...",
    "Related Work": "...",
    "Abstract": "...",
    "Experiments": "...",
    "Risk Factors and Limitations": "..."
}}
```

Ensure the JSON is properly formatted for automatic parsing.

Note: You should perform at least one literature search before finalizing your idea to ensure it is well-informed by existing research."""

# Define the initial idea generation prompt
idea_generation_prompt = """{workshop_description}

Here are the proposals that you have already generated:

'''
{prev_ideas_string}
'''

Begin by generating an interestingly new high-level research proposal that differs from what you have previously proposed.
"""

# Define the reflection prompt
idea_reflection_prompt = """Round {current_round}/{num_reflections}.

In your thoughts, first carefully consider the quality, novelty, and feasibility of the proposal you just created.
Include any other factors that you think are important in evaluating the proposal.
Ensure the proposal is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try to refine and improve your proposal.
Stick to the spirit of the original idea unless there are glaring issues.

If you have new information from tools, such as literature search results, incorporate them into your reflection and refine your proposal accordingly.

Results from your last action (if any):

{last_tool_results}
"""

TASK_DESCRIPTION = """# Title: Develop a Robust Algorithm for a Novel Task “Synthetic PolyRule Reasoning” 

## Background

**Synthetic PolyRule Reasoning (SPR)**

* SPR is a classification task distilled from complex reasoning patterns found in various real-world domains such as finance, academic publishing, and scientific discovery, where latent symbolic rules govern decision-making. Solving this task has the potential to greatly impact automated reasoning systems in practice. In SPR, each instance consists of a symbolic sequence $S = [s_1, s_2, \dots, s_L]$ of length $L$, where each token $s_i$ is composed of an abstract shape glyph from the set **{{▲, ■, ●, ◆}}** and a color glyph from the set **{{r, g, b, y}}**.


* A hidden *generation rule* $R$ governs the mapping from an input sequence $S$ to a binary label: *accept / reject*. This rule encapsulates a logical structure that governs how different symbol combinations should be interpreted, leading to the classification decision.

* The rules in SPR are **poly‑factor**, meaning each rule is the result of applying a logical AND across $k$ atomic predicates. These atomic predicates are derived from the following categories:
  1. **Shape-Count**: Refers to conditions based on the frequency of a specific shape within the sequence. For instance, a predicate could specify "exactly three ▲," meaning that the rule only holds if there are exactly three occurrences of the shape ▲.

  2. **Color-Position**: Conditions based on the color of a specific token at a defined position in the sequence. An example predicate might be "token 4 is **r**," meaning the fourth token in the sequence must be colored red for the rule to hold.

  3. **Parity**: Conditions involving the even or odd count of specific shapes or colors. For example, "the number of ■ is even" could be a rule, meaning that the total count of squares ■ in the sequence must be an even number for the sequence to be accepted.

  4. **Order**: Relational conditions on the order of specific tokens in the sequence. An example could be "the first ▲ precedes the first ●," meaning that the first occurrence of the shape ▲ must appear before the first occurrence of the shape ● in the sequence.

By solving the SPR task, we aim to unlock the ability to automatically identify and classify complex symbolic sequences that follow hidden, intricate rules. This has significant potential for automation in various domains where symbolic data patterns need to be understood, such as automating financial analysis, enhancing academic publishing workflows, or improving decision-making systems in complex environments.


## Task Instructions

1. **Design an Algorithm**: Develop an algorithm to solve the **SPR (Symbolic Pattern Recognition)** task. Your algorithm should decide whether a given \$L\$-token sequence of abstract symbols satisfies the hidden target rule.

2. **Benchmark Selection**: From the 20 available benchmarks listed in the following section, **select 4 benchmarks** to evaluate your algorithm and list the benchmark names. Clearly **list** the selected benchmark names and provide a **justification** for your choice of benchmarks based on their characteristics and how they align with your algorithm’s strengths.

3. **Training Procedure**:

   * Train your model using the **Train split** of each selected benchmark.
   * Tune your model on the **Dev split**.
   * The **Test split** labels are withheld, and you must report accuracy based on your model’s performance on this unseen data.
   * Note that **cross-benchmark training** is prohibited. Each model should be trained and evaluated independently for each chosen benchmark.

4. **Baseline Comparison**: Set the **SOTA (State-of-the-Art) accuracies** for each benchmark as a baseline. Your goal is to **compare your model’s performance** against these baselines and demonstrate improvements.

6. **Submission Requirements**: For each selected benchmark, submit a **separate model** along with the following:

   * The **final accuracy on the Test set**.
   * A comparison of your model’s performance against the **SOTA baseline** for that benchmark.

7. **Objective**: The goal of this task is to develop a **robust algorithm** that:

   * Outperforms the current **SOTA** benchmarks.
   * Demonstrates strong generalization across variations in **vocabulary sizes**, **sequence lengths**, and **rule complexities**.




## 20 Benchmarks from HuggingFace

In this experiment, we use 20 carefully curated benchmarks sourced from HuggingFace, each designed to evaluate the performance of machine learning models on a symbolic pattern recognition task. These benchmarks are unified in terms of dataset splitting, label distribution, and evaluation metrics, ensuring consistency across experiments. The benchmarks include a variety of symbolic patterns that challenge models to classify sequences of abstract symbols under varying conditions.


### **Fixed Global Dataset Parameters**

All benchmarks are standardized with the following fixed global parameters:


| Field                         | Value                                |
| ----------------------------- | ------------------------------------ |
| **Instances per split**       | Train = 2 000 Dev = 500 Test = 1 000 |
| **Label balance**             | 50 % accept / 50 % reject            |
| **Public leaderboard metric** | Label Accuracy on each benchmark     |

### Benchmark Results and SOTA Accuracy

Below are the names of the 20 benchmarks (encrypted for privacy).

{benchmarks}

The benchmarks are encrypted to preserve privacy, but the results indicate the variability in performance across different benchmarks. These benchmarks were designed to evaluate the robustness and generalization capabilities of models in detecting hidden patterns in symbolic sequences, making them a valuable resource for advancing machine learning techniques for symbolic reasoning.


### **Data Description**

```sql
SPR_BENCH/
 ├─ SFRFG/
 │   ├─ train.csv   (2 000 rows)
 │   ├─ dev.csv     (  500 rows)
 │   └─ test.csv    (1 000 rows)
 ├─ IJSJF/
 └─ …

```

Each `*.csv` has **UTF-8**, comma-separated columns:

| id           | sequence                | label |
| ------------ | ----------------------- | ----- |
| B01_train_17 | ▲r ■b ▲r ●g ◆r ■r ●y ◆r | 1     |

* A **token** is a shape glyph plus an optional 1-letter colour (`▲g`, `■`, …).
* **Label** = `1` (accept) or `0` (reject).
"""


def generate_temp_free_idea(
    idea_fname: str,
    client: Any,
    model: str,
    workshop_description: str,
    max_num_generations: int = 20,
    num_reflections: int = 5,
    reload_ideas: bool = False,
) -> List[Dict]:
    idea_str_archive = []
    # load ideas from file
    if reload_ideas and osp.exists(idea_fname):
        with open(idea_fname, "r") as f:
            idea_str_content = json.load(f)
            for idea in idea_str_content:
                idea_str_archive.append(json.dumps(idea))
            print(f"Loaded {len(idea_str_archive)} ideas from {idea_fname}")
    else:
        print(f"No ideas found in {idea_fname}. Starting from scratch.")

    for gen_idx in range(max_num_generations):
        print(f"Generating proposal {gen_idx + 1}/{max_num_generations}")
        
        # workshop_description = TASK_DESCRIPTION.format(W=randomize_markdown_table())
        # print(workshop_description)

        try:
            # prev_ideas_string = "\n\n".join(idea_str_archive)
            prev_ideas_string = ''

            last_tool_results = ""
            idea_finalized = False
            msg_history = []

            for reflection_round in range(num_reflections):
                if reflection_round == 0:
                    # Use the initial idea generation prompt
                    prompt_text = idea_generation_prompt.format(
                        workshop_description=workshop_description,
                        prev_ideas_string=prev_ideas_string,
                    )
                else:
                    # Use the reflection prompt, including tool results if any
                    prompt_text = idea_reflection_prompt.format(
                        current_round=reflection_round + 1,
                        num_reflections=num_reflections,
                        last_tool_results=last_tool_results or "No new results.",
                    )

                response_text, msg_history = get_response_from_llm(
                    prompt=prompt_text,
                    client=client,
                    model=model,
                    system_message=system_prompt,
                    msg_history=msg_history,
                )
                # print('-'*50)
                # print(response_text)
                # print('-'*50)
                
                # Parse the LLM's response
                try:
                    # Use regular expressions to extract the components
                    action_pattern = r"ACTION:\s*(.*?)\s*ARGUMENTS:"
                    arguments_pattern = r"ARGUMENTS:\s*(.*?)(?:$|\nTHOUGHT:|\n$)"

                    action_match = re.search(
                        action_pattern, response_text, re.DOTALL | re.IGNORECASE
                    )
                    arguments_match = re.search(
                        arguments_pattern, response_text, re.DOTALL | re.IGNORECASE
                    )

                    if not all([action_match, arguments_match]):
                        raise ValueError("Failed to parse the LLM response.")

                    action = action_match.group(1).strip()
                    arguments_text = arguments_match.group(1).strip()
                    # print(f"Action: {action}")
                    # print(f"Arguments: {arguments_text}")

                    # If arguments are wrapped in ```json blocks, extract the content
                    if arguments_text.startswith("```json"):
                        arguments_text = re.search(
                            r"```json\s*(.*?)\s*```", arguments_text, re.DOTALL
                        ).group(1)

                    # Process the action and arguments
                    if action in tools_dict:
                        # It's a tool we have defined
                        tool = tools_dict[action]
                        # Parse arguments
                        try:
                            arguments_json = json.loads(arguments_text)
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid arguments JSON for {action}.")

                        # Use the tool
                        try:
                            # Assuming the arguments match the parameters of the tool
                            result = tool.use_tool(**arguments_json)
                            last_tool_results = result
                        except Exception as e:
                            last_tool_results = f"Error using tool {action}: {str(e)}"
                    elif action == "FinalizeIdea":
                        # Parse arguments
                        try:
                            arguments_json = json.loads(arguments_text)
                            idea = arguments_json.get("idea")
                            if not idea:
                                raise ValueError("Missing 'idea' in arguments.")

                            # Append the idea to the archive
                            idea_str_archive.append(json.dumps(idea))
                            print(f"Proposal finalized: {idea}")
                            print(type(idea))
                            idea_finalized = True
                            break
                        except json.JSONDecodeError:
                            print(arguments_json)
                            # raise ValueError("Invalid arguments JSON for FinalizeIdea.")
                    else:
                        print(
                            "Invalid action. Please specify one of the available tools."
                        )
                        print(f"Available actions are: {tool_names_str}")
                except Exception as e:
                    print(
                        f"Failed to parse LLM response. Response text:\n{response_text}"
                    )
                    traceback.print_exc()
                    break  # Exit the loop if parsing fails

            if idea_finalized:
                
                if pathlib.Path(idea_fname).exists():
                    with open(idea_fname, "r") as f:
                        existing_ideas = json.load(f)
                else:
                    existing_ideas = []

                existing_ideas.append(idea)
                with open(idea_fname, "w") as f:
                    json.dump(existing_ideas, f, indent=4)
                    
                continue  # Move to the next idea

        except Exception as e:
            print("Failed to generate proposal:")
            traceback.print_exc()
            continue

    # Save ideas
    ideas = [json.loads(idea_str) for idea_str in idea_str_archive]

    # with open(idea_fname, "w") as f:
    #     json.dump(ideas, f, indent=4)
    # print(f"Stored {len(ideas)} ideas in {idea_fname}")
    return ideas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate AI scientist proposals - template free"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--max-num-generations",
        type=int,
        default=1,
        help="Maximum number of proposal generations.",
    )
    parser.add_argument(
        "--workshop-file",
        type=str,
        default="ideas/SPR.md",
        help="Path to the workshop description file.",
    )
    parser.add_argument(
        "--num-reflections",
        type=int,
        default=5,
        help="Number of reflection rounds per proposal.",
    )
    args = parser.parse_args()

    # Create the LLM client
    client, client_model = create_client(args.model)

    with open(args.workshop_file, "r") as f:
        workshop_description = f.read()
    # print(f"Using workshop description from {args.workshop_file} for idea generation.")
    # print(f"Workshop description:\n{workshop_description}")

    # Create output filename by replacing .md extension with .json
    idea_fname = args.workshop_file.replace(".md", ".json")
    print("Starting idea generation for", idea_fname)
    ideas = generate_temp_free_idea(
        idea_fname=idea_fname,
        client=client,
        model=client_model,
        workshop_description=workshop_description,
        max_num_generations=args.max_num_generations,
        num_reflections=args.num_reflections,
    )
    print(f"{args.workshop_file} generated {len(ideas)} ideas.")
