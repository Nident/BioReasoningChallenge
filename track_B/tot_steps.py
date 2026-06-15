from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

try:
    from track_B.tools.langchain_tools import LangChainTools
except ModuleNotFoundError:
    from tools.langchain_tools import LangChainTools


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / "config" / ".env")

Label = Literal["up", "down", "none"]
ANSWER_OPTIONS: tuple[tuple[str, Label], ...] = (
    ("A", "up"),
    ("B", "down"),
    ("C", "none"),
)


@dataclass
class Hypothesis:
    id: str
    parent_id: str
    answer: str
    label: Label
    hypothesis: str
    confidence: float


@dataclass
class ToolCall:
    tool_name: str
    args: dict[str, Any]
    reason: str


@dataclass
class Evaluation:
    hypothesis_id: str
    passed: bool
    score: float
    rationale: str
    failure_reasons: list[str]
    suggested_next_checks: list[str]


@dataclass
class MemoryItem:
    depth: int
    hypothesis: Hypothesis
    tool_plan: list[ToolCall]
    tool_results: list[dict[str, Any]]
    evaluation: Evaluation


@dataclass
class PruneDecision:
    survivor_ids: list[str]
    rejected_ids: list[str]
    rationale: str
    survivors: list[Hypothesis]


def get_model_api_key() -> SecretStr:
    api_key = os.getenv("MODEL_API_KEY")
    if not api_key:
        raise ValueError("MODEL_API_KEY is required in config/.env")
    return SecretStr(api_key)


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        api_key=get_model_api_key(),
        base_url=os.getenv("MODEL_BASE_URL") or None,
        temperature=float(os.getenv("MODEL_TEMPERATURE", "0")),
        max_completion_tokens=int(os.getenv("TOT_MODEL_MAX_TOKENS", "1200")),
        top_p=float(os.getenv("MODEL_TOP_P", "1")),
    )


def format_question(question: str, pert_gene: str, target_gene: str) -> str:
    return question.format(pert_gene=pert_gene, target_gene=target_gene)


def format_prompt(template: str, values: dict[str, Any]) -> str:
    return PromptTemplate.from_template(template).format(**values)


def tool_decision_memory_json(memory: list[MemoryItem], max_items: int) -> str:
    return compact_memory_json(memory, max_items)


def verify_memory_json(memory: list[MemoryItem], max_items: int) -> str:
    return compact_memory_json(memory, max_items)


def prune_memory_json(memory: list[MemoryItem], max_items: int) -> str:
    return compact_memory_json(memory, max_items)


def refine_memory_json(memory: list[MemoryItem], max_items: int) -> str:
    return compact_memory_json(memory, max_items)


def finalize_memory_json(memory: list[MemoryItem], max_items: int) -> str:
    return compact_memory_json(memory, max_items)


def compact_memory_json(memory: list[MemoryItem], max_items: int) -> str:
    return json.dumps(
        [compact_memory_item(item) for item in memory[-max_items:]],
        ensure_ascii=False,
    )


def compact_memory_item(item: MemoryItem) -> dict[str, Any]:
    return {
        "depth": item.depth,
        "hypothesis": compact_hypothesis(item.hypothesis),
        "tool_results": compact_tool_results(item.tool_results),
        "evaluation": compact_evaluation(item.evaluation),
    }


def compact_hypothesis(hypothesis: Hypothesis) -> dict[str, Any]:
    return {
        "id": hypothesis.id,
        "label": hypothesis.label,
        "hypothesis": hypothesis.hypothesis,
    }


def compact_evaluation(evaluation: Evaluation) -> dict[str, Any]:
    return {
        "hypothesis_id": evaluation.hypothesis_id,
        "passed": evaluation.passed,
        "score": evaluation.score,
        "rationale": evaluation.rationale,
        "failure_reasons": evaluation.failure_reasons,
        "suggested_next_checks": evaluation.suggested_next_checks,
    }


def compact_tool_results(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact_results = []

    for tool_result in tool_results:
        compact_result = {
            "tool_name": tool_result["tool_name"],
            "ok": tool_result["ok"],
            "reason": tool_result["reason"],
        }

        if tool_result["ok"]:
            compact_result["result"] = compact_tool_result_payload(
                tool_result["tool_name"],
                tool_result["result"],
            )
        else:
            compact_result["error"] = tool_result["error"]

        compact_results.append(compact_result)

    return compact_results


def compact_tool_result_payload(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "gene_pair_info":
        return {
            "status": result["status"],
            "pert": result["pert"],
            "gene": result["gene"],
            "missing_genes": result["missing_genes"],
            "compact_gene_summary": compact_gene_summary(result["compact_gene_summary"]),
            "compact_pair_report": result["compact_pair_report"],
        }

    if tool_name == "gene_graph_paths":
        return {
            "status": result["status"],
            "direct_edge": result["direct_edge"],
            "weighted_paths": result["weighted_paths"],
            "shortest_path": result["shortest_path"],
            "common_neighbors": result["common_neighbors"],
            "top_paths": result["top_paths"],
        }

    return result


def compact_gene_summary(summary: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        gene: {
            key: value
            for key, value in gene_summary.items()
            if key not in {"preferred_name", "string_id"}
        }
        for gene, gene_summary in summary.items()
    }


class ToTStepRunner:
    def __init__(self) -> None:
        self.tools = LangChainTools()

    def generate_initial_hypotheses(
        self,
        prompt_path: str,
        question: str,
        pert_gene: str,
        target_gene: str,
        depth: int,
    ) -> list[Hypothesis]:
        template = self.load_yaml_prompt(prompt_path)
        values = {
            "question": question,
            "pert_gene": pert_gene,
            "target_gene": target_gene,
        }
        data = self.invoke_json(template, values)

        hypotheses = [
            Hypothesis(
                id=f"H{index}",
                parent_id="",
                answer=item["answer"],
                label=item["label"],
                hypothesis=item["hypothesis"],
                confidence=1.0,
            )
            for index, item in enumerate(data["hypotheses"], start=1)
        ]

        self.save_prompt_and_json(
            depth=depth,
            name=f"initial_hypotheses_{pert_gene}_{target_gene}",
            prompt=format_prompt(template, values),
            response=data,
            normalized=[asdict(item) for item in hypotheses],
        )
        return hypotheses

    def decide_tools(
        self,
        prompt_path: str,
        question: str,
        hypothesis: Hypothesis,
        pert_gene: str,
        target_gene: str,
        memory: list[MemoryItem],
        max_tool_calls: int,
        max_memory_items: int,
        depth: int,
    ) -> list[ToolCall]:
        template = self.load_yaml_prompt(prompt_path)

        values = {
            "question": question,
            "hypothesis": json.dumps(asdict(hypothesis), ensure_ascii=False),
            "pert_gene": pert_gene,
            "target_gene": target_gene,
            "available_tools": self.tools.tool_descriptions_json(),
            "memory": tool_decision_memory_json(memory, max_memory_items),
            "max_tool_calls": max_tool_calls,
        }
        data = self.invoke_json(template, values)

        tool_calls = [
            ToolCall(
                tool_name=item["tool_name"],
                args={"pert": pert_gene, "gene": target_gene},
                reason=item["reason"],
            )
            for item in data["tool_calls"][:max_tool_calls]
        ]

        self.save_prompt_and_json(
            depth=depth,
            name=f"tool_decision_{hypothesis.id}",
            prompt=format_prompt(template, values),
            response=data,
            normalized=[asdict(item) for item in tool_calls],
        )
        return tool_calls

    def run_tools(
        self,
        tool_plan: list[ToolCall],
        hypothesis: Hypothesis,
        max_tool_calls: int,
        depth: int,
    ) -> list[dict[str, Any]]:
        tool_results = [
            {
                **self.tools.run_tool(tool_name=tool_call.tool_name, args=tool_call.args),
                "reason": tool_call.reason,
            }
            for tool_call in tool_plan[:max_tool_calls]
        ]

        self.save_json(
            depth=depth,
            name=f"tool_results_{hypothesis.id}",
            data=tool_results,
        )
        return tool_results

    def verify_hypothesis(
        self,
        prompt_path: str,
        question: str,
        hypothesis: Hypothesis,
        tool_results: list[dict[str, Any]],
        memory: list[MemoryItem],
        max_memory_items: int,
        depth: int,
    ) -> Evaluation:
        template = self.load_yaml_prompt(prompt_path)
        values = {
            "question": question,
            "hypothesis": json.dumps(asdict(hypothesis), ensure_ascii=False),
            "tool_results": json.dumps(compact_tool_results(tool_results), ensure_ascii=False),
            "memory": verify_memory_json(memory, max_memory_items),
        }
        data = self.invoke_json(template, values)

        evaluation = Evaluation(
            hypothesis_id=data["hypothesis_id"],
            passed=data["passed"],
            score=float(data["score"]),
            rationale=data["rationale"],
            failure_reasons=data["failure_reasons"],
            suggested_next_checks=data["suggested_next_checks"],
        )

        self.save_prompt_and_json(
            depth=depth,
            name=f"verify_hypothesis_{hypothesis.id}",
            prompt=format_prompt(template, values),
            response=data,
            normalized=asdict(evaluation),
        )
        return evaluation

    def prune_hypotheses(
        self,
        prompt_path: str,
        question: str,
        hypotheses: list[Hypothesis],
        evaluations: list[Evaluation],
        memory: list[MemoryItem],
        min_score: float,
        max_memory_items: int,
        depth: int,
    ) -> PruneDecision:
        template = self.load_yaml_prompt(prompt_path)
        values = {
            "question": question,
            "hypotheses": json.dumps([asdict(item) for item in hypotheses], ensure_ascii=False),
            "evaluations": json.dumps([asdict(item) for item in evaluations], ensure_ascii=False),
            "memory": prune_memory_json(memory, max_memory_items),
            "min_score": min_score,
        }
        data = self.invoke_json(template, values)

        by_id = {item.id: item for item in hypotheses}
        decision = PruneDecision(
            survivor_ids=data["survivor_ids"],
            rejected_ids=data["rejected_ids"],
            rationale=data["rationale"],
            survivors=[by_id[hypothesis_id] for hypothesis_id in data["survivor_ids"]],
        )

        self.save_prompt_and_json(
            depth=depth,
            name="prune_hypotheses",
            prompt=format_prompt(template, values),
            response=data,
            normalized=asdict(decision),
        )
        return decision

    def refine_hypotheses(
        self,
        prompt_path: str,
        question: str,
        survivors: list[Hypothesis],
        memory: list[MemoryItem],
        new_hypotheses_per_survivor: int,
        max_hypotheses: int,
        max_memory_items: int,
        depth: int,
    ) -> list[Hypothesis]:
        template = self.load_yaml_prompt(prompt_path)
        values = {
            "question": question,
            "survivors": json.dumps([asdict(item) for item in survivors], ensure_ascii=False),
            "memory": refine_memory_json(memory, max_memory_items),
            "new_hypotheses_per_survivor": new_hypotheses_per_survivor,
            "max_hypotheses": max_hypotheses,
        }
        data = self.invoke_json(template, values)

        hypotheses = [
            Hypothesis(
                id=item["id"],
                parent_id=item["parent_id"],
                answer=item["answer"],
                label=item["label"],
                hypothesis=item["hypothesis"],
                confidence=float(item["confidence"]),
            )
            for item in data["hypotheses"][:max_hypotheses]
        ]

        self.save_prompt_and_json(
            depth=depth,
            name="refine_hypotheses",
            prompt=format_prompt(template, values),
            response=data,
            normalized=[asdict(item) for item in hypotheses],
        )
        return hypotheses

    def finalize_hypothesis(
        self,
        prompt_path: str,
        question: str,
        hypothesis: Hypothesis,
        memory: list[MemoryItem],
        max_memory_items: int,
        depth: int,
    ) -> Hypothesis:
        template = self.load_yaml_prompt(prompt_path)
        values = {
            "question": question,
            "hypothesis": json.dumps(asdict(hypothesis), ensure_ascii=False),
            "memory": finalize_memory_json(memory, max_memory_items),
        }
        data = self.invoke_json(template, values)

        final_hypothesis = Hypothesis(
            id=data["id"],
            parent_id=data["parent_id"],
            answer=data["answer"],
            label=data["label"],
            hypothesis=data["hypothesis"],
            confidence=float(data["confidence"]),
        )

        self.save_prompt_and_json(
            depth=depth,
            name=f"finalize_hypothesis_{hypothesis.id}",
            prompt=format_prompt(template, values),
            response=data,
            normalized=asdict(final_hypothesis),
        )
        return final_hypothesis

    @staticmethod
    def select_best(memory: list[MemoryItem]) -> Hypothesis:
        return max(memory, key=lambda item: item.evaluation.score).hypothesis

    @staticmethod
    def load_yaml_prompt(prompt_path: str) -> str:
        path = Path(prompt_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return yaml.safe_load(path.read_text(encoding="utf-8"))["prompt"]

    @staticmethod
    def invoke_json(template: str, values: dict[str, Any]) -> dict[str, Any]:
        prompt = PromptTemplate.from_template(template)
        chain = prompt | build_llm() | JsonOutputParser()
        return chain.invoke(values)

    @staticmethod
    def save_prompt_and_json(
        depth: int,
        name: str,
        prompt: str,
        response: dict[str, Any],
        normalized: Any,
    ) -> None:
        debug_dir = PROJECT_ROOT / "track_B" / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        (debug_dir / f"d{depth}_{name}_prompt.txt").write_text(prompt, encoding="utf-8")
        (debug_dir / f"d{depth}_{name}_response.json").write_text(
            json.dumps(response, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (debug_dir / f"d{depth}_{name}.json").write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def save_json(depth: int, name: str, data: Any) -> None:
        debug_dir = PROJECT_ROOT / "track_B" / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / f"d{depth}_{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
