from __future__ import annotations

try:
    from track_B.tot_steps import Evaluation, Hypothesis, MemoryItem, ToolCall, ToTStepRunner, format_question
except ModuleNotFoundError:
    from tot_steps import Evaluation, Hypothesis, MemoryItem, ToolCall, ToTStepRunner, format_question


class TreeOfThought:
    def __init__(self) -> None:
        self.steps = ToTStepRunner()

    def recursive_tot(
        self,
        question: str,
        pert_gene: str,
        target_gene: str,
        max_depth: int = 4,
        max_hypotheses_per_step: int = 5,
        new_hypotheses_per_survivor: int = 2,
        min_score: float = 0.65,
        max_tool_calls_per_hypothesis: int = 3,
        max_total_tool_calls: int = 30,
        max_memory_items: int = 100,
        force_core_tools: bool = True,
        evidence_tool_names: tuple[str, ...] | None = None,
        init_yaml_prompt_path: str = "track_B/prompts/initial_hypoth.yaml",
        tool_decision_yaml_prompt_path: str = "track_B/prompts/tool_decision_prompt.yaml",
        verify_yaml_prompt_path: str = "track_B/prompts/verify_hypothesis_prompt.yaml",
        prune_yaml_prompt_path: str = "track_B/prompts/prune_hypotheses_prompt.yaml",
        refine_yaml_prompt_path: str = "track_B/prompts/refine_hypotheses_prompt.yaml",
        finalize_yaml_prompt_path: str = "track_B/prompts/finalize_hypothesis_prompt.yaml",
    ) -> Hypothesis:
        formatted_question = format_question(question, pert_gene, target_gene)
        memory: list[MemoryItem] = []

        pair_evidence = self.steps.collect_pair_evidence(
            pert_gene=pert_gene,
            target_gene=target_gene,
            depth=0,
            tool_names=evidence_tool_names,
        )

        hypotheses = self.steps.generate_initial_hypotheses(
            prompt_path=init_yaml_prompt_path,
            question=formatted_question,
            pert_gene=pert_gene,
            target_gene=target_gene,
            evidence=pair_evidence,
            depth=0,
        )

        for depth in range(max_depth):
            evaluations: list[Evaluation] = []

            for hypothesis in hypotheses:
                tool_plan: list[ToolCall] = []
                tool_results: list[dict] = []

                # Dynamic per-hypothesis tool calls are intentionally disabled.
                # The workflow now collects pair evidence once before any LLM
                # hypothesis generation and passes that evidence into every step.
                #
                # remaining_tool_calls = max(0, max_total_tool_calls - total_tool_calls)
                # tool_call_limit = min(max_tool_calls_per_hypothesis, remaining_tool_calls)
                # if tool_call_limit > 0:
                #     tool_plan = self.steps.decide_tools(
                #         prompt_path=tool_decision_yaml_prompt_path,
                #         question=formatted_question,
                #         hypothesis=hypothesis,
                #         pert_gene=pert_gene,
                #         target_gene=target_gene,
                #         memory=memory,
                #         max_tool_calls=tool_call_limit,
                #         max_memory_items=max_memory_items,
                #         depth=depth,
                #     )
                #     if force_core_tools:
                #         tool_plan = self.steps.add_core_tools(
                #             tool_plan=tool_plan,
                #             pert_gene=pert_gene,
                #             target_gene=target_gene,
                #             max_tool_calls=tool_call_limit,
                #         )
                # tool_results = self.steps.run_tools(
                #     tool_plan=tool_plan,
                #     hypothesis=hypothesis,
                #     max_tool_calls=tool_call_limit,
                #     depth=depth,
                # )
                # total_tool_calls += sum(
                #     1 for tool_result in tool_results
                #     if not tool_result.get("cached", False)
                # )

                evaluation = self.steps.verify_hypothesis(
                    prompt_path=verify_yaml_prompt_path,
                    question=formatted_question,
                    hypothesis=hypothesis,
                    tool_results=tool_results,
                    evidence=pair_evidence,
                    memory=memory,
                    max_memory_items=max_memory_items,
                    depth=depth,
                )
                evaluations.append(evaluation)
                memory.append(
                    MemoryItem(
                        depth=depth,
                        hypothesis=hypothesis,
                        tool_plan=tool_plan,
                        tool_results=tool_results,
                        evaluation=evaluation,
                    )
                )

            prune_decision = self.steps.prune_hypotheses(
                prompt_path=prune_yaml_prompt_path,
                question=formatted_question,
                hypotheses=hypotheses,
                evaluations=evaluations,
                evidence=pair_evidence,
                memory=memory,
                min_score=min_score,
                max_memory_items=max_memory_items,
                depth=depth,
            )

            if len(prune_decision.survivors) == 0:
                if depth + 1 < max_depth:
                    hypotheses = [self.steps.select_best(memory)]
                    continue

                best_hypothesis = self.steps.select_best(memory)
                return self.steps.finalize_hypothesis(
                    prompt_path=finalize_yaml_prompt_path,
                    question=formatted_question,
                    hypothesis=best_hypothesis,
                    evidence=pair_evidence,
                    memory=memory,
                    max_memory_items=max_memory_items,
                    depth=depth,
                )

            if depth + 1 >= max_depth:
                break

            hypotheses = self.steps.refine_hypotheses(
                prompt_path=refine_yaml_prompt_path,
                question=formatted_question,
                survivors=prune_decision.survivors,
                evidence=pair_evidence,
                memory=memory,
                new_hypotheses_per_survivor=new_hypotheses_per_survivor,
                max_hypotheses=max_hypotheses_per_step,
                max_memory_items=max_memory_items,
                depth=depth + 1,
            )

        best_hypothesis = self.steps.select_best(memory)
        return self.steps.finalize_hypothesis(
            prompt_path=finalize_yaml_prompt_path,
            question=formatted_question,
            hypothesis=best_hypothesis,
            evidence=pair_evidence,
            memory=memory,
            max_memory_items=max_memory_items,
            depth=max_depth,
        )


if __name__ == "__main__":
    tot = TreeOfThought()
    result = tot.recursive_tot(
        question="What is the effect of perturbation gene {pert_gene} on gene {target_gene}?",
        pert_gene="Chmp6",
        target_gene="Trmt10c",
    )
    print(result)
