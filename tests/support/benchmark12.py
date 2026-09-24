from __future__ import annotations

from bg6022.benchmark.dataset12 import (
    Benchmark12Item,
    ExpectedCheck,
    ExpectedDependency,
    ExpectedTaskRole,
    TaskGroundTruth,
)
from bg6022.benchmark.models import Benchmark12Observation, Benchmark12TurnObservation
from bg6022.models import (
    Artifact,
    InputReference,
    Plan,
    Request,
    Requirement,
    Result,
    ResultTarget,
    ScientificCheckResult,
    Step,
    Subject,
)


def t003_fixture(*, bad_dependency: bool = False, pre_confirmation: int = 0, legacy: bool = False):
    request = Request(
        id="req-1",
        description="water optimization and frequencies",
        source="chat",
        subjects={"subject_1": Subject(key="subject_1", molecule_query="water")},
        requirements=[
            Requirement(id="req-opt", subject_id="subject_1", capability="optimize_geometry"),
            Requirement(id="req-freq", subject_id="subject_1", capability="frequency"),
        ],
    )
    freq_input = (
        InputReference(artifact_id="geometry-initial")
        if bad_dependency
        else InputReference(step_id="step-opt", port="optimized_geometry")
    )
    plan = Plan(
        id="plan-1",
        request_id=request.id,
        steps=[
            Step(
                id="step-opt",
                tool="optimize_geometry",
                parameters={"method_profile": "r2scan3c"},
                inputs={"geometry": InputReference(artifact_id="geometry-initial")},
                requirement_id="req-opt",
                subject_id="subject_1",
            ),
            Step(
                id="step-freq",
                tool="frequency",
                parameters={"method_profile": "r2scan3c"},
                inputs={"geometry": freq_input},
                requirement_id="req-freq",
                subject_id="subject_1",
            ),
        ],
        requested_results=[ResultTarget(step_id="step-freq", field="vibrational_frequencies")],
    )
    artifacts = [
        Artifact(
            id="geometry-initial",
            artifact_type="molecular_geometry",
            role="input_geometry",
            relative_path="input.xyz",
            size_bytes=0,
            sha256="0" * 64,
            source="test",
            run_id="run-1",
        ),
        Artifact(
            id="geometry-opt",
            artifact_type="molecular_geometry",
            role="optimized_geometry",
            relative_path="opt/geometry.xyz",
            size_bytes=0,
            sha256="1" * 64,
            source="test",
            run_id="run-1",
            step_id="step-opt",
            attempt=1,
        ),
    ]
    results = [
        Result(
            run_id="run-1",
            step_id="step-opt",
            attempt=1,
            status="succeeded",
            values={"opt_final_electronic_energy": -76.1},
            output_ports={"optimized_geometry": "geometry-opt"},
        ),
        Result(
            run_id="run-1",
            step_id="step-freq",
            attempt=1,
            status="succeeded",
            values={"vibrational_frequencies": [1600.0, 3650.0, 3760.0]},
            scientific_checks={
                "frequency_complete": ScientificCheckResult(status="passed"),
                "local_minimum_supported": ScientificCheckResult(status="passed"),
            },
        ),
    ]
    gt = TaskGroundTruth(
        route="compute",
        roles=[
            ExpectedTaskRole(role="opt", capability="optimize_geometry", method_profile="r2scan3c"),
            ExpectedTaskRole(role="freq", capability="frequency", method_profile="r2scan3c"),
        ],
        dependencies=[
            ExpectedDependency(
                source_role="opt",
                source_property="molecular_geometry",
                target_role="freq",
                target_input_type="molecular_geometry",
            )
        ],
        answer_property="frequency",
        output_properties={
            "opt": ["molecular_geometry", "electronic_energy"],
            "freq": ["frequency"],
        },
        scientific_checks=[
            ExpectedCheck(role="freq", name="frequency_complete", status="passed"),
            ExpectedCheck(role="freq", name="local_minimum_supported", status="passed"),
        ],
        max_orca_attempts=2,
    )
    item = Benchmark12Item(
        id="T003__water__canonical",
        task_id="T003",
        object_id="water",
        variant_id="canonical",
        prompt="optimize water then run frequencies",
        ground_truth=gt,
        metadata={"kind": "scientific", "input_mode": "inline_xyz"},
    )
    calls = [{"purpose": "semantic", "category": "success", "usage": {}}]
    if legacy:
        calls.append({"purpose": "planner", "category": "success", "usage": {}})
    turns = [
        Benchmark12TurnObservation(
            index=1,
            user_message=item.prompt,
            response_text="Waiting for confirmation.",
            run_id="run-1",
            run_status="waiting",
            waiting_for="confirmation",
            llm_purposes=["semantic"],
            orca_attempts_after_turn=pre_confirmation,
        ),
        Benchmark12TurnObservation(
            index=2,
            user_message="确认",
            response_text="Computed and verified.",
            run_id="run-1",
            run_status="succeeded",
            llm_purposes=["answer"],
            orca_attempts_after_turn=2,
            delivery_status="complete",
            delivery_properties=["electronic_energy", "molecular_geometry", "frequency"],
        ),
    ]
    observation = Benchmark12Observation(
        item_id=item.id,
        run_index=1,
        turns=turns,
        request=request.model_dump(mode="json"),
        plan=plan.model_dump(mode="json"),
        results=[result.model_dump(mode="json") for result in results],
        artifacts=[artifact.model_dump(mode="json") for artifact in artifacts],
        final_response_text="Computed and verified.",
        llm_calls=calls,
        orca_attempts=2,
        orca_successes=2,
        final_status="succeeded",
        pre_confirmation_orca_attempts=pre_confirmation,
        post_confirmation_orca_attempts=2,
        confirmation_turn_index=2,
        confirmation_required=True,
        public_delivery_properties=["electronic_energy", "molecular_geometry", "frequency"],
        delivery_status="complete",
        legacy_control_plane_used=legacy,
        external_calls={"orca": 2},
    )
    return item, observation


def t005_fixture(*, dependent: bool = False):
    request = Request(
        id="req-5",
        description="independent dual optimization",
        source="chat",
        subjects={"subject_1": Subject(key="subject_1", molecule_query="water")},
        requirements=[
            Requirement(id="req-r2", subject_id="subject_1", capability="optimize_geometry"),
            Requirement(id="req-pbe0", subject_id="subject_1", capability="optimize_geometry"),
        ],
    )
    second_input = (
        InputReference(step_id="step-r2", port="optimized_geometry")
        if dependent
        else InputReference(artifact_id="geometry-initial")
    )
    plan = Plan(
        id="plan-5",
        request_id=request.id,
        steps=[
            Step(
                id="step-r2",
                tool="optimize_geometry",
                parameters={"method_profile": "r2scan3c"},
                inputs={"geometry": InputReference(artifact_id="geometry-initial")},
                requirement_id="req-r2",
                subject_id="subject_1",
            ),
            Step(
                id="step-pbe0",
                tool="optimize_geometry",
                parameters={"method_profile": "pbe0_d3bj_def2svp"},
                inputs={"geometry": second_input},
                requirement_id="req-pbe0",
                subject_id="subject_1",
            ),
        ],
        requested_results=[
            ResultTarget(step_id="step-r2", field="opt_final_electronic_energy"),
            ResultTarget(step_id="step-pbe0", field="opt_final_electronic_energy"),
        ],
    )
    artifacts = [
        Artifact(
            id="geometry-initial",
            artifact_type="molecular_geometry",
            role="input_geometry",
            relative_path="input.xyz",
            size_bytes=0,
            sha256="0" * 64,
            source="test",
            run_id="run-5",
        ),
        Artifact(
            id="geometry-r2",
            artifact_type="molecular_geometry",
            role="optimized_geometry",
            relative_path="r2/geometry.xyz",
            size_bytes=0,
            sha256="1" * 64,
            source="test",
            run_id="run-5",
            step_id="step-r2",
            attempt=1,
        ),
        Artifact(
            id="geometry-pbe0",
            artifact_type="molecular_geometry",
            role="optimized_geometry",
            relative_path="pbe0/geometry.xyz",
            size_bytes=0,
            sha256="2" * 64,
            source="test",
            run_id="run-5",
            step_id="step-pbe0",
            attempt=1,
        ),
    ]
    results = [
        Result(
            run_id="run-5",
            step_id="step-r2",
            attempt=1,
            status="succeeded",
            values={"opt_final_electronic_energy": -76.1},
            output_ports={"optimized_geometry": "geometry-r2"},
        ),
        Result(
            run_id="run-5",
            step_id="step-pbe0",
            attempt=1,
            status="succeeded",
            values={"opt_final_electronic_energy": -76.3},
            output_ports={"optimized_geometry": "geometry-pbe0"},
        ),
    ]
    gt = TaskGroundTruth(
        route="compute",
        roles=[
            ExpectedTaskRole(
                role="opt_r2", capability="optimize_geometry", method_profile="r2scan3c"
            ),
            ExpectedTaskRole(
                role="opt_pbe0",
                capability="optimize_geometry",
                method_profile="pbe0_d3bj_def2svp",
            ),
        ],
        same_initial_geometry_groups=[["opt_r2", "opt_pbe0"]],
        answer_property="electronic_energy",
        output_properties={
            "opt_r2": ["electronic_energy", "molecular_geometry"],
            "opt_pbe0": ["electronic_energy", "molecular_geometry"],
        },
        comparison_mode="side_by_side",
        max_orca_attempts=2,
    )
    item = Benchmark12Item(
        id="T005__water__canonical",
        task_id="T005",
        object_id="water",
        variant_id="canonical",
        prompt="optimize water independently with two methods",
        ground_truth=gt,
        metadata={"kind": "scientific", "input_mode": "inline_xyz"},
    )
    observation = Benchmark12Observation(
        item_id=item.id,
        run_index=1,
        turns=[
            Benchmark12TurnObservation(
                index=1,
                user_message=item.prompt,
                response_text="Waiting.",
                run_id="run-5",
                run_status="waiting",
                waiting_for="confirmation",
                orca_attempts_after_turn=0,
            ),
            Benchmark12TurnObservation(
                index=2,
                user_message="确认",
                response_text="Both calculations completed.",
                run_id="run-5",
                run_status="succeeded",
                delivery_status="complete",
                delivery_properties=["electronic_energy", "molecular_geometry"],
                orca_attempts_after_turn=2,
            ),
        ],
        request=request.model_dump(mode="json"),
        plan=plan.model_dump(mode="json"),
        results=[result.model_dump(mode="json") for result in results],
        artifacts=[artifact.model_dump(mode="json") for artifact in artifacts],
        final_response_text="Both calculations completed.",
        llm_calls=[{"purpose": "semantic", "category": "success", "usage": {}}],
        orca_attempts=2,
        orca_successes=2,
        final_status="succeeded",
        pre_confirmation_orca_attempts=0,
        post_confirmation_orca_attempts=2,
        confirmation_turn_index=2,
        confirmation_required=True,
        public_delivery_properties=["electronic_energy", "molecular_geometry"],
        delivery_status="complete",
        external_calls={"orca": 2},
    )
    return item, observation
