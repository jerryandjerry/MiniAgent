# Application design contract

## Client brief

Capture only facts the client owns:

- business goal and intended users;
- supported requests and business rules;
- context already available to the application;
- business systems, documents, or APIs the application may use;
- business constraints and priorities;
- representative requests and expected business outcomes.

The brief can be a short paragraph. Preserve its meaning in `BUSINESS_BRIEF.md`.

## Compiled application specification

The coding agent writes the application README as the authoritative implementation
specification. It contains:

1. product scope and success criteria;
2. one workflow state graph per supported goal;
3. an enumerated route table covering every reachable terminal path;
4. trace links from every supported request and business rule to its workflows;
5. context fields and user-supplied slots, including per-route initial availability
   and clarification order;
6. model-visible tools, argument schemas, result schemas, and dispatch policy;
7. behavior for success, empty evidence, provider error, completion, and refusal;
8. runtime architecture and provider boundaries;
9. data, training, evaluation, and promotion contracts;
10. exact commands and acceptance gates.

## Workflow compilation

Model each workflow as observable decisions and actions:

- start condition;
- context or slot decision;
- one follow-up action for each missing fact;
- tool action;
- branch on typed tool evidence;
- terminal user response.

Represent the graph in Mermaid. Label every route-defining edge. From the repository
root, run
`python3 skills/build-miniagent/scripts/compile_routes.py apps/<app>/README.md --tag W1`
and place the generated route table next to the graph. Select later workflow graphs
with `--block 2`, `--block 3`, and so on. Route tags fix structure; generated samples
vary wording and entities.

Manage combinatorial growth explicitly. Combine distinctions that produce the same
turn and action sequence. Split independent capabilities into separate workflows.

## Tool contracts

Derive tools from business actions. Each tool declares:

- stable name and purpose;
- typed arguments and ordered normalization operations from the tool schema;
- typed `ok`, `empty`, and `error` results;
- side-effect and idempotency behavior;
- timeout and retry policy;
- provider or synthetic-world implementation;
- authorization requirements.

Every argument has an explicit normalization rule. Every workflow binding names a
declared slot or prior tool result. Batched calls declare symmetric compatibility and
branch on every typed result combination.

The dispatch allowlist contains exactly the registered tool contracts. Workflow state
controls when each tool is eligible.

## Runtime boundary

The online request path is:

```text
API / CLI -> Agent -> Service -> Tool / Model
```

`src/` contains the installable project, dependency lock, runtime tests, policy parser,
business contracts, services, tools, selected model artifacts, knowledge-base runtime
artifacts, configuration, and release manifest. Data generation, training, and sealed
evaluation consume stable public contracts during development and stay outside the
runtime project.
