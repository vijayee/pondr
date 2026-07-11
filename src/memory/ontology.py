"""Seed ontology for the hippocampal memory system.

Stored in the WaveDB Graph layer at initialization as ``subClassOf`` triples
and evolved through discovery (GLiNER-Decoder invents labels → buffered →
promoted by Bonsai entailment) and periodic GNN refinement (architecture doc
§5, "The Evolving Ontology").

The seed has two halves that merge into one DAG:

* **Conversational** — the Episode-side taxonomy (entities, events, affective
  tones, topics) used to index conversations.
* **Code** — the code-artifact taxonomy (AST-level artifacts, version control,
  issues, tests, architecture, API, data, configuration, infrastructure,
  deployment, observability, quality, process) used to index code
  conversations. Drawn from the design discussion; it is a starting point, not
  an exhaustive taxonomy — new classes arrive at runtime via discovery.

Merge notes (resolved, flagged for review):
* The merged ontology is a **multi-parent DAG**, not a tree — a class may have
  several ``subClassOf`` parents. Overlaps between the two halves (e.g.
  ``Database`` is both ``Project``-side and ``Data``-side; ``Field`` is both
  ``CodeArtifact``-side and ``Data``-side; ``Cache`` under ``API``/``Data``/
  ``Infrastructure``; ``Configuration`` is a ``Topic`` AND a parent of config
  artifacts) are represented as multiple parents, which is what the Graph
  layer stores natively.
* The code ontology's source had a duplicate ``implements`` key
  (``Class→Interface`` and ``Commit→Feature``). Kept both: ``implements`` =
  ``Class→Interface`` and ``implements_feature`` = ``Commit→Feature``.
* Property domain/range types that were originally dangling
  (``Repository``, ``Team``, ``Message``, ``Permission``, ``Value``, plus the
  types introduced by the development/business extensions below) are defined
  as classes, so every relation endpoint is a real node.
* Beyond the code-artifact taxonomy, the seed also covers **development**
  (paradigms, algorithms, data structures, control flow, error handling,
  concurrency, documents, knowledge, communication, security) and
  **business/organizational** concepts (stakeholders, requirements, business
  rules, workflows, products, markets, regulations, organizations, roles) so
  conversations about code in any language, the development process, and the
  business/org context around it all index against a shared vocabulary.
"""

from typing import Any


# ── Conversational (Episode-side) taxonomy ──
CONVERSATIONAL_CLASSES: dict[str, list[str]] = {
    # Episode is the root: every recorded turn is an instance of Episode, and
    # the has* properties hang off it. Declared explicitly so it's a real node.
    "Episode": [],
    "Entity": ["Person", "Project", "Technology", "Concept"],
    "Project": ["Database", "Application", "Library"],
    "Technology": ["Protocol", "Language", "Framework"],
    # User = the agent's owner / a persona. A User owns Sessions and (via them)
    # Episodes, scoping a chat history to one owner so cross-chat recall is a
    # first-class query. SubClassOf Person so the existing Person-typed
    # relations (madeBy, explains, etc.) accept a User too.
    "Person": ["User"],
    "Event": ["Decision", "Explanation", "Question", "Conflict", "Session"],
    "AffectiveTone": ["Frustrated", "Excited", "Curious", "Neutral"],
    "Topic": [
        "DatabaseDesign", "Configuration", "Performance",
        "Security", "APIDesign", "AIArchitecture",
    ],
    # Statement is a leaf referenced by the `contradicts` property but not by
    # any subclass list, so it is declared explicitly rather than auto-created.
    "Statement": [],
}

CONVERSATIONAL_PROPERTIES: dict[str, dict[str, str]] = {
    "hasEntity":   {"domain": "Episode",   "range": "Entity"},
    "hasTopic":     {"domain": "Episode",   "range": "Topic"},
    "hasTone":      {"domain": "Episode",   "range": "AffectiveTone"},
    "hasDecision":  {"domain": "Episode",   "range": "Decision"},
    "madeBy":       {"domain": "Decision",  "range": "Person"},
    "about":        {"domain": "Decision",  "range": "Topic"},
    "explains":     {"domain": "Person",     "range": "Concept"},
    "contradicts":  {"domain": "Statement", "range": "Statement"},
    "follows":      {"domain": "Episode",    "range": "Episode"},
    "supersedes":   {"domain": "Episode",    "range": "Episode"},  # reconsolidation
    "superseded_by": {"domain": "Episode",   "range": "Episode"},  # back-pointer of reconsolidation
    "subClassOf":   {"domain": "Entity",    "range": "Entity"},    # taxonomy edges
    # ── User / Session hierarchy (global chat history) ──
    # A User owns Sessions; a Session contains Episodes. `follows` chains
    # episodes WITHIN a session; `follows_session` chains a user's sessions
    # (cross-chat temporal order).
    #
    # Literal data edges written as triples but intentionally NOT registered
    # here: `at_time`, `started_at`, `ended_at` (timestamps on episodes /
    # sessions), `state`, `validity_start`, `validity_end` (lifecycle /
    # supersession). They are data, not structure — a literal timestamp or
    # state string has no class-typed range, so registering it as a property
    # would add a meaningless domain/range row. Cross-session temporal queries
    # read them via a timestamp scan + MVCC validity windows (Phase 1c) rather
    # than a single global episode edge.
    "has_session":     {"domain": "User",    "range": "Session"},
    "has_episode":     {"domain": "Session", "range": "Episode"},
    "in_session":      {"domain": "Episode", "range": "Session"},
    "follows_session": {"domain": "Session", "range": "Session"},
}


# ── Code taxonomy (from the design discussion) ──
CODE_CLASSES: dict[str, list[str]] = {
    # Code structure (AST-level artifacts).
    "CodeArtifact": [
        "File", "Module", "Package",
        "Class", "Interface", "Trait", "Mixin", "Enum", "Struct",
        "Function", "Method", "Constructor", "Destructor",
        "Property", "Attribute", "Field",
        "Variable", "Constant", "Parameter",
        "Type", "Generic", "Union", "Alias",
        "Decorator", "Annotation",
        "Lambda", "Closure", "Generator",
        "Expression", "Statement", "Block",
    ],
    # Version control.
    "VersionControl": [
        "Repository",
        "Commit", "Branch", "Tag", "Release",
        "PullRequest", "MergeRequest", "Patch",
        "Merge", "Rebase", "CherryPick",
        "Conflict", "Diff", "Blame",
        "Fork", "Clone", "Remote",
        "Stash", "Worktree",
    ],
    # Issue tracking.
    "Issue": [
        "Bug", "Feature", "Enhancement", "Task",
        "TechnicalDebt", "Refactor",
        "PerformanceIssue", "SecurityVulnerability",
        "Regression", "BreakingChange",
        "Deprecation", "Migration",
    ],
    # Testing.
    "Test": [
        "UnitTest", "IntegrationTest", "EndToEndTest",
        "PerformanceTest", "SecurityTest", "RegressionTest",
        "Mock", "Stub", "Fixture", "TestSuite",
        "Coverage", "Assertion",
    ],
    # Architecture & design.
    "Architecture": [
        "DesignPattern", "ArchitecturalPattern",
        "Component", "Service", "Microservice",
        "Monolith", "Plugin", "Middleware",
        "Layer", "Tier", "Boundary",
        "Adapter", "Facade", "Proxy", "Bridge",
        "Factory", "Singleton", "Observer", "Strategy",
    ],
    # API.
    "API": [
        "Endpoint", "Route", "Controller",
        "Middleware", "Guard", "Interceptor",
        "Request", "Response", "DTO", "Schema",
        "Query", "Mutation", "Subscription",
        "REST", "GraphQL", "gRPC", "WebSocket",
        "RateLimit", "Throttle", "Cache",
    ],
    # Data.
    "Data": [
        "Database", "Table", "Collection",
        "Column", "Field", "Index",
        "PrimaryKey", "ForeignKey", "Constraint",
        "Query", "Migration", "Seed",
        "Schema", "Model", "Entity", "Relation",
        "Transaction", "Lock", "Deadlock",
        "Cache", "Session", "Connection",
    ],
    # Configuration.
    "Configuration": [
        "EnvironmentVariable", "ConfigFile",
        "Secret", "Credential", "APIKey",
        "FeatureFlag", "Toggle",
        "Profile", "BuildConfig", "Value",
    ],
    # Operations / DevOps.
    "Infrastructure": [
        "Server", "Container", "Pod", "Cluster",
        "LoadBalancer", "Proxy", "CDN",
        "Database", "Queue", "Cache",
        "Volume", "Network", "Firewall",
        "DNS", "Certificate",
    ],
    "Deployment": [
        "Pipeline", "Stage", "Job", "Step",
        "Build", "Test", "Deploy", "Rollback",
        "Artifact", "Image", "Registry",
        "Environment", "Namespace",
        "HelmChart", "Manifest", "Template",
    ],
    "Observability": [
        "Log", "Metric", "Trace", "Span",
        "Alert", "Incident", "Runbook",
        "Dashboard", "Monitor", "SLO",
        "Error", "Warning", "Debug",
    ],
    # Quality & process.
    "Quality": [
        "Lint", "Format", "StyleGuide",
        "CodeReview", "Audit",
        "Complexity", "Duplication",
        "Documentation", "README", "Changelog",
        "License", "Dependency",
        "Vulnerability", "CVE", "Patch",
    ],
    "Process": [
        "Sprint", "Milestone", "Roadmap",
        "Estimate", "StoryPoint",
        "Standup", "Retrospective",
        "OnCall", "IncidentResponse",
        "PostMortem", "RCA",
    ],
}

CODE_PROPERTIES: dict[str, dict[str, str]] = {
    # ── Code structure ──
    "contains":          {"domain": "CodeArtifact", "range": "CodeArtifact"},
    "defined_in":        {"domain": "CodeArtifact", "range": "File"},
    "declared_in":       {"domain": "CodeArtifact", "range": "CodeArtifact"},
    "calls":             {"domain": "Function",     "range": "Function"},
    "imports":           {"domain": "File",         "range": "Module"},
    "exports":           {"domain": "Module",       "range": "CodeArtifact"},
    "inherits":          {"domain": "Class",        "range": "Class"},
    "implements":        {"domain": "Class",        "range": "Interface"},
    "implements_feature": {"domain": "Commit",      "range": "Feature"},  # split from dup `implements`
    "overrides":         {"domain": "Method",      "range": "Method"},
    "uses":              {"domain": "CodeArtifact", "range": "CodeArtifact"},
    "instantiates":      {"domain": "CodeArtifact", "range": "Class"},
    "decorates":         {"domain": "Decorator",    "range": "CodeArtifact"},
    "annotates":         {"domain": "Annotation",   "range": "CodeArtifact"},
    "type_of":           {"domain": "Variable",     "range": "Type"},
    "returns":           {"domain": "Function",     "range": "Type"},
    "accepts":           {"domain": "Function",     "range": "Parameter"},
    "raises":            {"domain": "Function",     "range": "Type"},
    "catches":           {"domain": "Block",        "range": "Type"},

    # ── Version control ──
    "commits":           {"domain": "Branch",       "range": "Commit"},
    "parents":           {"domain": "Commit",       "range": "Commit"},
    "branches_from":     {"domain": "Branch",       "range": "Branch"},
    "merges_into":       {"domain": "Branch",       "range": "Branch"},
    "tags":              {"domain": "Commit",       "range": "Tag"},
    "releases":          {"domain": "Tag",          "range": "Release"},
    "resolves":          {"domain": "Merge",        "range": "Conflict"},
    "cherry_picks":      {"domain": "Commit",       "range": "Commit"},
    "reverts":           {"domain": "Commit",       "range": "Commit"},

    # ── Issue tracking ──
    "fixes":             {"domain": "Commit",       "range": "Bug"},
    "introduces":        {"domain": "Commit",       "range": "Bug"},
    "regresses":         {"domain": "Commit",       "range": "Regression"},
    "refactors":         {"domain": "Commit",       "range": "Refactor"},
    "addresses":         {"domain": "Commit",       "range": "Issue"},
    "closes":            {"domain": "PullRequest",  "range": "Issue"},
    "blocks":            {"domain": "Issue",        "range": "Issue"},
    "depends_on_issue":  {"domain": "Issue",        "range": "Issue"},
    "duplicates":        {"domain": "Issue",        "range": "Issue"},

    # ── Testing ──
    "tests":             {"domain": "Test",        "range": "CodeArtifact"},
    "covers":            {"domain": "TestSuite",    "range": "CodeArtifact"},
    "mocks":              {"domain": "Test",         "range": "CodeArtifact"},
    "asserts":            {"domain": "Test",         "range": "Assertion"},
    "fails":              {"domain": "Test",         "range": "Bug"},
    "regression_tests":  {"domain": "RegressionTest", "range": "Bug"},

    # ── Architecture & design ──
    "depends_on":         {"domain": "CodeArtifact", "range": "CodeArtifact"},
    "depends_on_module":   {"domain": "Module",      "range": "Module"},
    "depends_on_service":  {"domain": "Service",     "range": "Service"},
    "owns":                {"domain": "Team",        "range": "Service"},
    "communicates_with":   {"domain": "Service",     "range": "Service"},
    "proxies":             {"domain": "Proxy",       "range": "Service"},
    "balances":            {"domain": "LoadBalancer", "range": "Service"},
    "caches":              {"domain": "Cache",       "range": "Data"},
    "queues":              {"domain": "Queue",        "range": "Message"},
    "subscribes":          {"domain": "Service",      "range": "Event"},
    "publishes":           {"domain": "Service",      "range": "Event"},

    # ── API ──
    "routes_to":          {"domain": "Route",        "range": "Controller"},
    "handles":            {"domain": "Controller",   "range": "Endpoint"},
    "guards":             {"domain": "Guard",        "range": "Route"},
    "intercepts":         {"domain": "Interceptor",  "range": "Request"},
    "validates":          {"domain": "Middleware",    "range": "Schema"},
    "rate_limits":        {"domain": "RateLimit",    "range": "Endpoint"},
    "authenticates":      {"domain": "Guard",        "range": "Credential"},
    "authorizes":         {"domain": "Guard",        "range": "Permission"},

    # ── Data ──
    "persists":           {"domain": "Repository",   "range": "Entity"},
    "maps_to":            {"domain": "Entity",       "range": "Table"},
    "columns":            {"domain": "Table",        "range": "Column"},
    "references":         {"domain": "ForeignKey",   "range": "PrimaryKey"},
    "indexes":           {"domain": "Index",         "range": "Column"},
    "constrains":         {"domain": "Constraint",   "range": "Column"},
    "migrates":           {"domain": "Migration",    "range": "Schema"},
    "seeds":              {"domain": "Seed",         "range": "Table"},
    "transacts":          {"domain": "Transaction",   "range": "Database"},
    "locks":              {"domain": "Transaction",  "range": "Table"},

    # ── Configuration ──
    "configures":         {"domain": "ConfigFile",          "range": "CodeArtifact"},
    "sets":               {"domain": "EnvironmentVariable", "range": "Value"},
    "secrets":            {"domain": "Secret",       "range": "Credential"},
    "flags":              {"domain": "FeatureFlag",   "range": "Feature"},
    "profiles":           {"domain": "Profile",      "range": "Environment"},

    # ── Deployment ──
    "builds":             {"domain": "Pipeline",     "range": "Artifact"},
    "deploys_to":         {"domain": "Pipeline",     "range": "Environment"},
    "runs_on":            {"domain": "Job",          "range": "Infrastructure"},
    "produces":           {"domain": "Job",          "range": "Artifact"},
    "rolls_back":         {"domain": "Deploy",       "range": "Deploy"},
    "contains_stage":     {"domain": "Pipeline",     "range": "Stage"},
    "contains_job":       {"domain": "Stage",        "range": "Job"},
    "contains_step":      {"domain": "Job",          "range": "Step"},

    # ── Observability ──
    "logs":               {"domain": "CodeArtifact", "range": "Log"},
    "emits":              {"domain": "CodeArtifact", "range": "Metric"},
    "traces":             {"domain": "CodeArtifact", "range": "Span"},
    "alerts_on":          {"domain": "Monitor",      "range": "Metric"},
    "triggers":           {"domain": "Alert",        "range": "Incident"},
    "resolved_by":        {"domain": "Incident",     "range": "Runbook"},
    "caused_by":          {"domain": "Incident",     "range": "Deploy"},
    "postmortem_for":     {"domain": "PostMortem",   "range": "Incident"},

    # ── Quality ──
    "lints":              {"domain": "Lint",         "range": "CodeArtifact"},
    "formats":            {"domain": "Format",       "range": "CodeArtifact"},
    "reviews":            {"domain": "CodeReview",   "range": "PullRequest"},
    "documents":         {"domain": "Documentation", "range": "CodeArtifact"},
    "changelogs":         {"domain": "Changelog",    "range": "Release"},
    "depends_on_lib":     {"domain": "Module",       "range": "Dependency"},
    "vulnerable_in":      {"domain": "Vulnerability", "range": "Dependency"},
    "patches_vuln":       {"domain": "Patch",        "range": "Vulnerability"},

    # ── Process ──
    "scheduled_in":       {"domain": "Issue",        "range": "Sprint"},
    "milestoned_in":      {"domain": "Issue",        "range": "Milestone"},
    "estimated_at":       {"domain": "Issue",        "range": "Estimate"},
    "discussed_in":       {"domain": "Issue",        "range": "Standup"},
    "retrospected_in":    {"domain": "Sprint",       "range": "Retrospective"},
    "action_item_from":   {"domain": "Task",         "range": "Retrospective"},

    # ── Cross-cutting (code ↔ conversation) ──
    "discusses":          {"domain": "Episode",      "range": "CodeArtifact"},
    "modifies":           {"domain": "Episode",      "range": "File"},
    "produces_commit":    {"domain": "Episode",      "range": "Commit"},
    "reviews_code":       {"domain": "Episode",      "range": "PullRequest"},
    "debates":            {"domain": "Episode",      "range": "Issue"},
    "decides_on":         {"domain": "Episode",      "range": "Architecture"},
    "troubleshoots":      {"domain": "Episode",      "range": "Bug"},
    "incident_response":  {"domain": "Episode",      "range": "Incident"},
    "pairs_on":           {"domain": "Person",       "range": "CodeArtifact"},
    "owns_code":          {"domain": "Person",       "range": "CodeArtifact"},
    "reviews_work_of":    {"domain": "Person",       "range": "Person"},
}


# ── Development: language-agnostic code semantics, dev artifacts, security ──
DEVELOPMENT_CLASSES: dict[str, list[str]] = {
    # Programming paradigms (language-agnostic).
    "Paradigm": [
        "ObjectOriented", "Functional", "Procedural",
        "Declarative", "Reactive", "EventDriven",
    ],
    # Algorithms & complexity (abstract CS concepts).
    "Algorithm": [
        "Sorting", "Search", "DynamicProgramming",
        "Greedy", "DivideAndConquer", "Heuristic",
        "ComplexityClass",
    ],
    "ComplexityClass": [
        "Constant", "Logarithmic", "Linear",
        "Quadratic", "Exponential",
    ],
    # Abstract data structures (semantic; concrete storage types live under Data).
    "DataStructure": [
        "Array", "List", "Tree", "Graph",
        "HashTable", "Set", "Map",
        "Stack", "Queue", "Heap",
        "Trie", "Tuple", "Record",
    ],
    # Control flow concepts (semantic, not AST nodes).
    "ControlFlow": [
        "Conditional", "Loop", "Recursion",
        "Iteration", "SwitchCase", "Return",
    ],
    # Error / exception handling.
    "ErrorHandling": [
        "Exception", "Retry", "Fallback", "Recovery",
    ],
    # Concurrency & async.
    "Concurrency": [
        "Thread", "Coroutine", "Async", "Await",
        "Semaphore", "Mutex", "Future", "Promise",
        "Actor", "Pool",
    ],
    # Documents & knowledge artifacts.
    "Document": [
        "Specification", "DesignDoc", "RFC", "ADR",
        "Wiki", "Manual", "Tutorial", "Playbook",
        "Runbook", "README", "Changelog", "License",
    ],
    "Knowledge": [
        "BestPractice", "AntiPattern", "LessonLearned",
        "Pattern", "Convention", "Standard",
    ],
    # Communication / messaging.
    "Communication": [
        "Message", "Notification", "Channel", "Signal", "Event",
    ],
    # Security & access control.
    "Security": [
        "Permission", "Privilege", "AccessToken", "Scope",
        "ACL", "Identity", "Principal", "Threat", "Vulnerability",
    ],
}

DEVELOPMENT_PROPERTIES: dict[str, dict[str, str]] = {
    # ── Algorithms & complexity ──
    "uses_algorithm":     {"domain": "Function",     "range": "Algorithm"},
    "has_complexity":     {"domain": "Algorithm",    "range": "ComplexityClass"},

    # ── Documents & knowledge ──
    "describes":          {"domain": "DesignDoc",    "range": "Architecture"},
    "decides":            {"domain": "ADR",          "range": "Decision"},
    "references_doc":     {"domain": "Document",     "range": "CodeArtifact"},
    "documented_in":      {"domain": "CodeArtifact", "range": "Document"},
    "follows_practice":   {"domain": "CodeArtifact", "range": "BestPractice"},
    "avoids":             {"domain": "CodeArtifact", "range": "AntiPattern"},

    # ── Communication ──
    "delivers":           {"domain": "Channel",      "range": "Message"},
    "notifies":           {"domain": "Notification", "range": "Stakeholder"},
    "consumes":           {"domain": "Service",      "range": "Message"},
    "broadcasts":         {"domain": "Service",      "range": "Event"},

    # ── Security & access control ──
    "grants":             {"domain": "Role",          "range": "Permission"},
    "scoped_to":          {"domain": "AccessToken",  "range": "Scope"},
    "identifies":         {"domain": "Credential",   "range": "Identity"},
    "authenticates_with": {"domain": "Principal",    "range": "Credential"},
    "protects":           {"domain": "Security",      "range": "CodeArtifact"},
    "threatens":          {"domain": "Threat",       "range": "CodeArtifact"},
}


# ── Business & organizational concepts ──
BUSINESS_CLASSES: dict[str, list[str]] = {
    # People / stakeholders in the business context.
    "Stakeholder": ["Customer", "EndUser", "Sponsor", "ProductOwner", "Champion"],
    # Requirements engineering.
    "Requirement": [
        "FunctionalRequirement", "NonFunctionalRequirement",
        "AcceptanceCriterion", "UserStory", "UseCase",
    ],
    # Business rules & governance.
    "BusinessRule": [],
    "Regulation": ["Compliance", "Standard", "Policy", "Law", "SLA"],
    # Workflows / process modeling.
    "Workflow": ["TaskStep", "Transition", "Action", "Trigger", "Gate", "Lane"],
    # Product & market.
    "Product": [],
    "Market": ["Segment", "Competitor", "Trend"],
    "Domain": [],
    "KPI": [],
    # Organization & people structure.
    "Organization": [
        "Company", "Department", "Team",
        "Squad", "Tribe", "Chapter", "Guild",
    ],
    "Role": [
        "Architect", "Engineer", "Manager", "ProductManager",
        "Designer", "QAEngineer", "DevOpsEngineer", "Analyst", "Lead",
    ],
}

BUSINESS_PROPERTIES: dict[str, dict[str, str]] = {
    # ── Requirements ↔ delivery ──
    "requests":          {"domain": "Stakeholder", "range": "Feature"},
    "defines":           {"domain": "Stakeholder", "range": "Requirement"},
    "specifies":          {"domain": "Requirement", "range": "Feature"},
    "validated_by":      {"domain": "Requirement", "range": "AcceptanceCriterion"},
    "implemented_by":    {"domain": "Feature",     "range": "Commit"},

    # ── Organization & people ──
    "member_of":         {"domain": "Person",       "range": "Team"},
    "leads":              {"domain": "Person",       "range": "Team"},
    "reports_to":        {"domain": "Person",       "range": "Person"},
    "assigned_to":       {"domain": "Task",          "range": "Person"},
    "responsible_for":   {"domain": "Role",          "range": "CodeArtifact"},
    "employed_by":       {"domain": "Person",       "range": "Organization"},
    "owns_team":         {"domain": "Organization", "range": "Team"},

    # ── Business rules, workflow, compliance ──
    "automates":         {"domain": "Workflow",     "range": "Process"},
    "governs":           {"domain": "BusinessRule", "range": "Process"},
    "applies_to":        {"domain": "BusinessRule", "range": "Domain"},
    "complies_with":     {"domain": "CodeArtifact", "range": "Standard"},
    "complies":          {"domain": "Process",      "range": "Regulation"},
    "measured_by":       {"domain": "KPI",          "range": "Metric"},
}


def _merge(
    classes_parts: list[dict[str, list[str]]],
    properties_parts: list[dict[str, dict[str, str]]],
) -> dict[str, Any]:
    """Merge taxonomy halves into one multi-parent DAG.

    * Class subclass lists are unioned, so a class listed under multiple parents
      ends up with all of them (multi-parent DAG — what the Graph layer stores).
    * Any subclass name that has no explicit entry is auto-created as a leaf,
      so every ``subClassOf`` target is a real class.
    """
    classes: dict[str, set[str]] = {}
    for part in classes_parts:
        for name, subs in part.items():
            classes.setdefault(name, set())
            classes[name].update(subs)
    # Auto-create leaves for every referenced subclass.
    for subs in list(classes.values()):
        for c in subs:
            classes.setdefault(c, set())

    properties: dict[str, dict[str, str]] = {}
    for part in properties_parts:
        properties.update(part)

    return {
        "classes": {k: {"subclasses": sorted(v)} for k, v in classes.items()},
        "properties": properties,
    }


SEED_ONTOLOGY = _merge(
    [CONVERSATIONAL_CLASSES, CODE_CLASSES, DEVELOPMENT_CLASSES, BUSINESS_CLASSES],
    [
        CONVERSATIONAL_PROPERTIES,
        CODE_PROPERTIES,
        DEVELOPMENT_PROPERTIES,
        BUSINESS_PROPERTIES,
    ],
)